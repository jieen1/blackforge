#!/usr/bin/env python3
"""Format audit: record complete raw API responses for format comparison.

Sends 10 scenarios (5 OpenAI + 5 Anthropic) covering:
- Simple chat (non-streaming + streaming)
- Tool calls (non-streaming + streaming)
- Multi-turn with tool results
- Array content / system blocks

Records complete raw JSON (non-stream) or raw SSE event list (stream).
Usage: python3 tests/format_audit.py --base-url URL --output FILE
"""

import argparse
import json
import time

import requests

WEATHER_TOOL_OAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}
WEATHER_TOOL_ANT = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}

SCENARIOS = [
    # --- OpenAI ---
    {
        "id": "oai_chat",
        "label": "OpenAI: simple chat (non-stream)",
        "method": "POST",
        "path": "/v1/chat/completions",
        "body": {
            "model": "qwen3.6",
            "messages": [{"role": "user", "content": "Say hello in exactly one word."}],
            "max_tokens": 512,
        },
        "stream": False,
    },
    {
        "id": "oai_chat_stream",
        "label": "OpenAI: simple chat (stream)",
        "method": "POST",
        "path": "/v1/chat/completions",
        "body": {
            "model": "qwen3.6",
            "messages": [{"role": "user", "content": "Count from 1 to 5, one per line."}],
            "max_tokens": 512,
            "stream": True,
        },
        "stream": True,
    },
    {
        "id": "oai_tool",
        "label": "OpenAI: tool call (non-stream)",
        "method": "POST",
        "path": "/v1/chat/completions",
        "body": {
            "model": "qwen3.6",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo right now?"}],
            "tools": [WEATHER_TOOL_OAI],
            "max_tokens": 512,
        },
        "stream": False,
    },
    {
        "id": "oai_tool_stream",
        "label": "OpenAI: tool call (stream)",
        "method": "POST",
        "path": "/v1/chat/completions",
        "body": {
            "model": "qwen3.6",
            "messages": [{"role": "user", "content": "What is the weather in Beijing right now?"}],
            "tools": [WEATHER_TOOL_OAI],
            "max_tokens": 512,
            "stream": True,
        },
        "stream": True,
    },
    {
        "id": "oai_multi_turn",
        "label": "OpenAI: multi-turn with tool result",
        "method": "POST",
        "path": "/v1/chat/completions",
        "body": {
            "model": "qwen3.6",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_0001",
                    "content": "Paris: 22C, sunny, light breeze",
                },
                {"role": "user", "content": "Should I bring an umbrella today?"},
            ],
            "max_tokens": 512,
        },
        "stream": False,
    },
    # --- Anthropic ---
    {
        "id": "ant_chat",
        "label": "Anthropic: simple chat (non-stream)",
        "method": "POST",
        "path": "/v1/messages",
        "body": {
            "model": "qwen3.6",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Say hello in exactly one word."}],
        },
        "stream": False,
    },
    {
        "id": "ant_chat_stream",
        "label": "Anthropic: simple chat (stream)",
        "method": "POST",
        "path": "/v1/messages",
        "body": {
            "model": "qwen3.6",
            "max_tokens": 512,
            "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 5, one per line."}],
        },
        "stream": True,
    },
    {
        "id": "ant_tool",
        "label": "Anthropic: tool call (non-stream)",
        "method": "POST",
        "path": "/v1/messages",
        "body": {
            "model": "qwen3.6",
            "max_tokens": 512,
            "tools": [WEATHER_TOOL_ANT],
            "messages": [{"role": "user", "content": "What is the weather in Tokyo right now?"}],
        },
        "stream": False,
    },
    {
        "id": "ant_tool_stream",
        "label": "Anthropic: tool call (stream)",
        "method": "POST",
        "path": "/v1/messages",
        "body": {
            "model": "qwen3.6",
            "max_tokens": 512,
            "stream": True,
            "tools": [WEATHER_TOOL_ANT],
            "messages": [{"role": "user", "content": "What is the weather in Beijing right now?"}],
        },
        "stream": True,
    },
    {
        "id": "ant_multi_turn",
        "label": "Anthropic: multi-turn with tool result + array content",
        "method": "POST",
        "path": "/v1/messages",
        "body": {
            "model": "qwen3.6",
            "max_tokens": 512,
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful weather assistant.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the weather in Paris?"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_0001",
                            "name": "get_weather",
                            "input": {"location": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_0001",
                            "content": "Paris: 22C, sunny, light breeze",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Should I bring an umbrella today?"}],
                },
            ],
        },
        "stream": False,
    },
]


def record_non_stream(base_url, scenario):
    url = base_url + scenario["path"]
    resp = requests.post(url, json=scenario["body"], timeout=180)
    return {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "body": resp.json()
        if resp.headers.get("content-type", "").startswith("application/json")
        else resp.text,
    }


def record_stream(base_url, scenario):
    url = base_url + scenario["path"]
    resp = requests.post(url, json=scenario["body"], timeout=180, stream=True)
    events = []
    raw_lines = []
    for line in resp.iter_lines(decode_unicode=True):
        raw_lines.append(line)
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                events.append({"raw": "[DONE]"})
            else:
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    events.append({"raw": payload, "parse_error": True})
        elif line.startswith("event: "):
            events.append({"event_type": line[7:]})
    return {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "raw_lines": raw_lines,
        "parsed_events": events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="unknown")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    results = {
        "label": args.label,
        "base_url": base,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": {},
    }

    for sc in SCENARIOS:
        sid = sc["id"]
        print(
            "  [{}/{}] {} ...".format(SCENARIOS.index(sc) + 1, len(SCENARIOS), sc["label"]),
            end=" ",
            flush=True,
        )
        try:
            if sc["stream"]:
                rec = record_stream(base, sc)
            else:
                rec = record_non_stream(base, sc)
            results["scenarios"][sid] = rec
            print("OK (status={})".format(rec["status_code"]))
        except Exception as exc:
            results["scenarios"][sid] = {"error": str(exc)}
            print(f"ERROR: {exc}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
