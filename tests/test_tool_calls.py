"""C4: tests for tool call parsing and streaming deltas."""

import json

from server.formats.stream import StreamProcessor
from server.formats.tools import (
    _repair_json,
    convert_tools_to_chat_template,
    find_tool_call_start,
    format_tool_calls_anthropic,
    format_tool_calls_openai,
    parse_tool_calls,
)

TC_OPEN = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/tool_call" + chr(62)
FUNC_OPEN = chr(60) + "function="
FUNC_CLOSE = chr(60) + "/function" + chr(62)
PARAM_OPEN = chr(60) + "parameter="
PARAM_CLOSE = chr(60) + "/parameter" + chr(62)


def _make_tool_call(func_name, params):
    """Build a tool_call XML block from name and param dict."""
    parts = [TC_OPEN, FUNC_OPEN + func_name + chr(62)]
    for key, val in params.items():
        parts.append(PARAM_OPEN + key + chr(62))
        parts.append(val)
        parts.append(PARAM_CLOSE)
    parts.append(FUNC_CLOSE)
    parts.append(TC_CLOSE)
    return "".join(parts)


class TestParseToolCalls:
    def test_no_tool_calls(self):
        text = "Hello, world!"
        visible, calls = parse_tool_calls(text)
        assert visible == "Hello, world!"
        assert calls == []

    def test_single_tool_call(self):
        tc = _make_tool_call("get_weather", {"location": json.dumps("Paris")})
        text = "Sure! " + tc
        visible, calls = parse_tool_calls(text)
        assert visible == "Sure!"
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"] == {"location": "Paris"}

    def test_multiple_tool_calls(self):
        tc1 = _make_tool_call("foo", {"x": "1"})
        tc2 = _make_tool_call("bar", {"y": json.dumps("hello")})
        text = "Result: " + tc1 + " and " + tc2
        visible, calls = parse_tool_calls(text)
        assert visible == "Result:  and"
        assert len(calls) == 2
        assert calls[0]["name"] == "foo"
        assert calls[0]["arguments"] == {"x": 1}
        assert calls[1]["name"] == "bar"
        assert calls[1]["arguments"] == {"y": "hello"}

    def test_multiple_params(self):
        tc = _make_tool_call("search", {"query": json.dumps("python"), "limit": "10"})
        _, calls = parse_tool_calls(tc)
        assert calls[0]["arguments"] == {"query": "python", "limit": 10}

    def test_invalid_json_falls_back_to_string(self):
        tc = _make_tool_call("run", {"code": "not json at all"})
        _, calls = parse_tool_calls(tc)
        assert calls[0]["arguments"]["code"] == "not json at all"

    def test_json_repair_trailing_comma(self):
        tc = _make_tool_call("fn", {"items": "[1, 2, 3,]"})
        _, calls = parse_tool_calls(tc)
        assert calls[0]["arguments"]["items"] == [1, 2, 3]

    def test_json_repair_set_literal(self):
        repaired = _repair_json('{("key": "val")}')
        assert json.loads(repaired) == [{"key": "val"}]

    def test_empty_arguments(self):
        tc = _make_tool_call("no_args", {})
        _, calls = parse_tool_calls(tc)
        assert calls[0]["name"] == "no_args"
        assert calls[0]["arguments"] == {}


class TestFormatToolCallsOpenAI:
    def test_basic_format(self):
        calls = [{"name": "get_weather", "arguments": {"city": "London"}}]
        result = format_tool_calls_openai(calls)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "London"}
        assert result[0]["id"].startswith("call_")

    def test_start_id_offset(self):
        calls = [{"name": "fn", "arguments": {}}]
        result = format_tool_calls_openai(calls, start_id=5)
        assert result[0]["id"] == "call_0005"

    def test_multiple_calls_sequential_ids(self):
        calls = [
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {}},
        ]
        result = format_tool_calls_openai(calls)
        assert result[0]["id"] == "call_0000"
        assert result[1]["id"] == "call_0001"


class TestFormatToolCallsAnthropic:
    def test_basic_format(self):
        calls = [{"name": "get_weather", "arguments": {"city": "London"}}]
        result = format_tool_calls_anthropic(calls)
        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "get_weather"
        assert result[0]["input"] == {"city": "London"}
        assert result[0]["id"].startswith("toolu_")

    def test_unique_ids(self):
        calls = [{"name": "fn", "arguments": {}}]
        r1 = format_tool_calls_anthropic(calls)
        r2 = format_tool_calls_anthropic(calls)
        assert r1[0]["id"] != r2[0]["id"]


class TestConvertToolsToChatTemplate:
    def test_openai_format_passthrough(self):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        result = convert_tools_to_chat_template(tools)
        assert result == tools

    def test_anthropic_format_conversion(self):
        tools = [{"name": "test", "description": "A test", "input_schema": {"type": "object"}}]
        result = convert_tools_to_chat_template(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test"
        assert result[0]["function"]["parameters"] == {"type": "object"}

    def test_server_tools_filtered(self):
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "function", "function": {"name": "real_tool", "parameters": {}}},
        ]
        result = convert_tools_to_chat_template(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "real_tool"

    def test_none_input(self):
        assert convert_tools_to_chat_template(None) is None

    def test_empty_list(self):
        assert convert_tools_to_chat_template([]) is None

    def test_all_server_tools_returns_none(self):
        tools = [{"type": "code_execution_20250101", "name": "exec"}]
        assert convert_tools_to_chat_template(tools) is None


class TestFindToolCallStart:
    def test_full_tag(self):
        text = "Hello " + TC_OPEN + "stuff"
        assert find_tool_call_start(text) == 6

    def test_no_tag(self):
        assert find_tool_call_start("Hello world") == -1

    def test_partial_prefix_at_end(self):
        partial = TC_OPEN[:5]  # e.g. "<tool"
        text = "Hello " + partial
        assert find_tool_call_start(text) == 6

    def test_single_char_prefix(self):
        text = "Hello " + TC_OPEN[0]
        assert find_tool_call_start(text) == 6

    def test_empty_string(self):
        assert find_tool_call_start("") == -1

    def test_tag_at_start(self):
        text = TC_OPEN + "content"
        assert find_tool_call_start(text) == 0


# -- Streaming tool-call delta tests (C4) ------------------------------------


class _FakeTok:
    """Minimal tokenizer stub: maps token IDs to ASCII chars."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if 32 <= i < 127)


def _ids(text):
    return [ord(c) for c in text]


class TestStreamToolDeltas:
    def test_no_tool_call_no_deltas(self):
        proc = StreamProcessor(_FakeTok())
        close_think = chr(60) + "/think" + chr(62)
        proc.add_tokens(_ids("thinking" + close_think + "Just text"))
        proc.drain_content()
        assert proc.drain_tool_deltas() == []

    def test_tool_name_delta(self):
        proc = StreamProcessor(_FakeTok())
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        func_open = chr(60) + "function=get_weather" + chr(62)
        proc.add_tokens(_ids("think" + close_think + "Sure! " + tc_open + func_open))
        proc.drain_content()
        deltas = proc.drain_tool_deltas()
        name_deltas = [d for d in deltas if d["type"] == "name"]
        assert len(name_deltas) == 1
        assert name_deltas[0]["name"] == "get_weather"
        assert name_deltas[0]["index"] == 0

    def test_arguments_delta_incremental(self):
        proc = StreamProcessor(_FakeTok())
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        func_open = chr(60) + "function=fn" + chr(62)
        param_open = chr(60) + "parameter=x" + chr(62)
        # First chunk: name + partial args
        proc.add_tokens(_ids("t" + close_think + tc_open + func_open + param_open + "12"))
        proc.drain_content()
        deltas1 = proc.drain_tool_deltas()
        arg_deltas1 = [d for d in deltas1 if d["type"] == "arguments_delta"]
        assert len(arg_deltas1) >= 1
        # Second chunk: more args
        proc.add_tokens(_ids("34"))
        deltas2 = proc.drain_tool_deltas()
        arg_deltas2 = [d for d in deltas2 if d["type"] == "arguments_delta"]
        assert len(arg_deltas2) >= 1
        # The second delta should only contain the new text
        assert "34" in arg_deltas2[0]["delta"]
        assert "12" not in arg_deltas2[0]["delta"]

    def test_finalize_after_tool_deltas(self):
        proc = StreamProcessor(_FakeTok())
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        tc_close = chr(60) + "/tool_call" + chr(62)
        func_open = chr(60) + "function=add" + chr(62)
        func_close = chr(60) + "/function" + chr(62)
        param_open = chr(60) + "parameter=x" + chr(62)
        param_close = chr(60) + "/parameter" + chr(62)
        full = (
            "think"
            + close_think
            + "Result: "
            + tc_open
            + func_open
            + param_open
            + "42"
            + param_close
            + func_close
            + tc_close
        )
        proc.add_tokens(_ids(full))
        proc.drain_content()
        proc.drain_tool_deltas()
        visible, calls = proc.finalize()
        assert visible == "Result:"
        assert len(calls) == 1
        assert calls[0]["name"] == "add"
        assert calls[0]["arguments"] == {"x": 42}

    def test_content_frozen_after_tool_start(self):
        proc = StreamProcessor(_FakeTok())
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        proc.add_tokens(_ids("think" + close_think + "Hello " + tc_open))
        content = proc.drain_content()
        assert any("Hello" in c for c in content)
        # After tool call starts, content should be frozen
        proc.add_tokens(_ids("more text"))
        assert proc.drain_content() == []
