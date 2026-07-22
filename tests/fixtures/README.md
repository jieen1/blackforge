# Real-request regression fixtures

These are **real request bodies captured from the live server's raw-request
log** (`QSR_DEBUG_REQUESTS=1`, see `server/app.py`), plus one representative
Claude Desktop shape. They exist so that format-parsing or capacity regressions
that drop a user's message — or wrongly reject a long-context request — can
never recur silently.

- `anthropic_simple.json` — captured Anthropic `/v1/messages` (system array +
  user content array).
- `anthropic_claude_desktop.json` — representative Claude Desktop shape
  (system blocks with `cache_control`, user content array, tools with
  `input_schema`, `thinking` config, large `max_tokens`, `stream=true`). This
  is the shape that triggered the original "model didn't get the user message"
  report (actually a 67K capacity rejection — see `test_format_regression.py`).
- `openai_chat_simple.json` — captured OpenAI `/v1/chat/completions`.
- `openai_completions_simple.json` — representative OpenAI `/v1/completions`.

When a new real client request exposes a bug, capture its RAW REQUEST line from
the log and add it here verbatim, then add a case to `test_format_regression.py`.
