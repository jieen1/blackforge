"""Real-model bridge backend for the eager engine's OpRegistry.

This is the FIRST real (non-mock) ``prefill``/``decode`` implementation for
this runtime's control plane. It does not yet own GPU memory or drive
model layers directly (see ``notes/phase-3-real-forward.md`` for the scope
this leaves open). Instead it drives an isolated, real vLLM server -- the
same launch path
(``/home/bot/project/sm120-flash-attention/vllm_integration/launch_test_server.py``)
this machine already uses for kernel-level end-to-end validation -- over its
OpenAI-compatible HTTP API, running the real ``unsloth/Qwen3.6-27B-NVFP4``
checkpoint with the real 64-layer hybrid (16 full-attention + 48 GDN) graph.

This matches the phased-rollout intent in `项目实施规划.md`: "每个算子最开始
可以调用 vLLM/FlashInfer/torch，之后逐个替换成自研 kernel" -- vLLM's own engine
is itself a legitimate first "operation" implementation, not a mock.

Token bookkeeping: the OpenAI completions endpoint returns text, not raw
token ids, unless the server is started with ``--return-tokens-as-token-ids``
(not set for the current test server, restarting would cost several minutes
of model reload). Instead this module keeps a local ``transformers``
tokenizer and reconstructs token ids by encoding/decoding text locally. For
single-token decode steps this is exact in the overwhelming majority of
cases; BPE re-merge at a text boundary is a known, small residual risk -- see
the correctness gate in ``benchmarks/real_forward_smoke.py``, which verifies
via detokenized text content, not raw id equality, so it is not exposed to
this risk.
"""

from __future__ import annotations

import requests
from transformers import AutoTokenizer

from runtime.engine import ExecutionRequest


class VLLMBridgeError(RuntimeError):
    """Raised when the real server does not return a usable completion."""


class VLLMBridgeBackend:
    """Drive prefill/decode ops against a real, running vLLM server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        tokenizer_id: str,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._prompt_tokens: dict[str, tuple[int, ...]] = {}
        # This machine's shell exports an HTTP(S)_PROXY pointing at a Clash
        # proxy (see CLAUDE.md); its NO_PROXY list uses "127.*"-style glob
        # patterns that `requests` does not honor (it only matches exact
        # hostnames/domains), so localhost calls were silently routed
        # through the proxy and hung. Bypass proxy handling entirely for
        # this session -- it only ever talks to a local test server.
        self._session = requests.Session()
        self._session.trust_env = False

    def forget(self, request_id: str) -> None:
        """Drop cached prompt bookkeeping after a request is released."""
        self._prompt_tokens.pop(request_id, None)

    def prefill(self, context: ExecutionRequest) -> dict:
        self._prompt_tokens[context.request_id] = context.token_ids
        return self._step(context.request_id, context.token_ids)

    def decode(self, context: ExecutionRequest) -> dict:
        prompt = self._prompt_tokens.get(context.request_id)
        if prompt is None:
            raise VLLMBridgeError(
                f"decode called before prefill for request {context.request_id!r}"
            )
        full_prefix = prompt + context.generated_token_ids + context.token_ids
        return self._step(context.request_id, full_prefix)

    def _step(self, request_id: str, prefix_token_ids: tuple[int, ...]) -> dict:
        prefix_text = self._tokenizer.decode(prefix_token_ids)
        response = self._session.post(
            f"{self._base_url}/completions",
            json={
                "model": self._model,
                "prompt": prefix_text,
                "max_tokens": 1,
                "temperature": 0.0,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        new_text = choice["text"]
        finish_reason = choice["finish_reason"]
        if not new_text:
            if finish_reason == "stop":
                # The model reached a natural stop (e.g. EOS) with nothing
                # left to emit -- a real, valid outcome, not an error. There
                # is no new token id in this case; callers must not feed
                # this into another decode() call for the same request.
                return {
                    "text": "",
                    "token_id": None,
                    "finish_reason": finish_reason,
                    "usage": data.get("usage", {}),
                }
            raise VLLMBridgeError(
                f"empty completion for request {request_id!r} "
                f"(finish_reason={finish_reason!r})"
            )
        # `max_tokens=1` means the server emitted exactly one real token, and
        # `new_text` is that one token's own decoded string, so re-encoding
        # it in isolation reliably reproduces the single id for a byte-level
        # BPE vocabulary (Qwen's tokenizer): each token's text is
        # self-describing, independent of surrounding context. Encoding the
        # full (prefix + new_text) string instead would be *less* reliable
        # here, since a from-scratch BPE merge pass over the whole string can
        # legally choose a different global segmentation.
        new_token_ids = tuple(self._tokenizer.encode(new_text, add_special_tokens=False))
        if len(new_token_ids) != 1:
            raise VLLMBridgeError(
                f"expected exactly 1 local token id for a max_tokens=1 step, got "
                f"{new_token_ids} for text {new_text!r} (request {request_id!r})"
            )
        return {
            "text": new_text,
            "token_id": new_token_ids[0],
            "finish_reason": finish_reason,
            "usage": data.get("usage", {}),
        }
