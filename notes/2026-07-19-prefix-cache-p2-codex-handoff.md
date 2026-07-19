# Prefix-cache P2 handoff — why this exists and what to do next

This file exists because the coordinating session driving this work is rooted
in a **different, sibling git repository**
(`/home/bot/project/sm120-flash-attention`, the original sm120-kernel
research project), not in this repo. That mismatch caused a real, confirmed
bug when dispatching Codex for P2 (details in "What went wrong" below). The
user is starting a fresh agent session with this directory
(`/home/bot/project/qwen-sm120-runtime`) as its actual working directory,
specifically to avoid that bug. This document is the complete handoff so that
new session doesn't need to reconstruct context from the other repo's
conversation history.

## 1. What this project is

`qwen-sm120-runtime`: a from-scratch specialized inference runtime for
Qwen3.6-27B-NVFP4 on a single RTX PRO 6000 Blackwell (sm120, Max-Q, ~96GB
VRAM), built to beat native vLLM+FlashInfer serving the same model at a fixed
shape (≤4 concurrent requests, real multi-agent coding-agent workload, MTP
K=3 speculative decoding). By 2026-07-19 the core optimization campaign is
done and verified: the 4K/c=4 headline is at parity-or-better vs. native
(~148-166 tok/s depending on measurement, vs. native's 144.54), a real
OpenAI-compatible HTTP server (`server/app.py`, `server/engine.py`) wraps the
core `DirectModelRunner` engine, and 256K context at concurrency=4 is
confirmed genuinely achievable (see `notes/2026-07-18-session-review-and-
next-steps.md` §25 for the full real-number matrix at 64K/128K/200K/256K).

**Read `PROGRESS.md` first** — it is the up-to-date top-level pointer index
for everything in this project's history. Then read the tail of
`notes/2026-07-18-session-review-and-next-steps.md` (§20 onward, especially
§23-25) for the detailed recent history this handoff builds on.

## 2. Why prefix caching, and why it's the current top priority

§25.4 of `notes/2026-07-18-session-review-and-next-steps.md` documents a
real, measured finding: native vLLM's own already-enabled prefix cache gives
a **~15.4x** speedup on an exact-repeat 256K request (775.9s cold vs. 49.6s
warm, controlled comparison with a fresh server restart between runs). This
runtime currently has **no prefix caching at all** — it is a fixed-4-slot
architecture where each request gets a dedicated, private KV block range,
structurally unlike vLLM's PagedAttention-style shared block-table design.

The user has given an **explicit, high-priority directive**: prefix caching
is core to their real use case (multi-agent coding — many requests sharing
large common prefixes: system prompts, shared codebase/file context, and
especially the same conversation's growing history re-sent each turn) and
must be built to production quality, not skipped or minimally patched. An
earlier investigation's "don't build it now, instrument traffic first"
recommendation was explicitly overridden by the user after seeing the
tradeoffs — the decision to build it is final; the job now is to build it
well.

## 3. The architecture + phased plan (already designed, do not redesign)

`notes/prefix-cache-design.md` (854 lines) is the **authoritative,
already-completed design**, produced by a dedicated architecture-planning
pass that read this project's full history, the runtime's actual code, and
real proven designs (vLLM's own PagedAttention/prefix-caching) before
writing it. **Do not redesign or second-guess this document's architecture
choices without a very good reason** — it already resolved the hard
questions (GDN recurrent-state sharing granularity, correctness invariants,
CUDA-graph interaction, MTP draft/verify interaction, eviction under
capacity=4). Read it in full before starting any new phase.

Key structure:
- §1: problem framing, the two real sharing patterns (simultaneous fan-out
  vs. sequential per-conversation growth).
- §3: architecture — two co-indexed cache groups (attention KV block table
  with hashing/refcounting/LRU; GDN coarse-checkpoint snapshots at chunk
  boundaries, since GDN recurrent state cannot be shared at arbitrary
  positions the way attention KV can).
- §4: nine numbered correctness invariants (INV1-INV9) — every phase's
  verification must map to these explicitly.
- §5: the five-phase implementation plan, P0 through P4, each with a
  build description, a required dedicated test, and a rollback-safe
  boundary.
- §6: risk register (GDN state corruption is the single biggest risk,
  flagged with a specific mitigating test).

## 4. What's done so far (P0, P1 — both verified, both committed)

- **P0** (commit `4aabf1d`): block-table indirection substrate. Introduced
  `block_table[slot]: list[int]` in `runtime/direct_model_runner.py`,
  routing the attention-metadata build path, slot-mapping write path, and
  both CUDA-graph `_fill_buffers` methods through it. Behavior-identical
  refactor by design — verified bit-identical (4K/c=4 headline
  `total_committed_tokens=4116`/`draft_acceptance_rate_pct=70.29204431017119`
  unchanged, full 11-script `mtp_*_check.py` battery + `cudagraph_eager_
  parity_check.py` all PASS).
- **P1** (commit `8557355`): dynamic free-list allocator + reference
  counting. Replaced the static per-slot partition with a real `BlockPool`
  class (`allocate(n)`/`free(block_ids)`, FIFO free queue, `ref_cnt`
  bookkeeping, INV7 reserved-block-0 protection). Still zero cross-slot
  sharing this phase (`ref_cnt` never exceeds 1) — proves the block-table +
  CUDA-graph path tolerates genuinely non-contiguous block placement.
  Verified via a new `benchmarks/prefix_cache_allocator_check.py` plus
  deliberately-fragmented re-runs of `cudagraph_decode_regression.py` and
  `mtp_verify_cudagraph_check.py` (both PASS with provably non-contiguous
  block ids). Full battery + bit-identical headline confirmed again.

Both phases were implemented by a Sonnet general-purpose agent (not Codex).
The code was independently spot-checked by the coordinator (reading the
actual `BlockPool` implementation directly) and judged solid — defensive,
well-invariant-checked, matches the design doc's intent.

`notes/prefix-cache-implementation-log.md` is the running, detailed log for
this whole multi-phase effort — **append to it, do not replace it**, for
every future phase too.

## 5. What's next: P2 — fan-out fork (Pattern A, same-round sharing)

Per `notes/prefix-cache-design.md` §5's "P2" section (read it directly for
the authoritative spec; summarized here for orientation only):

- At admission time, when a same-round batch of requests (`admit_now`, up
  to 4) is being admitted together, detect a common token prefix shared
  among ≥2 of them via direct token comparison (cheap at this scale).
- Prefill the "group leader" fully, force a checkpoint boundary at the
  common-prefix length `Lc` (block-aligned), call the existing
  `snapshot_gdn_state` primitive there.
- For each "sibling": reference the leader's `[0, Lc)` attention blocks
  (`ref_cnt += 1`, all 17 attention layers — 16 target + 1 MTP draft),
  `restore_gdn_state` the snapshot, continue-prefill only the sibling's own
  suffix via the existing chunked-prefill continuation machinery.
- No persistent hash index, no eviction this phase — the shared entry only
  lives for this one admission event. Feature-flagged; must be
  byte-identical to P1 when the fan-out condition isn't met.

Verification must build `benchmarks/prefix_cache_fanout_check.py` covering
INV1 (cache-hit equivalence vs. an independent cold single-slot reference),
INV2 (signal-probe: no cross-sibling contamination in the suffix), and INV4
(MTP draft/verify safety across multiple rounds after the fork) — find the
exact invariant text in the design doc's §4, don't just approximate it. Plus
the standard gate: full 11-script regression battery + the 4K/c=4 headline
bit-identical to baseline.

## 6. What went wrong when Codex was dispatched from the OTHER repo's session

The user directed that P2 onward should be implemented by Codex (via the
`codex:codex-rescue` Claude Code plugin mechanism), not Sonnet, with careful
review of the first few rounds specifically. Two dispatch attempts from the
coordinating session (rooted in `/home/bot/project/sm120-flash-attention`,
**not** this repo) surfaced two real, root-caused problems — both confirmed
by reading the actual plugin source
(`~/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs`):

1. **Wrong workspace root.** The plugin resolves its Codex "workspace root"
   via `resolveWorkspaceRoot(cwd) → ensureGitRepository(cwd)` — i.e. it
   walks UP from whatever `cwd` it was given to find the nearest git repo.
   Since the dispatching session's own cwd was the *other* repo
   (`sm120-flash-attention`), Codex ended up treating that unrelated repo
   as its workspace instead of this one. The `task` command *does* support
   an explicit `--cwd <path>` argument (confirmed in `handleTask`'s own
   argument parser, `valueOptions: ["model", "effort", "cwd", "prompt-file"]`)
   — **this is exactly what starting a fresh session/agent with this
   directory as the actual cwd sidesteps entirely.**

2. **No GPU access in the Codex sandbox (confirmed structural, not a config
   fix).** When Codex actually got running (once the workspace-root issue
   was worked around enough to observe real behavior), its own diagnostic
   trace showed `nvidia-smi` failing with "GPU access blocked by the
   operating system", `/dev/nvidia*` and WSL2's `/dev/dxg` both absent from
   its sandbox's `/dev`, and `torch.cuda.is_available()` returning `False`.
   Reading `codex-companion.mjs` directly confirms this is **not** a
   fixable flag: the plugin hardcodes `sandbox: request.write ?
   "workspace-write" : "read-only"` (line ~491) — there is no third,
   more-permissive sandbox mode anywhere in this script. Every
   `codex:codex-rescue` write-capable task runs under `"workspace-write"`,
   and that mode has no GPU device passthrough, full stop.

**Implication for this new session**: fixing the cwd/workspace-root problem
(by simply running here) does *not* automatically fix the GPU problem —
these are two independent issues. **Before trusting a Codex-authored P2
implementation's own self-reported test results, verify GPU access is
actually available to it early** (e.g., have it run `nvidia-smi` and
`python -c "import torch; print(torch.cuda.is_available())"` as its very
first diagnostic step, the same way it already did unprompted in the failed
attempt). If GPU access is still unavailable even from a natively-rooted
session, the previously-discussed fallback plan applies: **let Codex author
the P2 implementation code, but have the actual GPU verification (the full
regression battery + headline + the new INV1/INV2/INV4 test) run through a
Sonnet agent or the coordinator's own direct tool calls** — this still
satisfies the spirit of "Codex does the development" while not blocking on
an environment limitation Codex itself cannot escalate past.

## 7. Standing review requirement (user's explicit instruction, do not relax)

The user explicitly said: for the first several rounds of using Codex for
actual development (not just review/planning) on this project, review its
output carefully — read the actual diff, run the actual tests yourself to
confirm they really test what they claim, don't just accept a self-reported
"passed" verdict. This project's established broader discipline (see
`PROGRESS.md`'s history) is the same standard applied to every sub-agent
report throughout this whole effort — nothing new, just explicitly
reaffirmed for Codex specifically since it's a new tool being trusted with
this codebase for the first time.

## 8. Practical next step for whoever picks this up

1. Confirm GPU/process state is idle (`nvidia-smi`, `pgrep -af`) before
   starting anything.
2. Dispatch P2 (the spec in §5 above / `notes/prefix-cache-design.md` §5)
   to Codex, now correctly rooted in this repo's own directory — no
   `--cwd` workaround needed since the session's own cwd already resolves
   correctly here.
3. Watch for the GPU-sandbox issue specifically; if it recurs even from
   here, fall back to the hybrid plan in §6 above rather than treating it
   as a retry-able transient failure (it isn't — it's a hardcoded plugin
   limitation as documented above).
4. On completion, review carefully per §7, verify the commit and test
   evidence directly, then continue with P3/P4 per the design doc.
