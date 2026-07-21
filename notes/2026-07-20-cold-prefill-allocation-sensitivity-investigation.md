# Investigation: allocation-history-sensitive cold prefill GDN state (P3.2 leader verification)

Date: 2026-07-20. Leader independent verification of P3.2 (eviction round), during
review of the executor's `chunk_boundary_partial_share` test.

## Status: OPEN — root cause not yet pinned. Read-only architect subagent dispatched.

## Symptom

For `b_prompt = base(18000 "nl") + diverge(1200)` (19200 tokens), comparing the
**hit path** (`mtp_prefill_with_cache`: restore A@16384 + single-forward continue
[16384,19200)) against a **cold path** (`mtp_prefill_batch(chunk_size=8192)`:
chunked [0,8192),[8192,16384),[16384,19200)) of the SAME prompt:

- The target-model GDN state @19200 differs by a **deterministic, reproducible**
  amount that depends on **slot/block allocation history**, NOT on prompt content
  (`_make_prompt_ids` is pure deterministic text-repeat + tokenize).
- The anchor (argmax at last prefill position) always matches (8581).
- Decode tokens sometimes diverge: hit decodes degenerate (`321,11,321,11,...`);
  cold decodes either degenerate (matching hit) or varied, depending on scenario.

## Reproduced scenarios (all under `/home/bot/.venvs/vllm/bin/python`, cudagraph off)

| Diagnostic (/tmp/) | between hit & cold | hit−cold GDN conv | tokens match |
|---|---|---|---|
| `diag_chunk4.py` | `_clear_persistent_cache` (resets ALL slots) | **47.015625** (repro'd 2x, bit-exact) | NO (cold varied) |
| `diag_chunk4_noclear.py` | nothing (no clear) | **58.765625** | YES (both degenerate) |
| `diag_decomp.py` | an extra slot-3 (X) prefill, no clear | **0.0 bytewise** (hit==X==cold) | YES (both degenerate) |
| `diag_chunk4_zero.py` | clear + **zero ALL attn+GDN KV** before cold | **47.015625** (unchanged) | NO (cold varied) |

`diag_decomp.py` introduces path X = fresh chunked [0,16384) + the SAME
single-forward continue the hit path uses. Result: **Hit==X==Cold, all bytewise
0.0** — three independent code paths agree exactly when slots 0/1/3 stay resident.

## What this rules OUT

- **NOT a restore/addressing bug**: Hit==X (0.0) proves restored [0,16384) attn KV
  + GDN checkpoint reproduce a fresh compute exactly.
- **NOT a continue-mechanism bug**: X==Cold (0.0) proves single-forward continue ==
  chunked continue exactly.
- **NOT stale block content**: `diag_chunk4_zero` zeroed every attention + GDN KV
  tensor before the cold prefill; the 47.015625 diff and varied cold tokens were
  UNCHANGED. So the cold prefill is not reading stale KV.
- **NOT physical-slot aliasing of the GDN measurement**: `_physical_slot(l)=l+1`
  is a stable offset; slot 2 GDN is always physical row 3.
- **NOT run-to-run non-determinism**: `diag_chunk4` reproduces 47.015625 bit-exact
  across separate process launches; `diag_decomp` reproduces 0.0.

## The open contradiction (for the architect)

The cold compute is `mtp_prefill_batch([2],[b_prompt],chunk_size=8192)` — identical
code in every scenario. Yet its GDN @19200 is H+47 (chunk4 scenario, after
`_reset_all` frees all blocks) vs H+0 (decomp scenario, slots resident), where H is
the (identical) hit GDN. Zeroing blocks doesn't change it ⇒ not content. But in
decomp, hit (A's restored blocks 1..1024 for [0,16384)) vs cold (fresh blocks
~2627.. for [0,16384)) use DIFFERENT physical block ids yet match 0.0 ⇒ block id
alone doesn't change the result either. So the cold result depends on allocation
history through some mechanism that is neither block content nor block id directly.

## Leading hypotheses to test

1. **fp8 attention reduction order tied to page-index layout**: the SM120 fp8
   attention kernel's split-K reduction order may depend on `kv_page_indices`,
   making the exact result layout-sensitive (fp8 non-associativity). BUT decomp's
   0.0 across different block ids argues against a simple page-index effect — needs
   the architect to read `vllm/v1/attention/backends/sm120_gqa.py` reduction order
   and `build_attention_metadata_batch`.
2. **A metadata/bookkeeping difference** in how `mtp_prefill_batch`'s chunked loop
   builds attention/GDN metadata under different `block_table[2]` contents
   (e.g. `kv_last_page_len`, `kv_split_size`/`max_num_splits`, `has_initial_state`,
   or `slot_gdn_initialized` interaction with the free-queue state).
3. **A subtle BlockPool free-queue ordering effect** after `_reset_all` (reverse-order
   free) that changes which blocks slot 2 gets AND thereby some metadata that affects
   the kernel numerically.

## Production relevance (must be determined)

INV1 requires a cache hit to reproduce a cold compute of the same prompt. If the
hit-vs-cold difference depends on slot churn / allocation history, this could be a
real INV1 violation in production (not just a test artifact). The decomp 0.0 shows
the paths CAN agree exactly; the chunk4 47 shows they sometimes don't. The architect
must determine whether this is (a) a real bug to fix, or (b) benign fp8-layout
non-associativity that the INV1 test must accommodate (compare like-for-like
allocation, or gate on a tolerance / on state+anchor rather than cross-history
bytewise tokens).

## Executor's original (incorrect) conclusion — for the record

The P3.2 executor attributed the 47 diff to "draft-model spec-row bookkeeping /
cross-path fp noise" and planned to weaken `chunk_boundary_partial_share` to gate on
state+anchor only. Leader review refuted the mechanism: the draft (MTP) model
registers **no GDN layers** (`direct_model_runner.py` ~L1502: `raise RuntimeError
("unexpected GDN layer in MTP draft model")`), and `clone_gdn` iterates target-only
`gdn_layer_names`, so draft bookkeeping cannot affect the measured target GDN diff.
The test-weakening is therefore NOT justified by that reasoning. Whether a
tolerance/state-based gate is nonetheless the correct methodology depends on the
architect's root-cause verdict.

## Diagnostics preserved

`/tmp/diag_decomp.py`, `/tmp/diag_chunk4.py`, `/tmp/diag_chunk4_noclear.py`,
`/tmp/diag_chunk4_zero.py` and their `.log` outputs. Model: unsloth/Qwen3.6-27B-NVFP4,
kv_cache_dtype=fp8_e4m3, block_size=16, blocks_per_slot=1408, num_slots=6, chunk=8192.
