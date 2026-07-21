# Root-cause & fix/test plan: "cold-prefill allocation-sensitivity" GDN anomaly

Date: 2026-07-20. Read-only architect/debugger. Status: **ROOT-CAUSED (PROVEN by GPU
diagnostics)**. Companion to `notes/2026-07-20-cold-prefill-allocation-sensitivity-investigation.md`.

All diagnostics ran under `/home/bot/.venvs/vllm/bin/python` from repo root, model
`unsloth/Qwen3.6-27B-NVFP4`, `kv_cache_dtype=fp8_e4m3`, `block_size=16`,
`blocks_per_slot=1408`, `num_slots=6`, `chunk=8192`, cudagraph off. Scripts/logs:
`/tmp/diag_rootcause.py(.log)`, `/tmp/diag_rc2..rc8.py(.log)`.
`/tmp/diag_rc9.py(.log)` and `/tmp/diag_rc10.py(.log)` add the decisive benign-proof diagnostics for
the secondary artifact (§3).
`/tmp/diag_rc11.py(.log)` is a single-fresh-process independent reproduction of the entire primary
chain (§2.2): cache hit at 16384, cold→A block-table swap, PRE-DECODE bytewise 0.0, the exact noclear
`58.7656` reproduced as a position drift, and like-for-like post-decode 0.0 with identical tokens.
`/tmp/diag_rc12.py(.log)` does the same for the chunk4 scenario (clear between hit and cold): cold is
pure fresh (`reconcile=0` after clear), PRE-DECODE bytewise 0.0 even with cleared cache, and the exact
headline `47.0156` (ssm 15.7302) reproduced as @19200-vs-@19245 position drift; cold decodes a varied
stream (kv_len 19245) vs hit's degenerate stream (kv_len 19235), target anchor identical (8581).
`/tmp/diag_rc13.py(.log)` + `/tmp/diag_rc14.py(.log)` directly decompose chunk4-cold vs decomp-cold
(§2.5): no prefill kernel input differs (committed conv rows [0,1,2] + ssm bytewise identical across
both colds and a fresh reference); the only PRE-DECODE difference is the §3 dead conv spec rows [3,4,5].
`/tmp/diag_rc15.py(.log)` runs the actual `chunk_boundary_partial_share` benchmark check end-to-end:
all checks True, INV1 restore-boundary conv+ssm diffs 0.0 bytewise, anchor 8581 (§6.4 proven).

---

## 0. Executive summary (5 lines)

1. The reported hit−cold GDN diffs (**noclear 58.765625**, **chunk4 47.015625**) are a
   **measurement-position bug in the diagnostics**, not a runtime defect: they compare the
   hit state cloned at **@19200 (pre-decode)** against the cold slot's **live** state read
   **after `_decode_from_prefill(cold_slot, n=12)` advanced it to @19200+35/45**.
2. Measured at the **same** position, the cold prefill reproduces the hit **bytewise**:
   `hit@19200 vs cold@19200 = (0.0, 0.0)` in noclear, chunk4 (fresh process), and decomp;
   like-for-like post-decode `hit@19235 vs cold@19235 = (0.0, 0.0)` with identical tokens.
3. **Verdict: NOT a real production INV1 violation.** The runtime is correct; INV1 holds
   bytewise (stronger than the design's near-tie bar) when the cache is present.
4. `decomp` reads 0.0 only because `diag_decomp.py` clones `cold_state` **before** decoding;
   `diag_chunk4*.py` read the live post-decode tensor — that asymmetry is the whole "anomaly".
5. A slot-reuse-after-decode GDN **conv**-state difference (~46.75–57.25 conv) was found en route and
   **PROVEN BENIGN** (`diag_rc10`): it lives only in dead spec-extension conv rows (token-positions
   3–5 for K=3) that the next decode overwrites before reading — with contamination present, decode
   tokens are byte-identical to clean and post-decode GDN converges to (0.0,0.0). It is a
   test-hygiene artifact, **not** a correctness bug, and does not cause this anomaly.

---

## 1. The anomaly, restated

For `b_prompt = base(18000 "nl") + diverge(1200)` (19200 tokens), the investigation compared
a HIT (`mtp_prefill_with_cache`: restore A@16384 + single-forward continue) against a COLD
(`mtp_prefill_batch(chunk_size=8192)`: chunked [0,8192),[8192,16384),[16384,19200)) and found a
"deterministic, allocation-history-dependent" target-GDN @19200 difference:

| diagnostic | between hit & cold | reported hit−cold GDN conv |
|---|---|---|
| `diag_chunk4.py`        | `_clear_persistent_cache` | **47.015625** (repro'd 2× bit-exact) |
| `diag_chunk4_noclear.py`| nothing                   | **58.765625** |
| `diag_decomp.py`        | extra slot-3 (X) prefill  | **0.0 bytewise** (hit==X==cold) |
| `diag_chunk4_zero.py`   | clear + zero all KV       | **47.015625** (unchanged) |

Anchor always 8581. The open contradiction: identical cold code yields H+47 vs H+0; zeroing
blocks doesn't change it; yet decomp's hit/cold use different block ids and match 0.0.

---

## 2. Root cause #1 (PRIMARY, PROVEN): the diagnostics compare two different sequence positions

### 2.1 The decisive code evidence

`diag_chunk4_noclear.py` / `diag_chunk4.py` (and `_zero`) all do, in order:

```python
hit_pr   = runner.mtp_prefill_with_cache([hit_slot], [b_prompt])
hit_state = clone_gdn(hit_slot)                                  # <-- HIT cloned @19200 (pre-decode)
hit_tokens = _decode_from_prefill(runner, hit_slot, hit_pr, n=12) # hit advances to @19200+dh
...
cold_pr = runner.mtp_prefill_batch([cold_slot], [b_prompt], chunk_size=chunk)
cold_tokens = _decode_from_prefill(runner, cold_slot, cold_pr, n=12)  # <-- COLD advances to @19200+dc
cp = _physical_slot(cold_slot)
for i,n in enumerate(runner.gdn_layer_names):
    conv, ssm = runner.kv_caches[n]; hc, hs = hit_state[n]
    maxc = max(maxc, float((hc.float()-conv[cp].float()).abs().max().item()))  # <-- reads COLD LIVE @19200+dc
```

The comparison is `hit_state` (a **clone taken at @19200**, before the hit's own decode) versus
`runner.kv_caches[n][0][cp]` — the cold slot's **live** tensor, read **after**
`_decode_from_prefill(cold_slot, n=12)` ran 12 MTP verify-commit rounds
(`benchmarks/prefix_cache_eviction_check.py:382` `_decode_rounds` → `mtp_verify_and_commit_batch`),
advancing the cold slot to `slot_kv_len = 19235` (noclear) / `19245` (chunk4). **It is a
@19200-vs-@19235/19245 comparison — two different points in the recurrence.** The GDN conv state
naturally drifts by tens of units over ~35-45 decoded tokens; that drift is the entire "58/47".

`diag_decomp.py` differs in exactly one decisive way — it clones the cold state **before**
decoding:

```python
cold_pr = runner.mtp_prefill_batch([cold_slot], [b_prompt], chunk_size=chunk)
cold_state = clone_gdn(cold_slot)            # <-- cloned @19200, BEFORE decode
cold_tokens = _decode_from_prefill(runner, cold_slot, cold_pr, n=12)
```

so it compares @19200-vs-@19200 and gets 0.0. **The decomp/noclear asymmetry is purely "clone
before decode" vs "read live after decode".**

### 2.2 The decisive experiment (PROVEN)

`/tmp/diag_rc3.py` reproduces noclear and measures at **both** positions in one run:

```
noclear:
  PRE-DECODE  hit@19200 vs cold@19200 : (0.0, 0.0)      <- correct INV1 compare
  hit decode -> kv_len 19235 ; cold decode -> kv_len 19235 ; tokens match True
  POST-DECODE hit@19200 vs cold@19235 : (58.7656, 23.4822) <- the diagnostic's ACTUAL compare == the "58"
  POST-DECODE hit@19235 vs cold@19235 : (0.0, 0.0)       <- like-for-like post-decode
```

`/tmp/diag_rc2.py` PART A reproduces noclear **exactly** (A, hit, decode-hit-12, reset 2, cold 2)
but clones the cold state pre-decode: `hit vs cold = (0.0, 0.0)`. `/tmp/diag_rc5.py` T1 reproduces
chunk4 in a fresh process and clones pre-decode: `hit@19200 vs cold@19200 = (0.0, 0.0)`; the
original chunk4 "47.015625" is reproduced exactly only when read post-decode
(`diag_rc3` chunk4 POST-DECODE = `(47.0156, 15.7302)`).

**Independent single-process reproduction (`/tmp/diag_rc11.py`, noclear-style).** One fresh process
runs the whole chain end-to-end: populate A → `reconcile_prefix_hit(b_prompt)=16384`; HIT
`mtp_prefill_with_cache([1],[b_prompt])` and COLD `mtp_prefill_batch([2],[b_prompt],chunk)` both reach
`slot_kv_len=19200` with anchor 8581; and **cold's first 1024 `block_table` entries == A's cached
blocks (True)** — direct confirmation of the §2.3 swap onto A's fp8 bytes. Then:

```
PRE-DECODE  hit@19200 vs cold@19200             : ((True,True), 0.0, 0.0)        <- correct INV1 compare
POST-DECODE hit@19200(pre) vs cold@19235(live)  : ((False,False), 58.7656, 23.4822)  <- reproduces noclear "58.765625" bit-exact
POST-DECODE hit@19235 vs cold@19235             : ((True,True), 0.0, 0.0)        <- like-for-like
decode tokens hit==cold (like-for-like)         : True
```

This reproduces the reported noclear value **58.765625 bit-exact** and localizes it entirely to the
@19200-vs-@19235 position drift; the same-position compare is bytewise 0.0 with layer-0 bytewise True.

The same experiment for the **chunk4** scenario (`/tmp/diag_rc12.py`) — which runs
`_clear_persistent_cache` between hit and cold so the cold is **pure fresh** (`reconcile_prefix_hit=0`
after the clear) — gives the identical conclusion and reproduces the headline value bit-exact:

```
reconcile after clear                           : 0   (cold is pure fresh, different blocks)
PRE-DECODE  hit@19200 vs cold@19200             : ((True,True), 0.0, 0.0)          <- INV1 holds EVEN with cleared cache
POST-DECODE hit@19200(pre) vs cold@19245(live)  : ((False,False), 47.0156, 15.7302)  <- reproduces chunk4 "47.015625" bit-exact
POST-DECODE kv_lens hit/cold                    : 19235 / 19245 (cold varied stream vs hit degenerate; anchor 8581 both)
```

So **both** reported numbers — noclear **58.765625** (`diag_rc11`) and chunk4 **47.015625**
(`diag_rc12`) — are independently reproduced as pure @19200-vs-post-decode position drift, while the
same-position prefill compare is bytewise 0.0 in both. The chunk4 cold's different kv_len advance
(19245 vs hit's 19235) and varied token stream are the draft model's block-id-sensitive split-K decode
proposals (§2.4); the **target** anchor (8581) and prefill-boundary state are identical.

**Conclusion (PROVEN):** at the same sequence position the cold prefill equals the hit bytewise.
The 47/58 are the GDN recurrence drift between @19200 and @19200+35/45, i.e. a measurement
artifact. Zeroing KV (`diag_chunk4_zero`) left 47 unchanged precisely because the artifact is a
position mismatch, not stale content.

### 2.3 Why the cold prefill is bit-exact equal to the hit (mechanism)

`mtp_prefill_batch`'s chunked loop publishes committed blocks at each block-aligned chunk boundary
and at completion (`runtime/direct_model_runner.py:3939` at the block-aligned chunk boundary and
`:3968` at completion), and
`_publish_committed_blocks` (`runtime/direct_model_runner.py:4229`, swap at `:4274-4276`) performs
**write-time dedup**: when the just-computed prefix block's chained hash already exists in the
content index (A's cached `base[0,16384)` blocks), it **swaps** `block_table[slot][i]` onto the
canonical cached block and frees the fresh duplicate. Consequently, when the cache is present
(noclear, decomp), the cold's chunks 2/3 attend over **A's very fp8 KV bytes** — the identical bytes
the hit reads after `restore_cached_prefix` (`:4362`) references A's blocks. Same bytes in ⇒ same
attention out ⇒ same hidden states ⇒ same GDN state, **bytewise**. Verified directly:
`diag_rootcause.py` PART 1/2 print `cold block_table[2][:1024] == A block_table[0][:1024] : True`
and `hit vs cold = (0.0, 0.0)`.

### 2.4 Why the prefill is block-id-insensitive (resolves the open contradiction)

The eager prefill path builds attention metadata with `kv_split_size = max(new_kv_lens)`,
`max_num_splits = 1` (`runtime/direct_model_runner.py:737`; `max_num_splits=1` at `:931`/`:941` for
the prefill path, decode split-K via `decode_fixed_max_num_splits` at `:1607`) — **no split-K**.
Split-K reduction order is a property of the **decode-specialized** kernel only
(`sm120_gqa.py`: `decode_qo_len`/`_MAX_DECODE_QO_LEN=16`, `flash_attn_sm120_decode_paged`); a
genuine prefill (`is_decode=False` ⇒ `decode_qo_len=0`) routes to the general chunked-prefix kernel
that reduces in page-list (position) order. So the prefill result is independent of which physical
block ids back each position. PROVEN empirically: `diag_rootcause.py` PART 3
`noclear_cold(blocks 1427..) vs decomp_cold(blocks 2627..) = (0.0, 0.0)` and PART 4 pure-fresh
cold (blocks 3165..) vs hit (A's blocks 1..1024) `= (0.0, 0.0)`; `diag_rc4.py` pure-fresh run1 vs
run2 on different blocks `= (0.0, 0.0)`. This is why decomp's hit (A's blocks 1..1024) and cold
(blocks 2627..) match 0.0 — block id was never the lever; the **measurement position** was.

### 2.5 Direct chunk4-cold vs decomp-cold decomposition — which kernel input differs? (rc13/rc14)

The investigation's open contradiction asks specifically which input to which kernel differs between
**chunk4-cold** (H+47) and **decomp-cold** (H+0). `/tmp/diag_rc13.py` runs both colds in one process
(decomp-style: cache present + extra X prefill, cold on slot 2 → blocks `[1,2,3,4,5,6]` = A's cached
blocks; chunk4-style: cache cleared, cold on slot 2 → fresh blocks `[1428,1502,1251,…]`) and compares
them at @19200:

```
block_table[:6]  decomp-cold [1,2,3,4,5,6]  vs  chunk4-cold [1428,1502,1251,1501,1500,1499]  (DIFFERENT blocks)
PRE-DECODE  decomp-cold@19200 vs chunk4-cold@19200 : ((False,True), 55.25, 0.0)   <- conv 55.25, ssm BYTEWISE equal
```

The PRE-DECODE diff is **not** 0.0 — but `/tmp/diag_rc14.py` proves it is entirely the §3 dead-spec-row
contamination, **not** a prefill-compute difference. Against a clean fresh-slot reference:

```
decomp-cold vs fresh-ref : (0.0, 0.0)    per-row [0,0,0, 0,0,0]              <- prefill compute identical, both clean
chunk4-cold vs fresh-ref : (55.25, 0.0)  per-row [0,0,0, 55.25,48.25,55.25]  <- ONLY dead spec rows [3,4,5]
decomp-cold vs chunk4    : (55.25, 0.0)  per-row [0,0,0, 55.25,48.25,55.25]  <- the rc13 55.25, spec rows only
```

**Conclusion (PROVEN): no prefill kernel input differs between chunk4-cold and decomp-cold.** The
committed conv rows **[0,1,2]** and the entire **ssm** state are bytewise identical across chunk4-cold,
decomp-cold, and a fresh reference — independent of physical block ids (`[1,2,3,…]` vs `[1428,…]`) and
of cache state (present vs cleared). The only difference is the dead conv **spec** rows **[3,4,5]**,
which reflect each slot's prior **decode** history (here chunk4-cold's slot had just been decoded as
decomp-cold before reuse — the §3 artifact), are never read before being overwritten (§3, `diag_rc10`),
and are irrelevant to INV1. The diagnostics' H+47 (chunk4) vs H+0 (decomp) is therefore **not** a
cold-compute difference at all: it is the §2 measurement-position artifact (chunk4 reads the cold live
post-decode @19245; decomp clones it pre-decode @19200), with the draft model's block-id-sensitive
decode (§2.4) setting the two post-decode positions/streams apart.

---

## 3. Root cause #2 (SECONDARY, PROVEN **BENIGN** artifact): slot-reuse-after-decode GDN conv dead-row staleness

Found while chasing the chunk4 PRE-DECODE component. **Not the cause of this anomaly** (the
diagnostics' cold slot is fresh/never-decoded). It is a real *tensor* difference but **not a
correctness bug** — it is confined to dead spec-extension conv rows and never reaches an output.

**Symptom:** if a logical slot runs a prefill **plus MTP decode** (spec-decode GDN path), then is
reset (`reset_slot`, which deliberately does **not** zero tensors — `runtime/direct_model_runner.py:2574`)
and reused for a fresh prefill, a **full-tensor** comparison of the new prefill's GDN **conv** state
versus a never-used slot shows ~46.75 (ssm 0.0). As proven below, this is stale data in **dead spec
rows only** — the live state and all outputs are correct.

**Evidence (`/tmp/diag_rc8.py`)** — pollute slot 2 with `b_prompt` prefill + 12 decode rounds, then
fresh `b_prompt` prefill on slot 2 vs a clean slot 5:

```
T1  reset_slot only (PRODUCTION reuse)            : (46.75, 0.0)   <- real production path
T2  test-only _clear_persistent_cache             : (46.75, 0.0)
T3  clear + ZERO conv row 3 only                  : (0.0, 0.0)     <- zeroing conv row fixes it
T4  clear + ZERO ssm spec rows {3,16,17,18} only  : (46.75, 0.0)   <- ssm spec rows NOT the cause
```

and `/tmp/diag_rc6.py`: pollute slot 2 with a **prefill only** (no decode, conv row norm 54.0) then
reuse → `(0.0, 0.0)`. So:

- the contaminant is stale **conv** data in the slot's own physical row (`_physical_slot(2)=3`),
  specifically the dead spec token-positions [3,4,5] (decisive experiment below); zeroing that whole
  physical row makes the full-tensor compare match (T3);
- a prior **prefill** leaves the conv row in a state the next prefill's `has_initial_state=False`
  correctly ignores (diag_rc6 = 0.0), but a prior **decode** (spec-decode GDN forward,
  `build_gdn_metadata_spec_batch` / `_ssm_spec_row`, `runtime/direct_model_runner.py:83`) leaves the
  conv row in a state `has_initial_state=False` does **not** clear ⇒ contamination.

**Why it didn't show up here:** the GDN conv state has only `total_physical_slots=7` physical rows
(`runtime/direct_model_runner.py:568-587`), each with 6 token-positions (3 committed + 3 spec for
K=3). The fresh prefill's chunk-1 (`has_initial_state=False`) reads zeros (kernel read path) and
correctly overwrites the committed token-positions 0–2, but leaves the dead spec positions 3–5 stale.
The anomaly diagnostics never hit this because their cold slot (2) is fresh in a fresh process
(`diag_rc5` T1 PRE-DECODE = 0.0), and — decisively — even when present it does not affect outputs.

**Production relevance — PROVEN BENIGN (decisive experiment `/tmp/diag_rc10.py`).** The stale bytes do
**not** reach any output. With contamination fully present (pollute slot 2 with a *different* prompt
`a_prompt` prefill+decode, `reset_slot`, fresh COLD `b_prompt` prefill on the contaminated slot):

```
PRE-decode conv diff (contamination magnitude) : (57.25, 0.0)   <- contamination IS present
anchor reused vs clean                         : 8581 == 8581
DECODE tokens reused == clean (12 MTP rounds)  : True           <- identical streams
first token divergence index                   : None
POST-decode GDN reused vs clean                : (0.0, 0.0)     <- states converge exactly
stale elements located                         : conv token-position rows [3,4,5] of 6; rows [0,1,2]
                                                 clean (per-row max [0,0,0,53.75,57.25,50.75])
```

The conv state has 6 token-position rows: rows **0–2** are the committed conv state (the last
`width-1` tokens, kernel width 4) that the next operation reads as initial state; rows **3–5** are the
K=3 spec-decode extension rows. The fresh prefill (non-spec path) writes only the committed rows 0–2;
the spec rows 3–5 keep the previous decode's data — but they are **dead**: the next decode reads only
the committed rows 0–2 (correct) and overwrites 3–5 before reading them. Hence identical decode tokens
and exact post-decode convergence. (`/tmp/diag_rc9.py` PART A confirms prompt-independence: a_prompt
pollution → 57.25; PART C confirms same-prompt pollution leaves 0 stale elements because the overwritten
data is identical; PART B same-prompt decode tokens identical, post-decode (0.0,0.0).)

**Kernel mechanism (PROVEN by code read of the read-only sibling repo).** The conv/ssm asymmetry is
explicit in `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` `_forward_core`:
- **SSM prefill** (`:1513-1531`) zeros fresh state in Python — `initial_state = ssm_state[prefill_state_indices];
  initial_state[~prefill_has_initial_state, ...] = 0` — then **fully overwrites**
  `ssm_state[prefill_state_indices] = last_recurrent_state`. Every SSM row is recomputed from zero
  regardless of prior content ⇒ ssm diff always 0.0.
- **CONV prefill** (`:1365-1372`) passes `has_initial_state` to `causal_conv1d_fn` with **no** Python-side
  zeroing. The kernel READ path is safe: `has_initial_state=False` ⇒ `col0..col3 = tl.zeros(...)`
  (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:184-191`), so the conv **output** never reads
  stale bytes. The kernel WRITE path (`causal_conv1d_fn` STEP 2) updates only the committed rows named by
  `cache_indices = non_spec_state_indices_tensor`; the spec-extension rows (written during the prior decode
  via `causal_conv1d_update(..., conv_state_indices=spec_state_indices_tensor[:,0])`, `:1344-1355`) are left
  stale. That asymmetry — SSM zeroed+overwritten in Python, conv only read-zeroed and committed-row-written —
  is the full mechanism, and the stale conv rows are dead ⇒ benign.

**Residual (test-hygiene only, NOT correctness):** a test that compares the **full** conv tensor
(including dead spec rows 3–5) bytewise across a slot-reuse-after-decode would see a spurious ~46–57
mismatch even though the runtime is correct. `_clear_persistent_cache`
(`benchmarks/prefix_cache_eviction_check.py:331`) resets counters and the content index but does not zero
GDN tensors. The live `chunk_boundary_partial_share` test compares prefill-boundary state on fresh slots,
so it is unaffected; see §5.3 for optional hardening.

---

## 4. Verdict

**The reported anomaly is NOT a real production INV1 violation.** It is a test-measurement bug
(§2). INV1 (`notes/prefix-cache-design.md:433`) — "a cache hit produces the same committed tokens as
a full cold prefill, within `NEAR_TIE_LOGIT_MARGIN=2.0` (fp8/batch non-associativity the only
permitted difference)" — in fact holds **bytewise** at the prefill boundary when the cache is
present (§2.3): `hit@19200 == cold@19200 == (0.0,0.0)`, anchor 8581 == 8581, and (noclear) the
decoded token streams match exactly. No runtime code change is required to make INV1 hold.

The chunk4 scenario additionally clears the cache between hit and cold, so its cold is a *pure
fresh* compute on different physical blocks; that introduces a benign fp8 non-associativity delta
versus the hit's restore+A-blocks path — but a cleared cache is **not** an INV1 scenario (there is
no hit to compare against; in production the cold path itself re-publishes the blocks a later hit
would restore, so hit==cold). The decode-token divergence seen in chunk4 (cold varied, hit
degenerate) is the **draft** model's block-id-sensitive decode proposals (split-K decode kernel,
§2.4) — the **target** state and anchor are identical; the draft is only a speedup.

The executor's original attribution ("draft-model spec-row bookkeeping / cross-path fp noise") is
**incorrect as a mechanism** (the draft registers no GDN layers; the target GDN diff was a position
mismatch), but see §5 for the gate assessment.

---

## 5. Test-methodology plan (this is the fix) + executor-gate assessment

The runtime stays as-is. The fix is in **how INV1 is measured**.

### 5.1 The correct INV1 comparison (like-for-like)

1. **Compare at the same sequence position.** Clone **both** hit and cold GDN states immediately
   after their prefills, **before any `_decode_from_prefill`**. Compare the clones. Never compare a
   pre-decode clone against a live post-decode tensor (the bug in `diag_chunk4*.py`).
   Also compare slots with the **same decode history** (or mask the dead conv spec rows [3,4,5]):
   `diag_rc13`/`diag_rc14` show a pre-decode **full-tensor** compare can otherwise show the §3
   dead-spec-row contamination (~55 conv, ssm 0.0) even though the live committed state is bytewise equal.
2. **Keep the cache present** (noclear-style) so the cold path's `_publish_committed_blocks` dedup
   swaps onto the cached prefix blocks (§2.3) ⇒ the comparison is bytewise `(0.0,0.0)`. Do **not**
   clear the cache between hit and cold for an INV1 assertion (a cleared cache forces a pure-fresh
   cold on different blocks → benign fp8 delta, not an INV1 signal).
3. **Gate (matches design R1/R6, `prefix-cache-design.md:708,713`):** GDN **layer-0 bytewise**
   exact (the R1 addressing proof — fp8 noise cannot hide a wrong-block/wrong-prefix read) **AND**
   full-48-layer-stack near-tie (`atol=rtol=1e-2`, the existing `_gdn_stack_compare`) **AND** anchor
   exact. This is exactly what `_run_chunk_boundary_partial_share`
   (`benchmarks/prefix_cache_eviction_check.py:558`) already does: it compares the restored `@L`
   state vs an independent true-cold chunked `B[0,L)` (`inv1_restore_state_matches_cold`) and the
   hit anchor vs a true-cold full anchor (`inv1_anchor_matches_cold`), and **deliberately does not**
   compare cross-path decode tokens (its docstring cites R6).
4. **Drop cross-path decode-token equality** as an INV1 bar. Decode tokens depend on the draft
   model's block-id-sensitive split-K proposals (§2.4) and on measuring at the same position; they
   are not a clean INV1 signal. If a decode-stream check is wanted, decode **both** hit and cold the
   same number of rounds from their (identical) prefill state and compare like-for-like — which
   passes bytewise (`diag_rc3` noclear `hit@19235 vs cold@19235 = (0.0,0.0)`, tokens match).

### 5.2 Is the executor's planned state+anchor gate correct?

**Correct in substance, wrong in rationale, and must be applied at the prefill boundary.** Gating on
state+anchor (not cross-path decode tokens) is exactly the design-sanctioned R1/R6 methodology and
is what `chunk_boundary_partial_share` already implements — so the gate should be kept. **But** it
must compare the **prefill-boundary** state (pre-decode clones, §5.1.1); a state+anchor gate that
reads the cold slot's live tensor *after* decoding would still "see" the 47/58 drift and either
fail or be wrongly waved away as "fp noise". **Verified directly:** `/tmp/diag_rc15.py` runs
`chunk_boundary_partial_share` end-to-end and it PASSES — restore @16384 vs cold @16384 conv+ssm diffs
**0.0 bytewise** (stronger than its near-tie gate), anchor 8581 — so the gate is correct in practice,
not just in design. The executor's *reason* (draft spec-row bookkeeping)
should be struck from the record; the *real* reason the token stream can't be compared cross-path is
the position mismatch (§2) plus draft split-K block-id sensitivity (§2.4). No weakening beyond the
already-correct state+anchor gate is needed — and the layer-0 bytewise addressing proof must be
**retained**, not relaxed.

### 5.3 Secondary artifact (§3) — optional test-hygiene hardening, NOT a correctness fix

§3 is proven benign (decode output is byte-identical with contamination present, `diag_rc10`), so **no
runtime change is required for correctness.** The only residual is test isolation: a full-tensor
bytewise GDN comparison across a slot-reuse-after-decode would spuriously see the dead spec rows
(~46–57). If the test battery ever does that, optional hardening (pick one): (a) compare only the
**live committed** conv rows 0–2 (mask out spec rows 3–5) — preferred, matches what the runtime actually
reads; or (b) zero the slot's conv spec rows in `reset_slot` / `_clear_persistent_cache` (cheap); or
(c) compare GDN **after one decode round**, which converges to (0.0,0.0) (`diag_rc10`); or (d) use fresh
slots per test. The `diag_rc8` T1 `(46.75,0.0)` is therefore a *hygiene* gate, not a correctness gate —
it need not become `(0.0,0.0)` unless a test chooses to compare full tensors across reuse.

---

## 6. Verification gates for the leader

Run under `/home/bot/.venvs/vllm/bin/python`, one GPU job at a time, filter
`2>&1 | grep -vE "INFO |WARNING |Loading safetensors|Completed \|"`.

1. **Artifact confirmed (re-run `/tmp/diag_rc3.py`):** noclear PRE-DECODE `hit@19200 vs cold@19200`
   = `(0.0,0.0)`; POST-DECODE `hit@19200 vs cold@19235` = `(58.77,23.48)`; like-for-like POST-DECODE
   `hit@19235 vs cold@19235` = `(0.0,0.0)`; chunk4 PRE-DECODE (fresh process, `/tmp/diag_rc5.py` T1)
   = `(0.0,0.0)` and POST-DECODE = `(47.0156,15.7302)`.
   `/tmp/diag_rc11.py` reproduces the entire noclear chain in ONE fresh process and is the cleanest
   single command: cache hit at 16384, cold `block_table[:1024]` == A's cached blocks (True),
   PRE-DECODE `((True,True),0.0,0.0)`, POST-DECODE buggy compare `58.7656` bit-exact, like-for-like
   `((True,True),0.0,0.0)`, decode tokens match.
   `/tmp/diag_rc12.py` is the chunk4 counterpart: clear between hit and cold (`reconcile=0` after),
   PRE-DECODE `((True,True),0.0,0.0)` even with cleared cache, POST-DECODE buggy compare `47.0156`
   (ssm 15.7302) bit-exact, cold kv_len 19245 (varied stream) vs hit 19235 (degenerate), anchor 8581.
   `/tmp/diag_rc13.py` + `/tmp/diag_rc14.py` answer the task's exact question (chunk4-cold vs
   decomp-cold, §2.5): committed conv rows [0,1,2] + ssm bytewise identical across both colds and a
   fresh ref (per-row `[0,0,0,0,0,0]`); the only PRE-DECODE diff is the §3 dead spec rows [3,4,5] (55.25).
2. **INV1 bytewise at boundary:** noclear-style (cache present) `hit@19200 vs cold@19200` layer-0
   bytewise True, full-stack `(0.0,0.0)`, anchors equal (8581). (`/tmp/diag_rc2.py` PART A;
   `/tmp/diag_rootcause.py` PART 1/2.)
3. **Block-id insensitivity of prefill:** pure-fresh cold on two different block ranges equal
   (`/tmp/diag_rootcause.py` PART 3/4; `/tmp/diag_rc4.py` run1 vs run2) = `(0.0,0.0)`.
4. **`chunk_boundary_partial_share` passes as written — PROVEN (`/tmp/diag_rc15.py` ran it directly):**
   all checks True; `inv1_restore_state_matches_cold` and `inv1_anchor_matches_cold` True; the restore
   @16384 vs cold @16384 conv AND ssm diffs are **0.0 bytewise** (stronger than the test's near-tie gate),
   `cold_anchor=8581`, `hit_reuses_cached_blocks=True`. The test is robust against the §3 dead-spec-row
   contamination because it compares prefill-boundary state on fresh slots and never decodes first.
5. **Secondary artifact proven benign:** `/tmp/diag_rc10.py` — with contamination present (PRE-decode
   conv `(57.25,0.0)`, a_prompt pollution), decode tokens are identical to clean and POST-decode GDN =
   `(0.0,0.0)`; stale elements confined to conv rows [3,4,5]. `/tmp/diag_rc9.py` PART A = `(57.25,0.0)`
   (prompt-independent), PART C same-prompt = 0 stale elements. No correctness fix required (§5.3).

---

## 7. Evidence ledger — PROVEN (diagnostic) vs INFERRED

**PROVEN by GPU diagnostic:**
- noclear/chunk4 47/58 are a @19200-vs-@19200+35/45 position mismatch; same-position compare is
  bytewise 0.0 (`diag_rc3`, `diag_rc2` A, `diag_rc5` T1, and `diag_rc11` — a single-fresh-process
  reproduction of the full noclear chain incl. the exact 58.7656 and the cold→A block-table swap; and
  `diag_rc12` — the chunk4 counterpart reproducing the exact 47.0156 with cleared cache, PRE-DECODE
  bytewise 0.0, cold varied stream kv_len 19245 vs hit degenerate 19235, anchor identical; and
  `diag_rc13`/`diag_rc14` — direct chunk4-cold-vs-decomp-cold decomposition proving NO prefill kernel
  input differs: committed conv rows [0,1,2] + ssm bytewise identical across both colds and a fresh ref,
  the only PRE-DECODE diff being the §3 dead spec rows [3,4,5] = 55.25).
- the actual `chunk_boundary_partial_share` test PASSES end-to-end (`diag_rc15`): all checks True,
  `inv1_restore_state_matches_cold`/`inv1_anchor_matches_cold` True, restore @16384 vs cold @16384 conv+ssm
  diffs 0.0 bytewise, `cold_anchor=8581` — converting §6.4 from inferred to proven and validating the §5
  test methodology; robust against §3 contamination (no decode before the boundary compare).
- decomp 0.0 is because `diag_decomp.py` clones cold pre-decode (code inspection + `diag_rc3`).
- cold prefill == hit bytewise at @19200 with cache present; cold swaps onto A's cached blocks
  (`diag_rootcause` PART 1/2, block_table equality).
- prefill is block-id-insensitive; split-K is decode-only (`diag_rootcause` PART 3/4, `diag_rc4`;
  `build_attention_metadata_batch` `max_num_splits=1`; `sm120_gqa.py` decode-gated split).
- slot-reuse-after-**decode** leaves stale GDN **conv** bytes (~46.75–57.25, ssm 0.0) but is **PROVEN
  BENIGN**: contamination is confined to dead spec-extension conv rows [3,4,5] (committed rows [0,1,2]
  clean); with contamination present, decode tokens are byte-identical to clean and post-decode GDN
  converges to (0.0,0.0) (`diag_rc10`); prompt-independent (`diag_rc9` A = 57.25); same-prompt pollution
  leaves 0 stale elements (`diag_rc9` C); prefill-only pollution does not contaminate (`diag_rc6`);
  zeroing the conv row masks it (`diag_rc8` T1-T4).
- kernel mechanism of the conv/ssm asymmetry (PROVEN by code read of the read-only sibling repo): SSM
  prefill zeros fresh state in Python and fully overwrites `ssm_state[prefill_state_indices]`
  (`qwen_gdn_linear_attn.py:1513-1531`); conv prefill is only read-zeroed by the kernel
  (`causal_conv1d.py:184-191`, `has_initial_state=False` ⇒ `col0..3 = tl.zeros(...)`) and writes only the
  committed rows via `non_spec_state_indices_tensor`, leaving the spec rows stale-but-dead
  (`qwen_gdn_linear_attn.py:1344-1372`).

**Citation audit (2026-07-20, read-only — every code-line reference above verified against source):**
`_ssm_spec_row` `runtime/direct_model_runner.py:83`; conv/ssm allocation `:560-587` (conv
`(total_physical_slots=7, 6, 10240)`, ssm `7×(1+K)=28` rows — grounds the §3 dead-row mechanism);
`build_attention_metadata_batch` `:737` with `max_num_splits=1` `:931`/`:941` (prefill) vs decode
split-K `decode_fixed_max_num_splits` `:1607`; `reset_slot` `:2574` (docstring: "Does not zero the
underlying tensors … has_initial_state=False … is what makes reuse correct"); `_publish_committed_blocks`
`:4229` with swap/dedup `:4274-4276`; `restore_cached_prefix` `:4362`; chunked-loop publish `:3939`/`:3968`;
`_clear_persistent_cache` `benchmarks/prefix_cache_eviction_check.py:331` → `_reset_all` `:325` (clears
content index + GDN ckpt pool, not `kv_caches`); `_run_chunk_boundary_partial_share` `:558` with
`inv1_restore_state_matches_cold` `:633` (`l0_exact and stack_near_tie`) and `inv1_anchor_matches_cold` `:642`.

**INFERRED:** none material. The two prior open items (which kernel instruction; production impact of §3)
are now resolved — the mechanism is kernel-traced (above) and the production impact is proven **nil**
(decode output byte-identical with contamination present, `diag_rc10`).

**Explicitly RULED OUT (consistent with the investigation's prior exclusions, now with the real cause):**
- restore/addressing bug, continue-mechanism bug, stale block content, physical-slot aliasing of the
  measurement, run-to-run non-determinism, draft/MTP GDN bookkeeping — all still excluded; the
  target prefill is bit-exact correct. The "allocation-history sensitivity" was an artifact of which
  position the diagnostics sampled, not of the cold compute.
