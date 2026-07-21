# P3 — Persistent Content-Addressed Prefix Cache: Implementation Plan

Status: **PLAN (Architect, read-only pass).** Sequels `notes/prefix-cache-design.md`
(§5 "P3") and `notes/prefix-cache-implementation-log.md` (P0/P1/P2 DONE).
The executor follows this document sub-round by sub-round; each sub-round is
rollback-safe and ships behind a dedicated test that must catch a silent bug
BEFORE the next sub-round starts. Line numbers in the design doc's §9 are
stale (the file is ~4614 lines now) — everything below names touch-points by
**function name**, located against the current `runtime/direct_model_runner.py`.

---

## TL;DR

P3 builds the persistent content-addressed cache on top of P0/P1/P2's proven
substrate (`block_table`, `BlockPool` alloc/free/`reference`/refcount,
`mtp_prefill_fanout_batch`, `restore_gdn_state(..., allow_cross_slot=True)`).
It is split into **four** ordered sub-rounds (the design's "machinery, then
long-context perf" refined): the machinery is front-loaded so the single
riskiest correctness fact — **a persistent GDN checkpoint restore + referenced
attention blocks + suffix continue-prefill reproduces a cold prefill** (the
INV1 reduction, R1's target) — is proven in **P3.1** against a directly-driven
hit path, *before* eviction, decode-position populate, and ragged/graph
composition are layered on. Production wiring (the engine actually serving
hits) lands in **P3.3**; the ≥64K Pattern-B speedup proof lands in **P3.4**.

The whole persistent cache sits behind a NEW finer flag
`enable_persistent_prefix_cache` (default `False`, requires the existing
`enable_prefix_cache`/`enable_block_table`). **Rollback spine:** persistent
lookup returning `L=0` ⇒ the existing P2 fan-out-or-cold path runs
byte-for-byte; `enable_persistent_prefix_cache=False` ⇒ byte-for-byte P2;
`enable_prefix_cache=False` ⇒ byte-for-byte P1.

| Pri | Sub-round | Build (one line) | Invariants | Dedicated test | Rollback boundary |
|---|---|---|---|---|---|
| 1 | **P3.1** Persistent-cache hit equivalence | Chained hashing + `hash_to_block` index + persistent GDN checkpoint pool + populate-on-completion + the restore-and-continue hit path (`mtp_prefill_with_cache`) | INV1, INV3, INV6, INV7, INV4 (multi-round after hit), R1, R10 | `prefix_cache_persistent_hit_check.py` (cold-vs-hit near-tie + **GDN-layer-0 exact**, several L, 20+ decode rounds, NL+code; populate keeps blocks alive across `reset_slot`) | hit path only driven by the test; production prefill unchanged ⇒ byte-for-byte P2; `L=0` ⇒ P2 |
| 2 | **P3.2** Eviction, lockstep GDN eviction, full populate | Intrusive LRU free queue + `touch`/evict + GDN byte-budget pool eviction in lockstep + decode-position populate + chunk-boundary checkpoints + write-time attention dedup | INV2, INV3, INV9, R4, R5, R7, R8 | `prefix_cache_eviction_check.py` (evict→re-request clean cold recompute; both halves evicted together; admission-under-pressure no live reclaim; 8192-stride partial share; `A>0,G=0` dedup; checkpoint byte-budget; no-leak churn) | eviction only triggers when free queue is empty ⇒ no pressure ⇒ byte-for-byte P3.1 |
| 3 | **P3.3** Production integration + full hit gate | Unified `mtp_prefill_with_cache` = persistent-hit + P2-fan-out + cold, ragged/mixed hit-cold, mid-flight admission, CUDA-graph parity, engine call-site swap | INV5, INV8 (+re-confirms INV1/2/3/4 through the production path) | `prefix_cache_hit_check.py` (the cumulative load-bearing gate: INV1/2/3/5/8/9 + eviction + ≥64K Pattern-B perf hook) | `enable_persistent_prefix_cache=False` ⇒ byte-for-byte P2; `L=0` ⇒ P2 fan-out/cold |
| 4 | **P3.4** Long-context perf validation | Checkpoint-placement tuning knob + hashing-overhead measurement + TTFT cold-vs-warm harness at ≥64K | R8 (memory), R9 (overhead) — perf, not new correctness | `prefix_cache_longctx_perf_check.py` (≥64K Pattern-B turn-2+ TTFT vs cold; peak-memory watchdog; hashing overhead) | perf measurement only; no behavior change |

---

## Sub-round detail

### P3.1 — Persistent-cache hit equivalence (the load-bearing correctness proof)

**Goal.** Prove the INV1 reduction with a *persistent* (cross-`reset_slot`,
cross-round) GDN checkpoint and content-addressed attention blocks: a request
served by "restore checkpoint @ L + reference the `[0,L)` attention blocks +
continue-prefill `[L, prompt)`" produces the same committed tokens as a cold
prefill, with **GDN layer 0 byte-exact** as the addressing proof (R1). No
eviction, no decode-position populate, no chunk-boundary checkpoints yet —
populate happens only at a cached prefix's **completion boundary**. The hit
path is exercised ONLY by the dedicated test this round; the production
prefill entrypoint is untouched, so production behavior is byte-for-byte P2.

**Build** (all in `runtime/direct_model_runner.py` unless noted):

1. **Chained hashing (module level).**
   - `NONE_HASH: int` — process-global seed (derive once from a fixed salt +
     `PYTHONHASHSEED` so a run is reproducible but cross-run collisions are
     impossible; R7).
   - `def hash_block_tokens(parent_hash: int | None, token_ids: list[int], extra_keys: tuple) -> int`
     — full-width hash (use `hashlib.blake2b(digest_size=16)` → 128-bit int;
     vLLM's `hash_block_tokens` shape, adapt don't copy). `parent_hash or
     NONE_HASH`. **`extra_keys` MUST carry `self.kv_cache_dtype`** (fp8 vs
     nvfp4 KV must never collide) — the runner stores this once at `__init__`.
   - `@dataclass(frozen=True) class BlockHash: value: int; num_tokens: int`
     — `num_tokens = (i+1)*block_size` enables the cheap paranoid first-block
     token-count verify on hit (R7). `Block.block_hash` (already present,
     unused since P1) now stores this.

2. **`BlockPool` index + refcount-keep-alive (no LRU yet).**
   - New field `self.hash_to_block: dict[int, Block] = {}` (keyed by
     `BlockHash.value`).
   - `def cache_block(self, block_id: int, block_hash: BlockHash) -> None` —
     set `blocks[block_id].block_hash = block_hash`; `hash_to_block[value] =
     block`. Idempotent guard: if `value` already present, this is the
     write-time-dedup signal (see step 6) — do NOT overwrite.
   - `def get_cached_block(self, hash_value: int) -> Block | None` — dict probe.
   - `free()` is **unchanged** (decrement; re-queue tail at `ref_cnt==0`;
     **retain `block_hash`** so a freed-but-published block stays hit-able —
     this is exactly what keeps a cached prefix alive across `reset_slot`).
   - P3.1 does NOT add LRU middle-removal: a hit can only reference a block
     that is either `ref_cnt>0` (live) or in the free queue with its hash.
     Because P3.1 has **no eviction**, a hashed block in the free queue is
     never handed out by `allocate` only if we yank it on hit — so add the
     minimal `touch` now:
   - `def touch(self, block_ids: list[int]) -> None` — `ref_cnt += 1`; if the
     block was `ref_cnt==0` it is in the `deque` free queue, so **remove it
     from the deque** (`self._free_queue.remove(block_id)` — O(n) on a deque,
     acceptable for P3.1's small block counts; P3.2 replaces the deque with an
     intrusive O(1) list). Mirror of `reference()` but legal for `ref_cnt==0`
     (revive) — `reference()` stays the same-round-fork primitive (it rejects
     `ref_cnt==0`), `touch()` is the cache-hit primitive.

3. **Persistent GDN checkpoint pool** (the genuinely new buffer; R8-aware from
   day one). New method `_allocate_gdn_checkpoint_pool()` called once at
   `__init__` (fixed-address discipline, like `_allocate_gdn_snapshot_buffers`):
   - `self.gdn_ckpt_conv: dict[str, torch.Tensor]` / `self.gdn_ckpt_ssm: ...`
     shaped `(max_checkpoints, *per_layer_state_shape)` — a small pool of
     **persistent** full-stack checkpoint slots, SEPARATE from the live
     per-slot `gdn_snapshot_*` buffers (those keep their existing MTP role).
   - `max_checkpoints = max(1, gdn_checkpoint_byte_budget // per_checkpoint_bytes)`
     where `per_checkpoint_bytes ≈ 151 MB` (the measured ~604 MB/4-slot figure)
     and `gdn_checkpoint_byte_budget` is a new `__init__` kwarg (default
     `8 * 2**30`). At 8 GB ⇒ ~53 slots. **Allocate lazily-but-bounded or up
     front; either way never reallocate after init.**
   - `self.gdn_ckpt_meta: dict[int, dict]` — per pool-slot `{key, hash_value,
     num_tokens, lru_node, bytes}`; plus a free-slot stack and an LRU order
     (a plain `collections.OrderedDict` keyed by pool-slot is enough for P3.1;
     P3.2 hardens eviction).
   - `def materialize_gdn_checkpoint(self, slot: int, key: int, hash_value: int, num_tokens: int) -> None`
     — `torch._foreach_copy_` the 48-layer live `kv_caches[name]` state at
     `_physical_slot(slot)` INTO a free pool slot (read-only source); record
     meta. Tag the checkpoint with `hash_value` so a wrong-prefix restore is
     REJECTED, not used (R1's "checkpoint-hash tag").
   - `def checkpoint_view(self, key: int) -> dict[str, tuple[Tensor, Tensor]] | None`
     — returns a snapshot-shaped dict (`{name: (conv, ssm)}` + `__slot__` tag
     set to the SOURCE slot the checkpoint was materialized from) so the
     EXISTING `restore_gdn_state(dest_slot, view, allow_cross_slot=True)`
     consumes it with zero changes to that primitive (its foreach_copy +
     cross-slot path already proven in P2). This is the reuse the design
     mandates: P3 does not write a second restore.
   - `def evict_gdn_checkpoint(self, key: int) -> None` — free the pool slot
     (P3.1: only called by the test / by lockstep eviction stub; P3.2 wires
     the byte-budget LRU).

4. **Per-slot hash-chain state** in `DirectModelRunner.__init__` (reset in
   `reset_slot`):
   - `self.slot_block_hashes: list[list[BlockHash]] = [[] for _ in range(num_slots)]`
     — growing chained hash per full block (`slot_block_hashes[s][i]` = hash
     of block `i`, depends on all tokens `0..(i+1)*block_size`).
   - `self.slot_published_blocks: list[int] = [0] * num_slots` — count of this
     slot's blocks already published to the index (write cursor).

5. **Write path — populate-on-completion (attention) + completion checkpoint (GDN).**
   - `def _publish_committed_blocks(self, slot: int, token_ids: list[int], committed_len: int) -> int`
     — publish full blocks `[slot_published_blocks[s], committed_len //
     block_size)`: for each new full block `i`, compute `h_i =
     hash_block_tokens(h_{i-1}, token_ids[i*16:(i+1)*16], extra_keys)`,
     append to `slot_block_hashes[s]`, and `cache_block(block_table[s][i],
     BlockHash(h_i, (i+1)*16))` (subject to step-6 dedup). Returns the deepest
     published boundary. **Only committed tokens are hashed/published** — the
     partial tail and any draft/verify tokens beyond commit are never touched
     (INV4; mirrors vLLM `kv_cache_manager.py:456-465`).
   - Wire it into the **cold prefill completion** point: inside
     `mtp_prefill_batch` (uniform + ragged paths) and `mtp_prefill` after the
     final chunk commits, when `enable_persistent_prefix_cache` is on, call
     `_publish_committed_blocks(slot, prompt, prompt_len)` then
     `materialize_gdn_checkpoint(slot, key=block_table[slot][G//16-1],
     hash_value=slot_block_hashes[slot][G//16-1].value, num_tokens=G)` where
     `G = (prompt_len // block_size) * block_size` capped at `prompt_len - 1`
     block-aligned (the **completion boundary**). In P3.1 the completion
     boundary is the ONLY checkpoint, so `G` = block-aligned `prompt_len-1`.
   - **Draft layer needs no separate publish**: it is the 17th member of the
     attention group (§3.1) — the same `block_table[slot]` blocks hold its KV,
     so publishing the attention blocks publishes the draft KV.

6. **Write-time attention dedup (minimal, collision-safe).** Inside
   `_publish_committed_blocks`, if `get_cached_block(h_i)` already returns a
   block `B'` (the `A>0, G=0` compute-miss case, or an incidental duplicate):
   verify `B'.block_hash.num_tokens == (i+1)*16` (cheap paranoid check, R7),
   then **do not publish the fresh block** — swap `block_table[slot][i]` to
   `B'.block_id`, `touch([B'.block_id])`, and `free([fresh_block_id])` (the
   fresh block had identical content written this forward; freeing reclaims
   the memory — this is the §3.8 memory reclamation for the compute-miss
   case). If no hit, publish fresh. (Full alloc-time dedup is NOT in P3.1;
   this publish-time swap is the minimal correct version.)

7. **Read path — reconciliation + restore-and-continue hit.**
   - `def _compute_prompt_block_hashes(self, token_ids: list[int], max_tokens: int) -> list[BlockHash]`
     — chained hashes of full blocks, **capped at `max_tokens = len(T) - 1`**
     (the last token is always recomputed for logits; vLLM
     `kv_cache_manager.py:225-231`). Pure CPU, O(blocks).
   - `def reconcile_prefix_hit(self, token_ids: list[int]) -> int` — the §3.4
     rule, specialized to two groups (no iterative solver):
     1. `hashes = _compute_prompt_block_hashes(T, len(T)-1)`.
     2. **Attention match `A`**: walk `hashes` left-to-right, stop at first
        miss in `hash_to_block`; `A = (matched_blocks) * block_size`.
        (Downward-closed: any prefix of a hit is a hit.)
     3. **GDN boundary `G`**: the largest checkpoint boundary `Lc ≤ A` with a
        GDN checkpoint under the SAME chained hash at `Lc` (probe
        `gdn_ckpt_meta` by `hashes[Lc//16 - 1].value`). In P3.1 checkpoints
        exist only at completion boundaries, so `G` is that boundary or `0`.
     4. **`L = G`** (always `≤ A`, always block-aligned). Return `L`.
   - `def restore_cached_prefix(self, slot: int, token_ids: list[int], L: int) -> None`
     — the §3.5 steps 1–4 for a fresh `slot`:
     1. `block_table[slot] = [b.block_id for b in matched_blocks[:L//16]]`;
        `touch(those ids)` (refcount + revive-from-free-queue; **reserve-
        and-touch BEFORE any forward** — R4/INV9).
     2. `restore_gdn_state(slot, checkpoint_view(key_at_L), allow_cross_slot=True)`;
        `slot_gdn_initialized[slot] = True`.
     3. `slot_draft_sync_len[slot] = L` (draft KV `[0,L)` already referenced
        in step 1).
     4. `slot_kv_len[slot] = L`; `slot_num_accepted_tokens[slot] = 1`;
        seed `slot_block_hashes[slot] = hashes[:L//16]`;
        `slot_published_blocks[slot] = L//16`.
   - `def mtp_prefill_with_cache(self, slots, prompts_per_slot, chunk_size=None) -> dict[int, dict]`
     — the new entrypoint (test-driven in P3.1; production-wired in P3.3).
     P3.1 body: for each slot compute `L = reconcile_prefix_hit(prompt)`;
     **hit slots** (`L>0`) → `restore_cached_prefix` then continue-prefill the
     suffix `[L, prompt_len)` via the EXACT validated continuation the P2
     fan-out sibling path uses (`_forward_batch([s],[suffix],[L], qo_len=
     suffix_len, commit=True, is_decode=False, logits_last_position_only=True)`
     + `_mtp_sync_and_propose_batch([s],[prompt[L+1:]+[anchor]], hidden,[L],
     num_new_tokens=suffix_len, k=K)`); **cold slots** (`L==0`) → delegate to
     `mtp_prefill_batch` (which now also populates on completion). Returns the
     same `{slot: {"anchor", "draft_tokens"}}` shape.
   - **R1 addressing proof hook:** `restore_cached_prefix` asserts the
     checkpoint's `hash_value == slot_block_hashes[slot][L//16-1].value`
     before restoring (a wrong-prefix checkpoint is rejected, not used).

**Invariants verified:** INV1 (cold-vs-hit equivalence), INV3 (attention/GDN
agree at `L=G≤A`), INV6 (publish only full blocks; private tail), INV7
(reserved block 0 — index never touches it), INV4 (multi-round MTP after a
hit; `slot_num_accepted_tokens` bootstrap = 1), R1 (GDN-layer-0 exact +
checkpoint-hash tag), R10 (`reset_slot` frees the slot's refs; published
blocks survive at `ref_cnt==0`).

**Dedicated test slice — `benchmarks/prefix_cache_persistent_hit_check.py`**
(style of `prefix_cache_fanout_check.py`: pure-Python part + real-GPU part,
near-tie `NEAR_TIE_LOGIT_MARGIN = 2.0` per R6, NOT bytewise — EXCEPT the
GDN-layer-0 addressing check which is exact per R1):
- Pure Python: `hash_chain_determinism` (same tokens ⇒ same chain; one-token
  divergence ⇒ every hash from divergence on differs; `extra_keys` dtype
  change ⇒ different hash); `index_keepalive` (publish → `reset_slot` →
  `ref_cnt==0` but `hash_to_block` still resolves; `touch` revives from the
  free queue; `free` of a published block retains its hash).
- Real GPU (one runner, `enable_block_table=True`, `enable_prefix_cache=True`,
  `enable_persistent_prefix_cache=True`, MTP K=3):
  - `inv1_cold_vs_hit` — cold-prefill prompt P in slot 0 (populates), `reset_slot(0)`,
    re-request P in slot 1 → hit at `L`; assert hit anchor + 20+ MTP decode
    rounds match an independent cold reference (near-tie). Repeat at several
    `L` (a 100-token prompt hitting 96; a 5000-token prompt hitting the
    block-aligned `len-1`; a partial-vs-full mix), and for a natural-language
    AND a code prompt.
  - `r1_gdn_layer0_exact` — at the hit boundary `L`, the restored GDN **layer
    0** `(conv_state, ssm_state)` is **bytewise identical** to a cold prefill
    of `[0,L)`'s layer-0 state (the addressing proof; noise cannot hide a
    wrong-block read). Then assert the full 48-layer stack is near-tie.
  - `inv3_mismatched_prefix` — request sharing only the first `Lc` tokens
    reuses EXACTLY `Lc` (not more); a request whose attention would match
    deeper than the only checkpoint reuses only up to that checkpoint
    (`L=G≤A`).
  - `inv4_multiround_after_hit` — 20+ decode rounds after a hit stay
    oracle-aligned (`draft_sync_len == kv_len` + near-tie next-token);
    `draft_acceptance_rate` sanity vs cold baseline.
  - `no_block_leak` — after hit + decode + `reset_slot`, free count returns to
    baseline minus the still-cached (ref_cnt==0, hashed) blocks; every live
    `ref_cnt==0`; `block_table[slot]==[]`.

**Rollback-safe boundary.** `mtp_prefill_with_cache` is built but NOT called by
any production path in P3.1 (the engine and `mtp_w1s_our_runtime_perf` still
call `mtp_prefill_batch`/`mtp_prefill_fanout_batch`); populate-on-completion
runs only under `enable_persistent_prefix_cache=True` (default `False`) and
only ADDS index/checkpoint writes — it never changes the tokens a cold
prefill produces. So: flag off ⇒ byte-for-byte P2; flag on but no hit ever
served in production ⇒ byte-for-byte P2 with extra (unconsulted) index state.
The hit path is proven solely by the dedicated test driving
`mtp_prefill_with_cache` directly.

**Risk mitigations (P3.1):** R1 (GDN-layer-0 exact addressing proof +
checkpoint-hash tag reject + reused `restore_gdn_state` guards); R7 (128-bit
hash, `extra_keys` dtype, paranoid num_tokens verify on dedup); R10 (publish
keeps blocks alive at `ref_cnt==0`; `reset_slot` only drops the slot's own
refs); R8 (checkpoint pool sized by byte budget at init, not unbounded).

---

### P3.2 — Eviction, lockstep GDN eviction, and the full populate path

**Goal.** Make the cache survive real memory pressure and capture all three
populate boundaries, without changing any token when no pressure exists. This
is the production-hardening round: LRU eviction with O(1) middle-removal,
GDN checkpoints evicted in lockstep with their keyed attention blocks, decode-
position populate, and chunk-boundary checkpoints (the Fork-2 coarse policy).

**Build:**

1. **Intrusive LRU free queue (replace the `deque`).** Add `prev_free`/
   `next_free` to `Block` (the design's §3.2 fields). New
   `class FreeBlockQueue` with sentinel head/tail, `append`/`appendleft`/
   `popleft`/`remove(block)` all O(1) (vLLM `FreeKVCacheBlockQueue` shape,
   adapt). `BlockPool._free_queue` becomes this. `touch()` now does
   `free_queue.remove(block)` in O(1) when reviving a `ref_cnt==0` block
   (replacing P3.1's O(n) deque remove).

2. **Eviction in `allocate`.** Rename/extend the allocator so that when the
   free queue has fewer than `n` blocks it **evicts** from the front:
   `def _evict_one(self) -> None` — `block = free_queue.popleft()`; if
   `block.block_hash is not None`, `del hash_to_block[value]` AND
   `evict_gdn_checkpoint(key=block_id)` (**lockstep** — the GDN checkpoint
   keyed by this attention tail block goes with it, INV3/R5); then the block
   is reusable (`block_hash=None`, `ref_cnt` stays 0). `allocate(n)` evicts
   until it can satisfy `n`, raising only when the queue is empty AND every
   block is `ref_cnt>0` (true exhaustion). Blocks freed WITHOUT a hash are
   `appendleft`-ed (evicted first); hashed blocks `append`-ed (LRU tail) —
   matching vLLM `free_blocks`. `reset_slot` frees in **reverse logical
   order** so deep-prefix tail blocks die first (keeps shallow shared prefixes
   longer; §3.2).

3. **GDN checkpoint byte-budget LRU.** `gdn_ckpt_meta`'s order becomes a real
   LRU (`OrderedDict.move_to_end` on every hit/restore). On
   `materialize_gdn_checkpoint`, if `total_bytes + per_checkpoint_bytes >
   gdn_checkpoint_byte_budget`, evict LRU checkpoints until it fits
   (`evict_gdn_checkpoint` frees the pool slot AND drops the co-keyed
   attention block's hash if that block is `ref_cnt==0` — lockstep both
   directions). `checkpoint_view`/restore `move_to_end` (MRU).

4. **Decode-position populate.** New
   `def publish_committed_decode_blocks(self, slot: int, committed_token_ids: list[int]) -> None`
   — after `mtp_verify_and_commit`/`mtp_verify_and_commit_batch` advance
   `slot_kv_len` by the REAL committed length, publish any newly-FULL committed
   blocks (`[slot_published_blocks[s], slot_kv_len[s]//16)`), extending
   `slot_block_hashes[s]` incrementally (parent = last published hash). The
   runner must keep the per-slot committed token ids available for hashing —
   add `self.slot_committed_tokens: list[list[int]]` (append on commit; reset
   in `reset_slot`). **Only committed tokens** (INV4): rejected drafts never
   reach this path because `slot_kv_len` only advances by the accepted count.
   Wire into both verify-commit methods under the flag.

5. **Chunk-boundary GDN checkpoints.** Inside `mtp_prefill_batch`'s chunked
   loop (the `while chunk_start < prompt_len` body), at each non-final
   `chunk_end` that is block-aligned (every `chunk_size` boundary; default
   `_DEFAULT_PREFILL_CHUNK_SIZE = 8192`), when the flag is on:
   `_publish_committed_blocks(slot, prompt, chunk_end)` then
   `materialize_gdn_checkpoint(slot, key=block_table[slot][chunk_end//16-1],
   hash_value=slot_block_hashes[slot][chunk_end//16-1].value, num_tokens=
   chunk_end)`. These boundaries are *free* (the GDN forward already
   materializes state there) and give 8192-granular cross-request partial
   sharing (Fork 2). The completion-boundary checkpoint (P3.1) remains.

6. **Alloc-time dedup upgrade (optional, if cheap):** move the P3.1
   publish-time swap earlier — in `_ensure_blocks`, compute the next block's
   hash and `touch` a cached match instead of `allocate`-ing fresh. Only do
   this if it does not perturb the `_ensure_blocks` call sites' growth
   targets; otherwise keep publish-time dedup (already correct). Flag as a
   contained optimization, not a requirement.

**Invariants verified:** INV2 (no stale/cross reads — eviction drops the hash
BEFORE re-hand-out; `ref_cnt>0` never in free queue), INV3 (lockstep eviction
— both halves go together), INV9 (eviction never races a live ref), R4
(reserve-and-touch-before-forward), R5 (`L=G≤A` + lockstep), R7 (paranoid
verify on every dedup/hit), R8 (byte-budget cap on checkpoints).

**Dedicated test slice — `benchmarks/prefix_cache_eviction_check.py`:**
- Pure Python: `lru_middle_removal` (intrusive queue O(1) remove of an
  arbitrary node; popleft order = oldest-freed-first); `evict_drops_hash`
  (a hashed block popped for reuse loses its `hash_to_block` entry);
  `lockstep_eviction` (evicting an attention tail block drops the co-keyed
  GDN checkpoint, and vice-versa — assert BOTH halves gone together, INV3);
  `refcnt_never_evicted` (a `ref_cnt>0` block is never popped, even when it is
  the LRU front — INV9); `byte_budget` (materialize past the budget ⇒ LRU
  checkpoint evicted, pool bytes stay ≤ budget).
- Real GPU:
  - `evict_then_recompute` — populate P, force P's blocks to be evicted
    (sponge the pool with other prefills until P's hash is gone), re-request P
    ⇒ clean COLD recompute (reconcile returns `L=0`), tokens match the
    original cold reference (INV1/INV3 — no half-evicted ghost hit).
  - `admission_under_pressure` — fill the pool so a new admission must evict
    while other slots are ACTIVE; assert no active slot's block is reclaimed
    (every active `ref_cnt>0` survives) and all slots' committed tokens stay
    correct (INV9). Reuse the `_fragment_pool_via_churn` sponge technique from
    `cudagraph_decode_regression.py`.
  - `chunk_boundary_partial_share` — request A = 20000-token prompt (cold;
    checkpoints at 8192/16384 + completion); request B shares A's first 18000
    tokens then diverges ⇒ B hits at `L=16384` (deepest chunk boundary ≤
    attention match), reuses exactly that (INV3), B's `[16384, B_len)`
    continue-prefill matches a cold reference (INV1).
  - `a_gt_0_g_eq_0_dedup` — construct the compute-miss case (attention blocks
    cached but no GDN checkpoint at a needed boundary): reconcile returns
    `L=0`, the fresh recompute's write-time dedup reclaims the duplicate
    attention blocks (free count recovers; no leak), tokens correct.
  - `no_leak_churn` — many admit/finish cycles under pressure; pool free count
    returns to baseline; `cuda_allocated_mib` flat (reuse the D3/sustained
    memory-flatness watchdog).

**Rollback-safe boundary.** Eviction code paths only execute when `allocate`
cannot satisfy from the free queue — with a generously-sized pool and no
pressure (the default test/headline shapes), `allocate` never evicts, so
behavior is byte-for-byte P3.1. Decode-position populate and chunk-boundary
checkpoints only ADD index/checkpoint state under the flag; they never change
produced tokens. `enable_persistent_prefix_cache=False` ⇒ byte-for-byte P2.

**Risk mitigations (P3.2):** R4 (single-event-loop serialization +
reserve-and-touch-before-forward; INV9 test); R5 (lockstep eviction, both
directions; INV3 test); R8 (byte-budget LRU + checkpoint only at
chunk/completion boundaries; byte-budget test + P3.4 memory watchdog); R7
(paranoid verify on dedup/hit).

---

### P3.3 — Production integration + the full hit gate

**Goal.** Wire the persistent cache into the real prefill path so the engine
and benchmarks actually serve hits; compose persistent-hit + P2-fan-out +
cold into one entrypoint; prove ragged/mixed hit-cold admission, mid-flight
admission of a hit slot, and CUDA-graph parity with hit-populated
non-contiguous tables. This round completes the cumulative load-bearing gate
`prefix_cache_hit_check.py` (the design's §5-P3 test).

**Build:**

1. **Unify the entrypoint.** Make `mtp_prefill_with_cache(slots,
   prompts_per_slot, chunk_size=None)` the single production prefill:
   - Per slot, `L = reconcile_prefix_hit(prompt)` (flag on).
   - **Hit set** (`L>0`): `restore_cached_prefix` each, then continue-prefill
     the (ragged) suffixes in ONE batched `_forward_batch` +
     `_mtp_sync_and_propose_batch` — exactly the P2 sibling-suffix batching,
     generalized to per-slot `L` (ragged `kv_lengths=[L_s]`, `qo_len=
     [suffix_len_s]`).
   - **Cold set** (`L==0`): hand to the EXISTING `mtp_prefill_fanout_batch`
     (which itself does P2 same-round fork detection among the cold slots,
     then falls back to `mtp_prefill_batch`). P2 is preserved as the explicit
     same-round fast path (Cross-cutting decision 1) — it runs ONLY over cold
     slots, so a persistent hit always wins over a same-round fork.
   - Merge the two result dicts. Return the unified `{slot: {anchor,
     draft_tokens}}`.
   - **Ragged-suffix-chunking guard (INV8 caveat):** if a hit slot's suffix
     individually exceeds `chunk_size` AND the hit batch is ragged, that
     inherits `mtp_prefill_batch`'s existing ragged+`chunk_size`
     `NotImplementedError`. v1 scope: such a slot **falls back to cold
     prefill** (`L=0` path) rather than mis-chunking (real hit suffixes are
     short; document this). Do NOT lift the ragged+chunk limit in P3.

2. **Engine call-site swap (the P4-deferred one-liner, now in scope for the
   runner-driving benchmarks; server wiring is still P4).** In
   `benchmarks/mtp_w1s_our_runtime_perf.py`'s `_run_batch_batched` and the
   sustained/hit checks, call `mtp_prefill_with_cache` instead of
   `mtp_prefill_batch` when the flag is on. (`server/engine.py:_step`'s
   `mtp_prefill_batch(new_slots, new_prompts)` → `mtp_prefill_with_cache(...)`
   is the P4 server-integration swap; P3 proves the runner mechanism, not the
   HTTP layer — keep the server diff out of P3 to hold the rollback boundary.)

3. **CUDA-graph parity (INV5).** No graph code change should be needed —
   `CapturedBatchDecodeGraph._fill_buffers` / `CapturedMTPDraftStepGraph
   ._fill_buffers` already source page ids from `runner.block_table[slot]`
   under `enable_block_table` and refill every replay (no baked-in ids;
   prefill, where hits happen, is eager). VERIFY: a hit slot's non-contiguous
   table (shared `[0,L)` ids + fresh private tail) replays correctly through
   the captured decode/verify graph. If parity EVER fails, the documented
   fallback is eager for hit slots (the graph path already has an eager
   fallback) — but the expectation (INV5) is that it holds.

**Invariants verified:** INV5 (eager-vs-graph parity with hit-populated
non-contiguous tables), INV8 (mixed hit/cold ragged + mid-flight admission of
a hit slot alongside long-running slots), and re-confirms INV1/INV2/INV3/INV4
THROUGH the production entrypoint (not just the directly-driven P3.1 path).

**Dedicated test slice — `benchmarks/prefix_cache_hit_check.py` (the
cumulative load-bearing gate; absorbs/extends the P3.1 and P3.2 checks):**
- **INV1** cold-vs-hit near-tie at several `L` (partial + full), 20+ decode
  rounds, natural-language + code prompt — now via `mtp_prefill_with_cache`.
- **INV2** signal-probe crosstalk with cache hits interleaved across all 4
  slots (per-slot marker tokens; zero leakage) — the fan-out check's
  `inv2_signal_probe` methodology, generalized to persistent hits.
- **INV3** mismatched/mixed-length prefixes reuse exactly the right depth
  (carry over P3.1/P3.2 cases).
- **INV5** eager-vs-graph parity with hit-populated non-contiguous tables —
  extend `cudagraph_eager_parity_check.py` with a hit-populated table; re-run
  `cudagraph_decode_regression --enable-block-table` and
  `mtp_verify_cudagraph_check --enable-block-table` with hit tables.
- **INV8** mixed hit/cold ragged admission + mid-flight admission of a hit
  slot alongside long-running slots; extend `mtp_ragged_prefill_check` /
  `mtp_async_arrival_check` patterns.
- **INV9** admission-under-pool-pressure forcing eviction with no live block
  reclaimed (carry over P3.2).
- **eviction correctness** — evict → re-request → clean cold recompute (carry
  over P3.2).
- **≥64K Pattern-B perf hook** — a real long-context re-run showing speedup vs
  the cold number (fully characterized in P3.4; the gate asserts correctness
  here, perf in P3.4).
- Plus full `mtp_*_check` fast battery + `cudagraph_eager_parity_check` +
  4K/c=4 headline bit-identical (all at the flags' default `False` — the
  "P3 changes nothing when off" half), AND a flag-on re-run of the same
  battery proving the flag-on path is also regression-free.

**Rollback-safe boundary.** `enable_persistent_prefix_cache=False` ⇒
`mtp_prefill_with_cache` delegates straight to `mtp_prefill_fanout_batch` ⇒
byte-for-byte P2. Flag on but a slot's lookup returns `L=0` ⇒ that slot takes
the P2 fan-out/cold path ⇒ byte-for-byte P2 for it. The engine call-site swap
is gated on the flag, so the server/benchmarks are unchanged when off.

**Risk mitigations (P3.3):** R3 (CUDA-graph assumptions under variable block
reuse — INV5 parity re-runs with hit tables; eager fallback documented); R4
(mid-flight admission still single-event-loop serialized; reserve-and-touch-
before-forward in `restore_cached_prefix`); R9 (the flag-on battery re-run
proves the cache does not regress the no-hit path).

---

### P3.4 — Long-context perf validation (≥64K Pattern-B)

**Goal.** Prove the actual user-facing win: a ≥64K Pattern-B (sequential
per-conversation growth) re-run shows turn-2+ TTFT approaching the 15.4×
exact-repeat ceiling vs the cold number, with bounded memory and acceptable
hashing overhead. This is perf validation, not new correctness — but it is the
reason the user mandated prefix caching, so it is a first-class sub-round.

**Build:**

1. **`benchmarks/prefix_cache_longctx_perf_check.py`** — Pattern-B harness:
   - Turn 1: cold-prefill a ≥64K prompt (`--fixture ctx64k`, `ctx128k`, …;
     `prompt_len=65536/131072/…`) via `mtp_prefill_with_cache` (flag on),
     measure TTFT (the perf bench already records `ttft_s`/`ttft_mean_ms`/
     `ttft_p99_ms`).
   - Turn 2..N: re-send the SAME prompt + a short appended turn (Pattern-B
     growth); each is a persistent hit at the prior completion boundary ⇒
     measure warm TTFT.
   - Report `cold_ttft`, `warm_ttft`, `speedup = cold/warm`, `hit_L`,
     `accepted_tok/s`, and `cuda_allocated_mib` first→last (climbing-trend
     watchdog). Target: turn-2+ TTFT speedup approaches the 15.4× ceiling
     (the §25.4 native-vLLM exact-repeat number); accept a documented fraction
     if the completion-boundary-only reuse leaves a short recompute tail.
   - Reuse `mtp_w1s_our_runtime_perf --batched --chunk-size 8192 --fixture
     ctx64k` as the cold baseline source; the check wraps both runs.

2. **Hashing-overhead measurement (R9).** Time `_compute_prompt_block_hashes`
   + `reconcile_prefix_hit` on the no-hit path at 4K/64K/256K; assert it is
   negligible vs prefill wall time (it is O(blocks) CPU dict probes on hashes
   computed once). If non-negligible at 256K, document and (only then)
   consider incremental hashing across chunks.

3. **Checkpoint-placement knob.** Expose `gdn_checkpoint_byte_budget` and the
   chunk stride as runner/bench knobs; sweep at 64K/128K to confirm the
   default (8192 stride, 8 GB budget) gives the best TTFT/memory tradeoff.
   No default change without data.

**Invariants verified:** R8 (peak memory bounded — checkpoint pool + KV under
the byte budget; flat `cuda_allocated_mib` across turns), R9 (hashing/lookup
overhead negligible; cache does not hurt the no-hit path).

**Dedicated test slice:** the perf check above IS the slice — it fails if
turn-2+ TTFT does not improve materially vs cold, if memory climbs, or if
hashing overhead exceeds a small fraction of prefill time.

**Rollback-safe boundary.** Pure measurement; no production behavior change.
The knobs it sweeps already exist from P3.1/P3.2.

**Risk mitigations (P3.4):** R8 (byte-budget cap validated under real 256K;
memory watchdog); R9 (overhead measured end-to-end, not assumed; no-hit path
re-checked).

---

## Cross-cutting decisions (decisive)

### 1. P2 fan-out vs persistent populate+hit — **keep P2 as the explicit same-round fast path** (confirms Fork 3)

Reasoning, now concrete: (a) **correctness surface is already proven** — P2's
`mtp_prefill_fanout_batch` is verified against independent cold references;
deleting it to "subsume" would re-open a validated surface for zero token-
quality gain. (b) **It avoids a same-tick race** — within ONE admission, the
persistent path would need "leader populates the index, THEN siblings look
up"; the single-event-loop makes this serializable, but the fan-out's direct
`_common_prefix_len` comparison sidesteps the ordering entirely and avoids
hashing a (possibly 100K+) shared prefix twice. (c) **Clean composition** —
in `mtp_prefill_with_cache`, persistent lookup runs first per slot; **only
`L=0` (cold) slots flow into `mtp_prefill_fanout_batch`**, so a persistent hit
always wins, and the fan-out optimizes the remaining cold siblings. The ONE
additive change to P2: when `enable_persistent_prefix_cache` is on, the
fan-out leader's prefill also runs `_publish_committed_blocks` + the
completion `materialize_gdn_checkpoint`, so a same-round fork ALSO populates
the persistent index for FUTURE rounds (purely additive; same-round behavior
unchanged). This is the only place P2's code is touched in P3.

### 2. Reconciliation algorithm — **`L = G ≤ A`, direct (no iterative solver)**

Adapted to this runtime's two cache groups: the attention group is
**downward-closed** (any prefix of a hit is a hit, so a single left-to-right
walk stopping at the first miss gives `A`), and the GDN group is **snapshot-
constrained** (one checkpoint, the deepest boundary `≤ A` under the same
chained hash, gives `G`). vLLM's `HybridKVCacheCoordinator` fixed-point
(`kv_cache_coordinator.py:631-742`) exists because it must handle N groups
with eagle-drop interactions that can shrink the length and require restart;
with exactly two groups where `G ≤ A` always, the fixed point collapses to
`L = G` in one pass. The `A>0, G=0` case (attention cached, no GDN checkpoint
at a usable boundary) is a **compute miss** (`L=0`, prefill fresh) — exactly
vLLM v1's rule — and write-time dedup (P3.1 step 6 / P3.2) reclaims the
recomputed attention blocks' memory. Checkpoints exist only at block-aligned
chunk boundaries (8192 stride) + completion boundaries, so `L` is always a
block boundary by construction (no partial-block sharing, no COW — §3.7/§3.8).

### 3. Where hashing happens + decode-position population — **prefill/commit, only-verified-tokens-cached**

Hashing is CPU-side, chained per full block: `h_i = H(h_{i-1},
token_ids[i*16:(i+1)*16], extra_keys)`, `extra_keys=(kv_cache_dtype,)`,
`NONE_HASH` seed. The per-slot chain (`slot_block_hashes`) grows
incrementally. **Prefill:** publish full blocks at the completion boundary
(P3.1) and at each chunk boundary (P3.2). **Decode:** publish newly-full
blocks after each verify-commit advances `slot_kv_len` by the ACCEPTED count
(P3.2) — rejected drafts never advance `slot_kv_len`, so they are never hashed
or published (INV4; mirrors vLLM `kv_cache_manager.py:456-465`). **Hit cap:**
lookup hashes are capped at `len(T)-1` so the last token is always recomputed
for logits (vLLM `kv_cache_manager.py:225-231`). The draft layer is the 17th
attention-group member, so publishing the attention blocks publishes draft KV
too — no separate draft hashing.

### 4. GDN checkpoint granularity, reuse, and pool sizing — **coarse (Fork 2), reuse `restore_gdn_state`, byte-budget pool**

Confirmed coarse: checkpoint at `chunk_size` boundaries (default
`_DEFAULT_PREFILL_CHUNK_SIZE = 8192`) + each cached prefix's completion
boundary. Dense per-block is unaffordable (~151 MB × 16K at 256K). **Reuse:**
`materialize_gdn_checkpoint` foreach_copies the 48-layer live state into a
persistent pool slot; `checkpoint_view(key)` returns a snapshot-shaped dict
that the EXISTING `restore_gdn_state(dest, view, allow_cross_slot=True)`
consumes unchanged (its cross-slot path + foreach_copy proven in P2) — P3
writes no second restore. **Pool sizing:** a fixed-address pool of
`byte_budget // ~151MB` slots (default 8 GB ⇒ ~53), allocated once at init;
LRU eviction by byte budget (P3.2), in lockstep with the keyed attention tail
block (both directions). The live per-slot `gdn_snapshot_*` buffers keep their
existing MTP role, untouched.

### 5. CUDA-graph interaction (INV5) — **hit slots use the captured graph; no eager fallback expected**

Prefill (where hits happen) is eager, never graph-captured, so the cache
machinery and the graph path are disjoint. A hit slot's decode/verify runs
through the SAME captured graph as a cold slot: `CapturedBatchDecodeGraph
._fill_buffers` / `CapturedMTPDraftStepGraph._fill_buffers` already source
`kv_page_indices`/`slot_mapping` from `runner.block_table[slot]` under
`enable_block_table`, refilled EVERY replay (no physical block id is baked
into the capture; P1 already proved non-contiguous tables replay correctly).
A hit table (shared `[0,L)` ids + fresh private tail) is just another
non-contiguous table. **Decision: hit slots use the captured graph from
P3.3.** P3.3's INV5 test re-runs `cudagraph_eager_parity_check` /
`cudagraph_decode_regression --enable-block-table` /
`mtp_verify_cudagraph_check --enable-block-table` with hit-populated tables;
the documented eager-for-hit-slots fallback exists ONLY if parity fails (it is
not expected to).

### 6. Flag surface + rollback spine

- `enable_block_table` (P0/P1, exists) → `enable_prefix_cache` (P2 fan-out,
  exists) → **`enable_persistent_prefix_cache` (NEW, P3, default `False`,
  requires `enable_prefix_cache=True`; constructor raises on the
  misconfiguration, matching the existing P2 guard).**
- Rollback spine: `enable_persistent_prefix_cache=False` ⇒ byte-for-byte P2;
  `enable_prefix_cache=False` ⇒ byte-for-byte P1; persistent lookup `L=0` ⇒
  the P2 fan-out/cold path runs byte-for-byte. Populate (index/checkpoint
  writes) under the flag never changes produced tokens, so a flag-on-no-hit
  run is token-identical to P2.
- Tuning knobs (P3.4): `gdn_checkpoint_byte_budget` (default `8*2**30`),
  chunk stride (existing `_DEFAULT_PREFILL_CHUNK_SIZE = 8192`).

---

## Risk-mitigation matrix

| # | Risk | Sub-round(s) | Mitigation (concrete) | Catching test |
|---|---|---|---|---|
| R1 | GDN state corruption on restore | **P3.1** (primary) | Reuse proven `restore_gdn_state(allow_cross_slot=True)`; **GDN-layer-0 bytewise-exact** as the addressing proof before trusting the 48-layer stack; checkpoint-hash tag asserted in `restore_cached_prefix` so a wrong-prefix checkpoint is REJECTED not used; keep the generation/slot/consumed guards | `prefix_cache_persistent_hit_check.py::r1_gdn_layer0_exact` + `inv1_cold_vs_hit` |
| R2 | MTP draft desync after a hit | P3.1, P3.3 | Draft layer restored as part of the attention group (same `block_table`); `slot_draft_sync_len=L`; `slot_num_accepted_tokens=1` bootstrap; suffix draft step-0 sync over `[L,prompt)` | `inv4_multiround_after_hit` (oracle-aligned per step + acceptance-rate sanity vs cold) |
| R3 | CUDA-graph assumptions under variable reuse | P3.3 | `_fill_buffers` already refills from `block_table` every replay (no baked ids); P1 proved non-contiguous replay; eager-for-hit fallback documented | INV5 parity re-runs with hit tables (`cudagraph_eager_parity_check`, `*_cudagraph_* --enable-block-table`) |
| R4 | Eviction race under concurrent admission | P3.2, P3.3 | Single-event-loop serialization; **reserve-and-touch BEFORE any forward** in `restore_cached_prefix`; free only in `reset_slot`/`_finish_request` | `prefix_cache_eviction_check.py::admission_under_pressure` (INV9) |
| R5 | Attention/GDN disagree on cached length | P3.1, P3.2 | `L=G≤A` by construction; GDN checkpoint keyed by the SAME chained hash; **lockstep eviction both directions** | `inv3_mismatched_prefix`, `lockstep_eviction`, `evict_then_recompute` |
| R6 | fp8 non-determinism masks/mimics a bug | P3.1+ | `NEAR_TIE_LOGIT_MARGIN=2.0` + signal-probe as the INV1 bar (NOT bytewise), BUT GDN-layer-0 EXACT as the addressing proof so noise can't hide a wrong-block read | near-tie methodology in every GPU check |
| R7 | Hash collision → wrong prefix served | P3.1, P3.2 | Full-width 128-bit hash; `extra_keys=(kv_cache_dtype,)`; `NONE_HASH` seed; paranoid `num_tokens` verify on every dedup/hit (optional first-block token verify in paranoid mode) | `hash_chain_determinism`; collision probability documented as negligible at ≤4-slot traffic |
| R8 | GDN checkpoint memory blow-up | P3.1 (cap), P3.2 (budget LRU), P3.4 (validate) | Byte-budget pool (`max_checkpoints = budget // ~151MB`); checkpoint only at chunk + completion boundaries; lockstep eviction; fixed-address (no fragmentation) | `byte_budget` unit test; P3.4 peak-memory climbing-trend watchdog at 64K–256K |
| R9 | Cache hurts throughput at low hit rate | P3.4 | Lookup is O(blocks) dict probes on hashes computed once; overhead measured end-to-end; no-hit path re-checked flag-on; session-affinity retention deferred to P4 (opt-in, default off) | `prefix_cache_longctx_perf_check.py` hashing-overhead + no-hit re-check |
| R10 | `reset_slot` leaks blocks | P3.1 | `reset_slot` already frees via `BlockPool.free` (P1); published blocks retain hash at `ref_cnt==0` (stay hit-able); reverse-order free in P3.2 | `no_block_leak` (P3.1) + `no_leak_churn` (P3.2) |

---

## Verification commands appendix

Run from the repo root `/home/bot/project/qwen-sm120-runtime`. Use the vLLM
venv (the repo `.venv` has NO vllm):

```bash
PY=/home/bot/.venvs/vllm/bin/python
```

**Per-sub-round dedicated gates (new scripts this plan adds):**
```bash
$PY -m benchmarks.prefix_cache_persistent_hit_check     # P3.1 (INV1/INV3/R1, GDN-layer-0 exact)
$PY -m benchmarks.prefix_cache_eviction_check           # P3.2 (INV2/INV3/INV9, lockstep, byte-budget)
$PY -m benchmarks.prefix_cache_hit_check                # P3.3 cumulative gate (INV1/2/3/5/8/9 + eviction)
$PY -m benchmarks.prefix_cache_longctx_perf_check --fixture ctx64k   # P3.4 (also ctx128k/ctx200k/ctx256k)
```

**Zero-regression fast battery (every sub-round, flags at default `False` —
the "P3 changes nothing when off" half; the established per-phase gate):**
```bash
for s in mtp_accept_reject_check mtp_async_arrival_check mtp_batch_verify_check \
         mtp_chunked_prefill_check mtp_gdn_rollback_check mtp_multiround_check \
         mtp_prior_kv_len_fix_check mtp_ragged_prefill_check \
         mtp_ragged_recompute_verify_check mtp_real_draft_check \
         mtp_verify_cudagraph_check cudagraph_eager_parity_check; do
  $PY -m benchmarks.$s || echo "FAILED: $s"
done
# Substrate checks (must stay green):
$PY -m benchmarks.prefix_cache_block_table_check
$PY -m benchmarks.prefix_cache_allocator_check
$PY -m benchmarks.prefix_cache_fanout_check            # P2 (now also exercises publish under the flag)
```

**4K/c=4 headline (must stay bit-identical: `total_committed_tokens=4116`,
`draft_acceptance_rate_pct=70.29204431017119`):**
```bash
$PY -m benchmarks.mtp_w1s_our_runtime_perf --batched --cudagraph --repeats 3 \
    --max-tokens 256 --concurrency 4 --fixture n16
```

**INV5 CUDA-graph parity with hit-populated (non-contiguous) tables (P3.3):**
```bash
$PY -m benchmarks.cudagraph_decode_regression --enable-block-table
$PY -m benchmarks.mtp_verify_cudagraph_check --enable-block-table
$PY -m benchmarks.cudagraph_eager_parity_check            # extended with a hit-populated table in P3.3
```

**Long-context Pattern-B cold baseline + warm re-run (P3.4):**
```bash
# Cold baseline (flag off), TTFT recorded by the bench:
$PY -m benchmarks.mtp_w1s_our_runtime_perf --batched --cudagraph --chunk-size 8192 \
    --fixture ctx64k --max-tokens 256 --concurrency 4
# Warm (flag on) turn-2+ via the new perf check (reports cold/warm TTFT + speedup):
$PY -m benchmarks.prefix_cache_longctx_perf_check --fixture ctx64k
$PY -m benchmarks.prefix_cache_longctx_perf_check --fixture ctx128k
```

**Slow E2E (optional, characterized; not part of the per-phase fast gate per
P0/P1/P2 precedent):**
```bash
$PY -m benchmarks.mtp_sustained_realistic_workload_check --duration-s 90   # representative slice
```

**Sequencing note for the executor.** Land P3.1 → verify → P3.2 → verify →
P3.3 → verify → P3.4. Do NOT start a sub-round until the previous sub-round's
dedicated gate AND the fast battery AND the headline are green. Append a P3
section to `notes/prefix-cache-implementation-log.md` per sub-round (problem,
what was built, verification evidence, next-sub-round readiness), matching the
P0/P1/P2 convention. The two highest-risk seams are P3.1's persistent GDN
restore (R1 — gated by the GDN-layer-0 exact proof) and P3.2's eviction
(R4/R5 — gated by admission-under-pressure + lockstep tests); do not relax
those gates under time pressure.

---

## Leader sign-off (2026-07-19)

Reviewed in full (TL;DR, sub-round table, P3.1 detail, cross-cutting
decisions, risk matrix). **Approved as the authoritative P3 spec.** The
architect's three flagged decisions are confirmed by the leader (none is a
major/ambiguous issue requiring user escalation):

1. **Finer `enable_persistent_prefix_cache` flag** (default off, requires
   `enable_prefix_cache`) — APPROVED. Matches the project's feature-flag
   convention and is what makes each sub-round byte-identical-to-prior until
   it passes (the rollback spine).
2. **Production wiring lands in P3.3; `server/engine.py:_step` swap stays
   P4** — APPROVED. Matches the established "P3 = runner mechanism, P4 =
   server wiring" boundary; P3.1/P3.2 prove the hit path via dedicated tests
   driving `mtp_prefill_with_cache` directly.
3. **INV8 v1 scope** — ragged hit suffixes that individually exceed
   `chunk_size` fall back to cold prefill rather than lifting the existing
   ragged+`chunk_size` `NotImplementedError` — APPROVED (real hit suffixes
   are short).
4. (Non-blocking) **8 GB GDN checkpoint byte-budget default** — APPROVED for
   now; P3.4 sweeps the knob. One-line change if a different KV/checkpoint
   headroom split is wanted.

Execution proceeds sub-round by sub-round (P3.1 → P3.2 → P3.3 → P3.4), each
implemented by an executor subagent and independently verified by the leader
before the next (the standing review discipline). P3.1 is the load-bearing
correctness proof and goes first.
