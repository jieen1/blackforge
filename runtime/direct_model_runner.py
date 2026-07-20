"""Direct (non-HTTP) model runner: this process owns GPU KV/GDN state itself
and drives ``model.forward()`` directly, replacing the HTTP bridge to a
separate vLLM server (``runtime/vllm_bridge_backend.py``, commit ``b28942c``).

Design and the four reused vLLM primitives this depends on (``EngineArgs
.create_engine_config()``, ``get_model()``, ``bind_kv_cache()``,
``set_forward_context()``) are documented in
``notes/direct-model-runner-design.md`` -- read that first.

Scope this round (see the design doc's "explicitly out of scope" section):
only slot 0 is exercised, no CUDA graph, no real multi-request batching, no
MTP. Metadata is hand-built for exactly one request at a time, not through
the production ``SM120GQAMetadataBuilder``/``GDNAttentionMetadataBuilder``
(those handle concerns -- persistent CUDA-graph-safe buffers, spec-decode,
multi-request batching -- this round's scope does not need).
"""

from __future__ import annotations

import array
import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch
import numpy as np
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.model_loader import get_model
from vllm.utils.network_utils import get_distributed_init_method, get_open_port
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.sm120_gqa import SM120GQAMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.utils import bind_kv_cache

NUM_SLOTS = 4
_SM120_BACKEND_PATH = "vllm.v1.attention.backends.sm120_gqa.SM120GQABackend"

# Physical index 0 (block index / GDN state index) is never used for real
# request data -- confirmed empirically from a real vLLM SchedulerOutput
# dump (block_ids=([1], [2], [3], [4]) for the first-ever scheduled
# request; see notes/direct-model-runner-design.md's "Stage C field diff"
# section). Root cause of the 100%-deterministic wrong output this round:
# our hand-built metadata hardcoded physical index = logical slot (so slot
# 0 -> physical index 0), which real vLLM's convention never produces --
# something about index 0 (padding/NULL_BLOCK_ID-adjacent) makes the model
# read/write the wrong state. Fix: reserve physical index 0 permanently
# and offset every logical slot by +1 when computing a physical address.
RESERVED_PHYSICAL_SLOTS = 1


def _physical_slot(logical_slot: int) -> int:
    return logical_slot + RESERVED_PHYSICAL_SLOTS


def _initial_block_table(logical_slot: int, blocks_per_slot: int) -> list[int]:
    """P0 "thin allocator" (``notes/prefix-cache-design.md`` sec 5, P0 --
    block-table indirection substrate): hands out the SAME contiguous
    physical block ids the arange-based addressing has always used --
    ``[first_block, first_block + blocks_per_slot)`` where ``first_block =
    _physical_slot(logical_slot) * blocks_per_slot``. This is the only
    thing P1's dynamic free-list allocator will change; every downstream
    consumer (``build_attention_metadata``/``_batch``, ``_slot_mapping``/
    ``_batch``, ``CapturedBatchDecodeGraph``/``CapturedMTPDraftStepGraph``
    ``._fill_buffers``) reads ``block_table[slot]`` without caring how it
    was populated -- behavior-identical by construction, since the values
    returned here are byte-identical to what the old formula computed
    inline.
    """
    first_block = _physical_slot(logical_slot) * blocks_per_slot
    return list(range(first_block, first_block + blocks_per_slot))


def _ssm_spec_row(logical_slot: int, col: int, total_physical_slots: int, num_spec: int) -> int:
    """Physical SSM-state row for MTP verify's K+1 candidate positions
    (Phase 2, 2026-07-18 -- ``notes/2026-07-17-post-ragged-round-next-steps.md``
    section 10/11). Re-derived and reconfirmed directly against the real
    kernel this round (``fused_sigmoid_gating_delta_rule_update_kernel``,
    ``vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py``): that
    kernel reads its initial state from
    ``ssm_state_indices[i_n, num_accepted_tokens[i_n] - 1]`` and
    unconditionally WRITES a result into ``ssm_state_indices[i_n, t]`` for
    every candidate position ``t`` in this round's batch -- i.e. real
    vLLM's own ``spec_state_indices_tensor`` is a FIXED per-(request,
    column) physical-row mapping (there, a slice of that request's own
    persistent block table, allocated once and never reshuffled across
    rounds), and it is ``num_accepted_tokens`` -- supplied fresh each
    round from the PREVIOUS round's real accept/reject outcome -- that
    selects which of those fixed rows holds the valid incoming state.
    This function is the fixed-slot-runtime analogue of that same fixed
    per-(slot, column) mapping: column 0 always resolves to
    ``_physical_slot(logical_slot)`` -- the SAME row the ordinary
    chunked/non-spec GDN path already writes to (this is what makes the
    bootstrap case -- the first spec verify round right after a real
    prefill -- correct: passing ``num_accepted_tokens_prev=1`` selects
    ``i_t=0``, i.e. column 0, i.e. exactly the row the prefill's chunked
    forward wrote into). Columns 1..num_spec each get their own
    dedicated row in a separate address range (past every slot's column-0
    row), written unconditionally every verify round regardless of which
    candidate later turns out to be the real one -- a rejected
    candidate's row is simply never selected by any future round's
    ``num_accepted_tokens``-derived read, which is what eliminates the
    need for any explicit snapshot/restore or recompute-forward repair.
    """
    physical = _physical_slot(logical_slot)
    if col == 0:
        return physical
    return total_physical_slots + physical * num_spec + (col - 1)


def _derive_none_hash() -> int:
    """Process-global seed for the chained block hash (P3, R7). Derived once
    from a fixed salt + ``PYTHONHASHSEED`` so hashing is reproducible within
    a run but cross-run collisions are impossible (different seed -> different
    chain). The persistent cache is per-process/in-memory, so the seed only
    needs to be a fixed, well-mixed, non-zero value within one process."""
    salt = b"qwen-sm120-runtime/prefix-cache/NONE_HASH"
    seed = os.environ.get("PYTHONHASHSEED", "0").encode("utf-8")
    return int.from_bytes(hashlib.blake2b(salt + seed, digest_size=16).digest(), "big")


NONE_HASH: int = _derive_none_hash()


def hash_block_tokens(parent_hash: int | None, token_ids: list[int], extra_keys: tuple) -> int:
    """Full-width 128-bit chained block hash (``notes/prefix-cache-design.md``
    sec 3.2; vLLM ``hash_block_tokens`` shape, adapted not copied). Block i's
    hash depends on the WHOLE prefix ``0..(i+1)*block_size`` via the chained
    ``parent_hash`` (or the ``NONE_HASH`` seed for block 0), so two prompts
    diverging at any earlier token get different hashes for every block from
    the divergence on -- which is exactly why prefix lookup can stop at the
    first miss.

    ``extra_keys`` MUST carry the ``kv_cache_dtype`` (the runner passes
    ``(self.kv_cache_dtype,)``) so fp8 vs nvfp4 KV never collide (R7). Uses
    ``hashlib.blake2b(digest_size=16)`` -> a 128-bit int; collision probability
    is negligible at this runtime's <=4-slot traffic (R7)."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update((parent_hash or NONE_HASH).to_bytes(16, "big"))
    hasher.update(array.array("Q", token_ids).tobytes())
    if extra_keys:
        hasher.update(repr(extra_keys).encode("utf-8"))
    return int.from_bytes(hasher.digest(), "big")


@dataclass(frozen=True)
class BlockHash:
    """Content identity of one full published attention-KV block (P3).
    ``value`` is the 128-bit chained hash; ``num_tokens = (i+1)*block_size``
    enables the cheap paranoid first-block token-count verify on a hit/dedup
    (R7). Stored on ``Block.block_hash`` for every published block."""

    value: int
    num_tokens: int


@dataclass
class Block:
    """One physical attention-KV block. ``ref_cnt``/``block_hash`` are the
    same fields ``notes/prefix-cache-design.md`` sec 3.2 specifies for
    vLLM-v1-style block-pool bookkeeping. ``block_hash`` is populated by P3's
    persistent content-addressed cache (``BlockPool.cache_block``) for every
    published full block and RETAINED when the block is freed back to
    ``ref_cnt == 0`` -- that retention is exactly what keeps a cached prefix
    hit-able across ``reset_slot`` (R10).

    ``prev_free``/``next_free`` (P3.2) are the intrusive doubly-linked-list
    pointers ``FreeBlockQueue`` manipulates in O(1) (the design doc sec 3.2
    fields, vLLM ``KVCacheBlock.prev_free_block``/``next_free_block`` shape).
    Both are ``None`` exactly when the block is NOT parked in the free queue
    (allocated at ``ref_cnt > 0``, or a just-popped/removed block); a block in
    the queue has both non-``None``. The two ``FreeBlockQueue`` sentinels are
    the only blocks whose links point at a sentinel."""

    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHash | None = None
    prev_free: Block | None = None
    next_free: Block | None = None


class FreeBlockQueue:
    """Intrusive doubly-linked LRU free-block queue (P3.2; design doc sec 3.2,
    vLLM ``FreeKVCacheBlockQueue`` shape, adapted). Replaces P1's plain
    ``collections.deque`` to add O(1) MIDDLE removal -- needed to revive a
    ``ref_cnt == 0`` cached block on a late hit (``BlockPool.touch`` yanks it
    out before reuse) without an O(n) scan.

    Order semantics (vLLM ``free_blocks``/``get_new_blocks`` faithful):
    front (head side, ``popleft``) = evict-next; tail = most-recently-freed.
    ``append`` parks a block at the TAIL (a freed-but-HASHED block stays
    hit-able as long as possible -- LRU tail, evicted last); ``appendleft``
    parks it at the HEAD (a freed block with NO hash carries no cached content,
    so it is evicted FIRST). ``popleft`` returns the evict-next block;
    ``remove(block)`` unlinkes an arbitrary node in O(1). Two sentinel
    ``Block`` s (``block_id`` -1/-2, never popped) eliminate edge branching.

    Links live ON the ``Block`` objects (``prev_free``/``next_free``); the
    queue holds a back-reference to ``blocks`` only so ``__contains__`` can
    resolve a bare ``block_id`` (the P3.1 gate probes ``blk in pool._free_queue``
    with an int)."""

    def __init__(self, blocks: list[Block]) -> None:
        self._blocks = blocks
        self.head = Block(block_id=-1)  # sentinel: head.next = evict-next
        self.tail = Block(block_id=-2)  # sentinel: tail.prev = MRU
        self.head.next_free = self.tail
        self.tail.prev_free = self.head
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def __contains__(self, item: Block | int) -> bool:
        block = item if isinstance(item, Block) else self._blocks[item]
        return block.prev_free is not None and block.next_free is not None

    def append(self, block: Block) -> None:
        """Park ``block`` at the TAIL (most-recently-freed / hashed LRU tail,
        evicted last). O(1)."""
        if block.prev_free is not None or block.next_free is not None:
            raise RuntimeError(f"block {block.block_id} is already linked in the free queue")
        last = self.tail.prev_free
        assert last is not None
        last.next_free = block
        block.prev_free = last
        block.next_free = self.tail
        self.tail.prev_free = block
        self._len += 1

    def appendleft(self, block: Block) -> None:
        """Park ``block`` at the HEAD (evict-next end; freed hashless blocks go
        here so they are reclaimed before any cached block). O(1)."""
        if block.prev_free is not None or block.next_free is not None:
            raise RuntimeError(f"block {block.block_id} is already linked in the free queue")
        first = self.head.next_free
        assert first is not None
        first.prev_free = block
        block.next_free = first
        block.prev_free = self.head
        self.head.next_free = block
        self._len += 1

    def popleft(self) -> Block:
        """Remove and return the evict-next (head) block. O(1)."""
        block = self.head.next_free
        if block is self.tail or block is None:
            raise RuntimeError("popleft from an empty free queue")
        self._unlink(block)
        return block

    def remove(self, block: Block) -> None:
        """Unlink an arbitrary node in O(1) (the late-hit revival primitive)."""
        if block.prev_free is None or block.next_free is None:
            raise RuntimeError(f"remove() on block {block.block_id} not in the free queue")
        self._unlink(block)

    def _unlink(self, block: Block) -> None:
        assert block.prev_free is not None and block.next_free is not None
        block.prev_free.next_free = block.next_free
        block.next_free.prev_free = block.prev_free
        block.prev_free = None
        block.next_free = None
        self._len -= 1


class BlockPool:
    """Dynamic free-list allocator + reference counting over the shared
    physical attention-KV block pool (``notes/prefix-cache-design.md`` sec
    5, "P1 -- Dynamic free-list allocator + reference counting").

    Replaces P0's ``_initial_block_table`` static per-slot partition (which
    just handed every logical slot the SAME contiguous
    ``[_physical_slot(slot) * blocks_per_slot, ...)`` range P0 was proven
    behavior-identical with the old arange addressing): blocks are now
    handed out from ONE shared free queue, on demand, to whichever slot
    asks next -- so a single slot's own block ids may end up non-contiguous
    over time (this is the intended, exercised behavior this phase proves
    safe, not a defect).

    **Free queue (P3.2)**: ``_free_queue`` is an intrusive ``FreeBlockQueue``
    (O(1) ``append``/``appendleft``/``popleft``/``remove``), replacing P1's
    plain ``collections.deque``. ``allocate`` pops from the front (evict-next);
    a popped block that still carries a published hash is EVICTED first -- its
    hash is dropped from ``hash_to_block`` and, in lockstep, the co-keyed GDN
    checkpoint is dropped via the ``_on_evict_block`` callback the runner wires
    after construction (INV2/INV3/R5). ``free`` re-queues a ``ref_cnt == 0``
    block at the TAIL if it keeps a hash (stays hit-able, LRU) or at the HEAD
    if hashless (no cached content, evicted first) -- vLLM ``free_blocks``
    order. ``touch`` yanks a revived ``ref_cnt == 0`` block out in O(1)
    (INV2/INV9). With a generously-sized pool and no pressure, ``allocate``
    never evicts a hashed block, so behavior is byte-for-byte P3.1.

    **INV7 (reserved physical slot 0)**: physical block ids
    ``[0, reserved)`` (``reserved=RESERVED_PHYSICAL_SLOTS`` by default,
    i.e. just block 0) are carved out at construction and NEVER enter the
    free queue via any path -- ``allocate``/``free`` both refuse to
    hand out or accept a reserved id, so a caller bug cannot leak block 0
    into real use, and neither can this class's own logic.
    """

    def __init__(self, num_blocks: int, reserved: int = RESERVED_PHYSICAL_SLOTS) -> None:
        if reserved < 1:
            raise ValueError("must reserve at least physical block 0 (INV7)")
        if num_blocks <= reserved:
            raise ValueError(f"num_blocks={num_blocks} must exceed reserved={reserved}")
        self.num_blocks = num_blocks
        self.reserved = reserved
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        # Free queue: intrusive LRU (FreeBlockQueue). Front (``popleft()``) =
        # evict-next; tail (``append()``) = most-recently-freed. Excludes every
        # reserved id by construction -- reserved ids are never appended by
        # ``free`` either (guarded there too, defense in depth). Initial blocks
        # are linked in ASCENDING id order (append each), so the no-pressure
        # ``popleft`` order matches P1's ``deque(range(reserved, num_blocks))``
        # exactly (byte-for-byte P3.1 when eviction never triggers).
        self._free_queue: FreeBlockQueue = FreeBlockQueue(self.blocks)
        for _block_id in range(reserved, num_blocks):
            self._free_queue.append(self.blocks[_block_id])
        # P3.2 lockstep eviction hook: when ``_evict_one`` reclaims a still-
        # hashed attention block, the runner drops the co-keyed GDN checkpoint
        # too (INV3/R5, both directions). ``None`` for a stand-alone BlockPool
        # (pure-Python tests); the runner sets it to ``evict_gdn_checkpoint``
        # after construction. Kept as a plain callable to avoid a BlockPool ->
        # DirectModelRunner import cycle.
        self._on_evict_block: Callable[[int], None] | None = None
        # P3 persistent content-addressed index (notes/prefix-cache-design.md
        # sec 3.2): maps a full block's 128-bit chained hash VALUE to the
        # physical Block holding that prefix's KV. Populated by cache_block at
        # a cached prefix's completion boundary; probed by get_cached_block on
        # admission. A published block keeps its block_hash (and thus stays in
        # this index) even when freed back to ref_cnt == 0 -- that is exactly
        # what keeps a cached prefix hit-able across reset_slot (R10); the hash
        # is dropped only on real eviction (P3.2).
        self.hash_to_block: dict[int, Block] = {}

    def num_free_blocks(self) -> int:
        return len(self._free_queue)

    def _evict_one(self) -> Block:
        """Pop the evict-next (free-queue front) block and make it reusable
        (P3.2, design doc sec 3.2 ``_maybe_evict`` + sec 3.9 lockstep). If it
        still carries a published hash (a freed-but-cached block at
        ``ref_cnt == 0``), drop that hash from ``hash_to_block`` FIRST -- only
        then is the cached content gone (INV2: a block is never both indexed as
        old content AND handed out for new) -- and, in LOCKSTEP, drop the
        co-keyed GDN checkpoint via ``_on_evict_block`` (INV3/R5: an attention
        tail block and its checkpoint are always evicted together; no
        half-evicted ghost can produce a wrong ``L``). Returns the block at
        ``ref_cnt == 0`` / ``block_hash is None``, ready to hand out. Only
        reachable under real pool pressure (a generous pool never evicts a
        hashed block in the no-pressure path => byte-for-byte P3.1)."""
        block = self._free_queue.popleft()
        if block.block_hash is not None:
            self.hash_to_block.pop(block.block_hash.value, None)
            block.block_hash = None
            if self._on_evict_block is not None:
                self._on_evict_block(block.block_id)
        return block

    def allocate(self, n: int) -> list[int]:
        """Hand out ``n`` blocks from the free-queue front, setting
        ``ref_cnt = 1`` on each and returning their ids in pop order
        (evict-next-first). Every free-queue block is ``ref_cnt == 0`` by
        construction (``free`` re-queues only at 0; ``touch``/``allocate``
        remove on revive), so a popped block is always safe to reuse -- but a
        popped block may still be CACHED (hash retained at ``ref_cnt == 0``),
        in which case ``_evict_one`` evicts it (drops the hash + lockstep GDN
        checkpoint) before hand-out.

        Raises only on TRUE exhaustion: the free queue holds fewer than ``n``
        blocks, i.e. every other block is ``ref_cnt > 0`` (held by an active
        slot) and thus NEVER evictable (INV9). Callers
        (``DirectModelRunner._ensure_blocks``) size the pool generously (this
        project's ``num_blocks = (num_slots + RESERVED_PHYSICAL_SLOTS) *
        blocks_per_slot`` sizing, unchanged from P0), so eviction is rare and
        true exhaustion rarer still."""
        if n < 0:
            raise ValueError(f"cannot allocate a negative count ({n})")
        if n > len(self._free_queue):
            raise RuntimeError(
                f"block pool exhausted: requested {n}, only "
                f"{len(self._free_queue)} free (of {self.num_blocks - self.reserved} total, "
                f"excluding {self.reserved} reserved); every other block is ref_cnt > 0"
            )
        ids: list[int] = []
        for _ in range(n):
            block = self._evict_one()
            if block.block_id < self.reserved:
                raise RuntimeError(
                    f"INV7 violation: reserved block {block.block_id} was in the free queue"
                )
            if block.ref_cnt != 0:
                raise RuntimeError(
                    f"block {block.block_id} was in the free queue with "
                    f"ref_cnt={block.ref_cnt} != 0"
                )
            block.ref_cnt = 1
            ids.append(block.block_id)
        return ids

    def free(self, block_ids: list[int]) -> None:
        """Release blocks back to the pool: ``ref_cnt -= 1``; at 0, re-queue
        (P3.2, vLLM ``free_blocks`` order): a block that KEEPS a published
        hash is ``append``-ed to the TAIL (stays hit-able, LRU -- evicted last
        among free blocks); a block with NO hash carries no cached content so
        it is ``appendleft``-ed to the HEAD (evicted FIRST, before any cached
        block). Written as a decrement (not a hard reset to 0) so a caller
        legitimately holding ``ref_cnt > 1`` (P2 fan-out / P3 hit sharing) is
        not silently masked -- only the final release re-queues."""
        for block_id in block_ids:
            if block_id < self.reserved:
                raise RuntimeError(f"INV7 violation: attempted to free reserved block {block_id}")
            if block_id >= self.num_blocks:
                raise RuntimeError(f"block {block_id} is out of range (num_blocks={self.num_blocks})")
            block = self.blocks[block_id]
            if block.ref_cnt <= 0:
                raise RuntimeError(f"double-free of block {block_id} (ref_cnt={block.ref_cnt})")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if block.block_hash is not None:
                    self._free_queue.append(block)
                else:
                    self._free_queue.appendleft(block)

    def reference(self, block_ids: list[int]) -> None:
        """Add a reference to already-allocated blocks (``ref_cnt += 1`` on
        each), the P2 fan-out primitive that lets a sibling slot share the
        leader's ``[0, Lc)`` attention blocks instead of recomputing them
        (``notes/prefix-cache-design.md`` sec 5, "P2 -- Fan-out fork", and
        sec 3.5 step 1: "touch each (ref_cnt += 1) for all 17 attention
        layers' shared namespace"). The 17 attention layers (16 target + 1
        MTP draft) share ONE block-id namespace, so referencing the block
        ids once covers every layer's KV for those token positions.

        Mirror-image of ``free``'s bookkeeping (which decrements and only
        re-queues at ``ref_cnt == 0``): a referenced block stays out of the
        free queue until EVERY referencer has freed it, so a shared block is
        never handed to ``allocate`` while any slot still references it
        (INV9). Carries the SAME INV7/range guards as ``allocate``/``free``
        (refuse reserved ids ``< reserved`` and out-of-range ids), plus a
        not-currently-allocated guard -- referencing a ``ref_cnt == 0``
        block would resurrect a block that may already be back in the free
        queue (and thus handed to another slot), an immediate aliasing bug.
        """
        for block_id in block_ids:
            if block_id < self.reserved:
                raise RuntimeError(
                    f"INV7 violation: attempted to reference reserved block {block_id}"
                )
            if block_id >= self.num_blocks:
                raise RuntimeError(
                    f"block {block_id} is out of range (num_blocks={self.num_blocks})"
                )
            block = self.blocks[block_id]
            if block.ref_cnt <= 0:
                raise RuntimeError(
                    f"cannot reference block {block_id} with ref_cnt={block.ref_cnt} "
                    "(not currently allocated -- it may already be back in the free queue)"
                )
            block.ref_cnt += 1

    def cache_block(self, block_id: int, block_hash: BlockHash) -> None:
        # Publish a full block to the content index (P3, sec 3.2): tag
        # blocks[block_id].block_hash and record hash_to_block[value].
        # Idempotent guard: if value is ALREADY present this is the write-time
        # dedup signal (see DirectModelRunner._publish_committed_blocks) -- the
        # existing entry is the canonical block for that prefix and is NOT
        # overwritten (the caller swaps onto it and frees the duplicate fresh
        # block instead).
        if block_id < self.reserved:
            raise RuntimeError(f"INV7 violation: attempted to publish reserved block {block_id}")
        if block_id >= self.num_blocks:
            raise RuntimeError(f"block {block_id} is out of range (num_blocks={self.num_blocks})")
        if block_hash.value in self.hash_to_block:
            return
        self.blocks[block_id].block_hash = block_hash
        self.hash_to_block[block_hash.value] = self.blocks[block_id]

    def get_cached_block(self, hash_value: int) -> Block | None:
        # Content-index probe (P3, sec 3.4 attention match): return the
        # published Block whose chained hash equals hash_value, or None on a
        # miss. O(1) dict lookup.
        return self.hash_to_block.get(hash_value)

    def touch(self, block_ids: list[int]) -> None:
        # Cache-hit reference primitive (P3, sec 3.2/3.5 step 1): ref_cnt += 1
        # on each block, REVIVING a block parked at ref_cnt == 0 in the free
        # queue by removing it first. Mirror of reference() but LEGAL for
        # ref_cnt == 0 -- reference() stays the same-round-fork primitive (it
        # rejects ref_cnt == 0), touch() is the persistent-cache hit primitive
        # (a published block freed back to ref_cnt == 0 by reset_slot is still
        # hit-able and must be yanked out of the free queue before reuse --
        # INV2/INV9). P3.2: the removal is O(1) via the intrusive FreeBlockQueue
        # (replacing P3.1's O(n) deque remove).
        for block_id in block_ids:
            if block_id < self.reserved:
                raise RuntimeError(f"INV7 violation: attempted to touch reserved block {block_id}")
            if block_id >= self.num_blocks:
                raise RuntimeError(f"block {block_id} is out of range (num_blocks={self.num_blocks})")
            block = self.blocks[block_id]
            if block.ref_cnt == 0:
                # Parked in the free queue (a freed-but-published block):
                # revive it so allocate never hands it to another slot while
                # this hit references it.
                self._free_queue.remove(block)
            block.ref_cnt += 1


def _ensure_sm120_backend_registered() -> None:
    """register_backend() is a plain dict write (see registry.py's
    _ATTN_OVERRIDES) -- safe to call more than once."""
    register_backend(AttentionBackendEnum.CUSTOM, _SM120_BACKEND_PATH)


def allocate_fixed_slot_kv_caches(
    static_forward_context: dict,
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    num_slots: int,
    block_size: int,
    blocks_per_slot: int,
    num_speculative_tokens: int = 0,
) -> dict[str, object]:
    """Allocate our own num_slots-fixed-slot KV (attention) and state (GDN)
    tensors and bind them via vLLM's own real ``bind_kv_cache()`` -- shared
    between ``DirectModelRunner`` (hand-built metadata) and
    ``runtime/vllm_stage_b_baseline.py`` (real vLLM metadata/scheduler,
    Stage B of the 2026-07-16 ownership-transfer ladder: this is the ONLY
    thing that differs from vLLM's own tensor allocation -- everything else
    stays real). Returns the same ``dict[str, tensor|tuple]`` bind_kv_cache
    expects, keyed by layer name.
    """
    attn_layer_names = []
    gdn_layer_names = []
    for name, layer in static_forward_context.items():
        if hasattr(layer, "get_state_shape"):
            gdn_layer_names.append(name)
        else:
            attn_layer_names.append(name)

    kv_caches: dict[str, object] = {}

    if attn_layer_names:
        any_attn = static_forward_context[attn_layer_names[0]]
        backend_cls = any_attn.get_attn_backend()
        num_kv_heads = any_attn.num_kv_heads
        head_size = any_attn.head_size
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        num_blocks = (num_slots + RESERVED_PHYSICAL_SLOTS) * blocks_per_slot
        shape = backend_cls.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )
        torch_dtype = any_attn.kv_cache_torch_dtype
        for name in attn_layer_names:
            kv_caches[name] = torch.zeros(shape, dtype=torch_dtype, device=device)

    for name in gdn_layer_names:
        layer = static_forward_context[name]
        conv_shape, ssm_shape = layer.get_state_shape()
        conv_dtype, ssm_dtype = layer.get_state_dtype()
        total_physical_slots = num_slots + RESERVED_PHYSICAL_SLOTS
        conv_state = torch.zeros((total_physical_slots, *conv_shape), dtype=conv_dtype, device=device)
        # Phase 2 (2026-07-18): SSM/recurrent state gets num_speculative_tokens
        # EXTRA dedicated rows per physical slot -- one per non-anchor MTP
        # candidate position -- on top of the ordinary one row per physical
        # slot ("column 0", shared with the non-spec/chunked/prefill path).
        # See _ssm_spec_row's docstring for the addressing scheme and its
        # direct verification against the real spec-decode GDN kernel.
        # num_speculative_tokens=0 (no MTP configured) reduces this to
        # exactly the previous allocation -- byte-for-byte unaffected.
        ssm_rows_per_slot = 1 + num_speculative_tokens
        ssm_state = torch.zeros(
            (total_physical_slots * ssm_rows_per_slot, *ssm_shape), dtype=ssm_dtype, device=device
        )
        kv_caches[name] = (conv_state, ssm_state)

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_caches, static_forward_context, runner_kv_caches)
    return kv_caches


def build_attention_metadata(
    *,
    prior_kv_len: int,
    num_new_tokens: int,
    is_decode: bool,
    slot: int,
    block_size: int,
    blocks_per_slot: int,
    device: torch.device,
    block_table: list[int] | None = None,
) -> SM120GQAMetadata:
    """Hand-built SM120GQAMetadata for one request in one fixed slot. Shared
    between ``DirectModelRunner`` (which tracks ``prior_kv_len`` itself via
    ``self.slot_kv_len``) and Stage C of the 2026-07-16 ownership-transfer
    ladder (``runtime/vllm_stage_c_baseline.py``, which derives
    ``prior_kv_len`` from vLLM's own real, scheduler-computed
    ``CommonAttentionMetadata`` instead) -- this is deliberately the exact
    same field-construction logic in both cases, so Stage C tests whether
    *this logic* is correct, not a second, independently-written copy of it.

    ``block_table`` (P0, 2026-07-19, ``notes/prefix-cache-design.md`` sec
    5): optional per-slot list of physical block ids, indexed by LOGICAL
    page position. ``None`` (the default) preserves the exact prior
    arange-based addressing byte-for-byte -- required for
    ``runtime/vllm_stage_c_baseline.py``, which does not pass this
    parameter and must remain untouched. ``DirectModelRunner`` passes its
    own ``self.block_table[slot]`` here only when constructed with
    ``enable_block_table=True``.
    """
    new_kv_len = prior_kv_len + num_new_tokens
    page_size = block_size
    first_block = _physical_slot(slot) * blocks_per_slot
    num_pages = (new_kv_len + page_size - 1) // page_size
    if num_pages > blocks_per_slot:
        raise RuntimeError(
            f"slot {slot} kv_len {new_kv_len} exceeds this slot's "
            f"{blocks_per_slot * page_size}-token capacity"
        )
    qo_indptr = torch.tensor([0, num_new_tokens], dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    if block_table is not None:
        kv_page_indices = torch.tensor(
            block_table[:num_pages], dtype=torch.int32, device=device
        )
    else:
        kv_page_indices = torch.arange(
            first_block, first_block + num_pages, dtype=torch.int32, device=device
        )
    last_page_len = new_kv_len - (num_pages - 1) * page_size
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    return SM120GQAMetadata(
        num_actual_tokens=num_new_tokens,
        num_reqs=1,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        page_size=page_size,
        is_pure_decode=is_decode and num_new_tokens == 1,
        kv_split_size=max(new_kv_len, 1),
        max_num_splits=1,
        decode_qo_len=num_new_tokens if is_decode else 0,
    )


def build_gdn_metadata(
    *,
    slot_initialized: bool,
    num_new_tokens: int,
    is_decode: bool,
    slot: int,
    device: torch.device,
) -> GDNAttentionMetadata:
    """Hand-built GDNAttentionMetadata for one request in one fixed slot --
    see ``build_attention_metadata``'s docstring for why this is a shared
    function, not a second copy, between ``DirectModelRunner`` and Stage C.
    """
    state_indices = torch.tensor([_physical_slot(slot)], dtype=torch.int32, device=device)
    if is_decode:
        assert num_new_tokens == 1
        non_spec_qsl = torch.tensor([0, 1], dtype=torch.int32, device=device)
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=1,
            num_decode_tokens=1,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=1,
            non_spec_query_start_loc=non_spec_qsl,
            non_spec_state_indices_tensor=state_indices,
        )

    query_start_loc = torch.tensor([0, num_new_tokens], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    has_initial_state = torch.tensor([slot_initialized], dtype=torch.bool, device=device)
    chunk_indices = prepare_chunk_indices(query_start_loc, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(query_start_loc, FLA_CHUNK_SIZE)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    return GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=num_new_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_new_tokens,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=query_start_loc,
        prefill_state_indices=state_indices,
        prefill_has_initial_state=has_initial_state,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


_MAX_DECODE_QO_LEN = 16
# Matches the real SM120GQAMetadataBuilder's own _MAX_DECODE_QO_LEN
# (vllm/v1/attention/backends/sm120_gqa.py) -- the decode/verify-shaped
# fast kernel's tested qo_len upper bound. Every real call in this project
# stays well under this (K+1 <= 4), so this cap was previously a latent,
# never-exercised gap (the old unconditional ``decode_qo_len = qo_len``
# formula had no cap at all) -- added here as part of the 2026-07-17
# ragged-qo_len generalization below, for faithfulness to the real
# formula, not because it currently changes any real call's outcome.

_DEFAULT_PREFILL_CHUNK_SIZE = 8192
# The suggested value for ``mtp_prefill_batch``'s ``chunk_size`` parameter
# (2026-07-19, chunked-prefill round) -- matches native vLLM's own
# ``--max-num-batched-tokens=8192`` default (``sm120-flash-attention/
# vllm_integration/launch_test_server.py``), so a chunked run and native's
# own chunked-prefill scheduler bound a single forward call's token count
# identically. Not itself used as ``mtp_prefill_batch``'s default (that
# stays ``None`` = unchunked, for full backward compatibility) -- callers
# that want chunking pass this explicitly.


def build_attention_metadata_batch(
    *,
    slots: list[int],
    prior_kv_lens: list[int],
    block_size: int,
    blocks_per_slot: int,
    device: torch.device,
    qo_len: int | list[int] = 1,
    is_decode: bool = True,
    fixed_kv_split_size: int | None = None,
    fixed_max_num_splits: int | None = None,
    block_tables: list[list[int]] | None = None,
) -> SM120GQAMetadata:
    """Hand-built SM120GQAMetadata for a real batch of requests spanning
    multiple fixed physical slots in a SINGLE metadata object, each
    contributing the SAME ``qo_len`` new query tokens this step (uniform
    across the batch -- the normal production case, since
    ``num_speculative_tokens`` is a global engine config, not
    per-request). ``qo_len=1`` (the default) is the batched analogue of
    ``build_attention_metadata``'s ``is_decode`` case (2026-07-16, verified
    through the full batch=1/2/3/4/varlen/reuse/continuous-generation
    ladder). ``qo_len>1`` is MTP/speculative-decode verify (K draft tokens
    + 1 bonus token per request, e.g. qo_len=4 for K=3) -- the real
    ``SM120GQAImpl.forward()`` dispatches this to
    ``flash_attn_sm120_fwd_v2_decode_fp8kv_paged`` (already
    production-hardened for qo_len in 2..4) whenever
    ``attn_metadata.decode_qo_len`` is in that range; this function only
    needs to construct the metadata correctly, not touch the kernel.

    Field construction (CSR qo_indptr/kv_page_indptr/kv_page_indices/
    kv_last_page_len, is_pure_decode, decode_qo_len) matches the real
    ``SM120GQAMetadataBuilder.build()`` (vllm/v1/attention/backends/
    sm120_gqa.py) generalized from vLLM's dense block-table -> CSR
    conversion to this project's fixed-slot addressing. At ``qo_len=1``
    every formula below reduces exactly to the previously-verified
    qo_len=1-only formulas (same numeric values, same tensors) -- this is
    a generalization, not a parallel implementation, so the existing
    ladder's results remain valid evidence for the qo_len=1 case.

    Kept as a SEPARATE function from ``build_attention_metadata`` (not a
    generalization of it) so this new batch path cannot regress the
    already-verified single-request path (2026-07-16 slot-0-reservation
    fix, Stage C/D 20/20) -- the two are cross-checked instead via the
    batch=1 equivalence test in ``benchmarks/batch_decode_regression.py``.

    ``is_decode`` (2026-07-17 addition, default ``True`` preserving
    ``decode_batch``/``verify_batch``'s existing behavior exactly): gates
    ``decode_qo_len``/``is_pure_decode`` exactly like
    ``build_attention_metadata``'s own ``is_decode`` parameter already
    does (``decode_qo_len = qo_len if is_decode else 0``,
    ``is_pure_decode = is_decode and qo_len == 1``) -- a real, pre-existing
    gap this function had before this fix: it used to set
    ``decode_qo_len = qo_len`` UNCONDITIONALLY, correct for every call site
    that existed before 2026-07-17 (``decode_batch``'s qo_len=1 decode and
    ``verify_batch``'s qo_len=k+1 MTP verify -- both genuinely
    decode/verify-shaped), but wrong for a genuine chunked/prefix PREFILL
    forward (e.g. ``mtp_prefill_batch``'s target-model call, or
    ``_mtp_sync_and_propose_batch``'s draft-model step-0 sync call when
    ``num_new_tokens > 1``): telling the kernel ``decode_qo_len=N`` for an
    N-token PREFILL falsely routes it through the decode/verify-shaped
    kernel path (confirmed against the real, authoritative
    ``SM120GQAMetadataBuilder.build()`` in ``vllm/v1/attention/backends/
    sm120_gqa.py``, whose own formula is
    ``decode_qo_len = cm.max_query_len if (is_uniform_qo_len and
    cm.max_query_len <= _MAX_DECODE_QO_LEN) else 0`` -- i.e. it is NEVER
    unconditional either). Caught via a real numerical-twin divergence
    (``benchmarks/mtp_batch_divergence_diag.py``): a batch=1 call through
    the new ``mtp_prefill_batch``/``_mtp_forward_batch`` path diverged
    from the long-verified single-slot ``mtp_prefill``/``_mtp_forward``
    path even at batch size 1, which is what proved this was a genuine
    formula gap and not a batch-size-dependent kernel numerics difference
    (the other candidate explanation this project's own established
    near-tie precedent would have made plausible).

    ``fixed_kv_split_size``/``fixed_max_num_splits`` (both None by
    default, preserving the existing per-call-derived behavior for the
    eager decode_batch/verify_batch paths): REQUIRED for CUDA-graph
    capture. Per this project's own read of
    ``vllm/v1/attention/backends/sm120_gqa.py``'s documented history (a
    real, previously-hit illegal-memory-access crash): a captured kernel
    launch's scalar arguments (kv_split_size/max_num_splits are plain
    Python ints, not device tensors) freeze to whatever value was live at
    capture time. If kv_split_size were still derived per-call from live
    ``new_kv_lens`` (as the default path does), replaying the SAME
    captured launch at a LARGER kv_len than capture time would silently
    use a stale, too-small split boundary -- the real backend's own fix
    (and this project's, when these are supplied) is to derive
    kv_split_size ONCE from a build-time-fixed upper bound L (this
    project's own ``blocks_per_slot * block_size``, i.e. the per-slot
    page-table limit THIS RUNTIME'S CALLER configured when constructing
    ``DirectModelRunner`` -- a software-chosen ceiling, not a GPU hardware
    limit; already enforced by the RuntimeError check below) via
    ``kv_split_size = ceil(L / target_splits)``,
    ``max_num_splits = target_splits``. Proof this stays correct for
    EVERY real kv_len from 1 up to L, not just the capture-time value:
    for split_size s = ceil(L/target_splits) and any real kv_len k <= L,
    num_splits(k) = ceil(k/s) <= ceil(L/s) <= target_splits (s >= L/target_splits
    by construction) -- so a single fixed pair is a valid upper bound for
    the entire decode lifetime of any request in this slot.

    ``qo_len`` as a RAGGED per-request list (2026-07-17 generalization,
    for the recompute-fallback batching round): previously a single
    scalar shared by the whole batch (the uniform case, e.g. verify's
    K+1 or a single decode token). A ``list[int]`` (one value per slot)
    is now also accepted -- this is what lets
    ``mtp_verify_and_commit_batch``'s recompute-fallback group (each
    slot needing a DIFFERENT number of real committed tokens replayed)
    batch into ONE call instead of one single-slot call per affected
    slot. A scalar ``qo_len`` is treated as a uniform list (broadcast to
    every slot) -- this is a strict generalization, not a parallel code
    path: every existing scalar call site produces byte-identical
    tensors to before.

    ``decode_qo_len``/``is_pure_decode`` now match the real
    ``SM120GQAMetadataBuilder.build()`` formula exactly, including its
    non-uniform-batch behavior: ``decode_qo_len = max(qo_lens) if
    (is_decode and is_uniform and max(qo_lens) <= _MAX_DECODE_QO_LEN)
    else 0``. A RAGGED (non-uniform) qo_lens list therefore ALWAYS gets
    ``decode_qo_len=0`` -- this deliberately routes ragged batches
    through the SAME general/chunked-prefix attention kernel
    (``flash_attn_sm120_fp8_kv_paged`` et al.) that this project's own
    genuine multi-token PREFILL calls already use, NOT a new kernel path
    invented for this round: that kernel is already documented (source
    comment in ``vllm/v1/attention/backends/sm120_gqa.py``) as "correct
    for pure prefill, chunked-prefill continuation, and ARBITRARY MIXED
    prefill+decode batches" -- i.e. real, ragged, per-request-varying
    query lengths within one batched call are exactly its designed use
    case, not new territory for the kernel itself, only for how THIS
    project's Python-side metadata construction reaches it.

    ``block_tables`` (P0, 2026-07-19, ``notes/prefix-cache-design.md`` sec
    5): optional list of per-slot physical block-id lists, ONE per entry in
    ``slots`` (same order), each indexed by LOGICAL page position. ``None``
    (the default) preserves the exact prior arange-based addressing
    byte-for-byte. Passed by ``DirectModelRunner`` only when constructed
    with ``enable_block_table=True``.
    """
    num_reqs = len(slots)
    if len(prior_kv_lens) != num_reqs:
        raise ValueError("slots and prior_kv_lens must have equal length")
    qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
    if len(qo_lens) != num_reqs:
        raise ValueError("qo_len list must have exactly one entry per slot")
    is_uniform = len(set(qo_lens)) <= 1

    page_size = block_size
    new_kv_lens = [kv_len + qo for kv_len, qo in zip(prior_kv_lens, qo_lens)]
    num_pages_per_req = [(kv_len + page_size - 1) // page_size for kv_len in new_kv_lens]
    for slot, kv_len, num_pages in zip(slots, new_kv_lens, num_pages_per_req):
        if num_pages > blocks_per_slot:
            raise RuntimeError(
                f"slot {slot} kv_len {kv_len} exceeds this slot's "
                f"{blocks_per_slot * page_size}-token capacity"
            )

    qo_indptr_list = [0]
    for qo in qo_lens:
        qo_indptr_list.append(qo_indptr_list[-1] + qo)
    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=device)

    kv_page_indptr_list = [0]
    for num_pages in num_pages_per_req:
        kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=device)

    if block_tables is not None:
        if len(block_tables) != num_reqs:
            raise ValueError("block_tables must have exactly one entry per slot")
        page_index_chunks = [
            torch.tensor(table[:num_pages], dtype=torch.int32, device=device)
            for table, num_pages in zip(block_tables, num_pages_per_req)
        ]
    else:
        page_index_chunks = [
            torch.arange(
                _physical_slot(slot) * blocks_per_slot,
                _physical_slot(slot) * blocks_per_slot + num_pages,
                dtype=torch.int32,
                device=device,
            )
            for slot, num_pages in zip(slots, num_pages_per_req)
        ]
    kv_page_indices = (
        torch.cat(page_index_chunks) if page_index_chunks else torch.empty(0, dtype=torch.int32, device=device)
    )
    kv_last_page_len = torch.tensor(
        [kv_len - (num_pages - 1) * page_size for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)],
        dtype=torch.int32,
        device=device,
    )
    if fixed_kv_split_size is not None:
        # CUDA-graph-safe path: fixed once from a build-time bound, never
        # from this call's live data -- see the docstring's proof.
        kv_split_size = fixed_kv_split_size
        max_num_splits = fixed_max_num_splits if fixed_max_num_splits is not None else 1
    else:
        # Conservative, correctness-first choice matching the single-request
        # function: kv_split_size >= every request's own new_kv_len forces
        # num_splits == 1 for all of them (no cross-request split-size tuning
        # yet -- a performance follow-on, not a correctness concern, exactly
        # the same tradeoff build_attention_metadata already makes). NOT
        # CUDA-graph-safe (see docstring) -- only used by the eager
        # decode_batch/verify_batch paths.
        kv_split_size = max(max(new_kv_lens, default=1), 1)
        max_num_splits = 1

    max_qo_len = max(qo_lens) if qo_lens else 0
    decode_qo_len = max_qo_len if (is_decode and is_uniform and max_qo_len <= _MAX_DECODE_QO_LEN) else 0
    return SM120GQAMetadata(
        num_actual_tokens=sum(qo_lens),
        num_reqs=num_reqs,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        page_size=page_size,
        is_pure_decode=(is_decode and max_qo_len == 1),
        kv_split_size=kv_split_size,
        max_num_splits=max_num_splits,
        decode_qo_len=decode_qo_len,
    )


def build_gdn_metadata_batch(
    *,
    slots: list[int],
    device: torch.device,
    qo_len: int | list[int] = 1,
    slot_initialized: list[bool] | None = None,
) -> GDNAttentionMetadata:
    """Hand-built GDNAttentionMetadata for a real batch of requests, each
    contributing the SAME ``qo_len`` new query tokens this step (see
    ``build_attention_metadata_batch``'s docstring for the uniform-qo_len
    scope rationale shared by both functions).

    ``qo_len=1`` (the default) is the batched analogue of
    ``build_gdn_metadata``'s ``is_decode`` case (2026-07-16, verified
    through the full batch ladder) -- matches the real
    ``GDNAttentionMetadataBuilder.build()``'s pure non-spec-decode branch
    (gdn_attn.py): only num_decodes/num_decode_tokens/
    non_spec_query_start_loc/non_spec_state_indices_tensor are populated,
    everything else stays None. At qo_len=1 the formulas below reduce
    exactly to the previously-verified values -- a generalization, not a
    parallel implementation.

    **2026-07-17 real bug, found via a batch=1 forced-reject equivalence
    test** (``benchmarks/mtp_ragged_recompute_verify_check.py``): an
    earlier version of this fast path was gated on
    ``isinstance(qo_len, int) and qo_len == 1`` -- i.e. a UNIFORM list
    where every entry happens to be 1 (exactly what the new
    ragged-recompute-batching caller always constructs, even for a
    single recompute slot with ``committed_len == 1``, since it always
    passes a list) fell through to the chunked/general branch instead,
    unlike a bare scalar ``1``. That asymmetry is NOT what
    ``build_attention_metadata_batch``'s own analogous ``decode_qo_len``
    logic does (it already treats a uniform list and a scalar
    identically, via ``is_uniform``/``max_qo_len``) -- this function's
    old condition was a real, narrower special-case that diverged from
    the sibling function's already-correct generalization, and the
    chunked/general GDN path is NOT a drop-in numerically-equivalent
    substitute for the fast single-token decode path (confirmed by the
    test finding real committed-content divergence between the two,
    not just a near-tie). Fixed below by making the fast-path condition
    VALUE-based (uniform, all entries equal to 1) instead of
    TYPE-based (bare scalar only) -- this treats scalar ``1`` and a
    uniform ``[1, 1, ...]`` list identically, exactly mirroring
    ``build_attention_metadata_batch``'s own already-correct pattern.

    ``qo_len>1`` (or a ragged per-request list, 2026-07-17 generalization
    -- see below) is MTP/speculative-decode verify. Rather than
    replicating the real builder's much more involved ``spec_decode``
    branch (accept/reject bookkeeping, sorting spec vs non-spec tokens --
    explicitly out of scope this round, see notes/direct-model-runner-
    design.md), this generalizes ``build_gdn_metadata``'s OTHER existing
    branch instead: the ``is_decode=False`` ("prefill"/chunked) case,
    which the real builder's own ``split_decodes_and_prefills`` would also
    select for any request with query_len>1 when no draft-acceptance info
    is supplied -- i.e. this treats an MTP verify step as an ordinary
    chunked continuation of ``qo_len`` new tokens per request, which is
    numerically correct (the chunked FLA kernel handles arbitrary query
    length, GDN state update included) even though it foregoes the
    real builder's spec-decode-specific optimizations.

    ``qo_len`` as a RAGGED per-request list (2026-07-17, for the
    recompute-fallback batching round -- mirrors
    ``build_attention_metadata_batch``'s identical generalization):
    ``query_start_loc`` is built from a per-request cumulative sum
    instead of ``arange(n+1) * qo_len`` -- a strict generalization (a
    scalar ``qo_len`` broadcasts to a uniform list, reducing to the exact
    same tensor as before). Crucially, this is NOT the same padding
    concern flagged when this generalization was first scoped: a
    genuinely ragged CSR construction feeds EVERY request EXACTLY its own
    real token count into the chunked FLA kernel (``prepare_chunk_indices``/
    ``prepare_chunk_offsets``/``compute_causal_conv1d_metadata`` are all
    already CSR/``cu_seqlens``-generic -- this project's own prior
    uniform-only usage was a special case of what these functions already
    support, not a hand-restriction on them). There is no padding token
    ever fed to any request's GDN state under this design, so the
    recurrent-state-corruption concern that made a padding-based ragged
    batch design hard does not apply here -- it only would have applied
    to a design that forced a shared qo_len via padding, which this is
    deliberately NOT doing.
    """
    num_reqs = len(slots)
    state_indices = torch.tensor(
        [_physical_slot(slot) for slot in slots], dtype=torch.int32, device=device
    )
    qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
    if len(qo_lens) != num_reqs:
        raise ValueError("qo_len list must have exactly one entry per slot")
    if num_reqs > 0 and all(qo == 1 for qo in qo_lens):
        non_spec_qsl = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device)
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=num_reqs,
            num_decode_tokens=num_reqs,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=num_reqs,
            non_spec_query_start_loc=non_spec_qsl,
            non_spec_state_indices_tensor=state_indices,
        )

    if slot_initialized is None or len(slot_initialized) != num_reqs:
        raise ValueError("slot_initialized (one bool per slot) is required when qo_len != 1")
    qsl_list = [0]
    for qo in qo_lens:
        qsl_list.append(qsl_list[-1] + qo)
    query_start_loc = torch.tensor(qsl_list, dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    has_initial_state = torch.tensor(slot_initialized, dtype=torch.bool, device=device)
    chunk_indices = prepare_chunk_indices(query_start_loc, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(query_start_loc, FLA_CHUNK_SIZE)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    num_actual_tokens = sum(qo_lens)
    return GDNAttentionMetadata(
        num_prefills=num_reqs,
        num_prefill_tokens=num_actual_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_actual_tokens,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=query_start_loc,
        prefill_state_indices=state_indices,
        prefill_has_initial_state=has_initial_state,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


def build_gdn_metadata_spec_batch(
    *,
    slots: list[int],
    device: torch.device,
    qo_len: int,
    num_accepted_tokens_prev: list[int],
    total_physical_slots: int,
    num_spec: int,
) -> GDNAttentionMetadata:
    """Real spec-decode GDN metadata (Phase 2, 2026-07-18) -- the
    ``num_prefills=0, num_decodes=0, num_spec_decodes=len(slots)`` branch
    of the real ``GDNAttentionMetadataBuilder.build()`` (``gdn_attn.py``),
    hand-built for our fixed-slot runtime instead of vLLM's paged block
    table. Replaces ``build_gdn_metadata_batch``'s chunked/prefill-shaped
    treatment of an MTP verify step -- originally for
    ``mtp_verify_and_commit_batch`` only (``mtp_verify_and_commit``, the
    singular/looped sibling, kept using the chunked path +
    snapshot/restore/recompute -- an intentional, documented divergence,
    see notes doc section 10/11), then (Phase B, same day) for
    ``mtp_verify_and_commit`` too, called at ``len(slots)==1``: both
    production verify paths now share this mechanism.

    ``qo_len`` here is always ``num_spec + 1`` (the K draft continuations
    + 1 bonus/anchor position) -- unlike ``build_gdn_metadata_batch``,
    this is NOT generalized to a ragged per-request list: every slot in
    one spec-decode verify call always submits the SAME K+1-token draft
    (a global engine config), matching ``verify_batch``'s own existing
    uniform-qo_len contract.

    ``spec_state_indices_tensor[i, col] = _ssm_spec_row(slots[i], col,
    total_physical_slots, num_spec)`` -- a FIXED per-(slot, column)
    physical-row mapping, unchanging round to round (see
    ``_ssm_spec_row``'s docstring for why this matches the real kernel's
    own block-table-derived addressing). ``num_accepted_tokens_prev[i]``
    is slot ``slots[i]``'s real committed length from its OWN previous
    verify round (or exactly ``1`` for a slot's first-ever verify right
    after a real prefill -- the bootstrap case, selecting column 0, the
    row the chunked prefill forward itself wrote into).

    ``spec_token_indx``/``non_spec_token_indx``/``has_initial_state``
    are deliberately left ``None`` -- confirmed by direct reading of
    ``qwen_gdn_linear_attn.py``'s ``_forward_core`` that the
    ``num_prefills=0 and num_decodes=0`` branch takes ``mixed_qkv_spec =
    mixed_qkv`` directly (no ``index_select``) and never reads
    ``has_initial_state`` at all in that branch (real drafts always
    resume from real prior state -- there is no "fresh" spec-decode
    request in this project's design, since every slot has already gone
    through a real ``mtp_prefill_batch`` before its first verify call)."""
    num_reqs = len(slots)
    if len(num_accepted_tokens_prev) != num_reqs:
        raise ValueError("num_accepted_tokens_prev must have exactly one entry per slot")
    spec_query_start_loc = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
    state_indices_list = [
        [_ssm_spec_row(slot, col, total_physical_slots, num_spec) for col in range(qo_len)]
        for slot in slots
    ]
    spec_state_indices_tensor = torch.tensor(state_indices_list, dtype=torch.int32, device=device)
    spec_sequence_masks = torch.ones(num_reqs, dtype=torch.bool, device=device)
    num_accepted_tokens = torch.tensor(num_accepted_tokens_prev, dtype=torch.int32, device=device)
    num_actual_tokens = num_reqs * qo_len
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_reqs,
        num_spec_decode_tokens=num_actual_tokens,
        num_actual_tokens=num_actual_tokens,
        spec_query_start_loc=spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_tensor,
        spec_sequence_masks=spec_sequence_masks,
        num_accepted_tokens=num_accepted_tokens,
    )


def _install_triton_norm_ops_once() -> None:
    """Install Triton-fused RMSNorm ops (vLLM C ext lacks them on this machine).
    Must be called AFTER create_engine_config() because that call resets
    IR op priorities via KernelConfig.ir_op_priority.set_priority()."""
    try:
        from runtime.triton_norm_ops import install_triton_norm_ops
        install_triton_norm_ops()
    except Exception:
        pass
    try:
        from runtime.gemma_norm_patch import patch_gemma_rms_norm
        patch_gemma_rms_norm()
    except Exception:
        pass


def build_vllm_config(
    *,
    model: str,
    kv_cache_dtype: str = "fp8_e4m3",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.5,
    speculative_config: dict | None = None,
) -> VllmConfig:
    _ensure_sm120_backend_registered()
    args = EngineArgs(
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        attention_backend=AttentionBackendEnum.CUSTOM,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        language_model_only=True,
        async_scheduling=False,
        speculative_config=speculative_config,
    )
    config = args.create_engine_config()
    _install_triton_norm_ops_once()
    return config


def determine_accept_reject(draft_tokens: list[int], verify_logits) -> dict:
    """Greedy MTP accept/reject (2026-07-17, moved here from
    ``benchmarks/mtp_accept_reject_check.py`` so the real
    ``mtp_verify_and_commit`` coordinator and that benchmark's regression
    test share ONE implementation, not two copies). ``draft_tokens`` has
    K+1 entries (anchor + K drafts); ``verify_logits`` is shaped
    ``[K+1, vocab]`` for ONE request. Returns ``num_accepted`` (0..K), the
    committed real token ids (accepted drafts, if any, plus exactly one
    recovery/bonus token), and the rejection position (``None`` if all K
    were accepted)."""
    k = len(draft_tokens) - 1
    committed: list[int] = []
    for p in range(k):
        predicted = int(verify_logits[p].argmax(dim=-1).item())
        if predicted == draft_tokens[p + 1]:
            committed.append(draft_tokens[p + 1])
        else:
            committed.append(predicted)
            return {"num_accepted": p, "committed": committed, "rejected_at": p}
    bonus = int(verify_logits[k].argmax(dim=-1).item())
    committed.append(bonus)
    return {"num_accepted": k, "committed": committed, "rejected_at": None}


def determine_accept_reject_batch(
    slots: list[int], drafts: dict[int, list[int]], verify_logits: torch.Tensor, k: int
) -> dict[int, dict]:
    """Batched analogue of ``determine_accept_reject`` -- computes the SAME
    greedy accept/reject decision for every slot in ONE vectorized GPU op
    plus exactly ONE host round-trip, instead of a Python loop calling
    ``determine_accept_reject`` once per slot (each of which does up to
    ``k+1`` sequential ``.item()`` calls -- 2026-07-17, Phase 3 of
    ``notes/2026-07-17-post-ragged-round-next-steps.md``, directly
    targeting that doc's section 7.4 finding that the compute-phase
    no-kernel gap is dominated by per-launch host dispatch, not GPU work).

    ``verify_logits`` is shaped ``[len(slots)*(k+1), vocab]`` in
    request-then-position order (``verify_batch``'s / the verify graph's
    own output convention). Returns a dict keyed by slot id, each value
    byte-for-byte the same shape as ``determine_accept_reject``'s own
    return dict (``num_accepted``/``committed``/``rejected_at``) -- this is
    a strict re-derivation of the same greedy rule, not a different one:
    for slot ``s`` with drafts ``d = drafts[s]`` (``k+1`` entries, anchor +
    k draft continuations) and per-position argmax predictions ``pred``,
    ``committed = [d[p+1] for p in range(num_accepted)] + [pred[num_accepted]]``
    is exactly what the original sequential version produces in EITHER
    branch (a genuine reject at position ``num_accepted < k``, where
    ``pred[num_accepted]`` is the recovery token; or a full accept where
    ``num_accepted == k`` and ``pred[k]`` is the bonus token) -- verified by
    direct comparison against ``determine_accept_reject`` in
    ``benchmarks/mtp_verify_cudagraph_check.py``.

    Vectorization: ``verify_logits.argmax(dim=-1)`` computes every
    position's greedy prediction in ONE kernel launch (instead of
    ``len(slots)*(k+1)`` separate ``.argmax().item()`` calls); comparing
    against each slot's own draft-continuation tokens and taking a
    cumulative-AND ("still matching every earlier position") over the
    position axis is a second vectorized op that yields ``num_accepted``
    for every slot at once. Only the FINAL small result tensor (shape
    ``[len(slots), k+2]``) is pulled to host via a single ``.tolist()`` --
    everything upstream of that stays on-GPU.
    """
    num_reqs = len(slots)
    predicted = verify_logits.argmax(dim=-1).view(num_reqs, k + 1)  # [num_reqs, k+1], int64
    draft_next = torch.tensor(
        [drafts[s][1:] for s in slots], dtype=predicted.dtype, device=predicted.device
    )  # [num_reqs, k] -- each slot's k candidate continuation tokens (drafts[s][1:])
    matches = predicted[:, :k] == draft_next  # [num_reqs, k] bool
    # True at position p iff every position <= p matched (the greedy
    # "still on the accepted prefix" condition) -- a cumulative product
    # over bools is exactly a running AND.
    still_matching = matches.cumprod(dim=1).bool() if k > 0 else matches.new_zeros((num_reqs, 0), dtype=torch.bool)
    num_accepted = still_matching.sum(dim=1)  # [num_reqs], int64, values 0..k

    # ONE combined host round-trip for the whole batch: num_accepted plus
    # every position's raw prediction (needed to build "committed" below).
    combined = torch.cat([num_accepted.unsqueeze(1), predicted], dim=1)  # [num_reqs, 1 + (k+1)]
    combined_list = combined.tolist()

    decisions: dict[int, dict] = {}
    for i, s in enumerate(slots):
        row = combined_list[i]
        na = row[0]
        pred_row = row[1:]
        committed = [drafts[s][p + 1] for p in range(na)] + [pred_row[na]]
        decisions[s] = {
            "num_accepted": na,
            "committed": committed,
            "rejected_at": na if na < k else None,
        }
    return decisions


class DirectModelRunner:
    """Owns the model, the 4-slot KV/GDN state tensors, and drives forward
    passes directly. This round: single request, slot 0 only."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_slots: int = NUM_SLOTS,
        block_size: int = 16,
        blocks_per_slot: int = 128,
        enable_cudagraph: bool = False,
        enable_block_table: bool = False,
        enable_prefix_cache: bool = False,
        enable_persistent_prefix_cache: bool = False,
        gdn_checkpoint_byte_budget: int = 8 * 2**30,
    ) -> None:
        # 2026-07-18, D3 memory-growth fix: this whole class is a pure
        # inference runtime (never computes a backward pass) but, unlike
        # real vLLM's ``GPUModelRunner`` (whose ``execute_model`` always
        # runs under ``@torch.inference_mode()``), NOTHING in this
        # hand-rolled runner ever disabled autograd -- confirmed by
        # grepping this whole file for "grad" before this fix: zero hits.
        # Every real (non-CUDA-graph) forward call (``_forward``/
        # ``_forward_batch``/``_mtp_forward``/``_mtp_forward_batch``,
        # exercised every round via the eager step-0 fallback whenever
        # active slots' committed lengths are ragged -- the common case at
        # real draft-acceptance rates < 100%) therefore built a full
        # autograd graph rooted at the model's own parameters
        # (``requires_grad=True`` by default, never explicitly frozen by
        # this project's loading path). Root-caused via
        # ``benchmarks/memory_growth_diag.py``: ``torch.cuda
        # .memory_allocated()`` (NOT just ``memory_reserved()``) grew
        # continuously and monotonically round over round with no
        # plateau -- real live-tensor growth, not allocator fragmentation
        # -- reaching 69055 MiB allocated / 97261 MiB nvidia-smi (99.3% of
        # the 97887 MiB card) after 3 W1-S passes, matching the review's
        # reported near-OOM figure almost exactly. ``torch.set_grad_enabled
        # (False)`` (process-global, not a context manager that needs a
        # matching exit -- this runner's process never needs grad) is the
        # standard fix for this exact class of bug and is set as early as
        # possible, before any model construction or forward call.
        torch.set_grad_enabled(False)

        self.vllm_config = vllm_config
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # P0 (2026-07-19, notes/prefix-cache-design.md sec 5 -- "P0 --
        # block-table indirection substrate"): block_table[slot] is a
        # per-logical-slot list of physical block ids, indexed by logical
        # page position. Built unconditionally (cheap: num_slots small
        # Python lists) so the dedicated equivalence tests can check it
        # regardless of enable_block_table; only CONSULTED by the
        # metadata/slot-mapping/CUDA-graph-fill code paths below when
        # enable_block_table=True. Default False preserves every existing
        # caller's behavior byte-for-byte (this project's established
        # feature-flag convention -- see enable_cudagraph above).
        #
        # P1 (2026-07-19, notes/prefix-cache-design.md sec 5 -- "P1 --
        # Dynamic free-list allocator + reference counting"): P0's
        # ``_initial_block_table`` static per-slot partition (every slot
        # pre-populated with its own fixed contiguous blocks_per_slot-sized
        # range, byte-identical to the old arange addressing) is REPLACED
        # here by a real ``BlockPool`` -- a free queue + ref-counting
        # allocator over the shared pool of physical blocks, excluding
        # reserved physical block 0 (INV7). Every slot now starts with an
        # EMPTY block_table and grows it ON DEMAND (see ``_ensure_blocks``,
        # called from every attention-metadata/slot-mapping/CUDA-graph-fill
        # call site that used to just read ``self.block_table[slot]``
        # as-is) as its kv_len actually grows, instead of every slot
        # permanently reserving its whole blocks_per_slot capacity
        # up front. ``_initial_block_table`` itself is kept, UNCHANGED, as
        # a standalone function -- ``benchmarks/prefix_cache_block_table_
        # check.py``'s arange-equivalence check still imports and calls it
        # directly (it never was, and still isn't, about what
        # DirectModelRunner's own initial state looks like) -- it is simply
        # no longer what populates ``self.block_table`` here.
        #
        # Still NO cross-slot sharing this phase: every block, once
        # allocated, has exactly one referencer (``Block.ref_cnt`` is
        # always 0 or 1) -- see ``BlockPool``'s docstring. This is what
        # keeps end-to-end *behavior* identical to P0/pre-P0 while making
        # *placement* genuinely dynamic (a single slot's own blocks may be
        # non-contiguous after any churn of allocate/free cycles -- the
        # thing this phase's own dedicated tests prove the block-table +
        # CUDA-graph path tolerates, not just P0's trivial contiguous case).
        self.enable_block_table = enable_block_table
        # P2 (2026-07-19, notes/prefix-cache-design.md sec 5, "P2 -- Fan-out
        # fork (Pattern A, same-round sharing)"): OPT-IN, default False --
        # preserves every existing caller's behavior byte-for-byte (this
        # project's established feature-flag convention, see enable_block_
        # table/enable_cudagraph above). When True, ``mtp_prefill_fanout_
        # batch`` detects a common token prefix among a same-round admit
        # batch and forks it (leader prefills the shared prefix once,
        # siblings reference the leader's [0, Lc) attention blocks + restore
        # the leader's GDN snapshot + continue-prefill only their own
        # suffixes). Requires ``enable_block_table=True`` (the fork reuses
        # the P1 block-table/ref-counting substrate -- it manipulates
        # ``block_table``/``BlockPool.reference`` directly); with the flag
        # off, OR when fewer than two same-round requests share at least one
        # full block of prefix, ``mtp_prefill_fanout_batch`` falls back to
        # the exact ``mtp_prefill_batch`` path -- byte-identical to P1.
        self.enable_prefix_cache = enable_prefix_cache
        if enable_prefix_cache and not enable_block_table:
            raise ValueError(
                "enable_prefix_cache=True requires enable_block_table=True "
                "(the fan-out fork reuses the P1 block-table/ref-counting substrate)"
            )
        # P3 persistent content-addressed prefix cache (notes/2026-07-19-p3-
        # implementation-plan.md, P3.1): OPT-IN, default False -- preserves
        # every existing caller's behavior byte-for-byte (rollback spine:
        # flag off => byte-for-byte P2; persistent lookup L=0 => P2 fan-out/
        # cold). Requires enable_prefix_cache=True (it builds on the P2 fan-out
        # substrate: block_table/BlockPool/restore_gdn_state(allow_cross_slot)),
        # raising on misconfiguration exactly like the P2 guard above. When on,
        # populate-on-completion writes a content index + persistent GDN
        # checkpoint pool, and mtp_prefill_with_cache serves restore-and-
        # continue hits -- exercised ONLY by the dedicated test in P3.1 (the
        # production prefill entrypoint is untouched this round).
        self.enable_persistent_prefix_cache = enable_persistent_prefix_cache
        if enable_persistent_prefix_cache and not enable_prefix_cache:
            raise ValueError(
                "enable_persistent_prefix_cache=True requires enable_prefix_cache=True "
                "(the persistent cache builds on the P2 fan-out/ref-counting substrate)"
            )
        # kv_cache_dtype is carried in every block's chained hash extra_keys so
        # fp8 vs nvfp4 KV can never collide on the same token prefix (R7).
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.gdn_checkpoint_byte_budget = gdn_checkpoint_byte_budget
        self.block_table: list[list[int]] = [[] for _ in range(num_slots)]
        self.block_pool = BlockPool(
            num_blocks=(num_slots + RESERVED_PHYSICAL_SLOTS) * blocks_per_slot,
            reserved=RESERVED_PHYSICAL_SLOTS,
        )

        # 2026-07-17, Phase 3 (notes/2026-07-17-post-ragged-round-next-steps.md):
        # OPT-IN, default False -- preserves every existing caller's
        # behavior byte-for-byte (every correctness suite in this project
        # constructs a runner with ``num_slots`` sized to its OWN real slot
        # count, no spare capacity reserved for a captured graph's
        # disposable warmup slots; turning this on unconditionally would
        # break them, since ``CapturedBatchDecodeGraph`` permanently
        # reserves the LAST ``batch_size`` logical slots of ``num_slots``
        # for its own warmup -- see that class's docstring -- and several
        # existing tests use those exact slot indices as real,
        # independent reference slots, e.g. ``mtp_batch_verify_check.py``'s
        # ``ref_slots = [4, 5, 6, 7]`` at ``num_slots=8``). A caller that
        # wants ``mtp_verify_and_commit_batch`` to graph-capture its verify
        # forward must pass ``enable_cudagraph=True`` AND size ``num_slots``
        # to at least twice the real concurrency it plans to use (the extra
        # half is reserved warmup capacity, never touched by real request
        # traffic) -- see ``_get_verify_graph``.
        self.enable_cudagraph = enable_cudagraph
        self._verify_graphs: dict[tuple[int, int], "CapturedBatchDecodeGraph"] = {}
        self._draft_step_graphs: dict[tuple[int, int], "CapturedMTPDraftStepGraph"] = {}

        with set_current_vllm_config(vllm_config):
            init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            init_worker_distributed_environment(
                vllm_config, rank=0, distributed_init_method=init_method, local_rank=0
            )
            self.model = get_model(vllm_config=vllm_config)

        sfc = vllm_config.compilation_config.static_forward_context
        self.static_forward_context = sfc
        self.attn_layer_names: list[str] = []
        self.gdn_layer_names: list[str] = []
        for name, layer in sfc.items():
            if hasattr(layer, "get_state_shape"):
                self.gdn_layer_names.append(name)
            else:
                self.attn_layer_names.append(name)
        if not self.attn_layer_names or not self.gdn_layer_names:
            raise RuntimeError(
                f"expected both attention and GDN layers, got "
                f"{len(self.attn_layer_names)} attn / {len(self.gdn_layer_names)} gdn"
            )

        # Real MTP draft model (2026-07-17, Phase 2 / sol's "Option A"),
        # loaded ONLY if the caller configured speculative decoding via
        # build_vllm_config(speculative_config=...). Uses vLLM's own real
        # loading mechanism (load_eagle_model -- also used by vLLM's real
        # MTPSpeculator, not just EAGLE) so embed_tokens/lm_head sharing
        # matches production exactly, nothing hand-rolled. Must load
        # BEFORE _allocate_and_bind_kv_caches() so the draft's own
        # attention layer registers into the SAME static_forward_context
        # this project's existing generic KV-cache-allocation machinery
        # already iterates over -- confirmed by reading vLLM's own
        # DraftModelSpeculator.load_model() (vllm/v1/worker/gpu/spec_decode
        # /speculator.py:153-170), which snapshots attention layer names
        # before/after loading the draft for the exact same reason (there
        # via get_layers_from_vllm_config(..., AttentionLayerBase); here
        # via a direct before/after diff of static_forward_context, which
        # is equivalent since every layer -- attention or GDN -- is
        # registered into that same dict).
        self.mtp_model = None
        self.mtp_attn_layer_names: list[str] = []
        self.num_speculative_tokens: int | None = None
        if vllm_config.speculative_config is not None:
            from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

            names_before = set(sfc.keys())
            with set_current_vllm_config(vllm_config):
                self.mtp_model = load_eagle_model(self.model, vllm_config)
            names_after = set(sfc.keys())
            self.mtp_attn_layer_names = sorted(names_after - names_before)
            if not self.mtp_attn_layer_names:
                raise RuntimeError("loading the MTP draft model registered no new layers")
            for name in self.mtp_attn_layer_names:
                if hasattr(sfc[name], "get_state_shape"):
                    raise RuntimeError(f"unexpected GDN layer in MTP draft model: {name}")
            self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

        self._allocate_and_bind_kv_caches()
        self._allocate_gdn_snapshot_buffers()
        if self.enable_persistent_prefix_cache:
            self._allocate_gdn_checkpoint_pool()
            # P3.2 lockstep eviction (INV3/R5, both directions): when
            # BlockPool._evict_one reclaims a still-hashed attention block, drop
            # the co-keyed GDN checkpoint too. evict_gdn_checkpoint is the
            # reverse direction as well (a budget/pool-driven checkpoint eviction
            # drops the co-keyed attention block's hash if that block is free).
            # Only wired under the flag: blocks are only ever hashed when the
            # persistent cache is on, so _evict_one never invokes this otherwise.
            self.block_pool._on_evict_block = self.evict_gdn_checkpoint

        # Per-slot bookkeeping: attention kv_len (tokens actually written into
        # the paged KV cache) and GDN "has state been initialized" flag.
        self.slot_kv_len = [0] * num_slots
        self.slot_gdn_initialized = [False] * num_slots

        # Per-slot MTP state (explicit fields, not implicit -- 2026-07-17
        # Codex-sol review asked for this precisely so a live multi-round
        # loop can't silently conflate "physically written" with
        # "committed"). ``slot_kv_len``/``slot_gdn_initialized`` above
        # ARE the target's committed_len/init-state -- no separate
        # "committed_len" field is added since that would just be a second
        # name for the same quantity; what's genuinely new is the DRAFT
        # model's own sync length (a different KV cache, tracked
        # separately) and the in-flight pending proposal.
        self.slot_draft_sync_len = [0] * num_slots
        self.slot_pending_draft_tokens: list[list[int] | None] = [None] * num_slots
        self.slot_gdn_snapshot_gen = [0] * num_slots

        # Phase 2 (2026-07-18): per-slot "real committed length from this
        # slot's own last spec-decode GDN verify round" -- read by
        # build_gdn_metadata_spec_batch to select which of the previous
        # round's K+1 dedicated SSM rows holds the valid state to resume
        # from (see _ssm_spec_row/build_gdn_metadata_spec_batch). Bootstrap
        # value is 1 (not 0) for a slot's first-ever spec verify right
        # after a real prefill -- selects column 0, the same physical row
        # the chunked prefill forward itself wrote into. Reset to 1 on
        # ``reset_slot`` and explicitly re-set to 1 in both
        # ``mtp_prefill_batch`` and ``mtp_prefill`` for defense in depth.
        # Phase B (2026-07-18): also read/updated by ``mtp_verify_and_commit``
        # (the singular/looped sibling) -- both production verify paths
        # share this bookkeeping now.
        self.slot_num_accepted_tokens = [1] * num_slots

        # P3 per-slot hash-chain state (notes/2026-07-19-p3-implementation-plan
        # .md, P3.1 step 4), reset in reset_slot. slot_block_hashes[s][i] is the
        # chained BlockHash of block i (depends on all tokens 0..(i+1)*block_size);
        # slot_published_blocks[s] is the count of this slot's blocks already
        # published to the content index (the write cursor for
        # _publish_committed_blocks). Built unconditionally (cheap small Python
        # lists) so the dedicated test can inspect them regardless of the flag;
        # only MUTATED by the persistent-cache write/read paths when the flag is
        # on.
        self.slot_block_hashes: list[list[BlockHash]] = [[] for _ in range(num_slots)]
        self.slot_published_blocks: list[int] = [0] * num_slots
        # P3.2 decode-position populate: the full committed token sequence for
        # each slot (positions [0, slot_kv_len[s])). Decode-produced blocks hash
        # tokens that may straddle the prompt tail + decode head, so the whole
        # sequence must be available. Seeded with the prompt at prefill (inside
        # _publish_committed_blocks), extended on each verify-commit (publish_
        # committed_decode_blocks), reset in reset_slot. Only mutated under the
        # flag; built unconditionally (cheap small lists) like slot_block_hashes.
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]

        # Split-KV parallelism for decode/verify-shaped batched kernel calls
        # (2026-07-17, found via direct source comparison after the
        # coordinator's own nvidia-smi monitoring caught persistently low
        # ~30% GPU utilization in the batched MTP path despite ~95%
        # CUDA-event-measured busy time -- a DIFFERENT dimension from
        # "is a kernel running right now" (busy%) than "how much of the
        # 188-SM array does any ONE kernel call actually occupy"
        # (occupancy), and it is this second dimension that was starved).
        # `build_attention_metadata_batch`'s DEFAULT (this eager path's
        # only caller, until now) derives `kv_split_size` from the
        # request's OWN live kv_len, which forces `max_num_splits == 1`
        # (literally zero split-KV parallelism) unconditionally -- the
        # real, production `SM120GQAMetadataBuilder.build()`
        # (`vllm/v1/attention/backends/sm120_gqa.py`) NEVER does this: it
        # always derives a FIXED `kv_split_size` from a build-time bound
        # (there, `max_model_len`; here, this runner's own real per-slot
        # capacity ceiling `blocks_per_slot * block_size`, the same L the
        # CUDA-graph-safety proof in `build_attention_metadata_batch`'s
        # docstring already establishes as a valid upper bound for every
        # real kv_len this runner will ever see) targeting
        # `_DECODE_TARGET_SPLITS_PER_REQ = 32` splits/request -- a value
        # that project's own sweep (kv_len 2000-131072) found best; this
        # project's OWN (not-yet-wired-into-production) `CapturedBatchDecodeGraph`
        # class used a stale `TARGET_SPLITS = 16` from an earlier round,
        # predating that later tuning -- 64 is used here to match the
        # CURRENT best-known value, not the stale one. Confirmed the SAME
        # underlying kernel is used on both sides of the W1-S native
        # comparison (`launch_test_server.py` defaults to
        # `--attention-backend CUSTOM`, this project's own SM120GQABackend
        # unless `--baseline-flashinfer` is passed) -- so this is a
        # same-kernel, different-launch-configuration gap, not a
        # different-kernel confound.
        _DECODE_TARGET_SPLITS_PER_REQ = 32
        capacity = self.blocks_per_slot * self.block_size
        self.decode_fixed_kv_split_size = max(1, -(-capacity // _DECODE_TARGET_SPLITS_PER_REQ))
        self.decode_fixed_max_num_splits = _DECODE_TARGET_SPLITS_PER_REQ

        self._warmup()

        # Pre-capture every real batch_size this runner's configured spare
        # capacity supports, so the one-time capture cost (a few extra
        # warmup forward passes per size -- see ``CapturedBatchDecodeGraph
        # .capture()``) happens HERE, during construction, not inside the
        # first few timed rounds of a real measurement (matches this
        # method's own "pay setup cost once at construction" philosophy).
        # Requires MTP to be configured (``num_speculative_tokens`` is
        # unknown otherwise, and this graph is only ever used from
        # ``mtp_verify_and_commit_batch``).
        if self.enable_cudagraph and self.num_speculative_tokens is not None:
            if self.num_slots >= 2 * self.num_slots:
                self._precapture_verify_graphs()
                self._precapture_draft_step_graphs()

    def _precapture_verify_graphs(self) -> None:
        # 2026-07-18, Phase 2 CUDA-graph reconciliation: only qo_len=k+1 is
        # ever needed now. The old rationale for precapturing every
        # qo_len in 1..k+1 (the recompute-forward graph-reuse path, which
        # needed a graph at whatever committed_len 1..k a ragged recompute
        # group happened to land on) no longer applies -- Phase 2 removed
        # the separate recompute forward entirely, so
        # mtp_verify_and_commit_batch's verify step now ALWAYS replays at
        # exactly qo_len=k+1, regardless of each slot's own accept/reject
        # outcome (see that method's docstring). Precapturing the other
        # qo_len values would just be wasted capture time/GPU memory for
        # shapes nothing calls anymore.
        max_batch = self.num_slots // 2
        for batch_size in range(1, max_batch + 1):
            self._get_verify_graph(batch_size, self.num_speculative_tokens + 1)

    def _precapture_draft_step_graphs(self) -> None:
        # 2026-07-17, Phase 3 round 2: precapture qo_len=1 (the K-1
        # continuation steps) AND every qo_len in 1..k+1 (step 0's own
        # shape for the full-accept group -- always k+1 -- and the
        # recompute group's uniform special case -- 1..k) so NEITHER step
        # 0 nor the continuation loop ever lazily captures during a real
        # timed round.
        max_batch = self.num_slots // 2
        for batch_size in range(1, max_batch + 1):
            for qo_len in range(1, self.num_speculative_tokens + 2):
                self._get_draft_step_graph(batch_size, qo_len)

    def _get_draft_step_graph(self, batch_size: int, qo_len: int = 1) -> "CapturedMTPDraftStepGraph | None":
        """Lazily construct + capture (and cache, keyed by
        ``(batch_size, qo_len)``) a ``CapturedMTPDraftStepGraph`` for the
        MTP draft model's qo_len=1 continuation step OR (2026-07-17,
        generalized) step 0's resync when its own ``num_new_tokens`` is
        uniform -- see that class's docstring. Same deliberate
        ``None``-on-insufficient-capacity fallback contract as
        ``_get_verify_graph``."""
        key = (batch_size, qo_len)
        cached = self._draft_step_graphs.get(key)
        if cached is not None:
            return cached
        if batch_size > self.num_slots or self.mtp_model is None:
            return None
        if self.num_slots >= 2 * batch_size:
            graph = CapturedMTPDraftStepGraph(self, batch_size=batch_size, qo_len=qo_len)
            graph.capture()
        else:
            warmup_slots = list(range(batch_size))
            graph = CapturedMTPDraftStepGraph(self, batch_size=batch_size, qo_len=qo_len,
                                              warmup_slots=warmup_slots)
            graph.capture()
        self._draft_step_graphs[key] = graph
        return graph

    def precapture_cuda_graphs(self, batch_sizes: list[int] | None = None,
                               qo_lens: list[int] | None = None) -> None:
        """Pre-capture CUDA graphs during initialization, before any real
        traffic. Uses real slots 0..batch_size-1 for warmup, then resets
        them so they are fresh for real traffic. This eliminates the need
        for permanently reserved warmup slots (which doubled KV cache
        memory)."""
        if not self.enable_cudagraph:
            return
        if batch_sizes is None:
            batch_sizes = [self.num_slots]
        if qo_lens is None:
            qo_lens = [1]
            if self.num_speculative_tokens is not None:
                qo_lens.append(self.num_speculative_tokens + 1)
        draft_qo_lens = qo_lens
        if self.mtp_model is not None and self.num_speculative_tokens is not None:
            draft_qo_lens = list(range(1, self.num_speculative_tokens + 2))
        for bs in batch_sizes:
            if bs > self.num_slots:
                raise ValueError(f"batch_size {bs} exceeds num_slots {self.num_slots}")
            warmup_slots = list(range(bs))
            for qo in qo_lens:
                key = (bs, qo)
                if key not in self._verify_graphs:
                    graph = CapturedBatchDecodeGraph(
                        self, bs, qo_len=qo, warmup_slots=warmup_slots,
                    )
                    graph.capture()
                    self._verify_graphs[key] = graph
        if self.mtp_model is not None:
            for bs in batch_sizes:
                warmup_slots = list(range(bs))
                for qo in draft_qo_lens:
                    key = (bs, qo)
                    if key not in self._draft_step_graphs:
                        graph = CapturedMTPDraftStepGraph(
                            self, bs, qo_len=qo, warmup_slots=warmup_slots,
                        )
                        graph.capture()
                        self._draft_step_graphs[key] = graph
        for bs in batch_sizes:
            for slot in range(bs):
                if self.slot_kv_len[slot] != 0:
                    self.reset_slot(slot)

    def _get_verify_graph(self, batch_size: int, qo_len: int) -> "CapturedBatchDecodeGraph | None":
        """Lazily construct + capture (and cache, keyed by
        ``(batch_size, qo_len)``) a ``CapturedBatchDecodeGraph`` for the
        target model's verify forward. Returns ``None`` -- a deliberate,
        documented eager-fallback signal, NOT an error -- when this runner
        wasn't configured with enough spare capacity
        (``num_slots >= 2*batch_size``) to reserve that graph's own
        disposable warmup slots. This is the expected, correct outcome for
        every existing (non-cudagraph) correctness suite in this project
        (``enable_cudagraph`` defaults to ``False`` there, so this method is
        never even called), and also the correct outcome for a genuinely
        unusual batch_size a graph-enabled caller never pre-captured (e.g.
        one bigger than ``num_slots // 2`` -- cannot happen from
        ``_precapture_verify_graphs``'s own range, but this method stays
        safe if called with an out-of-range size directly).

        Capturing a NEW graph resets its own reserved warmup slots
        (``runner.reset_slot``) immediately afterward -- ``capture()``
        requires its warmup slots to be fresh (``slot_kv_len == 0``), and
        different ``batch_size`` graphs' reserved-slot RANGES overlap
        (``CapturedBatchDecodeGraph`` reserves the LAST ``batch_size``
        logical slots of ``num_slots``, so e.g. batch_size=2 and
        batch_size=4 graphs share slots ``num_slots-2 .. num_slots-1``) --
        without this reset, capturing a second graph whose reserved range
        overlaps a previously-captured graph's would hit that freshness
        check and fail. This is safe because a graph's reserved slots are
        NEVER touched again after its own ``capture()`` call returns (never
        passed to ``replay()`` or any other runner method) -- resetting
        them costs nothing but bookkeeping."""
        key = (batch_size, qo_len)
        cached = self._verify_graphs.get(key)
        if cached is not None:
            return cached
        if batch_size > self.num_slots:
            return None
        if self.num_slots >= 2 * batch_size:
            graph = CapturedBatchDecodeGraph(self, batch_size=batch_size, qo_len=qo_len)
            graph.capture()
            for s in graph._warmup_slots:
                self.reset_slot(s)
        else:
            warmup_slots = list(range(batch_size))
            graph = CapturedBatchDecodeGraph(self, batch_size=batch_size, qo_len=qo_len,
                                             warmup_slots=warmup_slots)
            graph.capture()
        self._verify_graphs[key] = graph
        return graph

    def _warmup(self) -> None:
        """Real vLLM always runs a profiling/warmup forward before serving
        (see gpu_model_runner.py's warmup pass, and this project's own
        server logs: "Initial profiling/warmup run took N s"). Motivated by
        a real, isolated repro (see notes/direct-model-runner-design.md's
        "deep dive on the conv_state lead" section): causal_conv1d_fn's
        Triton kernel returns an all-zero result on its first-ever call in
        a process, in complete isolation, unrelated to this runtime's code.
        Kept here since it mirrors real vLLM's own behavior and cannot
        hurt, but -- reported honestly -- this alone does NOT fix the real
        model's wrong output (verified: neither a 1-token nor a
        shape-matched 5-token warmup changed the observed wrong completion
        for "The capital of France is"). The cold-start bug is real but
        evidently not the whole story; see the design doc for the
        follow-up isolated tests that show a messier, not-yet-characterized
        pattern (interleaved shapes don't self-correct the way repeating
        one shape does) and the next debugging steps."""
        try:
            self.prefill(0, [0, 0, 0, 0, 0])
        finally:
            self.reset_slot(0)

    def _allocate_and_bind_kv_caches(self) -> None:
        self.kv_caches = allocate_fixed_slot_kv_caches(
            self.static_forward_context,
            self.vllm_config,
            self.device,
            num_slots=self.num_slots,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            num_speculative_tokens=self.num_speculative_tokens or 0,
        )

    def _allocate_gdn_snapshot_buffers(self) -> None:
        """Preallocated, GPU-resident, fixed-address storage for
        ``snapshot_gdn_state``/``restore_gdn_state`` (2026-07-17, Phase 1 of
        ``notes/2026-07-17-post-ragged-round-next-steps.md``). Replaces the
        old per-call ``.detach().to("cpu", copy=True)`` -- Phase 0's real
        ``nsys`` ledger (that doc's section 7) measured this mechanism at
        89-117ms/round of pageable D2H/H2D memcpy-engine time alone, plus a
        comparable amount of host-dispatch gap in the same phases (~30-31%
        of round wall time combined, present in every round -- snapshot
        happens unconditionally for all active slots).

        Sizing rationale (verified against the real call pattern before
        relying on it, per this round's own instructions -- both
        ``mtp_verify_and_commit`` and ``mtp_verify_and_commit_batch`` snap
        each slot in the list AT MOST ONCE per round, and any restore for
        that slot happens later in that SAME round, before the next round's
        snapshot call for that slot can be issued): at most ONE snapshot
        per logical slot is ever outstanding at a time. One buffer entry per
        logical slot (indexed 0..num_slots-1) is therefore sufficient --
        NOT a literal ping-pong double buffer (which would double the VRAM
        cost to ~1.2GB); this is deliberately the plan doc's "~604MB"
        estimate, which already assumed exactly this one-copy-per-slot
        sizing (confirmed against Phase 0's own measured D2H byte count,
        ~604MB for a 4-slot round). The persistent buffer is safe to reuse
        round-over-round without an explicit double-buffer/generation-aware
        allocation scheme because everything here runs on ONE CUDA stream
        in strict Python-issued order: a later round's snapshot() write for
        slot S can only be enqueued after every earlier statement that
        reads slot S's snapshot (i.e. that round's own restore() call, if
        any) has already been issued -- CUDA's own per-stream FIFO
        ordering, not an extra synchronization primitive, is what makes
        this correct. The three safety invariants this class already
        enforces (slot-id tag, generation counter, consumed-once flag) are
        UNCHANGED and still checked before any tensor data is read on
        restore -- they continue to guard against a caller holding a STALE
        snapshot object across rounds, which would otherwise now silently
        alias newer data through the same buffer slot (the checks reject
        it before that data is ever used, exactly as before).

        Indexed directly by LOGICAL slot (0..num_slots-1), unlike
        ``kv_caches`` (which reserves physical index 0 -- see
        ``RESERVED_PHYSICAL_SLOTS``/``_physical_slot``): that reservation
        works around a real vLLM physical-block-addressing convention this
        private buffer is not subject to, so no such offset/reservation is
        needed here.

        Fixed-address discipline (never reallocated after ``__init__``,
        only ever written into via ``copy_``) matches this file's other
        persistent GPU buffers (see ``CapturedBatchDecodeGraph``'s class
        docstring) -- this code path does not currently run inside any CUDA
        graph capture region (``mtp_verify_and_commit``/``_batch`` are
        eager-only; ``CapturedBatchDecodeGraph`` is a separate, not-yet-
        wired-in mechanism per Phase 3 of the same plan doc), but following
        the same discipline now means Phase 3 does not have to revisit this
        buffer's allocation strategy later if GDN snapshot/restore is ever
        folded into a captured graph."""
        self.gdn_snapshot_conv: dict[str, torch.Tensor] = {}
        self.gdn_snapshot_ssm: dict[str, torch.Tensor] = {}
        for name in self.gdn_layer_names:
            conv_state, ssm_state = self.kv_caches[name]
            self.gdn_snapshot_conv[name] = torch.zeros(
                (self.num_slots, *conv_state.shape[1:]),
                dtype=conv_state.dtype,
                device=self.device,
            )
            self.gdn_snapshot_ssm[name] = torch.zeros(
                (self.num_slots, *ssm_state.shape[1:]),
                dtype=ssm_state.dtype,
                device=self.device,
            )

    def _allocate_gdn_checkpoint_pool(self) -> None:
        # Persistent full-stack GDN checkpoint pool (P3.1, notes/2026-07-19-p3-
        # implementation-plan.md step 3; R8-aware from day one). SEPARATE from
        # the live per-slot gdn_snapshot_* buffers above (those keep their
        # existing MTP role, untouched). Each checkpoint is a full 48-layer
        # (conv_state, ssm_state) snapshot at an exact prefix boundary -- the
        # recurrent state, so its size is INDEPENDENT of prefix length (~151 MB
        # measured, the ~604 MB/4-slot figure). Fixed-address discipline: a pool
        # slot's per-layer tensors are allocated once (lazily on first use) and
        # never reallocated; the pool is bounded by max_checkpoints =
        # byte_budget // per_checkpoint_bytes (default 8 GB => ~53 slots). Only
        # called when enable_persistent_prefix_cache is on, so the default-off
        # production path allocates nothing here (byte-for-byte P2).
        self._gdn_ckpt_conv_shape: dict[str, tuple] = {}
        self._gdn_ckpt_ssm_shape: dict[str, tuple] = {}
        self._gdn_ckpt_conv_dtype: dict[str, torch.dtype] = {}
        self._gdn_ckpt_ssm_dtype: dict[str, torch.dtype] = {}
        per_checkpoint_bytes = 0
        for name in self.gdn_layer_names:
            conv_state, ssm_state = self.kv_caches[name]
            # Column-0 row shapes (what snapshot_gdn_state captures): one row
            # per layer, shape shape[1:]. The K spec rows are per-slot scratch,
            # never cached (INV4 / MambaSpec.supports_eagle_cache_peek=False).
            self._gdn_ckpt_conv_shape[name] = tuple(conv_state.shape[1:])
            self._gdn_ckpt_ssm_shape[name] = tuple(ssm_state.shape[1:])
            self._gdn_ckpt_conv_dtype[name] = conv_state.dtype
            self._gdn_ckpt_ssm_dtype[name] = ssm_state.dtype
            conv_elems = 1
            for d in conv_state.shape[1:]:
                conv_elems *= int(d)
            ssm_elems = 1
            for d in ssm_state.shape[1:]:
                ssm_elems *= int(d)
            per_checkpoint_bytes += (
                conv_elems * conv_state.element_size() + ssm_elems * ssm_state.element_size()
            )
        self.gdn_ckpt_per_checkpoint_bytes = per_checkpoint_bytes
        self.gdn_ckpt_max_checkpoints = max(
            1, self.gdn_checkpoint_byte_budget // max(1, per_checkpoint_bytes)
        )
        # Per-layer pool-slot tensor lists, lazily allocated (None until first
        # materialize into that slot), bounded by gdn_ckpt_max_checkpoints.
        self.gdn_ckpt_conv: dict[str, list[torch.Tensor | None]] = {
            name: [None] * self.gdn_ckpt_max_checkpoints for name in self.gdn_layer_names
        }
        self.gdn_ckpt_ssm: dict[str, list[torch.Tensor | None]] = {
            name: [None] * self.gdn_ckpt_max_checkpoints for name in self.gdn_layer_names
        }
        # Meta keyed by the boundary tail block id ("key"): each entry records
        # {key, hash_value, num_tokens, pool_slot, bytes, __slot__}. The
        # hash_value tag is what makes a wrong-prefix restore REJECTED, not used
        # (R1). _gdn_ckpt_by_hash is the reverse index reconcile_prefix_hit
        # probes (sec 3.4 GDN boundary G). _gdn_ckpt_free is the free pool-slot
        # stack; _gdn_ckpt_lru (OrderedDict, oldest-first) is maintained now and
        # hardened into byte-budget eviction in P3.2 (here it is only the
        # bounded-pool safety valve when the pool is full).
        self.gdn_ckpt_meta: dict[int, dict] = {}
        self._gdn_ckpt_by_hash: dict[int, int] = {}
        self._gdn_ckpt_free: list[int] = list(range(self.gdn_ckpt_max_checkpoints))
        self._gdn_ckpt_lru: OrderedDict[int, None] = OrderedDict()

    def _gdn_ckpt_alloc_slot(self) -> int:
        # Pop a free pool slot, or -- only if the bounded pool is full -- evict
        # the LRU checkpoint to reclaim one (safety valve keeping the pool
        # bounded; P3.2 replaces this with real byte-budget LRU eviction in
        # lockstep with the attention index).
        if self._gdn_ckpt_free:
            return self._gdn_ckpt_free.pop()
        lru_key = next(iter(self._gdn_ckpt_lru))
        evicted_slot = self.gdn_ckpt_meta[lru_key]["pool_slot"]
        self.evict_gdn_checkpoint(lru_key)
        return evicted_slot

    def materialize_gdn_checkpoint(
        self, slot: int, key: int, hash_value: int, num_tokens: int
    ) -> None:
        # foreach_copy the 48-layer live state at _physical_slot(slot) (the
        # column-0 conv/ssm rows the just-completed forward wrote) INTO a free
        # pool slot, tagged with hash_value (R1's checkpoint-hash tag). The
        # source is read-only. Idempotent on (key, hash_value): re-materializing
        # the same boundary is a no-op. Mirrors snapshot_gdn_state's foreach_copy
        # (same column-0 rows), but into the PERSISTENT pool instead of the live
        # per-slot snapshot buffer.
        existing = self.gdn_ckpt_meta.get(key)
        if existing is not None:
            if existing["hash_value"] == hash_value:
                self._gdn_ckpt_lru.move_to_end(key)
                return
            # Same block id reused for a different prefix (post-eviction): drop
            # the stale entry first.
            self.evict_gdn_checkpoint(key)
        # P3.2 byte-budget LRU (R8): if adding this checkpoint would exceed
        # gdn_checkpoint_byte_budget, evict LRU checkpoints (lockstep with their
        # keyed attention blocks) until it fits. Checkpoints exist only at
        # chunk + completion boundaries, so this is a bounded, rare operation.
        self._evict_gdn_checkpoints_for_budget(self.gdn_ckpt_per_checkpoint_bytes)
        pool_slot = self._gdn_ckpt_alloc_slot()
        physical = _physical_slot(slot)
        conv_dsts, ssm_dsts, conv_srcs, ssm_srcs = [], [], [], []
        for name in self.gdn_layer_names:
            if self.gdn_ckpt_conv[name][pool_slot] is None:
                self.gdn_ckpt_conv[name][pool_slot] = torch.zeros(
                    self._gdn_ckpt_conv_shape[name],
                    dtype=self._gdn_ckpt_conv_dtype[name],
                    device=self.device,
                )
                self.gdn_ckpt_ssm[name][pool_slot] = torch.zeros(
                    self._gdn_ckpt_ssm_shape[name],
                    dtype=self._gdn_ckpt_ssm_dtype[name],
                    device=self.device,
                )
            conv_state, ssm_state = self.kv_caches[name]
            conv_dsts.append(self.gdn_ckpt_conv[name][pool_slot])
            ssm_dsts.append(self.gdn_ckpt_ssm[name][pool_slot])
            conv_srcs.append(conv_state[physical])
            ssm_srcs.append(ssm_state[physical])
        torch._foreach_copy_(conv_dsts, conv_srcs)
        torch._foreach_copy_(ssm_dsts, ssm_srcs)
        self.gdn_ckpt_meta[key] = {
            "key": key,
            "hash_value": hash_value,
            "num_tokens": num_tokens,
            "pool_slot": pool_slot,
            "bytes": self.gdn_ckpt_per_checkpoint_bytes,
            "__slot__": slot,
        }
        self._gdn_ckpt_by_hash[hash_value] = key
        self._gdn_ckpt_lru[key] = None
        self._gdn_ckpt_lru.move_to_end(key)

    def checkpoint_view(self, key: int) -> dict | None:
        # Return a snapshot-shaped dict for the checkpoint at boundary block
        # "key", consumable UNCHANGED by the EXISTING restore_gdn_state(dest,
        # view, allow_cross_slot=True) (P3 writes no second restore). The
        # __slot__ tag is the SOURCE slot whose state was checkpointed (the
        # cross-slot path only requires it to be non-None). Returns None if no
        # checkpoint exists for key. Revives the entry in the LRU.
        meta = self.gdn_ckpt_meta.get(key)
        if meta is None:
            return None
        pool_slot = meta["pool_slot"]
        view: dict = {"__slot__": meta["__slot__"]}
        for name in self.gdn_layer_names:
            view[name] = (
                self.gdn_ckpt_conv[name][pool_slot],
                self.gdn_ckpt_ssm[name][pool_slot],
            )
        self._gdn_ckpt_lru.move_to_end(key)
        return view

    def evict_gdn_checkpoint(self, key: int) -> None:
        # Drop the checkpoint at boundary block "key": remove its meta + hash
        # index + LRU entry and return its pool slot to the free stack (the
        # pool-slot tensors stay allocated for reuse).
        #
        # LOCKSTEP, reverse direction (INV3/R5): the checkpoint is keyed by the
        # attention tail block id == key, so dropping the checkpoint ALSO drops
        # that block's hash -- but ONLY if the block is free (ref_cnt == 0). The
        # two halves then never disagree about what is cached: a future
        # reconcile finds A shrunk below this boundary (compute miss L=0), never
        # a ghost attention hit with no GDN state. If the block is ref_cnt > 0
        # (an active slot still references it), its hash stays -- losing only the
        # checkpoint, which merely turns a future would-be hit into a safe
        # compute miss (L = G <= A still holds). The forward direction
        # (BlockPool._evict_one reclaiming the attention block) clears block_hash
        # BEFORE calling here, so this reverse step is a no-op there.
        meta = self.gdn_ckpt_meta.pop(key, None)
        if meta is None:
            return
        self._gdn_ckpt_by_hash.pop(meta["hash_value"], None)
        self._gdn_ckpt_lru.pop(key, None)
        self._gdn_ckpt_free.append(meta["pool_slot"])
        if 0 <= key < self.block_pool.num_blocks:
            block = self.block_pool.blocks[key]
            if block.ref_cnt == 0 and block.block_hash is not None:
                self.block_pool.hash_to_block.pop(block.block_hash.value, None)
                block.block_hash = None

    def _evict_gdn_checkpoints_for_budget(self, incoming_bytes: int) -> None:
        # P3.2 byte-budget LRU (R8): evict LRU checkpoints (oldest-first per
        # _gdn_ckpt_lru) until adding ``incoming_bytes`` fits within
        # gdn_checkpoint_byte_budget. Each eviction is lockstep (evict_gdn_
        # checkpoint drops the co-keyed attention block's hash if free). Pure
        # bookkeeping (no tensor ops), so it is unit-testable without a GPU.
        # Never evicts the entry about to be (re-)materialized: callers handle
        # the idempotent/stale-key cases before invoking this.
        total_bytes = sum(meta["bytes"] for meta in self.gdn_ckpt_meta.values())
        while (
            self.gdn_ckpt_meta
            and total_bytes + incoming_bytes > self.gdn_checkpoint_byte_budget
        ):
            lru_key = next(iter(self._gdn_ckpt_lru))
            total_bytes -= self.gdn_ckpt_meta[lru_key]["bytes"]
            self.evict_gdn_checkpoint(lru_key)


    def _ensure_blocks(self, slot: int, kv_len_needed: int) -> None:
        """P1 (notes/prefix-cache-design.md sec 5): grow
        ``self.block_table[slot]`` on demand from ``self.block_pool`` so it
        holds at least ``ceil(kv_len_needed / self.block_size)`` physical
        block ids -- called from every code path that is about to build
        attention metadata / a slot-mapping / a CUDA-graph fill for a write
        or read up to position ``kv_len_needed`` (single-request and
        batched target-model forward, single-request and batched MTP
        draft-model forward, both captured-graph ``_fill_buffers``
        methods). A no-op when the table already covers the request -- the
        common per-token-decode-step case, which only needs a fresh
        physical block once every ``block_size`` tokens, not every call.

        Every call site gates on ``self.enable_block_table`` before calling
        this (matching this file's existing per-call-site flag-branch
        convention) -- this method itself always consults ``self
        .block_pool`` unconditionally once called, it does not re-check the
        flag.

        Raises the same ``RuntimeError`` message shape as
        ``build_attention_metadata``/``_batch``'s own capacity check when
        ``kv_len_needed`` would need more than ``self.blocks_per_slot``
        pages -- checked here too (not just left to the metadata builder to
        catch after the fact) so a request that will be rejected anyway
        never consumes a block from the shared pool first."""
        num_pages_needed = (kv_len_needed + self.block_size - 1) // self.block_size
        if num_pages_needed > self.blocks_per_slot:
            raise RuntimeError(
                f"slot {slot} kv_len {kv_len_needed} exceeds this slot's "
                f"{self.blocks_per_slot * self.block_size}-token capacity"
            )
        table = self.block_table[slot]
        grow_by = num_pages_needed - len(table)
        if grow_by > 0:
            table.extend(self.block_pool.allocate(grow_by))

    def _attention_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> SM120GQAMetadata:
        if self.enable_block_table:
            self._ensure_blocks(slot, self.slot_kv_len[slot] + num_new_tokens)
        return build_attention_metadata(
            prior_kv_len=self.slot_kv_len[slot],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
            block_table=self.block_table[slot] if self.enable_block_table else None,
        )

    def _gdn_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> GDNAttentionMetadata:
        return build_gdn_metadata(
            slot_initialized=self.slot_gdn_initialized[slot],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            device=self.device,
        )

    def _slot_mapping(self, slot: int, start_pos: int, num_new_tokens: int) -> torch.Tensor:
        """Flat per-token KV-cache write index: block_id * block_size + offset
        -- the same convention vLLM's own paged attention backends use (see
        attention.py's do_kv_cache_update, which reads this from
        ``forward_context.slot_mapping[layer_name]``, NOT from
        ``attn_metadata`` -- easy to miss, and missing it means K/V are never
        written into the cache at all)."""
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )
        if self.enable_block_table:
            table = self.block_table[slot]
            block_ids = torch.tensor(
                [table[p // self.block_size] for p in range(start_pos, start_pos + num_new_tokens)],
                dtype=torch.long,
                device=self.device,
            )
        else:
            first_block = _physical_slot(slot) * self.blocks_per_slot
            block_ids = first_block + positions // self.block_size
        offsets = positions % self.block_size
        return (block_ids * self.block_size + offsets).to(torch.long)

    def _forward(
        self,
        slot: int,
        token_ids: list[int],
        start_pos: int,
        *,
        is_decode: bool,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        num_new_tokens = len(token_ids)
        attn_meta = self._attention_metadata(
            slot, num_new_tokens=num_new_tokens, is_decode=is_decode
        )
        gdn_meta = self._gdn_metadata(
            slot, num_new_tokens=num_new_tokens, is_decode=is_decode
        )
        attn_metadata_dict = {name: attn_meta for name in self.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in self.gdn_layer_names})
        slot_mapping = self._slot_mapping(slot, start_pos, num_new_tokens)
        slot_mapping_dict = {name: slot_mapping for name in self.attn_layer_names}

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = self.model.forward(input_ids, positions)
        # 2026-07-17, Phase 3 (notes/2026-07-17-post-ragged-round-next-steps.md):
        # the two ``torch.cuda.synchronize()`` calls that used to bracket
        # ``compute_logits`` here were removed -- they block the HOST
        # (Python) thread until every queued GPU op finishes, but neither
        # call was ever needed for CORRECTNESS: ``model.forward()`` and
        # ``compute_logits()`` are both issued on the SAME (default) CUDA
        # stream, so CUDA's own per-stream FIFO ordering already guarantees
        # ``compute_logits`` reads ``hidden_states`` only after `forward()`'s
        # kernels have written it -- exactly the same reasoning
        # ``CapturedBatchDecodeGraph.replay()``'s docstring already
        # established for removing ITS blanket sync (see that class,
        # 2026-07-17 correctness-review round). Any caller that actually
        # needs the values host-side (``.item()``/``.cpu()``/``torch.equal``)
        # already forces an implicit, narrowly-scoped sync at that read --
        # a blanket device-wide sync here was pure per-call dispatch
        # overhead (Phase 0's ``nsys`` ledger measured 3634 kernels/round in
        # the verify phase alone; every method in this file's hot path used
        # to insert two of these), not a safety requirement.
        logits = self.model.compute_logits(hidden_states)

        self.slot_kv_len[slot] += num_new_tokens
        self.slot_gdn_initialized[slot] = True
        if return_hidden:
            return logits, hidden_states
        return logits

    def prefill(self, slot: int, prompt_token_ids: list[int]) -> int:
        """Run the prompt through the model; returns the greedy next token id."""
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
        logits = self._forward(slot, prompt_token_ids, start_pos=0, is_decode=False)
        return int(logits[-1].argmax(dim=-1).item())

    def decode(self, slot: int, token_id: int) -> int:
        """Consume one token, return the greedy next token id."""
        start_pos = self.slot_kv_len[slot]
        logits = self._forward(slot, [token_id], start_pos=start_pos, is_decode=True)
        return int(logits[-1].argmax(dim=-1).item())

    def _slot_mapping_batch(
        self, slots: list[int], kv_lengths: list[int], qo_len: int | list[int] = 1
    ) -> torch.Tensor:
        """Batched analogue of ``_slot_mapping``: each request contributes
        ``qo_len`` new tokens starting at its own ``kv_lengths[i]``,
        flattened in the SAME per-request-contiguous order ``_forward_batch``
        uses for ``input_ids``/``positions`` (request 0's ``qo_len`` tokens,
        then request 1's, ...). At ``qo_len=1`` this reduces exactly to the
        previously-verified one-position-per-request mapping. ``qo_len`` may
        also be a per-slot RAGGED list (2026-07-17, mirrors
        ``build_attention_metadata_batch``'s identical generalization) --
        a scalar broadcasts to a uniform list, so every existing call site
        is unaffected."""
        num_reqs = len(slots)
        qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
        positions = [kv_len + j for kv_len, qo in zip(kv_lengths, qo_lens) for j in range(qo)]
        slots_per_token = [slot for slot, qo in zip(slots, qo_lens) for _ in range(qo)]
        if self.enable_block_table:
            block_ids = torch.tensor(
                [
                    self.block_table[slot][pos // self.block_size]
                    for slot, pos in zip(slots_per_token, positions)
                ],
                dtype=torch.long,
                device=self.device,
            )
        else:
            block_ids = torch.tensor(
                [
                    _physical_slot(slot) * self.blocks_per_slot + pos // self.block_size
                    for slot, pos in zip(slots_per_token, positions)
                ],
                dtype=torch.long,
                device=self.device,
            )
        offsets = torch.tensor(
            [pos % self.block_size for pos in positions], dtype=torch.long, device=self.device
        )
        return block_ids * self.block_size + offsets

    def _forward_batch(
        self,
        slot_ids: list[int],
        token_ids,
        kv_lengths: list[int],
        *,
        qo_len: int | list[int] = 1,
        commit: bool = True,
        return_hidden: bool = False,
        is_decode: bool = True,
        fixed_kv_split_size: int | None = None,
        fixed_max_num_splits: int | None = None,
        gdn_spec_num_accepted_tokens_prev: list[int] | None = None,
        logits_last_position_only: bool = False,
    ) -> torch.Tensor:
        """Real batched decode/verify: ONE batched attention/GDN metadata
        object and ONE ``model.forward()`` call covering every listed slot
        -- not a Python loop calling ``_forward``/``decode`` per slot.
        ``kv_lengths`` is the caller-asserted prior KV length (before this
        step's new tokens) for each slot; cross-checked against this
        runner's own ``self.slot_kv_len`` bookkeeping to catch drift early
        rather than silently addressing the wrong cache rows.

        ``qo_len=1`` (the default, unchanged from the original decode-only
        batch path): ``token_ids`` is a flat list, one token id per slot.
        ``qo_len>1`` (MTP/speculative-decode verify, uniform across the
        batch): ``token_ids`` is a list of per-slot token-id lists, each of
        length ``qo_len`` -- the K draft tokens + 1 bonus-position
        placeholder being verified in one batched call.
        Returns logits shaped ``[num_reqs * qo_len, vocab]``, flattened in
        request-then-position order (request 0's qo_len rows, then request
        1's, ...) -- the same order ``SM120GQAImpl.forward()``'s own
        ``q_decode.reshape(num_reqs, qo_len, ...)`` expects.

        ``commit`` (default ``True``, preserving the original decode_batch
        behavior exactly): whether to advance ``self.slot_kv_len`` by
        ``qo_len`` for every listed slot. The forward pass ALWAYS
        physically writes K/V for all ``qo_len`` positions regardless of
        this flag -- ``commit`` only controls this method's own
        bookkeeping. Real MTP verify calls (``verify_batch``/
        ``verify_batch_spec``) pass ``commit=False``, since the actual
        committed length is not known until the caller's accept/reject
        decision runs on the returned logits (2026-07-17, fixing the exact
        "physically-written vs. committed" conflation Codex-sol's review
        flagged) -- the caller (``mtp_verify_and_commit``/``_batch``) is
        responsible for advancing ``slot_kv_len`` by the REAL committed
        length afterward. Attention's own KV needs no explicit rollback
        either way (content/position addressed -- positions beyond the
        real committed length are simply never read again).

        **2026-07-18, Phase B update**: GDN's recurrent state used to need
        an explicit ``snapshot_gdn_state``/``restore_gdn_state`` + a real
        recompute-forward repair on a non-full-accept outcome -- that was
        true for both ``mtp_verify_and_commit`` and
        ``mtp_verify_and_commit_batch`` through 2026-07-18, then only for
        the singular path (Phase 2 migrated the batched path off it), and
        as of Phase B is no longer true for EITHER production verify path:
        both now go through the real spec-decode GDN mechanism
        (``gdn_spec_num_accepted_tokens_prev`` below), under which the
        recurrent state's per-position OUTPUT is already causally valid
        for every candidate position regardless of which are later
        accepted -- only the STATE COMMIT (which physical row survives to
        be read next round) is acceptance-aware, so no rollback is ever
        needed. ``snapshot_gdn_state``/``restore_gdn_state`` themselves are
        retained as tested, standalone primitives (still directly exercised
        by ``benchmarks/mtp_gdn_rollback_check.py`` and several other
        diagnostics -- see ``mtp_verify_and_commit``'s docstring), just no
        longer called from any production verify path.

        ``is_decode`` (2026-07-17 addition, default ``True`` preserving
        ``decode_batch``/``verify_batch``'s existing behavior byte-for-byte):
        forwarded to ``build_attention_metadata_batch``'s own ``is_decode``
        parameter -- see that function's docstring for the real gap this
        closes (``decode_qo_len`` must be 0 for a genuine chunked/prefix
        PREFILL call, not ``qo_len`` unconditionally). Only
        ``mtp_prefill_batch`` passes ``is_decode=False`` explicitly, for its
        genuine target-model prefill forward.

        ``fixed_kv_split_size``/``fixed_max_num_splits`` (both ``None`` by
        default, forwarded as-is to ``build_attention_metadata_batch``):
        without these, that function's default branch derives
        ``kv_split_size`` from this call's own live kv_len, which forces
        ``max_num_splits == 1`` -- literally zero split-KV parallelism.
        Real MTP callers now pass ``self.decode_fixed_kv_split_size``/
        ``self.decode_fixed_max_num_splits`` (computed once in
        ``__init__``, matching native's production
        ``SM120GQAMetadataBuilder``'s own fixed-from-build-time-bound
        derivation) so the SAME decode/verify kernel gets real split-KV
        parallelism here too -- see ``__init__``'s comment for the full
        story (2026-07-17, found after the coordinator's own nvidia-smi
        monitoring caught persistently low GPU utilization in the batched
        MTP path despite high CUDA-event-measured busy time).

        ``qo_len`` as a RAGGED per-request list (2026-07-17, for the
        recompute-fallback batching round): each slot may contribute a
        DIFFERENT number of new tokens this call -- forwarded as-is to
        ``build_attention_metadata_batch``/``build_gdn_metadata_batch``
        (both already generalized for this, see their docstrings) and
        used locally to build per-slot-correct ``positions``/kv_len
        bookkeeping. A scalar ``qo_len`` broadcasts to a uniform list, so
        every existing call site is byte-for-byte unaffected.

        ``gdn_spec_num_accepted_tokens_prev`` (2026-07-18, Phase 2, default
        ``None`` preserving every existing call site byte-for-byte): when
        given (one entry per slot), GDN metadata is built via the REAL
        spec-decode mechanism (``build_gdn_metadata_spec_batch``) instead
        of the chunked/prefill-shaped ``build_gdn_metadata_batch`` --
        K+1 dedicated SSM state rows per slot, acceptance-aware addressing
        selecting which row survives to be read next round, no
        snapshot/restore or recompute-forward needed. Requires a SCALAR,
        uniform ``qo_len`` (always ``num_speculative_tokens + 1`` in
        practice) -- unlike the chunked path this is not generalized to a
        ragged per-request list, since every real spec-decode verify call
        submits the same K+1-token draft for every slot. Only
        ``verify_batch_spec`` passes this.

        ``logits_last_position_only`` (2026-07-18, D1-followup fix, default
        ``False`` preserving every existing call site byte-for-byte): when
        ``True``, ``self.model.compute_logits(...)`` is applied to ONLY the
        last position of each slot's ``qo_len`` block (gathered via
        ``index_select`` right before the vocab-head projection), instead of
        every position -- the returned ``logits`` is then shaped
        ``[num_reqs, vocab]``, NOT ``[num_reqs * qo_len, vocab]``. The full,
        un-gathered ``hidden_states`` is still returned unchanged when
        ``return_hidden=True`` -- only the tensor fed into ``compute_logits``
        is sliced. Found via direct instrumentation
        (``benchmarks/mtp_prefill_batch_memory_diag.py``) profiling the
        16K-context/c=4 shape flagged in
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 12:
        at ``qo_len=16384``/``concurrency=4`` this call's own
        ``compute_logits`` alone allocates a 31040 MiB ``[65536, 248320]``
        bf16 tensor of which only 4 rows (0.006%) are ever read by any
        caller -- only ``mtp_prefill_batch`` needs the anchor logits, and
        only at each slot's OWN last prompt position. ``decode_batch``/
        ``verify_batch``/``verify_batch_spec`` genuinely need every
        position's logits (MTP verify checks every draft token against the
        target's own prediction) and MUST NOT pass this -- it is only safe
        when the caller already only reads the last row per slot, which is
        why only ``mtp_prefill_batch`` sets it.
        """
        num_reqs = len(slot_ids)
        qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
        if len(qo_lens) != num_reqs:
            raise ValueError("qo_len list must have exactly one entry per slot")

        if isinstance(qo_len, int) and qo_len == 1:
            if not (len(token_ids) == num_reqs and len(kv_lengths) == num_reqs):
                raise ValueError("slot_ids/token_ids/kv_lengths must have equal length")
            flat_token_ids = token_ids
        else:
            if not (
                len(token_ids) == num_reqs
                and len(kv_lengths) == num_reqs
                and all(len(t) == qo for t, qo in zip(token_ids, qo_lens))
            ):
                raise ValueError(
                    "slot_ids/token_ids/kv_lengths must have equal length, and "
                    "every token_ids[i] must have exactly qo_len[i] tokens"
                )
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        for slot, kv_len in zip(slot_ids, kv_lengths):
            if kv_len != self.slot_kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot}: caller-provided kv_length {kv_len} != "
                    f"tracked {self.slot_kv_len[slot]}"
                )
            # kv_len == 0 legitimately means "this slot's very first forward"
            # (matches ``prefill()``'s own "fresh slot" definition) -- 2026-07-17
            # relaxation for ``mtp_prefill_batch``, the first real caller that
            # needs a batched forward covering NEVER-forwarded slots.
            # ``build_gdn_metadata_batch``'s qo_len>1 branch already accepts a
            # per-slot ``slot_initialized`` list (passed below) and handles
            # ``False`` correctly (has_initial_state=False is exactly what a
            # fresh slot's chunked GDN forward needs) -- this guard was stricter
            # than the underlying kernel actually requires, a leftover of
            # ``_forward_batch`` previously only ever being called on
            # already-prefilled slots (``decode_batch``/``verify_batch``). Any
            # OTHER "not yet initialized" case (kv_len != 0) still raises,
            # unchanged -- that combination can only mean a caller skipped a
            # real prefill while lying about kv_len, exactly what this check
            # exists to catch.
            if not self.slot_gdn_initialized[slot] and kv_len != 0:
                raise RuntimeError(f"slot {slot} has no GDN state yet (needs a prior prefill)")

        # P1 (notes/prefix-cache-design.md sec 5): grow every listed slot's
        # block_table to cover this call's own new_kv_len (kv_len + qo)
        # BEFORE building metadata/slot-mapping below, which both read
        # self.block_table[slot] as-is.
        if self.enable_block_table:
            for slot, kv_len, qo in zip(slot_ids, kv_lengths, qo_lens):
                self._ensure_blocks(slot, kv_len + qo)

        attn_meta = build_attention_metadata_batch(
            slots=slot_ids,
            prior_kv_lens=kv_lengths,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
            qo_len=qo_len,
            is_decode=is_decode,
            fixed_kv_split_size=fixed_kv_split_size,
            fixed_max_num_splits=fixed_max_num_splits,
            block_tables=(
                [self.block_table[s] for s in slot_ids] if self.enable_block_table else None
            ),
        )
        if gdn_spec_num_accepted_tokens_prev is not None:
            if not isinstance(qo_len, int):
                raise ValueError("gdn_spec_num_accepted_tokens_prev requires a scalar qo_len")
            gdn_meta = build_gdn_metadata_spec_batch(
                slots=slot_ids,
                device=self.device,
                qo_len=qo_len,
                num_accepted_tokens_prev=gdn_spec_num_accepted_tokens_prev,
                total_physical_slots=self.num_slots + RESERVED_PHYSICAL_SLOTS,
                num_spec=self.num_speculative_tokens,
            )
        else:
            gdn_meta = build_gdn_metadata_batch(
                slots=slot_ids,
                device=self.device,
                qo_len=qo_len,
                slot_initialized=(
                    [self.slot_gdn_initialized[s] for s in slot_ids]
                    if not (isinstance(qo_len, int) and qo_len == 1)
                    else None
                ),
            )
        attn_metadata_dict = {name: attn_meta for name in self.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in self.gdn_layer_names})
        slot_mapping = self._slot_mapping_batch(slot_ids, kv_lengths, qo_len=qo_len)
        slot_mapping_dict = {name: slot_mapping for name in self.attn_layer_names}

        input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(
            [kv_len + j for kv_len, qo in zip(kv_lengths, qo_lens) for j in range(qo)],
            dtype=torch.long,
            device=self.device,
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = self.model.forward(input_ids, positions)
        # 2026-07-17, Phase 3: see ``_forward``'s docstring/comment for why
        # the two blanket ``torch.cuda.synchronize()`` calls that used to
        # bracket ``compute_logits`` here were removed -- same-stream
        # ordering already guarantees correctness, and this method (the
        # real per-round verify/recompute/decode hot path) is exactly
        # where Phase 0's ``nsys`` ledger measured the dominant no-kernel
        # gap this removal targets.
        if logits_last_position_only:
            # 2026-07-18, D1-followup fix: project only each slot's own
            # last position through the vocab head -- see this parameter's
            # docstring. ``qo_lens`` (already computed above) gives each
            # slot's own row count; cumulative sum minus 1 is that slot's
            # last row in the request-then-position-flattened layout
            # ``model.forward`` returned.
            last_idx = torch.tensor(
                [sum(qo_lens[: i + 1]) - 1 for i in range(num_reqs)],
                dtype=torch.long,
                device=self.device,
            )
            logits_hidden = hidden_states.index_select(0, last_idx)
        else:
            logits_hidden = hidden_states
        logits = self.model.compute_logits(logits_hidden)

        for slot, qo in zip(slot_ids, qo_lens):
            if commit:
                self.slot_kv_len[slot] += qo
            self.slot_gdn_initialized[slot] = True
        if return_hidden:
            return logits, hidden_states
        return logits

    def decode_batch(
        self, slot_ids: list[int], token_ids: list[int], kv_lengths: list[int]
    ) -> list[int]:
        """Decode one token for each of several active slots via a single
        real batched forward call. Returns the greedy next token id per
        slot, in the same order as ``slot_ids``."""
        logits = self._forward_batch(slot_ids, token_ids, kv_lengths)
        return [int(logits[i].argmax(dim=-1).item()) for i in range(len(slot_ids))]

    def verify_batch(
        self,
        slot_ids: list[int],
        draft_token_ids: list[list[int]],
        kv_lengths: list[int],
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """MTP/speculative-decode verify: submit ``qo_len`` draft tokens
        (K speculative + 1 bonus position) per active slot and run them all
        through ONE real batched forward call. ``draft_token_ids[i]`` is
        slot ``slot_ids[i]``'s own list of draft tokens (same length for
        every slot this step, since ``num_speculative_tokens`` is a global
        engine config). Returns raw logits shaped
        ``[num_reqs * qo_len, vocab]`` (request-then-position order) --
        accept/reject sampling against these logits is the caller's job
        (``determine_accept_reject``/``mtp_verify_and_commit``).
        ``commit=False`` is passed to ``_forward_batch`` unconditionally --
        a verify call's real committed length is never known until
        accept/reject runs on these logits, so ``slot_kv_len`` is
        deliberately NOT advanced here (2026-07-17 fix; see
        ``_forward_batch``'s docstring). Passes this runner's own fixed
        split-KV config (2026-07-17) so the decode/verify kernel gets real
        split-KV parallelism instead of collapsing to ``max_num_splits=1``
        -- see ``_forward_batch``'s docstring."""
        qo_len = len(draft_token_ids[0]) if draft_token_ids else 0
        return self._forward_batch(
            slot_ids,
            draft_token_ids,
            kv_lengths,
            qo_len=qo_len,
            commit=False,
            return_hidden=return_hidden,
            fixed_kv_split_size=self.decode_fixed_kv_split_size,
            fixed_max_num_splits=self.decode_fixed_max_num_splits,
        )

    def verify_batch_spec(
        self,
        slot_ids: list[int],
        draft_token_ids: list[list[int]],
        kv_lengths: list[int],
        *,
        num_accepted_tokens_prev: list[int],
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """MTP/speculative-decode verify via the REAL spec-decode GDN
        mechanism (Phase 2, 2026-07-18) -- ``verify_batch``'s sibling,
        originally for ``mtp_verify_and_commit_batch`` only, and (Phase B,
        same day) for ``mtp_verify_and_commit`` too (called at
        ``len(slot_ids)==1``) -- both production verify paths share this
        method now. Same call shape/return convention as ``verify_batch``
        (raw logits AND hidden states, request-then-position order,
        ``commit=False`` -- caller advances
        ``slot_kv_len``/``slot_num_accepted_tokens`` after accept/reject).
        The only difference is GDN metadata construction: K+1 dedicated
        SSM state rows per slot (``build_gdn_metadata_spec_batch``,
        ``_ssm_spec_row``) instead of the chunked/prefill-shaped path,
        so a partial reject needs no snapshot/restore or recompute-forward
        repair -- the "wrong" candidates' rows are simply never read by
        any future round. ``num_accepted_tokens_prev[i]`` is slot
        ``slot_ids[i]``'s real committed length from its own last verify
        round (or exactly 1 on a slot's first-ever verify right after a
        real prefill -- see ``build_gdn_metadata_spec_batch``'s
        docstring). See notes/2026-07-17-post-ragged-round-next-steps.md
        section 10/11 for the derivation and validation history that
        underlies this method."""
        qo_len = len(draft_token_ids[0]) if draft_token_ids else 0
        return self._forward_batch(
            slot_ids,
            draft_token_ids,
            kv_lengths,
            qo_len=qo_len,
            commit=False,
            return_hidden=return_hidden,
            fixed_kv_split_size=self.decode_fixed_kv_split_size,
            fixed_max_num_splits=self.decode_fixed_max_num_splits,
            gdn_spec_num_accepted_tokens_prev=num_accepted_tokens_prev,
        )

    def reset_slot(self, slot: int) -> None:
        """Release a slot for reuse by a new logical request. Does not zero
        the underlying tensors -- the next prefill's has_initial_state=False
        and kv_len bookkeeping starting from 0 is what makes reuse correct,
        matching this project's established fixed-slot-generation design.

        **2026-07-17 fix** (Codex-sol review, confirmed real): this used to
        leave ``slot_draft_sync_len``/``slot_pending_draft_tokens`` at
        whatever stale value the PREVIOUS logical request left behind. A
        fresh ``mtp_prefill()`` on this slot starts its real target KV at
        position 0, but its draft-sync step-0 call reads
        ``self.slot_draft_sync_len[slot]`` as ``prior_kv_len`` -- if that
        was never reset, the very first MTP cycle for the NEW request
        would build attention metadata against the OLD request's leftover
        history length, an immediate correctness bug for any slot that is
        ever reused (which is this project's whole fixed-slot-generation
        premise). Now cleared alongside the pre-existing fields, matching
        the same "every persistent per-slot MTP field must be reset on
        reuse" discipline.

        **P1 (2026-07-19, notes/prefix-cache-design.md sec 5, design doc's
        risk R10)**: also releases this slot's own physical attention
        blocks back to ``self.block_pool`` (``ref_cnt -= 1``, re-enters the
        free queue at 0) and clears ``self.block_table[slot]`` to ``[]`` --
        without this, P1's on-demand allocator would leak a block every
        time a slot is reused (``_ensure_blocks`` only ever grows, it never
        shrinks). Driven by ``self.block_table[slot]``'s own CONTENTS, not
        ``self.enable_block_table``'s current value -- correct regardless
        of whether the flag was on when these blocks were allocated (a
        slot that never grew any blocks, because the flag was off the
        whole time, has an empty list here and this is a no-op)."""
        if self.block_table[slot]:
            # P3.2 (design doc sec 3.2/3.9): free in REVERSE logical order so a
            # slot's deep-prefix (tail) blocks are enqueued ahead of its shallow
            # ones and die first under eviction -- keeping shallow, more-shared
            # prefixes cached longer. (Among hashed blocks, free appends to the
            # LRU tail in call order, so the first-freed deep tail lands closest
            # to the evict-next front.)
            self.block_pool.free(list(reversed(self.block_table[slot])))
            self.block_table[slot] = []
        self.slot_kv_len[slot] = 0
        self.slot_gdn_initialized[slot] = False
        self.slot_draft_sync_len[slot] = 0
        self.slot_pending_draft_tokens[slot] = None
        # Phase 2 (2026-07-18): bootstrap value for the spec-decode GDN
        # mechanism -- see __init__'s field comment.
        self.slot_num_accepted_tokens[slot] = 1
        # P3 (notes/2026-07-19-p3-implementation-plan.md step 4): reset this
        # slot's LOCAL hash-chain view. The published blocks themselves stay in
        # the global content index at ref_cnt == 0 (freed above, hash retained)
        # so they remain hit-able across this reset (R10) -- only the slot's own
        # cursor/chain is cleared for reuse by a new logical request.
        self.slot_block_hashes[slot] = []
        self.slot_published_blocks[slot] = 0
        # P3.2 decode-position populate: clear this slot's committed-token
        # record (the published blocks themselves stay in the global index at
        # ref_cnt == 0, hash retained -- only the slot-local sequence is reset).
        self.slot_committed_tokens[slot] = []

    def snapshot_gdn_state(self, slot: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Copy out this slot's ``(conv_state, ssm_state)`` for every GDN
        layer, keyed by layer name. Building block for MTP verify's GDN
        state commit/rollback (2026-07-17 round): unlike attention's paged
        KV cache (content-addressed by position, safe to just stop
        advancing ``slot_kv_len`` past a rejected boundary), GDN's
        recurrent/chunked state has no position index to truncate to -- it
        is a single accumulated value per slot that a verify call updates
        in place. Snapshotting before a verify call and restoring here on
        partial rejection (this class's chosen strategy -- "Option A" in
        notes/direct-model-runner-design.md's MTP-semantics design
        section) is the correctness-first approach: simple to reason about
        and to verify independently of the rest of MTP (see
        ``benchmarks/mtp_gdn_rollback_check.py``), at the cost of an extra
        state copy per verify call and a recompute forward pass on
        rejection.

        **2026-07-17, Phase 1 (GPU-resident double buffer)**: returns
        GPU-resident VIEWS into a preallocated, fixed-address per-slot
        buffer (``self.gdn_snapshot_conv``/``self.gdn_snapshot_ssm``, see
        ``_allocate_gdn_snapshot_buffers``) instead of fresh CPU clones --
        the data is copied via a single D2D ``copy_`` per layer (~0.4ms at
        HBM rates, measured; see notes/2026-07-17-post-ragged-round-next-
        steps.md's section 8) instead of a blocking pageable D2H memcpy
        (89-117ms/round, per that doc's section 7). API/return shape is
        UNCHANGED (same dict keys, same per-layer ``(conv, ssm)`` tuple
        shape) -- callers (``restore_gdn_state``, and, at the time, both
        ``mtp_verify_and_commit``/``_batch``) did not need to change.

        **2026-07-18, Phase B**: neither production verify path
        (``mtp_verify_and_commit``/``_batch``) calls this method any more
        -- both migrated to the real spec-decode GDN mechanism (see
        ``mtp_verify_and_commit``'s docstring), under which state commit is
        acceptance-aware and no snapshot/restore is ever needed. This
        method is retained as a tested, standalone primitive: a falsifier
        check (before Phase B's migration) confirmed
        ``benchmarks/mtp_gdn_rollback_check.py`` tests it directly
        (independent of any MTP verify call), and several other
        diagnostics (``mtp_real_draft_check.py``, ``mtp_trace_driven_probe.py``,
        ``mtp_slot_identity_pinpoint_diag.py``, ``mtp_batch_divergence_diag.py``,
        ``phase0_nsys_gap_ledger_diag.py``) call it directly too.

        Tags the snapshot with the SOURCE slot id and this slot's current
        generation counter (``self.slot_gdn_snapshot_gen``, bumped on
        every snapshot) -- 2026-07-17 addition per Codex-sol's explicit
        ask for explicit per-slot state so a STALE snapshot (e.g. a caller
        accidentally holding on to one from two rounds ago) can never be
        restored by mistake; ``restore_gdn_state`` rejects a generation
        mismatch. The slot-id tag was added in a follow-up fix the same
        day: without it, a caller mistakenly restoring slot A's snapshot
        into slot B could still pass the generation check (both slots
        typically climb their OWN counters in lockstep in a symmetric
        multi-slot workload, so equal generation numbers say nothing about
        SLOT identity) -- ``restore_gdn_state`` now also rejects a
        slot-id mismatch. Also marks the snapshot ``__consumed__`` on a
        successful restore -- restoring the SAME snapshot object a second
        time now raises instead of silently succeeding (idempotent in
        this specific case since both restores would write the same
        bytes, but a caller path that restores twice by mistake is exactly
        the kind of latent bug this project's "no silent passes" standard
        exists to catch). These three invariants are unchanged by the
        Phase 1 storage-medium change -- they are checked in
        ``restore_gdn_state`` BEFORE any tensor data is read, so a stale
        snapshot is still rejected even though the underlying GPU buffer
        may since have been overwritten by a newer generation's data (see
        ``_allocate_gdn_snapshot_buffers``'s docstring for why that's
        safe)."""
        physical = _physical_slot(slot)
        self.slot_gdn_snapshot_gen[slot] += 1
        snapshot: dict = {
            "__slot__": slot,
            "__generation__": self.slot_gdn_snapshot_gen[slot],
            "__consumed__": False,
        }
        # 2026-07-17, Phase 3 (round 2, coordinator-directed fast-iteration
        # pass): replaced the per-layer Python loop's 2*len(gdn_layer_names)
        # individual ``.copy_()`` kernel launches (96 for 48 layers, x4
        # slots/round = 384 -- Phase 0's ledger figure) with TWO
        # ``torch._foreach_copy_`` calls (one for all conv tensors, one for
        # all ssm tensors) -- PyTorch's multi-tensor-apply fuses the whole
        # list into a small constant number of kernel launches regardless
        # of layer count, cutting per-round host dispatch for this phase by
        # roughly 48x. Same D2D copy semantics as before (still fixed-address
        # buffers, no reallocation, no host round-trip) -- purely a launch-
        # count reduction, not a new mechanism.
        conv_dsts, ssm_dsts, conv_srcs, ssm_srcs = [], [], [], []
        for name in self.gdn_layer_names:
            conv_state, ssm_state = self.kv_caches[name]
            conv_dsts.append(self.gdn_snapshot_conv[name][slot])
            ssm_dsts.append(self.gdn_snapshot_ssm[name][slot])
            conv_srcs.append(conv_state[physical])
            ssm_srcs.append(ssm_state[physical])
        torch._foreach_copy_(conv_dsts, conv_srcs)
        torch._foreach_copy_(ssm_dsts, ssm_srcs)
        for name, snap_conv, snap_ssm in zip(self.gdn_layer_names, conv_dsts, ssm_dsts):
            snapshot[name] = (snap_conv, snap_ssm)
        return snapshot

    def restore_gdn_state(
        self,
        slot: int,
        snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        allow_cross_slot: bool = False,
    ) -> None:
        """Restore this slot's GDN state from a prior
        ``snapshot_gdn_state()`` call -- writes IN PLACE into the same
        persistent ``kv_caches`` tensors (never reallocates them), so this
        is safe to call between real forward passes without disturbing any
        other slot or any fixed-address buffer a CUDA-graph-captured call
        might depend on. Rejects a stale snapshot (generation counter
        mismatch), a snapshot taken for a DIFFERENT slot, or a snapshot
        that has already been consumed by a prior restore -- see
        ``snapshot_gdn_state``'s docstring for why each of these was
        added (2026-07-17, Codex-sol review), and (2026-07-17, Phase 1)
        for why they still hold with GPU-resident snapshot storage.

        **2026-07-17, Phase 1**: ``snapshot[name]`` is now already a
        GPU-resident tensor (a view into the fixed-address per-slot
        buffer), so the restore is a single D2D ``copy_`` per layer with
        no host round-trip and no ``.to(self.device)`` staging step -- the
        old CPU-clone path did both a D2H (in ``snapshot_gdn_state``) and
        an H2D (here) blocking pageable-memory copy per layer per slot."""
        if allow_cross_slot:
            # P2 fan-out fork (notes/prefix-cache-design.md sec 5, "P2 --
            # Fan-out fork", and sec 3.5 step 2): restore the LEADER's
            # snapshot into a SIBLING slot. The snapshot's ``__slot__`` is
            # the leader (the SOURCE of the recurrent state), which
            # legitimately differs from the destination ``slot`` here, so
            # the same-slot guard is relaxed. The generation counter is a
            # SAME-slot staleness guard (it catches restoring a slot's own
            # long-ago snapshot after that slot re-snapshotted itself); it
            # is meaningless across slots, where freshness is instead
            # guaranteed by the caller's synchronous structure --
            # ``mtp_prefill_fanout_batch`` snapshots the leader and restores
            # every sibling within ONE atomic admission tick, with no
            # intervening re-snapshot of the leader (and the MTP verify path
            # no longer calls snapshot_gdn_state at all, so the leader's
            # snapshot buffer cannot be clobbered mid-fork). ``__consumed__``
            # is deliberately NOT set below in this mode, so the ONE leader
            # snapshot can seed all N siblings: each restore is a read-only
            # D2D copy FROM the leader's fixed-address snapshot buffer INTO
            # this slot's own kv_caches row, and the source buffer stays
            # stable for the whole fork. R1 (GDN corruption on restore) is
            # still guarded structurally -- the foreach_copy below reads the
            # same 48-layer snapshot[name] tensors the leader populated.
            if snapshot.get("__slot__") is None:
                raise RuntimeError(
                    "cross-slot GDN restore requires a real snapshot (missing __slot__ tag)"
                )
        else:
            if snapshot.get("__slot__") != slot:
                raise RuntimeError(
                    f"GDN snapshot was taken for slot {snapshot.get('__slot__')}, "
                    f"not slot {slot} -- refusing a cross-slot restore"
                )
            if snapshot.get("__consumed__"):
                raise RuntimeError(f"GDN snapshot for slot {slot} was already restored once")
            gen = snapshot.get("__generation__")
            if gen != self.slot_gdn_snapshot_gen[slot]:
                raise RuntimeError(
                    f"stale GDN snapshot for slot {slot}: snapshot generation {gen} != "
                    f"current {self.slot_gdn_snapshot_gen[slot]}"
                )
        physical = _physical_slot(slot)
        # 2026-07-17, Phase 3 (round 2): same torch._foreach_copy_
        # launch-count reduction as snapshot_gdn_state's mirror-image
        # change above.
        conv_dsts, ssm_dsts, conv_srcs, ssm_srcs = [], [], [], []
        for name in self.gdn_layer_names:
            conv_state, ssm_state = self.kv_caches[name]
            snap_conv, snap_ssm = snapshot[name]
            conv_dsts.append(conv_state[physical])
            ssm_dsts.append(ssm_state[physical])
            conv_srcs.append(snap_conv)
            ssm_srcs.append(snap_ssm)
        torch._foreach_copy_(conv_dsts, conv_srcs)
        torch._foreach_copy_(ssm_dsts, ssm_srcs)
        if not allow_cross_slot:
            snapshot["__consumed__"] = True

    def _mtp_forward(
        self,
        slot: int,
        token_ids: list[int],
        hidden_states_in: torch.Tensor,
        start_pos: int,
        *,
        prior_kv_len: int,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Real draft-model (``Qwen3_5MTP``) forward for ONE slot -- the
        low-level primitive the centralized MTP-cycle coordinator methods
        (``mtp_prefill``/``mtp_verify_and_commit``) build on.

        ``prior_kv_len`` (2026-07-17 fix -- see below) is an EXPLICIT
        caller-supplied argument, NOT read from ``self.slot_draft_sync_len``
        internally as an earlier version of this method did. This method
        does NOT touch ``self.slot_draft_sync_len`` itself either way (the
        caller decides whether this call's advance represents the real
        synced history -- step 0, teacher-forced with the target's own
        just-computed hidden state -- or a throwaway exploratory propose
        step -- steps 1..K-1, autoregressive on the draft's own previous
        hidden state/token -- that must NOT be counted as committed). This
        is what makes the draft model's own KV cache need no explicit
        rollback on accept/reject, unlike GDN: an exploratory step's
        positions are simply overwritten by the next round's real sync
        call, exactly like attention's own content/position-addressed
        reasoning elsewhere in this file.

        **2026-07-17 real bug, caught by an independent Codex-sol review
        and independently re-verified by the coordinator before being
        relayed**: this method used to read ``prior_kv_len=self
        .slot_draft_sync_len[slot]`` directly. That field is deliberately
        NOT updated after step 0 (see above) -- correct for THAT field's
        job (tracking the real committed sync length across rounds), but
        WRONG when reused as this call's OWN attention-metadata history
        length for the exploratory loop's 2nd-and-later steps: those
        steps' actual physical write position (``start_pos``, which DOES
        advance every exploratory iteration in
        ``_mtp_sync_and_propose``) drifts away from the frozen
        ``slot_draft_sync_len``, so the attention metadata told the
        kernel a SMALLER history length than where the write actually
        landed -- the exploratory step's own query would then fail to
        attend to the PREVIOUS exploratory step's just-written K/V (it
        wasn't in the "prior" range the metadata declared), silently
        computing over an incomplete/wrong causal history for every
        exploratory step from the 2nd one onward. K=3 (this project's
        real production setting) has exactly 2 exploratory steps, so the
        1st (which happens to immediately follow step 0, where the frozen
        field and the real position still coincide) was fine, but the 2nd
        was not -- meaning every real K=3 proposal's 3rd draft token was
        computed against a subtly wrong causal history. Fixed by making
        the caller (``_mtp_sync_and_propose``) track its own LOCAL running
        prior-length counter that DOES advance every exploratory
        iteration, passed in here explicitly, while ``self
        .slot_draft_sync_len`` itself still only updates once (after step
        0) -- decoupling "what this call's attention needs" from "what the
        cross-round bookkeeping should remember" fixes both correctly at
        once. Not shape-checkable (see notes/direct-model-runner-design.md's
        2026-07-17 methodology-fix entry): this bug produced the right
        SHAPE and vocab-range output at every step, only the CONTENT was
        wrong from the 2nd exploratory step on -- exactly why the
        verification gradient's steps 3-4 (shape/length checks only) never
        caught it, and why the fix needed a per-step oracle-aligned logits
        comparison, not another shape check, to confirm.

        ``hidden_states_in`` must have exactly ``len(token_ids)`` rows
        (``Qwen3_5MultiTokenPredictor.forward()`` concatenates it against
        the embedded ``input_ids`` along the hidden-size dim, so the
        sequence-length dim must already match)."""
        if self.mtp_model is None:
            raise RuntimeError(
                "no MTP draft model loaded -- build_vllm_config(speculative_config=...) first"
            )
        num_new_tokens = len(token_ids)
        # P1 (notes/prefix-cache-design.md sec 5): the draft layer shares
        # the SAME block-id namespace as the target's attention group (sec
        # 3.1), so its own forward must grow self.block_table[slot] too --
        # using prior_kv_len (== start_pos at every real call site, see
        # this method's docstring) + num_new_tokens, exactly what
        # build_attention_metadata below uses as its own new_kv_len.
        if self.enable_block_table:
            self._ensure_blocks(slot, prior_kv_len + num_new_tokens)
        attn_meta = build_attention_metadata(
            prior_kv_len=prior_kv_len,
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
            block_table=self.block_table[slot] if self.enable_block_table else None,
        )
        attn_metadata_dict = {name: attn_meta for name in self.mtp_attn_layer_names}
        slot_mapping = self._slot_mapping(slot, start_pos, num_new_tokens)
        slot_mapping_dict = {name: slot_mapping for name in self.mtp_attn_layer_names}

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states_out = self.mtp_model.forward(input_ids, positions, hidden_states_in)
        # 2026-07-17, Phase 3: see ``_forward``'s docstring/comment -- same
        # same-stream-ordering reasoning applies to the draft model's own
        # forward+compute_logits pair.
        logits = self.mtp_model.compute_logits(hidden_states_out)
        return logits, hidden_states_out

    def _mtp_sync_and_propose(
        self,
        slot: int,
        shifted_input_ids: list[int],
        target_hidden_states: torch.Tensor,
        start_pos: int,
        num_new_tokens: int,
        k: int,
    ) -> list[int]:
        """The centralized sync+propose funnel every MTP-aware entry point
        (``mtp_prefill``/eventual ``mtp_decode``) routes through -- per
        2026-07-17's sol-refined design, this is the ONE place draft-sync
        logic lives, not duplicated per public entry point. Step 0 is the
        real sync (teacher-forced with the target's OWN just-computed
        hidden states, covering this step's FULL real query range --
        matches vLLM's real ``_prepare_prefill_inputs_kernel`` shift-by-one
        mechanism); steps 1..k-1 are genuinely autoregressive on the
        draft's own previous hidden state/token, and are NOT committed to
        ``self.slot_draft_sync_len`` (see ``_mtp_forward``'s docstring).

        **2026-07-17 fix**: tracks its OWN local ``running_prior_kv_len``
        counter, separate from ``self.slot_draft_sync_len`` -- the local
        counter advances every exploratory iteration (matching where each
        step's write actually lands), while the persistent field only
        ever advances once, after step 0. Passing the persistent field
        directly into every exploratory `_mtp_forward` call (the previous,
        buggy version) left steps 2..k-1's attention metadata pointing at
        a stale, non-advancing history length -- see `_mtp_forward`'s
        docstring for the full analysis of what that broke."""
        step0_logits, step0_hidden = self._mtp_forward(
            slot,
            shifted_input_ids,
            target_hidden_states,
            start_pos,
            prior_kv_len=self.slot_draft_sync_len[slot],
            is_decode=(num_new_tokens == 1),
        )
        self.slot_draft_sync_len[slot] += num_new_tokens
        draft_tokens = [int(step0_logits[-1].argmax(dim=-1).item())]
        prev_hidden = step0_hidden[-1:]
        prev_token = draft_tokens[0]
        next_pos = start_pos + num_new_tokens
        running_prior_kv_len = self.slot_draft_sync_len[slot]
        for _ in range(1, k):
            step_logits, step_hidden = self._mtp_forward(
                slot, [prev_token], prev_hidden, next_pos, prior_kv_len=running_prior_kv_len, is_decode=True
            )
            prev_token = int(step_logits[-1].argmax(dim=-1).item())
            draft_tokens.append(prev_token)
            prev_hidden = step_hidden[-1:]
            next_pos += 1
            running_prior_kv_len += 1
        return draft_tokens

    def mtp_prefill(self, slot: int, prompt_token_ids: list[int]) -> dict:
        """Unified MTP cycle funnel point for a fresh prefill: real target
        prefill (with hidden states) -> draft KV sync (step 0, teacher-
        forced shift over the WHOLE prompt) -> K-1 more autoregressive
        draft steps. Returns the anchor (target's own greedy next token,
        matching plain ``prefill()``'s contract -- not yet written into
        the target's own KV) and the K proposed draft tokens, ready for
        the caller to submit through ``mtp_verify_and_commit``."""
        if self.mtp_model is None or self.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        if self.slot_kv_len[slot] != 0 or self.slot_draft_sync_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh")
        # Phase 2/Phase B (2026-07-18), defense in depth: bootstrap value
        # for the spec-decode GDN mechanism's first-ever verify round on
        # this slot -- already 1 via __init__/reset_slot for any slot that
        # actually went through one of those, set explicitly here too so
        # this invariant holds regardless of how the slot got to "fresh"
        # (mirrors mtp_prefill_batch's identical defense-in-depth line).
        self.slot_num_accepted_tokens[slot] = 1
        target_logits, target_hidden = self._forward(
            slot, prompt_token_ids, start_pos=0, is_decode=False, return_hidden=True
        )
        anchor = int(target_logits[-1].argmax(dim=-1).item())
        shifted_input_ids = prompt_token_ids[1:] + [anchor]
        draft_tokens = self._mtp_sync_and_propose(
            slot,
            shifted_input_ids,
            target_hidden,
            start_pos=0,
            num_new_tokens=len(prompt_token_ids),
            k=self.num_speculative_tokens,
        )
        self.slot_pending_draft_tokens[slot] = draft_tokens
        # P3 populate-on-completion (attention half): publish this prefill's
        # full committed blocks to the content index. The GDN completion
        # checkpoint is NOT materialized here -- a single-shot prefill's live
        # GDN state is at prompt_len, not at the block-aligned completion
        # boundary G = block_align_down(prompt_len - 1), so a correct checkpoint
        # at G needs a forward that ENDS at G (mtp_prefill_with_cache's two-phase
        # cold path / the fan-out leader at Lc). Publishing attention only here
        # is safe: a later hit finds A>0 but G=0 => compute miss (L=0, cold
        # recompute), never a wrong-prefix serve (sec 3.4).
        if self.enable_persistent_prefix_cache:
            self._publish_committed_blocks(slot, prompt_token_ids, len(prompt_token_ids))
        return {"anchor": anchor, "draft_tokens": draft_tokens}

    def mtp_verify_and_commit(self, slot: int, anchor: int, draft_tokens: list[int]) -> dict:
        """Unified MTP cycle funnel point for verify+commit+resync+propose
        -- the ONE method a real multi-round loop calls repeatedly for a
        SINGLE slot (no separate "decode" coordinator needed; see the
        design note below on why).

        **2026-07-18, Phase B migration** (independent review's
        ``notes/2026-07-18-session-review-and-next-steps.md`` Phase B,
        option (a); see this session's own addendum for the falsifier
        check and the result): now uses the REAL spec-decode GDN mechanism
        (``verify_batch_spec``/``build_gdn_metadata_spec_batch``/
        ``_ssm_spec_row``) -- the exact same mechanism
        ``mtp_verify_and_commit_batch`` adopted in Phase 2, applied here at
        batch_size=1 -- instead of the old chunked-GDN-metadata +
        ``snapshot_gdn_state``/``restore_gdn_state`` + recompute-forward
        mechanism this method used through 2026-07-18. This was the last
        production call site of the old mechanism. ``snapshot_gdn_state``/
        ``restore_gdn_state`` are NOT deleted as a result: the falsifier
        check for this migration (before touching any code) confirmed
        ``benchmarks/mtp_gdn_rollback_check.py`` tests them directly as
        primitives (snapshot/restore around a real multi-step "detour",
        with no MTP verify call involved at all), and several other
        diagnostics (``mtp_real_draft_check.py``, ``mtp_trace_driven_probe.py``,
        ``mtp_slot_identity_pinpoint_diag.py``, ``mtp_batch_divergence_diag.py``,
        ``phase0_nsys_gap_ledger_diag.py``) call them directly too -- they
        remain in the codebase as tested, live (if no longer
        production-verify-path-connected) primitives. The old chunked
        ``verify_batch`` is retained for the same reason (still called
        directly by several diagnostics and by ``decode_batch``'s qo_len=1
        path via ``build_gdn_metadata_batch``).

        Why no accept/reject branch is needed any more (mirrors
        ``mtp_verify_and_commit_batch``'s own docstring): GDN's recurrent
        state, under the real spec-decode kernel, computes a causally-valid
        PER-POSITION output for every one of the K+1 candidate positions in
        a single verify forward, unconditionally -- only the recurrent
        STATE COMMIT (which physical row survives to be read next round) is
        acceptance-aware, via ``num_accepted_tokens``/``_ssm_spec_row``.
        This slot's hidden states for positions ``0..committed_len-1`` are
        therefore already sitting in ``verify_hidden``, correct, from the
        ONE verify forward this method issues -- a plain slice (never a
        second forward pass) is all the draft resync step needs, for a
        full accept exactly as much as for any partial reject.

        Persistent per-slot bookkeeping this method updates:
        ``self.slot_num_accepted_tokens[slot]`` (this slot's real committed
        length from ITS OWN last verify round, or bootstrap 1 right after a
        real ``mtp_prefill``) -- read by ``build_gdn_metadata_spec_batch``
        (via ``verify_batch_spec``) to select which of last round's K+1
        dedicated SSM rows holds the valid state to resume from.

        Recompute input alignment (2026-07-17, unchanged by this
        migration -- see notes/direct-model-runner-design.md): the token
        whose OWN K/V gets written at position ``kv_len_before + i`` is
        the i-th QUERY INPUT of the verify forward, matching
        ``verify_batch``/``verify_batch_spec``'s shared convention where
        ``draft[0]=anchor``'s K/V lands at ``kv_len_before`` (mirroring
        ``prefill()``/``decode()``'s established contract: the
        anchor/greedy-next token is NOT written into KV until it is fed
        back in as the FOLLOWING call's input). ``decision["committed"]``
        is ``[accepted_draft_0, ..., accepted_draft_{n-1}, recovery]`` --
        the recovery/bonus token is, symmetrically, NOT yet written into KV
        either (it becomes the next round's own anchor-equivalent). So the
        real input tokens for positions ``kv_len_before..+committed_len-1``
        are ``real_new_tokens = [anchor] + committed[:-1]`` (anchor +
        accepted drafts, dropping the not-yet-written recovery token) --
        NOT ``committed`` itself, which would silently write the WRONG
        token content into the KV cache while still looking correct on
        every shape/length/bookkeeping check.

        Draft catch-up + next-round propose, folded into ONE call
        (2026-07-17 multi-round design, unchanged by this migration): after
        committing, the draft's own KV is behind by exactly
        ``real_new_tokens`` (it was last synced at the END of the PREVIOUS
        round -- ``mtp_prefill``/this same method -- so
        ``slot_draft_sync_len`` always equals ``slot_kv_len`` from BEFORE
        this round's commit). Syncing the draft over ``real_new_tokens``
        (shifted by one, ending in the recovery/bonus token as the final
        candidate -- exactly ``_mtp_sync_and_propose``'s existing step-0
        pattern, just generalized from ``mtp_prefill``'s "whole prompt"
        range to "this round's newly committed range") both catches the
        draft's KV up to ``slot_kv_len`` again (restoring the invariant)
        AND, at that same call's LAST position (processing the
        recovery/bonus token as a candidate against the target's hidden
        state up through the last real position), produces the FIRST draft
        token for the NEXT round -- for free, no extra forward call.
        ``_mtp_sync_and_propose`` then runs the usual K-1 further
        autoregressive steps on top.

        Returns the accept/reject decision plus ``next_anchor`` (the
        recovery/bonus token -- feed this as ``anchor`` to the NEXT
        ``mtp_verify_and_commit`` call) and ``next_draft_tokens`` (K fresh
        proposed tokens for that next call)."""
        k = len(draft_tokens)
        draft = [anchor] + draft_tokens
        kv_len_before = self.slot_kv_len[slot]
        num_accepted_prev = self.slot_num_accepted_tokens[slot]

        verify_logits, verify_hidden = self.verify_batch_spec(
            [slot],
            [draft],
            [kv_len_before],
            num_accepted_tokens_prev=[num_accepted_prev],
            return_hidden=True,
        )
        decision = determine_accept_reject(draft, verify_logits)
        committed_len = decision["num_accepted"] + 1
        # Real input tokens for positions kv_len_before..+committed_len-1:
        # anchor followed by the accepted drafts (NOT the recovery token --
        # see the docstring above).
        real_new_tokens = [anchor] + decision["committed"][:-1]

        self.slot_kv_len[slot] = kv_len_before + committed_len
        self.slot_num_accepted_tokens[slot] = committed_len
        # P3.2 decode-position populate: publish any newly-FULL committed blocks
        # now that slot_kv_len advanced by the REAL committed length (only
        # committed tokens; INV4). No-op off-flag.
        self.publish_committed_decode_blocks(slot, real_new_tokens)
        # Ragged slice of the ONE verify forward's hidden states -- valid
        # for a full accept exactly as much as for any partial reject, see
        # the docstring above.
        real_new_hidden = verify_hidden[:committed_len]

        next_anchor = decision["committed"][-1]
        next_draft_tokens = self._mtp_sync_and_propose(
            slot,
            real_new_tokens[1:] + [next_anchor],
            real_new_hidden,
            start_pos=self.slot_draft_sync_len[slot],
            num_new_tokens=committed_len,
            k=k,
        )
        self.slot_pending_draft_tokens[slot] = next_draft_tokens
        return {**decision, "next_anchor": next_anchor, "next_draft_tokens": next_draft_tokens}

    # ------------------------------------------------------------------
    # True cross-slot batched MTP (2026-07-17 round): one shared forward
    # pass -- draft model included, not just the target -- across every
    # listed slot, instead of a Python loop calling the single-slot
    # methods above once per slot. Mirrors how ``_forward_batch``/
    # ``decode_batch``/``verify_batch`` already batch the plain
    # (non-MTP-aware) target-only path; these are the MTP analogues.
    # ------------------------------------------------------------------

    def _mtp_forward_batch(
        self,
        slots: list[int],
        token_ids,
        hidden_states_in: torch.Tensor,
        prior_kv_lens: list[int],
        start_pos_list: list[int],
        *,
        qo_len: int | list[int],
        is_decode: bool,
        logits_last_position_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analogue of ``_mtp_forward`` for the draft model
        (``Qwen3_5MTP``) -- ONE batched attention-metadata object (scoped to
        ``self.mtp_attn_layer_names``, no GDN metadata since the draft model
        registers no GDN layers -- see ``__init__``'s ``mtp_attn_layer_names``
        derivation) and ONE ``mtp_model.forward()`` call covering every
        listed slot. ``qo_len`` may be a per-slot RAGGED list (2026-07-17,
        for the recompute-fallback batching round -- mirrors
        ``build_attention_metadata_batch``'s identical generalization,
        forwarded through as-is); a scalar broadcasts to a uniform list,
        so every existing call site is byte-for-byte unaffected.

        ``prior_kv_lens``/``start_pos_list`` are explicit per-slot lists,
        kept as TWO separate parameters (not collapsed into one) to mirror
        ``_mtp_forward``'s own explicit-argument design (its 2026-07-17 bug
        fix -- see that method's docstring): at every real call site in this
        file the two values coincide, but they mean different things
        (attention-metadata history length vs. physical write/embedding
        position) and collapsing them would silently re-introduce the same
        class of bug that fix addressed, just in the batched path instead
        of the single-slot one.

        ``token_ids`` is a flat list (one token id per slot, in ``slots``
        order) when ``qo_len`` is the literal scalar ``1``, or a list of
        per-slot token-id lists (each of length ``qo_len[i]``, possibly
        ragged) otherwise -- same convention ``_forward_batch`` uses.
        Returns logits/hidden_states shaped ``[sum(qo_lens), ...]`` in
        request-then-position order.

        ``logits_last_position_only`` (2026-07-18, D1-followup fix, default
        ``False`` preserving every existing call site byte-for-byte): same
        contract as ``_forward_batch``'s identical parameter -- when
        ``True``, BOTH the returned ``logits`` AND the returned
        ``hidden_states_out`` are gathered down to only each slot's own
        last position (shape ``[num_reqs, ...]``, not
        ``[sum(qo_lens), ...]``) before/after ``compute_logits``. Safe here
        specifically because ``_mtp_sync_and_propose_batch`` (this
        method's only caller) already discards every non-last-position row
        of both return values via its own ``index_select`` immediately
        after calling this method, for every existing call site -- so
        gathering earlier (before the vocab-head projection, instead of
        after) changes nothing observable except removing the wasted
        compute/memory. Only ``_mtp_sync_and_propose_batch``'s step-0 call
        passes this, and only when its own caller (``mtp_prefill_batch``)
        requests it -- see that call site.
        """
        if self.mtp_model is None:
            raise RuntimeError(
                "no MTP draft model loaded -- build_vllm_config(speculative_config=...) first"
            )
        num_reqs = len(slots)
        if not (len(prior_kv_lens) == num_reqs and len(start_pos_list) == num_reqs):
            raise ValueError("slots/prior_kv_lens/start_pos_list must have equal length")
        qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
        if len(qo_lens) != num_reqs:
            raise ValueError("qo_len list must have exactly one entry per slot")
        if isinstance(qo_len, int) and qo_len == 1:
            if len(token_ids) != num_reqs:
                raise ValueError("token_ids must have one entry per slot when qo_len == 1")
            flat_token_ids = token_ids
        else:
            if not (len(token_ids) == num_reqs and all(len(t) == qo for t, qo in zip(token_ids, qo_lens))):
                raise ValueError("every slot's token_ids must have exactly qo_len[i] entries")
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        # P1 (notes/prefix-cache-design.md sec 5): same shared block-id
        # namespace reasoning as ``_mtp_forward`` -- grow every listed
        # slot's block_table to cover prior_kv_lens[i] + qo_lens[i] before
        # building metadata/slot-mapping below.
        if self.enable_block_table:
            for s, kv_len, qo in zip(slots, prior_kv_lens, qo_lens):
                self._ensure_blocks(s, kv_len + qo)

        attn_meta = build_attention_metadata_batch(
            slots=slots,
            prior_kv_lens=prior_kv_lens,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
            qo_len=qo_len,
            is_decode=is_decode,
            # 2026-07-17: always pass this runner's fixed split-KV config
            # (see __init__'s comment) -- harmless when is_decode=False
            # (decode_qo_len ends up 0 either way, so the decode-kernel
            # dispatch never reads kv_split_size/max_num_splits for that
            # call), and gives the draft model's own decode/verify-shaped
            # calls real split-KV parallelism instead of collapsing to
            # max_num_splits=1.
            fixed_kv_split_size=self.decode_fixed_kv_split_size,
            fixed_max_num_splits=self.decode_fixed_max_num_splits,
            block_tables=(
                [self.block_table[s] for s in slots] if self.enable_block_table else None
            ),
        )
        attn_metadata_dict = {name: attn_meta for name in self.mtp_attn_layer_names}
        # Reuses ``_slot_mapping_batch`` (built for the target model's own
        # batched path) unchanged: its ``kv_lengths`` parameter is used only
        # as "each request's own write-start position", i.e. exactly what
        # ``start_pos_list`` means here -- the formula is identical to
        # concatenating per-slot ``_slot_mapping(slot, start_pos, qo_len)``
        # calls, verified by inspection (both compute
        # ``_physical_slot(slot) * blocks_per_slot + pos // block_size``
        # over the same ``start_pos + j`` positions).
        slot_mapping = self._slot_mapping_batch(slots, start_pos_list, qo_len=qo_len)
        slot_mapping_dict = {name: slot_mapping for name in self.mtp_attn_layer_names}

        input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(
            [start_pos + j for start_pos, qo in zip(start_pos_list, qo_lens) for j in range(qo)],
            dtype=torch.long,
            device=self.device,
        )
        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states_out = self.mtp_model.forward(input_ids, positions, hidden_states_in)
        # 2026-07-17, Phase 3: see ``_forward``'s docstring/comment -- this
        # is the batched draft-model analogue, called up to K times per
        # round (previously 2 syncs each); same same-stream-ordering
        # reasoning applies.
        if logits_last_position_only:
            # 2026-07-18, D1-followup fix: see this parameter's docstring.
            last_idx = torch.tensor(
                [sum(qo_lens[: i + 1]) - 1 for i in range(num_reqs)],
                dtype=torch.long,
                device=self.device,
            )
            hidden_states_out = hidden_states_out.index_select(0, last_idx)
        logits = self.mtp_model.compute_logits(hidden_states_out)
        return logits, hidden_states_out

    def _mtp_run_continuation_steps(
        self,
        slots: list[int],
        draft_tokens: dict[int, list[int]],
        prev_tokens: list[int],
        prev_hidden: torch.Tensor,
        next_pos_list: list[int],
        running_prior_kv_len: list[int],
        k: int,
    ) -> None:
        """The k-1 batched autoregressive draft-continuation steps that
        follow step 0, appending each step's greedy token to
        ``draft_tokens[slot]`` IN PLACE (``draft_tokens`` must already hold
        step 0's own token for every slot in ``slots`` -- this method only
        APPENDS the remaining k-1).

        Extracted (2026-07-19, chunked-prefill round,
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 19)
        from ``_mtp_sync_and_propose_batch``'s own tail as a shared helper:
        pure code motion, not a behavior change -- ``_mtp_sync_and_propose_batch``'s
        call site below passes exactly the same local variables its old
        inlined body used, in the same order. This is what lets
        ``mtp_prefill_batch``'s new chunked path (which computes step 0
        itself, chunk by chunk, so each chunk's target hidden states can be
        fed to the draft model as soon as they exist, rather than needing
        the whole prompt's hidden states materialized at once) reuse the
        EXACT same, already-verified continuation logic afterward instead
        of a second, independently-written copy of it -- the two entry
        points (whole-prompt step 0 vs. chunked step 0) differ only in how
        ``prev_tokens``/``prev_hidden``/``next_pos_list``/
        ``running_prior_kv_len`` were produced, not in what happens to them
        next.

        ``next_pos_list``/``running_prior_kv_len`` are each slot's
        position/prior-kv-len immediately after step 0 -- numerically
        identical to each other at entry in every real call (both start
        equal right after step 0 and both advance by exactly 1 every
        iteration below), the same invariant
        ``CapturedMTPDraftStepGraph``'s own docstring documents."""
        num_reqs = len(slots)
        draft_step_graph = self._get_draft_step_graph(num_reqs) if self.enable_cudagraph else None
        for _ in range(1, k):
            if draft_step_graph is not None:
                step_logits, step_hidden = draft_step_graph.replay_incremental(slots, prev_tokens, prev_hidden, running_prior_kv_len)
            else:
                step_logits, step_hidden = self._mtp_forward_batch(
                    slots,
                    prev_tokens,
                    prev_hidden,
                    running_prior_kv_len,
                    next_pos_list,
                    qo_len=1,
                    is_decode=True,
                )
            # qo_len=1 uniform -> step_logits/step_hidden already have
            # exactly one row per slot in ``slots`` order (request-then-
            # position order degenerates to plain per-request order here),
            # so no index_select/cat needed -- a single batched argmax
            # over the whole tensor plus ONE ``.tolist()`` covers every
            # slot in this step.
            new_prev_tokens = step_logits.argmax(dim=-1).tolist()
            for i in range(num_reqs):
                draft_tokens[slots[i]].append(new_prev_tokens[i])
            prev_tokens = new_prev_tokens
            prev_hidden = step_hidden
            for i in range(num_reqs):
                next_pos_list[i] += 1
                running_prior_kv_len[i] += 1

    def _mtp_sync_and_propose_batch(
        self,
        slots: list[int],
        shifted_input_ids_per_slot: list[list[int]],
        target_hidden_states: torch.Tensor,
        start_pos_list: list[int],
        num_new_tokens: int | list[int],
        k: int,
        step0_logits_last_position_only: bool = False,
    ) -> dict[int, list[int]]:
        """Batched analogue of ``_mtp_sync_and_propose``: one batched step-0
        sync call, followed by ``k-1`` batched autoregressive steps (each
        contributing exactly 1 new token per slot -- always uniform,
        regardless of ``num_new_tokens``).

        ``num_new_tokens`` may be a per-slot RAGGED list (2026-07-17, for
        the recompute-fallback batching round -- mirrors
        ``_mtp_forward_batch``'s identical generalization): different
        slots' committed lengths from a partial-reject round genuinely
        differ, so a single shared scalar can no longer describe every
        real call site (``mtp_prefill_batch``'s uniform-prompt-length
        group and ``mtp_verify_and_commit_batch``'s full-accept group
        still pass a uniform value -- a scalar broadcasts to a uniform
        list, so those call sites are byte-for-byte unaffected;
        ``mtp_verify_and_commit_batch``'s recompute group is the new,
        genuinely-ragged caller).

        Tracks ONE ``running_prior_kv_len`` PER SLOT (a list, not a scalar)
        -- the direct batched generalization of the single-slot 2026-07-17
        fix documented on ``_mtp_forward``: each slot's own draft-sync
        length can differ even though every slot shares this call's ``k``,
        so a single shared counter would silently reintroduce the exact
        bug that fix addressed. Returns a dict keyed by slot id (not a
        positional list) so callers can freely pass a SUBSET of the
        runner's active slots (e.g. only the full-accept group from
        ``mtp_verify_and_commit_batch``) without index confusion.

        ``step0_logits_last_position_only`` (2026-07-18, D1-followup fix,
        default ``False`` preserving every existing call site byte-for-byte):
        forwarded to step 0's eager ``_mtp_forward_batch`` call (see its
        identical parameter's docstring) -- only ``mtp_prefill_batch`` sets
        this, since it already only ever reads each slot's own last-position
        draft token below, and it is the caller whose ``num_new_tokens``
        (a full prompt length) makes the full-sequence vocab-head
        projection this avoids a real cost. **Only actually takes effect
        when step 0 takes the EAGER branch** (``step0_graph is None``) --
        when a caller's ``num_new_tokens`` is uniform AND small enough for
        the captured-graph branch above (e.g. ``mtp_verify_cudagraph_check
        .py``'s deliberately-short-prompt regression test, with
        ``enable_cudagraph=True``), this parameter is silently a no-op: the
        graph path always returns the full, un-gathered shape (see
        ``step0_already_last_only`` below), which is harmless since that
        branch is only ever reached for ``num_new_tokens <=
        _MAX_DECODE_QO_LEN``, far too small for the projection cost to
        matter anyway.
        """
        num_reqs = len(slots)
        if not (len(shifted_input_ids_per_slot) == num_reqs and len(start_pos_list) == num_reqs):
            raise ValueError("slots/shifted_input_ids_per_slot/start_pos_list must have equal length")
        num_new_tokens_list = [num_new_tokens] * num_reqs if isinstance(num_new_tokens, int) else list(num_new_tokens)
        if len(num_new_tokens_list) != num_reqs:
            raise ValueError("num_new_tokens list must have exactly one entry per slot")
        if not all(len(t) == n for t, n in zip(shifted_input_ids_per_slot, num_new_tokens_list)):
            raise ValueError("every slot's shifted_input_ids must have exactly num_new_tokens[i] entries")

        prior_kv_lens_step0 = [self.slot_draft_sync_len[s] for s in slots]
        # 2026-07-17, Phase 3 round 2: step 0 (resync) is graph-capturable
        # whenever num_new_tokens is UNIFORM across this call's slots --
        # always true for the full-accept group (always k+1) and
        # sometimes true for the recompute group (see
        # ``CapturedMTPDraftStepGraph``'s docstring for the full
        # rationale, including why using the fast decode-kernel dispatch
        # here instead of eager's general-kernel choice is still
        # numerically correct). ``prior_kv_lens_step0 == start_pos_list``
        # always holds here (both derive from the same
        # ``slot_draft_sync_len`` snapshot), matching the class's own
        # single-length-list contract.
        # SAFETY (caught before ever running this in anger): mtp_prefill_batch
        # calls this SAME function with num_new_tokens=prompt_len -- a
        # LARGE uniform value (e.g. 4096 for a real W1-S prompt), which
        # would also pass the "uniform" check below. Forcing
        # decode_qo_len=qo_len unconditionally (this class's whole design,
        # matching CapturedBatchDecodeGraph's own convention) is only valid
        # within the real decode/verify kernel's tested range
        # (_MAX_DECODE_QO_LEN=16, the SAME bound
        # build_attention_metadata_batch's eager path already enforces
        # before routing to that kernel) -- MUST NOT be used for a genuine
        # long prefill, which needs the general/chunked kernel instead.
        step0_graph = None
        step0_qo_len_padded = 0
        if (
            self.enable_cudagraph
            and max(num_new_tokens_list) <= _MAX_DECODE_QO_LEN
        ):
            step0_qo_len_padded = max(num_new_tokens_list)
            step0_graph = self._get_draft_step_graph(num_reqs, step0_qo_len_padded)
        # 2026-07-18, D1-followup fix: whether step0's OWN return is already
        # gathered to last-position-only. NOT simply
        # ``step0_logits_last_position_only`` -- a caller (e.g.
        # ``mtp_prefill_batch`` invoked with a SHORT, uniform prompt, as
        # ``mtp_verify_cudagraph_check.py`` deliberately does to regression-
        # test this exact graph path) can request the optimization while
        # STILL legitimately taking the captured-graph branch above (short
        # prompt + ``enable_cudagraph=True`` -> ``num_new_tokens_list[0] <=
        # _MAX_DECODE_QO_LEN`` is true). The graph path always returns the
        # full, un-gathered ``[sum(qo_lens), ...]`` shape (its own
        # docstring/contract), so the optimization must only be treated as
        # "applied" when the eager ``_mtp_forward_batch`` branch actually
        # ran with the flag set -- otherwise this method's own last-index
        # bookkeeping below would silently read the wrong rows. Falling
        # back to full (correct, pre-existing) behavior on the graph path
        # is harmless: that path is only ever reached for small
        # ``num_new_tokens`` (<= ``_MAX_DECODE_QO_LEN``), where the
        # full-position vocab-head cost this fix targets was never large
        # enough to matter in the first place.
        step0_already_last_only = False
        if step0_graph is not None:
            step0_qo_len = step0_qo_len_padded
            is_uniform = len(set(num_new_tokens_list)) == 1
            if is_uniform:
                tokens_for_graph = (
                    [t[0] for t in shifted_input_ids_per_slot] if step0_qo_len == 1 else shifted_input_ids_per_slot
                )
                hidden_for_graph = target_hidden_states
            else:
                tokens_for_graph = []
                hidden_rows = []
                row_start = 0
                for i, n in enumerate(num_new_tokens_list):
                    slot_tokens = list(shifted_input_ids_per_slot[i])
                    if n < step0_qo_len:
                        slot_tokens = slot_tokens + [slot_tokens[-1]] * (step0_qo_len - n)
                    tokens_for_graph.append(slot_tokens)
                    slot_hidden = target_hidden_states[row_start:row_start + n]
                    if n < step0_qo_len:
                        pad_rows = slot_hidden[-1:].expand(step0_qo_len - n, -1)
                        slot_hidden = torch.cat([slot_hidden, pad_rows], dim=0)
                    hidden_rows.append(slot_hidden)
                    row_start += n
                hidden_for_graph = torch.cat(hidden_rows, dim=0)
            step0_logits, step0_hidden = step0_graph.replay(
                slots, tokens_for_graph, hidden_for_graph, prior_kv_lens_step0
            )
            if not is_uniform:
                real_last_indices = torch.tensor(
                    [i * step0_qo_len + num_new_tokens_list[i] - 1 for i in range(num_reqs)],
                    dtype=torch.long, device=step0_logits.device,
                )
                step0_logits = step0_logits.index_select(0, real_last_indices)
                step0_hidden = step0_hidden.index_select(0, real_last_indices)
                step0_already_last_only = True
        else:
            step0_logits, step0_hidden = self._mtp_forward_batch(
                slots,
                shifted_input_ids_per_slot,
                target_hidden_states,
                prior_kv_lens_step0,
                start_pos_list,
                qo_len=num_new_tokens,
                is_decode=all(n == 1 for n in num_new_tokens_list),
                logits_last_position_only=step0_logits_last_position_only,
            )
            step0_already_last_only = step0_logits_last_position_only
        for s, n in zip(slots, num_new_tokens_list):
            self.slot_draft_sync_len[s] += n

        # Per-request row offsets into step0's flattened [sum(num_new_tokens), ...]
        # output (request-then-position order) -- generalizes the old
        # ``i * num_new_tokens + num_new_tokens - 1`` uniform-stride formula
        # to a ragged cumulative-sum offset; reduces to the exact same
        # indices when num_new_tokens_list is uniform.
        row_offsets = [0]
        for n in num_new_tokens_list:
            row_offsets.append(row_offsets[-1] + n)

        # 2026-07-17, Phase 3: batched on-GPU argmax instead of a per-slot
        # Python loop each calling ``.argmax(dim=-1).item()`` separately
        # (``num_reqs`` sequential host round-trips per step, ``k`` steps
        # per round -- up to ``num_reqs * k`` total). ``index_select`` +
        # ONE ``.argmax(dim=-1)`` over every last-position row computes
        # every slot's step-0 draft token in one kernel launch; ONE
        # ``.tolist()`` is the single host round-trip for this step.
        draft_tokens: dict[int, list[int]] = {s: [] for s in slots}
        if step0_already_last_only:
            # Already gathered to [num_reqs, ...] (one row per slot, its
            # own last position) inside ``_mtp_forward_batch`` -- see that
            # method's docstring. Re-indexing with the OLD full-length
            # ``row_offsets`` formula here would read the wrong (or
            # out-of-bounds) rows, so use the tensors directly. Keyed off
            # ``step0_already_last_only`` (which branch ACTUALLY ran), not
            # the raw ``step0_logits_last_position_only`` parameter -- see
            # that variable's own definition above for why they can differ.
            last_logits = step0_logits
            prev_hidden = step0_hidden
        else:
            last_idx_tensor = torch.tensor(
                [row_offsets[i + 1] - 1 for i in range(num_reqs)], dtype=torch.long, device=step0_logits.device
            )
            last_logits = step0_logits.index_select(0, last_idx_tensor)
            prev_hidden = step0_hidden.index_select(0, last_idx_tensor)
        prev_tokens = last_logits.argmax(dim=-1).tolist()
        for i in range(num_reqs):
            draft_tokens[slots[i]].append(prev_tokens[i])

        next_pos_list = [sp + n for sp, n in zip(start_pos_list, num_new_tokens_list)]
        running_prior_kv_len = [prior_kv_lens_step0[i] + num_new_tokens_list[i] for i in range(num_reqs)]
        # 2026-07-17, Phase 3 round 2: this loop's ``prior_kv_lens`` and
        # ``start_pos_list`` are always numerically identical here (both
        # start equal right after step 0 and both advance by exactly 1
        # every iteration below) -- confirmed by inspection, see
        # ``CapturedMTPDraftStepGraph``'s docstring -- which is what makes
        # a single-length-list captured graph replay valid for this
        # specific loop.
        # 2026-07-19, chunked-prefill round: the actual k-1 loop body is now
        # ``_mtp_run_continuation_steps`` (pure code motion, see its
        # docstring) -- shared with ``mtp_prefill_batch``'s new chunked
        # path, which computes step 0 itself (chunk by chunk) and needs
        # this exact same, already-verified tail afterward.
        self._mtp_run_continuation_steps(
            slots, draft_tokens, prev_tokens, prev_hidden, next_pos_list, running_prior_kv_len, k
        )
        return draft_tokens

    def mtp_prefill_batch(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int | None = None,
    ) -> dict[int, dict]:
        """Batched analogue of ``mtp_prefill``: ONE real target prefill
        forward (``_forward_batch``, now able to accept never-forwarded
        slots -- see its 2026-07-17 GDN-init-guard relaxation) covering
        every listed slot, followed by ONE batched draft-sync+propose
        funnel (``_mtp_sync_and_propose_batch``).

        **2026-07-19, continuous-batching round (ragged-length prefill):**
        prompts no longer need the SAME length. Through this round, this
        method hard-asserted every slot's prompt was exactly the same
        length (true for the W1-S/W2-S frozen fixtures, but not for real
        async serving, where different requests genuinely arrive with
        different prompt lengths). This is now generalized -- see the
        ``elif is_uniform_len`` / ``else`` split in the body below -- by
        reusing the SAME per-slot-ragged ``qo_len``/``num_new_tokens`` LIST
        mechanism ``_forward_batch``/``_mtp_sync_and_propose_batch`` already
        built and verified for the 2026-07-17 recompute-fallback batching
        round (see their own docstrings: ``build_attention_metadata_batch``/
        ``build_gdn_metadata_batch`` already construct correct CSR/
        cu_seqlens-style metadata for a ragged per-request qo_len list; the
        general/chunked attention kernel this routes to is already
        documented, in ``vllm/v1/attention/backends/sm120_gqa.py``, as
        correct for "arbitrary mixed prefill+decode batches" -- ragged
        MULTI-REQUEST prefill lengths are exactly that kernel's designed use
        case, not new territory). No new kernel/metadata mechanism was
        needed for this -- confirmed by direct reading before writing this
        branch, the same discipline the chunking round below already
        established. Every EXISTING (uniform-length) caller takes the
        EXACT SAME code path as before, byte-for-byte (the ``elif
        is_uniform_len`` branch is the untouched pre-2026-07-19 code) --
        only a genuinely ragged batch exercises the new ``else`` branch.

        ``chunk_size`` is NOT YET generalized to ragged batches: the
        chunked loop below advances every slot's chunk boundary in
        lockstep from a single shared ``running_kv_len``/
        ``running_draft_len`` counter, which assumes uniform length by
        construction. Combining ragged lengths with ``chunk_size`` raises
        ``NotImplementedError`` with an explicit message rather than
        silently mis-chunking -- a precisely scoped, real follow-on (see
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 21),
        not something this round needed: real async admission prompts are
        far short of the 8K+ context where chunking's memory benefit
        matters, so this is a deliberate scope boundary, not a gap in the
        common case.

        **2026-07-18, D1-followup fix**: both the target model's prefill
        forward and the draft model's step-0 sync forward now pass
        ``logits_last_position_only=True`` -- this method only ever reads
        each slot's OWN last-position logits (the anchor / first draft
        token), never any other position's, so projecting every position
        through the vocab head was 100% wasted work. Found by direct
        instrumentation (``benchmarks/mtp_prefill_batch_memory_diag.py``)
        of the exact c=4/16K-context shape
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 12
        flagged as 4.85x slower than native and peaking at 99.2% of GPU
        memory: at that shape (``qo_len=16384``, ``concurrency=4``, vocab
        248320) EACH of the two ``compute_logits`` calls this method used
        to make allocated an unused ``[65536, 248320]`` bf16 tensor (31040
        MiB, ~30.3 GiB) of which only 4 rows were ever read -- ~60 GiB of
        pure waste, and the second such allocation (already competing with
        the first plus the model+KV-cache baseline for the remaining
        headroom) took 15.2s by itself, direct evidence of near-OOM
        allocator pressure compounding the waste. See that section's
        follow-up entry for the full before/after numbers.

        ``chunk_size`` (2026-07-19, chunked-prefill round, default ``None``
        preserving every existing call site byte-for-byte -- see
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 19
        for the full design/verification writeup): when given, and the
        prompt is longer than ``chunk_size``, splits the single giant
        target-model-forward-then-draft-model-step-0-forward this method
        otherwise does into multiple sequential ``chunk_size``-token
        pieces, so no one ``model.forward()`` call ever processes more
        than ``chunk_size * len(slots)`` tokens at once -- the fix for the
        16K/32K-context near-OOM activation-memory scaling section 12-18
        already root-caused and quantified (peak transient working set
        scales with ``qo_len * concurrency``; chunking bounds ``qo_len``
        to ``chunk_size`` regardless of total prompt length). Matches
        native vLLM's own ``--max-num-batched-tokens=8192`` chunked-prefill
        convention by default (``_DEFAULT_PREFILL_CHUNK_SIZE`` below).

        Both attention's paged KV cache and GDN's recurrent
        ``conv_state``/``ssm_state`` carry over correctly across chunks of
        the SAME slot's own prefill with **no new mechanism** -- both were
        already fully general, just never exercised this way before
        (confirmed by direct reading, not assumed, before writing this):
        ``_forward_batch``'s existing ``kv_lengths``/``commit=True``
        parameters already make each chunk's attention forward a genuine
        paged-KV continuation of the previous chunk (the SM120 general/
        FP8-KV kernel's own module docstring, ``vllm/v1/attention/backends
        /sm120_gqa.py``, already documents itself as correct for "pure
        prefill, chunked-prefill continuation, and arbitrary mixed
        prefill+decode batches" -- chunking here is new USAGE of that
        kernel path, not new kernel work); and GDN's ``has_initial_state``
        (built from ``self.slot_gdn_initialized`` inside
        ``build_gdn_metadata_batch``) is already False only for a
        genuinely fresh slot and unconditionally set True at the end of
        EVERY ``_forward_batch`` call regardless of ``qo_len`` (see that
        method's last few lines) -- so chunk 1 of a fresh slot's prefill
        correctly gets ``has_initial_state=False`` (matching today's
        single-shot behavior) and chunk 2 onward correctly gets
        ``has_initial_state=True``, reading back exactly the
        conv/ssm-state row chunk 1's own forward pass wrote, with zero new
        code in either metadata builder. This is the SAME per-physical-slot
        flag every decode/verify round already relies on for
        cross-ROUND state continuity -- generalized here, for the first
        time, to WITHIN-one-prefill continuity.

        The draft model's own step-0 sync is chunked in lockstep with the
        target model (same chunk boundaries, each chunk's target hidden
        states fed into that SAME chunk's draft forward) -- the other
        piece ``notes/2026-07-18-session-review-and-next-steps.md`` section
        18.6 identified as genuinely unbuilt. Only the LAST chunk's
        anchor/step-0 draft token are ever read; the K-1 further
        autoregressive draft continuation steps after step 0 are
        unaffected (still one uniform, small ``qo_len=1`` loop) and reuse
        ``_mtp_run_continuation_steps`` -- the exact same, already-verified
        tail ``_mtp_sync_and_propose_batch`` itself uses."""
        if self.mtp_model is None or self.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        num_reqs = len(slots)
        if len(prompts_per_slot) != num_reqs:
            raise ValueError("slots and prompts_per_slot must have equal length")
        if num_reqs == 0:
            return {}
        prompt_lens = [len(p) for p in prompts_per_slot]
        is_uniform_len = len(set(prompt_lens)) <= 1
        # 2026-07-19: chunking's loop below advances every slot's chunk
        # boundary from a SINGLE shared running counter -- a genuine
        # uniform-length assumption, not yet generalized (see this method's
        # docstring). Ragged batches that don't actually need chunking
        # (every prompt already fits in one chunk) are unaffected.
        needs_chunking = chunk_size is not None and max(prompt_lens) > chunk_size
        if needs_chunking and not is_uniform_len:
            raise NotImplementedError(
                "mtp_prefill_batch: chunk_size is not yet supported together with "
                "ragged (per-slot different-length) prompts -- either omit "
                "chunk_size (fine for real async-admission prompt lengths, far "
                "short of where chunking's memory benefit matters) or prefill "
                "this slot alone with chunk_size set. See "
                "notes/2026-07-18-session-review-and-next-steps.md section 21."
            )
        for s in slots:
            if self.slot_kv_len[s] != 0 or self.slot_draft_sync_len[s] != 0:
                raise RuntimeError(f"slot {s} is not fresh")
            # Phase 2 (2026-07-18), defense in depth: bootstrap value for
            # the spec-decode GDN mechanism's first-ever verify round on
            # this slot -- already 1 via __init__/reset_slot for any slot
            # that actually went through one of those, set explicitly here
            # too so this invariant holds regardless of how the slot got
            # to "fresh".
            self.slot_num_accepted_tokens[s] = 1

        if not needs_chunking and is_uniform_len:
            # Unchanged from before 2026-07-19: one giant forward each for
            # the target model and the draft model's step-0 sync. This
            # branch is byte-for-byte the prior implementation -- every
            # existing caller that never passes ``chunk_size`` (or whose
            # prompt already fits in one chunk) takes this exact path.
            prompt_len = prompt_lens[0]
            target_logits, target_hidden = self._forward_batch(
                slots,
                prompts_per_slot if prompt_len > 1 else [p[0] for p in prompts_per_slot],
                [0] * num_reqs,
                qo_len=prompt_len,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )
            anchors: dict[int, int] = {}
            shifted_per_slot = []
            for i, s in enumerate(slots):
                # target_logits is [num_reqs, vocab] (last-position-only, see
                # logits_last_position_only above) -- row i IS slot i's last
                # (and only returned) position, no further offset needed.
                row = target_logits[i]
                anchor = int(row.argmax(dim=-1).item())
                anchors[s] = anchor
                shifted_per_slot.append(prompts_per_slot[i][1:] + [anchor])

            draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
                slots,
                shifted_per_slot,
                target_hidden,
                [0] * num_reqs,
                num_new_tokens=prompt_len,
                k=self.num_speculative_tokens,
                step0_logits_last_position_only=True,
            )
            for s in slots:
                self.slot_pending_draft_tokens[s] = draft_tokens_by_slot[s]
            # P3 populate-on-completion (attention half): publish each slot's
            # full committed blocks. GDN completion checkpoint deferred to the
            # two-phase cold path (see mtp_prefill's identical note) -- a
            # single-shot prefill's live GDN state is at prompt_len, not G.
            if self.enable_persistent_prefix_cache:
                for i, s in enumerate(slots):
                    self._publish_committed_blocks(s, prompts_per_slot[i], prompt_lens[i])
            return {s: {"anchor": anchors[s], "draft_tokens": draft_tokens_by_slot[s]} for s in slots}

        if not needs_chunking:
            # NEW (2026-07-19, continuous-batching round): genuinely ragged
            # per-slot prompt lengths, single-shot (no chunking needed).
            # Mirrors the uniform branch above exactly, except ``qo_len``/
            # ``num_new_tokens`` are passed as a per-slot LIST instead of a
            # shared scalar -- both ``_forward_batch`` and
            # ``_mtp_sync_and_propose_batch`` already generalize to this
            # (built 2026-07-17 for the recompute-fallback batching round;
            # see this method's docstring), so no new mechanism is added
            # here, only new USAGE of an already-verified one. Always passes
            # ``prompts_per_slot``/``shifted_per_slot`` as nested per-slot
            # lists (never the uniform branch's flattened-scalar-1 special
            # case) -- correct for every prompt_len, including a degenerate
            # length-1 slot, since a per-slot qo_len list of all-1s is
            # already treated identically to the scalar case by both
            # ``build_attention_metadata_batch`` and ``build_gdn_metadata_
            # batch`` (value-based, not type-based -- see their docstrings).
            target_logits, target_hidden = self._forward_batch(
                slots,
                prompts_per_slot,
                [0] * num_reqs,
                qo_len=prompt_lens,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )
            anchors = {}
            shifted_per_slot = []
            for i, s in enumerate(slots):
                row = target_logits[i]
                anchor = int(row.argmax(dim=-1).item())
                anchors[s] = anchor
                shifted_per_slot.append(prompts_per_slot[i][1:] + [anchor])

            draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
                slots,
                shifted_per_slot,
                target_hidden,
                [0] * num_reqs,
                num_new_tokens=prompt_lens,
                k=self.num_speculative_tokens,
                step0_logits_last_position_only=True,
            )
            for s in slots:
                self.slot_pending_draft_tokens[s] = draft_tokens_by_slot[s]
            # P3 populate-on-completion (attention half): publish each slot's
            # full committed blocks. GDN completion checkpoint deferred to the
            # two-phase cold path (see mtp_prefill's identical note) -- a
            # single-shot prefill's live GDN state is at prompt_len, not G.
            if self.enable_persistent_prefix_cache:
                for i, s in enumerate(slots):
                    self._publish_committed_blocks(s, prompts_per_slot[i], prompt_lens[i])
            return {s: {"anchor": anchors[s], "draft_tokens": draft_tokens_by_slot[s]} for s in slots}

        # Chunked path (2026-07-19). Processes the prompt in
        # ``ceil(prompt_len / chunk_size)`` sequential pieces. Each chunk's
        # target-model forward is a genuine paged-KV-cache continuation of
        # the previous chunk (growing ``kv_lengths``, ``commit=True``) and
        # each chunk's draft-model forward is fed that SAME chunk's target
        # hidden states, mirroring the whole-prompt case's own
        # target_hidden -> draft-model wiring one chunk at a time.
        # Reaching here requires is_uniform_len (checked above), so
        # slots[0]'s own prompt length speaks for the whole batch.
        prompt_len = prompt_lens[0]
        anchors = {}
        step0_logits: torch.Tensor | None = None
        step0_hidden: torch.Tensor | None = None
        chunk_start = 0
        while chunk_start < prompt_len:
            chunk_end = min(chunk_start + chunk_size, prompt_len)
            this_chunk_len = chunk_end - chunk_start
            is_last_chunk = chunk_end == prompt_len
            chunk_tokens_per_slot = [p[chunk_start:chunk_end] for p in prompts_per_slot]

            # Uniform prompt length -> every slot's kv_len/draft_sync_len
            # advances identically chunk to chunk, so reading slots[0]'s
            # own counters is exactly this chunk's shared running value
            # for every slot (already asserted equal-length above).
            running_kv_len = self.slot_kv_len[slots[0]]
            target_logits_chunk, target_hidden_chunk = self._forward_batch(
                slots,
                chunk_tokens_per_slot if this_chunk_len > 1 else [p[0] for p in chunk_tokens_per_slot],
                [running_kv_len] * num_reqs,
                qo_len=this_chunk_len,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )

            if is_last_chunk:
                for i, s in enumerate(slots):
                    anchors[s] = int(target_logits_chunk[i].argmax(dim=-1).item())
                shifted_chunk_per_slot = [
                    prompts_per_slot[i][chunk_start + 1 : prompt_len] + [anchors[slots[i]]]
                    for i in range(num_reqs)
                ]
            else:
                # Not yet at the anchor position -- this chunk's shifted
                # (draft-model-input) tokens are simply the next real
                # prompt tokens, no anchor needed yet.
                shifted_chunk_per_slot = [
                    prompts_per_slot[i][chunk_start + 1 : chunk_end + 1] for i in range(num_reqs)
                ]

            running_draft_len = self.slot_draft_sync_len[slots[0]]
            draft_logits_chunk, draft_hidden_chunk = self._mtp_forward_batch(
                slots,
                # ``_mtp_forward_batch`` special-cases the literal scalar
                # ``qo_len == 1`` to expect a FLAT one-token-per-slot list
                # (matching ``_forward_batch``'s own convention, and the
                # equivalent flattening already applied to the target
                # model's call above) -- only matters for a final remainder
                # chunk of length 1 with a non-default chunk_size; every
                # real fixture/chunk-size combination this round uses
                # divides evenly, so this is a defensive correctness
                # guard, not something exercised by today's measurements.
                shifted_chunk_per_slot if this_chunk_len > 1 else [t[0] for t in shifted_chunk_per_slot],
                target_hidden_chunk,
                [running_draft_len] * num_reqs,
                [running_draft_len] * num_reqs,
                qo_len=this_chunk_len,
                is_decode=False,
                # Every chunk (not just the last) only needs each slot's
                # own last-position output kept -- earlier chunks' full
                # per-position hidden/logits are never read (the physical
                # KV-cache write that lets the NEXT chunk continue is a
                # side effect of the forward call itself, not something
                # this method needs the returned tensor for).
                logits_last_position_only=True,
            )
            for s in slots:
                self.slot_draft_sync_len[s] += this_chunk_len

            if is_last_chunk:
                step0_logits, step0_hidden = draft_logits_chunk, draft_hidden_chunk

            # P3.2 chunk-boundary GDN checkpoints (Fork-2 coarse, step 5): at
            # each NON-FINAL block-aligned chunk_end (every chunk_size boundary,
            # default 8192), the target GDN forward has just ended at chunk_end,
            # so its live state IS the state at chunk_end -- a FREE checkpoint
            # point (no extra forward) giving 8192-granular cross-request partial
            # sharing. Publish each slot's [.., chunk_end) attention blocks, then
            # materialize the persistent GDN checkpoint keyed by the chunk_end
            # tail block (same chained hash => INV3 agreement). The completion-
            # boundary checkpoint (two-phase cold path / fan-out leader) remains
            # separate. The chunked path is uniform-length (asserted above), so
            # every slot shares these boundaries; each slot checkpoints its OWN
            # physical GDN state.
            if (
                self.enable_persistent_prefix_cache
                and not is_last_chunk
                and chunk_end % self.block_size == 0
            ):
                num_chunk_blocks = chunk_end // self.block_size
                for i, s in enumerate(slots):
                    self._publish_committed_blocks(s, prompts_per_slot[i], chunk_end)
                    self.materialize_gdn_checkpoint(
                        s,
                        key=self.block_table[s][num_chunk_blocks - 1],
                        hash_value=self.slot_block_hashes[s][num_chunk_blocks - 1].value,
                        num_tokens=chunk_end,
                    )

            chunk_start = chunk_end

        assert step0_logits is not None and step0_hidden is not None
        prev_tokens = step0_logits.argmax(dim=-1).tolist()
        draft_tokens: dict[int, list[int]] = {s: [prev_tokens[i]] for i, s in enumerate(slots)}
        # Matches _mtp_sync_and_propose_batch's own invariant: both counters
        # equal ``prompt_len`` here (0 + prompt_len), identical to each
        # other, exactly what _mtp_run_continuation_steps expects at entry.
        next_pos_list = [self.slot_draft_sync_len[s] for s in slots]
        running_prior_kv_len = [self.slot_draft_sync_len[s] for s in slots]
        self._mtp_run_continuation_steps(
            slots, draft_tokens, prev_tokens, step0_hidden, next_pos_list, running_prior_kv_len,
            self.num_speculative_tokens,
        )
        for s in slots:
            self.slot_pending_draft_tokens[s] = draft_tokens[s]
        # P3 populate-on-completion (attention half) for the chunked path. The
        # GDN completion checkpoint at G is materialized only by a forward that
        # ends at G (two-phase cold path); chunk boundaries (P3.2) add the rest.
        if self.enable_persistent_prefix_cache:
            for i, s in enumerate(slots):
                self._publish_committed_blocks(s, prompts_per_slot[i], prompt_lens[i])
        return {s: {"anchor": anchors[s], "draft_tokens": draft_tokens[s]} for s in slots}

    @staticmethod
    def _common_prefix_len(prompts: list[list[int]]) -> int:
        """Longest token prefix shared by EVERY prompt in ``prompts`` (direct
        element-by-element comparison -- cheap for the <=4 same-round requests
        the fixed-slot runtime ever admits at once; ``notes/prefix-cache-design
        .md`` sec 5, "P2 -- Fan-out fork": "detect a common token prefix among
        the same-round admit_now batch by direct comparison")."""
        if not prompts:
            return 0
        first = prompts[0]
        max_len = min(len(p) for p in prompts)
        n = 0
        while n < max_len and all(p[n] == first[n] for p in prompts):
            n += 1
        return n

    def mtp_prefill_fanout_batch(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        min_shared_prefix_tokens: int | None = None,
    ) -> dict[int, dict]:
        """P2 fan-out fork -- Pattern A same-round prefix sharing
        (``notes/prefix-cache-design.md`` sec 5, "P2 -- Fan-out fork (Pattern
        A, same-round sharing; self-contained)", and sec 3.5/3.6).

        When ``enable_prefix_cache`` is on AND >=2 same-round requests share a
        token prefix of at least one full block, the shared prefix is computed
        ONCE and referenced by all siblings instead of being recomputed N
        times:

        1. Detect the common token prefix among the same-round admit batch by
           direct comparison (``_common_prefix_len``, cheap for <=4 requests).
        2. Prefill the group LEADER (``slots[0]``) over ``[0, Lc)`` -- where
           ``Lc`` is the block-aligned common-prefix length, capped at
           ``min(prompt_len) - 1`` so every request keeps >=1 suffix token to
           recompute for its own logits (sec 3.8) -- forcing a GDN checkpoint
           boundary there, then ``snapshot_gdn_state`` at ``Lc``.
        3. Continue-prefill the leader's own suffix ``[Lc, leader_len)`` and
           draft-sync the leader over its whole prompt (this writes the draft
           layer's KV for ``[0, Lc)`` into the same shared blocks -- the draft
           layer is in the attention group, sec 3.1).
        4. For each SIBLING: reference the leader's ``[0, Lc)`` attention
           blocks (``BlockPool.reference``, ``ref_cnt += 1``, all 17 attention
           layers share the one block-id namespace), ``restore_gdn_state`` the
           leader's snapshot (``allow_cross_slot=True``), set
           ``slot_draft_sync_len = Lc``, and continue-prefill ONLY the
           sibling's suffix ``[Lc, sibling_len)`` through the already-validated
           chunked-prefill continuation machinery (``_forward_batch`` with
           ``kv_lengths=[Lc]``/``commit``/``is_decode=False`` + the
           ``_mtp_sync_and_propose_batch`` draft funnel).

        No persistent hash index, no eviction -- the shared entry lives only
        for this one admission round (P3 builds the persistent cache). Reuses
        ONLY the P1 block-table/ref-counting substrate + the existing GDN
        snapshot primitive + the chunked-continuation path; nothing here is a
        parallel copy of those.

        **Rollback-safe / byte-identical-to-P1 gate**: with ``enable_prefix_
        cache`` off, OR fewer than 2 requests, OR a block-aligned common
        prefix shorter than ``min_shared_prefix_tokens`` (default
        ``block_size`` -- one full shareable block), this falls back to the
        exact ``mtp_prefill_batch`` path, byte-for-byte P1 behavior.

        Returns the same ``{slot: {"anchor": int, "draft_tokens": list[int]}}``
        shape ``mtp_prefill_batch`` returns, for the leader AND every sibling.
        """
        if self.mtp_model is None or self.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        num_reqs = len(slots)
        if len(prompts_per_slot) != num_reqs:
            raise ValueError("slots and prompts_per_slot must have equal length")
        if num_reqs == 0:
            return {}

        # Fork gate (rollback-safe boundary -- see docstring). Anything that
        # does not clear it takes the exact P1 path, byte-for-byte.
        if not self.enable_prefix_cache or num_reqs < 2:
            return self.mtp_prefill_batch(slots, prompts_per_slot)
        threshold = (
            self.block_size if min_shared_prefix_tokens is None else min_shared_prefix_tokens
        )
        common = self._common_prefix_len(prompts_per_slot)
        min_prompt_len = min(len(p) for p in prompts_per_slot)
        lc = (min(common, min_prompt_len - 1) // self.block_size) * self.block_size
        if lc < self.block_size or lc < threshold:
            return self.mtp_prefill_batch(slots, prompts_per_slot)

        # Defense in depth (R1): every prompt must really share [0, Lc) --
        # true by construction (Lc <= common), asserted cheaply for <=4 reqs.
        for p in prompts_per_slot:
            if p[:lc] != prompts_per_slot[0][:lc]:
                raise RuntimeError("fan-out fork: a request does not share the detected prefix")

        leader = slots[0]
        siblings = slots[1:]
        leader_prompt = prompts_per_slot[0]
        leader_len = len(leader_prompt)
        num_prefix_blocks = lc // self.block_size

        # Same fresh-slot contract as mtp_prefill_batch (every slot starts at
        # kv_len 0 / draft_sync_len 0).
        for s in slots:
            if self.slot_kv_len[s] != 0 or self.slot_draft_sync_len[s] != 0:
                raise RuntimeError(f"slot {s} is not fresh")
            self.slot_num_accepted_tokens[s] = 1

        # --- Leader phase 1: prefill the shared prefix [0, Lc), checkpoint
        # the GDN state there (the fork point). ---
        _, leader_hidden_prefix = self._forward_batch(
            [leader],
            [leader_prompt[:lc]],
            [0],
            qo_len=lc,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        leader_snapshot = self.snapshot_gdn_state(leader)
        shared_blocks = list(self.block_table[leader][:num_prefix_blocks])
        # P3 populate (cross-cutting decision 1): the fan-out leader's phase-1
        # forward ENDS at Lc, so its live GDN state IS the state at Lc -- a
        # correct completion checkpoint for the shared prefix. Publish the
        # leader's [0, Lc) attention blocks + materialize the persistent GDN
        # checkpoint at Lc so a FUTURE round (or another request sharing this
        # prefix) can hit it. Purely additive under the flag (P2 unchanged off).
        if self.enable_persistent_prefix_cache and num_prefix_blocks > 0:
            self._publish_committed_blocks(leader, leader_prompt, lc)
            self.materialize_gdn_checkpoint(
                leader,
                key=self.block_table[leader][num_prefix_blocks - 1],
                hash_value=self.slot_block_hashes[leader][num_prefix_blocks - 1].value,
                num_tokens=lc,
            )

        # --- Leader phase 2: continue-prefill the leader's own suffix
        # [Lc, leader_len) (validated chunked-prefill continuation). Lc is
        # capped at min_prompt_len - 1 <= leader_len - 1, so a non-empty
        # suffix always exists here. ---
        leader_logits_suffix, leader_hidden_suffix = self._forward_batch(
            [leader],
            [leader_prompt[lc:]],
            [lc],
            qo_len=leader_len - lc,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        leader_anchor = int(leader_logits_suffix[0].argmax(dim=-1).item())
        leader_hidden = torch.cat([leader_hidden_prefix, leader_hidden_suffix], dim=0)

        # Leader draft sync over the WHOLE prompt (step-0 resync + K-1
        # continuation steps) -- exactly mtp_prefill_batch's uniform-path
        # draft funnel. Writes the draft layer's KV for [0, leader_len) into
        # the leader's blocks, INCLUDING the shared [0, Lc) blocks the
        # siblings reference next (draft layer is in the attention group).
        leader_drafts = self._mtp_sync_and_propose_batch(
            [leader],
            [leader_prompt[1:] + [leader_anchor]],
            leader_hidden,
            [0],
            num_new_tokens=leader_len,
            k=self.num_speculative_tokens,
            step0_logits_last_position_only=True,
        )
        self.slot_pending_draft_tokens[leader] = leader_drafts[leader]
        # P3 populate: publish the leader's remaining full blocks [Lc, leader_len)
        # (the cursor advances from num_prefix_blocks); attention only -- the
        # leader's live GDN state is now at leader_len, not a block boundary.
        if self.enable_persistent_prefix_cache:
            self._publish_committed_blocks(leader, leader_prompt, leader_len)
        result: dict[int, dict] = {
            leader: {"anchor": leader_anchor, "draft_tokens": leader_drafts[leader]}
        }

        # --- Siblings: reference the leader's [0, Lc) attention blocks +
        # restore the leader's GDN snapshot, then continue-prefill each
        # sibling's own suffix [Lc, sibling_len). ---
        sibling_prompts = prompts_per_slot[1:]
        suffix_per_slot = [p[lc:] for p in sibling_prompts]
        suffix_lens = [len(sfx) for sfx in suffix_per_slot]
        for s in siblings:
            # Reference (ref_cnt += 1) the leader's [0, Lc) blocks -- all 17
            # attention layers share the one block-id namespace (sec 3.1), so
            # one reference call covers target + draft KV for the prefix.
            self.block_table[s] = list(shared_blocks)
            self.block_pool.reference(shared_blocks)
            # Restore the leader's GDN state at Lc into this sibling (sec 3.5
            # step 2); cross-slot by design (source = leader, dest = sibling).
            self.restore_gdn_state(s, leader_snapshot, allow_cross_slot=True)
            # Bookkeeping reproduces exactly the state computing [0, Lc) fresh
            # would have produced (sec 3.5 steps 3-4).
            self.slot_kv_len[s] = lc
            self.slot_gdn_initialized[s] = True
            self.slot_draft_sync_len[s] = lc
            self.slot_num_accepted_tokens[s] = 1

        # P3.2 decode-position populate: seed each sibling's committed-token
        # record (its full prompt) so a later verify-commit can hash decode-
        # produced blocks correctly (they may straddle the prompt tail + decode
        # head). The shared [0, Lc) blocks are the leader's, already published
        # by the leader under the flag; the incremental publish a decode round
        # triggers is idempotent for them (same chained hash) and fresh for the
        # suffix. Purely additive bookkeeping under the flag (the fan-out test
        # runs with the persistent flag off => complete no-op there).
        if self.enable_persistent_prefix_cache:
            for j, s in enumerate(siblings):
                self.slot_committed_tokens[s] = list(sibling_prompts[j])

        # Batched target continue-prefill over the (ragged) suffixes: the
        # validated chunked-prefill continuation (kv_lengths=[Lc], commit,
        # is_decode=False; GDN has_initial_state=True since every sibling's
        # slot_gdn_initialized is now True). Fresh suffix KV writes go to
        # freshly-allocated PRIVATE blocks appended to each sibling's table.
        sibling_logits, sibling_hidden = self._forward_batch(
            siblings,
            suffix_per_slot,
            [lc] * len(siblings),
            qo_len=suffix_lens,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        anchors = {s: int(sibling_logits[i].argmax(dim=-1).item()) for i, s in enumerate(siblings)}
        shifted_suffix_per_slot = [
            sibling_prompts[i][lc + 1 :] + [anchors[siblings[i]]] for i in range(len(siblings))
        ]

        # Batched draft step-0 sync over the suffixes + K-1 continuation
        # steps. prior_kv_lens_step0 = slot_draft_sync_len = Lc for each
        # sibling; the draft attends over the referenced [0, Lc) blocks and
        # writes its own suffix KV into fresh private blocks.
        sibling_drafts = self._mtp_sync_and_propose_batch(
            siblings,
            shifted_suffix_per_slot,
            sibling_hidden,
            [lc] * len(siblings),
            num_new_tokens=suffix_lens,
            k=self.num_speculative_tokens,
            step0_logits_last_position_only=True,
        )
        for s in siblings:
            self.slot_pending_draft_tokens[s] = sibling_drafts[s]
            result[s] = {"anchor": anchors[s], "draft_tokens": sibling_drafts[s]}
        return result

    # ------------------------------------------------------------------
    # P3.1 -- Persistent content-addressed prefix cache
    # (notes/2026-07-19-p3-implementation-plan.md, "P3.1 -- Persistent-cache
    # hit equivalence"). Write path: populate-on-completion (attention) +
    # completion GDN checkpoint. Read path: reconciliation (L = G <= A) +
    # restore-and-continue. All behind enable_persistent_prefix_cache
    # (default False => byte-for-byte P2; L=0 => P2 fan-out/cold).
    # ------------------------------------------------------------------

    def _publish_committed_blocks(self, slot: int, token_ids: list[int], committed_len: int) -> int:
        # Populate-on-completion (attention half, P3.1 step 5/6): publish the
        # full committed blocks [slot_published_blocks[slot], committed_len //
        # block_size) to the content index, growing this slot's chained hash.
        # ONLY committed tokens are hashed/published -- the partial tail and any
        # draft/verify tokens beyond commit are never touched (INV4; mirrors
        # vLLM kv_cache_manager.py:456-465). Write-time dedup (step 6, sec 3.8):
        # if get_cached_block(h_i) hits an existing B', paranoid-verify
        # num_tokens (R7), then swap block_table[slot][i] -> B', touch([B']),
        # free([fresh]) (the recomputed duplicate's memory is reclaimed -- the
        # A>0,G=0 compute-miss reclamation). Else publish fresh. Returns the
        # deepest published boundary in tokens. The draft layer needs no
        # separate publish: it is the 17th attention-group member, so the same
        # block_table[slot] blocks hold its KV (sec 3.1).
        if not self.enable_persistent_prefix_cache:
            return self.slot_published_blocks[slot] * self.block_size
        # P3.2: keep the slot's full committed-token sequence available for
        # hashing decode-produced blocks (which may straddle the prompt tail +
        # decode head). At prefill this seeds it from the prompt
        # (token_ids[:committed_len]); during decode populate the caller has
        # already extended it to slot_kv_len, so this is a no-op there.
        if len(self.slot_committed_tokens[slot]) < committed_len:
            self.slot_committed_tokens[slot] = list(token_ids[:committed_len])
        block_size = self.block_size
        extra_keys = (self.kv_cache_dtype,)
        full_blocks = committed_len // block_size
        cursor = self.slot_published_blocks[slot]
        parent_hash = self.slot_block_hashes[slot][cursor - 1].value if cursor > 0 else None
        for i in range(cursor, full_blocks):
            block_tokens = token_ids[i * block_size : (i + 1) * block_size]
            h_i = hash_block_tokens(parent_hash, block_tokens, extra_keys)
            block_hash = BlockHash(h_i, (i + 1) * block_size)
            self.slot_block_hashes[slot].append(block_hash)
            fresh_block_id = self.block_table[slot][i]
            existing = self.block_pool.get_cached_block(h_i)
            if existing is not None and existing.block_id != fresh_block_id:
                if (
                    existing.block_hash is None
                    or existing.block_hash.num_tokens != (i + 1) * block_size
                ):
                    raise RuntimeError(
                        f"prefix-cache dedup collision: block {existing.block_id} "
                        f"num_tokens={getattr(existing.block_hash, 'num_tokens', None)} "
                        f"!= {(i + 1) * block_size} for hash {h_i} (R7)"
                    )
                self.block_table[slot][i] = existing.block_id
                self.block_pool.touch([existing.block_id])
                self.block_pool.free([fresh_block_id])
            else:
                self.block_pool.cache_block(fresh_block_id, block_hash)
            parent_hash = h_i
        self.slot_published_blocks[slot] = full_blocks
        return full_blocks * block_size

    def publish_committed_decode_blocks(self, slot: int, committed_token_ids: list[int]) -> None:
        """Decode-position populate (attention half, P3.2 step 4). Called by
        both verify-commit funnels AFTER ``slot_kv_len`` advances by the REAL
        committed length: append the newly-committed tokens to the slot's
        committed sequence, then publish any newly-FULL committed blocks
        ``[slot_published_blocks[slot], slot_kv_len[slot] // block_size)``,
        chaining each hash from the last published block (via the incremental
        ``_publish_committed_blocks``).

        ``committed_token_ids`` are the tokens newly written into KV this round
        (``[anchor] + committed[:-1]`` -- the recovery/bonus token is NOT yet
        written, so it is excluded). ONLY committed tokens ever reach here
        (INV4): rejected drafts never advance ``slot_kv_len``, so they are never
        hashed or published (mirrors vLLM ``kv_cache_manager.py:456-465``).
        No-op when the flag is off, and a no-op publish when no NEW full block
        exists yet (the cursor simply does not advance)."""
        if not self.enable_persistent_prefix_cache:
            return
        self.slot_committed_tokens[slot].extend(committed_token_ids)
        self._publish_committed_blocks(
            slot, self.slot_committed_tokens[slot], self.slot_kv_len[slot]
        )

    def _compute_prompt_block_hashes(
        self, token_ids: list[int], max_tokens: int
    ) -> list[BlockHash]:
        # Chained hashes of full blocks, capped at max_tokens (= len(T) - 1 on
        # lookup so the last token is always recomputed for logits; vLLM
        # kv_cache_manager.py:225-231). Pure CPU, O(blocks). Block i's hash
        # depends on all tokens 0..(i+1)*block_size via the chain.
        block_size = self.block_size
        extra_keys = (self.kv_cache_dtype,)
        num_blocks = max_tokens // block_size if max_tokens > 0 else 0
        hashes: list[BlockHash] = []
        parent_hash = None
        for i in range(num_blocks):
            block_tokens = token_ids[i * block_size : (i + 1) * block_size]
            h_i = hash_block_tokens(parent_hash, block_tokens, extra_keys)
            hashes.append(BlockHash(h_i, (i + 1) * block_size))
            parent_hash = h_i
        return hashes

    def reconcile_prefix_hit(self, token_ids: list[int]) -> int:
        # Reconciliation (sec 3.4), specialized to two cache groups (no
        # iterative solver): L = G <= A.
        #   A = attention match -- walk hashes left-to-right, stop at first miss
        #       (the attention group is downward-closed: any prefix of a hit is
        #       a hit). A = matched_blocks * block_size.
        #   G = GDN boundary -- the largest checkpoint boundary Lc <= A with a
        #       GDN checkpoint under the SAME chained hash at Lc. In P3.1
        #       checkpoints exist only at completion boundaries, so G is that
        #       boundary or 0.
        #   L = G (always <= A, always block-aligned). A>0,G=0 => compute miss
        #       (L=0, prefill fresh -- vLLM v1's rule); write-time dedup still
        #       reclaims the recomputed attention blocks.
        if not self.enable_persistent_prefix_cache:
            return 0
        block_size = self.block_size
        hashes = self._compute_prompt_block_hashes(token_ids, len(token_ids) - 1)
        matched_blocks = 0
        for bh in hashes:
            if self.block_pool.get_cached_block(bh.value) is None:
                break
            matched_blocks += 1
        a = matched_blocks * block_size
        if a == 0:
            return 0
        g = 0
        for boundary_blocks in range(matched_blocks, 0, -1):
            hash_value = hashes[boundary_blocks - 1].value
            ckpt_key = self._gdn_ckpt_by_hash.get(hash_value)
            if ckpt_key is None:
                continue
            meta = self.gdn_ckpt_meta.get(ckpt_key)
            if meta is not None and meta["num_tokens"] == boundary_blocks * block_size:
                g = boundary_blocks * block_size
                break
        return g

    def restore_cached_prefix(self, slot: int, token_ids: list[int], L: int) -> None:
        # The sec 3.5 reuse steps 1-4 for a FRESH slot: reserve-and-touch the
        # [0, L) attention blocks BEFORE any forward (R4/INV9), restore the GDN
        # checkpoint at L (reusing the existing cross-slot restore -- P3 writes
        # no second restore), and set the bookkeeping to exactly what computing
        # [0, L) fresh would have produced. R1 addressing proof hook: the
        # checkpoint at L must be tagged with the SAME chained hash as this
        # prompt's boundary block at L -- a wrong-prefix checkpoint is REJECTED,
        # not used.
        block_size = self.block_size
        num_blocks = L // block_size
        if num_blocks <= 0:
            raise RuntimeError(f"restore_cached_prefix requires L >= block_size, got L={L}")
        if self.block_table[slot]:
            raise RuntimeError(f"restore_cached_prefix: slot {slot} is not fresh")
        hashes = self._compute_prompt_block_hashes(token_ids, len(token_ids) - 1)
        if len(hashes) < num_blocks:
            raise RuntimeError(
                f"restore_cached_prefix: prompt yields {len(hashes)} blocks < {num_blocks}"
            )
        matched_ids: list[int] = []
        for i in range(num_blocks):
            block = self.block_pool.get_cached_block(hashes[i].value)
            if block is None:
                raise RuntimeError(
                    f"prefix-cache hit lost block {i} (hash {hashes[i].value}) mid-restore"
                )
            matched_ids.append(block.block_id)
        boundary_hash = hashes[num_blocks - 1].value
        ckpt_key = self._gdn_ckpt_by_hash.get(boundary_hash)
        if ckpt_key is None:
            raise RuntimeError(
                f"prefix-cache hit at L={L} has no GDN checkpoint (hash {boundary_hash})"
            )
        meta = self.gdn_ckpt_meta[ckpt_key]
        if meta["hash_value"] != boundary_hash:
            raise RuntimeError(
                f"R1 reject: GDN checkpoint hash {meta['hash_value']} != prompt boundary "
                f"hash {boundary_hash} -- a wrong-prefix checkpoint is rejected, not used"
            )
        # Step 1: reference the [0, L) attention blocks (all 17 attention layers
        # share the one block-id namespace, sec 3.1). touch revives any block
        # parked at ref_cnt == 0 in the free queue (a freed-but-published block).
        self.block_table[slot] = list(matched_ids)
        self.block_pool.touch(matched_ids)
        # Step 2: restore the GDN checkpoint at L.
        view = self.checkpoint_view(ckpt_key)
        if view is None:
            raise RuntimeError(f"prefix-cache hit at L={L}: checkpoint view is None")
        self.restore_gdn_state(slot, view, allow_cross_slot=True)
        self.slot_gdn_initialized[slot] = True
        # Steps 3-4: bookkeeping reproduces computing [0, L) fresh.
        self.slot_draft_sync_len[slot] = L
        self.slot_kv_len[slot] = L
        self.slot_num_accepted_tokens[slot] = 1
        self.slot_block_hashes[slot] = list(hashes[:num_blocks])
        self.slot_published_blocks[slot] = num_blocks

    def _prefill_cold_with_populate(self, slot: int, prompt: list[int]) -> dict:
        # Two-phase cold prefill that materializes a CORRECT GDN completion
        # checkpoint at G = block_align_down(prompt_len - 1). A single-shot
        # prefill's live GDN state is at prompt_len, NOT at G -- so to capture
        # the state AT G, phase 1 prefills [0, G) (its GDN forward ENDS at G),
        # publishes [0, G//16) + materializes the checkpoint, then phase 2
        # continue-prefills [G, prompt_len). Token-identical to a single-shot
        # cold prefill (it IS chunked prefill with one boundary at G); mirrors
        # the proven P2 fan-out leader two-phase pattern. This is the dedicated
        # test's producing path (the only path that creates a correct completion
        # checkpoint in P3.1).
        if self.slot_kv_len[slot] != 0 or self.slot_draft_sync_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh")
        self.slot_num_accepted_tokens[slot] = 1
        prompt_len = len(prompt)
        k = self.num_speculative_tokens
        g = ((prompt_len - 1) // self.block_size) * self.block_size
        if g >= self.block_size:
            phase1_logits, phase1_hidden = self._forward_batch(
                [slot], [prompt[:g]], [0], qo_len=g, commit=True,
                return_hidden=True, is_decode=False, logits_last_position_only=True,
            )
            self._publish_committed_blocks(slot, prompt, g)
            num_g_blocks = g // self.block_size
            self.materialize_gdn_checkpoint(
                slot,
                key=self.block_table[slot][num_g_blocks - 1],
                hash_value=self.slot_block_hashes[slot][num_g_blocks - 1].value,
                num_tokens=g,
            )
            suffix_len = prompt_len - g
            suffix_tokens = prompt[g:]
            suffix_logits, suffix_hidden = self._forward_batch(
                [slot],
                [suffix_tokens] if suffix_len > 1 else [suffix_tokens[0]],
                [g], qo_len=suffix_len, commit=True,
                return_hidden=True, is_decode=False, logits_last_position_only=True,
            )
            anchor = int(suffix_logits[0].argmax(dim=-1).item())
            hidden = torch.cat([phase1_hidden, suffix_hidden], dim=0)
            draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
                [slot], [prompt[1:] + [anchor]], hidden, [0],
                num_new_tokens=prompt_len, k=k, step0_logits_last_position_only=True,
            )
            self._publish_committed_blocks(slot, prompt, prompt_len)
            self.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
            return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}
        # Prompt too short for a full-block boundary < prompt_len
        # (prompt_len <= block_size): plain single-shot cold prefill; publish
        # whatever full blocks exist (the completion checkpoint needs a forward
        # ending at G >= block_size, impossible here).
        target_logits, target_hidden = self._forward_batch(
            [slot],
            [prompt] if prompt_len > 1 else [prompt[0]],
            [0], qo_len=prompt_len, commit=True,
            return_hidden=True, is_decode=False, logits_last_position_only=True,
        )
        anchor = int(target_logits[0].argmax(dim=-1).item())
        draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
            [slot], [prompt[1:] + [anchor]], target_hidden, [0],
            num_new_tokens=prompt_len, k=k, step0_logits_last_position_only=True,
        )
        self._publish_committed_blocks(slot, prompt, prompt_len)
        self.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
        return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}

    def _prefill_hit_with_cache(self, slot: int, prompt: list[int], L: int) -> dict:
        # Restore-and-continue hit (sec 3.5): restore the [0, L) attention
        # blocks + GDN checkpoint at L (restore_cached_prefix), then continue-
        # prefill the suffix [L, prompt_len) via the EXACT validated continuation
        # the P2 fan-out sibling path uses (_forward_batch([s],[suffix],[L],
        # qo_len=suffix_len, commit, is_decode=False) + _mtp_sync_and_propose_
        # batch([s],[prompt[L+1:]+[anchor]], hidden,[L], num_new_tokens=
        # suffix_len, k=K)). L=0 never reaches here.
        if self.slot_kv_len[slot] != 0 or self.slot_draft_sync_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh")
        self.restore_cached_prefix(slot, prompt, L)
        prompt_len = len(prompt)
        suffix_len = prompt_len - L
        k = self.num_speculative_tokens
        suffix_tokens = prompt[L:]
        suffix_logits, suffix_hidden = self._forward_batch(
            [slot],
            [suffix_tokens] if suffix_len > 1 else [suffix_tokens[0]],
            [L], qo_len=suffix_len, commit=True,
            return_hidden=True, is_decode=False, logits_last_position_only=True,
        )
        anchor = int(suffix_logits[0].argmax(dim=-1).item())
        draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
            [slot], [prompt[L + 1 :] + [anchor]], suffix_hidden, [L],
            num_new_tokens=suffix_len, k=k, step0_logits_last_position_only=True,
        )
        # Publish the suffix's full committed blocks (attention) so future
        # longer requests can hit deeper. The GDN checkpoint at the new
        # completion boundary is deferred (live GDN state is at prompt_len, not
        # a block boundary -- a correct one needs a forward ending there).
        self._publish_committed_blocks(slot, prompt, prompt_len)
        self.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
        return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}

    def mtp_prefill_warm_continue(self, slot: int, prompt: list[int], prior_len: int) -> dict:
        # P4b session affinity -- zero-restore continuation of a WARM slot. The
        # slot already holds [0, prior_len) KV + GDN LIVE (turn-1's committed
        # content; it was retained, never reset_slot-ed), so prefill ONLY the
        # suffix [prior_len, prompt_len). Mirrors _prefill_hit_with_cache MINUS
        # the restore_cached_prefix call -- there is nothing to restore, because
        # the boundary state IS turn-1's live state (even more faithful than a
        # restore). Gated: only the server's session-affinity admission path ever
        # calls this; with the flag off it is never reached, so the frozen
        # P0-P3 + P4a paths stay byte-for-byte untouched. Reuses ONLY the private
        # primitives the hit path already uses (_forward_batch,
        # _mtp_sync_and_propose_batch, _publish_committed_blocks).
        if not self.enable_persistent_prefix_cache:
            raise RuntimeError("mtp_prefill_warm_continue requires the persistent prefix cache")
        if self.mtp_model is None or self.num_speculative_tokens is None:
            raise RuntimeError("mtp_prefill_warm_continue: no MTP draft model loaded")
        if self.slot_kv_len[slot] != prior_len or not self.slot_gdn_initialized[slot]:
            raise RuntimeError(
                f"mtp_prefill_warm_continue: slot {slot} is not warm at prior_len={prior_len} "
                f"(kv_len={self.slot_kv_len[slot]}, gdn_init={self.slot_gdn_initialized[slot]})"
            )
        # Authoritative prefix match: the slot's committed-token record (P1+C1)
        # must equal the new prompt through prior_len. Any mismatch => the caller
        # must fall back to the cold/restore path (the server catches this raise).
        if self.slot_committed_tokens[slot][:prior_len] != prompt[:prior_len]:
            raise RuntimeError(
                "mtp_prefill_warm_continue: prefix mismatch -- caller must fall back"
            )
        # Reset draft state to the committed boundary; discard turn-1's draft-ahead.
        # _mtp_sync_and_propose_batch requires prior_kv_lens_step0 == start_pos_list
        # (its own contract), so slot_draft_sync_len MUST be prior_len before the
        # call below with start_pos_list=[prior_len]. The committed draft KV
        # [0, prior_len) is valid (the draft layer is the 17th attention-group
        # member sharing block_table[slot]); the speculative [prior_len, old) is
        # overwritten by the suffix step-0 sync + K-1 continuation steps, and any
        # stale KV beyond the new draft_sync_len is never read. slot_num_accepted_
        # tokens=1 bootstraps the spec-decode GDN mechanism for the next verify
        # round, exactly as restore_cached_prefix and the fanout sibling do.
        self.slot_draft_sync_len[slot] = prior_len
        self.slot_num_accepted_tokens[slot] = 1
        self.slot_pending_draft_tokens[slot] = None
        prompt_len = len(prompt)
        suffix_len = prompt_len - prior_len
        k = self.num_speculative_tokens
        suffix_tokens = prompt[prior_len:]
        suffix_logits, suffix_hidden = self._forward_batch(
            [slot],
            [suffix_tokens] if suffix_len > 1 else [suffix_tokens[0]],
            [prior_len], qo_len=suffix_len, commit=True,
            return_hidden=True, is_decode=False, logits_last_position_only=True,
        )
        anchor = int(suffix_logits[0].argmax(dim=-1).item())
        draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
            [slot], [prompt[prior_len + 1 :] + [anchor]], suffix_hidden, [prior_len],
            num_new_tokens=suffix_len, k=k, step0_logits_last_position_only=True,
        )
        # Publish the suffix's full committed blocks (attention) so future longer
        # requests can hit deeper. The completion GDN checkpoint at the new boundary
        # is deferred (live GDN is at prompt_len, not a block boundary -- mirrors
        # the hit path); warm sessions continue in place anyway, and the content-
        # hash fallback still hits at turn-1's completion checkpoint.
        self._publish_committed_blocks(slot, prompt, prompt_len)
        self.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
        return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}

    def mtp_prefill_with_cache(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int | None = None,
    ) -> dict[int, dict]:
        # P3.3 -- UNIFIED production prefill entrypoint (test-driven in P3.1;
        # production-wired in P3.3a). ONE batched path composes persistent-hit
        # + P2 same-round fan-out + cold:
        #   * Per slot, reconcile_prefix_hit yields L.
        #   * HIT set (L>0): restore_cached_prefix each (references the [0,L_s)
        #     attention blocks + restores the GDN checkpoint at L_s), then
        #     continue-prefill ALL hit slots' RAGGED suffixes in ONE batched
        #     _forward_batch + ONE _mtp_sync_and_propose_batch -- the proven P2
        #     fan-out sibling ragged-suffix pattern (mtp_prefill_fanout_batch's
        #     sibling block) generalized from a SHARED Lc to PER-SLOT L_s
        #     (ragged kv_lengths=[L_s], qo_len=[suffix_len_s]). A single hit
        #     slot reduces TOKEN-IDENTICALLY to _prefill_hit_with_cache: a
        #     1-element qo_len LIST normalizes to the same flat tokens /
        #     positions / GDN metadata as the scalar qo_len the helper passes
        #     (qo_len==1 takes build_gdn_metadata_batch's decode branch either
        #     way; qo_len>1 resolves slot_initialized=[True] either way).
        #   * COLD set (L==0): >=2 cold slots hand to the EXISTING
        #     mtp_prefill_fanout_batch (P2 same-round fork detection among cold
        #     slots, falling back to mtp_prefill_batch); a single cold slot uses
        #     the two-phase _prefill_cold_with_populate so its COMPLETION GDN
        #     checkpoint is materialized (a future re-request must hit it --
        #     mtp_prefill_batch/fanout publish attention blocks but materialize
        #     no per-slot completion checkpoint, so routing a lone cold slot
        #     there would silently disable hit-after-cold). A persistent hit
        #     always wins over a same-round fork (hits are removed before the
        #     cold hand-off).
        # Returns the same {slot: {"anchor","draft_tokens"}} shape as
        # mtp_prefill_batch. Rollback spine: flag off => delegate to
        # mtp_prefill_batch (byte-for-byte P2); a slot's L=0 => the cold/fanout
        # path => byte-for-byte P2 OUTPUT for it.
        if self.mtp_model is None or self.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        if len(slots) != len(prompts_per_slot):
            raise ValueError("slots and prompts_per_slot must have equal length")
        if not self.enable_persistent_prefix_cache:
            return self.mtp_prefill_batch(slots, prompts_per_slot, chunk_size)
        if len(slots) == 0:
            return {}

        # Per-slot reconciliation (sec 3.4): L = G <= A.
        L_per_slot = [self.reconcile_prefix_hit(p) for p in prompts_per_slot]
        hit_idx = [i for i, L in enumerate(L_per_slot) if L > 0]
        cold_idx = [i for i, L in enumerate(L_per_slot) if L == 0]

        result: dict[int, dict] = {}

        # --- HIT set: restore each, then batched ragged continue-prefill. ---
        # When chunk_size is given and any hit suffix exceeds it, the hit
        # block switches to a chunked continue-prefill (Phase A INV8 lift,
        # 2026-07-20): uniform-suffix batches use the batched chunked pattern
        # from mtp_prefill_batch's cold chunked path; ragged-suffix batches
        # process each slot independently with chunking. Suffixes that fit
        # in one chunk take the EXISTING monolithic path, byte-for-byte.
        if hit_idx:
            hit_slots = [slots[i] for i in hit_idx]
            hit_prompts = [prompts_per_slot[i] for i in hit_idx]
            hit_L = [L_per_slot[i] for i in hit_idx]
            k = self.num_speculative_tokens
            # Same fresh-slot contract as _prefill_hit_with_cache; restore the
            # [0, L_s) attention blocks + GDN checkpoint at L_s (reserve-and-
            # touch before any forward, R4/INV9) and set the bookkeeping to
            # exactly what computing [0, L_s) fresh would have produced.
            for s, p, L in zip(hit_slots, hit_prompts, hit_L):
                if self.slot_kv_len[s] != 0 or self.slot_draft_sync_len[s] != 0:
                    raise RuntimeError(f"slot {s} is not fresh")
                self.restore_cached_prefix(s, p, L)
            # Ragged suffix continue-prefill (generalizes the fan-out sibling
            # block): kv_lengths=[L_s], qo_len=[suffix_len_s]. Fresh suffix KV
            # writes go to freshly-allocated PRIVATE blocks appended to each
            # slot's table; GDN has_initial_state=True (restore initialized it).
            suffix_per_slot = [p[L:] for p, L in zip(hit_prompts, hit_L)]
            suffix_lens = [len(sfx) for sfx in suffix_per_slot]

            use_chunked_hit = chunk_size is not None and max(suffix_lens) > chunk_size

            if not use_chunked_hit:
                # === MONOLITHIC PATH (byte-for-byte unchanged) ===
                suffix_logits, suffix_hidden = self._forward_batch(
                    hit_slots,
                    suffix_per_slot,
                    list(hit_L),
                    qo_len=suffix_lens,
                    commit=True,
                    return_hidden=True,
                    is_decode=False,
                    logits_last_position_only=True,
                )
                anchors = {
                    s: int(suffix_logits[i].argmax(dim=-1).item()) for i, s in enumerate(hit_slots)
                }
                shifted_suffix_per_slot = [
                    hit_prompts[i][hit_L[i] + 1 :] + [anchors[hit_slots[i]]]
                    for i in range(len(hit_slots))
                ]
                # Batched draft step-0 sync over the suffixes + K-1 continuation
                # steps. prior_kv_lens_step0 = slot_draft_sync_len = L_s for each
                # slot; the draft attends over the referenced [0, L_s) blocks and
                # writes its own suffix KV into fresh private blocks.
                hit_drafts = self._mtp_sync_and_propose_batch(
                    hit_slots,
                    shifted_suffix_per_slot,
                    suffix_hidden,
                    list(hit_L),
                    num_new_tokens=suffix_lens,
                    k=k,
                    step0_logits_last_position_only=True,
                )
                for i, s in enumerate(hit_slots):
                    # Publish the suffix's full committed blocks (attention) so
                    # future longer requests can hit deeper; the completion GDN
                    # checkpoint at the new boundary is deferred (live GDN state is
                    # at prompt_len, not a block boundary).
                    self._publish_committed_blocks(s, hit_prompts[i], len(hit_prompts[i]))
                    self.slot_pending_draft_tokens[s] = hit_drafts[s]
                    result[s] = {"anchor": anchors[s], "draft_tokens": hit_drafts[s]}

            elif len(set(suffix_lens)) == 1:
                # === UNIFORM SUFFIX CHUNKED PATH ===
                # All hit slots share the same suffix length (the common
                # benchmark scenario). Follows the EXACT pattern from
                # mtp_prefill_batch's cold chunked path, but with per-slot
                # running_kv_lens (hit_L values may differ per slot even
                # though suffix_len is uniform).
                suffix_len = suffix_lens[0]
                num_hit = len(hit_slots)
                # Bound TOTAL tokens per chunk at ~chunk_size (matching native
                # vLLM's max_num_batched_tokens): each slot gets chunk_size //
                # num_hit tokens per chunk, so the batched forward processes
                # num_hit × (chunk_size // num_hit) ≈ chunk_size tokens total.
                effective_chunk = max(1, chunk_size // num_hit)
                running_kv_lens = list(hit_L)
                running_draft_lens = list(hit_L)
                anchors = {}
                step0_logits = None
                step0_hidden = None
                chunk_start = 0
                while chunk_start < suffix_len:
                    chunk_end = min(chunk_start + effective_chunk, suffix_len)
                    this_chunk_len = chunk_end - chunk_start
                    is_last_chunk = chunk_end == suffix_len
                    chunk_tokens_per_slot = [
                        p[hit_L[i] + chunk_start : hit_L[i] + chunk_end]
                        for i, p in enumerate(hit_prompts)
                    ]

                    target_logits_chunk, target_hidden_chunk = self._forward_batch(
                        hit_slots,
                        chunk_tokens_per_slot if this_chunk_len > 1 else [t[0] for t in chunk_tokens_per_slot],
                        list(running_kv_lens),
                        qo_len=this_chunk_len,
                        commit=True,
                        return_hidden=True,
                        is_decode=False,
                        logits_last_position_only=True,
                    )
                    for i in range(num_hit):
                        running_kv_lens[i] += this_chunk_len

                    if is_last_chunk:
                        for i, s in enumerate(hit_slots):
                            anchors[s] = int(target_logits_chunk[i].argmax(dim=-1).item())
                        shifted_chunk_per_slot = [
                            hit_prompts[i][hit_L[i] + chunk_start + 1 : hit_L[i] + suffix_len] + [anchors[hit_slots[i]]]
                            for i in range(num_hit)
                        ]
                    else:
                        shifted_chunk_per_slot = [
                            hit_prompts[i][hit_L[i] + chunk_start + 1 : hit_L[i] + chunk_end + 1]
                            for i in range(num_hit)
                        ]

                    draft_logits_chunk, draft_hidden_chunk = self._mtp_forward_batch(
                        hit_slots,
                        shifted_chunk_per_slot if this_chunk_len > 1 else [t[0] for t in shifted_chunk_per_slot],
                        target_hidden_chunk,
                        list(running_draft_lens),
                        list(running_draft_lens),
                        qo_len=this_chunk_len,
                        is_decode=False,
                        logits_last_position_only=True,
                    )
                    for i, s in enumerate(hit_slots):
                        self.slot_draft_sync_len[s] += this_chunk_len
                        running_draft_lens[i] += this_chunk_len

                    if is_last_chunk:
                        step0_logits, step0_hidden = draft_logits_chunk, draft_hidden_chunk

                    if (
                        self.enable_persistent_prefix_cache
                        and not is_last_chunk
                    ):
                        for i, s in enumerate(hit_slots):
                            abs_chunk_end = hit_L[i] + chunk_end
                            if abs_chunk_end % self.block_size == 0:
                                num_chunk_blocks = abs_chunk_end // self.block_size
                                self._publish_committed_blocks(s, hit_prompts[i], abs_chunk_end)
                                self.materialize_gdn_checkpoint(
                                    s,
                                    key=self.block_table[s][num_chunk_blocks - 1],
                                    hash_value=self.slot_block_hashes[s][num_chunk_blocks - 1].value,
                                    num_tokens=abs_chunk_end,
                                )

                    chunk_start = chunk_end

                assert step0_logits is not None and step0_hidden is not None
                prev_tokens = step0_logits.argmax(dim=-1).tolist()
                hit_drafts: dict[int, list[int]] = {s: [prev_tokens[i]] for i, s in enumerate(hit_slots)}
                next_pos_list = [self.slot_draft_sync_len[s] for s in hit_slots]
                running_prior_kv_len = [self.slot_draft_sync_len[s] for s in hit_slots]
                self._mtp_run_continuation_steps(
                    hit_slots, hit_drafts, prev_tokens, step0_hidden, next_pos_list, running_prior_kv_len, k,
                )
                for i, s in enumerate(hit_slots):
                    self._publish_committed_blocks(s, hit_prompts[i], len(hit_prompts[i]))
                    self.slot_pending_draft_tokens[s] = hit_drafts[s]
                    result[s] = {"anchor": anchors[s], "draft_tokens": hit_drafts[s]}

            else:
                # === RAGGED SUFFIX PER-SLOT CHUNKED PATH ===
                # Different suffix lengths per slot: process each hit slot
                # independently with chunking. Simpler and correct, though
                # less efficient than batched ragged chunking.
                for idx, s in enumerate(hit_slots):
                    suffix_len = suffix_lens[idx]
                    L = hit_L[idx]
                    prompt = hit_prompts[idx]
                    running_kv_len = L
                    running_draft_len = L
                    chunk_start = 0
                    step0_logits_s = None
                    step0_hidden_s = None
                    anchor_s = None
                    while chunk_start < suffix_len:
                        chunk_end = min(chunk_start + chunk_size, suffix_len)
                        this_chunk_len = chunk_end - chunk_start
                        is_last_chunk = chunk_end == suffix_len
                        chunk_tokens = prompt[L + chunk_start : L + chunk_end]

                        target_logits_chunk, target_hidden_chunk = self._forward_batch(
                            [s],
                            [chunk_tokens] if this_chunk_len > 1 else [chunk_tokens[0]],
                            [running_kv_len],
                            qo_len=this_chunk_len,
                            commit=True,
                            return_hidden=True,
                            is_decode=False,
                            logits_last_position_only=True,
                        )
                        running_kv_len += this_chunk_len

                        if is_last_chunk:
                            anchor_s = int(target_logits_chunk[0].argmax(dim=-1).item())
                            shifted_chunk = prompt[L + chunk_start + 1 : L + suffix_len] + [anchor_s]
                        else:
                            shifted_chunk = prompt[L + chunk_start + 1 : L + chunk_end + 1]

                        draft_logits_chunk, draft_hidden_chunk = self._mtp_forward_batch(
                            [s],
                            [shifted_chunk] if this_chunk_len > 1 else [shifted_chunk[0]],
                            target_hidden_chunk,
                            [running_draft_len],
                            [running_draft_len],
                            qo_len=this_chunk_len,
                            is_decode=False,
                            logits_last_position_only=True,
                        )
                        self.slot_draft_sync_len[s] += this_chunk_len
                        running_draft_len += this_chunk_len

                        if is_last_chunk:
                            step0_logits_s, step0_hidden_s = draft_logits_chunk, draft_hidden_chunk

                        if (
                            self.enable_persistent_prefix_cache
                            and not is_last_chunk
                            and (L + chunk_end) % self.block_size == 0
                        ):
                            abs_end = L + chunk_end
                            num_blocks = abs_end // self.block_size
                            self._publish_committed_blocks(s, prompt, abs_end)
                            self.materialize_gdn_checkpoint(
                                s,
                                key=self.block_table[s][num_blocks - 1],
                                hash_value=self.slot_block_hashes[s][num_blocks - 1].value,
                                num_tokens=abs_end,
                            )

                        chunk_start = chunk_end

                    assert step0_logits_s is not None and step0_hidden_s is not None
                    prev_tokens_s = step0_logits_s.argmax(dim=-1).tolist()
                    draft_tokens_s: dict[int, list[int]] = {s: [prev_tokens_s[0]]}
                    next_pos_s = [self.slot_draft_sync_len[s]]
                    prior_kv_s = [self.slot_draft_sync_len[s]]
                    self._mtp_run_continuation_steps(
                        [s], draft_tokens_s, prev_tokens_s, step0_hidden_s, next_pos_s, prior_kv_s, k,
                    )
                    self._publish_committed_blocks(s, prompt, len(prompt))
                    self.slot_pending_draft_tokens[s] = draft_tokens_s[s]
                    result[s] = {"anchor": anchor_s, "draft_tokens": draft_tokens_s[s]}

        # --- COLD set: P2 same-round fan-out fork when >=2 cold slots; a lone
        #     cold slot uses the two-phase populate prefill (completion GDN
        #     checkpoint required for hit-after-cold; see docstring). ---
        if cold_idx:
            cold_slots = [slots[i] for i in cold_idx]
            cold_prompts = [prompts_per_slot[i] for i in cold_idx]
            if len(cold_slots) >= 2:
                result.update(self.mtp_prefill_fanout_batch(cold_slots, cold_prompts))
            else:
                result[cold_slots[0]] = self._prefill_cold_with_populate(
                    cold_slots[0], cold_prompts[0]
                )

        return result

    def mtp_verify_and_commit_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        draft_tokens: dict[int, list[int]],
    ) -> dict[int, dict]:
        """Batched analogue of ``mtp_verify_and_commit`` -- **Phase 2,
        2026-07-18 rewrite** (``notes/2026-07-17-post-ragged-round-next-steps.md``
        section 10/11): now uses the REAL spec-decode GDN mechanism
        (``verify_batch_spec`` / ``build_gdn_metadata_spec_batch`` /
        ``_ssm_spec_row``) instead of Phase 0-3's chunked-GDN-metadata +
        snapshot/restore + recompute-forward mechanism. This eliminates an
        entire class of extra work this function used to do on every
        partial-reject round (``benchmarks/mtp_batch_recompute_cost_diag.py``
        found that was 84.4% of real rounds, ~56% of round wall time):
        there is no more snapshot, no more restore, no more separate
        recompute forward pass, and (as a direct consequence) no more
        full-accept/recompute GROUP SPLIT at all -- every slot in this
        call, regardless of its own accept/reject outcome, is handled by
        the exact same code path below.

        Why one uniform code path is now correct (not just simpler): GDN's
        recurrent state, under the real spec-decode kernel
        (``fused_sigmoid_gating_delta_rule_update_kernel`` -- re-verified
        directly against source this round, see ``_ssm_spec_row``'s
        docstring), computes a causally-valid PER-POSITION output for
        every one of the K+1 candidate positions in a single verify
        forward, unconditionally, regardless of which candidates later
        turn out to be real -- exactly like attention already was
        (content/position-addressed, so a rejected position's KV is
        simply never read again). Only the recurrent STATE COMMIT (which
        physical row is read next round) is acceptance-aware, via
        ``num_accepted_tokens``/``_ssm_spec_row`` -- the per-position
        OUTPUT itself needs no rollback or recomputation. This means
        every slot's hidden states for positions ``0..committed_len-1``
        are already sitting in ``verify_hidden``, correct, from the ONE
        verify forward this method issues -- a straight ragged SLICE
        (never a second forward pass) is all the draft resync step needs,
        for full-accept slots (``committed_len == k+1``) exactly as much
        as for any partial-reject slot (``committed_len < k+1``).

        Persistent per-slot bookkeeping this round introduces:
        ``self.slot_num_accepted_tokens`` (this slot's real committed
        length from ITS OWN last verify round, or bootstrap 1 right after
        a real prefill) -- read by ``build_gdn_metadata_spec_batch`` to
        select which of last round's K+1 dedicated SSM rows holds the
        valid state to resume from, and updated here after every round.

        **2026-07-18, Phase 2 CUDA-graph reconciliation**: the verify
        forward now goes through a CUDA-graph replay
        (``CapturedBatchDecodeGraph``, via ``self._get_verify_graph``,
        rebuilt this round to fill its GDN metadata via the same
        spec-decode mechanism this method uses -- see that class's
        docstring) whenever ``self.enable_cudagraph`` is on AND this
        runner has enough spare slot capacity (``num_slots >= 2*len(slots)``).
        Falls back to the eager ``verify_batch_spec`` path -- correctly,
        not silently -- whenever the graph isn't available for this
        call's batch_size (``enable_cudagraph`` off, matching every
        existing correctness suite; or a caller's active-slot count isn't
        one this runner precaptured). Since Phase 2 removed the separate
        recompute forward entirely, there is only ONE verify-shaped
        forward per round now, always at ``qo_len=k+1`` -- unlike Phase 3's
        old recompute-forward-graph-reuse special case (which needed a
        DIFFERENT qo_len per ragged recompute group), this graph lookup is
        now always the exact same shape every round, for every slot.

        **2026-07-18, Phase B**: ``mtp_verify_and_commit`` (the
        singular/looped sibling) was, through this point, intentionally
        NOT migrated -- it used the old chunked + snapshot/restore +
        recompute-forward mechanism unconditionally. Phase B (see that
        method's own docstring) migrated it to this SAME spec-decode
        mechanism, applied at batch_size=1 -- both production verify paths
        now share one mechanism. ``snapshot_gdn_state``/``restore_gdn_state``
        remain in the codebase regardless (a falsifier check confirmed
        ``benchmarks/mtp_gdn_rollback_check.py`` and several other
        diagnostics test/use them directly, independent of either verify
        path), just no longer called from ANY production verify path.

        Returns a dict keyed by slot id, each value shaped exactly like
        ``mtp_verify_and_commit``'s own return dict (plus ``next_anchor``/
        ``next_draft_tokens``) -- the external contract is unchanged."""
        k = len(draft_tokens[slots[0]])
        drafts_by_slot = {s: [anchors[s]] + draft_tokens[s] for s in slots}
        drafts = [drafts_by_slot[s] for s in slots]
        kv_lens_before = {s: self.slot_kv_len[s] for s in slots}
        num_accepted_prev = [self.slot_num_accepted_tokens[s] for s in slots]

        graph = self._get_verify_graph(len(slots), k + 1) if self.enable_cudagraph else None
        if graph is not None:
            verify_logits, verify_hidden = graph.replay(
                slots,
                drafts,
                [kv_lens_before[s] for s in slots],
                commit=False,
                return_hidden=True,
                num_accepted_tokens_prev=num_accepted_prev,
            )
        else:
            verify_logits, verify_hidden = self.verify_batch_spec(
                slots,
                drafts,
                [kv_lens_before[s] for s in slots],
                num_accepted_tokens_prev=num_accepted_prev,
                return_hidden=True,
            )

        decisions = determine_accept_reject_batch(slots, drafts_by_slot, verify_logits, k)

        real_new_tokens = {s: [anchors[s]] + decisions[s]["committed"][:-1] for s in slots}
        next_anchors = {s: decisions[s]["committed"][-1] for s in slots}
        committed_lens = {s: decisions[s]["num_accepted"] + 1 for s in slots}

        for s in slots:
            self.slot_kv_len[s] = kv_lens_before[s] + committed_lens[s]
            self.slot_num_accepted_tokens[s] = committed_lens[s]
        # P3.2 decode-position populate (per slot): publish any newly-FULL
        # committed blocks now that slot_kv_len advanced by each slot's REAL
        # committed length (only committed tokens; INV4). No-op off-flag.
        if self.enable_persistent_prefix_cache:
            for s in slots:
                self.publish_committed_decode_blocks(s, real_new_tokens[s])

        # Ragged slice of the ONE verify forward's hidden states -- see
        # this method's docstring for why this is valid for EVERY slot
        # regardless of committed_len, not just full-accept ones.
        real_new_hidden: dict[int, torch.Tensor] = {}
        for i, s in enumerate(slots):
            real_new_hidden[s] = verify_hidden[i * (k + 1) : i * (k + 1) + committed_lens[s]]

        shifted = [real_new_tokens[s][1:] + [next_anchors[s]] for s in slots]
        hidden_concat = torch.cat([real_new_hidden[s] for s in slots], dim=0)
        start_pos_list = [self.slot_draft_sync_len[s] for s in slots]
        next_drafts_batch = self._mtp_sync_and_propose_batch(
            slots,
            shifted,
            hidden_concat,
            start_pos_list,
            num_new_tokens=[committed_lens[s] for s in slots],
            k=k,
        )

        result: dict[int, dict] = {}
        for s in slots:
            self.slot_pending_draft_tokens[s] = next_drafts_batch[s]
            result[s] = {
                **decisions[s],
                "next_anchor": next_anchors[s],
                "next_draft_tokens": next_drafts_batch[s],
            }
        return result


class CapturedBatchDecodeGraph:
    """CUDA-graph-captured batch decode/verify for a FIXED batch size and
    FIXED ``qo_len`` (1 = pure decode, >1 = MTP/speculative-decode verify,
    e.g. 4 for K=3 draft + 1 bonus token), replayable at ANY per-slot
    kv_len up to this runtime's per-slot capacity (``blocks_per_slot *
    block_size``) -- not just whatever dummy shape was used at capture
    time.

    2026-07-16, CUDA Graph round: this project's own read of
    ``vllm/v1/attention/backends/sm120_gqa.py``'s documented history (a
    real illegal-memory-access crash, root-caused to metadata tensors
    without fixed addresses) plus its OTHER documented lesson
    (``kv_split_size``/``max_num_splits`` frozen at capture time going
    stale under a later, larger real kv_len) directly motivate this
    class's two central design points:

    1. Every tensor a captured kernel launch reads (metadata CSR tensors,
       input_ids, positions, slot_mapping) is a PERSISTENT, fixed-address
       buffer, allocated once in ``__init__``. ``replay()`` writes freshly
       computed REAL values into these SAME buffers via ``.copy_()`` --
       it never reallocates them. This is what makes replaying at a
       kv_len the buffers were never filled with at capture time safe.
    2. ``kv_split_size``/``max_num_splits`` are derived ONCE from this
       runtime's configured per-slot page-table limit (``blocks_per_slot *
       block_size`` -- a software ceiling the caller chose when
       constructing ``DirectModelRunner``, NOT a GPU hardware limit), via
       ``build_attention_metadata_batch``'s ``fixed_kv_split_size``/
       ``fixed_max_num_splits`` parameters -- see that function's
       docstring for the correctness proof that this bounds every real
       kv_len up to that configured limit, not just the capture-time
       value.

    2026-07-16, MTP extension (qo_len>1), **superseded 2026-07-18 (Phase 2
    CUDA-graph reconciliation)**: originally this replicated GDN's
    chunked/"prefill" metadata (``build_gdn_metadata_batch``'s
    ``chunk_indices``/``chunk_offsets``/``nums_dict``/``batch_ptr``/
    ``token_chunk_offset_ptr``/``has_initial_state`` fields), matching
    what the eager verify path did at the time. The eager path has since
    moved to the REAL spec-decode GDN mechanism
    (``build_gdn_metadata_spec_batch`` -- K+1 dedicated SSM rows,
    acceptance-aware addressing, no snapshot/restore/recompute-forward;
    see ``mtp_verify_and_commit_batch``'s docstring) for the exact same
    reason this class exists at all: qo_len>1 here ONLY ever means MTP
    verify. This class now mirrors that mechanism instead:
    ``spec_query_start_loc``/``spec_sequence_masks`` are CONSTANT for a
    fixed (batch_size, qo_len) pair (computed once, same reasoning as
    ``qo_indptr`` above); ``spec_state_indices_tensor`` (a FIXED
    per-(slot, column) row mapping by construction -- see
    ``_ssm_spec_row`` -- but slot IDENTITY varies per replay, since one
    graph object is reused across calls with different active slot sets
    at the same batch_size) and ``num_accepted_tokens`` (the real
    per-round accept/reject outcome, inherently different every replay)
    are both refilled in ``_fill_buffers`` every replay, exactly like
    ``kv_page_indices``/``state_indices`` already were.

    A replayed CUDA graph is a pre-recorded sequence of GPU kernel
    launches, NOT a re-execution of Python control flow -- ``model
    .forward()`` (the Python function) is only ever actually called
    during ``capture()`` (plus its warmup iterations), never during
    ``replay()``. This means whatever kernel-dispatch branch
    ``SM120GQAImpl.forward()`` takes (decode-kernel vs general, FP8 vs
    NVFP4, MMA vs v2 vs scalar, ...) must be identical for every real
    kv_len this graph will ever replay at -- true here because dispatch
    depends only on ``qo_len``/kv-cache dtype/model config, all fixed for
    a given (batch_size, qo_len) graph, never on the live kv_len itself.

    2026-07-17, state-neutral capture (correctness fix, found via an
    independent review this project's coordinator commissioned and
    personally verified): ``capture()``'s warmup runs 3 REAL executions
    on a side stream before the graph trace (the trace itself, inside
    ``with torch.cuda.graph(g):``, executes nothing -- confirmed against
    the sibling project's own kernel-level CUDA-graph test). Attention's
    paged KV cache tolerates redundant warmup writes fine (same position,
    same value, overwritten harmlessly) -- but GDN's recurrent/chunked
    state update reads-old-state-and-writes-new-state each call, so it is
    NOT idempotent under repeated identical input; running warmup against
    slots a caller will later actually replay against silently advances
    those slots' real GDN state by 3 extra (unaccounted, un-bookkept)
    applications before any real replay happens. This was a genuine gap
    in this project's own initial qo_len=1/MTP test scripts (both reused
    real/twin-established slots for warmup) -- their empirical PASS
    results are not proof this doesn't matter, only evidence it didn't
    surface for that specific signal-probe task (plausibly because
    full-attention layers dominate identity recall, masking a GDN
    perturbation a GDN-sensitive task might not tolerate).

    Fix: this class now reserves ``batch_size`` of the runner's logical
    slots PERMANENTLY for its own exclusive, disposable warmup use (the
    LAST ``batch_size`` slots of ``runner.num_slots`` -- see
    ``self._warmup_slots``) -- ``capture()`` takes no external slot/token/
    kv_length arguments at all anymore, so this can no longer depend on
    caller discipline the way the original design implicitly did. Callers
    must size ``runner.num_slots >= 2 * batch_size`` and never pass this
    graph's reserved warmup slots to ``replay()`` or any other runner
    method.

    Also fixed the same round: ``replay()`` no longer calls
    ``torch.cuda.synchronize()`` (see that method's docstring for why this
    is safe and why the removed blanket device-wide sync worked against
    the whole point of using a captured graph to cut CPU-side dispatch
    overhead), and ``_fill_buffers`` now computes per-replay values via
    plain Python arithmetic instead of round-tripping through
    ``build_attention_metadata_batch``/``build_gdn_metadata_batch``/
    ``DirectModelRunner._slot_mapping_batch`` (which each construct several
    of their own intermediate GPU tensors -- real, avoidable per-replay
    allocation overhead on what should be a lean hot path). This is a
    partial mitigation (each static buffer's ``.copy_()`` source is still
    a freshly constructed small tensor, not a persistent pinned staging
    buffer written in place) -- a fully allocation-free version is a
    further optimization, not attempted this round.
    """

    def __init__(self, runner: "DirectModelRunner", batch_size: int, qo_len: int = 1,
                 warmup_slots: list[int] | None = None) -> None:
        if warmup_slots is not None:
            if len(warmup_slots) != batch_size:
                raise ValueError(
                    f"warmup_slots must have exactly batch_size ({batch_size}) "
                    f"entries, got {len(warmup_slots)}"
                )
            self._external_warmup = True
        else:
            if runner.num_slots < 2 * batch_size:
                raise ValueError(
                    f"runner.num_slots={runner.num_slots} must be >= 2*batch_size "
                    f"({2 * batch_size}) when warmup_slots is not provided"
                )
            warmup_slots = list(range(runner.num_slots - batch_size, runner.num_slots))
            self._external_warmup = False
        self.runner = runner
        self.batch_size = batch_size
        self.qo_len = qo_len
        device = runner.device
        block_size = runner.block_size
        blocks_per_slot = runner.blocks_per_slot
        # 2026-07-18, Phase 2 CUDA-graph reconciliation: this class used to
        # derive its own split-KV config from a stale local
        # ``TARGET_SPLITS = 16`` constant (a leftover from an earlier
        # round, predating the later real-production tuning to 64
        # splits/request -- see __init__'s split-KV comment on
        # ``DirectModelRunner``). That staleness was harmless as long as
        # this graph's verify-step replay was never actually exercised in
        # production (true through this round's reconciliation); wiring it
        # into the real ``mtp_verify_and_commit_batch`` path makes it
        # matter for real -- a genuinely DIFFERENT split-KV count changes
        # the attention kernel's reduction order, which this project has
        # already established can flip near-tie accept/reject decisions
        # (found via a real W1-S run: a 70.3%->76.7% draft-acceptance-rate
        # shift between the eager and graph-replayed paths, too large to
        # be ordinary near-tie noise, before this fix). Now uses the SAME
        # runner-computed, currently-tuned value every eager caller and
        # ``CapturedMTPDraftStepGraph`` already use, instead of maintaining
        # a second, independently-stale derivation.
        self.fixed_kv_split_size = runner.decode_fixed_kv_split_size
        self.fixed_max_num_splits = runner.decode_fixed_max_num_splits

        self._warmup_slots = warmup_slots
        self._last_num_pages_per_req: list[int] | None = None
        self._last_state_slot_ids: list[int] | None = None

        num_reqs = batch_size
        n_tokens = num_reqs * qo_len

        # Attention metadata static buffers -- worst-case sized (a request
        # could in principle use this slot's entire page capacity).
        # qo_indptr is CONSTANT for a fixed (batch_size, qo_len) pair
        # ([0, qo_len, 2*qo_len, ..., num_reqs*qo_len]) -- computed once,
        # never refilled.
        self.static_qo_indptr = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
        self.static_kv_page_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        self.static_kv_page_indices = torch.zeros(num_reqs * blocks_per_slot, dtype=torch.int32, device=device)
        self.static_kv_last_page_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # GDN metadata static buffers. non_spec_query_start_loc is
        # likewise constant; state_indices is per-replay-filled (depends
        # on slot_ids, not just batch_size/qo_len).
        self.static_state_indices = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        self.static_non_spec_qsl = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len

        # Model I/O static buffers.
        self.static_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_positions = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_slot_mapping = torch.zeros(n_tokens, dtype=torch.long, device=device)

        # 2026-07-18, Phase 2 CUDA-graph reconciliation: for qo_len>1 this
        # class's ONLY real remaining use is MTP verify, which the eager
        # path (``verify_batch_spec``) now does via the REAL spec-decode
        # GDN mechanism, not the old chunked/"prefill" one this class used
        # to replicate here (``_const_gdn_extra``, removed). Static buffers
        # mirror ``build_gdn_metadata_spec_batch``'s fields:
        # ``spec_query_start_loc``/``spec_sequence_masks`` are CONSTANT for
        # a fixed (batch_size, qo_len) pair (computed once, like
        # ``static_qo_indptr`` above); ``spec_state_indices_tensor`` (a
        # FIXED per-(slot, column) row mapping -- see ``_ssm_spec_row``'s
        # docstring -- but slot IDENTITY varies per replay, since this
        # class is reused across calls with potentially different active
        # slot sets at the same batch_size) and ``num_accepted_tokens``
        # (the real per-round accept/reject outcome, inherently different
        # every replay) are both per-replay-filled in ``_fill_buffers``.
        self.num_spec = runner.num_speculative_tokens
        self.total_physical_slots = runner.num_slots + RESERVED_PHYSICAL_SLOTS
        self.static_spec_query_start_loc: torch.Tensor | None = None
        self.static_spec_sequence_masks: torch.Tensor | None = None
        self.static_spec_state_indices: torch.Tensor | None = None
        self.static_num_accepted_tokens: torch.Tensor | None = None
        if qo_len > 1:
            if self.num_spec is None:
                raise RuntimeError(
                    "CapturedBatchDecodeGraph with qo_len>1 requires MTP "
                    "(runner.num_speculative_tokens) to be configured -- "
                    "this shape only ever means spec-decode verify"
                )
            self.static_spec_query_start_loc = (
                torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
            )
            self.static_spec_sequence_masks = torch.ones(num_reqs, dtype=torch.bool, device=device)
            self.static_spec_state_indices = torch.zeros((num_reqs, qo_len), dtype=torch.int32, device=device)
            self.static_num_accepted_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: torch.Tensor | None = None
        # 2026-07-17, Phase 3: captured alongside logits (see
        # ``_forward_no_sync``/``capture()``) so ``replay(..., return_hidden=True)``
        # can hand back the target model's own hidden states -- needed by
        # ``mtp_verify_and_commit_batch`` to feed the MTP draft model's next
        # resync step without an extra eager forward. ``None`` until
        # ``capture()`` has run.
        self._static_hidden_states: torch.Tensor | None = None
        # Test-observability only (2026-07-17, consolidation pass): counts
        # real replay() calls so a correctness test can directly confirm a
        # graph was actually EXERCISED, not merely constructed/precaptured
        # (precapture populates the cache regardless of whether a given
        # round ever replays that specific shape) -- see
        # benchmarks/mtp_verify_cudagraph_check.py.
        self.replay_count = 0

        cpu = torch.device("cpu")
        self._cpu_kv_page_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=cpu, pin_memory=True)
        self._cpu_kv_page_indices = torch.zeros(num_reqs * blocks_per_slot, dtype=torch.int32, device=cpu, pin_memory=True)
        self._cpu_kv_last_page_len = torch.zeros(num_reqs, dtype=torch.int32, device=cpu, pin_memory=True)
        self._cpu_state_indices = torch.zeros(num_reqs, dtype=torch.int32, device=cpu, pin_memory=True)
        self._cpu_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_positions = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_slot_mapping = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._np_kv_page_indptr = self._cpu_kv_page_indptr.numpy()
        self._np_kv_page_indices = self._cpu_kv_page_indices.numpy()
        self._np_kv_last_page_len = self._cpu_kv_last_page_len.numpy()
        self._np_state_indices = self._cpu_state_indices.numpy()
        self._np_input_ids = self._cpu_input_ids.numpy()
        self._np_positions = self._cpu_positions.numpy()
        self._np_slot_mapping = self._cpu_slot_mapping.numpy()
        if qo_len > 1:
            self._cpu_spec_state_indices = torch.zeros((num_reqs, qo_len), dtype=torch.int32, device=cpu, pin_memory=True)
            self._cpu_num_accepted_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=cpu, pin_memory=True)

    def _fill_buffers(
        self,
        slot_ids: list[int],
        token_ids,
        kv_lengths: list[int],
        *,
        num_accepted_tokens_prev: list[int] | None = None,
    ) -> None:
        """Write real, per-replay-varying values into the persistent static
        buffers. Computes everything via plain Python arithmetic (CPU-only,
        no GPU allocation) instead of calling
        ``build_attention_metadata_batch``/``build_gdn_metadata_batch``/
        ``DirectModelRunner._slot_mapping_batch`` -- those each construct
        several of their own intermediate GPU tensors (dataclass fields the
        caller doesn't need here), real avoidable overhead on a hot path
        meant to be lean. Each static buffer's ``.copy_()`` source below is
        still a freshly built small tensor (a partial mitigation, not a
        fully allocation-free design -- see the class docstring).

        ``num_accepted_tokens_prev`` (2026-07-18, Phase 2 CUDA-graph
        reconciliation, required when ``self.qo_len > 1``): each slot's
        real committed length from ITS OWN last verify round (or bootstrap
        1 right after a real prefill) -- see ``build_gdn_metadata_spec_batch``'s
        docstring. Used to fill ``static_num_accepted_tokens``;
        ``static_spec_state_indices`` is filled from ``slot_ids`` alone
        (``_ssm_spec_row`` is a fixed per-(slot, column) mapping, but slot
        IDENTITY genuinely varies per replay -- this class is reused
        across calls with potentially different active slot sets at the
        same batch_size)."""
        runner = self.runner
        device = runner.device
        qo_len = self.qo_len
        block_size = runner.block_size
        blocks_per_slot = runner.blocks_per_slot

        if qo_len == 1:
            flat_token_ids = token_ids
        else:
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        new_kv_lens = [kv_len + qo_len for kv_len in kv_lengths]
        num_pages_per_req = [(kv_len + block_size - 1) // block_size for kv_len in new_kv_lens]
        for slot, kv_len, num_pages in zip(slot_ids, new_kv_lens, num_pages_per_req):
            if num_pages > blocks_per_slot:
                raise RuntimeError(
                    f"slot {slot} kv_len {kv_len} exceeds this slot's "
                    f"{blocks_per_slot * block_size}-token capacity"
                )

        # P1 (notes/prefix-cache-design.md sec 5): grow every replayed
        # slot's block_table to cover this replay's own new_kv_len BEFORE
        # reading it below (INV5 -- the captured launch itself never bakes
        # in a physical block id, only this Python-side refill does, every
        # replay).
        if runner.enable_block_table:
            for slot, kv_len in zip(slot_ids, new_kv_lens):
                runner._ensure_blocks(slot, kv_len)

        pages_unchanged = (self._last_num_pages_per_req is not None
                           and num_pages_per_req == self._last_num_pages_per_req
                           and slot_ids == getattr(self, '_last_slot_ids', None))
        if not pages_unchanged:
            kv_page_indptr_list = [0]
            for num_pages in num_pages_per_req:
                kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)

            page_indices_list: list[int] = []
            for slot, num_pages in zip(slot_ids, num_pages_per_req):
                if runner.enable_block_table:
                    page_indices_list.extend(runner.block_table[slot][:num_pages])
                else:
                    first_block = _physical_slot(slot) * blocks_per_slot
                    page_indices_list.extend(range(first_block, first_block + num_pages))

            n_indptr = len(kv_page_indptr_list)
            self._np_kv_page_indptr[:n_indptr] = kv_page_indptr_list
            self.static_kv_page_indptr.copy_(self._cpu_kv_page_indptr, non_blocking=True)
            self.static_kv_page_indices.zero_()
            n_pages = len(page_indices_list)
            if n_pages:
                self._np_kv_page_indices[:n_pages] = page_indices_list
                self.static_kv_page_indices[:n_pages].copy_(self._cpu_kv_page_indices[:n_pages], non_blocking=True)
            self._last_num_pages_per_req = list(num_pages_per_req)
            self._last_slot_ids = list(slot_ids)

        last_page_len_list = [
            kv_len - (num_pages - 1) * block_size
            for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ]
        positions_list = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]

        slot_mapping_list: list[int] = []
        for slot, kv_len in zip(slot_ids, kv_lengths):
            table = runner.block_table[slot] if runner.enable_block_table else None
            first_block = _physical_slot(slot) * blocks_per_slot
            for j in range(qo_len):
                pos = kv_len + j
                block_id = table[pos // block_size] if table is not None else first_block + pos // block_size
                offset = pos % block_size
                slot_mapping_list.append(block_id * block_size + offset)
        n_reqs = len(last_page_len_list)
        self._np_kv_last_page_len[:n_reqs] = last_page_len_list
        self.static_kv_last_page_len.copy_(self._cpu_kv_last_page_len, non_blocking=True)
        slots_changed = (slot_ids != self._last_state_slot_ids)
        if slots_changed:
            state_indices_list = [_physical_slot(slot) for slot in slot_ids]
            self._np_state_indices[:n_reqs] = state_indices_list
            self.static_state_indices.copy_(self._cpu_state_indices, non_blocking=True)
            self._last_state_slot_ids = list(slot_ids)
        n_tok = len(flat_token_ids)
        self._np_input_ids[:n_tok] = flat_token_ids
        self.static_input_ids.copy_(self._cpu_input_ids, non_blocking=True)
        self._np_positions[:n_tok] = positions_list
        self.static_positions.copy_(self._cpu_positions, non_blocking=True)
        self._np_slot_mapping[:n_tok] = slot_mapping_list
        self.static_slot_mapping.copy_(self._cpu_slot_mapping, non_blocking=True)

        if qo_len > 1:
            if num_accepted_tokens_prev is None:
                raise ValueError(
                    "num_accepted_tokens_prev is required when qo_len > 1 "
                    "(this shape only ever means spec-decode verify)"
                )
            if slots_changed or not hasattr(self, '_spec_cached'):
                spec_indices_list = [
                    [_ssm_spec_row(slot, col, self.total_physical_slots, self.num_spec) for col in range(qo_len)]
                    for slot in slot_ids
                ]
                self._cpu_spec_state_indices[:n_reqs] = torch.tensor(spec_indices_list, dtype=torch.int32)
                self.static_spec_state_indices.copy_(self._cpu_spec_state_indices, non_blocking=True)
                self._spec_cached = True
            self._cpu_num_accepted_tokens[:n_reqs] = torch.tensor(num_accepted_tokens_prev, dtype=torch.int32)
            self.static_num_accepted_tokens.copy_(self._cpu_num_accepted_tokens, non_blocking=True)

    def _static_metadata_dicts(self) -> tuple[dict, dict]:
        runner = self.runner
        n_tokens = self.batch_size * self.qo_len
        attn_meta = SM120GQAMetadata(
            num_actual_tokens=n_tokens,
            num_reqs=self.batch_size,
            qo_indptr=self.static_qo_indptr,
            kv_page_indptr=self.static_kv_page_indptr,
            kv_page_indices=self.static_kv_page_indices,
            kv_last_page_len=self.static_kv_last_page_len,
            page_size=runner.block_size,
            is_pure_decode=(self.qo_len == 1),
            kv_split_size=self.fixed_kv_split_size,
            max_num_splits=self.fixed_max_num_splits,
            decode_qo_len=self.qo_len,
        )
        if self.qo_len == 1:
            gdn_meta = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=self.batch_size,
                num_decode_tokens=self.batch_size,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=self.batch_size,
                non_spec_query_start_loc=self.static_non_spec_qsl,
                non_spec_state_indices_tensor=self.static_state_indices,
            )
        else:
            # 2026-07-18, Phase 2 CUDA-graph reconciliation: this class's
            # only real remaining qo_len>1 use is MTP verify, which the
            # eager path now does via the REAL spec-decode GDN mechanism
            # (build_gdn_metadata_spec_batch), not the old chunked/
            # "prefill" one this branch used to replicate (removed along
            # with self._const_gdn_extra). See __init__'s comment for the
            # static-buffer rationale.
            gdn_meta = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=self.batch_size,
                num_spec_decode_tokens=n_tokens,
                num_actual_tokens=n_tokens,
                spec_query_start_loc=self.static_spec_query_start_loc,
                spec_state_indices_tensor=self.static_spec_state_indices,
                spec_sequence_masks=self.static_spec_sequence_masks,
                num_accepted_tokens=self.static_num_accepted_tokens,
            )
        attn_metadata_dict = {name: attn_meta for name in runner.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in runner.gdn_layer_names})
        slot_mapping_dict = {name: self.static_slot_mapping for name in runner.attn_layer_names}
        return attn_metadata_dict, slot_mapping_dict

    def _forward_no_sync(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Same op sequence as ``DirectModelRunner._forward_batch``, minus
        the ``torch.cuda.synchronize()`` calls -- calling those DURING
        capture is a documented CUDA-graph-capture violation (raises
        ``cudaErrorStreamCaptureUnsupported``), the same error class the
        sibling project already hit and documented for a different op (a
        boolean-mask-select) during its own CUDA Graph work.

        Returns ``(logits, hidden_states)`` (2026-07-17, Phase 3 addition --
        previously only logits were returned/captured; ``hidden_states`` is
        now ALSO captured so ``replay(..., return_hidden=True)`` can hand it
        back, matching ``_forward_batch``'s own ``return_hidden`` contract).
        This is a backward-compatible extension: the only caller inside
        ``capture()``'s warmup loop discards both return values either way."""
        runner = self.runner
        attn_metadata_dict, slot_mapping_dict = self._static_metadata_dicts()
        with set_forward_context(attn_metadata_dict, runner.vllm_config, slot_mapping=slot_mapping_dict):
            hidden_states = runner.model.forward(self.static_input_ids, self.static_positions)
        logits = runner.model.compute_logits(hidden_states)
        return logits, hidden_states

    def capture(self) -> None:
        """Warm up (uncaptured, on a side stream -- required by
        ``torch.cuda.graph`` before capture) then capture the graph, using
        this object's OWN permanently reserved, disposable warmup slots
        (``self._warmup_slots``) -- NEVER any slot a caller will later pass
        to ``replay()``. This is what makes capture state-neutral for real
        traffic: the only slots capture()'s 3 real warmup executions (plus
        the graph-trace call, which itself executes nothing -- see the
        class docstring) can touch are these reserved slots.

        Warmup content is disposable and never checked for correctness --
        any valid token id works, so a fixed dummy prompt is used (``[0,
        0, 0, 0, 0]``, matching ``DirectModelRunner._warmup``'s own
        convention). This is also why fixed-sizing kv_split_size/
        max_num_splits is required in the first place: the real kv_len
        distribution ``replay()`` sees is expected to differ, often
        drastically, from this disposable warmup shape."""
        if self._graph is not None:
            raise RuntimeError("already captured")
        runner = self.runner
        warmup_slots = self._warmup_slots
        for slot in warmup_slots:
            if runner.slot_kv_len[slot] != 0:
                raise RuntimeError(
                    f"reserved warmup slot {slot} is not fresh -- capture() "
                    "must run before anything else touches this graph's "
                    "own warmup slots, and exactly once per graph object"
                )
            runner.prefill(slot, [0, 0, 0, 0, 0])
        warmup_kv_lengths = [runner.slot_kv_len[s] for s in warmup_slots]
        if self.qo_len == 1:
            warmup_token_ids = [0] * self.batch_size
            warmup_num_accepted_tokens_prev = None
        else:
            warmup_token_ids = [[0] * self.qo_len for _ in range(self.batch_size)]
            # Bootstrap value (column 0 -- the row runner.prefill() above
            # just wrote into), matching real usage's own first-ever-verify
            # convention (see build_gdn_metadata_spec_batch's docstring).
            warmup_num_accepted_tokens_prev = [1] * self.batch_size
        self._fill_buffers(
            warmup_slots, warmup_token_ids, warmup_kv_lengths,
            num_accepted_tokens_prev=warmup_num_accepted_tokens_prev,
        )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward_no_sync()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._static_logits, self._static_hidden_states = self._forward_no_sync()
        self._graph = g

        if self._external_warmup:
            for slot in self._warmup_slots:
                runner.reset_slot(slot)

    def replay(
        self,
        slot_ids: list[int],
        token_ids,
        kv_lengths: list[int],
        *,
        commit: bool = True,
        return_hidden: bool = False,
        num_accepted_tokens_prev: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Replay the captured graph at REAL (slot_ids, token_ids,
        kv_lengths) data -- may (and, per this round's explicit test
        scope, deliberately does) differ drastically from capture()'s
        warmup data, including kv_len values much larger or smaller than
        whatever was used at capture time. Returns logits shaped
        ``[batch_size * qo_len, vocab]`` (request-then-position order).

        ``return_hidden`` (2026-07-17, Phase 3 addition, default ``False``
        preserving every existing caller's behavior byte-for-byte): when
        ``True``, returns ``(logits, hidden_states)`` instead of just
        ``logits`` -- mirrors ``_forward_batch``'s own ``return_hidden``
        parameter. Needed by ``mtp_verify_and_commit_batch`` so a
        graph-captured verify replay can still feed the MTP draft model's
        next resync step (which needs the target model's hidden states,
        not just its logits) without an extra eager forward call.

        ``num_accepted_tokens_prev`` (2026-07-18, Phase 2 CUDA-graph
        reconciliation, required when ``self.qo_len > 1``): each slot's
        real committed length from ITS OWN last verify round (bootstrap 1
        right after a real prefill) -- forwarded to ``_fill_buffers`` to
        select which of last round's K+1 dedicated SSM rows holds the
        valid state to resume from. Mirrors ``verify_batch_spec``'s own
        parameter of the same name on the eager path.

        ``commit`` (2026-07-17 fix, Codex-sol review, confirmed real):
        mirrors ``_forward_batch``'s own ``commit`` parameter -- this
        method used to advance ``self.runner.slot_kv_len`` by
        ``self.qo_len`` UNCONDITIONALLY, the exact physical-write-vs-
        committed conflation already fixed on the eager path (see
        ``_forward_batch``'s docstring) but left inconsistent here, since
        this class has its own separate captured-graph call path that
        never goes through ``_forward_batch``. For ``qo_len==1`` (plain
        decode) this default is harmless (never ambiguous). For
        ``qo_len>1`` (MTP verify), the caller (``mtp_verify_and_commit_batch``,
        2026-07-18) passes ``commit=False`` and applies the real
        committed-length correction itself (plus updates
        ``slot_num_accepted_tokens`` for the NEXT round) after determining
        accept/reject on the returned logits -- exactly like the eager
        ``verify_batch_spec`` path.

        No ``torch.cuda.synchronize()`` here (removed 2026-07-17, a
        correctness-review finding): ``_fill_buffers``'s ``.copy_()`` calls
        and ``self._graph.replay()`` are all issued on the SAME (default)
        CUDA stream, so CUDA's own stream-ordering already guarantees the
        graph's kernels observe the freshly-copied buffer contents, and
        that the NEXT call's ``_fill_buffers`` won't overwrite data this
        replay is still reading -- no explicit device-wide sync is needed
        for that. The caller gets an implicit, narrowly-scoped sync for
        free the moment it actually reads back a value (e.g.
        ``.argmax(dim=-1).item()`` on the returned logits). A blanket
        ``torch.cuda.synchronize()`` here would additionally block on any
        OTHER unrelated work queued on the device -- directly working
        against the whole point of using a captured graph to cut CPU-side
        launch/dispatch overhead."""
        if not self._external_warmup and (slot_ids == self._warmup_slots or set(slot_ids) & set(self._warmup_slots)):
            raise RuntimeError(
                f"slot(s) {set(slot_ids) & set(self._warmup_slots)} are this "
                "graph's own reserved warmup slots -- never replay() against "
                "them, they exist solely for capture()'s internal use"
            )
        if self._graph is None:
            raise RuntimeError("capture() must be called first")
        if not (len(slot_ids) == self.batch_size == len(token_ids) == len(kv_lengths)):
            raise ValueError("slot_ids/token_ids/kv_lengths must match batch_size")
        for slot, kv_len in zip(slot_ids, kv_lengths):
            if kv_len != self.runner.slot_kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot}: caller-provided kv_length {kv_len} != "
                    f"tracked {self.runner.slot_kv_len[slot]}"
                )
            if not self.runner.slot_gdn_initialized[slot]:
                raise RuntimeError(f"slot {slot} has no GDN state yet (needs a prior prefill)")
        self._fill_buffers(slot_ids, token_ids, kv_lengths, num_accepted_tokens_prev=num_accepted_tokens_prev)
        self._graph.replay()
        self.replay_count += 1
        for slot in slot_ids:
            if commit:
                self.runner.slot_kv_len[slot] += self.qo_len
            self.runner.slot_gdn_initialized[slot] = True
        if return_hidden:
            return self._static_logits, self._static_hidden_states
        return self._static_logits


class CapturedMTPDraftStepGraph:
    """CUDA-graph-captured decode/resync step for the MTP DRAFT model
    (``runner.mtp_model``), for a FIXED ``batch_size`` and FIXED ``qo_len``
    (1 = the autoregressive continuation steps 1..k-1; >1 = step 0's
    resync, when its ``num_new_tokens`` happens to be UNIFORM across the
    batch -- always true for the full-accept group, which is always
    exactly k+1, and sometimes true for the recompute group, when every
    recompute slot's committed_len happens to coincide).
    2026-07-17, Phase 3 round 2 (coordinator-directed fast-iteration pass,
    picking off ``notes/2026-07-17-post-ragged-round-next-steps.md``'s
    Phase 3 candidate 1: "then the K draft steps") -- generalized from an
    initial qo_len=1-only version (same day, same round) once it became
    clear step 0 is ALSO uniform-shaped often enough to be worth capturing,
    exactly mirroring how ``mtp_verify_and_commit_batch``'s recompute
    forward reuses ``CapturedBatchDecodeGraph`` for its own uniform special
    case (see that method's comment) -- this class is the draft-model
    analogue of that same idea, generalized in place rather than as a
    parallel copy.

    Narrower than ``CapturedBatchDecodeGraph`` in two ways, both because
    the draft model's own call shape is simpler than the target model's:
    1. No GDN metadata at all -- ``runner.mtp_attn_layer_names`` never
       includes a GDN layer (the draft model registers only its own small
       full-attention layer(s); see ``DirectModelRunner.__init__``).
    2. Takes an EXTRA static input, ``static_hidden_states_in`` (shape
       ``[batch_size*qo_len, hidden_size]``), since ``Qwen3_5MultiTokenPredictor
       .forward()`` needs the running hidden-state carry between steps
       that the target model's own ``forward()`` does not.

    Always dispatches via ``decode_qo_len=qo_len``/``is_pure_decode=(qo_len==1)``
    (the fast decode/verify kernel path) regardless of what the EAGER path
    would have used for the same call -- for step 0 specifically, the eager
    ``_mtp_forward_batch`` passes ``is_decode=all(n==1 for n in
    num_new_tokens_list)``, which is ``False`` whenever ``qo_len>1``,
    routing eager step-0 calls through the general/chunked attention
    kernel instead. This is a DELIBERATE dispatch difference, not a bug:
    the underlying math (causal attention over a query range against a KV
    cache) is identical either way -- this project's own
    ``build_attention_metadata_batch`` docstring already establishes that
    the decode-kernel and general-kernel paths are numerically
    equivalent, just different kernel choices -- and using the fast
    decode/verify kernel here is a legitimate additional (small) win on
    top of the graph-capture win itself. Verified via
    ``benchmarks/mtp_verify_cudagraph_check.py``'s real content comparison
    against an independent eager reference, not assumed.

    Stateless w.r.t. runner bookkeeping: unlike ``CapturedBatchDecodeGraph
    .replay()``, this class's ``replay()`` does NOT touch
    ``runner.slot_kv_len``/``slot_gdn_initialized`` at all -- mirroring
    ``_mtp_forward_batch`` itself, which also never does (the caller
    tracks its own running length counters as plain local Python
    variables, not persistent per-slot runner state). ``prior_kv_len ==
    start_pos`` always holds for every real call site in this codebase
    (confirmed by inspection: ``_mtp_sync_and_propose_batch`` always
    derives ``start_pos_list`` from the exact same ``slot_draft_sync_len``
    snapshot its own internal ``prior_kv_lens_step0`` uses, and the
    qo_len=1 continuation loop advances both counters by the same amount
    every iteration), so this class only needs ONE per-slot length list,
    not two.

    Same warmup-slot-reservation and state-neutral-capture discipline as
    ``CapturedBatchDecodeGraph`` (see that class's docstring for the full
    rationale): the last ``batch_size`` logical slots of ``runner.num_slots``
    are reserved, used only by this graph's own ``capture()``, and reset
    immediately afterward. Warmup content is fully disposable (a dummy
    all-zeros hidden-state tensor and dummy token ids) -- the draft
    model's own attention KV is content/position-addressed like the
    target model's, so redundant warmup writes are harmless, and there is
    no GDN-style non-idempotent recurrent state in this class's scope at
    all (point 1 above), so the correctness concern that motivated
    ``CapturedBatchDecodeGraph``'s state-neutral-capture fix does not even
    apply here -- reserved slots are used anyway, for consistency with the
    established pattern and so a caller can safely reuse the SAME
    physical reserved-slot range across both graph classes (each resets
    its own warmup slots right after its own ``capture()``)."""

    def __init__(self, runner: "DirectModelRunner", batch_size: int, qo_len: int = 1,
                 warmup_slots: list[int] | None = None) -> None:
        if runner.mtp_model is None:
            raise RuntimeError("no MTP draft model loaded")
        if warmup_slots is not None:
            if len(warmup_slots) != batch_size:
                raise ValueError(f"warmup_slots must have exactly batch_size ({batch_size}) entries")
            self._external_warmup = True
        else:
            if runner.num_slots < 2 * batch_size:
                raise ValueError(
                    f"runner.num_slots={runner.num_slots} must be >= 2*batch_size ({2 * batch_size})"
                )
            warmup_slots = list(range(runner.num_slots - batch_size, runner.num_slots))
            self._external_warmup = False
        self.runner = runner
        self.batch_size = batch_size
        self.qo_len = qo_len
        device = runner.device
        blocks_per_slot = runner.blocks_per_slot
        self.fixed_kv_split_size = runner.decode_fixed_kv_split_size
        self.fixed_max_num_splits = runner.decode_fixed_max_num_splits

        self._warmup_slots = warmup_slots

        n_tokens = batch_size * qo_len
        self.static_qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
        self.static_kv_page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.static_kv_page_indices = torch.zeros(batch_size * blocks_per_slot, dtype=torch.int32, device=device)
        self.static_kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.static_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_positions = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_slot_mapping = torch.zeros(n_tokens, dtype=torch.long, device=device)
        cpu = torch.device("cpu")
        self._cpu_kv_page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=cpu, pin_memory=True)
        self._cpu_kv_page_indices = torch.zeros(batch_size * blocks_per_slot, dtype=torch.int32, device=cpu, pin_memory=True)
        self._cpu_kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=cpu, pin_memory=True)
        self._cpu_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_positions = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_slot_mapping = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._np_kv_page_indptr = self._cpu_kv_page_indptr.numpy()
        self._np_kv_page_indices = self._cpu_kv_page_indices.numpy()
        self._np_kv_last_page_len = self._cpu_kv_last_page_len.numpy()
        self._np_input_ids = self._cpu_input_ids.numpy()
        self._np_positions = self._cpu_positions.numpy()
        self._np_slot_mapping = self._cpu_slot_mapping.numpy()
        self.hidden_size = runner.vllm_config.model_config.get_hidden_size()
        # Dtype/device matched lazily on first real hidden_states_in at
        # capture time (mirrors how the class discovers its own model's
        # activation dtype rather than assuming one).
        self.static_hidden_states_in: torch.Tensor | None = None

        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: torch.Tensor | None = None
        self._static_hidden_states_out: torch.Tensor | None = None
        # Test-observability only -- see CapturedBatchDecodeGraph's
        # identical field for the rationale.
        self.replay_count = 0
        self._last_slot_ids: list[int] | None = None
        self._last_kv_lengths: list[int] | None = None


    def _fill_buffers(self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]) -> None:
        runner = self.runner
        device = runner.device
        block_size = runner.block_size
        blocks_per_slot = runner.blocks_per_slot
        qo_len = self.qo_len

        if qo_len == 1:
            flat_token_ids = token_ids
        else:
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        new_kv_lens = [kv_len + qo_len for kv_len in kv_lengths]
        num_pages_per_req = [(kv_len + block_size - 1) // block_size for kv_len in new_kv_lens]
        for slot, kv_len, num_pages in zip(slot_ids, new_kv_lens, num_pages_per_req):
            if num_pages > blocks_per_slot:
                raise RuntimeError(f"slot {slot} kv_len {kv_len} exceeds this slot's {blocks_per_slot * block_size}-token capacity")

        # P1 (notes/prefix-cache-design.md sec 5): same reasoning as
        # CapturedBatchDecodeGraph._fill_buffers -- grow before reading.
        if runner.enable_block_table:
            for slot, kv_len in zip(slot_ids, new_kv_lens):
                runner._ensure_blocks(slot, kv_len)

        kv_page_indptr_list = [0]
        for num_pages in num_pages_per_req:
            kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)
        page_indices_list: list[int] = []
        for slot, num_pages in zip(slot_ids, num_pages_per_req):
            if runner.enable_block_table:
                page_indices_list.extend(runner.block_table[slot][:num_pages])
            else:
                first_block = _physical_slot(slot) * blocks_per_slot
                page_indices_list.extend(range(first_block, first_block + num_pages))
        last_page_len_list = [
            kv_len - (num_pages - 1) * block_size for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ]
        positions_list = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]
        slot_mapping_list = []
        for slot, kv_len in zip(slot_ids, kv_lengths):
            table = runner.block_table[slot] if runner.enable_block_table else None
            first_block = _physical_slot(slot) * blocks_per_slot
            for j in range(qo_len):
                pos = kv_len + j
                block_id = table[pos // block_size] if table is not None else first_block + pos // block_size
                offset = pos % block_size
                slot_mapping_list.append(block_id * block_size + offset)

        n_indptr = len(kv_page_indptr_list)
        self._np_kv_page_indptr[:n_indptr] = kv_page_indptr_list
        self.static_kv_page_indptr.copy_(self._cpu_kv_page_indptr, non_blocking=True)
        self.static_kv_page_indices.zero_()
        n_pages = len(page_indices_list)
        if n_pages:
            self._np_kv_page_indices[:n_pages] = page_indices_list
            self.static_kv_page_indices[:n_pages].copy_(self._cpu_kv_page_indices[:n_pages], non_blocking=True)
        self._np_kv_last_page_len[:len(last_page_len_list)] = last_page_len_list
        self.static_kv_last_page_len.copy_(self._cpu_kv_last_page_len, non_blocking=True)
        self._np_input_ids[:len(flat_token_ids)] = flat_token_ids
        self.static_input_ids.copy_(self._cpu_input_ids, non_blocking=True)
        self._np_positions[:len(positions_list)] = positions_list
        self.static_positions.copy_(self._cpu_positions, non_blocking=True)
        self._np_slot_mapping[:len(slot_mapping_list)] = slot_mapping_list
        self.static_slot_mapping.copy_(self._cpu_slot_mapping, non_blocking=True)
        if self.static_hidden_states_in is None:
            self.static_hidden_states_in = torch.zeros_like(hidden_states_in)
        self.static_hidden_states_in.copy_(hidden_states_in)

    def _static_metadata_dict(self) -> dict:
        runner = self.runner
        attn_meta = SM120GQAMetadata(
            num_actual_tokens=self.batch_size * self.qo_len,
            num_reqs=self.batch_size,
            qo_indptr=self.static_qo_indptr,
            kv_page_indptr=self.static_kv_page_indptr,
            kv_page_indices=self.static_kv_page_indices,
            kv_last_page_len=self.static_kv_last_page_len,
            page_size=runner.block_size,
            is_pure_decode=(self.qo_len == 1),
            kv_split_size=self.fixed_kv_split_size,
            max_num_splits=self.fixed_max_num_splits,
            decode_qo_len=self.qo_len,
        )
        return {name: attn_meta for name in runner.mtp_attn_layer_names}

    def _forward_no_sync(self) -> tuple[torch.Tensor, torch.Tensor]:
        runner = self.runner
        attn_metadata_dict = self._static_metadata_dict()
        slot_mapping_dict = {name: self.static_slot_mapping for name in runner.mtp_attn_layer_names}
        with set_forward_context(attn_metadata_dict, runner.vllm_config, slot_mapping=slot_mapping_dict):
            hidden_states_out = runner.mtp_model.forward(
                self.static_input_ids, self.static_positions, self.static_hidden_states_in
            )
        logits = runner.mtp_model.compute_logits(hidden_states_out)
        return logits, hidden_states_out

    def capture(self) -> None:
        if self._graph is not None:
            raise RuntimeError("already captured")
        runner = self.runner
        warmup_slots = self._warmup_slots
        dummy_prompt = [0, 0, 0, 0, 0]
        dummy_hidden = torch.zeros(
            len(warmup_slots) * len(dummy_prompt),
            self.hidden_size,
            dtype=runner.vllm_config.model_config.dtype,
            device=runner.device,
        )
        # Establishes the DRAFT model's own attention KV history on these
        # slots directly (bypassing the target model entirely -- warmup
        # content is disposable, see class docstring) so a subsequent
        # qo_len-token "step" at position len(dummy_prompt) is a genuine,
        # valid continuation rather than an out-of-range access.
        runner._mtp_forward_batch(
            warmup_slots,
            [dummy_prompt] * len(warmup_slots),
            dummy_hidden,
            [0] * len(warmup_slots),
            [0] * len(warmup_slots),
            qo_len=len(dummy_prompt),
            is_decode=False,
        )
        warmup_kv_lengths = [len(dummy_prompt)] * len(warmup_slots)
        if self.qo_len == 1:
            warmup_token_ids = [0] * self.batch_size
        else:
            warmup_token_ids = [[0] * self.qo_len for _ in range(self.batch_size)]
        warmup_hidden = torch.zeros(
            self.batch_size * self.qo_len, self.hidden_size, dtype=dummy_hidden.dtype, device=runner.device
        )
        self._fill_buffers(warmup_slots, warmup_token_ids, warmup_hidden, warmup_kv_lengths)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward_no_sync()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._static_logits, self._static_hidden_states_out = self._forward_no_sync()
        self._graph = g

        if self._external_warmup:
            for slot in self._warmup_slots:
                runner.reset_slot(slot)

    def replay(
        self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay at REAL data. Returns ``(logits, hidden_states_out)`` --
        unlike ``CapturedBatchDecodeGraph.replay()``, ``return_hidden`` is
        not optional here since every real caller needs the hidden state
        to feed the NEXT step (or, for a step-0-only recompute-group
        replay, to hand back to the caller). No runner bookkeeping is
        touched (see class docstring) -- purely a stateless tensor-in/
        tensor-out call, same as the eager ``_mtp_forward_batch`` it
        replaces. ``token_ids`` follows ``_mtp_forward_batch``'s own
        convention: flat (one id per slot) when ``qo_len==1``, else a
        list of per-slot ``qo_len``-length lists."""
        if not self._external_warmup and set(slot_ids) & set(self._warmup_slots):
            raise RuntimeError(
                f"slot(s) {set(slot_ids) & set(self._warmup_slots)} are this "
                "graph's own reserved warmup slots -- never replay() against them"
            )
        if self._graph is None:
            raise RuntimeError("capture() must be called first")
        if not (len(slot_ids) == self.batch_size == len(token_ids) == len(kv_lengths)):
            raise ValueError("slot_ids/token_ids/kv_lengths must match batch_size")
        self._fill_buffers(slot_ids, token_ids, hidden_states_in, kv_lengths)
        self._graph.replay()
        self.replay_count += 1
        self._last_slot_ids = slot_ids
        self._last_kv_lengths = kv_lengths
        return self._static_logits, self._static_hidden_states_out

    def replay_incremental(
        self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized replay for draft continuation steps where KV length
        increased by 1 from the previous replay. Skips rebuilding the
        expensive kv_page_indices array when no slot crossed a page boundary."""
        if self._graph is None:
            raise RuntimeError("capture() must be called first")
        runner = self.runner
        block_size = runner.block_size
        can_skip_pages = (
            self._last_slot_ids is not None
            and self._last_slot_ids == slot_ids
            and self._last_kv_lengths is not None
            and len(self._last_kv_lengths) == len(kv_lengths)
            and all(
                new - old == self.qo_len
                and (old + self.qo_len) // block_size == (new + self.qo_len) // block_size
                for old, new in zip(self._last_kv_lengths, kv_lengths)
            )
        )
        if can_skip_pages:
            self._fill_buffers_incremental(slot_ids, token_ids, hidden_states_in, kv_lengths)
        else:
            self._fill_buffers(slot_ids, token_ids, hidden_states_in, kv_lengths)
        self._graph.replay()
        self.replay_count += 1
        self._last_slot_ids = slot_ids
        self._last_kv_lengths = kv_lengths
        return self._static_logits, self._static_hidden_states_out

    def _fill_buffers_incremental(self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]) -> None:
        """Fast path: only update last_page_len, positions, slot_mapping,
        input_ids, hidden_states. Skip kv_page_indptr and kv_page_indices."""
        runner = self.runner
        block_size = runner.block_size
        qo_len = self.qo_len
        new_kv_lens = [kv_len + qo_len for kv_len in kv_lengths]
        num_pages_per_req = [(kv_len + block_size - 1) // block_size for kv_len in new_kv_lens]
        if runner.enable_block_table:
            for slot, kv_len in zip(slot_ids, new_kv_lens):
                runner._ensure_blocks(slot, kv_len)
        last_page_len_list = [
            kv_len - (num_pages - 1) * block_size for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ]
        positions_list = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]
        slot_mapping_list = []
        for slot, kv_len in zip(slot_ids, kv_lengths):
            table = runner.block_table[slot] if runner.enable_block_table else None
            first_block = _physical_slot(slot) * runner.blocks_per_slot
            for j in range(qo_len):
                pos = kv_len + j
                block_id = table[pos // block_size] if table is not None else first_block + pos // block_size
                slot_mapping_list.append(block_id * block_size + pos % block_size)
        if qo_len == 1:
            flat_token_ids = token_ids
        else:
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]
        self._np_kv_last_page_len[:len(last_page_len_list)] = last_page_len_list
        self.static_kv_last_page_len.copy_(self._cpu_kv_last_page_len, non_blocking=True)
        self._np_input_ids[:len(flat_token_ids)] = flat_token_ids
        self.static_input_ids.copy_(self._cpu_input_ids, non_blocking=True)
        self._np_positions[:len(positions_list)] = positions_list
        self.static_positions.copy_(self._cpu_positions, non_blocking=True)
        self._np_slot_mapping[:len(slot_mapping_list)] = slot_mapping_list
        self.static_slot_mapping.copy_(self._cpu_slot_mapping, non_blocking=True)
        if self.static_hidden_states_in is None:
            self.static_hidden_states_in = torch.zeros_like(hidden_states_in)
        self.static_hidden_states_in.copy_(hidden_states_in)
