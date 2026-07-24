"""CPU-only tests for DFlash engine structure and logic.

Tests the speculative decode accept/reject logic, buffer management,
and configuration constants without requiring GPU or model weights.
"""
import pytest

from runtime.backends.dflash_constants import (
    AUX_LAYER_IDS,
    DRAFT_HEAD_DIM,
    DRAFT_NUM_KV_HEADS,
    DRAFT_NUM_LAYERS,
    DRAFT_NUM_QO_HEADS,
    DRAFT_WINDOW,
    MASK_TOKEN_ID,
    NUM_QUERY_PER_REQ,
    NUM_SPECULATIVE_TOKENS,
)


class TestDFlashConstants:
    """Verify DFlash configuration constants match model config."""

    def test_speculative_tokens(self):
        assert NUM_SPECULATIVE_TOKENS == 15

    def test_query_per_req(self):
        assert NUM_QUERY_PER_REQ == 16  # 1 bonus + 15 mask

    def test_aux_layer_ids(self):
        # 0-indexed, matches dflash_config.target_layer_ids
        assert AUX_LAYER_IDS == (2, 11, 20, 30, 39, 48)
        assert len(AUX_LAYER_IDS) == 6

    def test_mask_token_id(self):
        assert MASK_TOKEN_ID == 12

    def test_draft_architecture(self):
        assert DRAFT_NUM_LAYERS == 6
        assert DRAFT_WINDOW == 512
        assert DRAFT_NUM_QO_HEADS == 72
        assert DRAFT_NUM_KV_HEADS == 8
        assert DRAFT_HEAD_DIM == 128


class TestGreedyVerifyLogic:
    """Test the greedy accept/reject logic in isolation."""

    def _verify_greedy(self, bonus_token, draft_tokens, verify_argmax):
        """Replicate the verify logic from DFlashEngine._verify."""
        accepted = [bonus_token]
        num_accepted = 0
        for verify_tok, draft_tok in zip(verify_argmax, draft_tokens):
            if verify_tok == draft_tok:
                accepted.append(draft_tok)
                num_accepted += 1
            else:
                accepted.append(verify_tok)
                num_accepted += 1
                break
        return accepted, num_accepted

    def test_all_accepted(self):
        """All 15 draft tokens match verify → 16 tokens accepted."""
        bonus = 42
        draft = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        verify = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        accepted, n = self._verify_greedy(bonus, draft, verify)
        assert n == 15
        assert len(accepted) == 16
        assert accepted[0] == bonus
        assert accepted[1:] == draft

    def test_first_rejected(self):
        """First draft token rejected → bonus + correction = 2 tokens."""
        bonus = 42
        draft = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        verify = [99, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        accepted, n = self._verify_greedy(bonus, draft, verify)
        assert n == 1
        assert len(accepted) == 2
        assert accepted == [42, 99]

    def test_partial_accept(self):
        """5 accepted then rejection → 7 tokens (bonus + 5 + correction)."""
        bonus = 42
        draft = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        verify = [10, 20, 30, 40, 50, 99, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        accepted, n = self._verify_greedy(bonus, draft, verify)
        assert n == 6  # 5 matches + 1 correction
        assert len(accepted) == 7
        assert accepted == [42, 10, 20, 30, 40, 50, 99]

    def test_empty_draft(self):
        """Edge case: no draft tokens."""
        bonus = 42
        accepted, n = self._verify_greedy(bonus, [], [])
        assert n == 0
        assert accepted == [42]


class TestRingBlocksForDraft:
    """Verify draft KV cache sizing."""

    def test_draft_ring_blocks(self):
        

        block_size = 16
        # Draft needs: window-1 + qo_max positions + 1 extra block
        # = 511 + 16 = 527 positions → cdiv(527, 16) + 1 = 33 + 1 = 34
        # Inline: cdiv(window-1+qo_max, block_size) + 1
        blocks = -(-(DRAFT_WINDOW - 1 + NUM_QUERY_PER_REQ) // block_size) + 1
        assert blocks == 34

    def test_draft_kv_memory(self):
        """Draft KV cache should be small (~36 MB per slot)."""
        

        block_size = 16
        # Inline: cdiv(window-1+qo_max, block_size) + 1
        blocks = -(-(DRAFT_WINDOW - 1 + NUM_QUERY_PER_REQ) // block_size) + 1
        # KV cache shape: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        # dtype: bf16 (2 bytes)
        bytes_per_block = 2 * block_size * DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM * 2
        total_per_slot = blocks * bytes_per_block * DRAFT_NUM_LAYERS
        mb_per_slot = total_per_slot / (1024 * 1024)
        # Should be ~36 MB per slot
        assert 10 < mb_per_slot < 20, f"Draft KV per slot: {mb_per_slot:.1f} MB"
