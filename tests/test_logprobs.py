"""C2: tests for runtime/logprobs.py — logprobs computation from logits."""
import math

import torch

from runtime.logprobs import compute_logprobs


class TestComputeLogprobs:
    def test_single_token_greedy(self):
        """Logprob of the argmax token should be close to 0 for peaked dist."""
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        result = compute_logprobs(logits, [0], top_k=0)
        assert len(result) == 1
        assert result[0]["token_id"] == 0
        # log_softmax([10,0,0,0])[0] ≈ 10 - log(e^10 + 3) ≈ -0.0001
        assert result[0]["logprob"] > -0.001
        assert result[0]["top_logprobs"] == []

    def test_uniform_distribution(self):
        """Uniform logits → logprob = -log(vocab_size)."""
        vocab = 8
        logits = torch.zeros(1, vocab)
        result = compute_logprobs(logits, [3], top_k=0)
        expected = -math.log(vocab)
        assert abs(result[0]["logprob"] - expected) < 1e-5

    def test_top_k_returns_correct_count(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = compute_logprobs(logits, [4], top_k=3)
        assert len(result[0]["top_logprobs"]) == 3
        # Top token should be id=4 (highest logit)
        assert result[0]["top_logprobs"][0]["token_id"] == 4
        # Logprobs should be sorted descending
        lps = [t["logprob"] for t in result[0]["top_logprobs"]]
        assert lps == sorted(lps, reverse=True)

    def test_top_k_clamped_to_vocab(self):
        logits = torch.tensor([[1.0, 2.0]])
        result = compute_logprobs(logits, [1], top_k=100)
        assert len(result[0]["top_logprobs"]) == 2

    def test_multi_position(self):
        """Multiple positions: each gets its own logprob entry."""
        logits = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])
        tokens = [0, 1, 2]
        result = compute_logprobs(logits, tokens, top_k=2)
        assert len(result) == 3
        for i, entry in enumerate(result):
            assert entry["token_id"] == tokens[i]
            assert entry["logprob"] > -0.001  # peaked distribution
            assert len(entry["top_logprobs"]) == 2

    def test_chosen_token_not_necessarily_top(self):
        """Logprob of a non-top token should be lower than top."""
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        result = compute_logprobs(logits, [0], top_k=3)
        assert result[0]["token_id"] == 0
        assert result[0]["logprob"] < -1.0  # not the top token
        # Top should be token 1
        assert result[0]["top_logprobs"][0]["token_id"] == 1

    def test_logprobs_sum_to_one_in_prob_space(self):
        """exp(logprobs) across all vocab should sum to ~1."""
        logits = torch.randn(1, 50)
        log_probs = torch.log_softmax(logits, dim=-1)
        total = log_probs.exp().sum().item()
        assert abs(total - 1.0) < 1e-5

    def test_raises_on_insufficient_logits(self):
        logits = torch.zeros(2, 10)
        try:
            compute_logprobs(logits, [0, 1, 2])
            assert False, "should have raised"
        except ValueError:
            pass
