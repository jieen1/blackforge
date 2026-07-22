"""C2: logprobs computation from logits.

Computes per-token log-probabilities and top-k alternatives from raw
logits tensors. Used by the engine when a request sets ``logprobs=True``.

All computation stays on GPU; only the final small result tensors are
moved to CPU for the API layer to format.
"""

from __future__ import annotations

import torch


def compute_logprobs(
    logits: torch.Tensor,
    token_ids: list[int],
    top_k: int = 0,
) -> list[dict]:
    """Compute logprobs for a sequence of tokens given their logits.

    Args:
        logits: shape ``[seq_len, vocab_size]`` — raw logits for each
            position. ``logits[i]`` is the distribution from which
            ``token_ids[i]`` was sampled/chosen.
        token_ids: the chosen token at each position.
        top_k: number of top alternatives to include (0 = chosen only).

    Returns:
        List of dicts, one per position::

            {
                "token_id": int,
                "logprob": float,
                "top_logprobs": [{"token_id": int, "logprob": float}, ...],
            }
    """
    seq_len = len(token_ids)
    if logits.shape[0] < seq_len:
        raise ValueError(
            f"logits has {logits.shape[0]} positions but {seq_len} tokens requested"
        )
    log_probs = torch.log_softmax(logits[:seq_len].float(), dim=-1)

    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    chosen_logprobs = log_probs.gather(1, token_tensor.unsqueeze(1)).squeeze(1)

    results: list[dict] = []
    effective_top_k = min(top_k, logits.shape[-1]) if top_k > 0 else 0

    if effective_top_k > 0:
        top_vals, top_ids = log_probs.topk(effective_top_k, dim=-1)
        top_vals_cpu = top_vals.cpu().tolist()
        top_ids_cpu = top_ids.cpu().tolist()
    else:
        top_vals_cpu = None
        top_ids_cpu = None

    chosen_cpu = chosen_logprobs.cpu().tolist()
    for i in range(seq_len):
        entry: dict = {
            "token_id": token_ids[i],
            "logprob": chosen_cpu[i],
        }
        if top_vals_cpu is not None:
            entry["top_logprobs"] = [
                {"token_id": top_ids_cpu[i][j], "logprob": top_vals_cpu[i][j]}
                for j in range(effective_top_k)
            ]
        else:
            entry["top_logprobs"] = []
        results.append(entry)
    return results
