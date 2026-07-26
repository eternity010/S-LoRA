"""Shared rank-calibrated workload helpers for RankFlow routing."""

DEFAULT_PROFILED_RANK_BETA = 0.00845


def rank_weighted_prompt_tokens(
    prompt_len: int,
    rank: int,
    beta: float = DEFAULT_PROFILED_RANK_BETA,
) -> int:
    """按离线 profiling 得到的 rank 成本对 prompt token 加权。"""
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}")
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    return int(prompt_len * (1.0 + beta * rank))
