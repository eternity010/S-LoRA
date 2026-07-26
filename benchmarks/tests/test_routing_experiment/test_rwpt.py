"""Tests for the profiled rank-weighted prompt-token model."""

import pytest

from slora.server.router.rwpt import rank_weighted_prompt_tokens


@pytest.mark.parametrize(
    ("prompt_len", "rank", "expected"),
    [
        (1000, 0, 1000),
        (1000, 16, 1135),
        (1000, 64, 1540),
    ],
)
def test_rank_weighted_prompt_tokens(prompt_len, rank, expected):
    assert rank_weighted_prompt_tokens(prompt_len, rank, 0.00845) == expected


@pytest.mark.parametrize(
    ("prompt_len", "rank", "beta"),
    [(-1, 16, 0.00845), (100, -1, 0.00845), (100, 16, -0.1)],
)
def test_rank_weighted_prompt_tokens_rejects_negative_inputs(
    prompt_len, rank, beta
):
    with pytest.raises(ValueError):
        rank_weighted_prompt_tokens(prompt_len, rank, beta)
