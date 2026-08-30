"""A fixed model on fixed seeds must always produce the same eval.

A score is only meaningful if it depends on the policy and nothing else --
otherwise two runs are not comparable and neither is a before/after. These
pin the properties that make that true.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402
from rewards import GameEnv  # noqa: E402

WORLDS, BATCH, SEED = 128, 64, 7   # 2 chunks, so chunking is exercised


@pytest.fixture(scope="module")
def fixed_policy():
    torch.manual_seed(0)
    return train.LSTMPolicy(obs_dim=463, act_dim=57, embed_dim=128,
                            hidden_dim=128, num_trunk_blocks=1)


def score(policy, seed=SEED):
    env = GameEnv(num_envs=BATCH, seed=seed, max_steps=64)
    try:
        return train.eval_policy(env, policy, torch.device("cpu"),
                                 worlds=WORLDS, seed=seed)
    finally:
        env.close()


def assert_same(a, b):
    xps_a, trunc_a, stats_a = a
    xps_b, trunc_b, stats_b = b
    assert np.array_equal(xps_a, xps_b), "per-world XP differs"
    assert trunc_a == trunc_b
    for key in stats_a:
        assert np.array_equal(stats_a[key], stats_b[key]), key


def test_repeated_eval_is_identical(fixed_policy):
    assert_same(score(fixed_policy), score(fixed_policy))


def test_callers_rng_state_cannot_change_the_score(fixed_policy):
    """Eval seeds torch itself, so whatever the caller was doing is irrelevant."""
    first = score(fixed_policy)
    torch.manual_seed(999)
    torch.rand(1000)
    assert_same(first, score(fixed_policy))


def test_eval_restores_the_callers_rng_stream(fixed_policy):
    """...and eval must not perturb the caller's stream either."""
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)
    score(fixed_policy)
    assert torch.equal(expected, torch.rand(4))


def test_a_different_seed_gives_a_different_score(fixed_policy):
    """Guard against the pinning being so aggressive the seed stops mattering."""
    xps_a, _, _ = score(fixed_policy, seed=SEED)
    xps_b, _, _ = score(fixed_policy, seed=SEED + 1)
    assert not np.array_equal(xps_a, xps_b)


def test_chunk_lands_on_the_worlds_a_wide_env_would_produce():
    """The chunking identity the bank rests on: chunk c reset with
    seed + c*batch*STRIDE gives exactly rows c*batch.. of a wide reset."""
    wide = GameEnv(num_envs=WORLDS, seed=1, max_steps=64)
    narrow = GameEnv(num_envs=BATCH, seed=1, max_steps=64)
    try:
        wide.reset(seed=SEED)
        narrow.reset(seed=(SEED + BATCH * train.WORLD_SEED_STRIDE) % 2**64)
        assert np.array_equal(wide.obs[BATCH:WORLDS], narrow.obs[:BATCH])
        narrow.reset(seed=SEED)
        assert np.array_equal(wide.obs[:BATCH], narrow.obs[:BATCH])
    finally:
        wide.close()
        narrow.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_repeated_eval_is_identical_on_gpu(fixed_policy):
    """cuBLAS/cuDNN pick reduction orders per launch; the score must not move."""
    device = torch.device("cuda")
    policy = fixed_policy.to(device)
    try:
        def gpu_score():
            env = GameEnv(num_envs=BATCH, seed=SEED, max_steps=64)
            try:
                return train.eval_policy(env, policy, device, worlds=WORLDS, seed=SEED)
            finally:
                env.close()
        assert_same(gpu_score(), gpu_score())
    finally:
        fixed_policy.to("cpu")
