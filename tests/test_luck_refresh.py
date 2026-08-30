"""Luck must always equal the jewelry greatness it is derived from.

`special_stats[LUCK]` is a cached derivation of equipped/bagged jewelry
greatness, but only the inventory kernel recomputes it. Combat raises
equipment greatness on every hit and buying jewelry writes it straight into
the equipment array, so the cache used to go stale until the next
equip/drop -- silently changing crit rates. This pins the invariant.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rewards import GameEnv  # noqa: E402

ITEM_FIELDS = 9
EQUIP_OFF, N_EQUIP = 14, 8
BAG_OFF, N_BAG = 86, 15
LUCK_COL = 13
SILVER_RING_ID = 4
NECK_SLOT, RING_SLOT = 6, 7


def greatness(block):
    """(greatness, id, slot) for one packed item row; greatness 0 if empty."""
    ids = block[..., 0]
    g = np.rint(block[..., 8] * 20.0) + 1.0
    return np.where(ids == 0, 0.0, g), ids, block[..., 3]


def expected_luck(obs):
    eq = obs[:, EQUIP_OFF:EQUIP_OFF + N_EQUIP * ITEM_FIELDS]
    eq = eq.reshape(len(obs), N_EQUIP, ITEM_FIELDS)
    bag = obs[:, BAG_OFF:BAG_OFF + N_BAG * ITEM_FIELDS]
    bag = bag.reshape(len(obs), N_BAG, ITEM_FIELDS)

    eq_g, eq_ids, _ = greatness(eq)
    bag_g, _, bag_slot = greatness(bag)

    neck = eq_g[:, NECK_SLOT]
    ring = eq_g[:, RING_SLOT]
    jewelry = np.isin(bag_slot, (7, 8))
    bagged = (bag_g * jewelry).sum(axis=1)
    silver = (eq_ids[:, RING_SLOT] == SILVER_RING_ID) * ring
    return neck + ring + bagged + silver


@pytest.mark.parametrize("seed", [0, 11])
def test_luck_tracks_jewelry_greatness(seed):
    env = GameEnv(num_envs=64, seed=seed, max_steps=400)
    try:
        env.reset(seed)
        rng = np.random.default_rng(seed)
        for step in range(400):
            # env.obs is a live view; read it before stepping.
            np.testing.assert_allclose(
                np.rint(env.obs[:, LUCK_COL] * 100.0), expected_luck(env.obs),
                atol=1e-6, err_msg=f"stale luck at step {step}")
            env.step(env.sample_masked_actions(rng))
    finally:
        env.close()


def test_luck_actually_rises_during_play():
    """Guards against the invariant passing because luck is always zero."""
    env = GameEnv(num_envs=64, seed=3, max_steps=400)
    try:
        env.reset(3)
        rng = np.random.default_rng(3)
        best = 0.0
        for _ in range(400):
            env.step(env.sample_masked_actions(rng))
            best = max(best, float(env.obs[:, LUCK_COL].max()))
        assert best > 0.0, "no env ever carried jewelry; the invariant is vacuous"
    finally:
        env.close()
