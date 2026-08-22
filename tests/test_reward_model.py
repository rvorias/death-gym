from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dmfast.reward import (
    ACTION_DIM,
    INFO_FIELDS,
    INFO_INDEX,
    OBS_BAG_SLICE,
    OBS_DIM,
    OBS_EQUIPMENT_SLICE,
    OBS_SIM_SLICE,
    Slot,
    StateDiffRewardModel,
)


def _flat_obs(*, phase: int = 0, sim_stats: list[float] | None = None) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = float(phase)
    if sim_stats is not None:
        obs[OBS_SIM_SLICE] = np.asarray(sim_stats, dtype=np.float32)
    return obs


def _set_equipment_item(obs: np.ndarray, slot_index: int, *, item_id: int, tier_norm: float, slot_value: int, greatness_norm: float) -> None:
    equipment = obs[OBS_EQUIPMENT_SLICE].reshape(8, 9)
    equipment[slot_index, 0] = float(item_id)
    equipment[slot_index, 1] = float(tier_norm)
    equipment[slot_index, 3] = float(slot_value)
    equipment[slot_index, 8] = float(greatness_norm)


def _set_bag_item(obs: np.ndarray, bag_index: int, *, item_id: int, tier_norm: float, slot_value: int, greatness_norm: float) -> None:
    bag = obs[OBS_BAG_SLICE].reshape(15, 9)
    bag[bag_index, 0] = float(item_id)
    bag[bag_index, 1] = float(tier_norm)
    bag[bag_index, 3] = float(slot_value)
    bag[bag_index, 8] = float(greatness_norm)


def _set_item_xp(obs: np.ndarray, *, equipment_index: int | None = None, bag_index: int | None = None, xp: float) -> None:
    if equipment_index is not None:
        obs[OBS_EQUIPMENT_SLICE].reshape(8, 9)[equipment_index, 7] = float(xp) / 400.0
    if bag_index is not None:
        obs[OBS_BAG_SLICE].reshape(15, 9)[bag_index, 7] = float(xp) / 400.0


def _info(
    *,
    level: float = 1.0,
    xp: float = 0.0,
    health: float = 100.0,
    gold: float = 0.0,
    beast_level: float = 0.0,
    beasts_killed: float = 0.0,
    items_bought: float = 0.0,
    potions_bought: float = 0.0,
    started_from_snapshot: float = 0.0,
) -> np.ndarray:
    row = np.zeros(len(INFO_FIELDS), dtype=np.float32)
    row[INFO_INDEX["adventurer_level"]] = float(level)
    row[INFO_INDEX["xp"]] = float(xp)
    row[INFO_INDEX["health"]] = float(health)
    row[INFO_INDEX["gold"]] = float(gold)
    row[INFO_INDEX["beast_level"]] = float(beast_level)
    row[INFO_INDEX["beasts_killed"]] = float(beasts_killed)
    row[INFO_INDEX["items_bought"]] = float(items_bought)
    row[INFO_INDEX["potions_bought"]] = float(potions_bought)
    row[INFO_INDEX["episode_started_from_curriculum_snapshot"]] = float(
        started_from_snapshot
    )
    return row


def _mask_with(action: int) -> np.ndarray:
    mask = np.zeros((1, ACTION_DIM), dtype=np.uint8)
    mask[0, int(action)] = 1
    return mask


def test_invalid_action_penalty_uses_pre_mask() -> None:
    model = StateDiffRewardModel({"invalid_action_penalty": -0.5})

    reward = model.compute_batch(
        prev_obs=_flat_obs().reshape(1, -1),
        prev_info=_info().reshape(1, -1),
        prev_mask=np.zeros((1, ACTION_DIM), dtype=np.uint8),
        actions=np.asarray([1], dtype=np.int32),
        post_obs=_flat_obs().reshape(1, -1),
        post_info=_info().reshape(1, -1),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )

    assert reward.shape == (1,)
    assert float(reward[0]) == -0.5


def test_xp_kill_and_death_rewards_use_info_deltas() -> None:
    model = StateDiffRewardModel({"xp_shaping_scale": 1024.0, "kill": 2.0, "death_penalty": -5.0})

    reward = model.compute_batch(
        prev_obs=_flat_obs(phase=4).reshape(1, -1),
        prev_info=_info(level=10, xp=100, beast_level=12, beasts_killed=0).reshape(1, -1),
        prev_mask=_mask_with(1),
        actions=np.asarray([1], dtype=np.int32),
        post_obs=_flat_obs(phase=1).reshape(1, -1),
        post_info=_info(level=10, xp=164, beast_level=0, beasts_killed=1, health=0).reshape(1, -1),
        terminated=np.asarray([1], dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )

    expected = (64.0 * 1024.0 / 32767.0) + (2.0 * 12.0) - 5.0
    assert np.isclose(float(reward[0]), expected, atol=1e-5)


def test_item_xp_gain_reward_uses_positive_held_item_xp_delta() -> None:
    model = StateDiffRewardModel({"item_xp_gain": 0.5})

    prev_obs = _flat_obs()
    post_obs = _flat_obs()
    _set_equipment_item(
        prev_obs,
        0,
        item_id=10,
        tier_norm=0.8,
        slot_value=int(Slot.WEAPON),
        greatness_norm=0.0,
    )
    _set_equipment_item(
        post_obs,
        0,
        item_id=10,
        tier_norm=0.8,
        slot_value=int(Slot.WEAPON),
        greatness_norm=0.0,
    )
    _set_item_xp(prev_obs, equipment_index=0, xp=40.0)
    _set_item_xp(post_obs, equipment_index=0, xp=100.0)

    reward = model.compute_batch(
        prev_obs=prev_obs.reshape(1, -1),
        prev_info=_info().reshape(1, -1),
        prev_mask=_mask_with(0),
        actions=np.asarray([0], dtype=np.int32),
        post_obs=post_obs.reshape(1, -1),
        post_info=_info().reshape(1, -1),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )

    assert np.isclose(float(reward[0]), 30.0, atol=1e-6)


def test_buy_rewards_come_from_inventory_deltas() -> None:
    model = StateDiffRewardModel(
        {
            "buy_item": 0.5,
            "buy_t1_item": 1.1,
            "buy_weapon_t1": 3.5,
            "buy_quality_delta": 0.06,
            "buy_asset_bank_delta": 0.045,
        }
    )

    prev_obs = _flat_obs(phase=5)
    post_obs = _flat_obs(phase=1)
    _set_bag_item(
        post_obs,
        0,
        item_id=42,
        tier_norm=1.0,
        slot_value=int(Slot.WEAPON),
        greatness_norm=0.0,
    )

    reward = model.compute_batch(
        prev_obs=prev_obs.reshape(1, -1),
        prev_info=_info(level=10, items_bought=0).reshape(1, -1),
        prev_mask=_mask_with(11),
        actions=np.asarray([11], dtype=np.int32),
        post_obs=post_obs.reshape(1, -1),
        post_info=_info(level=10, items_bought=1).reshape(1, -1),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )

    economy_scale = 1.35
    quality_delta = 52.0
    expected = economy_scale * (0.5 + 1.1 + 3.5 + 0.06 * quality_delta + 0.045 * quality_delta)
    assert np.isclose(float(reward[0]), expected, atol=1e-5)


def test_switch_survivability_uses_clipped_sim_score_delta() -> None:
    model = StateDiffRewardModel(
        {
            "switch_survivability": 0.5,
            "switch_surv_score_clip": 1.0,
            "switch_surv_min_level": 8,
            "switch_surv_max_level": 34,
        }
    )

    prev_obs = _flat_obs(phase=3, sim_stats=[1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    post_obs = _flat_obs(phase=4, sim_stats=[3.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _set_equipment_item(
        prev_obs,
        0,
        item_id=10,
        tier_norm=0.8,
        slot_value=int(Slot.WEAPON),
        greatness_norm=0.0,
    )
    _set_equipment_item(
        post_obs,
        0,
        item_id=11,
        tier_norm=1.0,
        slot_value=int(Slot.WEAPON),
        greatness_norm=0.0,
    )

    reward = model.compute_batch(
        prev_obs=prev_obs.reshape(1, -1),
        prev_info=_info(level=10).reshape(1, -1),
        prev_mask=_mask_with(11),
        actions=np.asarray([11], dtype=np.int32),
        post_obs=post_obs.reshape(1, -1),
        post_info=_info(level=10).reshape(1, -1),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )

    assert np.isclose(float(reward[0]), 0.5, atol=1e-6)
