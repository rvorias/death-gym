"""episode_stats reads gear milestones and final stats out of the last obs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402


def make_obs(rows=1):
    return np.zeros((rows, 463), dtype=np.float32)


def put_item(obs, row, slot, item_id=7, type_code=2, greatness=1):
    """Write one equipment slot the way ex_pack_obs packs it."""
    base = 14 + slot * train.ITEM_FIELDS
    obs[row, base + train.I_ID] = item_id
    obs[row, base + train.I_TYPE] = type_code
    obs[row, base + train.I_GREATNESS] = (greatness - 1) / 20.0


def test_gear_milestones():
    obs = make_obs(3)
    put_item(obs, 0, slot=0, greatness=15)   # weapon at 15
    put_item(obs, 1, slot=0, greatness=20)   # weapon at 20
    # row 2 stays empty
    s = train.episode_stats(obs)
    assert list(s["gear_lvl15"]) == [True, True, False]
    assert list(s["gear_lvl20"]) == [False, True, False]


def test_full_armour_needs_every_slot_and_one_material():
    obs = make_obs(3)
    for slot in range(1, 6):                       # chest head waist foot hand
        put_item(obs, 0, slot, type_code=3)        # all metal
        put_item(obs, 1, slot, type_code=3)
        put_item(obs, 2, slot, type_code=3)
    obs[1, 14 + 3 * train.ITEM_FIELDS + train.I_TYPE] = 2   # one hide piece
    obs[2, 14 + 3 * train.ITEM_FIELDS + train.I_ID] = 0     # one empty slot
    s = train.episode_stats(obs)
    assert list(s["full_metal"]) == [True, False, False]
    assert not s["full_hide"].any() and not s["full_cloth"].any()


def test_empty_slots_do_not_count_as_greatness_one():
    """An unequipped slot packs as zeros, which decodes to greatness 1."""
    s = train.episode_stats(make_obs(1))
    for key in train.GEAR_KEYS:
        assert not s[key].any(), key


def test_final_stats_are_denormalised():
    obs = make_obs(1)
    obs[0, 7] = 10 / 31.0      # str
    obs[0, 12] = 31 / 31.0     # cha at cap
    obs[0, 13] = 40 / 100.0    # luck
    s = train.episode_stats(obs)
    assert np.isclose(s["str"][0], 10.0, atol=1e-3)
    assert np.isclose(s["cha"][0], 31.0, atol=1e-3)
    assert np.isclose(s["luck"][0], 40.0, atol=1e-3)


def test_merge_concatenates_every_key():
    parts = [train.episode_stats(make_obs(2)), train.episode_stats(make_obs(3))]
    merged = train.merge_stats(parts)
    assert set(merged) == set(parts[0])
    assert all(len(v) == 5 for v in merged.values())
