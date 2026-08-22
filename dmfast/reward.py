from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Slot(IntEnum):
    """Equipment slot ids, mirroring the on-chain contract's Slot enum."""

    NONE = 0
    WEAPON = 1
    CHEST = 2
    HEAD = 3
    WAIST = 4
    FOOT = 5
    HAND = 6
    NECK = 7
    RING = 8


OBS_DIM = 463
ACTION_DIM = 57
INFO_FIELDS = (
    "adventurer_level",
    "xp",
    "health",
    "gold",
    "beast_level",
    "beasts_killed",
    "items_bought",
    "potions_bought",
    "episode_started_from_curriculum_snapshot",
)
INFO_INDEX = {name: idx for idx, name in enumerate(INFO_FIELDS)}

EX_MAX_XP = 32767.0

OBS_PHASE_SLICE = slice(0, 1)
OBS_EQUIPMENT_SLICE = slice(14, 86)
OBS_BAG_SLICE = slice(86, 221)
OBS_SIM_SLICE = slice(453, 463)

NUM_EQUIPMENT_SLOTS = 8
NUM_BAG_SLOTS = 15
NUM_ITEM_FIELDS = 9

ITEM_ID_IDX = 0
ITEM_TIER_IDX = 1
ITEM_SLOT_IDX = 3
ITEM_XP_IDX = 7
ITEM_GREATNESS_IDX = 8

DEFAULT_PARAMS: dict[str, float] = {
    "attack": 0.0,
    "explore": 0.0,
    "flee_penalty": 0.0,
    "buy_item": 0.50,
    "buy_potion": 0.90,
    "buy_t1_item": 1.10,
    "buy_weapon_t1": 3.50,
    "buy_quality_delta": 0.06,
    "buy_asset_bank_delta": 0.045,
    "buy_asset_bank_delta_max_level": 22.0,
    "switch_item": 0.0,
    "switch_survivability": 0.50,
    "switch_surv_score_clip": 2.0,
    "switch_surv_min_level": 8.0,
    "switch_surv_max_level": 34.0,
    "drop_item": 0.0,
    "stat_upgrade": 0.0,
    "kill": 1.0,
    "level_up": 0.0,
    "xp_shaping_scale": 0.0,
    "death_penalty": 0.0,
    "attack_safe_sim": 0.15,
    "flee_safe_sim_penalty": -0.10,
    "flee_hard_sim": 0.10,
    "sim_risk_safe_threshold": 0.95,
    "sim_risk_hard_threshold": 1.35,
    "phase_very_early_end": 5.0,
    "phase_early_end": 15.0,
    "phase_mid_end": 34.0,
    "phase_late_end": 70.0,
    "phase_scale_economy": 1.0,
    "phase_scale_stat": 1.0,
    "phase_scale_attack": 1.0,
    "phase_scale_kill": 1.0,
    "phase_scale_high_kill": 1.0,
    "phase_scale_flee": 1.0,
    "phase_scale_potion": 1.0,
    "phase_scale_xp": 1.0,
    "item_xp_gain": 0.0,
    "invalid_action_penalty": -0.002,
    # Death-audit (2026-06-11): 55% of deaths were explore-at-low-HP. Penalty
    # scales with the HP deficit below the threshold (ratio of max health).
    "explore_low_hp": 0.0,
    "explore_low_hp_threshold": 0.5,
    # Death-audit v2: deaths at ~full hp; vitality under-invested (max_health
    # ~150 @ level 13). Extra bonus when the chosen stat upgrade is vitality.
    "stat_upgrade_vit": 0.0,
    # Gear-compounding: bonus per held item crossing greatness 15 (suffix
    # regime threshold; humans max armor g=20 in all quartiles, agent dies ~13)
    "suffix_unlock": 0.0,
    # Kill-gated item XP: prices only item xp earned on kill steps (the 2x
    # channel); the flat term was arbitraged via riskless obstacles.
    "item_xp_kill_gain": 0.0,
    # Gold-hoard penalty: penalize holding gold while early + healthy -> forces
    # gold->gear (potions mask-gated on missing HP). coef * gold_norm.
    "gold_hoard": 0.0,
    "gold_hoard_level": 5.0,
    "gold_hoard_hp": 0.75,
    # Armor-score objective (2-stage curriculum): reward delta of total armor
    # score = sum over equipped+bag armor of greatness*(6-tier).
    "armor_score": 0.0,
}

CONFIG_KEYS = (
    "attack",
    "explore",
    "flee_penalty",
    "buy_item",
    "buy_potion",
    "buy_t1_item",
    "buy_weapon_t1",
    "buy_quality_delta",
    "buy_asset_bank_delta",
    "buy_asset_bank_delta_max_level",
    "switch_item",
    "switch_survivability",
    "switch_surv_score_clip",
    "switch_surv_min_level",
    "switch_surv_max_level",
    "drop_item",
    "stat_upgrade",
    "kill",
    "level_up",
    "xp_shaping_scale",
    "death_penalty",
    "attack_safe_sim",
    "flee_safe_sim_penalty",
    "flee_hard_sim",
    "sim_risk_safe_threshold",
    "sim_risk_hard_threshold",
    "phase_very_early_end",
    "phase_early_end",
    "phase_mid_end",
    "phase_late_end",
    "phase_scale_economy",
    "phase_scale_stat",
    "phase_scale_attack",
    "phase_scale_kill",
    "phase_scale_high_kill",
    "phase_scale_flee",
    "phase_scale_potion",
    "phase_scale_xp",
    "item_xp_gain",
    "invalid_action_penalty",
    "explore_low_hp",
    "explore_low_hp_threshold",
    "stat_upgrade_vit",
    "suffix_unlock",
    "item_xp_kill_gain",
    "gold_hoard",
    "gold_hoard_level",
    "gold_hoard_hp",
    "armor_score",
)
CONFIG_INDEX = {name: idx for idx, name in enumerate(CONFIG_KEYS)}

PHASE_TABLES: dict[str, np.ndarray] = {
    "economy": np.asarray([1.7, 1.35, 1.0, 0.6, 0.35], dtype=np.float32),
    "stat": np.asarray([1.8, 1.4, 1.1, 0.8, 0.6], dtype=np.float32),
    "attack": np.asarray([0.75, 1.0, 1.2, 1.35, 1.5], dtype=np.float32),
    "kill": np.asarray([0.8, 1.0, 1.3, 1.65, 2.0], dtype=np.float32),
    "high_kill": np.asarray([0.5, 0.8, 1.2, 1.7, 2.2], dtype=np.float32),
    "flee": np.asarray([0.7, 1.0, 1.25, 1.5, 1.8], dtype=np.float32),
    "potion": np.asarray([0.8, 1.0, 1.2, 1.35, 1.5], dtype=np.float32),
    "xp": np.asarray([0.8, 1.0, 1.1, 1.25, 1.4], dtype=np.float32),
}

ACTION_EXPLORE = 0
ACTION_ATTACK = 1
ACTION_FLEE = 2


def info_row_from_dict(info: dict) -> np.ndarray:
    row = np.zeros(len(INFO_FIELDS), dtype=np.float32)
    for idx, name in enumerate(INFO_FIELDS):
        row[idx] = float(info.get(name, 0.0))
    return row


def _item_quality_score(item_row: np.ndarray) -> float:
    item_id = int(round(float(item_row[ITEM_ID_IDX])))
    if item_id <= 0:
        return 0.0

    slot = int(round(float(item_row[ITEM_SLOT_IDX])))
    slot_bonus = 0.0
    if slot == int(Slot.WEAPON):
        slot_bonus = 1.0
    elif slot in {
        int(Slot.CHEST),
        int(Slot.HEAD),
        int(Slot.WAIST),
        int(Slot.FOOT),
        int(Slot.HAND),
    }:
        slot_bonus = 0.6
    elif slot in {int(Slot.NECK), int(Slot.RING)}:
        slot_bonus = 0.3

    tier_score = float(item_row[ITEM_TIER_IDX]) * 5.0
    greatness = 1.0 + float(item_row[ITEM_GREATNESS_IDX]) * 20.0
    return 10.0 * tier_score + greatness + slot_bonus


def _held_items_from_obs(obs_row: np.ndarray) -> np.ndarray:
    equipment = np.asarray(obs_row[OBS_EQUIPMENT_SLICE], dtype=np.float32).reshape(
        NUM_EQUIPMENT_SLOTS, NUM_ITEM_FIELDS
    )
    bag = np.asarray(obs_row[OBS_BAG_SLICE], dtype=np.float32).reshape(
        NUM_BAG_SLOTS, NUM_ITEM_FIELDS
    )
    return np.concatenate((equipment, bag), axis=0)


def _suffix_count(obs_row: np.ndarray) -> int:
    count = 0
    for base in [14 + i * 9 for i in range(8)] + [86 + i * 9 for i in range(15)]:
        if obs_row[base] > 0.5 and obs_row[base + 8] >= 0.699:
            count += 1
    return count


def _held_item_xp_total(obs_row: np.ndarray) -> float:
    total = 0.0
    for item_row in _held_items_from_obs(obs_row):
        item_id = int(round(float(item_row[ITEM_ID_IDX])))
        if item_id <= 0:
            continue
        total += float(item_row[ITEM_XP_IDX]) * 400.0
    return total


def _best_slot_asset_score(obs_row: np.ndarray) -> float:
    best_by_slot: dict[int, float] = {}
    for item_row in _held_items_from_obs(obs_row):
        item_id = int(round(float(item_row[ITEM_ID_IDX])))
        if item_id <= 0:
            continue
        slot = int(round(float(item_row[ITEM_SLOT_IDX])))
        if slot == int(Slot.NONE):
            continue
        score = _item_quality_score(item_row)
        prev = best_by_slot.get(slot, 0.0)
        if score > prev:
            best_by_slot[slot] = score
    return float(sum(best_by_slot.values()))


def _find_new_item(pre_obs_row: np.ndarray, post_obs_row: np.ndarray) -> np.ndarray | None:
    pre_ids = {
        int(round(float(item_row[ITEM_ID_IDX])))
        for item_row in _held_items_from_obs(pre_obs_row)
        if int(round(float(item_row[ITEM_ID_IDX]))) > 0
    }
    for item_row in _held_items_from_obs(post_obs_row):
        item_id = int(round(float(item_row[ITEM_ID_IDX])))
        if item_id > 0 and item_id not in pre_ids:
            return item_row
    return None


def _equipped_item_for_slot(obs_row: np.ndarray, slot: int) -> np.ndarray | None:
    equipment = np.asarray(obs_row[OBS_EQUIPMENT_SLICE], dtype=np.float32).reshape(
        NUM_EQUIPMENT_SLOTS, NUM_ITEM_FIELDS
    )
    for item_row in equipment:
        if int(round(float(item_row[ITEM_SLOT_IDX]))) == int(slot):
            return item_row
    return None


def _combat_swap_score(obs_row: np.ndarray) -> float:
    sim = np.asarray(obs_row[OBS_SIM_SLICE], dtype=np.float32)
    return (
        1.6 * float(sim[0])
        - 1.0 * float(sim[2])
        + 0.35 * float(sim[3])
    )


@dataclass(slots=True)
class StateDiffRewardModel:
    params: dict[str, float]

    def __init__(self, reward_config: dict[str, float] | None = None):
        self.params = dict(DEFAULT_PARAMS)
        if reward_config:
            for name, value in reward_config.items():
                self.set_param(name, value)

    def set_param(self, name: str, value: float) -> None:
        self.params[str(name)] = float(value)

    def config_array(self) -> np.ndarray:
        cfg = np.zeros(len(CONFIG_KEYS), dtype=np.float32)
        for idx, key in enumerate(CONFIG_KEYS):
            cfg[idx] = float(self.params.get(key, DEFAULT_PARAMS.get(key, 0.0)))
        return cfg

    def _phase_index(self, levels: np.ndarray) -> np.ndarray:
        very_early_end = float(self.params.get("phase_very_early_end", DEFAULT_PARAMS["phase_very_early_end"]))
        early_end = float(self.params.get("phase_early_end", DEFAULT_PARAMS["phase_early_end"]))
        mid_end = float(self.params.get("phase_mid_end", DEFAULT_PARAMS["phase_mid_end"]))
        late_end = float(self.params.get("phase_late_end", DEFAULT_PARAMS["phase_late_end"]))
        levels = np.asarray(levels, dtype=np.float32)
        return np.where(
            levels <= very_early_end,
            0,
            np.where(
                levels <= early_end,
                1,
                np.where(levels <= mid_end, 2, np.where(levels <= late_end, 3, 4)),
            ),
        ).astype(np.int32)

    def _phase_scale(self, family: str, levels: np.ndarray) -> np.ndarray:
        idx = self._phase_index(levels)
        table = PHASE_TABLES[family]
        base = table[idx]
        global_scale = float(self.params.get(f"phase_scale_{family}", 1.0))
        return base * global_scale

    def compute_batch(
        self,
        *,
        prev_obs: np.ndarray,
        prev_info: np.ndarray,
        prev_mask: np.ndarray,
        actions: np.ndarray,
        post_obs: np.ndarray,
        post_info: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> np.ndarray:
        prev_obs = np.asarray(prev_obs, dtype=np.float32).reshape(-1, OBS_DIM)
        prev_info = np.asarray(prev_info, dtype=np.float32).reshape(-1, len(INFO_FIELDS))
        prev_mask = np.asarray(prev_mask, dtype=np.uint8).reshape(-1, ACTION_DIM)
        actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        post_obs = np.asarray(post_obs, dtype=np.float32).reshape(-1, OBS_DIM)
        post_info = np.asarray(post_info, dtype=np.float32).reshape(-1, len(INFO_FIELDS))
        terminated = np.asarray(terminated, dtype=np.uint8).reshape(-1)
        truncated = np.asarray(truncated, dtype=np.uint8).reshape(-1)

        count = actions.shape[0]
        rewards = np.zeros(count, dtype=np.float32)

        safe_actions = np.clip(actions, 0, ACTION_DIM - 1)
        valid = (actions >= 0) & (actions < ACTION_DIM)
        valid &= prev_mask[np.arange(count), safe_actions] != 0
        rewards += np.where(
            valid,
            0.0,
            float(self.params.get("invalid_action_penalty", 0.0)),
        ).astype(np.float32)

        prev_level = prev_info[:, INFO_INDEX["adventurer_level"]]
        post_level = post_info[:, INFO_INDEX["adventurer_level"]]
        prev_beast_level = prev_info[:, INFO_INDEX["beast_level"]]
        xp_delta = post_info[:, INFO_INDEX["xp"]] - prev_info[:, INFO_INDEX["xp"]]
        kill_delta = np.maximum(
            0.0,
            post_info[:, INFO_INDEX["beasts_killed"]] - prev_info[:, INFO_INDEX["beasts_killed"]],
        )
        item_delta = np.maximum(
            0.0,
            post_info[:, INFO_INDEX["items_bought"]] - prev_info[:, INFO_INDEX["items_bought"]],
        )
        potion_delta = np.maximum(
            0.0,
            post_info[:, INFO_INDEX["potions_bought"]] - prev_info[:, INFO_INDEX["potions_bought"]],
        )
        level_delta = np.maximum(0.0, post_level - prev_level)

        attack_scale = self._phase_scale("attack", prev_level)
        economy_scale = self._phase_scale("economy", post_level)
        flee_scale = self._phase_scale("flee", prev_level)
        potion_scale = self._phase_scale("potion", post_level)
        kill_scale = self._phase_scale("kill", prev_level)
        stat_scale = self._phase_scale("stat", prev_level)
        xp_scale = self._phase_scale("xp", post_level)

        rewards += (
            np.maximum(0.0, xp_delta)
            * float(self.params.get("xp_shaping_scale", 0.0))
            / EX_MAX_XP
            * xp_scale
        ).astype(np.float32)
        item_xp_coeff = float(self.params.get("item_xp_gain", 0.0))
        if item_xp_coeff != 0.0:
            item_xp_delta = np.asarray(
                [
                    max(0.0, _held_item_xp_total(post_obs[idx]) - _held_item_xp_total(prev_obs[idx]))
                    for idx in range(count)
                ],
                dtype=np.float32,
            )
            rewards += item_xp_delta * item_xp_coeff
        suffix_coeff = float(self.params.get("suffix_unlock", 0.0))
        if suffix_coeff != 0.0:
            sd = np.asarray(
                [max(0, _suffix_count(post_obs[idx]) - _suffix_count(prev_obs[idx]))
                 for idx in range(count)], dtype=np.float32)
            rewards += sd * suffix_coeff
        asc = float(self.params.get("armor_score", 0.0))
        if asc != 0.0:
            def armor(o):
                tot = np.zeros(len(o), dtype=np.float32)
                for k in range(8 + 15):
                    base = 14 + k*9 if k < 8 else 86 + (k-8)*9
                    has = o[:, base] >= 0.5
                    slot = np.round(o[:, base+3]).astype(int)
                    is_armor = (slot >= 2) & (slot <= 6) & has
                    g = o[:, base+8]*20 + 1; tw = o[:, base+1]*5
                    tot += (is_armor * g * tw).astype(np.float32)
                return tot
            rewards += (asc * (armor(post_obs) - armor(prev_obs))).astype(np.float32)
        gh = float(self.params.get("gold_hoard", 0.0))
        if gh != 0.0:
            lvl_t = float(self.params.get("gold_hoard_level", 5.0))
            hp_t = float(self.params.get("gold_hoard_hp", 0.75))
            post_level = post_info[:, INFO_INDEX["adventurer_level"]]
            mh = np.maximum(post_obs[:, 2], 1e-9)
            hpr = post_obs[:, 1] / mh
            active = (post_level < lvl_t) & (hpr > hp_t)
            rewards += (active * gh * post_obs[:, 4]).astype(np.float32)
        kxp_coeff = float(self.params.get("item_xp_kill_gain", 0.0))
        if kxp_coeff != 0.0:
            kills_idx = INFO_INDEX["beasts_killed"]
            kd = post_info[:, kills_idx] - prev_info[:, kills_idx]
            kxp = np.asarray(
                [max(0.0, _held_item_xp_total(post_obs[idx]) - _held_item_xp_total(prev_obs[idx]))
                 if kd[idx] > 0 else 0.0 for idx in range(count)], dtype=np.float32)
            rewards += kxp * kxp_coeff
        rewards += (
            level_delta * float(self.params.get("level_up", 0.0))
        ).astype(np.float32)
        rewards += (
            kill_delta
            * prev_beast_level
            * float(self.params.get("kill", 0.0))
            * kill_scale
        ).astype(np.float32)
        rewards += (
            item_delta
            * float(self.params.get("buy_item", 0.0))
            * economy_scale
        ).astype(np.float32)
        rewards += (
            potion_delta
            * float(self.params.get("buy_potion", 0.0))
            * potion_scale
        ).astype(np.float32)

        prev_phase = np.rint(prev_obs[:, OBS_PHASE_SLICE.start]).astype(np.int32)
        prev_sim_risk = prev_obs[:, OBS_SIM_SLICE.start + 2]
        safe_threshold = float(
            self.params.get("sim_risk_safe_threshold", DEFAULT_PARAMS["sim_risk_safe_threshold"])
        )
        hard_threshold = float(
            self.params.get("sim_risk_hard_threshold", DEFAULT_PARAMS["sim_risk_hard_threshold"])
        )

        rewards += (
            (valid & (actions == ACTION_EXPLORE))
            * float(self.params.get("explore", 0.0))
            * economy_scale
        ).astype(np.float32)
        explore_low_hp = float(self.params.get("explore_low_hp", 0.0))
        if explore_low_hp != 0.0:
            thr = float(self.params.get("explore_low_hp_threshold", 0.5))
            if thr > 0.0:
                max_hp = np.maximum(prev_obs[:, 2], 1e-9)
                hp_ratio = prev_obs[:, 1] / max_hp
                deficit = np.maximum(0.0, thr - hp_ratio) / thr
                rewards += (
                    (valid & (actions == ACTION_EXPLORE))
                    * explore_low_hp * deficit
                ).astype(np.float32)
        rewards += (
            (valid & (actions == ACTION_ATTACK))
            * float(self.params.get("attack", 0.0))
            * attack_scale
        ).astype(np.float32)
        rewards += (
            (valid & (actions == ACTION_ATTACK) & (prev_sim_risk > 0.0) & (prev_sim_risk <= safe_threshold))
            * float(self.params.get("attack_safe_sim", 0.0))
            * attack_scale
        ).astype(np.float32)
        rewards += (
            (valid & (actions == ACTION_FLEE))
            * float(self.params.get("flee_penalty", 0.0))
            * flee_scale
        ).astype(np.float32)
        rewards += (
            (valid & (actions == ACTION_FLEE) & (prev_sim_risk > 0.0) & (prev_sim_risk <= safe_threshold))
            * float(self.params.get("flee_safe_sim_penalty", 0.0))
            * flee_scale
        ).astype(np.float32)
        rewards += (
            (valid & (actions == ACTION_FLEE) & (prev_sim_risk >= hard_threshold))
            * float(self.params.get("flee_hard_sim", 0.0))
            * flee_scale
        ).astype(np.float32)
        rewards += (
            (valid & (prev_phase == 0))
            * float(self.params.get("stat_upgrade", 0.0))
            * stat_scale
        ).astype(np.float32)
        rewards += (
            (valid & (prev_phase == 0) & (actions == 53))  # stat_vit
            * float(self.params.get("stat_upgrade_vit", 0.0))
            * stat_scale
        ).astype(np.float32)
        rewards += (
            (valid & (prev_phase == 2))
            * float(self.params.get("drop_item", 0.0))
            * economy_scale
        ).astype(np.float32)
        rewards += (
            (valid & (prev_phase == 3))
            * float(self.params.get("switch_item", 0.0))
            * economy_scale
        ).astype(np.float32)
        rewards += (
            (terminated != 0) * float(self.params.get("death_penalty", 0.0))
        ).astype(np.float32)

        asset_bank_coeff = float(self.params.get("buy_asset_bank_delta", 0.0))
        asset_bank_max_level = float(
            self.params.get("buy_asset_bank_delta_max_level", DEFAULT_PARAMS["buy_asset_bank_delta_max_level"])
        )
        quality_coeff = float(self.params.get("buy_quality_delta", 0.0))
        buy_t1_coeff = float(self.params.get("buy_t1_item", 0.0))
        buy_weapon_t1_coeff = float(self.params.get("buy_weapon_t1", 0.0))
        switch_coeff = float(self.params.get("switch_survivability", 0.0))
        switch_clip = float(
            self.params.get("switch_surv_score_clip", DEFAULT_PARAMS["switch_surv_score_clip"])
        )
        switch_min_level = float(
            self.params.get("switch_surv_min_level", DEFAULT_PARAMS["switch_surv_min_level"])
        )
        switch_max_level = float(
            self.params.get("switch_surv_max_level", DEFAULT_PARAMS["switch_surv_max_level"])
        )

        if (
            asset_bank_coeff != 0.0
            or quality_coeff != 0.0
            or buy_t1_coeff != 0.0
            or buy_weapon_t1_coeff != 0.0
            or switch_coeff != 0.0
        ):
            for idx in range(count):
                if item_delta[idx] > 0.0:
                    new_item = _find_new_item(prev_obs[idx], post_obs[idx])
                    if new_item is not None:
                        item_slot = int(round(float(new_item[ITEM_SLOT_IDX])))
                        tier_norm = float(new_item[ITEM_TIER_IDX])
                        is_t1 = tier_norm >= 0.999
                        if is_t1:
                            rewards[idx] += buy_t1_coeff * economy_scale[idx]
                            if item_slot == int(Slot.WEAPON):
                                rewards[idx] += buy_weapon_t1_coeff * economy_scale[idx]

                        if quality_coeff != 0.0 and item_slot != int(Slot.NONE):
                            equipped_pre = _equipped_item_for_slot(prev_obs[idx], item_slot)
                            pre_quality = (
                                _item_quality_score(equipped_pre)
                                if equipped_pre is not None
                                else 0.0
                            )
                            delta = _item_quality_score(new_item) - pre_quality
                            if delta > 0.0:
                                rewards[idx] += quality_coeff * delta * economy_scale[idx]

                    if asset_bank_coeff != 0.0 and post_level[idx] <= asset_bank_max_level:
                        delta = _best_slot_asset_score(post_obs[idx]) - _best_slot_asset_score(
                            prev_obs[idx]
                        )
                        if delta > 0.0:
                            rewards[idx] += asset_bank_coeff * delta * economy_scale[idx]

                if (
                    switch_coeff != 0.0
                    and valid[idx]
                    and prev_phase[idx] == 3
                    and switch_min_level <= post_level[idx] <= switch_max_level
                ):
                    delta = _combat_swap_score(post_obs[idx]) - _combat_swap_score(prev_obs[idx])
                    if switch_clip > 0.0:
                        delta = max(-switch_clip, min(switch_clip, delta))
                    if delta > 0.0:
                        rewards[idx] += switch_coeff * delta * attack_scale[idx]

        return rewards.astype(np.float32, copy=False)
