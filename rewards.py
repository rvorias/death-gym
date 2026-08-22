"""Reward API over the native engine.

Reward calculation lives in the native reward kernel (`engine/src/dmfast_reward.c`);
Python only selects and configures the reward presets handed to it.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dmfast import ACTION_DIM, INFO_DIM, OBS_DIM, BatchEnv
from dmfast.reward import DEFAULT_PARAMS


class Info:
    LEVEL = 0
    XP = 1
    HEALTH = 2
    GOLD = 3
    BEAST_LEVEL = 4
    BEASTS_KILLED = 5
    ITEMS_BOUGHT = 6
    POTIONS_BOUGHT = 7
    STARTED_FROM_CURRICULUM_SNAPSHOT = 8


RewardConfig = dict[str, float]

_NEUTRAL_PHASES: RewardConfig = {
    "phase_very_early_end": -1.0,
    "phase_early_end": 1.0e9,
    "phase_mid_end": 1.0e9,
    "phase_late_end": 1.0e9,
    "phase_scale_economy": 1.0,
    "phase_scale_stat": 1.0,
    "phase_scale_attack": 1.0,
    "phase_scale_kill": 1.0,
    "phase_scale_high_kill": 1.0,
    "phase_scale_flee": 1.0,
    "phase_scale_potion": 1.0,
    "phase_scale_xp": 1.0,
}


def reward_config(**overrides: float) -> RewardConfig:
    """Build a native reward config with zeroed shaping by default.

    The native engine has many optional shaping terms. Keep
    the API small and explicit: start from zero, keep invalid-action handling,
    neutralize phase tables, then opt into only the terms you want.
    """

    config: RewardConfig = {key: 0.0 for key in DEFAULT_PARAMS}
    config["invalid_action_penalty"] = float(DEFAULT_PARAMS["invalid_action_penalty"])
    config.update(_NEUTRAL_PHASES)
    config.update({key: float(value) for key, value in overrides.items()})
    return config


def item_xp_per_step_config(scale: float = 1.0) -> RewardConfig:
    """Reward only positive held-item XP gain, computed natively."""

    return reward_config(
        item_xp_gain=scale,
        invalid_action_penalty=0.0,
    )


def rich_reward_config() -> RewardConfig:
    """Use the engine's full default reward config (buy/kill/flee/sim shaping)."""
    return dict(DEFAULT_PARAMS)


def kills_and_death_config() -> RewardConfig:
    """Native kill shaping with a light beast-level multiplier and death penalty."""

    return reward_config(
        kill=0.02,
        death_penalty=-1.0,
    )


def kills_levelup_death_config() -> RewardConfig:
    """Native kill + level-up shaping preset close to the old Python preset."""

    return reward_config(
        kill=0.05,
        level_up=2.0,
        death_penalty=-0.5,
    )


class GameEnv:
    """Thin training-side wrapper over the native batch env.

    This keeps the training code stable while delegating both gameplay and
    reward calculation to the mirrored fast engine prototype.
    """

    def __init__(
        self,
        num_envs: int,
        seed: int = 42,
        max_steps: int = 512,
        reward_config: RewardConfig | None = None,
    ):
        self._engine = BatchEnv(
            num_envs=num_envs,
            seed=seed,
            max_steps=max_steps,
            reward_config=reward_config,
        )

        self.num_envs = self._engine.num_envs
        self.obs = self._engine.obs
        self.action_mask = self._engine.action_mask
        self.reward = self._engine.reward
        self.terminated = self._engine.terminated
        self.truncated = self._engine.truncated
        self.info = self._engine.info
        self.episode_return = self._engine.episode_return
        self.last_episode_info = self._engine.last_episode_info
        self.last_episode_return = self._engine.last_episode_return
        self.last_episode_length = self._engine.last_episode_length
        self.last_terminal_obs = self._engine.last_terminal_obs
        self.last_terminal_action_mask = self._engine.last_terminal_action_mask

    def reset(self, seed: int = 42):
        self._engine.reset(seed)

    def step(self, actions):
        self._engine.step(actions, auto_reset=True)

    def sample_masked_actions(self, rng=None):
        return self._engine.sample_masked_actions(rng)

    def set_reward_param(self, name: str, value: float):
        self._engine.set_reward_param(name, value)

    def set_engine_param(self, name: str, value: float):
        return self._engine.set_engine_param(name, value)

    def close(self):
        self._engine.close()

    def __del__(self):
        self.close()
