"""SB3-compatible vectorized environment for MaskablePPO training.

Usage:
    from dmfast import VecEnv
    from sb3_contrib import MaskablePPO

    env = VecEnv(num_envs=64, seed=42, max_steps=512)
    model = MaskablePPO("MultiInputPolicy", env, n_steps=512)
    model.learn(total_timesteps=10_000_000)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv as SB3VecEnv

from .env import BatchEnv, OBS_DIM, ACTION_DIM, INFO_FIELDS

# ── Obs layout: flat 463-dim → Dict slices ──────────────────────────────
# game_phase[1] + adventurer[13] + equipment[72] + bag[135] + market[225] + beast[7] + sim_stats[10]
_OBS_SLICES = {
    "game_phase":  (0,   1),
    "adventurer":  (1,   14),
    "equipment":   (14,  86),
    "bag":         (86,  221),
    "market":      (221, 446),
    "beast":       (446, 453),
    "sim_stats":   (453, 463),
}

# Pre-compute as tuple for faster iteration
_OBS_SLICE_KEYS = tuple(_OBS_SLICES.keys())
_OBS_SLICE_RANGES = tuple(_OBS_SLICES.values())


def _make_dict_obs_space() -> gym.spaces.Dict:
    """Build Dict observation space matching Python EngineEnv layout."""
    item_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    item_high = np.array([101, 5, 5, 8, 16, 69, 18, 1, 1], dtype=np.float32)
    return gym.spaces.Dict({
        "game_phase": gym.spaces.Box(
            low=np.array([0], dtype=np.float32),
            high=np.array([6], dtype=np.float32),
            dtype=np.float32,
        ),
        "adventurer": gym.spaces.Box(
            low=np.zeros(13, dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4], dtype=np.float32),
            dtype=np.float32,
        ),
        "equipment": gym.spaces.Box(
            low=np.tile(item_low, 8),
            high=np.tile(item_high, 8),
            dtype=np.float32,
        ),
        "bag": gym.spaces.Box(
            low=np.tile(item_low, 15),
            high=np.tile(item_high, 15),
            dtype=np.float32,
        ),
        "market": gym.spaces.Box(
            low=np.tile(item_low, 25),
            high=np.tile(item_high, 25),
            dtype=np.float32,
        ),
        "beast": gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 5, 16, 69, 18], dtype=np.float32),
            dtype=np.float32,
        ),
        "sim_stats": gym.spaces.Box(
            low=np.zeros(10, dtype=np.float32),
            high=np.array([10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32),
            dtype=np.float32,
        ),
    })


def _flat_to_dict(flat_obs: np.ndarray) -> dict[str, np.ndarray]:
    """Split flat (N, 463) obs into Dict of numpy arrays."""
    return {
        key: flat_obs[:, start:end]
        for key, (start, end) in _OBS_SLICES.items()
    }


class _LazyInfoDict(dict):
    """Lightweight info dict backed by shared numpy arrays.

    Defers per-env value extraction until first access. The shared arrays
    (info_buf row, phase value) are stored once; individual float values
    are only created when __getitem__ or .get() is called.
    """
    __slots__ = ("_row", "_phase", "_extras", "_populated")

    def __init__(self, info_row: np.ndarray, phase: int):
        # Don't call super().__init__() with data — start empty
        super().__init__()
        self._row = info_row        # (INFO_DIM,) float32 view
        self._phase = phase
        self._extras = None         # for episode/terminal_observation
        self._populated = False

    def _populate(self):
        if self._populated:
            return
        self._populated = True
        row = self._row
        for j, name in enumerate(INFO_FIELDS):
            super().__setitem__(name, float(row[j]))
        super().__setitem__("game_phase", self._phase)
        if self._extras:
            for k, v in self._extras.items():
                super().__setitem__(k, v)

    def _set_extra(self, key, value):
        """Set extra keys (episode, terminal_observation) without full populate."""
        if self._extras is None:
            self._extras = {}
        self._extras[key] = value
        if self._populated:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if not self._populated:
            self._populate()
        return super().__getitem__(key)

    def get(self, key, default=None):
        if not self._populated:
            # Fast path for SB3 hot-loop keys — avoid full populate
            if key == "game_phase":
                return self._phase
            if key == "episode" or key == "terminal_observation" or key == "TimeLimit.truncated" or key == "is_success":
                if self._extras and key in self._extras:
                    return self._extras[key]
                return default
            # INFO_FIELDS direct lookup without full populate
            for j, name in enumerate(INFO_FIELDS):
                if key == name:
                    return float(self._row[j])
            self._populate()
        return super().get(key, default)

    def __contains__(self, key):
        if not self._populated:
            if key == "game_phase":
                return True
            if self._extras and key in self._extras:
                return True
            if key in INFO_FIELDS:
                return True
            self._populate()
        return super().__contains__(key)

    def __iter__(self):
        if not self._populated:
            self._populate()
        return super().__iter__()

    def items(self):
        if not self._populated:
            self._populate()
        return super().items()

    def keys(self):
        if not self._populated:
            self._populate()
        return super().keys()

    def values(self):
        if not self._populated:
            self._populate()
        return super().values()


class VecEnv(SB3VecEnv):
    """SB3-compatible vectorized environment backed by the C batch engine.

    Implements the full VecEnv interface including action_masks() for
    MaskablePPO. All environments run in a single process via the C engine.

    Observations are returned as Dict matching the Python EngineEnv format.
    """

    def __init__(
        self,
        num_envs: int,
        seed: int = 42,
        max_steps: int = 512,
        reward_config: dict[str, float] | None = None,
    ):
        self._engine = BatchEnv(
            num_envs=num_envs,
            seed=seed,
            max_steps=max_steps,
            reward_config=reward_config,
        )

        observation_space = _make_dict_obs_space()
        action_space = gym.spaces.Discrete(ACTION_DIM)
        super().__init__(num_envs, observation_space, action_space)

        self._actions: np.ndarray | None = None

    # ── VecEnv abstract methods ──────────────────────────────────────────

    def reset(self) -> dict[str, np.ndarray]:
        self._engine.reset()
        return _flat_to_dict(self._engine.obs.copy())

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.int32)

    def step_wait(self):
        self._engine.step(self._actions, auto_reset=True)
        self._actions = None

        flat_obs = self._engine.obs.copy()
        obs = _flat_to_dict(flat_obs)
        rewards = self._engine.reward.copy()
        terminated = self._engine.terminated
        truncated = self._engine.truncated
        dones = (terminated | truncated).astype(bool)

        # Build lazy info dicts — O(1) per env, values extracted on demand
        info_buf = self._engine.info
        last_info_buf = self._engine.last_episode_info
        phase_arr = self._engine.phase
        n = self.num_envs
        infos = [None] * n

        # Only populate episode data for done envs (typically <5% of envs)
        done_indices = np.flatnonzero(dones)
        done_set = set(done_indices.tolist()) if done_indices.size > 0 else set()

        for i in range(n):
            # For done envs, use last_episode_info (terminal state before auto-reset)
            # instead of info (post-reset state which shows fresh episode values)
            row = last_info_buf[i] if i in done_set else info_buf[i]
            infos[i] = _LazyInfoDict(row, int(phase_arr[i]))

        if done_indices.size > 0:
            ler = self._engine.last_episode_return
            lel = self._engine.last_episode_length
            for i in done_indices:
                i = int(i)
                infos[i]._set_extra("episode", {"r": float(ler[i]), "l": int(lel[i])})
                infos[i]._set_extra("terminal_observation", {
                    _OBS_SLICE_KEYS[k]: self._engine.last_terminal_obs[i, _OBS_SLICE_RANGES[k][0]:_OBS_SLICE_RANGES[k][1]].copy()
                    for k in range(len(_OBS_SLICE_KEYS))
                })
                infos[i]._set_extra("TimeLimit.truncated", bool(truncated[i]))

        return obs, rewards, dones, infos

    def close(self) -> None:
        self._engine.close()

    def seed(self, seed=None):
        if seed is not None:
            self._engine.reset(seed=seed)
        return [seed] * self.num_envs

    def get_attr(self, attr_name: str, indices=None):
        idx = range(self.num_envs) if indices is None else indices
        if attr_name == "action_mask":
            return [self._engine.action_mask[i].copy() for i in idx]
        if attr_name == "render_mode":
            return [None for _ in idx]
        if attr_name == "action_masks":
            return [self.action_masks for _ in idx]
        raise AttributeError(f"Unknown attr: {attr_name}")

    def set_attr(self, attr_name: str, value, indices=None):
        raise AttributeError(f"Cannot set attr: {attr_name}")

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        idx = range(self.num_envs) if indices is None else indices
        if method_name in ("valid_action_mask", "action_masks"):
            return [self._engine.action_mask[i].copy() for i in idx]
        raise AttributeError(f"Unknown method: {method_name}")

    def env_is_wrapped(self, wrapper_class, indices=None):
        idx = range(self.num_envs) if indices is None else indices
        return [False for _ in idx]

    # ── MaskablePPO interface ────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return action masks for MaskablePPO. Shape: (num_envs, ACTION_DIM)."""
        return self._engine.action_mask.copy()

    # ── Convenience ──────────────────────────────────────────────────────

    def set_reward_param(self, name: str, value: float):
        """Set a reward shaping parameter."""
        self._engine.set_reward_param(name, value)


# Backwards compatibility
NativePPOVecEnv = VecEnv
