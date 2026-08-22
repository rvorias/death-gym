"""Core batch environment backed by the native C engine.

This is the primary API for running Death Mountain game simulations.
All environments share a single C-allocated batch buffer with zero-copy
numpy views for maximum throughput.

Usage:
    from dmfast import BatchEnv

    env = BatchEnv(num_envs=64, seed=42)
    for _ in range(1000):
        actions = env.sample_masked_actions(rng)
        env.step(actions)
        # env.obs, env.reward, env.terminated, env.action_mask are live views
    env.close()
"""
from __future__ import annotations

import ctypes
import numpy as np
from pathlib import Path

from .reward import INFO_FIELDS, StateDiffRewardModel

# ── Load shared library ─────────────────────────────────────────────────
# Built by `just build` (or `make -C engine`) into engine/build/, alongside this
# package. The engine is compiled from a clone rather than shipped in the wheel,
# so a non-editable install will not find it.
_LIB_PATH = Path(__file__).resolve().parent.parent / "engine" / "build" / "libdmfast.so"
if not _LIB_PATH.exists():
    raise ImportError(
        f"native engine not found at {_LIB_PATH}\n"
        "Build it first:  just build   (or: make -C engine)"
    )
_lib = ctypes.CDLL(str(_LIB_PATH))

# ── C function signatures ───────────────────────────────────────────────
_lib.dmfast_exact_create.restype = ctypes.c_void_p
_lib.dmfast_exact_create.argtypes = [ctypes.c_int32, ctypes.c_uint64, ctypes.c_int32]

_lib.dmfast_exact_destroy.restype = None
_lib.dmfast_exact_destroy.argtypes = [ctypes.c_void_p]

_lib.dmfast_exact_reset_all.restype = None
_lib.dmfast_exact_reset_all.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

_lib.dmfast_exact_step.restype = None
_lib.dmfast_exact_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]

_lib.dmfast_exact_state_size.restype = ctypes.c_int32
_lib.dmfast_exact_state_size.argtypes = []
_lib.dmfast_exact_save_state.restype = None
_lib.dmfast_exact_save_state.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
_lib.dmfast_exact_load_state.restype = None
_lib.dmfast_exact_load_state.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
_lib.dmfast_exact_broadcast_state.restype = None
_lib.dmfast_exact_broadcast_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64]

for _name in [
    "dmfast_exact_obs_ptr",
    "dmfast_exact_reward_ptr",
    "dmfast_exact_episode_return_ptr",
    "dmfast_exact_last_episode_return_ptr",
    "dmfast_exact_last_episode_info_ptr",
    "dmfast_exact_last_terminal_obs_ptr",
]:
    _fn = getattr(_lib, _name)
    _fn.restype = ctypes.POINTER(ctypes.c_float)
    _fn.argtypes = [ctypes.c_void_p]

for _name in [
    "dmfast_exact_action_mask_ptr",
    "dmfast_exact_terminated_ptr",
    "dmfast_exact_truncated_ptr",
    "dmfast_exact_last_terminal_action_mask_ptr",
]:
    _fn = getattr(_lib, _name)
    _fn.restype = ctypes.POINTER(ctypes.c_uint8)
    _fn.argtypes = [ctypes.c_void_p]

for _name in [
    "dmfast_exact_phase_ptr",
    "dmfast_exact_step_count_ptr",
    "dmfast_exact_last_episode_length_ptr",
]:
    _fn = getattr(_lib, _name)
    _fn.restype = ctypes.POINTER(ctypes.c_int32)
    _fn.argtypes = [ctypes.c_void_p]

_lib.dmfast_exact_info_ptr.restype = ctypes.POINTER(ctypes.c_float)
_lib.dmfast_exact_info_ptr.argtypes = [ctypes.c_void_p]

_lib.dmfast_exact_info_dim.restype = ctypes.c_int32
_lib.dmfast_exact_info_dim.argtypes = []

_lib.dmfast_exact_set_reward_param.restype = ctypes.c_int32
_lib.dmfast_exact_set_reward_param.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float,
]

_lib.dmfast_exact_sample_masked_actions.restype = None
_lib.dmfast_exact_sample_masked_actions.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
]

_lib.dmfast_reward_compute_batch.restype = None
_lib.dmfast_reward_compute_batch.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
]

# ── Constants ────────────────────────────────────────────────────────────
OBS_DIM = 463
ACTION_DIM = 57
INFO_DIM = int(_lib.dmfast_exact_info_dim())
# ── BatchEnv ─────────────────────────────────────────────────────────────
class BatchEnv:
    """Vectorized game environment backed by the native C engine.

    Runs `num_envs` independent game instances in a single C-allocated batch.
    All buffers (obs, reward, action_mask, etc.) are zero-copy numpy views
    into C memory — no per-step allocation overhead.

    Attributes:
        obs:              (num_envs, 463) float32  — observation vectors
        action_mask:      (num_envs, 57)  uint8    — valid action mask
        reward:           (num_envs,)     float32  — per-step reward (Python state-diff model)
        terminated:       (num_envs,)     uint8    — death flag
        truncated:        (num_envs,)     uint8    — max-steps flag
        phase:            (num_envs,)     int32    — game phase enum
        step_count:       (num_envs,)     int32    — steps in current episode
        episode_return:   (num_envs,)     float32  — cumulative reward (Python state-diff model)
        last_episode_return: (num_envs,)  float32  — return of last completed episode
        last_episode_length: (num_envs,)  int32    — length of last completed episode
        last_episode_info:   (num_envs, 8) float32 — terminal metrics of last completed episode
        last_terminal_obs:   (num_envs, 463) float32 — final obs before auto-reset for done envs
        last_terminal_action_mask: (num_envs, 57) uint8 — final mask before auto-reset for done envs
        info:             (num_envs, 8)   float32  — per-env metrics
    """

    def __init__(
        self,
        num_envs: int = 0,
        seed: int = 42,
        max_steps: int = 512,
        reward_config: dict[str, float] | None = None,
        *,
        batch_size: int = 0,  # backwards compat alias for num_envs
    ):
        if batch_size > 0 and num_envs == 0:
            num_envs = batch_size
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.batch_size = self.num_envs  # backwards compat
        self._handle = _lib.dmfast_exact_create(
            ctypes.c_int32(self.num_envs),
            ctypes.c_uint64(seed),
            ctypes.c_int32(max_steps),
        )
        if not self._handle:
            raise RuntimeError("Failed to create C batch engine")

        try:
            self._reward_model = StateDiffRewardModel(reward_config)

            # Zero-copy numpy views into C buffers
            n = self.num_envs
            self.obs = np.ctypeslib.as_array(
                _lib.dmfast_exact_obs_ptr(self._handle), shape=(n, OBS_DIM))
            self.action_mask = np.ctypeslib.as_array(
                _lib.dmfast_exact_action_mask_ptr(self._handle), shape=(n, ACTION_DIM))
            self._c_reward = np.ctypeslib.as_array(
                _lib.dmfast_exact_reward_ptr(self._handle), shape=(n,))
            self.terminated = np.ctypeslib.as_array(
                _lib.dmfast_exact_terminated_ptr(self._handle), shape=(n,))
            self.truncated = np.ctypeslib.as_array(
                _lib.dmfast_exact_truncated_ptr(self._handle), shape=(n,))
            self.phase = np.ctypeslib.as_array(
                _lib.dmfast_exact_phase_ptr(self._handle), shape=(n,))
            self.step_count = np.ctypeslib.as_array(
                _lib.dmfast_exact_step_count_ptr(self._handle), shape=(n,))
            self._c_episode_return = np.ctypeslib.as_array(
                _lib.dmfast_exact_episode_return_ptr(self._handle), shape=(n,))
            self._c_last_episode_return = np.ctypeslib.as_array(
                _lib.dmfast_exact_last_episode_return_ptr(self._handle), shape=(n,))
            self.last_episode_length = np.ctypeslib.as_array(
                _lib.dmfast_exact_last_episode_length_ptr(self._handle), shape=(n,))
            self.last_episode_info = np.ctypeslib.as_array(
                _lib.dmfast_exact_last_episode_info_ptr(self._handle), shape=(n, INFO_DIM))
            self.last_terminal_obs = np.ctypeslib.as_array(
                _lib.dmfast_exact_last_terminal_obs_ptr(self._handle), shape=(n, OBS_DIM))
            self.last_terminal_action_mask = np.ctypeslib.as_array(
                _lib.dmfast_exact_last_terminal_action_mask_ptr(self._handle), shape=(n, ACTION_DIM))
            self.info = np.ctypeslib.as_array(
                _lib.dmfast_exact_info_ptr(self._handle), shape=(n, INFO_DIM))
            self.reward = np.zeros(n, dtype=np.float32)
            self.episode_return = np.zeros(n, dtype=np.float32)
            self.last_episode_return = np.zeros(n, dtype=np.float32)
            self._reward_prev_obs = self.obs.copy()
            self._reward_prev_info = self.info.copy()
            self._reward_prev_mask = self.action_mask.copy()
        except Exception:
            self.close()
            raise

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    def set_reward_param(self, name: str, value: float):
        """Set a reward parameter for the Python-side state-diff reward model."""
        self._reward_model.set_param(name, value)

    def set_engine_param(self, name: str, value: float) -> int:
        """Set an EXACT-ENGINE config param (strcmp chain in dmfast_exact.c),
        e.g. the Go-Explore curriculum knobs (curriculum_snapshot_prob etc.)."""
        return int(_lib.dmfast_exact_set_reward_param(
            self._handle, name.encode(), ctypes.c_float(float(value))))

    def reset(self, seed: int = 42):
        """Reset all environments with the given seed."""
        _lib.dmfast_exact_reset_all(self._handle, ctypes.c_uint64(seed))
        self.reward.fill(0.0)
        self.episode_return.fill(0.0)
        self.last_episode_return.fill(0.0)
        self._reward_prev_obs[:] = self.obs
        self._reward_prev_info[:] = self.info
        self._reward_prev_mask[:] = self.action_mask

    def step(self, actions: np.ndarray, auto_reset: bool = True):
        """Step all environments. Rewards are derived from observable state diffs."""
        actions = np.asarray(actions, dtype=np.int32)
        prev_episode_return = self.episode_return.copy()
        _lib.dmfast_exact_step(
            self._handle,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(int(auto_reset)),
        )
        post_obs = self.obs.copy()
        post_info = self.info.copy()
        dones = (self.terminated | self.truncated).astype(bool)
        if auto_reset and np.any(dones):
            done_indices = np.flatnonzero(dones)
            post_obs[done_indices] = self.last_terminal_obs[done_indices]
            post_info[done_indices] = self.last_episode_info[done_indices]

        reward_config = self._reward_model.config_array()
        _lib.dmfast_reward_compute_batch(
            reward_config.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._reward_prev_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._reward_prev_info.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._reward_prev_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            post_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            post_info.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.terminated.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.truncated.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int32(self.num_envs),
            self.reward.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self.episode_return[:] = prev_episode_return + self.reward
        self.last_episode_return.fill(0.0)
        if np.any(dones):
            done_indices = np.flatnonzero(dones)
            self.last_episode_return[done_indices] = self.episode_return[done_indices]
            if auto_reset:
                self.episode_return[done_indices] = 0.0

        self._reward_prev_obs[:] = self.obs
        self._reward_prev_info[:] = self.info
        self._reward_prev_mask[:] = self.action_mask

    def action_masks(self) -> np.ndarray:
        """Return current action masks. Shape: (num_envs, ACTION_DIM)."""
        return self.action_mask.copy()

    # ── State snapshot / restore (search & expert-iteration support) ──

    @staticmethod
    def state_size() -> int:
        """Size in bytes of one env's opaque state blob."""
        return int(_lib.dmfast_exact_state_size())

    def save_state(self, idx: int) -> np.ndarray:
        """Snapshot env `idx` into an opaque uint8 blob (incl. its rng)."""
        buf = np.empty(self.state_size(), dtype=np.uint8)
        _lib.dmfast_exact_save_state(
            self._handle, ctypes.c_int32(idx),
            buf.ctypes.data_as(ctypes.c_void_p))
        return buf

    def load_state(self, idx: int, blob: np.ndarray) -> None:
        """Restore env `idx` from a snapshot; obs/mask refresh, done cleared."""
        blob = np.ascontiguousarray(blob, dtype=np.uint8)
        _lib.dmfast_exact_load_state(
            self._handle, ctypes.c_int32(idx),
            blob.ctypes.data_as(ctypes.c_void_p))

    def broadcast_state(self, blob: np.ndarray, count: int | None = None,
                        reseed: int = 1) -> None:
        """Clone ONE snapshot into envs [0, count) — the search primitive.

        reseed != 0 decorrelates each clone's rng so rollouts sample
        different futures (the right planning semantics); reseed == 0 keeps
        bit-identical replay."""
        blob = np.ascontiguousarray(blob, dtype=np.uint8)
        n = self.num_envs if count is None else int(count)
        _lib.dmfast_exact_broadcast_state(
            self._handle, blob.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(n), ctypes.c_uint64(reseed))

    def sample_masked_actions(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a random valid action per environment (C-accelerated)."""
        actions = np.zeros(self.num_envs, dtype=np.int32)
        # Use C kernel for vectorized sampling
        rng_state = ctypes.c_uint64(
            int(rng.integers(0, 2**64, dtype=np.uint64)) if rng is not None
            else np.random.default_rng().integers(0, 2**64, dtype=np.uint64)
        )
        _lib.dmfast_exact_sample_masked_actions(
            self._handle,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(rng_state),
        )
        return actions

    def close(self):
        """Free C resources."""
        if self._handle:
            _lib.dmfast_exact_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


# Backwards compatibility
ExactNativeBatchEnv = BatchEnv
