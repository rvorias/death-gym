"""dmfast — High-performance vectorized game engine for RL training.

Public API:
    BatchEnv  — Core batch environment (framework-agnostic)
    VecEnv    — SB3-compatible vectorized environment (MaskablePPO)

    OBS_DIM     = 463  (observation vector dimension)
    ACTION_DIM  = 57   (discrete action count)
    INFO_DIM    = 8    (per-env info fields)
    INFO_FIELDS = ["adventurer_level", "xp", "health", "gold",
                   "beast_level", "beasts_killed", "items_bought", "potions_bought"]

Usage:
    from dmfast import BatchEnv, VecEnv

    # Framework-agnostic (raw C engine):
    env = BatchEnv(num_envs=64, seed=42)
    env.step(actions)  # updates env.obs, env.reward, env.action_mask in-place

    # With SB3 MaskablePPO:
    env = VecEnv(num_envs=64, seed=42)
    model = MaskablePPO("MultiInputPolicy", env)
    model.learn(total_timesteps=10_000_000)
"""

from .env import BatchEnv, OBS_DIM, ACTION_DIM, INFO_DIM, INFO_FIELDS
from .env import ExactNativeBatchEnv  # backwards-compatible alias

__all__ = [
    "BatchEnv",
    "VecEnv",
    "OBS_DIM",
    "ACTION_DIM",
    "INFO_DIM",
    "INFO_FIELDS",
    # Backwards compatibility
    "ExactNativeBatchEnv",
    "NativePPOVecEnv",
]

# VecEnv is the Stable-Baselines3 adapter. It is imported lazily so the core
# engine stays usable with only numpy installed — `pip install .[sb3]` adds it.
_LAZY = {"VecEnv": "VecEnv", "NativePPOVecEnv": "NativePPOVecEnv"}


def __getattr__(name):
    if name in _LAZY:
        try:
            from . import vec_env
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                f"dmfast.{name} needs gymnasium and stable-baselines3. "
                "Install them with: pip install -e '.[sb3]'"
            ) from exc
        return getattr(vec_env, _LAZY[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
