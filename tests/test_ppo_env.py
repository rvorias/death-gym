"""Tests for the PPO-compatible vectorized environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from dmfast import VecEnv as NativePPOVecEnv


def _dict_obs_total_dim(obs: dict) -> int:
    """Sum total elements across all dict obs values."""
    return sum(v.shape[-1] for v in obs.values())


class TestNativePPOVecEnv:
    """Test PPO env interface compatibility."""

    def test_create_and_spaces(self):
        env = NativePPOVecEnv(num_envs=4, seed=42)
        # Dict observation space with 7 keys summing to 463 dims
        assert isinstance(env.observation_space, dict.__class__) or hasattr(env.observation_space, "spaces")
        total_dim = sum(s.shape[0] for s in env.observation_space.spaces.values())
        assert total_dim == 463
        assert env.action_space.n == 57
        assert env.num_envs == 4
        env.close()

    def test_reset_returns_obs(self):
        env = NativePPOVecEnv(num_envs=2, seed=42)
        obs = env.reset()
        assert isinstance(obs, dict)
        assert obs["game_phase"].shape == (2, 1)
        assert obs["adventurer"].shape == (2, 13)
        assert obs["equipment"].shape == (2, 72)
        assert obs["bag"].shape == (2, 135)
        assert obs["market"].shape == (2, 225)
        assert obs["beast"].shape == (2, 7)
        assert obs["sim_stats"].shape == (2, 10)
        # Total dims should be 463
        assert _dict_obs_total_dim(obs) == 463
        env.close()

    def test_step_returns_correct_shapes(self):
        env = NativePPOVecEnv(num_envs=4, seed=42, max_steps=256)
        env.reset()
        masks = env.action_masks()
        actions = np.array([
            np.random.choice(np.flatnonzero(masks[i])) for i in range(4)
        ], dtype=np.int32)
        obs, rewards, dones, infos = env.step(actions)
        assert isinstance(obs, dict)
        assert _dict_obs_total_dim(obs) == 463
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
        env.close()

    def test_action_masks(self):
        env = NativePPOVecEnv(num_envs=2, seed=42)
        env.reset()
        masks = env.action_masks()
        assert masks.shape == (2, 57)
        assert masks.dtype == np.uint8
        for i in range(2):
            assert masks[i].sum() > 0, f"Env {i} should have valid actions"
        env.close()

    def test_episode_tracking(self):
        """Run until at least one episode completes and check info dict."""
        env = NativePPOVecEnv(num_envs=8, seed=42, max_steps=50)
        env.reset()
        rng = np.random.default_rng(0)
        saw_episode = False
        for _ in range(200):
            masks = env.action_masks()
            actions = np.array([
                rng.choice(np.flatnonzero(masks[i])) if masks[i].sum() > 0 else 0
                for i in range(8)
            ], dtype=np.int32)
            obs, rewards, dones, infos = env.step(actions)
            for i in range(8):
                if dones[i]:
                    saw_episode = True
                    assert "episode" in infos[i]
                    assert "r" in infos[i]["episode"]
                    assert "l" in infos[i]["episode"]
                    assert infos[i]["episode"]["l"] > 0
        assert saw_episode, "Expected at least one episode to complete"
        env.close()

    def test_reward_config(self):
        """Test that reward config is applied."""
        env = NativePPOVecEnv(
            num_envs=2, seed=42, max_steps=256,
            reward_config={"kill": 5.0, "attack": 0.20}
        )
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(50):
            masks = env.action_masks()
            actions = np.array([
                rng.choice(np.flatnonzero(masks[i])) if masks[i].sum() > 0 else 0
                for i in range(2)
            ], dtype=np.int32)
            obs, rewards, dones, infos = env.step(actions)
            assert np.all(np.isfinite(rewards))
        env.close()

    def test_sb3_vecenv_subclass(self):
        """Verify it's a proper VecEnv subclass."""
        from stable_baselines3.common.vec_env import VecEnv
        env = NativePPOVecEnv(num_envs=2, seed=42)
        assert isinstance(env, VecEnv)
        env.close()

    def test_step_async_wait(self):
        """Test async step interface (used internally by SB3)."""
        env = NativePPOVecEnv(num_envs=2, seed=42, max_steps=256)
        env.reset()
        masks = env.action_masks()
        actions = np.array([
            np.random.choice(np.flatnonzero(masks[i])) for i in range(2)
        ], dtype=np.int32)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
        assert isinstance(obs, dict)
        assert _dict_obs_total_dim(obs) == 463
        assert rewards.shape == (2,)
        env.close()

    def test_maskable_ppo_smoke(self):
        """Smoke test: create MaskablePPO and run a few steps."""
        from sb3_contrib import MaskablePPO
        env = NativePPOVecEnv(num_envs=4, seed=42, max_steps=128)
        model = MaskablePPO(
            "MultiInputPolicy", env,
            n_steps=32, batch_size=16, n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=128)
        env.close()

    def test_info_dict_fields(self):
        """Test that step returns info dict with per-env metrics."""
        env = NativePPOVecEnv(num_envs=4, seed=42, max_steps=256)
        env.reset()
        rng = np.random.default_rng(0)
        masks = env.action_masks()
        actions = np.array([
            rng.choice(np.flatnonzero(masks[i])) for i in range(4)
        ], dtype=np.int32)
        obs, rewards, dones, infos = env.step(actions)
        expected_fields = [
            "adventurer_level", "xp", "health", "gold",
            "beast_level", "beasts_killed", "items_bought", "potions_bought",
            "game_phase",
        ]
        for info in infos:
            for field in expected_fields:
                assert field in info, f"Missing info field: {field}"
        # adventurer_level should be >= 1
        assert all(info["adventurer_level"] >= 1.0 for info in infos)
        env.close()

    def test_throughput_ppo_loop(self):
        """Simulate a PPO rollout collection loop."""
        import time
        num_envs = 64
        n_steps = 512
        env = NativePPOVecEnv(num_envs=num_envs, seed=42, max_steps=256)
        env.reset()
        rng = np.random.default_rng(0)

        t0 = time.perf_counter()
        for _ in range(n_steps):
            masks = env.action_masks()
            actions = np.array([
                rng.choice(np.flatnonzero(masks[i])) if masks[i].sum() > 0 else 0
                for i in range(num_envs)
            ], dtype=np.int32)
            obs, rewards, dones, infos = env.step(actions)

        dt = time.perf_counter() - t0
        total_steps = num_envs * n_steps
        print(f"\nPPO rollout throughput: {total_steps/dt:,.0f} env_steps/sec "
              f"({num_envs} envs × {n_steps} steps in {dt:.2f}s)")
        env.close()
