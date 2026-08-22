"""Smoke tests for the exact native batch engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from dmfast import BatchEnv as ExactNativeBatchEnv


class TestExactNativeSmoke:
    """Basic functionality: create, reset, step, auto-reset."""

    def test_create_and_shapes(self):
        env = ExactNativeBatchEnv(batch_size=4, seed=123, max_steps=256)
        assert env.obs.shape == (4, 463)
        assert env.action_mask.shape == (4, 57)
        assert env.reward.shape == (4,)
        assert env.terminated.shape == (4,)
        assert env.truncated.shape == (4,)
        env.close()

    def test_initial_obs_nonzero(self):
        env = ExactNativeBatchEnv(batch_size=2, seed=42, max_steps=256)
        # After reset, obs should be non-trivial
        assert env.obs.sum() > 0, "Obs should be non-zero after reset"
        # Health should be positive
        assert env.obs[0, 1] > 0, "Health should be positive"
        env.close()

    def test_initial_mask_has_valid_actions(self):
        env = ExactNativeBatchEnv(batch_size=2, seed=42, max_steps=256)
        for i in range(2):
            valid = np.flatnonzero(env.action_mask[i])
            assert valid.size > 0, f"Env {i} should have valid actions"
        env.close()

    def test_initial_phase_is_upgrade(self):
        env = ExactNativeBatchEnv(batch_size=4, seed=42, max_steps=256)
        # After start_game, phase should be UPGRADE (0)
        for i in range(4):
            assert env.phase[i] == 0, f"Initial phase should be UPGRADE (0), got {env.phase[i]}"
        env.close()

    def test_step_with_valid_action(self):
        env = ExactNativeBatchEnv(batch_size=2, seed=42, max_steps=256)
        actions = env.sample_masked_actions(np.random.default_rng(0))
        env.step(actions, auto_reset=True)
        # Should not crash; rewards should be finite
        assert np.all(np.isfinite(env.reward))
        env.close()

    def test_run_many_steps(self):
        """Run 200 random steps without crashing."""
        env = ExactNativeBatchEnv(batch_size=8, seed=42, max_steps=256)
        rng = np.random.default_rng(1337)
        completed_episodes = 0
        for _ in range(200):
            actions = env.sample_masked_actions(rng)
            env.step(actions, auto_reset=True)
            completed_episodes += int(env.terminated.sum() + env.truncated.sum())
            assert np.all(np.isfinite(env.obs)), "Obs has non-finite values"
            assert np.all(np.isfinite(env.reward)), "Reward has non-finite values"
            # Every env should have at least one valid action
            for i in range(8):
                if not env.terminated[i] and not env.truncated[i]:
                    assert env.action_mask[i].sum() > 0, f"Env {i} has no valid actions"
        print(f"Completed {completed_episodes} episodes in 200 steps across 8 envs")
        env.close()

    def test_auto_reset_preserves_buffers(self):
        """After auto-reset, obs and mask should be valid for the new episode."""
        env = ExactNativeBatchEnv(batch_size=4, seed=42, max_steps=50)
        rng = np.random.default_rng(0)
        saw_reset = False
        for _ in range(200):
            actions = env.sample_masked_actions(rng)
            env.step(actions, auto_reset=True)
            if env.terminated.any() or env.truncated.any():
                saw_reset = True
                for i in range(4):
                    if env.terminated[i] or env.truncated[i]:
                        # After auto-reset, should have valid obs and mask
                        assert env.obs[i].sum() > 0
                        assert env.action_mask[i].sum() > 0
                        assert env.last_episode_length[i] > 0
        assert saw_reset, "Expected at least one episode to complete"
        env.close()

    def test_episode_return_tracks(self):
        """Episode return should accumulate rewards."""
        env = ExactNativeBatchEnv(batch_size=1, seed=42, max_steps=256)
        total = 0.0
        for _ in range(20):
            actions = env.sample_masked_actions(np.random.default_rng(0))
            env.step(actions, auto_reset=False)
            total += float(env.reward[0])
            if env.terminated[0] or env.truncated[0]:
                break
        assert abs(env.episode_return[0] - total) < 0.01, \
            f"Episode return {env.episode_return[0]} doesn't match sum {total}"
        env.close()

    def test_obs_layout_game_phase(self):
        """First obs dimension should be game_phase (0=upgrade after reset)."""
        env = ExactNativeBatchEnv(batch_size=1, seed=42, max_steps=256)
        assert env.obs[0, 0] == 0.0, f"game_phase should be 0 (upgrade), got {env.obs[0, 0]}"
        env.close()

    def test_obs_layout_health_positive(self):
        """Second obs dimension (health / MAX_HEALTH) should be positive after reset."""
        env = ExactNativeBatchEnv(batch_size=1, seed=42, max_steps=256)
        hp_norm = env.obs[0, 1]
        assert 0.05 < hp_norm <= 1.0, f"Health should be in (0.05, 1.0], got {hp_norm}"
        env.close()

    def test_deterministic_with_same_seed(self):
        """Two envs with same seed should produce identical trajectories."""
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)
        env1 = ExactNativeBatchEnv(batch_size=1, seed=42, max_steps=256)
        env2 = ExactNativeBatchEnv(batch_size=1, seed=42, max_steps=256)
        np.testing.assert_array_equal(env1.obs, env2.obs)
        np.testing.assert_array_equal(env1.action_mask, env2.action_mask)
        for _ in range(50):
            a1 = env1.sample_masked_actions(rng1)
            a2 = env2.sample_masked_actions(rng2)
            np.testing.assert_array_equal(a1, a2)
            env1.step(a1, auto_reset=False)
            env2.step(a2, auto_reset=False)
            np.testing.assert_array_equal(env1.obs, env2.obs)
            np.testing.assert_array_equal(env1.reward, env2.reward)
            if env1.terminated[0] or env1.truncated[0]:
                break
        env1.close()
        env2.close()

    def test_reward_config(self):
        """Test that reward config parameters can be set."""
        env = ExactNativeBatchEnv(batch_size=2, seed=42, max_steps=256,
                                  reward_config={"kill": 2.0, "death_penalty": -5.0})
        rng = np.random.default_rng(0)
        for _ in range(50):
            actions = env.sample_masked_actions(rng)
            env.step(actions, auto_reset=True)
            assert np.all(np.isfinite(env.reward))
        env.close()

    def test_reward_config_set_param(self):
        """Test set_reward_param method."""
        env = ExactNativeBatchEnv(batch_size=1, seed=42, max_steps=256)
        env.set_reward_param("attack", 0.10)
        env.set_reward_param("flee_penalty", -0.50)
        rng = np.random.default_rng(0)
        for _ in range(20):
            actions = env.sample_masked_actions(rng)
            env.step(actions, auto_reset=False)
            if env.terminated[0] or env.truncated[0]:
                break
        env.close()

    def test_rewards_have_structure(self):
        """Run many steps and verify rewards have meaningful structure.
        S049 config: positive from kills/buys, all rewards should be finite."""
        env = ExactNativeBatchEnv(batch_size=8, seed=42, max_steps=256)
        rng = np.random.default_rng(1337)
        saw_positive = False
        max_reward = -float("inf")
        for _ in range(200):
            actions = env.sample_masked_actions(rng)
            env.step(actions, auto_reset=True)
            for i in range(8):
                r = float(env.reward[i])
                if r > 0.5:
                    saw_positive = True
                if r > max_reward:
                    max_reward = r
                assert np.isfinite(r), f"Reward not finite: {r}"
        assert saw_positive, "Should see large positive rewards (kills/buys)"
        assert max_reward > 1.0, f"Max reward too small: {max_reward}"
        env.close()

    def test_info_buffer(self):
        """Test that info buffer exposes correct metrics."""
        env = ExactNativeBatchEnv(batch_size=4, seed=42, max_steps=256)
        # After reset, info should be populated
        assert env.info.shape == (4, 9)
        assert env.info.dtype == np.float32
        # Level should be >= 1
        for i in range(4):
            assert env.info[i, 0] >= 1.0, "adventurer_level should be >= 1"
            assert env.info[i, 2] > 0, "health should be positive"
        # Run some steps and check info updates
        rng = np.random.default_rng(0)
        for _ in range(100):
            actions = env.sample_masked_actions(rng)
            env.step(actions, auto_reset=True)
        # After 100 steps, some envs should have killed beasts or bought items
        total_kills = env.info[:, 5].sum()
        total_items = env.info[:, 6].sum()
        # At least some activity should have happened
        assert env.info[:, 0].max() >= 1.0, "Level should still be >= 1"
        assert np.all(np.isfinite(env.info)), "Info values should be finite"
        env.close()

    def test_throughput(self):
        """Basic throughput benchmark."""
        batch_size = 64
        steps = 1000
        env = ExactNativeBatchEnv(batch_size=batch_size, seed=42, max_steps=256)
        rng = np.random.default_rng(0)
        import time
        t0 = time.perf_counter()
        for _ in range(steps):
            actions = env.sample_masked_actions(rng)
            env.step(actions, auto_reset=True)
        dt = time.perf_counter() - t0
        env_steps = batch_size * steps
        print(f"\nExact native throughput: {env_steps/dt:,.0f} env_steps/sec "
              f"({batch_size} envs × {steps} steps in {dt:.2f}s)")
        env.close()
