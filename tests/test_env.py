"""Env reset/step, obs shape, info keys, action→velocity, seeded smoke."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gym_env import GRID_SIZE, MovingAvoidanceEnv
from tests.conftest import INFO_KEYS


@pytest.fixture
def env_baseline():
    env = MovingAvoidanceEnv(scenario="baseline", reward_mode="old")
    yield env
    env.close()


def test_observation_space_and_reset_shape(env_baseline):
    assert env_baseline.observation_space.shape == (3, GRID_SIZE, GRID_SIZE)
    assert env_baseline.action_space.n == 8
    obs, info = env_baseline.reset(seed=0)
    assert obs.shape == (3, GRID_SIZE, GRID_SIZE)
    assert obs.dtype == np.float32
    assert INFO_KEYS.issubset(info.keys())
    assert info["scenario"] == "baseline"
    assert info["reward_mode"] == "old"
    assert info["steps"] == 0
    assert info["collision"] is False
    assert info["goal_reached"] is False


def test_step_returns_five_tuple_and_info(env_baseline):
    env_baseline.reset(seed=1)
    obs, reward, terminated, truncated, info = env_baseline.step(0)
    assert obs.shape == (3, GRID_SIZE, GRID_SIZE)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert INFO_KEYS.issubset(info.keys())
    assert info["steps"] == 1
    assert info["action"] == 0


def test_action_to_velocity_cardinal(env_baseline):
    speed = env_baseline.agent_speed
    # action 0 = east (0°)
    vx, vy = env_baseline._action_to_velocity(0)
    assert vx == pytest.approx(speed)
    assert vy == pytest.approx(0.0, abs=1e-9)
    # action 2 = north (90°)
    vx, vy = env_baseline._action_to_velocity(2)
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(speed)
    # action 4 = west (180°)
    vx, vy = env_baseline._action_to_velocity(4)
    assert vx == pytest.approx(-speed)
    assert vy == pytest.approx(0.0, abs=1e-9)


def test_seeded_reset_reproducible_poses():
    """Same seed → same agent/goal/obstacle poses for the first K movers."""
    env_a = MovingAvoidanceEnv(scenario="baseline", reward_mode="old")
    env_b = MovingAvoidanceEnv(scenario="baseline", reward_mode="old")
    try:
        obs_a, info_a = env_a.reset(seed=42)
        obs_b, info_b = env_b.reset(seed=42)
        assert np.allclose(obs_a, obs_b)
        assert info_a["goal_dist"] == pytest.approx(info_b["goal_dist"])
        assert env_a.goal_x == env_b.goal_x
        assert env_a.goal_y == env_b.goal_y
        assert env_a.agent.x == env_b.agent.x
        assert env_a.agent.y == env_b.agent.y
        assert len(env_a.obstacles) == len(env_b.obstacles) == 5
        for oa, ob in zip(env_a.obstacles, env_b.obstacles):
            assert oa.x == ob.x and oa.y == ob.y
            assert oa.vx == pytest.approx(ob.vx)
            assert oa.vy == pytest.approx(ob.vy)
    finally:
        env_a.close()
        env_b.close()


def test_reward_modes_differ_on_progress_step():
    """old vs new reward should diverge on a non-terminal step (living cost)."""
    env_old = MovingAvoidanceEnv(scenario="easy", reward_mode="old")
    env_new = MovingAvoidanceEnv(scenario="easy", reward_mode="new")
    try:
        env_old.reset(seed=7)
        env_new.reset(seed=7)
        # Align poses so Δd is comparable
        env_new.goal_x, env_new.goal_y = env_old.goal_x, env_old.goal_y
        for i, (a, b) in enumerate(zip(env_old.obstacles, env_new.obstacles)):
            b.x, b.y, b.vx, b.vy = a.x, a.y, a.vx, a.vy
        _, r_old, term_old, trunc_old, _ = env_old.step(0)
        _, r_new, term_new, trunc_new, _ = env_new.step(0)
        if term_old or trunc_old or term_new or trunc_new:
            pytest.skip("terminal on first step; retry not needed for CI")
        # Old living cost is −5 + Δd; new is progress − step_cost (± near-miss).
        assert r_old != pytest.approx(r_new)
    finally:
        env_old.close()
        env_new.close()


def test_timeout_truncates_when_max_steps_hit():
    env = MovingAvoidanceEnv(scenario="hard", reward_mode="old")
    env.max_steps = 3
    try:
        env.reset(seed=0)
        terminated = truncated = False
        info = {}
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        assert truncated is True or info.get("timeout") is True or terminated
        if truncated:
            assert info["timeout"] is True
            assert info["steps"] == 3
    finally:
        env.close()
