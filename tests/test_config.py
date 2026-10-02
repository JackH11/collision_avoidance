"""Config / scenario resolution tests."""

from __future__ import annotations

import pytest

from config import RAW_CONFIG, load_config, resolve_scenario
from gym_env import MovingAvoidanceEnv


def test_load_config_has_scenarios():
    cfg = load_config()
    assert "scenarios" in cfg
    assert set(cfg["scenarios"]) >= {"easy", "baseline", "hard", "large"}


def test_resolve_scenario_applies_knobs_without_bleed():
    hard = resolve_scenario(RAW_CONFIG, scenario="hard")
    easy = resolve_scenario(RAW_CONFIG, scenario="easy")
    assert hard["obstacle"]["count"] == 8
    assert easy["obstacle"]["count"] == 3
    assert hard["episode"]["max_steps"] == 600
    assert easy["episode"]["max_steps"] == 1200
    # Switching hard → easy must not keep hard max_speed
    assert easy["obstacle"].get("max_speed", 6.0) == pytest.approx(6.0)
    assert hard["obstacle"]["max_speed"] == pytest.approx(7.0)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        resolve_scenario(RAW_CONFIG, scenario="does_not_exist")


def test_env_respects_scenario_counts():
    for name, count in (("easy", 3), ("baseline", 5), ("hard", 8), ("large", 8)):
        env = MovingAvoidanceEnv(scenario=name)
        try:
            env.reset(seed=0)
            assert env.ITEM_COUNT == count
            assert len(env.obstacles) == count
            assert env.scenario_name == name
        finally:
            env.close()


def test_large_scenario_arena():
    env = MovingAvoidanceEnv(scenario="large")
    try:
        assert env.width == 400
        assert env.height == 400
        assert env.window_width == 520
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 30, 30)
    finally:
        env.close()
