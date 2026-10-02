"""Eval helpers: baselines, cone fraction, regression gate thresholds."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from eval_policy import _cone_risk_fraction, _percentile, evaluate
from evals.baselines import (
    BASELINE_FACTORIES,
    make_freeze_policy,
    make_greedy_policy,
    make_random_policy,
)
from evals.regression_gate import check_thresholds, load_gate_config
from gym_env import MovingAvoidanceEnv


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_factories_registered():
    assert set(BASELINE_FACTORIES) == {"random", "greedy", "freeze"}


def test_random_policy_in_range():
    predict = make_random_policy(seed=0)
    env = MovingAvoidanceEnv(scenario="easy")
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(20):
            a = predict(obs, env)
            assert 0 <= a < 8
    finally:
        env.close()


def test_freeze_policy_constant():
    predict = make_freeze_policy(3)
    assert predict(None, None) == 3


def test_greedy_policy_picks_toward_goal():
    env = MovingAvoidanceEnv(scenario="easy")
    try:
        env.reset(seed=0)
        # Place goal due east of agent
        env.agent.x = 50
        env.agent.y = 75
        env.goal_x = 120
        env.goal_y = 75
        predict = make_greedy_policy()
        action = predict(np.zeros((3, 30, 30), dtype=np.float32), env)
        assert action == 0  # east
    finally:
        env.close()


def test_cone_risk_fraction_and_percentile():
    obs = np.zeros((3, 30, 30), dtype=np.float32)
    assert _cone_risk_fraction(obs) == 0.0
    # Stamp prediction cells (code 1) in the center neighborhood
    obs[0, 14:17, 14:17] = 1.0
    frac = _cone_risk_fraction(obs, neighborhood=1)
    assert frac > 0.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert math.isnan(_percentile([], 10))


def test_gate_config_loads():
    cfg = load_gate_config(ROOT / "evals" / "gate_config.yaml")
    assert cfg["success_floor"] == pytest.approx(0.30)
    assert cfg["collision_ceiling"] == pytest.approx(0.70)
    assert cfg["scenario"] == "baseline"


def test_check_thresholds_pass_and_fail():
    ok, msgs = check_thresholds(
        {"success_rate": 0.48, "collision_rate": 0.52},
        success_floor=0.30,
        collision_ceiling=0.70,
    )
    assert ok is True
    ok, msgs = check_thresholds(
        {"success_rate": 0.02, "collision_rate": 0.98},
        success_floor=0.30,
        collision_ceiling=0.70,
    )
    assert ok is False
    assert any("success_rate" in m for m in msgs)


def test_gate_from_checked_in_json():
    path = ROOT / "evals" / "ppo_final_baseline_old.json"
    summary = json.loads(path.read_text())
    cfg = load_gate_config(ROOT / "evals" / "gate_config.yaml")
    ok, _ = check_thresholds(
        summary,
        success_floor=float(cfg["success_floor"]),
        collision_ceiling=float(cfg["collision_ceiling"]),
    )
    assert ok is True
    assert summary["success_rate"] >= 0.30


def test_evaluate_callable_baseline_smoke():
    """Short rollout with greedy policy — no model zip required."""
    summary = evaluate(
        model_path=None,
        episodes=3,
        seed=0,
        scenario="easy",
        reward_mode="old",
        predict_fn=make_greedy_policy(),
        policy_name="greedy",
    )
    assert summary["episodes"] == 3
    assert summary["policy_name"] == "greedy"
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert "near_miss_rate" in summary
    assert "mean_time_to_goal" in summary
