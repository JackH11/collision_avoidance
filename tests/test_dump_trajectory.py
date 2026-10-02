"""Smoke tests for trajectory dump used by Godot replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dump_trajectory import dump_trajectory


@pytest.fixture
def tiny_dump(tmp_path: Path):
    model = Path("dqn_avoidance_agent5.zip")
    if not model.exists() and not Path("dqn_avoidance_agent5").exists():
        pytest.skip("legacy dqn zip not present")
    payload = dump_trajectory(
        model_path=str(model if model.exists() else "dqn_avoidance_agent5"),
        algo="dqn",
        scenario="easy",
        seed=0,
        episodes=1,
        predictor="simple",
        reward_mode="old",
        include_predictions=True,
    )
    out = tmp_path / "t.json"
    out.write_text(json.dumps(payload))
    return payload, out


def test_dump_schema(tiny_dump):
    payload, _out = tiny_dump
    assert payload["version"] == 1
    assert "meta" in payload and "episodes" in payload
    assert payload["meta"]["boundary"]["width"] > 0
    assert len(payload["episodes"]) == 1
    ep = payload["episodes"][0]
    assert ep["outcome"] in {"success", "collision", "timeout", "other"}
    assert len(ep["frames"]) >= 2
    fr = ep["frames"][0]
    assert "agent" in fr and "goal" in fr and "obstacles" in fr
    assert "x" in fr["agent"] and "action" in fr["agent"]
