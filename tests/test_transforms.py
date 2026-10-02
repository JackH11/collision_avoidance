"""Lag feature builder (transforms/clean.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from transforms.clean import create_lag_features


def test_create_lag_features_adds_columns_and_drops_na():
    rows = []
    for ep in (0, 1):
        for item in ("a", "b"):
            for frame in range(5):
                rows.append(
                    {
                        "episode": ep,
                        "item": item,
                        "frame": frame,
                        "x": float(frame),
                        "y": float(frame) * 2,
                        "vx": 1.0,
                        "vy": 0.5,
                    }
                )
    df = pd.DataFrame(rows)
    out = create_lag_features(df, lag_steps=2)
    assert "x_lag_1" in out.columns
    assert "y_lag_2" in out.columns
    assert "vx_lag_1" in out.columns
    # First ``lag_steps`` frames per (item, episode) are dropped
    # 2 episodes × 2 items × (5 - 2) = 12 rows
    assert len(out) == 12
    sample = out[(out["episode"] == 0) & (out["item"] == "a") & (out["frame"] == 2)].iloc[
        0
    ]
    assert sample["x_lag_1"] == pytest.approx(1.0)
    assert sample["x_lag_2"] == pytest.approx(0.0)
