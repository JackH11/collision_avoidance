"""Shared helpers for headless env tests."""

from __future__ import annotations

import os

# Avoid opening a real display when pygame imports SDL.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

INFO_KEYS = {
    "collision",
    "goal_reached",
    "timeout",
    "goal_dist",
    "min_obstacle_dist",
    "steps",
    "action",
    "scenario",
    "reward_mode",
    "prediction_backend",
}
