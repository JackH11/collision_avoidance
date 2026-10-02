"""
Scripted baseline policies for honesty checks against learned agents.

- random: uniform Discrete-8
- greedy: always move toward the goal (closest of 8 directions)
- freeze: always action 0 (east) — weak “do nothing useful” control
  (Discrete-8 has no true stay; action 0 is a fixed direction)
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

PredictFn = Callable[[np.ndarray, Any], int]

# Match MovingAvoidanceEnv._action_to_angle
_ACTION_ANGLES_DEG = (0, 45, 90, 135, 180, 225, 270, 315)


def make_random_policy(seed: int = 0) -> PredictFn:
    rng = np.random.default_rng(seed)

    def predict(_obs: np.ndarray, _env: Any) -> int:
        return int(rng.integers(0, 8))

    return predict


def make_greedy_policy() -> PredictFn:
    """Choose the discrete action whose velocity best aligns with goal vector."""

    def predict(_obs: np.ndarray, env: Any) -> int:
        agent = env.agent
        dx = float(env.goal_x) - float(agent.x)
        dy = float(env.goal_y) - float(agent.y)
        if dx == 0.0 and dy == 0.0:
            return 0
        target = math.atan2(dy, dx)
        best_a = 0
        best_diff = float("inf")
        for a, deg in enumerate(_ACTION_ANGLES_DEG):
            ang = math.radians(deg)
            diff = abs(math.atan2(math.sin(ang - target), math.cos(ang - target)))
            if diff < best_diff:
                best_diff = diff
                best_a = a
        return int(best_a)

    return predict


def make_freeze_policy(action: int = 0) -> PredictFn:
    fixed = int(action) % 8

    def predict(_obs: np.ndarray, _env: Any) -> int:
        return fixed

    return predict


BASELINE_FACTORIES = {
    "random": lambda seed=0: make_random_policy(seed),
    "greedy": lambda seed=0: make_greedy_policy(),
    "freeze": lambda seed=0: make_freeze_policy(0),
}
