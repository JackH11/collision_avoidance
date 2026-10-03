#!/usr/bin/env python3
"""
Evaluate a saved SB3 / sb3-contrib policy (or a callable policy) over N episodes.

Reports success / collision / timeout rates, time-to-goal, clearance, and
safety-oriented near-miss / prediction-cone metrics.

Examples:
  python eval_policy.py --model dqn_avoidance_agent5 --episodes 50 --seed 0
  python eval_policy.py --model models/ppo_CnnPolicy_easy_s0/final_model.zip \\
      --algo ppo --scenario baseline --episodes 50
  python eval_policy.py --scenario hard --reward-mode old --episodes 20
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from collision_avoidance.env.gym_env import CELL_PRED, MovingAvoidanceEnv
from collision_avoidance.policy.model_loader import load_model

PredictFn = Callable[[np.ndarray, Any], int]

# Default near-miss distance matches config.yaml reward.near_miss_dist
DEFAULT_NEAR_MISS_DIST = 24.0
# Grid radius (cells) around agent center used for "in prediction cone" checks
DEFAULT_CONE_NEIGHBORHOOD = 2


def _cone_risk_fraction(obs: np.ndarray, neighborhood: int = DEFAULT_CONE_NEIGHBORHOOD) -> float:
    """Fraction of cells near grid center that are prediction-cone occupancy."""
    occ = np.asarray(obs)[0]
    h, w = occ.shape
    cy, cx = h // 2, w // 2
    r = int(neighborhood)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = occ[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch == CELL_PRED))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def evaluate(
    model_path: Optional[str] = None,
    episodes: int = 20,
    seed: int = 0,
    deterministic: bool = True,
    prediction_backend: str = "simple",
    scenario: str = "baseline",
    reward_mode: str = "old",
    algo: str | None = None,
    *,
    predict_fn: Optional[PredictFn] = None,
    policy_name: Optional[str] = None,
    near_miss_dist: float = DEFAULT_NEAR_MISS_DIST,
    cone_neighborhood: int = DEFAULT_CONE_NEIGHBORHOOD,
    env: Optional[MovingAvoidanceEnv] = None,
) -> dict:
    """
    Roll out ``episodes`` seeded episodes and return a metrics summary.

    Provide either ``model_path`` (SB3 zip) or ``predict_fn(obs, env) -> action``.
    """
    owns_env = env is None
    if env is None:
        env = MovingAvoidanceEnv(
            prediction_backend=prediction_backend,
            scenario=scenario,
            reward_mode=reward_mode,
        )

    resolved: Union[str, Path] = policy_name or "callable"
    algo_name = "callable"
    model = None
    if predict_fn is None:
        if not model_path:
            raise ValueError("Provide model_path or predict_fn")
        model, algo_name, resolved = load_model(model_path, algo=algo)

        def predict_fn(obs: np.ndarray, _env: Any) -> int:  # type: ignore[misc]
            action, _ = model.predict(obs, deterministic=deterministic)
            return int(action)

    assert predict_fn is not None

    outcomes = {"success": 0, "collision": 0, "timeout": 0, "other": 0}
    returns: list[float] = []
    goal_dists: list[float] = []
    min_clearances: list[float] = []
    lengths: list[int] = []
    time_to_goal: list[float] = []
    near_miss_episode_flags: list[bool] = []
    near_miss_step_fractions: list[float] = []
    cone_step_fractions: list[float] = []
    mean_cone_cell_fractions: list[float] = []

    t0 = time.perf_counter()
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_min_clearance = float(info.get("min_obstacle_dist", float("inf")))
        last_info = info
        steps = 0
        near_miss_steps = 0
        cone_risk_steps = 0
        cone_cell_sum = 0.0

        while not done:
            action = int(predict_fn(obs, env))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            steps += 1
            clearance = float(info.get("min_obstacle_dist", ep_min_clearance))
            if math.isfinite(clearance):
                ep_min_clearance = min(ep_min_clearance, clearance)
                if clearance < near_miss_dist:
                    near_miss_steps += 1
            cone_frac = _cone_risk_fraction(obs, cone_neighborhood)
            cone_cell_sum += cone_frac
            if cone_frac > 0.0:
                cone_risk_steps += 1
            last_info = info
            done = terminated or truncated

        returns.append(ep_return)
        ep_len = int(last_info.get("steps", steps))
        lengths.append(ep_len)
        goal_dists.append(float(last_info.get("goal_dist", float("nan"))))
        min_clearances.append(ep_min_clearance)
        near_miss_episode_flags.append(near_miss_steps > 0)
        near_miss_step_fractions.append(
            (near_miss_steps / ep_len) if ep_len > 0 else 0.0
        )
        cone_step_fractions.append(
            (cone_risk_steps / ep_len) if ep_len > 0 else 0.0
        )
        mean_cone_cell_fractions.append(
            (cone_cell_sum / ep_len) if ep_len > 0 else 0.0
        )

        if last_info.get("goal_reached"):
            outcomes["success"] += 1
            time_to_goal.append(float(ep_len))
        elif last_info.get("collision"):
            outcomes["collision"] += 1
        elif last_info.get("timeout"):
            outcomes["timeout"] += 1
        else:
            outcomes["other"] += 1

    if owns_env:
        env.close()
    elapsed = time.perf_counter() - t0
    n = float(episodes)
    success_ttg = time_to_goal  # only successful episodes

    summary = {
        "model": str(resolved),
        "algo": algo_name,
        "policy_name": policy_name or algo_name,
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "prediction_backend": prediction_backend
        if owns_env
        else getattr(env, "prediction_backend", prediction_backend),
        "scenario": scenario if owns_env else getattr(env, "scenario_name", scenario),
        "reward_mode": reward_mode if owns_env else getattr(env, "reward_mode", reward_mode),
        "near_miss_dist": float(near_miss_dist),
        "success_rate": outcomes["success"] / n,
        "collision_rate": outcomes["collision"] / n,
        "timeout_rate": outcomes["timeout"] / n,
        "other_rate": outcomes["other"] / n,
        "counts": outcomes,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "median_return": float(np.median(returns)) if returns else float("nan"),
        "mean_episode_length": float(np.mean(lengths)) if lengths else float("nan"),
        "mean_final_goal_dist": float(np.nanmean(goal_dists)) if goal_dists else float("nan"),
        "mean_min_obstacle_dist": float(np.mean(min_clearances)) if min_clearances else float("nan"),
        "median_min_obstacle_dist": float(np.median(min_clearances))
        if min_clearances
        else float("nan"),
        "worst_min_obstacle_dist": float(np.min(min_clearances))
        if min_clearances
        else float("nan"),
        "p10_min_obstacle_dist": _percentile(min_clearances, 10),
        # Time-to-goal (successful episodes only)
        "mean_time_to_goal": float(np.mean(success_ttg)) if success_ttg else float("nan"),
        "median_time_to_goal": float(np.median(success_ttg)) if success_ttg else float("nan"),
        "success_episodes_with_ttg": len(success_ttg),
        # Safety
        "near_miss_rate": float(np.mean(near_miss_episode_flags))
        if near_miss_episode_flags
        else float("nan"),
        "mean_near_miss_step_fraction": float(np.mean(near_miss_step_fractions))
        if near_miss_step_fractions
        else float("nan"),
        "mean_cone_risk_step_fraction": float(np.mean(cone_step_fractions))
        if cone_step_fractions
        else float("nan"),
        "mean_cone_cell_fraction": float(np.mean(mean_cone_cell_fractions))
        if mean_cone_cell_fractions
        else float("nan"),
        "elapsed_sec": elapsed,
        "sec_per_episode": elapsed / n if n else float("nan"),
    }
    return summary


def print_summary(summary: dict) -> None:
    print("=== Policy eval ===")
    print(f"model:              {summary['model']}")
    print(f"algo / policy:      {summary.get('policy_name', summary.get('algo'))}")
    print(f"episodes:           {summary['episodes']} (seed={summary['seed']})")
    print(f"scenario:           {summary['scenario']}")
    print(f"reward_mode:        {summary['reward_mode']}")
    print(f"prediction:         {summary['prediction_backend']}")
    print(f"success_rate:       {summary['success_rate']:.3f}")
    print(f"collision_rate:     {summary['collision_rate']:.3f}")
    print(f"timeout_rate:       {summary['timeout_rate']:.3f}")
    print(f"mean_return:        {summary['mean_return']:.2f}")
    print(f"median_return:      {summary['median_return']:.2f}")
    print(f"mean_ep_length:     {summary['mean_episode_length']:.1f}")
    print(f"mean_time_to_goal:  {summary['mean_time_to_goal']:.1f}")
    print(f"median_time_to_goal:{summary['median_time_to_goal']:.1f}")
    print(f"mean_final_goal_d:  {summary['mean_final_goal_dist']:.2f}")
    print(f"mean_min_clearance: {summary['mean_min_obstacle_dist']:.2f}")
    print(f"worst_min_clearance:{summary['worst_min_obstacle_dist']:.2f}")
    print(f"near_miss_rate:     {summary['near_miss_rate']:.3f}")
    print(f"mean_near_miss_frac:{summary['mean_near_miss_step_fraction']:.3f}")
    print(f"cone_risk_step_frac:{summary['mean_cone_risk_step_fraction']:.3f}")
    print(f"elapsed_sec:        {summary['elapsed_sec']:.1f}")
    print(f"sec_per_episode:    {summary['sec_per_episode']:.2f}")
    print(f"counts:             {summary['counts']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a collision-avoidance policy")
    parser.add_argument(
        "--model",
        default="dqn_avoidance_agent5",
        help="Path to SB3 zip (with or without .zip)",
    )
    parser.add_argument(
        "--algo",
        default=None,
        choices=["ppo", "qrdqn", "dqn"],
        help="Algorithm hint (auto-detected from filename when omitted)",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default: deterministic)",
    )
    parser.add_argument(
        "--prediction-backend",
        default="simple",
        choices=["simple", "nn", "nn_uncertainty"],
        help="Obstacle trajectory predictor (nn* requires TensorFlow)",
    )
    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=["easy", "baseline", "hard", "large"],
        help="Named scenario from config.yaml",
    )
    parser.add_argument(
        "--reward-mode",
        default="old",
        choices=["old", "new"],
        help="Reward variant (old = baseline-compatible)",
    )
    parser.add_argument(
        "--near-miss-dist",
        type=float,
        default=DEFAULT_NEAR_MISS_DIST,
        help="Clearance threshold for near-miss metrics (pixels)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the summary JSON",
    )
    args = parser.parse_args()

    summary = evaluate(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        prediction_backend=args.prediction_backend,
        scenario=args.scenario,
        reward_mode=args.reward_mode,
        algo=args.algo,
        near_miss_dist=args.near_miss_dist,
    )
    print_summary(summary)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
