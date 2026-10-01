#!/usr/bin/env python3
"""
Evaluate a saved SB3 policy over N episodes.

Prints success / collision / timeout rates plus mean return and distances.

Example:
  python eval_policy.py --model dqn_avoidance_agent5 --episodes 50 --seed 0
  python eval_policy.py --scenario hard --reward-mode new --episodes 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from gym_env import MovingAvoidanceEnv


def evaluate(
    model_path: str,
    episodes: int = 20,
    seed: int = 0,
    deterministic: bool = True,
    prediction_backend: str = "simple",
    scenario: str = "baseline",
    reward_mode: str = "old",
) -> dict:
    env = MovingAvoidanceEnv(
        prediction_backend=prediction_backend,
        scenario=scenario,
        reward_mode=reward_mode,
    )
    model = DQN.load(model_path)

    outcomes = {"success": 0, "collision": 0, "timeout": 0, "other": 0}
    returns = []
    goal_dists = []
    min_clearances = []
    lengths = []

    t0 = time.perf_counter()
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_min_clearance = float(info.get("min_obstacle_dist", float("inf")))
        last_info = info

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_min_clearance = min(
                ep_min_clearance, float(info.get("min_obstacle_dist", ep_min_clearance))
            )
            last_info = info
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(int(last_info.get("steps", 0)))
        goal_dists.append(float(last_info.get("goal_dist", float("nan"))))
        min_clearances.append(ep_min_clearance)

        if last_info.get("goal_reached"):
            outcomes["success"] += 1
        elif last_info.get("collision"):
            outcomes["collision"] += 1
        elif last_info.get("timeout"):
            outcomes["timeout"] += 1
        else:
            outcomes["other"] += 1

    env.close()
    elapsed = time.perf_counter() - t0
    n = float(episodes)

    summary = {
        "model": str(model_path),
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "prediction_backend": prediction_backend,
        "scenario": scenario,
        "reward_mode": reward_mode,
        "success_rate": outcomes["success"] / n,
        "collision_rate": outcomes["collision"] / n,
        "timeout_rate": outcomes["timeout"] / n,
        "other_rate": outcomes["other"] / n,
        "counts": outcomes,
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_final_goal_dist": float(np.nanmean(goal_dists)),
        "mean_min_obstacle_dist": float(np.mean(min_clearances)),
        "elapsed_sec": elapsed,
        "sec_per_episode": elapsed / n,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a collision-avoidance policy")
    parser.add_argument(
        "--model",
        default="dqn_avoidance_agent5",
        help="Path to SB3 zip (with or without .zip)",
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
        choices=["easy", "baseline", "hard"],
        help="Named scenario from config.yaml",
    )
    parser.add_argument(
        "--reward-mode",
        default="old",
        choices=["old", "new"],
        help="Reward variant (old = baseline-compatible)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the summary JSON",
    )
    args = parser.parse_args()

    model_path = args.model
    if not model_path.endswith(".zip") and not Path(model_path).exists():
        candidate = Path(f"{model_path}.zip")
        if candidate.exists():
            model_path = str(candidate)

    summary = evaluate(
        model_path=model_path,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        prediction_backend=args.prediction_backend,
        scenario=args.scenario,
        reward_mode=args.reward_mode,
    )

    print("=== Policy eval ===")
    print(f"model:              {summary['model']}")
    print(f"episodes:           {summary['episodes']} (seed={summary['seed']})")
    print(f"scenario:           {summary['scenario']}")
    print(f"reward_mode:        {summary['reward_mode']}")
    print(f"prediction:         {summary['prediction_backend']}")
    print(f"success_rate:       {summary['success_rate']:.3f}")
    print(f"collision_rate:     {summary['collision_rate']:.3f}")
    print(f"timeout_rate:       {summary['timeout_rate']:.3f}")
    print(f"mean_return:        {summary['mean_return']:.2f}")
    print(f"mean_ep_length:     {summary['mean_episode_length']:.1f}")
    print(f"mean_final_goal_d:  {summary['mean_final_goal_dist']:.2f}")
    print(f"mean_min_clearance: {summary['mean_min_obstacle_dist']:.2f}")
    print(f"elapsed_sec:        {summary['elapsed_sec']:.1f}")
    print(f"sec_per_episode:    {summary['sec_per_episode']:.2f}")
    print(f"counts:             {summary['counts']}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
