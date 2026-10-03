#!/usr/bin/env python3
"""
Dump agent trajectories to JSON for the Godot replay viewer.

Keeps Python as the sim/policy brain; Godot only renders.

Examples:
  python dump_trajectory.py --scenario large --episodes 5 --seed 0 \\
      --out media/trajectories/large_best.json
  python dump_trajectory.py --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip \\
      --algo qrdqn --scenario large --episodes 3 --out media/trajectories/run.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from collision_avoidance.env.gym_env import MovingAvoidanceEnv
from collision_avoidance.policy.model_loader import load_model
from collision_avoidance.prediction.model_prediction import resolve_prediction_backend

DEFAULT_MODEL_CANDIDATES = [
    "models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip",
    "models/ppo_CnnPolicy_easy_s0_cont/final_model.zip",
    "models/ppo_CnnPolicy_easy_s0/final_model.zip",
    "dqn_avoidance_agent5.zip",
    "dqn_avoidance_agent5",
]


def _pick_default_model() -> str:
    for path in DEFAULT_MODEL_CANDIDATES:
        if Path(path).exists() or Path(f"{path}.zip").exists():
            return path
    return "dqn_avoidance_agent5"


def _outcome(info: dict) -> str:
    if info.get("goal_reached"):
        return "success"
    if info.get("collision"):
        return "collision"
    if info.get("timeout"):
        return "timeout"
    return "other"


def _frame_snapshot(env: MovingAvoidanceEnv, action: int, predictions) -> Dict[str, Any]:
    preds = []
    if predictions is not None:
        for pred in predictions:
            px, py, sx, sy = pred
            preds.append(
                {
                    "x": float(px),
                    "y": float(py),
                    "std_x": float(sx),
                    "std_y": float(sy),
                }
            )
    obstacles = []
    for obs in env.obstacles:
        obstacles.append(
            {
                "x": float(obs.x),
                "y": float(obs.y),
                "vx": float(obs.vx),
                "vy": float(obs.vy),
                "noise": bool(getattr(obs, "add_noise", False)),
            }
        )
    return {
        "t": int(env.steps),
        "agent": {
            "x": float(env.agent.x),
            "y": float(env.agent.y),
            "vx": float(env.agent.vx),
            "vy": float(env.agent.vy),
            "action": int(action),
        },
        "goal": {"x": float(env.goal_x), "y": float(env.goal_y)},
        "obstacles": obstacles,
        "predictions": preds,
        "info": {
            "goal_dist": float(env._goal_dist()),
            "min_obstacle_dist": float(env._min_obstacle_dist()),
        },
    }


def dump_trajectory(
    *,
    model_path: str,
    algo: Optional[str],
    scenario: str,
    seed: int,
    episodes: int,
    predictor: str,
    reward_mode: Optional[str],
    include_predictions: bool,
) -> Dict[str, Any]:
    env = MovingAvoidanceEnv(
        scenario=scenario,
        reward_mode=reward_mode,
        prediction_backend=predictor,
    )
    policy, algo_name, resolved = load_model(model_path, algo=algo)
    _, predict_fn = resolve_prediction_backend(predictor)
    if hasattr(env.agent, "prediction_model"):
        # reset recreates agent; set after each reset too
        pass

    episodes_out: List[Dict[str, Any]] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        env.agent.prediction_model = predict_fn
        frames: List[Dict[str, Any]] = []
        ep_return = 0.0
        action = 0
        done = False
        # Initial pose before first action
        preds = (
            env.agent.make_predictions(env.obstacles) if include_predictions else None
        )
        frames.append(_frame_snapshot(env, action, preds))

        while not done:
            action_arr, _ = policy.predict(obs, deterministic=True)
            action = int(action_arr)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            preds = (
                env.agent.make_predictions(env.obstacles)
                if include_predictions
                else None
            )
            frames.append(_frame_snapshot(env, action, preds))
            done = bool(terminated or truncated)

        episodes_out.append(
            {
                "episode": ep,
                "seed": seed + ep,
                "outcome": _outcome(info),
                "return": float(ep_return),
                "steps": int(info.get("steps", len(frames) - 1)),
                "frames": frames,
            }
        )

    payload = {
        "version": 1,
        "meta": {
            "scenario": env.scenario_name,
            "algo": algo_name,
            "model": str(resolved),
            "seed": seed,
            "episodes": episodes,
            "predictor": env.prediction_backend,
            "reward_mode": env.reward_mode,
            "fps": 30,
            "boundary": {"width": int(env.width), "height": int(env.height)},
            "window": {
                "width": int(env.window_width),
                "height": int(env.window_height),
            },
            "agent_radius": float(env.item_radius),
            "obstacle_radius": float(env.item_radius),
            "goal_radius": float(env.goal_radius),
            "agent_speed": float(env.agent_speed),
        },
        "episodes": episodes_out,
    }
    env.close()
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="Dump trajectories for Godot replay")
    p.add_argument("--model", default=None)
    p.add_argument("--algo", default=None, choices=["ppo", "qrdqn", "dqn"])
    p.add_argument(
        "--scenario",
        default="large",
        choices=["easy", "baseline", "hard", "large"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument(
        "--predictor",
        default="simple",
        choices=["simple", "nn", "nn_uncertainty"],
    )
    p.add_argument("--reward-mode", default=None, choices=["old", "new"])
    p.add_argument(
        "--out",
        default="media/trajectories/large_best.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--no-predictions",
        action="store_true",
        help="Omit prediction cones (smaller files)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model or _pick_default_model()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = dump_trajectory(
        model_path=model_path,
        algo=args.algo,
        scenario=args.scenario,
        seed=args.seed,
        episodes=args.episodes,
        predictor=args.predictor,
        reward_mode=args.reward_mode,
        include_predictions=not args.no_predictions,
    )
    out.write_text(json.dumps(payload, separators=(",", ":")))
    n_frames = sum(len(ep["frames"]) for ep in payload["episodes"])
    outcomes = {ep["outcome"]: 0 for ep in payload["episodes"]}
    for ep in payload["episodes"]:
        outcomes[ep["outcome"]] += 1
    print(
        f"Wrote {out}  episodes={len(payload['episodes'])} frames={n_frames} "
        f"outcomes={outcomes} size_kb={out.stat().st_size / 1024:.1f}"
    )
    print(f"Replay in Godot: open godot_replay/ and set trajectory path to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
