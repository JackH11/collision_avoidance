#!/usr/bin/env python3
"""
Dump agent trajectories to JSON for the Godot replay viewer.

Keeps Python as the sim/policy brain; Godot only renders.

Examples:
  python -m collision_avoidance.demo.dump_trajectory --scenario large --episodes 5
  # Showcase: keep rolling until N solid successes
  python -m collision_avoidance.demo.dump_trajectory --scenario large \\
      --successes 4 --min-steps 70 --algo qrdqn \\
      --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip
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
    successes: Optional[int] = None,
    min_steps: int = 0,
    max_attempts: int = 64,
) -> Dict[str, Any]:
    env = MovingAvoidanceEnv(
        scenario=scenario,
        reward_mode=reward_mode,
        prediction_backend=predictor,
    )
    policy, algo_name, resolved = load_model(model_path, algo=algo)
    _, predict_fn = resolve_prediction_backend(predictor)

    episodes_out: List[Dict[str, Any]] = []
    attempt = 0
    target = successes if successes is not None else episodes
    # When collecting successes, keep rolling until we have enough (capped).
    while len(episodes_out) < target and attempt < max_attempts:
        obs, info = env.reset(seed=seed + attempt)
        env.agent.prediction_model = predict_fn
        frames: List[Dict[str, Any]] = []
        ep_return = 0.0
        action = 0
        done = False
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

        outcome = _outcome(info)
        steps = int(info.get("steps", len(frames) - 1))
        attempt += 1

        if successes is not None and (outcome != "success" or steps < min_steps):
            continue

        episodes_out.append(
            {
                "episode": len(episodes_out),
                "seed": seed + attempt - 1,
                "outcome": outcome,
                "return": float(ep_return),
                "steps": steps,
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
            "episodes": len(episodes_out),
            "attempts": attempt,
            "successes_requested": successes,
            "min_steps": min_steps,
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
        "--successes",
        type=int,
        default=None,
        help="Keep rolling until this many successful episodes (showcase mode)",
    )
    p.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="With --successes, discard shorter runs (avoids instant goal taps)",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=64,
        help="Cap on env resets when using --successes",
    )
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
        successes=args.successes,
        min_steps=args.min_steps,
        max_attempts=args.max_attempts,
    )
    out.write_text(json.dumps(payload, separators=(",", ":")))
    n_frames = sum(len(ep["frames"]) for ep in payload["episodes"])
    outcomes = {ep["outcome"]: 0 for ep in payload["episodes"]}
    for ep in payload["episodes"]:
        outcomes[ep["outcome"]] += 1
    print(
        f"Wrote {out}  episodes={len(payload['episodes'])} frames={n_frames} "
        f"outcomes={outcomes} attempts={payload['meta'].get('attempts')} "
        f"size_kb={out.stat().st_size / 1024:.1f}"
    )
    if args.successes and len(payload["episodes"]) < args.successes:
        print(
            f"WARNING: only collected {len(payload['episodes'])}/{args.successes} "
            f"successes in {payload['meta'].get('attempts')} attempts "
            f"(large-map transfer is imperfect — try more --max-attempts or baseline)"
        )
    print(f"Replay in Godot: open godot_replay/ and set trajectory path to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
