#!/usr/bin/env python3
"""
Regression gate for collision-avoidance policies (Phase 3).

Fails (exit 1) when baseline-scenario success drops below ``success_floor``
or collision exceeds ``collision_ceiling``.

Exit codes:
  0 — pass
  1 — metrics violate thresholds
  2 — configuration / missing model (unless --skip-if-missing)

Examples:
  python evals/regression_gate.py
  python evals/regression_gate.py --model models/ppo_CnnPolicy_easy_s0/final_model.zip
  python evals/regression_gate.py --from-json evals/ppo_final_baseline_old.json
  make regression-gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "evals" / "gate_config.yaml"

TRAIN_HINT = """\
Phase 2 checkpoints are gitignored under models/. Train then re-run the gate:

  python -m collision_avoidance.train --algo ppo --scenario easy --reward-mode new --timesteps 400000
  python -m collision_avoidance.train --algo qrdqn --scenario easy --reward-mode new --timesteps 300000

Or validate thresholds against a checked-in summary (no zip required):

  python evals/regression_gate.py --from-json evals/ppo_final_baseline_old.json
"""


def load_gate_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    required = ("success_floor", "collision_ceiling", "scenario")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"gate config missing keys: {missing}")
    return data


def resolve_model(cfg: dict[str, Any], explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if not p.exists() and not str(explicit).endswith(".zip"):
            alt = Path(f"{explicit}.zip")
            if alt.exists():
                return alt
        return p if p.exists() else None

    for cand in cfg.get("model_candidates") or []:
        p = Path(cand)
        if p.exists():
            return p
    return None


def check_thresholds(
    summary: dict[str, Any],
    success_floor: float,
    collision_ceiling: float,
) -> tuple[bool, list[str]]:
    success = float(summary["success_rate"])
    collision = float(summary["collision_rate"])
    messages: list[str] = []
    ok = True
    if success < success_floor:
        ok = False
        messages.append(
            f"success_rate {success:.3f} < success_floor {success_floor:.3f}"
        )
    else:
        messages.append(
            f"success_rate {success:.3f} >= success_floor {success_floor:.3f} (ok)"
        )
    if collision > collision_ceiling:
        ok = False
        messages.append(
            f"collision_rate {collision:.3f} > collision_ceiling {collision_ceiling:.3f}"
        )
    else:
        messages.append(
            f"collision_rate {collision:.3f} <= collision_ceiling {collision_ceiling:.3f} (ok)"
        )
    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 regression gate")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to gate_config.yaml",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model zip (default: first existing candidate in config)",
    )
    parser.add_argument(
        "--from-json",
        default=None,
        help="Skip live rollout; gate against an existing eval summary JSON",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episode count for live eval",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-if-missing",
        action="store_true",
        help="Exit 0 if no model zip is available (prints SKIP)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write gate result JSON",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_gate_config(cfg_path)
    success_floor = float(cfg["success_floor"])
    collision_ceiling = float(cfg["collision_ceiling"])

    summary: dict[str, Any]
    source: str

    if args.from_json:
        json_path = Path(args.from_json)
        if not json_path.exists():
            print(f"ERROR: summary JSON not found: {json_path}", file=sys.stderr)
            return 2
        summary = json.loads(json_path.read_text())
        source = f"json:{json_path}"
    else:
        model_path = resolve_model(cfg, args.model)
        if model_path is None:
            msg = "No Phase 2 model zip found for regression gate."
            print(f"ERROR: {msg}", file=sys.stderr)
            print(TRAIN_HINT, file=sys.stderr)
            if args.skip_if_missing:
                print("SKIP: regression gate (no model; --skip-if-missing)")
                return 0
            return 2

        from collision_avoidance.eval.eval_policy import evaluate

        summary = evaluate(
            model_path=str(model_path),
            episodes=int(args.episodes or cfg.get("episodes", 30)),
            seed=int(args.seed if args.seed is not None else cfg.get("seed", 0)),
            deterministic=True,
            prediction_backend=str(cfg.get("prediction_backend", "simple")),
            scenario=str(cfg.get("scenario", "baseline")),
            reward_mode=str(cfg.get("reward_mode", "old")),
            near_miss_dist=float(cfg.get("near_miss_dist", 24.0)),
        )
        source = f"live:{model_path}"

    ok, messages = check_thresholds(summary, success_floor, collision_ceiling)

    print("=== Regression gate ===")
    print(f"source:             {source}")
    print(f"scenario:           {summary.get('scenario', cfg.get('scenario'))}")
    print(f"reward_mode:        {summary.get('reward_mode', cfg.get('reward_mode'))}")
    print(f"episodes:           {summary.get('episodes', 'n/a')}")
    print(f"success_rate:       {float(summary['success_rate']):.3f}")
    print(f"collision_rate:     {float(summary['collision_rate']):.3f}")
    print(f"success_floor:      {success_floor:.3f}")
    print(f"collision_ceiling:  {collision_ceiling:.3f}")
    for line in messages:
        print(f"  - {line}")
    print(f"result:             {'PASS' if ok else 'FAIL'}")

    result = {
        "pass": ok,
        "source": source,
        "success_floor": success_floor,
        "collision_ceiling": collision_ceiling,
        "messages": messages,
        "summary": summary,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
