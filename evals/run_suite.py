#!/usr/bin/env python3
"""
Formal evaluation suite across scenarios and seeds (Phase 3).

Runs learned checkpoints (when present) plus scripted baselines, writes
JSON + CSV under ``evals/artifacts/``, and prints a comparison table.

Examples:
  python evals/run_suite.py
  python evals/run_suite.py --episodes 20 --seeds 0,1 --scenarios easy,baseline,hard
  python evals/run_suite.py --baselines-only
  make eval-suite
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collision_avoidance.eval.eval_policy import evaluate, print_summary  # noqa: E402
from evals.baselines import BASELINE_FACTORIES  # noqa: E402

DEFAULT_MODELS = [
    {
        "name": "dqn_legacy",
        "path": "dqn_avoidance_agent5.zip",
        "algo": "dqn",
        "reward_mode": "old",
    },
    {
        "name": "ppo_final",
        "path": "models/ppo_CnnPolicy_easy_s0/final_model.zip",
        "algo": "ppo",
        "reward_mode": "old",
    },
    {
        "name": "ppo_best",
        "path": "models/ppo_CnnPolicy_easy_s0/best_model.zip",
        "algo": "ppo",
        "reward_mode": "old",
    },
    {
        "name": "qrdqn_best",
        "path": "models/qrdqn_CnnPolicy_easy_s0/best_model.zip",
        "algo": "qrdqn",
        "reward_mode": "old",
    },
]

TABLE_COLS = [
    "name",
    "scenario",
    "seed",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_time_to_goal",
    "mean_min_obstacle_dist",
    "worst_min_obstacle_dist",
    "near_miss_rate",
    "mean_cone_risk_step_fraction",
    "mean_return",
    "episodes",
]


def _parse_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return f"{v:.3f}"
    return str(v)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = {
        c: max(len(c), max(len(_fmt(r.get(c, ""))) for r in rows)) for c in TABLE_COLS
    }
    header = " | ".join(c.ljust(widths[c]) for c in TABLE_COLS)
    sep = "-+-".join("-" * widths[c] for c in TABLE_COLS)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in TABLE_COLS))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in TABLE_COLS})


def write_markdown(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eval suite results",
        "",
        f"Generated: `{meta.get('generated_at')}`",
        f"Episodes/seed: **{meta.get('episodes')}**; "
        f"scenarios: `{meta.get('scenarios')}`; seeds: `{meta.get('seeds')}`; "
        f"predictor: `{meta.get('prediction_backend')}`",
        "",
        "| " + " | ".join(TABLE_COLS) + " |",
        "| " + " | ".join("---" for _ in TABLE_COLS) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in TABLE_COLS) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run_learned(
    models: Iterable[dict[str, Any]],
    scenarios: list[str],
    seeds: list[int],
    episodes: int,
    prediction_backend: str,
    skip_missing: bool,
    near_miss_dist: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for spec in models:
        path = Path(spec["path"])
        if not path.exists():
            msg = f"{spec['name']}: missing {path}"
            skipped.append(msg)
            if skip_missing:
                print(f"SKIP {msg}")
                continue
            raise FileNotFoundError(
                f"{msg}. Train with train.py or pass --skip-missing."
            )
        for scenario in scenarios:
            for seed in seeds:
                print(
                    f"\n>>> {spec['name']} | {scenario} | seed={seed} | "
                    f"N={episodes} | {path}"
                )
                summary = evaluate(
                    model_path=str(path),
                    episodes=episodes,
                    seed=seed,
                    deterministic=True,
                    prediction_backend=prediction_backend,
                    scenario=scenario,
                    reward_mode=str(spec.get("reward_mode", "old")),
                    algo=spec.get("algo"),
                    policy_name=spec["name"],
                    near_miss_dist=near_miss_dist,
                )
                print_summary(summary)
                row = {
                    "name": spec["name"],
                    "path": str(path),
                    **summary,
                }
                rows.append(row)
    return rows, skipped


def run_baselines(
    names: list[str],
    scenarios: list[str],
    seeds: list[int],
    episodes: int,
    prediction_backend: str,
    reward_mode: str,
    near_miss_dist: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        if name not in BASELINE_FACTORIES:
            raise ValueError(f"Unknown baseline '{name}'. Known: {sorted(BASELINE_FACTORIES)}")
        factory = BASELINE_FACTORIES[name]
        for scenario in scenarios:
            for seed in seeds:
                print(
                    f"\n>>> baseline:{name} | {scenario} | seed={seed} | N={episodes}"
                )
                predict_fn = factory(seed=seed)
                summary = evaluate(
                    model_path=None,
                    episodes=episodes,
                    seed=seed,
                    deterministic=True,
                    prediction_backend=prediction_backend,
                    scenario=scenario,
                    reward_mode=reward_mode,
                    predict_fn=predict_fn,
                    policy_name=f"baseline_{name}",
                    near_miss_dist=near_miss_dist,
                )
                print_summary(summary)
                rows.append(
                    {
                        "name": f"baseline_{name}",
                        "path": None,
                        **summary,
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal multi-scenario eval suite")
    parser.add_argument(
        "--scenarios",
        default="easy,baseline,hard",
        help="Comma-separated scenario names",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated integer seeds (episode i uses seed+i internally)",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--prediction-backend",
        default="simple",
        choices=["simple", "nn", "nn_uncertainty"],
    )
    parser.add_argument(
        "--reward-mode",
        default="old",
        choices=["old", "new"],
        help="Reward mode for baselines (learned models use their spec default)",
    )
    parser.add_argument(
        "--near-miss-dist",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip scripted baselines",
    )
    parser.add_argument(
        "--baselines",
        default="random,greedy,freeze",
        help="Comma-separated baseline names",
    )
    parser.add_argument(
        "--require-models",
        action="store_true",
        help="Fail if a configured learned model zip is missing (default: skip)",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only evaluate learned models (no baselines)",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Only evaluate scripted baselines",
    )
    parser.add_argument(
        "--out-dir",
        default="evals/artifacts",
        help="Directory for JSON/CSV/Markdown outputs",
    )
    args = parser.parse_args()

    scenarios = _parse_list(args.scenarios)
    seeds = _parse_ints(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    if not args.baselines_only:
        learned, skipped = run_learned(
            DEFAULT_MODELS,
            scenarios=scenarios,
            seeds=seeds,
            episodes=args.episodes,
            prediction_backend=args.prediction_backend,
            # Soft-skip missing Phase 2 zips by default (artifacts are gitignored).
            skip_missing=not args.require_models,
            near_miss_dist=args.near_miss_dist,
        )
        rows.extend(learned)

    include_baselines = (not args.models_only) and (not args.no_baselines)
    if args.baselines_only:
        include_baselines = True
    if include_baselines:
        rows.extend(
            run_baselines(
                names=_parse_list(args.baselines),
                scenarios=scenarios,
                seeds=seeds,
                episodes=args.episodes,
                prediction_backend=args.prediction_backend,
                reward_mode=args.reward_mode,
                near_miss_dist=args.near_miss_dist,
            )
        )

    meta = {
        "generated_at": stamp,
        "episodes": args.episodes,
        "scenarios": scenarios,
        "seeds": seeds,
        "prediction_backend": args.prediction_backend,
        "skipped_models": skipped,
    }

    table_rows = [{c: r.get(c) for c in TABLE_COLS} for r in rows]
    print("\n=== Suite comparison table ===")
    print_table(table_rows)

    json_path = out_dir / f"suite_{stamp}.json"
    csv_path = out_dir / f"suite_{stamp}.csv"
    md_path = out_dir / f"suite_{stamp}.md"
    latest_json = out_dir / "suite_latest.json"
    latest_csv = out_dir / "suite_latest.csv"
    latest_md = out_dir / "suite_latest.md"

    payload = {"meta": meta, "results": rows}
    json_path.write_text(json.dumps(payload, indent=2))
    latest_json.write_text(json.dumps(payload, indent=2))
    write_csv(csv_path, table_rows)
    write_csv(latest_csv, table_rows)
    write_markdown(md_path, table_rows, meta)
    write_markdown(latest_md, table_rows, meta)

    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"latest copies: {latest_json}, {latest_csv}, {latest_md}")
    if skipped:
        print("skipped models:")
        for s in skipped:
            print(f"  - {s}")
        print(
            "Train Phase 2 checkpoints with:\n"
            "  python train.py --algo ppo --scenario easy --timesteps 400000\n"
            "  python train.py --algo qrdqn --scenario easy --timesteps 300000"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
