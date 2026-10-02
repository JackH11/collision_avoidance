#!/usr/bin/env python3
"""
Phase 2 training harness (Option A): CNN policies + PPO / QR-DQN.

Config-driven hyperparameters, seeds, VecEnv, EvalCallback, checkpoints,
TensorBoard. Legacy ``train_dqn.py`` (DQN + MlpPolicy) remains as the
baseline-era reference only.

Examples:
  python train.py
  python train.py --algo ppo --scenario easy --timesteps 400000
  python train.py --algo qrdqn --scenario baseline --timesteps 300000 --seed 1
  python train.py --config train_config.yaml --algo ppo --n-envs 8

Continue training from a checkpoint:
  python train.py --algo ppo --resume models/ppo_CnnPolicy_easy_s0/best_model.zip \\
      --timesteps 200000 --run-name ppo_easy_continue
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

try:
    from sb3_contrib import QRDQN
except ImportError as exc:  # pragma: no cover
    QRDQN = None
    _QRDQN_IMPORT_ERROR = exc
else:
    _QRDQN_IMPORT_ERROR = None

from gym_env import MovingAvoidanceEnv
from policies import cnn_policy_kwargs

DEFAULT_CONFIG = Path(__file__).parent / "train_config.yaml"


def load_train_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_vec_env(
    n_envs: int,
    scenario: str,
    reward_mode: str,
    prediction_backend: str,
    seed: int,
    monitor_dir: Optional[Path] = None,
) -> VecEnv:
    """Build a DummyVecEnv of ``MovingAvoidanceEnv`` (one factory per rank)."""
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)

    def factory(rank: int):
        def _init():
            env = MovingAvoidanceEnv(
                scenario=scenario,
                reward_mode=reward_mode,
                prediction_backend=prediction_backend,
            )
            log = str(monitor_dir / f"monitor_{rank}") if monitor_dir else None
            return Monitor(env, filename=log)

        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


def make_model(algo: str, policy: str, env: VecEnv, cfg: Dict[str, Any], seed: int):
    tb_log = cfg.get("tensorboard_dir", "./tb_logs")
    features_dim = int(cfg.get("features_dim", 128))
    policy_kwargs = {}
    if policy == "CnnPolicy":
        policy_kwargs = cnn_policy_kwargs(features_dim=features_dim)

    if algo == "ppo":
        h = cfg.get("ppo", {})
        return PPO(
            policy=policy,
            env=env,
            learning_rate=float(h.get("learning_rate", 3e-4)),
            n_steps=int(h.get("n_steps", 1024)),
            batch_size=int(h.get("batch_size", 256)),
            n_epochs=int(h.get("n_epochs", 10)),
            gamma=float(h.get("gamma", 0.99)),
            gae_lambda=float(h.get("gae_lambda", 0.95)),
            clip_range=float(h.get("clip_range", 0.2)),
            ent_coef=float(h.get("ent_coef", 0.01)),
            vf_coef=float(h.get("vf_coef", 0.5)),
            max_grad_norm=float(h.get("max_grad_norm", 0.5)),
            policy_kwargs=policy_kwargs or None,
            tensorboard_log=tb_log,
            verbose=1,
            seed=seed,
        )

    if algo == "qrdqn":
        if QRDQN is None:
            raise ImportError(
                "sb3-contrib is required for QR-DQN. "
                f"Install with: pip install 'sb3-contrib>=2.3'. ({_QRDQN_IMPORT_ERROR})"
            )
        h = cfg.get("qrdqn", {})
        return QRDQN(
            policy=policy,
            env=env,
            learning_rate=float(h.get("learning_rate", 5e-4)),
            buffer_size=int(h.get("buffer_size", 100_000)),
            learning_starts=int(h.get("learning_starts", 5_000)),
            batch_size=int(h.get("batch_size", 64)),
            gamma=float(h.get("gamma", 0.99)),
            train_freq=int(h.get("train_freq", 4)),
            gradient_steps=int(h.get("gradient_steps", 1)),
            target_update_interval=int(h.get("target_update_interval", 1_000)),
            exploration_fraction=float(h.get("exploration_fraction", 0.3)),
            exploration_initial_eps=float(h.get("exploration_initial_eps", 1.0)),
            exploration_final_eps=float(h.get("exploration_final_eps", 0.05)),
            policy_kwargs={
                **policy_kwargs,
                "n_quantiles": int(h.get("n_quantiles", 200)),
            }
            if policy_kwargs
            else {"n_quantiles": int(h.get("n_quantiles", 200))},
            tensorboard_log=tb_log,
            verbose=1,
            seed=seed,
        )

    if algo == "dqn":
        # Ablation / baseline-era reference — prefer MlpPolicy
        h = cfg.get("dqn", {})
        return DQN(
            policy=policy,
            env=env,
            learning_rate=float(h.get("learning_rate", 1e-3)),
            buffer_size=int(h.get("buffer_size", 100_000)),
            learning_starts=int(h.get("learning_starts", 1_000)),
            batch_size=int(h.get("batch_size", 64)),
            gamma=float(h.get("gamma", 0.99)),
            train_freq=int(h.get("train_freq", 4)),
            target_update_interval=int(h.get("target_update_interval", 100)),
            exploration_fraction=float(h.get("exploration_fraction", 0.5)),
            exploration_initial_eps=float(h.get("exploration_initial_eps", 1.0)),
            exploration_final_eps=float(h.get("exploration_final_eps", 0.05)),
            policy_kwargs=policy_kwargs or None,
            tensorboard_log=tb_log,
            verbose=1,
            seed=seed,
        )

    raise ValueError(f"Unknown algo '{algo}'. Use ppo | qrdqn | dqn.")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 CNN training (PPO / QR-DQN)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="train_config.yaml path")
    p.add_argument("--algo", choices=["ppo", "qrdqn", "dqn"], default=None)
    p.add_argument("--policy", choices=["CnnPolicy", "MlpPolicy"], default=None)
    p.add_argument("--scenario", choices=["easy", "baseline", "hard"], default=None)
    p.add_argument("--reward-mode", choices=["old", "new"], default=None)
    p.add_argument("--prediction-backend", choices=["simple", "nn", "nn_uncertainty"], default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="Path to zip to continue training from")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (2k steps, 2 envs) for wiring checks",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_train_config(Path(args.config))

    algo = (args.algo or cfg.get("algo", "ppo")).lower()
    policy = args.policy or cfg.get("policy", "CnnPolicy")
    scenario = args.scenario or cfg.get("scenario", "easy")
    reward_mode = args.reward_mode or cfg.get("reward_mode", "new")
    prediction_backend = args.prediction_backend or cfg.get(
        "prediction_backend", "simple"
    )
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    n_envs = int(args.n_envs if args.n_envs is not None else cfg.get("n_envs", 8))
    total_timesteps = int(
        args.timesteps if args.timesteps is not None else cfg.get("total_timesteps", 400_000)
    )

    if args.smoke:
        n_envs = min(n_envs, 2)
        total_timesteps = min(total_timesteps, 2_000)
        cfg = deepcopy(cfg)
        cfg["eval_freq"] = 1_000
        cfg["checkpoint_freq"] = 1_000
        cfg["n_eval_episodes"] = 2

    # DQN ablation defaults to MLP unless explicitly overridden
    if algo == "dqn" and args.policy is None and cfg.get("policy") == "CnnPolicy":
        # Allow CnnPolicy DQN if user set it; otherwise prefer Mlp for ablation
        if not args.algo:
            pass
        # Keep config as-is — user can set policy: MlpPolicy in YAML for ablation

    run_name = args.run_name or cfg.get("run_name")
    if not run_name:
        run_name = f"{algo}_{policy}_{scenario}_s{seed}"

    run_root = Path(cfg.get("log_dir", "./runs")) / run_name
    ckpt_dir = Path(cfg.get("checkpoint_dir", "./checkpoints")) / run_name
    best_dir = Path(cfg.get("best_model_dir", "./models")) / run_name
    monitor_dir = run_root / "monitor"
    for d in (run_root, ckpt_dir, best_dir, Path(cfg.get("tensorboard_dir", "./tb_logs"))):
        d.mkdir(parents=True, exist_ok=True)

    meta = {
        "algo": algo,
        "policy": policy,
        "scenario": scenario,
        "reward_mode": reward_mode,
        "prediction_backend": prediction_backend,
        "seed": seed,
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "run_name": run_name,
        "features_dim": cfg.get("features_dim", 128),
        "hparams": {
            "ppo": cfg.get("ppo"),
            "qrdqn": cfg.get("qrdqn"),
            "dqn": cfg.get("dqn"),
        }.get(algo),
    }
    (run_root / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("=== Phase 2 train ===")
    for k, v in meta.items():
        if k != "hparams":
            print(f"  {k}: {v}")
    print(f"  hparams: {meta['hparams']}")
    print(f"  artifacts: run={run_root} ckpt={ckpt_dir} best={best_dir}")

    train_env = build_vec_env(
        n_envs=n_envs,
        scenario=scenario,
        reward_mode=reward_mode,
        prediction_backend=prediction_backend,
        seed=seed,
        monitor_dir=monitor_dir,
    )
    eval_env = build_vec_env(
        n_envs=1,
        scenario=scenario,
        reward_mode=reward_mode,
        prediction_backend=prediction_backend,
        seed=seed + 10_000,
        monitor_dir=run_root / "eval_monitor",
    )

    if args.resume:
        from model_loader import load_model

        model, loaded_algo, path = load_model(args.resume, algo=algo, env=train_env)
        print(f"Resumed {loaded_algo} from {path}")
        algo = loaded_algo
    else:
        model = make_model(algo, policy, train_env, cfg, seed)

    # EvalCallback save_freq is counted in calls; SB3 divides by n_envs for vec envs
    eval_freq = max(1, int(cfg.get("eval_freq", 10_000)) // n_envs)
    checkpoint_freq = max(1, int(cfg.get("checkpoint_freq", 50_000)) // n_envs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(run_root / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=int(cfg.get("n_eval_episodes", 10)),
        deterministic=bool(cfg.get("deterministic_eval", True)),
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(ckpt_dir),
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = CallbackList([eval_cb, ckpt_cb])

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=False,
        reset_num_timesteps=args.resume is None,
    )
    elapsed = time.perf_counter() - t0

    final_path = best_dir / "final_model"
    model.save(str(final_path))
    # Convenience copy at models/<run_name>.zip pointing at best if present
    best_zip = best_dir / "best_model.zip"
    latest = best_zip if best_zip.exists() else Path(str(final_path) + ".zip")
    summary = {
        **meta,
        "elapsed_sec": elapsed,
        "steps_per_sec": total_timesteps / elapsed if elapsed > 0 else None,
        "best_model": str(best_zip) if best_zip.exists() else None,
        "final_model": str(final_path) + ".zip",
        "continue_cmd": (
            f"python train.py --algo {algo} --resume {latest} "
            f"--timesteps {total_timesteps} --scenario {scenario} "
            f"--reward-mode {reward_mode} --n-envs {n_envs} --seed {seed} "
            f"--run-name {run_name}_cont"
        ),
    }
    (run_root / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("=== Training done ===")
    print(f"  elapsed_sec:     {elapsed:.1f}")
    print(f"  steps_per_sec:   {summary['steps_per_sec']:.1f}" if summary["steps_per_sec"] else "")
    print(f"  best_model:      {summary['best_model']}")
    print(f"  final_model:     {summary['final_model']}")
    print(f"  continue:        {summary['continue_cmd']}")

    train_env.close()
    eval_env.close()
    return summary


if __name__ == "__main__":
    main()
