# Collision Avoidance

Pygame + Gymnasium environment where an agent reaches a goal while avoiding
noisy moving obstacles. Obstacle motion is encoded as a local occupancy /
prediction grid; a **CNN** policy (PPO or QR-DQN) acts in discrete 8 directions.

Restore point: Git tag **`baseline`** @ `976b3c5` (`git checkout baseline`).
Do not move or delete that tag.

## Requirements

- Python **3.10 or 3.11** recommended (3.12 often works for train/demo)
- Core: Gymnasium, Stable-Baselines3, **sb3-contrib** (QR-DQN), PyTorch, Pygame,
  NumPy, PyYAML, pandas, scikit-learn, TensorBoard
- **Optional:** TensorFlow — only for the Keras NN trajectory predictor

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional NN predictor extras:

```bash
pip install -r requirements-predict.txt
```

Config lives in `config.yaml` (scenarios, reward mode, prediction backend) and
`train_config.yaml` (algo / hparams / seeds / n_envs). Prefer editing YAML over
hardcoding.

### Scenarios

Set `scenario:` in `config.yaml`, or pass `--scenario` / `MovingAvoidanceEnv(scenario=...)`:

| Name | Intent |
|------|--------|
| `easy` | 3 slower obstacles, no velocity noise, longer episodes (default train start) |
| `baseline` | Historical defaults (5 noisy movers) — use for fair compares |
| `hard` | 8 faster noisy movers, shorter episodes |

### Reward A/B

`reward.mode: old` keeps `Δgoal_distance − 5` for compatibility with
`dqn_avoidance_agent5`. `reward.mode: new` (default for Phase 2 training) uses
scaled progress + small step cost + optional near-miss penalty.

### Prediction backends

`prediction.backend: simple` | `nn_uncertainty` (`nn` alias). Same return
shape `(pred_x, pred_y, std_x, std_y)`.

## Train (Phase 2 — CNN + PPO / QR-DQN)

Primary path is **Option A**: spatial grid obs → custom CNN (`GridCnnExtractor`)
→ **PPO** and/or **QR-DQN** (sb3-contrib). Vanilla DQN + `MlpPolicy` is kept only
as the `baseline`-era reference (`train_dqn.py`).

```bash
# Default: PPO + CnnPolicy, scenario=easy, reward=new, 400k steps, 8 envs
python train.py

# QR-DQN + CNN
python train.py --algo qrdqn --scenario easy --timesteps 300000

# Train on baseline scenario
python train.py --algo ppo --scenario baseline --reward-mode new --timesteps 400000

# Smoke wiring check
python train.py --smoke
```

Artifacts (gitignored — do not commit large dumps):

| Path | Contents |
|------|----------|
| `models/<run_name>/best_model.zip` | EvalCallback best checkpoint |
| `models/<run_name>/final_model.zip` | End-of-run save |
| `checkpoints/<run_name>/` | Periodic checkpoints |
| `tb_logs/` | TensorBoard event files |
| `runs/<run_name>/` | Monitor logs, `run_meta.json`, `train_summary.json` |

View TensorBoard:

```bash
tensorboard --logdir ./tb_logs/
```

Continue training from a checkpoint (command also printed at end of `train.py`):

```bash
python train.py --algo ppo --resume models/ppo_CnnPolicy_easy_s0/best_model.zip \
  --timesteps 200000 --run-name ppo_CnnPolicy_easy_s0_cont
```

Legacy short DQN+MLP train (reference only):

```bash
python train_dqn.py
```

## Evaluate

Roll out a saved agent and print success / collision / timeout rates plus
safety metrics (time-to-goal, min clearance, near-miss, prediction-cone risk).
Auto-detects DQN / PPO / QR-DQN from the zip (or pass `--algo`):

```bash
# Legacy baseline agent (Phase 1 numbers: ~2% success on baseline/old)
python eval_policy.py --model dqn_avoidance_agent5 --episodes 50 --seed 0

# Phase 2 CNN checkpoint
python eval_policy.py --model models/ppo_CnnPolicy_easy_s0/final_model.zip \
  --algo ppo --scenario baseline --reward-mode old --episodes 50 --seed 0

python eval_policy.py --scenario hard --reward-mode old --episodes 20
```

Phase 2 CPU comparison table (N=50, seed=0, `baseline`/`old`, simple predictor)
is checked in under [`evals/COMPARISON.md`](evals/COMPARISON.md): legacy DQN
~2% success vs continued QR-DQN CNN **~74%** success / **~26%** collision
(PPO cont ~60%). Demo prefers `models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip`
when present.

### Formal eval suite & regression gate (Phase 3)

One command runs **easy / baseline / hard** (and optional seeds) for any local
Phase 2 checkpoints **plus** scripted baselines (`random`, `greedy`, `freeze`).
Writes JSON / CSV / Markdown under `evals/artifacts/`.

```bash
# Full suite (soft-skips missing models/*.zip — no multi-hour retrain required)
make eval-suite
# or:
python evals/run_suite.py --episodes 20 --seeds 0 --scenarios easy,baseline,hard

# Quick smoke
make eval-suite-smoke
```

**Regression gate** (baseline scenario): fails if `success_rate < 0.30` or
`collision_rate > 0.70`. Thresholds are documented in
[`evals/THRESHOLDS.md`](evals/THRESHOLDS.md) / [`evals/gate_config.yaml`](evals/gate_config.yaml)
and leave headroom under the measured ~48% / ~52% Phase 2 CNN numbers while
rejecting a collapse toward legacy DQN (~2% / ~98%).

```bash
# Live gate — needs a Phase 2 zip under models/ (exit 1 = metric fail, 2 = missing)
make regression-gate
python evals/regression_gate.py --model models/ppo_CnnPolicy_easy_s0/final_model.zip

# No zip required — validate gate math on checked-in Phase 2 summary (should PASS)
make regression-gate-json
```

If `models/` is empty, train first (artifacts are gitignored):

```bash
python train.py --algo ppo --scenario easy --reward-mode new --timesteps 400000
python train.py --algo qrdqn --scenario easy --reward-mode new --timesteps 300000
```

**Interpreting the gate:** PASS means the candidate stays in the Phase 2 CNN
performance band on `baseline`/`old`. FAIL means success dropped below the
floor or collisions exceeded the ceiling — do not merge training changes until
fixed or thresholds are deliberately revised with evidence.

## Demo

```bash
python main.py
python main.py --model models/ppo_CnnPolicy_easy_s0/best_model.zip --algo ppo
python main.py --scenario baseline --model dqn_avoidance_agent5 --algo dqn
```

**Canonical interactive renderer.** Prefers a Phase 2 `models/.../best_model.zip`
when present, else loads `dqn_avoidance_agent5`. Draws the policy grid with the
**simple** predictor. `env.render()` is intentionally a no-op stub.

## Predictor path (optional TF)

1. Collect trajectories via `main.py` with `SAVE = True` → `data/train_raw.csv`
2. Build lag features: `transforms/clean.py` → `data/train_lag.csv`
3. Train Keras model: `nn/nn.py` → artifacts under `nn/models/` (e.g. `j_10_5.keras`)
4. Set `prediction.backend: nn_uncertainty` in `config.yaml` after installing
   `requirements-predict.txt`

Default `prediction.backend: simple` does **not** import or load TensorFlow.

## Artifacts

Large training artifacts are gitignored (`agents/*`, `dqn_tensorboard/*`,
`checkpoints/*`, `tb_logs/*`, `runs/*`, `models/*`). Keep local zips /
TensorBoard runs out of git; reproduce with `train.py` or evaluate the
checked-in `dqn_avoidance_agent5.zip` when present.

## Restore baseline

```bash
git checkout baseline
```

## Project layout (high level)

| Path | Role |
|------|------|
| `gym_env.py` | `MovingAvoidanceEnv` (Gymnasium), movers, grid obs |
| `train.py` | Phase 2 config-driven CNN training (PPO / QR-DQN) |
| `train_config.yaml` | Training hyperparameters / seeds / paths |
| `policies.py` | Custom `GridCnnExtractor` for `(C,H,W)` float grids |
| `train_dqn.py` | Legacy SB3 DQN + MlpPolicy (baseline reference) |
| `eval_policy.py` | Offline eval + safety metrics |
| `evals/run_suite.py` | Multi-scenario suite → `evals/artifacts/` |
| `evals/regression_gate.py` | Success/collision regression gate |
| `evals/baselines.py` | random / greedy / freeze controls |
| `Makefile` | `eval-suite`, `regression-gate` targets |
| `model_loader.py` | Shared DQN/PPO/QR-DQN zip loader |
| `main.py` | Interactive Pygame demo |
| `model_prediction.py` | Simple + NN prediction backends |
| `config.yaml` | Single source of sim/config knobs |
