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

Roll out a saved agent and print success / collision / timeout rates.
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
~2% success vs PPO final / QR-DQN best ~48% success.

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
| `eval_policy.py` | Offline success/collision/timeout eval |
| `model_loader.py` | Shared DQN/PPO/QR-DQN zip loader |
| `main.py` | Interactive Pygame demo |
| `model_prediction.py` | Simple + NN prediction backends |
| `config.yaml` | Single source of sim/config knobs |
