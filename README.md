# Collision Avoidance

Pygame + Gymnasium environment where an agent reaches a goal while avoiding
noisy moving obstacles. Obstacle motion is encoded as a local occupancy /
prediction grid; a Stable-Baselines3 DQN policy acts in discrete 8 directions.

Restore point: Git tag **`baseline`** @ `976b3c5` (`git checkout baseline`).
Do not move or delete that tag.

## Requirements

- Python **3.10 or 3.11** recommended (3.12 often works for train/demo)
- Core: Gymnasium, Stable-Baselines3, PyTorch, Pygame, NumPy, PyYAML, pandas, scikit-learn
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

Config lives in `config.yaml` (scenarios, reward mode, prediction backend, etc.).
Prefer editing YAML over hardcoding.

### Scenarios

Set `scenario:` in `config.yaml`, or pass `--scenario` / `MovingAvoidanceEnv(scenario=...)`:

| Name | Intent |
|------|--------|
| `easy` | 3 slower obstacles, no velocity noise, longer episodes |
| `baseline` | Historical defaults (5 noisy movers) — use for fair compares |
| `hard` | 8 faster noisy movers, shorter episodes |

### Reward A/B

`reward.mode: old` (default) keeps `Δgoal_distance − 5` for compatibility with
`dqn_avoidance_agent5`. `reward.mode: new` uses scaled progress + small step
cost + optional near-miss penalty (`progress_scale`, `step_cost`,
`near_miss_dist`, `near_miss_penalty` in YAML).

### Prediction backends

`prediction.backend: simple` | `nn_uncertainty` (`nn` alias). Same return
shape `(pred_x, pred_y, std_x, std_y)`. Lag/window for the Keras path come from
`prediction.lag` / `prediction.window` (defaults aligned to `j_10_5`).

## Train (DQN)

```bash
python train_dqn.py
```

Default run: 20k timesteps, `MlpPolicy`, TensorBoard under `./dqn_tensorboard/`,
checkpoints under `./checkpoints/`, saves `dqn_avoidance_agent5.zip`.

View TensorBoard:

```bash
tensorboard --logdir ./dqn_tensorboard/
```

## Evaluate

Roll out a saved agent and print success / collision / timeout rates:

```bash
python eval_policy.py --model dqn_avoidance_agent5 --episodes 50 --seed 0
python eval_policy.py --scenario hard --reward-mode old --episodes 20
```

## Demo

```bash
python main.py
```

**Canonical interactive renderer.** Loads `dqn_avoidance_agent5` and draws the
policy grid with the **simple** predictor. `env.render()` is intentionally a
no-op stub — do not rely on Gymnasium human mode for visuals.

## Predictor path (optional TF)

1. Collect trajectories via `main.py` with `SAVE = True` → `data/train_raw.csv`
2. Build lag features: `transforms/clean.py` → `data/train_lag.csv`
3. Train Keras model: `nn/nn.py` → artifacts under `nn/models/` (e.g. `j_10_5.keras`)
4. Set `prediction.backend: nn_uncertainty` in `config.yaml` after installing
   `requirements-predict.txt`

Default `prediction.backend: simple` does **not** import or load TensorFlow.

## Artifacts

Large training artifacts are gitignored (`agents/*`, `dqn_tensorboard/*`,
`checkpoints/*`). Keep local zips / TensorBoard runs out of git; reproduce with
`train_dqn.py` or evaluate the checked-in `dqn_avoidance_agent5.zip` when present.

## Restore baseline

```bash
git checkout baseline
```

## Project layout (high level)

| Path | Role |
|------|------|
| `gym_env.py` | `MovingAvoidanceEnv` (Gymnasium), movers, grid obs |
| `train_dqn.py` | SB3 DQN training |
| `eval_policy.py` | Offline success/collision/timeout eval |
| `main.py` | Interactive Pygame demo |
| `model_prediction.py` | Simple + NN prediction backends |
| `config.yaml` | Single source of sim/config knobs |
