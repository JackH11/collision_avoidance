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

Config lives in `config.yaml` (obstacle count, speeds, episode `max_steps`,
prediction backend, etc.). Prefer editing YAML over hardcoding.

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

## Evaluate (numeric baseline)

Roll out a saved agent and print success / collision / timeout rates:

```bash
python eval_policy.py --model dqn_avoidance_agent5 --episodes 20 --seed 0
```

## Demo

```bash
python main.py
```

Loads `dqn_avoidance_agent5` and renders with the **simple** (constant-velocity)
predictor by default. Close the window or wait for the frame limit to exit.

## Predictor path (optional TF)

1. Collect trajectories via `main.py` with `SAVE = True` → `data/train_raw.csv`
2. Build lag features: `transforms/clean.py` → `data/train_lag.csv`
3. Train Keras model: `nn/nn.py` → artifacts under `nn/models/` (e.g. `j_10_5.keras`)
4. Set `prediction.backend: nn` in `config.yaml` (or pass `prediction_backend="nn"`)
   after installing `requirements-predict.txt`

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
