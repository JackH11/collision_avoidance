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
# Optional: unit-test extras
pip install -r requirements-dev.txt
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
| `large` | **400×400** arena showcase (8 movers); CNN obs still local 30×30 |

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

## Demo (Phase 5)

**Demo mode** (default): env-driven episodes, HUD with success / collision /
timeout counters, prediction cones on, policy-grid dots off.

**Research mode**: same HUD + policy occupancy grid dots (what the CNN sees).

```bash
# Interactive window
python main.py
python main.py --mode demo --scenario baseline --seed 0
python main.py --mode research --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn

# Best agent on the large 400×400 map
python main.py --scenario large --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn
python main.py --headless --scenario large --record media/demo_large.gif --episodes 5 --fps 30

# Headless clip for README / talks (.gif or .mp4; needs ffmpeg for mp4)
python main.py --headless --record media/demo.gif --episodes 3 --fps 30
make demo-record
```

| Flag | Purpose |
|------|---------|
| `--model` / `--algo` | Checkpoint (auto-picks Phase 2 best under `models/` if present) |
| `--scenario` / `--seed` | `easy` \| `baseline` \| `hard` |
| `--predictor` | `simple` (default) \| `nn_uncertainty` |
| `--mode` | `demo` \| `research` |
| `--zoom` | Visual scale (default **3×**): crop to playfield then enlarge |
| `--record` | Write `.gif` (Pillow) or `.mp4` (ffmpeg) |
| `--headless` | No window (`SDL_VIDEODRIVER=dummy`) |
| `--episodes` / `--max-frames` | Stop conditions (headless defaults to 3 episodes) |
| `--collect-data` | Dump obstacle trajectories (research / predictor data) |

**Keys (windowed):** `G` grid · `P` cones · `N` noise (next episode) · `R` reset · `Space` pause · `+`/`-` zoom · `Esc` quit.

Sample clips are **not** stored in git (large blobs). Generate locally:

```bash
make demo-record    # → media/demo.gif + media/demo.mp4
make demo-large     # → media/demo_large.*
```

### Godot replay (prettier renders)

Python dumps trajectories; a Godot 4 project replays them with trails, soft
disks, and camera follow — without porting the RL env.

```bash
python dump_trajectory.py --scenario large --episodes 5 --seed 0 \
  --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn
cp media/trajectories/large_best.json godot_replay/data/large_best.json
godot --path godot_replay
# or: make godot-movie   # → media/godot_large.mp4 (needs xvfb + Godot 4.3+)
```

See [`godot_replay/README.md`](godot_replay/README.md). Output stays local under
`media/` (gitignored).

### Research vs demo

| | Demo | Research |
|--|------|----------|
| Goal | Show avoidance clearly | Debug policy inputs |
| Grid dots | Off (toggle `G`) | On |
| Cones | On (toggle `P`) | On |
| Data dump | Off | `--collect-data` |
| Metrics | Live HUD + stdout episode lines | Same + eval suite / gate |

## Predictor path (optional TF)

1. Collect trajectories via `main.py` with `SAVE = True` → `data/train_raw.csv`
2. Build lag features: `transforms/clean.py` → `data/train_lag.csv`
3. Train Keras model: `nn/nn.py` → artifacts under `nn/models/` (e.g. `j_10_5.keras`)
4. Set `prediction.backend: nn_uncertainty` in `config.yaml` after installing
   `requirements-predict.txt`

Default `prediction.backend: simple` does **not** import or load TensorFlow.

## Tests & CI (Phase 4)

Headless unit/smoke tests cover env reset/step, obs shape, info keys,
action→velocity, scenario config, lag features, baselines, and gate helpers.
No large model zips required.

```bash
pip install -r requirements-dev.txt
make test
# or:
SDL_VIDEODRIVER=dummy python -m pytest
```

**What CI runs** (`.github/workflows/ci.yml` on push/PR to `main`):

1. `pip install -r requirements.txt -r requirements-dev.txt` (CPU torch)
2. `python -m pytest` — unit/smoke tests
3. `make regression-gate-json` — threshold check on checked-in
   `evals/ppo_final_baseline_old.json` (no model zip)
4. `evals/regression_gate.py --skip-if-missing` — live gate if `models/` exists,
   otherwise SKIP exit 0
5. `make eval-suite-smoke` — short baseline suite (N=5; soft-skips missing zips)

Local mirror of CI:

```bash
make ci
```

Seeded env determinism: same `reset(seed=…)` reproduces first-K obstacle/agent
poses under the simple predictor. TF/SB3 training RNG is not asserted.

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

Layout stays mostly flat (root CLIs) so existing train/eval paths keep working.
Dead stubs / Untitled notebooks are quarantined under `archive/` and
`notebooks/archive/`.

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
| `tests/` | pytest unit/smoke (Phase 4) |
| `Makefile` | `test`, `ci`, `eval-suite`, `regression-gate*` |
| `.github/workflows/ci.yml` | Install deps → tests → JSON gate → suite smoke |
| `model_loader.py` | Shared DQN/PPO/QR-DQN zip loader |
| `main.py` | Phase 5 demo (HUD, modes, headless record) |
| `dump_trajectory.py` | JSON dumps for Godot replay |
| `godot_replay/` | Godot 4.3 viewer (render-only) |
| `media/` | Local demo renders (gitignored) + trajectory sample |
| `model_prediction.py` | Simple + NN prediction backends |
| `config.yaml` | Single source of sim/config knobs |
| `archive/` | Quarantined dead stubs (not imported) |
