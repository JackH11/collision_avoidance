# Phase 2 eval comparison (seed=0, simple predictor)

Trained on CPU (`torch` no CUDA). Models live under `models/` (gitignored);
reproduce with `train.py` then re-run these commands.

## Headline vs legacy DQN (`baseline` scenario, `reward.mode=old`, N=50)

| Model | Steps (approx) | Success | Collision | Timeout | Mean return |
|-------|----------------|---------|-----------|---------|-------------|
| `dqn_avoidance_agent5` (DQN+MLP, legacy) | ~20k | **0.020** | **0.980** | 0.000 | -457.1 |
| PPO CNN `final` (first run) | 400k | 0.480 | 0.520 | 0.000 | -118.0 |
| QR-DQN CNN `best` (first run) | ≤300k | 0.480 | 0.520 | 0.000 | -60.6 |
| PPO CNN cont `final`/`best` | ~800k | 0.600 | 0.400 | 0.000 | ~-90 |
| **QR-DQN CNN cont `best`** | **~330k** | **0.740** | **0.260** | 0.000 | -133.3 |
| QR-DQN CNN cont `final` | ~330k | 0.700 | 0.300 | 0.000 | -147.1 |

**Current recommended demo / checkpoint:** `models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip`

## Train-condition check (`easy`, `reward.mode=new`, N=50)

| Model | Success | Collision | Mean return |
|-------|---------|-----------|-------------|
| PPO best (400k) | 0.700 | 0.300 | 80.2 |
| PPO cont best (~800k) | 0.780 | 0.220 | 94.1 |
| QR-DQN best (first) | 0.660 | 0.340 | 73.7 |
| **QR-DQN cont best** | **0.800** | **0.200** | 94.8 |

Prefer **EvalCallback `best_model`** for QR-DQN when final drifts.

## Commands

```bash
python eval_policy.py --model dqn_avoidance_agent5 --algo dqn \
  --scenario baseline --reward-mode old --episodes 50 --seed 0

python eval_policy.py --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn \
  --scenario baseline --reward-mode old --episodes 50 --seed 0

python train.py --algo qrdqn --resume models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip \
  --timesteps 300000 --run-name qrdqn_CnnPolicy_easy_s0_cont2
```

Raw JSON: files in this directory.
