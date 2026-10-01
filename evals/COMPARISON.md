# Phase 2 eval comparison (N=50 unless noted, seed=0, simple predictor)

Trained on CPU (`torch` no CUDA). Models live under `models/` (gitignored);
reproduce with `train.py` then re-run these commands.

## Headline vs legacy DQN (`baseline` scenario, `reward.mode=old`)

| Model | Success | Collision | Timeout | Mean return | Mean ep len |
|-------|---------|-----------|---------|-------------|-------------|
| `dqn_avoidance_agent5` (DQN+MLP, legacy) | **0.020** | **0.980** | 0.000 | -457.1 | 86.8 |
| `ppo_CnnPolicy_easy_s0/best_model` | 0.340 | 0.660 | 0.000 | -79.5 | 13.5 |
| `ppo_CnnPolicy_easy_s0/final_model` | **0.480** | **0.520** | 0.000 | -118.0 | 25.8 |
| `qrdqn_CnnPolicy_easy_s0/best_model` | **0.480** | **0.520** | 0.000 | -60.6 | 12.6 |

## Train-condition check (`easy`, `reward.mode=new`)

| Model | Success | Collision | Timeout | Mean return |
|-------|---------|-----------|---------|-------------|
| PPO best | **0.700** | 0.300 | 0.000 | 80.2 |
| QR-DQN best | 0.660 | 0.340 | 0.000 | 73.7 |
| QR-DQN final (N=30) | 0.100 | 0.900 | 0.000 | -73.9 |

Prefer **EvalCallback `best_model`** for QR-DQN (final overfit / degraded).
PPO **final** transferred better to `baseline` than its early `best_model`.

## Commands

```bash
python eval_policy.py --model dqn_avoidance_agent5 --algo dqn \
  --scenario baseline --reward-mode old --episodes 50 --seed 0

python eval_policy.py --model models/ppo_CnnPolicy_easy_s0/final_model.zip --algo ppo \
  --scenario baseline --reward-mode old --episodes 50 --seed 0

python eval_policy.py --model models/qrdqn_CnnPolicy_easy_s0/best_model.zip --algo qrdqn \
  --scenario baseline --reward-mode old --episodes 50 --seed 0
```

Raw JSON: files in this directory.
