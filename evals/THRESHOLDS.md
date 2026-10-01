# Phase 3 regression thresholds

Documented floors/ceilings for the **baseline** scenario (`reward.mode=old`,
simple predictor). Source of truth: [`gate_config.yaml`](gate_config.yaml).

## Measured reference (Phase 2, N=50, seed=0)

| Agent | Success | Collision |
|-------|---------|-----------|
| Legacy DQN+MLP (`dqn_avoidance_agent5`) | ~0.02 | ~0.98 |
| PPO CNN final / QR-DQN CNN best | **~0.48** | **~0.52** |

## Gate thresholds

| Knob | Value | Rationale |
|------|-------|-----------|
| `success_floor` | **0.30** | Below measured ~48% to allow seed/N noise; still far above legacy ~2% |
| `collision_ceiling` | **0.70** | Above measured ~52%; rejects collapse toward legacy ~98% collisions |
| Default live episodes | 30 | Faster than N=50; tighten locally with `--episodes 50` for release checks |

## Behavior

```bash
# Live eval of first existing candidate under models/ (exit 1 = fail, 2 = missing)
python evals/regression_gate.py
make regression-gate

# No zip required — gate math on checked-in Phase 2 summary (should PASS)
python evals/regression_gate.py --from-json evals/ppo_final_baseline_old.json
make regression-gate-json

# Soft-skip when artifacts absent (exit 0)
python evals/regression_gate.py --skip-if-missing
```

Fail conditions: `success_rate < success_floor` **or** `collision_rate > collision_ceiling`.

## Obtaining checkpoints

Phase 2 zips are **gitignored** (`models/`). Reproduce:

```bash
python train.py --algo ppo --scenario easy --reward-mode new --timesteps 400000
python train.py --algo qrdqn --scenario easy --reward-mode new --timesteps 300000
```

Then point the gate at a zip:

```bash
python evals/regression_gate.py --model models/ppo_CnnPolicy_easy_s0/final_model.zip
```
