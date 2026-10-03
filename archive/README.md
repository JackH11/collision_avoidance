# Archive / quarantine

Dead or superseded snippets kept out of the active import path.

| Path | Why |
|------|-----|
| `agent_stub.py` | Pre-`gym_env` ego stub (`perceive` / `predict` / `plan`); unused by train/demo/eval |
| `train_dqn.py` | Baseline-era DQN+`MlpPolicy` one-off; superseded by `python -m collision_avoidance.train` / `train.py` |

Active agent logic lives in `collision_avoidance.env` (`MovingAgent` / `MovingAvoidanceEnv`).
Untitled notebooks are under `notebooks/archive/`.

Root scripts (`main.py`, `train.py`, …) are thin shims into the package.
