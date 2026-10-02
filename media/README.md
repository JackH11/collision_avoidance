# Demo media (Phase 5)

Checked-in samples for README / talks:

| File | Notes |
|------|--------|
| `demo.gif` / `demo.mp4` | Baseline arena, 3× zoom crop |
| `demo_large.gif` / `demo_large.mp4` | **400×400** `large` scenario, best QR-DQN CNN |

Regenerate (needs a local Phase 2 zip under `models/`):

```bash
make demo-record
make demo-large
# or:
python main.py --headless --scenario large --record media/demo_large.gif \
  --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn \
  --episodes 5 --fps 30
```
