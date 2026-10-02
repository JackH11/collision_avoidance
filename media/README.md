# Demo media (Phase 5)

Checked-in samples for README / talks:

| File | Notes |
|------|--------|
| `demo.gif` | Short headless capture (`baseline`, QR-DQN CNN when available) |
| `demo.mp4` | Same pipeline, H.264 |

Regenerate (needs a local Phase 2 zip under `models/` or legacy `dqn_avoidance_agent5.zip`):

```bash
python main.py --headless --record media/demo.gif --scenario baseline --episodes 3 --fps 30
python main.py --headless --record media/demo.mp4 --scenario baseline --episodes 3 --fps 30
```
