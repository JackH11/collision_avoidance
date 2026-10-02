# Demo media (Phase 5)

Checked-in samples for README / talks:

| File | Notes |
|------|--------|
| `demo.gif` / `demo.mp4` | Baseline arena, 3× zoom crop |
| `demo_large.gif` / `demo_large.mp4` | **400×400** `large` scenario, best QR-DQN CNN |
| `godot_large.mp4` | Godot 4 replay of the same dump (prettier) |
| `trajectories/large_best.json` | Trajectory dump for Godot |

Regenerate (needs a local Phase 2 zip under `models/`):

```bash
make demo-record
make demo-large
make godot-movie   # Godot 4.3+ + xvfb-run + ffmpeg
```
