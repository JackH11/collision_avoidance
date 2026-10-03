# Demo media (Phase 5)

GIF/MP4 renders are **not** committed (gitignored). Generate locally or attach as
Cloud Agent / talk artifacts when you need a clip.

| File (local) | Notes |
|--------------|--------|
| `demo.gif` / `demo.mp4` | Baseline arena, 3× zoom crop |
| `demo_large.gif` / `demo_large.mp4` | **400×400** `large` scenario, best QR-DQN CNN |
| `godot_large.mp4` | Godot 4 replay of the same dump (prettier) |
| `trajectories/large_best.json` | Trajectory dump for Godot (checked in as sample) |

Regenerate (needs a local Phase 2 zip under `models/`):

```bash
make demo-record
make demo-large
make godot-movie   # Godot 4.3+ + xvfb-run + ffmpeg
```
