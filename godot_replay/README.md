# Godot trajectory replay (Phase 5)

Python stays the **sim + policy** brain. Godot only **renders** JSON dumps from
`dump_trajectory.py` — nicer lighting, trails, fixed full-board camera, prediction cones.

Default render is **1920×1920** with a **fixed full-board camera** (`--overview`).
Optional follow crops: `--frame agent_goal` or `--frame agent`.

## Quick start

```bash
# 1) Dump the best agent on the large map (needs a models/*.zip locally)
python -m collision_avoidance.demo.dump_trajectory --scenario large --episodes 5 --seed 0 \
  --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn \
  --out media/trajectories/large_best.json

# Copy (or symlink) into the Godot project data folder
mkdir -p godot_replay/data
cp media/trajectories/large_best.json godot_replay/data/large_best.json

# 2) Open / play in Godot 4.3+
godot --path godot_replay
# or open godot_replay/project.godot in the Godot editor
```

### CLI overrides

```bash
godot --path godot_replay -- --trajectory res://data/large_best.json
godot --path godot_replay -- --trajectory /abs/path/to/run.json --speed 0.85
# Fixed full board (default) vs follow crops; showcase filters:
godot --path godot_replay -- --overview --outcomes success --min-frames 60
godot --path godot_replay -- --frame agent_goal --min-view-span 280
godot --path godot_replay -- --frame agent --view-radius 160
```

### Record a video (Godot Movie Writer)

```bash
# Needs a display (use xvfb-run on headless CI/servers)
# Prefer make godot-movie (CRF 17 H.264). Manual:
xvfb-run -a godot --path godot_replay --write-movie ../media/godot_large.avi \
  --fixed-fps 30 -- \
  --trajectory res://data/large_best.json --quit-when-done \
  --overview --min-frames 60 --outcomes success --speed 0.8
ffmpeg -y -i media/godot_large.avi \
  -c:v libx264 -crf 17 -preset slow -pix_fmt yuv420p -movflags +faststart \
  media/godot_large.mp4
```

## Controls

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| N | Next episode |

## Layout

| Path | Role |
|------|------|
| `project.godot` | Godot 4.3 project (1920² + 2D MSAA) |
| `scenes/main.tscn` | Camera + HUD + replay root |
| `scripts/replay.gd` | JSON loader / playback / draw |
| `data/large_best.json` | Sample dump (generated; may be gitignored if huge) |

JSON schema version `1` — see `dump_trajectory.py` for fields (`meta`, `episodes[].frames[]` with agent / goal / obstacles / predictions).
