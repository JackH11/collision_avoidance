# Godot trajectory replay (Phase 5)

Python stays the **sim + policy** brain. Godot only **renders** JSON dumps from
`dump_trajectory.py` — nicer lighting, trails, camera follow, prediction cones.

## Quick start

```bash
# 1) Dump the best agent on the large map (needs a models/*.zip locally)
python dump_trajectory.py --scenario large --episodes 5 --seed 0 \
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
godot --path godot_replay -- --trajectory /abs/path/to/run.json --speed 1.5
```

### Record a video (Godot Movie Writer)

```bash
# Needs a display (use xvfb-run on headless CI/servers)
xvfb-run -a godot --path godot_replay --write-movie ../media/godot_large.avi \
  -- --trajectory res://data/large_best.json --quit-when-done --speed 1.0
# Then convert if desired:
ffmpeg -y -i media/godot_large.avi -c:v libx264 -pix_fmt yuv420p media/godot_large.mp4
```

## Controls

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| N | Next episode |

## Layout

| Path | Role |
|------|------|
| `project.godot` | Godot 4.3 project |
| `scenes/main.tscn` | Camera + HUD + replay root |
| `scripts/replay.gd` | JSON loader / playback / draw |
| `data/large_best.json` | Sample dump (generated; may be gitignored if huge) |

JSON schema version `1` — see `dump_trajectory.py` for fields (`meta`, `episodes[].frames[]` with agent / goal / obstacles / predictions).
