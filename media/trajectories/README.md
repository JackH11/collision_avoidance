# Trajectory dumps for Godot replay

| File | Notes |
|------|--------|
| `large_best.json` | Best QR-DQN on `large` map, 5 episodes (seed 0) |

Regenerate:

```bash
python dump_trajectory.py --scenario large --episodes 5 --seed 0 \
  --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn \
  --out media/trajectories/large_best.json
cp media/trajectories/large_best.json godot_replay/data/large_best.json
```

See `godot_replay/README.md` for playback / movie export.
