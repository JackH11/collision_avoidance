# Collision Avoidance — developer targets (Phase 3+)

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
GODOT ?= godot

.PHONY: help test test-fast eval-suite eval-suite-smoke regression-gate \
	regression-gate-json regression-gate-ci demo-record demo-large \
	dump-trajectory godot-movie ci

help:
	@echo "Targets:"
	@echo "  make test                  Unit/smoke tests (pytest)"
	@echo "  make eval-suite            Formal suite (scenarios × seeds → evals/artifacts/)"
	@echo "  make eval-suite-smoke      Short suite (N=5, baseline only) for quick checks"
	@echo "  make regression-gate       Live gate on Phase 2 checkpoint (fails if missing)"
	@echo "  make regression-gate-json  Gate against checked-in Phase 2 summary JSON"
	@echo "  make regression-gate-ci    JSON gate + live gate with --skip-if-missing"
	@echo "  make demo-record           Headless GIF+MP4 under media/ (needs model zip)"
	@echo "  make demo-large            Best agent on large 400×400 map → media/demo_large.*"
	@echo "  make dump-trajectory       JSON dump for Godot replay"
	@echo "  make godot-movie           Render Godot replay → media/godot_large.mp4"
	@echo "  make ci                    tests + regression-gate-ci + eval-suite-smoke"

# Fast unit/smoke tests (headless; no model zips required)
test:
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTEST)

test-fast: test

# Full formal suite: easy/baseline/hard × baselines + any local Phase 2 zips
eval-suite:
	SDL_VIDEODRIVER=dummy $(PYTHON) evals/run_suite.py --episodes 20 --seeds 0 --scenarios easy,baseline,hard

# Faster local / CI smoke (still writes artifacts; soft-skips missing models)
eval-suite-smoke:
	SDL_VIDEODRIVER=dummy $(PYTHON) evals/run_suite.py --episodes 5 --seeds 0 --scenarios baseline

# Live rollout gate — requires a models/*.zip from Phase 2 training
regression-gate:
	SDL_VIDEODRIVER=dummy $(PYTHON) evals/regression_gate.py

# Threshold check without retraining (uses evals/ppo_final_baseline_old.json)
regression-gate-json:
	$(PYTHON) evals/regression_gate.py --from-json evals/ppo_final_baseline_old.json

# CI-friendly: always run JSON gate; live gate skips cleanly when models/ absent
regression-gate-ci: regression-gate-json
	SDL_VIDEODRIVER=dummy $(PYTHON) evals/regression_gate.py --skip-if-missing

# Headless demo clip for README (GIF + MP4). Soft-fails if no model zip / ffmpeg.
demo-record:
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) -m collision_avoidance.demo \
		--headless --record media/demo.gif --scenario baseline --episodes 3 --fps 30
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) -m collision_avoidance.demo \
		--headless --record media/demo.mp4 --scenario baseline --episodes 3 --fps 30

# Best Phase 2 agent on the large 400×400 arena
demo-large:
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) -m collision_avoidance.demo \
		--headless --scenario large --record media/demo_large.gif \
		--model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn \
		--episodes 5 --fps 30 --seed 0
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) -m collision_avoidance.demo \
		--headless --scenario large --record media/demo_large.mp4 \
		--model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn \
		--episodes 5 --fps 30 --seed 0

# Trajectory JSON for Godot (Python = brain)
# Trajectory JSON for Godot (Python = brain) — success-only showcase on large map
dump-trajectory:
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) -m collision_avoidance.demo.dump_trajectory \
		--scenario large --successes 4 --min-steps 70 --max-attempts 80 --seed 0 \
		--model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn \
		--out media/trajectories/large_best.json
	mkdir -p godot_replay/data
	cp media/trajectories/large_best.json godot_replay/data/large_best.json

# Pretty Godot 4 replay → mp4 (requires Godot 4.3+ on PATH and xvfb-run)
# Success-only reel, agent↔goal framing, slower playback.
godot-movie: dump-trajectory
	xvfb-run -a $(GODOT) --path godot_replay --write-movie ../media/godot_large.avi \
		--fixed-fps 30 -- \
		--trajectory res://data/large_best.json --quit-when-done \
		--frame agent_goal --min-view-span 280 --min-frames 60 \
		--outcomes success --speed 0.8
	ffmpeg -y -i media/godot_large.avi \
		-c:v libx264 -crf 17 -preset slow -pix_fmt yuv420p -movflags +faststart \
		media/godot_large.mp4
	rm -f media/godot_large.avi

# What GitHub Actions runs (mirrors .github/workflows/ci.yml)
ci: test regression-gate-ci eval-suite-smoke
