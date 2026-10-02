# Collision Avoidance — developer targets (Phase 3+)

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help test test-fast eval-suite eval-suite-smoke regression-gate \
	regression-gate-json regression-gate-ci demo-record ci

help:
	@echo "Targets:"
	@echo "  make test                  Unit/smoke tests (pytest)"
	@echo "  make eval-suite            Formal suite (scenarios × seeds → evals/artifacts/)"
	@echo "  make eval-suite-smoke      Short suite (N=5, baseline only) for quick checks"
	@echo "  make regression-gate       Live gate on Phase 2 checkpoint (fails if missing)"
	@echo "  make regression-gate-json  Gate against checked-in Phase 2 summary JSON"
	@echo "  make regression-gate-ci    JSON gate + live gate with --skip-if-missing"
	@echo "  make demo-record           Headless GIF+MP4 under media/ (needs model zip)"
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
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) main.py \
		--headless --record media/demo.gif --scenario baseline --episodes 3 --fps 30
	SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 $(PYTHON) main.py \
		--headless --record media/demo.mp4 --scenario baseline --episodes 3 --fps 30

# What GitHub Actions runs (mirrors .github/workflows/ci.yml)
ci: test regression-gate-ci eval-suite-smoke
