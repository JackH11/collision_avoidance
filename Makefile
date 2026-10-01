# Collision Avoidance — developer targets (Phase 3+)

.PHONY: eval-suite eval-suite-smoke regression-gate regression-gate-json help

help:
	@echo "Targets:"
	@echo "  make eval-suite            Formal suite (scenarios × seeds → evals/artifacts/)"
	@echo "  make eval-suite-smoke      Short suite (N=5, baseline only) for quick checks"
	@echo "  make regression-gate       Live gate on Phase 2 checkpoint (fails if missing)"
	@echo "  make regression-gate-json  Gate against checked-in Phase 2 summary JSON"

# Full formal suite: easy/baseline/hard × baselines + any local Phase 2 zips
eval-suite:
	python evals/run_suite.py --episodes 20 --seeds 0 --scenarios easy,baseline,hard

# Faster local smoke (still writes artifacts)
eval-suite-smoke:
	python evals/run_suite.py --episodes 5 --seeds 0 --scenarios baseline

# Live rollout gate — requires a models/*.zip from Phase 2 training
regression-gate:
	python evals/regression_gate.py

# Threshold check without retraining (uses evals/ppo_final_baseline_old.json)
regression-gate-json:
	python evals/regression_gate.py --from-json evals/ppo_final_baseline_old.json
