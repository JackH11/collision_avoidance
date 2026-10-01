# Config loader with named-scenario deep merge.
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into a copy of base."""
    out = deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_scenario(
    config: Optional[Dict[str, Any]] = None,
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a config dict with the named scenario merged over YAML defaults.

    Always starts from the file-level defaults (or a provided raw config that
    still contains a ``scenarios`` catalog) so switching scenarios does not
    leak knobs from a previously applied scenario.
    """
    # Prefer pristine file defaults when caller passes an already-merged CONFIG.
    if config is None:
        raw = deepcopy(RAW_CONFIG if "RAW_CONFIG" in globals() else load_config())
    elif config.get("_resolved"):
        raw = deepcopy(RAW_CONFIG)
    else:
        raw = deepcopy(config)

    scenarios = raw.get("scenarios", {}) or {}
    # Work on a copy without baking nested scenario catalog into merges twice.
    base = deepcopy(raw)
    base.pop("scenarios", None)

    name = scenario if scenario is not None else base.get("scenario", "baseline")
    base["scenario"] = name

    if name not in scenarios:
        known = ", ".join(sorted(scenarios)) or "(none)"
        raise ValueError(f"Unknown scenario '{name}'. Known: {known}")

    merged = _deep_merge(base, scenarios[name])
    merged["scenarios"] = scenarios
    merged["_resolved"] = True
    return merged


# Pristine YAML (no scenario applied) + active scenario config for importers.
RAW_CONFIG = load_config()
CONFIG = resolve_scenario(RAW_CONFIG)
