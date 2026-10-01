"""
Load SB3 / sb3-contrib policies without hardcoding the algorithm class.

Used by ``eval_policy.py`` and ``main.py`` so CNN PPO / QR-DQN checkpoints
and the legacy DQN zip all load the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Type

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

try:
    from sb3_contrib import QRDQN
except ImportError:  # pragma: no cover
    QRDQN = None  # type: ignore


_ALGO_MAP: dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
}
if QRDQN is not None:
    _ALGO_MAP["qrdqn"] = QRDQN


def resolve_model_path(model: str) -> Path:
    """Accept paths with or without ``.zip``; search a few default dirs."""
    raw = Path(model)
    candidates = [raw]
    if raw.suffix != ".zip":
        candidates.append(Path(f"{model}.zip"))

    # Common artifact locations from Phase 2 training
    name = raw.name if raw.suffix else raw.name
    stem = name[:-4] if name.endswith(".zip") else name
    for folder in ("models", "checkpoints", "agents", "."):
        candidates.append(Path(folder) / f"{stem}.zip")
        candidates.append(Path(folder) / "best_model.zip")
        candidates.append(Path(folder) / stem / "best_model.zip")

    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Model not found: {model}. Tried: "
        + ", ".join(str(c) for c in candidates[:8])
        + ("..." if len(candidates) > 8 else "")
    )


def detect_algo(model_path: Path, hint: Optional[str] = None) -> str:
    """Infer algorithm from an explicit hint or the filename stem."""
    if hint:
        key = hint.lower().replace("-", "").replace("_", "")
        aliases = {"rainbow": "qrdqn", "qr": "qrdqn", "qrdqn": "qrdqn"}
        key = aliases.get(key, hint.lower())
        if key not in _ALGO_MAP:
            raise ValueError(f"Unknown algo '{hint}'. Known: {sorted(_ALGO_MAP)}")
        return key

    stem = model_path.stem.lower()
    # Prefer longer / more specific tokens first
    for name in ("qrdqn", "ppo", "dqn"):
        if name in stem:
            return name

    # Legacy Phase-0 artifact
    if "avoidance" in stem or stem.startswith("dqn"):
        return "dqn"

    # Default: try PPO then fall through at load time
    return "ppo"


def load_model(
    model: str,
    *,
    algo: Optional[str] = None,
    env: Any = None,
    device: str = "auto",
) -> Tuple[BaseAlgorithm, str, Path]:
    """
    Load a saved policy.

    Returns ``(model, algo_name, resolved_path)``.
    """
    path = resolve_model_path(model)
    algo_name = detect_algo(path, hint=algo)
    cls = _ALGO_MAP[algo_name]

    try:
        loaded = cls.load(str(path), env=env, device=device)
        return loaded, algo_name, path
    except Exception as first_err:
        # Filename hint wrong — try remaining algorithms
        errors = {algo_name: first_err}
        for name, other_cls in _ALGO_MAP.items():
            if name == algo_name:
                continue
            try:
                loaded = other_cls.load(str(path), env=env, device=device)
                return loaded, name, path
            except Exception as err:  # noqa: BLE001
                errors[name] = err
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(f"Failed to load {path} with any known algo. {detail}") from first_err
