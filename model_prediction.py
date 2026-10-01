"""
Pluggable obstacle trajectory prediction backends.

Backends share return shape: (pred_x, pred_y, std_x, std_y).
Supported names: ``simple`` | ``nn_uncertainty`` (alias: ``nn``).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Protocol, Sequence, Tuple

import numpy as np

Prediction = Tuple[float, float, float, float]
PredictFn = Callable[[object], Prediction]

_model_cache = None


class PredictionBackend(Protocol):
    """Callable that maps one mover → (pred_x, pred_y, std_x, std_y)."""

    def __call__(self, item: object) -> Prediction: ...


def agent_predict(model, item):
    """Synchronous prediction function (legacy helper)."""
    X = item.get_history(lag=5)
    y_pred = model.predict(np.array([X]), verbose=0)
    pred_x = y_pred[0][0]
    pred_y = y_pred[0][1]
    return pred_x, pred_y


def agent_uncertain_predict(model, item):
    X = item.get_history(lag=10, window=5)
    y_pred = model.predict(np.array([X]), verbose=0)
    pred_x = y_pred[0][0]
    pred_y = y_pred[0][1]
    std_x = y_pred[0][2]
    std_y = y_pred[0][3]
    return pred_x, pred_y, std_x, std_y


def update_item_async(item):
    """Update one item (name kept for main.py compatibility)."""
    item.update()
    return item


def get_cached_model():
    """Lazy-load the Keras predictor (imports TensorFlow only on first call)."""
    global _model_cache
    if _model_cache is None:
        from utils import get_model
        from nn.nn import nll_gaussian, ClippedLogVar
        from config import CONFIG

        model_name = CONFIG.get("prediction", {}).get("model_name", "j_10_5")
        _model_cache = get_model(
            model_name,
            custom_objects={
                "nll_gaussian": nll_gaussian,
                "ClippedLogVar": ClippedLogVar,
            },
            safe_mode=False,
        )
    return _model_cache


def make_nn_prediction(item) -> Prediction:
    """Single-item NN uncertainty prediction (lag/window aligned to j_10_5)."""
    from config import CONFIG

    pred_cfg = CONFIG.get("prediction", {})
    lag = int(pred_cfg.get("lag", 10))
    window = int(pred_cfg.get("window", 5))
    model = get_cached_model()
    X = item.get_history(lag=lag, window=window)
    y_pred = model.predict(np.array([X]), verbose=0)[0]
    return float(y_pred[0]), float(y_pred[1]), float(y_pred[2]), float(y_pred[3])


def make_nn_predictions_batch(items: Sequence[object]) -> List[Prediction]:
    """Batched NN predict — one model.predict for the whole obstacle set."""
    if not items:
        return []
    from config import CONFIG

    pred_cfg = CONFIG.get("prediction", {})
    lag = int(pred_cfg.get("lag", 10))
    window = int(pred_cfg.get("window", 5))
    model = get_cached_model()
    X = np.array([item.get_history(lag=lag, window=window) for item in items])
    y_pred = model.predict(X, verbose=0)
    return [
        (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        for row in y_pred
    ]


def make_simple_prediction(item) -> Prediction:
    from config import CONFIG

    horizon = float(CONFIG.get("prediction", {}).get("simple_horizon", 8))
    pred_x = item.x + item.vx * horizon
    pred_y = item.y + item.vy * horizon
    speed = (item.vx ** 2 + item.vy ** 2) ** 0.5
    std_x = 0.5 * speed
    std_y = 0.5 * speed
    return pred_x, pred_y, std_x, std_y


def normalize_backend_name(backend: str) -> str:
    name = (backend or "simple").strip().lower()
    if name in ("nn", "nn_uncertainty"):
        return "nn_uncertainty"
    if name == "simple":
        return "simple"
    raise ValueError(
        f"Unknown prediction backend '{backend}'. "
        "Use 'simple' or 'nn_uncertainty'."
    )


def resolve_prediction_backend(backend: str | None = None) -> tuple[str, PredictFn]:
    """
    Resolve backend name → (canonical_name, per-item predict fn).

    Prefer ``predict_batch`` for step loops; this returns the single-item fn
    for callers that still need one-off predictions.
    """
    from config import CONFIG

    if backend is None:
        backend = CONFIG.get("prediction", {}).get("backend", "simple")
    name = normalize_backend_name(backend)
    if name == "simple":
        return name, make_simple_prediction
    return name, make_nn_prediction


def predict_batch(
    items: Iterable[object],
    backend: str | None = None,
    predict_fn: PredictFn | None = None,
) -> List[Prediction]:
    """
    Sync batch predictions — no asyncio / no per-step event-loop nesting.

    For ``nn_uncertainty``, uses a single batched model.predict when possible.
    """
    items = list(items)
    if not items:
        return []

    if predict_fn is None or backend is not None:
        name, predict_fn = resolve_prediction_backend(backend)
    else:
        name = None
        # Infer NN path if the callable is the nn helper.
        if getattr(predict_fn, "__name__", "") == "make_nn_prediction":
            name = "nn_uncertainty"

    if name == "nn_uncertainty" or (
        predict_fn is not None
        and getattr(predict_fn, "__name__", "") == "make_nn_prediction"
    ):
        return make_nn_predictions_batch(items)

    assert predict_fn is not None
    return [predict_fn(item) for item in items]
