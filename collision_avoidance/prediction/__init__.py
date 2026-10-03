"""Prediction backends."""

from collision_avoidance.prediction.model_prediction import (
    predict_batch,
    resolve_prediction_backend,
)

__all__ = ["predict_batch", "resolve_prediction_backend"]
