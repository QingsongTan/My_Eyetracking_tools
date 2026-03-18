from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from gaze_toolkit.features import extract_features
from gaze_toolkit.types import GazeRecording

if TYPE_CHECKING:
    from gaze_toolkit.modeling import ModelResult


@dataclass
class OnlinePrediction:
    """A single online model output."""

    timestamp_ms: float
    prediction: Any
    features: dict[str, float]
    scores: dict[str, float] | None = None


class SlidingWindowPredictor:
    """Sliding-window online predictor for real-time gaze streams."""

    def __init__(self, model_result: ModelResult, window_ms: float = 1500.0, step_ms: float = 250.0) -> None:
        self.model_result = model_result
        self.window_ms = window_ms
        self.step_ms = step_ms
        self._buffer: deque[dict[str, Any]] = deque()
        self._last_emit_ms = float("-inf")

    def update(self, sample: dict[str, Any]) -> OnlinePrediction | None:
        """Append a sample and emit a prediction when the step interval is met."""
        if "timestamp_ms" not in sample:
            raise KeyError("Online samples must contain `timestamp_ms`.")

        current_time = float(sample["timestamp_ms"])
        self._buffer.append(sample)
        while self._buffer and current_time - float(self._buffer[0]["timestamp_ms"]) > self.window_ms:
            self._buffer.popleft()

        if current_time - self._last_emit_ms < self.step_ms or len(self._buffer) < 5:
            return None

        frame = pd.DataFrame(list(self._buffer))
        recording = GazeRecording(samples=frame)
        feature_row = extract_features(recording)
        matrix = pd.DataFrame(
            [{name: feature_row.get(name, 0.0) for name in self.model_result.feature_names}]
        ).fillna(0.0)

        raw_prediction = self.model_result.estimator.predict(matrix)[0]
        prediction = raw_prediction
        if self.model_result.label_encoder is not None:
            prediction = self.model_result.label_encoder.inverse_transform(np.asarray([int(raw_prediction)]))[0]

        scores = None
        if hasattr(self.model_result.estimator, "predict_proba"):
            probabilities = self.model_result.estimator.predict_proba(matrix)[0]
            if self.model_result.label_encoder is not None:
                labels = self.model_result.label_encoder.classes_
            else:
                labels = getattr(self.model_result.estimator, "classes_", range(len(probabilities)))
            scores = {str(label): float(score) for label, score in zip(labels, probabilities)}

        self._last_emit_ms = current_time
        return OnlinePrediction(
            timestamp_ms=current_time,
            prediction=prediction,
            features=feature_row,
            scores=scores,
        )

