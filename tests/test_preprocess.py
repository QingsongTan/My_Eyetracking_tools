from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_toolkit.preprocess import interpolate_missing, normalize_coordinates
from gaze_toolkit.types import GazeRecording


def test_interpolate_missing_preserves_invalid_flags() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 8.3, 16.6, 24.9],
            "x": [100.0, np.nan, 120.0, 130.0],
            "y": [200.0, np.nan, 220.0, 230.0],
            "pupil": [3.1, np.nan, 3.2, 3.3],
            "valid": [True, False, True, True],
        }
    )
    recording = GazeRecording(frame)

    processed = interpolate_missing(recording)

    assert not np.isnan(processed.samples.loc[1, "x"])
    assert not np.isnan(processed.samples.loc[1, "y"])
    assert processed.samples.loc[1, "valid"] == False


def test_normalize_coordinates_creates_centered_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 10.0, 20.0],
            "x": [0.0, 50.0, 100.0],
            "y": [0.0, 25.0, 50.0],
        }
    )
    recording = GazeRecording(frame)

    normalized = normalize_coordinates(recording)

    assert "x_norm" in normalized.samples.columns
    assert "y_norm" in normalized.samples.columns
    assert normalized.samples["x_norm"].between(-1.0, 1.0).all()
    assert normalized.samples["y_norm"].between(-1.0, 1.0).all()

