from __future__ import annotations

import pandas as pd

from gaze_toolkit.features import extract_features
from gaze_toolkit.pupil_preprocess import extract_pupil_load_features, preprocess_pupil_signal
from gaze_toolkit.types import GazeRecording


def test_preprocess_pupil_signal_interpolates_invalid_and_blink_ranges() -> None:
    recording = GazeRecording(
        samples=pd.DataFrame(
            {
                "timestamp_ms": [0, 100, 200, 300, 400, 500],
                "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "y": [0.2, 0.2, 0.3, 0.4, 0.5, 0.6],
                "pupil": [3.0, 3.1, 0.0, 0.0, 3.6, 3.7],
                "valid": [True, True, True, False, True, True],
            }
        )
    )
    result = preprocess_pupil_signal(recording, baseline_window_ms=200.0)

    assert result.cleaned_pupil.notna().all()
    assert result.blink_mask.iloc[2]
    assert result.blink_mask.iloc[3]
    assert result.metadata["interpolation_ratio"] > 0.0


def test_extract_pupil_load_features_returns_expected_keys() -> None:
    recording = GazeRecording(
        samples=pd.DataFrame(
            {
                "timestamp_ms": [0, 100, 200, 300, 400, 500],
                "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "y": [0.2, 0.2, 0.3, 0.4, 0.5, 0.6],
                "pupil": [3.0, 3.2, 3.4, 3.7, 3.6, 3.8],
                "valid": [True, True, True, True, True, True],
            }
        )
    )

    features = extract_pupil_load_features(recording)

    assert features["pupil_bc_peak"] > 0.0
    assert "pupil_phasic_peak" in features
    assert "pupil_dilation_latency_ms" in features


def test_extract_features_can_include_pupil_load_features() -> None:
    recording = GazeRecording(
        samples=pd.DataFrame(
            {
                "timestamp_ms": [0, 100, 200, 300, 400, 500],
                "x": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
                "y": [0.2, 0.22, 0.24, 0.28, 0.32, 0.4],
                "pupil": [3.0, 3.2, 3.5, 3.7, 3.4, 3.6],
                "valid": [True, True, True, True, True, True],
            }
        )
    )

    features = extract_features(recording, include_pupil_load_features=True)

    assert "pupil_bc_mean" in features
    assert "pupil_tonic_level" in features
    assert features["pupil_bc_peak"] >= 0.0


def test_extract_features_handles_missing_pupil_without_crashing() -> None:
    recording = GazeRecording(
        samples=pd.DataFrame(
            {
                "timestamp_ms": [0, 100, 200],
                "x": [0.1, 0.2, 0.3],
                "y": [0.2, 0.25, 0.3],
                "pupil": [None, None, None],
                "valid": [True, True, True],
            }
        )
    )

    features = extract_features(recording, include_pupil_load_features=True)

    assert features["pupil_bc_mean"] == 0.0
    assert features["pupil_phasic_peak"] == 0.0
