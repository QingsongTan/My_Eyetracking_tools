from __future__ import annotations

from gaze_toolkit.events import attach_events
from gaze_toolkit.features import extract_features
from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.preprocess import preprocess


def test_detected_events_feed_feature_extraction() -> None:
    recording = simulate_gaze_recording(duration_ms=3500, style="careful", seed=7)
    processed = preprocess(recording)
    enriched = attach_events(processed, velocity_threshold=600.0)
    features = extract_features(enriched)

    assert enriched.events
    assert features["fixation_count"] > 0.0
    assert features["saccade_count"] > 0.0
    assert "pupil_baseline" in features
    assert "x_approx_entropy" in features
