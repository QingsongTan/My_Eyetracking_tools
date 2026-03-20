from __future__ import annotations

import pandas as pd

from gaze_toolkit.events import attach_events, detect_events, has_labeled_events
from gaze_toolkit.types import GazeRecording


def test_detect_events_from_original_labels() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "x": [100.0, 100.5, 101.0, 120.0, 121.0, 121.5],
            "y": [200.0, 200.2, 200.3, 220.0, 221.0, 221.5],
            "event_label": ["fixation", "fixation", "fixation", "saccade", "saccade", "blink"],
            "valid": [True, True, True, True, True, False],
        }
    )
    recording = GazeRecording(frame)

    assert has_labeled_events(recording)
    events = detect_events(recording, source="labels", blink_min_duration_ms=0.0)
    enriched = attach_events(recording, source="auto", blink_min_duration_ms=0.0)

    assert [event.kind for event in events] == ["fixation", "saccade", "blink"]
    assert enriched.events[0].kind == "fixation"


def test_requesting_labels_without_label_column_raises() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 10.0, 20.0],
            "x": [100.0, 101.0, 102.0],
            "y": [200.0, 201.0, 202.0],
        }
    )
    recording = GazeRecording(frame)

    try:
        detect_events(recording, source="labels")
    except ValueError as exc:
        assert "no event label column" in str(exc).lower()
    else:
        raise AssertionError("Expected detect_events(..., source='labels') to fail without label data.")
