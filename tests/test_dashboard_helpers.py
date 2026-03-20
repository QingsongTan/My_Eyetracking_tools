from __future__ import annotations

import pandas as pd
import pytest

from gaze_toolkit.dashboard import DashboardControls, _build_segment_views, _parse_time_ranges
from gaze_toolkit.datasets import simulate_gaze_recording


def test_parse_time_ranges_supports_multiple_segments() -> None:
    ranges = _parse_time_ranges("0-800\n1200-2000")

    assert ranges == [(0.0, 800.0), (1200.0, 2000.0)]


def test_parse_time_ranges_rejects_descending_range() -> None:
    with pytest.raises(ValueError, match="结束时间不能早于开始时间"):
        _parse_time_ranges("800-200")


def test_build_segment_views_returns_segment_specific_analysis() -> None:
    recording = simulate_gaze_recording(duration_ms=3000, seed=21)
    recording.samples["marker"] = pd.NA
    recording.samples.loc[18, "marker"] = "cue"
    controls = DashboardControls(
        preprocess_params={"missing_strategy": "interpolate", "smooth_window": 5},
        event_params={
            "velocity_threshold": 850.0,
            "min_fixation_ms": 60.0,
            "blink_min_duration_ms": 75.0,
            "source": "thresholds",
        },
        feature_params={"include_complexity": True},
        segmentation_config={"strategy": "marker_windows", "marker_values": ["cue"], "pre_ms": 80.0, "post_ms": 120.0},
        segmentation_summary="marker window",
    )

    segment_views = _build_segment_views(recording, controls)

    assert len(segment_views) == 1
    assert segment_views[0].analysis.features["duration_ms"] < recording.duration_ms
