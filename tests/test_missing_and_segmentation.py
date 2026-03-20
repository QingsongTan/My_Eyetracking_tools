from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.pipeline import run_segmented_pipeline
from gaze_toolkit.preprocess import handle_missing_samples
from gaze_toolkit.segmentation import (
    segment_between_markers,
    segment_by_marker_windows,
    segment_by_time_ranges,
)
from gaze_toolkit.types import GazeRecording


def test_handle_missing_samples_drop_strategy_removes_invalid_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 10.0, 20.0, 30.0],
            "x": [100.0, np.nan, 120.0, 130.0],
            "y": [200.0, np.nan, 220.0, 230.0],
            "valid": [True, False, True, True],
        }
    )
    recording = GazeRecording(frame)

    cleaned = handle_missing_samples(recording, strategy="drop")

    assert len(cleaned.samples) == 3
    assert cleaned.samples["valid"].all()


def test_segment_by_marker_windows_and_between_markers() -> None:
    recording = simulate_gaze_recording(duration_ms=3000, seed=5)
    recording.samples["marker"] = pd.NA
    recording.samples.loc[10, "marker"] = "baseline"
    recording.samples.loc[40, "marker"] = "task_start"
    recording.samples.loc[100, "marker"] = "task_end"

    marker_segments = segment_by_marker_windows(
        recording,
        marker_values=["baseline"],
        pre_ms=50.0,
        post_ms=100.0,
    )
    paired_segments = segment_between_markers(recording, start_marker="task_start", end_marker="task_end")

    assert len(marker_segments) == 1
    assert marker_segments[0].recording.metadata["marker_value"] == "baseline"
    assert len(paired_segments) == 1
    assert paired_segments[0].start_time_ms < paired_segments[0].end_time_ms


def test_segment_by_time_ranges() -> None:
    recording = simulate_gaze_recording(duration_ms=3000, seed=6)
    segments = segment_by_time_ranges(recording, [(0.0, 500.0), (500.0, 1200.0)])

    assert len(segments) == 2
    assert segments[0].recording.duration_ms <= 500.0


def test_run_segmented_pipeline_extracts_features_per_segment(tmp_path: Path) -> None:
    recording = simulate_gaze_recording(duration_ms=3000, seed=8)
    recording.samples["marker"] = pd.NA
    recording.samples.loc[20, "marker"] = "cue"
    source = tmp_path / "segmented.csv"
    recording.samples.to_csv(source, index=False)

    feature_table = run_segmented_pipeline(
        source,
        segmentation={
            "strategy": "marker_windows",
            "marker_values": ["cue"],
            "pre_ms": 100.0,
            "post_ms": 150.0,
        },
        overrides={
            "io": {"sampling_rate_hz": 120},
            "preprocess": {"missing_strategy": "interpolate"},
            "events": {"source": "detected"},
        },
    )

    assert len(feature_table) == 1
    assert "segment_name" in feature_table.columns
    assert feature_table.loc[0, "sample_count"] > 0.0

