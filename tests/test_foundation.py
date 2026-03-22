from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gaze_toolkit.features import approximate_entropy
from gaze_toolkit.pipeline import build_feature_dataset
from gaze_toolkit.tables import FIXATION_TABLE_COLUMNS, compute_quality_grade, fixation_table
from gaze_toolkit.types import EyeEvent, GazeRecording


def test_fixation_table_columns() -> None:
    recording = _make_recording_with_events()

    table = fixation_table(recording)

    assert list(table.columns) == FIXATION_TABLE_COLUMNS


def test_fixation_table_only_fixations() -> None:
    recording = _make_recording_with_events()

    table = fixation_table(recording)

    assert len(table) == 2
    assert table["event_index"].tolist() == [0, 2]
    assert table["centroid_x"].tolist() == [110.0, 138.0]
    assert table["centroid_y"].tolist() == [205.0, 212.0]


def test_build_feature_dataset_design_columns() -> None:
    first = _make_dataset_recording(
        subject_id="S01",
        condition="careful",
        trial="T1",
        valid_flags=[True] * 9 + [False],
    )
    second = _make_dataset_recording(
        subject_id="S02",
        condition="skim",
        trial="T2",
        valid_flags=[True] * 4 + [False] * 4,
    )

    dataset = build_feature_dataset([first, second])

    assert {"subject_id", "condition", "trial", "quality_grade", "segment_name"}.issubset(dataset.columns)
    assert dataset["subject_id"].tolist() == ["S01", "S02"]
    assert dataset["condition"].tolist() == ["careful", "skim"]
    assert dataset["trial"].tolist() == ["T1", "T2"]
    assert dataset["quality_grade"].tolist() == ["优", "可用"]


def test_quality_grade_thresholds() -> None:
    excellent = _make_quality_recording([True] * 9 + [False])
    good = _make_quality_recording([True] * 3 + [False])
    usable = _make_quality_recording([True, True, False, False])
    reject = _make_quality_recording([True, False, False, False])

    assert compute_quality_grade(excellent) == "优"
    assert compute_quality_grade(good) == "良"
    assert compute_quality_grade(usable) == "可用"
    assert compute_quality_grade(reject) == "建议剔除"


def test_approximate_entropy_large_input() -> None:
    signal = np.sin(np.linspace(0.0, 40.0 * np.pi, 5000, dtype=float))

    reduced = approximate_entropy(signal, max_samples=2000)
    indices = np.floor(np.arange(2000, dtype=float) * (len(signal) / 2000.0)).astype(int)
    indices[-1] = len(signal) - 1
    expected = approximate_entropy(signal[indices], max_samples=2000)

    assert np.isfinite(reduced)
    assert 0.0 <= reduced < 5.0
    assert reduced == pytest.approx(expected)


def _make_recording_with_events() -> GazeRecording:
    recording = _make_quality_recording(
        [True, True, True, True, True, True],
        metadata={
            "session_id": "session-01",
            "subject_id": "P01",
            "condition": "careful",
            "trial": "trial-01",
        },
    )
    recording.events = [
        EyeEvent(
            kind="fixation",
            start_time_ms=0.0,
            end_time_ms=80.0,
            start_index=0,
            end_index=1,
            metadata={"centroid_x": 110.0, "centroid_y": 205.0},
        ),
        EyeEvent(
            kind="saccade",
            start_time_ms=80.0,
            end_time_ms=120.0,
            start_index=2,
            end_index=3,
            metadata={"centroid_x": 120.0, "centroid_y": 208.0},
        ),
        EyeEvent(
            kind="fixation",
            start_time_ms=120.0,
            end_time_ms=220.0,
            start_index=4,
            end_index=5,
            metadata={"centroid_x": 138.0, "centroid_y": 212.0},
        ),
    ]
    return recording


def _make_dataset_recording(
    *,
    subject_id: str,
    condition: str,
    trial: str,
    valid_flags: list[bool],
) -> GazeRecording:
    return _make_quality_recording(
        valid_flags,
        metadata={
            "subject_id": subject_id,
            "condition": condition,
            "trial": trial,
            "intent_label": condition,
            "segment_name": f"segment-{trial}",
        },
    )


def _make_quality_recording(
    valid_flags: list[bool],
    metadata: dict[str, object] | None = None,
) -> GazeRecording:
    timestamps = np.arange(len(valid_flags), dtype=float) * 16.0
    x_values = np.linspace(100.0, 140.0, len(valid_flags), dtype=float)
    y_values = np.linspace(200.0, 220.0, len(valid_flags), dtype=float)
    valid_array = np.asarray(valid_flags, dtype=bool)

    frame = pd.DataFrame(
        {
            "timestamp_ms": timestamps,
            "x": np.where(valid_array, x_values, np.nan),
            "y": np.where(valid_array, y_values, np.nan),
            "pupil": np.where(valid_array, 3.2, np.nan),
            "valid": valid_array,
        }
    )
    return GazeRecording(samples=frame, metadata=dict(metadata or {}))
