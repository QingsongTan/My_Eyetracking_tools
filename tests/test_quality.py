from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gaze_toolkit.quality import QualityReport, assess_quality, find_missing_segments, format_quality_cards
from gaze_toolkit.tables import compute_quality_grade
from gaze_toolkit.types import EyeEvent, GazeRecording

np.random.seed(42)


def test_assess_quality_perfect_data() -> None:
    recording = _make_quality_recording([True] * 10)

    report = assess_quality(recording)

    assert report.tracking_ratio == pytest.approx(1.0)
    assert report.missing_segments == 0
    assert report.max_gap_duration_ms == pytest.approx(0.0)
    assert report.quality_grade == "优"


def test_assess_quality_with_gaps() -> None:
    recording = _make_quality_recording(
        [True, False, False, True, False, False, False, True],
        time_step_ms=10.0,
    )

    report = assess_quality(recording)

    assert report.missing_segments == 2
    assert report.max_gap_duration_ms == pytest.approx(20.0)


def test_assess_quality_with_preprocessed() -> None:
    recording = _make_quality_recording([True, False, False, True, True])
    preprocessed = _make_quality_recording([True, True, True, True, True])

    report = assess_quality(recording, preprocessed=preprocessed)

    assert report.interpolated_ratio > 0.0


def test_assess_quality_blink_count() -> None:
    recording = _make_quality_recording([True] * 6)
    recording.events = [
        EyeEvent(kind="blink", start_time_ms=10.0, end_time_ms=40.0, start_index=1, end_index=2),
        EyeEvent(kind="fixation", start_time_ms=50.0, end_time_ms=80.0, start_index=3, end_index=4),
        EyeEvent(kind="blink", start_time_ms=90.0, end_time_ms=120.0, start_index=4, end_index=5),
    ]

    report = assess_quality(recording)

    assert report.blink_count == 2


def test_assess_quality_sampling_rate() -> None:
    timestamps = np.linspace(0.0, 1000.0, 100, dtype=float)
    recording = _make_quality_recording([True] * 100, timestamps=timestamps)

    report = assess_quality(recording)

    assert report.sampling_rate_actual == pytest.approx(99.0, rel=1e-6)


def test_assess_quality_empty_recording() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_ms": pd.Series(dtype=float),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
            "pupil": pd.Series(dtype=float),
            "valid": pd.Series(dtype=bool),
        }
    )
    recording = GazeRecording(samples=frame)

    report = assess_quality(recording)

    assert report.total_samples == 0
    assert report.valid_samples == 0
    assert report.tracking_ratio == pytest.approx(0.0)
    assert report.missing_segments == 0
    assert report.max_gap_duration_ms == pytest.approx(0.0)
    assert report.sampling_rate_actual == pytest.approx(0.0)


def test_find_missing_segments_none() -> None:
    recording = _make_quality_recording([True] * 6)

    segments = find_missing_segments(recording.samples)

    assert segments == []


def test_find_missing_segments_two_gaps() -> None:
    recording = _make_quality_recording(
        [True, False, False, True, False, False, False, True],
        time_step_ms=10.0,
    )

    segments = find_missing_segments(recording.samples)

    assert len(segments) == 2
    assert segments[0]["start_index"] == 1
    assert segments[0]["end_index"] == 2
    assert segments[0]["start_time_ms"] == pytest.approx(10.0)
    assert segments[0]["end_time_ms"] == pytest.approx(20.0)
    assert segments[0]["duration_ms"] == pytest.approx(10.0)
    assert segments[0]["sample_count"] == 2
    assert segments[1]["start_index"] == 4
    assert segments[1]["end_index"] == 6
    assert segments[1]["duration_ms"] == pytest.approx(20.0)
    assert segments[1]["sample_count"] == 3


def test_format_quality_cards_status() -> None:
    good_report = _make_quality_report(0.95)
    warn_report = _make_quality_report(0.80)
    bad_report = _make_quality_report(0.40)

    good_cards = {card["label"]: card for card in format_quality_cards(good_report)}
    warn_cards = {card["label"]: card for card in format_quality_cards(warn_report)}
    bad_cards = {card["label"]: card for card in format_quality_cards(bad_report)}

    assert good_cards["追踪率"]["status"] == "good"
    assert warn_cards["追踪率"]["status"] == "warn"
    assert bad_cards["追踪率"]["status"] == "bad"


def test_quality_report_grade_consistency() -> None:
    recording = _make_quality_recording([True, True, True, False])

    report = assess_quality(recording)

    assert report.quality_grade == compute_quality_grade(recording)


def _make_quality_recording(
    valid_flags: list[bool],
    *,
    time_step_ms: float = 16.0,
    timestamps: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> GazeRecording:
    valid_array = np.asarray(valid_flags, dtype=bool)
    sample_count = len(valid_array)
    if timestamps is None:
        timestamps = np.arange(sample_count, dtype=float) * time_step_ms
    x_values = np.linspace(100.0, 140.0, sample_count, dtype=float) if sample_count else np.array([], dtype=float)
    y_values = np.linspace(200.0, 220.0, sample_count, dtype=float) if sample_count else np.array([], dtype=float)

    frame = pd.DataFrame(
        {
            "timestamp_ms": np.asarray(timestamps, dtype=float),
            "x": np.where(valid_array, x_values, np.nan),
            "y": np.where(valid_array, y_values, np.nan),
            "pupil": np.where(valid_array, 3.2, np.nan),
            "valid": valid_array,
        }
    )
    return GazeRecording(samples=frame, metadata=dict(metadata or {}))


def _make_quality_report(tracking_ratio: float) -> QualityReport:
    if tracking_ratio >= 0.9:
        quality_grade = "优"
    elif tracking_ratio >= 0.75:
        quality_grade = "良"
    elif tracking_ratio >= 0.5:
        quality_grade = "可用"
    else:
        quality_grade = "建议剔除"

    report = QualityReport(
        tracking_ratio=tracking_ratio,
        total_samples=100,
        valid_samples=int(tracking_ratio * 100),
        missing_segments=1,
        max_gap_duration_ms=200.0,
        interpolated_ratio=0.02,
        blink_count=2,
        recording_duration_s=12.0,
        sampling_rate_actual=120.0,
        quality_grade=quality_grade,
    )
    setattr(report, "expected_sampling_rate_hz", 120.0)
    return report
