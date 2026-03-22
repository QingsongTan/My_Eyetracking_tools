from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gaze_toolkit.aoi import (
    assign_fixations_to_aoi,
    compute_aoi_metrics,
    compute_transition_matrix,
    define_aoi,
    define_polygon_aoi,
)
from gaze_toolkit.tables import FIXATION_TABLE_COLUMNS


def test_assign_fixations_rectangle() -> None:
    fixations = _make_fixation_frame([(50.0, 50.0), (250.0, 50.0), (420.0, 420.0)])
    aois = [
        define_aoi("左上", 0.0, 0.0, 100.0, 100.0),
        define_aoi("右上", 200.0, 0.0, 300.0, 100.0),
    ]

    assigned = assign_fixations_to_aoi(fixations, aois)

    assert assigned["aoi_name"].tolist() == ["左上", "右上", None]


def test_assign_fixations_polygon() -> None:
    fixations = _make_fixation_frame([(10.0, 10.0), (80.0, 80.0)])
    aois = [define_polygon_aoi("三角区", [(0.0, 0.0), (120.0, 0.0), (0.0, 120.0)])]

    assigned = assign_fixations_to_aoi(fixations, aois)

    assert assigned.loc[0, "aoi_name"] == "三角区"
    assert pd.isna(assigned.loc[1, "aoi_name"])


def test_aoi_metrics_ttff_and_dwell() -> None:
    fixations = _make_fixation_frame(
        [(10.0, 10.0), (210.0, 10.0), (220.0, 20.0)],
        starts=[100.0, 200.0, 350.0],
        durations=[50.0, 100.0, 50.0],
    )
    aois = [define_aoi("目标区", 200.0, 0.0, 300.0, 100.0)]
    assigned = assign_fixations_to_aoi(fixations, aois)

    metrics = compute_aoi_metrics(assigned, aois, total_duration=500.0)["目标区"]

    assert metrics.first_fixation_time == pytest.approx(100.0)
    assert metrics.total_dwell_time == pytest.approx(150.0)
    assert metrics.dwell_proportion == pytest.approx(0.3)
    assert metrics.fixation_count == 2
    assert metrics.mean_fixation_duration == pytest.approx(75.0)


def test_aoi_metrics_visit_and_revisit() -> None:
    aois = [
        define_aoi("A", 0.0, 0.0, 100.0, 100.0),
        define_aoi("B", 200.0, 0.0, 300.0, 100.0),
    ]
    fixations = _make_fixation_frame(
        [(10.0, 10.0), (220.0, 10.0), (20.0, 20.0)],
        starts=[0.0, 120.0, 240.0],
        durations=[80.0, 80.0, 80.0],
    )
    assigned = assign_fixations_to_aoi(fixations, aois)

    metrics = compute_aoi_metrics(assigned, aois, total_duration=320.0)

    assert metrics["A"].visit_count == 2
    assert metrics["A"].revisit_count == 1
    assert metrics["B"].visit_count == 1
    assert metrics["B"].revisit_count == 0


def test_transition_matrix_normalization() -> None:
    fixations = _fixations_with_aoi(["A", "B", "A", "C"])

    matrix = compute_transition_matrix(fixations, ["A", "B", "C"])

    assert matrix.loc["A"].sum() == pytest.approx(1.0)
    assert matrix.loc["B"].sum() == pytest.approx(1.0)
    assert matrix.loc["C"].sum() == pytest.approx(0.0)


def test_transition_matrix_no_self_transitions() -> None:
    fixations = _fixations_with_aoi(["A", "A", "B", "B", "A"])

    matrix = compute_transition_matrix(fixations, ["A", "B"])

    assert matrix.loc["A", "A"] == pytest.approx(0.0)
    assert matrix.loc["B", "B"] == pytest.approx(0.0)
    assert matrix.loc["A", "B"] == pytest.approx(1.0)
    assert matrix.loc["B", "A"] == pytest.approx(1.0)


def test_empty_aoi_returns_zero_metrics() -> None:
    fixations = _make_fixation_frame([(400.0, 400.0)], starts=[0.0], durations=[60.0])
    aois = [define_aoi("未命中", 0.0, 0.0, 100.0, 100.0)]
    assigned = assign_fixations_to_aoi(fixations, aois)

    metrics = compute_aoi_metrics(assigned, aois, total_duration=300.0)["未命中"]

    assert metrics.first_fixation_time is None
    assert metrics.total_dwell_time == pytest.approx(0.0)
    assert metrics.dwell_proportion == pytest.approx(0.0)
    assert metrics.fixation_count == 0
    assert metrics.visit_count == 0
    assert metrics.revisit_count == 0
    assert metrics.mean_fixation_duration == pytest.approx(0.0)


def _make_fixation_frame(
    points: list[tuple[float, float]],
    *,
    starts: list[float] | None = None,
    durations: list[float] | None = None,
) -> pd.DataFrame:
    starts = starts or [float(index * 100) for index in range(len(points))]
    durations = durations or [80.0] * len(points)
    rows: list[dict[str, object]] = []

    for index, ((x_pos, y_pos), start_time, duration) in enumerate(zip(points, starts, durations, strict=True)):
        rows.append(
            {
                "event_index": index,
                "start_time_ms": float(start_time),
                "end_time_ms": float(start_time + duration),
                "duration_ms": float(duration),
                "centroid_x": float(x_pos),
                "centroid_y": float(y_pos),
                "session_id": None,
                "subject_id": None,
                "condition": None,
                "trial": None,
            }
        )

    return pd.DataFrame(rows, columns=FIXATION_TABLE_COLUMNS)


def _fixations_with_aoi(sequence: list[str]) -> pd.DataFrame:
    frame = _make_fixation_frame(
        [(float(index * 10), float(index * 10)) for index in range(len(sequence))],
        starts=[float(index * 100) for index in range(len(sequence))],
        durations=[80.0] * len(sequence),
    )
    frame["aoi_name"] = sequence
    return frame
