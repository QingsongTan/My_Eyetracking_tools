from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gaze_toolkit.aoi import define_aoi
from gaze_toolkit.batch import (
    batch_analyze,
    batch_analyze_recordings,
    export_html_report,
    export_markdown_report,
)
from gaze_toolkit.types import GazeRecording


def test_batch_analyze_recordings_basic() -> None:
    recordings = [
        _make_quality_recording([True] * 10, metadata={"subject_id": "S01", "condition": "careful", "trial": "T1"}),
        _make_quality_recording([True] * 8 + [False] * 2, metadata={"subject_id": "S02", "condition": "skim", "trial": "T1"}),
        _make_quality_recording([True] * 6 + [False] * 4, metadata={"subject_id": "S03", "condition": "careful", "trial": "T2"}),
    ]

    result = batch_analyze_recordings(recordings)

    assert len(result) == 3
    assert {"file_path", "error", "quality_grade"}.issubset(result.columns)
    assert result["file_path"].fillna("").tolist() == ["", "", ""]
    assert result["error"].isna().all()


def test_batch_analyze_recordings_with_error() -> None:
    valid = _make_quality_recording([True] * 10, metadata={"subject_id": "S01"})
    invalid = _make_empty_recording(metadata={"subject_id": "S02"})
    result = batch_analyze_recordings([valid, invalid])

    assert len(result) == 2
    assert result.loc[0, "error"] is None or pd.isna(result.loc[0, "error"])
    assert isinstance(result.loc[1, "error"], str)
    assert result.loc[1, "error"]


def test_batch_analyze_recordings_with_aois() -> None:
    recordings = [_make_quality_recording([True] * 10, metadata={"subject_id": "S01"})]
    aois = [define_aoi("目标区", 90.0, 190.0, 150.0, 230.0)]

    result = batch_analyze_recordings(recordings, aois=aois)

    assert "目标区_dwell_ms" in result.columns
    assert "目标区_visit_count" in result.columns
    assert float(result.loc[0, "目标区_fixation_count"]) >= 1.0


def test_batch_analyze_recordings_no_complexity() -> None:
    recordings = [_make_quality_recording([True] * 12, metadata={"subject_id": "S01"})]

    result = batch_analyze_recordings(recordings, include_complexity=False)

    assert "x_approx_entropy" not in result.columns
    assert "y_approx_entropy" not in result.columns


def test_export_html_report_creates_file(tmp_path: Path) -> None:
    batch_df = batch_analyze_recordings([_make_quality_recording([True] * 10, metadata={"subject_id": "S01"})])
    output_path = tmp_path / "report.html"

    exported = export_html_report(batch_df, output_path, scenario_name="测试场景")
    content = output_path.read_text(encoding="utf-8")

    assert Path(exported).exists()
    assert "<html" in content
    assert "<table" in content


def test_export_markdown_report_creates_file(tmp_path: Path) -> None:
    batch_df = batch_analyze_recordings([_make_quality_recording([True] * 10, metadata={"subject_id": "S01"})])
    output_path = tmp_path / "report.md"

    exported = export_markdown_report(batch_df, output_path, scenario_name="测试场景")
    content = output_path.read_text(encoding="utf-8")

    assert Path(exported).exists()
    assert "|" in content
    assert "# 眼动批量分析报告" in content


def test_batch_file_error_does_not_abort(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.csv"
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 16.0, 32.0, 48.0],
            "x": [100.0, 102.0, 104.0, 106.0],
            "y": [200.0, 202.0, 204.0, 206.0],
            "valid": [True, True, True, True],
            "pupil": [3.0, 3.1, 3.2, 3.2],
        }
    )
    frame.to_csv(valid_path, index=False)

    missing_path = tmp_path / "missing.csv"
    result = batch_analyze([valid_path, missing_path])

    assert len(result) == 2
    assert result.loc[result["file_path"] == str(valid_path), "error"].isna().all()
    missing_error = result.loc[result["file_path"] == str(missing_path), "error"].iloc[0]
    assert isinstance(missing_error, str)
    assert missing_error


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


def _make_empty_recording(metadata: dict[str, object] | None = None) -> GazeRecording:
    frame = pd.DataFrame(
        {
            "timestamp_ms": pd.Series(dtype=float),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
            "pupil": pd.Series(dtype=float),
            "valid": pd.Series(dtype=bool),
        }
    )
    return GazeRecording(samples=frame, metadata=dict(metadata or {}))
