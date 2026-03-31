from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gaze_toolkit.io import load
from gaze_toolkit.pymovements_adapter import HAS_PYMOVEMENTS, from_pymovements, load_with_pymovements


pytestmark = pytest.mark.skipif(not HAS_PYMOVEMENTS, reason="pymovements not installed")


def test_from_pymovements_flattens_pixel_samples() -> None:
    import polars as pl
    import pymovements as pm

    samples = pl.DataFrame(
        {
            "timestamp_ms": [0, 10, 20],
            "x": [100, 110, 120],
            "y": [200, 210, 220],
            "pupil": [3.1, 3.2, 3.3],
            "valid": [True, True, False],
        }
    )
    gaze = pm.Gaze(samples=samples, time_column="timestamp_ms", time_unit="ms", pixel_columns=["x", "y"])
    recording = from_pymovements(gaze, metadata={"subject_id": "p01"})

    assert list(recording.samples.columns)[:5] == ["timestamp_ms", "x", "y", "pupil", "valid"]
    assert recording.samples["x"].tolist() == [100.0, 110.0, 120.0]
    assert recording.samples["y"].tolist() == [200.0, 210.0, 220.0]
    assert recording.metadata["subject_id"] == "p01"
    assert "pymovements_metadata" in recording.metadata


def test_load_with_pymovements_reads_csv(tmp_path: Path) -> None:
    source = tmp_path / "pm_demo.csv"
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0, 10, 20],
            "x": [300, 310, 320],
            "y": [400, 410, 420],
            "pupil": [3.0, 3.2, 3.1],
            "valid": [True, True, True],
        }
    )
    frame.to_csv(source, index=False)

    recording = load_with_pymovements(source)

    assert recording.source_format == "pymovements"
    assert recording.samples["timestamp_ms"].tolist() == [0.0, 10.0, 20.0]


def test_io_load_can_route_to_pymovements_loader(tmp_path: Path) -> None:
    source = tmp_path / "pm_io.csv"
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0, 16, 32],
            "x": [10, 20, 30],
            "y": [15, 25, 35],
        }
    )
    frame.to_csv(source, index=False)

    recording = load(source, format="pymovements")

    assert recording.source_format == "pymovements"
    assert recording.samples["x"].tolist() == [10.0, 20.0, 30.0]
