from __future__ import annotations

from pathlib import Path

from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.pipeline import run_pipeline


def test_run_pipeline_on_csv(tmp_path: Path) -> None:
    recording = simulate_gaze_recording(seed=21)
    source = tmp_path / "demo.csv"
    recording.samples.to_csv(source, index=False)

    features = run_pipeline(source)

    assert features["sample_count"] > 0.0
    assert "fixation_count" in features

