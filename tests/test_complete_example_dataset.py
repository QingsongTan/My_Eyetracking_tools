from __future__ import annotations

from pathlib import Path

from gaze_toolkit.datasets import create_complete_example_recording, write_complete_example_csv


def test_complete_example_recording_contains_expected_columns_and_signals() -> None:
    recording = create_complete_example_recording()
    frame = recording.samples

    assert {
        "timestamp_ms",
        "x",
        "y",
        "pupil",
        "valid",
        "marker",
        "event_label",
        "label",
        "trial",
    }.issubset(frame.columns)
    assert {"fixation", "saccade", "blink", "smooth_pursuit"}.issubset(set(frame["event_label"].dropna()))
    assert frame["marker"].dropna().shape[0] >= 8
    assert (~frame["valid"]).any()
    assert frame.loc[~frame["valid"], "x"].isna().all()


def test_write_complete_example_csv_creates_uploadable_file(tmp_path: Path) -> None:
    output = tmp_path / "complete_eye_tracking_example.csv"
    written = write_complete_example_csv(output)

    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert "timestamp_ms,x,y,pupil,valid,marker,event_label,label,trial" in text.splitlines()[0]
