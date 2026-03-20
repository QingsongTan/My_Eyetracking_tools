from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from gaze_toolkit.cli import main as cli_main
from gaze_toolkit.config import load_config, merge_config
from gaze_toolkit.dashboard_launcher import main as launcher_main
from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.io import from_frame, load


def test_cli_train_demo_outputs_metrics(capsys) -> None:
    exit_code = cli_main(["train-demo", "--sessions", "12"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["metrics"]["accuracy"] >= 0.0


def test_config_load_and_merge(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io:\n  sampling_rate_hz: 120\nmodel:\n  model_name: random_forest\n", encoding="utf-8")

    loaded = load_config(config_path)
    merged = merge_config(loaded, {"model": {"test_size": 0.2}})

    assert loaded["io"]["sampling_rate_hz"] == 120
    assert merged["model"]["model_name"] == "random_forest"
    assert merged["model"]["test_size"] == 0.2


def test_io_from_frame_and_asc_loading(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 8.3, 16.6],
            "gaze_x": [100.0, 110.0, 120.0],
            "gaze_y": [200.0, 210.0, 220.0],
            "pupil_size": [3.1, 3.2, 3.3],
        }
    )
    recording = from_frame(frame)
    assert list(recording.samples.columns[:4]) == ["timestamp_ms", "x", "y", "pupil"]

    asc_path = tmp_path / "demo.asc"
    asc_path.write_text("0 100 200 3.1\n8.3 110 210 3.2\n16.6 120 220 3.3\n", encoding="utf-8")
    loaded = load(asc_path, format="asc")
    assert len(loaded.samples) == 3


def test_from_frame_accepts_nan_coordinates_and_marks_invalid() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 8.3, 16.6],
            "x": [100.0, None, 120.0],
            "y": [200.0, None, 220.0],
            "valid": [1, 1, "0"],
        }
    )

    recording = from_frame(frame)

    assert recording.samples["valid"].tolist() == [True, False, False]
    assert recording.samples.loc[1, "x"] != recording.samples.loc[1, "x"]


def test_dashboard_launcher_sets_streamlit_args(monkeypatch) -> None:
    from streamlit.web import cli as stcli

    def fake_main() -> int:
        assert sys.argv[0] == "streamlit"
        assert sys.argv[1] == "run"
        assert sys.argv[2].endswith("dashboard.py")
        return 0

    monkeypatch.setattr(stcli, "main", fake_main)
    assert launcher_main() == 0
