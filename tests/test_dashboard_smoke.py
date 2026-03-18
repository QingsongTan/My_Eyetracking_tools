from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_smoke() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "gaze_toolkit" / "dashboard.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=120)

    assert not app.exception
