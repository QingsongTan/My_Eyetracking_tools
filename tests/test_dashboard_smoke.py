from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _load_dashboard() -> AppTest:
    app_path = Path(__file__).resolve().parents[1] / "src" / "gaze_toolkit" / "dashboard.py"
    app = AppTest.from_file(str(app_path))
    app.run(timeout=300)
    return app


def _tab_container(app: AppTest):
    root = app._tree[0]
    for element in root.children.values():
        if type(element).__name__ == "Block" and getattr(element, "type", "") == "tab_container":
            return element
    raise AssertionError("Dashboard tab container not found")


def test_dashboard_smoke() -> None:
    app = _load_dashboard()

    assert not app.exception


def test_dashboard_single_session_has_one_primary_visualization_row() -> None:
    app = _load_dashboard()

    tabs = _tab_container(app).children
    single_session_tab = tabs[1]
    chart_row = single_session_tab.children[8]

    assert len(chart_row.children) == 2
    for column in chart_row.children.values():
        assert len(column.children) == 3
        assert type(column.children[0]).__name__ == "Markdown"
        assert type(column.children[1]).__name__ == "Caption"
        assert type(column.children[2]).__name__ == "UnknownElement"


def test_dashboard_can_switch_to_light_theme() -> None:
    app = _load_dashboard()

    theme_select = next(box for box in app.selectbox if box.label == "界面主题")
    theme_select.set_value("浅色模式")
    app.run(timeout=300)

    assert not app.exception
    assert any("#f9fdff" in getattr(markdown, "value", "") for markdown in app.markdown)
