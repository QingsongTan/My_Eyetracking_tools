from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest
from PIL import Image
from streamlit.elements import image as st_image
from streamlit.elements.lib import image_utils

from gaze_toolkit.dashboard import (
    DashboardControls,
    _build_canvas_initial_drawing,
    _build_segment_views,
    _ensure_streamlit_drawable_canvas_compatibility,
    _load_canvas_background_image,
    _parse_canvas_rectangles_to_aois,
    _parse_time_ranges,
    _should_restore_canvas_draft,
)
from gaze_toolkit.datasets import simulate_gaze_recording


def test_parse_time_ranges_supports_multiple_segments() -> None:
    ranges = _parse_time_ranges("0-800\n1200-2000")

    assert ranges == [(0.0, 800.0), (1200.0, 2000.0)]


def test_parse_time_ranges_rejects_descending_range() -> None:
    with pytest.raises(ValueError, match="结束时间不能早于开始时间"):
        _parse_time_ranges("800-200")


def test_build_segment_views_returns_segment_specific_analysis() -> None:
    recording = simulate_gaze_recording(duration_ms=3000, seed=21)
    recording.samples["marker"] = pd.NA
    recording.samples.loc[18, "marker"] = "cue"
    controls = DashboardControls(
        preprocess_params={"missing_strategy": "interpolate", "smooth_window": 5},
        event_params={
            "velocity_threshold": 850.0,
            "min_fixation_ms": 60.0,
            "blink_min_duration_ms": 75.0,
            "source": "thresholds",
        },
        feature_params={"include_complexity": True},
        segmentation_config={"strategy": "marker_windows", "marker_values": ["cue"], "pre_ms": 80.0, "post_ms": 120.0},
        segmentation_summary="marker window",
    )

    segment_views = _build_segment_views(recording, controls)

    assert len(segment_views) == 1
    assert segment_views[0].analysis.features["duration_ms"] < recording.duration_ms


def test_load_canvas_background_image_supports_file_like_object() -> None:
    buffer = BytesIO()
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(buffer, format="PNG")
    buffer.seek(0)

    loaded = _load_canvas_background_image(buffer)

    assert loaded is not None
    assert loaded.size == (12, 8)
    assert loaded.mode == "RGBA"


def test_ensure_streamlit_drawable_canvas_compatibility_patches_legacy_image_to_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    original_image_to_url = getattr(st_image, "image_to_url", None)

    def fake_image_to_url(
        image: object,
        *,
        layout_config: object,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
    ) -> str:
        captured["image"] = image
        captured["layout_width"] = getattr(layout_config, "width", None)
        captured["clamp"] = clamp
        captured["channels"] = channels
        captured["output_format"] = output_format
        captured["image_id"] = image_id
        return "/media/test-image"

    monkeypatch.delattr(st_image, "image_to_url", raising=False)
    monkeypatch.setattr(image_utils, "image_to_url", fake_image_to_url)

    _ensure_streamlit_drawable_canvas_compatibility()

    assert hasattr(st_image, "image_to_url")
    assert st_image.image_to_url("img", 640, True, "RGB", "PNG", "canvas-bg") == "/media/test-image"
    assert captured == {
        "image": "img",
        "layout_width": 640,
        "clamp": True,
        "channels": "RGB",
        "output_format": "PNG",
        "image_id": "canvas-bg",
    }

    if original_image_to_url is None:
        monkeypatch.delattr(st_image, "image_to_url", raising=False)
    else:
        monkeypatch.setattr(st_image, "image_to_url", original_image_to_url)


def test_should_restore_canvas_draft_only_when_entering_canvas_mode() -> None:
    assert not _should_restore_canvas_draft(previous_mode=None, current_mode="手动输入")
    assert _should_restore_canvas_draft(previous_mode="手动输入", current_mode="鼠标绘制")
    assert not _should_restore_canvas_draft(previous_mode="鼠标绘制", current_mode="鼠标绘制")


def test_build_canvas_initial_drawing_avoids_live_feedback_loop() -> None:
    draft = {"objects": [{"type": "rect"}]}

    assert _build_canvas_initial_drawing(draft, restore_draft=False) is None
    assert _build_canvas_initial_drawing(None, restore_draft=True) is None
    assert _build_canvas_initial_drawing(draft, restore_draft=True) == draft


def test_parse_canvas_rectangles_to_aois_scales_coordinates() -> None:
    aois = _parse_canvas_rectangles_to_aois(
        {
            "objects": [
                {
                    "type": "rect",
                    "left": 10.0,
                    "top": 5.0,
                    "width": 20.0,
                    "height": 10.0,
                    "scaleX": 1.5,
                    "scaleY": 2.0,
                }
            ]
        },
        canvas_width=100,
        canvas_height=50,
        screen_size=(200, 100),
    )

    assert len(aois) == 1
    assert aois[0].name == "AOI 1"
    assert aois[0].region == pytest.approx((20.0, 10.0, 80.0, 50.0))


def test_parse_canvas_rectangles_to_aois_ignores_invalid_objects() -> None:
    aois = _parse_canvas_rectangles_to_aois(
        {
            "objects": [
                {"type": "line", "left": 0, "top": 0, "width": 30, "height": 20},
                {"type": "rect", "left": 2, "top": 2, "width": 4, "height": 7},
                {"type": "rect", "left": 5, "top": 5, "width": 20, "height": 20, "angle": 15},
                {"type": "rect", "left": 8, "top": 12, "width": 30, "height": 16},
            ]
        },
        canvas_width=100,
        canvas_height=100,
        screen_size=(1000, 1000),
    )

    assert len(aois) == 1
    assert aois[0].name == "AOI 1"
    assert aois[0].region == pytest.approx((80.0, 120.0, 380.0, 280.0))
