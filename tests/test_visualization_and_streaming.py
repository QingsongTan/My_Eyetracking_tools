from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaze_toolkit.analysis import analyze_recording
from gaze_toolkit.datasets import simulate_gaze_recording, simulate_intent_dataset
from gaze_toolkit.modeling import train_model
from gaze_toolkit.streaming import SlidingWindowPredictor
from gaze_toolkit.visualization import (
    plot_feature_importance,
    plot_heatmap,
    plot_interactive_heatmap,
    plot_interactive_scanpath,
    plot_scanpath,
    plot_signal_overview,
)


def test_visualization_helpers_return_renderable_objects() -> None:
    recording = simulate_gaze_recording(seed=9)
    analysis = analyze_recording(recording)

    figure, axis = plt.subplots()
    returned_axis = plot_scanpath(recording, ax=axis)
    assert returned_axis.get_title() == "扫描路径轨迹"
    plt.close(figure)

    figure, axis = plt.subplots()
    returned_axis = plot_heatmap(recording, ax=axis)
    assert returned_axis.get_title() == "核密度注意力热力图"
    plt.close(figure)

    interactive_scanpath = plot_interactive_scanpath(analysis.enriched_recording)
    assert interactive_scanpath.layout.title.text == "交互式眼动扫描路径 (Scanpath)"
    assert len(interactive_scanpath.data) >= 1

    interactive_heatmap = plot_interactive_heatmap(analysis.enriched_recording)
    assert interactive_heatmap.layout.title.text == "注意力核密度热力图 (Attention KDE Heatmap)"
    assert len(interactive_heatmap.data) == 1

    signal_figure = plot_signal_overview(recording)
    assert len(signal_figure.axes) == 4
    assert signal_figure.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb("#08172d")
    assert signal_figure.axes[0].get_facecolor()[:3] == plt.matplotlib.colors.to_rgb("#08172d")
    plt.close(signal_figure)

    importance = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance_mean": [0.3, 0.2, 0.1],
            "importance_std": [0.01, 0.01, 0.01],
        }
    )
    figure, axis = plt.subplots()
    returned_axis = plot_feature_importance(importance, ax=axis)
    assert returned_axis.get_title() == "关键特征重要性"
    plt.close(figure)


def test_interactive_scanpath_attaches_uploaded_background(tmp_path) -> None:
    recording = analyze_recording(simulate_gaze_recording(seed=12)).enriched_recording
    background = np.zeros((24, 24, 3), dtype=float)
    background[..., 1] = 0.6
    background_path = tmp_path / "stimulus.png"
    plt.imsave(background_path, background)

    figure = plot_interactive_scanpath(
        recording,
        background_image=background_path,
        screen_size=(1920, 1080),
    )

    assert len(figure.layout.images) == 1
    assert str(figure.layout.images[0].source).startswith("data:image/png;base64,")
    assert figure.layout.plot_bgcolor == "rgba(8,23,45,0.16)"
    assert list(figure.layout.xaxis.range) == [0.0, 1920.0]
    assert list(figure.layout.yaxis.range) == [1080.0, 0.0]
    assert float(figure.layout.images[0].x) == 0.0
    assert float(figure.layout.images[0].y) == 0.0
    assert float(figure.layout.images[0].sizex) == 1920.0
    assert float(figure.layout.images[0].sizey) == 1080.0


def test_interactive_heatmap_attaches_uploaded_background(tmp_path) -> None:
    recording = analyze_recording(simulate_gaze_recording(seed=18)).enriched_recording
    background = np.zeros((24, 24, 3), dtype=float)
    background[..., 0] = 0.35
    background[..., 2] = 0.7
    background_path = tmp_path / "stimulus_heatmap.png"
    plt.imsave(background_path, background)

    figure = plot_interactive_heatmap(
        recording,
        background_image=background_path,
        screen_size=(1366, 768),
    )

    assert len(figure.layout.images) == 1
    assert str(figure.layout.images[0].source).startswith("data:image/png;base64,")
    assert figure.layout.plot_bgcolor == "rgba(8,23,45,0.12)"
    assert len(figure.data) == 1
    assert list(figure.layout.xaxis.range) == [0.0, 1366.0]
    assert list(figure.layout.yaxis.range) == [768.0, 0.0]
    assert float(figure.layout.images[0].x) == 0.0
    assert float(figure.layout.images[0].y) == 0.0
    assert float(figure.layout.images[0].sizex) == 1366.0
    assert float(figure.layout.images[0].sizey) == 768.0


def test_interactive_scanpath_without_fixations_shows_empty_state(tmp_path) -> None:
    recording = simulate_gaze_recording(seed=22)
    background = np.zeros((24, 24, 3), dtype=float)
    background[..., 1] = 0.45
    background_path = tmp_path / "scanpath_empty.png"
    plt.imsave(background_path, background)

    figure = plot_interactive_scanpath(
        recording,
        background_image=background_path,
        screen_size=(1920, 1080),
    )

    assert len(figure.data) == 0
    assert len(figure.layout.images) == 1
    assert len(figure.layout.annotations) == 1
    assert "未检测到注视事件" in str(figure.layout.annotations[0].text)


def test_interactive_heatmap_without_fixations_shows_empty_state(tmp_path) -> None:
    recording = simulate_gaze_recording(seed=28)
    background = np.zeros((24, 24, 3), dtype=float)
    background[..., 0] = 0.3
    background[..., 2] = 0.55
    background_path = tmp_path / "heatmap_empty.png"
    plt.imsave(background_path, background)

    figure = plot_interactive_heatmap(
        recording,
        background_image=background_path,
        screen_size=(1920, 1080),
    )

    assert len(figure.data) == 0
    assert len(figure.layout.images) == 1
    assert len(figure.layout.annotations) == 1
    assert "未检测到可用于热图的注视事件" in str(figure.layout.annotations[0].text)


def test_visualizations_support_light_theme() -> None:
    recording = analyze_recording(simulate_gaze_recording(seed=31)).enriched_recording

    figure, axis = plt.subplots()
    returned_scanpath = plot_scanpath(recording, ax=axis, theme_name="light")
    assert returned_scanpath.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb("#f7fbff")
    plt.close(figure)

    figure, axis = plt.subplots()
    returned_heatmap = plot_heatmap(recording, ax=axis, theme_name="light")
    assert returned_heatmap.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb("#f7fbff")
    plt.close(figure)

    scanpath = plot_interactive_scanpath(recording, theme_name="light")
    assert scanpath.layout.paper_bgcolor == "#f7fbff"
    assert scanpath.layout.font.color == "#0f2748"

    heatmap = plot_interactive_heatmap(recording, theme_name="light")
    assert heatmap.layout.paper_bgcolor == "#f7fbff"
    assert heatmap.layout.font.color == "#0f2748"

    signal_figure = plot_signal_overview(recording, theme_name="light")
    assert signal_figure.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb("#f7fbff")
    plt.close(signal_figure)


def test_interactive_visualizations_accept_palette_and_opacity_overrides() -> None:
    recording = analyze_recording(simulate_gaze_recording(seed=44)).enriched_recording

    default_scanpath = plot_interactive_scanpath(recording)
    custom_scanpath = plot_interactive_scanpath(
        recording,
        palette="sunset",
        fixation_opacity=0.55,
    )
    assert custom_scanpath.data[-1].marker.opacity == 0.55
    assert tuple(custom_scanpath.data[-1].marker.colorscale) != tuple(default_scanpath.data[-1].marker.colorscale)

    default_heatmap = plot_interactive_heatmap(recording)
    custom_heatmap = plot_interactive_heatmap(
        recording,
        palette="violet",
        heatmap_opacity=0.48,
    )
    assert custom_heatmap.data[0].opacity == 0.48
    assert tuple(custom_heatmap.data[0].colorscale) != tuple(default_heatmap.data[0].colorscale)


def test_sliding_window_predictor_emits_prediction() -> None:
    dataset = simulate_intent_dataset(num_sessions=16, random_state=4)
    result = train_model(dataset, target="intent_label")
    predictor = SlidingWindowPredictor(result, window_ms=5000.0, step_ms=0.0)
    recording = simulate_gaze_recording(seed=4)

    emitted = None
    for sample in recording.samples.to_dict("records"):
        emitted = predictor.update(sample)
        if emitted is not None:
            break

    assert emitted is not None
    assert emitted.prediction in {"careful", "skim"}
