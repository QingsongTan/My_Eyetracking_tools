from __future__ import annotations

import base64
import importlib
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay

from gaze_toolkit.types import EyeEvent, GazeRecording

DARK_VISUAL_THEME: dict[str, Any] = {
    "plot_template": "plotly_dark",
    "panel_bg": "#08172d",
    "panel_grid": "rgba(0, 243, 255, 0.12)",
    "panel_grid_mpl": (0 / 255, 243 / 255, 255 / 255, 0.12),
    "panel_text": "#eaf7ff",
    "panel_muted": "#9bc6de",
    "panel_line": "rgba(0, 243, 255, 0.44)",
    "scanpath_colorscale": [
        [0.0, "#103a6a"],
        [0.28, "#0f7bd8"],
        [0.58, "#00F3FF"],
        [0.82, "#00FF9D"],
        [1.0, "#e9fff8"],
    ],
    "heatmap_colorscale": [
        [0.00, "rgba(8,23,45,0.00)"],
        [0.08, "rgba(8,23,45,0.10)"],
        [0.20, "#0f2b50"],
        [0.42, "#1266b5"],
        [0.64, "#00F3FF"],
        [0.84, "#00FF9D"],
        [1.00, "#effffd"],
    ],
    "scanpath_bg_overlay": "rgba(8,23,45,0.16)",
    "heatmap_bg_overlay": "rgba(8,23,45,0.12)",
    "annotation_bg": "rgba(8,23,45,0.78)",
    "annotation_border": "rgba(234,247,255,0.18)",
    "marker_line": "rgba(234,247,255,0.90)",
    "marker_text": "white",
    "series_x": "#7d9eff",
    "series_y": "#6fe0b6",
    "series_pupil": "#ffb55a",
    "series_velocity": "#ff6a6a",
    "metric_bar": "#2ca02c",
    "feature_bar": "#00c8ff",
}

LIGHT_VISUAL_THEME: dict[str, Any] = {
    "plot_template": "plotly_white",
    "panel_bg": "#f7fbff",
    "panel_grid": "rgba(17, 111, 178, 0.14)",
    "panel_grid_mpl": (17 / 255, 111 / 255, 178 / 255, 0.14),
    "panel_text": "#0f2748",
    "panel_muted": "#5f7a94",
    "panel_line": "rgba(0, 184, 235, 0.42)",
    "scanpath_colorscale": [
        [0.0, "#dbeeff"],
        [0.28, "#8ccfff"],
        [0.58, "#00d9ff"],
        [0.82, "#00d991"],
        [1.0, "#123256"],
    ],
    "heatmap_colorscale": [
        [0.00, "rgba(247,251,255,0.00)"],
        [0.08, "rgba(247,251,255,0.28)"],
        [0.20, "#dcefff"],
        [0.42, "#94d2ff"],
        [0.64, "#18d9ff"],
        [0.84, "#00e89a"],
        [1.00, "#123256"],
    ],
    "scanpath_bg_overlay": "rgba(255,255,255,0.72)",
    "heatmap_bg_overlay": "rgba(255,255,255,0.76)",
    "annotation_bg": "rgba(255,255,255,0.94)",
    "annotation_border": "rgba(17, 111, 178, 0.18)",
    "marker_line": "rgba(18,50,86,0.26)",
    "marker_text": "#0f2748",
    "series_x": "#4284ff",
    "series_y": "#00b38f",
    "series_pupil": "#ff9e2c",
    "series_velocity": "#ff5b7f",
    "metric_bar": "#10b981",
    "feature_bar": "#00bfe8",
}

SCANPATH_COLOR_PRESETS: dict[str, list[list[Any]]] = {
    "aurora": [
        [0.0, "#173d7a"],
        [0.35, "#00c2ff"],
        [0.7, "#00f0b5"],
        [1.0, "#e8fff9"],
    ],
    "glacier": [
        [0.0, "#dceeff"],
        [0.3, "#7cc7ff"],
        [0.7, "#3af1ff"],
        [1.0, "#0c355d"],
    ],
    "sunset": [
        [0.0, "#53264d"],
        [0.32, "#ff7b54"],
        [0.68, "#ffb74d"],
        [1.0, "#fff0d6"],
    ],
    "violet": [
        [0.0, "#251554"],
        [0.34, "#7446ff"],
        [0.7, "#ff5edb"],
        [1.0, "#ffe6fb"],
    ],
}

HEATMAP_COLOR_PRESETS: dict[str, list[list[Any]]] = {
    "aurora": [
        [0.00, "rgba(8,23,45,0.00)"],
        [0.18, "#123d7a"],
        [0.44, "#00b8ff"],
        [0.72, "#00f2c3"],
        [1.00, "#f4fff9"],
    ],
    "glacier": [
        [0.00, "rgba(247,251,255,0.00)"],
        [0.18, "#dceeff"],
        [0.44, "#89cbff"],
        [0.72, "#30e9ff"],
        [1.00, "#0d365d"],
    ],
    "sunset": [
        [0.00, "rgba(44,20,35,0.00)"],
        [0.18, "#5b2c51"],
        [0.44, "#ff7b54"],
        [0.72, "#ffb74d"],
        [1.00, "#fff1d7"],
    ],
    "violet": [
        [0.00, "rgba(24,12,54,0.00)"],
        [0.18, "#2c1766"],
        [0.44, "#7446ff"],
        [0.72, "#ff5edb"],
        [1.00, "#ffe9fd"],
    ],
}


def _visual_theme(theme_name: str = "dark") -> dict[str, Any]:
    return LIGHT_VISUAL_THEME if theme_name == "light" else DARK_VISUAL_THEME


def _resolve_scanpath_colorscale(theme_name: str, palette: str) -> list[list[Any]]:
    if palette == "theme_default":
        return _visual_theme(theme_name)["scanpath_colorscale"]
    return SCANPATH_COLOR_PRESETS.get(palette, _visual_theme(theme_name)["scanpath_colorscale"])


def _resolve_heatmap_colorscale(theme_name: str, palette: str) -> list[list[Any]]:
    if palette == "theme_default":
        return _visual_theme(theme_name)["heatmap_colorscale"]
    return HEATMAP_COLOR_PRESETS.get(palette, _visual_theme(theme_name)["heatmap_colorscale"])


def _apply_matplotlib_panel_style(
    figure: Any,
    axes: Any,
    *,
    theme_name: str = "dark",
    title_color: str | None = None,
    grid_alpha: float = 0.14,
) -> None:
    """Apply the active panel theme to one or more Matplotlib axes."""
    theme = _visual_theme(theme_name)
    title_color = title_color or theme["panel_text"]
    figure.patch.set_facecolor(theme["panel_bg"])
    figure.patch.set_edgecolor(theme["panel_bg"])
    axis_list = list(np.atleast_1d(axes))
    for axis in axis_list:
        axis.set_facecolor(theme["panel_bg"])
        axis.tick_params(colors=theme["panel_muted"])
        axis.xaxis.label.set_color(theme["panel_text"])
        axis.yaxis.label.set_color(theme["panel_text"])
        axis.title.set_color(title_color)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.grid(alpha=grid_alpha, color=theme["panel_grid_mpl"])


def _style_matplotlib_colorbar(colorbar: Any, theme: dict[str, Any], *, label: str | None = None) -> None:
    """Apply the active theme to a Matplotlib colorbar."""
    if label is not None:
        colorbar.set_label(label, color=theme["panel_text"])
    colorbar.ax.tick_params(colors=theme["panel_muted"])
    colorbar.outline.set_visible(False)
    colorbar.ax.set_facecolor(theme["panel_bg"])


def plot_scanpath(
    recording: GazeRecording,
    background_image: str | Path | Any | None = None,
    ax: Any | None = None,
    theme_name: str = "dark",
) -> Any:
    """Plot a scanpath with time-ordered points using Matplotlib."""
    theme = _visual_theme(theme_name)
    frame = _valid_frame(recording)
    ax = ax or plt.subplots(figsize=(8, 5))[1]

    if background_image is not None:
        if isinstance(background_image, (str, Path)):
            image = plt.imread(Path(background_image))
        else:
            if hasattr(background_image, "seek"):
                background_image.seek(0)
            image = plt.imread(background_image)
        ax.imshow(image)

    ax.set_facecolor(theme["panel_bg"])
    ax.plot(frame["x"], frame["y"], color="#45a8ff", linewidth=1.5, alpha=0.78)
    scatter = ax.scatter(frame["x"], frame["y"], c=frame["timestamp_ms"], cmap="viridis", s=18)
    colorbar = plt.colorbar(scatter, ax=ax)
    _style_matplotlib_colorbar(colorbar, theme, label="时间戳（ms）")
    ax.set_title("扫描路径轨迹")
    ax.set_xlabel("Screen X (px)")
    ax.set_ylabel("Screen Y (px)")
    ax.tick_params(colors=theme["panel_muted"])
    ax.xaxis.label.set_color(theme["panel_text"])
    ax.yaxis.label.set_color(theme["panel_text"])
    ax.title.set_color(theme["panel_text"])
    ax.grid(False)
    ax.xaxis.grid(True, alpha=0.18, color=theme["panel_grid_mpl"])
    ax.invert_yaxis()
    return ax


def plot_heatmap(
    recording: GazeRecording,
    bins: int = 40,
    ax: Any | None = None,
    theme_name: str = "dark",
) -> Any:
    """Plot a gaze density heatmap using Matplotlib."""
    theme = _visual_theme(theme_name)
    frame = _valid_frame(recording)
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    ax.set_facecolor(theme["panel_bg"])
    heat = ax.hist2d(frame["x"], frame["y"], bins=bins, cmap="magma")
    colorbar = plt.colorbar(heat[3], ax=ax)
    _style_matplotlib_colorbar(colorbar, theme, label="密度")
    ax.set_title("核密度注意力热力图")
    ax.set_xlabel("Screen X")
    ax.set_ylabel("Screen Y")
    ax.tick_params(colors=theme["panel_muted"])
    ax.xaxis.label.set_color(theme["panel_text"])
    ax.yaxis.label.set_color(theme["panel_text"])
    ax.title.set_color(theme["panel_text"])
    ax.grid(False)
    ax.xaxis.grid(True, alpha=0.10, color=theme["panel_grid_mpl"])
    ax.invert_yaxis()
    return ax


def plot_interactive_scanpath(
    recording: GazeRecording,
    background_image: str | Path | Any | None = None,
    screen_size: tuple[int, int] | None = None,
    figure_height: int = 420,
    theme_name: str = "dark",
    palette: str = "theme_default",
    fixation_opacity: float = 0.72,
) -> go.Figure:
    """Plot a fixation-centered interactive scanpath."""
    theme = _visual_theme(theme_name)
    scanpath_colorscale = _resolve_scanpath_colorscale(theme_name, palette)
    fixation_opacity = float(np.clip(fixation_opacity, 0.15, 1.0))
    line_opacity = float(np.clip(fixation_opacity * 0.8, 0.18, 0.92))
    valid_frame = _valid_frame(recording)
    fixations = _fixation_frame(recording)
    figure = go.Figure()
    background = _load_background_image(background_image)
    x_range, y_range = _resolve_plot_ranges(
        valid_frame,
        screen_size=screen_size,
        use_screen_space=background is not None,
    )

    if fixations.empty:
        _attach_background_image(figure, background=background, x_range=x_range, y_range=y_range, opacity=0.68)
        figure.update_layout(
            template=theme["plot_template"],
            paper_bgcolor=theme["panel_bg"],
            plot_bgcolor=theme["scanpath_bg_overlay"] if background is not None else theme["panel_bg"],
            title="交互式眼动扫描路径 (Scanpath)",
            height=figure_height,
            margin={"l": 28, "r": 20, "t": 50, "b": 30},
            font={"color": theme["panel_text"], "family": "Aptos, Microsoft YaHei, sans-serif"},
            xaxis={
                "title": "Screen X (px)",
                "gridcolor": theme["panel_grid"],
                "zeroline": False,
                "showline": False,
                "range": x_range,
            },
            yaxis={
                "title": "Screen Y (px)",
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "range": y_range,
            },
        )
        _add_empty_state_annotation(figure, "当前配置下未检测到注视事件", theme_name=theme_name)
        return figure

    figure.add_trace(
        go.Scatter(
            x=fixations["x"],
            y=fixations["y"],
            mode="lines",
            line={"color": _with_alpha(theme["panel_line"], line_opacity), "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=fixations["x"],
            y=fixations["y"],
            mode="markers+text",
            text=fixations["order"].astype(str),
            textposition="middle center",
            textfont={"color": theme["marker_text"], "size": 11},
            marker={
                "size": fixations["marker_size"],
                "color": fixations["progress"],
                "colorscale": scanpath_colorscale,
                "cmin": 0.0,
                "cmax": 1.0,
                "showscale": True,
                "colorbar": {
                    "title": {"text": "时序进度", "font": {"color": theme["panel_text"]}},
                    "tickcolor": theme["panel_muted"],
                    "tickfont": {"color": theme["panel_text"]},
                    "thickness": 18,
                },
                "line": {"color": theme["marker_line"], "width": 1.2},
                "opacity": fixation_opacity,
            },
            customdata=fixations[["duration_ms", "start_time_ms", "end_time_ms"]],
            hovertemplate=(
                "注视 %{text}<br>"
                "X=%{x:.1f}<br>"
                "Y=%{y:.1f}<br>"
                "持续=%{customdata[0]:.0f} ms<br>"
                "开始=%{customdata[1]:.0f} ms<br>"
                "结束=%{customdata[2]:.0f} ms<extra></extra>"
            ),
            showlegend=False,
        )
    )

    _attach_background_image(figure, background=background, x_range=x_range, y_range=y_range, opacity=0.68)

    figure.update_layout(
        title="交互式眼动扫描路径 (Scanpath)",
        height=figure_height,
        template=theme["plot_template"],
        paper_bgcolor=theme["panel_bg"],
        plot_bgcolor=theme["scanpath_bg_overlay"] if background is not None else theme["panel_bg"],
        margin={"l": 28, "r": 20, "t": 50, "b": 30},
        font={"color": theme["panel_text"], "family": "Aptos, Microsoft YaHei, sans-serif"},
        xaxis={
            "title": "Screen X (px)",
            "gridcolor": theme["panel_grid"],
            "zeroline": False,
            "showline": False,
            "range": x_range,
        },
        yaxis={
            "title": "Screen Y (px)",
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "range": y_range,
        },
    )
    return figure


def plot_interactive_heatmap(
    recording: GazeRecording,
    background_image: str | Path | Any | None = None,
    screen_size: tuple[int, int] | None = None,
    bins: int = 140,
    figure_height: int = 420,
    theme_name: str = "dark",
    palette: str = "theme_default",
    heatmap_opacity: float = 0.62,
) -> go.Figure:
    """Plot a smoothed density heatmap with a dark interactive style."""
    theme = _visual_theme(theme_name)
    heatmap_colorscale = _resolve_heatmap_colorscale(theme_name, palette)
    heatmap_opacity = float(np.clip(heatmap_opacity, 0.15, 1.0))
    frame = _valid_frame(recording)
    figure = go.Figure()
    background = _load_background_image(background_image)
    x_range, y_range = _resolve_plot_ranges(
        frame,
        screen_size=screen_size,
        use_screen_space=background is not None,
    )

    if frame.empty:
        _attach_background_image(figure, background=background, x_range=x_range, y_range=y_range, opacity=0.82)
        figure.update_layout(
            template=theme["plot_template"],
            paper_bgcolor=theme["panel_bg"],
            plot_bgcolor=theme["heatmap_bg_overlay"] if background is not None else theme["panel_bg"],
            title="注意力核密度热力图 (Attention KDE Heatmap)",
            height=figure_height,
            margin={"l": 28, "r": 20, "t": 50, "b": 30},
            font={"color": theme["panel_text"], "family": "Aptos, Microsoft YaHei, sans-serif"},
            xaxis={
                "title": "Screen X",
                "gridcolor": theme["panel_grid"],
                "zeroline": False,
                "showline": False,
                "range": x_range,
            },
            yaxis={
                "title": "Screen Y",
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "range": y_range,
            },
        )
        _add_empty_state_annotation(figure, "当前记录中没有可用于热图的有效样本", theme_name=theme_name)
        return figure

    x_centers = np.linspace(min(x_range), max(x_range), bins)
    y_centers = np.linspace(min(y_range), max(y_range), bins)
    density = _build_attention_density(recording, x_centers=x_centers, y_centers=y_centers)
    if not np.any(density > 0):
        _attach_background_image(figure, background=background, x_range=x_range, y_range=y_range, opacity=0.82)
        figure.update_layout(
            title="娉ㄦ剰鍔涙牳瀵嗗害鐑姏鍥?(Attention KDE Heatmap)",
            height=figure_height,
            template=theme["plot_template"],
            paper_bgcolor=theme["panel_bg"],
            plot_bgcolor=theme["heatmap_bg_overlay"] if background is not None else theme["panel_bg"],
            margin={"l": 28, "r": 20, "t": 50, "b": 30},
            font={"color": theme["panel_text"], "family": "Aptos, Microsoft YaHei, sans-serif"},
            xaxis={
                "title": "Screen X",
                "gridcolor": theme["panel_grid"],
                "zeroline": False,
                "showline": False,
                "range": x_range,
            },
            yaxis={
                "title": "Screen Y",
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "range": y_range,
            },
        )
        _add_empty_state_annotation(figure, "当前配置下未检测到可用于热图的注视事件", theme_name=theme_name)
        return figure
    density = _smooth_density(density, passes=5)
    positive = density[density > 0]
    if positive.size:
        density[density < np.percentile(positive, 18)] = 0.0
    zmax = max(float(np.percentile(density[density > 0], 99.5)) if np.any(density > 0) else 1.0, 1e-6)

    figure.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=density,
            colorscale=heatmap_colorscale,
            zmin=0.0,
            zmax=zmax,
            zsmooth="best",
            opacity=heatmap_opacity,
            hovertemplate="Screen X=%{x:.0f}<br>Screen Y=%{y:.0f}<br>密度=%{z:.3f}<extra></extra>",
            showscale=False,
        )
    )

    _attach_background_image(figure, background=background, x_range=x_range, y_range=y_range, opacity=0.82)

    figure.update_layout(
        title="注意力核密度热力图 (Attention KDE Heatmap)",
        height=figure_height,
        template=theme["plot_template"],
        paper_bgcolor=theme["panel_bg"],
        plot_bgcolor=theme["heatmap_bg_overlay"] if background is not None else theme["panel_bg"],
        margin={"l": 28, "r": 20, "t": 50, "b": 30},
        font={"color": theme["panel_text"], "family": "Aptos, Microsoft YaHei, sans-serif"},
        xaxis={
            "title": "Screen X",
            "gridcolor": theme["panel_grid"],
            "zeroline": False,
            "showline": False,
            "range": x_range,
        },
        yaxis={
            "title": "Screen Y",
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "range": y_range,
        },
    )
    return figure


def plot_image_saliency_heatmap(
    saliency_map: np.ndarray,
    background_image: str | Path | Any | None = None,
    screen_size: tuple[int, int] | None = None,
    figure_height: int = 420,
    theme_name: str = "dark",
    palette: str = "theme_default",
    heatmap_opacity: float = 0.62,
    title: str = "图片快速显著性热力图 (OpenCV Fast Saliency)",
) -> go.Figure:
    """Render an image-derived saliency map as an interactive heatmap."""
    theme = _visual_theme(theme_name)
    heatmap_colorscale = _resolve_heatmap_colorscale(theme_name, palette)
    heatmap_opacity = float(np.clip(heatmap_opacity, 0.15, 1.0))
    density = np.nan_to_num(np.asarray(saliency_map, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if density.ndim != 2:
        raise ValueError("saliency_map must be a 2D array.")

    figure = go.Figure()
    background = _load_background_image(background_image)
    map_height, map_width = density.shape
    plot_size = screen_size or (map_width, map_height)
    x_range = [0.0, float(plot_size[0])]
    y_range = [float(plot_size[1]), 0.0]

    if not np.any(density > 0):
        _attach_background_image(figure, background=background, x_range=x_range, y_range=y_range, opacity=0.82)
        figure.update_layout(
            title=title,
            height=figure_height,
            template=theme["plot_template"],
            paper_bgcolor=theme["panel_bg"],
            plot_bgcolor=theme["heatmap_bg_overlay"] if background is not None else theme["panel_bg"],
            margin={"l": 28, "r": 20, "t": 50, "b": 30},
            font={"color": theme["panel_text"], "family": "Aptos, Microsoft YaHei, sans-serif"},
            xaxis={
                "title": "Screen X",
                "gridcolor": theme["panel_grid"],
                "zeroline": False,
                "showline": False,
                "range": x_range,
            },
            yaxis={
                "title": "Screen Y",
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "range": y_range,
            },
        )
        _add_empty_state_annotation(figure, "当前图片没有生成可用的显著性密度图", theme_name=theme_name)
        return figure

    x_centers = np.linspace(0.0, float(plot_size[0]), density.shape[1])
    y_centers = np.linspace(0.0, float(plot_size[1]), density.shape[0])
    positive = density[density > 0]
    zmax = max(float(np.percentile(positive, 99.5)) if positive.size else 1.0, 1e-6)

    figure.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=density,
            colorscale=heatmap_colorscale,
            zmin=0.0,
            zmax=zmax,
            zsmooth="best",
            opacity=heatmap_opacity,
            hovertemplate="Screen X=%{x:.0f}<br>Screen Y=%{y:.0f}<br>显著性=%{z:.3f}<extra></extra>",
            showscale=False,
        )
    )

    _attach_background_image(figure, background=background, x_range=x_range, y_range=y_range, opacity=0.82)

    figure.update_layout(
        title=title,
        height=figure_height,
        template=theme["plot_template"],
        paper_bgcolor=theme["panel_bg"],
        plot_bgcolor=theme["heatmap_bg_overlay"] if background is not None else theme["panel_bg"],
        margin={"l": 28, "r": 20, "t": 50, "b": 30},
        font={"color": theme["panel_text"], "family": "Aptos, Microsoft YaHei, sans-serif"},
        xaxis={
            "title": "Screen X",
            "gridcolor": theme["panel_grid"],
            "zeroline": False,
            "showline": False,
            "range": x_range,
        },
        yaxis={
            "title": "Screen Y",
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "range": y_range,
        },
    )
    return figure


def plot_signal_overview(
    recording: GazeRecording,
    velocity: pd.Series | None = None,
    theme_name: str = "dark",
) -> Any:
    """Plot core gaze signals and velocity over time."""
    theme = _visual_theme(theme_name)
    frame = recording.samples.copy()
    velocity = velocity if velocity is not None else pd.Series(np.zeros(len(frame)))
    figure, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(frame["timestamp_ms"], frame["x"], color=theme["series_x"], linewidth=1.1)
    axes[0].set_ylabel("X")
    axes[0].set_title("核心信号总览")

    axes[1].plot(frame["timestamp_ms"], frame["y"], color=theme["series_y"], linewidth=1.1)
    axes[1].set_ylabel("Y")

    axes[2].plot(frame["timestamp_ms"], frame["pupil"], color=theme["series_pupil"], linewidth=1.1)
    axes[2].set_ylabel("Pupil")

    axes[3].plot(frame["timestamp_ms"], velocity, color=theme["series_velocity"], linewidth=1.1)
    axes[3].set_ylabel("Velocity")
    axes[3].set_xlabel("时间戳（ms）")

    _apply_matplotlib_panel_style(figure, axes, theme_name=theme_name, grid_alpha=0.12)
    figure.tight_layout()
    return figure


def plot_feature_correlations(frame: pd.DataFrame, ax: Any | None = None, theme_name: str = "dark") -> Any:
    """Plot a correlation matrix for numeric features."""
    numeric = frame.select_dtypes(include=[np.number])
    corr = numeric.corr()
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("特征相关性")
    _apply_matplotlib_panel_style(ax.figure, ax, theme_name=theme_name, grid_alpha=0.0)
    return ax


def plot_metrics(metrics: dict[str, float], ax: Any | None = None, theme_name: str = "dark") -> Any:
    """Plot scalar metrics as a bar chart."""
    theme = _visual_theme(theme_name)
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    names = list(metrics)
    values = [metrics[name] for name in names]
    ax.bar(names, values, color=theme["metric_bar"])
    ax.set_title("模型指标")
    ax.tick_params(axis="x", rotation=45)
    _apply_matplotlib_panel_style(ax.figure, ax, theme_name=theme_name, grid_alpha=0.10)
    return ax


def plot_feature_importance(
    importance: pd.DataFrame,
    top_k: int = 12,
    ax: Any | None = None,
    theme_name: str = "dark",
) -> Any:
    """Plot the top feature importances."""
    theme = _visual_theme(theme_name)
    subset = importance.head(top_k).iloc[::-1]
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    ax.barh(subset["feature"], subset["importance_mean"], color=theme["feature_bar"])
    ax.set_title("关键特征重要性")
    ax.set_xlabel("置换重要性")
    _apply_matplotlib_panel_style(ax.figure, ax, theme_name=theme_name, grid_alpha=0.10)
    return ax


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Any | None = None,
    theme_name: str = "dark",
) -> Any:
    """Plot a confusion matrix."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("混淆矩阵")
    _apply_matplotlib_panel_style(ax.figure, ax, theme_name=theme_name, grid_alpha=0.0)
    return ax


def _valid_frame(recording: GazeRecording) -> pd.DataFrame:
    return recording.samples.loc[recording.samples["valid"]].copy()


def _fixation_frame(recording: GazeRecording) -> pd.DataFrame:
    fixations = [event for event in recording.events if event.kind == "fixation"]
    if fixations:
        frame = pd.DataFrame(
            {
                "order": np.arange(1, len(fixations) + 1),
                "x": [float(event.metadata.get("centroid_x", np.nan)) for event in fixations],
                "y": [float(event.metadata.get("centroid_y", np.nan)) for event in fixations],
                "duration_ms": [event.duration_ms for event in fixations],
                "start_time_ms": [event.start_time_ms for event in fixations],
                "end_time_ms": [event.end_time_ms for event in fixations],
            }
        ).dropna(subset=["x", "y"])
    else:
        return pd.DataFrame(columns=["order", "x", "y", "duration_ms", "start_time_ms", "end_time_ms"])

    if frame.empty:
        return frame

    duration_scale = frame["duration_ms"].clip(lower=1.0)
    marker_size = 18.0 + 24.0 * (duration_scale / duration_scale.max())
    progress = np.linspace(0.0, 1.0, len(frame))
    frame["marker_size"] = marker_size
    frame["progress"] = progress
    return frame


def _heatmap_seed_frame(recording: GazeRecording) -> pd.DataFrame:
    fixations = _fixation_frame(recording)
    if not fixations.empty:
        seed = fixations[["x", "y", "duration_ms"]].copy()
        median_duration = max(float(seed["duration_ms"].median()), 40.0)
        seed["weight"] = seed["duration_ms"].clip(lower=40.0) / median_duration
        return seed[["x", "y", "weight"]]
    return pd.DataFrame(columns=["x", "y", "weight"])


def _axis_ranges(frame: pd.DataFrame) -> tuple[list[float], list[float]]:
    x_min = float(frame["x"].min())
    x_max = float(frame["x"].max())
    y_min = float(frame["y"].min())
    y_max = float(frame["y"].max())
    x_pad = max((x_max - x_min) * 0.18, 120.0)
    y_pad = max((y_max - y_min) * 0.20, 120.0)
    return [x_min - x_pad, x_max + x_pad], [y_max + y_pad, y_min - y_pad]


def _resolve_plot_ranges(
    frame: pd.DataFrame,
    screen_size: tuple[int, int] | None = None,
    use_screen_space: bool = False,
) -> tuple[list[float], list[float]]:
    if use_screen_space and screen_size is not None:
        screen_width, screen_height = screen_size
        return [0.0, float(screen_width)], [float(screen_height), 0.0]
    if frame.empty:
        if screen_size is not None:
            screen_width, screen_height = screen_size
            return [0.0, float(screen_width)], [float(screen_height), 0.0]
        return [0.0, 1.0], [1.0, 0.0]
    return _axis_ranges(frame)


def _attach_background_image(
    figure: go.Figure,
    background: str | None,
    x_range: list[float],
    y_range: list[float],
    opacity: float,
) -> None:
    if background is None:
        return
    figure.add_layout_image(
        dict(
            source=background,
            xref="x",
            yref="y",
            x=min(x_range),
            y=min(y_range),
            sizex=max(x_range) - min(x_range),
            sizey=max(y_range) - min(y_range),
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            opacity=opacity,
            layer="below",
        )
    )


def _add_empty_state_annotation(figure: go.Figure, message: str, theme_name: str = "dark") -> None:
    theme = _visual_theme(theme_name)
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font={"size": 14, "color": theme["panel_text"]},
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderpad=8,
    )


def _smooth_density(values: np.ndarray, passes: int = 3) -> np.ndarray:
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel = kernel / kernel.sum()
    smoothed = values.astype(float, copy=True)

    for _ in range(passes):
        smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=0, arr=smoothed)
        smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=smoothed)
    return smoothed


def _build_attention_density(
    recording: GazeRecording,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> np.ndarray:
    seeds = _heatmap_seed_frame(recording)
    density = np.zeros((len(y_centers), len(x_centers)), dtype=float)
    if seeds.empty:
        return density

    x_span = max(float(x_centers.max() - x_centers.min()), 1.0)
    y_span = max(float(y_centers.max() - y_centers.min()), 1.0)
    sigma_x = max(x_span * 0.032, 24.0)
    sigma_y = max(y_span * 0.032, 24.0)
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    for seed in seeds.itertuples(index=False):
        exponent = (
            ((x_grid - float(seed.x)) ** 2) / (2.0 * sigma_x ** 2)
            + ((y_grid - float(seed.y)) ** 2) / (2.0 * sigma_y ** 2)
        )
        density += float(seed.weight) * np.exp(-exponent)

    peak = float(density.max()) if density.size else 0.0
    if peak > 0.0:
        density = density / peak
    return density


def _load_background_image(background_image: str | Path | Any | None) -> str | None:
    if background_image is None:
        return None

    try:
        pil_image = importlib.import_module("PIL.Image")
    except ModuleNotFoundError:
        return None

    if isinstance(background_image, (str, Path)):
        image = pil_image.open(Path(background_image))
    else:
        if hasattr(background_image, "seek"):
            background_image.seek(0)
        image = pil_image.open(background_image)

    image = image.convert("RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _with_alpha(color: str, alpha: float) -> str:
    value = color.strip()
    if value.startswith("rgba("):
        channels = [component.strip() for component in value[5:-1].split(",")]
        red, green, blue = [int(float(component)) for component in channels[:3]]
        return f"rgba({red}, {green}, {blue}, {alpha:.3f})"
    if value.startswith("rgb("):
        channels = [component.strip() for component in value[4:-1].split(",")]
        red, green, blue = [int(float(component)) for component in channels[:3]]
        return f"rgba({red}, {green}, {blue}, {alpha:.3f})"

    red_f, green_f, blue_f = matplotlib.colors.to_rgb(value)
    red = int(round(red_f * 255))
    green = int(round(green_f * 255))
    blue = int(round(blue_f * 255))
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"
