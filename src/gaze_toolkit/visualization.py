from __future__ import annotations

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
from sklearn.metrics import ConfusionMatrixDisplay

from gaze_toolkit.types import GazeRecording


def plot_scanpath(
    recording: GazeRecording,
    background_image: str | Path | Any | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot a scanpath with time-ordered points."""
    frame = recording.samples.loc[recording.samples["valid"]]
    ax = ax or plt.subplots(figsize=(8, 5))[1]

    if background_image is not None:
        if isinstance(background_image, (str, Path)):
            image = plt.imread(Path(background_image))
        else:
            if hasattr(background_image, "seek"):
                background_image.seek(0)
            image = plt.imread(background_image)
        ax.imshow(image)

    ax.plot(frame["x"], frame["y"], color="#1f77b4", linewidth=1.5, alpha=0.8)
    scatter = ax.scatter(frame["x"], frame["y"], c=frame["timestamp_ms"], cmap="viridis", s=18)
    plt.colorbar(scatter, ax=ax, label="时间戳（ms）")
    ax.set_title("扫描路径")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    return ax


def plot_heatmap(recording: GazeRecording, bins: int = 40, ax: Any | None = None) -> Any:
    """Plot a gaze density heatmap."""
    frame = recording.samples.loc[recording.samples["valid"]]
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    heat = ax.hist2d(frame["x"], frame["y"], bins=bins, cmap="magma")
    plt.colorbar(heat[3], ax=ax, label="密度")
    ax.set_title("注视热图")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    return ax


def plot_signal_overview(recording: GazeRecording, velocity: pd.Series | None = None) -> Any:
    """Plot core gaze signals and velocity over time."""
    frame = recording.samples.copy()
    velocity = velocity if velocity is not None else pd.Series(np.zeros(len(frame)))
    figure, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(frame["timestamp_ms"], frame["x"], color="#0f4c81", linewidth=1.1)
    axes[0].set_ylabel("X")
    axes[0].set_title("信号总览")

    axes[1].plot(frame["timestamp_ms"], frame["y"], color="#a23b72", linewidth=1.1)
    axes[1].set_ylabel("Y")

    axes[2].plot(frame["timestamp_ms"], frame["pupil"], color="#4c956c", linewidth=1.1)
    axes[2].set_ylabel("Pupil")

    axes[3].plot(frame["timestamp_ms"], velocity, color="#ff7f11", linewidth=1.1)
    axes[3].set_ylabel("Velocity")
    axes[3].set_xlabel("时间戳（ms）")

    figure.tight_layout()
    return figure


def plot_feature_correlations(frame: pd.DataFrame, ax: Any | None = None) -> Any:
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
    return ax


def plot_metrics(metrics: dict[str, float], ax: Any | None = None) -> Any:
    """Plot scalar metrics as a bar chart."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    names = list(metrics)
    values = [metrics[name] for name in names]
    ax.bar(names, values, color="#2ca02c")
    ax.set_title("模型指标")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_feature_importance(
    importance: pd.DataFrame,
    top_k: int = 12,
    ax: Any | None = None,
) -> Any:
    """Plot the top feature importances."""
    subset = importance.head(top_k).iloc[::-1]
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    ax.barh(subset["feature"], subset["importance_mean"], color="#c8553d")
    ax.set_title("关键特征重要性")
    ax.set_xlabel("置换重要性")
    return ax


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, ax: Any | None = None) -> Any:
    """Plot a confusion matrix."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("混淆矩阵")
    return ax
