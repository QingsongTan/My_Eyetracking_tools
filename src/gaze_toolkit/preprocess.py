from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np
import pandas as pd

from gaze_toolkit.errors import OptionalDependencyError
from gaze_toolkit.types import GazeRecording


def interpolate_missing(
    recording: GazeRecording,
    method: str = "linear",
    columns: Iterable[str] = ("x", "y", "pupil"),
) -> GazeRecording:
    """Interpolate missing values while preserving original validity flags."""
    clone = recording.copy()
    invalid_mask = ~clone.samples["valid"]
    for column in columns:
        if column not in clone.samples.columns:
            continue
        series = clone.samples[column].mask(invalid_mask)
        clone.samples[column] = series.interpolate(method=method, limit_direction="both")
    return clone


def drop_invalid_samples(recording: GazeRecording) -> GazeRecording:
    """Remove invalid samples entirely."""
    clone = recording.copy()
    clone.samples = clone.samples.loc[clone.samples["valid"]].reset_index(drop=True)
    clone.validate()
    return clone


def handle_missing_samples(
    recording: GazeRecording,
    strategy: str = "interpolate",
    interpolation_method: str = "linear",
    columns: Iterable[str] = ("x", "y", "pupil"),
) -> GazeRecording:
    """Resolve NaN and invalid samples according to the chosen strategy."""
    normalized_strategy = strategy.lower()
    if normalized_strategy == "interpolate":
        return interpolate_missing(recording, method=interpolation_method, columns=columns)
    if normalized_strategy in {"drop", "clean"}:
        return drop_invalid_samples(recording)
    if normalized_strategy in {"keep", "none"}:
        return recording.copy()
    raise ValueError(f"Unsupported missing data strategy: {strategy}")


def smooth_signal(
    recording: GazeRecording,
    method: str = "moving_average",
    window: int = 5,
    columns: Iterable[str] = ("x", "y", "pupil"),
    polyorder: int = 2,
) -> GazeRecording:
    """Apply a smoothing operator to numeric columns."""
    clone = recording.copy()
    for column in columns:
        if column not in clone.samples.columns:
            continue
        series = clone.samples[column]
        if method == "moving_average":
            clone.samples[column] = series.rolling(window=window, center=True, min_periods=1).mean()
        elif method == "savitzky_golay":
            try:
                scipy_signal = importlib.import_module("scipy.signal", package=None)
            except ModuleNotFoundError as exc:
                raise OptionalDependencyError("Install gaze-toolkit[signal] to use Savitzky-Golay smoothing.")
            window_length = max(window, polyorder + 2)
            if window_length % 2 == 0:
                window_length += 1
            clone.samples[column] = scipy_signal.savgol_filter(
                series.to_numpy(),
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            )
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")
    return clone


def normalize_coordinates(
    recording: GazeRecording,
    bounds: tuple[float, float, float, float] | None = None,
    centered: bool = True,
) -> GazeRecording:
    """Normalize coordinates to 0-1 or -1 to 1 space."""
    clone = recording.copy()
    if bounds is None:
        x_min = float(clone.samples["x"].min())
        x_max = float(clone.samples["x"].max())
        y_min = float(clone.samples["y"].min())
        y_max = float(clone.samples["y"].max())
    else:
        x_min, x_max, y_min, y_max = bounds

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    clone.samples["x_norm"] = (clone.samples["x"] - x_min) / x_span
    clone.samples["y_norm"] = (clone.samples["y"] - y_min) / y_span

    if centered:
        clone.samples["x_norm"] = clone.samples["x_norm"] * 2.0 - 1.0
        clone.samples["y_norm"] = clone.samples["y_norm"] * 2.0 - 1.0

    return clone


def preprocess(
    recording: GazeRecording,
    missing_strategy: str = "interpolate",
    interpolation_method: str = "linear",
    smooth_method: str = "moving_average",
    smooth_window: int = 5,
    normalize_coordinates_flag: bool = True,
) -> GazeRecording:
    """Run the default preprocessing pipeline."""
    processed = handle_missing_samples(
        recording,
        strategy=missing_strategy,
        interpolation_method=interpolation_method,
    )
    processed = smooth_signal(processed, method=smooth_method, window=smooth_window)
    if normalize_coordinates_flag:
        processed = normalize_coordinates(processed)
    return processed
