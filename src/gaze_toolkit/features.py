from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from gaze_toolkit.events import compute_velocity, detect_events
from gaze_toolkit.types import EyeEvent, GazeRecording

FeatureFn = Callable[[GazeRecording], dict[str, float]]

_FEATURE_REGISTRY: dict[str, FeatureFn] = {}


def register_feature(name: str, fn: FeatureFn) -> None:
    """Register a custom feature function."""
    _FEATURE_REGISTRY[name] = fn


def extract_features(
    recording: GazeRecording,
    window_ms: float = 500.0,
    include_complexity: bool = True,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract a portfolio-friendly baseline feature set."""
    frame = recording.samples.copy()
    valid = frame.loc[frame["valid"]].reset_index(drop=True)
    velocity = compute_velocity(recording)
    events = recording.events or detect_events(recording)

    features: dict[str, float] = {
        "duration_ms": float(recording.duration_ms),
        "sample_count": float(len(frame)),
        "valid_ratio": float(frame["valid"].mean()),
        "path_length": float(
            np.sqrt(frame["x"].diff().pow(2) + frame["y"].diff().pow(2)).fillna(0.0).sum()
        ),
        "velocity_mean": float(velocity.mean()),
        "velocity_peak": float(velocity.max()),
    }

    _add_event_features(features, events, recording.duration_ms)
    _add_signal_statistics(features, valid, window_ms=window_ms, include_complexity=include_complexity)

    pupil = valid["pupil"].dropna()
    if not pupil.empty:
        baseline_window = max(1, int(len(pupil) * 0.1))
        features["pupil_baseline"] = float(pupil.iloc[:baseline_window].mean())
        time_delta = valid["timestamp_ms"].diff().replace(0.0, np.nan) / 1000.0
        pupil_rate = valid["pupil"].diff().abs() / time_delta
        features["pupil_change_rate"] = float(pupil_rate.replace([np.inf, -np.inf], np.nan).mean())
    else:
        features["pupil_baseline"] = 0.0
        features["pupil_change_rate"] = 0.0

    registered_names = feature_names or list(_FEATURE_REGISTRY)
    for name in registered_names:
        fn = _FEATURE_REGISTRY[name]
        features.update(fn(recording))

    return {key: _nan_safe(value) for key, value in features.items()}


def _add_event_features(features: dict[str, float], events: list[EyeEvent], duration_ms: float) -> None:
    by_kind = {
        "fixation": [event for event in events if event.kind == "fixation"],
        "saccade": [event for event in events if event.kind == "saccade"],
        "blink": [event for event in events if event.kind == "blink"],
    }

    fixations = by_kind["fixation"]
    saccades = by_kind["saccade"]
    blinks = by_kind["blink"]
    duration_seconds = max(duration_ms / 1000.0, 1e-6)

    features["fixation_count"] = float(len(fixations))
    features["fixation_duration_mean"] = _mean(event.duration_ms for event in fixations)
    features["fixation_duration_total"] = float(sum(event.duration_ms for event in fixations))
    features["fixation_density"] = float(len(fixations) / duration_seconds)

    features["saccade_count"] = float(len(saccades))
    features["saccade_amplitude_mean"] = _mean(event.amplitude for event in saccades)
    features["saccade_peak_velocity_mean"] = _mean(event.peak_velocity for event in saccades)
    features["saccade_latency_mean"] = _mean(
        saccades[index].start_time_ms - fixations[index].end_time_ms
        for index in range(min(len(saccades), len(fixations)))
    )

    features["blink_count"] = float(len(blinks))
    features["blink_rate_hz"] = float(len(blinks) / duration_seconds)
    features["blink_duration_mean"] = _mean(event.duration_ms for event in blinks)


def _add_signal_statistics(
    features: dict[str, float],
    frame: pd.DataFrame,
    window_ms: float,
    include_complexity: bool,
) -> None:
    if frame.empty:
        return

    median_dt_ms = float(frame["timestamp_ms"].diff().dropna().median()) if len(frame) > 1 else window_ms
    window_samples = max(int(round(window_ms / max(median_dt_ms, 1.0))), 2)

    for column in ("x", "y", "pupil"):
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if series.empty:
            continue
        features[f"{column}_mean"] = float(series.mean())
        features[f"{column}_std"] = float(series.std(ddof=0))
        features[f"{column}_min"] = float(series.min())
        features[f"{column}_max"] = float(series.max())
        features[f"{column}_q25"] = float(series.quantile(0.25))
        features[f"{column}_q75"] = float(series.quantile(0.75))
        features[f"{column}_skew"] = float(series.skew())
        features[f"{column}_kurtosis"] = float(series.kurtosis())

        rolling_mean = series.rolling(window_samples, min_periods=1).mean()
        rolling_std = series.rolling(window_samples, min_periods=1).std().fillna(0.0)
        features[f"{column}_rolling_mean_std"] = float(rolling_mean.std(ddof=0))
        features[f"{column}_rolling_std_mean"] = float(rolling_std.mean())

        if include_complexity and len(series) >= 12:
            features[f"{column}_approx_entropy"] = float(approximate_entropy(series.to_numpy()))


def approximate_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: float | None = None,
    max_samples: int = 2000,
) -> float:
    """Compute approximate entropy for a one-dimensional signal."""
    values = np.asarray(signal, dtype=float)
    values = values[~np.isnan(values)]
    if max_samples > 0 and len(values) > max_samples:
        values = _downsample_evenly(values, target_size=max_samples)
    if len(values) <= m + 1:
        return 0.0

    tolerance = r if r is not None else 0.2 * np.std(values)
    tolerance = max(float(tolerance), 1e-6)

    def _phi(order: int) -> float:
        vectors = np.array([values[index : index + order] for index in range(len(values) - order + 1)])
        distances = np.max(np.abs(vectors[:, None] - vectors[None, :]), axis=2)
        counts = np.mean(distances <= tolerance, axis=0)
        counts = np.clip(counts, 1e-12, None)
        return float(np.mean(np.log(counts)))

    return float(_phi(m) - _phi(m + 1))


def _downsample_evenly(values: np.ndarray, target_size: int) -> np.ndarray:
    """按等间隔保留样本，避免超长序列在复杂度特征上 OOM。"""
    if target_size <= 0 or len(values) <= target_size:
        return values

    step = len(values) / float(target_size)
    indices = np.floor(np.arange(target_size, dtype=float) * step).astype(int)
    indices[-1] = len(values) - 1
    return values[indices]


def _mean(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return float(np.mean(collected))


def _nan_safe(value: float) -> float:
    return 0.0 if pd.isna(value) or np.isinf(value) else float(value)
