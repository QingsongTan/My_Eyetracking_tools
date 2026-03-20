from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from gaze_toolkit.types import EyeEvent, GazeRecording

DEFAULT_EVENT_LABEL_MAP = {
    "fixation": "fixation",
    "fix": "fixation",
    "fix_start": "fixation",
    "fix_end": "fixation",
    "saccade": "saccade",
    "sac": "saccade",
    "sacc": "saccade",
    "saccade_start": "saccade",
    "saccade_end": "saccade",
    "blink": "blink",
    "bl": "blink",
    "blink_start": "blink",
    "blink_end": "blink",
    "smooth_pursuit": "smooth_pursuit",
    "pursuit": "smooth_pursuit",
    "sp": "smooth_pursuit",
}


def compute_velocity(recording: GazeRecording) -> pd.Series:
    """Compute point-to-point gaze velocity in px/s."""
    frame = recording.samples
    dt_seconds = frame["timestamp_ms"].diff().replace(0.0, np.nan) / 1000.0
    dx = frame["x"].diff()
    dy = frame["y"].diff()
    velocity = np.sqrt(dx.pow(2) + dy.pow(2)) / dt_seconds
    velocity = velocity.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    velocity.loc[~frame["valid"]] = 0.0
    return velocity


def detect_events(
    recording: GazeRecording,
    velocity_threshold: float = 850.0,
    min_fixation_ms: float = 60.0,
    blink_min_duration_ms: float = 75.0,
    source: str = "auto",
    label_column: str | None = None,
    label_map: Mapping[str, str] | None = None,
) -> list[EyeEvent]:
    """Detect events from original labels or custom thresholds."""
    normalized_source = source.lower()

    if normalized_source in {"auto", "labels", "original"} and has_labeled_events(recording, label_column=label_column):
        return detect_events_from_labels(
            recording,
            min_fixation_ms=min_fixation_ms,
            blink_min_duration_ms=blink_min_duration_ms,
            label_column=label_column,
            label_map=label_map,
        )

    if normalized_source in {"labels", "original"}:
        raise ValueError("Requested original-label events, but no event label column was found.")

    return detect_events_with_thresholds(
        recording,
        velocity_threshold=velocity_threshold,
        min_fixation_ms=min_fixation_ms,
        blink_min_duration_ms=blink_min_duration_ms,
    )


def detect_events_with_thresholds(
    recording: GazeRecording,
    velocity_threshold: float = 850.0,
    min_fixation_ms: float = 60.0,
    blink_min_duration_ms: float = 75.0,
) -> list[EyeEvent]:
    """Detect fixations, saccades, and blinks using a simple I-VT rule."""
    frame = recording.samples.copy()
    frame["velocity_px_s"] = compute_velocity(recording)

    labels = np.where(
        ~frame["valid"],
        "blink",
        np.where(frame["velocity_px_s"] >= velocity_threshold, "saccade", "fixation"),
    )

    return _events_from_grouped_labels(
        frame=frame,
        labels=pd.Series(labels, index=frame.index),
        min_fixation_ms=min_fixation_ms,
        blink_min_duration_ms=blink_min_duration_ms,
    )


def detect_events_from_labels(
    recording: GazeRecording,
    min_fixation_ms: float = 60.0,
    blink_min_duration_ms: float = 75.0,
    label_column: str | None = None,
    label_map: Mapping[str, str] | None = None,
    apply_duration_filters: bool = False,
) -> list[EyeEvent]:
    """Build events from row-level original eye-tracker event labels."""
    frame = recording.samples.copy()
    frame["velocity_px_s"] = compute_velocity(recording)
    resolved_column = _resolve_label_column(frame, requested=label_column)
    if resolved_column is None:
        return []

    normalized_labels = frame[resolved_column].map(lambda value: _normalize_event_label(value, label_map=label_map))
    return _events_from_grouped_labels(
        frame=frame,
        labels=normalized_labels,
        min_fixation_ms=min_fixation_ms if apply_duration_filters else 0.0,
        blink_min_duration_ms=blink_min_duration_ms if apply_duration_filters else 0.0,
    )


def has_labeled_events(recording: GazeRecording, label_column: str | None = None) -> bool:
    """Return whether the recording contains usable sample-level event labels."""
    resolved = _resolve_label_column(recording.samples, requested=label_column)
    if resolved is None:
        return False
    labels = recording.samples[resolved].map(lambda value: _normalize_event_label(value))
    return labels.notna().any()


def attach_events(recording: GazeRecording, **kwargs: Any) -> GazeRecording:
    """Return a copy of the recording with detected events attached."""
    return recording.with_events(detect_events(recording, **kwargs))


def _events_from_grouped_labels(
    frame: pd.DataFrame,
    labels: pd.Series,
    min_fixation_ms: float,
    blink_min_duration_ms: float,
) -> list[EyeEvent]:
    grouped_labels = labels.where(labels.notna(), other=pd.NA)
    group_keys = grouped_labels.fillna("__missing__")
    groups = group_keys.ne(group_keys.shift()).cumsum()
    events: list[EyeEvent] = []

    for _, segment in frame.groupby(groups):
        kind = grouped_labels.loc[segment.index[0]]
        if pd.isna(kind):
            continue

        start_index = int(segment.index[0])
        end_index = int(segment.index[-1])
        start_time = float(segment["timestamp_ms"].iloc[0])
        end_time = float(segment["timestamp_ms"].iloc[-1])
        duration_ms = end_time - start_time

        if kind == "fixation" and duration_ms < min_fixation_ms:
            continue
        if kind == "blink" and duration_ms < blink_min_duration_ms:
            continue

        events.append(_build_event(segment, kind=str(kind)))

    return events


def _build_event(segment: pd.DataFrame, kind: str) -> EyeEvent:
    start_index = int(segment.index[0])
    end_index = int(segment.index[-1])
    start_time = float(segment["timestamp_ms"].iloc[0])
    end_time = float(segment["timestamp_ms"].iloc[-1])
    peak_velocity = float(segment.get("velocity_px_s", pd.Series([0.0])).max())

    amplitude = 0.0
    centroid_x = np.nan
    centroid_y = np.nan
    if kind != "blink":
        dx = float(segment["x"].iloc[-1] - segment["x"].iloc[0])
        dy = float(segment["y"].iloc[-1] - segment["y"].iloc[0])
        amplitude = math.hypot(dx, dy)
        centroid_x = float(segment["x"].mean())
        centroid_y = float(segment["y"].mean())

    return EyeEvent(
        kind=kind,
        start_time_ms=start_time,
        end_time_ms=end_time,
        start_index=start_index,
        end_index=end_index,
        amplitude=amplitude,
        peak_velocity=peak_velocity,
        metadata={
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "source_label": segment.get("event_label", pd.Series([pd.NA])).iloc[0] if "event_label" in segment.columns else pd.NA,
        },
    )


def _resolve_label_column(frame: pd.DataFrame, requested: str | None = None) -> str | None:
    if requested is not None and requested in frame.columns:
        return requested
    for candidate in ("event_label", "label"):
        if candidate in frame.columns:
            return candidate
    return None


def _normalize_event_label(value: Any, label_map: Mapping[str, str] | None = None) -> str | None:
    if value is None or pd.isna(value):
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    lower = normalized.lower()
    mapping = {**DEFAULT_EVENT_LABEL_MAP, **{str(key).lower(): val for key, val in (label_map or {}).items()}}
    if lower in mapping:
        return mapping[lower]

    for alias, target in mapping.items():
        if alias in lower:
            return target

    return lower
