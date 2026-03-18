from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gaze_toolkit.types import EyeEvent, GazeRecording


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
) -> list[EyeEvent]:
    """Detect fixations, saccades, and blinks using a simple I-VT rule."""
    frame = recording.samples.copy()
    frame["velocity_px_s"] = compute_velocity(recording)

    labels = np.where(
        ~frame["valid"],
        "blink",
        np.where(frame["velocity_px_s"] >= velocity_threshold, "saccade", "fixation"),
    )

    groups = pd.Series(labels).ne(pd.Series(labels).shift()).cumsum()
    events: list[EyeEvent] = []

    for _, segment in frame.groupby(groups):
        kind = labels[segment.index[0]]
        start_index = int(segment.index[0])
        end_index = int(segment.index[-1])
        start_time = float(segment["timestamp_ms"].iloc[0])
        end_time = float(segment["timestamp_ms"].iloc[-1])
        duration_ms = end_time - start_time

        if kind == "fixation" and duration_ms < min_fixation_ms:
            continue
        if kind == "blink" and duration_ms < blink_min_duration_ms:
            continue

        amplitude = 0.0
        peak_velocity = float(segment["velocity_px_s"].max())
        if kind != "blink":
            dx = float(segment["x"].iloc[-1] - segment["x"].iloc[0])
            dy = float(segment["y"].iloc[-1] - segment["y"].iloc[0])
            amplitude = math.hypot(dx, dy)

        events.append(
            EyeEvent(
                kind=kind,
                start_time_ms=start_time,
                end_time_ms=end_time,
                start_index=start_index,
                end_index=end_index,
                amplitude=amplitude,
                peak_velocity=peak_velocity,
                metadata={
                    "centroid_x": float(segment["x"].mean()) if kind != "blink" else np.nan,
                    "centroid_y": float(segment["y"].mean()) if kind != "blink" else np.nan,
                },
            )
        )

    return events


def attach_events(recording: GazeRecording, **kwargs: float) -> GazeRecording:
    """Return a copy of the recording with detected events attached."""
    return recording.with_events(detect_events(recording, **kwargs))

