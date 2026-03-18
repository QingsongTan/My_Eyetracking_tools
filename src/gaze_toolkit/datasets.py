from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from gaze_toolkit.features import extract_features
from gaze_toolkit.types import GazeRecording

ReadingStyle = Literal["careful", "skim"]

_STYLE_PROFILES = {
    "careful": {
        "fixation_ms": (180, 280),
        "saccade_ms": (20, 40),
        "jump_px": 110,
        "jitter_px": 7,
        "blink_probability": 0.08,
        "pupil_baseline": 3.4,
    },
    "skim": {
        "fixation_ms": (75, 145),
        "saccade_ms": (18, 32),
        "jump_px": 230,
        "jitter_px": 12,
        "blink_probability": 0.03,
        "pupil_baseline": 3.0,
    },
}


def simulate_gaze_recording(
    duration_ms: int = 5000,
    sampling_rate_hz: int = 120,
    style: ReadingStyle = "careful",
    seed: int | None = None,
) -> GazeRecording:
    """Generate a synthetic reading-like gaze trace."""
    rng = np.random.default_rng(seed)
    profile = _STYLE_PROFILES[style]
    dt_ms = 1000.0 / sampling_rate_hz

    current_x = 140.0
    current_y = 260.0
    current_time = 0.0
    rows: list[dict[str, float | bool]] = []

    while current_time < duration_ms:
        fixation_duration = int(rng.integers(*profile["fixation_ms"]))
        fixation_samples = max(1, int(round(fixation_duration / dt_ms)))
        for _ in range(fixation_samples):
            if current_time >= duration_ms:
                break
            rows.append(
                {
                    "timestamp_ms": current_time,
                    "x": current_x + rng.normal(0.0, profile["jitter_px"]),
                    "y": current_y + rng.normal(0.0, profile["jitter_px"] / 2.0),
                    "pupil": profile["pupil_baseline"] + rng.normal(0.0, 0.08),
                    "valid": True,
                }
            )
            current_time += dt_ms

        if current_time >= duration_ms:
            break

        if rng.random() < profile["blink_probability"]:
            blink_duration = int(rng.integers(85, 150))
            blink_samples = max(1, int(round(blink_duration / dt_ms)))
            for _ in range(blink_samples):
                if current_time >= duration_ms:
                    break
                rows.append(
                    {
                        "timestamp_ms": current_time,
                        "x": np.nan,
                        "y": np.nan,
                        "pupil": np.nan,
                        "valid": False,
                    }
                )
                current_time += dt_ms

        if current_time >= duration_ms:
            break

        next_x = float(np.clip(current_x + rng.normal(profile["jump_px"], profile["jump_px"] * 0.2), 80, 1820))
        next_y = float(
            np.clip(
                current_y + rng.normal(0.0, 20.0 if style == "careful" else 45.0),
                120,
                960,
            )
        )
        saccade_duration = int(rng.integers(*profile["saccade_ms"]))
        saccade_samples = max(2, int(round(saccade_duration / dt_ms)))
        for sample_index in range(saccade_samples):
            if current_time >= duration_ms:
                break
            alpha = (sample_index + 1) / saccade_samples
            rows.append(
                {
                    "timestamp_ms": current_time,
                    "x": current_x + alpha * (next_x - current_x),
                    "y": current_y + alpha * (next_y - current_y),
                    "pupil": profile["pupil_baseline"] + rng.normal(0.0, 0.04),
                    "valid": True,
                }
            )
            current_time += dt_ms

        current_x = next_x
        current_y = next_y

    frame = pd.DataFrame(rows).reset_index(drop=True)
    return GazeRecording(
        samples=frame,
        sampling_rate_hz=float(sampling_rate_hz),
        metadata={"intent_label": style},
        source_format="synthetic",
    )


def simulate_intent_dataset(num_sessions: int = 24, random_state: int = 42) -> pd.DataFrame:
    """Generate a labeled feature dataset for intent classification demos."""
    rows: list[dict[str, float | str | int]] = []
    for session_id, recording in enumerate(simulate_intent_recordings(num_sessions, random_state=random_state)):
        features = extract_features(recording)
        features["session_id"] = session_id
        features["intent_label"] = str(recording.metadata["intent_label"])
        rows.append(features)

    return pd.DataFrame(rows)


def simulate_intent_recordings(num_sessions: int = 24, random_state: int = 42) -> list[GazeRecording]:
    """Generate a cohort of labeled recordings for downstream experiments."""
    styles: list[ReadingStyle] = ["careful", "skim"]
    recordings: list[GazeRecording] = []

    for session_id in range(num_sessions):
        style = styles[session_id % len(styles)]
        recordings.append(simulate_gaze_recording(style=style, seed=random_state + session_id))
    return recordings


def simulate_heart_rate_signal(recording: GazeRecording, seed: int | None = None) -> pd.DataFrame:
    """Generate a low-frequency heart-rate signal aligned to a recording."""
    rng = np.random.default_rng(seed)
    timestamps = recording.samples["timestamp_ms"].iloc[:: max(len(recording.samples) // 30, 1)].to_numpy()
    base_hr = 72.0 if recording.metadata.get("intent_label") == "careful" else 78.0
    values = base_hr + 2.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(timestamps))) + rng.normal(0.0, 0.6, len(timestamps))
    return pd.DataFrame({"timestamp_ms": timestamps, "heart_rate_bpm": values})


def extract_heart_rate_features(signal: pd.DataFrame) -> dict[str, float]:
    """Extract lightweight heart-rate features for multimodal demos."""
    values = pd.to_numeric(signal["heart_rate_bpm"], errors="coerce").dropna()
    if values.empty:
        return {
            "heart_rate_mean": 0.0,
            "heart_rate_std": 0.0,
            "heart_rate_min": 0.0,
            "heart_rate_max": 0.0,
            "heart_rate_rmssd": 0.0,
        }

    diffs = values.diff().dropna()
    rmssd = float(np.sqrt(np.mean(np.square(diffs)))) if not diffs.empty else 0.0
    return {
        "heart_rate_mean": float(values.mean()),
        "heart_rate_std": float(values.std(ddof=0)),
        "heart_rate_min": float(values.min()),
        "heart_rate_max": float(values.max()),
        "heart_rate_rmssd": rmssd,
    }


def simulate_multimodal_intent_dataset(num_sessions: int = 24, random_state: int = 42) -> pd.DataFrame:
    """Generate a multimodal cohort with gaze and heart-rate summary features."""
    rows: list[dict[str, float | str | int]] = []
    recordings = simulate_intent_recordings(num_sessions=num_sessions, random_state=random_state)

    for session_id, recording in enumerate(recordings):
        row = extract_features(recording)
        heart_signal = simulate_heart_rate_signal(recording, seed=random_state + session_id)
        row.update(extract_heart_rate_features(heart_signal))
        row["session_id"] = session_id
        row["intent_label"] = str(recording.metadata["intent_label"])
        rows.append(row)

    return pd.DataFrame(rows)
