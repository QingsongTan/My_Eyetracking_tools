from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pathlib import Path

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


def create_complete_example_recording(
    sampling_rate_hz: int = 120,
    seed: int = 2026,
) -> GazeRecording:
    """Build a dashboard-ready example recording with markers, labels, and NaN blink spans."""
    rng = np.random.default_rng(seed)
    dt_ms = 1000.0 / sampling_rate_hz
    current_time = 0.0
    current_x = 280.0
    current_y = 300.0
    rows: list[dict[str, float | bool | int | str | None]] = []

    def append_segment(
        kind: str,
        duration_ms: float,
        target_x: float | None = None,
        target_y: float | None = None,
        pupil_baseline: float = 3.35,
        jitter_px: float = 5.0,
        marker: str | None = None,
        trial: int = 0,
    ) -> None:
        nonlocal current_time, current_x, current_y

        samples = max(1, int(round(duration_ms / dt_ms)))
        start_x = current_x
        start_y = current_y
        next_x = current_x if target_x is None else float(target_x)
        next_y = current_y if target_y is None else float(target_y)

        for sample_index in range(samples):
            row_marker = marker if sample_index == 0 and marker else None
            if kind == "blink":
                rows.append(
                    {
                        "timestamp_ms": round(current_time, 3),
                        "x": np.nan,
                        "y": np.nan,
                        "pupil": np.nan,
                        "valid": False,
                        "marker": row_marker,
                        "event_label": "blink",
                        "label": "blink",
                        "trial": trial,
                    }
                )
            elif kind == "saccade":
                alpha = (sample_index + 1) / samples
                rows.append(
                    {
                        "timestamp_ms": round(current_time, 3),
                        "x": round(start_x + alpha * (next_x - start_x), 3),
                        "y": round(start_y + alpha * (next_y - start_y), 3),
                        "pupil": round(pupil_baseline + rng.normal(0.0, 0.03), 3),
                        "valid": True,
                        "marker": row_marker,
                        "event_label": "saccade",
                        "label": "saccade",
                        "trial": trial,
                    }
                )
            elif kind == "smooth_pursuit":
                alpha = (sample_index + 1) / samples
                rows.append(
                    {
                        "timestamp_ms": round(current_time, 3),
                        "x": round(start_x + alpha * (next_x - start_x) + rng.normal(0.0, 1.2), 3),
                        "y": round(start_y + alpha * (next_y - start_y) + rng.normal(0.0, 0.8), 3),
                        "pupil": round(pupil_baseline + rng.normal(0.0, 0.035), 3),
                        "valid": True,
                        "marker": row_marker,
                        "event_label": "smooth_pursuit",
                        "label": "smooth_pursuit",
                        "trial": trial,
                    }
                )
            else:
                rows.append(
                    {
                        "timestamp_ms": round(current_time, 3),
                        "x": round(next_x + rng.normal(0.0, jitter_px), 3),
                        "y": round(next_y + rng.normal(0.0, jitter_px * 0.55), 3),
                        "pupil": round(pupil_baseline + rng.normal(0.0, 0.06), 3),
                        "valid": True,
                        "marker": row_marker,
                        "event_label": "fixation",
                        "label": "fixation",
                        "trial": trial,
                    }
                )
            current_time += dt_ms

        if kind != "blink":
            current_x = next_x
            current_y = next_y

    append_segment("fixation", 800, target_x=280, target_y=300, marker="baseline_start", trial=0)
    append_segment("fixation", 450, target_x=360, target_y=300, marker="baseline_end", trial=0)

    append_segment("saccade", 36, target_x=520, target_y=320, marker="trial1_start", trial=1)
    append_segment("fixation", 520, target_x=520, target_y=320, marker="cue_A", trial=1)
    append_segment("blink", 110, marker="blink_1", trial=1)
    append_segment("fixation", 640, target_x=560, target_y=336, marker="pre_response_A", trial=1)
    append_segment("saccade", 28, target_x=760, target_y=360, trial=1)
    append_segment("fixation", 460, target_x=760, target_y=360, marker="response_A", trial=1)
    append_segment("smooth_pursuit", 300, target_x=880, target_y=382, marker="tracking_A", trial=1)
    append_segment("fixation", 420, target_x=900, target_y=386, marker="trial1_end", trial=1)

    append_segment("saccade", 34, target_x=480, target_y=520, marker="trial2_start", trial=2)
    append_segment("fixation", 500, target_x=480, target_y=520, marker="cue_B", trial=2)
    append_segment("blink", 120, marker="blink_2", trial=2)
    append_segment("fixation", 620, target_x=540, target_y=530, marker="pre_response_B", trial=2)
    append_segment("saccade", 24, target_x=820, target_y=548, trial=2)
    append_segment("fixation", 480, target_x=820, target_y=548, marker="response_B", trial=2)
    append_segment("smooth_pursuit", 280, target_x=940, target_y=560, marker="tracking_B", trial=2)
    append_segment("fixation", 430, target_x=960, target_y=564, marker="trial2_end", trial=2)
    append_segment("fixation", 450, target_x=980, target_y=566, marker="task_end", trial=2)

    frame = pd.DataFrame(rows)
    return GazeRecording(
        samples=frame,
        sampling_rate_hz=float(sampling_rate_hz),
        metadata={
            "intent_label": "careful",
            "example_name": "complete_eye_tracking_example",
            "description": "Includes markers, row-level event labels, blink NaN spans, and trial ids.",
        },
        source_format="synthetic_complete_example",
    )


def write_complete_example_csv(
    output_path: str | Path,
    sampling_rate_hz: int = 120,
    seed: int = 2026,
) -> Path:
    """Write the complete example recording to a CSV file and return the output path."""
    recording = create_complete_example_recording(sampling_rate_hz=sampling_rate_hz, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    recording.samples.to_csv(output, index=False, encoding="utf-8")
    return output
