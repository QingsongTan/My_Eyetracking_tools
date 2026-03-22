from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gaze_toolkit.config import load_config, merge_config
from gaze_toolkit.events import attach_events
from gaze_toolkit.features import extract_features
from gaze_toolkit.io import load
from gaze_toolkit.preprocess import preprocess
from gaze_toolkit.segmentation import build_segment_feature_table, segment_recording
from gaze_toolkit.tables import compute_quality_grade
from gaze_toolkit.types import GazeRecording


def build_feature_dataset(recordings: list[GazeRecording], target_key: str = "intent_label") -> pd.DataFrame:
    """Convert multiple recordings into a feature matrix."""
    rows: list[dict[str, Any]] = []
    for index, recording in enumerate(recordings):
        processed = preprocess(recording)
        enriched = attach_events(processed)
        row = extract_features(enriched)
        metadata = recording.metadata
        row["session_id"] = metadata.get("session_id", index)
        row["subject_id"] = metadata.get("subject_id")
        row["condition"] = metadata.get("condition")
        row["trial"] = metadata.get("trial")
        row["quality_grade"] = metadata.get("quality_grade", compute_quality_grade(recording))
        row["segment_name"] = metadata.get("segment_name")
        if target_key in recording.metadata:
            row[target_key] = recording.metadata[target_key]
        rows.append(row)
    return pd.DataFrame(rows)


def run_pipeline(
    input_path: str | Path,
    config: str | Path | dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run load -> preprocess -> detect events -> extract features from config."""
    base_config: dict[str, Any] = {}
    if isinstance(config, (str, Path)):
        base_config = load_config(config)
    elif isinstance(config, dict):
        base_config = dict(config)

    effective = merge_config(base_config, overrides or {})

    io_config = effective.get("io", {})
    recording = load(
        input_path,
        format=io_config.get("format"),
        sampling_rate_hz=io_config.get("sampling_rate_hz"),
    )

    preprocess_config = effective.get("preprocess", {})
    processed = preprocess(
        recording,
        missing_strategy=preprocess_config.get("missing_strategy", "interpolate"),
        interpolation_method=preprocess_config.get("interpolation_method", "linear"),
        smooth_method=preprocess_config.get("smooth_method", "moving_average"),
        smooth_window=int(preprocess_config.get("smooth_window", 5)),
        normalize_coordinates_flag=bool(preprocess_config.get("normalize_coordinates", True)),
    )

    event_config = effective.get("events", {})
    enriched = attach_events(
        processed,
        velocity_threshold=float(event_config.get("velocity_threshold", 850.0)),
        min_fixation_ms=float(event_config.get("min_fixation_ms", 60.0)),
        blink_min_duration_ms=float(event_config.get("blink_min_duration_ms", 75.0)),
        source=event_config.get("source", "auto"),
        label_column=event_config.get("label_column"),
    )

    feature_config = effective.get("features", {})
    return extract_features(
        enriched,
        window_ms=float(feature_config.get("window_ms", 500.0)),
        include_complexity=bool(feature_config.get("include_complexity", True)),
    )


def run_segmented_pipeline(
    input_path: str | Path,
    segmentation: dict[str, Any],
    config: str | Path | dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run preprocess/events/features over one or more segments."""
    base_config: dict[str, Any] = {}
    if isinstance(config, (str, Path)):
        base_config = load_config(config)
    elif isinstance(config, dict):
        base_config = dict(config)

    effective = merge_config(base_config, overrides or {})
    io_config = effective.get("io", {})
    recording = load(
        input_path,
        format=io_config.get("format"),
        sampling_rate_hz=io_config.get("sampling_rate_hz"),
    )

    segments = segment_recording(recording, **segmentation)
    preprocess_config = effective.get("preprocess", {})
    event_config = effective.get("events", {})
    feature_config = effective.get("features", {})

    return build_segment_feature_table(
        segments,
        preprocess_fn=preprocess,
        attach_events_fn=attach_events,
        extract_features_fn=extract_features,
        preprocess_params={
            "missing_strategy": preprocess_config.get("missing_strategy", "interpolate"),
            "interpolation_method": preprocess_config.get("interpolation_method", "linear"),
            "smooth_method": preprocess_config.get("smooth_method", "moving_average"),
            "smooth_window": int(preprocess_config.get("smooth_window", 5)),
            "normalize_coordinates_flag": bool(preprocess_config.get("normalize_coordinates", True)),
        },
        event_params={
            "velocity_threshold": float(event_config.get("velocity_threshold", 850.0)),
            "min_fixation_ms": float(event_config.get("min_fixation_ms", 60.0)),
            "blink_min_duration_ms": float(event_config.get("blink_min_duration_ms", 75.0)),
            "source": event_config.get("source", "auto"),
            "label_column": event_config.get("label_column"),
        },
        feature_params={
            "window_ms": float(feature_config.get("window_ms", 500.0)),
            "include_complexity": bool(feature_config.get("include_complexity", True)),
        },
    )
