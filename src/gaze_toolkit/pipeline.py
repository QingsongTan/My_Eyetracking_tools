from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gaze_toolkit.config import load_config, merge_config
from gaze_toolkit.events import attach_events
from gaze_toolkit.features import extract_features
from gaze_toolkit.io import load
from gaze_toolkit.preprocess import preprocess
from gaze_toolkit.types import GazeRecording


def build_feature_dataset(recordings: list[GazeRecording], target_key: str = "intent_label") -> pd.DataFrame:
    """Convert multiple recordings into a feature matrix."""
    rows: list[dict[str, Any]] = []
    for index, recording in enumerate(recordings):
        processed = preprocess(recording)
        enriched = attach_events(processed)
        row = extract_features(enriched)
        row["session_id"] = index
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
    )

    feature_config = effective.get("features", {})
    return extract_features(
        enriched,
        window_ms=float(feature_config.get("window_ms", 500.0)),
        include_complexity=bool(feature_config.get("include_complexity", True)),
    )

