from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from gaze_toolkit.errors import DataValidationError, UnsupportedFormatError
from gaze_toolkit.pymovements_adapter import load_with_pymovements
from gaze_toolkit.types import GazeRecording

LoaderFn = Callable[[Path, float | None, dict[str, str] | None], pd.DataFrame]

_LOADER_REGISTRY: dict[str, LoaderFn] = {}

_COLUMN_ALIASES = {
    "timestamp_ms": ["timestamp_ms", "timestamp", "time", "t", "sample_time"],
    "x": ["x", "gaze_x", "screen_x", "x_pos", "x_position"],
    "y": ["y", "gaze_y", "screen_y", "y_pos", "y_position"],
    "pupil": ["pupil", "pupil_size", "pupil_diameter", "diameter"],
    "valid": ["valid", "is_valid", "tracking_valid"],
    "marker": ["marker", "marker_name", "trigger", "message", "annotation", "stimulus_marker"],
    "event_label": ["event_label", "event_type", "eye_event", "gaze_event", "evt", "event"],
}


def register_loader(name: str, loader: LoaderFn) -> None:
    """Register a custom file loader."""
    _LOADER_REGISTRY[name.lower()] = loader


def load(
    path: str | Path,
    format: str | None = None,
    sampling_rate_hz: float | None = None,
    column_map: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GazeRecording:
    """Load a gaze recording and normalize its columns."""
    source = Path(path)
    source_format = (format or source.suffix.lstrip(".") or "csv").lower()
    if source_format == "auto":
        source_format = source.suffix.lstrip(".").lower() or "csv"

    if source_format in _LOADER_REGISTRY:
        frame = _LOADER_REGISTRY[source_format](source, sampling_rate_hz, column_map)
    elif source_format in {"pymovements", "pm"}:
        return load_with_pymovements(
            source,
            sampling_rate_hz=sampling_rate_hz,
            metadata=metadata,
        )
    elif source_format in {"csv", "tsv", "txt"}:
        frame = _read_tabular(source, sampling_rate_hz, column_map)
    elif source_format == "asc":
        frame = _read_eyelink_asc(source, sampling_rate_hz, column_map)
    elif source_format == "edf":
        raise UnsupportedFormatError(
            "EDF parsing is not bundled in the MVP. Convert to CSV/ASC first or register a custom loader."
        )
    else:
        raise UnsupportedFormatError(f"Unsupported format: {source_format}")

    return GazeRecording(
        samples=frame,
        sampling_rate_hz=sampling_rate_hz,
        metadata=metadata or {},
        source_format=source_format,
    )


def from_frame(
    frame: pd.DataFrame,
    sampling_rate_hz: float | None = None,
    column_map: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    source_format: str = "dataframe",
) -> GazeRecording:
    """Create a recording directly from an in-memory frame."""
    normalized = _normalize_columns(frame, sampling_rate_hz=sampling_rate_hz, column_map=column_map)
    return GazeRecording(
        samples=normalized,
        sampling_rate_hz=sampling_rate_hz,
        metadata=metadata or {},
        source_format=source_format,
    )


def _read_tabular(
    path: Path,
    sampling_rate_hz: float | None,
    column_map: dict[str, str] | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    return _normalize_columns(frame, sampling_rate_hz=sampling_rate_hz, column_map=column_map)


def _read_eyelink_asc(
    path: Path,
    sampling_rate_hz: float | None,
    column_map: dict[str, str] | None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue

            values: list[float] = []
            for token in tokens[:6]:
                try:
                    values.append(float(token))
                except ValueError:
                    break

            if len(values) >= 3:
                row = {"timestamp_ms": values[0], "x": values[1], "y": values[2]}
                if len(values) >= 4:
                    row["pupil"] = values[3]
                rows.append(row)

    if not rows:
        raise DataValidationError("No sample rows were detected in the ASC file.")

    frame = pd.DataFrame(rows)
    return _normalize_columns(frame, sampling_rate_hz=sampling_rate_hz, column_map=column_map)


def _normalize_columns(
    frame: pd.DataFrame,
    sampling_rate_hz: float | None,
    column_map: dict[str, str] | None,
) -> pd.DataFrame:
    normalized = frame.copy()
    lookup = {column.lower(): column for column in normalized.columns}

    if column_map:
        for canonical, original in column_map.items():
            if original in normalized.columns:
                normalized[canonical] = normalized[original]

    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in normalized.columns:
            continue
        for alias in aliases:
            source = lookup.get(alias.lower())
            if source is not None:
                normalized[canonical] = normalized[source]
                break

    if "timestamp_ms" not in normalized.columns:
        if sampling_rate_hz is None:
            raise DataValidationError(
                "Missing timestamp column. Provide `sampling_rate_hz` so timestamps can be synthesized."
            )
        interval_ms = 1000.0 / sampling_rate_hz
        normalized["timestamp_ms"] = normalized.index.to_series(dtype="float64") * interval_ms

    keep_columns = ["timestamp_ms", "x", "y", "pupil", "valid", "marker", "event_label", "label", "trial"]
    available = [column for column in keep_columns if column in normalized.columns]
    return normalized[available]
