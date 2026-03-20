from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gaze_toolkit.types import GazeRecording


@dataclass
class RecordingSegment:
    """A named segment extracted from a recording."""

    name: str
    recording: GazeRecording
    start_time_ms: float
    end_time_ms: float


def segment_recording(
    recording: GazeRecording,
    strategy: str = "whole",
    time_ranges: list[tuple[float, float]] | None = None,
    marker_values: list[str] | None = None,
    pre_ms: float = 0.0,
    post_ms: float = 0.0,
    start_marker: str | None = None,
    end_marker: str | None = None,
    marker_column: str = "marker",
) -> list[RecordingSegment]:
    """Segment a recording by whole file, time ranges, or markers."""
    normalized = strategy.lower()
    if normalized == "whole":
        return [_slice_segment(recording, "whole_file", recording.samples["timestamp_ms"].min(), recording.samples["timestamp_ms"].max())]
    if normalized in {"time_ranges", "time"}:
        if not time_ranges:
            raise ValueError("`time_ranges` is required for time-range segmentation.")
        return [
            _slice_segment(recording, f"time_{index}", start_ms, end_ms)
            for index, (start_ms, end_ms) in enumerate(time_ranges, start=1)
        ]
    if normalized in {"marker_windows", "marker_window"}:
        return segment_by_marker_windows(
            recording,
            marker_values=marker_values,
            pre_ms=pre_ms,
            post_ms=post_ms,
            marker_column=marker_column,
        )
    if normalized in {"between_markers", "marker_pairs"}:
        if start_marker is None or end_marker is None:
            raise ValueError("`start_marker` and `end_marker` are required for between-marker segmentation.")
        return segment_between_markers(
            recording,
            start_marker=start_marker,
            end_marker=end_marker,
            marker_column=marker_column,
        )
    raise ValueError(f"Unsupported segmentation strategy: {strategy}")


def segment_by_time_ranges(
    recording: GazeRecording,
    time_ranges: list[tuple[float, float]],
) -> list[RecordingSegment]:
    """Slice a recording by explicit time ranges."""
    return [
        _slice_segment(recording, f"time_{index}", start_ms, end_ms)
        for index, (start_ms, end_ms) in enumerate(time_ranges, start=1)
    ]


def segment_by_marker_windows(
    recording: GazeRecording,
    marker_values: list[str] | None = None,
    pre_ms: float = 0.0,
    post_ms: float = 0.0,
    marker_column: str = "marker",
) -> list[RecordingSegment]:
    """Segment around each matching marker with a pre/post window."""
    markers = _marker_rows(recording, marker_column=marker_column)
    if marker_values is not None:
        normalized_values = {str(value) for value in marker_values}
        markers = markers.loc[markers[marker_column].astype(str).isin(normalized_values)]

    segments: list[RecordingSegment] = []
    for index, row in enumerate(markers.itertuples(index=False), start=1):
        marker_time = float(getattr(row, "timestamp_ms"))
        marker_value = str(getattr(row, marker_column))
        segments.append(
            _slice_segment(
                recording,
                f"marker_window_{index}_{marker_value}",
                marker_time - pre_ms,
                marker_time + post_ms,
                metadata={"marker_value": marker_value},
            )
        )
    return segments


def segment_between_markers(
    recording: GazeRecording,
    start_marker: str,
    end_marker: str,
    marker_column: str = "marker",
) -> list[RecordingSegment]:
    """Segment from each start marker to the next matching end marker."""
    markers = _marker_rows(recording, marker_column=marker_column).reset_index(drop=True)
    starts = markers.loc[markers[marker_column].astype(str) == str(start_marker)]
    ends = markers.loc[markers[marker_column].astype(str) == str(end_marker)]

    segments: list[RecordingSegment] = []
    for index, start_row in enumerate(starts.itertuples(index=False), start=1):
        start_time = float(getattr(start_row, "timestamp_ms"))
        candidate_end = ends.loc[ends["timestamp_ms"] > start_time, "timestamp_ms"]
        if candidate_end.empty:
            continue
        end_time = float(candidate_end.iloc[0])
        segments.append(
            _slice_segment(
                recording,
                f"between_{index}_{start_marker}_to_{end_marker}",
                start_time,
                end_time,
                metadata={"start_marker": start_marker, "end_marker": end_marker},
            )
        )
    return segments


def build_segment_feature_table(
    segments: list[RecordingSegment],
    preprocess_fn: Any,
    attach_events_fn: Any,
    extract_features_fn: Any,
    preprocess_params: dict[str, Any] | None = None,
    event_params: dict[str, Any] | None = None,
    feature_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run preprocessing and feature extraction for a list of segments."""
    preprocess_params = preprocess_params or {}
    event_params = event_params or {}
    feature_params = feature_params or {}

    rows: list[dict[str, Any]] = []
    for segment in segments:
        processed = preprocess_fn(segment.recording, **preprocess_params)
        enriched = attach_events_fn(processed, **event_params)
        feature_map = extract_features_fn(enriched, **feature_params)
        feature_map["segment_name"] = segment.name
        feature_map["segment_start_ms"] = segment.start_time_ms
        feature_map["segment_end_ms"] = segment.end_time_ms
        rows.append(feature_map)
    return pd.DataFrame(rows)


def _marker_rows(recording: GazeRecording, marker_column: str) -> pd.DataFrame:
    if marker_column not in recording.samples.columns:
        raise ValueError(f"Marker column not found: {marker_column}")
    markers = recording.samples.copy()
    marker_series = markers[marker_column]
    return markers.loc[marker_series.notna() & (marker_series.astype(str).str.strip() != "")]


def _slice_segment(
    recording: GazeRecording,
    name: str,
    start_time_ms: float,
    end_time_ms: float,
    metadata: dict[str, Any] | None = None,
) -> RecordingSegment:
    bounded_start = max(float(start_time_ms), float(recording.samples["timestamp_ms"].min()))
    bounded_end = min(float(end_time_ms), float(recording.samples["timestamp_ms"].max()))
    if bounded_end < bounded_start:
        bounded_end = bounded_start

    subset = recording.samples.loc[
        (recording.samples["timestamp_ms"] >= bounded_start)
        & (recording.samples["timestamp_ms"] <= bounded_end)
    ].reset_index(drop=True)

    segment_recording = GazeRecording(
        samples=subset,
        sampling_rate_hz=recording.sampling_rate_hz,
        metadata={
            **recording.metadata,
            "segment_name": name,
            "segment_start_ms": bounded_start,
            "segment_end_ms": bounded_end,
            **(metadata or {}),
        },
        source_format=recording.source_format,
    )
    return RecordingSegment(
        name=name,
        recording=segment_recording,
        start_time_ms=bounded_start,
        end_time_ms=bounded_end,
    )
