from __future__ import annotations

import pandas as pd

from gaze_toolkit.quality import compute_quality_grade as compute_quality_grade
from gaze_toolkit.types import GazeRecording

FIXATION_TABLE_COLUMNS = [
    "event_index",
    "start_time_ms",
    "end_time_ms",
    "duration_ms",
    "centroid_x",
    "centroid_y",
    "session_id",
    "subject_id",
    "condition",
    "trial",
]


def fixation_table(recording: GazeRecording) -> pd.DataFrame:
    """将 fixation 事件整理为统一的公共表结构。"""
    rows: list[dict[str, object]] = []
    metadata = recording.metadata

    for event_index, event in enumerate(recording.events):
        if event.kind != "fixation":
            continue

        rows.append(
            {
                "event_index": event_index,
                "start_time_ms": event.start_time_ms,
                "end_time_ms": event.end_time_ms,
                "duration_ms": event.duration_ms,
                "centroid_x": event.metadata.get("centroid_x"),
                "centroid_y": event.metadata.get("centroid_y"),
                "session_id": metadata.get("session_id"),
                "subject_id": metadata.get("subject_id"),
                "condition": metadata.get("condition"),
                "trial": metadata.get("trial"),
            }
        )

    return pd.DataFrame(rows, columns=FIXATION_TABLE_COLUMNS)
