from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from gaze_toolkit.types import GazeRecording


def compute_quality_grade(recording: GazeRecording) -> str:
    """基于 tracking ratio 计算基础质量等级。"""
    tracking_ratio = float(recording.samples["valid"].mean()) if not recording.samples.empty else 0.0

    if tracking_ratio >= 0.9:
        return "优"
    if tracking_ratio >= 0.75:
        return "良"
    if tracking_ratio >= 0.5:
        return "可用"
    return "建议剔除"


@dataclass
class QualityReport:
    """数据质量完整报告。"""

    tracking_ratio: float
    total_samples: int
    valid_samples: int
    missing_segments: int
    max_gap_duration_ms: float
    interpolated_ratio: float
    blink_count: int
    recording_duration_s: float
    sampling_rate_actual: float
    quality_grade: str


def find_missing_segments(samples: pd.DataFrame) -> list[dict[str, Any]]:
    """找出样本数据中的所有连续缺失段。"""
    required_columns = {"timestamp_ms", "valid"}
    missing_columns = required_columns.difference(samples.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"samples 必须包含 timestamp_ms 和 valid 列，缺失: {missing_text}")

    if samples.empty:
        return []

    frame = samples.loc[:, ["timestamp_ms", "valid"]].copy()
    frame["timestamp_ms"] = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    frame["valid"] = frame["valid"].fillna(False).astype(bool)

    invalid_mask = ~frame["valid"]
    if not invalid_mask.any():
        return []

    segment_starts = invalid_mask & ~invalid_mask.shift(fill_value=False)
    segment_ids = segment_starts.cumsum().where(invalid_mask)
    invalid_frame = frame.loc[invalid_mask].copy()
    invalid_frame["segment_id"] = segment_ids.loc[invalid_mask].astype(int)

    segments: list[dict[str, Any]] = []
    for _, group in invalid_frame.groupby("segment_id", sort=True):
        start_index = int(group.index[0])
        end_index = int(group.index[-1])
        start_time_ms = float(group["timestamp_ms"].iloc[0])
        end_time_ms = float(group["timestamp_ms"].iloc[-1])
        duration_ms = max(0.0, end_time_ms - start_time_ms)
        segments.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "duration_ms": duration_ms,
                "sample_count": int(len(group)),
            }
        )

    return segments


def assess_quality(
    recording: GazeRecording,
    preprocessed: GazeRecording | None = None,
) -> QualityReport:
    """评估眼动记录数据质量。"""
    samples = recording.samples

    if samples.empty:
        tracking_ratio = 0.0
        total_samples = 0
        valid_samples = 0
        missing_segments_data: list[dict[str, Any]] = []
    else:
        tracking_ratio = float(samples["valid"].mean())
        total_samples = int(len(samples))
        valid_samples = int(samples["valid"].sum())
        missing_segments_data = find_missing_segments(samples)

    max_gap_duration_ms = (
        float(max(segment["duration_ms"] for segment in missing_segments_data))
        if missing_segments_data
        else 0.0
    )

    preprocessed_ratio = 0.0
    if preprocessed is not None and not preprocessed.samples.empty:
        preprocessed_ratio = float(preprocessed.samples["valid"].mean())
    interpolated_ratio = preprocessed_ratio - tracking_ratio if preprocessed is not None else 0.0

    events = recording.events or []
    blink_count = sum(1 for event in events if getattr(event, "kind", "") == "blink")

    recording_duration_s = float(recording.duration_ms) / 1000.0
    sampling_rate_actual = (
        float(total_samples - 1) / recording_duration_s
        if total_samples >= 2 and recording_duration_s > 0.0
        else 0.0
    )

    report = QualityReport(
        tracking_ratio=tracking_ratio,
        total_samples=total_samples,
        valid_samples=valid_samples,
        missing_segments=len(missing_segments_data),
        max_gap_duration_ms=max_gap_duration_ms,
        interpolated_ratio=interpolated_ratio,
        blink_count=int(blink_count),
        recording_duration_s=recording_duration_s,
        sampling_rate_actual=sampling_rate_actual,
        quality_grade=compute_quality_grade(recording),
    )

    expected_sampling_rate = recording.metadata.get("sampling_rate_hz", recording.sampling_rate_hz)
    try:
        setattr(report, "expected_sampling_rate_hz", float(expected_sampling_rate))
    except (TypeError, ValueError):
        setattr(report, "expected_sampling_rate_hz", None)
    return report


def format_quality_cards(report: QualityReport) -> list[dict[str, str]]:
    """将质量报告格式化为 Dashboard 指标卡片数据。"""

    def _status_from_ratio(value: float, good_threshold: float, warn_threshold: float) -> str:
        if value >= good_threshold:
            return "good"
        if value >= warn_threshold:
            return "warn"
        return "bad"

    def _status_from_reverse(value: float, good_threshold: float, warn_threshold: float) -> str:
        if value < good_threshold:
            return "good"
        if value <= warn_threshold:
            return "warn"
        return "bad"

    def _format_duration_ms(value_ms: float) -> str:
        if value_ms >= 1000.0:
            return f"{value_ms / 1000.0:.1f} s"
        return f"{value_ms:.0f} ms"

    expected_sampling_rate = getattr(report, "expected_sampling_rate_hz", None)
    if expected_sampling_rate is None or expected_sampling_rate <= 0:
        sampling_status = "good"
    else:
        delta_ratio = abs(report.sampling_rate_actual - expected_sampling_rate) / expected_sampling_rate
        if delta_ratio < 0.05:
            sampling_status = "good"
        elif delta_ratio <= 0.10:
            sampling_status = "warn"
        else:
            sampling_status = "bad"

    if report.quality_grade in {"优", "良"}:
        grade_status = "good"
    elif report.quality_grade == "可用":
        grade_status = "warn"
    else:
        grade_status = "bad"

    if report.missing_segments <= 2:
        missing_status = "good"
    elif report.missing_segments <= 5:
        missing_status = "warn"
    else:
        missing_status = "bad"

    if report.recording_duration_s >= 10.0:
        duration_status = "good"
    elif report.recording_duration_s >= 5.0:
        duration_status = "warn"
    else:
        duration_status = "bad"

    cards = [
        {
            "label": "追踪率",
            "value": f"{report.tracking_ratio:.1%}",
            "status": _status_from_ratio(report.tracking_ratio, 0.90, 0.75),
        },
        {
            "label": "采样率",
            "value": f"{report.sampling_rate_actual:.1f} Hz",
            "status": sampling_status,
        },
        {
            "label": "记录时长",
            "value": f"{report.recording_duration_s:.1f} s",
            "status": duration_status,
        },
        {
            "label": "缺失段",
            "value": f"{report.missing_segments} 段",
            "status": missing_status,
        },
        {
            "label": "最大缺失段",
            "value": _format_duration_ms(report.max_gap_duration_ms),
            "status": _status_from_reverse(report.max_gap_duration_ms, 500.0, 2000.0),
        },
        {
            "label": "质量等级",
            "value": report.quality_grade,
            "status": grade_status,
        },
        {
            "label": "眨眼次数",
            "value": f"{report.blink_count} 次",
            "status": "info",
        },
        {
            "label": "插值比例",
            "value": f"{report.interpolated_ratio:.1%}",
            "status": _status_from_reverse(abs(report.interpolated_ratio), 0.05, 0.15),
        },
    ]
    return cards
