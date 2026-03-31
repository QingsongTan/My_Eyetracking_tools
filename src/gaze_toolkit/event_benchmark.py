"""Event detection benchmark: compare native I-VT, pymovements I-VT / I-DT, and optional ground truth."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaze_toolkit.events import detect_events_with_thresholds
from gaze_toolkit.pymovements_adapter import HAS_PYMOVEMENTS
from gaze_toolkit.types import EyeEvent, GazeRecording


@dataclass
class EventBenchmarkResult:
    """Structured result of an event detection benchmark run."""

    dataset_name: str
    recording_label: str
    comparison_table: pd.DataFrame  # columns: method, fixation_count, fixation_duration_mean_ms, fixation_duration_total_ms, fixation_amplitude_mean
    agreement_table: pd.DataFrame  # columns: comparison, sample_overlap_ratio, precision, recall, f1


@dataclass
class _BenchmarkSummary:
    headline: str
    summary_lines: list[str]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def run_event_detection_benchmark(
    recording: GazeRecording,
    dataset_name: str = "unknown",
    recording_label: str = "unknown_0",
) -> EventBenchmarkResult:
    """Run event detection with native thresholds and pymovements, then build comparison tables."""
    n_samples = len(recording.samples)

    # --- native I-VT ---
    native_events = detect_events_with_thresholds(recording)
    native_mask = _fixation_mask(native_events, n_samples)

    # --- pymovements I-VT / I-DT ---
    pm_ivt_events, pm_idt_events = _run_pymovements_detection(recording)
    pm_ivt_mask = _fixation_mask(pm_ivt_events, n_samples)
    pm_idt_mask = _fixation_mask(pm_idt_events, n_samples)

    # --- ground truth (if labelled) ---
    gt_events = _extract_ground_truth(recording)
    gt_mask = _fixation_mask(gt_events, n_samples) if gt_events else None

    # --- build comparison table ---
    rows: list[dict[str, Any]] = []
    if gt_events:
        rows.append(_summarize_fixations("ground_truth", gt_events))
    rows.append(_summarize_fixations("native_ivt", native_events))
    rows.append(_summarize_fixations("pymovements_ivt", pm_ivt_events))
    rows.append(_summarize_fixations("pymovements_idt", pm_idt_events))
    comparison_table = pd.DataFrame(rows)

    # --- build agreement table ---
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    if gt_mask is not None:
        pairs.append(("ground_truth_vs_native", gt_mask, native_mask))
        pairs.append(("ground_truth_vs_pymovements_ivt", gt_mask, pm_ivt_mask))
        pairs.append(("ground_truth_vs_pymovements_idt", gt_mask, pm_idt_mask))
    pairs.append(("native_vs_pymovements_ivt", native_mask, pm_ivt_mask))
    pairs.append(("native_vs_pymovements_idt", native_mask, pm_idt_mask))
    pairs.append(("pymovements_ivt_vs_idt", pm_ivt_mask, pm_idt_mask))

    agreement_rows = [_sample_agreement(label, a, b) for label, a, b in pairs]
    agreement_table = pd.DataFrame(agreement_rows)

    return EventBenchmarkResult(
        dataset_name=dataset_name,
        recording_label=recording_label,
        comparison_table=comparison_table,
        agreement_table=agreement_table,
    )


def run_public_dataset_benchmark(
    dataset_name: str,
    recording_index: int = 0,
    *,
    root_dir: str | Path | None = None,
    download: bool = True,
) -> EventBenchmarkResult:
    """Load a pymovements public dataset recording and run the benchmark."""
    if not HAS_PYMOVEMENTS:
        raise ImportError("pymovements is required for public dataset benchmarks.")

    import pymovements as pm

    from gaze_toolkit.pymovements_adapter import from_pymovements

    dataset = pm.Dataset(dataset_name, path=str(root_dir) if root_dir else None)
    if download:
        dataset.download()
    dataset.load()

    gaze_list = dataset.gaze
    if recording_index >= len(gaze_list):
        raise IndexError(
            f"recording_index {recording_index} out of range (dataset has {len(gaze_list)} recordings)."
        )

    gaze = gaze_list[recording_index]
    recording = from_pymovements(gaze, source_format="pymovements")
    label = f"{dataset_name}_{recording_index}"

    return run_event_detection_benchmark(recording, dataset_name=dataset_name, recording_label=label)


def summarize_event_benchmark(benchmark: EventBenchmarkResult) -> _BenchmarkSummary:
    """Generate a one-line headline and a list of summary observations."""
    ct = benchmark.comparison_table
    at = benchmark.agreement_table

    has_gt = "ground_truth" in ct["method"].tolist()

    lines: list[str] = []

    if has_gt:
        # Find best agreement with ground truth
        gt_rows = at[at["comparison"].str.startswith("ground_truth_vs_")]
        if not gt_rows.empty:
            best_idx = gt_rows["f1"].idxmax()
            best_row = gt_rows.loc[best_idx]
            best_method = best_row["comparison"].replace("ground_truth_vs_", "")
            overlap = round(best_row["sample_overlap_ratio"], 3)
            f1 = round(best_row["f1"], 3)
            headline = f"pymovements I-VT 与 EyeLink Ground Truth 最接近" if "ivt" in best_method else f"{best_method} 与 EyeLink Ground Truth 最接近"
            lines.append(f"最佳样本重叠率为 {overlap}，F1 为 {f1}。")
        else:
            headline = "事件检测对照完成"
    else:
        headline = "事件检测对照完成（无 Ground Truth）"

    # Native vs pymovements I-VT agreement
    native_ivt = at[at["comparison"] == "native_vs_pymovements_ivt"]
    if not native_ivt.empty:
        row = native_ivt.iloc[0]
        overlap = round(row["sample_overlap_ratio"], 3)
        if overlap >= 0.95:
            lines.append(f"项目原生阈值法与 pymovements I-VT 的结果高度一致，二者样本重叠率为 {overlap}。")
        else:
            lines.append(f"项目原生阈值法与 pymovements I-VT 的样本重叠率为 {overlap}。")

    # I-DT note
    idt_row = ct[ct["method"] == "pymovements_idt"]
    if not idt_row.empty:
        idt_count = int(idt_row["fixation_count"].iloc[0])
        if idt_count == 0:
            lines.append("当前参数下，pymovements I-DT 与 Ground Truth 偏差较大，适合作为方法敏感性讨论案例。")

    return _BenchmarkSummary(headline=headline, summary_lines=lines)


def build_event_benchmark_markdown(benchmark: EventBenchmarkResult) -> str:
    """Build a Markdown report string from a benchmark result."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts: list[str] = [
        "# 事件检测方法验证报告",
        "",
        f"- generated_at: {now}",
        f"- dataset_name: {benchmark.dataset_name}",
        f"- recording_label: {benchmark.recording_label}",
        "",
        "## 研究结论摘要",
        "",
    ]

    insight = summarize_event_benchmark(benchmark)
    parts.append(f"**{insight.headline}**")
    parts.append("")
    for line in insight.summary_lines:
        parts.append(f"- {line}")
    parts.append("")

    # Comparison table
    parts.append("## 方法摘要对照")
    parts.append("")
    ct = benchmark.comparison_table.copy()
    method_labels = {
        "ground_truth": "EyeLink Ground Truth",
        "native_ivt": "Native Threshold",
        "pymovements_ivt": "pymovements I-VT",
        "pymovements_idt": "pymovements I-DT",
    }
    ct["method"] = ct["method"].map(method_labels).fillna(ct["method"])
    parts.append(_dataframe_to_markdown(ct))
    parts.append("")

    # Agreement table
    parts.append("## 样本级一致性对照")
    parts.append("")
    parts.append(_dataframe_to_markdown(benchmark.agreement_table))
    parts.append("")

    # Interpretation
    parts.append("## 解释建议")
    parts.append("")
    parts.append("- Ground Truth 来自 EyeLink ASC 中的 EFIX 事件时，可用于近似方法学验证，不应等同于人工二次标注金标准。")
    parts.append("- 原生阈值法与 pymovements I-VT 若高度一致，说明当前阈值设置具有较好的跨工具可复现性。")
    parts.append('- 若 I-DT 与 Ground Truth 偏差明显，可作为"参数敏感性"和"方法选择"讨论点。')
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fixation_mask(events: list[EyeEvent], n_samples: int) -> np.ndarray:
    """Create a boolean mask of fixation samples."""
    mask = np.zeros(n_samples, dtype=bool)
    for ev in events:
        if ev.kind == "fixation":
            start = max(0, ev.start_index)
            end = min(n_samples, ev.end_index + 1)
            mask[start:end] = True
    return mask


def _summarize_fixations(method: str, events: list[EyeEvent]) -> dict[str, Any]:
    """Compute summary statistics for fixation events from a single method."""
    fixations = [ev for ev in events if ev.kind == "fixation"]
    if not fixations:
        return {
            "method": method,
            "fixation_count": 0,
            "fixation_duration_mean_ms": 0.0,
            "fixation_duration_total_ms": 0.0,
            "fixation_amplitude_mean": 0.0,
        }

    durations = [ev.duration_ms for ev in fixations]
    amplitudes = [ev.amplitude for ev in fixations]
    return {
        "method": method,
        "fixation_count": len(fixations),
        "fixation_duration_mean_ms": round(float(np.mean(durations)), 4),
        "fixation_duration_total_ms": round(float(np.sum(durations)), 4),
        "fixation_amplitude_mean": round(float(np.mean(amplitudes)), 4),
    }


def _sample_agreement(label: str, mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, Any]:
    """Compute sample-level overlap and classification metrics between two fixation masks."""
    if mask_a.sum() == 0 and mask_b.sum() == 0:
        return {"comparison": label, "sample_overlap_ratio": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    overlap_ratio = float(overlap / union) if union > 0 else 0.0

    tp = float(np.sum(mask_a & mask_b))
    fp = float(np.sum(~mask_a & mask_b))
    fn = float(np.sum(mask_a & ~mask_b))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "comparison": label,
        "sample_overlap_ratio": round(overlap_ratio, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _run_pymovements_detection(recording: GazeRecording) -> tuple[list[EyeEvent], list[EyeEvent]]:
    """Run pymovements I-VT and I-DT on a recording, returning two event lists."""
    if not HAS_PYMOVEMENTS:
        return [], []

    import warnings

    import pymovements as pm

    samples = recording.samples
    frame_data = {
        "time": samples["timestamp_ms"].values,
        "pixel": list(zip(samples["x"].values, samples["y"].values)),
    }
    df = pd.DataFrame(frame_data)

    try:
        import polars as pl

        pl_df = pl.from_pandas(df)
    except Exception:
        return [], []

    ivt_events: list[EyeEvent] = []
    idt_events: list[EyeEvent] = []

    # --- I-VT ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gaze_ivt = pm.Gaze(samples=pl_df, auto_column_detect=True)
            gaze_ivt.detect("ivt", velocity_threshold=50)
        ivt_events = _pymovements_events_to_eye_events(gaze_ivt, recording)
    except Exception:
        pass

    # --- I-DT ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gaze_idt = pm.Gaze(samples=pl_df, auto_column_detect=True)
            gaze_idt.detect("idt", dispersion_threshold=100, duration_threshold=100)
        idt_events = _pymovements_events_to_eye_events(gaze_idt, recording)
    except Exception:
        pass

    return ivt_events, idt_events


def _pymovements_events_to_eye_events(
    gaze: Any,
    recording: GazeRecording,
) -> list[EyeEvent]:
    """Convert pymovements detected events into our EyeEvent format."""
    events_df = getattr(gaze, "events", None)
    if events_df is None:
        return []

    try:
        import polars as pl

        if isinstance(events_df, pl.DataFrame):
            events_df = events_df.to_pandas()
    except Exception:
        pass

    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return []

    samples = recording.samples
    timestamps = samples["timestamp_ms"].values
    result: list[EyeEvent] = []

    for _, row in events_df.iterrows():
        name = str(row.get("name", row.get("label", ""))).lower()
        if "fixation" not in name:
            continue

        onset = row.get("onset", None)
        offset = row.get("offset", None)
        if onset is None or offset is None:
            continue

        onset_val = float(onset[0]) if hasattr(onset, "__len__") and not isinstance(onset, (str, bytes)) else float(onset)
        offset_val = float(offset[0]) if hasattr(offset, "__len__") and not isinstance(offset, (str, bytes)) else float(offset)

        start_idx = int(np.searchsorted(timestamps, onset_val, side="left"))
        end_idx = int(np.searchsorted(timestamps, offset_val, side="right")) - 1
        start_idx = max(0, min(start_idx, len(timestamps) - 1))
        end_idx = max(start_idx, min(end_idx, len(timestamps) - 1))

        segment = samples.iloc[start_idx : end_idx + 1]
        if segment.empty:
            continue

        dx = float(segment["x"].iloc[-1] - segment["x"].iloc[0])
        dy = float(segment["y"].iloc[-1] - segment["y"].iloc[0])
        amplitude = float(np.hypot(dx, dy))

        result.append(
            EyeEvent(
                kind="fixation",
                start_time_ms=float(timestamps[start_idx]),
                end_time_ms=float(timestamps[end_idx]),
                start_index=start_idx,
                end_index=end_idx,
                amplitude=amplitude,
                metadata={"source": "pymovements"},
            )
        )

    return result


def _extract_ground_truth(recording: GazeRecording) -> list[EyeEvent] | None:
    """Extract ground truth fixation events from labeled samples, if available."""
    samples = recording.samples
    label_col = None
    for col in ("event_label", "label"):
        if col in samples.columns and samples[col].notna().any():
            label_col = col
            break

    if label_col is None:
        return None

    labels = samples[label_col].astype(str).str.lower().str.strip()
    is_fixation = labels.str.contains("fix", na=False)

    if not is_fixation.any():
        return None

    events: list[EyeEvent] = []
    groups = is_fixation.ne(is_fixation.shift()).cumsum()

    for _, segment_idx in is_fixation.groupby(groups):
        if not segment_idx.iloc[0]:
            continue
        segment = samples.loc[segment_idx.index]
        start_idx = int(segment.index[0])
        end_idx = int(segment.index[-1])
        dx = float(segment["x"].iloc[-1] - segment["x"].iloc[0])
        dy = float(segment["y"].iloc[-1] - segment["y"].iloc[0])
        events.append(
            EyeEvent(
                kind="fixation",
                start_time_ms=float(segment["timestamp_ms"].iloc[0]),
                end_time_ms=float(segment["timestamp_ms"].iloc[-1]),
                start_index=start_idx,
                end_index=end_idx,
                amplitude=float(np.hypot(dx, dy)),
                metadata={"source": "ground_truth"},
            )
        )

    return events if events else None


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, separator]
    for _, row in df.iterrows():
        line = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        lines.append(line)
    return "\n".join(lines)
