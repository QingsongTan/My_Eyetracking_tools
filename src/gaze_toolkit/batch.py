"""批量分析与报告导出模块。"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from gaze_toolkit.analysis import analyze_recording
from gaze_toolkit.aoi import assign_fixations_to_aoi, compute_aoi_metrics
from gaze_toolkit.config import load_config, merge_config
from gaze_toolkit.io import load
from gaze_toolkit.tables import compute_quality_grade, fixation_table
from gaze_toolkit.types import GazeRecording

KEY_METRICS = [
    "fixation_count",
    "fixation_duration_mean",
    "saccade_count",
    "valid_ratio",
    "path_length",
]
AOI_METRIC_SUFFIXES = ["ttff_ms", "dwell_ms", "dwell_prop", "visit_count", "fixation_count"]


def batch_analyze(
    file_paths: list[str | Path],
    config: str | Path | dict[str, Any] | None = None,
    include_complexity: bool = False,
    aois: list | None = None,
) -> pd.DataFrame:
    """
    批量分析多个眼动记录文件。

    基于 foundation 层公共接口，对每个文件执行完整分析流程。
    """
    io_params, preprocess_params, event_params, feature_params = _resolve_analysis_params(
        config=config,
        include_complexity=include_complexity,
    )

    rows: list[dict[str, Any]] = []
    for file_path in file_paths:
        source = Path(file_path)
        try:
            recording = load(
                source,
                format=io_params.get("format"),
                sampling_rate_hz=io_params.get("sampling_rate_hz"),
                metadata={"file_path": str(source)},
            )
            row = _analyze_recording_row(
                recording,
                file_path=str(source),
                preprocess_params=preprocess_params,
                event_params=event_params,
                feature_params=feature_params,
                aois=aois,
            )
        except Exception as exc:
            row = _build_error_row(
                file_path=str(source),
                error=str(exc),
                aois=aois,
            )
        rows.append(row)

    return pd.DataFrame(rows)


def batch_analyze_recordings(
    recordings: list[GazeRecording],
    include_complexity: bool = False,
    aois: list | None = None,
) -> pd.DataFrame:
    """
    批量分析已加载的 GazeRecording 对象列表。

    与 batch_analyze 相同逻辑，但跳过文件 IO。
    """
    _, preprocess_params, event_params, feature_params = _resolve_analysis_params(
        config=None,
        include_complexity=include_complexity,
    )

    rows: list[dict[str, Any]] = []
    for recording in recordings:
        file_path = str(recording.metadata.get("file_path", ""))
        try:
            row = _analyze_recording_row(
                recording,
                file_path=file_path,
                preprocess_params=preprocess_params,
                event_params=event_params,
                feature_params=feature_params,
                aois=aois,
            )
        except Exception as exc:
            row = _build_error_row(
                file_path=file_path,
                error=str(exc),
                recording=recording,
                aois=aois,
            )
        rows.append(row)

    return pd.DataFrame(rows)


def export_html_report(
    batch_df: pd.DataFrame,
    output_path: str | Path,
    scenario_name: str = "",
) -> str:
    """
    从批量分析结果生成可分享的 HTML 报告。
    """
    content = build_html_report_content(batch_df, scenario_name=scenario_name)
    target_path = Path(output_path).expanduser().resolve()
    _write_text_atomic(target_path, content)
    return str(target_path)


def export_markdown_report(
    batch_df: pd.DataFrame,
    output_path: str | Path,
    scenario_name: str = "",
) -> str:
    """
    从批量分析结果生成 Markdown 格式报告。
    """
    content = build_markdown_report_content(batch_df, scenario_name=scenario_name)
    target_path = Path(output_path).expanduser().resolve()
    _write_text_atomic(target_path, content)
    return str(target_path)


def build_html_report_content(batch_df: pd.DataFrame, scenario_name: str = "") -> str:
    """生成 HTML 报告内容字符串。"""
    frame = batch_df.copy()
    successful = _successful_rows(frame)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quality_chart = _quality_distribution_chart_html(successful)
    descriptive = _descriptive_stats_frame(successful)
    quality_table = _quality_distribution_frame(successful)
    aoi_summary = _aoi_summary_frame(successful)

    scenario_text = scenario_name or "通用批量分析"
    valid_count = int(len(successful))
    total_count = int(len(frame))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>眼动批量分析报告</title>
  <style>
    body {{
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      margin: 0;
      padding: 32px;
      color: #10213a;
      background: #f4f8fc;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
    }}
    .hero {{
      background: linear-gradient(135deg, #0a1a2f, #113a67);
      color: #f5fbff;
      border-radius: 18px;
      padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: 0 18px 44px rgba(10, 26, 47, 0.18);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    .hero p {{
      margin: 8px 0;
      opacity: 0.94;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }}
    .metric-card {{
      background: #ffffff;
      border-radius: 16px;
      padding: 18px 20px;
      box-shadow: 0 10px 24px rgba(18, 41, 69, 0.08);
    }}
    .metric-card .label {{
      color: #54708b;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric-card .value {{
      font-size: 28px;
      font-weight: 700;
      color: #0f2e4d;
    }}
    .section {{
      background: #ffffff;
      border-radius: 16px;
      padding: 22px 24px;
      margin-bottom: 20px;
      box-shadow: 0 10px 24px rgba(18, 41, 69, 0.08);
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 22px;
      color: #0f2e4d;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d7e2ee;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #edf4fb;
      color: #163b61;
      font-weight: 600;
    }}
    tr.error-row td {{
      background: #fff1f1;
      color: #952d2d;
    }}
    details {{
      margin-top: 12px;
    }}
    summary {{
      cursor: pointer;
      color: #0b69b8;
      font-weight: 600;
    }}
    img.chart {{
      max-width: 360px;
      width: 100%;
      display: block;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>眼动批量分析报告</h1>
      <p><strong>场景：</strong> {html.escape(scenario_text)}</p>
      <p><strong>生成时间：</strong> {html.escape(generated_at)}</p>
    </div>

    <div class="metrics">
      <div class="metric-card"><div class="label">总文件数</div><div class="value">{total_count}</div></div>
      <div class="metric-card"><div class="label">有效记录数</div><div class="value">{valid_count}</div></div>
      <div class="metric-card"><div class="label">失败记录数</div><div class="value">{total_count - valid_count}</div></div>
    </div>

    <div class="section">
      <h2>数据概览</h2>
      {quality_chart}
    </div>

    <div class="section">
      <h2>描述性统计表</h2>
      {_dataframe_to_html_table(descriptive)}
    </div>

    <div class="section">
      <h2>质量等级分布表</h2>
      {_dataframe_to_html_table(quality_table)}
    </div>

    {f'<div class="section"><h2>AOI 指标汇总表</h2>{_dataframe_to_html_table(aoi_summary)}</div>' if not aoi_summary.empty else ''}

    <div class="section">
      <h2>原始数据表</h2>
      <details>
        <summary>展开查看完整结果表</summary>
        {_dataframe_to_html_table(frame)}
      </details>
    </div>
  </div>
</body>
</html>
"""


def build_markdown_report_content(batch_df: pd.DataFrame, scenario_name: str = "") -> str:
    """生成 Markdown 报告内容字符串。"""
    frame = batch_df.copy()
    successful = _successful_rows(frame)
    scenario_text = scenario_name or "通用批量分析"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    descriptive = _descriptive_stats_frame(successful)
    quality_table = _quality_distribution_frame(successful)
    aoi_summary = _aoi_summary_frame(successful)

    sections = [
        "# 眼动批量分析报告",
        "",
        f"- 场景：{scenario_text}",
        f"- 生成时间：{generated_at}",
        f"- 总文件数：{len(frame)}",
        f"- 有效记录数：{len(successful)}",
        f"- 失败记录数：{len(frame) - len(successful)}",
        "",
        "## 描述性统计表",
        "",
        _dataframe_to_markdown_table(descriptive),
        "",
        "## 质量等级分布表",
        "",
        _dataframe_to_markdown_table(quality_table),
    ]

    if not aoi_summary.empty:
        sections.extend(
            [
                "",
                "## AOI 指标汇总表",
                "",
                _dataframe_to_markdown_table(aoi_summary),
            ]
        )

    sections.extend(
        [
            "",
            "## 原始数据表",
            "",
            "<details><summary>展开查看完整结果表</summary>",
            "",
            _dataframe_to_markdown_table(frame),
            "",
            "</details>",
        ]
    )
    return "\n".join(sections)


def _resolve_analysis_params(
    *,
    config: str | Path | dict[str, Any] | None,
    include_complexity: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(config, (str, Path)):
        base_config = load_config(config)
    elif isinstance(config, dict):
        base_config = dict(config)
    else:
        base_config = {}

    effective = merge_config(base_config, {"features": {"include_complexity": include_complexity}})
    io_config = dict(effective.get("io", {}))
    preprocess_config = {
        "missing_strategy": effective.get("preprocess", {}).get("missing_strategy", "interpolate"),
        "interpolation_method": effective.get("preprocess", {}).get("interpolation_method", "linear"),
        "smooth_method": effective.get("preprocess", {}).get("smooth_method", "moving_average"),
        "smooth_window": int(effective.get("preprocess", {}).get("smooth_window", 5)),
        "normalize_coordinates_flag": bool(effective.get("preprocess", {}).get("normalize_coordinates", True)),
    }
    event_config = {
        "velocity_threshold": float(effective.get("events", {}).get("velocity_threshold", 850.0)),
        "min_fixation_ms": float(effective.get("events", {}).get("min_fixation_ms", 60.0)),
        "blink_min_duration_ms": float(effective.get("events", {}).get("blink_min_duration_ms", 75.0)),
        "source": effective.get("events", {}).get("source", "auto"),
        "label_column": effective.get("events", {}).get("label_column"),
    }
    feature_config = {
        "window_ms": float(effective.get("features", {}).get("window_ms", 500.0)),
        "include_complexity": bool(effective.get("features", {}).get("include_complexity", include_complexity)),
    }
    return io_config, preprocess_config, event_config, feature_config


def _analyze_recording_row(
    recording: GazeRecording,
    *,
    file_path: str,
    preprocess_params: dict[str, Any],
    event_params: dict[str, Any],
    feature_params: dict[str, Any],
    aois: list | None,
) -> dict[str, Any]:
    if recording.samples.empty:
        raise ValueError("Recording contains no samples.")

    analysis = analyze_recording(
        recording,
        preprocess_params=preprocess_params,
        event_params=event_params,
        feature_params=feature_params,
    )

    metadata = recording.metadata
    row: dict[str, Any] = dict(analysis.features)
    row["session_id"] = metadata.get("session_id")
    row["subject_id"] = metadata.get("subject_id")
    row["condition"] = metadata.get("condition")
    row["trial"] = metadata.get("trial")
    row["quality_grade"] = metadata.get("quality_grade", compute_quality_grade(recording))
    row["segment_name"] = metadata.get("segment_name")
    row["file_path"] = file_path
    row["error"] = None

    if aois:
        row.update(_compute_aoi_columns(analysis.enriched_recording, aois))

    return row


def _build_error_row(
    *,
    file_path: str,
    error: str,
    recording: GazeRecording | None = None,
    aois: list | None = None,
) -> dict[str, Any]:
    metadata = recording.metadata if recording is not None else {}
    quality_grade: str | None
    try:
        quality_grade = metadata.get("quality_grade") if recording is None else metadata.get(
            "quality_grade",
            compute_quality_grade(recording),
        )
    except Exception:
        quality_grade = None

    row: dict[str, Any] = {
        "session_id": metadata.get("session_id"),
        "subject_id": metadata.get("subject_id"),
        "condition": metadata.get("condition"),
        "trial": metadata.get("trial"),
        "quality_grade": quality_grade,
        "segment_name": metadata.get("segment_name"),
        "file_path": file_path,
        "error": error,
    }
    if aois:
        for aoi in aois:
            row.update(_empty_aoi_metrics(aoi.name))
    return row


def _compute_aoi_columns(recording: GazeRecording, aois: list) -> dict[str, Any]:
    fixations = fixation_table(recording)
    assigned = assign_fixations_to_aoi(fixations, aois)
    metrics = compute_aoi_metrics(assigned, aois, total_duration=recording.duration_ms)

    columns: dict[str, Any] = {}
    for aoi in aois:
        metric = metrics[aoi.name]
        columns[f"{aoi.name}_ttff_ms"] = metric.first_fixation_time
        columns[f"{aoi.name}_dwell_ms"] = metric.total_dwell_time
        columns[f"{aoi.name}_dwell_prop"] = metric.dwell_proportion
        columns[f"{aoi.name}_visit_count"] = metric.visit_count
        columns[f"{aoi.name}_fixation_count"] = metric.fixation_count
    return columns


def _empty_aoi_metrics(aoi_name: str) -> dict[str, Any]:
    return {
        f"{aoi_name}_ttff_ms": pd.NA,
        f"{aoi_name}_dwell_ms": pd.NA,
        f"{aoi_name}_dwell_prop": pd.NA,
        f"{aoi_name}_visit_count": pd.NA,
        f"{aoi_name}_fixation_count": pd.NA,
    }


def _successful_rows(batch_df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in batch_df.columns:
        return batch_df.copy()
    error_text = batch_df["error"].fillna("").astype(str).str.strip()
    return batch_df.loc[error_text.eq("")].copy()


def _descriptive_stats_frame(batch_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in KEY_METRICS:
        if column not in batch_df.columns:
            continue
        series = pd.to_numeric(batch_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "metric": column,
                "mean": round(float(series.mean()), 4),
                "sd": round(float(series.std(ddof=1)) if len(series) > 1 else 0.0, 4),
                "n": int(len(series)),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "mean", "sd", "n"])


def _quality_distribution_frame(batch_df: pd.DataFrame) -> pd.DataFrame:
    if "quality_grade" not in batch_df.columns:
        return pd.DataFrame(columns=["quality_grade", "count"])

    distribution = (
        batch_df["quality_grade"]
        .fillna("未知")
        .astype(str)
        .value_counts()
        .rename_axis("quality_grade")
        .reset_index(name="count")
    )
    return distribution


def _aoi_summary_frame(batch_df: pd.DataFrame) -> pd.DataFrame:
    aoi_columns = [
        column
        for column in batch_df.columns
        if any(column.endswith(f"_{suffix}") for suffix in AOI_METRIC_SUFFIXES)
    ]
    if not aoi_columns:
        return pd.DataFrame(columns=["metric", "mean", "sd", "n"])

    rows: list[dict[str, Any]] = []
    for column in aoi_columns:
        series = pd.to_numeric(batch_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "metric": column,
                "mean": round(float(series.mean()), 4),
                "sd": round(float(series.std(ddof=1)) if len(series) > 1 else 0.0, 4),
                "n": int(len(series)),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "mean", "sd", "n"])


def _quality_distribution_chart_html(batch_df: pd.DataFrame) -> str:
    distribution = _quality_distribution_frame(batch_df)
    if distribution.empty:
        return "<p>当前没有可用于绘制质量分布图的有效记录。</p>"

    figure = Figure(figsize=(4.8, 3.6))
    axis = figure.subplots()
    axis.pie(
        distribution["count"].to_numpy(dtype=float),
        labels=distribution["quality_grade"].astype(str).tolist(),
        autopct="%1.0f%%",
        startangle=90,
    )
    axis.set_title("质量等级分布")
    axis.axis("equal")

    image = _figure_to_base64(figure)
    return f'<img class="chart" src="data:image/png;base64,{image}" alt="质量等级分布图" />'


def _figure_to_base64(figure: Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _dataframe_to_html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>无数据。</p>"

    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body_rows: list[str] = []
    for _, row in frame.fillna("").iterrows():
        is_error_row = bool(str(row.get("error", "")).strip()) if "error" in frame.columns else False
        class_name = ' class="error-row"' if is_error_row else ""
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        body_rows.append(f"<tr{class_name}>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"

    safe_frame = frame.fillna("").copy()
    header = "| " + " | ".join(_escape_markdown_text(column) for column in safe_frame.columns.astype(str)) + " |"
    separator = "| " + " | ".join("---" for _ in safe_frame.columns) + " |"
    rows = [
        "| " + " | ".join(_escape_markdown_text(value) for value in row.astype(str).tolist()) + " |"
        for _, row in safe_frame.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _escape_markdown_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"报告导出失败：{exc}") from exc
