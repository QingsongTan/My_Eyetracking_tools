from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gaze_toolkit.aoi import AOI
from gaze_toolkit.types import GazeRecording
from gaze_toolkit.visualization import plot_interactive_scanpath


@dataclass
class VisualControls:
    """Visual customization options for linked charts."""

    scanpath_palette: str
    heatmap_palette: str
    fixation_opacity: float
    heatmap_opacity: float
    show_saliency: bool = True
    show_deepgaze: bool = True
    show_heatmap: bool = True


def _localize_feature_name(name: str) -> str:
    direct_map = {
        "duration_ms": "总时长(ms)",
        "sample_count": "样本数",
        "valid_ratio": "有效比例",
        "path_length": "路径长度",
        "velocity_mean": "速度均值",
        "velocity_peak": "速度峰值",
        "fixation_count": "注视次数",
        "fixation_duration_mean": "注视平均时长",
        "fixation_duration_total": "注视总时长",
        "fixation_density": "注视密度",
        "saccade_count": "扫视次数",
        "saccade_amplitude_mean": "扫视幅度均值",
        "saccade_peak_velocity_mean": "扫视峰值速度均值",
        "saccade_latency_mean": "扫视潜伏期均值",
        "blink_count": "眨眼次数",
        "blink_rate_hz": "眨眼频率",
        "blink_duration_mean": "眨眼平均时长",
        "pupil_baseline": "瞳孔基线水平",
        "pupil_change_rate": "瞳孔变化率",
        "pupil_bc_mean": "基线校正后瞳孔均值",
        "pupil_bc_std": "基线校正后瞳孔标准差",
        "pupil_bc_peak": "基线校正后瞳孔峰值",
        "pupil_bc_q75": "基线校正后瞳孔 75 分位",
        "pupil_tonic_level": "瞳孔持续性水平",
        "pupil_phasic_mean": "瞳孔相位性反应均值",
        "pupil_phasic_peak": "瞳孔相位性反应峰值",
        "pupil_dilation_latency_ms": "瞳孔扩张潜伏期（ms）",
        "pupil_blink_ratio": "眨眼样本占比",
        "pupil_interpolation_ratio": "插值样本占比",
        "cognitive_load_level": "工作负荷等级",
        "cognitive_load_score": "工作负荷评分",
        "heart_rate_mean": "心率均值",
        "heart_rate_std": "心率标准差",
        "heart_rate_min": "最低心率",
        "heart_rate_max": "最高心率",
        "heart_rate_rmssd": "心率RMSSD",
    }
    if name in direct_map:
        return direct_map[name]

    replacements = {
        "fixation": "注视",
        "saccade": "扫视",
        "blink": "眨眼",
        "pupil": "瞳孔",
        "velocity": "速度",
        "duration": "时长",
        "count": "次数",
        "mean": "均值",
        "peak": "峰值",
        "baseline": "基线",
        "density": "密度",
        "latency": "潜伏期",
        "rolling": "滑窗",
        "std": "标准差",
        "min": "最小值",
        "max": "最大值",
        "skew": "偏度",
        "kurtosis": "峰度",
        "approx": "近似",
        "entropy": "熵",
        "q25": "25分位",
        "q75": "75分位",
        "x": "X",
        "y": "Y",
    }
    localized = name
    for source, target in replacements.items():
        localized = localized.replace(source, target)
    return localized.replace("_", "·")


def _extract_metric_groups(
    *,
    feature_df: pd.DataFrame,
    metric: str,
    selected_conditions: list[str],
    paired: bool,
    subject_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    subset_columns = ["condition", metric]
    if subject_col is not None:
        subset_columns.append(subject_col)
    subset = feature_df[subset_columns].copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna(subset=["condition", metric])

    if paired:
        if subject_col is None:
            raise ValueError("配对检验需要 subject_id 列。")
        pivot = subset.pivot(index=subject_col, columns="condition", values=metric)
        pivot = pivot.reindex(columns=selected_conditions).dropna()
        if len(pivot) < 2:
            raise ValueError("当前数据不足以形成有效的配对样本。")
        return (
            pivot[selected_conditions[0]].to_numpy(dtype=float),
            pivot[selected_conditions[1]].to_numpy(dtype=float),
        )

    left = subset.loc[subset["condition"] == selected_conditions[0], metric].dropna().to_numpy(dtype=float)
    right = subset.loc[subset["condition"] == selected_conditions[1], metric].dropna().to_numpy(dtype=float)
    if len(left) < 2 or len(right) < 2:
        raise ValueError("每个 condition 至少需要 2 个有效观测。")
    return left, right


def _significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_p_for_display(p_value: float) -> str:
    if p_value < 0.001:
        return "< .001"
    text = f"{p_value:.3f}"
    return text[1:] if text.startswith("0") else text


def _overlay_aoi_shapes(figure: go.Figure, aois: list[AOI]) -> None:
    palette = [
        ("rgba(0, 243, 255, 0.14)", "rgba(0, 243, 255, 0.88)"),
        ("rgba(0, 255, 157, 0.14)", "rgba(0, 255, 157, 0.88)"),
        ("rgba(255, 181, 90, 0.16)", "rgba(255, 181, 90, 0.88)"),
        ("rgba(125, 158, 255, 0.16)", "rgba(125, 158, 255, 0.88)"),
    ]

    for index, aoi in enumerate(aois):
        fill_color, line_color = palette[index % len(palette)]
        if aoi.region_type == "rectangle":
            x_min, y_min, x_max, y_max = aoi.region
            figure.add_shape(
                type="rect",
                x0=x_min,
                y0=y_min,
                x1=x_max,
                y1=y_max,
                line={"color": line_color, "width": 2},
                fillcolor=fill_color,
                layer="above",
            )
            label_x = float(x_min) + 8.0
            label_y = float(y_min) + 12.0
        else:
            vertices = [(float(x), float(y)) for x, y in aoi.region]
            path = "M " + " L ".join(f"{x},{y}" for x, y in vertices) + " Z"
            figure.add_shape(
                type="path",
                path=path,
                line={"color": line_color, "width": 2},
                fillcolor=fill_color,
                layer="above",
            )
            label_x = float(np.mean([x for x, _ in vertices]))
            label_y = float(np.mean([y for _, y in vertices]))

        figure.add_annotation(
            x=label_x,
            y=label_y,
            text=aoi.name,
            showarrow=False,
            bgcolor="rgba(8, 23, 45, 0.82)",
            bordercolor=line_color,
            borderpad=5,
            font={"color": "#EAF7FF", "size": 11},
        )


def _build_aoi_scanpath_figure(
    *,
    recording: GazeRecording,
    aois: list[AOI],
    stimulus_image: str | Path | Any | None,
    screen_size: tuple[int, int],
    theme_name: str,
    visual_controls: VisualControls,
) -> go.Figure:
    figure = plot_interactive_scanpath(
        recording,
        background_image=stimulus_image,
        screen_size=screen_size,
        theme_name=theme_name,
        palette=visual_controls.scanpath_palette,
        fixation_opacity=visual_controls.fixation_opacity,
    )
    figure.update_xaxes(range=[0.0, float(screen_size[0])])
    figure.update_yaxes(range=[float(screen_size[1]), 0.0])
    figure.update_layout(title="AOI 叠加 Scanpath")
    _overlay_aoi_shapes(figure, aois)
    return figure


def _build_aoi_transition_figure(matrix: pd.DataFrame, *, theme_name: str) -> go.Figure:
    template = "plotly_white" if theme_name == "light" else "plotly_dark"
    text_values = matrix.map(lambda value: f"{value:.2f}")
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=matrix.to_numpy(dtype=float),
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                zmin=0.0,
                zmax=1.0,
                colorscale="Blues",
                text=text_values.to_numpy(),
                texttemplate="%{text}",
                hovertemplate="来源=%{y}<br>目标=%{x}<br>概率=%{z:.2f}<extra></extra>",
                colorbar={"title": "转移概率"},
            )
        ]
    )
    figure.update_layout(
        title="AOI 转移矩阵热力图",
        template=template,
        height=380,
        margin={"l": 32, "r": 20, "t": 48, "b": 28},
        xaxis={"title": "目标 AOI"},
        yaxis={"title": "来源 AOI"},
    )
    return figure


def _build_regression_prediction_figure(
    holdout: pd.DataFrame,
    *,
    theme_name: str = "dark",
) -> go.Figure:
    theme = "plotly_white" if theme_name == "light" else "plotly_dark"
    frame = holdout.copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame["residual"] = pd.to_numeric(frame.get("residual"), errors="coerce")
    frame = frame.dropna(subset=["y_true", "y_pred"])
    figure = go.Figure()
    if frame.empty:
        figure.update_layout(
            title="回归预测对照",
            template=theme,
            height=360,
            margin={"l": 24, "r": 16, "t": 48, "b": 36},
        )
        return figure

    min_value = float(min(frame["y_true"].min(), frame["y_pred"].min()))
    max_value = float(max(frame["y_true"].max(), frame["y_pred"].max()))
    figure.add_trace(
        go.Scatter(
            x=frame["y_true"],
            y=frame["y_pred"],
            mode="markers",
            marker={"size": 10, "color": "#00bfe8", "line": {"width": 1, "color": "rgba(255,255,255,0.22)"}},
            customdata=frame[["residual"]] if "residual" in frame.columns else None,
            hovertemplate="真实值=%{x:.3f}<br>预测值=%{y:.3f}<br>残差=%{customdata[0]:.3f}<extra></extra>",
            name="Holdout 样本",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            line={"dash": "dash", "color": "rgba(255,255,255,0.45)"},
            name="理想预测线",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title="回归预测对照",
        template=theme,
        height=360,
        margin={"l": 24, "r": 16, "t": 48, "b": 36},
        xaxis={"title": "真实值"},
        yaxis={"title": "预测值"},
    )
    return figure


def _build_statistics_boxplot(
    *,
    feature_df: pd.DataFrame,
    selected_conditions: list[str],
    selected_metrics: list[str],
    theme_name: str,
) -> go.Figure:
    figure = go.Figure()
    for condition in selected_conditions:
        condition_frame = feature_df.loc[feature_df["condition"] == condition]
        for metric in selected_metrics:
            values = pd.to_numeric(condition_frame[metric], errors="coerce").dropna()
            figure.add_trace(
                go.Box(
                    x=[_localize_feature_name(metric)] * len(values),
                    y=values,
                    name=str(condition),
                    boxmean=True,
                    legendgroup=str(condition),
                    offsetgroup=str(condition),
                )
            )

    figure.update_layout(
        title="条件间箱线图对比",
        template="plotly_white" if theme_name == "light" else "plotly_dark",
        height=420,
        margin={"l": 24, "r": 16, "t": 48, "b": 36},
        xaxis={"title": "指标"},
        yaxis={"title": "数值"},
        boxmode="group",
    )
    return figure


def _effect_ci_scale(group1: np.ndarray, group2: np.ndarray) -> float:
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))
    pooled_var = (((n1 - 1) * var1) + ((n2 - 1) * var2)) / max(n1 + n2 - 2, 1)
    return float(np.sqrt(max(pooled_var, 0.0)))


def _estimate_effect_ci(
    *,
    feature_df: pd.DataFrame,
    metric: str,
    effect_size: float,
    ci_lower: float,
    ci_upper: float,
    selected_conditions: list[str],
    paired: bool,
    subject_col: str | None,
) -> tuple[float, float]:
    if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        return (float("nan"), float("nan"))

    group1, group2 = _extract_metric_groups(
        feature_df=feature_df,
        metric=metric,
        selected_conditions=selected_conditions,
        paired=paired,
        subject_col=subject_col,
    )
    if paired:
        differences = group2 - group1
        scale = float(np.std(differences, ddof=1))
    else:
        scale = _effect_ci_scale(group1, group2)

    if not np.isfinite(scale) or scale == 0.0:
        return (float(effect_size), float(effect_size))
    return (float(ci_lower / scale), float(ci_upper / scale))


def _build_effect_size_forest(
    *,
    feature_df: pd.DataFrame,
    results: pd.DataFrame,
    selected_conditions: list[str],
    paired: bool,
    subject_col: str | None,
    theme_name: str,
) -> go.Figure:
    plot_frame = results.copy()
    if plot_frame.empty:
        return go.Figure()

    ci_bounds = [
        _estimate_effect_ci(
            feature_df=feature_df,
            metric=row.metric,
            effect_size=float(row.effect_size),
            ci_lower=float(row.ci_lower),
            ci_upper=float(row.ci_upper),
            selected_conditions=selected_conditions,
            paired=paired,
            subject_col=subject_col,
        )
        for row in plot_frame.itertuples(index=False)
    ]
    plot_frame["effect_ci_lower"] = [bounds[0] for bounds in ci_bounds]
    plot_frame["effect_ci_upper"] = [bounds[1] for bounds in ci_bounds]
    plot_frame["metric_label"] = plot_frame["metric"].map(_localize_feature_name)

    error_minus = np.where(
        np.isfinite(plot_frame["effect_ci_lower"]),
        plot_frame["effect_size"] - plot_frame["effect_ci_lower"],
        0.0,
    )
    error_plus = np.where(
        np.isfinite(plot_frame["effect_ci_upper"]),
        plot_frame["effect_ci_upper"] - plot_frame["effect_size"],
        0.0,
    )

    figure = go.Figure(
        data=[
            go.Scatter(
                x=plot_frame["effect_size"],
                y=plot_frame["metric_label"],
                mode="markers",
                marker={"size": 11, "color": "#00bfe8"},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": error_plus,
                    "arrayminus": error_minus,
                    "visible": True,
                },
                customdata=plot_frame[["effect_size_name", "conclusion"]],
                hovertemplate="指标=%{y}<br>效应量=%{x:.2f}<br>类型=%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            )
        ]
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color="rgba(255,255,255,0.35)")
    figure.update_layout(
        title="效应量森林图",
        template="plotly_white" if theme_name == "light" else "plotly_dark",
        height=420,
        margin={"l": 24, "r": 16, "t": 48, "b": 36},
        xaxis={"title": "效应量"},
        yaxis={"title": "指标"},
    )
    return figure


def _build_batch_quality_figure(quality_counts: pd.DataFrame, *, theme_name: str) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Bar(
                x=quality_counts["quality_grade"],
                y=quality_counts["count"],
                marker={"color": ["#0fb8ad", "#5fbf68", "#f0a63a", "#d15a5a", "#8a9cb4"][: len(quality_counts)]},
                hovertemplate="质量等级=%{x}<br>记录数=%{y}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="质量分布",
        template="plotly_white" if theme_name == "light" else "plotly_dark",
        height=320,
        margin={"l": 20, "r": 12, "t": 46, "b": 24},
        xaxis={"title": "质量等级"},
        yaxis={"title": "记录数"},
    )
    return figure
