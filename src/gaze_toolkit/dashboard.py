from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from gaze_toolkit.analysis import (
    RecordingAnalysis,
    analyze_recording,
    compare_modalities,
    run_intent_experiment,
    synthesize_heart_rate_preview,
)
from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.events import has_labeled_events
from gaze_toolkit.io import from_frame
from gaze_toolkit.saliency import (
    COGNITIVE_SALIENCY_BACKEND,
    FAST_SALIENCY_BACKEND,
    get_saliency_backend_status,
    probe_deepgaze_runtime,
    predict_image_attention,
)
from gaze_toolkit.segmentation import segment_recording
from gaze_toolkit.types import GazeRecording
from gaze_toolkit.visualization import (
    plot_confusion,
    plot_feature_importance,
    plot_image_saliency_heatmap,
    plot_interactive_heatmap,
    plot_interactive_scanpath,
    plot_metrics,
    plot_signal_overview,
)

ACCENT = "#00F3FF"
PAPER = "#0A1A2F"
INK = "#EAF7FF"
UI_THEME_OPTIONS = {"深色科技": "dark", "浅色模式": "light"}
STYLE_OPTIONS = {"精读": "careful", "略读": "skim"}
MODEL_OPTIONS = {
    "随机森林": "random_forest",
    "梯度提升树": "gradient_boosting",
    "支持向量机": "svm",
    "逻辑回归": "logistic_regression",
}
MISSING_OPTIONS = {
    "自动插值": "interpolate",
    "清洗无效样本": "drop",
    "保留原始缺失": "keep",
}
EVENT_SOURCE_OPTIONS = {
    "自动优先原始标签": "auto",
    "仅使用原始标签": "labels",
    "使用自定义阈值识别": "thresholds",
}
SEGMENTATION_OPTIONS = {
    "整个文件": "whole",
    "按时间段": "time_ranges",
    "按 Marker 前后窗口": "marker_windows",
    "按 Marker1 到 Marker2": "between_markers",
}
TIME_RANGE_PATTERN = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:-|~|to|:)\s*([+-]?\d+(?:\.\d+)?)\s*$")
SCREEN_RESOLUTION_PRESETS: dict[str, tuple[int, int] | None] = {
    "1920 x 1080": (1920, 1080),
    "1366 x 768": (1366, 768),
    "1440 x 900": (1440, 900),
    "1600 x 900": (1600, 900),
    "2560 x 1440": (2560, 1440),
    "3840 x 2160": (3840, 2160),
    "自定义": None,
}
DEFAULT_SCREEN_SIZE = (1920, 1080)
SCANPATH_PALETTE_OPTIONS = {
    "主题默认": "theme_default",
    "极光蓝青": "aurora",
    "冰川蓝": "glacier",
    "霓虹紫": "violet",
    "落日橙": "sunset",
}
HEATMAP_PALETTE_OPTIONS = {
    "主题默认": "theme_default",
    "极光蓝青": "aurora",
    "冰川蓝": "glacier",
    "霓虹紫": "violet",
    "落日橙": "sunset",
}
IMAGE_ATTENTION_MODEL_LABELS = {
    FAST_SALIENCY_BACKEND: "OpenCV Fast Saliency",
    COGNITIVE_SALIENCY_BACKEND: "PyTorch + PySaliency + DeepGaze",
}


@dataclass
class DashboardControls:
    """UI selections that drive the analysis pipeline."""

    preprocess_params: dict[str, Any]
    event_params: dict[str, Any]
    feature_params: dict[str, Any]
    segmentation_config: dict[str, Any] | None
    segmentation_summary: str
    segmentation_warning: str | None = None


@dataclass
class SegmentView:
    """A segment plus its derived analysis objects."""

    name: str
    analysis: RecordingAnalysis
    start_time_ms: float
    end_time_ms: float
    marker_value: str = ""
    start_marker: str = ""
    end_marker: str = ""


@dataclass
class VisualControls:
    """Visual customization options for linked charts."""

    scanpath_palette: str
    heatmap_palette: str
    fixation_opacity: float
    heatmap_opacity: float


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="眼动人因智能实验台",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme_name = _select_theme_mode()
    _inject_styles(theme_name)

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">研究型眼动数据分析示例</div>
          <h1>眼动与多模态行为分析实验台</h1>
          <p>
            用于整理采样级眼动记录，并结合事件识别、分段观察、可视化和建模实验，
            对单次会话与多模态线索进行探索性分析。
          </p>
          <div class="hero-credit">Powered by 谭青松</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "上传含 `timestamp/x/y` 的 CSV 即可开始分析；若同时提供 `valid`、`marker`、`event_label/label`、`pupil`，"
        "系统会自动接入缺失值处理、原始事件复用和分段视图。"
    )
    _render_overview_strip()

    recording, stimulus_image, screen_size = _build_recording_from_sidebar()
    controls = _build_analysis_controls(recording)
    visual_controls = _build_visual_controls()

    try:
        full_analysis = analyze_recording(
            recording,
            preprocess_params=controls.preprocess_params,
            event_params=controls.event_params,
            feature_params=controls.feature_params,
        )
        segment_views = _build_segment_views(recording, controls)
        segment_table = _segment_views_to_frame(segment_views)
    except ValueError as exc:
        st.error(f"当前配置无法完成分析：{exc}")
        st.stop()

    tabs = st.tabs(
        [
            "研究概览",
            "单次会话分析",
            "意图建模实验",
            "多模态融合",
            "项目解读",
        ]
    )

    with tabs[0]:
        _render_capability_story(full_analysis, controls, segment_table)

    with tabs[1]:
        _render_single_session(
            full_analysis,
            controls=controls,
            segment_views=segment_views,
            segment_table=segment_table,
            stimulus_image=stimulus_image,
            screen_size=screen_size,
            theme_name=theme_name,
            visual_controls=visual_controls,
        )

    with tabs[2]:
        _render_modeling_workbench(theme_name)

    with tabs[3]:
        _render_multimodal_tab(recording, theme_name)

    with tabs[4]:
        _render_portfolio_talking_points()


def _render_overview_strip() -> None:
    st.markdown(
        """
        <div class="overview-strip">
          <div class="overview-card">
            <div class="overview-kicker">Data</div>
            <div class="overview-title">采样点整理</div>
            <p>统一时间戳、屏幕坐标和有效性字段，把原始记录变成可分析对象。</p>
          </div>
          <div class="overview-card">
            <div class="overview-kicker">Events</div>
            <div class="overview-title">事件识别</div>
            <p>支持设备原始标签复用，也支持按阈值重算注视、扫视和眨眼。</p>
          </div>
          <div class="overview-card">
            <div class="overview-kicker">Segments</div>
            <div class="overview-title">分段联动</div>
            <p>Marker、时间窗和整段记录可以直接联动到 scanpath、heatmap 和事件表。</p>
          </div>
          <div class="overview-card">
            <div class="overview-kicker">Models</div>
            <div class="overview-title">建模扩展</div>
            <p>保留多模态与基线实验入口，便于把观察结果继续接到状态分析和模型验证。</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _select_theme_mode() -> str:
    theme_label = st.sidebar.selectbox(
        "界面主题",
        options=list(UI_THEME_OPTIONS),
        index=0,
        help="切换当前仪表板的深色或浅色视觉模式。",
    )
    return UI_THEME_OPTIONS[theme_label]


def _build_recording_from_sidebar() -> tuple[GazeRecording, str | Path | None, tuple[int, int]]:
    st.sidebar.header("实验输入设置")
    source = st.sidebar.radio("数据来源", options=["合成演示数据", "上传 CSV 文件"], index=0)
    sampling_rate_hz = float(st.sidebar.number_input("采样率（Hz）", min_value=30, max_value=1000, value=120))
    st.sidebar.caption("必需列：timestamp/x/y；可选列：valid、pupil、marker、event_label/label。")
    screen_size = _select_screen_size()
    stimulus_image: str | Path | None = None

    if source == "合成演示数据":
        style_label = st.sidebar.selectbox("阅读意图模式", options=list(STYLE_OPTIONS), index=0)
        duration_ms = int(st.sidebar.slider("时长（毫秒）", min_value=2000, max_value=12000, value=5000, step=500))
        seed = int(st.sidebar.slider("随机种子", min_value=1, max_value=999, value=42))
        recording = simulate_gaze_recording(
            duration_ms=duration_ms,
            sampling_rate_hz=int(sampling_rate_hz),
            style=STYLE_OPTIONS[style_label],
            seed=seed,
        )
        recording.metadata["intent_label"] = STYLE_OPTIONS[style_label]
        return recording, None, screen_size

    uploaded = st.sidebar.file_uploader("上传眼动 CSV", type=["csv"])
    uploaded_stimulus = st.sidebar.file_uploader("上传刺激图片（可选）", type=["png", "jpg", "jpeg"])
    if uploaded_stimulus is not None:
        stimulus_image = uploaded_stimulus

    if uploaded is None:
        st.sidebar.warning("尚未上传 CSV，系统将回退到默认合成演示数据。")
        return simulate_gaze_recording(seed=42), stimulus_image, screen_size

    frame = pd.read_csv(uploaded)
    recording = from_frame(frame, sampling_rate_hz=sampling_rate_hz, source_format="csv_upload")
    recording.metadata["intent_label"] = "unknown"
    return recording, stimulus_image, screen_size


def _select_screen_size() -> tuple[int, int]:
    resolution_label = st.sidebar.selectbox(
        "屏幕分辨率",
        options=list(SCREEN_RESOLUTION_PRESETS),
        index=0,
        help="用于把上传的刺激图片映射到 scanpath 和 heatmap 的屏幕坐标系，默认 1920 x 1080。",
    )
    preset = SCREEN_RESOLUTION_PRESETS[resolution_label]
    if preset is not None:
        st.sidebar.caption(f"当前刺激图坐标映射：{preset[0]} x {preset[1]}")
        return preset

    width_col, height_col = st.sidebar.columns(2)
    width = int(
        width_col.number_input(
            "宽度(px)",
            min_value=320,
            max_value=7680,
            value=DEFAULT_SCREEN_SIZE[0],
            step=10,
        )
    )
    height = int(
        height_col.number_input(
            "高度(px)",
            min_value=240,
            max_value=4320,
            value=DEFAULT_SCREEN_SIZE[1],
            step=10,
        )
    )
    st.sidebar.caption(f"当前刺激图坐标映射：{width} x {height}")
    return width, height


def _build_analysis_controls(recording: GazeRecording) -> DashboardControls:
    st.sidebar.header("分析流程控制")
    missing_label = st.sidebar.selectbox("缺失值处理", options=list(MISSING_OPTIONS), index=0)
    smooth_window = int(st.sidebar.slider("平滑窗口", min_value=3, max_value=21, step=2, value=5))
    include_complexity = st.sidebar.toggle("包含复杂度特征", value=True)

    has_original_labels = has_labeled_events(recording)
    if has_original_labels:
        source_label = st.sidebar.selectbox("事件识别来源", options=list(EVENT_SOURCE_OPTIONS), index=0)
        event_source = EVENT_SOURCE_OPTIONS[source_label]
    else:
        event_source = "thresholds"
        st.sidebar.caption("当前文件未检测到原始事件标签，事件识别将使用自定义阈值。")

    velocity_threshold = 850.0
    if event_source == "thresholds":
        velocity_threshold = float(
            st.sidebar.slider("扫视速度阈值", min_value=200, max_value=1800, value=850, step=50)
        )

    min_fixation_ms = float(st.sidebar.slider("最小注视时长（ms）", min_value=30, max_value=200, value=60, step=10))
    blink_min_duration_ms = float(
        st.sidebar.slider("最小眨眼时长（ms）", min_value=40, max_value=250, value=75, step=5)
    )

    segmentation_config, segmentation_summary, segmentation_warning = _build_segmentation_controls(recording)

    return DashboardControls(
        preprocess_params={
            "missing_strategy": MISSING_OPTIONS[missing_label],
            "smooth_window": smooth_window,
        },
        event_params={
            "velocity_threshold": velocity_threshold,
            "min_fixation_ms": min_fixation_ms,
            "blink_min_duration_ms": blink_min_duration_ms,
            "source": event_source,
        },
        feature_params={"include_complexity": include_complexity},
        segmentation_config=segmentation_config,
        segmentation_summary=segmentation_summary,
        segmentation_warning=segmentation_warning,
    )


def _build_visual_controls() -> VisualControls:
    st.sidebar.header("可视化样式")
    scanpath_palette_label = st.sidebar.selectbox("Scanpath 色系", options=list(SCANPATH_PALETTE_OPTIONS), index=0)
    heatmap_palette_label = st.sidebar.selectbox("Heatmap 色系", options=list(HEATMAP_PALETTE_OPTIONS), index=0)
    fixation_opacity = float(st.sidebar.slider("注视层透明度", min_value=0.20, max_value=0.95, value=0.72, step=0.05))
    heatmap_opacity = float(st.sidebar.slider("热图透明度", min_value=0.20, max_value=0.95, value=0.60, step=0.05))
    st.sidebar.caption("注视节点和热图默认半透明，叠加刺激图时不会完全遮住背景。")
    return VisualControls(
        scanpath_palette=SCANPATH_PALETTE_OPTIONS[scanpath_palette_label],
        heatmap_palette=HEATMAP_PALETTE_OPTIONS[heatmap_palette_label],
        fixation_opacity=fixation_opacity,
        heatmap_opacity=heatmap_opacity,
    )


def _build_segmentation_controls(
    recording: GazeRecording,
) -> tuple[dict[str, Any] | None, str, str | None]:
    st.sidebar.header("分段设置")
    marker_values = _available_marker_values(recording)
    if marker_values:
        st.sidebar.caption(f"当前检测到 {len(marker_values)} 个 Marker：{', '.join(marker_values[:6])}")
    else:
        st.sidebar.caption("当前文件未检测到可用 Marker。")

    segmentation_label = st.sidebar.selectbox("分段方式", options=list(SEGMENTATION_OPTIONS), index=0)
    strategy = SEGMENTATION_OPTIONS[segmentation_label]

    if strategy == "whole":
        return {"strategy": "whole"}, "整个文件作为一个分段进入分析。", None

    if strategy == "time_ranges":
        default_end = min(max(float(recording.duration_ms), 500.0), 1500.0)
        raw_ranges = st.sidebar.text_area(
            "时间段（毫秒）",
            value=f"0-{default_end:.0f}",
            help="多个时间段用分号或换行分隔，例如：0-800；1200-2000",
        )
        try:
            time_ranges = _parse_time_ranges(raw_ranges)
        except ValueError as exc:
            return None, "按时间段分段。", str(exc)
        return {"strategy": "time_ranges", "time_ranges": time_ranges}, f"按时间段分段，共 {len(time_ranges)} 段。", None

    if strategy == "marker_windows":
        if not marker_values:
            return None, "按 Marker 前后窗口分段。", "当前文件没有可用 Marker，无法使用该分段方式。"
        selected_markers = st.sidebar.multiselect("选择 Marker", options=marker_values, default=marker_values[:1])
        pre_ms = float(st.sidebar.number_input("Marker 前窗口（ms）", min_value=0.0, value=200.0, step=50.0))
        post_ms = float(st.sidebar.number_input("Marker 后窗口（ms）", min_value=0.0, value=800.0, step=50.0))
        if not selected_markers:
            return None, "按 Marker 前后窗口分段。", "请至少选择一个 Marker。"
        return {
            "strategy": "marker_windows",
            "marker_values": selected_markers,
            "pre_ms": pre_ms,
            "post_ms": post_ms,
        }, f"围绕 {len(selected_markers)} 个 Marker 生成前后窗口分段。", None

    if not marker_values:
        return None, "按 Marker1 到 Marker2 分段。", "当前文件没有可用 Marker，无法使用该分段方式。"

    start_marker = st.sidebar.selectbox("起始 Marker", options=marker_values, index=0)
    end_index = 1 if len(marker_values) > 1 else 0
    end_marker = st.sidebar.selectbox("结束 Marker", options=marker_values, index=end_index)
    return {
        "strategy": "between_markers",
        "start_marker": start_marker,
        "end_marker": end_marker,
    }, f"从每个 {start_marker} 到其后第一个 {end_marker} 形成分段。", None


def _build_segment_views(recording: GazeRecording, controls: DashboardControls) -> list[SegmentView]:
    if controls.segmentation_config is None:
        return []

    segments = segment_recording(recording, **controls.segmentation_config)
    views: list[SegmentView] = []
    for segment in segments:
        analysis = analyze_recording(
            segment.recording,
            preprocess_params=controls.preprocess_params,
            event_params=controls.event_params,
            feature_params=controls.feature_params,
        )
        views.append(
            SegmentView(
                name=segment.name,
                analysis=analysis,
                start_time_ms=segment.start_time_ms,
                end_time_ms=segment.end_time_ms,
                marker_value=str(segment.recording.metadata.get("marker_value", "")),
                start_marker=str(segment.recording.metadata.get("start_marker", "")),
                end_marker=str(segment.recording.metadata.get("end_marker", "")),
            )
        )
    return views


def _segment_views_to_frame(segment_views: list[SegmentView]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for view in segment_views:
        row = dict(view.analysis.features)
        row.update(
            {
                "segment_name": view.name,
                "segment_start_ms": view.start_time_ms,
                "segment_end_ms": view.end_time_ms,
                "marker_value": view.marker_value,
                "start_marker": view.start_marker,
                "end_marker": view.end_marker,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _render_capability_story(
    analysis: RecordingAnalysis,
    controls: DashboardControls,
    segment_table: pd.DataFrame,
) -> None:
    left, right = st.columns([1.1, 1.3], gap="large")

    with left:
        st.subheader("这一页关注什么")
        st.write(
            "这里更关注研究链路是否完整、观察层级是否清晰，以及不同分析视图能否在同一份记录上联动切换。"
        )
        st.markdown(
            """
            - 原始采样点可以先整理成统一结构，再进入事件识别、分段观察和特征摘要。
            - 事件识别既可复用设备原始标签，也可按阈值重新计算，便于不同数据源保持一致。
            - 缺失样本、眨眼片段和坏段可以按插值、清洗或保留原样的方式分别处理。
            - 记录可以围绕时间窗或 Marker 分段，并把当前分段同步投影到 scanpath、heatmap 和事件表。
            - 眼动链路可以继续接入心率等时间序列，为后续状态分析或多模态实验预留接口。
            """
        )

        st.subheader("当前会话摘要")
        metrics = analysis.quality_summary
        metric_cols = st.columns(4)
        metric_cols[0].metric("有效样本占比", f"{metrics['valid_ratio']:.2%}")
        metric_cols[1].metric("注视次数", int(metrics["fixation_count"]))
        metric_cols[2].metric("扫视次数", int(metrics["saccade_count"]))
        metric_cols[3].metric("眨眼次数", int(metrics["blink_count"]))

        st.caption(f"分段策略：{controls.segmentation_summary}")
        if not segment_table.empty:
            st.caption(f"当前配置共得到 {len(segment_table)} 个分段。")

    with right:
        st.subheader("当前分析范围")
        coverage_frame = pd.DataFrame(
            [
                ["数据整理", "可用", "支持采样点标准化、字段映射和基础校验"],
                ["缺失与坏段处理", "可用", "支持插值、清洗删除和保留原样"],
                ["事件来源", "可用", "可复用设备标签，也可按阈值重算"],
                ["分段方式", "可用", "支持整段、时间窗、Marker 窗口和 Marker 对"],
                ["视图联动", "可用", "分段切换会同步更新 scanpath、heatmap 和事件表"],
                ["特征摘要", "可用", "覆盖注视、扫视、眨眼、瞳孔和复杂度指标"],
                ["建模实验", "可用", "提供基线分类实验和特征重要性查看"],
                ["多模态接口", "可用", "可与心率等时间序列按时间轴对齐"],
            ],
            columns=["模块", "当前支持", "说明"],
        )
        _render_panel_table(coverage_frame, hide_index=True, max_height_px=360)


def _render_single_session(
    analysis: RecordingAnalysis,
    controls: DashboardControls,
    segment_views: list[SegmentView],
    segment_table: pd.DataFrame,
    stimulus_image: str | Path | None = None,
    screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
    theme_name: str = "dark",
    visual_controls: VisualControls | None = None,
) -> None:
    st.subheader("单次会话分析链路")
    st.caption("输入数据 -> 预处理 -> 事件识别 -> 分段选择 -> Scanpath / Heatmap / 特征解释")

    overall = analysis.features
    metric_cols = st.columns(5)
    metric_cols[0].metric("总时长", f"{overall['duration_ms'] / 1000:.1f}s")
    metric_cols[1].metric("路径长度", f"{overall['path_length']:.0f}")
    metric_cols[2].metric("平均速度", f"{overall['velocity_mean']:.1f}")
    metric_cols[3].metric("眨眼频率", f"{overall['blink_rate_hz']:.2f} Hz")
    metric_cols[4].metric("瞳孔基线", f"{overall['pupil_baseline']:.2f}")

    strategy_cols = st.columns(4)
    strategy_cols[0].metric("缺失值策略", _label_for_value(MISSING_OPTIONS, controls.preprocess_params["missing_strategy"]))
    strategy_cols[1].metric("事件来源", _event_source_copy(controls.event_params["source"]))
    strategy_cols[2].metric("分段方式", controls.segmentation_summary)

    strategy_cols[3].metric("屏幕分辨率", f"{screen_size[0]} x {screen_size[1]}")

    selected_view = _render_segment_selector(segment_views)
    display_analysis = selected_view.analysis if selected_view is not None else analysis
    display_title = _segment_title(selected_view)

    display_metrics = display_analysis.features
    st.markdown("**当前可视化对象**")
    st.caption(f"当前图形与事件表展示的是：{display_title}")

    focus_cols = st.columns(4)
    focus_cols[0].metric("当前时长", f"{display_metrics['duration_ms'] / 1000:.2f}s")
    focus_cols[1].metric("当前样本数", int(display_metrics["sample_count"]))
    focus_cols[2].metric("当前注视数", int(display_metrics["fixation_count"]))
    focus_cols[3].metric("当前眨眼数", int(display_metrics["blink_count"]))

    visual_controls = visual_controls or VisualControls(
        scanpath_palette="theme_default",
        heatmap_palette="theme_default",
        fixation_opacity=0.72,
        heatmap_opacity=0.60,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### 扫描路径轨迹 (Scanpath)")
        st.caption("通过点线连接逐个注视中心，并用圆点大小表示停留时长，用颜色表示时序进度。")
        st.plotly_chart(
            plot_interactive_scanpath(
                display_analysis.enriched_recording,
                background_image=stimulus_image,
                screen_size=screen_size,
                theme_name=theme_name,
                palette=visual_controls.scanpath_palette,
                fixation_opacity=visual_controls.fixation_opacity,
            ),
            key="single-session-scanpath",
            width="stretch",
            config={"displaylogo": False},
        )

    with right:
        st.markdown("### 核密度注意力热力图 (Heatmap)")
        st.caption("依据当前展示分段中的注视密度，投射出屏幕上的视觉“重心区”和“信息盲区”。")
        st.plotly_chart(
            plot_interactive_heatmap(
                display_analysis.enriched_recording,
                background_image=stimulus_image,
                screen_size=screen_size,
                theme_name=theme_name,
                palette=visual_controls.heatmap_palette,
                heatmap_opacity=visual_controls.heatmap_opacity,
            ),
            key="single-session-heatmap",
            width="stretch",
            config={"displaylogo": False},
        )

    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        st.markdown("**核心信号总览**")
        signal_figure = plot_signal_overview(
            display_analysis.enriched_recording,
            velocity=display_analysis.velocity_profile,
            theme_name=theme_name,
        )
        st.pyplot(signal_figure, clear_figure=True, width="stretch")

    with lower_right:
        st.markdown("**事件表**")
        event_table = display_analysis.event_table.rename(
            columns={
                "kind": "事件类型",
                "start_time_ms": "开始时间(ms)",
                "end_time_ms": "结束时间(ms)",
                "duration_ms": "持续时间(ms)",
                "amplitude": "幅度",
                "peak_velocity": "峰值速度",
                "centroid_x": "中心X",
                "centroid_y": "中心Y",
            }
        )
        if not event_table.empty:
            event_table["事件类型"] = event_table["事件类型"].replace(
                {
                    "fixation": "注视",
                    "saccade": "扫视",
                    "blink": "眨眼",
                    "smooth_pursuit": "平滑追随",
                }
            )
        _render_panel_table(event_table.head(24), max_height_px=360)

    if stimulus_image is not None:
        _render_stimulus_attention_section(
            stimulus_image=stimulus_image,
            recording=display_analysis.enriched_recording,
            screen_size=screen_size,
            theme_name=theme_name,
            visual_controls=visual_controls,
        )

    st.markdown("**当前展示对象关键特征**")
    top_features = (
        pd.Series(display_analysis.features, name="数值")
        .sort_values(ascending=False)
        .head(18)
        .rename_axis("特征")
        .reset_index()
    )
    top_features["特征"] = top_features["特征"].map(_localize_feature_name)
    _render_panel_table(top_features, hide_index=True, max_height_px=420)

    st.markdown("**分段结果预览**")
    if controls.segmentation_warning:
        st.warning(controls.segmentation_warning)
    elif segment_table.empty:
        st.info("当前配置下没有生成可用分段。")
    else:
        st.caption("切换上方“当前展示分段”后，scanpath、heatmap、信号总览和事件表都会同步刷新。")
        _render_panel_table(_format_segment_table(segment_table), hide_index=True, max_height_px=300)


def _render_stimulus_attention_section(
    stimulus_image: str | Path | Any,
    *,
    recording: GazeRecording,
    screen_size: tuple[int, int],
    theme_name: str,
    visual_controls: VisualControls,
) -> None:
    st.markdown("**基于图片内容的先验注意分布**")
    st.caption("这部分不依赖上传的眼动轨迹，只根据图片自身的颜色对比、局部反差和边缘结构，快速估计底层视觉显著性。")

    try:
        result = predict_image_attention(stimulus_image, backend=FAST_SALIENCY_BACKEND)
    except (ImportError, TypeError, ValueError) as exc:
        st.warning(f"图片显著性生成失败：{exc}")
        return

    fast_status = get_saliency_backend_status(FAST_SALIENCY_BACKEND)
    future_status = get_saliency_backend_status(COGNITIVE_SALIENCY_BACKEND)
    runtime_ok = False
    runtime_payload: dict[str, Any] = {}
    try:
        runtime_ok, runtime_payload = probe_deepgaze_runtime()
    except RuntimeError as exc:
        runtime_payload = {"ok": False, "error": str(exc)}
    attention_center = (
        f"({result.metadata['attention_center_x']:.0f}, {result.metadata['attention_center_y']:.0f})"
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("当前模型", IMAGE_ATTENTION_MODEL_LABELS[result.backend])
    metric_cols[1].metric("输入分辨率", f"{result.width} x {result.height}")
    metric_cols[2].metric("高显著区域", f"{result.metadata['hotspot_ratio'] * 100:.1f}%")
    metric_cols[3].metric("注意质心", attention_center)
    st.caption(
        f"{fast_status.detail} 推理耗时约 {result.metadata['inference_ms']:.1f} ms，"
        f"峰值显著性 {result.metadata['peak_saliency']:.2f}。"
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### 图片快速显著性热力图")
        st.caption("适合在没有真实眼动数据时，先用作刺激图的快速注意先验。")
        st.plotly_chart(
            plot_image_saliency_heatmap(
                result.saliency_map,
                background_image=stimulus_image,
                screen_size=screen_size,
                theme_name=theme_name,
                palette=visual_controls.heatmap_palette,
                heatmap_opacity=visual_controls.heatmap_opacity,
            ),
            key="stimulus-fast-saliency",
            width="stretch",
            config={"displaylogo": False},
        )

    with right:
        st.markdown("### 认知模型热力图 (DeepGaze)")
        st.caption("有真实注视历史时优先使用 DeepGazeIII；没有 fixation history 时会回退到 DeepGazeIIE。")
        if runtime_ok:
            try:
                cognitive = predict_image_attention(
                    stimulus_image,
                    backend=COGNITIVE_SALIENCY_BACKEND,
                    recording=recording,
                )
            except RuntimeError as exc:
                st.warning(f"DeepGaze 推理失败：{exc}")
                st.info(future_status.detail)
            else:
                st.plotly_chart(
                    plot_image_saliency_heatmap(
                        cognitive.saliency_map,
                        background_image=stimulus_image,
                        screen_size=screen_size,
                        theme_name=theme_name,
                        palette=visual_controls.heatmap_palette,
                        heatmap_opacity=visual_controls.heatmap_opacity,
                        title="认知模型注意热力图 (DeepGaze)",
                    ),
                    key="stimulus-cognitive-saliency",
                    width="stretch",
                    config={"displaylogo": False},
                )
                details = []
                model_name = cognitive.metadata.get("deepgaze_model")
                if model_name:
                    details.append(f"模型：{model_name}")
                if "conditioning_fixation_count" in cognitive.metadata:
                    details.append(f"条件 fixation 数：{int(cognitive.metadata['conditioning_fixation_count'])}")
                if "nss_mean" in cognitive.metadata:
                    details.append(f"NSS：{cognitive.metadata['nss_mean']:.3f}")
                if "sim" in cognitive.metadata:
                    details.append(f"SIM：{cognitive.metadata['sim']:.3f}")
                if "kl_divergence" in cognitive.metadata:
                    details.append(f"KL：{cognitive.metadata['kl_divergence']:.3f}")
                if details:
                    st.caption(" | ".join(details))
        else:
            st.warning(runtime_payload.get("error", future_status.detail))
            st.info(
                "\n".join(
                    [
                        f"backend 名称：`{COGNITIVE_SALIENCY_BACKEND}`",
                        "需要可用的独立 DeepGaze Python 运行时。",
                        "默认查找：项目根目录下 `.deepgaze-py312/Scripts/python.exe`。",
                        "也可以通过环境变量 `GAZE_TOOLKIT_DEEPGAZE_PYTHON` 指定解释器。",
                    ]
                )
            )


def _render_segment_selector(segment_views: list[SegmentView]) -> SegmentView | None:
    if not segment_views:
        return None
    if len(segment_views) == 1:
        st.caption(f"当前仅有 1 个分段：{_segment_label(segment_views[0])}")
        return segment_views[0]

    selected_index = st.selectbox(
        "当前展示分段",
        options=list(range(len(segment_views))),
        index=0,
        format_func=lambda index: _segment_label(segment_views[index]),
    )
    return segment_views[selected_index]


def _render_panel_table(
    frame: pd.DataFrame,
    *,
    hide_index: bool = False,
    max_height_px: int = 360,
) -> None:
    if frame.empty:
        st.info("当前没有可显示的数据。")
        return

    display = frame.copy().fillna("")
    header_cells = "".join(f"<th>{html.escape(str(column))}</th>" for column in display.columns)
    rows: list[str] = []
    for index, row in display.iterrows():
        index_cell = "" if hide_index else f"<td class='panel-table-index'>{html.escape(str(index))}</td>"
        value_cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{index_cell}{value_cells}</tr>")

    index_header = "" if hide_index else "<th class='panel-table-index'>#</th>"
    table_html = f"""
    <div class="panel-table-shell" style="max-height:{max_height_px}px;">
      <table class="panel-table-grid">
        <thead>
          <tr>{index_header}{header_cells}</tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _render_modeling_workbench(theme_name: str = "dark") -> None:
    st.subheader("意图建模实验台")
    controls = st.columns([1, 1, 1])
    num_sessions = int(controls[0].slider("合成会话数", min_value=12, max_value=80, value=32, step=4))
    model_label = controls[1].selectbox("模型类型", options=list(MODEL_OPTIONS), index=0)
    random_state = int(controls[2].slider("实验随机种子", min_value=1, max_value=999, value=42))

    report = run_intent_experiment(
        num_sessions=num_sessions,
        model_name=MODEL_OPTIONS[model_label],
        random_state=random_state,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("准确率", f"{report.result.metrics.get('accuracy', 0.0):.3f}")
    metric_cols[1].metric("F1 宏平均", f"{report.result.metrics.get('f1_macro', 0.0):.3f}")
    metric_cols[2].metric("ROC AUC", f"{report.result.metrics.get('roc_auc', 0.0):.3f}")

    left, right = st.columns(2, gap="large")
    with left:
        figure, axis = plt.subplots(figsize=(6, 4))
        plot_metrics(report.result.metrics, ax=axis, theme_name=theme_name)
        st.pyplot(figure, clear_figure=True, width="stretch")

        if not report.holdout_predictions.empty:
            figure, axis = plt.subplots(figsize=(5, 5))
            plot_confusion(
                report.holdout_predictions["y_true"].to_numpy(),
                report.holdout_predictions["y_pred"].to_numpy(),
                ax=axis,
                theme_name=theme_name,
            )
            st.pyplot(figure, clear_figure=True, width="stretch")

    with right:
        figure, axis = plt.subplots(figsize=(6, 4.8))
        localized_importance = report.feature_importance.copy()
        localized_importance["feature"] = localized_importance["feature"].map(_localize_feature_name)
        plot_feature_importance(localized_importance, ax=axis, theme_name=theme_name)
        st.pyplot(figure, clear_figure=True, width="stretch")
        importance = localized_importance.head(15).rename(
            columns={
                "feature": "特征",
                "importance_mean": "平均重要性",
                "importance_std": "重要性标准差",
            }
        )
        _render_panel_table(importance, hide_index=True, max_height_px=360)


def _render_multimodal_tab(recording: GazeRecording, theme_name: str = "dark") -> None:
    st.subheader("多模态融合演示")
    st.caption("默认模态组合：眼动 + 模拟心率信号")

    heart_signal, heart_features = synthesize_heart_rate_preview(recording, seed=99)
    comparison = compare_modalities(num_sessions=32, model_name="random_forest", random_state=42)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("**当前会话心率预览**")
        heart_signal_cn = heart_signal.rename(columns={"heart_rate_bpm": "心率(bpm)"})
        st.line_chart(heart_signal_cn.set_index("timestamp_ms"), height=260)
        heart_feature_frame = pd.Series(heart_features, name="数值").rename_axis("特征").reset_index()
        heart_feature_frame["特征"] = heart_feature_frame["特征"].map(_localize_feature_name)
        _render_panel_table(heart_feature_frame, hide_index=True, max_height_px=320)

    with right:
        st.markdown("**基线结果对比**")
        summary = comparison.summary.rename(
            columns={
                "modality": "模态方案",
                "accuracy": "准确率",
                "f1_macro": "F1 宏平均",
                "roc_auc": "ROC AUC",
                "confusion_matrix_trace": "混淆矩阵对角和",
                "feature_count": "特征数量",
            }
        )
        summary["模态方案"] = summary["模态方案"].replace(
            {
                "gaze_only": "仅眼动",
                "gaze_plus_heart_rate": "眼动 + 心率",
            }
        )
        _render_panel_table(summary, hide_index=True, max_height_px=320)
        delta = (
            comparison.multimodal.result.metrics.get("accuracy", 0.0)
            - comparison.gaze_only.result.metrics.get("accuracy", 0.0)
        )
        st.metric("引入心率后的准确率变化", f"{delta:+.3f}")

    st.markdown("**如何理解这一页**")
    st.markdown(
        """
        这一页不是为了把心率平台做复杂，而是为了证明当前架构已经能承接第二路生理信号，
        完成时间对齐、特征构建，以及单模态和多模态建模对比。
        当前心率是模拟信号，用来证明工程链路；换成真实 HR、EDA、EEG 或行为日志时，
        更像是替换数据源，而不是重写整个系统。
        """
    )


def _render_portfolio_talking_points() -> None:
    st.subheader("项目解读")
    st.markdown(
        """
        **适合的使用方式**
        这套原型既可以读取真实眼动时序，也可以生成可重复的合成会话数据，适合先验证链路，再逐步替换成真实实验数据。

        **数据链路**
        数据先进入缺失值处理，再做平滑与标准化；随后按“原始标签”或“自定义阈值”识别事件，
        并支持围绕 Marker 或时间段做分段提取与视图联动。

        **观察层级**
        可以从整段记录切到单个时间窗、Marker 窗口或 Marker 对，再回看 scanpath、heatmap、事件表和特征摘要，
        便于把统计结果和具体片段对应起来。

        **扩展方向**
        当前界面已经预留了与心率等时间序列对齐的方式，也保留了建模实验入口，后续可以继续接入更多生理或行为信号。

        **使用前提**
        默认假设上传文件能映射到 `timestamp/x/y`，并且 gaze 坐标与当前屏幕分辨率处于同一像素坐标系。
        如果是厂商专有导出格式或局部刺激区域，通常还需要补一层字段映射或刺激区域配置。
        """
    )

    st.warning(
        "当前提醒：如果真实实验中的刺激只占屏幕一部分，后续建议补充刺激区域边界配置，"
        "这样 scanpath 和 heatmap 的底图映射会更精确。"
    )
    st.caption("项目作者：谭青松")


def _available_marker_values(recording: GazeRecording) -> list[str]:
    if "marker" not in recording.samples.columns:
        return []
    values = recording.samples["marker"].dropna().astype(str).str.strip()
    values = [value for value in values if value]
    return list(dict.fromkeys(values))


def _parse_time_ranges(raw: str) -> list[tuple[float, float]]:
    if not raw.strip():
        raise ValueError("请至少输入一个时间段，例如：0-800；1200-2000。")

    ranges: list[tuple[float, float]] = []
    for chunk in [item.strip() for item in re.split(r"[;\n]+", raw) if item.strip()]:
        match = TIME_RANGE_PATTERN.match(chunk)
        if match is None:
            raise ValueError(f"无法解析时间段：{chunk}。请使用“开始-结束”的格式。")
        start_ms = float(match.group(1))
        end_ms = float(match.group(2))
        if end_ms < start_ms:
            raise ValueError(f"时间段结束时间不能早于开始时间：{chunk}")
        ranges.append((start_ms, end_ms))
    return ranges


def _format_segment_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "segment_name",
        "marker_value",
        "start_marker",
        "end_marker",
        "segment_start_ms",
        "segment_end_ms",
        "duration_ms",
        "sample_count",
        "valid_ratio",
        "fixation_count",
        "saccade_count",
        "blink_count",
        "pupil_baseline",
    ]
    available = [column for column in columns if column in frame.columns]
    display = frame[available].copy()
    display = display.rename(
        columns={
            "segment_name": "分段名称",
            "marker_value": "Marker 值",
            "start_marker": "起始 Marker",
            "end_marker": "结束 Marker",
            "segment_start_ms": "分段开始(ms)",
            "segment_end_ms": "分段结束(ms)",
            "duration_ms": "时长(ms)",
            "sample_count": "样本数",
            "valid_ratio": "有效占比",
            "fixation_count": "注视数",
            "saccade_count": "扫视数",
            "blink_count": "眨眼数",
            "pupil_baseline": "瞳孔基线",
        }
    )
    return display.round(3)


def _segment_label(view: SegmentView) -> str:
    parts = [view.name, f"{view.start_time_ms:.0f}-{view.end_time_ms:.0f} ms"]
    if view.marker_value:
        parts.append(f"Marker={view.marker_value}")
    if view.start_marker and view.end_marker:
        parts.append(f"{view.start_marker}->{view.end_marker}")
    return " | ".join(parts)


def _segment_title(view: SegmentView | None) -> str:
    if view is None:
        return "整体会话"
    return _segment_label(view)


def _event_source_copy(source: str) -> str:
    return {
        "auto": "自动优先原始标签",
        "labels": "仅使用原始标签",
        "thresholds": "自定义阈值",
    }.get(source, source)


def _label_for_value(mapping: dict[str, str], value: str) -> str:
    for label, mapped in mapping.items():
        if mapped == value:
            return label
    return value


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
        "pupil_baseline": "瞳孔基线",
        "pupil_change_rate": "瞳孔变化率",
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


def _inject_styles(theme_name: str = "dark") -> None:
    light_overrides = _light_mode_overrides() if theme_name == "light" else ""
    st.markdown(
        f"""
        <style>
        :root {{
          --space-0: #071221;
          --space-1: #0A1A2F;
          --space-2: #102744;
          --space-3: #15365d;
          --holo: #00F3FF;
          --signal: #00FF9D;
          --ink: #EAF7FF;
          --muted: rgba(207, 232, 255, 0.70);
          --panel: rgba(10, 26, 47, 0.58);
          --panel-strong: rgba(12, 34, 60, 0.78);
          --line: rgba(0, 243, 255, 0.22);
          --line-strong: rgba(0, 243, 255, 0.38);
          --glow: 0 0 0 1px rgba(0, 243, 255, 0.12), 0 22px 60px rgba(0, 12, 32, 0.48), 0 0 36px rgba(0, 243, 255, 0.10);
        }}
        .block-container {{
          max-width: 1480px;
          padding-top: 1.15rem;
          padding-bottom: 2.8rem;
          position: relative;
          z-index: 1;
        }}
        [data-testid="stAppViewContainer"] {{
          position: relative !important;
          min-height: 100vh;
          overflow: visible !important;
        }}
        .stApp {{
          position: relative;
          background:
            radial-gradient(circle at 12% 18%, rgba(0, 243, 255, 0.16), transparent 20%),
            radial-gradient(circle at 82% 10%, rgba(0, 255, 157, 0.10), transparent 18%),
            radial-gradient(circle at 92% 82%, rgba(0, 243, 255, 0.09), transparent 18%),
            linear-gradient(145deg, #040b16 0%, #081325 18%, {PAPER} 52%, #0d213b 100%);
          color: {INK};
        }}
        .hero {{
          position: relative;
          overflow: hidden;
          padding: 1.95rem 2.1rem 1.85rem;
          border-radius: 30px;
          border: 1px solid rgba(0, 243, 255, 0.18);
          background:
            linear-gradient(135deg, rgba(13, 28, 50, 0.88), rgba(13, 35, 61, 0.76) 48%, rgba(9, 24, 44, 0.88) 100%);
          box-shadow: var(--glow);
          backdrop-filter: blur(22px) saturate(140%);
          margin-bottom: 1rem;
        }}
        .hero::selection {{
          background: rgba(0, 243, 255, 0.26);
        }}
        .hero::before {{
          content: "";
          position: absolute;
          inset: auto -6rem -6rem auto;
          width: 22rem;
          height: 22rem;
          background: radial-gradient(circle, rgba(0,243,255,0.24), rgba(0,243,255,0.03) 62%, transparent 72%);
          filter: blur(18px);
          pointer-events: none;
          animation: aurora-pulse 6.8s ease-in-out infinite;
        }}
        .hero::after {{
          content: "";
          position: absolute;
          top: -5rem;
          right: 16rem;
          width: 18rem;
          height: 18rem;
          background:
            radial-gradient(circle, rgba(0,255,157,0.16), rgba(0,255,157,0.02) 58%, transparent 72%);
          filter: blur(16px);
          pointer-events: none;
          animation: aurora-pulse 9s ease-in-out infinite reverse;
        }}
        .hero .st-emotion-cache-zy6yx3, .hero .st-emotion-cache-10trblm {{
          color: inherit;
        }}
        .hero > * {{
          position: relative;
          z-index: 1;
        }}
        .hero > *::selection {{
          background: rgba(0, 243, 255, 0.26);
        }}
        .hero:has(:hover) {{
          border-color: rgba(0, 243, 255, 0.28);
        }}
        .hero-kicker {{
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.34rem 0.7rem;
          border-radius: 999px;
          border: 1px solid rgba(0, 243, 255, 0.24);
          background: linear-gradient(180deg, rgba(0,243,255,0.12), rgba(0,243,255,0.05));
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: {ACCENT};
          font-size: 0.75rem;
          margin-bottom: 0.55rem;
          font-weight: 700;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 0 22px rgba(0,243,255,0.12);
        }}
        .hero h1 {{
          font-family: "Segoe UI Variable Display", "Bahnschrift", "Aptos Display", "Microsoft YaHei UI", sans-serif;
          font-size: clamp(2.45rem, 3.9vw, 4rem);
          line-height: 1;
          letter-spacing: -0.045em;
          margin: 0 0 0.6rem 0;
          max-width: none;
          white-space: nowrap;
          color: #f1fbff;
          text-shadow: 0 0 18px rgba(0,243,255,0.12);
        }}
        .hero p {{
          max-width: 58rem;
          font-size: 1.03rem;
          line-height: 1.7;
          margin: 0.2rem 0 0 0;
          color: rgba(218, 239, 255, 0.78);
        }}
        .hero p, .stMarkdown, .stCaption, .stDataFrame {{
          font-family: "Aptos", "Segoe UI Variable Text", "Microsoft YaHei UI", sans-serif;
        }}
        .hero-credit {{
          display: inline-flex;
          align-items: center;
          margin-top: 1rem;
          padding: 0.34rem 0.72rem;
          border-radius: 999px;
          border: 1px solid rgba(0, 243, 255, 0.16);
          background: rgba(7, 24, 44, 0.52);
          font-size: 0.84rem;
          color: rgba(221, 246, 255, 0.64);
          font-weight: 600;
        }}
        .overview-strip {{
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.85rem;
          margin: 0.15rem 0 1.4rem 0;
        }}
        .overview-card {{
          position: relative;
          overflow: hidden;
          min-height: 152px;
          padding: 1rem 1rem 0.95rem;
          border-radius: 22px;
          border: 1px solid var(--line);
          background: linear-gradient(180deg, rgba(10, 26, 47, 0.64), rgba(12, 34, 60, 0.76));
          box-shadow: var(--glow);
          backdrop-filter: blur(18px) saturate(130%);
          transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }}
        .overview-card::after,
        div[data-testid="stPlotlyChart"]::after,
        div[data-testid="stDataFrame"]::after,
        div[data-testid="stMetric"]::after {{
          content: "";
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(0,243,255,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,243,255,0.05) 1px, transparent 1px);
          background-size: 18px 18px;
          opacity: 0;
          transition: opacity 180ms ease;
          pointer-events: none;
        }}
        .overview-card:hover,
        div[data-testid="stPlotlyChart"]:hover,
        div[data-testid="stDataFrame"]:hover,
        div[data-testid="stMetric"]:hover {{
          transform: translateY(-2px);
          border-color: var(--line-strong);
          box-shadow: 0 0 0 1px rgba(0,243,255,0.16), 0 30px 70px rgba(0, 8, 28, 0.50), 0 0 28px rgba(0,243,255,0.12);
        }}
        .overview-card:hover::after,
        div[data-testid="stPlotlyChart"]:hover::after,
        div[data-testid="stDataFrame"]:hover::after,
        div[data-testid="stMetric"]:hover::after {{
          opacity: 0.22;
        }}
        .overview-kicker {{
          font-size: 0.73rem;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: rgba(0, 243, 255, 0.92);
          font-weight: 700;
        }}
        .overview-title {{
          margin-top: 0.35rem;
          font-family: "Segoe UI Variable Display", "Bahnschrift", "Aptos Display", "Microsoft YaHei UI", sans-serif;
          font-size: 1.22rem;
          color: #effbff;
        }}
        .overview-card p {{
          margin: 0.55rem 0 0 0;
          color: rgba(214, 237, 255, 0.72);
          font-size: 0.94rem;
          line-height: 1.6;
        }}
        [data-testid="stSidebar"] {{
          background:
            linear-gradient(180deg, rgba(7, 18, 33, 0.94), rgba(10, 26, 47, 0.97));
          border-right: 1px solid rgba(0, 243, 255, 0.12);
          box-shadow: inset -1px 0 0 rgba(255,255,255,0.03);
        }}
        [data-testid="stSidebar"] * {{
          color: rgba(232, 247, 255, 0.92);
        }}
        [data-baseweb="select"] > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="base-input"] > div,
        [data-testid="stNumberInput"] > div > div {{
          position: relative;
          border-radius: 18px;
          border-color: rgba(0, 243, 255, 0.14) !important;
          background: rgba(12, 34, 60, 0.72) !important;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }}
        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="input"]:focus-within,
        [data-baseweb="base-input"]:focus-within,
        [data-baseweb="base-input"] > div:focus-within,
        [data-testid="stNumberInput"] > div > div:focus-within {{
          border-color: rgba(0, 255, 157, 0.36) !important;
          box-shadow: 0 0 0 1px rgba(0,255,157,0.12), 0 0 18px rgba(0,255,157,0.10);
        }}
        [data-baseweb="select"] input,
        [data-baseweb="base-input"] input,
        [data-baseweb="select"] span,
        [data-testid="stNumberInput"] input {{
          color: rgba(232, 247, 255, 0.92) !important;
          caret-color: #00F3FF !important;
        }}
        [data-baseweb="select"] svg,
        [data-baseweb="base-input"] svg,
        [data-testid="stNumberInput"] svg {{
          fill: rgba(208, 236, 255, 0.78) !important;
        }}
        [data-testid="stNumberInput"] button {{
          background: rgba(12, 34, 60, 0.94) !important;
          color: rgba(232, 247, 255, 0.92) !important;
          border-color: rgba(0, 243, 255, 0.12) !important;
        }}
        [data-testid="stNumberInput"] button:hover {{
          background: rgba(17, 45, 77, 0.96) !important;
          border-color: rgba(0, 255, 157, 0.24) !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
          position: relative;
          border-radius: 18px;
          border-color: rgba(0, 243, 255, 0.14) !important;
          background: rgba(12, 34, 60, 0.72);
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
        [data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
        [data-testid="stSidebar"] [data-baseweb="base-input"]:focus-within,
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div:focus-within,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {{
          border-color: rgba(0, 255, 157, 0.36) !important;
          box-shadow: 0 0 0 1px rgba(0,255,157,0.12), 0 0 18px rgba(0,255,157,0.10);
        }}
        [data-testid="stFileUploaderDropzone"] button {{
          background: rgba(14, 34, 57, 0.92) !important;
          color: rgba(232, 247, 255, 0.92) !important;
          border: 1px solid rgba(0, 243, 255, 0.14) !important;
          box-shadow: none !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
          background: rgba(18, 43, 72, 0.98) !important;
          border-color: rgba(0, 255, 157, 0.24) !important;
          color: #f0ffff !important;
        }}
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span {{
          color: rgba(214, 237, 255, 0.72) !important;
        }}
        div[data-testid="stMetric"] {{
          position: relative;
          overflow: hidden;
          background: linear-gradient(180deg, rgba(10, 26, 47, 0.66), rgba(11, 31, 55, 0.82));
          border: 1px solid var(--line);
          border-radius: 22px;
          box-shadow: var(--glow);
          padding: 0.78rem 0.92rem;
          min-height: 116px;
        }}
        div[data-testid="stMetricValue"] {{
          color: #f2ffff;
          text-shadow: 0 0 18px rgba(0,243,255,0.10);
        }}
        div[data-testid="stMetricLabel"] {{
          color: rgba(207, 232, 255, 0.70);
        }}
        div[data-testid="stMetric"] label {{
          letter-spacing: 0.01em;
        }}
        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"] {{
          position: relative;
          overflow: hidden;
          background: linear-gradient(180deg, rgba(9, 23, 42, 0.64), rgba(12, 34, 60, 0.78));
          border: 1px solid var(--line);
          border-radius: 24px;
          padding: 0.55rem;
          box-shadow: var(--glow);
          backdrop-filter: blur(16px);
        }}
        div[data-testid="stPlotlyChart"] > div {{
          border-radius: 18px;
          overflow: hidden;
        }}
        div[data-testid="stImage"] {{
          background: transparent;
        }}
        div[data-testid="stImageContainer"] {{
          border-radius: 18px;
          overflow: hidden;
          background: rgba(8, 23, 45, 0.96);
          box-shadow: inset 0 0 0 1px rgba(0, 243, 255, 0.08);
        }}
        div[data-testid="stImageContainer"] img {{
          display: block;
          border-radius: 18px;
          background: rgba(8, 23, 45, 0.96);
        }}
        div[data-testid="stDataFrameResizable"] {{
          border-radius: 18px;
          overflow: hidden;
          background: rgba(8, 23, 45, 0.94);
          box-shadow: inset 0 0 0 1px rgba(0, 243, 255, 0.08);
        }}
        .stDataFrameGlideDataEditor {{
          --gdg-accent-color: #00F3FF !important;
          --gdg-accent-fg: #04101f !important;
          --gdg-accent-light: rgba(0, 243, 255, 0.14) !important;
          --gdg-text-dark: rgba(234, 247, 255, 0.94) !important;
          --gdg-text-medium: rgba(214, 237, 255, 0.86) !important;
          --gdg-text-light: rgba(182, 215, 235, 0.58) !important;
          --gdg-text-bubble: rgba(214, 237, 255, 0.86) !important;
          --gdg-bg-icon-header: rgba(0, 243, 255, 0.24) !important;
          --gdg-fg-icon-header: #eaf7ff !important;
          --gdg-text-header: rgba(0, 243, 255, 0.92) !important;
          --gdg-text-group-header: rgba(198, 231, 248, 0.78) !important;
          --gdg-bg-group-header: rgba(10, 26, 47, 0.98) !important;
          --gdg-bg-group-header-hovered: rgba(0, 243, 255, 0.12) !important;
          --gdg-text-header-selected: #eaf7ff !important;
          --gdg-bg-cell: rgba(8, 23, 45, 0.96) !important;
          --gdg-bg-cell-medium: rgba(11, 31, 55, 0.98) !important;
          --gdg-bg-header: rgba(10, 26, 47, 0.98) !important;
          --gdg-bg-header-has-focus: rgba(0, 243, 255, 0.10) !important;
          --gdg-bg-header-hovered: rgba(0, 243, 255, 0.08) !important;
          --gdg-bg-bubble: rgba(11, 31, 55, 0.98) !important;
          --gdg-bg-bubble-selected: rgba(0, 243, 255, 0.18) !important;
          --gdg-bg-search-result: rgba(0, 255, 157, 0.18) !important;
          --gdg-border-color: rgba(0, 243, 255, 0.12) !important;
          --gdg-horizontal-border-color: rgba(0, 243, 255, 0.10) !important;
          --gdg-drilldown-border: rgba(0, 243, 255, 0.18) !important;
          --gdg-link-color: #00F3FF !important;
          --gdg-resize-indicator-color: #00FF9D !important;
          --gdg-font-family: "Aptos", "Segoe UI Variable Text", "Microsoft YaHei UI", sans-serif !important;
        }}
        div[data-testid="stElementToolbarButtonContainer"] {{
          background: rgba(8, 23, 45, 0.94) !important;
          border: 1px solid rgba(0, 243, 255, 0.14);
          border-radius: 12px;
          box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
        }}
        button[data-testid="stBaseButton-elementToolbar"] {{
          background: transparent !important;
          color: rgba(226, 246, 255, 0.76) !important;
        }}
        button[data-testid="stBaseButton-elementToolbar"]:hover {{
          color: #00F3FF !important;
          background: rgba(0, 243, 255, 0.08) !important;
        }}
        [data-testid="stElementToolbarButtonIcon"] {{
          color: inherit !important;
          fill: currentColor !important;
        }}
        div[data-testid="stDataFrame"] canvas {{
          background: rgba(8, 23, 45, 0.96) !important;
        }}
        div[data-testid="stDataFrame"] [role="grid"] {{
          background: rgba(8, 23, 45, 0.96) !important;
          color: rgba(234, 247, 255, 0.92) !important;
        }}
        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td {{
          background: rgba(8, 23, 45, 0.96) !important;
          color: rgba(234, 247, 255, 0.92) !important;
          border-color: rgba(0, 243, 255, 0.10) !important;
        }}
        div[data-testid="stDataFrame"] th {{
          color: rgba(0, 243, 255, 0.92) !important;
        }}
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
          gap: 0.45rem;
          width: fit-content;
          padding: 0.34rem;
          border-radius: 999px;
          border: 1px solid rgba(0, 243, 255, 0.16);
          background: rgba(7, 24, 44, 0.56);
          box-shadow: 0 14px 30px rgba(0, 6, 22, 0.34);
          backdrop-filter: blur(14px);
        }}
        div[data-testid="stTabs"] button {{
          border-radius: 999px !important;
          padding: 0.52rem 0.95rem !important;
          font-weight: 600;
          color: rgba(214, 237, 255, 0.68);
          transition: all 0.18s ease;
        }}
        div[data-testid="stTabs"] button[aria-selected="true"] {{
          background: linear-gradient(135deg, rgba(0, 243, 255, 0.22), rgba(0, 255, 157, 0.20)) !important;
          color: #f0ffff !important;
          box-shadow: 0 0 0 1px rgba(0,243,255,0.22), 0 0 22px rgba(0,243,255,0.14);
        }}
        h3 {{
          letter-spacing: -0.02em;
          color: #f0fbff;
        }}
        .stCaption {{
          color: rgba(207, 232, 255, 0.58);
        }}
        .stAlert {{
          border-radius: 20px !important;
          border: 1px solid rgba(0,243,255,0.16);
          box-shadow: 0 14px 30px rgba(0, 6, 22, 0.30);
          background: linear-gradient(180deg, rgba(9, 24, 44, 0.72), rgba(12, 34, 60, 0.80));
        }}
        ul {{
          line-height: 1.72;
        }}
        .stMarkdown a, .stCaption a {{
          color: var(--holo);
        }}
        .panel-table-shell {{
          overflow: auto;
          border-radius: 20px;
          border: 1px solid rgba(0, 243, 255, 0.18);
          background: linear-gradient(180deg, rgba(8, 23, 45, 0.94), rgba(10, 26, 47, 0.98));
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), 0 12px 28px rgba(0, 8, 28, 0.22);
        }}
        .panel-table-grid {{
          width: 100%;
          border-collapse: collapse;
          font-size: 0.94rem;
          color: rgba(234, 247, 255, 0.92);
          background: transparent;
        }}
        .panel-table-grid thead th {{
          position: sticky;
          top: 0;
          z-index: 1;
          background: rgba(10, 26, 47, 0.98);
          color: rgba(0, 243, 255, 0.94);
          text-align: left;
          font-weight: 700;
          letter-spacing: 0.01em;
        }}
        .panel-table-grid th,
        .panel-table-grid td {{
          padding: 0.7rem 0.85rem;
          border-bottom: 1px solid rgba(0, 243, 255, 0.10);
          white-space: nowrap;
        }}
        .panel-table-grid tbody tr:nth-child(odd) {{
          background: rgba(11, 31, 55, 0.62);
        }}
        .panel-table-grid tbody tr:nth-child(even) {{
          background: rgba(8, 23, 45, 0.92);
        }}
        .panel-table-grid tbody tr:hover {{
          background: rgba(0, 243, 255, 0.08);
        }}
        .panel-table-grid .panel-table-index {{
          color: rgba(182, 215, 235, 0.56);
          width: 3.2rem;
        }}
        button:focus-visible,
        [role="tab"]:focus-visible,
        input:focus-visible,
        textarea:focus-visible {{
          outline: 1px solid rgba(0,255,157,0.70) !important;
          box-shadow: 0 0 0 4px rgba(0,255,157,0.10) !important;
        }}
        @keyframes aurora-pulse {{
          0%, 100% {{ opacity: 0.62; transform: scale(0.96); }}
          50% {{ opacity: 1; transform: scale(1.03); }}
        }}
        @media (max-width: 1200px) {{
          .overview-strip {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }}
          .hero h1 {{
            font-size: 2.95rem;
          }}
        }}
        @media (min-width: 961px) {{
          html, body, .stApp {{
            height: 100%;
            overflow: hidden;
          }}
          [data-testid="stAppViewContainer"] {{
            height: 100vh !important;
            min-height: 100vh !important;
            overflow: hidden !important;
            align-items: stretch;
          }}
          [data-testid="stSidebar"] {{
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
          }}
          [data-testid="stSidebar"] > div,
          [data-testid="stSidebarContent"] {{
            height: 100vh !important;
            max-height: 100vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior: contain;
          }}
          [data-testid="stMain"] {{
            height: 100vh !important;
            max-height: 100vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            overscroll-behavior: contain;
          }}
          .block-container {{
            min-height: auto;
          }}
        }}
        @media (max-width: 760px) {{
          .overview-strip {{
            grid-template-columns: 1fr;
          }}
          .hero {{
            padding: 1.35rem 1.2rem 1.25rem;
            border-radius: 22px;
          }}
          .hero h1 {{
            font-size: 2.15rem;
            white-space: normal;
            max-width: 100%;
            line-height: 1.04;
            text-wrap: balance;
          }}
        }}
        {light_overrides}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _light_mode_overrides() -> str:
    return """
        .stApp {
          background:
            radial-gradient(circle at 12% 18%, rgba(0, 243, 255, 0.12), transparent 20%),
            radial-gradient(circle at 82% 10%, rgba(0, 255, 157, 0.08), transparent 18%),
            radial-gradient(circle at 92% 82%, rgba(0, 243, 255, 0.08), transparent 18%),
            linear-gradient(145deg, #f9fdff 0%, #f1f8ff 30%, #e8f5ff 70%, #f7fcff 100%) !important;
          color: #0f2748 !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .block-container {
          color: #0f2748 !important;
        }
        [data-testid="stHeader"] {
          background: rgba(249, 253, 255, 0.80) !important;
          border-bottom: 1px solid rgba(0, 200, 255, 0.10) !important;
          backdrop-filter: blur(12px) saturate(130%) !important;
        }
        .hero {
          background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(241,250,255,0.94) 52%, rgba(227,244,255,0.92) 100%) !important;
          border-color: rgba(0, 200, 255, 0.20) !important;
          box-shadow: 0 18px 40px rgba(18, 78, 138, 0.10), 0 0 0 1px rgba(255,255,255,0.52) inset !important;
        }
        .hero-kicker,
        .overview-kicker {
          color: #00a7d6 !important;
          border-color: rgba(0, 200, 255, 0.18) !important;
        }
        .hero h1 {
          color: #0d284a !important;
          text-shadow: 0 0 16px rgba(0, 205, 255, 0.08) !important;
        }
        .hero p {
          color: rgba(15, 39, 72, 0.76) !important;
        }
        .hero-credit {
          background: rgba(255,255,255,0.72) !important;
          border-color: rgba(0, 200, 255, 0.14) !important;
          color: rgba(15, 39, 72, 0.58) !important;
        }
        .overview-card,
        div[data-testid="stMetric"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"] {
          background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(239,248,255,0.84)) !important;
          border-color: rgba(0, 200, 255, 0.16) !important;
          box-shadow: 0 18px 36px rgba(18, 78, 138, 0.08), 0 0 0 1px rgba(255,255,255,0.64) inset !important;
        }
        .overview-title,
        h3,
        div[data-testid="stMetricValue"] {
          color: #0d284a !important;
          text-shadow: none !important;
        }
        p,
        li,
        strong,
        label,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stMarkdownContainer"] strong {
          color: rgba(15, 39, 72, 0.88) !important;
        }
        .overview-card p,
        .stCaption,
        div[data-testid="stMetricLabel"] {
          color: rgba(15, 39, 72, 0.64) !important;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(248,252,255,0.98), rgba(233,244,255,0.99)) !important;
          border-right-color: rgba(0, 200, 255, 0.12) !important;
          box-shadow: inset -1px 0 0 rgba(255,255,255,0.76) !important;
        }
        [data-testid="stSidebar"] * {
          color: #15365c !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"],
        [data-testid="stSidebar"] [data-baseweb="base-input"] > div,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
          background: rgba(255,255,255,0.86) !important;
          border-color: rgba(0, 200, 255, 0.16) !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="base-input"] > div,
        [data-testid="stNumberInput"] > div > div {
          background: rgba(255,255,255,0.90) !important;
          border-color: rgba(0, 200, 255, 0.16) !important;
          box-shadow: 0 10px 24px rgba(18, 78, 138, 0.06), 0 0 0 1px rgba(255,255,255,0.66) inset !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="base-input"] input,
        [data-baseweb="select"] span,
        [data-testid="stNumberInput"] input {
          color: #15365c !important;
          caret-color: #00bfe8 !important;
        }
        [data-baseweb="select"] svg,
        [data-baseweb="base-input"] svg,
        [data-testid="stNumberInput"] svg {
          fill: rgba(21, 54, 92, 0.68) !important;
        }
        [data-testid="stNumberInput"] button {
          background: rgba(255,255,255,0.92) !important;
          color: #15365c !important;
          border-color: rgba(0, 200, 255, 0.12) !important;
        }
        [data-testid="stNumberInput"] button:hover {
          background: rgba(243,250,255,0.98) !important;
          border-color: rgba(0, 200, 255, 0.20) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
          background: rgba(255,255,255,0.90) !important;
          color: #15365c !important;
          border: 1px solid rgba(0, 200, 255, 0.14) !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
          background: rgba(244,250,255,0.98) !important;
          border-color: rgba(0, 200, 255, 0.22) !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
          background: rgba(255,255,255,0.78) !important;
          border-color: rgba(0, 200, 255, 0.14) !important;
          box-shadow: 0 12px 28px rgba(18, 78, 138, 0.08) !important;
        }
        div[data-testid="stTabs"] button {
          color: rgba(15, 39, 72, 0.66) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
          background: linear-gradient(135deg, rgba(0, 243, 255, 0.18), rgba(0, 255, 157, 0.16)) !important;
          color: #0d284a !important;
          box-shadow: 0 0 0 1px rgba(0, 200, 255, 0.16), 0 10px 20px rgba(18, 78, 138, 0.10) !important;
        }
        .stAlert {
          background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(239,248,255,0.88)) !important;
          border-color: rgba(0, 200, 255, 0.14) !important;
          box-shadow: 0 12px 26px rgba(18, 78, 138, 0.08) !important;
        }
        div[data-testid="stImageContainer"],
        div[data-testid="stDataFrameResizable"] {
          background: rgba(255,255,255,0.92) !important;
          box-shadow: inset 0 0 0 1px rgba(0, 200, 255, 0.10) !important;
        }
        div[data-testid="stImageContainer"] img {
          background: rgba(255,255,255,0.92) !important;
        }
        .stDataFrameGlideDataEditor {
          --gdg-accent-color: #00bfe8 !important;
          --gdg-accent-fg: #ffffff !important;
          --gdg-accent-light: rgba(0, 191, 232, 0.12) !important;
          --gdg-text-dark: rgba(15, 39, 72, 0.92) !important;
          --gdg-text-medium: rgba(15, 39, 72, 0.82) !important;
          --gdg-text-light: rgba(15, 39, 72, 0.48) !important;
          --gdg-text-bubble: rgba(15, 39, 72, 0.86) !important;
          --gdg-bg-icon-header: rgba(0, 191, 232, 0.16) !important;
          --gdg-fg-icon-header: #0f2748 !important;
          --gdg-text-header: #009fd9 !important;
          --gdg-text-group-header: rgba(15, 39, 72, 0.64) !important;
          --gdg-bg-group-header: rgba(241, 248, 255, 0.98) !important;
          --gdg-bg-group-header-hovered: rgba(0, 191, 232, 0.08) !important;
          --gdg-text-header-selected: #0d284a !important;
          --gdg-bg-cell: rgba(255, 255, 255, 0.96) !important;
          --gdg-bg-cell-medium: rgba(244, 250, 255, 0.98) !important;
          --gdg-bg-header: rgba(241, 248, 255, 0.98) !important;
          --gdg-bg-header-has-focus: rgba(0, 191, 232, 0.10) !important;
          --gdg-bg-header-hovered: rgba(0, 191, 232, 0.06) !important;
          --gdg-bg-bubble: rgba(244, 250, 255, 0.98) !important;
          --gdg-bg-bubble-selected: rgba(0, 191, 232, 0.14) !important;
          --gdg-bg-search-result: rgba(16, 185, 129, 0.12) !important;
          --gdg-border-color: rgba(0, 200, 255, 0.10) !important;
          --gdg-horizontal-border-color: rgba(0, 200, 255, 0.08) !important;
          --gdg-drilldown-border: rgba(0, 200, 255, 0.14) !important;
          --gdg-link-color: #00a7d6 !important;
          --gdg-resize-indicator-color: #10b981 !important;
        }
        div[data-testid="stElementToolbarButtonContainer"] {
          background: rgba(255,255,255,0.88) !important;
          border-color: rgba(0, 200, 255, 0.12) !important;
          box-shadow: 0 8px 20px rgba(18, 78, 138, 0.08) !important;
        }
        button[data-testid="stBaseButton-elementToolbar"] {
          color: rgba(15, 39, 72, 0.66) !important;
        }
        button[data-testid="stBaseButton-elementToolbar"]:hover {
          color: #00a7d6 !important;
          background: rgba(0, 191, 232, 0.08) !important;
        }
        div[data-testid="stDataFrame"] canvas,
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td {
          background: rgba(255,255,255,0.96) !important;
          color: rgba(15, 39, 72, 0.88) !important;
          border-color: rgba(0, 200, 255, 0.08) !important;
        }
        div[data-testid="stDataFrame"] th {
          color: #009fd9 !important;
        }
        code {
          color: #0d6ea9 !important;
          background: rgba(0, 191, 232, 0.08) !important;
        }
        hr {
          border-color: rgba(0, 200, 255, 0.10) !important;
        }
        .panel-table-shell {
          background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(241,250,255,0.92)) !important;
          border-color: rgba(0, 200, 255, 0.16) !important;
          box-shadow: 0 12px 26px rgba(18, 78, 138, 0.08), 0 0 0 1px rgba(255,255,255,0.68) inset !important;
        }
        .panel-table-grid {
          color: rgba(15, 39, 72, 0.88) !important;
        }
        .panel-table-grid thead th {
          background: rgba(236,247,255,0.96) !important;
          color: #009fd9 !important;
        }
        .panel-table-grid tbody tr:nth-child(odd) {
          background: rgba(252,254,255,0.98) !important;
        }
        .panel-table-grid tbody tr:nth-child(even) {
          background: rgba(244,250,255,0.96) !important;
        }
        .panel-table-grid tbody tr:hover {
          background: rgba(0, 243, 255, 0.08) !important;
        }
        .panel-table-grid th,
        .panel-table-grid td {
          border-bottom-color: rgba(0, 200, 255, 0.10) !important;
        }
        .panel-table-grid .panel-table-index {
          color: rgba(15, 39, 72, 0.46) !important;
        }
    """


if __name__ == "__main__":
    main()
