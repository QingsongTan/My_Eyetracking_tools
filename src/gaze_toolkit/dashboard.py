from __future__ import annotations

import html
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from PIL import Image

from gaze_toolkit.aoi import (
    AOI,
    assign_fixations_to_aoi,
    compute_aoi_metrics,
    compute_transition_matrix,
    define_aoi,
)
from gaze_toolkit.analysis import (
    RecordingAnalysis,
    analyze_recording,
    compare_modalities,
    run_intent_experiment,
    synthesize_heart_rate_preview,
)
from gaze_toolkit.batch import (
    batch_analyze,
    build_html_report_content,
    build_markdown_report_content,
)
from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.events import has_labeled_events
from gaze_toolkit.io import from_frame
from gaze_toolkit.pipeline import build_feature_dataset
from gaze_toolkit.quality import QualityReport, assess_quality, find_missing_segments, format_quality_cards
from gaze_toolkit.saliency import (
    COGNITIVE_SALIENCY_BACKEND,
    FAST_SALIENCY_BACKEND,
    get_saliency_backend_status,
    probe_deepgaze_runtime,
    predict_image_attention,
)
from gaze_toolkit.scenarios import (
    SCENARIOS_DIR,
    ScenarioTask,
    get_scenario_aois,
    list_scenarios,
    load_scenario,
)
from gaze_toolkit.segmentation import segment_recording
from gaze_toolkit.statistics import (
    compare_conditions,
    descriptive_table,
    independent_t_test,
    mann_whitney_test,
    paired_t_test,
    wilcoxon_test,
)
from gaze_toolkit.tables import fixation_table
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


def _ensure_streamlit_drawable_canvas_compatibility() -> None:
    """Patch Streamlit image helpers so streamlit-drawable-canvas works on newer releases."""
    try:
        from streamlit.elements import image as st_image

        if hasattr(st_image, "image_to_url"):
            return

        from streamlit.elements.lib.image_utils import image_to_url as streamlit_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig

        def _legacy_image_to_url(
            image: Any,
            width: int | str | None,
            clamp: bool,
            channels: str,
            output_format: str,
            image_id: str,
        ) -> str:
            return streamlit_image_to_url(
                image,
                layout_config=LayoutConfig(width=width),
                clamp=clamp,
                channels=channels,
                output_format=output_format,
                image_id=image_id,
            )

        st_image.image_to_url = _legacy_image_to_url
    except Exception:
        return


try:
    _ensure_streamlit_drawable_canvas_compatibility()
    from streamlit_drawable_canvas import st_canvas

    _HAS_AOI_CANVAS = True
except Exception:
    st_canvas = None
    _HAS_AOI_CANVAS = False

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
AOI_STATE_KEY = "dashboard_aois"
AOI_CANVAS_DRAFT_KEY = "dashboard_aoi_canvas"
AOI_CANVAS_MODE_KEY = "dashboard_aoi_canvas_mode"
AOI_CANVAS_RESTORE_KEY = "dashboard_aoi_canvas_restore"
AOI_SCANPATH_KEY = "single-session-aoi-scanpath"
AOI_TRANSITION_KEY = "single-session-aoi-transition"
DASHBOARD_ACTIVE_TAB_KEY = "dashboard_active_tab"
SCENARIO_LINKED_KEY = "dashboard_linked_scenario"
SCENARIO_IMPORT_NOTICE_KEY = "dashboard_scenario_import_notice"
BATCH_RESULTS_KEY = "dashboard_batch_results"
AOI_CANVAS_WIDTH = 640


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

    tab_labels = [
        "研究概览",
        "单次会话分析",
        "意图建模实验",
        "多模态融合",
        "批量分析",
        "项目解读",
        "产品评测场景",
    ]
    default_tab = st.session_state.pop(DASHBOARD_ACTIVE_TAB_KEY, None)
    if default_tab not in tab_labels:
        default_tab = None
    tabs = st.tabs(tab_labels, default=default_tab, key="dashboard-main-tabs")

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
        _render_batch_tab(theme_name=theme_name)

    with tabs[5]:
        _render_portfolio_talking_points()

    with tabs[6]:
        _render_scenario_tab()


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


def _render_quality_cards(report: QualityReport, theme_name: str) -> None:
    del theme_name

    status_labels = {
        "good": "🟢 good",
        "warn": "🟡 warn",
        "bad": "🔴 bad",
        "info": "⚪ info",
    }
    cards = format_quality_cards(report)
    primary_cards = cards[:6]
    secondary_cards = cards[6:]

    st.markdown("### 数据质量速览")
    for card_group in [primary_cards, secondary_cards]:
        if not card_group:
            continue
        columns = st.columns(len(card_group), gap="small")
        for column, card in zip(columns, card_group, strict=False):
            with column:
                st.metric(card["label"], card["value"])
                st.caption(status_labels.get(card["status"], "⚪ info"))


def _render_quality_detail(recording: GazeRecording, report: QualityReport, theme_name: str) -> None:
    with st.expander("数据质量详情", expanded=False):
        missing_segments = find_missing_segments(recording.samples)
        if not missing_segments:
            st.info("无数据缺失")
            return

        detail_frame = pd.DataFrame(missing_segments)
        detail_frame["start_time_s"] = detail_frame["start_time_ms"] / 1000.0
        detail_frame["end_time_s"] = detail_frame["end_time_ms"] / 1000.0
        total_duration_ms = max(report.recording_duration_s * 1000.0, 0.0)
        detail_frame["percentage_of_total"] = (
            detail_frame["duration_ms"] / total_duration_ms if total_duration_ms > 0.0 else 0.0
        )

        timeline_frame = detail_frame.sort_values("start_time_ms").reset_index(drop=True)
        figure = go.Figure(
            data=[
                go.Bar(
                    x=timeline_frame["duration_ms"] / 1000.0,
                    y=["缺失段"] * len(timeline_frame),
                    base=timeline_frame["start_time_s"],
                    orientation="h",
                    marker={"color": "#FF7A59"},
                    hovertemplate=(
                        "开始=%{base:.3f} s<br>"
                        "持续=%{x:.3f} s<br>"
                        "结束=%{customdata[0]:.3f} s<br>"
                        "样本数=%{customdata[1]}<extra></extra>"
                    ),
                    customdata=timeline_frame[["end_time_s", "sample_count"]].to_numpy(),
                    name="缺失段",
                )
            ]
        )
        figure.update_layout(
            title="缺失段时间轴分布",
            template="plotly_white" if theme_name == "light" else "plotly_dark",
            height=240,
            margin={"l": 20, "r": 20, "t": 44, "b": 24},
            showlegend=False,
            xaxis={"title": "时间 (s)"},
            yaxis={"title": "", "showticklabels": False},
        )
        st.plotly_chart(
            figure,
            key="single-session-quality-missing-timeline",
            width="stretch",
            config={"displaylogo": False},
        )

        summary_frame = detail_frame.sort_values("duration_ms", ascending=False).reset_index(drop=True)
        summary_frame = summary_frame.loc[
            :,
            ["start_time_s", "end_time_s", "duration_ms", "percentage_of_total"],
        ].copy()
        summary_frame["start_time_s"] = summary_frame["start_time_s"].round(3)
        summary_frame["end_time_s"] = summary_frame["end_time_s"].round(3)
        summary_frame["duration_ms"] = summary_frame["duration_ms"].round(1)
        summary_frame["percentage_of_total"] = summary_frame["percentage_of_total"].map(lambda value: f"{value:.1%}")
        st.dataframe(summary_frame, use_container_width=True, hide_index=True)


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
    import_notice = st.session_state.pop(SCENARIO_IMPORT_NOTICE_KEY, None)
    if import_notice:
        st.success(import_notice)

    quality_report = assess_quality(
        recording=analysis.raw_recording,
        preprocessed=analysis.processed_recording,
    )
    _render_quality_cards(quality_report, theme_name)
    _render_quality_detail(analysis.raw_recording, quality_report, theme_name)

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

    _render_aoi_section(
        recording=display_analysis.enriched_recording,
        stimulus_image=stimulus_image,
        screen_size=screen_size,
        theme_name=theme_name,
        visual_controls=visual_controls,
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
                        "如运行时目录已损坏，可执行 `powershell -File scripts/setup-deepgaze-runtime.ps1 -ForceRecreate` 重建。",
                    ]
                )
            )

def _render_aoi_section(
    *,
    recording: GazeRecording,
    stimulus_image: str | Path | Any | None,
    screen_size: tuple[int, int],
    theme_name: str,
    visual_controls: VisualControls,
) -> None:
    with st.expander("兴趣区域 (AOI) 分析", expanded=False):
        if not recording.events:
            st.info("当前还没有可用于 AOI 分析的事件，请先完成数据加载和事件识别。")
            return

        fixations = fixation_table(recording)
        if fixations.empty:
            st.info("当前记录没有可用于 AOI 分析的 fixation 事件。")
            return

        if AOI_STATE_KEY not in st.session_state:
            st.session_state[AOI_STATE_KEY] = []
        if AOI_CANVAS_DRAFT_KEY not in st.session_state:
            st.session_state[AOI_CANVAS_DRAFT_KEY] = None
        if AOI_CANVAS_MODE_KEY not in st.session_state:
            st.session_state[AOI_CANVAS_MODE_KEY] = None
        if AOI_CANVAS_RESTORE_KEY not in st.session_state:
            st.session_state[AOI_CANVAS_RESTORE_KEY] = False

        left, right = st.columns([1.05, 1.35], gap="large")

        with left:
            st.markdown("### AOI 定义区")
            st.caption("AOI 分析基于 fixation_table() 的注视中心，不直接操作原始 samples。")

            mode = st.radio("AOI 定义方式", options=["手动输入", "鼠标绘制"], key="aoi-mode", horizontal=True)
            previous_mode = st.session_state.get(AOI_CANVAS_MODE_KEY)
            if _should_restore_canvas_draft(previous_mode=previous_mode, current_mode=mode):
                st.session_state[AOI_CANVAS_RESTORE_KEY] = True
            st.session_state[AOI_CANVAS_MODE_KEY] = mode
            if mode == "手动输入":
                name = st.text_input(
                    "AOI 名称",
                    value=f"AOI {len(st.session_state[AOI_STATE_KEY]) + 1}",
                    key="aoi-name",
                )
                x_col, y_col, x2_col, y2_col = st.columns(4)
                x_min = float(
                    x_col.number_input(
                        "x_min",
                        min_value=0.0,
                        max_value=float(screen_size[0]),
                        value=0.0,
                        step=10.0,
                    )
                )
                y_min = float(
                    y_col.number_input(
                        "y_min",
                        min_value=0.0,
                        max_value=float(screen_size[1]),
                        value=0.0,
                        step=10.0,
                    )
                )
                x_max = float(
                    x2_col.number_input(
                        "x_max",
                        min_value=0.0,
                        max_value=float(screen_size[0]),
                        value=float(screen_size[0] / 2),
                        step=10.0,
                    )
                )
                y_max = float(
                    y2_col.number_input(
                        "y_max",
                        min_value=0.0,
                        max_value=float(screen_size[1]),
                        value=float(screen_size[1] / 2),
                        step=10.0,
                    )
                )

                if st.button("添加 AOI", key="aoi-add-button", use_container_width=True):
                    normalized_name = name.strip() or f"AOI {len(st.session_state[AOI_STATE_KEY]) + 1}"
                    st.session_state[AOI_STATE_KEY] = [
                        *st.session_state[AOI_STATE_KEY],
                        define_aoi(normalized_name, x_min, y_min, x_max, y_max),
                    ]
            else:
                _render_canvas_aoi_builder(stimulus_image=stimulus_image, screen_size=screen_size)
                st.caption("示例 AOI 和手动输入会直接作用于分析结果，不自动回填到画布。")

            example_cols = st.columns(3)
            if example_cols[0].button("示例: 屏幕四象限", key="aoi-example-quadrants", use_container_width=True):
                st.session_state[AOI_STATE_KEY] = _example_quadrant_aois(screen_size)
            if example_cols[1].button("示例: 顶栏+内容区", key="aoi-example-layout", use_container_width=True):
                st.session_state[AOI_STATE_KEY] = _example_layout_aois(screen_size)
            if example_cols[2].button("清空 AOI", key="aoi-clear", use_container_width=True):
                st.session_state[AOI_STATE_KEY] = []

            aois = list(st.session_state[AOI_STATE_KEY])
            if aois:
                st.dataframe(_aoi_definition_frame(aois), use_container_width=True, hide_index=True)
                _render_aoi_name_editor(aois)
                aois = list(st.session_state[AOI_STATE_KEY])
                st.plotly_chart(
                    _build_aoi_scanpath_figure(
                        recording=recording,
                        aois=aois,
                        stimulus_image=stimulus_image,
                        screen_size=screen_size,
                        theme_name=theme_name,
                        visual_controls=visual_controls,
                    ),
                    key=AOI_SCANPATH_KEY,
                    width="stretch",
                    config={"displaylogo": False},
                )
            else:
                st.info("先手动添加 AOI，或点击上方示例按钮快速生成一组区域。")

        with right:
            st.markdown("### AOI 分析结果")
            aois = list(st.session_state[AOI_STATE_KEY])
            if not aois:
                st.info("定义至少一个 AOI 后，系统会显示指标汇总表和转移矩阵。")
                return

            assigned = assign_fixations_to_aoi(fixations, aois)
            metrics = compute_aoi_metrics(assigned, aois, total_duration=recording.duration_ms)
            metric_frame = _aoi_metrics_frame(metrics)
            st.dataframe(metric_frame, use_container_width=True, hide_index=True)

            transition_matrix = compute_transition_matrix(assigned, [aoi.name for aoi in aois])
            st.plotly_chart(
                _build_aoi_transition_figure(transition_matrix, theme_name=theme_name),
                key=AOI_TRANSITION_KEY,
                width="stretch",
                config={"displaylogo": False},
            )


def _render_canvas_aoi_builder(
    *,
    stimulus_image: str | Path | Any | None,
    screen_size: tuple[int, int],
) -> None:
    if not _HAS_AOI_CANVAS or st_canvas is None:
        st.warning("当前环境未启用鼠标绘制组件，请继续使用手动输入模式。")
        return

    canvas_width, canvas_height = _aoi_canvas_dimensions(screen_size)
    canvas_background = _load_canvas_background_image(stimulus_image)
    restore_draft = bool(st.session_state.get(AOI_CANVAS_RESTORE_KEY))
    canvas_result = st_canvas(
        fill_color="rgba(0, 243, 255, 0.14)",
        stroke_width=2,
        stroke_color="rgba(0, 243, 255, 0.90)",
        background_image=canvas_background,
        background_color=PAPER,
        width=canvas_width,
        height=canvas_height,
        drawing_mode="rect",
        display_toolbar=True,
        update_streamlit=True,
        initial_drawing=_build_canvas_initial_drawing(
            st.session_state.get(AOI_CANVAS_DRAFT_KEY),
            restore_draft=restore_draft,
        ),
        key="aoi-canvas-v1",
    )
    if restore_draft:
        st.session_state[AOI_CANVAS_RESTORE_KEY] = False
    current_canvas_json = canvas_result.json_data if canvas_result is not None else None
    if current_canvas_json is not None:
        st.session_state[AOI_CANVAS_DRAFT_KEY] = current_canvas_json

    st.caption("仅支持矩形。绘制完成后点击下方按钮，才会覆盖当前 AOI 分析结果。")
    if st.button("用画布覆盖当前 AOI", key="aoi-apply-canvas", use_container_width=True):
        aois = _parse_canvas_rectangles_to_aois(
            current_canvas_json or st.session_state.get(AOI_CANVAS_DRAFT_KEY),
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            screen_size=screen_size,
        )
        if not aois:
            st.warning("请先绘制至少一个矩形 AOI。")
        else:
            st.session_state[AOI_STATE_KEY] = aois


def _should_restore_canvas_draft(*, previous_mode: str | None, current_mode: str) -> bool:
    return current_mode == "鼠标绘制" and previous_mode != current_mode


def _build_canvas_initial_drawing(
    draft: dict[str, Any] | None,
    *,
    restore_draft: bool,
) -> dict[str, Any] | None:
    if not restore_draft or not isinstance(draft, dict):
        return None
    return draft


def _render_aoi_name_editor(aois: list[AOI]) -> None:
    st.caption("已应用 AOI 可在此重命名。重新从画布覆盖后，将按当前矩形列表重新生成默认名称。")
    with st.form("aoi-rename-form"):
        renamed: list[str] = []
        for index, aoi in enumerate(aois):
            renamed.append(
                st.text_input(
                    f"AOI {index + 1} 名称",
                    value=aoi.name,
                    key=f"aoi-rename-{index}-{_aoi_name_editor_suffix(aoi)}",
                )
            )
        submitted = st.form_submit_button("更新 AOI 名称", use_container_width=True)

    if submitted:
        st.session_state[AOI_STATE_KEY] = [
            AOI(
                name=(name.strip() or f"AOI {index + 1}"),
                region=aoi.region,
                region_type=aoi.region_type,
            )
            for index, (aoi, name) in enumerate(zip(aois, renamed))
        ]


def _aoi_name_editor_suffix(aoi: AOI) -> str:
    if aoi.region_type == "rectangle":
        return "-".join(f"{float(value):.1f}" for value in aoi.region)
    return "-".join(f"{float(x):.1f}-{float(y):.1f}" for x, y in aoi.region)


def _aoi_canvas_dimensions(screen_size: tuple[int, int]) -> tuple[int, int]:
    canvas_width = AOI_CANVAS_WIDTH
    canvas_height = max(180, round(canvas_width * screen_size[1] / max(screen_size[0], 1)))
    return canvas_width, canvas_height


def _load_canvas_background_image(stimulus_image: str | Path | Any | None) -> Image.Image | None:
    if stimulus_image is None:
        return None

    if isinstance(stimulus_image, (str, Path)):
        image = Image.open(Path(stimulus_image))
    else:
        if hasattr(stimulus_image, "seek"):
            stimulus_image.seek(0)
        image = Image.open(stimulus_image)
    return image.convert("RGBA")


def _parse_canvas_rectangles_to_aois(
    json_data: dict[str, Any] | None,
    *,
    canvas_width: int,
    canvas_height: int,
    screen_size: tuple[int, int],
) -> list[AOI]:
    if not json_data:
        return []

    objects = json_data.get("objects")
    if not isinstance(objects, list):
        return []

    scale_x = screen_size[0] / max(canvas_width, 1)
    scale_y = screen_size[1] / max(canvas_height, 1)
    aois: list[AOI] = []
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "rect":
            continue
        if float(obj.get("angle", 0.0)) != 0.0:
            continue

        width = float(obj.get("width", 0.0))
        height = float(obj.get("height", 0.0))
        scaled_width = width * float(obj.get("scaleX", 1.0))
        scaled_height = height * float(obj.get("scaleY", 1.0))
        if scaled_width < 8.0 or scaled_height < 8.0:
            continue

        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        aois.append(
            define_aoi(
                f"AOI {len(aois) + 1}",
                left * scale_x,
                top * scale_y,
                (left + scaled_width) * scale_x,
                (top + scaled_height) * scale_y,
            )
        )
    return aois


def _example_quadrant_aois(screen_size: tuple[int, int]) -> list[AOI]:
    width, height = screen_size
    mid_x = width / 2
    mid_y = height / 2
    return [
        define_aoi("左上", 0, 0, mid_x, mid_y),
        define_aoi("右上", mid_x, 0, width, mid_y),
        define_aoi("左下", 0, mid_y, mid_x, height),
        define_aoi("右下", mid_x, mid_y, width, height),
    ]


def _example_layout_aois(screen_size: tuple[int, int]) -> list[AOI]:
    width, height = screen_size
    header_height = height * 0.16
    side_width = width * 0.28
    return [
        define_aoi("顶部导航", 0, 0, width, header_height),
        define_aoi("左侧内容", 0, header_height, side_width, height),
        define_aoi("主内容区", side_width, header_height, width, height),
    ]


def _aoi_definition_frame(aois: list[AOI]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for aoi in aois:
        rows.append(
            {
                "AOI 名称": aoi.name,
                "类型": "矩形" if aoi.region_type == "rectangle" else "多边形",
                "区域": _describe_aoi_region(aoi),
            }
        )
    return pd.DataFrame(rows)


def _describe_aoi_region(aoi: AOI) -> str:
    if aoi.region_type == "rectangle":
        x_min, y_min, x_max, y_max = aoi.region
        return f"({x_min:.0f}, {y_min:.0f}) -> ({x_max:.0f}, {y_max:.0f})"
    return " -> ".join(f"({x:.0f}, {y:.0f})" for x, y in aoi.region)


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


def _aoi_metrics_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics.values():
        rows.append(
            {
                "AOI": metric.aoi_name,
                "TTFF (ms)": np.nan if metric.first_fixation_time is None else metric.first_fixation_time,
                "总驻留时长 (ms)": metric.total_dwell_time,
                "驻留占比": metric.dwell_proportion,
                "注视次数": metric.fixation_count,
                "访问次数": metric.visit_count,
                "回视次数": metric.revisit_count,
                "平均注视时长 (ms)": metric.mean_fixation_duration,
            }
        )
    return pd.DataFrame(rows)


def _build_aoi_transition_figure(matrix: pd.DataFrame, *, theme_name: str) -> go.Figure:
    template = "plotly_white" if theme_name == "light" else "plotly_dark"
    text_values = matrix.map(lambda value: f"{value:.2f}")
    figure = go.Figure(
        data=
        [
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
    linked_scenario = st.session_state.get(SCENARIO_LINKED_KEY)
    if linked_scenario:
        st.caption(f"当前联动研究模板：{linked_scenario}")
    controls = st.columns([1, 1, 1])
    num_sessions = int(controls[0].slider("合成会话数", min_value=12, max_value=80, value=32, step=4))
    model_label = controls[1].selectbox("模型类型", options=list(MODEL_OPTIONS), index=0)
    random_state = int(controls[2].slider("实验随机种子", min_value=1, max_value=999, value=42))
    feature_df = _build_statistics_feature_dataset(num_sessions=num_sessions, random_state=random_state)

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

    _render_statistics_section(feature_df=feature_df, theme_name=theme_name)


def _build_statistics_feature_dataset(num_sessions: int, random_state: int) -> pd.DataFrame:
    subject_count = max(2, num_sessions // 2)
    recordings: list[GazeRecording] = []
    session_id = 0

    for subject_index in range(subject_count):
        for condition_index, condition in enumerate(("careful", "skim")):
            recording = simulate_gaze_recording(style=condition, seed=random_state + subject_index * 31 + condition_index)
            recording.metadata.update(
                {
                    "session_id": session_id,
                    "subject_id": f"S{subject_index + 1:02d}",
                    "condition": condition,
                    "trial": 1,
                    "intent_label": condition,
                }
            )
            recordings.append(recording)
            session_id += 1

    return build_feature_dataset(recordings, target_key="intent_label")


def _render_statistics_section(*, feature_df: pd.DataFrame, theme_name: str) -> None:
    with st.expander("统计分析", expanded=False):
        if len(feature_df) < 2:
            st.info("至少需要 2 条 recording 才能进行统计分析。")
            return
        if "condition" not in feature_df.columns:
            st.info("当前特征表没有 condition 列，无法进行条件对比。")
            return

        condition_values = [value for value in feature_df["condition"].dropna().astype(str).unique().tolist()]
        if len(condition_values) < 2:
            st.info("当前数据中少于 2 个 condition，无法进行统计分析。")
            return

        numeric_columns = [
            column
            for column in feature_df.select_dtypes(include=[np.number]).columns
            if column not in {"session_id", "trial"}
        ]
        if not numeric_columns:
            st.info("当前特征表没有可用于统计分析的数值指标。")
            return

        st.caption("统计分析基于 build_feature_dataset() 产出的 subject × condition 粒度特征表。")
        controls_top = st.columns([1.2, 1.6, 1.1, 1.0], gap="large")
        selected_conditions = controls_top[0].multiselect(
            "条件选择",
            options=condition_values,
            default=condition_values[:2],
            max_selections=2,
        )
        selected_metrics = controls_top[1].multiselect(
            "指标选择",
            options=numeric_columns,
            default=[metric for metric in ["fixation_count", "fixation_duration_mean", "path_length"] if metric in numeric_columns]
            or numeric_columns[:3],
            format_func=_localize_feature_name,
        )
        test_mode = controls_top[2].selectbox("检验类型", options=["自动选择", "参数检验", "非参数检验"], index=0)
        paired = controls_top[3].toggle("配对设计", value=True)

        if len(selected_conditions) != 2:
            st.info("请选择恰好 2 个 condition。")
            return
        if not selected_metrics:
            st.info("请至少选择 1 个指标。")
            return

        filtered_df = feature_df.loc[feature_df["condition"].isin(selected_conditions)].copy()
        subject_col = "subject_id" if "subject_id" in filtered_df.columns else None

        try:
            descriptive = descriptive_table(filtered_df, "condition", selected_metrics)
            results = _run_statistics_comparison(
                feature_df=filtered_df,
                selected_conditions=selected_conditions,
                selected_metrics=selected_metrics,
                test_mode=test_mode,
                paired=paired,
                subject_col=subject_col,
            )
        except ImportError as exc:
            st.warning(str(exc))
            return
        except ValueError as exc:
            st.warning(str(exc))
            return

        st.markdown("**描述性统计表**")
        st.dataframe(_format_descriptive_table(descriptive), use_container_width=True, hide_index=True)

        st.markdown("**检验结果汇总**")
        result_display = _format_statistics_results(results)
        st.dataframe(result_display, use_container_width=True, hide_index=True)

        lower_left, lower_right = st.columns(2, gap="large")
        with lower_left:
            st.markdown("**箱线图对比**")
            st.plotly_chart(
                _build_statistics_boxplot(
                    feature_df=filtered_df,
                    selected_conditions=selected_conditions,
                    selected_metrics=selected_metrics,
                    theme_name=theme_name,
                ),
                key="statistics-boxplot",
                width="stretch",
                config={"displaylogo": False},
            )
        with lower_right:
            st.markdown("**效应量森林图**")
            st.plotly_chart(
                _build_effect_size_forest(
                    feature_df=filtered_df,
                    results=results,
                    selected_conditions=selected_conditions,
                    paired=paired,
                    subject_col=subject_col,
                    theme_name=theme_name,
                ),
                key="statistics-effect-forest",
                width="stretch",
                config={"displaylogo": False},
            )


def _run_statistics_comparison(
    *,
    feature_df: pd.DataFrame,
    selected_conditions: list[str],
    selected_metrics: list[str],
    test_mode: str,
    paired: bool,
    subject_col: str | None,
) -> pd.DataFrame:
    if test_mode == "自动选择":
        return compare_conditions(
            feature_df,
            condition_col="condition",
            metric_cols=selected_metrics,
            paired=paired,
            subject_col=subject_col,
        )

    rows: list[dict[str, object]] = []
    _validate_dashboard_granularity(feature_df, condition_col="condition", subject_col=subject_col)

    for metric in selected_metrics:
        group1, group2 = _extract_metric_groups(
            feature_df=feature_df,
            metric=metric,
            selected_conditions=selected_conditions,
            paired=paired,
            subject_col=subject_col,
        )
        if test_mode == "参数检验":
            result = paired_t_test(group1, group2, var_name=metric) if paired else independent_t_test(group1, group2, var_name=metric)
        else:
            result = wilcoxon_test(group1, group2, var_name=metric) if paired else mann_whitney_test(group1, group2, var_name=metric)

        rows.append(
            {
                "metric": metric,
                "test_name": result.test_name,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "effect_size_name": result.effect_size_name,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "conclusion": result.conclusion,
            }
        )

    return pd.DataFrame(rows)


def _validate_dashboard_granularity(feature_df: pd.DataFrame, *, condition_col: str, subject_col: str | None) -> None:
    if subject_col is not None and subject_col in feature_df.columns:
        duplicate_mask = feature_df.duplicated([subject_col, condition_col], keep=False)
        if duplicate_mask.any():
            raise ValueError("输入必须是 subject × condition 粒度的汇总表")


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


def _format_descriptive_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if display.empty:
        return display
    display["metric"] = display["metric"].map(_localize_feature_name)
    display["condition"] = display["condition"].astype(str)
    rename_map = {
        "condition": "条件",
        "metric": "指标",
        "mean": "均值",
        "sd": "标准差",
        "median": "中位数",
        "min": "最小值",
        "max": "最大值",
        "n": "样本量",
        "ci_lower": "CI 下界",
        "ci_upper": "CI 上界",
    }
    return display.rename(columns=rename_map)


def _format_statistics_results(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if display.empty:
        return display
    display["metric"] = display["metric"].map(_localize_feature_name)
    display["显著性"] = display["p_value"].map(_significance_stars)
    display["p 值"] = display["p_value"].map(_format_p_for_display)
    display = display.rename(
        columns={
            "metric": "指标",
            "test_name": "检验",
            "statistic": "统计量",
            "effect_size": "效应量",
            "effect_size_name": "效应量类型",
            "ci_lower": "CI 下界",
            "ci_upper": "CI 上界",
            "conclusion": "结论",
        }
    )
    return display[
        ["指标", "检验", "统计量", "p 值", "显著性", "效应量", "效应量类型", "CI 下界", "CI 上界", "结论"]
    ]


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


def _effect_ci_scale(group1: np.ndarray, group2: np.ndarray) -> float:
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))
    pooled_var = (((n1 - 1) * var1) + ((n2 - 1) * var2)) / max(n1 + n2 - 2, 1)
    return float(np.sqrt(max(pooled_var, 0.0)))


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


def _render_batch_tab(theme_name: str = "dark") -> None:
    st.subheader("批量分析")
    uploaded_files = st.file_uploader(
        "上传多个眼动数据文件",
        type=["csv", "tsv", "txt"],
        accept_multiple_files=True,
        key="batch-file-uploader",
    )

    current_aois = list(st.session_state.get(AOI_STATE_KEY, []))
    config_cols = st.columns(2, gap="large")
    include_complexity = config_cols[0].checkbox(
        "包含复杂度特征（ApEn，大文件慎选）",
        value=False,
        key="batch-include-complexity",
    )
    use_current_aois = config_cols[1].checkbox(
        "使用当前已定义的 AOI（如果 session_state 中存在）",
        value=bool(current_aois),
        disabled=not bool(current_aois),
        key="batch-use-current-aois",
    )

    if st.button("开始批量分析", key="batch-run-button", use_container_width=True):
        if not uploaded_files:
            st.warning("请先上传至少一个眼动数据文件。")
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="gaze-toolkit-batch-"))
            aois = current_aois if use_current_aois and current_aois else None
            progress = st.progress(0.0)
            try:
                saved_paths = _save_uploaded_batch_files(uploaded_files, temp_dir)
                result_frames: list[pd.DataFrame] = []
                with st.status("准备批量分析...", expanded=True) as status:
                    total_files = len(saved_paths)
                    for index, path in enumerate(saved_paths, start=1):
                        status.update(label=f"正在分析 {path.name} ({index}/{total_files})", state="running")
                        status.write(f"已提交文件：{path.name}")
                        result_frames.append(
                            batch_analyze(
                                [path],
                                include_complexity=include_complexity,
                                aois=aois,
                            )
                        )
                        progress.progress(index / max(total_files, 1))

                    batch_df = (
                        pd.concat(result_frames, ignore_index=True)
                        if result_frames
                        else pd.DataFrame(columns=["file_path", "error"])
                    )
                    st.session_state[BATCH_RESULTS_KEY] = batch_df
                    status.update(label="批量分析完成", state="complete")
            except Exception as exc:
                st.error(f"批量分析执行失败：{exc}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    batch_df = st.session_state.get(BATCH_RESULTS_KEY)
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        st.info("上传多个文件并点击“开始批量分析”后，这里会显示批量结果、CSV 下载和报告导出。")
        return

    st.markdown("---")
    st.markdown("### 分析结果")
    success_df = _successful_batch_rows(batch_df)
    metric_cols = st.columns(4)
    metric_cols[0].metric("总文件数", int(len(batch_df)))
    metric_cols[1].metric("成功", int(len(success_df)))
    metric_cols[2].metric("失败", int(len(batch_df) - len(success_df)))
    metric_cols[3].metric("平均质量等级", _mean_quality_grade_label(success_df))

    quality_counts = _quality_counts_frame(batch_df)
    if not quality_counts.empty:
        st.plotly_chart(
            _build_batch_quality_figure(quality_counts, theme_name=theme_name),
            key="batch-quality-chart",
            width="stretch",
            config={"displaylogo": False},
        )

    st.markdown("**关键指标描述统计表**")
    st.dataframe(_build_batch_summary_table(batch_df), use_container_width=True, hide_index=True)

    st.markdown("**完整特征表**")
    styled_batch_df = _style_batch_results(batch_df)
    st.dataframe(
        styled_batch_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "file_path": st.column_config.TextColumn("file_path", width="large"),
            "error": st.column_config.TextColumn("error", width="large"),
        },
    )

    st.download_button(
        "下载 CSV",
        data=batch_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="gaze_toolkit_batch_results.csv",
        mime="text/csv",
        use_container_width=True,
        key="batch-download-csv",
    )

    st.markdown("### 报告导出")
    scenario_name = str(st.session_state.get(SCENARIO_LINKED_KEY, ""))
    html_content = build_html_report_content(batch_df, scenario_name=scenario_name)
    markdown_content = build_markdown_report_content(batch_df, scenario_name=scenario_name)
    export_cols = st.columns(2, gap="large")
    export_cols[0].download_button(
        "导出 HTML 报告",
        data=html_content,
        file_name="gaze_toolkit_batch_report.html",
        mime="text/html",
        use_container_width=True,
        key="batch-download-html",
    )
    export_cols[1].download_button(
        "导出 Markdown 报告",
        data=markdown_content,
        file_name="gaze_toolkit_batch_report.md",
        mime="text/markdown",
        use_container_width=True,
        key="batch-download-markdown",
    )


def _save_uploaded_batch_files(uploaded_files: list[Any], temp_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for index, uploaded in enumerate(uploaded_files, start=1):
        original_name = Path(uploaded.name or f"recording_{index}.csv")
        safe_name = f"{index:03d}_{original_name.name}"
        destination = temp_dir / safe_name
        destination.write_bytes(uploaded.getbuffer())
        paths.append(destination)
    return paths


def _successful_batch_rows(batch_df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in batch_df.columns:
        return batch_df.copy()
    errors = batch_df["error"].fillna("").astype(str).str.strip()
    return batch_df.loc[errors.eq("")].copy()


def _quality_counts_frame(batch_df: pd.DataFrame) -> pd.DataFrame:
    if "quality_grade" not in batch_df.columns:
        return pd.DataFrame(columns=["quality_grade", "count"])

    order = ["优", "良", "可用", "建议剔除", "未知"]
    distribution = (
        batch_df["quality_grade"]
        .fillna("未知")
        .astype(str)
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("quality_grade")
        .reset_index(name="count")
    )
    return distribution.loc[distribution["count"] > 0].reset_index(drop=True)


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


def _build_batch_summary_table(batch_df: pd.DataFrame) -> pd.DataFrame:
    success_df = _successful_batch_rows(batch_df)
    rows: list[dict[str, object]] = []
    for metric in ["fixation_count", "fixation_duration_mean", "saccade_count", "valid_ratio"]:
        if metric not in success_df.columns:
            continue
        values = pd.to_numeric(success_df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": _localize_feature_name(metric),
                "mean": round(float(values.mean()), 4),
                "sd": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4),
                "n": int(len(values)),
                "note": "",
            }
        )

    if "quality_grade" in success_df.columns and not success_df.empty:
        quality_mode = success_df["quality_grade"].dropna().astype(str)
        rows.append(
            {
                "metric": "质量等级",
                "mean": "",
                "sd": "",
                "n": int(len(quality_mode)),
                "note": quality_mode.mode().iloc[0] if not quality_mode.empty else "未知",
            }
        )

    return pd.DataFrame(rows, columns=["metric", "mean", "sd", "n", "note"])


def _mean_quality_grade_label(batch_df: pd.DataFrame) -> str:
    if batch_df.empty or "quality_grade" not in batch_df.columns:
        return "无"

    score_map = {"建议剔除": 1.0, "可用": 2.0, "良": 3.0, "优": 4.0}
    reverse_map = {1: "建议剔除", 2: "可用", 3: "良", 4: "优"}
    scores = batch_df["quality_grade"].map(score_map).dropna()
    if scores.empty:
        return "未知"
    rounded = int(min(max(round(float(scores.mean())), 1), 4))
    return reverse_map[rounded]


def _style_batch_results(batch_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _style_row(row: pd.Series) -> list[str]:
        has_error = bool(str(row.get("error", "")).strip()) if not pd.isna(row.get("error", "")) else False
        if has_error:
            return ["background-color: #fff0f0; color: #8f2d2d;" for _ in row]
        return ["" for _ in row]

    return batch_df.style.apply(_style_row, axis=1)


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


def _render_scenario_tab() -> None:
    st.subheader("产品评测场景")
    scenario_names = list_scenarios()
    if not scenario_names:
        st.info("当前还没有可用的产品评测场景模板，请先在 `configs/scenarios/` 目录中添加 YAML 文件。")
        return

    option_labels = _scenario_option_labels(scenario_names)
    selected_name = st.selectbox(
        "场景选择",
        options=scenario_names,
        format_func=lambda value: option_labels.get(value, value),
    )

    try:
        scenario = load_scenario(selected_name)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        st.error(str(exc))
        return

    st.markdown("### 研究场景概览")
    st.markdown(
        f"**产品：** {scenario.product}  |  **设计类型：** {scenario.research_design.type}  |  "
        f"**样本量：** {scenario.research_design.sample_size}"
    )
    st.markdown(f"**研究目标：** {scenario.research_goal}")
    st.markdown(f"**华为相关性：** {scenario.huawei_relevance}")

    design_col, plan_col = st.columns(2, gap="large")
    with design_col:
        st.markdown("### 实验设计")
        st.markdown(f"**自变量：** {scenario.research_design.iv}")
        for dv_group, items in scenario.research_design.dv.items():
            st.markdown(f"**因变量（{_scenario_dv_label(dv_group)}）：**")
            for item in items:
                st.markdown(f"- {item}")
        st.markdown(f"**样本量计算：** {scenario.research_design.sample_size}")
        st.markdown(f"**平衡策略：** {scenario.research_design.counterbalancing}")

    with plan_col:
        st.markdown("### 分析计划")
        st.markdown("**主要分析：**")
        for item in scenario.analysis_plan.get("primary", []):
            st.markdown(f"- {item}")
        st.markdown("**次要分析：**")
        for item in scenario.analysis_plan.get("secondary", []):
            st.markdown(f"- {item}")
        if st.button("与当前数据联动分析", key=f"scenario-link-{selected_name}", use_container_width=True):
            st.session_state[SCENARIO_LINKED_KEY] = scenario.name
            st.session_state[DASHBOARD_ACTIVE_TAB_KEY] = "意图建模实验"
            st.rerun()

    st.markdown("### 任务与 AOI 配置")
    if not scenario.tasks:
        st.info("当前场景还没有配置任务。")
    elif len(scenario.tasks) <= 5:
        task_tabs = st.tabs([task.id for task in scenario.tasks], key=f"scenario-task-tabs-{selected_name}")
        for task, task_tab in zip(scenario.tasks, task_tabs, strict=True):
            with task_tab:
                _render_scenario_task_panel(scenario_name=scenario.name, selected_name=selected_name, task=task)
    else:
        selected_task_id = st.radio(
            "任务切换",
            options=[task.id for task in scenario.tasks],
            horizontal=True,
            key=f"scenario-task-radio-{selected_name}",
        )
        task = next(task for task in scenario.tasks if task.id == selected_task_id)
        _render_scenario_task_panel(scenario_name=scenario.name, selected_name=selected_name, task=task)

    with st.expander("查看完整 YAML 配置", expanded=False):
        st.code(_scenario_yaml_text(selected_name), language="yaml")


def _render_scenario_task_panel(*, scenario_name: str, selected_name: str, task: ScenarioTask) -> None:
    st.markdown(f"**当前任务：{task.id} - {task.description}**")
    st.dataframe(_scenario_task_frame(task), use_container_width=True, hide_index=True)
    if st.button(
        "导入到 AOI 分析",
        key=f"scenario-import-{selected_name}-{task.id}",
        use_container_width=True,
    ):
        scenario = load_scenario(selected_name)
        st.session_state[AOI_STATE_KEY] = get_scenario_aois(scenario, task.id)
        st.session_state[SCENARIO_LINKED_KEY] = scenario_name
        st.session_state[SCENARIO_IMPORT_NOTICE_KEY] = f"已导入 {task.id} 的 AOI 配置，请继续查看单次会话分析。"
        st.session_state[DASHBOARD_ACTIVE_TAB_KEY] = "单次会话分析"
        st.rerun()


def _scenario_option_labels(scenario_names: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for scenario_name in scenario_names:
        try:
            labels[scenario_name] = load_scenario(scenario_name).name
        except (FileNotFoundError, ValueError, yaml.YAMLError):
            labels[scenario_name] = scenario_name
    return labels


def _scenario_dv_label(group_name: str) -> str:
    return {
        "eye_tracking": "眼动",
        "behavior": "行为",
        "subjective": "主观",
    }.get(group_name, group_name)


def _scenario_task_frame(task: ScenarioTask) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in task.aoi_regions:
        x_min, y_min, x_max, y_max = region.region
        rows.append(
            {
                "name": region.name,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )
    return pd.DataFrame(rows, columns=["name", "x_min", "y_min", "x_max", "y_max"])


def _scenario_yaml_text(scenario_name: str) -> str:
    scenario_path = SCENARIOS_DIR / f"{scenario_name}.yaml"
    if not scenario_path.exists():
        return f"# 未找到场景配置：{scenario_path}"
    return scenario_path.read_text(encoding="utf-8")


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
