from __future__ import annotations

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
from gaze_toolkit.io import from_frame
from gaze_toolkit.types import GazeRecording
from gaze_toolkit.visualization import (
    plot_confusion,
    plot_feature_importance,
    plot_heatmap,
    plot_metrics,
    plot_scanpath,
    plot_signal_overview,
)

ACCENT = "#c8553d"
PAPER = "#f5f1e8"
INK = "#1f1f1f"
GRID = "#d9cfbc"
STYLE_OPTIONS = {"精读": "careful", "略读": "skim"}
MODEL_OPTIONS = {
    "随机森林": "random_forest",
    "梯度提升树": "gradient_boosting",
    "支持向量机": "svm",
    "逻辑回归": "logistic_regression",
}


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="眼动人因智能实验台",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">面向人因研究面试的作品集演示</div>
          <h1>眼动与多模态人因智能实验台</h1>
          <p>
            用可运行的研究型界面，展示眼动数据处理、多模态特征融合、状态与意图建模，
            以及面向人因专家的工程化分析能力。
          </p>
          <div class="hero-credit">Powered by 谭青松的求职作品集</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assumptions = (
        "说明：这个 MVP 优先服务本地演示场景，默认使用 CSV 或模拟眼动数据，并用轻量心率信号展示多模态扩展能力。"
        "EDF 原生解析和更重的深度基础模型当前保留为扩展点，而不是默认依赖。"
    )
    st.info(assumptions)

    recording, stimulus_image = _build_recording_from_sidebar()
    analysis = _run_sidebar_analysis(recording)

    tabs = st.tabs(
        [
            "能力证明",
            "单次会话分析",
            "意图建模实验",
            "多模态融合",
            "面试讲述提纲",
        ]
    )

    with tabs[0]:
        _render_capability_story(analysis)

    with tabs[1]:
        _render_single_session(analysis, stimulus_image=stimulus_image)

    with tabs[2]:
        _render_modeling_workbench()

    with tabs[3]:
        _render_multimodal_tab(recording)

    with tabs[4]:
        _render_portfolio_talking_points()


def _build_recording_from_sidebar() -> tuple[GazeRecording, str | Path | None]:
    st.sidebar.header("实验输入设置")
    source = st.sidebar.radio("数据来源", options=["合成演示数据", "上传 CSV 文件"], index=0)
    sampling_rate_hz = float(st.sidebar.number_input("采样率（Hz）", min_value=30, max_value=1000, value=120))
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
        return recording, None

    uploaded = st.sidebar.file_uploader("上传眼动 CSV", type=["csv"])
    uploaded_stimulus = st.sidebar.file_uploader("上传刺激图片（可选）", type=["png", "jpg", "jpeg"])
    if uploaded_stimulus is not None:
        stimulus_image = uploaded_stimulus

    if uploaded is None:
        st.sidebar.warning("尚未上传 CSV，系统将回退到默认合成演示数据。")
        return simulate_gaze_recording(seed=42), stimulus_image

    frame = pd.read_csv(uploaded)
    recording = from_frame(frame, sampling_rate_hz=sampling_rate_hz, source_format="csv_upload")
    recording.metadata["intent_label"] = "unknown"
    return recording, stimulus_image


def _run_sidebar_analysis(recording: GazeRecording) -> RecordingAnalysis:
    st.sidebar.header("分析流程控制")
    smooth_window = int(st.sidebar.slider("平滑窗口", min_value=3, max_value=21, step=2, value=5))
    velocity_threshold = float(
        st.sidebar.slider("扫视速度阈值", min_value=200, max_value=1800, value=850, step=50)
    )
    min_fixation_ms = float(st.sidebar.slider("最小注视时长（ms）", min_value=30, max_value=200, value=60, step=10))
    blink_min_duration_ms = float(
        st.sidebar.slider("最小眨眼时长（ms）", min_value=40, max_value=250, value=75, step=5)
    )
    include_complexity = st.sidebar.toggle("包含复杂度特征", value=True)

    return analyze_recording(
        recording,
        preprocess_params={"smooth_window": smooth_window},
        event_params={
            "velocity_threshold": velocity_threshold,
            "min_fixation_ms": min_fixation_ms,
            "blink_min_duration_ms": blink_min_duration_ms,
        },
        feature_params={"include_complexity": include_complexity},
    )


def _render_capability_story(analysis: RecordingAnalysis) -> None:
    left, right = st.columns([1.1, 1.3], gap="large")

    with left:
        st.subheader("这个界面要证明什么")
        st.markdown(
            """
            这个界面是为面试场景专门设计的，重点证明：

            - 我能把原始眼动数据组织成可复用的研究对象。
            - 我能自动提取人因研究常用特征，而不是只做清洗和画图。
            - 我能构建状态/意图模型，而不是停留在可视化层。
            - 我能把眼动链路扩展到多模态用户信号。
            - 我能把研究流程封装成别人可直接操作的工具界面。
            """
        )

        st.subheader("当前单次会话证据")
        metrics = analysis.quality_summary
        metric_cols = st.columns(4)
        metric_cols[0].metric("有效样本占比", f"{metrics['valid_ratio']:.2%}")
        metric_cols[1].metric("注视次数", int(metrics["fixation_count"]))
        metric_cols[2].metric("扫视次数", int(metrics["saccade_count"]))
        metric_cols[3].metric("眨眼次数", int(metrics["blink_count"]))

    with right:
        st.subheader("能力矩阵")
        capability_frame = pd.DataFrame(
            [
                ["眼动预处理", "已实现", "插值、平滑、坐标归一化"],
                ["事件检测", "已实现", "I-VT 注视、扫视、眨眼"],
                ["特征自动化", "已实现", "注视、扫视、瞳孔、复杂度特征"],
                ["意图建模", "已实现", "精读 vs 略读基线实验"],
                ["多模态融合", "已实现", "眼动 + 心率早期融合基线"],
                ["研究界面", "已实现", "可交互 Streamlit 实验台"],
                ["EDF / 深度基础模型", "扩展点", "当前没有强行打进默认 MVP"],
            ],
            columns=["能力项", "状态", "说明"],
        )
        st.dataframe(capability_frame, hide_index=True, use_container_width=True)


def _render_single_session(analysis: RecordingAnalysis, stimulus_image: str | Path | None = None) -> None:
    st.subheader("单次会话分析链路")
    st.caption("输入数据 -> 预处理 -> 事件检测 -> 特征提取 -> 研究摘要")

    metric_cols = st.columns(5)
    feature_map = analysis.features
    metric_cols[0].metric("总时长", f"{feature_map['duration_ms'] / 1000:.1f}s")
    metric_cols[1].metric("路径长度", f"{feature_map['path_length']:.0f}")
    metric_cols[2].metric("平均速度", f"{feature_map['velocity_mean']:.1f}")
    metric_cols[3].metric("眨眼频率", f"{feature_map['blink_rate_hz']:.2f} Hz")
    metric_cols[4].metric("基线瞳孔值", f"{feature_map['pupil_baseline']:.2f}")

    left, right = st.columns(2, gap="large")
    with left:
        figure, axis = plt.subplots(figsize=(7, 4.2))
        plot_scanpath(analysis.enriched_recording, background_image=stimulus_image, ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

        signal_figure = plot_signal_overview(analysis.enriched_recording, velocity=analysis.velocity_profile)
        st.pyplot(signal_figure, clear_figure=True, use_container_width=True)

    with right:
        figure, axis = plt.subplots(figsize=(7, 4.2))
        plot_heatmap(analysis.enriched_recording, ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

        st.markdown("**事件表**")
        event_table = analysis.event_table.rename(
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
        event_table["事件类型"] = event_table["事件类型"].replace(
            {"fixation": "注视", "saccade": "扫视", "blink": "眨眼"}
        )
        st.dataframe(event_table.head(20), use_container_width=True)

    st.markdown("**本次会话关键特征**")
    top_features = (
        pd.Series(analysis.features, name="value")
        .sort_values(ascending=False)
        .head(18)
        .rename_axis("特征")
        .reset_index()
    )
    top_features = top_features.rename(columns={"value": "数值"})
    top_features["特征"] = top_features["特征"].map(_localize_feature_name)
    st.dataframe(top_features, hide_index=True, use_container_width=True)


def _render_modeling_workbench() -> None:
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
        plot_metrics(report.result.metrics, ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)

        if not report.holdout_predictions.empty:
            figure, axis = plt.subplots(figsize=(5, 5))
            plot_confusion(
                report.holdout_predictions["y_true"].to_numpy(),
                report.holdout_predictions["y_pred"].to_numpy(),
                ax=axis,
            )
            st.pyplot(figure, clear_figure=True, use_container_width=True)

    with right:
        figure, axis = plt.subplots(figsize=(6, 4.8))
        localized_importance = report.feature_importance.copy()
        localized_importance["feature"] = localized_importance["feature"].map(_localize_feature_name)
        plot_feature_importance(localized_importance, ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)
        importance = localized_importance.head(15).rename(
            columns={
                "feature": "特征",
                "importance_mean": "平均重要性",
                "importance_std": "重要性标准差",
            }
        )
        st.dataframe(importance, hide_index=True, use_container_width=True)


def _render_multimodal_tab(recording: GazeRecording) -> None:
    st.subheader("多模态融合演示")
    st.caption("默认模态组合：眼动 + 模拟心率信号")

    heart_signal, heart_features = synthesize_heart_rate_preview(recording, seed=99)
    comparison = compare_modalities(num_sessions=32, model_name="random_forest", random_state=42)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("**当前会话心率预览**")
        heart_signal_cn = heart_signal.rename(columns={"heart_rate_bpm": "心率（bpm）"})
        st.line_chart(heart_signal_cn.set_index("timestamp_ms"), height=260)
        heart_feature_frame = pd.Series(heart_features, name="数值").rename_axis("特征").reset_index()
        heart_feature_frame["特征"] = heart_feature_frame["特征"].map(_localize_feature_name)
        st.dataframe(heart_feature_frame, hide_index=True, use_container_width=True)

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
        st.dataframe(summary, hide_index=True, use_container_width=True)
        delta = (
            comparison.multimodal.result.metrics.get("accuracy", 0.0)
            - comparison.gaze_only.result.metrics.get("accuracy", 0.0)
        )
        st.metric("引入心率后的准确率变化", f"{delta:+.3f}")

    st.markdown("**如何解读这部分**")
    st.markdown(
        """
        这一页的目标不是制造一个复杂的生理平台，而是证明当前架构已经能够接入第二路生理信号，
        完成时间对齐、模态特征构建，以及单模态和多模态建模的对比。
        当前心率是模拟信号，用来证明工程链路；换成真实 HR、EDA、EEG 或交互行为数据时，
        更像是数据源替换，而不是重写架构。
        """
    )


def _render_portfolio_talking_points() -> None:
    st.subheader("面试讲述提纲")
    st.markdown(
        """
        **输入**
        这个原型既能接收真实眼动时序，也能生成可重复的合成会话数据用于演示。

        **处理流程**
        它会先做坐标标准化、缺失值插值、平滑滤波，再识别注视、扫视、眨眼，并自动提取建模特征。

        **状态变化**
        整条链路显式保留原始记录、预处理记录、事件增强记录、特征表和模型输出，便于复核与解释。

        **输出**
        研究人员可以在同一个界面里看到扫描路径、热力图、信号曲线、事件表、特征摘要、基线指标和多模态对比结果。

        **上下游影响**
        上游只要求时间戳和基本坐标；下游则可以继续接 notebook、批量实验、在线预测和更复杂的深度模型。
        """
    )

    st.warning(
        "未验证前提：当前界面假设上传的 CSV 能被映射到 timestamp/x/y/（可选 pupil、valid）列。"
        "如果是厂商专有导出格式，仍需要预转换或定制 loader。"
    )
    st.caption("Powered by 谭青松的求职作品集")


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


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
          background:
            radial-gradient(circle at top left, rgba(200, 85, 61, 0.12), transparent 28%),
            linear-gradient(180deg, {PAPER} 0%, #f9f6ef 45%, #efe7d9 100%);
          color: {INK};
        }}
        .hero {{
          padding: 1.6rem 1.8rem;
          border: 1px solid rgba(31, 31, 31, 0.14);
          background: rgba(255, 252, 246, 0.86);
          box-shadow: 0 12px 42px rgba(75, 55, 30, 0.08);
          margin-bottom: 1.25rem;
        }}
        .hero-kicker {{
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: {ACCENT};
          font-size: 0.75rem;
          margin-bottom: 0.25rem;
          font-weight: 700;
        }}
        .hero h1 {{
          font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
          font-size: 3rem;
          line-height: 1;
          margin: 0 0 0.5rem 0;
        }}
        .hero p, .stMarkdown, .stCaption, .stDataFrame {{
          font-family: "Aptos", "Segoe UI", sans-serif;
        }}
        .hero-credit {{
          margin-top: 0.9rem;
          font-size: 0.95rem;
          color: {ACCENT};
          font-weight: 600;
        }}
        [data-testid="stSidebar"] {{
          background:
            linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245, 238, 225, 0.96));
          border-right: 1px solid rgba(31, 31, 31, 0.08);
        }}
        div[data-testid="stMetric"] {{
          background: rgba(255,255,255,0.72);
          border: 1px solid rgba(31,31,31,0.08);
          padding: 0.7rem 0.85rem;
        }}
        div[data-testid="stTabs"] button {{
          font-weight: 600;
        }}
        .stAlert {{
          border-radius: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
