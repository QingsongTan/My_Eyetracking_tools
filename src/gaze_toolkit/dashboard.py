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


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Human Factors AI Lab",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Portfolio Demo for Human Factors AI</div>
          <h1>Human Factors AI Lab</h1>
          <p>
            A research-style console for eye-tracking analytics, multimodal feature fusion,
            intent modeling, and portfolio-grade evidence generation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assumptions = (
        "Assumption: this MVP is optimized for local demonstration to human factors researchers and interviewers, "
        "using CSV or synthetic gaze data plus a lightweight heart-rate modality. EDF parsing and deeper foundation "
        "models remain extension points rather than bundled defaults."
    )
    st.info(assumptions)

    recording, stimulus_image = _build_recording_from_sidebar()
    analysis = _run_sidebar_analysis(recording)

    tabs = st.tabs(
        [
            "Capability Story",
            "Single Session Analysis",
            "Intent Modeling",
            "Multimodal Fusion",
            "Portfolio Talking Points",
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
    st.sidebar.header("Research Session Setup")
    source = st.sidebar.radio("Input source", options=["Synthetic demo", "Upload CSV"], index=0)
    sampling_rate_hz = float(st.sidebar.number_input("Sampling rate (Hz)", min_value=30, max_value=1000, value=120))
    stimulus_image: str | Path | None = None

    if source == "Synthetic demo":
        style = st.sidebar.selectbox("Reading intent pattern", options=["careful", "skim"], index=0)
        duration_ms = int(st.sidebar.slider("Duration (ms)", min_value=2000, max_value=12000, value=5000, step=500))
        seed = int(st.sidebar.slider("Random seed", min_value=1, max_value=999, value=42))
        recording = simulate_gaze_recording(
            duration_ms=duration_ms,
            sampling_rate_hz=int(sampling_rate_hz),
            style=style,
            seed=seed,
        )
        recording.metadata["intent_label"] = style
        return recording, None

    uploaded = st.sidebar.file_uploader("Upload gaze CSV", type=["csv"])
    uploaded_stimulus = st.sidebar.file_uploader("Optional stimulus image", type=["png", "jpg", "jpeg"])
    if uploaded_stimulus is not None:
        stimulus_image = uploaded_stimulus

    if uploaded is None:
        st.sidebar.warning("No CSV uploaded yet. Falling back to a synthetic demo session.")
        return simulate_gaze_recording(seed=42), stimulus_image

    frame = pd.read_csv(uploaded)
    recording = from_frame(frame, sampling_rate_hz=sampling_rate_hz, source_format="csv_upload")
    recording.metadata["intent_label"] = "unknown"
    return recording, stimulus_image


def _run_sidebar_analysis(recording: GazeRecording) -> RecordingAnalysis:
    st.sidebar.header("Pipeline Controls")
    smooth_window = int(st.sidebar.slider("Smooth window", min_value=3, max_value=21, step=2, value=5))
    velocity_threshold = float(
        st.sidebar.slider("Saccade velocity threshold", min_value=200, max_value=1800, value=850, step=50)
    )
    min_fixation_ms = float(st.sidebar.slider("Min fixation (ms)", min_value=30, max_value=200, value=60, step=10))
    blink_min_duration_ms = float(
        st.sidebar.slider("Min blink duration (ms)", min_value=40, max_value=250, value=75, step=5)
    )
    include_complexity = st.sidebar.toggle("Include complexity features", value=True)

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
        st.subheader("Why this demo exists")
        st.markdown(
            """
            This interface is designed to prove a specific point in an interview:

            - I can structure raw gaze data into reusable research objects.
            - I can automate feature extraction for human factors questions.
            - I can train intent/state models rather than stop at visualization.
            - I can extend the same pipeline toward multimodal user signals.
            - I can wrap the whole workflow into a tool other researchers can operate.
            """
        )

        st.subheader("Current single-session evidence")
        metrics = analysis.quality_summary
        metric_cols = st.columns(4)
        metric_cols[0].metric("Valid ratio", f"{metrics['valid_ratio']:.2%}")
        metric_cols[1].metric("Fixations", int(metrics["fixation_count"]))
        metric_cols[2].metric("Saccades", int(metrics["saccade_count"]))
        metric_cols[3].metric("Blinks", int(metrics["blink_count"]))

    with right:
        st.subheader("Capability matrix")
        capability_frame = pd.DataFrame(
            [
                ["Eye tracking preprocessing", "Implemented", "Interpolation, smoothing, coordinate normalization"],
                ["Event detection", "Implemented", "I-VT fixations, saccades, blinks"],
                ["Feature automation", "Implemented", "Fixation, saccade, pupil, complexity metrics"],
                ["Intent modeling", "Implemented", "Synthetic careful-vs-skim baseline"],
                ["Multimodal fusion", "Implemented", "Eye + heart rate early-fusion baseline"],
                ["Research UI", "Implemented", "Interactive Streamlit workbench"],
                ["EDF / deep foundation models", "Extension point", "Kept out of the default MVP"],
            ],
            columns=["Capability", "Status", "Notes"],
        )
        st.dataframe(capability_frame, hide_index=True, use_container_width=True)


def _render_single_session(analysis: RecordingAnalysis, stimulus_image: str | Path | None = None) -> None:
    st.subheader("Single-session analysis chain")
    st.caption("Input -> preprocess -> event detection -> feature extraction -> research summary")

    metric_cols = st.columns(5)
    feature_map = analysis.features
    metric_cols[0].metric("Duration", f"{feature_map['duration_ms'] / 1000:.1f}s")
    metric_cols[1].metric("Path length", f"{feature_map['path_length']:.0f}")
    metric_cols[2].metric("Mean velocity", f"{feature_map['velocity_mean']:.1f}")
    metric_cols[3].metric("Blink rate", f"{feature_map['blink_rate_hz']:.2f} Hz")
    metric_cols[4].metric("Pupil baseline", f"{feature_map['pupil_baseline']:.2f}")

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

        st.markdown("**Event table**")
        st.dataframe(analysis.event_table.head(20), use_container_width=True)

    st.markdown("**Top features from this session**")
    top_features = (
        pd.Series(analysis.features, name="value")
        .sort_values(ascending=False)
        .head(18)
        .rename_axis("feature")
        .reset_index()
    )
    st.dataframe(top_features, hide_index=True, use_container_width=True)


def _render_modeling_workbench() -> None:
    st.subheader("Intent modeling workbench")
    controls = st.columns([1, 1, 1])
    num_sessions = int(controls[0].slider("Synthetic sessions", min_value=12, max_value=80, value=32, step=4))
    model_name = controls[1].selectbox(
        "Model",
        options=["random_forest", "gradient_boosting", "svm", "logistic_regression"],
        index=0,
    )
    random_state = int(controls[2].slider("Experiment seed", min_value=1, max_value=999, value=42))

    report = run_intent_experiment(num_sessions=num_sessions, model_name=model_name, random_state=random_state)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Accuracy", f"{report.result.metrics.get('accuracy', 0.0):.3f}")
    metric_cols[1].metric("F1 macro", f"{report.result.metrics.get('f1_macro', 0.0):.3f}")
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
        plot_feature_importance(report.feature_importance, ax=axis)
        st.pyplot(figure, clear_figure=True, use_container_width=True)
        st.dataframe(report.feature_importance.head(15), hide_index=True, use_container_width=True)


def _render_multimodal_tab(recording: GazeRecording) -> None:
    st.subheader("Multimodal fusion demo")
    st.caption("Default modality pairing: eye tracking + synthetic heart-rate signal")

    heart_signal, heart_features = synthesize_heart_rate_preview(recording, seed=99)
    comparison = compare_modalities(num_sessions=32, model_name="random_forest", random_state=42)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("**Current session heart-rate preview**")
        st.line_chart(heart_signal.set_index("timestamp_ms"), height=260)
        st.dataframe(
            pd.Series(heart_features, name="value").rename_axis("feature").reset_index(),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.markdown("**Baseline comparison**")
        st.dataframe(comparison.summary, hide_index=True, use_container_width=True)
        delta = (
            comparison.multimodal.result.metrics.get("accuracy", 0.0)
            - comparison.gaze_only.result.metrics.get("accuracy", 0.0)
        )
        st.metric("Accuracy gain from heart-rate fusion", f"{delta:+.3f}")

    st.markdown("**Interpretation**")
    st.markdown(
        """
        This tab is intentionally narrow in scope: it proves the pipeline can fuse a second physiological stream,
        align time series, derive modality-specific features, and compare gaze-only versus multimodal modeling.
        The physiology signal is synthetic in this MVP; replacing it with real HR, EDA, EEG, or interaction traces
        is a data-source change rather than an architectural rewrite.
        """
    )


def _render_portfolio_talking_points() -> None:
    st.subheader("Interview talking points")
    st.markdown(
        """
        **Input**
        This prototype accepts raw gaze streams from CSV-like exports or generates controlled synthetic sessions for reproducible demos.

        **Processing flow**
        It standardizes coordinates, interpolates missing values, smooths samples, detects fixations/saccades/blinks, and derives reusable hand-crafted features.

        **State changes**
        The pipeline explicitly moves through raw recording, processed recording, enriched recording with events, feature table, and model output.

        **Output**
        Researchers get scanpaths, heatmaps, signal traces, event tables, feature summaries, baseline intent metrics, and multimodal comparisons in one interface.

        **Upstream / downstream impact**
        Upstream, the only hard requirement is time-stamped gaze data. Downstream, the same objects can feed notebooks, batch experiments, online prediction, or richer deep models.
        """
    )

    st.warning(
        "Unverified prerequisite: this demo assumes uploaded CSV files can be mapped onto timestamp/x/y/(optional pupil, valid) columns. "
        "Vendor-specific EDF/Tobii exports still require either conversion or a custom loader."
    )


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
