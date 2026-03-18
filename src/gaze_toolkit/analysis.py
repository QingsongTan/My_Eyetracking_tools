from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gaze_toolkit.datasets import (
    extract_heart_rate_features,
    simulate_heart_rate_signal,
    simulate_intent_dataset,
    simulate_multimodal_intent_dataset,
)
from gaze_toolkit.events import attach_events, compute_velocity
from gaze_toolkit.features import extract_features
from gaze_toolkit.modeling import ModelResult, permutation_feature_importance, train_model
from gaze_toolkit.preprocess import preprocess
from gaze_toolkit.types import EyeEvent, GazeRecording


@dataclass
class RecordingAnalysis:
    """Unified output for a single-recording research analysis."""

    raw_recording: GazeRecording
    processed_recording: GazeRecording
    enriched_recording: GazeRecording
    features: dict[str, float]
    event_table: pd.DataFrame
    quality_summary: dict[str, float]
    velocity_profile: pd.Series


@dataclass
class ExperimentReport:
    """Reusable output for intent-model demo experiments."""

    dataset: pd.DataFrame
    result: ModelResult
    feature_importance: pd.DataFrame
    holdout_predictions: pd.DataFrame
    modality_name: str


@dataclass
class ModalityComparison:
    """Comparison of gaze-only and multimodal baseline results."""

    summary: pd.DataFrame
    gaze_only: ExperimentReport
    multimodal: ExperimentReport


def analyze_recording(
    recording: GazeRecording,
    preprocess_params: dict[str, Any] | None = None,
    event_params: dict[str, float] | None = None,
    feature_params: dict[str, Any] | None = None,
) -> RecordingAnalysis:
    """Run the single-recording analysis path used by CLI, notebooks, and UI."""
    preprocess_params = preprocess_params or {}
    event_params = event_params or {}
    feature_params = feature_params or {}

    processed = preprocess(recording, **preprocess_params)
    enriched = attach_events(processed, **event_params)
    feature_map = extract_features(enriched, **feature_params)
    velocity = compute_velocity(enriched)

    event_table = _events_to_frame(enriched.events)
    quality_summary = {
        "sample_count": float(len(recording.samples)),
        "duration_ms": float(recording.duration_ms),
        "valid_ratio": float(recording.samples["valid"].mean()),
        "invalid_ratio": float(1.0 - recording.samples["valid"].mean()),
        "fixation_count": float((event_table["kind"] == "fixation").sum()) if not event_table.empty else 0.0,
        "saccade_count": float((event_table["kind"] == "saccade").sum()) if not event_table.empty else 0.0,
        "blink_count": float((event_table["kind"] == "blink").sum()) if not event_table.empty else 0.0,
        "velocity_peak": float(velocity.max()),
    }

    return RecordingAnalysis(
        raw_recording=recording,
        processed_recording=processed,
        enriched_recording=enriched,
        features=feature_map,
        event_table=event_table,
        quality_summary=quality_summary,
        velocity_profile=velocity,
    )


def run_intent_experiment(
    num_sessions: int = 32,
    model_name: str = "random_forest",
    random_state: int = 42,
    multimodal: bool = False,
) -> ExperimentReport:
    """Train a synthetic intent model for portfolio demos."""
    dataset = (
        simulate_multimodal_intent_dataset(num_sessions=num_sessions, random_state=random_state)
        if multimodal
        else simulate_intent_dataset(num_sessions=num_sessions, random_state=random_state)
    )
    result = train_model(
        dataset,
        target="intent_label",
        model_name=model_name,
        task="classification",
        random_state=random_state,
    )
    importance = permutation_feature_importance(result, dataset[result.feature_names], dataset["intent_label"])
    holdout = _build_holdout_frame(result)

    return ExperimentReport(
        dataset=dataset,
        result=result,
        feature_importance=importance,
        holdout_predictions=holdout,
        modality_name="multimodal" if multimodal else "gaze_only",
    )


def compare_modalities(
    num_sessions: int = 32,
    model_name: str = "random_forest",
    random_state: int = 42,
) -> ModalityComparison:
    """Compare gaze-only and eye+heart-rate baselines."""
    gaze_only = run_intent_experiment(
        num_sessions=num_sessions,
        model_name=model_name,
        random_state=random_state,
        multimodal=False,
    )
    multimodal = run_intent_experiment(
        num_sessions=num_sessions,
        model_name=model_name,
        random_state=random_state,
        multimodal=True,
    )

    summary = pd.DataFrame(
        [
            {
                "modality": "gaze_only",
                **gaze_only.result.metrics,
                "feature_count": len(gaze_only.result.feature_names),
            },
            {
                "modality": "gaze_plus_heart_rate",
                **multimodal.result.metrics,
                "feature_count": len(multimodal.result.feature_names),
            },
        ]
    )
    return ModalityComparison(summary=summary, gaze_only=gaze_only, multimodal=multimodal)


def synthesize_heart_rate_preview(recording: GazeRecording, seed: int | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate a heart-rate signal and summary features for UI preview."""
    signal = simulate_heart_rate_signal(recording, seed=seed)
    return signal, extract_heart_rate_features(signal)


def _events_to_frame(events: list[EyeEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=[
                "kind",
                "start_time_ms",
                "end_time_ms",
                "duration_ms",
                "amplitude",
                "peak_velocity",
                "centroid_x",
                "centroid_y",
            ]
        )

    rows = []
    for event in events:
        rows.append(
            {
                "kind": event.kind,
                "start_time_ms": event.start_time_ms,
                "end_time_ms": event.end_time_ms,
                "duration_ms": event.duration_ms,
                "amplitude": event.amplitude,
                "peak_velocity": event.peak_velocity,
                "centroid_x": event.metadata.get("centroid_x", np.nan),
                "centroid_y": event.metadata.get("centroid_y", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _build_holdout_frame(result: ModelResult) -> pd.DataFrame:
    if result.y_test is None or result.y_pred is None:
        return pd.DataFrame(columns=["y_true", "y_pred"])

    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.y_pred)
    if result.label_encoder is not None:
        y_true_labels = result.label_encoder.inverse_transform(y_true.astype(int))
        y_pred_labels = result.label_encoder.inverse_transform(y_pred.astype(int))
    else:
        y_true_labels = y_true
        y_pred_labels = y_pred

    holdout = pd.DataFrame({"y_true": y_true_labels, "y_pred": y_pred_labels})
    if result.y_score is not None:
        scores = np.asarray(result.y_score)
        if scores.ndim == 1:
            holdout["score"] = scores
        else:
            for index in range(scores.shape[1]):
                holdout[f"score_{index}"] = scores[:, index]
    return holdout
