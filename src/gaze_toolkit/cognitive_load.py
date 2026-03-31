"""Cognitive load experiment module for classification and regression workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gaze_toolkit.features import extract_features
from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.modeling import ModelResult, permutation_feature_importance, train_model

DEMO_CLASSIFICATION_TARGET = "cognitive_load_level"
DEMO_REGRESSION_TARGET = "cognitive_load_score"

_METADATA_COLUMNS = frozenset({
    "session_id",
    "subject_id",
    "condition",
    "trial",
    "intent_label",
    "source",
    "file_path",
    DEMO_CLASSIFICATION_TARGET,
    DEMO_REGRESSION_TARGET,
})


@dataclass
class CognitiveLoadExperimentReport:
    """Output for a cognitive load classification or regression experiment."""

    dataset: pd.DataFrame
    result: ModelResult
    feature_importance: pd.DataFrame
    holdout_predictions: pd.DataFrame
    target: str
    task: str
    data_source: str
    note: str


def simulate_cognitive_load_dataset(
    num_sessions: int = 36,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset with cognitive load labels derived from gaze and pupil features."""
    rng = np.random.default_rng(random_state)
    styles = ["careful", "skim"]
    rows: list[dict[str, Any]] = []

    for session_id in range(num_sessions):
        style = styles[session_id % len(styles)]
        recording = simulate_gaze_recording(style=style, seed=random_state + session_id)
        features = extract_features(recording)
        features["session_id"] = session_id

        # Synthesize a continuous cognitive load score from pupil + gaze features
        pupil_component = features.get("pupil_baseline", 3.2) * 8.0
        fixation_component = features.get("fixation_duration_mean", 200.0) / 80.0
        blink_component = features.get("blink_rate_hz", 0.1) * 5.0
        noise = rng.normal(0.0, 2.5)
        score = float(np.clip(pupil_component + fixation_component + blink_component + noise, 0.0, 100.0))
        features[DEMO_REGRESSION_TARGET] = round(score, 2)

        # Discretize into load levels
        if score < 30.0:
            level = "low"
        elif score < 60.0:
            level = "medium"
        else:
            level = "high"
        features[DEMO_CLASSIFICATION_TARGET] = level

        rows.append(features)

    return pd.DataFrame(rows)


def candidate_target_columns(dataset: pd.DataFrame) -> list[str]:
    """Return column names that have at least 2 distinct non-null values (potential targets)."""
    candidates: list[str] = []
    for column in dataset.columns:
        values = dataset[column].dropna()
        if len(values.unique()) >= 2:
            candidates.append(column)
    return candidates


def infer_task_from_target(series: pd.Series) -> str:
    """Infer 'classification' or 'regression' from the target column dtype and cardinality."""
    non_null = series.dropna()
    if non_null.empty:
        return "classification"

    # If the column is numeric and has many unique values relative to length, treat as regression
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        unique_ratio = len(numeric.unique()) / max(len(numeric), 1)
        if unique_ratio > 0.3 and len(numeric.unique()) > 10:
            return "regression"

    return "classification"


def run_cognitive_load_experiment(
    dataset: pd.DataFrame,
    *,
    target: str = DEMO_CLASSIFICATION_TARGET,
    task: str = "classification",
    model_name: str = "random_forest",
    random_state: int = 42,
    data_source: str = "demo",
    note: str = "",
) -> CognitiveLoadExperimentReport:
    """Train a cognitive load model and return a structured experiment report."""
    # Determine feature columns: all numeric columns except target and metadata
    feature_columns = [
        col for col in dataset.select_dtypes(include=[np.number]).columns
        if col != target and col not in _METADATA_COLUMNS
    ]

    result = train_model(
        dataset,
        target=target,
        task=task,
        model_name=model_name,
        random_state=random_state,
        feature_columns=feature_columns,
    )
    importance = permutation_feature_importance(
        result,
        dataset[result.feature_names],
        dataset[target],
    )
    holdout = _build_holdout_frame(result, task=task)

    return CognitiveLoadExperimentReport(
        dataset=dataset,
        result=result,
        feature_importance=importance,
        holdout_predictions=holdout,
        target=target,
        task=task,
        data_source=data_source,
        note=note,
    )


def _build_holdout_frame(result: ModelResult, task: str = "classification") -> pd.DataFrame:
    """Build a holdout prediction DataFrame from ModelResult, including residuals for regression."""
    if result.y_test is None or result.y_pred is None:
        return pd.DataFrame(columns=["y_true", "y_pred"])

    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.y_pred)

    if task == "classification" and result.label_encoder is not None:
        y_true_labels = result.label_encoder.inverse_transform(y_true.astype(int))
        y_pred_labels = result.label_encoder.inverse_transform(y_pred.astype(int))
    else:
        y_true_labels = y_true
        y_pred_labels = y_pred

    holdout = pd.DataFrame({"y_true": y_true_labels, "y_pred": y_pred_labels})

    if task == "regression":
        holdout["residual"] = pd.to_numeric(holdout["y_true"], errors="coerce") - pd.to_numeric(
            holdout["y_pred"], errors="coerce"
        )

    if result.y_score is not None:
        scores = np.asarray(result.y_score)
        if scores.ndim == 1:
            holdout["score"] = scores
        else:
            for index in range(scores.shape[1]):
                holdout[f"score_{index}"] = scores[:, index]

    return holdout
