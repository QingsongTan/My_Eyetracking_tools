from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR

from gaze_toolkit.errors import OptionalDependencyError

ModelFactory = Callable[..., Any]

_MODEL_REGISTRY: dict[tuple[str, str], ModelFactory] = {}


@dataclass
class ModelResult:
    """Trained estimator bundle."""

    estimator: Any
    metrics: dict[str, float]
    feature_names: list[str]
    task: str
    model_name: str
    label_encoder: LabelEncoder | None = None
    X_test: pd.DataFrame | None = None
    y_test: np.ndarray | pd.Series | None = None
    y_pred: np.ndarray | None = None
    y_score: np.ndarray | None = None


def register_model(name: str, task: str, factory: ModelFactory) -> None:
    """Register a custom model builder."""
    _MODEL_REGISTRY[(task.lower(), name.lower())] = factory


def build_model(
    task: str = "classification",
    model_name: str = "random_forest",
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """Instantiate a supported model."""
    normalized_task = task.lower()
    normalized_name = model_name.lower()
    key = (normalized_task, normalized_name)

    if key in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[key](random_state=random_state, **kwargs)

    if normalized_task == "classification":
        factories = {
            "random_forest": lambda: RandomForestClassifier(random_state=random_state, n_estimators=200, **kwargs),
            "gradient_boosting": lambda: GradientBoostingClassifier(random_state=random_state, **kwargs),
            "svm": lambda: SVC(probability=True, random_state=random_state, **kwargs),
            "logistic_regression": lambda: LogisticRegression(
                random_state=random_state,
                max_iter=2000,
                **kwargs,
            ),
        }
    else:
        factories = {
            "random_forest": lambda: RandomForestRegressor(random_state=random_state, n_estimators=200, **kwargs),
            "gradient_boosting": lambda: GradientBoostingRegressor(random_state=random_state, **kwargs),
            "svm": lambda: SVR(**kwargs),
        }

    if normalized_name in factories:
        return factories[normalized_name]()

    if normalized_name in {"xgboost", "lightgbm"}:
        return _build_optional_booster(normalized_name, normalized_task, random_state=random_state, **kwargs)

    raise ValueError(f"Unsupported model `{model_name}` for task `{task}`.")


def train_model(
    dataset: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_name: str = "random_forest",
    test_size: float = 0.25,
    random_state: int = 42,
    feature_columns: list[str] | None = None,
    **model_kwargs: Any,
) -> ModelResult:
    """Train a baseline model and compute holdout metrics."""
    if target not in dataset.columns:
        raise KeyError(f"Target column not found: {target}")

    feature_names = feature_columns or [column for column in dataset.columns if column != target]
    numeric_frame = dataset[feature_names].select_dtypes(include=[np.number]).fillna(0.0)
    y_raw = dataset[target]

    label_encoder: LabelEncoder | None = None
    y: np.ndarray | pd.Series
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0.0)

    stratify = y if task == "classification" and len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        numeric_frame,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    estimator = build_model(task=task, model_name=model_name, random_state=random_state, **model_kwargs)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    scores = _predict_scores(estimator, X_test, task=task)

    if task == "classification":
        metrics = _classification_metrics(estimator, X_test, y_test, predictions, scores)
    else:
        metrics = _regression_metrics(y_test, predictions)

    return ModelResult(
        estimator=estimator,
        metrics=metrics,
        feature_names=list(numeric_frame.columns),
        task=task,
        model_name=model_name,
        label_encoder=label_encoder,
        X_test=X_test,
        y_test=y_test,
        y_pred=predictions,
        y_score=scores,
    )


def predict(result: ModelResult, features: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a fitted model result."""
    matrix = features[result.feature_names].fillna(0.0)
    predicted = result.estimator.predict(matrix)
    if result.label_encoder is None:
        return predicted
    return result.label_encoder.inverse_transform(predicted.astype(int))


def permutation_feature_importance(
    result: ModelResult,
    features: pd.DataFrame,
    target: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Estimate permutation feature importance on a trained model."""
    matrix = features[result.feature_names].fillna(0.0)
    y = target
    if result.label_encoder is not None:
        y = result.label_encoder.transform(target.astype(str))

    importance = permutation_importance(
        result.estimator,
        matrix,
        y,
        n_repeats=n_repeats,
        random_state=42,
    )
    return pd.DataFrame(
        {
            "feature": result.feature_names,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def explain_with_shap(result: ModelResult, features: pd.DataFrame) -> Any:
    """Create a SHAP explainer when the optional dependency is installed."""
    try:
        shap = importlib.import_module("shap")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Install gaze-toolkit[explain] to enable SHAP explanations.") from exc

    matrix = features[result.feature_names].fillna(0.0)
    explainer = shap.Explainer(result.estimator, matrix)
    return explainer(matrix)


def _classification_metrics(
    estimator: Any,
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2 and y_score is not None:
        probabilities = y_score[:, 1] if y_score.ndim > 1 else y_score
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix_trace"] = float(np.trace(cm))
    return metrics


def _predict_scores(estimator: Any, X_test: pd.DataFrame, task: str) -> np.ndarray | None:
    """Return probabilistic or decision scores when available."""
    if task != "classification":
        return None
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X_test))
    return None


def _regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_optional_booster(name: str, task: str, random_state: int, **kwargs: Any) -> Any:
    if name == "xgboost":
        try:
            xgb = importlib.import_module("xgboost")
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError("Install xgboost to use the xgboost model backend.") from exc
        if task == "classification":
            return xgb.XGBClassifier(random_state=random_state, eval_metric="logloss", **kwargs)
        return xgb.XGBRegressor(random_state=random_state, **kwargs)

    try:
        lgbm = importlib.import_module("lightgbm")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Install lightgbm to use the lightgbm model backend.") from exc
    if task == "classification":
        return lgbm.LGBMClassifier(random_state=random_state, **kwargs)
    return lgbm.LGBMRegressor(random_state=random_state, **kwargs)
