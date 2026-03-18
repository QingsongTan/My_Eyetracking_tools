from __future__ import annotations

from gaze_toolkit.datasets import simulate_intent_dataset
from gaze_toolkit.modeling import predict, train_model


def test_train_demo_classifier() -> None:
    dataset = simulate_intent_dataset(num_sessions=20, random_state=3)
    result = train_model(dataset, target="intent_label", model_name="random_forest")
    predictions = predict(result, dataset[result.feature_names])

    assert result.feature_names
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert len(predictions) == len(dataset)

