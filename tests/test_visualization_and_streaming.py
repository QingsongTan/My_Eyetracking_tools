from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from gaze_toolkit.datasets import simulate_gaze_recording, simulate_intent_dataset
from gaze_toolkit.modeling import train_model
from gaze_toolkit.streaming import SlidingWindowPredictor
from gaze_toolkit.visualization import (
    plot_feature_importance,
    plot_heatmap,
    plot_scanpath,
    plot_signal_overview,
)


def test_visualization_helpers_return_renderable_objects() -> None:
    recording = simulate_gaze_recording(seed=9)

    figure, axis = plt.subplots()
    returned_axis = plot_scanpath(recording, ax=axis)
    assert returned_axis.get_title() == "Scanpath"
    plt.close(figure)

    figure, axis = plt.subplots()
    returned_axis = plot_heatmap(recording, ax=axis)
    assert returned_axis.get_title() == "Gaze Heatmap"
    plt.close(figure)

    signal_figure = plot_signal_overview(recording)
    assert len(signal_figure.axes) == 4
    plt.close(signal_figure)

    importance = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance_mean": [0.3, 0.2, 0.1],
            "importance_std": [0.01, 0.01, 0.01],
        }
    )
    figure, axis = plt.subplots()
    returned_axis = plot_feature_importance(importance, ax=axis)
    assert returned_axis.get_title() == "Top Feature Importance"
    plt.close(figure)


def test_sliding_window_predictor_emits_prediction() -> None:
    dataset = simulate_intent_dataset(num_sessions=16, random_state=4)
    result = train_model(dataset, target="intent_label")
    predictor = SlidingWindowPredictor(result, window_ms=5000.0, step_ms=0.0)
    recording = simulate_gaze_recording(seed=4)

    emitted = None
    for sample in recording.samples.to_dict("records"):
        emitted = predictor.update(sample)
        if emitted is not None:
            break

    assert emitted is not None
    assert emitted.prediction in {"careful", "skim"}

