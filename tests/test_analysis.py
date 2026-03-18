from __future__ import annotations

from gaze_toolkit.analysis import analyze_recording, compare_modalities, run_intent_experiment
from gaze_toolkit.datasets import simulate_gaze_recording


def test_analyze_recording_returns_closed_loop_outputs() -> None:
    recording = simulate_gaze_recording(seed=12)
    analysis = analyze_recording(recording)

    assert analysis.features["sample_count"] > 0.0
    assert "valid_ratio" in analysis.quality_summary
    assert analysis.event_table is not None
    assert len(analysis.velocity_profile) == len(analysis.enriched_recording.samples)


def test_compare_modalities_builds_two_reports() -> None:
    comparison = compare_modalities(num_sessions=16, random_state=12)

    assert len(comparison.summary) == 2
    assert "modality" in comparison.summary.columns
    assert comparison.gaze_only.result.metrics["accuracy"] >= 0.0
    assert comparison.multimodal.result.metrics["accuracy"] >= 0.0


def test_run_intent_experiment_exposes_holdout_predictions() -> None:
    report = run_intent_experiment(num_sessions=16, random_state=6)

    assert not report.holdout_predictions.empty
    assert {"y_true", "y_pred"}.issubset(report.holdout_predictions.columns)
