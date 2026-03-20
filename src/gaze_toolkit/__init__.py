"""Public package interface for gaze-toolkit."""

from gaze_toolkit.analysis import (
    ExperimentReport,
    ModalityComparison,
    RecordingAnalysis,
    analyze_recording,
    compare_modalities,
    run_intent_experiment,
)
from gaze_toolkit.config import load_config
from gaze_toolkit.datasets import create_complete_example_recording, write_complete_example_csv
from gaze_toolkit.events import attach_events, detect_events
from gaze_toolkit.features import extract_features, register_feature
from gaze_toolkit.io import from_frame, load, register_loader
from gaze_toolkit.modeling import (
    ModelResult,
    build_model,
    explain_with_shap,
    permutation_feature_importance,
    predict,
    register_model,
    train_model,
)
from gaze_toolkit.multimodal import MultiModalData, late_fusion
from gaze_toolkit.pipeline import build_feature_dataset, run_pipeline, run_segmented_pipeline
from gaze_toolkit.preprocess import handle_missing_samples, preprocess
from gaze_toolkit.segmentation import (
    RecordingSegment,
    build_segment_feature_table,
    segment_between_markers,
    segment_by_marker_windows,
    segment_by_time_ranges,
    segment_recording,
)
from gaze_toolkit.streaming import OnlinePrediction, SlidingWindowPredictor
from gaze_toolkit.types import EyeEvent, GazeRecording
from gaze_toolkit.visualization import plot_heatmap, plot_scanpath

__all__ = [
    "EyeEvent",
    "ExperimentReport",
    "GazeRecording",
    "ModelResult",
    "ModalityComparison",
    "MultiModalData",
    "OnlinePrediction",
    "RecordingAnalysis",
    "RecordingSegment",
    "SlidingWindowPredictor",
    "analyze_recording",
    "attach_events",
    "build_feature_dataset",
    "build_segment_feature_table",
    "build_model",
    "compare_modalities",
    "create_complete_example_recording",
    "detect_events",
    "explain_with_shap",
    "extract_features",
    "from_frame",
    "handle_missing_samples",
    "late_fusion",
    "load",
    "load_config",
    "permutation_feature_importance",
    "plot_heatmap",
    "plot_scanpath",
    "preprocess",
    "predict",
    "register_feature",
    "register_loader",
    "register_model",
    "run_intent_experiment",
    "run_pipeline",
    "run_segmented_pipeline",
    "segment_between_markers",
    "segment_by_marker_windows",
    "segment_by_time_ranges",
    "segment_recording",
    "train_model",
    "write_complete_example_csv",
]
