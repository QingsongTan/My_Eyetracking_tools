"""Public package interface for gaze-toolkit."""

from gaze_toolkit.aoi import (
    AOI,
    AOIMetrics,
    assign_fixations_to_aoi,
    compute_aoi_metrics,
    compute_transition_matrix,
    define_aoi,
    define_polygon_aoi,
)
from gaze_toolkit.analysis import (
    ExperimentReport,
    ModalityComparison,
    RecordingAnalysis,
    analyze_recording,
    compare_modalities,
    run_intent_experiment,
)
from gaze_toolkit.batch import (
    batch_analyze,
    batch_analyze_recordings,
    export_html_report,
    export_markdown_report,
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
from gaze_toolkit.quality import QualityReport, assess_quality, format_quality_cards
from gaze_toolkit.saliency import (
    COGNITIVE_SALIENCY_BACKEND,
    FAST_SALIENCY_BACKEND,
    ImageSaliencyResult,
    SaliencyBackendStatus,
    get_saliency_backend_status,
    list_saliency_backends,
    probe_deepgaze_runtime,
    predict_image_attention,
)
from gaze_toolkit.scenarios import (
    AOIRegion,
    ResearchDesign,
    ScenarioTask,
    ScenarioTemplate,
    get_scenario_aois,
    list_scenarios,
    load_scenario,
)
from gaze_toolkit.segmentation import (
    RecordingSegment,
    build_segment_feature_table,
    segment_between_markers,
    segment_by_marker_windows,
    segment_by_time_ranges,
    segment_recording,
)
from gaze_toolkit.statistics import (
    StatTestResult,
    compare_conditions,
    descriptive_table,
    independent_t_test,
    mann_whitney_test,
    paired_t_test,
    repeated_measures_anova,
    wilcoxon_test,
)
from gaze_toolkit.streaming import OnlinePrediction, SlidingWindowPredictor
from gaze_toolkit.tables import compute_quality_grade, fixation_table
from gaze_toolkit.types import EyeEvent, GazeRecording
from gaze_toolkit.visualization import plot_heatmap, plot_scanpath

__all__ = [
    "AOI",
    "AOIMetrics",
    "AOIRegion",
    "EyeEvent",
    "ExperimentReport",
    "GazeRecording",
    "ModelResult",
    "ModalityComparison",
    "MultiModalData",
    "OnlinePrediction",
    "ResearchDesign",
    "RecordingAnalysis",
    "RecordingSegment",
    "ScenarioTask",
    "ScenarioTemplate",
    "SlidingWindowPredictor",
    "StatTestResult",
    "analyze_recording",
    "assign_fixations_to_aoi",
    "attach_events",
    "batch_analyze",
    "batch_analyze_recordings",
    "build_feature_dataset",
    "build_segment_feature_table",
    "build_model",
    "compare_modalities",
    "compute_quality_grade",
    "create_complete_example_recording",
    "compute_aoi_metrics",
    "compute_transition_matrix",
    "compare_conditions",
    "define_aoi",
    "define_polygon_aoi",
    "descriptive_table",
    "detect_events",
    "explain_with_shap",
    "export_html_report",
    "export_markdown_report",
    "extract_features",
    "fixation_table",
    "from_frame",
    "get_scenario_aois",
    "handle_missing_samples",
    "late_fusion",
    "load",
    "load_config",
    "load_scenario",
    "list_saliency_backends",
    "list_scenarios",
    "independent_t_test",
    "mann_whitney_test",
    "permutation_feature_importance",
    "paired_t_test",
    "plot_heatmap",
    "plot_scanpath",
    "preprocess",
    "predict",
    "predict_image_attention",
    "probe_deepgaze_runtime",
    "QualityReport",
    "register_feature",
    "register_loader",
    "register_model",
    "repeated_measures_anova",
    "run_intent_experiment",
    "run_pipeline",
    "run_segmented_pipeline",
    "assess_quality",
    "format_quality_cards",
    "segment_between_markers",
    "segment_by_marker_windows",
    "segment_by_time_ranges",
    "segment_recording",
    "get_saliency_backend_status",
    "train_model",
    "wilcoxon_test",
    "write_complete_example_csv",
    "COGNITIVE_SALIENCY_BACKEND",
    "FAST_SALIENCY_BACKEND",
    "ImageSaliencyResult",
    "SaliencyBackendStatus",
]
