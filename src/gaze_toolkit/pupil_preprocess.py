from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import importlib
import warnings

import numpy as np
import pandas as pd

from gaze_toolkit.types import EyeEvent, GazeRecording


@dataclass
class PupilProcessingResult:
    """Standardized pupil-only preprocessing output."""

    cleaned_pupil: pd.Series
    baseline_corrected_pupil: pd.Series
    blink_mask: pd.Series
    metadata: dict[str, Any]


def preprocess_pupil_signal(
    recording: GazeRecording,
    *,
    baseline_mode: str = "trial_start",
    baseline_window_ms: float = 500.0,
    zscore_within_recording: bool = False,
    smoothing_window: int = 5,
    backend: str = "auto",
) -> PupilProcessingResult:
    """Clean a pupil trace and derive a baseline-corrected version."""
    frame = recording.samples.copy()
    pupil = pd.to_numeric(frame.get("pupil"), errors="coerce")
    timestamps = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    valid = frame.get("valid", pd.Series(True, index=frame.index)).fillna(False).astype(bool)

    blink_mask = (~valid) | pupil.isna() | (pupil <= 0.0)
    blink_mask |= _build_event_blink_mask(recording.events, timestamps)

    masked_pupil = pupil.mask(blink_mask)
    cleaned = masked_pupil.interpolate(method="linear", limit_direction="both").ffill().bfill()
    cleaned = cleaned.fillna(0.0)

    if smoothing_window > 1:
        cleaned = cleaned.rolling(int(smoothing_window), min_periods=1, center=True).mean()

    baseline_value = _estimate_baseline(
        cleaned=cleaned,
        timestamps=timestamps,
        blink_mask=blink_mask,
        baseline_mode=baseline_mode,
        baseline_window_ms=baseline_window_ms,
    )
    baseline_corrected = cleaned - baseline_value

    if zscore_within_recording:
        scale = float(baseline_corrected.std(ddof=0))
        if scale > 1e-6:
            baseline_corrected = (baseline_corrected - float(baseline_corrected.mean())) / scale

    metadata = {
        "available": bool(pupil.notna().any()),
        "backend_requested": backend,
        "backend_used": _resolve_backend(backend),
        "pypillometry_importable": _pypillometry_importable(),
        "baseline_mode": baseline_mode,
        "baseline_window_ms": float(baseline_window_ms),
        "baseline_value": float(baseline_value),
        "blink_sample_ratio": float(blink_mask.mean()) if len(blink_mask) else 0.0,
        "blink_event_count": float(sum(1 for event in recording.events if event.kind == "blink")),
        "interpolation_ratio": float(masked_pupil.isna().mean()) if len(masked_pupil) else 0.0,
    }

    return PupilProcessingResult(
        cleaned_pupil=cleaned.astype(float),
        baseline_corrected_pupil=baseline_corrected.astype(float),
        blink_mask=blink_mask.astype(bool),
        metadata=metadata,
    )


def extract_pupil_load_features(
    recording: GazeRecording,
    pupil_result: PupilProcessingResult | None = None,
    *,
    window_ms: float = 1000.0,
) -> dict[str, float]:
    """Compute workload-oriented pupil features from a cleaned trace."""
    result = pupil_result or preprocess_pupil_signal(recording)
    cleaned = result.cleaned_pupil.astype(float)
    baseline_corrected = result.baseline_corrected_pupil.astype(float)
    timestamps = pd.to_numeric(recording.samples["timestamp_ms"], errors="coerce")

    if cleaned.empty or baseline_corrected.empty:
        return _empty_pupil_features()

    tonic = cleaned.rolling(_window_samples(recording, window_ms), min_periods=1).mean()
    phasic = baseline_corrected - baseline_corrected.rolling(
        _window_samples(recording, window_ms),
        min_periods=1,
    ).mean()
    phasic_positive = phasic.clip(lower=0.0)

    positive_bc = baseline_corrected[baseline_corrected > 0.0]
    peak_positive = float(positive_bc.max()) if not positive_bc.empty else 0.0
    peak_threshold = peak_positive * 0.5
    latency_ms = 0.0
    if peak_threshold > 0.0:
        onset_candidates = timestamps[(baseline_corrected >= peak_threshold).fillna(False)]
        if not onset_candidates.empty:
            latency_ms = float(onset_candidates.iloc[0] - timestamps.iloc[0])

    return {
        "pupil_bc_mean": float(baseline_corrected.mean()),
        "pupil_bc_std": float(baseline_corrected.std(ddof=0)),
        "pupil_bc_peak": peak_positive,
        "pupil_bc_q75": float(baseline_corrected.quantile(0.75)),
        "pupil_tonic_level": float(tonic.mean()),
        "pupil_phasic_mean": float(phasic_positive.mean()),
        "pupil_phasic_peak": float(phasic_positive.max()),
        "pupil_dilation_latency_ms": latency_ms,
        "pupil_blink_ratio": float(result.metadata.get("blink_sample_ratio", 0.0)),
        "pupil_interpolation_ratio": float(result.metadata.get("interpolation_ratio", 0.0)),
    }


def _estimate_baseline(
    *,
    cleaned: pd.Series,
    timestamps: pd.Series,
    blink_mask: pd.Series,
    baseline_mode: str,
    baseline_window_ms: float,
) -> float:
    valid_cleaned = cleaned[~blink_mask]
    if valid_cleaned.empty:
        return 0.0

    if baseline_mode == "recording_median":
        return float(valid_cleaned.median())

    window_end = float(timestamps.iloc[0] + baseline_window_ms)
    window_mask = (timestamps <= window_end) & (~blink_mask)
    window = cleaned[window_mask]
    if not window.empty:
        return float(window.mean())
    return float(valid_cleaned.iloc[: max(1, min(len(valid_cleaned), 10))].mean())


def _build_event_blink_mask(events: list[EyeEvent], timestamps: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=timestamps.index, dtype=bool)
    for event in events:
        if event.kind != "blink":
            continue
        in_range = (timestamps >= event.start_time_ms) & (timestamps <= event.end_time_ms)
        mask |= in_range.fillna(False)
    return mask


def _window_samples(recording: GazeRecording, window_ms: float) -> int:
    timestamps = pd.to_numeric(recording.samples["timestamp_ms"], errors="coerce")
    if len(timestamps) <= 1:
        return 2
    median_dt_ms = float(timestamps.diff().dropna().median())
    return max(int(round(window_ms / max(median_dt_ms, 1.0))), 2)


def _empty_pupil_features() -> dict[str, float]:
    return {
        "pupil_bc_mean": 0.0,
        "pupil_bc_std": 0.0,
        "pupil_bc_peak": 0.0,
        "pupil_bc_q75": 0.0,
        "pupil_tonic_level": 0.0,
        "pupil_phasic_mean": 0.0,
        "pupil_phasic_peak": 0.0,
        "pupil_dilation_latency_ms": 0.0,
        "pupil_blink_ratio": 0.0,
        "pupil_interpolation_ratio": 0.0,
    }


def _resolve_backend(backend: str) -> str:
    if backend == "pypillometry" and _pypillometry_importable():
        return "pypillometry"
    if backend == "auto" and _pypillometry_importable():
        return "pypillometry"
    return "internal_heuristic"


@lru_cache(maxsize=1)
def _pypillometry_importable() -> bool:
    if importlib.util.find_spec("pypillometry") is None:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module("pypillometry")
    except Exception:
        return False
    return True
