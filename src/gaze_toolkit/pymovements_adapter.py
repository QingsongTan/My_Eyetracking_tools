from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from gaze_toolkit.types import GazeRecording

try:
    import polars as pl
    import pymovements as pm

    HAS_PYMOVEMENTS = True
except Exception:  # pragma: no cover - import failure path depends on environment
    pl = None
    pm = None
    HAS_PYMOVEMENTS = False


def from_pymovements(
    gaze: Any,
    *,
    sampling_rate_hz: float | None = None,
    metadata: dict[str, Any] | None = None,
    source_format: str = "pymovements",
) -> GazeRecording:
    """Convert a pymovements gaze object or compatible frame into ``GazeRecording``."""
    from gaze_toolkit.io import from_frame

    frame = _coerce_samples_frame(gaze)
    normalized = _flatten_pymovements_frame(frame)
    merged_metadata = dict(metadata or {})
    if HAS_PYMOVEMENTS and isinstance(gaze, pm.Gaze):
        merged_metadata.setdefault("pymovements_metadata", dict(getattr(gaze, "metadata", {}) or {}))
    return from_frame(
        normalized,
        sampling_rate_hz=sampling_rate_hz,
        metadata=merged_metadata,
        source_format=source_format,
    )


def load_with_pymovements(
    path: str | Path,
    *,
    sampling_rate_hz: float | None = None,
    metadata: dict[str, Any] | None = None,
    separator: str | None = None,
) -> GazeRecording:
    """Load a tabular gaze file through pymovements, then convert to ``GazeRecording``."""
    if not HAS_PYMOVEMENTS:
        raise ImportError("pymovements is not installed. Install the optional dependency first.")

    source = Path(path)
    read_kwargs: dict[str, Any] = {}
    if separator is not None:
        read_kwargs["separator"] = separator
    samples = pl.read_csv(source, **read_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gaze = pm.Gaze(samples=samples, auto_column_detect=True)
    return from_pymovements(
        gaze,
        sampling_rate_hz=sampling_rate_hz,
        metadata=metadata,
        source_format="pymovements",
    )


def _coerce_samples_frame(gaze: Any) -> pd.DataFrame:
    if HAS_PYMOVEMENTS and isinstance(gaze, pm.Gaze):
        return gaze.samples.to_pandas()
    if HAS_PYMOVEMENTS and pl is not None and isinstance(gaze, pl.DataFrame):
        return gaze.to_pandas()
    if isinstance(gaze, pd.DataFrame):
        return gaze.copy()
    if hasattr(gaze, "to_pandas"):
        return gaze.to_pandas()
    raise TypeError("Expected a pymovements.Gaze, polars DataFrame, or pandas DataFrame.")


def _flatten_pymovements_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "time" in normalized.columns and "timestamp_ms" not in normalized.columns:
        normalized["timestamp_ms"] = normalized["time"]

    if "pixel" in normalized.columns:
        pixel_values = normalized["pixel"].apply(_coerce_pixel_pair)
        normalized["x"] = pixel_values.apply(lambda value: value[0])
        normalized["y"] = pixel_values.apply(lambda value: value[1])

    if "message" in normalized.columns and "marker" not in normalized.columns:
        normalized["marker"] = normalized["message"]

    if "name" in normalized.columns and "trial" not in normalized.columns:
        normalized["trial"] = normalized["name"]

    return normalized


def _coerce_pixel_pair(value: Any) -> tuple[float, float]:
    if value is None:
        return (float("nan"), float("nan"))
    if isinstance(value, dict):
        x = value.get("x", value.get("pixel_x", float("nan")))
        y = value.get("y", value.get("pixel_y", float("nan")))
        return float(x), float(y)
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return (float("nan"), float("nan"))
