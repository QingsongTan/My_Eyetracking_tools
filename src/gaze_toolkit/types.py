from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from gaze_toolkit.errors import DataValidationError

REQUIRED_COLUMNS = ("timestamp_ms", "x", "y")
OPTIONAL_COLUMNS = ("pupil", "valid", "marker", "event_label", "label", "trial")


@dataclass(slots=True)
class EyeEvent:
    """Structured eye-movement event."""

    kind: str
    start_time_ms: float
    end_time_ms: float
    start_index: int
    end_index: int
    amplitude: float = 0.0
    peak_velocity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration derived from event boundaries."""
        return float(self.end_time_ms - self.start_time_ms)


@dataclass
class GazeRecording:
    """Canonical eye-tracking recording container."""

    samples: pd.DataFrame
    sampling_rate_hz: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_format: str = "csv"
    events: list[EyeEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.samples = self.samples.copy()
        self.validate()

    def validate(self) -> None:
        """Validate and normalize required columns."""
        missing = [column for column in REQUIRED_COLUMNS if column not in self.samples.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

        normalized = self.samples.sort_values("timestamp_ms").reset_index(drop=True)

        for column in REQUIRED_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        if normalized["timestamp_ms"].isna().any():
            raise DataValidationError("All samples must contain a valid timestamp.")

        coordinate_mask = normalized[["x", "y"]].notna().all(axis=1)
        if "valid" not in normalized.columns:
            normalized["valid"] = coordinate_mask
        else:
            normalized["valid"] = _coerce_valid_column(normalized["valid"]) & coordinate_mask

        valid_subset = normalized.loc[normalized["valid"], ["x", "y"]]
        if valid_subset.isna().any().any():
            raise DataValidationError("Valid samples must contain numeric x/y coordinates.")

        if "pupil" not in normalized.columns:
            normalized["pupil"] = np.nan
        else:
            normalized["pupil"] = pd.to_numeric(normalized["pupil"], errors="coerce")

        for column in ("marker", "event_label", "label", "trial"):
            if column not in normalized.columns:
                normalized[column] = pd.NA

        self.samples = normalized

    @property
    def duration_ms(self) -> float:
        """Total recording duration in milliseconds."""
        if self.samples.empty:
            return 0.0
        return float(self.samples["timestamp_ms"].iloc[-1] - self.samples["timestamp_ms"].iloc[0])

    def copy(self) -> GazeRecording:
        """Deep-ish copy preserving metadata and events."""
        return GazeRecording(
            samples=self.samples.copy(),
            sampling_rate_hz=self.sampling_rate_hz,
            metadata=dict(self.metadata),
            source_format=self.source_format,
            events=list(self.events),
        )

    def with_events(self, events: list[EyeEvent]) -> GazeRecording:
        """Return a new recording with attached events."""
        clone = self.copy()
        clone.events = events
        return clone


def _coerce_valid_column(values: pd.Series) -> pd.Series:
    """Normalize common validity encodings to boolean flags."""
    if is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    numeric = pd.to_numeric(values, errors="coerce")
    normalized = pd.Series(False, index=values.index, dtype="bool")

    numeric_mask = numeric.notna()
    if numeric_mask.any():
        normalized.loc[numeric_mask] = numeric.loc[numeric_mask].astype(float) != 0.0

    text = values.astype("string").str.strip().str.lower()
    truthy_tokens = {"true", "t", "yes", "y", "valid"}
    falsy_tokens = {"false", "f", "no", "n", "invalid", "", "nan", "none", "null", "<na>"}
    text_mask = ~numeric_mask & text.notna()
    if text_mask.any():
        normalized.loc[text_mask] = text.loc[text_mask].isin(truthy_tokens)
        normalized.loc[text_mask & text.isin(falsy_tokens)] = False

    return normalized.fillna(False).astype(bool)
