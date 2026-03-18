from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MultiModalData:
    """Container for time-aligned multimodal signals."""

    modalities: dict[str, pd.DataFrame] = field(default_factory=dict)
    timestamp_column: str = "timestamp_ms"
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_modality(self, name: str, frame: pd.DataFrame) -> None:
        """Register a modality frame with a shared timestamp column."""
        if self.timestamp_column not in frame.columns:
            raise ValueError(f"{name} is missing timestamp column `{self.timestamp_column}`.")
        self.modalities[name] = frame.sort_values(self.timestamp_column).reset_index(drop=True).copy()

    def align(self, reference: str | None = None, tolerance_ms: float = 50.0) -> pd.DataFrame:
        """Align modalities onto a reference timeline using nearest-neighbor joins."""
        if not self.modalities:
            return pd.DataFrame()

        names = list(self.modalities)
        reference_name = reference or names[0]
        if reference_name not in self.modalities:
            raise KeyError(f"Unknown reference modality: {reference_name}")

        aligned = self._prefixed_frame(reference_name)
        for name, frame in self.modalities.items():
            if name == reference_name:
                continue
            prefixed = self._prefixed_frame(name)
            aligned = pd.merge_asof(
                aligned.sort_values(self.timestamp_column),
                prefixed.sort_values(self.timestamp_column),
                on=self.timestamp_column,
                direction="nearest",
                tolerance=tolerance_ms,
            )

        return aligned

    def early_fusion(self, reference: str | None = None, tolerance_ms: float = 50.0) -> pd.DataFrame:
        """Return an aligned and imputed multimodal feature table."""
        aligned = self.align(reference=reference, tolerance_ms=tolerance_ms)
        if aligned.empty:
            return aligned
        return aligned.ffill().bfill()

    def feature_level_concat(self, features_by_modality: dict[str, pd.Series | dict[str, float]]) -> pd.Series:
        """Concatenate per-modality feature vectors into a single series."""
        fused: dict[str, float] = {}
        for modality, features in features_by_modality.items():
            if isinstance(features, pd.Series):
                items = features.to_dict()
            else:
                items = dict(features)
            fused.update({f"{modality}_{key}": float(value) for key, value in items.items()})
        return pd.Series(fused)

    def _prefixed_frame(self, name: str) -> pd.DataFrame:
        frame = self.modalities[name].copy()
        rename_map = {
            column: f"{name}_{column}"
            for column in frame.columns
            if column != self.timestamp_column
        }
        return frame.rename(columns=rename_map)


def late_fusion(
    predictions: dict[str, np.ndarray | list[float] | pd.Series],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Combine probabilistic outputs from multiple modalities."""
    if not predictions:
        return np.array([])

    names = list(predictions)
    normalized_weights = weights or {name: 1.0 for name in names}
    arrays = []
    weight_values = []
    for name in names:
        arrays.append(np.asarray(predictions[name], dtype=float))
        weight_values.append(float(normalized_weights.get(name, 1.0)))

    stacked = np.vstack(arrays)
    return np.average(stacked, axis=0, weights=np.asarray(weight_values, dtype=float))

