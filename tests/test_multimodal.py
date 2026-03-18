from __future__ import annotations

import numpy as np

from gaze_toolkit.datasets import simulate_gaze_recording, simulate_heart_rate_signal
from gaze_toolkit.multimodal import MultiModalData, late_fusion


def test_multimodal_alignment_and_fusion() -> None:
    recording = simulate_gaze_recording(seed=11)
    heart_rate = simulate_heart_rate_signal(recording, seed=11)

    container = MultiModalData()
    container.add_modality("gaze", recording.samples[["timestamp_ms", "x", "y", "pupil"]])
    container.add_modality("hr", heart_rate)

    fused = container.early_fusion(reference="gaze")

    assert "hr_heart_rate_bpm" in fused.columns
    assert len(fused) == len(recording.samples)

    combined = late_fusion(
        {
            "gaze": np.array([0.2, 0.8]),
            "hr": np.array([0.4, 0.6]),
        }
    )
    assert combined.shape == (2,)

