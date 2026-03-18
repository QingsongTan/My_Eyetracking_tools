from __future__ import annotations

from gaze_toolkit.datasets import simulate_gaze_recording, simulate_heart_rate_signal
from gaze_toolkit.multimodal import MultiModalData


def main() -> None:
    recording = simulate_gaze_recording(style="skim", seed=9)
    heart_rate = simulate_heart_rate_signal(recording, seed=9)

    multimodal = MultiModalData()
    multimodal.add_modality("gaze", recording.samples[["timestamp_ms", "x", "y", "pupil"]])
    multimodal.add_modality("heart", heart_rate)

    fused = multimodal.early_fusion(reference="gaze")
    print(fused.head())


if __name__ == "__main__":
    main()

