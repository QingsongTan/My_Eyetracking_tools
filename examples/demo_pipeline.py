from __future__ import annotations

from pprint import pprint

from gaze_toolkit.datasets import simulate_gaze_recording, simulate_intent_dataset
from gaze_toolkit.events import attach_events
from gaze_toolkit.features import extract_features
from gaze_toolkit.modeling import train_model
from gaze_toolkit.preprocess import preprocess


def main() -> None:
    recording = simulate_gaze_recording(style="careful", seed=42)
    processed = preprocess(recording)
    enriched = attach_events(processed)
    feature_map = extract_features(enriched)

    dataset = simulate_intent_dataset(num_sessions=24, random_state=42)
    result = train_model(dataset, target="intent_label", model_name="random_forest")

    print("Single recording features:")
    pprint(dict(list(feature_map.items())[:10]))
    print("\nDemo model metrics:")
    pprint(result.metrics)


if __name__ == "__main__":
    main()

