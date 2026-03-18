from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gaze_toolkit.datasets import simulate_gaze_recording, simulate_intent_dataset
from gaze_toolkit.modeling import train_model
from gaze_toolkit.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""
    parser = argparse.ArgumentParser(prog="gaze-toolkit", description="Eye-tracking analytics toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Generate a synthetic gaze recording.")
    simulate.add_argument("--output", required=True, help="Destination CSV path.")
    simulate.add_argument("--style", default="careful", choices=["careful", "skim"])
    simulate.add_argument("--duration-ms", type=int, default=5000)
    simulate.add_argument("--sampling-rate", type=int, default=120)
    simulate.add_argument("--seed", type=int, default=42)

    features = subparsers.add_parser("features", help="Extract features from a gaze file.")
    features.add_argument("--input", required=True, help="Input gaze file.")
    features.add_argument("--config", help="Optional YAML config.")
    features.add_argument("--sampling-rate", type=float, help="Sampling rate when timestamps are absent.")
    features.add_argument("--output", help="Optional JSON output path.")

    train_demo = subparsers.add_parser("train-demo", help="Train a demo intent classifier on synthetic data.")
    train_demo.add_argument("--sessions", type=int, default=24)
    train_demo.add_argument("--model-name", default="random_forest")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        recording = simulate_gaze_recording(
            duration_ms=args.duration_ms,
            sampling_rate_hz=args.sampling_rate,
            style=args.style,
            seed=args.seed,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        recording.samples.to_csv(output, index=False)
        print(f"Saved synthetic recording to {output}")
        return 0

    if args.command == "features":
        overrides = {"io": {}}
        if args.sampling_rate:
            overrides["io"]["sampling_rate_hz"] = args.sampling_rate
        feature_map = run_pipeline(args.input, config=args.config, overrides=overrides)
        payload = json.dumps(feature_map, ensure_ascii=False, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    dataset = simulate_intent_dataset(num_sessions=args.sessions)
    result = train_model(
        dataset,
        target="intent_label",
        task="classification",
        model_name=args.model_name,
    )
    payload = {
        "model_name": result.model_name,
        "feature_count": len(result.feature_names),
        "metrics": result.metrics,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
