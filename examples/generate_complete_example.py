from __future__ import annotations

from pathlib import Path

from gaze_toolkit.datasets import write_complete_example_csv


def main() -> None:
    output = Path(__file__).resolve().parent / "data" / "complete_eye_tracking_example.csv"
    path = write_complete_example_csv(output)
    print(f"Saved complete example dataset to {path}")


if __name__ == "__main__":
    main()
