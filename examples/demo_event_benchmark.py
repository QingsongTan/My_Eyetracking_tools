from __future__ import annotations

from pathlib import Path

from gaze_toolkit.event_benchmark import build_event_benchmark_markdown, run_public_dataset_benchmark, summarize_event_benchmark


def main() -> None:
    root_dir = Path(".cache") / "pymovements" / "ToyDatasetEyeLink"
    result = run_public_dataset_benchmark(
        dataset_name="ToyDatasetEyeLink",
        root_dir=root_dir,
        recording_index=0,
        download=True,
    )
    insight = summarize_event_benchmark(result)

    print(insight.headline)
    for line in insight.summary_lines:
        print(f"- {line}")

    report_path = Path("event-benchmark-report.md")
    report_path.write_text(build_event_benchmark_markdown(result), encoding="utf-8")
    print(f"\n报告已导出到: {report_path.resolve()}")


if __name__ == "__main__":
    main()
