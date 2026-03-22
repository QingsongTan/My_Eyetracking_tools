from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _build_stimulus() -> np.ndarray:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(image, (90, 120), 45, (255, 255, 255), -1)
    cv2.rectangle(image, (210, 60), (290, 180), (0, 180, 255), -1)
    return image


def _run_worker(payload: dict[str, object], *, repo_root: Path, runtime_python: Path) -> dict[str, object]:
    worker_script = repo_root / "src" / "deepgaze_worker.py"

    with tempfile.TemporaryDirectory(prefix="deepgaze-full-validation-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        image_path = temp_dir / "stimulus.png"
        request_path = temp_dir / "request.json"
        response_path = temp_dir / "response.json"

        cv2.imwrite(str(image_path), _build_stimulus())
        request = dict(payload)
        request["image_path"] = str(image_path)
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

        completed = subprocess.run(
            [str(runtime_python), str(worker_script), "--request", str(request_path), "--response", str(response_path)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=900,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "worker exited without details"
            if response_path.exists():
                try:
                    response_payload = json.loads(response_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    response_payload = None
                if isinstance(response_payload, dict) and response_payload.get("error"):
                    detail = str(response_payload["error"])
            raise RuntimeError(detail)

        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not response.get("ok", True):
            raise RuntimeError(str(response.get("error", "worker returned an unsuccessful payload")))
        return response


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_python = Path(sys.executable).resolve()

    no_history = _run_worker(
        {
            "conditioning_fixations_x": [],
            "conditioning_fixations_y": [],
            "evaluation_fixations_x": [],
            "evaluation_fixations_y": [],
            "recording_fixation_count": 0,
        },
        repo_root=repo_root,
        runtime_python=runtime_python,
    )
    if no_history.get("metadata", {}).get("deepgaze_model") != "DeepGazeIIE":
        raise RuntimeError(
            f"Expected DeepGazeIIE without fixation history, got {no_history.get('metadata', {}).get('deepgaze_model')!r}"
        )

    with_history = _run_worker(
        {
            "conditioning_fixations_x": [90.0, 249.0],
            "conditioning_fixations_y": [120.0, 122.0],
            "evaluation_fixations_x": [90.0, 249.0],
            "evaluation_fixations_y": [120.0, 122.0],
            "recording_fixation_count": 2,
        },
        repo_root=repo_root,
        runtime_python=runtime_python,
    )
    if with_history.get("metadata", {}).get("deepgaze_model") != "DeepGazeIII":
        raise RuntimeError(
            f"Expected DeepGazeIII with fixation history, got {with_history.get('metadata', {}).get('deepgaze_model')!r}"
        )

    report = {
        "ok": True,
        "runtime_python": str(runtime_python),
        "deepgaze_iie": {
            "label": no_history.get("label"),
            "model": no_history.get("metadata", {}).get("deepgaze_model"),
            "conditioning_fixation_count": no_history.get("metadata", {}).get("conditioning_fixation_count"),
            "peak_saliency": no_history.get("metadata", {}).get("peak_saliency"),
        },
        "deepgaze_iii": {
            "label": with_history.get("label"),
            "model": with_history.get("metadata", {}).get("deepgaze_model"),
            "conditioning_fixation_count": with_history.get("metadata", {}).get("conditioning_fixation_count"),
            "peak_saliency": with_history.get("metadata", {}).get("peak_saliency"),
            "nss_mean": with_history.get("metadata", {}).get("nss_mean"),
            "sim": with_history.get("metadata", {}).get("sim"),
            "kl_divergence": with_history.get("metadata", {}).get("kl_divergence"),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
