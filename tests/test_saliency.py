from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import gaze_toolkit.saliency as saliency_module
from gaze_toolkit.analysis import analyze_recording
from gaze_toolkit.datasets import simulate_gaze_recording
from gaze_toolkit.saliency import (
    COGNITIVE_SALIENCY_BACKEND,
    FAST_SALIENCY_BACKEND,
    list_saliency_backends,
    probe_deepgaze_runtime,
    predict_image_attention,
)
from gaze_toolkit.visualization import plot_image_saliency_heatmap


def test_predict_image_attention_fast_backend_returns_normalized_map() -> None:
    image = np.full((140, 220, 3), fill_value=18, dtype=np.uint8)
    image[24:102, 30:96] = np.array([245, 48, 48], dtype=np.uint8)
    image[42:94, 132:186] = np.array([250, 250, 250], dtype=np.uint8)

    result = predict_image_attention(image, backend=FAST_SALIENCY_BACKEND)

    assert result.backend == FAST_SALIENCY_BACKEND
    assert result.saliency_map.shape == image.shape[:2]
    assert float(result.saliency_map.min()) >= 0.0
    assert float(result.saliency_map.max()) <= 1.0
    assert result.metadata["peak_saliency"] > 0.0
    assert result.metadata["hotspot_ratio"] > 0.0
    assert result.saliency_map[30:100, 30:96].mean() > result.saliency_map[110:135, 0:25].mean()


def test_plot_image_saliency_heatmap_supports_background_and_opacity(tmp_path) -> None:
    background = np.zeros((48, 72, 3), dtype=float)
    background[..., 0] = 0.30
    background[..., 1] = 0.55
    background_path = tmp_path / "stimulus_saliency.png"
    plt.imsave(background_path, background)

    result = predict_image_attention((background * 255).astype(np.uint8), backend=FAST_SALIENCY_BACKEND)
    figure = plot_image_saliency_heatmap(
        result.saliency_map,
        background_image=background_path,
        screen_size=(960, 540),
        theme_name="light",
        palette="sunset",
        heatmap_opacity=0.44,
    )

    assert "OpenCV Fast Saliency" in figure.layout.title.text
    assert len(figure.data) == 1
    assert figure.data[0].opacity == 0.44
    assert len(figure.layout.images) == 1
    assert list(figure.layout.xaxis.range) == [0.0, 960.0]
    assert list(figure.layout.yaxis.range) == [540.0, 0.0]


def test_list_saliency_backends_marks_deepgaze_available_when_runtime_exists(monkeypatch, tmp_path) -> None:
    runtime_python = tmp_path / "python.exe"
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(saliency_module, "_resolve_deepgaze_python", lambda **_: runtime_python)

    status = list_saliency_backends(project_root=tmp_path)[COGNITIVE_SALIENCY_BACKEND]

    assert status.implemented is True
    assert status.available is True
    assert str(runtime_python) in status.detail


def test_probe_deepgaze_runtime_surfaces_subprocess_failure(monkeypatch, tmp_path) -> None:
    runtime_python = tmp_path / "python.exe"
    runtime_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(saliency_module, "_require_deepgaze_python", lambda **_: runtime_python)

    def fake_run(*args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout=json.dumps({"ok": False, "error": "torch import failed"}),
            stderr="",
        )

    monkeypatch.setattr(saliency_module.subprocess, "run", fake_run)

    ok, payload = probe_deepgaze_runtime(project_root=tmp_path)

    assert ok is False
    assert payload["error"] == "torch import failed"


def test_predict_image_attention_cognitive_backend_uses_worker_bridge(monkeypatch, tmp_path) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[8:24, 12:28] = 255
    recording = analyze_recording(simulate_gaze_recording(seed=14)).enriched_recording
    runtime_python = tmp_path / "python.exe"
    runtime_python.write_text("", encoding="utf-8")
    captured_request: dict[str, object] = {}

    monkeypatch.setattr(saliency_module, "_require_deepgaze_python", lambda **_: runtime_python)

    def fake_worker(*, request_path: Path, response_path: Path, project_root: Path, runtime_python: Path, timeout_sec: float):
        del project_root, runtime_python, timeout_sec
        captured_request.update(json.loads(request_path.read_text(encoding="utf-8")))
        saliency_map = np.full((32, 48), fill_value=0.25, dtype=np.float32)
        saliency_map[8:24, 12:28] = 1.0
        saliency_map_path = response_path.with_suffix(".npy")
        np.save(saliency_map_path, saliency_map)
        return {
            "saliency_map_path": str(saliency_map_path),
            "label": "DeepGazeIII Cognitive Saliency",
            "metadata": {
                "deepgaze_model": "DeepGazeIII",
                "nss_mean": 1.42,
                "conditioning_fixation_count": len(captured_request["conditioning_fixations_x"]),
            },
        }

    monkeypatch.setattr(saliency_module, "_invoke_deepgaze_worker", fake_worker)

    result = predict_image_attention(
        image,
        backend=COGNITIVE_SALIENCY_BACKEND,
        recording=recording,
        project_root=tmp_path,
    )

    assert result.backend == COGNITIVE_SALIENCY_BACKEND
    assert result.saliency_map.shape == (32, 48)
    assert result.metadata["deepgaze_model"] == "DeepGazeIII"
    assert captured_request["recording_fixation_count"] >= 0
    assert isinstance(captured_request["conditioning_fixations_x"], list)


def test_invoke_deepgaze_worker_prefers_response_payload_on_failure(monkeypatch, tmp_path) -> None:
    runtime_python = tmp_path / "python.exe"
    runtime_python.write_text("", encoding="utf-8")
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text("{}", encoding="utf-8")
    response_path.write_text(
        json.dumps({"ok": False, "error": "negative dimensions are not allowed"}),
        encoding="utf-8",
    )

    def fake_run(*args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="",
            stderr="loud stderr that should not win",
        )

    monkeypatch.setattr(saliency_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="negative dimensions are not allowed"):
        saliency_module._invoke_deepgaze_worker(
            request_path=request_path,
            response_path=response_path,
            project_root=tmp_path,
            runtime_python=runtime_python,
            timeout_sec=3.0,
        )


def test_predict_image_attention_cognitive_backend_supports_missing_fixations() -> None:
    ok, payload = probe_deepgaze_runtime()
    if not ok:
        pytest.skip(payload.get("error", "DeepGaze runtime unavailable"))

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[24:92, 18:66] = np.array([255, 255, 255], dtype=np.uint8)
    image[36:84, 98:142] = np.array([64, 180, 255], dtype=np.uint8)

    result = predict_image_attention(
        image,
        backend=COGNITIVE_SALIENCY_BACKEND,
        timeout_sec=900,
    )

    assert result.backend == COGNITIVE_SALIENCY_BACKEND
    assert result.metadata["deepgaze_model"] == "DeepGazeIIE"
    assert result.metadata["evaluation_fixation_count"] == 0
    assert result.metadata["conditioning_fixation_count"] == 0
    assert result.metadata["pysaliency_fixation_count"] == 0
    assert result.saliency_map.shape == image.shape[:2]
