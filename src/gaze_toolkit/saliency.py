from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from gaze_toolkit.types import GazeRecording

FAST_SALIENCY_BACKEND = "opencv_fast"
COGNITIVE_SALIENCY_BACKEND = "deepgaze"


@dataclass(frozen=True)
class SaliencyBackendStatus:
    """Runtime status for one saliency backend."""

    backend: str
    label: str
    implemented: bool
    available: bool
    dependencies: tuple[str, ...]
    detail: str


@dataclass
class ImageSaliencyResult:
    """Normalized attention prior predicted from an image."""

    saliency_map: np.ndarray
    backend: str
    label: str
    width: int
    height: int
    metadata: dict[str, Any] = field(default_factory=dict)


def list_saliency_backends(project_root: str | Path | None = None) -> dict[str, SaliencyBackendStatus]:
    """Describe supported image-attention backends."""
    has_cv2 = _module_available("cv2")
    deepgaze_python = _resolve_deepgaze_python(project_root=project_root)
    deepgaze_available = deepgaze_python is not None
    deepgaze_detail = (
        f"DeepGaze subprocess runtime detected at {deepgaze_python}."
        if deepgaze_available
        else (
            "DeepGaze backend is implemented through a separate Python runtime. "
            "Set GAZE_TOOLKIT_DEEPGAZE_PYTHON or create .deepgaze-py312/Scripts/python.exe."
        )
    )

    return {
        FAST_SALIENCY_BACKEND: SaliencyBackendStatus(
            backend=FAST_SALIENCY_BACKEND,
            label="OpenCV Fast Saliency",
            implemented=True,
            available=has_cv2,
            dependencies=("cv2",),
            detail=(
                "Uses OpenCV contrast, local variation, and edge structure to build a fast bottom-up saliency map."
                if has_cv2
                else "OpenCV (cv2) is missing, so the fast saliency backend is unavailable."
            ),
        ),
        COGNITIVE_SALIENCY_BACKEND: SaliencyBackendStatus(
            backend=COGNITIVE_SALIENCY_BACKEND,
            label="PyTorch + PySaliency + DeepGaze",
            implemented=True,
            available=deepgaze_available,
            dependencies=("torch", "torchvision", "pysaliency", "deepgaze_pytorch"),
            detail=deepgaze_detail,
        ),
    }


def get_saliency_backend_status(
    backend: str,
    *,
    project_root: str | Path | None = None,
) -> SaliencyBackendStatus:
    """Return the status for one backend."""
    registry = list_saliency_backends(project_root=project_root)
    if backend not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown saliency backend '{backend}'. Available backends: {available}")
    return registry[backend]


def probe_deepgaze_runtime(
    *,
    project_root: str | Path | None = None,
    runtime_python: str | Path | None = None,
    timeout_sec: float = 45.0,
) -> tuple[bool, dict[str, Any]]:
    """Run a subprocess self-check for the DeepGaze runtime."""
    project_root_path = _resolve_project_root(project_root)
    python_path = _require_deepgaze_python(project_root=project_root_path, runtime_python=runtime_python)
    worker_script = project_root_path / "src" / "deepgaze_worker.py"
    command = [str(python_path), str(worker_script), "--self-check"]
    env = _prepare_deepgaze_env(project_root=project_root_path, runtime_python=python_path)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
        cwd=str(project_root_path),
    )
    payload_text = completed.stdout.strip() or completed.stderr.strip()
    payload: dict[str, Any]
    try:
        payload = json.loads(payload_text) if payload_text else {}
    except json.JSONDecodeError:
        payload = {"ok": False, "error": payload_text or "DeepGaze runtime check produced no output."}

    if completed.returncode != 0:
        payload.setdefault("ok", False)
        payload.setdefault("error", completed.stderr.strip() or completed.stdout.strip() or "DeepGaze runtime self-check failed.")
        return False, payload

    return bool(payload.get("ok", True)), payload


def predict_image_attention(
    image: str | Path | bytes | bytearray | memoryview | np.ndarray | Any,
    *,
    backend: str = FAST_SALIENCY_BACKEND,
    recording: GazeRecording | None = None,
    max_side: int = 480,
    center_bias_weight: float = 0.10,
    project_root: str | Path | None = None,
    runtime_python: str | Path | None = None,
    timeout_sec: float = 240.0,
) -> ImageSaliencyResult:
    """Predict an image-based attention prior."""
    if backend == FAST_SALIENCY_BACKEND:
        return _predict_fast_saliency(
            image,
            max_side=max_side,
            center_bias_weight=center_bias_weight,
        )
    if backend == COGNITIVE_SALIENCY_BACKEND:
        return _predict_cognitive_saliency(
            image,
            recording=recording,
            project_root=project_root,
            runtime_python=runtime_python,
            timeout_sec=timeout_sec,
        )

    available = ", ".join(sorted(list_saliency_backends(project_root=project_root)))
    raise ValueError(f"Unknown saliency backend '{backend}'. Available backends: {available}")


def _predict_fast_saliency(
    image: str | Path | bytes | bytearray | memoryview | np.ndarray | Any,
    *,
    max_side: int,
    center_bias_weight: float,
) -> ImageSaliencyResult:
    cv2 = _require_cv2()
    bgr = _load_image_bgr(image, cv2=cv2)
    height, width = bgr.shape[:2]

    started = perf_counter()
    working, resize_scale = _resize_for_compute(bgr, cv2=cv2, max_side=max_side)
    saliency = _compute_opencv_saliency(
        working,
        cv2=cv2,
        center_bias_weight=center_bias_weight,
    )
    if saliency.shape != (height, width):
        saliency = cv2.resize(saliency, (width, height), interpolation=cv2.INTER_CUBIC)
    saliency = np.clip(saliency.astype(np.float32), 0.0, 1.0)
    elapsed_ms = (perf_counter() - started) * 1000.0

    return ImageSaliencyResult(
        saliency_map=saliency,
        backend=FAST_SALIENCY_BACKEND,
        label="OpenCV Fast Saliency",
        width=width,
        height=height,
        metadata=_summarize_saliency_map(
            saliency,
            algorithm="opencv_frequency_contrast",
            center_bias_weight=float(np.clip(center_bias_weight, 0.0, 0.4)),
            inference_ms=elapsed_ms,
            resize_scale=resize_scale,
        ),
    )


def _predict_cognitive_saliency(
    image: str | Path | bytes | bytearray | memoryview | np.ndarray | Any,
    *,
    recording: GazeRecording | None,
    project_root: str | Path | None,
    runtime_python: str | Path | None,
    timeout_sec: float,
) -> ImageSaliencyResult:
    project_root_path = _resolve_project_root(project_root)
    python_path = _require_deepgaze_python(project_root=project_root_path, runtime_python=runtime_python)
    fixation_payload = _extract_fixation_payload(recording)

    with tempfile.TemporaryDirectory(prefix="gaze-toolkit-deepgaze-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        image_path = temp_dir / "stimulus.png"
        request_path = temp_dir / "request.json"
        response_path = temp_dir / "response.json"

        height, width = _write_image_png(image, image_path)
        request_payload = {
            "image_path": str(image_path),
            "conditioning_fixations_x": fixation_payload["conditioning_fixations_x"],
            "conditioning_fixations_y": fixation_payload["conditioning_fixations_y"],
            "evaluation_fixations_x": fixation_payload["evaluation_fixations_x"],
            "evaluation_fixations_y": fixation_payload["evaluation_fixations_y"],
            "recording_fixation_count": fixation_payload["fixation_count"],
        }
        request_path.write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        response = _invoke_deepgaze_worker(
            request_path=request_path,
            response_path=response_path,
            project_root=project_root_path,
            runtime_python=python_path,
            timeout_sec=timeout_sec,
        )
        saliency_map = np.load(response["saliency_map_path"])
        metadata = dict(response.get("metadata", {}))
        metadata.setdefault("recording_fixation_count", fixation_payload["fixation_count"])
        metadata.setdefault("conditioning_fixation_count", len(fixation_payload["conditioning_fixations_x"]))
        return ImageSaliencyResult(
            saliency_map=saliency_map.astype(np.float32),
            backend=COGNITIVE_SALIENCY_BACKEND,
            label=response.get("label", "DeepGaze"),
            width=width,
            height=height,
            metadata=metadata,
        )


def _invoke_deepgaze_worker(
    *,
    request_path: Path,
    response_path: Path,
    project_root: Path,
    runtime_python: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    worker_script = project_root / "src" / "deepgaze_worker.py"
    command = [
        str(runtime_python),
        str(worker_script),
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    env = _prepare_deepgaze_env(project_root=project_root, runtime_python=runtime_python)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
        cwd=str(project_root),
    )
    if completed.returncode != 0:
        if response_path.exists():
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                detail = payload.get("error") or payload.get("traceback")
                if detail:
                    raise RuntimeError(f"DeepGaze worker failed: {detail}")
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "DeepGaze worker exited without details."
        raise RuntimeError(f"DeepGaze worker failed: {detail}")

    if not response_path.exists():
        raise RuntimeError("DeepGaze worker finished without writing a response payload.")

    return json.loads(response_path.read_text(encoding="utf-8"))


def _prepare_deepgaze_env(*, project_root: Path, runtime_python: Path) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries = [str(project_root / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    runtime_root = runtime_python.resolve().parents[1]
    torch_lib_dir = runtime_root / "Lib" / "site-packages" / "torch" / "lib"
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    sanitized_entries: list[str] = []
    for entry in path_entries:
        entry_path = Path(entry)
        if entry_path.resolve() == torch_lib_dir.resolve():
            continue
        if (entry_path / "libiomp5md.dll").exists() and "torch" not in entry.lower():
            continue
        sanitized_entries.append(entry)

    if torch_lib_dir.exists():
        sanitized_entries.insert(0, str(torch_lib_dir))
    env["PATH"] = os.pathsep.join(sanitized_entries)
    return env


def _resolve_project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[2]


def _resolve_deepgaze_python(
    *,
    project_root: str | Path | None = None,
    runtime_python: str | Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    project_root_path = _resolve_project_root(project_root)
    if runtime_python is not None:
        candidates.append(Path(runtime_python))

    env_value = os.getenv("GAZE_TOOLKIT_DEEPGAZE_PYTHON")
    if env_value:
        candidates.append(Path(env_value))

    candidates.extend(
        [
            project_root_path / ".deepgaze-py312" / "Scripts" / "python.exe",
            project_root_path / ".venv-deepgaze" / "Scripts" / "python.exe",
            project_root_path / ".venv-deepgaze312" / "Scripts" / "python.exe",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _require_deepgaze_python(
    *,
    project_root: str | Path | None = None,
    runtime_python: str | Path | None = None,
) -> Path:
    python_path = _resolve_deepgaze_python(project_root=project_root, runtime_python=runtime_python)
    if python_path is None:
        raise RuntimeError(
            "DeepGaze runtime was not found. Set GAZE_TOOLKIT_DEEPGAZE_PYTHON or create "
            ".deepgaze-py312/Scripts/python.exe under the project root."
        )
    return python_path


def _extract_fixation_payload(recording: GazeRecording | None) -> dict[str, Any]:
    if recording is None:
        return {
            "conditioning_fixations_x": [],
            "conditioning_fixations_y": [],
            "evaluation_fixations_x": [],
            "evaluation_fixations_y": [],
            "fixation_count": 0,
        }

    fixations: list[tuple[float, float]] = []
    for event in recording.events:
        if event.kind != "fixation":
            continue
        x = event.metadata.get("centroid_x")
        y = event.metadata.get("centroid_y")
        if x is None or y is None:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        fixations.append((float(x), float(y)))

    xs = [point[0] for point in fixations]
    ys = [point[1] for point in fixations]
    return {
        "conditioning_fixations_x": xs,
        "conditioning_fixations_y": ys,
        "evaluation_fixations_x": xs,
        "evaluation_fixations_y": ys,
        "fixation_count": len(fixations),
    }


def _compute_opencv_saliency(
    bgr_image: np.ndarray,
    *,
    cv2: Any,
    center_bias_weight: float,
) -> np.ndarray:
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    blurred_lab = cv2.GaussianBlur(lab, (0, 0), sigmaX=3.0, sigmaY=3.0)
    global_mean = blurred_lab.reshape(-1, 3).mean(axis=0)
    color_contrast = np.linalg.norm(blurred_lab - global_mean, axis=2)

    local_mean = cv2.GaussianBlur(blurred_lab, (0, 0), sigmaX=13.0, sigmaY=13.0)
    local_contrast = np.linalg.norm(blurred_lab - local_mean, axis=2)

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.magnitude(grad_x, grad_y)
    edge_strength = cv2.GaussianBlur(edge_strength, (0, 0), sigmaX=1.2, sigmaY=1.2)

    saliency = (
        0.52 * _robust_normalize(color_contrast)
        + 0.28 * _robust_normalize(local_contrast)
        + 0.20 * _robust_normalize(edge_strength)
    )

    center_bias = _make_center_bias(saliency.shape)
    weight = float(np.clip(center_bias_weight, 0.0, 0.4))
    if weight > 0.0:
        saliency = (1.0 - weight) * saliency + weight * center_bias

    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), sigmaX=4.5, sigmaY=4.5)
    saliency = _robust_normalize(saliency)
    saliency = np.power(saliency, 0.92, dtype=np.float32)
    return np.clip(saliency, 0.0, 1.0)


def _summarize_saliency_map(
    saliency_map: np.ndarray,
    *,
    algorithm: str,
    center_bias_weight: float,
    inference_ms: float,
    resize_scale: float,
) -> dict[str, Any]:
    height, width = saliency_map.shape
    total_mass = float(saliency_map.sum())
    hotspot_ratio = float((saliency_map >= 0.75).mean())
    peak_index = np.unravel_index(int(np.argmax(saliency_map)), saliency_map.shape)
    peak_y, peak_x = int(peak_index[0]), int(peak_index[1])

    if total_mass > 1e-8:
        yy, xx = np.indices(saliency_map.shape, dtype=np.float32)
        center_x = float((xx * saliency_map).sum() / total_mass)
        center_y = float((yy * saliency_map).sum() / total_mass)
    else:
        center_x = width / 2.0
        center_y = height / 2.0

    flat = saliency_map.reshape(-1).astype(np.float64)
    flat_sum = float(flat.sum())
    if flat_sum > 1e-12 and flat.size > 1:
        probabilities = flat / flat_sum
        entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum() / np.log2(flat.size))
    else:
        entropy = 0.0

    return {
        "algorithm": algorithm,
        "center_bias_weight": center_bias_weight,
        "inference_ms": float(inference_ms),
        "resize_scale": float(resize_scale),
        "peak_saliency": float(saliency_map.max(initial=0.0)),
        "mean_saliency": float(saliency_map.mean()),
        "hotspot_ratio": hotspot_ratio,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "attention_center_x": center_x,
        "attention_center_y": center_y,
        "entropy": entropy,
    }


def _load_image_bgr(
    image: str | Path | bytes | bytearray | memoryview | np.ndarray | Any,
    *,
    cv2: Any,
) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return _numpy_image_to_bgr(image, cv2=cv2)

    if isinstance(image, (str, Path)):
        buffer = np.fromfile(str(Path(image)), dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Unable to decode image from path.")
        return decoded

    if isinstance(image, (bytes, bytearray, memoryview)):
        payload = bytes(image)
    elif hasattr(image, "getvalue"):
        payload = bytes(image.getvalue())
        if hasattr(image, "seek"):
            image.seek(0)
    elif hasattr(image, "read"):
        if hasattr(image, "seek"):
            image.seek(0)
        payload = bytes(image.read())
        if hasattr(image, "seek"):
            image.seek(0)
    else:
        raise TypeError("Unsupported image source type for saliency prediction.")

    if not payload:
        raise ValueError("Received an empty image payload.")

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unable to decode image payload.")
    return decoded


def _write_image_png(
    image: str | Path | bytes | bytearray | memoryview | np.ndarray | Any,
    destination: Path,
) -> tuple[int, int]:
    cv2 = _require_cv2()
    bgr = _load_image_bgr(image, cv2=cv2)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Unable to encode stimulus image as PNG.")
    destination.write_bytes(encoded.tobytes())
    height, width = bgr.shape[:2]
    return height, width


def _numpy_image_to_bgr(image: np.ndarray, *, cv2: Any) -> np.ndarray:
    if image.ndim == 2:
        gray = _to_uint8(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3:
        raise ValueError("Expected a 2D or 3D numpy image array.")

    normalized = _to_uint8(image)
    channels = normalized.shape[2]
    if channels == 1:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(normalized, cv2.COLOR_RGBA2BGR)
    raise ValueError("Unsupported channel count for numpy image array.")


def _resize_for_compute(bgr_image: np.ndarray, *, cv2: Any, max_side: int) -> tuple[np.ndarray, float]:
    height, width = bgr_image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return bgr_image, 1.0

    scale = float(max_side / float(longest_side))
    resized = cv2.resize(
        bgr_image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _make_center_bias(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    bias = np.exp(-((x_grid**2) / (2.0 * 0.58**2) + (y_grid**2) / (2.0 * 0.52**2)))
    return _robust_normalize(bias)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    normalized = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if normalized.size == 0:
        return normalized

    upper = float(np.percentile(normalized, 98.0))
    lower = float(np.percentile(normalized, 2.0))
    if upper <= lower + 1e-8:
        peak = float(normalized.max(initial=0.0))
        return normalized / peak if peak > 0.0 else np.zeros_like(normalized, dtype=np.float32)

    clipped = np.clip(normalized, lower, upper)
    scaled = (clipped - lower) / (upper - lower)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.copy()
    float_array = array.astype(np.float32)
    peak = float(np.nanmax(float_array)) if float_array.size else 0.0
    if peak <= 1.0:
        float_array = float_array * 255.0
    return np.clip(float_array, 0.0, 255.0).astype(np.uint8)


def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _require_cv2() -> Any:
    if not _module_available("cv2"):
        raise ImportError("OpenCV (cv2) is required for the fast saliency backend.")

    import cv2

    return cv2
