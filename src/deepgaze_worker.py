from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaze-toolkit-deepgaze-worker")
    parser.add_argument("--request", help="Path to request JSON.")
    parser.add_argument("--response", help="Path to response JSON.")
    parser.add_argument("--self-check", action="store_true", help="Only verify the runtime and print JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_check:
        return _run_self_check()

    if not args.request or not args.response:
        parser.error("--request and --response are required unless --self-check is used.")

    request_path = Path(args.request)
    response_path = Path(args.response)
    payload = json.loads(request_path.read_text(encoding="utf-8"))

    try:
        response = _run_inference(payload, response_path=response_path)
    except Exception as exc:  # pragma: no cover - exercised in manual runtime checks
        error_payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        response_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(error_payload, ensure_ascii=False))
        return 1

    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _run_self_check() -> int:
    try:
        _prepare_torch_runtime_environment()
        import cv2
        import deepgaze_pytorch
        import pysaliency
        import torch

        payload = {
            "ok": True,
            "torch_version": torch.__version__,
            "deepgaze_module": str(Path(deepgaze_pytorch.__file__).resolve()),
            "pysaliency_module": str(Path(pysaliency.__file__).resolve()),
            "cv2_version": cv2.__version__,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - environment dependent
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 1


def _run_inference(payload: dict[str, Any], *, response_path: Path) -> dict[str, Any]:
    _prepare_torch_runtime_environment()

    import cv2
    import deepgaze_pytorch
    import pysaliency
    import torch
    from pysaliency.metrics import NSS, SIM, convert_saliency_map_to_density, image_based_kl_divergence
    from scipy.ndimage import zoom
    from scipy.special import logsumexp

    image_path = Path(payload["image_path"])
    image_bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to decode image at {image_path}.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    conditioning_x = np.asarray(payload.get("conditioning_fixations_x", []), dtype=np.float32)
    conditioning_y = np.asarray(payload.get("conditioning_fixations_y", []), dtype=np.float32)
    evaluation_x = np.asarray(payload.get("evaluation_fixations_x", []), dtype=np.float32)
    evaluation_y = np.asarray(payload.get("evaluation_fixations_y", []), dtype=np.float32)

    started = perf_counter()
    centerbias_path = _ensure_centerbias_template()
    centerbias_template = np.load(centerbias_path)
    centerbias = zoom(
        centerbias_template,
        (height / centerbias_template.shape[0], width / centerbias_template.shape[1]),
        order=0,
        mode="nearest",
    )
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor(image_rgb.transpose(2, 0, 1)[None], dtype=torch.float32)
    centerbias_tensor = torch.tensor(centerbias[None], dtype=torch.float32)

    if conditioning_x.size > 0 and conditioning_y.size > 0:
        model = deepgaze_pytorch.DeepGazeIII(pretrained=True)
        history_x, history_y = _prepare_scanpath_history(
            conditioning_x=conditioning_x,
            conditioning_y=conditioning_y,
            width=width,
            height=height,
            included_fixations=model.included_fixations,
        )
        x_hist_tensor = torch.tensor(history_x[None], dtype=torch.float32)
        y_hist_tensor = torch.tensor(history_y[None], dtype=torch.float32)
        log_density = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
        model_label = "DeepGazeIII"
    else:
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
        log_density = model(image_tensor, centerbias_tensor)
        model_label = "DeepGazeIIE"

    log_density_map = log_density.detach().cpu().numpy()[0, 0]
    saliency_map = np.exp(log_density_map - logsumexp(log_density_map)).astype(np.float32)
    inference_ms = (perf_counter() - started) * 1000.0

    metadata: dict[str, Any] = {
        "algorithm": "deepgaze",
        "deepgaze_model": model_label,
        "recording_fixation_count": int(payload.get("recording_fixation_count", 0)),
        "conditioning_fixation_count": int(conditioning_x.size),
        "evaluation_fixation_count": int(evaluation_x.size),
        "inference_ms": float(inference_ms),
        "peak_saliency": float(saliency_map.max(initial=0.0)),
        "mean_saliency": float(saliency_map.mean()),
        "hotspot_ratio": float((saliency_map >= np.percentile(saliency_map, 90.0)).mean()),
        "attention_center_x": float((np.indices(saliency_map.shape)[1] * saliency_map).sum() / max(saliency_map.sum(), 1e-8)),
        "attention_center_y": float((np.indices(saliency_map.shape)[0] * saliency_map).sum() / max(saliency_map.sum(), 1e-8)),
    }

    stimuli = pysaliency.Stimuli([image_rgb])
    metadata["pysaliency_stimulus_count"] = int(len(stimuli))
    metadata["pysaliency_fixation_count"] = int(evaluation_x.size)

    if evaluation_x.size > 0 and evaluation_y.size > 0:
        fixation_trains = pysaliency.FixationTrains.from_fixation_trains(
            [evaluation_x.tolist()],
            [evaluation_y.tolist()],
            [np.arange(len(evaluation_x), dtype=float).tolist()],
            [0],
            [0],
        )
        metadata["pysaliency_fixation_count"] = int(len(fixation_trains.x))
        evaluation_x_int = np.clip(np.round(evaluation_x).astype(int), 0, width - 1)
        evaluation_y_int = np.clip(np.round(evaluation_y).astype(int), 0, height - 1)
        nss_values = NSS(saliency_map, evaluation_x_int, evaluation_y_int)
        empirical_density = _build_empirical_fixation_density(
            xs=evaluation_x_int,
            ys=evaluation_y_int,
            width=width,
            height=height,
            cv2=cv2,
        )
        predicted_density = convert_saliency_map_to_density(saliency_map.copy(), minimum_value=1e-12)
        metadata.update(
            {
                "nss_mean": float(np.mean(nss_values)),
                "nss_std": float(np.std(nss_values)),
                "sim": float(SIM(predicted_density, empirical_density)),
                "kl_divergence": float(image_based_kl_divergence(predicted_density, empirical_density)),
            }
        )

    saliency_map_path = response_path.with_suffix(".npy")
    np.save(saliency_map_path, saliency_map)
    return {
        "ok": True,
        "label": f"{model_label} Cognitive Saliency",
        "metadata": metadata,
        "saliency_map_path": str(saliency_map_path),
    }


def _prepare_torch_runtime_environment() -> None:
    python_path = Path(sys.executable).resolve()
    runtime_root = python_path.parents[1]
    torch_lib_dir = runtime_root / "Lib" / "site-packages" / "torch" / "lib"

    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    sanitized_entries: list[str] = []
    for entry in path_entries:
        entry_path = Path(entry)
        if torch_lib_dir.exists() and entry_path.resolve() == torch_lib_dir.resolve():
            continue
        if (entry_path / "libiomp5md.dll").exists() and "torch" not in entry.lower():
            continue
        sanitized_entries.append(entry)

    if torch_lib_dir.exists():
        sanitized_entries.insert(0, str(torch_lib_dir))
    os.environ["PATH"] = os.pathsep.join(sanitized_entries)


def _ensure_centerbias_template() -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "gaze-toolkit-deepgaze-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "centerbias_mit1003.npy"
    if not target.exists():
        url = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
        target.write_bytes(urllib.request.urlopen(url, timeout=90).read())
    return target


def _prepare_scanpath_history(
    *,
    conditioning_x: np.ndarray,
    conditioning_y: np.ndarray,
    width: int,
    height: int,
    included_fixations: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    required_length = len(included_fixations)
    center_x = width / 2.0
    center_y = height / 2.0
    padded_x = [center_x] * max(required_length - len(conditioning_x), 0) + conditioning_x.tolist()
    padded_y = [center_y] * max(required_length - len(conditioning_y), 0) + conditioning_y.tolist()
    history_x = np.asarray(padded_x[-required_length:], dtype=np.float32)
    history_y = np.asarray(padded_y[-required_length:], dtype=np.float32)
    return history_x[included_fixations], history_y[included_fixations]


def _build_empirical_fixation_density(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    cv2: Any,
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    if xs.size == 0:
        density.fill(1.0 / density.size)
        return density

    density[ys, xs] += 1.0
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=28.0, sigmaY=28.0)
    density_sum = float(density.sum())
    if density_sum <= 1e-12:
        density.fill(1.0 / density.size)
    else:
        density /= density_sum
    return density


if __name__ == "__main__":
    raise SystemExit(main())
