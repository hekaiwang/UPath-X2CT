#!/usr/bin/env python3
"""Fit train-only intensity calibration for the differentiable cone projector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from upathx2ct.sota.cone_projection import (
    ConeProjectorConfig,
    DifferentiableConeProjector,
)
from upathx2ct.sota.data import UPathLIDC128Dataset
from upathx2ct.sota.geometry import ConeBeamGeometry
from upathx2ct.sota.projective_lifting import PhysicalGrid
from upathx2ct.sota.protocol import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-size", type=int, default=128)
    parser.add_argument("--ray-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def fit_view(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
    x = np.concatenate([value.reshape(-1) for value in predictions]).astype(
        np.float64
    )
    y = np.concatenate([value.reshape(-1) for value in targets]).astype(np.float64)
    design = np.stack((x, np.ones_like(x)), axis=1)
    gain, bias = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = gain * x + bias
    residual = fitted - y
    correlation = float(np.corrcoef(x, y)[0, 1])
    nrmse = float(np.sqrt(np.mean(residual**2)) / max(np.std(y), 1e-8))
    target_log = np.log1p(np.maximum(y, 0.0))
    return {
        "gain": float(gain),
        "bias": float(bias),
        "pearson": correlation,
        "normalized_rmse": nrmse,
        "target_log1p_mean": float(target_log.mean()),
        "target_log1p_std": float(target_log.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "samples": int(len(y)),
    }


def main() -> None:
    args = parse_args()
    if args.max_cases <= 0:
        raise ValueError("Calibration requires a positive train-case sample")
    torch.manual_seed(args.seed)
    dataset = UPathLIDC128Dataset(
        args.dataset_root,
        "train",
        include_raw_drr=True,
    )
    generator = np.random.default_rng(args.seed)
    selected_indices = np.sort(
        generator.choice(
            len(dataset),
            size=min(args.max_cases, len(dataset)),
            replace=False,
        )
    )
    full_geometry = ConeBeamGeometry(
        source_detector_distance_mm=1800.0,
        source_object_distance_mm=1500.0,
        detector_shape=(512, 512),
        detector_pixel_size_mm=(0.9, 0.9),
        principal_point_px=(256.0, 256.0),
    )
    projector = DifferentiableConeProjector(
        PhysicalGrid(
            (128, 128, 128),
            (-190.5, -190.5, -190.5),
            (3.0, 3.0, 3.0),
        ),
        full_geometry,
        ConeProjectorConfig(
            detector_shape=(args.detector_size, args.detector_size),
            ray_samples=args.ray_samples,
        ),
    ).to(args.device)
    scale = 512 // args.detector_size
    if scale * args.detector_size != 512:
        raise ValueError("Detector size must evenly divide 512")
    predicted: list[list[np.ndarray]] = [[], []]
    target: list[list[np.ndarray]] = [[], []]
    with torch.inference_mode():
        for completed, index in enumerate(selected_indices, start=1):
            item = dataset[int(index)]
            ct = item["ct"].unsqueeze(0).to(args.device)
            projections = projector(ct)
            raw = (item["xray_ap_raw"], item["xray_lat_raw"])
            for view in range(2):
                downsampled = F.avg_pool2d(
                    raw[view].unsqueeze(0),
                    kernel_size=scale,
                    stride=scale,
                )
                predicted[view].append(
                    projections[view][0, 0].float().cpu().numpy()
                )
                target[view].append(downsampled[0, 0].numpy())
            print(
                f"[{completed}/{len(selected_indices)}] {item['case_id']}",
                flush=True,
            )
    views = {
        name: fit_view(predicted[index], target[index])
        for index, name in enumerate(("frontal_ap", "left_lateral"))
    }
    checks = {
        "train_only": True,
        "both_views_pearson_at_least_0_98": all(
            values["pearson"] >= 0.98 for values in views.values()
        ),
        "both_views_normalized_rmse_at_most_0_20": all(
            values["normalized_rmse"] <= 0.20 for values in views.values()
        ),
        "positive_gain": all(values["gain"] > 0.0 for values in views.values()),
    }
    result = {
        "protocol_id": "UPATH-LIDC-v1.0",
        "gate": "P1-projector-calibration",
        "status": "pass" if all(checks.values()) else "fail",
        "definition_split": "train",
        "case_count": len(selected_indices),
        "selection_seed": args.seed,
        "detector_shape": [args.detector_size, args.detector_size],
        "ray_samples": args.ray_samples,
        "views": views,
        "checks": checks,
        "provenance": {
            "training_pack_summary_sha256": sha256_file(
                args.dataset_root / "summary.json"
            ),
            "training_pack_manifest_sha256": sha256_file(
                args.dataset_root / "manifest.csv"
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit("Differentiable projector calibration failed")


if __name__ == "__main__":
    main()
