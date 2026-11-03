#!/usr/bin/env python3
"""Cache frozen B0 coarse latents and conditions in GPU shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from upathx2ct.sota.canonical import stable_balanced_shards
from upathx2ct.sota.data import UPathLIDC128Dataset
from upathx2ct.sota.model import load_b0_coarse_checkpoint
from upathx2ct.sota.protocol import sha256_file, write_json
from upathx2ct.b0.training import autocast_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_cache_atomic(
    path: Path,
    coarse: np.ndarray,
    condition: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            coarse_latent=coarse.astype(np.float16),
            condition=condition.astype(np.float16),
        )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid B0-cache shard declaration")
    rows = read_csv(args.dataset_root / "manifest.csv")
    by_id = {row["case_id"]: row for row in rows}
    selected_ids = stable_balanced_shards(by_id, args.num_shards)[args.shard_index]
    if args.max_cases:
        selected_ids = selected_ids[: args.max_cases]
    dataset = UPathLIDC128Dataset(
        args.dataset_root,
        "train",
        latent_root=args.latent_root,
    )
    dataset.case_ids = selected_ids
    coarse_hash = sha256_file(args.coarse_checkpoint)
    model = load_b0_coarse_checkpoint(
        args.coarse_checkpoint,
        args.device,
    )
    model_variant = "B0"
    protocol_id = "UPATH-LIDC-v1.0"
    model.eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_summary_path = args.output_dir / "summary.json"
    if args.resume and prior_summary_path.is_file():
        prior = json.loads(
            prior_summary_path.read_text(encoding="utf-8")
        )
        prior_provenance = prior.get("provenance", {})
        if (
            prior_provenance.get("coarse_checkpoint_sha256")
            != coarse_hash
            or int(prior.get("num_shards", -1)) != args.num_shards
            or int(prior.get("shard_index", -1)) != args.shard_index
        ):
            raise ValueError(
                "Refusing to resume a cache built from different "
                "checkpoints or shard settings"
            )

    records = []
    errors = []
    residual_sum = None
    residual_sum_square = None
    train_voxels_per_channel = 0
    for index in range(len(dataset)):
        item = dataset[index]
        case_id = item["case_id"]
        row = by_id[case_id]
        output_path = args.output_dir / "cases" / f"{case_id}.npz"
        try:
            if output_path.is_file() and args.resume:
                with np.load(output_path) as handle:
                    coarse = np.asarray(
                        handle["coarse_latent"],
                        dtype=np.float32,
                    )
                    condition = np.asarray(
                        handle["condition"],
                        dtype=np.float32,
                    )
            else:
                with torch.inference_mode(), autocast_context(
                    torch.device(args.device),
                    args.precision,
                ):
                    output = model(
                        item["xray_ap"].unsqueeze(0).to(args.device),
                        item["xray_lat"].unsqueeze(0).to(args.device),
                        item["projection_matrices"].unsqueeze(0).to(args.device),
                    )
                coarse = output["coarse_latent"][0].float().cpu().numpy()
                condition = output["condition"][0].float().cpu().numpy()
                save_cache_atomic(output_path, coarse, condition)
                with np.load(output_path) as handle:
                    coarse = np.asarray(
                        handle["coarse_latent"],
                        dtype=np.float32,
                    )
                    condition = np.asarray(
                        handle["condition"],
                        dtype=np.float32,
                    )
            if row["split"] == "train":
                target = item["target_latent"].numpy()
                residual = (target - coarse).astype(np.float64).reshape(
                    coarse.shape[0],
                    -1,
                )
                current_sum = residual.sum(axis=1)
                current_square = np.square(residual).sum(axis=1)
                residual_sum = (
                    current_sum
                    if residual_sum is None
                    else residual_sum + current_sum
                )
                residual_sum_square = (
                    current_square
                    if residual_sum_square is None
                    else residual_sum_square + current_square
                )
                train_voxels_per_channel += residual.shape[1]
            records.append(
                {
                    "case_id": case_id,
                    "patient_id": row["patient_id"],
                    "series_uid": row["series_uid"],
                    "split": row["split"],
                    "cache_path": str(output_path.resolve()),
                    "cache_sha256": sha256_file(output_path),
                    "coarse_shape": "x".join(str(value) for value in coarse.shape),
                    "condition_shape": "x".join(
                        str(value) for value in condition.shape
                    ),
                }
            )
            print(f"[{index + 1}/{len(dataset)}] {case_id}", flush=True)
        except Exception as error:
            errors.append(
                {
                    "case_id": case_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"ERROR {case_id}: {errors[-1]['error']}", flush=True)
    records.sort(key=lambda row: (row["split"], row["patient_id"], row["series_uid"]))
    if records:
        write_csv(args.output_dir / "manifest.csv", records)
    if errors:
        write_csv(args.output_dir / "failures.csv", errors)
    checks = {
        "all_selected_cases_cached": len(records) == len(selected_ids) and not errors,
        "train_residual_moments_available": train_voxels_per_channel > 0,
        "all_cache_hashes_recorded": all(
            len(row["cache_sha256"]) == 64 for row in records
        ),
    }
    result = {
        "protocol_id": protocol_id,
        "gate": f"{model_variant}-cache-shard",
        "status": "pass" if all(checks.values()) else "fail",
        "model_variant": model_variant,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "checks": checks,
        "counts": {
            "selected": len(selected_ids),
            "cached": len(records),
            "errors": len(errors),
        },
        "train_residual_moments": {
            "voxels_per_channel": train_voxels_per_channel,
            "sum": residual_sum.tolist() if residual_sum is not None else [],
            "sum_square": (
                residual_sum_square.tolist()
                if residual_sum_square is not None
                else []
            ),
        },
        "provenance": {
            "training_pack_manifest_sha256": sha256_file(
                args.dataset_root / "manifest.csv"
            ),
            "latent_summary_sha256": sha256_file(
                args.latent_root / "summary.json"
            ),
            "coarse_checkpoint_sha256": coarse_hash,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(f"{model_variant} condition-cache shard failed")


if __name__ == "__main__":
    main()
