#!/usr/bin/env python3
"""Cache raw CT autoencoder means in deterministic GPU shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from upathx2ct.sota.canonical import stable_balanced_shards
from upathx2ct.sota.protocol import sha256_file, write_json
from upathx2ct.b0.autoencoder import load_autoencoder_checkpoint
from upathx2ct.b0.training import autocast_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
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


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid latent-cache shard declaration")
    summary = json.loads(
        (args.dataset_root / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "pass":
        raise RuntimeError("Training pack has not passed")
    rows = read_csv(args.dataset_root / "manifest.csv")
    by_id = {row["case_id"]: row for row in rows}
    selected_ids = stable_balanced_shards(by_id, args.num_shards)[args.shard_index]
    if args.max_cases:
        selected_ids = selected_ids[: args.max_cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_autoencoder_checkpoint(
        str(args.autoencoder_checkpoint),
        device="cpu",
    ).to(args.device)
    model.eval().requires_grad_(False)

    records = []
    errors = []
    channel_sum = None
    channel_sum_square = None
    train_voxels_per_channel = 0
    for index, case_id in enumerate(selected_ids, start=1):
        row = by_id[case_id]
        output_path = args.output_dir / "cases" / f"{case_id}.npy"
        try:
            if output_path.is_file() and args.resume:
                latent = np.load(output_path).astype(np.float32, copy=False)
            else:
                with h5py.File(row["packed_path"], "r") as handle:
                    ct = np.asarray(handle["ct"], dtype=np.float32)
                tensor = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0).to(args.device)
                with torch.inference_mode(), autocast_context(
                    torch.device(args.device),
                    args.precision,
                ):
                    mean, _ = model.encode_distribution(tensor)
                latent = mean[0].float().cpu().numpy()
                save_npy_atomic(output_path, latent.astype(np.float16))
                latent = np.load(output_path).astype(np.float32, copy=False)
            if row["split"] == "train":
                values = latent.astype(np.float64).reshape(latent.shape[0], -1)
                current_sum = values.sum(axis=1)
                current_sum_square = np.square(values).sum(axis=1)
                channel_sum = (
                    current_sum
                    if channel_sum is None
                    else channel_sum + current_sum
                )
                channel_sum_square = (
                    current_sum_square
                    if channel_sum_square is None
                    else channel_sum_square + current_sum_square
                )
                train_voxels_per_channel += values.shape[1]
            records.append(
                {
                    "case_id": case_id,
                    "patient_id": row["patient_id"],
                    "series_uid": row["series_uid"],
                    "split": row["split"],
                    "latent_path": str(output_path.resolve()),
                    "latent_sha256": sha256_file(output_path),
                    "latent_shape": "x".join(str(value) for value in latent.shape),
                }
            )
            print(f"[{index}/{len(selected_ids)}] {case_id}", flush=True)
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
        "all_latent_hashes_recorded": all(
            len(row["latent_sha256"]) == 64 for row in records
        ),
        "train_moments_available": train_voxels_per_channel > 0,
    }
    result = {
        "protocol_id": "UPATH-LIDC-v1.0",
        "gate": "L0-cache-shard",
        "status": "pass" if all(checks.values()) else "fail",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "checks": checks,
        "counts": {
            "selected": len(selected_ids),
            "cached": len(records),
            "errors": len(errors),
        },
        "train_moments": {
            "voxels_per_channel": train_voxels_per_channel,
            "sum": channel_sum.tolist() if channel_sum is not None else [],
            "sum_square": (
                channel_sum_square.tolist()
                if channel_sum_square is not None
                else []
            ),
        },
        "provenance": {
            "training_pack_summary_sha256": sha256_file(
                args.dataset_root / "summary.json"
            ),
            "autoencoder_checkpoint_sha256": sha256_file(
                args.autoencoder_checkpoint
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit("Latent-cache shard failed")


if __name__ == "__main__":
    main()
