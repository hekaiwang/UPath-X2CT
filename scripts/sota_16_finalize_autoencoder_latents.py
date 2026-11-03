#!/usr/bin/env python3
"""Merge latent-cache shards and freeze train-only latent statistics."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from upathx2ct.sota.protocol import sha256_file, write_json
from upathx2ct.b0.ema import resolve_checkpoint_model_state
from upathx2ct.b0.training import atomic_torch_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"Existing latent differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    expected_rows = read_csv(args.dataset_root / "manifest.csv")
    expected_ids = {row["case_id"] for row in expected_rows}
    rows = []
    summaries = []
    total_sum = None
    total_sum_square = None
    total_voxels = 0
    for shard_index in range(args.num_shards):
        shard = args.latent_root / f"shard_{shard_index:02d}"
        summary_path = shard / "summary.json"
        manifest_path = shard / "manifest.csv"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete latent shard: {shard}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "pass":
            raise RuntimeError(f"Latent shard {shard_index} did not pass")
        summaries.append(summary)
        rows.extend(read_csv(manifest_path))
        moments = summary["train_moments"]
        values = np.asarray(moments["sum"], dtype=np.float64)
        squares = np.asarray(moments["sum_square"], dtype=np.float64)
        total_sum = values if total_sum is None else total_sum + values
        total_sum_square = (
            squares
            if total_sum_square is None
            else total_sum_square + squares
        )
        total_voxels += int(moments["voxels_per_channel"])

    observed_ids = [row["case_id"] for row in rows]
    duplicates = sorted(
        case_id
        for case_id, count in Counter(observed_ids).items()
        if count > 1
    )
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    if total_voxels <= 0:
        raise RuntimeError("No train latent moments were accumulated")
    mean = total_sum / total_voxels
    variance = np.maximum(total_sum_square / total_voxels - mean**2, 1e-12)
    std = np.sqrt(variance)
    rows.sort(key=lambda row: (row["split"], row["patient_id"], row["series_uid"]))
    for row in rows:
        source = Path(row["latent_path"])
        target = args.latent_root / "cases" / f"{row['case_id']}.npy"
        link_or_copy(source, target)
        row["latent_path"] = str(target.resolve())
    write_csv(args.latent_root / "manifest.csv", rows)
    for split in ("train", "val", "test"):
        case_ids = [row["case_id"] for row in rows if row["split"] == split]
        (args.latent_root / f"{split}.txt").write_text(
            "\n".join(case_ids) + ("\n" if case_ids else ""),
            encoding="utf-8",
        )
    latent_stats = {
        "protocol_id": "UPATH-LIDC-v1.0",
        "definition_split": "train",
        "channels": len(mean),
        "voxels_per_channel": total_voxels,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    write_json(args.latent_root / "latent_stats.json", latent_stats)

    checkpoint = torch.load(args.autoencoder_checkpoint, map_location="cpu")
    state = resolve_checkpoint_model_state(checkpoint, prefer_ema=True)
    state["latent_mean"] = torch.from_numpy(mean.astype(np.float32)).reshape(
        1,
        -1,
        1,
        1,
        1,
    )
    state["latent_std"] = torch.from_numpy(std.astype(np.float32)).reshape(
        1,
        -1,
        1,
        1,
        1,
    )
    frozen_checkpoint = args.latent_root / "autoencoder_frozen.pt"
    atomic_torch_save(
        {
            "model": state,
            "autoencoder_config": checkpoint["autoencoder_config"],
            "latent_stats": latent_stats,
            "source_best_metrics": checkpoint.get(
                "best_ceiling_metrics",
                checkpoint.get("best_metrics", {}),
            ),
            "source_checkpoint_sha256": sha256_file(
                args.autoencoder_checkpoint
            ),
        },
        frozen_checkpoint,
    )
    checks = {
        "all_shards_passed": all(
            summary.get("status") == "pass" for summary in summaries
        ),
        "all_cases_present": not missing,
        "no_unexpected_cases": not unexpected,
        "no_duplicate_cases": not duplicates,
        "finite_positive_latent_std": bool(
            np.isfinite(std).all() and np.all(std > 0)
        ),
        "frozen_autoencoder_written": frozen_checkpoint.is_file(),
    }
    result = {
        "protocol_id": "UPATH-LIDC-v1.0",
        "gate": "L0-finalize",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "expected": len(expected_ids),
            "observed": len(rows),
        },
        "failures": {
            "missing": missing,
            "unexpected": unexpected,
            "duplicates": duplicates,
        },
        "latent_stats": latent_stats,
        "provenance": {
            "training_pack_manifest_sha256": sha256_file(
                args.dataset_root / "manifest.csv"
            ),
            "latent_manifest_sha256": sha256_file(
                args.latent_root / "manifest.csv"
            ),
            "source_autoencoder_sha256": sha256_file(
                args.autoencoder_checkpoint
            ),
            "frozen_autoencoder_sha256": sha256_file(frozen_checkpoint),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(args.latent_root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit("Latent finalization failed")


if __name__ == "__main__":
    main()
