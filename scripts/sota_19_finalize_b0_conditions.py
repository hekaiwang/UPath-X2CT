#!/usr/bin/env python3
"""Merge compatible B0 condition shards and freeze residual scales."""

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

from upathx2ct.sota.protocol import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
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
            raise ValueError(f"Existing cache differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    expected = read_csv(args.dataset_root / "manifest.csv")
    expected_ids = {row["case_id"] for row in expected}
    rows = []
    summaries = []
    summary_paths = []
    total_sum = None
    total_sum_square = None
    total_voxels = 0
    for shard_index in range(args.num_shards):
        shard = args.cache_root / f"shard_{shard_index:02d}"
        summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "pass":
            raise RuntimeError(f"Cache shard {shard_index} did not pass")
        if (
            int(summary.get("shard_index", -1)) != shard_index
            or int(summary.get("num_shards", -1)) != args.num_shards
        ):
            raise RuntimeError(
                f"Cache shard declaration differs for shard {shard_index}"
            )
        summaries.append(summary)
        summary_paths.append(shard / "summary.json")
        rows.extend(read_csv(shard / "manifest.csv"))
        moments = summary["train_residual_moments"]
        values = np.asarray(moments["sum"], dtype=np.float64)
        squares = np.asarray(moments["sum_square"], dtype=np.float64)
        total_sum = values if total_sum is None else total_sum + values
        total_sum_square = (
            squares
            if total_sum_square is None
            else total_sum_square + squares
        )
        total_voxels += int(moments["voxels_per_channel"])
    protocol_ids = {summary.get("protocol_id") for summary in summaries}
    model_variants = {
        summary.get("model_variant", "B0") for summary in summaries
    }
    source_signatures = {
        summary.get("provenance", {}).get("coarse_checkpoint_sha256")
        for summary in summaries
    }
    if (
        len(protocol_ids) != 1
        or len(model_variants) != 1
        or len(source_signatures) != 1
    ):
        raise RuntimeError(
            "Refusing to merge cache shards from different protocols, "
            "model variants or checkpoints"
        )
    protocol_id = str(next(iter(protocol_ids)))
    model_variant = str(next(iter(model_variants)))
    if model_variant != "B0":
        raise RuntimeError("B0 cache finalization received a non-B0 shard")
    observed_ids = [row["case_id"] for row in rows]
    duplicates = sorted(
        case_id
        for case_id, count in Counter(observed_ids).items()
        if count > 1
    )
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    residual_mean = total_sum / total_voxels
    residual_variance = np.maximum(
        total_sum_square / total_voxels - residual_mean**2,
        1e-8,
    )
    residual_std = np.sqrt(residual_variance)
    rows.sort(key=lambda row: (row["split"], row["patient_id"], row["series_uid"]))
    for row in rows:
        source = Path(row["cache_path"])
        target = args.cache_root / "cases" / f"{row['case_id']}.npz"
        link_or_copy(source, target)
        row["cache_path"] = str(target.resolve())
    write_csv(args.cache_root / "manifest.csv", rows)
    residual_stats = {
        "protocol_id": protocol_id,
        "definition_split": "train",
        "voxels_per_channel": total_voxels,
        "mean": residual_mean.tolist(),
        "std": residual_std.tolist(),
    }
    write_json(args.cache_root / "residual_stats.json", residual_stats)
    checks = {
        "all_shards_passed": all(
            summary.get("status") == "pass" for summary in summaries
        ),
        "all_cases_present": not missing,
        "no_unexpected_cases": not unexpected,
        "no_duplicate_cases": not duplicates,
        "finite_positive_residual_std": bool(
            np.isfinite(residual_std).all() and np.all(residual_std > 0)
        ),
    }
    result = {
        "protocol_id": protocol_id,
        "gate": f"{model_variant}-cache-finalize",
        "status": "pass" if all(checks.values()) else "fail",
        "model_variant": model_variant,
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
        "residual_stats": residual_stats,
        "provenance": {
            "training_pack_manifest_sha256": sha256_file(
                args.dataset_root / "manifest.csv"
            ),
            "cache_manifest_sha256": sha256_file(
                args.cache_root / "manifest.csv"
            ),
            "source_checkpoint_sha256": {
                "coarse": next(iter(source_signatures))[0],
                "evidence": next(iter(source_signatures))[1],
                "pathology": next(iter(source_signatures))[2],
            },
            "shard_summary_sha256": [
                sha256_file(path) for path in summary_paths
            ],
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(args.cache_root / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(f"{model_variant} cache finalization failed")


if __name__ == "__main__":
    main()
