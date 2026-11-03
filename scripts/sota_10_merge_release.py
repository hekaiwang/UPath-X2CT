#!/usr/bin/env python3
"""Merge and audit completed UPath-LIDC-v1 release shards without copying data."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from upathx2ct.sota.protocol import sha256_file, write_json


REQUIRED_CASE_FILES = (
    "ct_full_256.nii.gz",
    "ct_global_128.nii.gz",
    "body_mask_full_256.nii.gz",
    "lung_mask_full_256.nii.gz",
    "lesion_proxy_full_256.nii.gz",
    "body_mask_global_128.nii.gz",
    "lung_mask_global_128.nii.gz",
    "lesion_proxy_global_128.nii.gz",
    "drrs_ap_left_lateral.npz",
    "projection_matrices_centered_lps.npz",
    "metadata.json",
    "record.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/processed/upath_lidc_v1_release_20260724"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/manifests/upath_lidc_v1.csv"),
    )
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def patient_role_map(rows: list[dict[str, str]]) -> tuple[dict[str, str], list[str]]:
    roles: dict[str, str] = {}
    conflicts = []
    for row in rows:
        patient = row["patient_id"]
        role = row["split"]
        previous = roles.setdefault(patient, role)
        if previous != role:
            conflicts.append(f"{patient}:{previous}/{role}")
    return roles, conflicts


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    release_root = (
        args.release_root if args.release_root.is_absolute() else root / args.release_root
    )
    source_manifest = (
        args.source_manifest
        if args.source_manifest.is_absolute()
        else root / args.source_manifest
    )
    if args.num_shards <= 0:
        raise ValueError("Number of release shards must be positive")
    source_rows = read_csv(source_manifest)
    source_by_uid = {row["series_uid"]: row for row in source_rows}
    if len(source_by_uid) != len(source_rows):
        raise ValueError("Source manifest contains duplicate series UIDs")

    shard_summaries = []
    release_rows: list[dict[str, Any]] = []
    missing_files = []
    hash_failures = []
    for shard_index in range(args.num_shards):
        shard_dir = release_root / f"shard_{shard_index:02d}"
        summary_path = shard_dir / "summary.json"
        manifest_path = shard_dir / "case_manifest.csv"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete release shard: {shard_dir}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_summaries.append(summary)
        for row in read_csv(manifest_path):
            case_dir = shard_dir / "cases" / row["case_id"]
            absent = [
                name for name in REQUIRED_CASE_FILES if not (case_dir / name).is_file()
            ]
            if absent:
                missing_files.append(
                    {
                        "series_uid": row["series_uid"],
                        "case_dir": str(case_dir),
                        "missing": absent,
                    }
                )
            if args.verify_hashes and (case_dir / "metadata.json").is_file():
                metadata = json.loads(
                    (case_dir / "metadata.json").read_text(encoding="utf-8")
                )
                for name, expected in metadata.get("hashes", {}).items():
                    path = case_dir / name
                    if not path.is_file() or sha256_file(path) != expected:
                        hash_failures.append(
                            {
                                "series_uid": row["series_uid"],
                                "file": str(path),
                            }
                        )
            source = source_by_uid.get(row["series_uid"])
            if source is None:
                raise ValueError(f"Release series absent from D0: {row['series_uid']}")
            release_rows.append(
                {
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "series_uid": row["series_uid"],
                    "split": row["split"],
                    "has_nodule": source["has_nodule"],
                    "nodule_count": row["nodule_count"],
                    "selection_evidence_tier": row["selection_evidence_tier"],
                    "case_dir": str(case_dir.resolve()),
                    "record_path": str((case_dir / "record.json").resolve()),
                    "metadata_path": str((case_dir / "metadata.json").resolve()),
                }
            )

    release_rows.sort(key=lambda row: (row["patient_id"], row["series_uid"]))
    release_uids = [row["series_uid"] for row in release_rows]
    duplicate_uids = sorted(
        uid for uid, count in Counter(release_uids).items() if count > 1
    )
    source_uids = set(source_by_uid)
    release_uid_set = set(release_uids)
    missing_uids = sorted(source_uids - release_uid_set)
    unexpected_uids = sorted(release_uid_set - source_uids)
    roles, role_conflicts = patient_role_map(
        [
            {"patient_id": row["patient_id"], "split": row["split"]}
            for row in release_rows
        ]
    )
    source_roles, source_role_conflicts = patient_role_map(source_rows)
    role_mismatches = sorted(
        patient
        for patient, role in roles.items()
        if source_roles.get(patient) != role
    )
    checks = {
        "all_shards_passed": all(
            summary.get("status") == "pass" for summary in shard_summaries
        ),
        "exactly_1018_series": len(release_rows) == 1018,
        "no_duplicate_series": not duplicate_uids,
        "no_missing_series": not missing_uids,
        "no_unexpected_series": not unexpected_uids,
        "all_required_case_files_present": not missing_files,
        "verified_hashes_match": not hash_failures,
        "release_patient_roles_disjoint": not role_conflicts,
        "source_patient_roles_disjoint": not source_role_conflicts,
        "release_roles_match_d0": not role_mismatches,
    }
    status = "pass" if all(checks.values()) else "fail"
    write_csv(release_root / "manifest.csv", release_rows)
    for split in ("train", "val", "test"):
        values = [
            row["case_id"] for row in release_rows if row["split"] == split
        ]
        (release_root / f"{split}.txt").write_text(
            "\n".join(values) + ("\n" if values else ""),
            encoding="utf-8",
        )
    summary = {
        "protocol_id": "UPATH-LIDC-v1.0",
        "gate": "R0",
        "status": status,
        "checks": checks,
        "hash_verification_requested": args.verify_hashes,
        "counts": {
            "series": len(release_rows),
            "patients": len(roles),
            "train_series": sum(row["split"] == "train" for row in release_rows),
            "val_series": sum(row["split"] == "val" for row in release_rows),
            "test_series": sum(row["split"] == "test" for row in release_rows),
            "positive_series": sum(row["has_nodule"] == "1" for row in release_rows),
            "negative_series": sum(row["has_nodule"] == "0" for row in release_rows),
        },
        "failures": {
            "duplicate_uids": duplicate_uids,
            "missing_uids": missing_uids,
            "unexpected_uids": unexpected_uids,
            "missing_files": missing_files,
            "hash_failures": hash_failures,
            "role_conflicts": role_conflicts,
            "source_role_conflicts": source_role_conflicts,
            "role_mismatches": role_mismatches,
        },
        "provenance": {
            "source_manifest_sha256": sha256_file(source_manifest),
            "merged_manifest_sha256": sha256_file(release_root / "manifest.csv"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "shard_summary_sha256": [
                sha256_file(
                    release_root / f"shard_{index:02d}" / "summary.json"
                )
                for index in range(args.num_shards)
            ],
        },
    }
    write_json(release_root / "summary.json", summary)
    (release_root / "README.md").write_text(
        "# UPath-LIDC-v1 full synchronized release\n\n"
        f"Status: **{status.upper()}**.\n\n"
        "The merged manifest references immutable case directories in each "
        "deterministic shard; no medical data are duplicated during merge.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if status != "pass":
        raise SystemExit("R0 merge failed; O1 and B0 remain locked")


if __name__ == "__main__":
    main()
