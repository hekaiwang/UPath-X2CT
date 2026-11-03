"""Training-ready UPath-LIDC-v1 HDF5 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_drr(
    image: np.ndarray,
    mean: float,
    std: float,
    clip: float = 6.0,
) -> np.ndarray:
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("DRR normalization statistics must be finite and nonzero")
    transformed = np.log1p(np.maximum(np.asarray(image, dtype=np.float32), 0.0))
    return np.clip((transformed - mean) / std, -clip, clip).astype(
        np.float32,
        copy=False,
    )


class UPathProjectionOnlyDataset(Dataset):
    """Load stored AP/LAT projections without opening any target-CT dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        max_samples: int = 0,
    ) -> None:
        self.root = Path(root)
        split_path = self.root / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        self.split = split
        self.case_ids = [
            value.strip()
            for value in split_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if max_samples > 0:
            self.case_ids = self.case_ids[:max_samples]
        if not self.case_ids:
            raise ValueError(f"No cases selected from {split_path}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def case_path(self, case_id: str) -> Path:
        return self.root / "cases" / f"{case_id}.h5"

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id = self.case_ids[index]
        path = self.case_path(case_id)
        with h5py.File(path, "r") as handle:
            ap_raw = np.asarray(
                handle["drr_frontal_ap"],
                dtype=np.float32,
            )
            lat_raw = np.asarray(
                handle["drr_left_lateral"],
                dtype=np.float32,
            )
            stored_split = str(handle.attrs["split"])
        if stored_split != self.split:
            raise ValueError(
                f"Split mismatch for {case_id}: "
                f"manifest={self.split}, HDF5={stored_split}"
            )
        if ap_raw.shape != lat_raw.shape or ap_raw.ndim != 2:
            raise ValueError(
                f"Invalid AP/LAT shapes for {case_id}: "
                f"{ap_raw.shape}, {lat_raw.shape}"
            )
        if not np.isfinite(ap_raw).all() or not np.isfinite(lat_raw).all():
            raise ValueError(f"Nonfinite AP/LAT projection for {case_id}")
        return {
            "case_id": case_id,
            "split": stored_split,
            "xray_ap_raw": torch.from_numpy(
                np.ascontiguousarray(ap_raw)
            ).unsqueeze(0),
            "xray_lat_raw": torch.from_numpy(
                np.ascontiguousarray(lat_raw)
            ).unsqueeze(0),
            "path": str(path),
        }


class UPathLIDC128Dataset(Dataset):
    """Load the immutable 128-cubed global reconstruction representation."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        max_samples: int = 0,
        ct_only_augmentation: bool = False,
        include_raw_drr: bool = False,
        load_projections: bool = True,
        latent_root: str | Path | None = None,
        coarse_cache_root: str | Path | None = None,
        lesion_sidecar_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        split_path = self.root / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        self.case_ids = [
            value.strip()
            for value in split_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if max_samples > 0:
            self.case_ids = self.case_ids[:max_samples]
        if not self.case_ids:
            raise ValueError(f"No cases selected from {split_path}")
        stats = json.loads(
            (self.root / "projection_stats.json").read_text(encoding="utf-8")
        )
        if stats.get("definition_split") != "train":
            raise ValueError("Projection statistics were not defined from train only")
        self.projection_stats = stats["views"]
        self.ct_only_augmentation = ct_only_augmentation
        self.include_raw_drr = include_raw_drr
        self.load_projections = load_projections
        self.latent_root = Path(latent_root) if latent_root is not None else None
        self.coarse_cache_root = (
            Path(coarse_cache_root) if coarse_cache_root is not None else None
        )
        self.lesion_sidecar_root = (
            Path(lesion_sidecar_root)
            if lesion_sidecar_root is not None
            else None
        )
        if self.lesion_sidecar_root is not None:
            sidecar_summary = json.loads(
                (self.lesion_sidecar_root / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            if sidecar_summary.get("status") != "pass":
                raise ValueError("LIDC XML lesion sidecar has not passed")
            sidecar_cases = {
                value.strip()
                for value in (
                    self.lesion_sidecar_root / f"{split}.txt"
                ).read_text(encoding="utf-8").splitlines()
                if value.strip()
            }
            missing_sidecars = sorted(set(self.case_ids) - sidecar_cases)
            if missing_sidecars:
                raise ValueError(
                    "LIDC XML lesion sidecar is missing split cases: "
                    f"{missing_sidecars[:5]}"
                )
        self.latent_mean = None
        self.latent_std = None
        if self.latent_root is not None:
            latent_stats = json.loads(
                (self.latent_root / "latent_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            if latent_stats.get("definition_split") != "train":
                raise ValueError("Latent statistics were not defined from train only")
            self.latent_mean = np.asarray(
                latent_stats["mean"],
                dtype=np.float32,
            ).reshape(-1, 1, 1, 1)
            self.latent_std = np.asarray(
                latent_stats["std"],
                dtype=np.float32,
            ).reshape(-1, 1, 1, 1)
            if (
                not np.isfinite(self.latent_mean).all()
                or not np.isfinite(self.latent_std).all()
                or np.any(self.latent_std <= 0)
            ):
                raise ValueError("Invalid cached latent normalization statistics")

    def __len__(self) -> int:
        return len(self.case_ids)

    def case_path(self, case_id: str) -> Path:
        return self.root / "cases" / f"{case_id}.h5"

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id = self.case_ids[index]
        path = self.case_path(case_id)
        with h5py.File(path, "r") as handle:
            ct = np.asarray(handle["ct"], dtype=np.float32)
            body = np.asarray(handle["body_mask"], dtype=np.float32)
            lung = np.asarray(handle["lung_mask"], dtype=np.float32)
            lesion = np.asarray(handle["lesion_proxy"], dtype=np.float32)
            if self.load_projections:
                ap_raw = np.asarray(handle["drr_frontal_ap"], dtype=np.float32)
                lat_raw = np.asarray(handle["drr_left_lateral"], dtype=np.float32)
                matrices = np.asarray(
                    handle["projection_matrices"],
                    dtype=np.float32,
                )
            nodule_count = int(handle.attrs["nodule_count"])
            split = str(handle.attrs["split"])
            patient_id = str(handle.attrs["patient_id"])
            series_uid = str(handle.attrs["series_uid"])
            source_case_dir = str(handle.attrs["source_case_dir"])
            metadata_json = str(handle.attrs["metadata_json"])

        flipped_axes = []
        if self.ct_only_augmentation:
            for axis in range(3):
                if torch.rand(()) < 0.5:
                    flipped_axes.append(axis)
                    ct = np.flip(ct, axis=axis).copy()
                    body = np.flip(body, axis=axis).copy()
                    lung = np.flip(lung, axis=axis).copy()
                    lesion = np.flip(lesion, axis=axis).copy()
        result: dict[str, Any] = {
            "case_id": case_id,
            "patient_id": patient_id,
            "series_uid": series_uid,
            "split": split,
            "ct": torch.from_numpy(np.ascontiguousarray(ct)).unsqueeze(0),
            "body_mask": torch.from_numpy(np.ascontiguousarray(body)).unsqueeze(0),
            "lung_mask": torch.from_numpy(np.ascontiguousarray(lung)).unsqueeze(0),
            "lesion_mask": torch.from_numpy(
                np.ascontiguousarray(lesion)
            ).unsqueeze(0),
            "has_lesion_label": torch.tensor(
                float(nodule_count > 0),
                dtype=torch.float32,
            ),
            "metadata_json": metadata_json,
            "source_case_dir": source_case_dir,
            "path": str(path),
        }
        if self.load_projections:
            ap = normalize_drr(
                ap_raw,
                float(self.projection_stats["frontal_ap"]["log1p_mean"]),
                float(self.projection_stats["frontal_ap"]["log1p_std"]),
            )
            lat = normalize_drr(
                lat_raw,
                float(self.projection_stats["left_lateral"]["log1p_mean"]),
                float(self.projection_stats["left_lateral"]["log1p_std"]),
            )
            result.update(
                xray_ap=torch.from_numpy(np.ascontiguousarray(ap)).unsqueeze(0),
                xray_lat=torch.from_numpy(np.ascontiguousarray(lat)).unsqueeze(0),
                projection_matrices=torch.from_numpy(
                    np.ascontiguousarray(matrices)
                ),
            )
        if self.load_projections and self.include_raw_drr:
            result["xray_ap_raw"] = torch.from_numpy(
                np.ascontiguousarray(ap_raw)
            ).unsqueeze(0)
            result["xray_lat_raw"] = torch.from_numpy(
                np.ascontiguousarray(lat_raw)
            ).unsqueeze(0)
        if self.latent_root is not None:
            latent_path = self.latent_root / "cases" / f"{case_id}.npy"
            latent = np.load(latent_path).astype(np.float32, copy=False)
            if latent.shape[0] != self.latent_mean.shape[0]:
                raise ValueError(
                    f"Latent channels differ for {case_id}: {latent.shape}"
                )
            normalized_latent = (latent - self.latent_mean) / self.latent_std
            result["target_latent"] = torch.from_numpy(
                np.ascontiguousarray(normalized_latent)
            )
        if self.coarse_cache_root is not None:
            cache_path = self.coarse_cache_root / "cases" / f"{case_id}.npz"
            with np.load(cache_path) as handle:
                coarse = np.asarray(handle["coarse_latent"], dtype=np.float32)
                condition = np.asarray(handle["condition"], dtype=np.float32)
            result["coarse_latent"] = torch.from_numpy(
                np.ascontiguousarray(coarse)
            )
            result["condition"] = torch.from_numpy(
                np.ascontiguousarray(condition)
            )
        if self.lesion_sidecar_root is not None:
            sidecar_path = (
                self.lesion_sidecar_root / "cases" / f"{case_id}.h5"
            )
            with h5py.File(sidecar_path, "r") as handle:
                if str(handle.attrs["case_id"]) != case_id:
                    raise ValueError(f"Lesion sidecar case mismatch: {sidecar_path}")
                probability = np.asarray(
                    handle["probability_global_128"],
                    dtype=np.float32,
                )
                consensus = np.asarray(
                    handle["consensus_global_128"],
                    dtype=np.float32,
                )
                support = np.asarray(
                    handle["reader_support_global_128"],
                    dtype=np.float32,
                )
                target_count = int(handle.attrs["eligible_target_count"])
                source_target_count = int(
                    handle.attrs["source_luna_target_count"]
                )
            if (
                probability.shape != ct.shape
                or consensus.shape != ct.shape
                or support.shape != ct.shape
            ):
                raise ValueError(
                    f"Lesion sidecar/grid mismatch for {case_id}: "
                    f"{probability.shape}, {consensus.shape}, "
                    f"{support.shape}, {ct.shape}"
                )
            for axis in flipped_axes:
                probability = np.flip(probability, axis=axis).copy()
                consensus = np.flip(consensus, axis=axis).copy()
                support = np.flip(support, axis=axis).copy()
            result.update(
                lesion_probability=torch.from_numpy(
                    np.ascontiguousarray(probability)
                ).unsqueeze(0),
                lesion_consensus_mask=torch.from_numpy(
                    np.ascontiguousarray(consensus)
                ).unsqueeze(0),
                lesion_reader_support=torch.from_numpy(
                    np.ascontiguousarray(support)
                ).unsqueeze(0),
                has_xml_contour_target=torch.tensor(
                    float(target_count > 0),
                    dtype=torch.float32,
                ),
                xml_contour_target_count=torch.tensor(
                    target_count,
                    dtype=torch.int64,
                ),
                source_luna_target_count=torch.tensor(
                    source_target_count,
                    dtype=torch.int64,
                ),
                lesion_sidecar_path=str(sidecar_path),
            )
        return result
