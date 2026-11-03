"""Frozen data and provenance helpers for SOTA parity runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

CT_STORED_MIN = 0.0
CT_STORED_MAX = 2500.0
HU_OFFSET = -1000.0


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_case_ids(path: Path) -> list[str]:
    case_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs in {path}")
    return case_ids


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0 or count > size:
        raise ValueError(f"Invalid evenly spaced selection: size={size}, count={count}")
    indices = np.rint(np.linspace(0, size - 1, count)).astype(np.int64).tolist()
    if len(indices) != len(set(indices)):
        raise ValueError(f"Selection produced duplicate indices: {indices}")
    return indices


def resolve_h5_path(dataset_root: Path, case_id: str) -> Path:
    candidates = (
        dataset_root / "LIDC-HDF5-256" / case_id / "ct_xray_data.h5",
        dataset_root / case_id / "ct_xray_data.h5",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No HDF5 file found for {case_id} under {dataset_root}")


def load_stored_ct(dataset_root: Path, case_id: str) -> np.ndarray:
    path = resolve_h5_path(dataset_root, case_id)
    with h5py.File(path, "r") as handle:
        if "ct" not in handle:
            raise KeyError(f"Missing ct dataset in {path}")
        volume = np.asarray(handle["ct"], dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"Invalid CT in {path}: shape={volume.shape}")
    return volume


def resize_volume(
    volume: np.ndarray,
    shape: tuple[int, int, int] = (128, 128, 128),
    implementation: Literal["torch", "scipy"] = "torch",
) -> np.ndarray:
    source = np.asarray(volume, dtype=np.float32)
    if source.shape == shape:
        return np.ascontiguousarray(source)
    if implementation == "torch":
        tensor = torch.from_numpy(np.ascontiguousarray(source))[None, None]
        resized = F.interpolate(
            tensor,
            size=shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0].numpy()
    elif implementation == "scipy":
        zoom = np.asarray(shape, dtype=np.float64) / np.asarray(source.shape, dtype=np.float64)
        resized = ndimage.zoom(source, zoom=zoom, order=1)
        if resized.shape != shape:
            raise ValueError(f"SciPy resize produced {resized.shape}, expected {shape}")
    else:
        raise ValueError(f"Unsupported resize implementation: {implementation}")
    return np.ascontiguousarray(resized, dtype=np.float32)


def load_normalized_ct(
    dataset_root: Path,
    case_id: str,
    shape: tuple[int, int, int] = (128, 128, 128),
    implementation: Literal["torch", "scipy"] = "torch",
) -> np.ndarray:
    stored = load_stored_ct(dataset_root, case_id)
    resized = resize_volume(stored, shape=shape, implementation=implementation)
    return np.ascontiguousarray(
        np.clip(resized, CT_STORED_MIN, CT_STORED_MAX) / (CT_STORED_MAX - CT_STORED_MIN),
        dtype=np.float32,
    )


def normalized_to_hu(volume: np.ndarray) -> np.ndarray:
    return np.asarray(volume, dtype=np.float32) * (CT_STORED_MAX - CT_STORED_MIN) + HU_OFFSET


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
