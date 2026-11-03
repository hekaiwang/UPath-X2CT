"""Metric primitives shared by the frozen B0 evaluation pipeline."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _validate_pair(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"Expected same-shape 3D volumes, got {prediction.shape} and "
            f"{target.shape}"
        )
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Volumes contain non-finite values")
    tolerance = 1e-5
    if prediction.min() < -tolerance or prediction.max() > 1.0 + tolerance:
        raise ValueError("Prediction is outside [0, 1]")
    if target.min() < -tolerance or target.max() > 1.0 + tolerance:
        raise ValueError("Target is outside [0, 1]")
    return np.clip(prediction, 0.0, 1.0), np.clip(target, 0.0, 1.0)


def psnr_from_mse(
    mse: float,
    data_range: float = 1.0,
    identical_value: float = 100.0,
) -> float:
    if mse < 0 or not math.isfinite(mse):
        raise ValueError(f"Invalid MSE: {mse}")
    if mse == 0:
        return identical_value
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def psnr_3d(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> float:
    prediction, target = _validate_pair(prediction, target)
    mse = float(np.mean(np.square(prediction - target)))
    return psnr_from_mse(mse, data_range=data_range)


def summarize_rows(
    rows: Iterable[dict[str, float | str]],
) -> dict[str, dict[str, float]]:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot summarize an empty metric table")
    metric_keys = sorted(
        key
        for key, value in rows[0].items()
        if key != "case_id"
        and isinstance(value, (int, float, np.integer, np.floating))
    )
    summary: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        standard_error = (
            float(np.std(finite, ddof=1) / math.sqrt(finite.size))
            if finite.size > 1
            else 0.0
        )
        mean = float(np.mean(finite))
        summary[key] = {
            "mean": mean,
            "std": float(np.std(finite)),
            "ci95_low": mean - 1.96 * standard_error,
            "ci95_high": mean + 1.96 * standard_error,
            "n": int(finite.size),
        }
    return summary
