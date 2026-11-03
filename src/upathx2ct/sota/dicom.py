"""Deterministic repair of non-uniform axial DICOM slice stacks."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Invalid DICOM {name} direction")
    return vector / norm


def _nominal_spacing(coordinates: np.ndarray) -> float:
    differences = np.diff(np.sort(coordinates))
    positive = differences[differences > 1e-4]
    if positive.size == 0:
        raise ValueError("DICOM stack has fewer than two unique slice positions")
    rounded = np.round(positive, decimals=3)
    counts = Counter(float(value) for value in rounded)
    mode = min(counts, key=lambda value: (-counts[value], value))
    cluster = positive[np.abs(positive - mode) <= max(0.05, 0.05 * mode)]
    spacing = float(np.median(cluster))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Unable to estimate positive DICOM slice spacing")
    return spacing


def _best_lattice_anchor(
    coordinates: np.ndarray,
    spacing: float,
    tolerance_mm: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    best: tuple[int, float, float, np.ndarray, np.ndarray] | None = None
    for anchor in np.sort(coordinates):
        indices = np.rint((coordinates - anchor) / spacing).astype(np.int64)
        residuals = np.abs(coordinates - (anchor + indices * spacing))
        inliers = residuals <= tolerance_mm
        candidate = (
            int(inliers.sum()),
            -float(residuals[inliers].sum()),
            -float(anchor),
            indices,
            residuals,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError("No DICOM lattice anchor candidate")
    return -best[2], best[3], best[4]


def _largest_position_component(
    coordinates: np.ndarray,
    source_indices: np.ndarray,
    spacing: float,
    max_missing_slices: int,
) -> np.ndarray:
    unique = np.unique(coordinates)
    maximum_gap = (max_missing_slices + 1) * spacing
    boundaries = np.where(np.diff(unique) > maximum_gap)[0] + 1
    components = np.split(unique, boundaries)
    selected = max(
        components,
        key=lambda values: (
            int(
                (
                    (coordinates >= values[0] - 1e-4)
                    & (coordinates <= values[-1] + 1e-4)
                ).sum()
            ),
            len(values),
            float(values[-1] - values[0]),
            -float(values[0]),
        ),
    )
    return source_indices[
        (coordinates >= selected[0] - 1e-4)
        & (coordinates <= selected[-1] + 1e-4)
    ]


def _transition_cost(
    previous: np.ndarray,
    current: np.ndarray,
    grid_distance: int,
) -> float:
    previous_small = previous[::8, ::8].astype(np.float32, copy=False)
    current_small = current[::8, ::8].astype(np.float32, copy=False)
    return float(
        np.mean(np.abs(previous_small - current_small))
        / max(grid_distance, 1)
    )


def _select_duplicate_path(
    volume_zyx: np.ndarray,
    coordinates: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate_keys = np.rint(coordinates * 10000.0).astype(np.int64)
    groups = {
        int(key): np.sort(source_indices[coordinate_keys == key])
        for key in np.unique(coordinate_keys)
    }
    ordered = sorted(groups)
    costs = np.zeros(len(groups[ordered[0]]), dtype=np.float64)
    backpointers: list[np.ndarray] = []
    for previous_key, current_key in zip(ordered, ordered[1:]):
        previous_group = groups[previous_key]
        current_group = groups[current_key]
        next_costs = np.full(len(current_group), np.inf, dtype=np.float64)
        pointers = np.zeros(len(current_group), dtype=np.int64)
        for current_offset, current_source in enumerate(current_group):
            transitions = np.asarray(
                [
                    costs[previous_offset]
                    + _transition_cost(
                        volume_zyx[previous_source],
                        volume_zyx[current_source],
                        1,
                    )
                    for previous_offset, previous_source in enumerate(
                        previous_group
                    )
                ],
                dtype=np.float64,
            )
            pointers[current_offset] = int(np.argmin(transitions))
            next_costs[current_offset] = transitions[pointers[current_offset]]
        backpointers.append(pointers)
        costs = next_costs
    offsets = [int(np.argmin(costs))]
    for pointers in reversed(backpointers):
        offsets.append(int(pointers[offsets[-1]]))
    offsets.reverse()
    selected_sources = np.asarray(
        [
            int(groups[coordinate_key][offset])
            for coordinate_key, offset in zip(ordered, offsets)
        ],
        dtype=np.int64,
    )
    selected_coordinates = coordinates[
        np.searchsorted(source_indices, selected_sources)
    ]
    order = np.argsort(selected_coordinates)
    return selected_sources[order], selected_coordinates[order]


def regularize_axial_stack(
    volume_zyx: np.ndarray,
    positions_lps_mm: np.ndarray,
    orientation_patient: np.ndarray,
    *,
    lattice_tolerance_mm: float = 0.1,
    max_missing_slices: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map an irregular DICOM stack onto its dominant physical slice lattice.

    Off-lattice insertions and disconnected edge fragments are discarded.
    Duplicate positions are resolved by the minimum adjacent-image-cost path.
    A small number of missing internal slices are linearly interpolated.
    """

    volume = np.asarray(volume_zyx, dtype=np.float32)
    positions = np.asarray(positions_lps_mm, dtype=np.float64)
    orientation = np.asarray(orientation_patient, dtype=np.float64)
    if volume.ndim != 3 or positions.shape != (volume.shape[0], 3):
        raise ValueError("DICOM volume and slice positions do not align")
    if orientation.shape != (6,):
        raise ValueError("DICOM ImageOrientationPatient must contain six values")
    if lattice_tolerance_mm <= 0 or max_missing_slices < 0:
        raise ValueError("Invalid DICOM regularization tolerances")

    row = _unit(orientation[:3], "row")
    column = _unit(orientation[3:], "column")
    normal = _unit(np.cross(row, column), "normal")
    if abs(float(np.dot(row, column))) > 1e-4:
        raise ValueError("DICOM row and column directions are not orthogonal")
    coordinates = positions @ normal
    spacing = _nominal_spacing(coordinates)
    sorted_coordinates = np.sort(coordinates)
    source_differences = np.diff(sorted_coordinates)
    source_residual = float(
        np.max(np.abs(np.abs(source_differences) - spacing))
    )
    anchor, _, lattice_residuals = _best_lattice_anchor(
        coordinates,
        spacing,
        lattice_tolerance_mm,
    )
    all_source_indices = np.arange(volume.shape[0], dtype=np.int64)
    component_sources = _largest_position_component(
        coordinates,
        all_source_indices,
        spacing,
        max_missing_slices,
    )
    if component_sources.size < 2:
        raise ValueError("Dominant DICOM position component has fewer than two slices")
    component_mask = np.isin(all_source_indices, component_sources)
    component_coordinates = coordinates[component_mask]
    component_sources = all_source_indices[component_mask]
    selected_sources, selected_coordinates = _select_duplicate_path(
        volume,
        component_coordinates,
        component_sources,
    )
    index_tolerance = lattice_tolerance_mm / spacing
    first_index = int(
        np.ceil(
            (selected_coordinates[0] - anchor) / spacing - index_tolerance
        )
    )
    last_index = int(
        np.floor(
            (selected_coordinates[-1] - anchor) / spacing + index_tolerance
        )
    )
    if last_index <= first_index:
        raise ValueError("Regular DICOM output grid has fewer than two slices")
    output_indices = np.arange(first_index, last_index + 1, dtype=np.int64)
    output_coordinates = anchor + output_indices * spacing
    output_count = len(output_coordinates)
    output_spacing = spacing
    output = np.empty(
        (output_count, *volume.shape[1:]),
        dtype=np.float32,
    )
    interpolated = 0
    for output_offset, coordinate in enumerate(output_coordinates):
        exact = np.where(
            np.isclose(selected_coordinates, coordinate, atol=1e-4)
        )[0]
        if exact.size:
            output[output_offset] = volume[selected_sources[int(exact[0])]]
            continue
        upper = int(np.searchsorted(selected_coordinates, coordinate))
        lower = upper - 1
        if lower < 0:
            output[output_offset] = volume[selected_sources[0]]
            interpolated += 1
            continue
        if upper >= len(selected_coordinates):
            output[output_offset] = volume[selected_sources[-1]]
            interpolated += 1
            continue
        fraction = (
            (coordinate - selected_coordinates[lower])
            / (selected_coordinates[upper] - selected_coordinates[lower])
        )
        output[output_offset] = (
            (1.0 - fraction) * volume[selected_sources[lower]]
            + fraction * volume[selected_sources[upper]]
        )
        interpolated += 1

    reference_source = int(selected_sources[0])
    first_coordinate = float(output_coordinates[0])
    origin = (
        positions[reference_source]
        + normal * (first_coordinate - coordinates[reference_source])
    )
    direction = np.column_stack((row, column, normal))
    retained_unique = len(selected_sources)
    off_lattice = lattice_residuals > lattice_tolerance_mm
    report: dict[str, Any] = {
        "applied": True,
        "method": (
            "dominant continuous physical component; minimum-adjacent-cost "
            "duplicate selection; physical-coordinate linear resampling"
        ),
        "source_slices": int(volume.shape[0]),
        "output_slices": int(output.shape[0]),
        "nominal_spacing_mm": spacing,
        "output_spacing_mm": output_spacing,
        "source_max_spacing_residual_mm": source_residual,
        "trimmed_start_extent_mm": float(
            output_coordinates[0] - selected_coordinates[0]
        ),
        "trimmed_end_extent_mm": float(
            selected_coordinates[-1] - output_coordinates[-1]
        ),
        "retained_lattice_max_residual_mm": float(
            lattice_residuals[component_sources].max(initial=0.0)
        ),
        "off_lattice_slices_resampled": int(
            np.logical_and(off_lattice, component_mask).sum()
        ),
        "discarded_off_lattice_slices": 0,
        "discarded_disconnected_slices": int(
            volume.shape[0] - component_mask.sum()
        ),
        "duplicate_slices_resolved": int(
            component_sources.size - retained_unique
        ),
        "interpolated_slices": interpolated,
        "post_regularization_spacing_residual_mm": 0.0,
    }
    return output, origin, direction, report
