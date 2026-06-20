from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import gmsh


"""DEFAULT_MESH_DEF_FILE = (
    Path(__file__).resolve().parent.parent / "V3" / "msh_def_FlapletV2_WKS.json"
)"""
DEFAULT_MESH_DEF_FILE = (
    Path(__file__).resolve().parent.parent / "V3" / "msh_def_original_WKS.json"
)

class CurveConstraint:
    def __init__(self, n_pts: int, mesh_type: str, coef: float) -> None:
        self.n_pts = n_pts
        self.mesh_type = mesh_type
        self.coef = coef


class MeshZoneError(RuntimeError):
    def __init__(self, zone_name: str, error: Exception) -> None:
        self.zone_name = zone_name
        super().__init__(f"Mesh zone {zone_name!r}: {error}")


@dataclass(frozen=True)
class CurveSpec:
    curve_ids: int | list[int]
    invert_directions: bool | list[bool]
    group_invert_direction: bool
    n_pts: int
    mesh_type: str
    coef: float


@dataclass(frozen=True)
class TracedLoop:
    curve_ids: list[int]
    points: list[int]
    blunt_curve_ids: list[int]
    blunt_points: list[int]


@dataclass(frozen=True)
class LongitudinalPath:
    points: list[int]
    curve_segments: list[list[int]]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"Expected boolean invert_direction entry, got {value!r}.")


def _as_list(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _as_bool_list(value: bool | list[bool], n_items: int) -> list[bool]:
    if isinstance(value, bool):
        return [value] * n_items

    result = [_as_bool(item) for item in value]
    if len(result) != n_items:
        raise ValueError(
            f"invert_direction group length {len(result)} does not match "
            f"curve group length {n_items}."
        )

    return result


def _is_list_value(value: Any) -> bool:
    return isinstance(value, list)


def _per_spec_value(
    entry: dict[str, Any],
    key: str,
    index: int,
    spec_count: int,
    default: Any = None,
) -> Any:
    if key not in entry:
        return default

    value = entry[key]
    if not _is_list_value(value):
        return value

    if len(value) != spec_count:
        raise ValueError(
            f"{key} length {len(value)} does not match curve/group count "
            f"{spec_count} in {entry.get('name', entry)!r}."
        )

    return value[index]


def _curve_spec_discretization(
    entry: dict[str, Any],
    index: int,
    spec_count: int,
    curve_ids: int | list[int],
) -> tuple[int, str, float]:
    mesh_type = str(_per_spec_value(entry, "type", index, spec_count, "Progression"))
    mesh_size_mode = str(
        _per_spec_value(entry, "mesh size mode", index, spec_count, "npts")
    ).lower()

    if mesh_size_mode in ("npts", "n_pts", "points"):
        n_pts = int(_per_spec_value(entry, "n_pts", index, spec_count))
        coef = float(_per_spec_value(entry, "Parameter", index, spec_count, 1.0))
    elif mesh_size_mode in ("ele size", "element size", "element_size"):
        curve_length = _curve_group_length(curve_ids)
        target_size_1 = float(
            _per_spec_value(entry, "target ele size 1", index, spec_count)
        )
        target_size_2 = float(
            _per_spec_value(entry, "target ele size 2", index, spec_count)
        )
        mesh_type_normalized = mesh_type.lower()
        if mesh_type_normalized == "progression":
            n_pts, coef = _progression_from_endpoint_sizes(
                curve_length, target_size_1, target_size_2
            )
        elif mesh_type_normalized == "bump":
            n_pts, coef = _bump_from_corner_middle_sizes(
                curve_length, target_size_1, target_size_2
            )
        else:
            raise ValueError(
                f"Element-size mode supports Progression and Bump, got "
                f"{mesh_type!r} in {entry.get('name', entry)!r}."
            )
    else:
        raise ValueError(
            f"Unsupported mesh size mode {mesh_size_mode!r} in "
            f"{entry.get('name', entry)!r}."
        )

    if n_pts < 2:
        raise ValueError(f"Transfinite curve point count must be >= 2: {entry}")

    return n_pts, mesh_type, coef


def _entry_mesh_size_mode(entry: dict[str, Any], index: int, spec_count: int) -> str:
    return str(
        _per_spec_value(entry, "mesh size mode", index, spec_count, "npts")
    ).lower()


def _uses_element_size_mode(entry: dict[str, Any], spec_count: int) -> bool:
    return any(
        _entry_mesh_size_mode(entry, index, spec_count)
        in ("ele size", "element size", "element_size")
        for index in range(spec_count)
    )


def _as_vector(value: Sequence[float]) -> list[float]:
    vector = [float(item) for item in value]
    if len(vector) != 3:
        raise ValueError(f"Expected a 3D vector, got {value!r}.")
    if math.sqrt(sum(component * component for component in vector)) <= 0.0:
        raise ValueError("Tracing vector must be non-zero.")
    return vector


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _normalized(vector: Sequence[float]) -> list[float]:
    norm = _vector_norm(vector)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return [component / norm for component in vector]


def _dot(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(vector_a, vector_b))


def _cross(vector_a: Sequence[float], vector_b: Sequence[float]) -> list[float]:
    return [
        vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
        vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0],
    ]


def _point_coord(point_id: int) -> list[float]:
    return [float(value) for value in gmsh.model.getValue(0, int(point_id), [0])]


def _point_vector(start_point: int, end_point: int) -> list[float]:
    start_coord = _point_coord(start_point)
    end_coord = _point_coord(end_point)
    return [end_coord[index] - start_coord[index] for index in range(3)]


def _other_curve_point(curve_id: int, point_id: int) -> int:
    endpoints = _curve_endpoints(curve_id)
    if point_id == endpoints[0]:
        return endpoints[1]
    if point_id == endpoints[1]:
        return endpoints[0]
    raise ValueError(f"Point {point_id} is not on curve {curve_id}.")


def _curve_length(curve_id: int) -> float:
    return gmsh.model.occ.getMass(1, abs(curve_id))


def _curve_group_length(curve_ids: int | list[int]) -> float:
    return sum(_curve_length(curve_id) for curve_id in _as_list(curve_ids))


def _progression_sum(first_size: float, ratio: float, n_elements: int) -> float:
    if n_elements <= 0:
        return 0.0
    if abs(ratio - 1.0) < 1e-14:
        return n_elements * first_size
    return first_size * (ratio**n_elements - 1.0) / (ratio - 1.0)


def _progression_length_error(
    curve_length: float,
    target_size_1: float,
    target_size_2: float,
    n_elements: int,
) -> float:
    if n_elements == 1:
        estimated_length = 0.5 * (target_size_1 + target_size_2)
    else:
        ratio = (target_size_2 / target_size_1) ** (1.0 / (n_elements - 1))
        estimated_length = _progression_sum(target_size_1, ratio, n_elements)
    return abs(estimated_length - curve_length)


def _progression_from_endpoint_sizes(
    curve_length: float,
    target_size_1: float,
    target_size_2: float,
) -> tuple[int, float]:
    if curve_length <= 0.0:
        raise ValueError(f"Curve length must be positive, got {curve_length}.")
    if target_size_1 <= 0.0 or target_size_2 <= 0.0:
        raise ValueError(
            f"Target element sizes must be positive, got "
            f"{target_size_1} and {target_size_2}."
        )

    if abs(target_size_1 - target_size_2) < 1e-14:
        n_elements = max(1, round(curve_length / target_size_1))
        return n_elements + 1, 1.0

    upper = 2
    while (
        _progression_length_error(
            curve_length, target_size_1, target_size_2, upper
        )
        > _progression_length_error(
            curve_length, target_size_1, target_size_2, upper + 1
        )
        and upper < 100000
    ):
        upper *= 2

    upper = min(max(upper * 2, 4), 100000)
    n_elements = min(
        range(1, upper + 1),
        key=lambda candidate: _progression_length_error(
            curve_length, target_size_1, target_size_2, candidate
        ),
    )

    if n_elements == 1:
        return 2, 1.0

    coef = (target_size_2 / target_size_1) ** (1.0 / (n_elements - 1))
    return n_elements + 1, coef


def _progression_node_positions(n_pts: int, coef: float, invert: bool) -> list[float]:
    n_elements = n_pts - 1
    ratio = 1.0 / coef if invert else coef

    if ratio <= 0.0:
        raise ValueError(f"Progression coefficient must be positive, got {coef}.")

    if abs(ratio - 1.0) < 1e-14:
        return [i / n_elements for i in range(n_pts)]

    denominator = ratio**n_elements - 1.0
    return [(ratio**i - 1.0) / denominator for i in range(n_pts)]


def _bump_node_positions(n_pts: int, coef: float) -> list[float]:
    n_elements = n_pts - 1

    if abs(coef - 1.0) < 1e-14:
        return [i / n_elements for i in range(n_pts)]

    if coef > 1.0:
        sqrt_a = math.sqrt(coef - 1.0)
        angle = math.atan(sqrt_a)
        return [
            0.5 * (1.0 - math.tan(angle * (1.0 - 2.0 * i / n_elements)) / sqrt_a)
            for i in range(n_pts)
        ]

    if coef > 0.0:
        sqrt_a = math.sqrt(1.0 - coef)
        angle = math.atanh(sqrt_a)
        return [
            0.5
            * (1.0 - math.tanh(angle * (1.0 - 2.0 * i / n_elements)) / sqrt_a)
            for i in range(n_pts)
        ]

    raise ValueError(f"Bump coefficient must be positive, got {coef}.")


def _bump_corner_middle_sizes(
    curve_length: float,
    n_elements: int,
    coef: float,
) -> tuple[float, float]:
    if abs(coef - 1.0) < 1e-14:
        return curve_length / n_elements, curve_length / n_elements

    if coef > 1.0:
        sqrt_a = math.sqrt(coef - 1.0)
        angle = math.atan(sqrt_a)

        def node_position(index: int) -> float:
            return 0.5 * (
                1.0 - math.tan(angle * (1.0 - 2.0 * index / n_elements)) / sqrt_a
            )
    elif coef > 0.0:
        sqrt_a = math.sqrt(1.0 - coef)
        angle = math.atanh(sqrt_a)

        def node_position(index: int) -> float:
            return 0.5 * (
                1.0 - math.tanh(angle * (1.0 - 2.0 * index / n_elements)) / sqrt_a
            )
    else:
        raise ValueError(f"Bump coefficient must be positive, got {coef}.")

    def element_size(index: int) -> float:
        return curve_length * (node_position(index + 1) - node_position(index))

    first_size = element_size(0)
    last_size = element_size(n_elements - 1)
    corner_size = 0.5 * (first_size + last_size)

    middle_index = n_elements // 2
    if n_elements % 2 == 0:
        middle_size = 0.5 * (
            element_size(middle_index - 1)
            + element_size(middle_index)
        )
    else:
        middle_size = element_size(middle_index)

    return corner_size, middle_size


def _bump_size_error(
    curve_length: float,
    target_corner_size: float,
    target_middle_size: float,
    n_elements: int,
    log_coef: float,
) -> float:
    corner_size, middle_size = _bump_corner_middle_sizes(
        curve_length, n_elements, math.exp(log_coef)
    )
    corner_error = math.log(corner_size / target_corner_size)
    middle_error = math.log(middle_size / target_middle_size)
    return corner_error * corner_error + middle_error * middle_error


def _best_bump_coef_for_element_count(
    curve_length: float,
    target_corner_size: float,
    target_middle_size: float,
    n_elements: int,
) -> tuple[float, float]:
    left = -30.0
    right = 30.0
    for _ in range(80):
        left_mid = left + (right - left) / 3.0
        right_mid = right - (right - left) / 3.0
        if _bump_size_error(
            curve_length,
            target_corner_size,
            target_middle_size,
            n_elements,
            left_mid,
        ) < _bump_size_error(
            curve_length,
            target_corner_size,
            target_middle_size,
            n_elements,
            right_mid,
        ):
            right = right_mid
        else:
            left = left_mid

    log_coef = 0.5 * (left + right)
    return (
        _bump_size_error(
            curve_length,
            target_corner_size,
            target_middle_size,
            n_elements,
            log_coef,
        ),
        math.exp(log_coef),
    )


def _candidate_bump_element_counts(
    lower: int,
    upper: int,
    curve_length: float,
    target_corner_size: float,
    target_middle_size: float,
) -> list[int]:
    if upper - lower <= 500:
        return list(range(lower, upper + 1))

    candidates = {
        lower,
        upper,
        round(curve_length / target_corner_size),
        round(curve_length / target_middle_size),
        round(curve_length / math.sqrt(target_corner_size * target_middle_size)),
        round(curve_length / (0.5 * (target_corner_size + target_middle_size))),
    }
    step = (upper - lower) / 500.0
    candidates.update(round(lower + step * index) for index in range(501))
    return sorted(
        max(lower, min(upper, int(candidate)))
        for candidate in candidates
        if int(candidate) >= 1
    )


def _bump_from_corner_middle_sizes(
    curve_length: float,
    target_corner_size: float,
    target_middle_size: float,
) -> tuple[int, float]:
    if curve_length <= 0.0:
        raise ValueError(f"Curve length must be positive, got {curve_length}.")
    if target_corner_size <= 0.0 or target_middle_size <= 0.0:
        raise ValueError(
            f"Target element sizes must be positive, got "
            f"{target_corner_size} and {target_middle_size}."
        )

    if abs(target_corner_size - target_middle_size) < 1e-14:
        n_elements = max(1, round(curve_length / target_corner_size))
        return n_elements + 1, 1.0

    lower = max(1, math.floor(curve_length / max(target_corner_size, target_middle_size)))
    upper = max(lower, math.ceil(curve_length / min(target_corner_size, target_middle_size)))
    candidates = _candidate_bump_element_counts(
        lower,
        upper,
        curve_length,
        target_corner_size,
        target_middle_size,
    )
    best_n_elements, (best_error, best_coef) = min(
        (
            (
                n_elements,
                _best_bump_coef_for_element_count(
                    curve_length,
                    target_corner_size,
                    target_middle_size,
                    n_elements,
                ),
            )
            for n_elements in candidates
        ),
        key=lambda item: item[1][0],
    )

    refinement_radius = max(10, math.ceil((upper - lower) / 500))
    refined_lower = max(lower, best_n_elements - refinement_radius)
    refined_upper = min(upper, best_n_elements + refinement_radius)
    best_n_elements, (best_error, best_coef) = min(
        (
            (
                n_elements,
                _best_bump_coef_for_element_count(
                    curve_length,
                    target_corner_size,
                    target_middle_size,
                    n_elements,
                ),
            )
            for n_elements in range(refined_lower, refined_upper + 1)
        ),
        key=lambda item: item[1][0],
    )

    return best_n_elements + 1, best_coef


def _distribution_node_positions(
    entry: dict[str, Any],
    n_pts: int,
    invert_group: bool,
    mesh_type: str | None = None,
    coef: float | None = None,
) -> list[float]:
    mesh_type = str(
        mesh_type if mesh_type is not None else entry.get("type", "Progression")
    ).lower()
    coef = float(coef if coef is not None else entry.get("Parameter", 1.0))

    if mesh_type == "progression":
        return _progression_node_positions(n_pts, coef, invert_group)
    if mesh_type == "bump":
        return _bump_node_positions(n_pts, coef)

    raise ValueError(
        f"Arclength-based split curves currently support Progression and Bump, "
        f"got {entry.get('type')!r}."
    )


def _nearest_node_index(node_positions: list[float], target_position: float) -> int:
    return min(
        range(1, len(node_positions) - 1),
        key=lambda index: abs(node_positions[index] - target_position),
    )


def _greedy_compound_split_indices(
    lengths: list[float],
    node_positions: list[float],
) -> list[int]:
    total_length = sum(lengths)
    n_elements = len(node_positions) - 1
    split_indices = [0]
    accumulated_length = 0.0

    for i, length in enumerate(lengths[:-1]):
        accumulated_length += length
        target_position = accumulated_length / total_length
        nearest_index = _nearest_node_index(node_positions, target_position)

        min_index = split_indices[-1] + 1
        max_index = n_elements - (len(lengths) - i - 1)
        split_indices.append(min(max(nearest_index, min_index), max_index))

    split_indices.append(n_elements)
    return split_indices


def _compound_split_indices(
    lengths: list[float],
    node_positions: list[float],
) -> list[int]:
    n_elements = len(node_positions) - 1
    n_curves = len(lengths)
    if n_curves == 1:
        return [0, n_elements]

    # The dynamic program avoids accumulated rounding errors on compound edges
    # with many short CAD subcurves. Fall back to the old greedy split for very
    # large cases where the quadratic search would be disproportionate.
    if n_curves * n_elements * n_elements > 20_000_000:
        return _greedy_compound_split_indices(lengths, node_positions)

    total_length = sum(lengths)
    epsilon = 1.0e-300
    best_by_end_index: dict[int, tuple[float, list[int]]] = {0: (0.0, [0])}

    for curve_index, curve_length in enumerate(lengths):
        remaining_curves = n_curves - curve_index - 1
        next_best_by_end_index: dict[int, tuple[float, list[int]]] = {}

        for start_index, (current_cost, current_path) in best_by_end_index.items():
            min_end_index = start_index + 1
            max_end_index = n_elements - remaining_curves

            for end_index in range(min_end_index, max_end_index + 1):
                represented_length = total_length * (
                    node_positions[end_index] - node_positions[start_index]
                )
                split_cost = math.log(
                    max(represented_length, epsilon)
                    / max(curve_length, epsilon)
                ) ** 2
                candidate_cost = current_cost + split_cost

                best_candidate = next_best_by_end_index.get(end_index)
                if best_candidate is None or candidate_cost < best_candidate[0]:
                    next_best_by_end_index[end_index] = (
                        candidate_cost,
                        current_path + [end_index],
                    )

        best_by_end_index = next_best_by_end_index

    if n_elements not in best_by_end_index:
        return _greedy_compound_split_indices(lengths, node_positions)

    return best_by_end_index[n_elements][1]


def _curve_node_counts(
    curve_ids: int | list[int],
    invert_directions: bool | list[bool],
    group_invert_direction: bool,
    n_pts: int,
    entry: dict[str, Any],
    mesh_type: str | None = None,
    coef: float | None = None,
) -> list[tuple[int, int, bool, float]]:
    curve_group = _as_list(curve_ids)
    invert_group = _as_bool_list(invert_directions, len(curve_group))
    mesh_type = str(
        mesh_type if mesh_type is not None else entry.get("type", "Progression")
    )
    coef = float(coef if coef is not None else entry.get("Parameter", 1.0))

    if len(curve_group) == 1:
        curve_coef = coef
        if mesh_type.lower() == "progression" and group_invert_direction:
            if coef == 0.0:
                raise ValueError("Progression coefficient cannot be zero.")
            curve_coef = 1.0 / coef
        return [(curve_group[0], n_pts, invert_group[0], curve_coef)]

    n_elements = n_pts - 1
    if n_elements < len(curve_group):
        raise ValueError(
            f"Need at least {len(curve_group) + 1} points for {len(curve_group)} "
            f"split curves, got {n_pts}."
        )

    lengths = [_curve_length(curve_id) for curve_id in curve_group]
    total_length = sum(lengths)
    if total_length <= 0.0:
        raise ValueError(f"Split curve group has non-positive total length: {curve_group}")

    # The virtual grouped curve distribution is defined in traced curve order.
    # Geometry orientation is handled per CAD subcurve when setting Gmsh's local
    # progression coefficient, not when choosing where to split the compound.
    node_positions = _distribution_node_positions(
        entry, n_pts, group_invert_direction, mesh_type, coef
    )
    split_indices = _compound_split_indices(lengths, node_positions)
    element_counts = [
        end_index - start_index
        for start_index, end_index in zip(split_indices, split_indices[1:])
    ]

    curve_coefs = []
    for start_index, end_index in zip(split_indices, split_indices[1:]):
        curve_elements = end_index - start_index
        if mesh_type.lower() != "progression" or curve_elements <= 1:
            curve_coefs.append(coef)
            continue

        first_size = node_positions[start_index + 1] - node_positions[start_index]
        last_size = node_positions[end_index] - node_positions[end_index - 1]
        if first_size <= 0.0 or last_size <= 0.0:
            curve_coefs.append(coef)
            continue

        curve_coefs.append((last_size / first_size) ** (1.0 / (curve_elements - 1)))

    return [
        (curve_id, n_elements + 1, invert, curve_coef)
        for curve_id, n_elements, invert, curve_coef in zip(
            curve_group, element_counts, invert_group, curve_coefs
        )
    ]


def _iter_curve_entries(
    section: dict[str, Any] | list[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    if isinstance(section, list):
        yield from section
        return

    for key in ("rows", "columns"):
        yield from section.get(key, [])


def _iter_surface_entries(
    section: dict[str, Any] | list[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    if isinstance(section, list):
        yield from section
        return

    yield from section.get("surfaces", [])


def _has_trace_definition(entry: dict[str, Any]) -> bool:
    trace_keys = {"start_pt", "v_pointing", "n_subcurvs"}
    trigger_keys = {"start_pt", "v_pointing"}
    present_keys = trace_keys.intersection(entry)

    if not trigger_keys.intersection(entry):
        return False

    if present_keys and present_keys != trace_keys:
        missing_keys = sorted(trace_keys - present_keys)
        raise ValueError(
            f"Incomplete traced curve definition in {entry.get('name', entry)!r}; "
            f"missing {missing_keys}."
        )

    return bool(present_keys)


def _oriented_curve_endpoints(curve_id: int) -> tuple[int, int]:
    boundary = gmsh.model.getBoundary([(1, abs(curve_id))], oriented=True)
    points = [abs(tag) for dim, tag in boundary if dim == 0]
    if len(points) != 2:
        raise ValueError(f"Curve {curve_id} does not have exactly two endpoints.")
    return points[0], points[1]


def _curve_tangent_at_point(curve_id: int, point_id: int) -> list[float] | None:
    try:
        point_coord = gmsh.model.getValue(0, int(point_id), [0])
        curve_parameter = gmsh.model.getParametrization(
            1, abs(int(curve_id)), point_coord
        )
        derivative = gmsh.model.getDerivative(1, abs(int(curve_id)), curve_parameter)
    except Exception:
        return None

    tangent = [float(value) for value in derivative[:3]]
    if _vector_norm(tangent) <= 0.0:
        return None
    return tangent


def _curve_traversal_invert_direction(
    curve_id: int,
    start_point: int,
    end_point: int,
) -> bool:
    traversal_vector = _point_vector(start_point, end_point)
    tangent = _curve_tangent_at_point(curve_id, start_point)
    if tangent is not None:
        alignment = _dot(tangent, traversal_vector)
        if abs(alignment) > 1.0e-12 * _vector_norm(tangent) * _vector_norm(
            traversal_vector
        ):
            return alignment < 0.0

    natural_start, natural_end = _oriented_curve_endpoints(curve_id)
    if natural_start == start_point and natural_end == end_point:
        return False
    if natural_start == end_point and natural_end == start_point:
        return True

    natural_vector = _point_vector(natural_start, natural_end)
    return _dot(natural_vector, traversal_vector) < 0.0


def _curve_tangent_for_traversal(
    curve_id: int,
    point_id: int,
    previous_point_id: int | None = None,
) -> list[float] | None:
    tangent = _curve_tangent_at_point(curve_id, point_id)
    if tangent is None:
        return None

    if previous_point_id is not None:
        traversal_vector = _point_vector(previous_point_id, point_id)
        if _dot(tangent, traversal_vector) < 0.0:
            tangent = [-value for value in tangent]

    return tangent


def _curve_tangent_towards_point(
    curve_id: int,
    point_id: int,
    next_point_id: int,
) -> list[float] | None:
    tangent = _curve_tangent_at_point(curve_id, point_id)
    if tangent is None:
        return None

    traversal_vector = _point_vector(point_id, next_point_id)
    if _dot(tangent, traversal_vector) < 0.0:
        tangent = [-value for value in tangent]
    return tangent


def _best_adjacent_tangent(
    point_id: int,
    direction: Sequence[float],
    excluded_curves: set[int],
) -> list[float] | None:
    direction_unit = _normalized(direction)
    best_curve = None
    best_next_point = None
    best_score = 0.0

    for curve_id in _adjacent_curve_ids(point_id):
        if abs(curve_id) in excluded_curves:
            continue

        next_point = _other_curve_point(curve_id, point_id)
        curve_vector = _point_vector(point_id, next_point)
        curve_norm = _vector_norm(curve_vector)
        if curve_norm <= 0.0:
            continue

        score = abs(_dot(_normalized(curve_vector), direction_unit))
        if score > best_score:
            best_curve = curve_id
            best_next_point = next_point
            best_score = score

    if best_curve is None or best_next_point is None:
        return None
    return _curve_tangent_towards_point(best_curve, point_id, best_next_point)


def _local_blunt_direction(
    start_point: int,
    first_curve_id: int,
    first_curve_end_point: int,
    longitudinal_direction: Sequence[float],
    excluded_curves: set[int],
) -> list[float]:
    first_tangent = _curve_tangent_towards_point(
        first_curve_id,
        start_point,
        first_curve_end_point,
    )
    if first_tangent is None:
        first_tangent = _point_vector(start_point, first_curve_end_point)

    longitudinal_tangent = _best_adjacent_tangent(
        start_point,
        longitudinal_direction,
        set(excluded_curves).union({abs(first_curve_id)}),
    )
    if longitudinal_tangent is None:
        longitudinal_tangent = _as_vector(longitudinal_direction)

    blunt_direction = _cross(longitudinal_tangent, first_tangent)
    if _vector_norm(blunt_direction) <= 0.0:
        raise ValueError(
            "Cannot infer blunt trailing-edge direction from parallel circumferential "
            f"and longitudinal tangents at point {start_point}."
        )
    return blunt_direction


def _trace_surface_edge_curves(
    start_pt: int,
    v_pointing: Sequence[float],
) -> list[tuple[int, bool]]:
    current_pt = int(start_pt)
    current_coord = gmsh.model.getValue(0, current_pt, [0])
    pointing = _as_vector(v_pointing)
    traced_curves = []
    visited_curves = set()

    while True:
        adjacent_curves, _ = gmsh.model.getAdjacencies(0, current_pt)
        best_curve = None
        best_next_pt = None
        best_vector = None
        best_dot = 0.0

        for adjacent_curve in adjacent_curves:
            curve_id = int(adjacent_curve)
            if abs(curve_id) in visited_curves:
                continue

            endpoints = _curve_endpoints(curve_id)
            if current_pt not in endpoints:
                continue

            next_pt = endpoints[1] if endpoints[0] == current_pt else endpoints[0]
            next_coord = gmsh.model.getValue(0, next_pt, [0])
            adjacent_vector = [
                next_coord[index] - current_coord[index] for index in range(3)
            ]
            dot_product = sum(
                adjacent_vector[index] * pointing[index] for index in range(3)
            )

            if dot_product > best_dot:
                best_curve = curve_id
                best_next_pt = next_pt
                best_vector = adjacent_vector
                best_dot = dot_product

        if best_curve is None or best_next_pt is None or best_vector is None:
            break

        invert_direction = _curve_traversal_invert_direction(
            best_curve, current_pt, best_next_pt
        )
        traced_curves.append((best_curve, invert_direction))
        visited_curves.add(abs(best_curve))
        current_pt = best_next_pt
        current_coord = gmsh.model.getValue(0, current_pt, [0])
        pointing = best_vector

    return traced_curves


def _adjacent_curve_ids(point_id: int) -> list[int]:
    adjacent_curves, _ = gmsh.model.getAdjacencies(0, int(point_id))
    return [int(curve_id) for curve_id in adjacent_curves]


def _select_adjacent_curve(
    point_id: int,
    direction: Sequence[float],
    excluded_curves: set[int],
    *,
    min_score: float = 0.0,
    reject_direction: Sequence[float] | None = None,
    reject_alignment: float = 0.5,
) -> tuple[int, int, list[float]] | None:
    direction = _normalized(direction)
    best_curve = None
    best_next_point = None
    best_vector = None
    best_score = min_score
    reject_unit = (
        _normalized(reject_direction) if reject_direction is not None else None
    )

    for curve_id in _adjacent_curve_ids(point_id):
        if abs(curve_id) in excluded_curves:
            continue

        next_point = _other_curve_point(curve_id, point_id)
        curve_vector = _point_vector(point_id, next_point)
        curve_norm = _vector_norm(curve_vector)
        if curve_norm <= 0.0:
            continue

        candidate_vector = _curve_tangent_towards_point(curve_id, point_id, next_point)
        if candidate_vector is None or _vector_norm(candidate_vector) <= 0.0:
            candidate_vector = curve_vector

        candidate_unit = _normalized(candidate_vector)
        if (
            reject_unit is not None
            and abs(_dot(candidate_unit, reject_unit)) > reject_alignment
        ):
            continue

        score = _dot(candidate_unit, direction)
        if score > best_score:
            best_curve = curve_id
            best_next_point = next_point
            best_vector = curve_vector
            best_score = score

    if best_curve is None or best_next_point is None or best_vector is None:
        return None

    return best_curve, best_next_point, best_vector


def _connecting_curve(point_a: int, point_b: int) -> int | None:
    adjacent_a = {abs(curve_id) for curve_id in _adjacent_curve_ids(point_a)}
    for curve_id in _adjacent_curve_ids(point_b):
        if abs(curve_id) in adjacent_a:
            return abs(curve_id)
    return None


def _curve_path_alignment(
    curve_ids: Sequence[int],
    point_path: Sequence[int],
    direction: Sequence[float],
) -> float:
    if not curve_ids or len(point_path) < 2:
        return -1.0

    direction_unit = _normalized(direction)
    length_sum = 0.0
    weighted_alignment = 0.0
    for curve_index, curve_id in enumerate(curve_ids):
        start_point = point_path[curve_index]
        end_point = point_path[curve_index + 1]
        curve_vector = _point_vector(start_point, end_point)
        curve_norm = _vector_norm(curve_vector)
        if curve_norm <= 0.0:
            continue
        curve_length = _curve_length(curve_id)
        length_sum += curve_length
        weighted_alignment += curve_length * abs(
            _dot(_normalized(curve_vector), direction_unit)
        )

    if length_sum <= 0.0:
        return -1.0
    return weighted_alignment / length_sum


def _has_only_simple_intermediate_points(point_path: Sequence[int]) -> bool:
    return all(_point_valence(point_id) <= 2 for point_id in point_path[1:-1])


def _known_intermediate_points(
    point_path: Sequence[int],
    known_points: set[int],
) -> list[int]:
    return [point_id for point_id in point_path[1:-1] if point_id in known_points]


def _trace_circumferential_loop(
    start_point: int,
    circumferential_direction: Sequence[float],
    has_blunt_te: bool,
    longitudinal_direction: Sequence[float],
    excluded_curves: set[int],
    expected_blunt_curve_count: int | None = None,
    expected_total_curve_count: int | None = None,
    closed_circumferential_loop: bool = True,
) -> TracedLoop:
    current_point = int(start_point)
    pointing = _as_vector(circumferential_direction)
    effective_blunt_direction = None
    curve_ids = []
    points = [current_point]
    local_excluded = set(excluded_curves)
    expected_body_curve_count = None
    if expected_total_curve_count is not None:
        expected_body_curve_count = int(expected_total_curve_count)
        if closed_circumferential_loop and has_blunt_te:
            expected_body_curve_count -= int(expected_blunt_curve_count or 0)
        if expected_body_curve_count < 1:
            raise ValueError(
                "Expected circumferential loop curve count must leave at least one "
                f"non-blunt curve, got {expected_total_curve_count}."
            )

    while True:
        if (
            expected_body_curve_count is not None
            and len(curve_ids) >= expected_body_curve_count
        ):
            if not closed_circumferential_loop or not has_blunt_te:
                return TracedLoop(curve_ids, points, [], [])

            blunt_curves, blunt_points = _curve_path_between_points(
                current_point,
                start_point,
                local_excluded,
            )
            if (
                expected_blunt_curve_count is not None
                and len(blunt_curves) != expected_blunt_curve_count
            ):
                raise ValueError(
                    f"Expected {expected_blunt_curve_count} blunt trailing-edge "
                    f"curves from point {current_point} to {start_point}, got "
                    f"{len(blunt_curves)}: {blunt_curves}."
                )
            if (
                effective_blunt_direction is not None
                and _curve_path_alignment(
                    blunt_curves,
                    blunt_points,
                    effective_blunt_direction,
                )
                <= 0.2
            ):
                raise ValueError(
                    f"Closing curve path from {current_point} to {start_point} "
                    f"is not aligned with the local blunt trailing-edge direction: "
                    f"{blunt_curves}."
                )
            return TracedLoop(curve_ids, points, blunt_curves, blunt_points)

        if curve_ids and expected_body_curve_count is None:
            blunt_curve = _connecting_curve(current_point, start_point)
            if blunt_curve is not None and blunt_curve not in local_excluded:
                if not has_blunt_te:
                    return TracedLoop(
                        curve_ids + [blunt_curve],
                        points + [start_point],
                        [],
                        [],
                    )
                if expected_blunt_curve_count in (None, 1):
                    blunt_vector = _point_vector(current_point, start_point)
                    if (
                        abs(
                            _dot(
                                _normalized(blunt_vector),
                                _normalized(effective_blunt_direction),
                            )
                        )
                        > 0.2
                    ):
                        return TracedLoop(
                            curve_ids,
                            points,
                            [blunt_curve],
                            [current_point, start_point],
                        )

            if has_blunt_te:
                try:
                    blunt_curves, blunt_points = _curve_path_between_points(
                        current_point,
                        start_point,
                        local_excluded,
                    )
                except ValueError:
                    blunt_curves = []
                    blunt_points = []
                if len(blunt_curves) > 1 and (
                    (
                        _has_only_simple_intermediate_points(blunt_points)
                        or len(blunt_curves) == expected_blunt_curve_count
                    )
                    and
                    _curve_path_alignment(
                        blunt_curves,
                        blunt_points,
                        effective_blunt_direction,
                    )
                    > 0.2
                ):
                    return TracedLoop(curve_ids, points, blunt_curves, blunt_points)
            else:
                try:
                    closing_curves, closing_points = _curve_path_between_points(
                        current_point,
                        start_point,
                        local_excluded,
                    )
                except ValueError:
                    closing_curves = []
                    closing_points = []
                if len(closing_curves) > 1 and _has_only_simple_intermediate_points(
                    closing_points
                ):
                    return TracedLoop(
                        curve_ids + closing_curves,
                        points + closing_points[1:],
                        [],
                        [],
                    )

        selected = _select_adjacent_curve(
            current_point,
            pointing,
            local_excluded,
            min_score=-1.0,
            reject_direction=(
                None if expected_body_curve_count is not None else longitudinal_direction
            ),
            reject_alignment=0.85,
        )
        if selected is None:
            raise ValueError(
                f"Could not continue circumferential loop trace from point {current_point}."
            )

        curve_id, next_point, curve_vector = selected
        curve_ids.append(curve_id)
        points.append(next_point)
        local_excluded.add(abs(curve_id))
        if len(curve_ids) == 1 and has_blunt_te:
            effective_blunt_direction = _local_blunt_direction(
                start_point,
                curve_id,
                next_point,
                longitudinal_direction,
                local_excluded,
            )
        tangent = _curve_tangent_for_traversal(curve_id, next_point, current_point)
        current_point = next_point
        pointing = tangent if tangent is not None else curve_vector


def _curve_group_size(curve_group: int | list[int]) -> int:
    if isinstance(curve_group, int):
        return 1
    return len(curve_group)


def _group_traced_curves(
    traced_curve_ids: list[int],
    template_entry: dict[str, Any],
    inferred_n_subcurvs: list[int] | None = None,
) -> list[int | list[int]]:
    if inferred_n_subcurvs is not None:
        group_sizes = inferred_n_subcurvs
    elif "n_subcurvs" in template_entry:
        group_sizes = [int(group_size) for group_size in template_entry["n_subcurvs"]]
    elif "curve_ids" in template_entry:
        group_sizes = [
            _curve_group_size(curve_group)
            for curve_group in template_entry["curve_ids"]
        ]
    else:
        group_sizes = [1] * len(traced_curve_ids)

    expected_curve_count = sum(group_sizes)
    if expected_curve_count != len(traced_curve_ids):
        raise ValueError(
            f"Automatic trace for {template_entry.get('name', template_entry)!r} "
            f"found {len(traced_curve_ids)} curves, but the template groups "
            f"require {expected_curve_count}."
        )

    grouped_curves = []
    index = 0
    for group_size in group_sizes:
        group = traced_curve_ids[index : index + group_size]
        index += group_size
        grouped_curves.append(group[0] if group_size == 1 else group)
    return grouped_curves


def _group_values_by_curve_groups(
    values: list[Any],
    curve_groups: list[int | list[int]],
) -> list[Any]:
    grouped_values = []
    index = 0
    for curve_group in curve_groups:
        group_size = _curve_group_size(curve_group)
        group_values = values[index : index + group_size]
        index += group_size
        grouped_values.append(group_values[0] if group_size == 1 else group_values)
    return grouped_values


def _grouped_template_invert_directions(
    template_entry: dict[str, Any],
    group_sizes: list[int],
    total_curves: int,
) -> list[bool | list[bool]] | None:
    template_key = (
        "invert_direction"
        if "invert_direction" in template_entry
        else "_invert_direction"
    )
    if template_key not in template_entry:
        return None

    value = template_entry[template_key]
    if isinstance(value, list) and len(value) == len(group_sizes):
        return copy.deepcopy(value)

    template_group_sizes = _template_group_sizes(template_entry, total_curves)
    expanded_values = _expand_template_values(
        template_entry,
        template_key,
        template_group_sizes,
        total_curves,
    )
    if expanded_values is None:
        return None
    if len(expanded_values) < total_curves:
        expanded_values.extend(
            copy.deepcopy(expanded_values[-1])
            for _ in range(total_curves - len(expanded_values))
        )

    return _regroup_template_values(
        expanded_values,
        group_sizes,
        keep_group_lists=True,
    )


def _group_invert_direction(value: bool | list[bool] | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value

    group = [_as_bool(item) for item in value]
    if not group:
        return False
    if len(set(group)) == 1:
        return group[0]
    return group[0]


def _combine_grouped_invert_directions(
    inferred_directions: list[bool | list[bool]],
    user_directions: list[bool | list[bool]] | None,
    curve_groups: list[int | list[int]],
) -> list[bool | list[bool]]:
    if user_directions is None:
        return inferred_directions

    combined_directions = []
    for curve_group, inferred_direction, user_direction in zip(
        curve_groups,
        inferred_directions,
        user_directions,
    ):
        group_size = _curve_group_size(curve_group)
        inferred_group = _as_bool_list(inferred_direction, group_size)
        user_group = _as_bool_list(user_direction, group_size)
        combined_group = [
            bool(inferred) != bool(user_invert)
            for inferred, user_invert in zip(inferred_group, user_group)
        ]
        combined_directions.append(
            combined_group[0] if group_size == 1 else combined_group
        )

    return combined_directions


def _infer_n_subcurvs(
    traced_loop: TracedLoop,
    split_longitudinal_curve_count: int,
) -> list[int]:
    loop_curve_set = {abs(curve_id) for curve_id in _loop_all_curve_ids(traced_loop)}
    group_sizes = []
    current_group_size = 1

    for curve_index, point_id in enumerate(traced_loop.points[1:-1], start=1):
        adjacent_non_loop_curves = [
            curve_id
            for curve_id in _adjacent_curve_ids(point_id)
            if abs(curve_id) not in loop_curve_set
        ]

        if len(adjacent_non_loop_curves) >= split_longitudinal_curve_count:
            group_sizes.append(current_group_size)
            current_group_size = 1
        else:
            current_group_size += 1

    group_sizes.append(current_group_size)
    group_sizes.append(len(traced_loop.blunt_curve_ids))
    return group_sizes


def _template_group_count(template_entry: dict[str, Any]) -> int | None:
    for key in (
        "n_subcurvs_per_compound_curve",
        "n_subcurvs",
        "n_pts",
        "type",
        "mesh size mode",
        "target ele size 1",
        "target ele size 2",
        "Parameter",
        "invert_direction",
        "_invert_direction",
    ):
        value = template_entry.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _infer_n_subcurvs_for_template(
    traced_loop: TracedLoop,
    template_entry: dict[str, Any],
    default_split_longitudinal_curve_count: int,
    previous_group_sizes: list[int] | None = None,
) -> list[int]:
    target_group_count = _template_group_count(template_entry)
    candidate_thresholds = [default_split_longitudinal_curve_count, 1, 2, 3]
    candidate_group_sizes = []

    for threshold in candidate_thresholds:
        if threshold < 1:
            continue
        group_sizes = _infer_n_subcurvs(traced_loop, threshold)
        if group_sizes not in candidate_group_sizes:
            candidate_group_sizes.append(group_sizes)
        if target_group_count is not None and len(group_sizes) == target_group_count:
            return group_sizes

    if (
        target_group_count is not None
        and previous_group_sizes is not None
        and len(previous_group_sizes) == target_group_count
        and sum(previous_group_sizes)
        == len(traced_loop.curve_ids) + 1
    ):
        return previous_group_sizes

    if target_group_count is not None:
        best_group_sizes = min(
            candidate_group_sizes,
            key=lambda group_sizes: abs(len(group_sizes) - target_group_count),
        )
        return best_group_sizes

    return candidate_group_sizes[0]


def _unique_ordered(values: Iterable[int]) -> list[int]:
    unique_values = []
    seen_values = set()
    for value in values:
        int_value = int(value)
        if int_value in seen_values:
            continue
        unique_values.append(int_value)
        seen_values.add(int_value)
    return unique_values


def _point_valence(point_id: int) -> int:
    return len(_adjacent_curve_ids(point_id))


def _loop_ordered_points(traced_loop: TracedLoop) -> list[int]:
    return _unique_ordered(traced_loop.points)


def _loop_all_curve_ids(traced_loop: TracedLoop) -> list[int]:
    return traced_loop.curve_ids + traced_loop.blunt_curve_ids


def _format_traced_loop_curve_ids(traced_loop: TracedLoop) -> str:
    return (
        f"curve_ids={traced_loop.curve_ids}, "
        f"blunt_curve_ids={traced_loop.blunt_curve_ids}, "
        f"all_curve_ids={_loop_all_curve_ids(traced_loop)}"
    )


def _format_compound_curve_groups(
    traced_loop: TracedLoop,
    group_sizes: list[int],
) -> str:
    all_curve_ids = _loop_all_curve_ids(traced_loop)
    grouped_curve_ids: list[Any] = []
    curve_index = 0
    for group_size in group_sizes:
        if group_size < 1:
            grouped_curve_ids.append({"invalid_group_size": group_size})
            continue
        next_curve_index = curve_index + group_size
        grouped_curve_ids.append(all_curve_ids[curve_index:next_curve_index])
        curve_index = next_curve_index

    if curve_index < len(all_curve_ids):
        grouped_curve_ids.append(
            {"unassigned_curve_ids": all_curve_ids[curve_index:]}
        )
    elif curve_index > len(all_curve_ids):
        grouped_curve_ids.append(
            {"missing_curve_count": curve_index - len(all_curve_ids)}
        )

    return str(grouped_curve_ids)


def _loop_split_points(traced_loop: TracedLoop) -> list[int]:
    return [
        point_id
        for point_id in _loop_ordered_points(traced_loop)
        if _point_valence(point_id) >= 3
    ]


def _zone_compound_subcurve_counts(
    config: dict[str, Any],
    traced_loop: TracedLoop,
) -> list[int]:
    for key in ("n_subcurvs_per_compound_curve", "n_subcurvs"):
        if key in config:
            group_sizes = [int(value) for value in config[key]]
            break
    else:
        group_sizes = [1] * len(traced_loop.curve_ids)
        if traced_loop.blunt_curve_ids:
            group_sizes.append(len(traced_loop.blunt_curve_ids))

    traced_curve_ids = _format_traced_loop_curve_ids(traced_loop)
    compound_curve_groups = _format_compound_curve_groups(traced_loop, group_sizes)

    if not group_sizes:
        raise ValueError(
            "n_subcurvs_per_compound_curve must define at least one compound "
            f"curve. Got {group_sizes}. Traced first loop: {traced_curve_ids}."
        )

    if any(group_size < 1 for group_size in group_sizes):
        raise ValueError(
            "n_subcurvs_per_compound_curve values must be >= 1: "
            f"{group_sizes}. Traced first loop: {traced_curve_ids}. "
            f"Compound groups from these counts: {compound_curve_groups}."
        )

    expected_total = len(traced_loop.curve_ids) + len(traced_loop.blunt_curve_ids)
    if sum(group_sizes) != expected_total:
        raise ValueError(
            "n_subcurvs_per_compound_curve must sum to the number of curves in "
            f"the first traced loop ({expected_total}), got {sum(group_sizes)}: "
            f"{group_sizes}. Traced first loop: {traced_curve_ids}. "
            f"Compound groups from these counts: {compound_curve_groups}."
        )
    if group_sizes[-1] != len(traced_loop.blunt_curve_ids):
        if traced_loop.blunt_curve_ids:
            raise ValueError(
                "The final n_subcurvs_per_compound_curve value must describe the "
                f"blunt trailing-edge group ({len(traced_loop.blunt_curve_ids)}), "
                f"got {group_sizes[-1]}. Traced first loop: {traced_curve_ids}. "
                f"Compound groups from these counts: {compound_curve_groups}."
            )

    return group_sizes


def _explicit_blunt_curve_count(config: dict[str, Any]) -> int | None:
    for key in ("n_subcurvs_per_compound_curve", "n_subcurvs"):
        if key in config:
            group_sizes = [int(value) for value in config[key]]
            if group_sizes:
                return group_sizes[-1]
            return None
    return None


def _explicit_total_curve_count(config: dict[str, Any]) -> int | None:
    for key in ("n_subcurvs_per_compound_curve", "n_subcurvs"):
        if key in config:
            return sum(int(value) for value in config[key])
    return None


def _closed_circumferential_loop(config: dict[str, Any]) -> bool:
    return bool(config.get("closed_circumferential_loop", True))


def _expected_circumferential_trace_curve_count(
    config: dict[str, Any],
    circumferential_template: dict[str, Any],
    closed_circumferential_loop: bool,
) -> int | None:
    explicit_curve_count = _explicit_total_curve_count(config)
    if explicit_curve_count is not None:
        return explicit_curve_count

    if closed_circumferential_loop:
        return None

    return _entry_group_count(circumferential_template)


def _has_blunt_te(config: dict[str, Any]) -> bool:
    return bool(config.get("has_blunt_te", False))


def _zone_direction(
    config: dict[str, Any],
    key: str,
    default: Sequence[float],
) -> list[float]:
    if key in config:
        return _as_vector(config[key])
    return _as_vector(default)


def _loop_split_points_from_group_sizes(
    traced_loop: TracedLoop,
    group_sizes: list[int],
    closed_circumferential_loop: bool = True,
) -> list[int]:
    split_points = [traced_loop.points[0]]
    curve_index = 0
    split_group_sizes = group_sizes[:-1] if closed_circumferential_loop else group_sizes
    for group_size in split_group_sizes:
        curve_index += group_size
        split_points.append(traced_loop.points[curve_index])
    return split_points


def _select_adjacent_curve_excluding(
    point_id: int,
    direction: Sequence[float],
    excluded_curves: set[int],
    forbidden_curves: set[int],
    *,
    min_score: float = -0.2,
) -> tuple[int, int, list[float]] | None:
    filtered_excluded = set(excluded_curves).union(forbidden_curves)
    return _select_adjacent_curve(
        point_id,
        direction,
        filtered_excluded,
        min_score=min_score,
    )


def _trace_longitudinal_segment_to_crossing(
    start_point: int,
    direction: Sequence[float],
    excluded_curves: set[int],
    forbidden_curves: set[int],
) -> tuple[list[int], int, list[float]] | None:
    current_point = int(start_point)
    current_direction = _as_vector(direction)
    local_excluded = set(excluded_curves)
    segment_curves = []
    last_vector = current_direction

    while True:
        selected = _select_adjacent_curve_excluding(
            current_point,
            current_direction,
            local_excluded,
            forbidden_curves,
        )
        if selected is None:
            return None

        curve_id, next_point, curve_vector = selected
        segment_curves.append(curve_id)
        local_excluded.add(abs(curve_id))
        tangent = _curve_tangent_for_traversal(curve_id, next_point, current_point)
        current_point = next_point
        current_direction = tangent if tangent is not None else curve_vector
        last_vector = current_direction

        if _point_valence(current_point) >= 3:
            return segment_curves, current_point, last_vector


def _trace_longitudinal_path_by_crossings(
    start_point: int,
    direction: Sequence[float],
    n_segments: int,
    excluded_curves: set[int],
    forbidden_curves: set[int],
) -> LongitudinalPath:
    points = [int(start_point)]
    curve_segments = []
    current_point = int(start_point)
    current_direction = _as_vector(direction)
    local_excluded = set(excluded_curves)

    for _ in range(n_segments):
        segment = _trace_longitudinal_segment_to_crossing(
            current_point,
            current_direction,
            local_excluded,
            forbidden_curves,
        )
        if segment is None:
            raise ValueError(
                f"Could not continue longitudinal path from point {current_point}."
            )

        segment_curves, next_point, next_direction = segment
        curve_segments.append(segment_curves)
        local_excluded.update(abs(curve_id) for curve_id in segment_curves)
        points.append(next_point)
        current_point = next_point
        current_direction = next_direction

    return LongitudinalPath(points, curve_segments)


def _longitudinal_direction_at_path_point(
    path: LongitudinalPath,
    point_index: int,
    fallback_direction: Sequence[float],
) -> list[float]:
    if point_index > 0:
        segment = path.curve_segments[point_index - 1]
        if segment:
            curve_id = segment[-1]
            point_id = path.points[point_index]
            previous_point = _other_curve_point(curve_id, point_id)
            tangent = _curve_tangent_for_traversal(
                curve_id,
                point_id,
                previous_point,
            )
            if tangent is not None:
                return tangent
        previous_point = path.points[point_index - 1]
        current_point = path.points[point_index]
        longitudinal_vector = _point_vector(previous_point, current_point)
        if _vector_norm(longitudinal_vector) > 0.0:
            return longitudinal_vector

    if point_index < len(path.curve_segments):
        segment = path.curve_segments[point_index]
        if segment:
            curve_id = segment[0]
            point_id = path.points[point_index]
            next_point = _other_curve_point(curve_id, point_id)
            tangent = _curve_tangent_towards_point(curve_id, point_id, next_point)
            if tangent is not None:
                return tangent
        current_point = path.points[point_index]
        next_point = path.points[point_index + 1]
        longitudinal_vector = _point_vector(current_point, next_point)
        if _vector_norm(longitudinal_vector) > 0.0:
            return longitudinal_vector

    return _as_vector(fallback_direction)


def _group_loop_curves_by_split_points(
    traced_loop: TracedLoop,
    split_points: list[int],
    closed_circumferential_loop: bool = True,
) -> tuple[list[int | list[int]], list[int]]:
    split_points = _unique_ordered(split_points)
    if not split_points or traced_loop.points[0] != split_points[0]:
        raise ValueError(
            "Circumferential split points must start at the circumferential loop start point."
        )

    split_point_set = set(split_points)
    curve_groups = []
    group_sizes = []
    current_group = []

    for curve_id, end_point in zip(traced_loop.curve_ids, traced_loop.points[1:]):
        current_group.append(curve_id)
        if end_point in split_point_set:
            curve_groups.append(
                current_group[0] if len(current_group) == 1 else list(current_group)
            )
            group_sizes.append(len(current_group))
            current_group = []

    if current_group:
        raise ValueError(
            f"Could not close circumferential curve group at split points {split_points}."
        )
    if traced_loop.blunt_curve_ids:
        curve_groups.append(
            traced_loop.blunt_curve_ids[0]
            if len(traced_loop.blunt_curve_ids) == 1
            else list(traced_loop.blunt_curve_ids)
        )
        group_sizes.append(len(traced_loop.blunt_curve_ids))
    expected_group_count = (
        len(split_points) if closed_circumferential_loop else len(split_points) - 1
    )
    if len(curve_groups) != expected_group_count:
        raise ValueError(
            f"Split points define {expected_group_count} groups, but traced loop "
            f"produced {len(curve_groups)} groups."
        )

    return curve_groups, group_sizes


def _curve_path_between_points(
    start_point: int,
    end_point: int,
    forbidden_curves: set[int],
) -> tuple[list[int], list[int]]:
    start_point = int(start_point)
    end_point = int(end_point)
    queue: list[tuple[float, int, list[int], list[int]]] = [
        (0.0, start_point, [], [start_point])
    ]
    best_distance = {start_point: 0.0}

    while queue:
        distance, point_id, curve_path, point_path = heapq.heappop(queue)
        if point_id == end_point:
            return curve_path, point_path
        if distance > best_distance.get(point_id, math.inf):
            continue

        for curve_id in _adjacent_curve_ids(point_id):
            abs_curve_id = abs(curve_id)
            if abs_curve_id in forbidden_curves:
                continue

            next_point = _other_curve_point(curve_id, point_id)
            next_distance = distance + _curve_length(curve_id)
            if next_distance >= best_distance.get(next_point, math.inf):
                continue

            best_distance[next_point] = next_distance
            heapq.heappush(
                queue,
                (
                    next_distance,
                    next_point,
                    curve_path + [curve_id],
                    point_path + [next_point],
                ),
            )

    raise ValueError(f"Could not find curve path from point {start_point} to {end_point}.")


def _trace_circumferential_loop_through_split_points(
    split_points: list[int],
    forbidden_curves: set[int],
    longitudinal_direction: Sequence[float] | None = None,
    has_blunt_te: bool = False,
    known_loop_points: set[int] | None = None,
    closed_circumferential_loop: bool = True,
) -> TracedLoop:
    split_points = _unique_ordered(split_points)
    if len(split_points) < 2:
        raise ValueError("Need at least two split points to trace an circumferential loop.")

    curve_ids = []
    points = [split_points[0]]
    local_forbidden = set(forbidden_curves)

    for start_point, end_point in zip(split_points[:-1], split_points[1:]):
        path_curves, path_points = _curve_path_between_points(
            start_point, end_point, local_forbidden
        )
        curve_ids.extend(path_curves)
        points.extend(path_points[1:])
        local_forbidden.update(abs(curve_id) for curve_id in path_curves)

    if not closed_circumferential_loop:
        return TracedLoop(curve_ids, points, [], [])

    if curve_ids and longitudinal_direction is not None and has_blunt_te:
        closing_curve = _connecting_curve(split_points[-1], split_points[0])
        if closing_curve is not None and closing_curve not in local_forbidden:
            local_blunt_direction = _local_blunt_direction(
                split_points[0],
                curve_ids[0],
                points[1],
                longitudinal_direction,
                local_forbidden,
            )
            closing_vector = _point_vector(split_points[-1], split_points[0])
            if (
                abs(
                    _dot(
                        _normalized(closing_vector),
                        _normalized(local_blunt_direction),
                    )
                )
                > 0.2
            ):
                return TracedLoop(
                    curve_ids,
                    points,
                    [closing_curve],
                    [split_points[-1], split_points[0]],
                )

    closing_curves, closing_points = _curve_path_between_points(
        split_points[-1], split_points[0], local_forbidden
    )
    if not has_blunt_te:
        curve_ids.extend(closing_curves)
        points.extend(closing_points[1:])
        return TracedLoop(curve_ids, points, [], [])

    known_intermediate_points = _known_intermediate_points(
        closing_points,
        known_loop_points or set(split_points),
    )
    if known_intermediate_points:
        raise ValueError(
            f"Closing blunt trailing-edge curve path from {split_points[-1]} to "
            f"{split_points[0]} passes through already identified loop/longitudinal "
            f"points {known_intermediate_points}: {closing_points}."
        )
    if (
        longitudinal_direction is not None
        and has_blunt_te
        and curve_ids
    ):
        local_blunt_direction = _local_blunt_direction(
            split_points[0],
            curve_ids[0],
            points[1],
            longitudinal_direction,
            local_forbidden,
        )
        if (
            _curve_path_alignment(
                closing_curves,
                closing_points,
                local_blunt_direction,
            )
            <= 0.2
        ):
            raise ValueError(
                f"Closing curve path from {split_points[-1]} to {split_points[0]} "
                f"is not aligned with the local blunt trailing-edge direction: "
                f"{closing_curves}."
            )
    if not closing_curves:
        raise ValueError(
            f"Expected a closing blunt trailing-edge curve path from "
            f"{split_points[-1]} to {split_points[0]}, got {closing_curves}."
        )

    return TracedLoop(curve_ids, points, closing_curves, closing_points)


def _entry_group_count(entry: dict[str, Any]) -> int | None:
    return _template_group_count(entry)


def _best_loop_split_points(
    traced_loop: TracedLoop,
    defining_split_points: list[int],
    template_entry: dict[str, Any],
    closed_circumferential_loop: bool = True,
) -> list[int]:
    target_group_count = _entry_group_count(template_entry)
    target_split_point_count = (
        None
        if target_group_count is None
        else target_group_count if closed_circumferential_loop else target_group_count + 1
    )
    split_points = [
        point_id
        for point_id in defining_split_points
        if point_id in set(_loop_ordered_points(traced_loop))
    ]
    if (
        target_split_point_count is not None
        and len(split_points) != target_split_point_count
    ):
        local_split_points = _loop_split_points(traced_loop)
        if len(local_split_points) == target_split_point_count:
            return local_split_points
    return split_points


def _circumferential_entry_from_grouped_loop(
    traced_loop: TracedLoop,
    circumferential_template: dict[str, Any],
    curve_groups: list[int | list[int]],
    group_sizes: list[int],
) -> dict[str, Any]:
    traced_curve_ids = _loop_all_curve_ids(traced_loop)
    traced_invert_directions = [
        _curve_traversal_invert_direction(
            curve_id,
            traced_loop.points[curve_index],
            traced_loop.points[curve_index + 1],
        )
        for curve_index, curve_id in enumerate(traced_loop.curve_ids)
    ]
    traced_invert_directions.extend(
        _curve_traversal_invert_direction(
            curve_id,
            traced_loop.blunt_points[curve_index],
            traced_loop.blunt_points[curve_index + 1],
        )
        for curve_index, curve_id in enumerate(traced_loop.blunt_curve_ids)
    )

    circumferential_entry = copy.deepcopy(circumferential_template)
    circumferential_entry["curve_ids"] = curve_groups
    inferred_invert_directions = _group_values_by_curve_groups(
        traced_invert_directions, curve_groups
    )
    _remap_template_values_to_groups(
        circumferential_entry,
        circumferential_template,
        group_sizes,
        len(traced_curve_ids),
    )
    user_invert_directions = _grouped_template_invert_directions(
        circumferential_template,
        group_sizes,
        len(traced_curve_ids),
    )
    circumferential_entry["invert_direction"] = inferred_invert_directions
    _normalize_invert_directions_for_curve_groups(circumferential_entry)
    circumferential_entry["_group_invert_direction"] = [
        _group_invert_direction(user_invert_direction)
        for user_invert_direction in (
            user_invert_directions or [False] * len(curve_groups)
        )
    ]
    return circumferential_entry


def _template_group_sizes(template_entry: dict[str, Any], total_curves: int) -> list[int]:
    if "n_subcurvs" in template_entry:
        return [int(group_size) for group_size in template_entry["n_subcurvs"]]
    if "curve_ids" in template_entry:
        return [
            _curve_group_size(curve_group)
            for curve_group in template_entry["curve_ids"]
        ]
    return [1] * total_curves


def _expand_template_values(
    template_entry: dict[str, Any],
    key: str,
    template_group_sizes: list[int],
    total_curves: int,
) -> list[Any] | None:
    if key not in template_entry:
        return None

    value = template_entry[key]
    if not isinstance(value, list):
        return [copy.deepcopy(value) for _ in range(total_curves)]

    if len(value) != len(template_group_sizes):
        return [copy.deepcopy(value_item) for value_item in value]

    expanded_values = []
    for group_value, group_size in zip(value, template_group_sizes):
        if key in ("invert_direction", "_invert_direction") and isinstance(
            group_value, list
        ):
            expanded_values.extend(copy.deepcopy(group_value))
        else:
            expanded_values.extend(copy.deepcopy(group_value) for _ in range(group_size))

    return expanded_values


def _regroup_template_values(
    expanded_values: list[Any],
    group_sizes: list[int],
    *,
    keep_group_lists: bool,
) -> list[Any]:
    regrouped_values = []
    index = 0
    for group_size in group_sizes:
        group_values = expanded_values[index : index + group_size]
        index += group_size
        if keep_group_lists and group_size > 1:
            regrouped_values.append(group_values)
        else:
            regrouped_values.append(group_values[0])
    return regrouped_values


def _remap_template_values_to_groups(
    entry: dict[str, Any],
    template_entry: dict[str, Any],
    group_sizes: list[int],
    total_curves: int,
) -> None:
    template_group_sizes = _template_group_sizes(template_entry, total_curves)

    for key in ("type", "Parameter", "n_pts", "invert_direction"):
        template_key = f"_{key}" if key not in template_entry else key
        value = template_entry.get(template_key)
        if isinstance(value, list) and len(value) == len(group_sizes):
            entry[key] = copy.deepcopy(value)
            continue

        expanded_values = _expand_template_values(
            template_entry, template_key, template_group_sizes, total_curves
        )
        if expanded_values is None:
            continue
        if len(expanded_values) < total_curves:
            expanded_values.extend(
                copy.deepcopy(expanded_values[-1])
                for _ in range(total_curves - len(expanded_values))
            )
        entry[key] = _regroup_template_values(
            expanded_values,
            group_sizes,
            keep_group_lists=(key == "invert_direction"),
        )


def _normalize_invert_directions_for_curve_groups(entry: dict[str, Any]) -> None:
    if "curve_ids" not in entry or "invert_direction" not in entry:
        return

    curve_groups = entry["curve_ids"]
    invert_directions = entry["invert_direction"]
    if isinstance(invert_directions, bool):
        return

    normalized_directions = []
    for curve_group, invert_direction in zip(curve_groups, invert_directions):
        group_size = _curve_group_size(curve_group)
        if isinstance(invert_direction, bool):
            normalized_directions.append(invert_direction)
            continue

        if group_size == 1:
            normalized_directions.append(_as_bool(invert_direction[0]))
            continue

        direction_group = [_as_bool(value) for value in invert_direction]
        if len(direction_group) < group_size:
            direction_group.extend([direction_group[-1]] * (group_size - len(direction_group)))
        normalized_directions.append(direction_group[:group_size])

    entry["invert_direction"] = normalized_directions


def _trace_longitudinal_curves(
    point_ids: list[int],
    directions: list[list[float]],
    excluded_curves: set[int],
) -> tuple[list[int], list[bool], list[int], list[list[float]]]:
    curve_ids = []
    invert_directions = []
    next_points = []
    next_directions = []

    for index, point_id in enumerate(point_ids):
        direction = directions[min(index, len(directions) - 1)]
        current_point = point_id
        current_direction = direction
        chain_curves = []
        chain_inverts = []
        chain_excluded = set(excluded_curves)
        last_vector = None

        while True:
            selected = _select_adjacent_curve(
                current_point, current_direction, chain_excluded
            )
            if selected is None:
                break

            curve_id, next_point, curve_vector = selected
            chain_curves.append(curve_id)
            chain_inverts.append(
                _curve_traversal_invert_direction(curve_id, current_point, next_point)
            )
            chain_excluded.add(abs(curve_id))
            tangent = _curve_tangent_for_traversal(curve_id, next_point, current_point)
            current_point = next_point
            current_direction = tangent if tangent is not None else curve_vector
            last_vector = current_direction

            continuing_curves = [
                candidate_curve
                for candidate_curve in _adjacent_curve_ids(current_point)
                if abs(candidate_curve) not in chain_excluded
                and abs(candidate_curve) not in excluded_curves
            ]
            if len(continuing_curves) != 1:
                break

        if not chain_curves or last_vector is None:
            continue

        curve_ids.extend(chain_curves)
        invert_directions.extend(chain_inverts)
        next_points.append(current_point)
        next_directions.append(last_vector)

    return curve_ids, invert_directions, next_points, next_directions


def _mesh_zones(mesh_def: dict[str, Any]) -> list[dict[str, Any]]:
    zones = mesh_def.get("mesh_zones")
    if not isinstance(zones, list) or not zones:
        raise ValueError("Mesh definition must contain a non-empty 'mesh_zones' list.")

    for index, zone in enumerate(zones, start=1):
        if not isinstance(zone, dict):
            raise TypeError(f"mesh_zones[{index}] must be an object.")

    return zones


def _mesh_zone_name(zone: dict[str, Any], zone_index: int) -> str:
    return str(zone.get("name", f"mesh zone {zone_index}"))


def _mesh_zone_curve_source(zone: dict[str, Any], zone_name: str) -> str:
    source = zone.get("curve_definition", zone.get("curve source", zone.get("type")))
    if source is None:
        raise ValueError(
            f"Mesh zone {zone_name!r} must define 'curve_definition' as "
            "'automatic' or 'manual'."
        )
    if isinstance(source, dict):
        source = source.get("type", source.get("mode"))
    source = str(source).lower()
    if source not in {"automatic", "manual"}:
        raise ValueError(
            f"Mesh zone {zone_name!r} has unsupported curve_definition "
            f"{source!r}; expected 'automatic' or 'manual'."
        )
    return source


def _merge_transfinite_template(
    common_template: dict[str, Any],
    local_template: dict[str, Any],
    excluded_common_keys: set[str],
) -> dict[str, Any]:
    merged_template = {
        key: copy.deepcopy(value)
        for key, value in common_template.items()
        if key not in excluded_common_keys
    }
    merged_template.update(copy.deepcopy(local_template))
    return merged_template


def _merge_longitudinal_transfinite_template(
    longitudinal_template: dict[str, Any],
    section_template: dict[str, Any],
) -> dict[str, Any]:
    merged_template = _merge_transfinite_template(
        longitudinal_template,
        section_template,
        {"sections", "default type"},
    )
    if "type" not in merged_template and "default type" in longitudinal_template:
        merged_template["type"] = copy.deepcopy(longitudinal_template["default type"])
    return merged_template


def _automatic_circumferential_templates(
    config: dict[str, Any],
    loop_count: int | None = None,
) -> list[dict[str, Any]]:
    circumferential_def = config.get("circumferential_transfinite_def", {})
    if not circumferential_def:
        return []

    loop_defs = circumferential_def.get("loops")
    if isinstance(loop_defs, list) and loop_defs:
        return [
            _merge_transfinite_template(circumferential_def, loop_def, {"loops"})
            for loop_def in loop_defs
        ]

    if loop_count is None:
        return []

    common_template = {
        key: copy.deepcopy(value)
        for key, value in circumferential_def.items()
        if key != "loops"
    }
    return [
        {
            **copy.deepcopy(common_template),
            "name": f"circumferential loop {loop_index + 1}",
        }
        for loop_index in range(loop_count)
    ]


def _automatic_longitudinal_templates(config: dict[str, Any]) -> list[dict[str, Any]]:
    longitudinal_def = config.get("longitudinal_transfinite_def")
    if longitudinal_def is None:
        longitudinal_def = config.get("circumferential_transfinite_def", {}).get(
            "longitudinal_transfinite_def", {}
        )
    if not longitudinal_def:
        return []

    section_defs = longitudinal_def.get("sections")
    if isinstance(section_defs, list) and section_defs:
        return [
            _merge_longitudinal_transfinite_template(longitudinal_def, section_def)
            for section_def in section_defs
        ]

    target_sizes = longitudinal_def.get("target ele sizes")
    if isinstance(target_sizes, list) and len(target_sizes) >= 2:
        common_template = {
            key: copy.deepcopy(value)
            for key, value in longitudinal_def.items()
            if key != "target ele sizes"
        }
        common_template.setdefault("mesh size mode", "ele size")
        return [
            {
                **copy.deepcopy(common_template),
                "name": f"longitudinal section {index + 1}",
                "target ele size 1": target_sizes[index],
                "target ele size 2": target_sizes[index + 1],
            }
            for index in range(len(target_sizes) - 1)
        ]

    return []


def _automatic_curve_sequences_for_zone(
    config: dict[str, Any],
    zone_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start_point = int(config.get("start_pt", config.get("start point", 66)))
    circumferential_direction = _zone_direction(
        config,
        "circumferential_direction",
        [-1.0, 0.0, 0.0],
    )
    longitudinal_direction = _zone_direction(
        config,
        "longitudinal_direction",
        [0.0, 1.0, 0.0],
    )
    has_blunt_te = _has_blunt_te(config)
    closed_circumferential_loop = _closed_circumferential_loop(config)
    longitudinal_templates = [
        copy.deepcopy(entry)
        for entry in _iter_curve_entries(config.get("longitudinal_curve_sequences", []))
    ]
    if not longitudinal_templates:
        longitudinal_templates = _automatic_longitudinal_templates(config)
    circumferential_templates = [
        copy.deepcopy(entry)
        for entry in _iter_curve_entries(config.get("circumferential_curve_sequences", []))
    ]
    if not circumferential_templates:
        circumferential_templates = _automatic_circumferential_templates(
            config, len(longitudinal_templates) + 1 if longitudinal_templates else None
        )
    if not circumferential_templates:
        raise ValueError(
            f"Automatic curve discovery in mesh zone {zone_name!r} requires "
            "circumferential templates."
        )

    n_longitudinal_sections = len(longitudinal_templates)
    n_circumferential_loops = len(circumferential_templates)
    if n_circumferential_loops != n_longitudinal_sections + 1:
        raise ValueError(
            f"Automatic tracing needs one more circumferential template than longitudinal "
            f"sections, got {n_circumferential_loops} and {n_longitudinal_sections}."
        )

    expected_circumferential_curve_count = _expected_circumferential_trace_curve_count(
        config,
        circumferential_templates[0],
        closed_circumferential_loop,
    )
    if not closed_circumferential_loop and expected_circumferential_curve_count is None:
        raise ValueError(
            f"Open circumferential loop tracing in mesh zone {zone_name!r} requires a "
            "curve count from 'n_subcurvs_per_compound_curve', 'n_subcurvs', or "
            "a list-valued circumferential transfinite definition."
        )

    first_loop = _trace_circumferential_loop(
        start_point,
        circumferential_direction,
        has_blunt_te,
        longitudinal_direction,
        set(),
        (
            _explicit_blunt_curve_count(config)
            if closed_circumferential_loop and has_blunt_te
            else None
        ),
        expected_circumferential_curve_count,
        closed_circumferential_loop,
    )
    first_loop_group_sizes = _zone_compound_subcurve_counts(config, first_loop)
    first_loop_group_count = _entry_group_count(circumferential_templates[0])
    if (
        first_loop_group_count is not None
        and len(first_loop_group_sizes) != first_loop_group_count
    ):
        raise ValueError(
            "n_subcurvs_per_compound_curve defines "
            f"{len(first_loop_group_sizes)} compound curves, but the first "
            f"circumferential template defines {first_loop_group_count}. Traced "
            f"first loop: {_format_traced_loop_curve_ids(first_loop)}. "
            "Compound groups from these counts: "
            f"{_format_compound_curve_groups(first_loop, first_loop_group_sizes)}."
        )
    first_loop_split_points = _loop_split_points_from_group_sizes(
        first_loop,
        first_loop_group_sizes,
        closed_circumferential_loop,
    )
    if len(first_loop_split_points) < 2:
        raise ValueError(
            "Could not identify compound curve defining points on the first "
            "circumferential loop."
        )

    first_loop_curves = {abs(curve_id) for curve_id in _loop_all_curve_ids(first_loop)}
    defining_paths = [
        _trace_longitudinal_path_by_crossings(
            split_point,
            longitudinal_direction,
            n_longitudinal_sections,
            set(),
            first_loop_curves,
        )
        for split_point in first_loop_split_points
    ]
    known_loop_points = {
        point_id for defining_path in defining_paths for point_id in defining_path.points
    }
    loop_start_points = defining_paths[0].points
    traced_loops = [first_loop]
    circumferential_curve_set = set(first_loop_curves)

    for loop_index in range(1, n_circumferential_loops):
        loop_split_points = [
            defining_path.points[loop_index] for defining_path in defining_paths
        ]
        loop_longitudinal_direction = _longitudinal_direction_at_path_point(
            defining_paths[0],
            loop_index,
            longitudinal_direction,
        )
        try:
            traced_loop = _trace_circumferential_loop_through_split_points(
                loop_split_points,
                circumferential_curve_set,
                loop_longitudinal_direction,
                has_blunt_te,
                known_loop_points,
                closed_circumferential_loop,
            )
        except ValueError as exc:
            warnings.warn(
                f"Stopping automatic circumferential-loop tracing before loop "
                f"{loop_index + 1}: {exc}",
                stacklevel=2,
            )
            break
        traced_loops.append(traced_loop)
        circumferential_curve_set.update(
            abs(curve_id) for curve_id in _loop_all_curve_ids(traced_loop)
        )

    circumferential_templates = circumferential_templates[: len(traced_loops)]
    longitudinal_templates = longitudinal_templates[: max(0, len(traced_loops) - 1)]

    circumferential_entries = []
    for loop_index, traced_loop in enumerate(traced_loops):
        split_points = [
            defining_path.points[loop_index] for defining_path in defining_paths
        ]
        split_points = _best_loop_split_points(
            traced_loop,
            split_points,
            circumferential_templates[loop_index],
            closed_circumferential_loop,
        )
        curve_groups, group_sizes = _group_loop_curves_by_split_points(
            traced_loop,
            split_points,
            closed_circumferential_loop,
        )
        circumferential_entries.append(
            _circumferential_entry_from_grouped_loop(
                traced_loop,
                circumferential_templates[loop_index],
                curve_groups,
                group_sizes,
            )
        )

    longitudinal_entries = []
    used_longitudinal_curves: set[int] = set()
    for section_index, longitudinal_template in enumerate(longitudinal_templates):
        section_curve_ids = []
        section_invert_directions = []
        section_direction = _point_vector(
            defining_paths[0].points[section_index],
            defining_paths[0].points[section_index + 1],
        )

        for start_point_candidate in _loop_ordered_points(traced_loops[section_index]):
            segment = _trace_longitudinal_segment_to_crossing(
                start_point_candidate,
                section_direction,
                used_longitudinal_curves,
                circumferential_curve_set,
            )
            if segment is None:
                continue

            segment_curves, end_point, _ = segment
            next_loop_points = set(_loop_ordered_points(traced_loops[section_index + 1]))
            if end_point not in next_loop_points:
                continue

            segment_invert_directions = []
            current_point = start_point_candidate
            for curve_id in segment_curves:
                next_point = _other_curve_point(curve_id, current_point)
                segment_invert_directions.append(
                    _curve_traversal_invert_direction(
                        curve_id, current_point, next_point
                    )
                )
                used_longitudinal_curves.add(abs(curve_id))
                current_point = next_point
            section_curve_ids.append(
                segment_curves[0] if len(segment_curves) == 1 else list(segment_curves)
            )
            section_invert_directions.append(
                segment_invert_directions[0]
                if len(segment_invert_directions) == 1
                else segment_invert_directions
            )

        longitudinal_entry = copy.deepcopy(longitudinal_template)
        longitudinal_entry["curve_ids"] = section_curve_ids
        if (
            "_invert_direction" in longitudinal_entry
            and "invert_direction" not in longitudinal_entry
        ):
            spec_count = len(section_curve_ids)
            longitudinal_entry["invert_direction"] = section_invert_directions
            longitudinal_entry["_group_invert_direction"] = [
                _group_invert_direction(
                    _per_spec_value(
                        longitudinal_entry,
                        "_invert_direction",
                        spec_index,
                        spec_count,
                        False,
                    )
                )
                for spec_index in range(spec_count)
            ]
        else:
            longitudinal_entry["invert_direction"] = section_invert_directions
            longitudinal_entry["_group_invert_direction"] = [
                False for _ in longitudinal_entry["invert_direction"]
            ]
        longitudinal_entries.append(longitudinal_entry)

    return circumferential_entries, longitudinal_entries


def _zone_curve_entries(zone: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [copy.deepcopy(entry) for entry in _iter_curve_entries(zone.get(key, []))]


def _zone_surface_entries(zone: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(entry)
        for entry in _iter_surface_entries(zone.get("transfinite_surfaces", []))
    ]


def _tag_zone_curve_entries(
    entries: list[dict[str, Any]],
    *,
    zone_name: str,
    curve_source: str,
) -> list[dict[str, Any]]:
    tagged_entries = []
    for entry in entries:
        tagged_entry = copy.deepcopy(entry)
        tagged_entry["_mesh_zone_name"] = zone_name
        tagged_entry["_mesh_zone_curve_source"] = curve_source
        tagged_entries.append(tagged_entry)
    return tagged_entries


def _tag_zone_surface_entries(
    entries: list[dict[str, Any]],
    *,
    zone_name: str,
) -> list[dict[str, Any]]:
    tagged_entries = []
    for entry in entries:
        tagged_entry = copy.deepcopy(entry)
        tagged_entry["_mesh_zone_name"] = zone_name
        tagged_entries.append(tagged_entry)
    return tagged_entries


def _mesh_zone_name_from_entry(entry: dict[str, Any]) -> str | None:
    zone_name = entry.get("_mesh_zone_name")
    return None if zone_name is None else str(zone_name)


def _raise_with_mesh_zone(entry: dict[str, Any], error: Exception) -> None:
    zone_name = _mesh_zone_name_from_entry(entry)
    if zone_name is not None:
        raise MeshZoneError(zone_name, error) from error
    raise error


def expand_mesh_zones(mesh_def: dict[str, Any]) -> dict[str, Any]:
    expanded_mesh_def = copy.deepcopy(mesh_def)
    circumferential_entries: list[dict[str, Any]] = []
    longitudinal_entries: list[dict[str, Any]] = []
    explicit_entries: list[dict[str, Any]] = []
    surface_entries: list[dict[str, Any]] = []

    for zone_index, zone in enumerate(_mesh_zones(mesh_def), start=1):
        zone_name = _mesh_zone_name(zone, zone_index)
        try:
            curve_source = _mesh_zone_curve_source(zone, zone_name)

            if curve_source == "automatic":
                zone_circumferential_entries, zone_longitudinal_entries = (
                    _automatic_curve_sequences_for_zone(zone, zone_name)
                )
                circumferential_entries.extend(
                    _tag_zone_curve_entries(
                        zone_circumferential_entries,
                        zone_name=zone_name,
                        curve_source=curve_source,
                    )
                )
                longitudinal_entries.extend(
                    _tag_zone_curve_entries(
                        zone_longitudinal_entries,
                        zone_name=zone_name,
                        curve_source=curve_source,
                    )
                )
                continue

            manual_entries = (
                _zone_curve_entries(zone, "circumferential_curve_sequences")
                + _zone_curve_entries(zone, "longitudinal_curve_sequences")
                + _zone_curve_entries(zone, "explicit_curve_sequences")
            )
            zone_surface_entries = _zone_surface_entries(zone)
            if not manual_entries and not zone_surface_entries:
                raise ValueError(
                    f"Manual mesh zone {zone_name!r} must define at least one curve "
                    "sequence or transfinite surface."
                )

            circumferential_entries.extend(
                _tag_zone_curve_entries(
                    _zone_curve_entries(zone, "circumferential_curve_sequences"),
                    zone_name=zone_name,
                    curve_source=curve_source,
                )
            )
            longitudinal_entries.extend(
                _tag_zone_curve_entries(
                    _zone_curve_entries(zone, "longitudinal_curve_sequences"),
                    zone_name=zone_name,
                    curve_source=curve_source,
                )
            )
            explicit_entries.extend(
                _tag_zone_curve_entries(
                    _zone_curve_entries(zone, "explicit_curve_sequences"),
                    zone_name=zone_name,
                    curve_source=curve_source,
                )
            )
            surface_entries.extend(
                _tag_zone_surface_entries(
                    zone_surface_entries,
                    zone_name=zone_name,
                )
            )
        except Exception as error:
            raise MeshZoneError(zone_name, error) from error

    expanded_mesh_def["circumferential_curve_sequences"] = circumferential_entries
    expanded_mesh_def["longitudinal_curve_sequences"] = longitudinal_entries
    expanded_mesh_def["explicit_curve_sequences"] = explicit_entries
    if surface_entries:
        expanded_mesh_def["transfinite_surfaces"] = (
            copy.deepcopy(
                list(_iter_surface_entries(mesh_def.get("transfinite_surfaces", {})))
            )
            + surface_entries
        )
    return expanded_mesh_def


def _flatten_curve_group(curve_group: int | list[int]) -> list[int]:
    if isinstance(curve_group, int):
        return [curve_group]
    return [int(curve_id) for curve_id in curve_group]


def _entry_curve_groups(entry: dict[str, Any]) -> list[list[int]]:
    return [_flatten_curve_group(group) for group in entry["curve_ids"]]


def _ordered_group_point_pairs(
    curve_groups: list[list[int]],
) -> list[tuple[list[int], int, int]]:
    first_curve = curve_groups[0][0]
    last_curve = curve_groups[-1][-1]
    first_endpoints = set(_curve_endpoints(first_curve))
    last_endpoints = set(_curve_endpoints(last_curve))
    shared_points = first_endpoints.intersection(last_endpoints)
    current_point = next(iter(shared_points)) if shared_points else _curve_endpoints(first_curve)[0]

    group_point_pairs = []
    for curve_group in curve_groups:
        group_start = current_point
        for curve_id in curve_group:
            current_point = _other_curve_point(curve_id, current_point)
        group_point_pairs.append((curve_group, group_start, current_point))

    return group_point_pairs


def _flat_group_point_pairs(
    curve_groups: list[list[int]],
) -> list[tuple[list[int], int, int]]:
    flat_groups = [[curve_id] for curve_group in curve_groups for curve_id in curve_group]
    return _ordered_group_point_pairs(flat_groups)


def _surface_boundary_curve_map() -> dict[frozenset[int], int]:
    surface_map = {}
    for _, surface_id in gmsh.model.getEntities(2):
        boundary_curves = frozenset(_surface_boundary_curves(surface_id))
        surface_map[boundary_curves] = surface_id
    return surface_map


def _surface_ordered_boundary_points(surface_id: int) -> list[int]:
    ordered_points, _ = _surface_ordered_boundary(surface_id)
    return ordered_points


def _surface_ordered_boundary(surface_id: int) -> tuple[list[int], list[int]]:
    boundary_curves = _surface_boundary_curves(surface_id)
    if not boundary_curves:
        return [], []

    unused_curves = set(boundary_curves)
    first_curve = boundary_curves[0]
    start_point, current_point = _curve_endpoints(first_curve)
    ordered_points = [start_point, current_point]
    ordered_curves = [first_curve]
    unused_curves.remove(first_curve)

    while unused_curves and current_point != start_point:
        next_curve = None
        next_point = None
        for curve_id in unused_curves:
            endpoints = _curve_endpoints(curve_id)
            if current_point == endpoints[0]:
                next_curve = curve_id
                next_point = endpoints[1]
                break
            if current_point == endpoints[1]:
                next_curve = curve_id
                next_point = endpoints[0]
                break

        if next_curve is None or next_point is None:
            break

        unused_curves.remove(next_curve)
        current_point = next_point
        ordered_curves.append(next_curve)
        ordered_points.append(current_point)

    return ordered_points, ordered_curves


def _curve_incident_boundary_points(curve_ids: Iterable[int]) -> dict[int, set[int]]:
    incident_curves: dict[int, set[int]] = {}
    for curve_id in curve_ids:
        for point_id in _curve_endpoints(curve_id):
            incident_curves.setdefault(point_id, set()).add(curve_id)
    return incident_curves


def _surface_transfinite_corners(
    surface_id: int,
    circumferential_curves: set[int],
    longitudinal_curves: set[int],
) -> list[int]:
    boundary_curves = _surface_boundary_curves(surface_id)
    boundary_circumferential_curves = [
        curve_id for curve_id in boundary_curves if curve_id in circumferential_curves
    ]
    boundary_longitudinal_curves = [
        curve_id for curve_id in boundary_curves if curve_id in longitudinal_curves
    ]
    incident_circumferential = _curve_incident_boundary_points(boundary_circumferential_curves)
    incident_longitudinal = _curve_incident_boundary_points(boundary_longitudinal_curves)
    corner_set = set(incident_circumferential).intersection(incident_longitudinal)

    ordered_corners = []
    for point_id in _surface_ordered_boundary_points(surface_id):
        if point_id in corner_set and point_id not in ordered_corners:
            ordered_corners.append(point_id)

    return ordered_corners


def _longitudinal_curve_map(curve_ids: list[int]) -> dict[int, int]:
    curve_map = {}
    for curve_id in curve_ids:
        point_a, point_b = _curve_endpoints(curve_id)
        curve_map[point_a] = curve_id
        curve_map[point_b] = curve_id
    return curve_map


def _longitudinal_connector_curves(longitudinal_entries: Iterable[dict[str, Any]]) -> set[int]:
    connector_curves = set()
    for entry in longitudinal_entries:
        curve_ids = [
            abs(curve_id)
            for curve_group in _entry_curve_groups(entry)
            for curve_id in curve_group
        ]
        if not curve_ids:
            continue

        curve_lengths = [_curve_length(curve_id) for curve_id in curve_ids]
        short_threshold = _short_curve_threshold(curve_lengths)
        connector_curves.update(
            curve_id
            for curve_id, curve_length in zip(curve_ids, curve_lengths)
            if curve_length <= short_threshold
        )
    return connector_curves


def _automatic_surfaces_enabled(mesh_def: dict[str, Any]) -> bool:
    return any(
        _mesh_zone_curve_source(zone, _mesh_zone_name(zone, zone_index)) == "automatic"
        for zone_index, zone in enumerate(_mesh_zones(mesh_def), start=1)
    )


def _automatic_zone_curve_entries(
    section: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in _iter_curve_entries(section)
        if entry.get("_mesh_zone_curve_source") == "automatic"
    ]


def _surface_meshing_algorithm_surface_ids(mesh_def: dict[str, Any]) -> set[int]:
    entries = mesh_def.get("surface_meshing_algorithms", [])
    if not entries:
        return set()
    if not isinstance(entries, list):
        raise TypeError("surface_meshing_algorithms must be a list.")

    surface_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(
                "Each surface_meshing_algorithms entry must be an object."
            )
        surface_ids.update(int(surface_id) for surface_id in entry.get("surfaces", []))
    return surface_ids


def apply_automatic_transfinite_surfaces(mesh_def: dict[str, Any]) -> dict[str, Any]:
    if not _automatic_surfaces_enabled(mesh_def):
        return mesh_def

    circumferential_entries = _automatic_zone_curve_entries(
        mesh_def.get("circumferential_curve_sequences", [])
    )
    longitudinal_entries = _automatic_zone_curve_entries(
        mesh_def.get("longitudinal_curve_sequences", [])
    )
    if len(circumferential_entries) < 2 or not longitudinal_entries:
        return mesh_def

    circumferential_curves = {
        abs(curve_id)
        for entry in circumferential_entries
        for curve_group in _entry_curve_groups(entry)
        for curve_id in curve_group
    }
    longitudinal_curves = {
        abs(curve_id)
        for entry in longitudinal_entries
        for curve_group in _entry_curve_groups(entry)
        for curve_id in curve_group
    }
    transfinite_curves = circumferential_curves.union(longitudinal_curves)
    transfinite_surfaces = []
    transfinite_surface_ids = set()
    algorithm_surface_ids = _surface_meshing_algorithm_surface_ids(mesh_def)
    manual_surfaces = copy.deepcopy(
        list(_iter_surface_entries(mesh_def.get("transfinite_surfaces", {})))
    )
    manual_surface_ids = {int(surface["id"]) for surface in manual_surfaces}

    for _, surface_id in gmsh.model.getEntities(2):
        if surface_id in algorithm_surface_ids or surface_id in manual_surface_ids:
            continue

        boundary_curves = set(_surface_boundary_curves(surface_id))
        boundary_circumferential_curves = boundary_curves.intersection(circumferential_curves)
        boundary_longitudinal_curves = boundary_curves.intersection(longitudinal_curves)
        if (
            not boundary_curves.issubset(transfinite_curves)
            or not boundary_circumferential_curves
            or not boundary_longitudinal_curves
        ):
            continue

        boundary_points = _surface_transfinite_corners(
            surface_id, circumferential_curves, longitudinal_curves
        )
        if len(boundary_points) == 4:
            transfinite_surfaces.append(
                {
                    "id": surface_id,
                    "Arrangement": "left",
                    "boundary points": boundary_points,
                }
            )
            transfinite_surface_ids.add(surface_id)

    all_surface_ids = {surface_id for _, surface_id in gmsh.model.getEntities(2)}
    auto_mesh_def = copy.deepcopy(mesh_def)
    auto_mesh_def["transfinite_surfaces"] = manual_surfaces + transfinite_surfaces
    auto_mesh_def["unstructured_surfaces"] = sorted(
        all_surface_ids - transfinite_surface_ids - manual_surface_ids
    )
    return auto_mesh_def


def _iter_traced_curve_specs(
    entry: dict[str, Any],
) -> Iterable[CurveSpec]:
    traced_curves = _trace_surface_edge_curves(entry["start_pt"], entry["v_pointing"])
    n_subcurvs = [int(value) for value in entry["n_subcurvs"]]

    if any(value < 1 for value in n_subcurvs):
        raise ValueError(f"n_subcurvs values must be >= 1: {entry!r}")

    expected_curve_count = sum(n_subcurvs)
    if expected_curve_count != len(traced_curves):
        raise ValueError(
            f"Trace from point {entry['start_pt']} produced {len(traced_curves)} "
            f"curves, but n_subcurvs requests {expected_curve_count}: {n_subcurvs}."
        )

    index = 0
    spec_count = len(n_subcurvs)
    for spec_index, group_size in enumerate(n_subcurvs):
        group = traced_curves[index : index + group_size]
        index += group_size
        curve_ids = [curve_id for curve_id, _ in group]
        invert_directions = [invert_direction for _, invert_direction in group]
        n_pts, mesh_type, coef = _curve_spec_discretization(
            entry, spec_index, spec_count, curve_ids
        )

        user_invert_direction = _per_spec_value(
            entry, "invert_direction", spec_index, spec_count, False
        )
        user_invert_group = _as_bool_list(user_invert_direction, group_size)
        invert_directions = [
            bool(inferred) != bool(user_invert)
            for inferred, user_invert in zip(invert_directions, user_invert_group)
        ]

        if group_size == 1:
            yield CurveSpec(
                curve_ids[0],
                invert_directions[0],
                _group_invert_direction(user_invert_direction),
                n_pts,
                mesh_type,
                coef,
            )
        else:
            yield CurveSpec(
                curve_ids,
                invert_directions,
                _group_invert_direction(user_invert_direction),
                n_pts,
                mesh_type,
                coef,
            )


def _legacy_invert_direction(entry: dict[str, Any]) -> bool:
    direction = str(entry.get("direction", "normal")).lower()

    if direction == "normal":
        return False
    if direction == "flipped":
        return True

    raise ValueError(f"Unsupported curve direction {direction!r}.")


def _iter_curve_specs(
    entry: dict[str, Any],
) -> Iterable[CurveSpec]:
    if _has_trace_definition(entry):
        yield from _iter_traced_curve_specs(entry)
        return

    curve_ids = entry["curve_ids"]
    spec_count = len(curve_ids)

    if "invert_direction" in entry:
        invert_direction = entry["invert_direction"]
        if isinstance(invert_direction, bool):
            invert_direction = [invert_direction] * spec_count
        elif len(invert_direction) == spec_count:
            pass
        elif spec_count == 1:
            invert_direction = [invert_direction]
        else:
            raise ValueError(
                f"invert_direction length {len(invert_direction)} does not match "
                f"curve_ids length {len(curve_ids)} in {entry.get('name', entry)!r}."
            )
    else:
        invert_direction = [_legacy_invert_direction(entry)] * spec_count

    for spec_index, (spec_curve_ids, spec_invert_direction) in enumerate(
        zip(curve_ids, invert_direction)
    ):
        if "_group_invert_direction" in entry:
            group_invert_direction = _as_bool(
                _per_spec_value(
                    entry,
                    "_group_invert_direction",
                    spec_index,
                    spec_count,
                    False,
                )
            )
        else:
            group_invert_direction = _group_invert_direction(spec_invert_direction)
        n_pts, mesh_type, coef = _curve_spec_discretization(
            entry, spec_index, spec_count, spec_curve_ids
        )
        yield CurveSpec(
            spec_curve_ids,
            spec_invert_direction,
            group_invert_direction,
            n_pts,
            mesh_type,
            coef,
        )


def _curve_coef(mesh_type: str, base_coef: float, invert: bool) -> float:
    if mesh_type.lower() == "progression" and invert:
        if base_coef == 0.0:
            raise ValueError("Progression coefficient cannot be zero.")
        return 1.0 / base_coef

    return base_coef


def _automatic_circumferential_zone_name(entry: dict[str, Any]) -> str | None:
    if entry.get("_mesh_zone_curve_source") != "automatic":
        return None
    zone_name = entry.get("_mesh_zone_name")
    return None if zone_name is None else str(zone_name)


def _propagate_circumferential_element_size_specs(
    entry: dict[str, Any],
    curve_specs: list[CurveSpec],
    reference_specs_by_zone: dict[str, list[CurveSpec]],
) -> list[CurveSpec]:
    zone_name = _automatic_circumferential_zone_name(entry)
    if zone_name is None:
        return curve_specs

    reference_specs = reference_specs_by_zone.get(zone_name)
    if reference_specs is None:
        reference_specs_by_zone[zone_name] = curve_specs
        return curve_specs

    if len(reference_specs) != len(curve_specs):
        raise ValueError(
            f"Automatic circumferential loop {entry.get('name', entry)!r} in mesh zone "
            f"{zone_name!r} has {len(curve_specs)} curve groups, but the first "
            f"loop has {len(reference_specs)}."
        )

    spec_count = len(curve_specs)
    propagated_specs = []
    for index, curve_spec in enumerate(curve_specs):
        if _entry_mesh_size_mode(entry, index, spec_count) in (
            "ele size",
            "element size",
            "element_size",
        ):
            reference_spec = reference_specs[index]
            propagated_specs.append(
                replace(
                    curve_spec,
                    n_pts=reference_spec.n_pts,
                    mesh_type=reference_spec.mesh_type,
                    coef=reference_spec.coef,
                )
            )
        else:
            propagated_specs.append(curve_spec)

    return propagated_specs


def _curve_endpoints(curve_id: int) -> tuple[int, int]:
    boundary = gmsh.model.getBoundary([(1, abs(curve_id))], oriented=False)
    points = [tag for dim, tag in boundary if dim == 0]
    if len(points) != 2:
        raise ValueError(f"Curve {curve_id} does not have exactly two endpoints.")
    return points[0], points[1]


def _surface_boundary_curves(surface_id: int) -> list[int]:
    boundary = gmsh.model.getBoundary([(2, surface_id)], oriented=False)
    return [tag for dim, tag in boundary if dim == 1]


def _surface_corner_points(surface_id: int) -> list[int]:
    corners = []

    for curve_id in _surface_boundary_curves(surface_id):
        for point_id in _curve_endpoints(curve_id):
            if point_id not in corners:
                corners.append(point_id)

    return corners


def _corner_edge_index(
    endpoints: tuple[int, int],
    corner_edges: list[frozenset[int]],
) -> int | None:
    endpoint_set = frozenset(endpoints)
    for edge_index, corner_edge in enumerate(corner_edges):
        if endpoint_set == corner_edge:
            return edge_index
    return None


def _surface_edge_curves(surface_id: int, corners: list[int]) -> list[list[int]]:
    if len(corners) != 4:
        return [[], [], [], []]

    ordered_points, ordered_curves = _surface_ordered_boundary(surface_id)
    if len(ordered_points) < 2 or len(ordered_curves) < 1:
        return [[], [], [], []]

    if ordered_points[-1] == ordered_points[0]:
        ordered_points = ordered_points[:-1]
    corner_indices = [ordered_points.index(corner) for corner in corners]
    edge_curves = []

    for edge_index in range(4):
        start_index = corner_indices[edge_index]
        end_index = corner_indices[(edge_index + 1) % 4]
        curves = []
        cursor = start_index
        while cursor != end_index:
            curves.append(ordered_curves[cursor % len(ordered_curves)])
            cursor = (cursor + 1) % len(ordered_points)
        edge_curves.append(curves)

    return edge_curves


def _is_triangular_transfinite_surface(corners: list[int]) -> bool:
    return len(set(corners)) == 3


def _edge_divisions(
    curves: list[int],
    constraints: dict[int, CurveConstraint],
) -> int | None:
    divisions = 0
    for curve_id in curves:
        constraint = constraints.get(abs(curve_id))
        if constraint is None:
            return None
        divisions += constraint.n_pts - 1
    return divisions


def _curve_constraint_or_default(
    curve_id: int,
    constraints: dict[int, CurveConstraint],
) -> CurveConstraint:
    return constraints.get(abs(curve_id), CurveConstraint(2, "Progression", 1.0))


def _set_curve_constraint(
    curve_id: int,
    constraint: CurveConstraint,
    constraints: dict[int, CurveConstraint],
) -> None:
    gmsh.model.mesh.setTransfiniteCurve(
        curve_id,
        constraint.n_pts,
        constraint.mesh_type,
        constraint.coef,
    )
    constraints[abs(curve_id)] = constraint


def _short_curve_threshold(curve_lengths: list[float]) -> float:
    nonzero_lengths = sorted(length for length in curve_lengths if length > 0.0)
    if not nonzero_lengths:
        return 0.0
    median_length = nonzero_lengths[len(nonzero_lengths) // 2]
    return max(1.0e-9, median_length * 1.0e-3)


def _normalize_longitudinal_sequence_counts(
    mesh_def: dict[str, Any],
    constraints: dict[int, CurveConstraint],
) -> None:
    for entry in _iter_curve_entries(mesh_def.get("longitudinal_curve_sequences", [])):
        curve_ids = [
            abs(curve_id)
            for curve_group in _entry_curve_groups(entry)
            for curve_id in curve_group
        ]
        if not curve_ids:
            continue

        curve_lengths = [_curve_length(curve_id) for curve_id in curve_ids]
        short_threshold = _short_curve_threshold(curve_lengths)
        primary_curves = [
            curve_id
            for curve_id, curve_length in zip(curve_ids, curve_lengths)
            if curve_length > short_threshold
        ]
        if not primary_curves:
            continue

        target_n_pts = max(
            _curve_constraint_or_default(curve_id, constraints).n_pts
            for curve_id in primary_curves
        )
        target_n_pts = max(target_n_pts, 2)

        for curve_id, curve_length in zip(curve_ids, curve_lengths):
            current_constraint = _curve_constraint_or_default(curve_id, constraints)
            n_pts = current_constraint.n_pts if curve_length <= short_threshold else target_n_pts
            _set_curve_constraint(
                curve_id,
                CurveConstraint(
                    n_pts,
                    current_constraint.mesh_type,
                    current_constraint.coef,
                ),
                constraints,
            )


def apply_transfinite_curves(mesh_def: dict[str, Any]) -> dict[int, CurveConstraint]:
    constraints = {}
    circumferential_reference_specs_by_zone: dict[str, list[CurveSpec]] = {}

    for section_name in (
        "circumferential_curve_sequences",
        "longitudinal_curve_sequences",
        "explicit_curve_sequences",
    ):
        section = mesh_def.get(section_name, {})
        for entry in _iter_curve_entries(section):
            try:
                curve_specs = list(_iter_curve_specs(entry))
                if section_name == "circumferential_curve_sequences":
                    curve_specs = _propagate_circumferential_element_size_specs(
                        entry,
                        curve_specs,
                        circumferential_reference_specs_by_zone,
                    )
                if (
                    section_name == "longitudinal_curve_sequences"
                    and curve_specs
                    and _uses_element_size_mode(entry, len(curve_specs))
                ):
                    first_spec = curve_specs[0]
                    curve_specs = [
                        replace(
                            curve_spec,
                            n_pts=first_spec.n_pts,
                            mesh_type=first_spec.mesh_type,
                            coef=first_spec.coef,
                        )
                        for curve_spec in curve_specs
                    ]

                for curve_spec in curve_specs:
                    for curve_id, curve_n_pts, invert, curve_base_coef in _curve_node_counts(
                        curve_spec.curve_ids,
                        curve_spec.invert_directions,
                        curve_spec.group_invert_direction,
                        curve_spec.n_pts,
                        entry,
                        curve_spec.mesh_type,
                        curve_spec.coef,
                    ):
                        curve_coef = _curve_coef(
                            curve_spec.mesh_type, curve_base_coef, invert
                        )
                        gmsh.model.mesh.setTransfiniteCurve(
                            curve_id,
                            curve_n_pts,
                            curve_spec.mesh_type,
                            curve_coef,
                        )
                        constraints[abs(curve_id)] = CurveConstraint(
                            curve_n_pts, curve_spec.mesh_type, curve_coef
                        )
            except Exception as error:
                _raise_with_mesh_zone(entry, error)

    return constraints


def complete_surface_boundary_curves(
    mesh_def: dict[str, Any],
    constraints: dict[int, CurveConstraint],
) -> None:
    for _ in range(100):
        previous_counts = {
            curve_id: constraint.n_pts
            for curve_id, constraint in constraints.items()
        }
        _complete_surface_boundary_curves_once(mesh_def, constraints)
        current_counts = {
            curve_id: constraint.n_pts
            for curve_id, constraint in constraints.items()
        }
        if current_counts == previous_counts:
            return

    warnings.warn(
        "Surface boundary curve completion did not converge after 100 iterations.",
        stacklevel=2,
    )


def _complete_surface_boundary_curves_once(
    mesh_def: dict[str, Any],
    constraints: dict[int, CurveConstraint],
) -> None:
    section = mesh_def.get("transfinite_surfaces", {})
    unstructured_surfaces = _unstructured_surface_ids(mesh_def)
    circumferential_curve_loop_indices = _circumferential_curve_loop_indices(mesh_def)
    circumferential_curve_groups = _circumferential_curve_groups(mesh_def)
    protected_curve_counts = _protected_curve_counts(mesh_def)

    for surface in _iter_surface_entries(section):
        try:
            _complete_surface_boundary_curves_for_surface(
                mesh_def,
                surface,
                constraints,
                unstructured_surfaces,
                circumferential_curve_loop_indices,
                circumferential_curve_groups,
                protected_curve_counts,
            )
        except Exception as error:
            _raise_with_mesh_zone(surface, error)


def _complete_surface_boundary_curves_for_surface(
    mesh_def: dict[str, Any],
    surface: dict[str, Any],
    constraints: dict[int, CurveConstraint],
    unstructured_surfaces: set[int],
    circumferential_curve_loop_indices: dict[int, int],
    circumferential_curve_groups: dict[int, list[int]],
    protected_curve_counts: set[int],
) -> None:
    surface_id = int(surface["id"])
    if surface_id in unstructured_surfaces:
        return

    corners = [int(point) for point in surface.get("boundary points", [])]

    if len(corners) != 4:
        return

    if _is_triangular_transfinite_surface(corners):
        return

    edge_curves = _surface_edge_curves(surface_id, corners)
    if _has_protected_opposite_edge_mismatch(
        edge_curves, constraints, protected_curve_counts
    ):
        _add_unstructured_surface(mesh_def, surface_id)
        unstructured_surfaces.add(surface_id)
        return

    for edge_index, curves in enumerate(edge_curves):
        if len(curves) != 1 or curves[0] in constraints:
            continue

        opposite_curves = edge_curves[(edge_index + 2) % 4]
        if len(opposite_curves) != 1:
            continue

        opposite_constraint = constraints.get(opposite_curves[0])
        if opposite_constraint is None:
            continue

        curve_id = curves[0]
        _set_curve_constraint(
            curve_id,
            opposite_constraint,
            constraints,
        )

    _align_opposite_edge_divisions(
        edge_curves,
        constraints,
        circumferential_curve_loop_indices,
        circumferential_curve_groups,
        protected_curve_counts,
    )


def _circumferential_curve_loop_indices(mesh_def: dict[str, Any]) -> dict[int, int]:
    loop_indices = {}
    for loop_index, entry in enumerate(
        _iter_curve_entries(mesh_def.get("circumferential_curve_sequences", []))
    ):
        for curve_group in entry.get("curve_ids", []):
            for curve_id in _as_list(curve_group):
                loop_indices[abs(curve_id)] = loop_index
    return loop_indices


def _circumferential_curve_groups(mesh_def: dict[str, Any]) -> dict[int, list[int]]:
    curve_groups = {}
    for entry in _iter_curve_entries(mesh_def.get("circumferential_curve_sequences", [])):
        for curve_group in entry.get("curve_ids", []):
            group = [abs(curve_id) for curve_id in _as_list(curve_group)]
            for curve_id in group:
                curve_groups[curve_id] = group
    return curve_groups


def _protected_curve_counts(mesh_def: dict[str, Any]) -> set[int]:
    protected_curves = set()
    sections = (
        mesh_def.get("longitudinal_curve_sequences", []),
        mesh_def.get("explicit_curve_sequences", []),
    )
    for section in sections:
        for entry in _iter_curve_entries(section):
            curve_specs = list(_iter_curve_specs(entry))
            spec_count = len(curve_specs)
            for index, curve_spec in enumerate(curve_specs):
                if _entry_mesh_size_mode(entry, index, spec_count) != "npts":
                    continue
                for curve_id in _as_list(curve_spec.curve_ids):
                    protected_curves.add(abs(curve_id))
    return protected_curves


def _has_protected_opposite_edge_mismatch(
    edge_curves: list[list[int]],
    constraints: dict[int, CurveConstraint],
    protected_curve_counts: set[int],
) -> bool:
    for edge_index in range(2):
        curves_a = edge_curves[edge_index]
        curves_b = edge_curves[edge_index + 2]
        divisions_a = _edge_divisions(curves_a, constraints)
        divisions_b = _edge_divisions(curves_b, constraints)
        if divisions_a is None or divisions_b is None or divisions_a == divisions_b:
            continue

        protected_a = any(abs(curve_id) in protected_curve_counts for curve_id in curves_a)
        protected_b = any(abs(curve_id) in protected_curve_counts for curve_id in curves_b)
        if protected_a and protected_b:
            return True

    return False


def _add_unstructured_surface(mesh_def: dict[str, Any], surface_id: int) -> None:
    unstructured_surfaces = {
        int(existing_surface_id)
        for existing_surface_id in mesh_def.get("unstructured_surfaces", [])
    }
    unstructured_surfaces.add(int(surface_id))
    mesh_def["unstructured_surfaces"] = sorted(unstructured_surfaces)


def _align_opposite_edge_divisions(
    edge_curves: list[list[int]],
    constraints: dict[int, CurveConstraint],
    circumferential_curve_loop_indices: dict[int, int],
    circumferential_curve_groups: dict[int, list[int]],
    protected_curve_counts: set[int],
) -> None:
    for edge_index in range(2):
        curves_a = edge_curves[edge_index]
        curves_b = edge_curves[edge_index + 2]
        divisions_a = _edge_divisions(curves_a, constraints)
        divisions_b = _edge_divisions(curves_b, constraints)
        if divisions_a is None or divisions_b is None or divisions_a == divisions_b:
            continue

        loop_indices_a = [
            circumferential_curve_loop_indices.get(abs(curve_id)) for curve_id in curves_a
        ]
        loop_indices_b = [
            circumferential_curve_loop_indices.get(abs(curve_id)) for curve_id in curves_b
        ]
        circumferential_a = all(loop_index is not None for loop_index in loop_indices_a)
        circumferential_b = all(loop_index is not None for loop_index in loop_indices_b)

        if circumferential_a and circumferential_b:
            _align_opposite_circumferential_edge_divisions(
                curves_a,
                divisions_a,
                loop_indices_a,
                curves_b,
                divisions_b,
                loop_indices_b,
                constraints,
                circumferential_curve_groups,
            )
            continue
        if circumferential_a:
            source_divisions = divisions_a
            target_edge_divisions = divisions_b
            target_curves = curves_b
        elif circumferential_b:
            source_divisions = divisions_b
            target_edge_divisions = divisions_a
            target_curves = curves_a
        else:
            source_divisions = max(divisions_a, divisions_b)
            target_edge_divisions = min(divisions_a, divisions_b)
            target_curves = curves_a if divisions_a < divisions_b else curves_b

        non_circumferential_target_curves = [
            abs(curve_id)
            for curve_id in target_curves
            if abs(curve_id) not in circumferential_curve_loop_indices
            and abs(curve_id) not in protected_curve_counts
        ]
        if not non_circumferential_target_curves:
            warnings.warn(
                "Could not align opposite surface edge divisions without changing "
                f"circumferential or protected curves {target_curves}. Leaving counts "
                "unchanged.",
                stacklevel=2,
            )
            continue

        target_curve = max(
            non_circumferential_target_curves,
            key=_curve_length,
        )

        target_constraint = constraints.get(target_curve)
        if target_constraint is None:
            continue

        missing_divisions = source_divisions - target_edge_divisions
        if missing_divisions == 0:
            continue

        target_n_pts = max(2, target_constraint.n_pts + missing_divisions)
        _set_curve_constraint(
            target_curve,
            CurveConstraint(
                target_n_pts,
                target_constraint.mesh_type,
                target_constraint.coef,
            ),
            constraints,
        )


def _align_opposite_circumferential_edge_divisions(
    curves_a: list[int],
    divisions_a: int,
    loop_indices_a: list[int | None],
    curves_b: list[int],
    divisions_b: int,
    loop_indices_b: list[int | None],
    constraints: dict[int, CurveConstraint],
    circumferential_curve_groups: dict[int, list[int]],
) -> None:
    mean_loop_a = sum(
        loop_index for loop_index in loop_indices_a if loop_index is not None
    ) / len(loop_indices_a)
    mean_loop_b = sum(
        loop_index for loop_index in loop_indices_b if loop_index is not None
    ) / len(loop_indices_b)
    if mean_loop_a == mean_loop_b:
        return

    if mean_loop_a > mean_loop_b:
        source_divisions = divisions_b
        target_edge_divisions = divisions_a
        target_curves = curves_a
    else:
        source_divisions = divisions_a
        target_edge_divisions = divisions_b
        target_curves = curves_b

    missing_divisions = source_divisions - target_edge_divisions
    if missing_divisions == 0:
        return

    target_curve = max((abs(curve_id) for curve_id in target_curves), key=_curve_length)
    target_group = circumferential_curve_groups.get(target_curve, [target_curve])
    compensation_curves = [
        curve_id
        for curve_id in target_group
        if curve_id != target_curve
        and curve_id not in {abs(target_item) for target_item in target_curves}
    ]
    if not compensation_curves:
        warnings.warn(
            "Opposite circumferential surface edges have inconsistent divisions, but "
            f"curve {target_curve} is not part of a larger grouped curve. "
            "Leaving circumferential group total unchanged.",
            stacklevel=2,
        )
        return

    target_constraint = constraints.get(target_curve)
    if target_constraint is None:
        return

    compensation_curve = max(
        compensation_curves,
        key=lambda curve_id: constraints.get(
            curve_id, CurveConstraint(2, "Progression", 1.0)
        ).n_pts,
    )
    compensation_constraint = constraints.get(compensation_curve)
    if compensation_constraint is None:
        return

    target_n_pts = max(2, target_constraint.n_pts + missing_divisions)
    compensation_n_pts = compensation_constraint.n_pts - missing_divisions
    if compensation_n_pts < 2:
        warnings.warn(
            "Cannot align circumferential subcurve divisions while preserving grouped "
            f"curve total for group {target_group}. Leaving counts unchanged.",
            stacklevel=2,
        )
        return

    _set_curve_constraint(
        target_curve,
        CurveConstraint(
            target_n_pts,
            target_constraint.mesh_type,
            target_constraint.coef,
        ),
        constraints,
    )
    _set_curve_constraint(
        compensation_curve,
        CurveConstraint(
            compensation_n_pts,
            compensation_constraint.mesh_type,
            compensation_constraint.coef,
        ),
        constraints,
    )


def _unstructured_surface_ids(mesh_def: dict[str, Any]) -> set[int]:
    return {
        int(surface_id)
        for surface_id in mesh_def.get("unstructured_surfaces", [])
    }.union(_surface_meshing_algorithm_surface_ids(mesh_def))


def _transfinite_surface_definitions(mesh_def: dict[str, Any]) -> dict[int, dict[str, Any]]:
    section = mesh_def.get("transfinite_surfaces", {})
    return {int(surface["id"]): surface for surface in _iter_surface_entries(section)}


def apply_transfinite_surfaces(mesh_def: dict[str, Any]) -> None:
    unstructured_surfaces = _unstructured_surface_ids(mesh_def)
    surface_definitions = _transfinite_surface_definitions(mesh_def)

    for surface_id, surface in surface_definitions.items():
        try:
            if surface_id in unstructured_surfaces:
                continue

            arrangement = str(surface.get("Arrangement", "Left")).capitalize()
            boundary_points = [int(point) for point in surface.get("boundary points", [])]

            gmsh.model.mesh.setTransfiniteSurface(
                surface_id, arrangement, boundary_points
            )
        except Exception as error:
            _raise_with_mesh_zone(surface, error)


def apply_geometry_healing(mesh_def: dict[str, Any]) -> None:
    healing_def = mesh_def.get("geometry_healing", {})
    if not healing_def or not bool(healing_def.get("enabled", False)):
        return

    gmsh.model.occ.healShapes(
        [],
        float(healing_def.get("tolerance", 1.0e-6)),
        bool(healing_def.get("fix_degenerated", True)),
        bool(healing_def.get("fix_small_edges", True)),
        bool(healing_def.get("fix_small_faces", True)),
        bool(healing_def.get("sew_faces", True)),
        bool(healing_def.get("make_solids", False)),
    )


def _as_point(value: Sequence[float], *, name: str) -> list[float]:
    point = [float(item) for item in value]
    if len(point) != 3:
        raise ValueError(f"{name} must be a 3D point/vector, got {value!r}.")
    return point


def _as_xyz_values(value: Any, *, name: str, default: float) -> list[float]:
    if value is None:
        return [default, default, default]

    if isinstance(value, (int, float)):
        scalar = float(value)
        return [scalar, scalar, scalar]

    if isinstance(value, dict):
        return [
            float(value.get("x", default)),
            float(value.get("y", default)),
            float(value.get("z", default)),
        ]

    values = [float(item) for item in value]
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar, x/y/z object, or 3-item list.")
    return values


def _imported_geometry_dim_tags(imported_entities: list[tuple[int, int]]) -> list[tuple[int, int]]:
    dim_tags = list(imported_entities or []) or gmsh.model.occ.getEntities()
    if not dim_tags:
        return []

    max_dim = max(dim for dim, _ in dim_tags)
    return [(dim, tag) for dim, tag in dim_tags if dim == max_dim]


def apply_geometry_preprocessing(
    mesh_def: dict[str, Any],
    imported_entities: list[tuple[int, int]],
) -> None:
    preprocessing_def = mesh_def.get("geometry_preprocessing", {})
    if not preprocessing_def or not bool(preprocessing_def.get("enabled", True)):
        return

    dim_tags = _imported_geometry_dim_tags(imported_entities)
    if not dim_tags:
        return

    origin = _as_point(preprocessing_def.get("origin", [0.0, 0.0, 0.0]), name="origin")
    scale = _as_xyz_values(preprocessing_def.get("scale"), name="scale", default=1.0)
    if any(abs(value) <= 0.0 for value in scale):
        raise ValueError("geometry_preprocessing scale values must be non-zero.")

    if any(not math.isclose(value, 1.0) for value in scale):
        gmsh.model.occ.dilate(dim_tags, *origin, *scale)

    rotation_deg = preprocessing_def.get(
        "rotation_deg",
        preprocessing_def.get("rotation_degrees", preprocessing_def.get("rotation")),
    )
    rotation_rad = preprocessing_def.get("rotation_rad")
    if rotation_rad is not None:
        rotations = _as_xyz_values(rotation_rad, name="rotation_rad", default=0.0)
    else:
        rotations = [
            math.radians(value)
            for value in _as_xyz_values(rotation_deg, name="rotation_deg", default=0.0)
        ]

    axes = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for angle, axis in zip(rotations, axes):
        if math.isclose(angle, 0.0):
            continue
        gmsh.model.occ.rotate(dim_tags, *origin, *axis, angle)

    translation = _as_point(
        preprocessing_def.get("translation", [0.0, 0.0, 0.0]),
        name="translation",
    )
    if any(not math.isclose(value, 0.0) for value in translation):
        gmsh.model.occ.translate(dim_tags, *translation)


def _surface_meshing_algorithm_code(algorithm: Any) -> int:
    if isinstance(algorithm, int):
        return algorithm

    algorithm_name = str(algorithm).strip().lower()
    algorithm_map = {
        "meshadapt": 1,
        "automatic": 2,
        "initialmeshonly": 3,
        "delaunay": 5,
        "frontal-delaunay": 6,
        "frontaldelaunay": 6,
        "bamg": 7,
        "frontal-delaunay for quads": 8,
        "frontaldelaunayforquads": 8,
        "packing of parallelograms": 9,
        "packingofparallelograms": 9,
        "quasi-structured quad": 11,
        "quasistructuredquad": 11,
    }
    if algorithm_name not in algorithm_map:
        raise ValueError(
            f"Unsupported surface meshing algorithm {algorithm!r}. "
            f"Supported names: {sorted(algorithm_map)}."
        )
    return algorithm_map[algorithm_name]


def apply_surface_meshing_algorithms(mesh_def: dict[str, Any]) -> None:
    entries = mesh_def.get("surface_meshing_algorithms", [])
    if not entries:
        return
    if not isinstance(entries, list):
        raise TypeError("surface_meshing_algorithms must be a list.")

    surface_ids = {surface_id for _, surface_id in gmsh.model.getEntities(2)}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(
                "Each surface_meshing_algorithms entry must be an object."
            )

        target_surfaces = [int(surface_id) for surface_id in entry.get("surfaces", [])]
        if not target_surfaces:
            continue

        unknown_surfaces = [
            surface_id for surface_id in target_surfaces if surface_id not in surface_ids
        ]
        if unknown_surfaces:
            raise ValueError(
                f"surface_meshing_algorithms references unknown surfaces "
                f"{unknown_surfaces}."
            )

        algorithm_code = _surface_meshing_algorithm_code(entry["algorithm"])
        for surface_id in target_surfaces:
            gmsh.model.mesh.setAlgorithm(2, surface_id, algorithm_code)


def apply_structured_surface_recombination(mesh_def: dict[str, Any]) -> None:
    unstructured_surfaces = _unstructured_surface_ids(mesh_def)
    structured_surface_ids = [
        surface_id
        for _, surface_id in gmsh.model.getEntities(2)
        if surface_id not in unstructured_surfaces
    ]
    for surface_id in structured_surface_ids:
        gmsh.model.mesh.setRecombine(2, surface_id)


def _surface_is_on_y0_plane(surface_id: int, tolerance: float = 1.0e-3) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surface_id)
    return abs(ymin) <= tolerance and abs(ymax) <= tolerance


def _unique_physical_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    suffix = 2
    while f"{name}_{suffix}" in used_names:
        suffix += 1

    unique_name = f"{name}_{suffix}"
    used_names.add(unique_name)
    return unique_name


def _curve_to_circumferential_group_map(
    mesh_def: dict[str, Any],
) -> dict[int, tuple[int, int]]:
    curve_map = {}
    for loop_index, entry in enumerate(
        _iter_curve_entries(mesh_def.get("circumferential_curve_sequences", [])),
        start=1,
    ):
        for group_index, curve_group in enumerate(_entry_curve_groups(entry), start=1):
            for curve_id in curve_group:
                curve_map[abs(curve_id)] = (loop_index, group_index)
    return curve_map


def _curve_to_longitudinal_section_map(mesh_def: dict[str, Any]) -> dict[int, int]:
    curve_map = {}
    for section_index, entry in enumerate(
        _iter_curve_entries(mesh_def.get("longitudinal_curve_sequences", [])),
        start=1,
    ):
        for curve_group in _entry_curve_groups(entry):
            for curve_id in curve_group:
                curve_map[abs(curve_id)] = section_index
    return curve_map


def _blunt_trailing_edge_group_index(mesh_def: dict[str, Any]) -> int | None:
    circumferential_entries = list(_iter_curve_entries(mesh_def.get("circumferential_curve_sequences", [])))
    if not circumferential_entries:
        return None
    return max(len(_entry_curve_groups(entry)) for entry in circumferential_entries)


def _structured_surface_export_names(mesh_def: dict[str, Any]) -> dict[int, str]:
    surface_definitions = _transfinite_surface_definitions(mesh_def)
    circumferential_curve_map = _curve_to_circumferential_group_map(mesh_def)
    longitudinal_curve_map = _curve_to_longitudinal_section_map(mesh_def)
    blunt_group_index = _blunt_trailing_edge_group_index(mesh_def)
    structured_names = {}

    for surface_id, surface_def in surface_definitions.items():
        corners = [int(point) for point in surface_def.get("boundary points", [])]
        if len(corners) != 4:
            continue

        boundary_curves = [abs(curve_id) for curve_id in _surface_boundary_curves(surface_id)]
        circumferential_groups = {
            circumferential_curve_map[curve_id][1]
            for curve_id in boundary_curves
            if curve_id in circumferential_curve_map
        }
        longitudinal_sections = {
            longitudinal_curve_map[curve_id]
            for curve_id in boundary_curves
            if curve_id in longitudinal_curve_map
        }

        if len(circumferential_groups) != 1 or len(longitudinal_sections) != 1:
            continue

        circumferential_index = next(iter(circumferential_groups))
        longitudinal_index = next(iter(longitudinal_sections))
        if blunt_group_index is not None and circumferential_index == blunt_group_index:
            structured_names[surface_id] = f"struct_bluntTE_S{longitudinal_index:02d}"
        else:
            structured_names[surface_id] = (
                f"struct_surf_S{longitudinal_index:02d}_C{circumferential_index:02d}"
            )

    return structured_names


def apply_export_surface_names(mesh_def: dict[str, Any]) -> None:
    symmetry_surfaces = []
    other_surfaces = []
    tolerance = 1.0e-3
    structured_surface_names = _structured_surface_export_names(mesh_def)

    for _, surface_id in gmsh.model.getEntities(2):
        if _surface_is_on_y0_plane(surface_id, tolerance):
            symmetry_surfaces.append(surface_id)
        else:
            other_surfaces.append(surface_id)

    physical_groups = gmsh.model.getPhysicalGroups(2)
    for _, physical_tag in physical_groups:
        gmsh.model.removePhysicalGroups([(2, physical_tag)])

    used_names = set()
    if symmetry_surfaces:
        physical_tag = gmsh.model.addPhysicalGroup(2, symmetry_surfaces)
        gmsh.model.setPhysicalName(
            2,
            physical_tag,
            _unique_physical_name("symm face", used_names),
        )

    named_surface_groups: dict[str, list[int]] = {}
    unnamed_surfaces: list[int] = []
    for surface_id in other_surfaces:
        entity_name = structured_surface_names.get(surface_id)
        if entity_name is None:
            unnamed_surfaces.append(surface_id)
            continue
        named_surface_groups.setdefault(entity_name, []).append(surface_id)

    for entity_name, surface_ids in named_surface_groups.items():
        physical_tag = gmsh.model.addPhysicalGroup(2, surface_ids)
        gmsh.model.setPhysicalName(
            2,
            physical_tag,
            _unique_physical_name(entity_name, used_names),
        )

    for surface_id in unnamed_surfaces:
        physical_tag = gmsh.model.addPhysicalGroup(2, [surface_id])
        entity_name = gmsh.model.getEntityName(2, surface_id).strip()
        if not entity_name:
            entity_name = f"surface_{surface_id:05d}"
        gmsh.model.setPhysicalName(
            2,
            physical_tag,
            _unique_physical_name(entity_name, used_names),
        )


def show_gmsh() -> None:
    """Open the Gmsh GUI for inspecting the generated mesh."""
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.fltk.run()


def _mesh_zone_name_from_error(error: BaseException) -> str | None:
    current: BaseException | None = error
    while current is not None:
        zone_name = getattr(current, "zone_name", None)
        if zone_name is not None:
            return str(zone_name)
        current = current.__cause__
    return None


def _root_error_message(error: BaseException) -> str:
    current: BaseException = error
    while isinstance(current, MeshZoneError) and current.__cause__ is not None:
        current = current.__cause__
    return str(current)


def show_geometry_after_error(error: Exception) -> None:
    """Open Gmsh with geometry only, preserving the original failure."""
    zone_name = _mesh_zone_name_from_error(error)
    prefix = f"Mesh zone {zone_name!r}: " if zone_name is not None else ""
    print(
        f"{prefix}Mesh generation failed. Opening Gmsh with the imported geometry for "
        f"inspection before re-raising the error: {_root_error_message(error)}"
    )
    try:
        gmsh.model.mesh.clear()
        gmsh.model.occ.synchronize()
        show_gmsh()
    except Exception as show_error:
        print(f"Could not open Gmsh for error inspection: {show_error}")


def load_mesh_def(mesh_def_file: Path) -> dict[str, Any]:
    with mesh_def_file.open("r", encoding="utf-8") as file:
        return normalize_mesh_def(json.load(file))


def normalize_mesh_def(mesh_def: dict[str, Any]) -> dict[str, Any]:
    expected_sections = {"geometry definition", "mesh definition"}
    actual_sections = set(mesh_def)
    if actual_sections != expected_sections:
        missing_sections = sorted(expected_sections - actual_sections)
        extra_sections = sorted(actual_sections - expected_sections)
        details = []
        if missing_sections:
            details.append(f"missing {missing_sections}")
        if extra_sections:
            details.append(f"unexpected top-level keys {extra_sections}")
        raise ValueError(
            "Mesh definition JSON must contain exactly the top-level sections "
            "'geometry definition' and 'mesh definition'"
            + (f" ({'; '.join(details)})." if details else ".")
        )

    geometry_def = mesh_def["geometry definition"]
    mesh_settings = mesh_def["mesh definition"]

    if not isinstance(geometry_def, dict):
        raise TypeError("'geometry definition' must be an object.")
    if not isinstance(mesh_settings, dict):
        raise TypeError("'mesh definition' must be an object.")

    normalized: dict[str, Any] = {}
    normalized.update(copy.deepcopy(geometry_def))
    normalized.update(copy.deepcopy(mesh_settings))

    top_level_curve_sequence_keys = {
        "automatic_curve_sequences",
        "circumferential_curve_sequences",
        "longitudinal_curve_sequences",
        "explicit_curve_sequences",
    }
    deprecated_keys = sorted(top_level_curve_sequence_keys.intersection(normalized))
    if deprecated_keys:
        raise ValueError(
            "Old top-level curve sequence keys are no longer supported: "
            f"{deprecated_keys}. Define curve sequences inside 'mesh_zones' instead."
        )
    if "mesh_zones" not in normalized:
        raise ValueError("Mesh definition must define curve sequences in 'mesh_zones'.")

    return normalized


def mesh_def_step_file(mesh_def: dict[str, Any], mesh_def_file: Path) -> Path:
    step_file_value = mesh_def.get("step_file")
    if not step_file_value:
        raise ValueError(
            f"No STEP file specified. Add 'step_file' to {mesh_def_file} "
            "or pass --step."
        )

    step_file = Path(str(step_file_value))
    if not step_file.is_absolute():
        step_file = mesh_def_file.resolve().parent / step_file
    return step_file


def generate_surface_mesh(
    step_file: Path,
    mesh_def: dict[str, Any],
    output_file: Path | None,
    *,
    recombine: bool = True,
    mesh_format: str = "msh2",
    show: bool = True,
    mesh: bool = True,
) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        if output_file is not None and output_file.suffix.lower() == ".msh":
            gmsh.option.setNumber(
                "Mesh.MshFileVersion", 2.2 if mesh_format == "msh2" else 4.1
        )

        gmsh.model.add(step_file.stem)
        imported_entities = gmsh.model.occ.importShapes(str(step_file))
        apply_geometry_preprocessing(mesh_def, imported_entities)
        apply_geometry_healing(mesh_def)
        gmsh.model.occ.synchronize()

        if not mesh:
            if show:
                show_gmsh()
            return

        generate_1d_mesh = bool(mesh_def.get("generate_1d_mesh", True))
        generate_2d_mesh = bool(mesh_def.get("generate_2d_mesh", True))
        if generate_2d_mesh and not generate_1d_mesh:
            raise ValueError(
                "generate_2d_mesh requires generate_1d_mesh, because Gmsh "
                "builds surface meshes from curve meshes."
            )
        if not generate_1d_mesh and not generate_2d_mesh:
            if show:
                show_gmsh()
            return

        mesh_def = expand_mesh_zones(mesh_def)
        curve_constraints = apply_transfinite_curves(mesh_def)
        if generate_2d_mesh:
            mesh_def = apply_automatic_transfinite_surfaces(mesh_def)
            complete_surface_boundary_curves(mesh_def, curve_constraints)
            apply_transfinite_surfaces(mesh_def)
            apply_surface_meshing_algorithms(mesh_def)
            if recombine:
                apply_structured_surface_recombination(mesh_def)

        gmsh.model.mesh.generate(2 if generate_2d_mesh else 1)

        if generate_2d_mesh:
            apply_export_surface_names(mesh_def)
        if output_file is not None:
            gmsh.write(str(output_file))

        if show:
            show_gmsh()
    except Exception as error:
        if show:
            show_geometry_after_error(error)
        raise
    finally:
        gmsh.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Gmsh surface mesh from a STEP file and JSON transfinite definitions."
    )
    parser.add_argument(
        "--step",
        type=Path,
        default=None,
        help="Path to the STEP file. Overrides the 'step_file' value in the JSON file.",
    )
    parser.add_argument(
        "--mesh-def",
        type=Path,
        default=DEFAULT_MESH_DEF_FILE,
        help="Path to the JSON mesh definition file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output mesh path. Defaults to <step-file>.cgns.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Generate the mesh but do not write an output file.",
    )
    parser.add_argument(
        "--recombine",
        action="store_true",
        dest="recombine",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-recombine",
        action="store_false",
        dest="recombine",
        help="Do not recombine 2D mesh elements after mesh generation.",
    )
    parser.set_defaults(recombine=True)
    parser.add_argument(
        "--format",
        choices=("msh2", "msh4"),
        default="msh2",
        help="Gmsh MSH file format version.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the Gmsh GUI after writing the mesh.",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Only import and optionally heal the STEP geometry; do not generate a mesh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_def = load_mesh_def(args.mesh_def)
    step_file = args.step or mesh_def_step_file(mesh_def, args.mesh_def)
    output_file = None if args.no_write else (args.out or step_file.with_suffix(".cgns"))

    generate_surface_mesh(
        step_file,
        mesh_def,
        output_file,
        recombine=args.recombine,
        mesh_format=args.format,
        show=not args.no_show,
        mesh=not args.no_mesh,
    )

    if args.no_mesh:
        print("Imported geometry without generating a mesh.")
    elif output_file is None:
        print("Generated mesh without writing an output file.")
    else:
        print(f"Wrote surface mesh to {output_file}")


if __name__ == "__main__":
    main()
