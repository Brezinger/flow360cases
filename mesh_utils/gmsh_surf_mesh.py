from __future__ import annotations

import argparse
import copy
import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import gmsh


DEFAULT_STEP_FILE = Path(
    r"C:\OneDrive\OneDrive - Achleitner Aerospace GmbH"
    r"\Achleitner Aerospace GmbH Allgemein - General"
    r"\01_Projekte\04_Flaplets\03_CAD\Aerodynamikmodelle"
    r"\Ventus3 FlapletV2\WKS\Ventus3_FlapletV2_WKS_B.stp"
)
DEFAULT_MESH_DEF_FILE = (
    Path(__file__).resolve().parent.parent / "V3" / "msh_def_FlapletV2_WKS.json"
)


class CurveConstraint:
    def __init__(self, n_pts: int, mesh_type: str, coef: float) -> None:
        self.n_pts = n_pts
        self.mesh_type = mesh_type
        self.coef = coef


@dataclass(frozen=True)
class CurveSpec:
    curve_ids: int | list[int]
    invert_directions: bool | list[bool]
    n_pts: int
    mesh_type: str
    coef: float


@dataclass(frozen=True)
class TracedLoop:
    curve_ids: list[int]
    points: list[int]
    blunt_curve_id: int


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
        if mesh_type.lower() != "progression":
            raise ValueError(
                f"Element-size mode currently requires Progression, got "
                f"{mesh_type!r} in {entry.get('name', entry)!r}."
            )

        curve_length = _curve_group_length(curve_ids)
        target_size_1 = float(
            _per_spec_value(entry, "target ele size 1", index, spec_count)
        )
        target_size_2 = float(
            _per_spec_value(entry, "target ele size 2", index, spec_count)
        )
        n_pts, coef = _progression_from_endpoint_sizes(
            curve_length, target_size_1, target_size_2
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


def _curve_node_counts(
    curve_ids: int | list[int],
    invert_directions: bool | list[bool],
    n_pts: int,
    entry: dict[str, Any],
    mesh_type: str | None = None,
    coef: float | None = None,
) -> list[tuple[int, int, bool]]:
    curve_group = _as_list(curve_ids)
    invert_group = _as_bool_list(invert_directions, len(curve_group))

    if len(curve_group) == 1:
        return [(curve_group[0], n_pts, invert_group[0])]

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

    # The first sub-curve defines the virtual grouped curve direction. The
    # per-sub-curve booleans are still applied below when setting Gmsh's local
    # progression coefficient on each CAD curve.
    node_positions = _distribution_node_positions(
        entry, n_pts, invert_group[0], mesh_type, coef
    )
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
    element_counts = [
        end_index - start_index
        for start_index, end_index in zip(split_indices, split_indices[1:])
    ]

    return [
        (curve_id, n_elements + 1, invert)
        for curve_id, n_elements, invert in zip(curve_group, element_counts, invert_group)
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
    natural_start, natural_end = _oriented_curve_endpoints(curve_id)
    if natural_start == start_point and natural_end == end_point:
        return False
    if natural_start == end_point and natural_end == start_point:
        return True

    traversal_vector = _point_vector(start_point, end_point)
    tangent = _curve_tangent_at_point(curve_id, start_point)
    if tangent is not None:
        return _dot(tangent, traversal_vector) < 0.0

    natural_vector = _point_vector(natural_start, natural_end)
    return _dot(natural_vector, traversal_vector) < 0.0


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

        curve_unit = _normalized(curve_vector)
        if (
            reject_unit is not None
            and abs(_dot(curve_unit, reject_unit)) > reject_alignment
        ):
            continue

        score = _dot(curve_unit, direction)
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


def _trace_airfoil_loop(
    start_point: int,
    chordwise_direction: Sequence[float],
    blunt_direction: Sequence[float],
    spanwise_direction: Sequence[float],
    excluded_curves: set[int],
) -> TracedLoop:
    current_point = int(start_point)
    pointing = _as_vector(chordwise_direction)
    curve_ids = []
    points = [current_point]
    local_excluded = set(excluded_curves)

    while True:
        if curve_ids:
            blunt_curve = _connecting_curve(current_point, start_point)
            if blunt_curve is not None and blunt_curve not in local_excluded:
                blunt_vector = _point_vector(current_point, start_point)
                if (
                    abs(_dot(_normalized(blunt_vector), _normalized(blunt_direction)))
                    > 0.2
                ):
                    return TracedLoop(curve_ids, points, blunt_curve)

        selected = _select_adjacent_curve(
            current_point,
            pointing,
            local_excluded,
            min_score=-1.0,
            reject_direction=spanwise_direction,
            reject_alignment=0.85,
        )
        if selected is None:
            raise ValueError(
                f"Could not continue airfoil loop trace from point {current_point}."
            )

        curve_id, next_point, curve_vector = selected
        curve_ids.append(curve_id)
        points.append(next_point)
        local_excluded.add(abs(curve_id))
        current_point = next_point
        pointing = curve_vector


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


def _infer_n_subcurvs(
    traced_loop: TracedLoop,
    split_spanwise_curve_count: int,
) -> list[int]:
    loop_curve_set = {
        abs(curve_id)
        for curve_id in traced_loop.curve_ids + [traced_loop.blunt_curve_id]
    }
    group_sizes = []
    current_group_size = 1

    for curve_index, point_id in enumerate(traced_loop.points[1:-1], start=1):
        adjacent_non_loop_curves = [
            curve_id
            for curve_id in _adjacent_curve_ids(point_id)
            if abs(curve_id) not in loop_curve_set
        ]

        if len(adjacent_non_loop_curves) >= split_spanwise_curve_count:
            group_sizes.append(current_group_size)
            current_group_size = 1
        else:
            current_group_size += 1

    group_sizes.append(current_group_size)
    group_sizes.append(1)
    return group_sizes


def _template_group_count(template_entry: dict[str, Any]) -> int | None:
    for key in ("n_pts", "type", "Parameter", "invert_direction", "_invert_direction"):
        value = template_entry.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _infer_n_subcurvs_for_template(
    traced_loop: TracedLoop,
    template_entry: dict[str, Any],
    default_split_spanwise_curve_count: int,
    previous_group_sizes: list[int] | None = None,
) -> list[int]:
    target_group_count = _template_group_count(template_entry)
    candidate_thresholds = [default_split_spanwise_curve_count, 1, 2, 3]
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


def _trace_spanwise_curves(
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
            current_point = next_point
            current_direction = curve_vector
            last_vector = curve_vector

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


def _automatic_config(mesh_def: dict[str, Any]) -> dict[str, Any] | None:
    config = mesh_def.get("automatic_curve_sequences")
    if config is None:
        return None
    if config is True:
        return {}
    if not isinstance(config, dict):
        raise TypeError("automatic_curve_sequences must be an object or true.")
    if config.get("enabled", True) is False:
        return None
    return config


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


def _automatic_chordwise_templates(
    config: dict[str, Any],
    loop_count: int | None = None,
) -> list[dict[str, Any]]:
    chordwise_def = config.get("chordwise_transfinite_def", {})
    if not chordwise_def:
        return []

    loop_defs = chordwise_def.get("loops")
    if isinstance(loop_defs, list) and loop_defs:
        return [
            _merge_transfinite_template(chordwise_def, loop_def, {"loops"})
            for loop_def in loop_defs
        ]

    if loop_count is None:
        return []

    common_template = {
        key: copy.deepcopy(value)
        for key, value in chordwise_def.items()
        if key != "loops"
    }
    return [
        {
            **copy.deepcopy(common_template),
            "name": f"airfoil loop {loop_index + 1}",
        }
        for loop_index in range(loop_count)
    ]


def _automatic_spanwise_templates(config: dict[str, Any]) -> list[dict[str, Any]]:
    spanwise_def = config.get("spanwise_transfinite_def")
    if spanwise_def is None:
        spanwise_def = config.get("chordwise_transfinite_def", {}).get(
            "spanwise_transfinite_def", {}
        )
    if not spanwise_def:
        return []

    section_defs = spanwise_def.get("sections")
    if isinstance(section_defs, list) and section_defs:
        return [
            _merge_transfinite_template(spanwise_def, section_def, {"sections"})
            for section_def in section_defs
        ]

    target_sizes = spanwise_def.get("target ele sizes")
    if isinstance(target_sizes, list) and len(target_sizes) >= 2:
        common_template = {
            key: copy.deepcopy(value)
            for key, value in spanwise_def.items()
            if key != "target ele sizes"
        }
        common_template.setdefault("mesh size mode", "ele size")
        return [
            {
                **copy.deepcopy(common_template),
                "name": f"spanwise section {index + 1}",
                "target ele size 1": target_sizes[index],
                "target ele size 2": target_sizes[index + 1],
            }
            for index in range(len(target_sizes) - 1)
        ]

    return []


def apply_automatic_curve_sequences(mesh_def: dict[str, Any]) -> dict[str, Any]:
    config = _automatic_config(mesh_def)
    if config is None:
        return mesh_def

    start_point = int(config.get("start_pt", config.get("start point", 66)))
    chordwise_direction = _as_vector(
        config.get(
            "chordwise_direction",
            config.get("v_chordwise", [-1.0, 0.0, 0.0]),
        )
    )
    spanwise_direction = _as_vector(
        config.get(
            "spanwise_direction",
            config.get("v_spanwise", [0.0, 1.0, 0.0]),
        )
    )
    blunt_direction = _as_vector(
        config.get(
            "blunt_te_direction",
            config.get("v_blunt_te", [0.0, 0.0, -1.0]),
        )
    )
    infer_n_subcurvs = bool(config.get("infer_n_subcurvs", True))

    spanwise_templates = [
        copy.deepcopy(entry)
        for entry in _iter_curve_entries(mesh_def.get("spanwise_curve_sequences", []))
    ]
    if not spanwise_templates:
        spanwise_templates = _automatic_spanwise_templates(config)
    chordwise_templates = [
        copy.deepcopy(entry)
        for entry in _iter_curve_entries(mesh_def.get("chordwise_curve_sequences", []))
    ]
    if not chordwise_templates:
        chordwise_templates = _automatic_chordwise_templates(
            config, len(spanwise_templates) + 1 if spanwise_templates else None
        )
    if not chordwise_templates:
        raise ValueError("Automatic curve discovery requires chordwise templates.")

    auto_mesh_def = copy.deepcopy(mesh_def)
    used_curves: set[int] = set()
    chordwise_entries = []
    spanwise_entries = []
    loop_start_point = start_point
    spanwise_directions = [spanwise_direction]
    previous_chordwise_group_sizes = None

    for loop_index, chordwise_template in enumerate(chordwise_templates):
        if loop_index > 0 and spanwise_directions:
            first_chord = _select_adjacent_curve(
                loop_start_point, chordwise_direction, used_curves
            )
            if first_chord is None:
                raise ValueError(
                    f"Could not identify first chordwise curve at point "
                    f"{loop_start_point}."
                )
            first_chord_vector = _point_vector(
                loop_start_point,
                _other_curve_point(first_chord[0], loop_start_point),
            )
            cross_blunt_direction = _cross(first_chord_vector, spanwise_directions[0])
            if _vector_norm(cross_blunt_direction) > 0.0:
                blunt_direction = cross_blunt_direction

        traced_loop = _trace_airfoil_loop(
            loop_start_point,
            chordwise_direction,
            blunt_direction,
            spanwise_directions[0] if spanwise_directions else spanwise_direction,
            used_curves,
        )
        traced_curve_ids = traced_loop.curve_ids + [traced_loop.blunt_curve_id]
        traced_invert_directions = [
            _curve_traversal_invert_direction(
                curve_id,
                traced_loop.points[curve_index],
                traced_loop.points[curve_index + 1],
            )
            for curve_index, curve_id in enumerate(traced_loop.curve_ids)
        ]
        traced_invert_directions.append(
            _curve_traversal_invert_direction(
                traced_loop.blunt_curve_id,
                traced_loop.points[-1],
                traced_loop.points[0],
            )
        )
        used_curves.update(abs(curve_id) for curve_id in traced_curve_ids)

        chordwise_entry = copy.deepcopy(chordwise_template)
        split_spanwise_curve_count = (
            1 if loop_index in (0, len(chordwise_templates) - 1) else 2
        )
        inferred_n_subcurvs = (
            _infer_n_subcurvs_for_template(
                traced_loop,
                chordwise_template,
                split_spanwise_curve_count,
                previous_chordwise_group_sizes,
            )
            if infer_n_subcurvs
            else None
        )
        chordwise_entry["curve_ids"] = _group_traced_curves(
            traced_curve_ids,
            chordwise_template,
            inferred_n_subcurvs,
        )
        chordwise_entry["invert_direction"] = _group_values_by_curve_groups(
            traced_invert_directions, chordwise_entry["curve_ids"]
        )
        if inferred_n_subcurvs is not None:
            _remap_template_values_to_groups(
                chordwise_entry,
                chordwise_template,
                inferred_n_subcurvs,
                len(traced_curve_ids),
            )
        _normalize_invert_directions_for_curve_groups(chordwise_entry)
        chordwise_entries.append(chordwise_entry)
        if inferred_n_subcurvs is not None:
            previous_chordwise_group_sizes = inferred_n_subcurvs

        if loop_index >= len(spanwise_templates):
            break

        (
            spanwise_curve_ids,
            spanwise_invert_directions,
            next_points,
            next_directions,
        ) = _trace_spanwise_curves(traced_loop.points, spanwise_directions, used_curves)
        if not spanwise_curve_ids:
            break

        used_curves.update(abs(curve_id) for curve_id in spanwise_curve_ids)
        spanwise_entry = copy.deepcopy(spanwise_templates[loop_index])
        spanwise_entry["curve_ids"] = spanwise_curve_ids
        if (
            "_invert_direction" in spanwise_entry
            and "invert_direction" not in spanwise_entry
        ):
            spanwise_entry["invert_direction"] = copy.deepcopy(
                spanwise_entry["_invert_direction"]
            )
        else:
            spanwise_entry["invert_direction"] = spanwise_invert_directions
        spanwise_entries.append(spanwise_entry)

        loop_start_point = next_points[0]
        spanwise_directions = next_directions or [spanwise_direction]

    auto_mesh_def["chordwise_curve_sequences"] = chordwise_entries
    auto_mesh_def["spanwise_curve_sequences"] = spanwise_entries
    return auto_mesh_def


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
    chordwise_curves: set[int],
    spanwise_curves: set[int],
) -> list[int]:
    boundary_curves = _surface_boundary_curves(surface_id)
    boundary_chordwise_curves = [
        curve_id for curve_id in boundary_curves if curve_id in chordwise_curves
    ]
    boundary_spanwise_curves = [
        curve_id for curve_id in boundary_curves if curve_id in spanwise_curves
    ]
    incident_chordwise = _curve_incident_boundary_points(boundary_chordwise_curves)
    incident_spanwise = _curve_incident_boundary_points(boundary_spanwise_curves)
    corner_set = set(incident_chordwise).intersection(incident_spanwise)

    ordered_corners = []
    for point_id in _surface_ordered_boundary_points(surface_id):
        if point_id in corner_set and point_id not in ordered_corners:
            ordered_corners.append(point_id)

    return ordered_corners


def _spanwise_curve_map(curve_ids: list[int]) -> dict[int, int]:
    curve_map = {}
    for curve_id in curve_ids:
        point_a, point_b = _curve_endpoints(curve_id)
        curve_map[point_a] = curve_id
        curve_map[point_b] = curve_id
    return curve_map


def _spanwise_connector_curves(spanwise_entries: Iterable[dict[str, Any]]) -> set[int]:
    connector_curves = set()
    for entry in spanwise_entries:
        curve_ids = [abs(curve_id) for curve_id in _as_list(entry.get("curve_ids", []))]
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
    if "automatic_transfinite_surfaces" in mesh_def:
        return bool(mesh_def["automatic_transfinite_surfaces"])
    return _automatic_config(mesh_def) is not None


def apply_automatic_transfinite_surfaces(mesh_def: dict[str, Any]) -> dict[str, Any]:
    if not _automatic_surfaces_enabled(mesh_def):
        return mesh_def

    chordwise_entries = list(
        _iter_curve_entries(mesh_def.get("chordwise_curve_sequences", []))
    )
    spanwise_entries = list(
        _iter_curve_entries(mesh_def.get("spanwise_curve_sequences", []))
    )
    if len(chordwise_entries) < 2 or not spanwise_entries:
        return mesh_def

    chordwise_curves = {
        abs(curve_id)
        for entry in chordwise_entries
        for curve_group in _entry_curve_groups(entry)
        for curve_id in curve_group
    }
    spanwise_curves = {
        abs(curve_id)
        for entry in spanwise_entries
        for curve_id in _as_list(entry["curve_ids"])
    }
    transfinite_curves = chordwise_curves.union(spanwise_curves)
    transfinite_surfaces = []
    transfinite_surface_ids = set()

    for _, surface_id in gmsh.model.getEntities(2):
        boundary_curves = set(_surface_boundary_curves(surface_id))
        boundary_chordwise_curves = boundary_curves.intersection(chordwise_curves)
        boundary_spanwise_curves = boundary_curves.intersection(spanwise_curves)
        if (
            not boundary_curves.issubset(transfinite_curves)
            or not boundary_chordwise_curves
            or not boundary_spanwise_curves
        ):
            continue

        boundary_points = _surface_transfinite_corners(
            surface_id, chordwise_curves, spanwise_curves
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
    auto_mesh_def["transfinite_surfaces"] = transfinite_surfaces
    auto_mesh_def["unstructured_surfaces"] = sorted(
        all_surface_ids - transfinite_surface_ids
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
                curve_ids[0], invert_directions[0], n_pts, mesh_type, coef
            )
        else:
            yield CurveSpec(curve_ids, invert_directions, n_pts, mesh_type, coef)


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
        n_pts, mesh_type, coef = _curve_spec_discretization(
            entry, spec_index, spec_count, spec_curve_ids
        )
        yield CurveSpec(
            spec_curve_ids, spec_invert_direction, n_pts, mesh_type, coef
        )


def _curve_coef(mesh_type: str, base_coef: float, invert: bool) -> float:
    if mesh_type.lower() == "progression" and invert:
        if base_coef == 0.0:
            raise ValueError("Progression coefficient cannot be zero.")
        return 1.0 / base_coef

    return base_coef


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


def _normalize_spanwise_sequence_counts(
    mesh_def: dict[str, Any],
    constraints: dict[int, CurveConstraint],
) -> None:
    for entry in _iter_curve_entries(mesh_def.get("spanwise_curve_sequences", [])):
        curve_ids = [abs(curve_id) for curve_id in _as_list(entry.get("curve_ids", []))]
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

    for section_name in ("chordwise_curve_sequences", "spanwise_curve_sequences"):
        section = mesh_def.get(section_name, {})
        for entry in _iter_curve_entries(section):
            curve_specs = list(_iter_curve_specs(entry))
            if (
                section_name == "spanwise_curve_sequences"
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
                for curve_id, curve_n_pts, invert in _curve_node_counts(
                    curve_spec.curve_ids,
                    curve_spec.invert_directions,
                    curve_spec.n_pts,
                    entry,
                    curve_spec.mesh_type,
                    curve_spec.coef,
                ):
                    curve_coef = _curve_coef(
                        curve_spec.mesh_type, curve_spec.coef, invert
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
    chordwise_curve_loop_indices = _chordwise_curve_loop_indices(mesh_def)
    chordwise_curve_groups = _chordwise_curve_groups(mesh_def)
    protected_curve_counts = _protected_curve_counts(mesh_def)

    for surface in _iter_surface_entries(section):
        surface_id = int(surface["id"])
        if surface_id in unstructured_surfaces:
            continue

        corners = [int(point) for point in surface.get("boundary points", [])]

        if len(corners) != 4:
            continue

        edge_curves = _surface_edge_curves(surface_id, corners)
        if _has_protected_opposite_edge_mismatch(
            edge_curves, constraints, protected_curve_counts
        ):
            _add_unstructured_surface(mesh_def, surface_id)
            unstructured_surfaces.add(surface_id)
            continue

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
            chordwise_curve_loop_indices,
            chordwise_curve_groups,
        )


def _chordwise_curve_loop_indices(mesh_def: dict[str, Any]) -> dict[int, int]:
    loop_indices = {}
    for loop_index, entry in enumerate(
        _iter_curve_entries(mesh_def.get("chordwise_curve_sequences", []))
    ):
        for curve_group in entry.get("curve_ids", []):
            for curve_id in _as_list(curve_group):
                loop_indices[abs(curve_id)] = loop_index
    return loop_indices


def _chordwise_curve_groups(mesh_def: dict[str, Any]) -> dict[int, list[int]]:
    curve_groups = {}
    for entry in _iter_curve_entries(mesh_def.get("chordwise_curve_sequences", [])):
        for curve_group in entry.get("curve_ids", []):
            group = [abs(curve_id) for curve_id in _as_list(curve_group)]
            for curve_id in group:
                curve_groups[curve_id] = group
    return curve_groups


def _protected_curve_counts(mesh_def: dict[str, Any]) -> set[int]:
    protected_curves = set()
    for entry in _iter_curve_entries(mesh_def.get("spanwise_curve_sequences", [])):
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
    chordwise_curve_loop_indices: dict[int, int],
    chordwise_curve_groups: dict[int, list[int]],
) -> None:
    for edge_index in range(2):
        curves_a = edge_curves[edge_index]
        curves_b = edge_curves[edge_index + 2]
        divisions_a = _edge_divisions(curves_a, constraints)
        divisions_b = _edge_divisions(curves_b, constraints)
        if divisions_a is None or divisions_b is None or divisions_a == divisions_b:
            continue

        loop_indices_a = [
            chordwise_curve_loop_indices.get(abs(curve_id)) for curve_id in curves_a
        ]
        loop_indices_b = [
            chordwise_curve_loop_indices.get(abs(curve_id)) for curve_id in curves_b
        ]
        chordwise_a = all(loop_index is not None for loop_index in loop_indices_a)
        chordwise_b = all(loop_index is not None for loop_index in loop_indices_b)

        if chordwise_a and chordwise_b:
            _align_opposite_chordwise_edge_divisions(
                curves_a,
                divisions_a,
                loop_indices_a,
                curves_b,
                divisions_b,
                loop_indices_b,
                constraints,
                chordwise_curve_groups,
            )
            continue
        if chordwise_a:
            source_divisions = divisions_a
            target_edge_divisions = divisions_b
            target_curves = curves_b
        elif chordwise_b:
            source_divisions = divisions_b
            target_edge_divisions = divisions_a
            target_curves = curves_a
        else:
            source_divisions = max(divisions_a, divisions_b)
            target_edge_divisions = min(divisions_a, divisions_b)
            target_curves = curves_a if divisions_a < divisions_b else curves_b

        non_chordwise_target_curves = [
            abs(curve_id)
            for curve_id in target_curves
            if abs(curve_id) not in chordwise_curve_loop_indices
        ]
        if not non_chordwise_target_curves:
            warnings.warn(
                "Could not align opposite surface edge divisions without changing "
                f"chordwise curves {target_curves}. Leaving counts unchanged.",
                stacklevel=2,
            )
            continue

        target_curve = max(
            non_chordwise_target_curves,
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


def _align_opposite_chordwise_edge_divisions(
    curves_a: list[int],
    divisions_a: int,
    loop_indices_a: list[int | None],
    curves_b: list[int],
    divisions_b: int,
    loop_indices_b: list[int | None],
    constraints: dict[int, CurveConstraint],
    chordwise_curve_groups: dict[int, list[int]],
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
    target_group = chordwise_curve_groups.get(target_curve, [target_curve])
    compensation_curves = [
        curve_id
        for curve_id in target_group
        if curve_id != target_curve
        and curve_id not in {abs(target_item) for target_item in target_curves}
    ]
    if not compensation_curves:
        warnings.warn(
            "Opposite chordwise surface edges have inconsistent divisions, but "
            f"curve {target_curve} is not part of a larger grouped curve. "
            "Leaving chordwise group total unchanged.",
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
            "Cannot align chordwise subcurve divisions while preserving grouped "
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
    return {int(surface_id) for surface_id in mesh_def.get("unstructured_surfaces", [])}


def _transfinite_surface_definitions(mesh_def: dict[str, Any]) -> dict[int, dict[str, Any]]:
    section = mesh_def.get("transfinite_surfaces", {})
    return {int(surface["id"]): surface for surface in _iter_surface_entries(section)}


def apply_transfinite_surfaces(mesh_def: dict[str, Any]) -> None:
    unstructured_surfaces = _unstructured_surface_ids(mesh_def)
    surface_definitions = _transfinite_surface_definitions(mesh_def)

    for _, surface_id in gmsh.model.getEntities(2):
        if surface_id in unstructured_surfaces:
            continue

        surface = surface_definitions.get(surface_id, {})
        explicitly_defined = surface_id in surface_definitions
        if not explicitly_defined:
            corner_points = _surface_corner_points(surface_id)
            if len(corner_points) > 4:
                warnings.warn(
                    f"Skipping automatic transfinite surface {surface_id}: "
                    f"surface has {len(corner_points)} corners {corner_points}. "
                    f"Add it explicitly to transfinite_surfaces or list it in "
                    f"unstructured_surfaces.",
                    stacklevel=2,
                )
                continue

        arrangement = str(surface.get("Arrangement", "Left")).capitalize()
        boundary_points = [int(point) for point in surface.get("boundary points", [])]

        gmsh.model.mesh.setTransfiniteSurface(
            surface_id, arrangement, boundary_points
        )


def show_gmsh() -> None:
    """Open the Gmsh GUI for inspecting the generated mesh."""
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.fltk.run()


def generate_surface_mesh(
    step_file: Path,
    mesh_def_file: Path,
    output_file: Path,
    *,
    recombine: bool = True,
    mesh_format: str = "msh2",
    show: bool = True,
) -> None:
    with mesh_def_file.open("r", encoding="utf-8") as file:
        mesh_def = json.load(file)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2 if mesh_format == "msh2" else 4.1)

        gmsh.model.add(step_file.stem)
        gmsh.model.occ.importShapes(str(step_file))
        gmsh.model.occ.synchronize()

        mesh_def = apply_automatic_curve_sequences(mesh_def)
        mesh_def = apply_automatic_transfinite_surfaces(mesh_def)
        curve_constraints = apply_transfinite_curves(mesh_def)
        complete_surface_boundary_curves(mesh_def, curve_constraints)
        apply_transfinite_surfaces(mesh_def)

        gmsh.model.mesh.generate(2)
        if recombine:
            gmsh.model.mesh.recombine()

        gmsh.write(str(output_file))

        if show:
            show_gmsh()
    finally:
        gmsh.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Gmsh surface mesh from a STEP file and JSON transfinite definitions."
    )
    parser.add_argument(
        "--step",
        type=Path,
        default=DEFAULT_STEP_FILE,
        help="Path to the STEP file.",
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
        help="Output mesh path. Defaults to <step-file>.msh.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = args.out or args.step.with_suffix(".msh")

    generate_surface_mesh(
        args.step,
        args.mesh_def,
        output_file,
        recombine=args.recombine,
        mesh_format=args.format,
        show=not args.no_show,
    )

    print(f"Wrote surface mesh to {output_file}")


if __name__ == "__main__":
    main()
