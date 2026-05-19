from __future__ import annotations

import argparse
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
    r"\Ventus3 FlapletV2\WKS\Ventus3_FlapletV2_WKS.stp"
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
    present_keys = trace_keys.intersection(entry)

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

        natural_start, natural_end = _oriented_curve_endpoints(best_curve)
        invert_direction = not (
            natural_start == current_pt and natural_end == best_next_pt
        )
        traced_curves.append((best_curve, invert_direction))
        visited_curves.add(abs(best_curve))
        current_pt = best_next_pt
        current_coord = gmsh.model.getValue(0, current_pt, [0])
        pointing = best_vector

    return traced_curves


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
    section = mesh_def.get("transfinite_surfaces", {})
    unstructured_surfaces = _unstructured_surface_ids(mesh_def)

    for surface in _iter_surface_entries(section):
        surface_id = int(surface["id"])
        if surface_id in unstructured_surfaces:
            continue

        corners = [int(point) for point in surface.get("boundary points", [])]

        if len(corners) != 4:
            continue

        corner_edges = [
            frozenset((corners[0], corners[1])),
            frozenset((corners[1], corners[2])),
            frozenset((corners[2], corners[3])),
            frozenset((corners[3], corners[0])),
        ]
        edge_curves: list[list[int]] = [[], [], [], []]

        for curve_id in _surface_boundary_curves(surface_id):
            edge_index = _corner_edge_index(_curve_endpoints(curve_id), corner_edges)
            if edge_index is not None:
                edge_curves[edge_index].append(curve_id)

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
            gmsh.model.mesh.setTransfiniteCurve(
                curve_id,
                opposite_constraint.n_pts,
                opposite_constraint.mesh_type,
                opposite_constraint.coef,
            )
            constraints[curve_id] = opposite_constraint


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
