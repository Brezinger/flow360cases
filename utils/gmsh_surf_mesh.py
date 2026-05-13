from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import gmsh


DEFAULT_STEP_FILE = Path(
    r"C:\OneDrive\OneDrive - Achleitner Aerospace GmbH"
    r"\Achleitner Aerospace GmbH Allgemein - General"
    r"\01_Projekte\04_Flaplets\03_CAD\Aerodynamikmodelle"
    r"\Ventus3 FlapletV2\WKS\Ventus3_FlapletV2_WKS.stp"
)
DEFAULT_MESH_DEF_FILE = DEFAULT_STEP_FILE.with_name("msh_def.json")


class CurveConstraint:
    def __init__(self, n_pts: int, mesh_type: str, coef: float) -> None:
        self.n_pts = n_pts
        self.mesh_type = mesh_type
        self.coef = coef


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


def _curve_length(curve_id: int) -> float:
    return gmsh.model.occ.getMass(1, abs(curve_id))


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
) -> list[float]:
    mesh_type = str(entry.get("type", "Progression")).lower()
    coef = float(entry.get("Parameter", 1.0))

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
    node_positions = _distribution_node_positions(entry, n_pts, invert_group[0])
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


def _iter_curve_entries(section: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("rows", "columns"):
        yield from section.get(key, [])


def _legacy_invert_direction(entry: dict[str, Any]) -> bool:
    direction = str(entry.get("direction", "normal")).lower()

    if direction == "normal":
        return False
    if direction == "flipped":
        return True

    raise ValueError(f"Unsupported curve direction {direction!r}.")


def _iter_curve_specs(
    entry: dict[str, Any],
) -> Iterable[tuple[int | list[int], bool | list[bool]]]:
    curve_ids = entry["curve_ids"]

    if "invert_direction" in entry:
        invert_direction = entry["invert_direction"]
        if isinstance(invert_direction, bool):
            invert_direction = [invert_direction] * len(curve_ids)
        elif len(invert_direction) != len(curve_ids):
            raise ValueError(
                f"invert_direction length {len(invert_direction)} does not match "
                f"curve_ids length {len(curve_ids)} in {entry.get('name', entry)!r}."
            )
    else:
        invert_direction = [_legacy_invert_direction(entry)] * len(curve_ids)

    yield from zip(curve_ids, invert_direction)


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
            n_pts = int(entry["n_pts"])
            mesh_type = str(entry.get("type", "Progression"))
            coef = float(entry.get("Parameter", 1.0))

            if n_pts < 2:
                raise ValueError(f"Transfinite curve point count must be >= 2: {entry}")

            for curve_ids, invert_directions in _iter_curve_specs(entry):
                for curve_id, curve_n_pts, invert in _curve_node_counts(
                    curve_ids, invert_directions, n_pts, entry
                ):
                    curve_coef = _curve_coef(mesh_type, coef, invert)
                    gmsh.model.mesh.setTransfiniteCurve(
                        curve_id,
                        curve_n_pts,
                        mesh_type,
                        curve_coef,
                    )
                    constraints[abs(curve_id)] = CurveConstraint(
                        curve_n_pts, mesh_type, curve_coef
                    )

    return constraints


def complete_surface_boundary_curves(
    mesh_def: dict[str, Any],
    constraints: dict[int, CurveConstraint],
) -> None:
    section = mesh_def.get("transfinite_surfaces", {})
    unstructured_surfaces = _unstructured_surface_ids(mesh_def)

    for surface in section.get("surfaces", []):
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
    return {int(surface["id"]): surface for surface in section.get("surfaces", [])}


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
