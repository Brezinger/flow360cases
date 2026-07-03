from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gmsh

from gmsh_surf_mesh import (
    apply_geometry_healing,
    apply_geometry_preprocessing,
    load_mesh_def,
    mesh_def_step_file,
)


DEFAULT_POINT_TOLERANCE = 1.0e-4
DEFAULT_SURFACE_POINT_MATCH_RATIO = 0.8

_TRANSFINITE_CURVE_RE = re.compile(
    r"^\s*Transfinite\s+(?P<kind>Curve|Line)\s*\{(?P<curves>[^}]*)\}(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_TRANSFINITE_SURFACE_RE = re.compile(
    r"""
    ^\s*Transfinite\s+Surface\s*
    \{(?P<surfaces>[^}]*)\}
    (?P<between>\s*=\s*\{(?P<points>[^}]*)\})?
    (?P<tail>.*)$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)
_RECOMBINE_SURFACE_RE = re.compile(
    r"^\s*Recombine\s+Surface\s*\{(?P<surfaces>[^}]*)\}(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_MESH_ALGORITHM_SURFACE_RE = re.compile(
    r"""
    ^\s*(?P<prefix>(?:Mesh\.Algorithm|MeshAlgorithm)\s+Surface\s*)
    \{(?P<surfaces>[^}]*)\}
    (?P<tail>.*)$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class GeometryEntities:
    points: dict[int, tuple[float, float, float]]
    curves: dict[int, tuple[int, int]]
    curve_lengths: dict[int, float]
    surfaces: dict[int, set[int]]
    surface_points: dict[int, set[int]]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _load_geometry(mesh_def_file: Path) -> GeometryEntities:
    mesh_def = load_mesh_def(mesh_def_file)
    step_file = mesh_def_step_file(mesh_def, mesh_def_file)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(step_file.stem)
        imported_entities = gmsh.model.occ.importShapes(str(step_file))
        apply_geometry_preprocessing(mesh_def, imported_entities)
        apply_geometry_healing(mesh_def)
        gmsh.model.occ.synchronize()

        points = {
            tag: tuple(float(value) for value in gmsh.model.getValue(0, tag, [0]))
            for _, tag in gmsh.model.getEntities(0)
        }
        curves = {
            tag: _curve_endpoints(tag)
            for _, tag in gmsh.model.getEntities(1)
        }
        curve_lengths = {
            tag: float(gmsh.model.occ.getMass(1, tag))
            for _, tag in gmsh.model.getEntities(1)
        }
        surfaces = {
            tag: _surface_boundary_curves(tag)
            for _, tag in gmsh.model.getEntities(2)
        }
        surface_points = {
            tag: _surface_boundary_points(tag)
            for _, tag in gmsh.model.getEntities(2)
        }
    finally:
        gmsh.finalize()

    return GeometryEntities(points, curves, curve_lengths, surfaces, surface_points)


def _curve_endpoints(curve_id: int) -> tuple[int, int]:
    boundary = gmsh.model.getBoundary([(1, curve_id)], oriented=False, recursive=False)
    endpoints = sorted(tag for dim, tag in boundary if dim == 0)
    if len(endpoints) != 2:
        raise ValueError(f"Curve {curve_id} has {len(endpoints)} endpoints, expected 2.")
    return endpoints[0], endpoints[1]


def _surface_boundary_curves(surface_id: int) -> set[int]:
    boundary = gmsh.model.getBoundary([(2, surface_id)], oriented=False, recursive=False)
    return {abs(tag) for dim, tag in boundary if dim == 1}


def _surface_boundary_points(surface_id: int) -> set[int]:
    boundary = gmsh.model.getBoundary([(2, surface_id)], oriented=False, recursive=True)
    return {abs(tag) for dim, tag in boundary if dim == 0}


def correlate_points(
    old_points: dict[int, tuple[float, float, float]],
    new_points: dict[int, tuple[float, float, float]],
    tolerance: float,
) -> dict[int, int | None]:
    candidates: list[tuple[float, int, int]] = []
    for old_id, old_coord in old_points.items():
        for new_id, new_coord in new_points.items():
            distance = _distance(old_coord, new_coord)
            if distance <= tolerance:
                candidates.append((distance, old_id, new_id))

    candidates.sort()
    point_map: dict[int, int | None] = {old_id: None for old_id in old_points}
    used_new_points: set[int] = set()
    for _, old_id, new_id in candidates:
        if point_map[old_id] is not None or new_id in used_new_points:
            continue
        point_map[old_id] = new_id
        used_new_points.add(new_id)

    return point_map


def correlate_curves(
    old_curves: dict[int, tuple[int, int]],
    new_curves: dict[int, tuple[int, int]],
    point_map: dict[int, int | None],
) -> dict[int, int | None]:
    new_by_endpoints: dict[frozenset[int], list[int]] = {}
    for new_curve_id, endpoints in new_curves.items():
        new_by_endpoints.setdefault(frozenset(endpoints), []).append(new_curve_id)

    curve_map: dict[int, int | None] = {}
    for old_curve_id, endpoints in old_curves.items():
        mapped_endpoints = [point_map.get(point_id) for point_id in endpoints]
        if any(point_id is None for point_id in mapped_endpoints):
            curve_map[old_curve_id] = None
            continue

        matches = new_by_endpoints.get(frozenset(int(point_id) for point_id in mapped_endpoints), [])
        curve_map[old_curve_id] = matches[0] if len(matches) == 1 else None

    return curve_map


def correlate_surfaces(
    old_surfaces: dict[int, set[int]],
    old_surface_points: dict[int, set[int]],
    new_surfaces: dict[int, set[int]],
    new_surface_points: dict[int, set[int]],
    point_map: dict[int, int | None],
    curve_map: dict[int, int | None],
    point_match_ratio: float,
) -> dict[int, int | None]:
    new_by_curves: dict[frozenset[int], list[int]] = {}
    for new_surface_id, curves in new_surfaces.items():
        new_by_curves.setdefault(frozenset(curves), []).append(new_surface_id)

    surface_map: dict[int, int | None] = {}
    for old_surface_id, old_curves in old_surfaces.items():
        mapped_curves = {
            mapped_curve
            for old_curve in old_curves
            if (mapped_curve := curve_map.get(old_curve)) is not None
        }
        if len(mapped_curves) == len(old_curves):
            matches = new_by_curves.get(frozenset(mapped_curves), [])
            if len(matches) == 1:
                surface_map[old_surface_id] = matches[0]
                continue

        surface_map[old_surface_id] = _surface_match_by_points(
            old_surface_points.get(old_surface_id, set()),
            new_surface_points,
            point_map,
            point_match_ratio,
        )

    return surface_map


def _surface_match_by_points(
    old_points: set[int],
    new_surface_points: dict[int, set[int]],
    point_map: dict[int, int | None],
    point_match_ratio: float,
) -> int | None:
    mapped_points = {
        mapped_point
        for old_point in old_points
        if (mapped_point := point_map.get(old_point)) is not None
    }
    if not mapped_points:
        return None

    best_surface_id: int | None = None
    best_score = 0.0
    second_best_score = 0.0
    for new_surface_id, candidate_points in new_surface_points.items():
        intersection_count = len(mapped_points.intersection(candidate_points))
        union_count = len(mapped_points.union(candidate_points))
        score = intersection_count / union_count if union_count else 0.0
        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_surface_id = new_surface_id
        elif score > second_best_score:
            second_best_score = score

    if best_score < point_match_ratio or math.isclose(best_score, second_best_score):
        return None
    return best_surface_id


def _mapped_id(old_id: int, entity_map: dict[int, int | None]) -> int | str:
    sign = -1 if old_id < 0 else 1
    mapped_id = entity_map.get(abs(old_id))
    if mapped_id is None:
        return f"{old_id}_unidentified"
    return sign * mapped_id


def _map_id_value(value: Any, entity_map: dict[int, int | None]) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return _mapped_id(value, entity_map)
    if isinstance(value, list):
        return [_map_id_value(item, entity_map) for item in value]
    return value


def update_mesh_definition_ids(
    mesh_def: Any,
    point_map: dict[int, int | None],
    curve_map: dict[int, int | None],
    surface_map: dict[int, int | None],
) -> Any:
    updated = copy.deepcopy(mesh_def)
    _update_node(updated, point_map, curve_map, surface_map)
    return updated


def _update_node(
    node: Any,
    point_map: dict[int, int | None],
    curve_map: dict[int, int | None],
    surface_map: dict[int, int | None],
) -> None:
    if isinstance(node, list):
        for item in node:
            _update_node(item, point_map, curve_map, surface_map)
        return

    if not isinstance(node, dict):
        return

    for key, value in list(node.items()):
        if key in {"start_pt", "boundary points"}:
            node[key] = _map_id_value(value, point_map)
        elif key == "curve_ids":
            node[key] = _map_id_value(value, curve_map)
        elif key in {"surfaces", "unstructured_surfaces"}:
            node[key] = _map_id_value(value, surface_map)
        elif key == "id" and _looks_like_transfinite_surface(node):
            node[key] = _map_id_value(value, surface_map)
        else:
            _update_node(value, point_map, curve_map, surface_map)


def _looks_like_transfinite_surface(node: dict[str, Any]) -> bool:
    return "Arrangement" in node or "boundary points" in node


def _count_identified(entity_map: dict[int, int | None]) -> tuple[int, int]:
    identified = sum(mapped_id is not None for mapped_id in entity_map.values())
    return identified, len(entity_map)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def write_correlated_geo_file(
    old_geo_file: Path,
    new_geo_file: Path,
    point_map: dict[int, int | None],
    curve_map: dict[int, int | None],
    surface_map: dict[int, int | None],
) -> tuple[int, int]:
    translated_statements, skipped_count = correlate_geo_text(
        old_geo_file.read_text(encoding="utf-8"),
        point_map,
        curve_map,
        surface_map,
    )
    with new_geo_file.open("w", encoding="utf-8") as file:
        for statement in translated_statements:
            file.write("//+\n")
            file.write(statement)
            file.write(";\n")
    return len(translated_statements), skipped_count


def correlate_geo_text(
    geo_text: str,
    point_map: dict[int, int | None],
    curve_map: dict[int, int | None],
    surface_map: dict[int, int | None],
) -> tuple[list[str], int]:
    translated_statements: list[str] = []
    skipped_count = 0
    for statement in _geo_statements(geo_text):
        translated_statement = _correlate_geo_statement(
            statement,
            point_map,
            curve_map,
            surface_map,
        )
        if translated_statement is None:
            skipped_count += 1
            continue
        translated_statements.append(translated_statement)
    return translated_statements, skipped_count


def _correlate_geo_statement(
    statement: str,
    point_map: dict[int, int | None],
    curve_map: dict[int, int | None],
    surface_map: dict[int, int | None],
) -> str | None:
    if match := _TRANSFINITE_CURVE_RE.match(statement):
        mapped_curves = _mapped_geo_id_list(match.group("curves"), curve_map)
        if mapped_curves is None:
            return None
        return (
            f"Transfinite {match.group('kind').capitalize()} "
            f"{{{_format_geo_id_list(mapped_curves)}}}{match.group('tail')}"
        )

    if match := _TRANSFINITE_SURFACE_RE.match(statement):
        mapped_surfaces = _mapped_geo_id_list(match.group("surfaces"), surface_map)
        if mapped_surfaces is None:
            return None

        points = match.group("points")
        if points is None:
            point_clause = ""
        else:
            mapped_points = _mapped_geo_id_list(points, point_map)
            if mapped_points is None:
                return None
            point_clause = f" = {{{_format_geo_id_list(mapped_points)}}}"

        return (
            f"Transfinite Surface {{{_format_geo_id_list(mapped_surfaces)}}}"
            f"{point_clause}{match.group('tail')}"
        )

    if match := _RECOMBINE_SURFACE_RE.match(statement):
        mapped_surfaces = _mapped_geo_id_list(match.group("surfaces"), surface_map)
        if mapped_surfaces is None:
            return None
        return f"Recombine Surface {{{_format_geo_id_list(mapped_surfaces)}}}{match.group('tail')}"

    if match := _MESH_ALGORITHM_SURFACE_RE.match(statement):
        mapped_surfaces = _mapped_geo_id_list(match.group("surfaces"), surface_map)
        if mapped_surfaces is None:
            return None
        return (
            f"{match.group('prefix')}{{{_format_geo_id_list(mapped_surfaces)}}}"
            f"{match.group('tail')}"
        )

    return None


def _geo_statements(geo_text: str) -> list[str]:
    without_block_comments = re.sub(r"/\*.*?\*/", "", geo_text, flags=re.DOTALL)
    lines = []
    for line in without_block_comments.splitlines():
        lines.append(line.split("//", 1)[0])
    return [
        statement.strip()
        for statement in "\n".join(lines).split(";")
        if statement.strip()
    ]


def _mapped_geo_id_list(
    value: str,
    entity_map: dict[int, int | None],
) -> list[int] | None:
    mapped_ids: list[int] = []
    for entity_id in _parse_geo_int_list(value):
        mapped_id = entity_map.get(abs(entity_id))
        if mapped_id is None:
            return None
        mapped_ids.append(-mapped_id if entity_id < 0 else mapped_id)
    return mapped_ids


def _parse_geo_int_list(value: str) -> list[int]:
    ids: list[int] = []
    for raw_item in re.split(r"[,\s]+", value.strip()):
        if not raw_item:
            continue
        ids.append(int(raw_item))
    return ids


def _format_geo_id_list(values: Iterable[int]) -> str:
    return ", ".join(str(value) for value in values)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate Gmsh point/curve/surface IDs between two STEP-backed mesh "
            "definition JSON files after applying the same preprocessing and healing "
            "used by gmsh_surf_mesh.py."
        )
    )
    parser.add_argument("old_mesh_def", type=Path, help="Mesh definition with known IDs.")
    parser.add_argument("new_mesh_def", type=Path, help="Mesh definition to rewrite.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to modifying new_mesh_def in place.",
    )
    parser.add_argument(
        "--point-tolerance",
        type=float,
        default=DEFAULT_POINT_TOLERANCE,
        help=f"Maximum coordinate distance for point matching. Default: {DEFAULT_POINT_TOLERANCE:g}.",
    )
    parser.add_argument(
        "--surface-point-match-ratio",
        type=float,
        default=DEFAULT_SURFACE_POINT_MATCH_RATIO,
        help=(
            "Minimum Jaccard score for surface point-set fallback matching. "
            f"Default: {DEFAULT_SURFACE_POINT_MATCH_RATIO:g}."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file when modifying new_mesh_def in place.",
    )
    parser.add_argument(
        "--geo-in",
        type=Path,
        help="Old .geo file whose supported mesh statements should be mapped to the new model.",
    )
    parser.add_argument(
        "--geo-out",
        type=Path,
        help="Output .geo path for the mapped new-model statements. Requires --geo-in.",
    )
    parser.add_argument(
        "--skip-json-update",
        action="store_true",
        help="Only write the correlated .geo file; do not rewrite the new mesh JSON.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    old_mesh_def_file = args.old_mesh_def.resolve()
    new_mesh_def_file = args.new_mesh_def.resolve()
    output_file = args.output.resolve() if args.output else new_mesh_def_file

    if args.geo_out is not None and args.geo_in is None:
        raise ValueError("--geo-out requires --geo-in.")
    if args.skip_json_update and args.geo_in is None:
        raise ValueError("--skip-json-update requires --geo-in.")

    old_entities = _load_geometry(old_mesh_def_file)
    new_entities = _load_geometry(new_mesh_def_file)

    point_map = correlate_points(old_entities.points, new_entities.points, args.point_tolerance)
    curve_map = correlate_curves(old_entities.curves, new_entities.curves, point_map)
    surface_map = correlate_surfaces(
        old_entities.surfaces,
        old_entities.surface_points,
        new_entities.surfaces,
        new_entities.surface_points,
        point_map,
        curve_map,
        args.surface_point_match_ratio,
    )

    if not args.skip_json_update:
        with old_mesh_def_file.open("r", encoding="utf-8") as file:
            old_mesh_def_json = json.load(file)
        with new_mesh_def_file.open("r", encoding="utf-8") as file:
            new_mesh_def_json = json.load(file)

        rewrite_template = copy.deepcopy(old_mesh_def_json)
        rewrite_template["geometry definition"] = copy.deepcopy(
            new_mesh_def_json["geometry definition"]
        )
        updated_mesh_def = update_mesh_definition_ids(
            rewrite_template,
            point_map,
            curve_map,
            surface_map,
        )

        if output_file == new_mesh_def_file and not args.no_backup:
            backup_file = new_mesh_def_file.with_suffix(new_mesh_def_file.suffix + ".bak")
            backup_file.write_text(
                new_mesh_def_file.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        _write_json(output_file, updated_mesh_def)

    if args.geo_in is not None:
        geo_in_file = args.geo_in.resolve()
        geo_out_file = (
            args.geo_out.resolve()
            if args.geo_out is not None
            else geo_in_file.with_name(f"{geo_in_file.stem}_correlated.geo")
        )
        geo_written_count, geo_skipped_count = write_correlated_geo_file(
            geo_in_file,
            geo_out_file,
            point_map,
            curve_map,
            surface_map,
        )

    point_count = _count_identified(point_map)
    curve_count = _count_identified(curve_map)
    surface_count = _count_identified(surface_map)
    print(f"Point IDs identified: {point_count[0]} / {point_count[1]}")
    print(f"Curve IDs identified: {curve_count[0]} / {curve_count[1]}")
    print(f"Surface IDs identified: {surface_count[0]} / {surface_count[1]}")
    if not args.skip_json_update:
        print(f"Wrote updated mesh definition to {output_file}")
    if args.geo_in is not None:
        print(f"Wrote correlated .geo file to {geo_out_file}")
        print(f".geo statements written: {geo_written_count}")
        print(f".geo statements skipped: {geo_skipped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
