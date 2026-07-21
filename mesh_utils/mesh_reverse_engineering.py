from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_CGNS = Path("DUC/duc_case1_mg13w2.cgns")
DEFAULT_ZONE = "zone_r1"
DEFAULT_SECTIONS = (
    "tri_blade1",
    "quad_blade1",
    "tri_blade2",
    "quad_blade2",
    "tri_blade3",
    "quad_blade3",
    "tri_blade4",
    "quad_blade4",
    "tri_hub",
    "quad_hub",
)
DEFAULT_GROUPS = (
    "blade1:tri_blade1,quad_blade1",
    "blade2:tri_blade2,quad_blade2",
    "blade3:tri_blade3,quad_blade3",
    "blade4:tri_blade4,quad_blade4",
    "hub:tri_hub,quad_hub",
)


@dataclass(frozen=True)
class SurfaceSection:
    name: str
    nodes: np.ndarray
    faces: np.ndarray
    face_size: int


def _decode_node_data(group: h5py.Group) -> str:
    data = group[" data"][()]
    return bytes(np.asarray(data, dtype=np.int8)).decode("utf-8", errors="replace").strip()


def _element_type_face_size(element_type: int) -> int:
    # CGNS element type numbers: TRI_3 = 5, QUAD_4 = 7.
    if element_type == 5:
        return 3
    if element_type == 7:
        return 4
    raise ValueError(f"Unsupported surface element type {element_type}; expected TRI_3 or QUAD_4.")


def _read_surface_section(zone: h5py.Group, section_name: str) -> SurfaceSection:
    section = zone[section_name]
    element_type = int(section[" data"][()][0])
    face_size = _element_type_face_size(element_type)
    conn = section["ElementConnectivity"][" data"][()]
    if conn.size % face_size:
        raise ValueError(f"{section_name} connectivity length is not divisible by {face_size}.")
    faces = conn.reshape((-1, face_size)).astype(np.int64) - 1
    nodes = np.unique(faces)
    return SurfaceSection(section_name, nodes, faces, face_size)


def _read_coordinates(zone: h5py.Group, nodes: np.ndarray) -> np.ndarray:
    gc = zone["GridCoordinates"]
    coords = np.empty((nodes.size, 3), dtype=np.float64)
    for i, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
        coords[:, i] = gc[name][" data"][nodes]
    return coords


def _section_coordinates(zone: h5py.Group, section: SurfaceSection) -> tuple[np.ndarray, np.ndarray]:
    coords = _read_coordinates(zone, section.nodes)
    local_index = np.empty(int(section.nodes[-1]) + 1, dtype=np.int64)
    local_index[section.nodes] = np.arange(section.nodes.size)
    local_faces = local_index[section.faces]
    return coords, local_faces


def _face_edges(face_size: int) -> tuple[tuple[int, int], ...]:
    if face_size == 3:
        return ((0, 1), (1, 2), (2, 0))
    return ((0, 1), (1, 2), (2, 3), (3, 0))


def _edge_lengths(coords: np.ndarray, faces: np.ndarray, face_size: int) -> np.ndarray:
    edge_chunks = []
    for a, b in _face_edges(face_size):
        pa = coords[faces[:, a]]
        pb = coords[faces[:, b]]
        edge_chunks.append(np.linalg.norm(pa - pb, axis=1))
    return np.concatenate(edge_chunks)


def _polygon_edges(face: np.ndarray) -> Iterable[tuple[int, int]]:
    for index in range(len(face)):
        a = int(face[index])
        b = int(face[(index + 1) % len(face)])
        yield tuple(sorted((a, b)))


def _mixed_edge_lengths(coords: np.ndarray, face_groups: list[np.ndarray]) -> np.ndarray:
    chunks = []
    for faces in face_groups:
        face_size = faces.shape[1]
        chunks.append(_edge_lengths(coords, faces, face_size))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    percentiles = np.percentile(values, [1, 5, 10, 50, 90, 95, 99])
    return {
        "count": float(values.size),
        "min": float(np.min(values)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _format_stats(stats: dict[str, float]) -> str:
    return (
        f"count={int(stats['count'])}, min={stats['min']:.6g}, p01={stats['p01']:.6g}, "
        f"p05={stats['p05']:.6g}, p10={stats['p10']:.6g}, p50={stats['p50']:.6g}, "
        f"p90={stats['p90']:.6g}, p95={stats['p95']:.6g}, p99={stats['p99']:.6g}, "
        f"max={stats['max']:.6g}, mean={stats['mean']:.6g}"
    )


def _face_normals(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = coords[faces[:, 0]]
    p1 = coords[faces[:, 1]]
    p2 = coords[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 0.0
    normals[good] /= lengths[good, None]
    return normals


def _mixed_face_normals(coords: np.ndarray, face_groups: list[np.ndarray]) -> list[np.ndarray]:
    return [_face_normals(coords, faces) for faces in face_groups]


def _feature_edges(
    coords: np.ndarray,
    faces: np.ndarray,
    face_size: int,
    *,
    min_dihedral_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = _face_normals(coords, faces)
    adjacency: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for a, b in _face_edges(face_size):
            edge = tuple(sorted((int(face[a]), int(face[b]))))
            adjacency.setdefault(edge, []).append(face_index)

    edges = []
    lengths = []
    angles = []
    min_cos = math.cos(math.radians(min_dihedral_deg))
    for (a, b), face_indices in adjacency.items():
        if len(face_indices) == 1:
            edges.append((a, b))
            lengths.append(float(np.linalg.norm(coords[a] - coords[b])))
            angles.append(180.0)
            continue
        if len(face_indices) != 2:
            continue
        dot = float(np.clip(np.dot(normals[face_indices[0]], normals[face_indices[1]]), -1.0, 1.0))
        if dot <= min_cos:
            edges.append((a, b))
            lengths.append(float(np.linalg.norm(coords[a] - coords[b])))
            angles.append(math.degrees(math.acos(dot)))

    if not edges:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return np.asarray(edges, dtype=np.int64), np.asarray(lengths), np.asarray(angles)


def _mixed_feature_edges(
    coords: np.ndarray,
    face_groups: list[np.ndarray],
    *,
    min_dihedral_deg: float,
    include_boundary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal_groups = _mixed_face_normals(coords, face_groups)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for group_index, faces in enumerate(face_groups):
        for face_index, face in enumerate(faces):
            for edge in _polygon_edges(face):
                adjacency.setdefault(edge, []).append((group_index, face_index))

    edges = []
    lengths = []
    angles = []
    growth_ratios = []
    side_lengths = []
    opposite_lengths = []
    min_cos = math.cos(math.radians(min_dihedral_deg))
    for (a, b), face_refs in adjacency.items():
        if len(face_refs) == 1:
            if include_boundary:
                edges.append((a, b))
                edge_length = float(np.linalg.norm(coords[a] - coords[b]))
                lengths.append(edge_length)
                angles.append(180.0)
                growth_ratios.append(np.nan)
                side_lengths.extend((np.nan, np.nan, np.nan, np.nan))
                opposite_lengths.append(np.nan)
            continue
        if len(face_refs) != 2:
            continue
        g0, f0 = face_refs[0]
        g1, f1 = face_refs[1]
        dot = float(np.clip(np.dot(normal_groups[g0][f0], normal_groups[g1][f1]), -1.0, 1.0))
        if dot <= min_cos:
            edges.append((a, b))
            edge_length = float(np.linalg.norm(coords[a] - coords[b]))
            lengths.append(edge_length)
            angles.append(math.degrees(math.acos(dot)))
            adjacent_lengths = []
            feature_side_lengths = []
            feature_opposite_lengths = []
            for group_index, face_index in face_refs:
                face = face_groups[group_index][face_index]
                for edge in _polygon_edges(face):
                    if edge == (a, b):
                        continue
                    length = float(np.linalg.norm(coords[edge[0]] - coords[edge[1]]))
                    adjacent_lengths.append(length)
                    if edge[0] in (a, b) or edge[1] in (a, b):
                        feature_side_lengths.append(length)
                    else:
                        feature_opposite_lengths.append(length)
            larger_adjacent = [value for value in adjacent_lengths if value >= edge_length]
            reference = min(larger_adjacent) if larger_adjacent else min(adjacent_lengths, default=edge_length)
            growth_ratios.append(reference / edge_length if edge_length else np.nan)
            side_lengths.extend(feature_side_lengths)
            opposite_lengths.extend(feature_opposite_lengths)

    if not edges:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.asarray(edges, dtype=np.int64),
        np.asarray(lengths),
        np.asarray(angles),
        np.asarray(growth_ratios),
        np.asarray(side_lengths),
        np.asarray(opposite_lengths),
    )


def _axis_spread(coords: np.ndarray) -> str:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    return (
        f"bbox_min=({mins[0]:.6g}, {mins[1]:.6g}, {mins[2]:.6g}), "
        f"bbox_max=({maxs[0]:.6g}, {maxs[1]:.6g}, {maxs[2]:.6g}), "
        f"span=({span[0]:.6g}, {span[1]:.6g}, {span[2]:.6g})"
    )


def list_cgns(cgns_file: Path) -> None:
    with h5py.File(cgns_file, "r") as handle:
        base = handle["Base"]
        for zone_name, zone in base.items():
            if not isinstance(zone, h5py.Group) or zone.attrs.get("label") != b"Zone_t":
                continue
            zone_data = zone[" data"][()]
            print(f"\n{zone_name}: vertices={int(zone_data[0, 0])}, cells={int(zone_data[1, 0])}")
            for name, group in zone.items():
                if isinstance(group, h5py.Group) and group.attrs.get("label") == b"Elements_t":
                    element_type = int(group[" data"][()][0])
                    element_range = group["ElementRange"][" data"][()]
                    count = int(element_range[1] - element_range[0] + 1)
                    print(f"  {name}: type={element_type}, count={count}")


def analyze_surfaces(
    cgns_file: Path,
    *,
    zone_name: str,
    sections: Iterable[str],
    min_dihedral_deg: float,
) -> None:
    with h5py.File(cgns_file, "r") as handle:
        zone = handle["Base"][zone_name]
        print(f"CGNS file: {cgns_file}")
        print(f"Zone: {zone_name}")
        print(f"Trailing-edge/feature threshold: dihedral >= {min_dihedral_deg:g} deg")
        for section_name in sections:
            if section_name not in zone:
                print(f"\n{section_name}: missing")
                continue
            section = _read_surface_section(zone, section_name)
            coords, faces = _section_coordinates(zone, section)
            edge_lengths = _edge_lengths(coords, faces, section.face_size)
            feature_edges, feature_lengths, feature_angles = _feature_edges(
                coords,
                faces,
                section.face_size,
                min_dihedral_deg=min_dihedral_deg,
            )
            print(f"\n{section_name}")
            print(f"  faces={faces.shape[0]}, unique_nodes={coords.shape[0]}, face_size={section.face_size}")
            print(f"  geometry: {_axis_spread(coords)}")
            print(f"  surface edge length: {_format_stats(_stats(edge_lengths))}")
            if feature_lengths.size:
                feature_coords = coords[np.unique(feature_edges)]
                print(f"  feature/trailing-edge candidate count={feature_lengths.size}")
                print(f"  feature edge length: {_format_stats(_stats(feature_lengths))}")
                print(f"  feature dihedral angle: {_format_stats(_stats(feature_angles))}")
                print(f"  feature region: {_axis_spread(feature_coords)}")
            else:
                print("  feature/trailing-edge candidate count=0")


def _parse_group(value: str) -> tuple[str, list[str]]:
    if ":" not in value:
        return value, [value]
    name, raw_sections = value.split(":", 1)
    return name, [item.strip() for item in raw_sections.split(",") if item.strip()]


def analyze_groups(
    cgns_file: Path,
    *,
    zone_name: str,
    groups: Iterable[str],
    min_dihedral_deg: float,
    include_boundary_features: bool,
) -> None:
    with h5py.File(cgns_file, "r") as handle:
        zone = handle["Base"][zone_name]
        print(f"CGNS file: {cgns_file}")
        print(f"Zone: {zone_name}")
        print(f"Sharp-feature threshold: dihedral >= {min_dihedral_deg:g} deg")
        print(f"Boundary edges included as features: {include_boundary_features}")
        for group_spec in groups:
            group_name, section_names = _parse_group(group_spec)
            sections = [_read_surface_section(zone, name) for name in section_names if name in zone]
            if not sections:
                print(f"\n{group_name}: no matching sections")
                continue

            global_nodes = np.unique(np.concatenate([section.nodes for section in sections]))
            coords = _read_coordinates(zone, global_nodes)
            local_index = np.empty(int(global_nodes[-1]) + 1, dtype=np.int64)
            local_index[global_nodes] = np.arange(global_nodes.size)
            face_groups = [local_index[section.faces] for section in sections]
            face_count = sum(faces.shape[0] for faces in face_groups)
            edge_lengths = _mixed_edge_lengths(coords, face_groups)
            (
                feature_edges,
                feature_lengths,
                feature_angles,
                growth_ratios,
                feature_side_lengths,
                feature_opposite_lengths,
            ) = _mixed_feature_edges(
                coords,
                face_groups,
                min_dihedral_deg=min_dihedral_deg,
                include_boundary=include_boundary_features,
            )

            print(f"\n{group_name} ({', '.join(section_names)})")
            print(f"  faces={face_count}, unique_nodes={coords.shape[0]}")
            print(f"  geometry: {_axis_spread(coords)}")
            print(f"  surface edge length: {_format_stats(_stats(edge_lengths))}")
            if feature_lengths.size:
                feature_coords = coords[np.unique(feature_edges)]
                print(f"  sharp/trailing-edge candidate count={feature_lengths.size}")
                print(f"  sharp/trailing-edge edge length: {_format_stats(_stats(feature_lengths))}")
                print(f"  sharp/trailing-edge dihedral angle: {_format_stats(_stats(feature_angles))}")
                valid_growth = growth_ratios[np.isfinite(growth_ratios)]
                if valid_growth.size:
                    print(f"  sharp/trailing-edge adjacent growth ratio: {_format_stats(_stats(valid_growth))}")
                valid_side_lengths = feature_side_lengths[np.isfinite(feature_side_lengths)]
                valid_opposite_lengths = feature_opposite_lengths[np.isfinite(feature_opposite_lengths)]
                if valid_side_lengths.size:
                    print(f"  sharp/trailing-edge side-away edge length: {_format_stats(_stats(valid_side_lengths))}")
                if valid_opposite_lengths.size:
                    print(f"  sharp/trailing-edge opposite edge length: {_format_stats(_stats(valid_opposite_lengths))}")
                print(f"  sharp/trailing-edge region: {_axis_spread(feature_coords)}")
            else:
                print("  sharp/trailing-edge candidate count=0")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse-engineer surface mesh sizing/refinement statistics from a CGNS file."
    )
    parser.add_argument("cgns_file", nargs="?", type=Path, default=DEFAULT_CGNS)
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument("--sections", nargs="*", default=list(DEFAULT_SECTIONS))
    parser.add_argument(
        "--groups",
        nargs="*",
        default=list(DEFAULT_GROUPS),
        help="Combined surface groups as name:section1,section2. Used unless --per-section is set.",
    )
    parser.add_argument("--per-section", action="store_true")
    parser.add_argument("--list", action="store_true", help="List zones and element sections only.")
    parser.add_argument(
        "--feature-dihedral-deg",
        type=float,
        default=35.0,
        help="Minimum adjacent-face dihedral angle for feature/trailing-edge candidates.",
    )
    parser.add_argument(
        "--include-boundary-features",
        action="store_true",
        help="Include one-sided boundary edges in feature statistics.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.list:
        list_cgns(args.cgns_file)
    elif args.per_section:
        analyze_surfaces(
            args.cgns_file,
            zone_name=args.zone,
            sections=args.sections,
            min_dihedral_deg=args.feature_dihedral_deg,
        )
    else:
        analyze_groups(
            args.cgns_file,
            zone_name=args.zone,
            groups=args.groups,
            min_dihedral_deg=args.feature_dihedral_deg,
            include_boundary_features=args.include_boundary_features,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
