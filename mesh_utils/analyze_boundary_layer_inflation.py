from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


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


def _read_coords(zone: h5py.Group, nodes: np.ndarray) -> np.ndarray:
    gc = zone["GridCoordinates"]
    coords = np.empty((nodes.size, 3), dtype=np.float64)
    for i, coord_name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
        coords[:, i] = gc[coord_name][" data"][nodes]
    return coords


def _read_surface_faces(zone: h5py.Group, sections: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    faces_by_size = []
    labels = []
    for section_name in sections:
        if section_name not in zone:
            continue
        section = zone[section_name]
        element_type = int(section[" data"][()][0])
        face_size = {5: 3, 7: 4}.get(element_type)
        if face_size is None:
            continue
        conn = section["ElementConnectivity"][" data"][()]
        faces = conn.reshape((-1, face_size)).astype(np.int64) - 1
        if face_size == 3:
            faces = np.column_stack((faces, faces[:, 2]))
        faces_by_size.append(faces)
        labels.extend([section_name] * faces.shape[0])
    if not faces_by_size:
        raise ValueError("No requested wall surface sections were found.")
    return np.concatenate(faces_by_size), np.asarray(labels)


def _face_samples(zone: h5py.Group, faces: np.ndarray, labels: np.ndarray, max_faces: int, seed: int):
    rng = np.random.default_rng(seed)
    if faces.shape[0] > max_faces:
        sample_indices = np.sort(rng.choice(faces.shape[0], size=max_faces, replace=False))
        faces = faces[sample_indices]
        labels = labels[sample_indices]

    nodes = np.unique(faces)
    coords = _read_coords(zone, nodes)
    local = np.empty(int(nodes[-1]) + 1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    face_coords = coords[local[faces]]

    centroids = np.mean(face_coords, axis=1)
    normals = np.cross(face_coords[:, 1] - face_coords[:, 0], face_coords[:, 2] - face_coords[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    keep = normal_lengths > 0.0
    normals[keep] /= normal_lengths[keep, None]
    return centroids[keep], normals[keep], labels[keep]


def _sample_near_wall_nodes(
    zone: h5py.Group,
    wall_tree: cKDTree,
    *,
    radius: float,
    stride: int,
    chunk_nodes: int,
) -> np.ndarray:
    total_nodes = int(zone[" data"][()][0, 0])
    gc = zone["GridCoordinates"]
    chunks = []
    for start in range(0, total_nodes, chunk_nodes):
        stop = min(start + chunk_nodes, total_nodes)
        indices = np.arange(start, stop, stride, dtype=np.int64)
        if indices.size == 0:
            continue
        coords = np.empty((indices.size, 3), dtype=np.float64)
        for i, coord_name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
            coords[:, i] = gc[coord_name][" data"][indices]
        distances, _ = wall_tree.query(coords, k=1, workers=-1)
        chunks.append(coords[distances <= radius])
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks)


def _stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values) & (values > 0.0)]
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
    if not stats:
        return "count=0"
    return (
        f"count={int(stats['count'])}, min={stats['min']:.6g}, p01={stats['p01']:.6g}, "
        f"p05={stats['p05']:.6g}, p10={stats['p10']:.6g}, p50={stats['p50']:.6g}, "
        f"p90={stats['p90']:.6g}, p95={stats['p95']:.6g}, p99={stats['p99']:.6g}, "
        f"max={stats['max']:.6g}, mean={stats['mean']:.6g}"
    )


def _layer_distances(
    centroids: np.ndarray,
    normals: np.ndarray,
    near_nodes: np.ndarray,
    *,
    query_radius: float,
    max_layers: int,
    min_tangent_ratio: float,
) -> np.ndarray:
    node_tree = cKDTree(near_nodes)
    layers = np.full((centroids.shape[0], max_layers), np.nan, dtype=np.float64)
    for i, (centroid, normal) in enumerate(zip(centroids, normals, strict=True)):
        candidate_ids = node_tree.query_ball_point(centroid, query_radius, workers=-1)
        if len(candidate_ids) < max_layers:
            continue
        rel = near_nodes[candidate_ids] - centroid
        signed = rel @ normal
        normal_distance = np.abs(signed)
        tangent_distance = np.linalg.norm(rel - signed[:, None] * normal, axis=1)
        mask = (normal_distance > 1.0e-10) & (tangent_distance <= min_tangent_ratio * normal_distance)
        values = np.sort(normal_distance[mask])
        if values.size < max_layers:
            continue
        unique_values = [values[0]]
        for value in values[1:]:
            if value > unique_values[-1] * 1.08:
                unique_values.append(value)
                if len(unique_values) == max_layers:
                    break
        if len(unique_values) == max_layers:
            layers[i] = unique_values
    return layers


def analyze(
    cgns_file: Path,
    *,
    zone_name: str,
    sections: tuple[str, ...],
    max_faces: int,
    near_radius: float,
    query_radius: float,
    node_stride: int,
    max_layers: int,
    seed: int,
) -> None:
    with h5py.File(cgns_file, "r") as handle:
        zone = handle["Base"][zone_name]
        faces, labels = _read_surface_faces(zone, sections)
        centroids, normals, labels = _face_samples(zone, faces, labels, max_faces, seed)
        wall_tree = cKDTree(centroids)
        near_nodes = _sample_near_wall_nodes(
            zone,
            wall_tree,
            radius=near_radius,
            stride=node_stride,
            chunk_nodes=2_000_000,
        )
        print(f"CGNS file: {cgns_file}")
        print(f"Zone: {zone_name}")
        print(f"Wall faces available={faces.shape[0]}, sampled={centroids.shape[0]}")
        print(f"Near-wall volume nodes sampled={near_nodes.shape[0]} (stride={node_stride}, radius={near_radius:g} m)")
        if near_nodes.shape[0] == 0:
            raise ValueError("No near-wall nodes were sampled; increase --near-radius or reduce --node-stride.")

        layers = _layer_distances(
            centroids,
            normals,
            near_nodes,
            query_radius=query_radius,
            max_layers=max_layers,
            min_tangent_ratio=0.75,
        )
        valid_rows = np.all(np.isfinite(layers), axis=1)
        print(f"Usable wall-normal profiles={int(np.count_nonzero(valid_rows))}")
        if not np.any(valid_rows):
            raise ValueError("No usable wall-normal profiles. Increase --near-radius/query-radius or reduce stride.")

        layer_values = layers[valid_rows]
        for layer_index in range(max_layers):
            print(f"Layer node distance {layer_index + 1} [m]: {_format_stats(_stats(layer_values[:, layer_index]))}")

        first_cell = layer_values[:, 0]
        print(f"\nEstimated y1 / first wall-normal node distance [m]: {_format_stats(_stats(first_cell))}")
        ratios = layer_values[:, 1:] / layer_values[:, :-1]
        first_growth = ratios[:, 0]
        median_growth = np.nanmedian(ratios, axis=1)
        print(f"First growth ratio d2/d1: {_format_stats(_stats(first_growth))}")
        print(f"Median profile growth ratio: {_format_stats(_stats(median_growth))}")

        for prefix in ("blade", "hub"):
            mask = np.char.find(labels[valid_rows].astype(str), prefix) >= 0
            if np.any(mask):
                print(f"\n{prefix}:")
                print(f"  y1 [m]: {_format_stats(_stats(first_cell[mask]))}")
                print(f"  first growth ratio: {_format_stats(_stats(first_growth[mask]))}")
                print(f"  median growth ratio: {_format_stats(_stats(median_growth[mask]))}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cgns_file", type=Path)
    parser.add_argument("--zone", default="zone_r1")
    parser.add_argument("--sections", nargs="*", default=list(DEFAULT_SECTIONS))
    parser.add_argument("--max-faces", type=int, default=20_000)
    parser.add_argument("--near-radius", type=float, default=0.0015)
    parser.add_argument("--query-radius", type=float, default=0.0015)
    parser.add_argument("--node-stride", type=int, default=4)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    analyze(
        args.cgns_file,
        zone_name=args.zone,
        sections=tuple(args.sections),
        max_faces=args.max_faces,
        near_radius=args.near_radius,
        query_radius=args.query_radius,
        node_stride=args.node_stride,
        max_layers=args.layers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
