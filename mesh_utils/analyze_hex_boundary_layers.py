from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DEFAULT_SECTIONS = (
    "quad_blade1",
    "quad_blade2",
    "quad_blade3",
    "quad_blade4",
    "quad_hub",
)

# CGNS HEXA_8 vertex numbering. Opposite face pairs are (0, 5), (1, 3), (2, 4).
HEX_FACES = (
    (0, 3, 2, 1),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (0, 4, 7, 3),
    (4, 5, 6, 7),
)
OPPOSITE_FACE = (5, 3, 4, 1, 2, 0)


@dataclass
class Profile:
    label: str
    first_key: bytes
    previous_key: bytes | None
    current_key: bytes
    normal: np.ndarray
    wall_centroid: np.ndarray
    distances: list[float]


def _key_rows(faces: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(np.sort(faces.astype(np.int64), axis=1))
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def _coords(zone: h5py.Group, nodes: np.ndarray) -> np.ndarray:
    unique = np.asarray(np.sort(np.unique(nodes)), dtype=np.int64)
    gc = zone["GridCoordinates"]
    values = np.empty((unique.size, 3), dtype=np.float64)
    for i, name in enumerate(("CoordinateX", "CoordinateY", "CoordinateZ")):
        values[:, i] = gc[name][" data"][unique]
    inverse = np.searchsorted(unique, nodes)
    return values[inverse]


def _read_wall_quads(zone: h5py.Group, sections: tuple[str, ...], max_faces: int, seed: int):
    rng = np.random.default_rng(seed)
    faces = []
    labels = []
    for section_name in sections:
        if section_name not in zone:
            continue
        section = zone[section_name]
        if int(section[" data"][()][0]) != 7:
            continue
        conn = section["ElementConnectivity"][" data"][()]
        section_faces = conn.reshape((-1, 4)).astype(np.int64) - 1
        faces.append(section_faces)
        labels.extend([section_name] * section_faces.shape[0])
    if not faces:
        raise ValueError("No quad wall sections found.")
    all_faces = np.vstack(faces)
    all_labels = np.asarray(labels)
    if all_faces.shape[0] > max_faces:
        sample = np.sort(rng.choice(all_faces.shape[0], size=max_faces, replace=False))
        all_faces = all_faces[sample]
        all_labels = all_labels[sample]
    face_coords = _coords(zone, all_faces.ravel()).reshape((-1, 4, 3))
    centroids = face_coords.mean(axis=1)
    normals = np.cross(face_coords[:, 1] - face_coords[:, 0], face_coords[:, 2] - face_coords[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 0.0
    normals[keep] /= lengths[keep, None]
    return all_faces[keep], all_labels[keep], centroids[keep], normals[keep]


def _stats(values: np.ndarray) -> str:
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return "count=0"
    p = np.percentile(values, [1, 5, 10, 50, 90, 95, 99])
    return (
        f"count={values.size}, min={np.min(values):.6g}, p01={p[0]:.6g}, "
        f"p05={p[1]:.6g}, p10={p[2]:.6g}, p50={p[3]:.6g}, p90={p[4]:.6g}, "
        f"p95={p[5]:.6g}, p99={p[6]:.6g}, max={np.max(values):.6g}, mean={np.mean(values):.6g}"
    )


def _scan_layer(zone: h5py.Group, profiles: list[Profile], chunk_cells: int) -> int:
    active = {profile.current_key: i for i, profile in enumerate(profiles)}
    if not active:
        return 0

    section = zone["HexElements"]
    conn = section["ElementConnectivity"][" data"]
    cell_count = conn.size // 8
    active_keys = np.frombuffer(b"".join(active.keys()), dtype=np.dtype((np.void, 32)))
    matched = 0
    for start in range(0, cell_count, chunk_cells):
        stop = min(start + chunk_cells, cell_count)
        cells = conn[start * 8 : stop * 8].reshape((-1, 8)).astype(np.int64) - 1
        for face_index, face_nodes in enumerate(HEX_FACES):
            faces = cells[:, face_nodes]
            keys = _key_rows(faces)
            mask = np.isin(keys, active_keys, assume_unique=False)
            if not np.any(mask):
                continue
            for row, key in zip(np.nonzero(mask)[0], keys[mask], strict=True):
                profile = profiles[active[bytes(key)]]
                opposite = cells[row, HEX_FACES[OPPOSITE_FACE[face_index]]]
                opposite_key = bytes(_key_rows(opposite.reshape(1, 4))[0])
                if profile.previous_key is not None and opposite_key == profile.previous_key:
                    continue
                profile.previous_key = profile.current_key
                profile.current_key = opposite_key
                pts = _coords(zone, opposite)
                distance = abs(float((pts.mean(axis=0) - profile.wall_centroid) @ profile.normal))
                profile.distances.append(distance)
                matched += 1
                active.pop(profile.previous_key, None)
                if not active:
                    return matched
            if active:
                active_keys = np.frombuffer(b"".join(active.keys()), dtype=np.dtype((np.void, 32)))
    return matched


def analyze(cgns_file: Path, zone_name: str, sections: tuple[str, ...], max_faces: int, layers: int, chunk_cells: int, seed: int) -> None:
    with h5py.File(cgns_file, "r") as handle:
        zone = handle["Base"][zone_name]
        faces, labels, centroids, normals = _read_wall_quads(zone, sections, max_faces, seed)
        keys = _key_rows(faces)
        profiles = [
            Profile(str(label), bytes(key), None, bytes(key), normal, centroid, [])
            for label, key, normal, centroid in zip(labels, keys, normals, centroids, strict=True)
        ]
        print(f"CGNS file: {cgns_file}")
        print(f"Zone: {zone_name}")
        print(f"Sampled wall quads={len(profiles)}")
        for layer in range(layers):
            before = sum(len(profile.distances) == layer for profile in profiles)
            matched = _scan_layer(
                zone,
                [profile for profile in profiles if len(profile.distances) == layer],
                chunk_cells,
            )
            after = sum(len(profile.distances) == layer + 1 for profile in profiles)
            print(f"Layer {layer + 1}: matched={matched}, complete_profiles={after} (started={before})")
            if matched == 0:
                break

        complete = [profile for profile in profiles if len(profile.distances) >= layers]
        if not complete:
            complete = [profile for profile in profiles if len(profile.distances) >= 2]
        distances = np.asarray([profile.distances for profile in complete], dtype=np.float64)
        labels = np.asarray([profile.label for profile in complete])
        print(f"Profiles used for statistics={distances.shape[0]}, layers_per_profile={distances.shape[1] if distances.size else 0}")
        for i in range(distances.shape[1]):
            print(f"Layer face distance {i + 1} [m]: {_stats(distances[:, i])}")
        if distances.shape[1] >= 2:
            thicknesses = np.diff(np.column_stack((np.zeros(distances.shape[0]), distances)), axis=1)
            ratios = thicknesses[:, 1:] / thicknesses[:, :-1]
            print(f"\nFirst cell thickness / y1 [m]: {_stats(thicknesses[:, 0])}")
            print(f"Growth ratio t2/t1: {_stats(ratios[:, 0])}")
            print(f"Median profile growth ratio: {_stats(np.median(ratios, axis=1))}")
        for prefix in ("quad_blade", "quad_hub"):
            mask = np.char.find(labels.astype(str), prefix) >= 0
            if np.any(mask):
                print(f"\n{prefix}:")
                print(f"  y1 [m]: {_stats(distances[mask, 0])}")
                if distances.shape[1] >= 2:
                    local_distances = distances[mask]
                    thicknesses = np.diff(np.column_stack((np.zeros(local_distances.shape[0]), local_distances)), axis=1)
                    ratios = thicknesses[:, 1:] / thicknesses[:, :-1]
                    print(f"  t2/t1: {_stats(ratios[:, 0])}")
                    print(f"  median growth: {_stats(np.median(ratios, axis=1))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cgns_file", type=Path)
    parser.add_argument("--zone", default="zone_r1")
    parser.add_argument("--sections", nargs="*", default=list(DEFAULT_SECTIONS))
    parser.add_argument("--max-faces", type=int, default=5000)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--chunk-cells", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    analyze(args.cgns_file, args.zone, tuple(args.sections), args.max_faces, args.layers, args.chunk_cells, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
