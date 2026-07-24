from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


TRI_3 = 5
QUAD_4 = 7
BC_WALL = 4000


def _cgns_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_+.-]", "_", name)
    return clean[:32] if clean else "boundary"


def _string_data(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("ascii"), dtype=np.int8)


def _fixed_attr(value: str, length: int) -> np.bytes_:
    encoded = value.encode("ascii")
    if len(encoded) > length:
        raise ValueError(f"CGNS attribute {value!r} is longer than {length} bytes.")
    return np.bytes_(encoded.ljust(length, b"\0"))


def _next_child_order(parent: h5py.Group) -> int:
    return sum(
        1
        for name, child in parent.items()
        if isinstance(child, h5py.Group) and not name.startswith(" ")
    )


def _add_node(parent: h5py.Group, name: str, label: str, data=None, cgns_type: str = "MT"):
    group = parent.create_group(name, track_order=True)
    group.attrs["name"] = _fixed_attr(name, 33)
    group.attrs["label"] = _fixed_attr(label, 33)
    group.attrs["type"] = _fixed_attr(cgns_type, 3)
    group.attrs[" order"] = np.array([_next_child_order(parent)], dtype=np.int32)
    if data is not None:
        group.create_dataset(" data", data=data)
    return group


def _mapbc_file_for_ugrid(ugrid_file: Path) -> Path:
    file_name = ugrid_file.name
    if file_name.endswith(".lb8.ugrid"):
        return ugrid_file.with_name(file_name.removesuffix(".lb8.ugrid") + ".mapbc")
    if file_name.endswith(".b8.ugrid"):
        return ugrid_file.with_name(file_name.removesuffix(".b8.ugrid") + ".mapbc")
    return ugrid_file.with_suffix(".mapbc")


def _read_mapbc(mapbc_file: Path) -> dict[int, tuple[int, str]]:
    lines = mapbc_file.read_text(encoding="ascii").splitlines()
    if not lines:
        raise ValueError(f"MAPBC file is empty: {mapbc_file}")
    patch_count = int(lines[0].strip())
    patches = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid MAPBC row in {mapbc_file}: {line!r}")
        patch_id = int(parts[0])
        patches[patch_id] = (int(parts[1]), parts[2])
    if len(patches) != patch_count:
        raise ValueError(f"MAPBC header says {patch_count} patches but {len(patches)} rows were found.")
    return patches


def _ugrid_endian(ugrid_file: Path) -> str:
    if ".lb8." in ugrid_file.name:
        return "<"
    if ".b8." in ugrid_file.name:
        return ">"
    raise ValueError(f"Cannot infer binary UGRID endianness from {ugrid_file.name}")


def read_surface_ugrid(ugrid_file: Path):
    endian = _ugrid_endian(ugrid_file)
    int_dtype = np.dtype(endian + "i4")
    float_dtype = np.dtype(endian + "f8")
    with ugrid_file.open("rb") as file:
        header = np.fromfile(file, dtype=int_dtype, count=7)
        if header.size != 7:
            raise ValueError(f"Could not read UGRID header from {ugrid_file}")
        n_nodes, n_tris, n_quads, n_tets, n_pyrs, n_prisms, n_hexes = header.tolist()
        if n_tets or n_pyrs or n_prisms or n_hexes:
            raise ValueError(f"Expected a surface-only UGRID, got header {header.tolist()}")
        coords = np.fromfile(file, dtype=float_dtype, count=n_nodes * 3).reshape((n_nodes, 3))
        tris = np.fromfile(file, dtype=int_dtype, count=n_tris * 3).reshape((n_tris, 3))
        quads = np.fromfile(file, dtype=int_dtype, count=n_quads * 4).reshape((n_quads, 4))
        tri_ids = np.fromfile(file, dtype=int_dtype, count=n_tris)
        quad_ids = np.fromfile(file, dtype=int_dtype, count=n_quads)
    return coords, tris, quads, tri_ids, quad_ids


def _section_groups(connectivity: np.ndarray, patch_ids: np.ndarray, element_type: int):
    groups = defaultdict(list)
    for index, patch_id in enumerate(patch_ids):
        groups[int(patch_id)].append(index)
    for patch_id, indices in sorted(groups.items()):
        selected = np.asarray(indices, dtype=np.int64)
        yield patch_id, element_type, connectivity[selected]


def _write_section(zone: h5py.Group, name: str, element_type: int, start: int, connectivity: np.ndarray) -> int:
    end = start + len(connectivity) - 1
    section = _add_node(zone, name, "Elements_t", np.array([element_type, 0], dtype=np.int32), "I4")
    _add_node(
        section,
        "ElementRange",
        "IndexRange_t",
        np.array([start, end], dtype=np.int32),
        "I4",
    )
    _add_node(
        section,
        "ElementConnectivity",
        "DataArray_t",
        connectivity.astype(np.int32, copy=False).reshape(-1),
        "I4",
    )
    return end


def _write_bc(zone_bc: h5py.Group, name: str, start: int, end: int) -> None:
    bc = _add_node(zone_bc, name, "BC_t", _string_data("FamilySpecified"), "C1")
    _add_node(bc, "PointRange", "IndexRange_t", np.array([[start], [end]], dtype=np.int32), "I4")
    _add_node(bc, "GridLocation", "GridLocation_t", _string_data("CellCenter"), "C1")
    _add_node(bc, "FamilyName", "FamilyName_t", _string_data(name), "C1")


def write_surface_cgns(
    ugrid_file: Path,
    cgns_file: Path,
    mapbc_file: Path | None = None,
    split_boundaries: bool = True,
) -> Path:
    if mapbc_file is None:
        mapbc_file = _mapbc_file_for_ugrid(ugrid_file)
    patches = _read_mapbc(mapbc_file) if split_boundaries else {}
    coords, tris, quads, tri_ids, quad_ids = read_surface_ugrid(ugrid_file)
    total_faces = len(tris) + len(quads)
    output_patch_ids = tri_ids

    with h5py.File(cgns_file, "w", track_order=True) as handle:
        handle.attrs["name"] = np.bytes_("HDF5 MotherNode")
        handle.attrs["label"] = np.bytes_("Root Node of HDF5 File")
        handle.attrs["type"] = np.bytes_("MT")
        handle.create_dataset(" format", data=_string_data("IEEE_LITTLE_32\0"))
        handle.create_dataset(" hdf5version", data=_string_data("HDF5 Version 1.8\0".ljust(33, "\0")))
        _add_node(
            handle,
            "CGNSLibraryVersion",
            "CGNSLibraryVersion_t",
            np.array([3.4], dtype=np.float32),
            "R4",
        )
        base = _add_node(handle, "Base", "CGNSBase_t", np.array([2, 3], dtype=np.int32), "I4")
        zone = _add_node(
            base,
            "Zone1",
            "Zone_t",
            np.array([[len(coords)], [total_faces], [0]], dtype=np.int32),
            "I4",
        )
        _add_node(zone, "ZoneType", "ZoneType_t", _string_data("Unstructured"), "C1")
        grid = _add_node(zone, "GridCoordinates", "GridCoordinates_t")
        _add_node(grid, "CoordinateX", "DataArray_t", coords[:, 0], "R8")
        _add_node(grid, "CoordinateY", "DataArray_t", coords[:, 1], "R8")
        _add_node(grid, "CoordinateZ", "DataArray_t", coords[:, 2], "R8")

        if split_boundaries:
            if len(quads):
                raise ValueError("Split-boundary CGNS output currently supports triangular surface UGRID files only.")
            zone_bc = _add_node(zone, "ZoneBC", "ZoneBC_t")
            element_start = 1
            split_patch_ids = []
            for patch_id, element_type, connectivity in _section_groups(tris, tri_ids, TRI_3):
                if patch_id not in patches:
                    raise ValueError(f"Patch ID {patch_id} is present in UGRID but missing in {mapbc_file}.")
                bc_code, patch_name = patches[patch_id]
                if bc_code != BC_WALL:
                    raise ValueError(f"Unsupported MAPBC code {bc_code} for patch {patch_id} ({patch_name})")
                section_name = _cgns_name(patch_name)
                element_end = _write_section(zone, section_name, element_type, element_start, connectivity)
                _write_bc(zone_bc, section_name, element_start, element_end)
                _add_node(base, section_name, "Family_t")
                split_patch_ids.append(np.full(len(connectivity), patch_id, dtype=np.int32))
                element_start = element_end + 1
            output_patch_ids = np.concatenate(split_patch_ids)
        else:
            if len(quads):
                raise ValueError("Simple CGNS output currently supports triangular surface UGRID files only.")
            section = _add_node(zone, "GridElements", "Elements_t", np.array([TRI_3, 0], dtype=np.int32), "I4")
            _add_node(
                section,
                "ElementRange",
                "IndexRange_t",
                np.array([1, len(tris)], dtype=np.int32),
                "I4",
            )
            _add_node(
                section,
                "ElementConnectivity",
                "DataArray_t",
                tris.astype(np.int32, copy=False).reshape(-1),
                "I4",
            )

        solution = _add_node(zone, "CellData", "FlowSolution_t")
        _add_node(solution, "GridLocation", "GridLocation_t", _string_data("CellCenter"), "C1")
        _add_node(solution, "PatchID", "DataArray_t", output_patch_ids.astype(np.int32, copy=False), "I4")

    return cgns_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a binary surface UGRID + MAPBC to CGNS/HDF5.")
    parser.add_argument("ugrid_file", type=Path)
    parser.add_argument("--mapbc", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--simple", action="store_true", help="Write one triangle section and only PatchID cell data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cgns_file = args.out or args.ugrid_file.with_name(
        args.ugrid_file.name.replace(".lb8.ugrid", ".cgns").replace(".b8.ugrid", ".cgns")
    )
    output = write_surface_cgns(args.ugrid_file, cgns_file, args.mapbc, split_boundaries=not args.simple)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
