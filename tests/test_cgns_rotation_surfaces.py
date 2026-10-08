"""Geometry and CGNS-name checks without cloud access."""

import tempfile
import unittest
import runpy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from flow360 import u

from flow360_utils.flow360_tools import cgns_surfaces_inside_rotation_region


def node(parent, name, label, value=None):
    group = parent.create_group(name)
    group.attrs["label"] = np.bytes_(label)
    if value is not None:
        if isinstance(value, str):
            value = np.frombuffer(value.encode(), dtype=np.int8)
        group.create_dataset(" data", data=value)
    return group


def add_zone(base, name, coordinates):
    zone = node(base, name, "Zone_t")
    node(zone, "ZoneType", "ZoneType_t", "Unstructured")
    grid = node(zone, "GridCoordinates", "GridCoordinates_t")
    for i, axis in enumerate("XYZ"):
        node(grid, f"Coordinate{axis}", "DataArray_t", np.asarray(coordinates)[:, i])
    return zone, node(zone, "ZoneBC", "ZoneBC_t")


def vertex_bc(zone_bc, name, indices):
    bc = node(zone_bc, name, "BC_t", "BCWall")
    node(bc, "PointList", "IndexArray_t", [indices])
    return bc


class RotationSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "surface.cgns"
        self.cylinder = SimpleNamespace(
            center=np.array([1., 2., 3.]) * u.m,
            axis=(0., 0., 2.),
            height=2 * u.m,
            outer_radius=1 * u.m,
            inner_radius=0 * u.m,
        )

    def select(self, names, **kwargs):
        entities = [SimpleNamespace(name=name) for name in names]
        return [s.name for s in cgns_surfaces_inside_rotation_region(
            self.path, entities, self.cylinder, **kwargs
        )]

    def test_axial_radial_partial_and_touching_surfaces_in_mm(self):
        with h5py.File(self.path, "w") as mesh:
            base = node(mesh, "Base", "CGNSBase_t")
            _, bc = add_zone(base, "Zone", [
                [1000, 2000, 3000], [1500, 2000, 3500],  # inside
                [1000, 2000, 4100],  # beyond end plane
                [2100, 2000, 3000],  # beyond radius
                [2000, 2000, 4000],  # on end plane and radius
            ])
            vertex_bc(bc, "inside", [1, 2])
            vertex_bc(bc, "axial_outside", [3])
            vertex_bc(bc, "radial_outside", [4])
            vertex_bc(bc, "partial", [1, 4])
            vertex_bc(bc, "touching", [5])
        self.assertEqual(self.select([
            "inside", "axial_outside", "radial_outside", "partial", "touching"
        ]), ["inside", "touching"])

    def test_arbitrary_axis_and_vertex_point_range(self):
        self.cylinder.axis = (1., 1., 0.)
        direction = np.array([1., 1., 0.]) / np.sqrt(2)
        with h5py.File(self.path, "w") as mesh:
            base = node(mesh, "Base", "CGNSBase_t")
            _, bc = add_zone(base, "Zone", [
                np.array([1., 2., 3.]) + 0.9 * direction,
                np.array([1., 2., 3.]) - 0.9 * direction,
                np.array([1., 2., 3.]) + 1.1 * direction,
            ])
            patch = node(bc, "inside", "BC_t")
            node(patch, "PointRange", "IndexRange_t", [[1, 2]])
            vertex_bc(bc, "outside", [3])
        self.assertEqual(self.select(["inside", "outside"], mesh_length_unit="m"), ["inside"])

    def test_gmsh_cell_center_quad_family_aliases_and_multiple_zones(self):
        with h5py.File(self.path, "w") as mesh:
            base = node(mesh, "Base", "CGNSBase_t")
            for zone_name, z, family_name in (("Rotor", 3000, "S_1"), ("Stator", 5000, "S_2")):
                zone, bc = add_zone(base, zone_name, [
                    [1000, 2000, z], [1100, 2000, z],
                    [1100, 2100, z], [1000, 2100, z],
                ])
                section = node(zone, "Faces", "Elements_t", [7, 0])
                node(section, "ElementRange", "IndexRange_t", [10, 10])
                node(section, "ElementConnectivity", "DataArray_t", [1, 2, 3, 4])
                patch = node(bc, family_name, "BC_t")
                node(patch, "GridLocation", "GridLocation_t", "CellCenter")
                node(patch, "PointRange", "IndexRange_t", [[10, 10]])
                node(patch, "FamilyName", "FamilyName_t", family_name)
                family = node(base, family_name, "Family_t")
                node(family, "physical_group", "FamilyName_t", "shared_group")
        self.assertEqual(self.select([
            "S_1", "Rotor/S_1", "S_2", "shared_group", "Rotor/shared_group"
        ]), ["S_1", "Rotor/S_1", "Rotor/shared_group"])

    def test_face_center_triangle_point_list_uses_element_numbers(self):
        with h5py.File(self.path, "w") as mesh:
            base = node(mesh, "Base", "CGNSBase_t")
            zone, bc = add_zone(base, "Zone", [
                [1000, 2000, 3000], [1100, 2000, 3000],
                [1000, 2100, 3000], [1000, 2000, 5000],
            ])
            section = node(zone, "Faces", "Elements_t", [5, 0])
            node(section, "ElementRange", "IndexRange_t", [101, 102])
            node(section, "ElementConnectivity", "DataArray_t", [1, 2, 3, 1, 2, 4])
            for name, index in (("inside", 101), ("partial", 102)):
                patch = vertex_bc(bc, name, [index])
                node(patch, "GridLocation", "GridLocation_t", "FaceCenter")
        self.assertEqual(self.select(["inside", "partial"]), ["inside"])

    def test_unknown_name_fails_instead_of_silently_omitting_surface(self):
        with h5py.File(self.path, "w") as mesh:
            base = node(mesh, "Base", "CGNSBase_t")
            _, bc = add_zone(base, "Zone", [[1000, 2000, 3000]])
            vertex_bc(bc, "inside", [1])
        with self.assertRaisesRegex(ValueError, "could not be resolved"):
            self.select(["missing"])

    def test_pusher_keeps_unmatched_stator_as_wall_with_named_rotor_groups(self):
        script = Path(__file__).resolve().parents[1] / "DUC/Pusher/DUC_EVE_Pusher_noise.py"
        wall_entities = runpy.run_path(str(script), run_name="validation")["_wall_entities"]
        names = ["hub_front", "blade1", "blade2", "blade3", "blade4", "stator_main",
                 "zone_s1/farfield", "zone_r1/Interface_zone_s1"]

        class Mesh:
            boundary_names = names

            def __getitem__(self, name):
                return SimpleNamespace(name=name)

        groups = wall_entities(Mesh())
        self.assertEqual([surface.name for surface in groups["other"]], ["stator_main"])
        self.assertEqual(sum(len(surfaces) for surfaces in groups.values()), 6)


if __name__ == "__main__":
    unittest.main()
