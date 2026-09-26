import json
import unittest
from pathlib import Path
from unittest.mock import patch

import gmsh_surf_mesh


def _anisotropic_refinement_entities(dimension: int):
    if dimension == 1:
        return [(1, 96), (1, 407), (1, 416), (1, 420)]
    if dimension == 2:
        return [(2, 83), (2, 90), (2, 92)]
    return []


class DefaultGmshThreadCountTests(unittest.TestCase):
    @patch("gmsh_surf_mesh.psutil.cpu_count", return_value=16)
    def test_uses_one_less_than_physical_core_count(self, _cpu_count) -> None:
        self.assertEqual(gmsh_surf_mesh.default_gmsh_thread_count(), 15)

    @patch("gmsh_surf_mesh.psutil.cpu_count", return_value=1)
    def test_keeps_at_least_one_thread(self, _cpu_count) -> None:
        self.assertEqual(gmsh_surf_mesh.default_gmsh_thread_count(), 1)

    @patch("gmsh_surf_mesh.os.cpu_count", return_value=8)
    @patch("gmsh_surf_mesh.psutil", None)
    def test_falls_back_to_logical_cpu_count(self, _cpu_count) -> None:
        self.assertEqual(gmsh_surf_mesh.default_gmsh_thread_count(), 7)


class GmshPointDisplayTests(unittest.TestCase):
    @patch("gmsh_surf_mesh.gmsh.option.setNumber")
    def test_configures_spherical_points_at_size_six(self, set_number) -> None:
        gmsh_surf_mesh.configure_gmsh_point_display()

        set_number.assert_any_call("Geometry.PointType", 1)
        set_number.assert_any_call("Geometry.PointSize", 6)


class GmshOptionTests(unittest.TestCase):
    @patch("gmsh_surf_mesh.gmsh.option.setNumber")
    @patch("gmsh_surf_mesh.default_gmsh_thread_count", return_value=15)
    def test_serializes_per_surface_bamg_meshing(
        self, _default_thread_count, set_number
    ) -> None:
        gmsh_surf_mesh.apply_gmsh_thread_limit(
            {"surface_meshing_algorithms": [{"surfaces": [95], "algorithm": "bamg"}]}
        )

        self.assertEqual(
            set_number.call_args_list,
            [
                (("General.NumThreads", 15),),
                (("Mesh.MaxNumThreads1D", 15),),
                (("Mesh.MaxNumThreads2D", 15),),
                (("Mesh.MaxNumThreads3D", 15),),
                (("Mesh.MaxNumThreads2D", 1),),
            ],
        )

    @patch("gmsh_surf_mesh.gmsh.option.setNumber")
    @patch("gmsh_surf_mesh.default_gmsh_thread_count", return_value=15)
    def test_serializes_anisotropic_refinement_meshing(
        self, _default_thread_count, set_number
    ) -> None:
        gmsh_surf_mesh.apply_gmsh_thread_limit(
            {"anisotropic_curve_refinement": {"surfaces": [83]}}
        )

        set_number.assert_any_call("Mesh.MaxNumThreads1D", 15)
        set_number.assert_any_call("Mesh.MaxNumThreads2D", 1)

    @patch("gmsh_surf_mesh.gmsh.option.setNumber")
    @patch("gmsh_surf_mesh.default_gmsh_thread_count", return_value=15)
    def test_preserves_default_limit_without_bamg(
        self, _default_thread_count, set_number
    ) -> None:
        gmsh_surf_mesh.apply_gmsh_thread_limit(
            {
                "surface_meshing_algorithms": [
                    {"surfaces": [95], "algorithm": "frontal-delaunay"}
                ],
            }
        )

        self.assertEqual(set_number.call_count, 4)
        set_number.assert_any_call("Mesh.MaxNumThreads2D", 15)

    @patch("gmsh_surf_mesh.gmsh.option.setNumber")
    @patch("gmsh_surf_mesh.default_gmsh_thread_count", return_value=15)
    def test_serializes_global_numeric_bamg_meshing(
        self, _default_thread_count, set_number
    ) -> None:
        gmsh_surf_mesh.apply_gmsh_thread_limit(
            {"surface_meshing_algorithm": 7}
        )

        set_number.assert_any_call("Mesh.MaxNumThreads2D", 1)

    @patch("gmsh_surf_mesh.gmsh.option.setNumber")
    def test_applies_requested_global_surface_algorithm(self, set_number) -> None:
        gmsh_surf_mesh.apply_global_surface_meshing_algorithm(
            {"surface_meshing_algorithm": "frontal-delaunay"}
        )

        set_number.assert_called_once_with("Mesh.Algorithm", 6)


class SurfaceMeshGenerationTests(unittest.TestCase):
    def test_generates_curves_with_tangent_field_before_anisotropic_fields(self) -> None:
        events = []
        mesh_def = {"mesh_zones": [{"curve_definition": "manual"}]}

        with (
            patch("gmsh_surf_mesh.gmsh") as gmsh,
            patch("gmsh_surf_mesh.configure_gmsh_point_display"),
            patch("gmsh_surf_mesh.apply_gmsh_thread_limit"),
            patch("gmsh_surf_mesh.apply_geometry_preprocessing"),
            patch("gmsh_surf_mesh.apply_geometry_healing"),
            patch("gmsh_surf_mesh.apply_degree_two_curve_compounds"),
            patch("gmsh_surf_mesh.apply_mesh_size_bounds"),
            patch("gmsh_surf_mesh.apply_surface_size_limits"),
            patch("gmsh_surf_mesh.expand_mesh_zones", return_value=mesh_def),
            patch("gmsh_surf_mesh.apply_transfinite_curves", return_value={}),
            patch("gmsh_surf_mesh.apply_automatic_transfinite_surfaces", return_value=mesh_def),
            patch("gmsh_surf_mesh.complete_surface_boundary_curves"),
            patch("gmsh_surf_mesh.apply_transfinite_surfaces"),
            patch("gmsh_surf_mesh.apply_global_surface_meshing_algorithm"),
            patch("gmsh_surf_mesh.apply_surface_meshing_algorithms"),
            patch("gmsh_surf_mesh.apply_boundary_layers"),
            patch(
                "gmsh_surf_mesh.configure_anisotropic_curve_tangential_field",
                side_effect=lambda _mesh_def: events.append("tangent field") or 42,
            ),
            patch(
                "gmsh_surf_mesh.generate_anisotropic_surface_mesh",
                side_effect=lambda _mesh_def: events.append("anisotropic fields"),
            ),
        ):
            gmsh.model.mesh.generate.side_effect = lambda dimension: events.append(
                f"generate({dimension})"
            )
            gmsh.model.mesh.field.setAsBackgroundMesh.side_effect = (
                lambda field_id: events.append(f"background({field_id})")
            )
            gmsh.model.occ.importShapes.return_value = []

            gmsh_surf_mesh.generate_surface_mesh(
                Path("model.step"), mesh_def, None, recombine=False, show=False
            )

        self.assertEqual(
            events,
            ["tangent field", "generate(1)", "background(0)", "anisotropic fields"],
        )

    @patch("gmsh_surf_mesh.gmsh.model.mesh.field.setAsBackgroundMesh")
    @patch("gmsh_surf_mesh._configure_anisotropic_curve_refinement_fields")
    @patch("gmsh_surf_mesh.gmsh.model.mesh.generate")
    @patch("gmsh_surf_mesh.gmsh.option.setNumber")
    @patch("gmsh_surf_mesh.gmsh.model.setVisibility")
    @patch("gmsh_surf_mesh.gmsh.model.getVisibility", side_effect=[1, 0, 1])
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        return_value=[(2, 83), (2, 90), (2, 92)],
    )
    def test_meshes_anisotropic_surfaces_before_remaining_empty_surfaces(
        self,
        _get_entities,
        get_visibility,
        set_visibility,
        set_number,
        generate,
        configure_fields,
        set_background,
    ) -> None:
        gmsh_surf_mesh.generate_anisotropic_surface_mesh(
            {"anisotropic_curve_refinement": {"surfaces": [83, 92]}}
        )

        self.assertEqual(generate.call_args_list, [((2,),), ((2,),)])
        configure_fields.assert_called_once()
        set_background.assert_called_once_with(0)
        self.assertEqual(
            [call.args for call in set_number.call_args_list],
            [
                ("Mesh.MeshOnlyVisible", 1),
                ("Mesh.MeshOnlyEmpty", 1),
                ("Mesh.MeshOnlyVisible", 0),
                ("Mesh.MeshOnlyEmpty", 1),
                ("Mesh.MeshOnlyVisible", 0),
                ("Mesh.MeshOnlyEmpty", 0),
            ],
        )
        self.assertEqual(
            set_visibility.call_args_list,
            [
                (([(2, 83), (2, 90), (2, 92)], 0),),
                (([(2, 83), (2, 92)], 1),),
                (([(2, 83), (2, 90), (2, 92)], 1),),
                (([(2, 83)], 1),),
                (([(2, 90)], 0),),
                (([(2, 92)], 1),),
            ],
        )
        self.assertEqual(get_visibility.call_count, 3)

    def test_groups_overlapping_refinement_surfaces(self) -> None:
        first = {"surfaces": [83, 90]}
        second = {"surfaces": [90, 92]}
        third = {"surfaces": [95]}

        groups = gmsh_surf_mesh._anisotropic_refinement_surface_groups(
            {"anisotropic_curve_refinements": [first, second, third]}
        )

        self.assertEqual(
            [(len(entries), surfaces) for entries, surfaces in groups],
            [(2, {83, 90, 92}), (1, {95})],
        )


class AnisotropicCurveRefinementTests(unittest.TestCase):
    def _refinement(self) -> dict[str, object]:
        return {
            "curves": [407, 416, 420],
            "surfaces": [83, 90],
            "sampling": 1000,
            "size_min_normal": 0.1,
            "size_min_tangent": 3.0,
            "size_max_normal": 3.0,
            "size_max_tangent": 3.0,
            "dist_min": 1.0,
            "dist_max": 20.0,
        }

    @patch("gmsh_surf_mesh.gmsh.model.mesh.field")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_configures_anisotropic_background_field(self, _get_entities, field) -> None:
        field.add.return_value = 12

        gmsh_surf_mesh.apply_anisotropic_curve_refinement(
            {"anisotropic_curve_refinement": self._refinement()}
        )

        self.assertEqual(
            field.add.call_args_list,
            [(('AttractorAnisoCurve',),)],
        )
        field.setNumbers.assert_any_call(12, "CurvesList", [407, 416, 420])
        field.setNumber.assert_any_call(12, "Sampling", 1000)
        field.setNumber.assert_any_call(12, "SizeMinNormal", 0.1)
        field.setNumber.assert_any_call(12, "SizeMaxTangent", 3.0)
        field.setNumber.assert_any_call(12, "DistMin", 1.0)
        field.setNumber.assert_any_call(12, "DistMax", 20.0)
        field.setAsBackgroundMesh.assert_called_once_with(12)

    @patch("gmsh_surf_mesh._surface_boundary_curves", side_effect=[[407, 416], [420]])
    @patch("gmsh_surf_mesh.gmsh.model.mesh.field")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_configures_tangent_field_on_target_boundary_curves(
        self, _get_entities, field, _boundary_curves
    ) -> None:
        field.add.side_effect = [11, 12, 13]
        refinement = self._refinement()
        refinement["size_min_tangent"] = 0.3
        refinement["size_max_tangent"] = 1.0

        field_id = gmsh_surf_mesh.configure_anisotropic_curve_tangential_field(
            {"anisotropic_curve_refinement": refinement}
        )

        self.assertEqual(field_id, 13)
        self.assertEqual(
            [call.args for call in field.add.call_args_list],
            [("Distance",), ("Threshold",), ("Restrict",)],
        )
        field.setNumbers.assert_any_call(11, "CurvesList", [407, 416, 420])
        field.setNumber.assert_any_call(11, "Sampling", 1000)
        field.setNumber.assert_any_call(12, "SizeMin", 0.3)
        field.setNumber.assert_any_call(12, "SizeMax", 1.0)
        field.setNumber.assert_any_call(12, "DistMin", 1.0)
        field.setNumber.assert_any_call(12, "DistMax", 20.0)
        field.setNumbers.assert_any_call(13, "CurvesList", [407, 416, 420])
        field.setAsBackgroundMesh.assert_called_once_with(13)

    @patch("gmsh_surf_mesh._surface_boundary_curves", side_effect=[[407], [416], [416], [96]])
    @patch("gmsh_surf_mesh.gmsh.model.mesh.field")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_merges_tangent_fields_for_shared_boundary_curves(
        self, _get_entities, field, _boundary_curves
    ) -> None:
        field.add.side_effect = [11, 12, 13, 14, 15, 16, 17]
        second = self._refinement()
        second["curves"] = [96]
        second["surfaces"] = [90, 92]

        field_id = gmsh_surf_mesh.configure_anisotropic_curve_tangential_field(
            {"anisotropic_curve_refinements": [self._refinement(), second]}
        )

        self.assertEqual(field_id, 17)
        field.setNumbers.assert_any_call(13, "CurvesList", [407, 416])
        field.setNumbers.assert_any_call(16, "CurvesList", [96, 416])
        field.setNumbers.assert_any_call(17, "FieldsList", [13, 16])
        field.setAsBackgroundMesh.assert_called_once_with(17)

    @patch("gmsh_surf_mesh.gmsh.model.mesh.field")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_combines_multiple_anisotropic_background_fields(
        self, _get_entities, field
    ) -> None:
        second_refinement = self._refinement()
        second_refinement["curves"] = [96]
        second_refinement["surfaces"] = [92]
        field.add.side_effect = [12, 13, 14]

        gmsh_surf_mesh.apply_anisotropic_curve_refinement(
            {
                "anisotropic_curve_refinements": [
                    self._refinement(),
                    second_refinement,
                ]
            }
        )

        self.assertEqual(
            field.add.call_args_list,
            [
                (('AttractorAnisoCurve',),),
                (('AttractorAnisoCurve',),),
                (('MinAniso',),),
            ],
        )
        field.setNumbers.assert_any_call(12, "CurvesList", [407, 416, 420])
        field.setNumbers.assert_any_call(13, "CurvesList", [96])
        field.setNumbers.assert_any_call(14, "FieldsList", [12, 13])
        field.setAsBackgroundMesh.assert_called_once_with(14)

    def test_rejects_singular_and_plural_refinement_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "either"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {
                    "anisotropic_curve_refinement": self._refinement(),
                    "anisotropic_curve_refinements": [self._refinement()],
                }
            )

    def test_rejects_empty_plural_refinements(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinements": []}
            )

    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_rejects_inverted_distance_range(self, _get_entities) -> None:
        refinement = self._refinement()
        refinement["dist_min"] = 20.0
        refinement["dist_max"] = 1.0

        with self.assertRaisesRegex(ValueError, "dist_max"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinement": refinement}
            )

    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_rejects_unknown_refinement_curve(self, _get_entities) -> None:
        refinement = self._refinement()
        refinement["curves"] = [999]

        with self.assertRaisesRegex(ValueError, "unknown curves"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinement": refinement}
            )

    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_rejects_invalid_sampling(self, _get_entities) -> None:
        refinement = self._refinement()
        refinement["sampling"] = 0

        with self.assertRaisesRegex(ValueError, "sampling"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinement": refinement}
            )

    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_rejects_non_positive_field_size(self, _get_entities) -> None:
        refinement = self._refinement()
        refinement["size_min_normal"] = 0.0

        with self.assertRaisesRegex(ValueError, "size_min_normal"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinement": refinement}
            )

    def test_requires_refinement_surfaces(self) -> None:
        refinement = self._refinement()
        del refinement["surfaces"]

        with self.assertRaisesRegex(ValueError, "non-empty 'surfaces'"):
            gmsh_surf_mesh._anisotropic_refinement_surface_ids(
                {"anisotropic_curve_refinement": refinement}
            )

    def test_rejects_invalid_refinement_name(self) -> None:
        refinement = self._refinement()
        refinement["name"] = " "

        with self.assertRaisesRegex(ValueError, "name must be a non-empty string"):
            gmsh_surf_mesh._anisotropic_refinement_surface_ids(
                {"anisotropic_curve_refinement": refinement}
            )

    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_uses_refinement_name_in_validation_errors(self, _get_entities) -> None:
        refinement = self._refinement()
        refinement["name"] = "tip surfaces"
        refinement["curves"] = [999]

        with self.assertRaisesRegex(ValueError, "anisotropic curve refinement 'tip surfaces'"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinement": refinement}
            )

    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=_anisotropic_refinement_entities,
    )
    def test_rejects_unknown_refinement_surface(self, _get_entities) -> None:
        refinement = self._refinement()
        refinement["surfaces"] = [999]

        with self.assertRaisesRegex(ValueError, "unknown surfaces"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinement": refinement}
            )


class AnisotropicCurveFieldIntegrationTests(unittest.TestCase):
    def test_tangent_fields_size_curves_and_preserve_explicit_counts(self) -> None:
        gmsh = gmsh_surf_mesh.gmsh
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
            gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
            gmsh.model.add("anisotropic_curve_field_test")
            surfaces = [
                gmsh.model.occ.addRectangle(x, 0, 0, 1, 1)
                for x in (0, 3, 6, 9)
            ]
            gmsh.model.occ.synchronize()
            source_curves = [
                min(
                    tag
                    for dim, tag in gmsh.model.getBoundary(
                        [(2, surface)], oriented=False
                    )
                    if dim == 1
                )
                for surface in surfaces
            ]
            refinements = [
                {
                    "curves": [source_curves[index]],
                    "surfaces": [surfaces[index]],
                    "sampling": 100,
                    "size_min_normal": 0.05,
                    "size_min_tangent": tangent_size,
                    "size_max_normal": 0.1,
                    "size_max_tangent": 0.5,
                    "dist_min": 0.01,
                    "dist_max": 0.5,
                }
                for index, tangent_size in enumerate((0.2, 0.05, 0.05))
            ]
            mesh_def = {"anisotropic_curve_refinements": refinements}
            gmsh.model.mesh.setTransfiniteCurve(source_curves[2], 4)
            gmsh_surf_mesh.apply_surface_meshing_algorithms(mesh_def)
            gmsh_surf_mesh.configure_anisotropic_curve_tangential_field(mesh_def)
            gmsh.model.mesh.generate(1)
            gmsh.model.mesh.field.setAsBackgroundMesh(0)

            def element_count(dim: int, tag: int) -> int:
                return sum(len(tags) for tags in gmsh.model.mesh.getElements(dim, tag)[1])

            self.assertEqual(
                [element_count(1, curve) for curve in source_curves[:3]],
                [5, 20, 3],
            )
            self.assertLess(element_count(1, source_curves[3]), 5)
            gmsh_surf_mesh.generate_anisotropic_surface_mesh(mesh_def)
            self.assertTrue(all(element_count(2, surface) > 0 for surface in surfaces))
            self.assertEqual(
                [element_count(1, curve) for curve in source_curves[:3]],
                [5, 20, 3],
            )
        finally:
            gmsh.finalize()


class AnisotropicSurfaceAlgorithmTests(unittest.TestCase):
    @patch("gmsh_surf_mesh.gmsh.model.mesh.setAlgorithm")
    @patch("gmsh_surf_mesh.gmsh.model.getEntities", return_value=[(2, 83), (2, 90)])
    def test_selects_bamg_for_refinement_surfaces(
        self, _get_entities, set_algorithm
    ) -> None:
        mesh_def = {"anisotropic_curve_refinement": {"surfaces": [90, 83]}}

        gmsh_surf_mesh.apply_surface_meshing_algorithms(mesh_def)

        self.assertEqual(
            set_algorithm.call_args_list,
            [((2, 83, 7),), ((2, 90, 7),)],
        )
        self.assertEqual(
            gmsh_surf_mesh._surface_meshing_algorithm_surface_ids(mesh_def), {83, 90}
        )

    @patch("gmsh_surf_mesh.gmsh.model.mesh.setAlgorithm")
    @patch("gmsh_surf_mesh.gmsh.model.getEntities", return_value=[(2, 83)])
    def test_rejects_non_bamg_algorithm_on_refinement_surface(
        self, _get_entities, _set_algorithm
    ) -> None:
        with self.assertRaisesRegex(ValueError, "must use BAMG"):
            gmsh_surf_mesh.apply_surface_meshing_algorithms(
                {
                    "anisotropic_curve_refinement": {"surfaces": [83]},
                    "surface_meshing_algorithms": [
                        {"surfaces": [83], "algorithm": "frontal-delaunay"}
                    ],
                }
            )


class ManualCurveInversionTests(unittest.TestCase):
    @patch("gmsh_surf_mesh.gmsh.model.mesh.setTransfiniteCurve")
    def test_inverts_manual_curve_progression_once(self, set_transfinite_curve) -> None:
        gmsh_surf_mesh.apply_transfinite_curves(
            {
                "explicit_curve_sequences": [
                    {
                        "curve_ids": [408, 421],
                        "type": "Progression",
                        "n_pts": 10,
                        "Parameter": 2.0,
                        "invert_direction": [False, True],
                    }
                ]
            }
        )

        self.assertEqual(
            set_transfinite_curve.call_args_list,
            [
                ((408, 10, "Progression", 2.0),),
                ((421, 10, "Progression", 0.5),),
            ],
        )

    def test_preserves_explicit_group_inversion(self) -> None:
        specs = list(
            gmsh_surf_mesh._iter_curve_specs(
                {
                    "curve_ids": [408],
                    "type": "Progression",
                    "n_pts": 10,
                    "Parameter": 2.0,
                    "invert_direction": [False],
                    "_group_invert_direction": [True],
                }
            )
        )

        self.assertTrue(specs[0].group_invert_direction)


class Poc2ConfigurationTests(unittest.TestCase):
    def test_configures_anisotropic_tip_refinement_without_boundary_layers(self) -> None:
        case_dir = Path(__file__).resolve().parent.parent / "DUC"
        with (case_dir / "msh_def_POC2.json").open(encoding="utf-8") as file:
            mesh_def = json.load(file)["mesh definition"]

        self.assertEqual(mesh_def["surface_meshing_algorithm"], "frontal-delaunay")
        self.assertNotIn("surface_meshing_algorithms", mesh_def)
        self.assertNotIn("max_num_threads_2d", mesh_def)
        self.assertNotIn("anisotropic_curve_refinement", mesh_def)
        self.assertNotIn("boundary_layers", mesh_def)
        refinements = mesh_def["anisotropic_curve_refinements"]
        self.assertTrue(
            all(
                isinstance(refinement.get("name"), str) and refinement["name"].strip()
                for refinement in refinements
            )
        )
        self.assertTrue(
            all(
                isinstance(refinement.get("curves"), list)
                and refinement["curves"]
                and isinstance(refinement.get("surfaces"), list)
                and refinement["surfaces"]
                for refinement in refinements
            )
        )
        manual_zone = next(
            zone
            for zone in mesh_def["mesh_zones"]
            if zone["name"] == "tip_curve_constraints"
        )
        self.assertEqual(
            manual_zone,
            {
                "name": "tip_curve_constraints",
                "curve_definition": "manual",
                "explicit_curve_sequences": [
                    {
                        "curve_ids": [408, 421, 952, 948, 1209, 1222, 693, 689],
                        "type": "Progression",
                        "mesh size mode": "ele size",
                        "target ele size 1": 0.3,
                        "target ele size 2": 0.1,
                        "invert_direction": [False, True, False, True, False, True, False, True],
                    },
                    {
                        "curve_ids": [409, 422, 953, 950, 1210, 1223, 694, 691],
                        "mesh size mode": "ele size",
                        "target ele size 1": 0.3,
                        "target ele size 2": 0.1,
                        "invert_direction": [False, True, False, True, False, True, False, True],
                    }
                ],
            },
        )


class BoundaryLayerTests(unittest.TestCase):
    @patch("gmsh_surf_mesh.gmsh.model.mesh.field")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getAdjacencies",
        return_value=([8, 95], []),
    )
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        side_effect=[[(1, 108), (1, 429)], [(2, 8), (2, 95)]],
    )
    def test_configures_boundary_layer_field(
        self, _get_entities, _get_adjacencies, field
    ) -> None:
        field.add.return_value = 7

        gmsh_surf_mesh.apply_boundary_layers(
            {
                "boundary_layers": [
                    {
                        "curves": [429, 108],
                        "surfaces": [95],
                        "size": 1.0,
                        "thickness": 20.0,
                        "n_layers": 10,
                        "ratio": 1.2,
                        "quads": True,
                        "size_far": 3.0,
                    }
                ]
            }
        )

        field.setNumbers.assert_any_call(7, "CurvesList", [429, 108])
        field.setNumbers.assert_any_call(7, "ExcludedSurfacesList", [8])
        field.setNumber.assert_any_call(7, "NbLayers", 10)
        field.setNumber.assert_any_call(7, "Quads", 1)
        field.setAsBoundaryLayer.assert_called_once_with(7)


class SurfaceSizeLimitTests(unittest.TestCase):
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        return_value=[(2, 7), (2, 8), (2, 94), (2, 95)],
    )
    def test_configures_surface_maximum_size_callback(
        self, _get_entities
    ) -> None:
        with patch("gmsh_surf_mesh.gmsh.model.mesh.setSizeCallback") as callback:
            gmsh_surf_mesh.apply_surface_size_limits(
                {
                    "surface_size_limits": [
                        {"surfaces": [7, 95], "max_element_size": 5.0},
                        {"surfaces": [8, 94], "max_element_size": 10.0},
                    ]
                }
            )

        size_callback = callback.call_args.args[0]
        self.assertEqual(size_callback(2, 7, 0.0, 0.0, 0.0, 8.0), 5.0)
        self.assertEqual(size_callback(2, 95, 0.0, 0.0, 0.0, 3.0), 3.0)
        self.assertEqual(size_callback(2, 8, 0.0, 0.0, 0.0, 12.0), 10.0)
        self.assertEqual(size_callback(1, 108, 0.0, 0.0, 0.0, 12.0), 12.0)


if __name__ == "__main__":
    unittest.main()
