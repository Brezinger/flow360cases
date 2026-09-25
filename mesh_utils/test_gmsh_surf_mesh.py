import json
import unittest
from pathlib import Path
from unittest.mock import patch

import gmsh_surf_mesh


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
    def test_generates_curves_before_installing_anisotropic_fields(self) -> None:
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
                "gmsh_surf_mesh.apply_anisotropic_curve_refinement",
                side_effect=lambda _mesh_def: events.append("anisotropic fields"),
            ),
        ):
            gmsh.model.mesh.generate.side_effect = lambda dimension: events.append(
                f"generate({dimension})"
            )
            gmsh.model.occ.importShapes.return_value = []

            gmsh_surf_mesh.generate_surface_mesh(
                Path("model.step"), mesh_def, None, recombine=False, show=False
            )

        self.assertEqual(
            events,
            ["generate(1)", "anisotropic fields", "generate(2)"],
        )


class AnisotropicCurveRefinementTests(unittest.TestCase):
    def _refinement(self) -> dict[str, object]:
        return {
            "curves": [407, 416, 420],
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
        return_value=[(1, 407), (1, 416), (1, 420)],
    )
    def test_configures_anisotropic_background_field(self, _get_entities, field) -> None:
        field.add.return_value = 12

        gmsh_surf_mesh.apply_anisotropic_curve_refinement(
            {"anisotropic_curve_refinement": self._refinement()}
        )

        field.add.assert_called_once_with("AttractorAnisoCurve")
        field.setNumbers.assert_called_once_with(12, "CurvesList", [407, 416, 420])
        field.setNumber.assert_any_call(12, "Sampling", 1000)
        field.setNumber.assert_any_call(12, "SizeMinNormal", 0.1)
        field.setNumber.assert_any_call(12, "SizeMaxTangent", 3.0)
        field.setNumber.assert_any_call(12, "DistMin", 1.0)
        field.setNumber.assert_any_call(12, "DistMax", 20.0)
        field.setAsBackgroundMesh.assert_called_once_with(12)

    @patch("gmsh_surf_mesh._curve_length", side_effect=[92.265, 5.0])
    @patch("gmsh_surf_mesh.gmsh.model.mesh.setTransfiniteCurve")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        return_value=[(1, 407), (1, 416)],
    )
    def test_sets_tangential_constraints_for_unconstrained_refinement_curves(
        self, _get_entities, set_transfinite_curve, _curve_length
    ) -> None:
        first_refinement = self._refinement()
        first_refinement["curves"] = [407]
        first_refinement["size_min_tangent"] = 1.0
        second_refinement = self._refinement()
        second_refinement["curves"] = [416]
        second_refinement["size_min_tangent"] = 2.0
        constraints = {}

        gmsh_surf_mesh.apply_anisotropic_curve_tangential_constraints(
            {
                "anisotropic_curve_refinements": [
                    first_refinement,
                    second_refinement,
                ]
            },
            constraints,
        )

        self.assertEqual(
            set_transfinite_curve.call_args_list,
            [
                ((407, 94, "Progression", 1.0),),
                ((416, 4, "Progression", 1.0),),
            ],
        )
        self.assertEqual(constraints[407].n_pts, 94)
        self.assertEqual(constraints[416].n_pts, 4)

    @patch("gmsh_surf_mesh._curve_length")
    @patch("gmsh_surf_mesh.gmsh.model.mesh.setTransfiniteCurve")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        return_value=[(1, 407)],
    )
    def test_preserves_existing_tangential_constraint(
        self, _get_entities, set_transfinite_curve, _curve_length
    ) -> None:
        existing_constraint = gmsh_surf_mesh.CurveConstraint(10, "Progression", 1.2)
        constraints = {407: existing_constraint}
        refinement = self._refinement()
        refinement["curves"] = [407]

        gmsh_surf_mesh.apply_anisotropic_curve_tangential_constraints(
            {"anisotropic_curve_refinement": refinement}, constraints
        )

        set_transfinite_curve.assert_not_called()
        _curve_length.assert_not_called()
        self.assertIs(constraints[407], existing_constraint)

    @patch("gmsh_surf_mesh.gmsh.model.mesh.field")
    @patch(
        "gmsh_surf_mesh.gmsh.model.getEntities",
        return_value=[(1, 96), (1, 407), (1, 416), (1, 420)],
    )
    def test_combines_multiple_anisotropic_background_fields(
        self, _get_entities, field
    ) -> None:
        second_refinement = self._refinement()
        second_refinement["curves"] = [96]
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
            [(('AttractorAnisoCurve',),), (('AttractorAnisoCurve',),), (('MinAniso',),)],
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
        return_value=[(1, 407), (1, 416), (1, 420)],
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
        return_value=[(1, 407), (1, 416), (1, 420)],
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
        return_value=[(1, 407), (1, 416), (1, 420)],
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
        return_value=[(1, 407), (1, 416), (1, 420)],
    )
    def test_rejects_non_positive_field_size(self, _get_entities) -> None:
        refinement = self._refinement()
        refinement["size_min_normal"] = 0.0

        with self.assertRaisesRegex(ValueError, "size_min_normal"):
            gmsh_surf_mesh.apply_anisotropic_curve_refinement(
                {"anisotropic_curve_refinement": refinement}
            )


class Poc2ConfigurationTests(unittest.TestCase):
    def test_configures_anisotropic_tip_refinement_without_boundary_layers(self) -> None:
        case_dir = Path(__file__).resolve().parent.parent / "DUC"
        with (case_dir / "msh_def_POC2.json").open(encoding="utf-8") as file:
            mesh_def = json.load(file)["mesh definition"]

        self.assertEqual(mesh_def["surface_meshing_algorithm"], "frontal-delaunay")
        self.assertEqual(
            mesh_def["surface_meshing_algorithms"],
            [{"surfaces": [83, 89, 93, 90, 87, 92], "algorithm": "bamg"}],
        )
        self.assertNotIn("max_num_threads_2d", mesh_def)
        self.assertNotIn("anisotropic_curve_refinement", mesh_def)
        self.assertNotIn("boundary_layers", mesh_def)
        self.assertEqual(
            mesh_def["anisotropic_curve_refinements"],
            [
                {
                    "curves": [407, 416, 420],
                    "sampling": 1000,
                    "size_min_normal": 0.1,
                    "size_min_tangent": 1.0,
                    "size_max_normal": 0.3,
                    "size_max_tangent": 1.0,
                    "dist_min": 1.0,
                    "dist_max": 5.0,
                }
            ],
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
