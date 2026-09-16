import unittest

from correlate_gmsh_entities import (
    GeometryEntities,
    _ordered_curve_endpoints_from_parametrization,
    _parse_args,
    correlate_curves_with_orientation,
    correlate_entities_sequentially,
    correlate_geo_text,
    shift_vector,
)


def _entities(
    points: dict[int, tuple[float, float, float]],
    curves: dict[int, tuple[int, int]],
    surfaces: dict[int, set[int]],
) -> GeometryEntities:
    return GeometryEntities(
        points=points,
        curves=curves,
        curve_lengths={curve_id: 1.0 for curve_id in curves},
        surfaces=surfaces,
        surface_points={
            surface_id: {point_id for curve_id in curve_ids for point_id in curves[curve_id]}
            for surface_id, curve_ids in surfaces.items()
        },
    )


class SequentialCorrelationTests(unittest.TestCase):
    def test_parametrization_bounds_determine_curve_endpoint_order(self) -> None:
        endpoints = _ordered_curve_endpoints_from_parametrization(
            10,
            {
                1: (0.0, 0.0, 0.0),
                2: (0.0, 10.0, 0.0),
            },
            start_coordinate=(0.0, 10.0, 0.0),
            end_coordinate=(0.0, 0.0, 0.0),
        )
        self.assertEqual(endpoints, (2, 1))

    def test_parametrization_bounds_reject_unmatched_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match a boundary endpoint"):
            _ordered_curve_endpoints_from_parametrization(
                10,
                {
                    1: (0.0, 0.0, 0.0),
                    2: (0.0, 10.0, 0.0),
                },
                start_coordinate=(1.0, 20.0, 0.0),
                end_coordinate=(0.0, 0.0, 0.0),
            )

    def test_later_shift_overwrites_and_unmatched_entities_are_preserved(self) -> None:
        old = _entities(
            {1: (0.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 3.0, 0.0), 4: (0.0, 4.0, 0.0)},
            {10: (1, 2), 11: (3, 4)},
            {100: {10}, 101: {11}},
        )
        new = _entities(
            {
                101: (0.0, 10.0, 0.0), 102: (0.0, 11.0, 0.0),
                103: (0.0, 13.0, 0.0), 104: (0.0, 14.0, 0.0),
                201: (0.0, 20.0, 0.0), 202: (0.0, 21.0, 0.0),
            },
            {110: (102, 101), 111: (103, 104), 210: (201, 202)},
            {300: {110}, 301: {111}, 400: {210}},
        )

        point_map, curve_map, surface_map, curve_reversed, pass_results = correlate_entities_sequentially(
            old,
            new,
            tolerance=1.0e-9,
            surface_point_match_ratio=0.8,
            translations=[(0.0, 10.0, 0.0), (0.0, 20.0, 0.0)],
        )

        self.assertEqual(point_map, {1: 201, 2: 202, 3: 103, 4: 104})
        self.assertEqual(curve_map, {10: 210, 11: 111})
        self.assertEqual(curve_reversed, {10: False, 11: False})
        self.assertEqual(surface_map, {100: 400, 101: 301})
        self.assertEqual(pass_results[1].point_overwrites, 2)
        self.assertEqual(pass_results[1].curve_overwrites, 1)
        self.assertEqual(pass_results[1].surface_overwrites, 1)

    def test_final_unshifted_pass_overwrites_shifted_mapping(self) -> None:
        old = _entities(
            {1: (0.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0)},
            {10: (1, 2)},
            {100: {10}},
        )
        new = _entities(
            {
                101: (0.0, 10.0, 0.0), 102: (0.0, 11.0, 0.0),
                201: (0.0, 0.0, 0.0), 202: (0.0, 1.0, 0.0),
            },
            {110: (101, 102), 210: (202, 201)},
            {300: {110}, 400: {210}},
        )

        point_map, curve_map, surface_map, curve_reversed, pass_results = correlate_entities_sequentially(
            old,
            new,
            tolerance=1.0e-9,
            surface_point_match_ratio=0.8,
            translations=[(0.0, 10.0, 0.0)],
        )

        self.assertEqual(point_map, {1: 201, 2: 202})
        self.assertEqual(curve_map, {10: 210})
        self.assertEqual(curve_reversed, {10: True})
        self.assertEqual(surface_map, {100: 400})
        self.assertEqual(pass_results[-1].translation, (0.0, 0.0, 0.0))
        self.assertEqual(pass_results[-1].point_overwrites, 2)
        self.assertEqual(pass_results[-1].curve_overwrites, 1)
        self.assertEqual(pass_results[-1].surface_overwrites, 1)

    def test_shift_vector_uses_yz_plane_and_degrees(self) -> None:
        self.assertEqual(shift_vector(12.5, 0.0), (0.0, 12.5, 0.0))
        vector = shift_vector(12.5, 90.0)
        self.assertAlmostEqual(vector[0], 0.0)
        self.assertAlmostEqual(vector[1], 0.0)
        self.assertAlmostEqual(vector[2], 12.5)

    def test_parser_accepts_a_shift_for_each_moved_component(self) -> None:
        args = _parse_args(
            [
                "old.json",
                "new.json",
                "--entity-shift",
                "12.5",
                "35",
                "--entity-shift",
                "7",
                "-20",
            ]
        )
        self.assertEqual(args.entity_shift, [[12.5, 35.0], [7.0, -20.0]])

    def test_curve_correlation_detects_reversed_endpoint_order(self) -> None:
        curve_map, curve_reversed = correlate_curves_with_orientation(
            {10: (1, 2)},
            {20: (102, 101)},
            {1: 101, 2: 102},
        )
        self.assertEqual(curve_map, {10: 20})
        self.assertEqual(curve_reversed, {10: True})

    def test_geo_progression_is_inverted_for_a_reversed_curve(self) -> None:
        statements, skipped_count = correlate_geo_text(
            "Transfinite Curve {10} = 42 Using Progression 1.25;",
            point_map={},
            curve_map={10: 20},
            surface_map={},
            curve_reversed={10: True},
        )
        self.assertEqual(skipped_count, 0)
        self.assertEqual(
            statements,
            ["Transfinite Curve {20} = 42 Using Progression 0.8"],
        )

    def test_geo_mixed_curve_directions_are_split_and_bumps_are_unchanged(self) -> None:
        statements, skipped_count = correlate_geo_text(
            """
            Transfinite Curve {10, 11} = 42 Using Progression 1.25;
            Transfinite Curve {10} = 42 Using Bump 0.3;
            """,
            point_map={},
            curve_map={10: 20, 11: 21},
            surface_map={},
            curve_reversed={10: True, 11: False},
        )
        self.assertEqual(skipped_count, 0)
        self.assertEqual(
            statements,
            [
                "Transfinite Curve {20} = 42 Using Progression 0.8",
                "Transfinite Curve {21} = 42 Using Progression 1.25",
                "Transfinite Curve {20} = 42 Using Bump 0.3",
            ],
        )


if __name__ == "__main__":
    unittest.main()
