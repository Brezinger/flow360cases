import unittest

from correlate_gmsh_entities import (
    _unidentified_reference_paths,
    correlate_points,
    correlate_points_with_match_modes,
)


class CorrelatePointsTests(unittest.TestCase):
    def test_direct_matching_is_unchanged(self) -> None:
        old_points = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)}
        new_points = {10: (0.0, 0.0, 0.0), 20: (1.0, 0.0, 0.0)}

        self.assertEqual(correlate_points(old_points, new_points, 1.0e-6), {1: 10, 2: 20})

    def test_direct_and_multiple_translated_matches_coexist(self) -> None:
        old_points = {
            1: (0.0, 0.0, 0.0),
            2: (10.0, 0.0, 0.0),
            3: (20.0, 0.0, 0.0),
        }
        new_points = {
            10: (0.0, 0.0, 0.0),
            20: (10.0, -4.0, 0.0),
            30: (20.0, -4.0, 2.0),
        }
        first_translation = (0.0, -4.0, 0.0)
        second_translation = (0.0, -4.0, 2.0)

        point_map, match_modes = correlate_points_with_match_modes(
            old_points,
            new_points,
            1.0e-6,
            [first_translation, second_translation],
        )

        self.assertEqual(point_map, {1: 10, 2: 20, 3: 30})
        self.assertIsNone(match_modes[1])
        self.assertEqual(match_modes[2], first_translation)
        self.assertEqual(match_modes[3], second_translation)

    def test_unidentified_references_are_reported_with_paths(self) -> None:
        mesh_definition = {
            "mesh definition": {"start_pt": "369_unidentified", "curve_ids": [10]}
        }

        self.assertEqual(
            _unidentified_reference_paths(mesh_definition),
            ["$.mesh definition.start_pt"],
        )


if __name__ == "__main__":
    unittest.main()
