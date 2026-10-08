"""Observer geometry and independent Q40LT group selection."""

import json
import runpy
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from flow360_schema.models.entities.surface_entities import Surface


class PusherObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[1] / "DUC/Pusher/DUC_EVE_Pusher_noise.py"
        cls.module = runpy.run_path(str(script), run_name="validation")

    def test_arc_radius_angles_and_centerline_plane(self):
        cfg = self.module["CONFIG"]
        observers = self.module["_make_centerline_arc_observers"](cfg)
        self.assertEqual(len(observers), 5)
        axis = np.asarray(cfg.rotation_axis)
        axis = axis / np.linalg.norm(axis)
        for angle_deg, observer in zip((20, 45, 90, 135, 160), observers):
            offset = observer.position.to("m").value - np.asarray(cfg.rotation_center)
            self.assertAlmostEqual(np.linalg.norm(offset), 304.8)
            self.assertAlmostEqual(offset[1], 0)
            self.assertLess(offset[2], 0)
            measured_angle = np.degrees(np.arccos(np.dot(offset, axis) / 304.8))
            self.assertAlmostEqual(measured_angle, angle_deg)
            self.assertEqual(observer.group_name, "centerline_arc")

    def test_q40lt_groups_toggle_independently_and_arc_is_always_added(self):
        for group1, group2 in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(group1=group1, group2=group2):
                cfg = replace(self.module["CONFIG"],
                              include_aeroacoustic_observer_group_1=group1,
                              include_aeroacoustic_observer_group_2=group2)
                output = self.module["_make_aeroacoustic_output"](cfg)
                self.assertEqual(output.patch_type, "solid")
                self.assertIsNone(output.permeable_surfaces)
                expected = {"centerline_arc": 5}
                if group1:
                    expected["1"] = 5
                if group2:
                    expected["2"] = 120
                self.assertEqual(Counter(o.group_name for o in output.observers), expected)

    def test_legacy_ungrouped_observers_respect_group1_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            source_file = Path(directory) / "legacy.json"
            source_file.write_text(json.dumps({"aeroacousticOutput": {
                "observers": [[1, 2, 3]], "writePerSurfaceOutput": True,
            }}), encoding="utf-8")
            for include1, count in ((True, 6), (False, 5)):
                cfg = replace(self.module["CONFIG"], include_aeroacoustic_observer_group_1=include1)
                output = self.module["_make_aeroacoustic_output"](
                    cfg, source_file
                )
                self.assertEqual(len(output.observers), count)

    def test_solid_output_needs_no_sliding_interface_mesh(self):
        cfg = replace(self.module["CONFIG"], step=self.module["STEP_SETTINGS"][3])
        outputs = self.module["_make_outputs"]([Surface(name="blade1")], cfg)
        acoustic = [output for output in outputs if output.output_type == "AeroAcousticOutput"]
        self.assertEqual(len(acoustic), 1)
        self.assertEqual(acoustic[0].patch_type, "solid")
        self.assertIsNone(acoustic[0].permeable_surfaces)


if __name__ == "__main__":
    unittest.main()
