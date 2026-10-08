"""Signed force axes, reference lengths, and force-file discovery."""

import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ForcePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "DUC/results_postprocessing/plot_forces_timesteps.py"
        cls.module = runpy.run_path(str(path), run_name="validation")
        cls.globals = cls.module["configure_case"].__globals__

    def setUp(self):
        self.module["configure_case"]("Pusher")

    def test_signed_axes_and_final_pseudo_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step3_total_forces_v2.csv"
            path.write_text("physical_step,CFx,CMx,CFz,CMz\n1,-2,3,99,98\n1,-4,5,97,96\n")
            actual = self.module["_read_final_coefficients_by_physical_step"](path)
            self.assertEqual(actual, {1: (4.0, -5.0)})
            self.globals["THRUST_COMPONENT"] = "+cfz"
            self.globals["TORQUE_COMPONENT"] = "cmz"
            self.assertEqual(self.module["_read_final_coefficients_by_physical_step"](path), {1: (97.0, 96.0)})

    def test_moment_length_follows_selected_axis(self):
        self.globals["MOMENT_LENGTHS_M"] = (1.0, 2.0, 3.0)
        cfg = self.module["RunConfig"](2186, 1756)
        _, x_torque = self.module["_propeller_coefficients"]([1.0], [1.0], cfg)
        self.globals["TORQUE_COMPONENT"] = "CMy"
        _, y_torque = self.module["_propeller_coefficients"]([1.0], [1.0], cfg)
        self.assertAlmostEqual(y_torque[0], 2 * x_torque[0])

    def test_flat_and_nested_filenames(self):
        label = self.module["_run_label"]
        self.assertEqual(label(Path("step3_total_forces_v2.csv")), "step3")
        self.assertEqual(label(Path("step3_case-123_results_total_forces_v2.csv")), "step3")
        self.assertEqual(label(Path("step3/case-123_total_forces_v2.csv")), "step3")

    def test_poc2_keeps_original_axes_and_conversion(self):
        self.module["configure_case"]("POC2")
        self.assertEqual(self.globals["THRUST_COMPONENT"], "CFz")
        self.assertEqual(self.globals["TORQUE_COMPONENT"], "CMz")
        ct, cq = self.module["_propeller_coefficients"]([1.0], [1.0], self.module["RunConfig"](1000, 120))
        self.assertAlmostEqual(ct[0], np.pi**3 / 8)
        self.assertAlmostEqual(cq[0], np.pi**3 / 16)


if __name__ == "__main__":
    unittest.main()
