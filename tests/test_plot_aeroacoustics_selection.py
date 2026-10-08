"""Physical metadata and consistent cuts on observer and surface time grids."""
import contextlib
import csv
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from DUC.results_postprocessing import plot_aeroacoustics as a


class SelectionTests(unittest.TestCase):
    def dataset(self, values=None, source="POC2/step3/total.csv", origin=0):
        values = np.asarray(values if values is not None else [0]*7 + list(range(1, 301)) + [0]*3, float)
        return a.AcousticDataset(
            Path(source), (origin + np.arange(len(values))) * .01 * a._speed_of_sound_m_s(298.15),
            {1: values}, {1: values*.25}, {1: values*.75},
            60, 200, 1.2, 10, 298.15, 1,
        )

    def test_pusher_metadata_matches_setup_and_step3b_continuation(self):
        for step, count in a.PUSHER_STEPS_PER_REVOLUTION.items():
            path = Path(f"Pusher/aeroacoustics/{step}/total.csv")
            self.assertEqual(a._timing_for_file(path, None, None), (2186, count))
            rho, velocity, temperature, length = a._reference_values_for_file(path, None, None, None, None)
            self.assertAlmostEqual(rho, 1.1839124829)
            self.assertAlmostEqual(velocity, 2*np.pi*1.06*2186/60)
            self.assertEqual((temperature, length), (298.15, .001))
        self.assertEqual([a._observer_angle_deg(path, o) for o in range(1, 6)], [20, 45, 90, 135, 160])

    def test_explicit_reference_overrides_are_retained(self):
        path = Path("Pusher/step3/total.csv")
        self.assertEqual(a._timing_for_file(path, 2000, 4000), (2000, 4000))
        self.assertEqual(a._reference_values_for_file(path, 1, 2, 300, 1), (1, 2, 300, 1))

    def test_revolution_cut_origin_is_signal_start(self):
        d = self.dataset()
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = a.prepare_acoustic_dataset(d, revolution_range=a.RevolutionRange(0, 2, "0-2"))
        self.assertEqual(prepared.selection_windows_by_observer[1], (7, 207))
        self.assertEqual(len(a._pressure_time_history(prepared, 1)[0]), 200)

    def test_surface_uses_total_physical_window_instead_of_its_own_signal_origin(self):
        d = self.dataset()
        surface = self.dataset([0]*20 + list(range(1, 291)), origin=0)
        with contextlib.redirect_stdout(io.StringIO()):
            total = a.prepare_acoustic_dataset(d, 10, "acoustic", a.RevolutionRange(0, 2, "0-2"))
        cut = a._apply_total_selection(surface, total)
        self.assertEqual(cut.selection_windows_by_observer[1], (17, 217))
        self.assertEqual(len(a._pressure_time_history(cut, 1)[0]), 200)
        np.testing.assert_array_equal(cut.pressure_by_observer[1][17:217], surface.pressure_by_observer[1][17:217])

    def test_validation_rejects_nonfinite_irregular_and_nonincreasing_inputs(self):
        d = self.dataset()
        for time in (np.full(len(d.time), np.nan), d.time[::-1], np.concatenate(([0], d.time[1:]+10))):
            with self.assertRaises(ValueError):
                replace(d, time=time)
        for refs in ({"rpm": 0}, {"reference_density_kg_m3": -1}, {"reference_temperature_k": np.nan}):
            with self.assertRaises(ValueError):
                replace(d, **refs)
        with self.assertRaises(ValueError):
            replace(d, loading_by_observer={1: np.full(len(d.time), np.inf)})

    def test_independent_inference_rejects_incompatible_combination(self):
        first = self.dataset()
        calls = []
        def load(source, observers, **kwargs):
            calls.append(kwargs)
            return replace(first, source_file=Path(source), **kwargs)
        with patch.object(a, "load_acoustic_dataset", side_effect=load), self.assertRaisesRegex(ValueError, "matching RPM"):
            a.make_plots(csv_files=[Path("POC2/step3/a.csv"), Path("Q40LT/step3/b.csv")], observers=(1,))
        self.assertEqual(calls[0]["rpm"], 1000)
        self.assertEqual(calls[1]["rpm"], 1002)
        self.assertEqual(calls[1]["l_grid_unit_m"], 1)

    def test_loader_and_prepare_use_identical_revolution_origin(self):
        d = self.dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"total.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time", "observer_0_pressure", "observer_0_thickness", "observer_0_loading"])
                writer.writerows(zip(d.time, d.pressure_by_observer[1], d.thickness_by_observer[1], d.loading_by_observer[1]))
            refs = dict(rpm=60, time_steps_per_revolution=200, reference_density_kg_m3=1.2,
                        reference_velocity_m_s=10, reference_temperature_k=298.15, l_grid_unit_m=1)
            rev = a.RevolutionRange(0, 2, "0-2")
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = a.load_acoustic_dataset(path, (1,), revolution_range=rev, **refs)
                prepared = a.prepare_acoustic_dataset(a.load_acoustic_dataset(path, (1,), **refs), revolution_range=rev)
            np.testing.assert_array_equal(loaded.pressure_by_observer[1], prepared.pressure_by_observer[1])
            self.assertEqual(loaded.selection_windows_by_observer, prepared.selection_windows_by_observer)


if __name__ == "__main__":
    unittest.main()
