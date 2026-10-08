"""Continuation alignment and observer-specific transient exclusion."""

import contextlib
import io
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from DUC.results_postprocessing import plot_aeroacoustics as a


class CombinedAcousticTests(unittest.TestCase):
    def dataset(self, start=0, values=None, spacing=.02):
        values = np.asarray(values if values is not None else np.arange(1, 21), dtype=float)
        speed = a._speed_of_sound_m_s(298.15)
        return a.AcousticDataset(
            source_file=Path(f"part{start}.csv"),
            time=(start + np.arange(len(values))) * spacing * speed,
            pressure_by_observer={1: values, 2: values.copy()},
            thickness_by_observer={1: values * .25, 2: values * .25},
            loading_by_observer={1: values * .75, 2: values * .75},
            rpm=60, time_steps_per_revolution=100,
            reference_density_kg_m3=1.2, reference_velocity_m_s=10,
            reference_temperature_k=298.15, l_grid_unit_m=1,
        )

    def test_continuation_preserves_grid_and_components(self):
        first, second = self.dataset(), self.dataset(start=20)
        merged = a.combine_acoustic_datasets([second, first])
        np.testing.assert_allclose(merged.time_seconds, np.arange(40) * .02)
        np.testing.assert_array_equal(merged.pressure_by_observer[1], np.tile(np.arange(1, 21), 2))
        np.testing.assert_allclose(merged.pressure_by_observer[1],
                                   merged.thickness_by_observer[1] + merged.loading_by_observer[1])

    def test_overlapping_contributions_are_summed(self):
        merged = a.combine_acoustic_datasets([self.dataset(), self.dataset()])
        np.testing.assert_array_equal(merged.pressure_by_observer[1], np.arange(1, 21) * 2)

    def test_incompatible_sampling_and_references_are_rejected(self):
        for incompatible in (self.dataset(spacing=.03), replace(self.dataset(), rpm=61)):
            with self.assertRaises(ValueError):
                a.combine_acoustic_datasets([self.dataset(), incompatible])

    def test_unaligned_timestamps_and_signal_gaps_are_rejected(self):
        for second in (self.dataset(start=20.5), self.dataset(start=21)):
            with self.assertRaises(ValueError):
                a.combine_acoustic_datasets([self.dataset(), second])

    def test_cfd_cut_starts_at_each_observers_signal(self):
        d = self.dataset(values=[0, 0, *range(1, 21), 0, 0])
        for component in ("pressure_by_observer", "thickness_by_observer", "loading_by_observer"):
            getattr(d, component)[2] = np.roll(getattr(d, component)[2], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            cut = a.prepare_acoustic_dataset(d, 4, "cfd")
        self.assertEqual(np.flatnonzero(cut.pressure_by_observer[1])[0], 4)
        self.assertEqual(np.flatnonzero(cut.pressure_by_observer[2])[0], 5)
        self.assertEqual(np.flatnonzero(d.pressure_by_observer[1])[0], 2)
        np.testing.assert_allclose(cut.pressure_by_observer[1],
                                   cut.thickness_by_observer[1] + cut.loading_by_observer[1])

    def test_acoustic_cut_preserves_internal_zero(self):
        d = self.dataset()
        d.pressure_by_observer[1][10] = 0
        with contextlib.redirect_stdout(io.StringIO()):
            cut = a.prepare_acoustic_dataset(d, 3, "acoustic")
        self.assertEqual(np.flatnonzero(cut.pressure_by_observer[1])[0], 3)
        self.assertEqual(cut.pressure_by_observer[1][10], 0)
        self.assertEqual(len(cut.time), len(d.time))

    def test_excessive_cut_is_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            a.prepare_acoustic_dataset(self.dataset(), 30, "acoustic")


if __name__ == "__main__":
    unittest.main()
