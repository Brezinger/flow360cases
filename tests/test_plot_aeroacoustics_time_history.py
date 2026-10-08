"""Pressure traces retain timing and interior zeros when edge padding is removed."""

import unittest
from types import SimpleNamespace

import numpy as np

from DUC.results_postprocessing import plot_aeroacoustics as acoustics


class PressureTimeHistoryTests(unittest.TestCase):
    def test_clipping_keeps_matching_timestamps_and_interior_zero(self):
        dataset = SimpleNamespace(
            time_seconds=np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5]),
            pressure_by_observer={1: np.array([0.0, 2.0, 0.0, -3.0, 0.0, 0.0])},
            pressure_scale_pa=4.0,
        )
        time, pressure = acoustics._pressure_time_history(dataset, 1)
        np.testing.assert_allclose(time, [0.0, 0.1, 0.2])
        np.testing.assert_array_equal(pressure, [8.0, 0.0, -12.0])

    def test_all_zero_trace_retains_full_record(self):
        dataset = SimpleNamespace(
            time_seconds=np.array([2.0, 2.01, 2.02]),
            pressure_by_observer={1: np.zeros(3)},
            pressure_scale_pa=4.0,
        )
        time, pressure = acoustics._pressure_time_history(dataset, 1)
        np.testing.assert_allclose(time, [0.0, 0.01, 0.02])
        np.testing.assert_array_equal(pressure, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
