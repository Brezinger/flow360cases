"""Regression tests for the three-case acoustic comparison."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from DUC.results_postprocessing import plot_aeroacoustics as acoustics


class ComparisonTests(unittest.TestCase):
    def test_q40lt_uses_its_own_physical_reference_and_rpm(self):
        source = Path("F:/WDIR/flow360/DUC/Q40LT/duc_case1_step4/total_acoustics_v3.csv")
        rpm, steps_per_revolution = acoustics._timing_for_file(source, None, None)
        density, velocity, temperature, grid_unit = acoustics._reference_values_for_file(
            source, None, None, None, None
        )
        self.assertEqual(rpm, 1002.0)
        self.assertEqual(steps_per_revolution, acoustics.DEFAULT_TIME_STEPS_PER_REVOLUTION)
        self.assertEqual((density, velocity, temperature, grid_unit),
                         (1.149, 146.8587100942248, 283.275, 1.0))

    def test_legend_labels_are_used_verbatim(self):
        source = Path("F:/WDIR/flow360/DUC/Q40LT/duc_case1_step4/total_acoustics_v3.csv")
        with patch.object(acoustics, "LEGEND_LABELS_BY_FILE", {source.resolve(): "flow360 WMLES Q40LT"}):
            self.assertEqual(acoustics._legend_label(source), "flow360 WMLES Q40LT")

    def test_poc2_results_file_contains_measured_values_and_signed_errors(self):
        dataset = SimpleNamespace(source_file=Path("POC2/step4/total_acoustics_v3.csv"))
        with tempfile.TemporaryDirectory() as directory:
            output = acoustics.write_poc2_test_results(
                dataset,
                {1: 99.0, 2: 57.0, 3: 56.0, 4: 63.0},
                {2: 55.87, 3: 57.61, 4: 61.68},
                Path(directory),
            )
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([row["observer"] for row in rows], ["2", "3", "4"])
        self.assertEqual([row["simulation_minus_experiment_dba"] for row in rows],
                         ["+1.13", "-1.61", "+1.32"])


if __name__ == "__main__":
    unittest.main()
