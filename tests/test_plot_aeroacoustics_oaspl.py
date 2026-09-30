"""Tests for reporting simulated OASPL against measured observers."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from DUC.results_postprocessing import plot_aeroacoustics as acoustics


class OasplDifferenceOutputTests(unittest.TestCase):
    def test_signed_differences_for_each_case_and_matching_observer(self):
        measured = {2: 70.0, 3: 80.0, 4: 90.0}
        output = io.StringIO()
        with redirect_stdout(output):
            acoustics.print_oaspl_differences(
                SimpleNamespace(source_file=Path("step3_total_acoustics_v3.csv")),
                {1: 99.0, 2: 72.0, 3: 77.5, 5: 88.0},
                measured,
                (1, 2, 3, 4, 5),
            )
            acoustics.print_oaspl_differences(
                SimpleNamespace(source_file=Path("step4_total_acoustics_v3.csv")),
                {2: 69.0, 4: 91.25},
                measured,
                (2, 4),
            )

        text = output.getvalue()
        self.assertIn("step3 (simulation - measurement)", text)
        self.assertIn("Observer 2: simulation 72.00 dBA, measurement 70.00 dBA, difference +2.00 dBA", text)
        self.assertIn("Observer 3: simulation 77.50 dBA, measurement 80.00 dBA, difference -2.50 dBA", text)
        self.assertIn("step4 (simulation - measurement)", text)
        self.assertIn("Observer 4: simulation 91.25 dBA, measurement 90.00 dBA, difference +1.25 dBA", text)
        self.assertNotIn("Observer 1:", text)
        self.assertNotIn("Observer 5:", text)

    def test_no_comparable_measurements_produces_no_difference_section(self):
        output = io.StringIO()
        with redirect_stdout(output):
            acoustics.print_oaspl_differences(
                SimpleNamespace(source_file=Path("step4_total_acoustics_v3.csv")),
                {1: 66.0},
                {2: 70.0},
                (1, 2),
            )
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
