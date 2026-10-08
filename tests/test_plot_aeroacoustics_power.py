"""Physical power normalization, independent of tone-amplitude normalization."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import get_window, welch

from DUC.results_postprocessing import plot_aeroacoustics as a


class SpectralPowerTests(unittest.TestCase):
    fs = 32000.0

    def signal(self, frequency=1000.0):
        return np.sqrt(2) * np.cos(2 * np.pi * frequency * np.arange(32000) / self.fs)

    def test_known_one_pascal_rms_tone_and_a_weighting(self):
        pressure = self.signal()
        expected = 20 * np.log10(1 / a.ACOUSTIC_PRESSURE_REFERENCE_PA)
        self.assertAlmostEqual(a.oaspl_unweighted_db(pressure, self.fs), expected, places=8)
        self.assertAlmostEqual(a.oaspl_a_weighted_db(pressure, self.fs), expected, places=4)
        self.assertAlmostEqual(a.tone_spl_db(pressure, self.fs, 1000)[1], expected, places=8)
        self.assertAlmostEqual(a.harmonic_third_octave_spl_db(pressure, self.fs, 1000), expected, places=8)
        centers, levels = a.third_octave_spectrum_db(pressure, self.fs, False)
        self.assertAlmostEqual(levels[np.argmin(abs(centers - 1000))], expected, places=8)

    def test_off_bin_tone_power_is_recovered(self):
        pressure = self.signal(1000.37)
        _, powers = a._welch_mean_square_spectrum(pressure, self.fs)
        self.assertAlmostEqual(float(powers.sum()), 1, places=5)

    def test_broadband_power_matches_window_normalized_time_record(self):
        pressure = np.random.default_rng(42).normal(size=32000)
        fluctuating = pressure - pressure.mean()
        window = get_window('hann', len(pressure))
        expected = np.sum((window * fluctuating)**2) / np.sum(window**2)
        _, powers = a._welch_mean_square_spectrum(pressure, self.fs)
        self.assertAlmostEqual(float(powers.sum()), expected, places=12)

    def test_legacy_shift_and_narrowband_amplitudes(self):
        pressure = self.signal() + np.random.default_rng(8).normal(size=32000)
        for segment in ('All', 4096):
            with patch.object(a, 'NPERSEG', segment):
                frequencies, corrected = a._welch_mean_square_spectrum(pressure, self.fs)
                _, legacy = welch(pressure - pressure.mean(), fs=self.fs, window='hann',
                                  nperseg=a._nperseg(len(pressure)),
                                  noverlap=a._nperseg(len(pressure)) // 2, scaling='spectrum')
                np.testing.assert_allclose(legacy, 1.5 * corrected, rtol=1e-12)
                mask = (frequencies >= 40) & (frequencies <= 10000)
                self.assertAlmostEqual(10*np.log10(legacy[mask].sum()/corrected[mask].sum()),
                                       1.7609125905568124, places=12)
                _, narrow = a.spl_spectrum_db(pressure, self.fs, False)
                np.testing.assert_allclose(narrow, a._db_from_power(legacy[mask]))

    def test_spectrum_display_includes_high_tone_peaks(self):
        fig, ax = a.plt.subplots()
        try:
            ax.plot([100, 1000, 10000], [20, 80, 30])
            a._set_spectrum_limits(ax)
            self.assertGreater(ax.get_ylim()[1], 80)
        finally:
            a.plt.close(fig)


if __name__ == '__main__':
    unittest.main()
