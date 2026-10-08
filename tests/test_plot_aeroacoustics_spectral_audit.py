"""Independent signal and bandwidth checks for the acoustic mathematical audit."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from DUC.results_postprocessing import plot_aeroacoustics as a


class SpectralAuditTests(unittest.TestCase):
    fs = 32000.0

    def tone(self, frequency, count=32000, rms=1.0, phase=0.37):
        return np.sqrt(2.0) * rms * np.cos(2 * np.pi * frequency * np.arange(count) / self.fs + phase)

    def test_zero_and_negative_power_have_distinct_semantics(self):
        self.assertEqual(a._db_from_power(0.0), -np.inf)
        self.assertTrue(np.isnan(a._db_from_power(np.nan)))
        with self.assertRaises(ValueError):
            a._db_from_power(np.array([1.0, -1.0]))

    def test_iec_exact_centres_and_display_interval(self):
        nominal = np.array([50, 63, 100, 160, 1000, 10000])
        np.testing.assert_allclose(a.exact_third_octave_centers(nominal),
                                   1000 * 10.0 ** np.array([-1.3, -1.2, -1, -.8, 0, 1]))
        centers, lower, upper = a._third_octave_bands(40, 10000)
        self.assertEqual(centers.size, 24)
        self.assertGreaterEqual(centers[0], 40)
        self.assertEqual(centers[-1], 10000)
        np.testing.assert_allclose(upper[:-1], lower[1:])
        self.assertAlmostEqual(upper[-1], 11220.184543019635)

    def test_nonstandard_nominal_labels_are_not_silently_mapped(self):
        with self.assertRaisesRegex(ValueError, 'preferred nominal'):
            a.exact_third_octave_centers(np.array([50, 51]))
        exact = 1000 * 10.0 ** np.array([-1.3, -1.2, -1, -.8, 0, 1])
        np.testing.assert_allclose(a.exact_third_octave_centers(exact), exact)

    def test_valid_edge_zeros_do_not_change_selected_time_window(self):
        from scipy.signal import get_window

        pressure = np.r_[np.zeros(100), np.random.default_rng(83).normal(size=800), np.zeros(100)]
        frequencies, powers = a._welch_mean_square_spectrum(pressure, self.fs)
        self.assertEqual(len(frequencies), len(pressure) // 2 + 1)
        self.assertAlmostEqual(frequencies[1] - frequencies[0], self.fs / len(pressure))
        window = get_window('hann', len(pressure))
        expected = np.sum(((pressure - pressure.mean()) * window) ** 2) / np.sum(window ** 2)
        self.assertAlmostEqual(powers.sum(), expected, places=12)

    def test_top_band_contains_power_above_display_cutoff(self):
        pressure = self.tone(10500)
        centers, levels = a.third_octave_spectrum_db(pressure, self.fs, False)
        self.assertAlmostEqual(levels[-1], a._db_from_power(1.0), places=8)
        self.assertLess(a.oaspl_unweighted_db(pressure, self.fs), 0)
        np.testing.assert_allclose(levels, a.third_octave_spectrum_at_centers_db(
            pressure, self.fs, centers, False))

    def test_iec_edge_assignment_and_a_weighting(self):
        # 177.2 Hz lies in nominal 160 Hz IEC band, above the historical binary edge.
        # Use df=.1 Hz so the Hann main lobe also remains wholly inside the band.
        pressure = self.tone(177.2, count=320000)
        levels = a.third_octave_spectrum_at_centers_db(pressure, self.fs, [160], True,
                                                       centers_are_nominal=True)
        expected = a._db_from_power(1.0) + a._a_weighting_db(np.array([177.2]))[0]
        self.assertAlmostEqual(levels[0], expected, places=4)

    def test_unresolved_band_is_unavailable(self):
        pressure = self.tone(1000, count=1758)  # df=18.2 Hz: no bin in nominal 63 Hz band.
        with self.assertWarnsRegex(RuntimeWarning, 'no FFT bins'):
            levels = a.third_octave_spectrum_at_centers_db(pressure, self.fs, [63], False,
                                                           centers_are_nominal=True)
        self.assertTrue(np.isnan(levels[0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaisesRegex(ValueError, 'no FFT bins'):
                a.measurement_matched_oaspl_db(pressure, self.fs, [63])

    def test_frequency_support_is_not_silently_truncated(self):
        pressure = np.ones(32000)
        with self.assertRaisesRegex(ValueError, 'Nyquist'):
            a.oaspl_unweighted_db(pressure, 8000)
        with self.assertRaisesRegex(ValueError, 'Nyquist'):
            a.third_octave_spectrum_at_centers_db(pressure, 20000, [10000])
        with self.assertRaisesRegex(ValueError, 'Nyquist'):
            a.tone_spl_db(pressure, self.fs, 20000)

    def test_empty_narrowband_interval_is_rejected(self):
        for evaluator in (a.oaspl_unweighted_db, a.spl_spectrum_db, a.psd_spectrum_db):
            with self.subTest(evaluator=evaluator.__name__), self.assertRaisesRegex(ValueError, 'No frequencies'):
                evaluator(np.ones(8), 100000)

    def test_display_frequency_range_is_independent_of_literal_oaspl_interval(self):
        pressure = self.tone(13000)
        with patch.object(a, "SPECTRUM_MIN_FREQUENCY_HZ", 12000), patch.object(a, "SPECTRUM_MAX_FREQUENCY_HZ", 14000):
            for evaluator in (a.spl_spectrum_db, a.psd_spectrum_db):
                frequencies, _ = evaluator(pressure, self.fs, False)
                self.assertTrue(np.all((frequencies >= 12000) & (frequencies <= 14000)))
            centers, levels = a.third_octave_spectrum_db(pressure, self.fs, False)
            self.assertEqual(len(centers), 1)
            self.assertAlmostEqual(levels[0], a._db_from_power(1), places=8)
        self.assertLess(a.oaspl_unweighted_db(pressure, self.fs), -100)

    def test_measurement_matched_overall_uses_full_test_bandwidth(self):
        pressure = self.tone(10500, rms=.8) + self.tone(1000, rms=.6)
        nominal = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                            800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])
        self.assertAlmostEqual(a.measurement_matched_oaspl_db(pressure, self.fs, nominal, False),
                               a._db_from_power(1.0), places=8)
        self.assertAlmostEqual(a.oaspl_unweighted_db(pressure, self.fs),
                               a._db_from_power(.36), places=8)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            a.measurement_matched_oaspl_db(pressure, self.fs, [63, 63])

    def test_off_bin_known_frequency_fit_recovers_amplitude_and_dc(self):
        target = 1000.37
        actual_frequency, level = a.tone_spl_db(self.tone(target, count=7013) + 5.0, self.fs, target)
        self.assertEqual(actual_frequency, target)
        self.assertAlmostEqual(level, a._db_from_power(1.0), places=10)

    def test_joint_fit_does_not_assign_ninth_tone_to_tenth_harmonic(self):
        pressure = self.tone(900, count=9001) + self.tone(1000, count=9001, rms=.01)
        frequencies, levels = a.harmonic_tone_spl_db(pressure, self.fs, 100)
        self.assertEqual(frequencies[-1], 1000)
        self.assertAlmostEqual(levels[8], a._db_from_power(1), places=10)
        self.assertAlmostEqual(levels[9], a._db_from_power(.0001), places=8)
        # The legitimate, overlapping band neighbourhood contains both tones.
        self.assertGreater(a.harmonic_third_octave_spl_db(pressure, self.fs, 1000), levels[9] + 30)

    def test_density_and_amplitude_are_related_by_hann_noise_bandwidth(self):
        pressure = np.random.default_rng(817).normal(size=32000)
        with patch.object(a, 'NPERSEG', 4096):
            frequencies, amplitude = a.spl_spectrum_db(pressure, self.fs, False)
            density_frequencies, density = a.psd_spectrum_db(pressure, self.fs, False)
        np.testing.assert_array_equal(frequencies, density_frequencies)
        np.testing.assert_allclose(amplitude - density, 10 * np.log10(1.5 * self.fs / 4096), atol=1e-12)

    def test_invalid_inputs_and_segment_configuration_are_rejected(self):
        for pressure in [np.array([1, 2, np.nan] * 10), np.ones((4, 4)), np.ones(7)]:
            with self.subTest(shape=pressure.shape), self.assertRaises(ValueError):
                a._welch_mean_square_spectrum(pressure, self.fs)
        for fs in [0, -2, np.nan, np.inf]:
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                a._welch_mean_square_spectrum(np.ones(100), fs)
        for setting in [1, 0, -1, 7.5, True, 'invalid']:
            with self.subTest(setting=setting), patch.object(a, 'NPERSEG', setting), self.assertRaises(ValueError):
                a._welch_mean_square_spectrum(np.ones(100), self.fs)


if __name__ == '__main__':
    unittest.main()
