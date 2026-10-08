"""Physical resolution, density units and time axes for acoustic diagnostics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import get_window

from DUC.results_postprocessing import plot_aeroacoustics as a


def make_dataset(pressure, sample_rate_hz=32000.0, rpm=1000.0):
    pressure = np.asarray(pressure, dtype=float)
    temperature_k = 298.15
    speed_of_sound = a._speed_of_sound_m_s(temperature_k)
    grid_unit_m = 0.001
    time = np.arange(len(pressure)) / sample_rate_hz * speed_of_sound / grid_unit_m
    return a.AcousticDataset(
        source_file=Path("POC2/aeroacoustics/step3/test_total_acoustics_v3.csv"),
        time=time,
        pressure_by_observer={1: pressure},
        thickness_by_observer={1: pressure * 0.4},
        loading_by_observer={1: pressure * 0.6},
        rpm=rpm,
        time_steps_per_revolution=3840,
        reference_density_kg_m3=1.1839124829,
        reference_velocity_m_s=146.607657,
        reference_temperature_k=temperature_k,
        l_grid_unit_m=grid_unit_m,
    )


class SpectrogramResolutionTests(unittest.TestCase):
    fs = 32000.0

    def power_from_levels(self, frequencies, levels):
        density = a.ACOUSTIC_PRESSURE_REFERENCE_PA**2 * 10.0**(levels / 10.0)
        return np.sum(density, axis=0) * (frequencies[1] - frequencies[0])

    def test_one_pascal_rms_bpf_retains_power(self):
        # One 1000 RPM revolution contains four full BPF cycles. This window
        # resolves the BPF instead of suppressing it through local mean removal.
        duration_s = 60.0 / 1000.0
        pressure = np.sqrt(2) * np.cos(
            2 * np.pi * (4 * 1000.0 / 60.0) * np.arange(19200) / self.fs
        )
        frequencies, times, levels = a._spectrogram_spl_db(
            pressure, self.fs, window_duration_s=duration_s,
        )
        self.assertGreater(len(times), 1)
        self.assertAlmostEqual(frequencies[1] - frequencies[0], 1 / duration_s)
        np.testing.assert_allclose(self.power_from_levels(frequencies, levels), 1.0, atol=1e-12)

    def test_global_mean_removed_without_segment_mean_suppression(self):
        pressure = 3.0 + np.cos(2 * np.pi * 5.0 * np.arange(32000) / self.fs)
        duration_s = 0.05
        frequencies, times, levels = a._spectrogram_spl_db(
            pressure, self.fs, window_duration_s=duration_s,
        )
        nperseg = int(round(duration_s * self.fs))
        window = get_window(a.WINDOW, nperseg)
        fluctuating = pressure - np.mean(pressure)
        expected = []
        for center_time in times:
            start = int(round(center_time * self.fs - nperseg / 2))
            segment = fluctuating[start:start + nperseg]
            expected.append(np.sum((segment * window)**2) / np.sum(window**2))
        np.testing.assert_allclose(
            self.power_from_levels(frequencies, levels), expected, rtol=1e-12, atol=1e-12,
        )

    def test_physical_window_has_same_resolution_at_different_sample_rates(self):
        resolutions = []
        for sample_rate_hz in (32000.0, 64000.0):
            pressure = np.cos(2 * np.pi * 1000 * np.arange(int(sample_rate_hz)) / sample_rate_hz)
            frequencies, times, _ = a._spectrogram_spl_db(
                pressure, sample_rate_hz, window_duration_s=0.05,
            )
            resolutions.append(frequencies[1] - frequencies[0])
            self.assertAlmostEqual(times[0], 0.025)
        np.testing.assert_allclose(resolutions, [20.0, 20.0], rtol=1e-12)

    def test_known_tone_density_peak_has_density_units(self):
        pressure = np.sqrt(2) * np.cos(2 * np.pi * 1000 * np.arange(32000) / self.fs)
        frequencies, _, levels = a._spectrogram_spl_db(
            pressure, self.fs, window_duration_s=0.05,
        )
        index = int(np.argmin(abs(frequencies - 1000)))
        window = get_window(a.WINDOW, 1600)
        enbw_hz = self.fs * np.sum(window**2) / np.sum(window)**2
        expected_level = 10 * np.log10(1 / (enbw_hz * a.ACOUSTIC_PRESSURE_REFERENCE_PA**2))
        np.testing.assert_allclose(levels[index], expected_level, atol=1e-10)


class ResolutionRendererTests(unittest.TestCase):
    def test_psd_spectrum_density_is_separate_from_amplitude_normalization(self):
        pressure = np.sqrt(2) * np.cos(2 * np.pi * 1000 * np.arange(32000) / 32000.0)
        for segment in (3200, 32000):
            with patch.object(a, "NPERSEG", segment):
                frequencies, density_levels = a.psd_spectrum_db(pressure, 32000.0, False)
                _, amplitude_levels = a.spl_spectrum_db(pressure, 32000.0, False)
            index = int(np.argmin(abs(frequencies - 1000)))
            window = get_window(a.WINDOW, segment)
            enbw_hz = 32000.0 * np.sum(window**2) / np.sum(window)**2
            expected_amplitude = 10 * np.log10(1 / a.ACOUSTIC_PRESSURE_REFERENCE_PA**2)
            self.assertAlmostEqual(amplitude_levels[index], expected_amplitude, places=10)
            self.assertAlmostEqual(
                density_levels[index], expected_amplitude - 10 * np.log10(enbw_hz), places=10,
            )

    def test_psd_output_units_and_filenames_are_separate_from_amplitude(self):
        pressure = np.random.default_rng(42).normal(size=32000)
        dataset = make_dataset(pressure)
        captured = []

        def capture(fig, filename):
            captured.append((Path(filename).name, fig.axes[0].get_ylabel()))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(a, "_save_figure", side_effect=capture):
                output_files = a.render_psd_plots([(dataset, {})], (1,), Path(directory))
        self.assertEqual(len(output_files), 1)
        self.assertTrue(captured[0][0].startswith("psd_spectrum"))
        self.assertRegex(captured[0][1].lower(), "density|psd")
        self.assertIn("Hz", captured[0][1])

    def test_spectrogram_has_seconds_and_density_colorbar(self):
        pressure = np.cos(2 * np.pi * 1000 * np.arange(32000) / 32000.0)
        dataset = make_dataset(pressure)
        captured = []

        def capture(fig, filename):
            captured.append({
                "filename": Path(filename).name,
                "xlabel": fig.axes[0].get_xlabel(),
                "colorbar_label": fig.axes[-1].get_ylabel(),
                "xlim": fig.axes[0].get_xlim(),
            })

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(a, "_save_figure", side_effect=capture):
                a.render_spectrogram_plots([(dataset, {})], (1,), Path(directory))
        self.assertEqual(len(captured), 2)
        for figure in captured:
            self.assertRegex(figure["xlabel"].lower(), r"time.*\[s\]")
            self.assertRegex(figure["colorbar_label"].lower(), "density|psd")
            self.assertIn("Hz", figure["colorbar_label"])

    def test_single_frame_spectrogram_has_visible_time_width(self):
        # A physical window capped to a very short record yields one frame.
        # Center-only pcolormesh coordinates would collapse this to zero width.
        pressure = np.cos(2 * np.pi * 1000 * np.arange(64) / 32000.0)
        dataset = make_dataset(pressure)
        mesh_widths = []

        def capture(fig, filename):
            coordinates = fig.axes[0].collections[0].get_coordinates()
            mesh_widths.append(float(np.ptp(coordinates[:, :, 0])))

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(a, "_save_figure", side_effect=capture):
                a.render_spectrogram_plots([(dataset, {})], (1,), Path(directory))
        self.assertEqual(len(mesh_widths), 2)
        for width in mesh_widths:
            self.assertGreater(width, 0)

    def test_metadata_discloses_segment_bandwidth_and_two_time_scales(self):
        signal = np.cos(2 * np.pi * 1000 * np.arange(4096) / 32000.0)
        padded = np.concatenate((np.zeros(7), signal, np.zeros(12)))
        dataset = make_dataset(padded)
        with patch.object(a, "NPERSEG", 1024):
            metadata = a.spectral_analysis_metadata(dataset, 1)
        self.assertEqual(metadata["retained_samples"], len(signal))
        self.assertAlmostEqual(metadata["resolution_hz"], 31.25)
        self.assertAlmostEqual(metadata["equivalent_noise_bandwidth_hz"], 46.875)
        self.assertEqual(metadata["welch_segments"], 7)
        self.assertAlmostEqual(metadata["duration_s"], 4096 / 32000.0, delta=1 / 32000.0)
        self.assertAlmostEqual(metadata["cfd_samples_per_10khz_period"], 6.4)
        self.assertAlmostEqual(metadata["acoustic_samples_per_10khz_period"], 3.2)
        self.assertIn("band_resolution", metadata)


if __name__ == "__main__":
    unittest.main()
