"""Read-only mathematical audit of the current acoustic postprocessor.

Run from any directory using the project's configured interpreter. Only files
beside this script are written; production code and plot sets are unchanged.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy
from scipy.signal import get_window, spectrogram, welch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import plot_aeroacoustics as a
import correct_lifter4b_acoustics as calibration


def source(study, step):
    return next((ROOT / study / "aeroacoustics" / step).glob("*total_acoustics_v3.csv"))


PUSHER = dict(rpm=2186., time_steps_per_revolution=1756.,
              reference_density_kg_m3=1.1839124828745407,
              reference_velocity_m_s=242.65242777307085,
              reference_temperature_k=298.15, l_grid_unit_m=.001)


def load(path, pusher=False):
    if pusher:
        return a.load_acoustic_dataset(path, **PUSHER)
    rpm, steps = a._timing_for_file(path, None, None)
    rho, velocity, temp, grid = a._reference_values_for_file(path, None, None, None, None)
    return a.load_acoustic_dataset(path, rpm=rpm, time_steps_per_revolution=steps,
                                  reference_density_kg_m3=rho, reference_velocity_m_s=velocity,
                                  reference_temperature_k=temp, l_grid_unit_m=grid)


def db(power):
    return a._db_from_power(power)


def band_levels(f, power, centers, factor):
    return np.array([db(power[(f >= fc / factor) & (f < fc * factor)].sum()) for fc in centers])


def check_dataset(dataset):
    fs = dataset.sample_rate_hz
    dt = np.diff(dataset.time_seconds)
    centers, lower, upper = a._third_octave_bands(40, 10000)
    base10_centers = 1000 * 10. ** (np.arange(-13, 11) / 10.)
    observers = []
    for obs in dataset.pressure_by_observer:
        raw = dataset.pressure_by_observer[obs] * dataset.pressure_scale_pa
        nonzero = np.flatnonzero(raw != 0)
        p = a._filter_zero_signal_edges(raw)
        x = p - p.mean()
        f, power = a._welch_mean_square_spectrum(raw, fs)
        w = get_window(a.WINDOW, len(p))
        time_power = np.sum((w*x)**2) / np.sum(w*w)
        np.testing.assert_allclose(power.sum(), time_power, rtol=1e-12)
        wa = 10 ** (a._a_weighting_db(f) / 10)
        band = (f >= 40) & (f <= 10000)
        full = band_levels(f, power, centers, 2 ** (1/6))
        full_a = band_levels(f, power*wa, centers, 2 ** (1/6))
        _, plotted = a.third_octave_spectrum_db(raw, fs, False)
        _, plotted_a = a.third_octave_spectrum_db(raw, fs, True)
        base10 = band_levels(f, power, base10_centers, 10 ** .05)
        base10_a = band_levels(f, power*wa, base10_centers, 10 ** .05)
        bpf = 4 * dataset.rpm / 60
        # A unit-RMS synthetic tone with this actual sample rate and duration
        # isolates single-bin scalloping from CFD amplitude/nonstationarity.
        synthetic = np.sqrt(2) * np.cos(2*np.pi*bpf*np.arange(len(p))/fs + .4)
        tone_f, tone_level = a.tone_spl_db(synthetic, fs, bpf)
        _, box_power = welch(x, fs=fs, window="boxcar", nperseg=len(x), scaling="density")
        box_power *= f[1] - f[0]
        residual = raw - (dataset.thickness_by_observer[obs] + dataset.loading_by_observer[obs]) * dataset.pressure_scale_pa
        covered = np.zeros_like(f, dtype=bool)
        for lo, hi in zip(lower, upper):
            covered |= (f >= lo) & (f < hi) & band
        observers.append(dict(
            observer=obs, retained_samples=len(p), leading_zeros=int(nonzero[0]),
            trailing_zeros=len(raw)-int(nonzero[-1])-1, duration_s=len(p)/fs,
            retained_revolutions=len(p)/fs*dataset.rpm/60, df_hz=float(f[1]-f[0]),
            bins_first_third_band=int(np.count_nonzero((f >= lower[0]) & (f < upper[0]))),
            empty_plotted_band_centers_hz=[float(fc) for fc,lo,hi in zip(centers,lower,upper)
                if not np.any(band & (f >= lo) & (f < hi))],
            total_component_relative_max_error=float(np.max(abs(residual))/max(np.max(abs(raw)), 1e-300)),
            parseval_relative_error=float(abs(power.sum()-time_power)/time_power),
            top_band_plotted_db=float(plotted[-1]), top_band_full_db=float(full[-1]),
            top_band_plotted_dba=float(plotted_a[-1]), top_band_full_dba=float(full_a[-1]),
            full_minus_plotted_top_db=float(full[-1]-plotted[-1]),
            full_minus_plotted_top_dba=float(full_a[-1]-plotted_a[-1]),
            full_50_to_10000_centers_db=float(10*np.log10(np.sum(10**(full/10)))),
            full_50_to_10000_centers_dba=float(10*np.log10(np.sum(10**(full_a/10)))),
            script_oaspl_db=float(db(power[band].sum())), script_oaspl_dba=float(db((power*wa)[band].sum())),
            displayed_band_sum_db=float(10*np.log10(np.sum(10**(plotted/10)))),
            uncovered_40_44_power_fraction=float(power[band & ~covered].sum()/power[band].sum()),
            base10_minus_binary_full_db=(base10-full).tolist(),
            base10_minus_binary_full_dba=(base10_a-full_a).tolist(),
            synthetic_bpf_nearest_frequency_hz=tone_f,
            synthetic_bpf_single_bin_error_db=tone_level-float(db(1.)),
            hann_minus_rectangular_band_db=float(db(power[band].sum())-db(box_power[band].sum())),
            hann_minus_rectangular_band_dba=float(db((power*wa)[band].sum())-db((box_power*wa)[band].sum())),
        ))
    return dict(source=str(dataset.source_file), source_files=[str(x) for x in dataset.source_files],
                rpm=dataset.rpm, sample_rate_hz=fs, nyquist_hz=fs/2,
                timestep_relative_max_deviation=float(np.max(abs(dt-dt.mean()))/dt.mean()),
                timestamps_increasing=bool(np.all(dt > 0)), all_pressures_finite=bool(np.all([
                    np.all(np.isfinite(p)) for p in dataset.pressure_by_observer.values()])),
                spectrogram_df_hz=fs/512, spectrogram_window_s=512/fs,
                bpf_hz=dataset.rpm/60*4, observers=observers)


def synthetic_checks():
    fs, n = 32000., 32000
    t = np.arange(n)/fs
    pressure = np.sqrt(2)*np.cos(2*np.pi*1000*t)
    expected = float(db(1.))
    physical = dict(expected_unit_rms_level_db=expected,
                    oaspl_db=a.oaspl_unweighted_db(pressure, fs),
                    oaspl_dba=a.oaspl_a_weighted_db(pressure, fs))
    np.testing.assert_allclose([physical['oaspl_db'], physical['oaspl_dba']], expected, atol=1e-8)
    checks = dict(unit_rms_1khz=physical)
    checks['hann_half_bin_tone_error_db'] = a.tone_spl_db(
        np.sqrt(2)*np.cos(2*np.pi*1000.5*t), fs, 1000.5)[1]-expected
    # Same stationary white-noise record, different analysis bandwidth.
    noise = np.random.default_rng(293).normal(size=n)
    spectra = []
    for segment in (1024, 8192, "All"):
        with patch.object(a, 'NPERSEG', segment):
            f, amp_db = a.spl_spectrum_db(noise, fs, False)
            ff, power = a._welch_mean_square_spectrum(noise, fs)
            mask = (f >= 2000) & (f <= 8000)
            # mean linear power, converted to dB, avoids log-estimator bias
            spectra.append(dict(segment=segment, df_hz=float(f[1]-f[0]),
                                broadband_amplitude_level_db=float(10*np.log10(np.mean(10**(amp_db[mask]/10)))),
                                broadband_psd_db=float(db(np.mean(power[(ff >= 2000)&(ff <= 8000)]/(ff[1]-ff[0])))),
                                oaspl_db=a.oaspl_unweighted_db(noise, fs)))
    checks['white_noise_segment_comparison'] = spectra
    normalization_checks = []
    for length in (8, 9, 257, 1024):
        values = np.random.default_rng(length).normal(size=length)
        for window_name in ('hann', 'hamming', 'boxcar', 'flattop'):
            for segment in ('All', max(4, length//3)):
                with patch.object(a, 'WINDOW', window_name), patch.object(a, 'NPERSEG', segment):
                    f, power = a._welch_mean_square_spectrum(values, fs)
                segment_length = length if segment == 'All' else segment
                stride = segment_length-segment_length//2
                w = get_window(window_name, segment_length)
                expected_powers = []
                for start in range(0, length-segment_length+1, stride):
                    cut = values[start:start+segment_length]
                    cut = cut-cut.mean()
                    expected_powers.append(np.sum((w*cut)**2)/np.sum(w*w))
                expected_power = np.mean(expected_powers)
                np.testing.assert_allclose(power.sum(), expected_power, rtol=1e-12)
                normalization_checks.append(dict(length=length, window=window_name, segment=segment,
                    relative_error=float(abs(power.sum()-expected_power)/expected_power)))
    checks['normalization_cases'] = normalization_checks
    aw_f = np.array([31.5,63,125,250,500,1000,2000,4000,8000,10000])
    checks['a_weighting_db'] = dict(frequencies_hz=aw_f.tolist(), levels_db=a._a_weighting_db(aw_f).tolist())
    # Strong 9th harmonic lies inside the nominal 10th harmonic's 1/3 band.
    bpf = 100.
    neighbor = np.sqrt(2)*np.cos(2*np.pi*(9*bpf)*t)
    checks['harmonic_10_with_only_harmonic_9_present_db'] = a.harmonic_third_octave_spl_db(neighbor, fs, 10*bpf)
    checks['harmonic_10_band_hz'] = [10*bpf/2**(1/6), 10*bpf*2**(1/6)]
    checks['adjacent_harmonic_band_overlap_pairs'] = [[h,h+1] for h in range(1,10)
        if h*2**(1/6) > (h+1)/2**(1/6)]
    # A Fourier bin can have power assigned into a band that doesn't contain
    # the physical tone. Here the physical tone is above the 62.5 Hz band.
    edge_fs, edge_n, edge_f = 12000., 600, 71.
    edge = np.sqrt(2)*np.cos(2*np.pi*edge_f*np.arange(edge_n)/edge_fs)
    centers, edge_levels = a.third_octave_spectrum_db(edge, edge_fs, False)
    checks['unresolved_band_edge_tone'] = dict(tone_hz=edge_f, df_hz=edge_fs/edge_n,
        nominal_63_upper_hz=62.5*2**(1/6), nominal_63_level_db=float(edge_levels[np.argmin(abs(centers-62.5))]),
        true_unit_rms_level_db=expected)
    checks['empty_band_counts'] = int(np.count_nonzero(edge_levels < -1000))
    checks['zero_power_level_db'] = db(0.)
    checks['negative_power_level_db'] = db(-1.)
    checks['out_of_nyquist_target'] = dict(requested_hz=20000., returned_hz=a.tone_spl_db(pressure, fs, 20000)[0])
    # Short spectrogram window relative to the 66.7 Hz POC2 BPF.
    sg_fs, sg_n, sg_f = 133332.92442101915, 22968, 4*1000/60
    sg_tone = np.sqrt(2)*np.cos(2*np.pi*sg_f*np.arange(sg_n)/sg_fs)
    f, times, sg = spectrogram(sg_tone, fs=sg_fs, window='hann', nperseg=512,
                              noverlap=256, scaling='density', mode='psd')
    sg_power = sg.sum(axis=0)*(f[1]-f[0])
    checks['poc2_bpf_short_spectrogram'] = dict(df_hz=float(f[1]-f[0]), first_positive_bin_hz=float(f[1]),
        mean_power_relative_to_unit_rms_db=float(10*np.log10(np.mean(sg_power))), window_s=512/sg_fs)
    # Uniform sampling is not validated by the single-file path.
    d = a.AcousticDataset(Path('synthetic.csv'), np.arange(16.), {1:np.arange(1.,17.)},
        {1:np.arange(1.,17.)*.25}, {1:np.arange(1.,17.)*.75}, 60., 100., 1.2, 10., 298.15, 1.)
    irregular = replace(d, time=d.time.copy())
    irregular.time[8] += 2.5
    try:
        level = a.oaspl_unweighted_db(irregular.pressure_by_observer[1], irregular.sample_rate_hz)
        checks['nonmonotonic_time_accepted'] = dict(accepted=True, reported_fs_hz=irregular.sample_rate_hz, level_db=level)
    except ValueError as error:
        checks['nonmonotonic_time_accepted'] = dict(accepted=False, error=str(error))
    with patch.object(a, 'OASPL_MAX_FREQUENCY_HZ', 10000.):
        low_fs_tone = np.sqrt(2)*np.cos(2*np.pi*1000*np.arange(8000)/8000)
        checks['nyquist_below_oaspl_upper_accepted'] = dict(
            fs_hz=8000., upper_band_hz=10000., level_db=a.oaspl_unweighted_db(low_fs_tone, 8000.))
    # Same --revolutions, different preprocessing path => different origin.
    x = np.sqrt(2)*np.cos(2*np.pi*100*np.arange(7000)/1000 + .3)
    values = np.r_[np.zeros(700), x, np.zeros(300)]
    demo = replace(d, time=np.arange(len(values))*.001*d.speed_of_sound_m_s,
        pressure_by_observer={1:values}, thickness_by_observer={1:values*.25},
        loading_by_observer={1:values*.75}, rpm=60.)
    rr = a.RevolutionRange(0, 2, 'zero:0-2')
    sample_count = a._samples_per_revolution(demo.time, demo.rpm, demo.reference_temperature_k, demo.l_grid_unit_m)
    ordinary = a._slice_revolutions(values, rr, sample_count, demo.source_file)
    with contextlib.redirect_stdout(io.StringIO()):
        prepared = a.prepare_acoustic_dataset(demo, 0, 'acoustic', rr)
    checks['revolution_origin_mismatch'] = dict(
        ordinary_retained_nonpadding_samples=len(a._filter_zero_signal_edges(ordinary)),
        prepared_retained_nonpadding_samples=len(a._filter_zero_signal_edges(prepared.pressure_by_observer[1])))
    class CaptureComplete(Exception):
        pass
    captured = []
    def capture_load(path, observers, **kwargs):
        captured.append(dict(source=str(path), **kwargs))
        return demo
    def capture_combine(datasets):
        raise CaptureComplete()
    with patch.object(a, 'load_acoustic_dataset', capture_load), patch.object(a, 'combine_acoustic_datasets', capture_combine):
        try:
            a.make_plots(csv_files=[ROOT/'POC2/step4/first.csv', ROOT/'Q40LT/step4/second.csv'])
        except CaptureComplete:
            pass
    checks['combined_cli_inferred_references'] = captured
    return checks


def main():
    before = hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
    data = {f'{study}_{step}': load(source(study,step))
            for study in ('POC2','Lifter4B') for step in ('step3','step4')}
    data['Pusher_step3'] = load(source('Pusher','step3'), True)
    data['Pusher_step3B_only'] = load(source('Pusher','step3B'), True)
    with contextlib.redirect_stdout(io.StringIO()):
        data['Pusher_combined_skip150'] = a.prepare_acoustic_dataset(a.combine_acoustic_datasets(
            [data['Pusher_step3'], data['Pusher_step3B_only']]), 150, 'cfd')
    results = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                   audited_script_sha256=before, synthetic=synthetic_checks(),
                   datasets={name:check_dataset(d) for name,d in data.items()})
    push_default = load(source('Pusher','step3'))
    intended = data['Pusher_step3']
    p = push_default.pressure_by_observer[3]*push_default.pressure_scale_pa
    pg = intended.pressure_by_observer[3]*intended.pressure_scale_pa
    results['pusher_defaults'] = dict(default_rpm=push_default.rpm, correct_rpm=intended.rpm,
        default_steps=push_default.time_steps_per_revolution, correct_steps=intended.time_steps_per_revolution,
        default_rho=push_default.reference_density_kg_m3, correct_rho=intended.reference_density_kg_m3,
        default_temperature_k=push_default.reference_temperature_k, correct_temperature_k=intended.reference_temperature_k,
        pressure_scale_error_db=float(20*np.log10(push_default.pressure_scale_pa/intended.pressure_scale_pa)),
        sample_rate_ratio=push_default.sample_rate_hz/intended.sample_rate_hz,
        observer3_oaspl_default_minus_correct_db=a.oaspl_unweighted_db(p,push_default.sample_rate_hz)-a.oaspl_unweighted_db(pg,intended.sample_rate_hz),
        observer3_oaspl_default_minus_correct_dba=a.oaspl_a_weighted_db(p,push_default.sample_rate_hz)-a.oaspl_a_weighted_db(pg,intended.sample_rate_hz))
    # Actual surface workflow omits the total dataset's initial cut.
    lifter = data['Lifter4B_step4']
    with contextlib.redirect_stdout(io.StringIO()):
        cut = a.prepare_acoustic_dataset(lifter, 150, 'acoustic')
        surface_uncut = a.load_surface_resolved_datasets([(cut,{})], groups=('blade1',), observers=(3,))[0].dataset
        surface_correct_cut = a.prepare_acoustic_dataset(surface_uncut, 150, 'acoustic')
    pu = surface_uncut.pressure_by_observer[3]*surface_uncut.pressure_scale_pa
    pc = surface_correct_cut.pressure_by_observer[3]*surface_correct_cut.pressure_scale_pa
    results['surface_skip_mismatch'] = dict(
        uncut_samples=len(a._filter_zero_signal_edges(pu)), cut_samples=len(a._filter_zero_signal_edges(pc)),
        uncut_minus_cut_oaspl_db=a.oaspl_unweighted_db(pu,surface_uncut.sample_rate_hz)-a.oaspl_unweighted_db(pc,surface_correct_cut.sample_rate_hz))
    # Workbook header and exact energy-sum checks using the existing source.
    measured = a.load_poc2x_1000rpm_measurement_data(calibration.DEFAULT_MEASUREMENT_EXCEL)
    results['measurement'] = {
        str(obs): dict(frequencies_hz=m.frequencies_hz.tolist(), reported_oaspl_dba=measured.oaspl_dba_by_observer[obs],
                      band_energy_sum_dba=calibration.energy_sum_db(m.spl_db))
        for obs,m in measured.spectra_by_observer.items()}
    maxima = []
    for name, checked in results['datasets'].items():
        if name == 'Pusher_step3B_only':
            continue
        d = data[name]
        for row in checked['observers']:
            differences = np.asarray(row['base10_minus_binary_full_dba'])
            index = int(np.argmax(abs(differences)))
            fc2 = 1000*2**((index-13)/3)
            fc10 = 1000*10**((index-13)/10)
            f,p = a._welch_mean_square_spectrum(d.pressure_by_observer[row['observer']]*d.pressure_scale_pa,d.sample_rate_hz)
            mask2 = (f >= fc2/2**(1/6)) & (f < fc2*2**(1/6))
            mask10 = (f >= fc10/10**.05) & (f < fc10*10**.05)
            maxima.append(dict(dataset=name, observer=row['observer'], binary_center_hz=fc2,
                base10_center_hz=fc10, base10_minus_binary_dba=float(differences[index]),
                binary_bins_hz=f[mask2].tolist(), base10_bins_hz=f[mask10].tolist()))
    results['largest_band_convention_changes'] = sorted(maxima,key=lambda r:abs(r['base10_minus_binary_dba']),reverse=True)
    # Provenance and coverage of the actual Pusher continuation sum. The
    # source-window semantics still require the solver metadata, not just CSV.
    continuation = []
    for obs in a.DEFAULT_OBSERVERS:
        d1, d2 = data['Pusher_step3'],data['Pusher_step3B_only']
        i1 = np.flatnonzero(d1.pressure_by_observer[obs])
        i2 = np.flatnonzero(d2.pressure_by_observer[obs])
        start1,end1 = d1.time_seconds[i1[0]],d1.time_seconds[i1[-1]]
        start2,end2 = d2.time_seconds[i2[0]],d2.time_seconds[i2[-1]]
        continuation.append(dict(observer=obs, step3_window_s=[float(start1),float(end1)],
            step3B_window_s=[float(start2),float(end2)], arrival_overlap_s=float(min(end1,end2)-max(start1,start2))))
    results['pusher_continuation_windows'] = continuation
    assert before == hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
    (HERE/'audit_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(dict(synthetic=results['synthetic'], pusher_defaults=results['pusher_defaults'],
                         surface_skip_mismatch=results['surface_skip_mismatch']), indent=2))
    for name,d in results['datasets'].items():
        rows = d['observers']
        print(name, 'fs',round(d['sample_rate_hz'],3),
              'full-minus-plotted-top', [round(r['full_minus_plotted_top_dba'],3) for r in rows],
              'BPF tone scalloping',[round(r['synthetic_bpf_single_bin_error_db'],3) for r in rows])


if __name__ == '__main__':
    main()
