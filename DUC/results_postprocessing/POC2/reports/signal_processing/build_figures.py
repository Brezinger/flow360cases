"""Reproduce report figures from the real POC2 step4 pressure CSV."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import get_window, welch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE/'source_snapshot' if (HERE/'source_snapshot/plot_aeroacoustics.py').exists() else ROOT))
import plot_aeroacoustics as a

SOURCE = ROOT / 'POC2/aeroacoustics/step4/case-fb40c043-c99a-4cce-95cf-de9630dba98d_total_acoustics_v3.csv'
if not SOURCE.exists():
    SOURCE = HERE / 'input/POC2/aeroacoustics/step4' / SOURCE.name
BLUE, RED, GREY = '#006B91', '#B64732', '#646464'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#292929', 'text.color': '#292929',
                     'axes.titleweight': 'bold', 'figure.constrained_layout.use': True,
                     'savefig.dpi': 200, 'axes.grid': True, 'grid.alpha': .22})
(HERE / 'figures').mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(HERE / f'figures/{name}.pdf')
    fig.savefig(HERE / f'figures/{name}.png')
    plt.close(fig)


def db(values):
    return a._db_from_power(values)


def main():
    rpm, steps = a._timing_for_file(SOURCE, None, None)
    rho, velocity, temperature, grid = a._reference_values_for_file(SOURCE, None, None, None, None)
    data = a.load_acoustic_dataset(SOURCE, rpm=rpm, time_steps_per_revolution=steps,
                                  reference_density_kg_m3=rho, reference_velocity_m_s=velocity,
                                  reference_temperature_k=temperature, l_grid_unit_m=grid)
    observer = 3
    raw = data.pressure_by_observer[observer] * data.pressure_scale_pa
    active = np.flatnonzero(raw != 0)
    first, last = int(active[0]), int(active[-1])
    p = raw[first:last+1]
    t_abs = data.time_seconds
    t = t_abs[first:last+1] - t_abs[first]
    x = p - p.mean()
    fs = data.sample_rate_hz
    n = len(x)
    window = get_window('hann', n)
    f, density = welch(x, fs=fs, window='hann', nperseg=n, noverlap=n//2,
                       scaling='density')
    _, spectrum = welch(x, fs=fs, window='hann', nperseg=n, noverlap=n//2,
                        scaling='spectrum')
    delta_f = f[1] - f[0]
    bin_power = density * delta_f
    enbw = fs * np.sum(window**2) / np.sum(window)**2
    band = (f >= 40) & (f <= 10000)
    linear_a = 10**(a._a_weighting_db(f) / 10)
    np.testing.assert_allclose(spectrum, bin_power * 1.5, rtol=1e-9, atol=1e-20)
    np.testing.assert_allclose(bin_power.sum(), np.sum((window*x)**2)/np.sum(window**2), rtol=1e-12)
    centers, lower, upper = a._third_octave_bands(40, 10000)
    third = np.array([bin_power[band & (f >= lo) & (f < hi)].sum() for lo, hi in zip(lower, upper)])
    third_a = np.array([(bin_power*linear_a)[band & (f >= lo) & (f < hi)].sum()
                        for lo, hi in zip(lower, upper)])
    covered = np.zeros_like(f, dtype=bool)
    for lo, hi in zip(lower, upper):
        covered |= band & (f >= lo) & (f < hi)
    rows = []
    for obs in a.DEFAULT_OBSERVERS:
        pp = a._filter_zero_signal_edges(data.pressure_by_observer[obs]*data.pressure_scale_pa)
        ff, corrected = a._welch_mean_square_spectrum(pp, fs)
        mask = (ff >= 40) & (ff <= 10000)
        weight = 10**(a._a_weighting_db(ff)/10)
        good, good_a = db(corrected[mask].sum()), db((corrected*weight)[mask].sum())
        _, original = a._welch_spectrum(pp, fs, scaling='spectrum')
        rows.append(dict(observer=obs, csv_index=obs-1, angle_deg=a.OBSERVER_ANGLES_DEG[obs],
                         corrected_db=good, erroneous_db=db(original[mask].sum()),
                         corrected_dba=good_a, erroneous_dba=db((original*weight)[mask].sum())))
    stats = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 plotting_script_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
                 python_version=platform.python_version(), scipy_version=scipy.__version__,
                 matplotlib_version=matplotlib.__version__,
                 numpy_version=np.__version__, selected_observer=observer, csv_column='observer_2_pressure',
                 density_kg_m3=rho, temperature_k=temperature, speed_of_sound_m_s=data.speed_of_sound_m_s,
                 grid_unit_m=grid, pressure_scale_pa=data.pressure_scale_pa, rpm=rpm,
                 solver_steps_per_rev=steps, raw_samples=len(raw), signal_samples=n,
                 leading_padding_samples=first, trailing_padding_samples=len(raw)-last-1,
                 signal_first_csv_row_zero_based=first, signal_last_csv_row_zero_based=last,
                 absolute_signal_start_s=float(t_abs[first]), absolute_signal_end_s=float(t_abs[last]),
                 signal_elapsed_span_s=float(t[-1]), dft_record_duration_s=n/fs,
                 dt_s=1/fs, sample_rate_hz=fs, nyquist_hz=fs/2, frequency_spacing_hz=float(delta_f),
                 sample_interval_relative_max_deviation=float(np.max(abs(np.diff(t_abs)-1/fs))/(1/fs)),
                 enbw_hz=float(enbw), enbw_bins=float(enbw/delta_f), pressure_mean_pa=float(p.mean()),
                 raw_fluctuation_rms_pa=float(np.sqrt(np.mean(x*x))),
                 window_normalized_rms_pa=float(np.sqrt(bin_power.sum())),
                 full_band_window_power_pa2=float(bin_power.sum()),
                 raw_fluctuation_power_pa2=float(np.mean(x*x)),
                 integrated_40_10000_pa2=float(bin_power[band].sum()),
                 integrated_40_10000_db=float(db(bin_power[band].sum())),
                 integrated_40_10000_dba=float(db((bin_power*linear_a)[band].sum())),
                 erroneous_40_10000_db=float(db(spectrum[band].sum())),
                 erroneous_40_10000_dba=float(db((spectrum*linear_a)[band].sum())),
                 third_octave_sum_db=float(db(third.sum())), third_octave_sum_dba=float(db(third_a.sum())),
                 first_center_hz=float(centers[0]), last_center_hz=float(centers[-1]),
                 first_lower_edge_hz=float(lower[0]), last_upper_edge_hz=float(upper[-1]),
                 uncovered_low_band_power_pa2=float(bin_power[band & ~covered].sum()),
                 blade_passing_frequency_hz=4*rpm/60, observer_levels=rows)
    (HERE / 'analysis_results.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    with (HERE / 'observer_oaspl.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (HERE / 'third_octave_observer3.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['center_hz','lower_hz','upper_hz','corrected_pa2','corrected_db','erroneous_db',
                         'corrected_a_weighted_pa2','corrected_dba','erroneous_dba'])
        writer.writerows(zip(centers, lower, upper, third, db(third), db(1.5*third),
                             third_a, db(third_a), db(1.5*third_a)))
    # Export the full retained real signal and spectral arrays for reproducibility.
    np.savez_compressed(HERE/'poc2_observer3_processing.npz', time_s=t, pressure_pa=p,
                        fluctuation_pa=x, hann=window, frequency_hz=f,
                        psd_pa2_hz=density, amplitude_spectrum_pa2=spectrum, bin_power_pa2=bin_power)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 3.6))
    axes[0].plot(t_abs-t_abs[0], raw, color=BLUE, lw=.7)
    axes[0].axvspan(t_abs[first]-t_abs[0], t_abs[last]-t_abs[0], color=BLUE, alpha=.08)
    axes[0].set(title='Downloaded record: padded arrival-time interval', xlabel='CSV elapsed time [s]', ylabel='Pressure [Pa]')
    axes[1].plot(t, p, color=BLUE, lw=.7)
    axes[1].set(title='Retained observer signal', xlabel='Time since first retained sample [s]', ylabel='Pressure [Pa]')
    zoom = t <= .06
    axes[2].plot(t[zoom], x[zoom], color=BLUE, lw=.8)
    axes[2].set(title='Mean removed: first propeller revolution', xlabel='Elapsed time [s]', ylabel="p\N{PRIME} [Pa]")
    save(fig, '01_pressure_time')

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.2), sharex=True)
    axes[0].plot(t, x, color=GREY, lw=.5, label='Pressure fluctuation')
    axes[0].plot(t, x*window, color=BLUE, lw=.6, label='Hann-windowed fluctuation')
    axes[0].set(ylabel='Pressure [Pa]'); axes[0].legend(loc='upper right', fontsize=8)
    axes[1].plot(t, window, color=BLUE)
    axes[1].set(xlabel='Elapsed time [s]', ylabel='Hann weight', ylim=(-.03,1.08))
    save(fig, '02_hann_window')

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.7))
    for ax in axes:
        ax.plot(f[band], db(spectrum[band]), color=BLUE, lw=.65, label='Amplitude-normalized spectrum S')
        ax.plot(f[band], db(bin_power[band]), color=RED, lw=.65, alpha=.85, label='Additive bin power G x df')
        ax.set(ylabel='Level re (20 microPa)$^2$ [dB]', xlabel='Frequency [Hz]')
    axes[0].set(xscale='log', xlim=(40,10000), ylim=(-20,75), title='Same FFT, two normalizations')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[1].set(xlim=(50,85), title='Blade-passing-frequency region: 66.67 Hz', ylim=(0,75))
    save(fig, '03_narrowband')

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.0), sharex=True)
    for ax, powers, unit in zip(axes, (third,third_a), ('dB','dBA')):
        ax.plot(centers, db(powers*1.5), color=RED, ls='--', marker='o', ms=3, lw=1,
                label='Erroneous: summed amplitude spectrum')
        ax.plot(centers, db(powers), color=BLUE, marker='o', ms=3, lw=1,
                label='Correct: integrated PSD')
        ax.set(xscale='log', xlim=(44,11200), ylabel=f'Band level [{unit}]',
               title='Unweighted' if unit=='dB' else 'A-weighted')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[1].set_xlabel('Exact binary third-octave center frequency [Hz]')
    save(fig, '04_third_octave')

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=False)
    pos = np.arange(5)
    for ax, key, title in zip(axes, ('db','dba'), ('Unweighted OASPL [dB]','A-weighted OASPL [dBA]')):
        good = [row['corrected_'+key] for row in rows]
        bad = [row['erroneous_'+key] for row in rows]
        ax.bar(pos-.18, bad, .34, color=RED, label='Erroneous')
        ax.bar(pos+.18, good, .34, color=BLUE, label='Correct')
        ax.set(xticks=pos, xticklabels=[f"{row['observer']}\n{row['angle_deg']}\N{DEGREE SIGN}" for row in rows],
               ylabel=title, xlabel='Plot observer / angle', ylim=(min(good)-3,max(bad)+3))
    axes[0].legend(loc='upper left', fontsize=8)
    save(fig, '05_oaspl')
    # Supply numeric LaTeX macros and the table from the actual calculation.
    macros = {
        'RawSamples': str(len(raw)), 'SignalSamples': str(n), 'LeadPadding': str(first),
        'TailPadding': str(len(raw)-last-1), 'SignalSpan': f'{t[-1]:.6f}',
        'RecordDuration': f'{n/fs:.6f}', 'SampleRate': f'{fs:.3f}', 'SampleInterval': f'{1e6/fs:.6f}',
        'FrequencySpacing': f'{delta_f:.6f}', 'ENBW': f'{enbw:.6f}',
        'SpeedSound': f'{data.speed_of_sound_m_s:.6f}', 'PressureScale': f'{data.pressure_scale_pa:.3f}',
        'MeanPressure': f'{p.mean():.6f}', 'RawRMS': f'{np.sqrt(np.mean(x*x)):.6f}',
        'WindowRMS': f'{np.sqrt(bin_power.sum()):.6f}', 'PowerTime': f'{np.sum((window*x)**2)/np.sum(window**2):.8f}',
        'PowerFrequency': f'{bin_power.sum():.8f}', 'BandPower': f'{bin_power[band].sum():.8f}',
        'CorrectDB': f'{db(bin_power[band].sum()):.4f}', 'WrongDB': f'{db(spectrum[band].sum()):.4f}',
        'CorrectDBA': f'{db((bin_power*linear_a)[band].sum()):.4f}',
        'WrongDBA': f'{db((spectrum*linear_a)[band].sum()):.4f}',
        'ThirdSumDB': f'{db(third.sum()):.4f}', 'ThirdSumDBA': f'{db(third_a.sum()):.4f}',
        'FirstEdge': f'{lower[0]:.4f}', 'LastCenter': f'{centers[-1]:.4f}',
        'RawHash': stats['source_sha256'], 'ScriptHash': stats['plotting_script_sha256'],
        'ScipyVersion': scipy.__version__, 'SignalStart': f'{t_abs[first]:.6f}',
        'SignalStop': f'{t_abs[last]:.6f}',
    }
    tex = '\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n'
    (HERE/'numbers.tex').write_text(tex, encoding='utf-8')
    table = '\n'.join(f"{r['observer']} / {r['csv_index']} & {r['angle_deg']}$^\\circ$ & "
                      f"{r['erroneous_db']:.2f} & {r['corrected_db']:.2f} & "
                      f"{r['erroneous_dba']:.2f} & {r['corrected_dba']:.2f}\\\\\\hline" for r in rows)
    (HERE/'observer_table.tex').write_text('\\newcommand{\\ObserverRows}{%\n'+table+'%\n}\n', encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
