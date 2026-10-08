"""Regenerate the local POC2, Lifter4B and Pusher acoustic plot suite."""
from __future__ import annotations

import contextlib
import csv
import functools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

try:
    from . import plot_aeroacoustics as a
    from . import correct_lifter4b_acoustics as calibration
except ImportError:  # Direct CLI execution.
    import plot_aeroacoustics as a
    import correct_lifter4b_acoustics as calibration

ROOT = Path(__file__).resolve().parent


def source(study: str, step: str) -> Path:
    files = list((ROOT / study / 'aeroacoustics' / step).glob('*total_acoustics_v3.csv'))
    if len(files) != 1:
        raise ValueError(f'Expected one total CSV for {study}/{step}: {files}')
    return files[0]


def load_source(csv_file: Path, observers: tuple[int, ...] = a.DEFAULT_OBSERVERS) -> a.AcousticDataset:
    """Resolve physical references independently from each study's source file."""
    rpm, steps = a._timing_for_file(csv_file, None, None)
    rho, velocity, temperature, grid = a._reference_values_for_file(csv_file, None, None, None, None)
    return a.load_acoustic_dataset(
        csv_file, observers, rpm=rpm, time_steps_per_revolution=steps,
        reference_density_kg_m3=rho, reference_velocity_m_s=velocity,
        reference_temperature_k=temperature, l_grid_unit_m=grid,
    )


def remove_obsolete_measurement_oaspl(output_dir: Path, sources: list[Path]) -> list[Path]:
    """Remove only known old overlays after their matched-band replacements exist."""
    output_dir = Path(output_dir)
    removed = []
    for csv_file in sources:
        prefix = a._case_label(Path(csv_file)) + '_'
        replacement = output_dir / 'a-weighted' / (
            prefix + 'oaspl_a_weighted_measurement_matched_observers_1_5.png'
        )
        if replacement.is_file():
            for old in (output_dir / (prefix + a.PLOTS[1].filename),
                        output_dir / 'a-weighted' / (prefix + a.PLOTS[1].filename)):
                if old.is_file():
                    old.unlink()
                    removed.append(old)
    comparison = Path(a.COMPARISON_OUTPUT_FILE)
    replacement = output_dir / (comparison.stem + '_measurement_matched.png')
    if replacement.is_file() and (output_dir / comparison).is_file():
        (output_dir / comparison).unlink()
        removed.append(output_dir / comparison)
    old_results = output_dir / 'poc2_test_results.csv'
    if (output_dir / 'poc2_test_results_measurement_matched.csv').is_file() and old_results.is_file():
        old_results.unlink()
        removed.append(old_results)
    return removed


def update_pusher_comparison() -> None:
    """Refresh the numeric step3 versus continuation comparison alongside plots."""
    original = load_source(source('Pusher', 'step3'))
    continuation = load_source(source('Pusher', 'step3B'))
    prepared = a.prepare_acoustic_dataset(a.combine_acoustic_datasets([original, continuation]), 150, 'cfd')
    original = a.prepare_acoustic_dataset(original)
    rows = []
    for observer in a.DEFAULT_OBSERVERS:
        row = {'observer': observer}
        for label, dataset in (('original', original), ('processed', prepared)):
            pressure = a._observer_pressure(dataset, observer)
            frequencies, powers = a._welch_mean_square_spectrum(pressure, dataset.sample_rate_hz)
            mask = (frequencies >= 2000) & (frequencies <= 10000)
            row[label + '_oaspl_db'] = a.oaspl_unweighted_db(pressure, dataset.sample_rate_hz)
            row[label + '_oaspl_dba'] = a.oaspl_a_weighted_db(pressure, dataset.sample_rate_hz)
            row[label + '_band_2_10_khz_db'] = a._db_from_power(np.sum(powers[mask]))
        for key in ('oaspl_db', 'oaspl_dba', 'band_2_10_khz_db'):
            row['change_' + key] = row['processed_' + key] - row['original_' + key]
        rows.append(row)
    output = ROOT / 'Pusher/aeroacoustic_plots/step3B'
    with (output / 'comparison_with_step3.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_file = output / 'analysis_manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    manifest['comparison_with_step3'] = rows
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def refresh_spectral_displays() -> None:
    """Refresh spectral displays and historic copies using the recorded suite inputs."""
    summary = json.loads((ROOT / 'aeroacoustic_regeneration_summary.json').read_text(encoding='utf-8'))
    measurement = a.load_poc2x_1000rpm_measurement_data(calibration.DEFAULT_MEASUREMENT_EXCEL)
    for job in summary['jobs']:
        output = Path(job['output_dir'])
        observers = tuple(job.get('observers', (2, 3, 4) if output.name == 'overlay_POC2x_measurements'
                                  else a.DEFAULT_OBSERVERS))
        datasets = []
        for source_file in map(Path, job['sources']):
            datasets.append(load_source(source_file, observers))
        a.LEGEND_LABEL_OVERRIDE = None
        a.LEGEND_LABELS_BY_FILE = {}
        combined = job.get('combine_csv', 'Pusher' in output.parts and output.name == 'step3B')
        skip = job.get('n_timesteps_skip', 150 if combined else 0)
        skip_unit = job.get('skip_unit', 'cfd')
        revolution_range = a._parse_revolution_range(job.get('revolutions', 'all'))
        if combined:
            datasets = [a.prepare_acoustic_dataset(a.combine_acoustic_datasets(datasets), skip, skip_unit, revolution_range)]
        else:
            datasets = [a.prepare_acoustic_dataset(dataset, skip, skip_unit, revolution_range) for dataset in datasets]
        if 'Pusher' in output.parts:
            if output.name == 'step3B':
                a.LEGEND_LABEL_OVERRIDE = 'Pusher step3 + step3B, first 150 CFD steps excluded'
            else:
                a.LEGEND_LABEL_OVERRIDE = 'Pusher step3 (2186 RPM)'
        if output.name == 'comparison_POC2_Lifter4B_Q40LT':
            a.LEGEND_LABELS_BY_FILE = {
                dataset.source_file.resolve(): 'flow360 WMLES ' + name
                for dataset, name in zip(datasets, ('POC2', 'Lifter4B', 'Q40LT'))}
        results = [(dataset, {}) for dataset in datasets]
        print(f'Checking spectral display ranges: {output}', flush=True)
        for weighted in (True, False):
            a.APPLY_A_WEIGHTING_TO_SPECTRA = weighted
            files = a.render_spectrum_plots(results, observers, output)
            files += a.render_psd_plots(results, observers, output)
            files += a.render_third_octave_plots(
                results, observers, output,
                measurement_spectra_by_observer=measurement.spectra_by_observer
                if weighted and 'measurement' in output.name else None)
            files += a.render_surface_pair_third_octave_plots(results, observers, output)
            for file in files:
                legacy = output / file.name
                if legacy.exists():
                    shutil.copy2(file, legacy)
        if 'measurement' in output.name:
            remove_obsolete_measurement_oaspl(output, list(map(Path, job['sources'])))


def main() -> None:
    # Reuse the same immutable input arrays between weighted/unweighted plots.
    a.load_acoustic_dataset = functools.lru_cache(maxsize=64)(a.load_acoustic_dataset)
    poc3, poc4 = source('POC2', 'step3'), source('POC2', 'step4')
    lif3, lif4 = source('Lifter4B', 'step3'), source('Lifter4B', 'step4')
    push3, pushb = source('Pusher', 'step3'), source('Pusher', 'step3B')
    workbook = calibration.DEFAULT_MEASUREMENT_EXCEL
    if not workbook.exists():
        raise FileNotFoundError(workbook)
    jobs = [
        ('POC2/aeroacoustic_plots/step3_step4_comparison', [poc3, poc4], []),
        ('POC2/aeroacoustic_plots/step4_measurements_comparison', [poc4],
         ['--measurement-excel', str(workbook)]),
        ('Lifter4B/aeroacoustic_plots', [lif4], []),
        ('Lifter4B/aeroacoustic_plots/step3', [lif3], []),
        ('Lifter4B/aeroacoustic_plots/overlay_POC2x_measurements', [lif4],
         ['--measurement-excel', str(workbook), '--observers', '2,3,4']),
        ('Pusher/aeroacoustic_plots/step3', [push3],
         ['--legend-label', 'Pusher step3 (2186 RPM)']),
        ('Pusher/aeroacoustic_plots/step3B', [push3, pushb],
         ['--combine-csv', '--n-timesteps-skip', '150', '--skip-unit', 'cfd',
          '--legend-label', 'Pusher step3 + step3B, first 150 CFD steps excluded']),
    ]
    q40 = Path('F:/WDIR/flow360/DUC/Q40LT/duc_case1_step4/total_acoustics_v3.csv')
    if q40.exists():
        jobs.append(('comparison_POC2_Lifter4B_Q40LT', [poc4, lif4, q40],
                     ['--legend-labels', 'flow360 WMLES POC2,flow360 WMLES Lifter4B,flow360 WMLES Q40LT']))
    existing = {relative: list((ROOT / relative).glob('*.png')) for relative, _, _ in jobs}
    baseline_path = ROOT / 'Lifter4B_corrected/oaspl_comparison.csv'
    baseline = list(csv.DictReader(baseline_path.open(encoding='utf-8'))) if baseline_path.exists() else []
    started = time.time()
    commands = []
    records = []
    for relative, files, settings in jobs:
        output = ROOT / relative
        output.mkdir(parents=True, exist_ok=True)
        for weighting in ('--a-weighting', '--no-a-weighting'):
            argv = ['--output-dir', str(output), '--csv-observer-offset', '-1']
            for file in files:
                argv += ['--csv', str(file)]
            argv += settings + [weighting]
            command = subprocess.list2cmdline([sys.executable, str(Path(a.__file__).resolve()), *argv])
            commands.append(command)
            print(f'Regenerating {relative}: {weighting}', flush=True)
            sys.argv = [a.__file__, *argv]
            with (output / ('regeneration_a_weighted.log' if weighting == '--a-weighting'
                            else 'regeneration_unweighted.log')).open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log):
                    a.main()
        if output.name == 'step3B' and 'Pusher' in output.parts:
            for suffix in ('a_weighted', 'unweighted'):
                shutil.copy2(output / f'regeneration_{suffix}.log', output / f'analysis_{suffix}.log')
        # Refresh historic flat-layout copies without leaving biased band plots there.
        for old in existing[relative]:
            candidates = [output / subdir / old.name for subdir in ('a-weighted', 'unweighted')]
            for candidate in candidates:
                if candidate.exists() and candidate.stat().st_mtime >= started:
                    shutil.copy2(candidate, old)
                    break
        if '--measurement-excel' in settings:
            remove_obsolete_measurement_oaspl(output, files)
        combined = '--combine-csv' in settings
        skip = int(settings[settings.index('--n-timesteps-skip') + 1]) if '--n-timesteps-skip' in settings else 0
        selected_observers = tuple(map(int, settings[settings.index('--observers') + 1].split(','))) \
            if '--observers' in settings else a.DEFAULT_OBSERVERS
        revolutions = settings[settings.index('--revolutions') + 1] if '--revolutions' in settings else 'all'
        records.append({'output_dir': str(output), 'sources': [str(f) for f in files],
                        'observers': list(selected_observers), 'combine_csv': combined,
                        'n_timesteps_skip': skip, 'skip_unit': 'cfd', 'revolutions': revolutions,
                        'png_count': len([p for p in output.rglob('*.png') if p.stat().st_mtime >= started])})
    print('Regenerating POC2-calibrated Lifter4B estimates', flush=True)
    with (ROOT / 'Lifter4B_corrected/regeneration.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            calibration.run(workbook, poc4, lif4, ROOT / 'Lifter4B_corrected')
    commands.append(subprocess.list2cmdline([sys.executable, str(Path(calibration.__file__).resolve()),
                                           '--measurement-excel', str(workbook)]))
    update_pusher_comparison()
    current = list(csv.DictReader(baseline_path.open(encoding='utf-8')))
    changes = []
    for old, new in zip(baseline, current):
        changes.append({'observer': int(new['observer']), **{
            key + '_change_db': float(new[key]) - float(old[key])
            for key in ('poc2_cfd_dba', 'lifter4b_cfd_dba', 'lifter4b_corrected_dba')
        }})
    summary = {'power_normalization': 'Welch PSD times frequency-bin width',
               'legacy_bias_db': 1.7609125905568124,
               'narrowband_normalization': 'Amplitude spectrum for tone peaks plus PSD for broadband comparison',
               'third_octave_convention': 'IEC base-10 centers, full bands; measurement convention assumed',
               'measurement_comparison': 'A-weighted spectra only; overall levels matched to measurement bands',
               'selection_origin': 'Each observer active acoustic interval',
               'jobs': records, 'lifter_calibration_changes': changes, 'commands': commands}
    (ROOT / 'aeroacoustic_regeneration_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (ROOT / 'aeroacoustic_regeneration_commands.txt').write_text('\n'.join(commands) + '\n', encoding='utf-8')
    print(json.dumps(summary['lifter_calibration_changes'], indent=2), flush=True)
    print('Completed all requested acoustic plot sets.', flush=True)


if __name__ == '__main__':
    main()
