"""Estimate Lifter4B noise using the measured-minus-CFD POC2 band error.

The estimate is made on A-weighted third-octave *band energies*. No pressure
time series is synthesized, and no correction is inferred for unmeasured angles.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import plot_aeroacoustics as acoustics
except ImportError:  # Direct CLI execution.
    import plot_aeroacoustics as acoustics


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "Lifter4B_corrected"
DEFAULT_MEASUREMENT_EXCEL = (
    Path("C:/OneDrive/OneDrive - Achleitner Aerospace GmbH/")
    / "Achleitner Aerospace GmbH Allgemein - General/01_Projekte/27_DUC_Noise/POC_2x"
    / "SPL DUC_POC2_2X RPM_SWEEP_SBD_ASSESSMENT dBA.xlsx"
)
DEFAULT_POC2_CSV = (
    ROOT / "POC2" / "aeroacoustics" / "step4"
    / "case-fb40c043-c99a-4cce-95cf-de9630dba98d_total_acoustics_v3.csv"
)
DEFAULT_LIFTER_CSV = (
    ROOT / "Lifter4B" / "aeroacoustics" / "step4"
    / "case-cf29e49e-da21-43ad-80e2-61a6f97bec80_results_total_acoustics_v3.csv"
)
OBSERVERS = (2, 3, 4)


def nominal_to_exact_centers(frequencies_hz: np.ndarray) -> np.ndarray:
    """Use the plotting script's IEC base-10 bands for nominal measurement labels."""
    centers = acoustics.exact_third_octave_centers(frequencies_hz)
    if (centers.ndim != 1 or not centers.size or np.unique(centers).size != centers.size
            or np.any(np.diff(centers) <= 0)):
        raise ValueError("Measurement bands must be nonempty, unique and ascending")
    return centers


def energy_sum_db(band_levels_db: np.ndarray) -> float:
    """Sum band mean-square pressure, including any A-weighting already present."""
    band_levels_db = np.asarray(band_levels_db, dtype=float)
    if band_levels_db.ndim != 1 or not band_levels_db.size:
        raise ValueError("Need a nonempty one-dimensional band-level array")
    if np.any(np.isnan(band_levels_db)) or np.any(np.isposinf(band_levels_db)):
        raise ValueError("Cannot sum unavailable or invalid band levels")
    power = float(np.sum(10.0 ** (band_levels_db / 10.0)))
    return -np.inf if power == 0.0 else float(10.0 * np.log10(power))


def cfd_band_levels(dataset: acoustics.AcousticDataset, observer: int, centers_hz: np.ndarray) -> np.ndarray:
    pressure_pa = acoustics._observer_pressure(dataset, observer)
    levels = acoustics.third_octave_spectrum_at_centers_db(
        pressure_pa, dataset.sample_rate_hz, centers_hz, apply_a_weighting=True
    )
    if not np.all(np.isfinite(levels)):
        unavailable = np.asarray(centers_hz)[~np.isfinite(levels)]
        raise ValueError(
            "Cannot calibrate bands with unavailable or zero CFD power: "
            + ", ".join(f"{center:.1f} Hz" for center in unavailable)
        )
    return levels


def load_cfd(csv_file: Path) -> acoustics.AcousticDataset:
    rpm, steps_per_revolution = acoustics._timing_for_file(csv_file, None, None)
    density, velocity, temperature, length_unit = acoustics._reference_values_for_file(
        csv_file, None, None, None, None
    )
    return acoustics.load_acoustic_dataset(
        csv_file,
        OBSERVERS,
        rpm=rpm,
        time_steps_per_revolution=steps_per_revolution,
        reference_density_kg_m3=density,
        reference_velocity_m_s=velocity,
        reference_temperature_k=temperature,
        l_grid_unit_m=length_unit,
    )


def write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def plot_spectrum(
    output_file: Path,
    observer: int,
    frequency_hz: np.ndarray,
    measurement_db: np.ndarray,
    poc2_cfd_db: np.ndarray,
    lifter_cfd_db: np.ndarray,
    corrected_db: np.ndarray,
    measurement_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    series = (
        (measurement_db, measurement_label, "black", "x", "--"),
        (poc2_cfd_db, "POC2 CFD", "#1976a3", "o", "-"),
        (lifter_cfd_db, "Lifter4B CFD", "#e3771c", "s", "-"),
        (corrected_db, "Lifter4B POC2-calibrated estimate", "#a5286b", "D", "-"),
    )
    for levels, label, color, marker, style in series:
        ax.plot(frequency_hz, levels, label=label, color=color, marker=marker,
                markerfacecolor="none", markersize=5, linewidth=1.6, linestyle=style)
    ax.set(xscale="log", xlim=(45, 11200), xlabel="Third-octave center frequency [Hz]",
           ylabel="A-weighted band level [dBA]",
           title=f"POC2 calibration transferred to Lifter4B — observer {observer} "
                 f"({acoustics.OBSERVER_ANGLES_DEG[observer]}°)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_file.unlink(missing_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_oaspl(output_file: Path, rows: list[tuple], measurement_label: str) -> None:
    angles = np.asarray([row[1] for row in rows], dtype=float)
    columns = (2, 3, 4, 5)
    labels = (measurement_label, "POC2 CFD", "Lifter4B CFD", "Lifter4B calibrated")
    colors = ("black", "#1976a3", "#e3771c", "#a5286b")
    x = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for index, (column, label, color) in enumerate(zip(columns, labels, colors)):
        bars = ax.bar(x + (index - 1.5) * 0.19, [row[column] for row in rows],
                      width=0.18, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=[f"{angle:g}°" for angle in angles],
           xlabel="Observer angle", ylabel="A-weighted OASPL, 50 Hz–10 kHz band centers [dBA]",
           title="POC2-calibrated Lifter4B OASPL (measurements as reported)")
    ax.set_ylim(0, max(row[column] for row in rows for column in columns) + 8)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_file.unlink(missing_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def run(measurement_excel: Path, poc2_csv: Path, lifter_csv: Path,
        output_dir: Path) -> None:
    measurement = acoustics.load_poc2x_1000rpm_measurement_data(measurement_excel)
    if set(measurement.spectra_by_observer) != set(OBSERVERS):
        raise ValueError(f"Need POC2 measurements for observers {OBSERVERS}")
    for observer, spectrum in measurement.spectra_by_observer.items():
        if spectrum.weighting != "A" or spectrum.band_base != "10":
            raise ValueError(
                f"Observer {observer} calibration needs A-weighted measurement bands "
                "using the IEC base-10 comparison convention"
            )
    poc2 = load_cfd(poc2_csv)
    lifter = load_cfd(lifter_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_label = "POC2 measurement (as reported)"
    oaspl_rows = []
    measurement_rows = []
    for observer in OBSERVERS:
        measured = measurement.spectra_by_observer[observer]
        nominal_hz = measured.frequencies_hz
        centers_hz = nominal_to_exact_centers(nominal_hz)
        measured_db = measured.spl_db
        if not np.isclose(energy_sum_db(measured_db), measurement.oaspl_dba_by_observer[observer], atol=0.01):
            raise ValueError(f"Measurement OASPL does not equal band-energy sum for observer {observer}")
        poc2_db = cfd_band_levels(poc2, observer, centers_hz)
        lifter_db = cfd_band_levels(lifter, observer, centers_hz)
        correction_db = measured_db - poc2_db
        corrected_db = lifter_db + correction_db
        rows = [
            (f"{frequency:.0f}", f"{measured_level:.6f}", f"{baseline:.6f}",
             f"{target:.6f}", f"{correction:.6f}", f"{estimate:.6f}")
            for frequency, measured_level, baseline, target, correction, estimate in zip(
                nominal_hz, measured_db, poc2_db, lifter_db,
                correction_db, corrected_db
            )
        ]
        write_csv(
            output_dir / f"third_octave_observer_{observer}.csv",
            ("nominal_frequency_hz", "poc2_measurement_dba",
             "poc2_cfd_dba", "lifter4b_cfd_dba", "poc2_correction_db", "lifter4b_corrected_dba"),
            rows,
        )
        measurement_rows.extend((observer, acoustics.OBSERVER_ANGLES_DEG[observer],
                                 f"{frequency:.0f}", f"{level:.6f}")
                                for frequency, level in zip(nominal_hz, measured_db))
        plot_spectrum(output_dir / f"third_octave_observer_{observer}.png", observer,
                      nominal_hz, measured_db, poc2_db, lifter_db, corrected_db,
                      measurement_label)
        oaspl_rows.append((
            observer, acoustics.OBSERVER_ANGLES_DEG[observer],
            energy_sum_db(measured_db), energy_sum_db(poc2_db),
            energy_sum_db(lifter_db), energy_sum_db(corrected_db),
        ))
    write_csv(output_dir / "poc2_measurements_1000rpm.csv",
              ("observer", "angle_deg", "nominal_frequency_hz", "measured_band_dba"),
              measurement_rows)
    write_csv(output_dir / "oaspl_comparison.csv",
              ("observer", "angle_deg", "poc2_measurement_dba", "poc2_cfd_dba",
               "lifter4b_cfd_dba", "lifter4b_corrected_dba"),
              [(row[0], row[1], *(f"{value:.6f}" for value in row[2:])) for row in oaspl_rows])
    plot_oaspl(output_dir / "oaspl_comparison.png", oaspl_rows, measurement_label)
    readme = f"""# Lifter4B POC2-calibrated acoustic estimate

Source measurement: `{measurement_excel}`; 1000 RPM, A-weighted third-octave bands,
50–10,000 Hz, observers 2–4 (90°, 105°, 135°). Source CFD: `{poc2_csv}` and
`{lifter_csv}`. The 24 bands use IEC base-10 exact third-octave edges; reported nominal
frequencies are retained in the tables. CFD pressures are dimensionalized and
Welch bin powers are computed as PSD times frequency-bin width, with window-energy
normalization, as in `plot_aeroacoustics.py`. OASPL is recomputed
from the 24 A-weighted band energies, not corrected as a single dB value.
The top band is centered at 10 kHz and includes its full upper half-band.
IEC base-10 bands are assumed for the measurement's nominal labels; the workbook
does not establish the analyser filter convention. FFT-bin integration is not
equivalent to a standards-certified filter bank, especially for short records.

The measured levels are used *as reported* by the workbook, matching the
existing `plot_aeroacoustics.py --measurement-excel ... --a-weighting` comparison.
No distance or atmospheric-absorption adjustment is applied. The separate
microphone-position file gives a radius around 15.45 in its coordinate units;
those units and the workbook's normalization basis have not been established.
The POC2 CFD setup specifies observers around 100 m; therefore this transfer
is conditional on the workbook already representing the intended comparison
basis. Its levels should not be interpreted as a validated prediction at a
specified distance without resolving that basis.

For every observer and third-octave band: `correction = POC2 measurement as
reported - POC2 CFD`; `Lifter4B estimate = Lifter4B CFD + correction`.
The same relation is equivalent to multiplying band mean-square pressures by
the POC2 measured/CFD power ratio. Tones and broadband were not separated,
because the measurement source provides only third-octave levels. POC2 CFD is
at {poc2.rpm:g} RPM and Lifter4B CFD at {lifter.rpm:g} RPM; the calibrated
estimate assumes the POC2 model discrepancy transfers across this operating
point difference. It is an engineering estimate, not an independent validation.

Run again with:

`python correct_lifter4b_acoustics.py --measurement-excel \"{measurement_excel}\"`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    for row in oaspl_rows:
        print(f"Observer {row[0]} ({row[1]}°): measured {row[2]:.2f}, "
              f"POC2 CFD {row[3]:.2f}, Lifter4B CFD {row[4]:.2f}, "
              f"corrected {row[5]:.2f} dBA")
    print(f"Wrote {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-excel", type=Path, default=DEFAULT_MEASUREMENT_EXCEL)
    parser.add_argument("--poc2-csv", type=Path, default=DEFAULT_POC2_CSV)
    parser.add_argument("--lifter-csv", type=Path, default=DEFAULT_LIFTER_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.measurement_excel, args.poc2_csv, args.lifter_csv,
        args.output_dir)


if __name__ == "__main__":
    main()
