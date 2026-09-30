#!/usr/bin/env python3
"""
Acoustic signal post-processing for Flow360 acoustic CSV.

Usage:
python spl_spectra_calculator.py \
    --input case_total_acoustics.csv \
    --observer 7 \
    --apply-a-weighting
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, MultipleLocator, FuncFormatter
import argparse

# ============================================================
# USER INPUTS (hardcoded)
# ============================================================
AIR_SPECIFIC_HEAT_RATIO = 1.4
AIR_GAS_CONSTANT_J_KG_K = 287.05287

# Local Lifter4B Flow360 reference quantities.
"""REFERENCE_TEMPERATURE_K = 303.275
REFERENCE_DENSITY_KG_M3 = 1.064099
L_GRID_UNIT = 0.001"""

# POC2x2 reference quantities.
REFERENCE_TEMPERATURE_K = 283.275
REFERENCE_DENSITY_KG_M3 = 1.149
L_GRID_UNIT = 1.0

U_SCALE = (AIR_SPECIFIC_HEAT_RATIO * AIR_GAS_CONSTANT_J_KG_K * REFERENCE_TEMPERATURE_K) ** 0.5

FMIN = 40.0
FMAX = 10000.0

P_REF = 20e-6  # Pa
APPLY_A_WEIGHTING_DEFAULT = True

# Use "All" to use the maximum possible Welch segment length (= all samples)
# or set it to an integer, for example 32768
NPERSEG = "All"

WINDOW = "hann"


# ============================================================
# HELPERS
# ============================================================
def a_weighting_db(frequencies):
    frequencies = np.asarray(frequencies, dtype=float)
    f_squared = np.square(frequencies)
    ra = (12200.0**2 * f_squared**2) / (
        (f_squared + 20.6**2) *
        (f_squared + 12200.0**2) *
        np.sqrt((f_squared + 107.7**2) * (f_squared + 737.9**2))
    )
    epsilon = 1e-12
    ra = np.maximum(ra, epsilon)
    ra_1khz = (12200.0**2 * (1000.0**2)**2) / (
        (1000.0**2 + 20.6**2) *
        (1000.0**2 + 12200.0**2) *
        np.sqrt((1000.0**2 + 107.7**2) * (1000.0**2 + 737.9**2))
    )
    a_db = 20.0 * np.log10(ra / ra_1khz)
    a_db = np.where(frequencies > 0.0, a_db, -np.inf)
    return a_db


def weighting_db(frequencies, apply_a):
    if apply_a:
        return a_weighting_db(frequencies)
    return np.zeros_like(np.asarray(frequencies, dtype=float))


def filter_zero_signal_edges(time, pressure):
    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    nz = np.where(pressure != 0.0)[0]
    if nz.size == 0:
        return time, pressure
    return time[nz[0]:nz[-1] + 1], pressure[nz[0]:nz[-1] + 1]


def third_octave_centers(fmin, fmax):
    centers = []
    k_min = int(np.floor(3.0 * np.log2(max(fmin, 1e-12) / 1000.0)))
    k_max = int(np.ceil(3.0 * np.log2(fmax / 1000.0)))
    for k in range(k_min, k_max + 1):
        fc = 1000.0 * (2.0 ** (k / 3.0))
        if fmin <= fc <= fmax:
            centers.append(fc)
    return np.array(centers, dtype=float)


def third_octave_bands(fmin, fmax):
    centers = third_octave_centers(fmin, fmax)
    factor = 2.0 ** (1.0 / 6.0)
    return centers, centers / factor, centers * factor


def db_from_power(p2):
    tiny = np.finfo(float).tiny
    return 10.0 * np.log10(np.maximum(p2, tiny) / (P_REF**2))


def get_nperseg(n_samples):
    if isinstance(NPERSEG, str) and NPERSEG.strip().lower() == "all":
        nperseg = n_samples
        nperseg_label = "All"
    else:
        nperseg = min(int(NPERSEG), n_samples)
        nperseg_label = str(int(NPERSEG))
    return nperseg, nperseg_label


def sci_formatter(x, pos):
    if x == 0:
        return "0"
    return f"{x:.0f}"


# ============================================================
# SPECTRA CALCULATION
# ============================================================
def spectra_calculation(input_path, observer_index, apply_a):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    if "time" not in df.columns:
        raise ValueError("Missing 'time' column")

    pcol = f"observer_{observer_index}_pressure"
    if pcol not in df.columns:
        raise ValueError(f"{pcol} not found")

    t_raw = df["time"].values
    p_raw = df[pcol].values

    t_raw, p_raw = filter_zero_signal_edges(t_raw, p_raw)

    if len(t_raw) < 2:
        raise ValueError("Not enough non-zero samples after trimming.")

    t = t_raw * (L_GRID_UNIT / U_SCALE)
    p = p_raw * (REFERENCE_DENSITY_KG_M3 * U_SCALE**2)

    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    nperseg, nperseg_label = get_nperseg(len(p))
    if nperseg < 8:
        raise ValueError("Too few samples for Welch calculation.")

    f, p2 = welch(
        p - np.mean(p),
        fs=fs,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="spectrum",
    )

    mask = (f >= FMIN) & (f <= FMAX)
    f_nb = f[mask]
    p2_nb = p2[mask]

    if len(f_nb) == 0:
        raise ValueError("No frequencies remain inside [FMIN, FMAX]. Increase NPERSEG or adjust FMIN/FMAX.")

    w_nb = weighting_db(f_nb, apply_a)
    p2_nb_w = p2_nb * (10.0 ** (w_nb / 10.0))

    spl_nb = db_from_power(p2_nb)
    spl_nb_out = db_from_power(p2_nb_w) if apply_a else spl_nb

    centers, fl, fu = third_octave_bands(FMIN, FMAX)
    p2_oct = []
    p2_oct_w = []

    for lo, hi in zip(fl, fu):
        m = (f_nb >= lo) & (f_nb < hi)
        band_p2 = np.sum(p2_nb[m])
        p2_oct.append(band_p2)

        if np.any(m):
            band_p2_w = np.sum(p2_nb[m] * (10.0 ** (w_nb[m] / 10.0)))
        else:
            band_p2_w = 0.0
        p2_oct_w.append(band_p2_w)

    p2_oct = np.array(p2_oct)
    p2_oct_w = np.array(p2_oct_w)

    spl_oct = db_from_power(p2_oct)
    spl_oct_out = db_from_power(p2_oct_w) if apply_a else spl_oct

    oaspl_nb = db_from_power(np.sum(p2_nb_w)) if apply_a else db_from_power(np.sum(p2_nb))
    oaspl_oct = db_from_power(np.sum(p2_oct_w)) if apply_a else db_from_power(np.sum(p2_oct))

    return {
        "input_path": input_path,
        "observer_index": observer_index,
        "pressure_column": pcol,
        "apply_a_weighting": apply_a,
        "samples_after_trim": len(p_raw),
        "dt": dt,
        "fs": fs,
        "nperseg": nperseg,
        "nperseg_label": nperseg_label,
        "f_nb": f_nb,
        "p2_nb": p2_nb,
        "p2_nb_w": p2_nb_w,
        "spl_nb": spl_nb,
        "spl_nb_out": spl_nb_out,
        "w_nb": w_nb,
        "centers": centers,
        "lower_band": fl,
        "upper_band": fu,
        "p2_oct": p2_oct,
        "p2_oct_w": p2_oct_w,
        "spl_oct": spl_oct,
        "spl_oct_out": spl_oct_out,
        "oaspl_nb": oaspl_nb,
        "oaspl_oct": oaspl_oct,
    }


# ============================================================
# SPECTRA WRITE
# ============================================================
def spectra_write(results):
    input_path = results["input_path"]
    observer_index = results["observer_index"]
    apply_a = results["apply_a_weighting"]

    output_prefix = f"{input_path.stem}_observer{observer_index}"

    narrowband_df = pd.DataFrame({
        "frequency_hz": results["f_nb"],
        "band_power_pa2": results["p2_nb_w"] if apply_a else results["p2_nb"],
        "spl_dba" if apply_a else "spl_db": results["spl_nb_out"],
    })
    narrowband_csv = f"{output_prefix}_narrowband.csv"
    narrowband_df.to_csv(narrowband_csv, index=False)

    octave_df = pd.DataFrame({
        "center_frequency_hz": results["centers"],
        "lower_frequency_hz": results["lower_band"],
        "upper_frequency_hz": results["upper_band"],
        "band_power_pa2": results["p2_oct_w"] if apply_a else results["p2_oct"],
        "spl_dba" if apply_a else "spl_db": results["spl_oct_out"],
    })
    octave_csv = f"{output_prefix}_third_octave.csv"
    octave_df.to_csv(octave_csv, index=False)

    summary_df = pd.DataFrame([{
        "input_csv": str(input_path),
        "observer_index": observer_index,
        "pressure_column": results["pressure_column"],
        "apply_a_weighting": apply_a,
        "u_scale_m_per_s": U_SCALE,
        "density_kg_per_m3": REFERENCE_DENSITY_KG_M3,
        "temperature_k": REFERENCE_TEMPERATURE_K,
        "l_grid_unit_m": L_GRID_UNIT,
        "fmin_hz": FMIN,
        "fmax_hz": FMAX,
        "nperseg": results["nperseg_label"],
        "sampling_rate_hz": results["fs"],
        "dt_seconds_mean": results["dt"],
        "n_samples_after_zero_trim": results["samples_after_trim"],
        "oaspl_narrowband_dba" if apply_a else "oaspl_narrowband_db": results["oaspl_nb"],
        "oaspl_third_octave_dba" if apply_a else "oaspl_third_octave_db": results["oaspl_oct"],
    }])
    summary_csv = f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    return {
        "output_prefix": output_prefix,
        "narrowband_csv": narrowband_csv,
        "octave_csv": octave_csv,
        "summary_csv": summary_csv,
    }


# ============================================================
# SPECTRA PLOT
# ============================================================
def spectra_plot(results):
    input_path = results["input_path"]
    observer_index = results["observer_index"]
    apply_a = results["apply_a_weighting"]
    output_prefix = f"{input_path.stem}_observer{observer_index}"

    yunit = "dBA" if apply_a else "dB"

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.semilogx(
        results["f_nb"],
        results["spl_nb_out"],
        label=f"Narrowband OASPL={results['oaspl_nb']:.2f} {yunit}",
        linewidth=2,
    )
    ax.semilogx(
        results["centers"],
        results["spl_oct_out"],
        marker="o",
        linestyle="-",
        label=f"1/3 Octave OASPL={results['oaspl_oct']:.2f} {yunit}",
    )

    ax.set_xlabel("Frequency [Hz]", fontsize=16)
    ax.set_ylabel(f"SPL [{yunit}]", fontsize=16)
    ax.set_title(f"caseID: {input_path.stem} - Observer {observer_index}", fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle="--", alpha=0.35)

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 4, 6, 8), numticks=5))
    ax.xaxis.set_major_formatter(FuncFormatter(sci_formatter))
    ax.xaxis.set_minor_formatter(FuncFormatter(sci_formatter))
    
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', labelsize=8)

    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor', labelsize=8)

    ax.yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(FuncFormatter(sci_formatter))

    ax.legend(fontsize=12)
    plt.tight_layout()
    ax.set_xlim(FMIN, FMAX)

    fig_path = f"{output_prefix}_spectrum.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return fig_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Acoustic SPL post-processing")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--observer", required=True, type=int, help="Observer index (0..N)")
    parser.add_argument("--apply-a-weighting", action="store_true", help="Apply A-weighting")
    args = parser.parse_args()

    apply_a = args.apply_a_weighting if args.apply_a_weighting else APPLY_A_WEIGHTING_DEFAULT

    results = spectra_calculation(
        input_path=args.input,
        observer_index=args.observer,
        apply_a=apply_a,
    )

    written_files = spectra_write(results)
    fig_path = spectra_plot(results)

    yunit = "dBA" if apply_a else "dB"

    print("=== Acoustic post-processing summary ===")
    print(f"Input CSV              : {results['input_path']}")
    print(f"Observer index         : {results['observer_index']}")
    print(f"Pressure column        : {results['pressure_column']}")
    print(f"A-weighting applied    : {results['apply_a_weighting']}")
    print(f"U_scale [m/s]          : {U_SCALE}")
    print(f"Density [kg/m^3]       : {REFERENCE_DENSITY_KG_M3}")
    print(f"Temperature [K]        : {REFERENCE_TEMPERATURE_K}")
    print(f"L_grid_unit [m]        : {L_GRID_UNIT}")
    print(f"Welch segments         : {results['nperseg_label']}")
    print(f"Samples after trim     : {results['samples_after_trim']}")
    print(f"Sampling rate [Hz]     : {results['fs']:.2f}")
    print(f"Mean dt [s]            : {results['dt']:.4e}")
    print(f"Frequency range [Hz]   : {FMIN:.1f} to {FMAX:.1f}")
    print(f"Narrowband OASPL [{yunit}] : {results['oaspl_nb']:.2f}")
    print(f"1/3 octave OASPL [{yunit}] : {results['oaspl_oct']:.2f}")
    print()
    print("Wrote:")
    print(f"  {written_files['narrowband_csv']}")
    print(f"  {written_files['octave_csv']}")
    print(f"  {written_files['summary_csv']}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
