from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent
AEROACOUSTICS_DIR = RESULTS_DIR / "aeroacoustics"
PLOTS_DIR = RESULTS_DIR / "aeroacoustic_plots"

ACOUSTIC_PRESSURE_REFERENCE_PA = 20.0e-6
AIR_SPECIFIC_HEAT_RATIO = 1.4
AIR_GAS_CONSTANT_J_KG_K = 287.05287
DEFAULT_OBSERVERS = (1, 2, 3, 4, 5)
OBSERVER_ANGLES_DEG = {
    1: 0,
    2: 90,
    3: 105,
    4: 135,
    5: 180,
}
FLOW360_OBSERVER_NUMBERING_OFFSET = -1
DEFAULT_RPM = 1063.677
STEP3_TIME_STEPS_PER_REVOLUTION = 120 * 32
STEP4_TIME_STEPS_PER_REVOLUTION = STEP3_TIME_STEPS_PER_REVOLUTION * 25 / 12
DEFAULT_TIME_STEPS_PER_REVOLUTION = STEP4_TIME_STEPS_PER_REVOLUTION
DEFAULT_BLADE_COUNT = 4
LOCAL_RESULTS_REFERENCE_DENSITY_KG_M3 = 1.064099
LOCAL_RESULTS_REFERENCE_VELOCITY_M_S = 155.9432
LOCAL_RESULTS_REFERENCE_TEMPERATURE_K = 303.275
POC2X2_REFERENCE_DENSITY_KG_M3 = 1.149
POC2X2_REFERENCE_VELOCITY_M_S = 146.857683
POC2X2_REFERENCE_TEMPERATURE_K = 283.275
CASE_TIMING_BY_PREFIX = {
    "step3": (DEFAULT_RPM, STEP3_TIME_STEPS_PER_REVOLUTION),
    "step4": (DEFAULT_RPM, STEP4_TIME_STEPS_PER_REVOLUTION),
}
COMPARISON_OUTPUT_FILE = "oaspl_observers_1_5_comparison.png"
OASPL_MIN_FREQUENCY_HZ = 40.0
OASPL_MAX_FREQUENCY_HZ = 1_000.0
SPECTRUM_MIN_FREQUENCY_HZ = 40.0
SPECTRUM_MAX_FREQUENCY_HZ = 10_000.0
SPECTRUM_X_TICKS_HZ = (40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000)
SPECTRUM_Y_LIMITS_DB = (-20.0, 70.0)
HARMONIC_INDICES = tuple(range(1, 11))
HARMONIC_Y_LIMITS_DB = (0.0, 110.0)
THIRD_OCTAVE_CENTER_FREQUENCIES_HZ = (
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
)


@dataclass(frozen=True)
class AcousticDataset:
    source_file: Path
    time: np.ndarray
    pressure_by_observer: dict[int, np.ndarray]
    thickness_by_observer: dict[int, np.ndarray]
    loading_by_observer: dict[int, np.ndarray]
    rpm: float
    time_steps_per_revolution: float
    reference_density_kg_m3: float
    reference_velocity_m_s: float
    reference_temperature_k: float

    @property
    def sample_spacing_s(self) -> float:
        return 60.0 / (self.rpm * self.time_steps_per_revolution)

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.sample_spacing_s

    @property
    def pressure_scale_pa(self) -> float:
        return (
            AIR_SPECIFIC_HEAT_RATIO
            * self.reference_density_kg_m3
            * AIR_GAS_CONSTANT_J_KG_K
            * self.reference_temperature_k
        )


@dataclass(frozen=True)
class PlotDefinition:
    name: str
    filename: str
    evaluator: Callable[[np.ndarray, float], float]
    ylabel: str
    title: str


def _csv_observer_index(user_observer_index: int) -> int:
    return user_observer_index + FLOW360_OBSERVER_NUMBERING_OFFSET


def _default_acoustic_files() -> list[Path]:
    search_dirs = [
        AEROACOUSTICS_DIR,
        RESULTS_DIR.parent / "POC2x2" / "results" / "aeroacoustics",
    ]
    candidates = []
    for results_dir in search_dirs:
        candidates.extend(sorted(results_dir.glob("*_results_total_acoustics_v*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No Flow360 total-acoustics CSV files found in {search_dirs}")
    return candidates


def load_acoustic_dataset(
    csv_file: Path | None = None,
    observers: tuple[int, ...] = DEFAULT_OBSERVERS,
    rpm: float = DEFAULT_RPM,
    time_steps_per_revolution: float = DEFAULT_TIME_STEPS_PER_REVOLUTION,
    reference_density_kg_m3: float = LOCAL_RESULTS_REFERENCE_DENSITY_KG_M3,
    reference_velocity_m_s: float = LOCAL_RESULTS_REFERENCE_VELOCITY_M_S,
    reference_temperature_k: float = LOCAL_RESULTS_REFERENCE_TEMPERATURE_K,
) -> AcousticDataset:
    csv_file = _default_acoustic_files()[-1] if csv_file is None else Path(csv_file)
    pressure_columns = {
        observer: f"observer_{_csv_observer_index(observer)}_pressure"
        for observer in observers
    }
    thickness_columns = {
        observer: f"observer_{_csv_observer_index(observer)}_thickness"
        for observer in observers
    }
    loading_columns = {
        observer: f"observer_{_csv_observer_index(observer)}_loading"
        for observer in observers
    }
    time_values: list[float] = []
    pressure_values = {observer: [] for observer in observers}
    thickness_values = {observer: [] for observer in observers}
    loading_values = {observer: [] for observer in observers}

    with csv_file.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        required_columns = [
            *pressure_columns.values(),
            *thickness_columns.values(),
            *loading_columns.values(),
        ]
        missing_columns = [column for column in required_columns if column not in reader.fieldnames]
        if missing_columns:
            raise KeyError(f"Missing acoustic pressure columns in {csv_file}: {missing_columns}")

        for row in reader:
            time_values.append(float(row["time"]))
            for observer, column in pressure_columns.items():
                pressure_values[observer].append(float(row[column]))
            for observer, column in thickness_columns.items():
                thickness_values[observer].append(float(row[column]))
            for observer, column in loading_columns.items():
                loading_values[observer].append(float(row[column]))

    if not time_values:
        raise ValueError(f"No acoustic samples found in {csv_file}")

    return AcousticDataset(
        source_file=csv_file,
        time=np.asarray(time_values, dtype=float),
        pressure_by_observer={
            observer: np.asarray(values, dtype=float)
            for observer, values in pressure_values.items()
        },
        thickness_by_observer={
            observer: np.asarray(values, dtype=float)
            for observer, values in thickness_values.items()
        },
        loading_by_observer={
            observer: np.asarray(values, dtype=float)
            for observer, values in loading_values.items()
        },
        rpm=rpm,
        time_steps_per_revolution=time_steps_per_revolution,
        reference_density_kg_m3=reference_density_kg_m3,
        reference_velocity_m_s=reference_velocity_m_s,
        reference_temperature_k=reference_temperature_k,
    )


def _fluctuating_pressure(pressure: np.ndarray) -> np.ndarray:
    return pressure - np.mean(pressure)


def oaspl_unweighted_db(pressure: np.ndarray, sample_rate_hz: float) -> float:
    frequencies, mean_square_by_bin = _band_limited_mean_square_spectrum(pressure, sample_rate_hz)
    del frequencies
    mean_square_pressure = float(np.sum(mean_square_by_bin))
    return 10.0 * np.log10(mean_square_pressure / ACOUSTIC_PRESSURE_REFERENCE_PA**2)


def _a_weighting_db(frequency_hz: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=float)
    frequency_squared = frequency**2
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = (12194.0**2) * frequency_squared**2
        denominator = (
            (frequency_squared + 20.6**2)
            * np.sqrt((frequency_squared + 107.7**2) * (frequency_squared + 737.9**2))
            * (frequency_squared + 12194.0**2)
        )
        weighting = 20.0 * np.log10(numerator / denominator) + 2.0
    weighting[~np.isfinite(weighting)] = -np.inf
    return weighting


def _one_sided_mean_square_spectrum(pressure: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    pressure_fluctuation = _fluctuating_pressure(pressure)
    sample_count = len(pressure_fluctuation)
    if sample_count < 2:
        raise ValueError("At least two samples are required for spectral OASPL.")

    spectrum = np.fft.rfft(pressure_fluctuation)
    mean_square_by_bin = (np.abs(spectrum) / sample_count) ** 2
    if sample_count % 2 == 0:
        mean_square_by_bin[1:-1] *= 2.0
    else:
        mean_square_by_bin[1:] *= 2.0
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_hz)
    return frequencies, mean_square_by_bin


def _band_limited_mean_square_spectrum(pressure: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    frequencies, mean_square_by_bin = _one_sided_mean_square_spectrum(pressure, sample_rate_hz)
    frequency_mask = (
        (frequencies >= OASPL_MIN_FREQUENCY_HZ)
        & (frequencies <= OASPL_MAX_FREQUENCY_HZ)
    )
    return frequencies[frequency_mask], mean_square_by_bin[frequency_mask]


def oaspl_a_weighted_db(pressure: np.ndarray, sample_rate_hz: float) -> float:
    frequencies, mean_square_by_bin = _band_limited_mean_square_spectrum(pressure, sample_rate_hz)
    linear_a_weight = 10.0 ** (_a_weighting_db(frequencies) / 10.0)
    weighted_mean_square_pressure = float(np.sum(mean_square_by_bin * linear_a_weight))
    return 10.0 * np.log10(weighted_mean_square_pressure / ACOUSTIC_PRESSURE_REFERENCE_PA**2)


def spl_spectrum_db(pressure: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    frequencies, mean_square_by_bin = _one_sided_mean_square_spectrum(pressure, sample_rate_hz)
    positive_frequency_mask = frequencies > 0.0
    frequencies = frequencies[positive_frequency_mask]
    mean_square_by_bin = np.maximum(mean_square_by_bin[positive_frequency_mask], np.finfo(float).tiny)
    spl_db = 10.0 * np.log10(mean_square_by_bin / ACOUSTIC_PRESSURE_REFERENCE_PA**2)
    return frequencies, spl_db


def tone_spl_db(pressure: np.ndarray, sample_rate_hz: float, target_frequency_hz: float) -> tuple[float, float]:
    frequencies, mean_square_by_bin = _one_sided_mean_square_spectrum(pressure, sample_rate_hz)
    valid_frequency_mask = frequencies > 0.0
    frequencies = frequencies[valid_frequency_mask]
    mean_square_by_bin = mean_square_by_bin[valid_frequency_mask]
    nearest_index = int(np.argmin(np.abs(frequencies - target_frequency_hz)))
    mean_square_pressure = max(float(mean_square_by_bin[nearest_index]), np.finfo(float).tiny)
    spl_db = 10.0 * np.log10(mean_square_pressure / ACOUSTIC_PRESSURE_REFERENCE_PA**2)
    return float(frequencies[nearest_index]), float(spl_db)


def third_octave_spectrum_db(pressure: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    frequencies, mean_square_by_bin = _one_sided_mean_square_spectrum(pressure, sample_rate_hz)
    center_frequencies = np.asarray(THIRD_OCTAVE_CENTER_FREQUENCIES_HZ, dtype=float)
    exact_center_frequencies = 1000.0 * 10.0 ** (np.round(10.0 * np.log10(center_frequencies / 1000.0)) / 10.0)
    lower_band_edges = exact_center_frequencies / 10.0 ** (1.0 / 20.0)
    upper_band_edges = exact_center_frequencies * 10.0 ** (1.0 / 20.0)
    band_levels = []

    for lower_edge, upper_edge in zip(lower_band_edges, upper_band_edges):
        band_mask = (frequencies >= lower_edge) & (frequencies < upper_edge)
        mean_square_pressure = float(np.sum(mean_square_by_bin[band_mask]))
        if mean_square_pressure <= 0.0:
            band_levels.append(np.nan)
        else:
            band_levels.append(10.0 * np.log10(mean_square_pressure / ACOUSTIC_PRESSURE_REFERENCE_PA**2))

    return center_frequencies, np.asarray(band_levels, dtype=float)


PLOTS = (
    PlotDefinition(
        name="oaspl_unweighted",
        filename="oaspl_unweighted_observers_1_5.png",
        evaluator=oaspl_unweighted_db,
        ylabel="OASPL [dB]",
        title="Unweighted OASPL, 50-8000 Hz",
    ),
    PlotDefinition(
        name="oaspl_a_weighted",
        filename="oaspl_a_weighted_observers_1_5.png",
        evaluator=oaspl_a_weighted_db,
        ylabel="A-weighted OASPL [dBA]",
        title="A-weighted OASPL, 50-8000 Hz",
    ),
)


def evaluate_plot(dataset: AcousticDataset, plot: PlotDefinition) -> dict[int, float]:
    return {
        observer: plot.evaluator(pressure * dataset.pressure_scale_pa, dataset.sample_rate_hz)
        for observer, pressure in dataset.pressure_by_observer.items()
    }


def print_oaspl_values(
    dataset: AcousticDataset,
    values_by_plot: dict[str, dict[int, float]],
    observers: tuple[int, ...],
) -> None:
    print(f"OASPL values for {_case_label(dataset.source_file)}:")
    for observer in observers:
        unweighted = values_by_plot["oaspl_unweighted"][observer]
        a_weighted = values_by_plot["oaspl_a_weighted"][observer]
        print(f"  Observer {observer}: {unweighted:.2f} dB, {a_weighted:.2f} dBA")


def _case_label(source_file: Path) -> str:
    return source_file.name.split("_results_total_acoustics", 1)[0]


def _step_label(source_file: Path) -> str:
    case_label = _case_label(source_file)
    for separator in ("_case-", "_"):
        if separator in case_label:
            return case_label.split(separator, 1)[0]
    return case_label


def _legend_label(source_file: Path) -> str:
    step_label = _step_label(source_file)
    if "poc2x2" in {part.lower() for part in source_file.resolve().parts}:
        return f"total - Case: POC2x2 {step_label}"
    return f"total - Case: {step_label}"


def _timing_for_file(csv_file: Path, rpm: float | None, time_steps_per_revolution: float | None) -> tuple[float, float]:
    if rpm is not None and time_steps_per_revolution is not None:
        return rpm, time_steps_per_revolution

    step_label = _step_label(csv_file)
    default_rpm, default_steps_per_revolution = CASE_TIMING_BY_PREFIX.get(
        step_label,
        (DEFAULT_RPM, DEFAULT_TIME_STEPS_PER_REVOLUTION),
    )
    return (
        default_rpm if rpm is None else rpm,
        default_steps_per_revolution if time_steps_per_revolution is None else time_steps_per_revolution,
    )


def _reference_values_for_file(
    csv_file: Path,
    reference_density_kg_m3: float | None,
    reference_velocity_m_s: float | None,
    reference_temperature_k: float | None,
) -> tuple[float, float, float]:
    if (
        reference_density_kg_m3 is not None
        and reference_velocity_m_s is not None
        and reference_temperature_k is not None
    ):
        return reference_density_kg_m3, reference_velocity_m_s, reference_temperature_k

    path_parts = {part.lower() for part in csv_file.resolve().parts}
    if "poc2x2" in path_parts:
        default_density = POC2X2_REFERENCE_DENSITY_KG_M3
        default_velocity = POC2X2_REFERENCE_VELOCITY_M_S
        default_temperature = POC2X2_REFERENCE_TEMPERATURE_K
    else:
        default_density = LOCAL_RESULTS_REFERENCE_DENSITY_KG_M3
        default_velocity = LOCAL_RESULTS_REFERENCE_VELOCITY_M_S
        default_temperature = LOCAL_RESULTS_REFERENCE_TEMPERATURE_K

    return (
        default_density if reference_density_kg_m3 is None else reference_density_kg_m3,
        default_velocity if reference_velocity_m_s is None else reference_velocity_m_s,
        default_temperature if reference_temperature_k is None else reference_temperature_k,
    )


def render_bar_plot(
    values_by_observer: dict[int, float],
    plot: PlotDefinition,
    source_file: Path,
    output_dir: Path = PLOTS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    observers = sorted(values_by_observer)
    values = [values_by_observer[observer] for observer in observers]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(observer) for observer in observers], values, color="#4477aa")
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=9)
    ax.set_xlabel("Observer")
    ax.set_ylabel(plot.ylabel)
    ax.set_title(f"{plot.title} - {_case_label(source_file)}")
    ax.grid(axis="y", alpha=0.3)
    y_min = min(values)
    y_max = max(values)
    padding = 3.0 if y_min == y_max else 0.15 * (y_max - y_min)
    ax.set_ylim(y_min - padding, y_max + padding)
    fig.tight_layout()

    output_file = output_dir / f"{_case_label(source_file)}_{plot.filename}"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def render_comparison_plot(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), sharex=True)
    style_cycle = [
        {"color": "green", "marker": "o", "linestyle": "-"},
        {"color": "red", "marker": ">", "linestyle": "-"},
        {"color": "tab:blue", "marker": "s", "linestyle": "-"},
        {"color": "tab:orange", "marker": "D", "linestyle": "-"},
    ]

    for ax, plot in zip(axes, PLOTS):
        all_values = []
        for index, (dataset, values_by_plot) in enumerate(results_by_case):
            values_by_observer = values_by_plot[plot.name]
            values = [values_by_observer[observer] for observer in observers]
            all_values.extend(values)
            ax.plot(
                observers,
                values,
                linewidth=1.2,
                markersize=5,
                label=_legend_label(dataset.source_file),
                **style_cycle[index % len(style_cycle)],
            )

        y_min = min(all_values)
        y_max = max(all_values)
        padding = 0.2 * (y_max - y_min) if y_max > y_min else 3.0
        y_lower = y_min - max(padding, 8.0)
        y_upper = y_max + padding
        ax.set_ylim(y_lower, y_upper)

        angle_label_y = y_lower + 0.28 * (y_upper - y_lower)
        for observer in observers:
            angle = OBSERVER_ANGLES_DEG.get(observer)
            if angle is None:
                continue
            ax.vlines(observer, y_lower, angle_label_y - 0.02 * (y_upper - y_lower), colors="black", linewidth=1.2)
            ax.text(
                observer - 0.1,
                angle_label_y,
                f"{angle} deg",
                ha="left",
                va="bottom",
                fontsize=10,
            )

        ax.set_title(plot.title)
        ax.set_xlabel("Observer")
        ax.set_ylabel(plot.ylabel)
        ax.set_xlim(0, max(observers) + 1)
        ax.set_xticks(range(0, max(observers) + 2))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    output_file = output_dir / COMPARISON_OUTPUT_FILE
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def _frequency_tick_label(value: float, position: int) -> str:
    del position
    if value >= 1000.0:
        return f"{int(value)}"
    return f"{value:g}"


def render_spectrum_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    style_cycle = [
        {"color": "green", "linestyle": "-", "linewidth": 1.2},
        {"color": "red", "linestyle": "-", "linewidth": 1.2},
        {"color": "tab:blue", "linestyle": "-", "linewidth": 1.2},
        {"color": "tab:orange", "linestyle": "-", "linewidth": 1.2},
    ]

    for observer in observers:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for index, (dataset, _) in enumerate(results_by_case):
            pressure_pa = dataset.pressure_by_observer[observer] * dataset.pressure_scale_pa
            frequencies, spl_db = spl_spectrum_db(pressure_pa, dataset.sample_rate_hz)
            frequency_mask = (
                (frequencies >= SPECTRUM_MIN_FREQUENCY_HZ)
                & (frequencies <= SPECTRUM_MAX_FREQUENCY_HZ)
            )
            ax.plot(
                frequencies[frequency_mask],
                spl_db[frequency_mask],
                label=_legend_label(dataset.source_file),
                **style_cycle[index % len(style_cycle)],
            )

        ax.set_title(f"Spectra for observer ID: {observer}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("SPL [dB]")
        ax.set_xscale("log")
        ax.set_xlim(SPECTRUM_MIN_FREQUENCY_HZ, SPECTRUM_MAX_FREQUENCY_HZ)
        ax.set_xticks(SPECTRUM_X_TICKS_HZ)
        ax.xaxis.set_major_formatter(FuncFormatter(_frequency_tick_label))
        ax.set_ylim(*SPECTRUM_Y_LIMITS_DB)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()

        output_file = output_dir / f"spectrum_observer_{observer}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        output_files.append(output_file)
    return output_files


def render_third_octave_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    style_cycle = [
        {"color": "green", "marker": "o", "markerfacecolor": "none", "linestyle": "-", "linewidth": 1.2},
        {"color": "red", "marker": "s", "markerfacecolor": "none", "linestyle": "-", "linewidth": 1.2},
        {"color": "tab:blue", "marker": "D", "markerfacecolor": "none", "linestyle": "-", "linewidth": 1.2},
        {"color": "tab:orange", "marker": "^", "markerfacecolor": "none", "linestyle": "-", "linewidth": 1.2},
    ]

    for observer in observers:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for index, (dataset, _) in enumerate(results_by_case):
            pressure_pa = dataset.pressure_by_observer[observer] * dataset.pressure_scale_pa
            center_frequencies, band_spl_db = third_octave_spectrum_db(pressure_pa, dataset.sample_rate_hz)
            valid_band_mask = np.isfinite(band_spl_db)
            ax.plot(
                center_frequencies[valid_band_mask],
                band_spl_db[valid_band_mask],
                label=_legend_label(dataset.source_file),
                **style_cycle[index % len(style_cycle)],
            )

        ax.set_title(f"1/3 Octave Spectra for observer ID: {observer}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"SPL$_{1/3}$ [dB]")
        ax.set_xscale("log")
        ax.set_xlim(SPECTRUM_MIN_FREQUENCY_HZ, SPECTRUM_MAX_FREQUENCY_HZ)
        ax.set_xticks(SPECTRUM_X_TICKS_HZ)
        ax.xaxis.set_major_formatter(FuncFormatter(_frequency_tick_label))
        ax.set_ylim(*SPECTRUM_Y_LIMITS_DB)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()

        output_file = output_dir / f"third_octave_spectrum_observer_{observer}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        output_files.append(output_file)
    return output_files


def _bpf_frequency_hz(dataset: AcousticDataset, harmonic: int, blade_count: int = DEFAULT_BLADE_COUNT) -> float:
    return harmonic * blade_count * dataset.rpm / 60.0


def render_bpf_component_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
    blade_count: int = DEFAULT_BLADE_COUNT,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    component_definitions = (
        ("Thickness pressure", "thickness_by_observer"),
        ("Loading pressure", "loading_by_observer"),
        ("Total pressure", "pressure_by_observer"),
    )
    style_cycle = [
        {"color": "green", "marker": "^", "markerfacecolor": "green", "linestyle": "-", "linewidth": 1.2},
        {"color": "black", "marker": ">", "markerfacecolor": "black", "linestyle": "-", "linewidth": 1.2},
        {"color": "red", "marker": "s", "markerfacecolor": "none", "linestyle": "-", "linewidth": 1.2},
        {"color": "tab:blue", "marker": "o", "markerfacecolor": "none", "linestyle": "-", "linewidth": 1.2},
    ]

    for harmonic, harmonic_label in ((1, "first"), (2, "second")):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
        all_component_values = []
        actual_frequencies = []

        for ax, (component_title, attribute_name) in zip(axes, component_definitions):
            for index, (dataset, _) in enumerate(results_by_case):
                target_frequency = _bpf_frequency_hz(dataset, harmonic, blade_count)
                component_by_observer = getattr(dataset, attribute_name)
                values = []
                nearest_frequencies = []
                for observer in observers:
                    component_pressure = component_by_observer[observer] * dataset.pressure_scale_pa
                    nearest_frequency, spl_db = tone_spl_db(
                        component_pressure,
                        dataset.sample_rate_hz,
                        target_frequency,
                    )
                    values.append(spl_db)
                    nearest_frequencies.append(nearest_frequency)
                all_component_values.extend(values)
                actual_frequencies.append(float(np.mean(nearest_frequencies)))
                ax.plot(
                    values,
                    observers,
                    label=_legend_label(dataset.source_file),
                    **style_cycle[index % len(style_cycle)],
                )

            ax.set_title(component_title)
            ax.set_xlabel("SPL [dB]")
            ax.set_ylim(0, max(observers) + 1)
            ax.set_yticks(range(0, max(observers) + 2))
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Observer ID")
        axes[-1].legend(loc="lower left")
        if all_component_values:
            x_min = min(all_component_values)
            x_max = max(all_component_values)
            x_padding = max(4.0, 0.15 * (x_max - x_min))
            for ax in axes:
                ax.set_xlim(x_min - x_padding, x_max + x_padding)

        average_frequency = float(np.mean(actual_frequencies)) if actual_frequencies else 0.0
        fig.suptitle(
            f"Thickness and Loading Component for the {harmonic_label.title()} BPF "
            f"({average_frequency:.1f} Hz)",
            y=0.98,
        )
        fig.tight_layout()

        output_file = output_dir / f"thickness_loading_{harmonic_label}_bpf.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        output_files.append(output_file)

    return output_files


def render_harmonic_content_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
    blade_count: int = DEFAULT_BLADE_COUNT,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    style_cycle = [
        {"color": "green", "marker": "^", "markerfacecolor": "green"},
        {"color": "black", "marker": ">", "markerfacecolor": "black"},
        {"color": "red", "marker": "s", "markerfacecolor": "none"},
        {"color": "tab:blue", "marker": "o", "markerfacecolor": "none"},
    ]

    harmonics = np.asarray(HARMONIC_INDICES, dtype=int)
    for observer in observers:
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        for index, (dataset, _) in enumerate(results_by_case):
            bpf_frequency = _bpf_frequency_hz(dataset, 1, blade_count)
            harmonic_levels = []
            for harmonic in harmonics:
                target_frequency = float(harmonic * bpf_frequency)
                pressure_pa = dataset.pressure_by_observer[observer] * dataset.pressure_scale_pa
                _, spl_db = tone_spl_db(pressure_pa, dataset.sample_rate_hz, target_frequency)
                harmonic_levels.append(spl_db)

            style = style_cycle[index % len(style_cycle)]
            ax.vlines(
                harmonics,
                0.0,
                harmonic_levels,
                colors=style["color"],
                linewidth=4.0,
                alpha=0.95,
            )
            ax.plot(
                harmonics,
                harmonic_levels,
                linestyle="none",
                markersize=8,
                label=_legend_label(dataset.source_file),
                **style,
            )

        ax.set_title(f"Harmonic content for observer ID: {observer}")
        ax.set_xlabel("Harmonics")
        ax.set_ylabel("SPL [dB]")
        ax.set_xlim(0, max(HARMONIC_INDICES) + 1)
        ax.set_xticks(range(0, max(HARMONIC_INDICES) + 2))
        ax.set_ylim(*HARMONIC_Y_LIMITS_DB)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()

        output_file = output_dir / f"harmonic_content_observer_{observer}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        output_files.append(output_file)

    return output_files


def make_plots(
    csv_file: Path | None = None,
    observers: tuple[int, ...] = DEFAULT_OBSERVERS,
    plots: tuple[PlotDefinition, ...] = PLOTS,
    output_dir: Path = PLOTS_DIR,
    rpm: float | None = None,
    time_steps_per_revolution: float | None = None,
    reference_density_kg_m3: float | None = None,
    reference_velocity_m_s: float | None = None,
    reference_temperature_k: float | None = None,
) -> list[Path]:
    resolved_csv_file = _default_acoustic_files()[-1] if csv_file is None else Path(csv_file)
    rpm, time_steps_per_revolution = _timing_for_file(
        resolved_csv_file,
        rpm,
        time_steps_per_revolution,
    )
    reference_density_kg_m3, reference_velocity_m_s, reference_temperature_k = _reference_values_for_file(
        resolved_csv_file,
        reference_density_kg_m3,
        reference_velocity_m_s,
        reference_temperature_k,
    )
    dataset = load_acoustic_dataset(
        resolved_csv_file,
        observers,
        rpm=rpm,
        time_steps_per_revolution=time_steps_per_revolution,
        reference_density_kg_m3=reference_density_kg_m3,
        reference_velocity_m_s=reference_velocity_m_s,
        reference_temperature_k=reference_temperature_k,
    )
    print(
        f"Loaded {dataset.source_file.name}: "
        f"{len(dataset.time)} samples, "
        f"rpm={dataset.rpm:.3f}, "
        f"steps/rev={dataset.time_steps_per_revolution:g}, "
        f"sample_rate={dataset.sample_rate_hz:.3f} Hz, "
        f"rho={dataset.reference_density_kg_m3:.6f} kg/m^3, "
        f"T={dataset.reference_temperature_k:.3f} K, "
        f"U_ref={dataset.reference_velocity_m_s:.6f} m/s, "
        f"pressure_scale=gamma*p_inf={dataset.pressure_scale_pa:.3f} Pa"
    )

    output_files = []
    values_by_plot = {}
    for plot in plots:
        values_by_observer = evaluate_plot(dataset, plot)
        values_by_plot[plot.name] = values_by_observer
        output_file = render_bar_plot(values_by_observer, plot, dataset.source_file, output_dir)
        output_files.append(output_file)
        print(f"Wrote {output_file}")
    if all(plot.name in values_by_plot for plot in PLOTS):
        print_oaspl_values(dataset, values_by_plot, observers)
    return output_files


def make_comparison_plot(
    csv_files: list[Path] | None = None,
    observers: tuple[int, ...] = DEFAULT_OBSERVERS,
    plots: tuple[PlotDefinition, ...] = PLOTS,
    output_dir: Path = PLOTS_DIR,
) -> Path:
    csv_files = _default_acoustic_files() if csv_files is None else [Path(path) for path in csv_files]
    results_by_case = []
    for csv_file in csv_files:
        rpm, time_steps_per_revolution = _timing_for_file(csv_file, None, None)
        reference_density_kg_m3, reference_velocity_m_s, reference_temperature_k = _reference_values_for_file(
            csv_file,
            None,
            None,
            None,
        )
        dataset = load_acoustic_dataset(
            csv_file,
            observers,
            rpm=rpm,
            time_steps_per_revolution=time_steps_per_revolution,
            reference_density_kg_m3=reference_density_kg_m3,
            reference_velocity_m_s=reference_velocity_m_s,
            reference_temperature_k=reference_temperature_k,
        )
        values_by_plot = {plot.name: evaluate_plot(dataset, plot) for plot in plots}
        results_by_case.append((dataset, values_by_plot))
        print(
            f"Loaded {dataset.source_file.name}: "
            f"{len(dataset.time)} samples, "
            f"rpm={dataset.rpm:.3f}, "
            f"steps/rev={dataset.time_steps_per_revolution:g}, "
            f"sample_rate={dataset.sample_rate_hz:.3f} Hz, "
            f"pressure_scale=gamma*p_inf={dataset.pressure_scale_pa:.3f} Pa"
        )
        if all(plot.name in values_by_plot for plot in PLOTS):
            print_oaspl_values(dataset, values_by_plot, observers)

    output_file = render_comparison_plot(results_by_case, observers, output_dir)
    print(f"Wrote {output_file}")
    for spectrum_output_file in render_spectrum_plots(results_by_case, observers, output_dir):
        print(f"Wrote {spectrum_output_file}")
    for third_octave_output_file in render_third_octave_plots(results_by_case, observers, output_dir):
        print(f"Wrote {third_octave_output_file}")
    for bpf_output_file in render_bpf_component_plots(results_by_case, observers, output_dir):
        print(f"Wrote {bpf_output_file}")
    for harmonic_output_file in render_harmonic_content_plots(results_by_case, observers, output_dir):
        print(f"Wrote {harmonic_output_file}")
    return output_file


def _parse_observers(value: str) -> tuple[int, ...]:
    observers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not observers:
        raise argparse.ArgumentTypeError("Provide at least one observer index.")
    return observers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Flow360 aeroacoustic OASPL results.")
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=None,
        help="Path to *_results_total_acoustics_v*.csv. Can be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--observers", type=_parse_observers, default=DEFAULT_OBSERVERS)
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--steps-per-rev", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None, help="Density for dimensionalizing acoustic pressure.")
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=None,
        help="Static temperature for dimensionalizing acoustic pressure with gamma*p_inf.",
    )
    parser.add_argument("--u-ref", type=float, default=None, help="Reference velocity retained for reporting.")
    parser.add_argument(
        "--csv-observer-offset",
        type=int,
        default=FLOW360_OBSERVER_NUMBERING_OFFSET,
        help="Offset from user observer number to CSV observer index. Default maps observer 1 to observer_0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global FLOW360_OBSERVER_NUMBERING_OFFSET
    FLOW360_OBSERVER_NUMBERING_OFFSET = args.csv_observer_offset
    if (
        args.rpm is not None
        or args.steps_per_rev is not None
        or args.rho is not None
        or args.temperature_k is not None
        or args.u_ref is not None
    ):
        if args.csv is not None and len(args.csv) > 1:
            raise ValueError(
                "Manual --rpm/--steps-per-rev/--rho/--temperature-k/--u-ref overrides "
                "are only supported for one CSV."
            )
        make_plots(
            csv_file=args.csv[0] if args.csv else None,
            observers=args.observers,
            output_dir=args.output_dir,
            rpm=args.rpm,
            time_steps_per_revolution=args.steps_per_rev,
            reference_density_kg_m3=args.rho,
            reference_velocity_m_s=args.u_ref,
            reference_temperature_k=args.temperature_k,
        )
    else:
        make_comparison_plot(
            csv_files=args.csv,
            observers=args.observers,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
