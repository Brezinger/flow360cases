from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator
import numpy as np
from scipy.signal import spectrogram, welch


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
LOCAL_RESULTS_L_GRID_UNIT_M = 0.001
POC2X2_REFERENCE_DENSITY_KG_M3 = 1.149
POC2X2_REFERENCE_VELOCITY_M_S = 146.857683
POC2X2_REFERENCE_TEMPERATURE_K = 283.275
POC2X2_L_GRID_UNIT_M = 1.0
CASE_TIMING_BY_PREFIX = {
    "step3": (DEFAULT_RPM, STEP3_TIME_STEPS_PER_REVOLUTION),
    "step4": (DEFAULT_RPM, STEP4_TIME_STEPS_PER_REVOLUTION),
}
COMPARISON_OUTPUT_FILE = "oaspl_observers_1_5_comparison.png"
OASPL_MIN_FREQUENCY_HZ = 40.0
OASPL_MAX_FREQUENCY_HZ = 10_000.0
SPECTRUM_MIN_FREQUENCY_HZ = 40.0
SPECTRUM_MAX_FREQUENCY_HZ = 10_000.0
SPECTRUM_X_TICKS_HZ = (40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000)
SPECTRUM_Y_LIMITS_DB = (-20.0, 70.0)
SPECTROGRAM_MAX_FREQUENCY_HZ = 5_000.0
SPECTROGRAM_LIMITS_DB = (0.0, 120.0)
SPECTROGRAM_NPERSEG = 512
HARMONIC_INDICES = tuple(range(1, 11))
HARMONIC_Y_LIMITS_DB = (0.0, 110.0)
APPLY_A_WEIGHTING_TO_SPECTRA = True
LEGEND_LABEL_OVERRIDE: str | None = None
NPERSEG = "All"
WINDOW = "hann"


def _default_line_colors() -> list[str]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _is_local_step4(source_file: Path) -> bool:
    path_parts = {part.lower() for part in source_file.resolve().parts}
    return _step_label(source_file) == "step4" and "poc2x2" not in path_parts


def _line_color(source_file: Path, index: int) -> str:
    colors = _default_line_colors()
    if _is_local_step4(source_file):
        return colors[0]
    return colors[(index + 1) % len(colors)]


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
    l_grid_unit_m: float

    @property
    def sample_spacing_s(self) -> float:
        if len(self.time) < 2:
            raise ValueError("At least two time samples are required to compute the acoustic sample spacing.")
        return float(np.mean(np.diff(self.time * (self.l_grid_unit_m / self.speed_of_sound_m_s))))

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.sample_spacing_s

    @property
    def time_seconds(self) -> np.ndarray:
        return self.time * (self.l_grid_unit_m / self.speed_of_sound_m_s)

    @property
    def pressure_scale_pa(self) -> float:
        return self.reference_density_kg_m3 * self.speed_of_sound_m_s**2

    @property
    def speed_of_sound_m_s(self) -> float:
        return float(
            np.sqrt(
                AIR_SPECIFIC_HEAT_RATIO
                * AIR_GAS_CONSTANT_J_KG_K
                * self.reference_temperature_k
            )
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
    l_grid_unit_m: float = LOCAL_RESULTS_L_GRID_UNIT_M,
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
        l_grid_unit_m=l_grid_unit_m,
    )


def _fluctuating_pressure(pressure: np.ndarray) -> np.ndarray:
    return pressure - np.mean(pressure)


def _filter_zero_signal_edges(pressure: np.ndarray) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=float)
    nonzero_indices = np.where(pressure != 0.0)[0]
    if nonzero_indices.size == 0:
        return pressure
    return pressure[nonzero_indices[0]:nonzero_indices[-1] + 1]


def _db_from_power(mean_square_pressure: np.ndarray | float) -> np.ndarray | float:
    mean_square_pressure = np.asarray(mean_square_pressure, dtype=float)
    spl_db = 10.0 * np.log10(
        np.maximum(mean_square_pressure, np.finfo(float).tiny)
        / ACOUSTIC_PRESSURE_REFERENCE_PA**2
    )
    if spl_db.ndim == 0:
        return float(spl_db)
    return spl_db


def _nperseg(sample_count: int) -> int:
    if isinstance(NPERSEG, str) and NPERSEG.strip().lower() == "all":
        return sample_count
    return min(int(NPERSEG), sample_count)


def oaspl_unweighted_db(pressure: np.ndarray, sample_rate_hz: float) -> float:
    frequencies, mean_square_by_bin = _band_limited_mean_square_spectrum(pressure, sample_rate_hz)
    del frequencies
    mean_square_pressure = float(np.sum(mean_square_by_bin))
    return float(_db_from_power(mean_square_pressure))


def _a_weighting_db(frequency_hz: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=float)
    frequency_squared = np.square(frequency)
    ra = (12200.0**2 * frequency_squared**2) / (
        (frequency_squared + 20.6**2)
        * (frequency_squared + 12200.0**2)
        * np.sqrt((frequency_squared + 107.7**2) * (frequency_squared + 737.9**2))
    )
    ra = np.maximum(ra, 1.0e-12)
    ra_1khz = (12200.0**2 * (1000.0**2) ** 2) / (
        (1000.0**2 + 20.6**2)
        * (1000.0**2 + 12200.0**2)
        * np.sqrt((1000.0**2 + 107.7**2) * (1000.0**2 + 737.9**2))
    )
    weighting = 20.0 * np.log10(ra / ra_1khz)
    return np.where(frequency > 0.0, weighting, -np.inf)


def _weighting_db(frequency_hz: np.ndarray, apply_a_weighting: bool) -> np.ndarray:
    if apply_a_weighting:
        return _a_weighting_db(frequency_hz)
    return np.zeros_like(np.asarray(frequency_hz, dtype=float))


def _welch_mean_square_spectrum(pressure: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    pressure = _filter_zero_signal_edges(pressure)
    sample_count = len(pressure)
    if sample_count < 8:
        raise ValueError("Too few samples for Welch calculation.")

    nperseg = _nperseg(sample_count)
    frequencies, mean_square_by_bin = welch(
        _fluctuating_pressure(pressure),
        fs=sample_rate_hz,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="spectrum",
    )
    return frequencies, mean_square_by_bin


def _band_limited_mean_square_spectrum(pressure: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    frequencies, mean_square_by_bin = _welch_mean_square_spectrum(pressure, sample_rate_hz)
    frequency_mask = (
        (frequencies >= OASPL_MIN_FREQUENCY_HZ)
        & (frequencies <= OASPL_MAX_FREQUENCY_HZ)
    )
    if not np.any(frequency_mask):
        raise ValueError("No frequencies remain inside the configured frequency band.")
    return frequencies[frequency_mask], mean_square_by_bin[frequency_mask]


def oaspl_a_weighted_db(pressure: np.ndarray, sample_rate_hz: float) -> float:
    frequencies, mean_square_by_bin = _band_limited_mean_square_spectrum(pressure, sample_rate_hz)
    linear_a_weight = 10.0 ** (_weighting_db(frequencies, apply_a_weighting=True) / 10.0)
    weighted_mean_square_pressure = float(np.sum(mean_square_by_bin * linear_a_weight))
    return float(_db_from_power(weighted_mean_square_pressure))


def spl_spectrum_db(
    pressure: np.ndarray,
    sample_rate_hz: float,
    apply_a_weighting: bool = APPLY_A_WEIGHTING_TO_SPECTRA,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies, mean_square_by_bin = _band_limited_mean_square_spectrum(pressure, sample_rate_hz)
    linear_weight = 10.0 ** (_weighting_db(frequencies, apply_a_weighting) / 10.0)
    return frequencies, _db_from_power(mean_square_by_bin * linear_weight)


def tone_spl_db(pressure: np.ndarray, sample_rate_hz: float, target_frequency_hz: float) -> tuple[float, float]:
    frequencies, mean_square_by_bin = _welch_mean_square_spectrum(pressure, sample_rate_hz)
    valid_frequency_mask = frequencies > 0.0
    frequencies = frequencies[valid_frequency_mask]
    mean_square_by_bin = mean_square_by_bin[valid_frequency_mask]
    nearest_index = int(np.argmin(np.abs(frequencies - target_frequency_hz)))
    spl_db = _db_from_power(mean_square_by_bin[nearest_index])
    return float(frequencies[nearest_index]), float(spl_db)


def harmonic_third_octave_spl_db(
    pressure: np.ndarray,
    sample_rate_hz: float,
    target_frequency_hz: float,
) -> float:
    frequencies, mean_square_by_bin = _welch_mean_square_spectrum(pressure, sample_rate_hz)
    band_factor = 2.0 ** (1.0 / 6.0)
    band_mask = (
        (frequencies >= target_frequency_hz / band_factor)
        & (frequencies < target_frequency_hz * band_factor)
    )
    return float(_db_from_power(np.sum(mean_square_by_bin[band_mask])))


def _third_octave_centers(fmin_hz: float, fmax_hz: float) -> np.ndarray:
    centers = []
    k_min = int(np.floor(3.0 * np.log2(max(fmin_hz, 1.0e-12) / 1000.0)))
    k_max = int(np.ceil(3.0 * np.log2(fmax_hz / 1000.0)))
    for k in range(k_min, k_max + 1):
        center_frequency = 1000.0 * (2.0 ** (k / 3.0))
        if center_frequency >= fmin_hz:
            centers.append(center_frequency)
    return np.asarray(centers, dtype=float)


def _third_octave_bands(fmin_hz: float, fmax_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = _third_octave_centers(fmin_hz, fmax_hz)
    factor = 2.0 ** (1.0 / 6.0)
    return centers, centers / factor, centers * factor


def third_octave_spectrum_db(
    pressure: np.ndarray,
    sample_rate_hz: float,
    apply_a_weighting: bool = APPLY_A_WEIGHTING_TO_SPECTRA,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies, mean_square_by_bin = _band_limited_mean_square_spectrum(pressure, sample_rate_hz)
    linear_weight = 10.0 ** (_weighting_db(frequencies, apply_a_weighting) / 10.0)
    center_frequencies, lower_band_edges, upper_band_edges = _third_octave_bands(
        OASPL_MIN_FREQUENCY_HZ,
        OASPL_MAX_FREQUENCY_HZ,
    )
    mean_square_by_band = []

    for lower_edge, upper_edge in zip(lower_band_edges, upper_band_edges):
        band_mask = (frequencies >= lower_edge) & (frequencies < upper_edge)
        mean_square_by_band.append(float(np.sum(mean_square_by_bin[band_mask] * linear_weight[band_mask])))

    return center_frequencies, _db_from_power(np.asarray(mean_square_by_band, dtype=float))


PLOTS = (
    PlotDefinition(
        name="oaspl_unweighted",
        filename="oaspl_unweighted_observers_1_5.png",
        evaluator=oaspl_unweighted_db,
        ylabel="OASPL [dB]",
        title="Unweighted OASPL",
    ),
    PlotDefinition(
        name="oaspl_a_weighted",
        filename="oaspl_a_weighted_observers_1_5.png",
        evaluator=oaspl_a_weighted_db,
        ylabel="A-weighted OASPL [dBA]",
        title="A-weighted OASPL",
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
    if LEGEND_LABEL_OVERRIDE is not None:
        return LEGEND_LABEL_OVERRIDE
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
    l_grid_unit_m: float | None,
) -> tuple[float, float, float, float]:
    if (
        reference_density_kg_m3 is not None
        and reference_velocity_m_s is not None
        and reference_temperature_k is not None
        and l_grid_unit_m is not None
    ):
        return reference_density_kg_m3, reference_velocity_m_s, reference_temperature_k, l_grid_unit_m

    path_parts = {part.lower() for part in csv_file.resolve().parts}
    if "poc2x2" in path_parts:
        default_density = POC2X2_REFERENCE_DENSITY_KG_M3
        default_velocity = POC2X2_REFERENCE_VELOCITY_M_S
        default_temperature = POC2X2_REFERENCE_TEMPERATURE_K
        default_l_grid_unit_m = POC2X2_L_GRID_UNIT_M
    else:
        default_density = LOCAL_RESULTS_REFERENCE_DENSITY_KG_M3
        default_velocity = LOCAL_RESULTS_REFERENCE_VELOCITY_M_S
        default_temperature = LOCAL_RESULTS_REFERENCE_TEMPERATURE_K
        default_l_grid_unit_m = LOCAL_RESULTS_L_GRID_UNIT_M

    return (
        default_density if reference_density_kg_m3 is None else reference_density_kg_m3,
        default_velocity if reference_velocity_m_s is None else reference_velocity_m_s,
        default_temperature if reference_temperature_k is None else reference_temperature_k,
        default_l_grid_unit_m if l_grid_unit_m is None else l_grid_unit_m,
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
    markers = ("o", ">", "s", "D")

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
                color=_line_color(dataset.source_file, index),
                marker=markers[index % len(markers)],
                linestyle="-",
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
    if value == 0:
        return "0"
    return f"{value:.0f}"


def _weighting_filename_part() -> str:
    return "_a_weighted" if APPLY_A_WEIGHTING_TO_SPECTRA else ""


def _style_frequency_axis(ax) -> None:
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle="--", alpha=0.35)

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 4, 6, 8), numticks=5))
    ax.xaxis.set_major_formatter(FuncFormatter(_frequency_tick_label))
    ax.xaxis.set_minor_formatter(FuncFormatter(_frequency_tick_label))

    ax.tick_params(axis="x", which="major", labelsize=10)
    ax.tick_params(axis="x", which="minor", labelsize=8)
    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.tick_params(axis="y", which="minor", labelsize=8)

    ax.yaxis.set_major_formatter(FuncFormatter(_frequency_tick_label))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_formatter(FuncFormatter(_frequency_tick_label))


def render_spectrum_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

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
                color=_line_color(dataset.source_file, index),
                linestyle="-",
                linewidth=1.2,
            )

        ax.set_title(f"Spectra for observer ID: {observer}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("SPL [dBA]" if APPLY_A_WEIGHTING_TO_SPECTRA else "SPL [dB]")
        ax.set_xscale("log")
        ax.set_xlim(SPECTRUM_MIN_FREQUENCY_HZ, SPECTRUM_MAX_FREQUENCY_HZ)
        ax.set_ylim(*SPECTRUM_Y_LIMITS_DB)
        _style_frequency_axis(ax)
        ax.legend(loc="upper right", fontsize=12)
        fig.tight_layout()

        output_file = output_dir / f"spectrum{_weighting_filename_part()}_observer_{observer}.png"
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
    markers = ("o", "s", "D", "^")

    for observer in observers:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        upper_frequency_limit = SPECTRUM_MAX_FREQUENCY_HZ
        for index, (dataset, _) in enumerate(results_by_case):
            pressure_pa = dataset.pressure_by_observer[observer] * dataset.pressure_scale_pa
            center_frequencies, band_spl_db = third_octave_spectrum_db(pressure_pa, dataset.sample_rate_hz)
            upper_frequency_limit = max(upper_frequency_limit, float(np.max(center_frequencies)))
            valid_band_mask = np.isfinite(band_spl_db)
            ax.plot(
                center_frequencies[valid_band_mask],
                band_spl_db[valid_band_mask],
                label=_legend_label(dataset.source_file),
                color=_line_color(dataset.source_file, index),
                marker=markers[index % len(markers)],
                markerfacecolor="none",
                linestyle="-",
                linewidth=1.2,
            )

        ax.set_title(f"1/3 Octave Spectra for observer ID: {observer}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"SPL$_{1/3}$ [dBA]" if APPLY_A_WEIGHTING_TO_SPECTRA else r"SPL$_{1/3}$ [dB]")
        ax.set_xscale("log")
        ax.set_xlim(SPECTRUM_MIN_FREQUENCY_HZ, upper_frequency_limit)
        ax.set_ylim(*SPECTRUM_Y_LIMITS_DB)
        _style_frequency_axis(ax)
        ax.legend(loc="upper right", fontsize=12)
        fig.tight_layout()

        output_file = output_dir / f"third_octave_spectrum{_weighting_filename_part()}_observer_{observer}.png"
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
    markers = ("^", ">", "s", "o")

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
                    color=_line_color(dataset.source_file, index),
                    marker=markers[index % len(markers)],
                    markerfacecolor=_line_color(dataset.source_file, index),
                    linestyle="-",
                    linewidth=1.2,
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
    markers = ("^", ">", "s", "o")

    harmonics = np.asarray(HARMONIC_INDICES, dtype=int)
    for observer in observers:
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        for index, (dataset, _) in enumerate(results_by_case):
            bpf_frequency = _bpf_frequency_hz(dataset, 1, blade_count)
            harmonic_levels = []
            for harmonic in harmonics:
                target_frequency = float(harmonic * bpf_frequency)
                pressure_pa = dataset.pressure_by_observer[observer] * dataset.pressure_scale_pa
                spl_db = harmonic_third_octave_spl_db(
                    pressure_pa,
                    dataset.sample_rate_hz,
                    target_frequency,
                )
                harmonic_levels.append(spl_db)

            color = _line_color(dataset.source_file, index)
            ax.vlines(
                harmonics,
                0.0,
                harmonic_levels,
                colors=color,
                linewidth=4.0,
                alpha=0.95,
            )
            ax.plot(
                harmonics,
                harmonic_levels,
                linestyle="none",
                markersize=8,
                label=_legend_label(dataset.source_file),
                color=color,
                marker=markers[index % len(markers)],
                markerfacecolor=color,
            )

        ax.set_title(f"Harmonic content for observer ID: {observer}")
        ax.set_xlabel("Harmonics")
        ax.set_ylabel(r"SPL$_{1/3}$ [dB]")
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


def render_pressure_time_history_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    for observer in observers:
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        all_pressure_values = []

        for index, (dataset, _) in enumerate(results_by_case):
            time_seconds = dataset.time_seconds
            time_span = time_seconds[-1] - time_seconds[0]
            if time_span <= 0.0:
                raise ValueError(f"Non-positive acoustic time span in {dataset.source_file}")

            time_fraction = (time_seconds - time_seconds[0]) / time_span
            pressure_pa = dataset.pressure_by_observer[observer] * dataset.pressure_scale_pa
            all_pressure_values.extend(pressure_pa)
            ax.plot(
                time_fraction,
                pressure_pa,
                label=_legend_label(dataset.source_file),
                color=_line_color(dataset.source_file, index),
                linestyle="-",
                linewidth=1.2,
            )

        ax.set_title(f"Pressure-time History for observer ID: {observer}")
        ax.set_xlabel("Time Fraction")
        ax.set_ylabel("Pressure [Pa]")
        ax.set_xlim(0.0, 1.0)
        if all_pressure_values:
            pressure_min = min(all_pressure_values)
            pressure_max = max(all_pressure_values)
            padding = 0.15 * (pressure_max - pressure_min) if pressure_max > pressure_min else 0.01
            ax.set_ylim(pressure_min - padding, pressure_max + padding)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()

        output_file = output_dir / f"pressure_time_history_observer_{observer}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        output_files.append(output_file)

    return output_files


def _average_revolutions(signal: np.ndarray, revolution_count: int = 3) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    samples_per_revolution = len(signal) // revolution_count
    if samples_per_revolution < 2:
        raise ValueError(
            f"Need at least {2 * revolution_count} samples to average {revolution_count} revolutions."
        )
    trimmed_sample_count = samples_per_revolution * revolution_count
    revolutions = signal[:trimmed_sample_count].reshape(revolution_count, samples_per_revolution)
    return np.mean(revolutions, axis=0)


def render_average_pressure_time_history_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
    revolution_count: int = 3,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    for observer in observers:
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        all_pressure_values = []

        for index, (dataset, _) in enumerate(results_by_case):
            pressure_pa = dataset.pressure_by_observer[observer] * dataset.pressure_scale_pa
            averaged_pressure_pa = _average_revolutions(
                pressure_pa,
                revolution_count=revolution_count,
            )
            all_pressure_values.extend(averaged_pressure_pa)
            time_fraction = np.linspace(0.0, 1.0, len(averaged_pressure_pa))

            ax.plot(
                time_fraction,
                averaged_pressure_pa,
                label=_legend_label(dataset.source_file),
                color=_line_color(dataset.source_file, index),
                linestyle="-",
                linewidth=1.2,
            )

        ax.set_title(f"Average Pressure-time History for observer ID: {observer}")
        ax.set_xlabel("Time Fraction")
        ax.set_ylabel("Pressure [Pa]")
        ax.set_xlim(0.0, 1.0)
        if all_pressure_values:
            pressure_min = min(all_pressure_values)
            pressure_max = max(all_pressure_values)
            padding = 0.15 * (pressure_max - pressure_min) if pressure_max > pressure_min else 0.01
            ax.set_ylim(pressure_min - padding, pressure_max + padding)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()

        output_file = output_dir / f"average_pressure_time_history_observer_{observer}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        output_files.append(output_file)

    return output_files


def _spectrogram_spl_db(
    pressure_pa: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pressure_pa = _filter_zero_signal_edges(pressure_pa)
    nperseg = min(SPECTROGRAM_NPERSEG, len(pressure_pa))
    if nperseg < 8:
        raise ValueError("Too few samples for spectrogram calculation.")

    frequencies, times, power_by_time = spectrogram(
        _fluctuating_pressure(pressure_pa),
        fs=sample_rate_hz,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="spectrum",
        mode="psd",
    )
    spl_db = _db_from_power(power_by_time)
    return frequencies, times, spl_db


def render_spectrogram_plots(
    results_by_case: list[tuple[AcousticDataset, dict[str, dict[int, float]]]],
    observers: tuple[int, ...],
    output_dir: Path = PLOTS_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    component_definitions = (
        ("PRESSURE", "pressure_by_observer", "total_pressure"),
        ("LOADING", "loading_by_observer", "loading_pressure"),
    )

    for dataset, _ in results_by_case:
        case_label = _case_label(dataset.source_file)
        for observer in observers:
            for component_title, attribute_name, filename_part in component_definitions:
                component_by_observer = getattr(dataset, attribute_name)
                pressure_pa = component_by_observer[observer] * dataset.pressure_scale_pa
                frequencies, times, spl_db = _spectrogram_spl_db(
                    pressure_pa,
                    dataset.sample_rate_hz,
                )

                frequency_mask = frequencies <= SPECTROGRAM_MAX_FREQUENCY_HZ
                frequencies = frequencies[frequency_mask]
                spl_db = spl_db[frequency_mask, :]
                time_fraction = times / times[-1] if times[-1] > 0.0 else times

                fig, ax = plt.subplots(figsize=(10.5, 6.0))
                mesh = ax.pcolormesh(
                    time_fraction,
                    frequencies,
                    spl_db,
                    shading="nearest",
                    cmap="jet",
                    vmin=SPECTROGRAM_LIMITS_DB[0],
                    vmax=SPECTROGRAM_LIMITS_DB[1],
                )
                colorbar = fig.colorbar(mesh, ax=ax, pad=0.10)
                colorbar.set_label("SPL [dB]")

                ax.set_title(f"{component_title} - Spectrogram for Observer ID: {observer}")
                ax.set_xlabel("Time Fraction")
                ax.set_ylabel("Frequency [Hz]")
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, SPECTROGRAM_MAX_FREQUENCY_HZ)
                ax.grid(True, color="white", linewidth=0.5, alpha=0.5)
                fig.tight_layout()

                output_file = output_dir / (
                    f"spectrogram_{filename_part}_{case_label}_observer_{observer}.png"
                )
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
    l_grid_unit_m: float | None = None,
) -> list[Path]:
    resolved_csv_file = _default_acoustic_files()[-1] if csv_file is None else Path(csv_file)
    rpm, time_steps_per_revolution = _timing_for_file(
        resolved_csv_file,
        rpm,
        time_steps_per_revolution,
    )
    reference_density_kg_m3, reference_velocity_m_s, reference_temperature_k, l_grid_unit_m = _reference_values_for_file(
        resolved_csv_file,
        reference_density_kg_m3,
        reference_velocity_m_s,
        reference_temperature_k,
        l_grid_unit_m,
    )
    dataset = load_acoustic_dataset(
        resolved_csv_file,
        observers,
        rpm=rpm,
        time_steps_per_revolution=time_steps_per_revolution,
        reference_density_kg_m3=reference_density_kg_m3,
        reference_velocity_m_s=reference_velocity_m_s,
        reference_temperature_k=reference_temperature_k,
        l_grid_unit_m=l_grid_unit_m,
    )
    print(
        f"Loaded {dataset.source_file.name}: "
        f"{len(dataset.time)} samples, "
        f"rpm={dataset.rpm:.3f}, "
        f"steps/rev={dataset.time_steps_per_revolution:g}, "
        f"sample_rate={dataset.sample_rate_hz:.3f} Hz, "
        f"rho={dataset.reference_density_kg_m3:.6f} kg/m^3, "
        f"T={dataset.reference_temperature_k:.3f} K, "
        f"a={dataset.speed_of_sound_m_s:.6f} m/s, "
        f"L_grid_unit={dataset.l_grid_unit_m:g} m, "
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
    for pressure_time_output_file in render_pressure_time_history_plots(
        [(dataset, values_by_plot)],
        observers,
        output_dir,
    ):
        output_files.append(pressure_time_output_file)
        print(f"Wrote {pressure_time_output_file}")
    for average_pressure_time_output_file in render_average_pressure_time_history_plots(
        [(dataset, values_by_plot)],
        observers,
        output_dir,
    ):
        output_files.append(average_pressure_time_output_file)
        print(f"Wrote {average_pressure_time_output_file}")
    for spectrogram_output_file in render_spectrogram_plots(
        [(dataset, values_by_plot)],
        observers,
        output_dir,
    ):
        output_files.append(spectrogram_output_file)
        print(f"Wrote {spectrogram_output_file}")
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
        reference_density_kg_m3, reference_velocity_m_s, reference_temperature_k, l_grid_unit_m = _reference_values_for_file(
            csv_file,
            None,
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
            l_grid_unit_m=l_grid_unit_m,
        )
        values_by_plot = {plot.name: evaluate_plot(dataset, plot) for plot in plots}
        results_by_case.append((dataset, values_by_plot))
        print(
            f"Loaded {dataset.source_file.name}: "
            f"{len(dataset.time)} samples, "
            f"rpm={dataset.rpm:.3f}, "
            f"steps/rev={dataset.time_steps_per_revolution:g}, "
            f"sample_rate={dataset.sample_rate_hz:.3f} Hz, "
            f"a={dataset.speed_of_sound_m_s:.6f} m/s, "
            f"L_grid_unit={dataset.l_grid_unit_m:g} m, "
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
    for pressure_time_output_file in render_pressure_time_history_plots(results_by_case, observers, output_dir):
        print(f"Wrote {pressure_time_output_file}")
    for average_pressure_time_output_file in render_average_pressure_time_history_plots(
        results_by_case,
        observers,
        output_dir,
    ):
        print(f"Wrote {average_pressure_time_output_file}")
    for spectrogram_output_file in render_spectrogram_plots(results_by_case, observers, output_dir):
        print(f"Wrote {spectrogram_output_file}")
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
        "--l-grid-unit",
        type=float,
        default=None,
        help="Flow360 grid-unit length in meters for dimensionalizing acoustic CSV time.",
    )
    parser.add_argument(
        "--csv-observer-offset",
        type=int,
        default=FLOW360_OBSERVER_NUMBERING_OFFSET,
        help="Offset from user observer number to CSV observer index. Default maps observer 1 to observer_0.",
    )
    parser.add_argument(
        "--legend-label",
        type=str,
        default=None,
        help="Override the plotted case label in legends.",
    )
    weighting_group = parser.add_mutually_exclusive_group()
    weighting_group.add_argument(
        "--a-weighting",
        dest="apply_a_weighting_to_spectra",
        action="store_true",
        default=APPLY_A_WEIGHTING_TO_SPECTRA,
        help="Use A-weighting for spectrum and third-octave plots.",
    )
    weighting_group.add_argument(
        "--no-a-weighting",
        dest="apply_a_weighting_to_spectra",
        action="store_false",
        help="Use unweighted spectrum and third-octave plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global APPLY_A_WEIGHTING_TO_SPECTRA, FLOW360_OBSERVER_NUMBERING_OFFSET, LEGEND_LABEL_OVERRIDE
    APPLY_A_WEIGHTING_TO_SPECTRA = args.apply_a_weighting_to_spectra
    FLOW360_OBSERVER_NUMBERING_OFFSET = args.csv_observer_offset
    LEGEND_LABEL_OVERRIDE = args.legend_label
    if (
        args.rpm is not None
        or args.steps_per_rev is not None
        or args.rho is not None
        or args.temperature_k is not None
        or args.u_ref is not None
        or args.l_grid_unit is not None
    ):
        if args.csv is not None and len(args.csv) > 1:
            raise ValueError(
                "Manual --rpm/--steps-per-rev/--rho/--temperature-k/--u-ref/--l-grid-unit overrides "
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
            l_grid_unit_m=args.l_grid_unit,
        )
    else:
        make_comparison_plot(
            csv_files=args.csv,
            observers=args.observers,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
