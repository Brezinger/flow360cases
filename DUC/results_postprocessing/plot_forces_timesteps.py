from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "Pusher" / "forces"
STEP_CONFIG_OVERRIDES = {
    "step0": {"rpm": 2186.0, "time_steps_per_revolution": 54},
    "step1": {"rpm": 2186.0, "time_steps_per_revolution": 329},
    "step2": {"rpm": 2186.0, "time_steps_per_revolution": 878},
    "step3": {"rpm": 2186.0, "time_steps_per_revolution": 1756},
    "step4": {"rpm": 2186.0, "time_steps_per_revolution": 3660},
}

# Previous POC2 configuration:
# RESULTS_DIR = SCRIPT_DIR / "POC2" / "forces"
# STEP_CONFIG_OVERRIDES = {
#     "step0": {"rpm": 1000.0, "time_steps_per_revolution": 120},
#     "step1": {"rpm": 1000.0, "time_steps_per_revolution": 120 * 6},
#     "step2": {"rpm": 1000.0, "time_steps_per_revolution": 120 * 16},
#     "step3": {"rpm": 1000.0, "time_steps_per_revolution": 120 * 32},
#     "step4": {"rpm": 1000.0, "time_steps_per_revolution": 120 * 32 * 25 / 12},
# }
# REFERENCE_DENSITY_KG_M3 = 1.111635
# REFERENCE_SPEED_OF_SOUND_M_S = 349.237
# PROPELLER_RADIUS_M = 1.4
# REFERENCE_AREA_M2 = np.pi * PROPELLER_RADIUS_M**2
# MOMENT_LENGTHS_M = (1.4, 1.4, 1.4)
# THRUST_COMPONENT = "CFz"
# TORQUE_COMPONENT = "CMz"

"""RESULTS_DIR = Path(__file__).resolve().parent / "Lifter4B" / "forces"
STEP_CONFIG_OVERRIDES = {
    "step0C": {"rpm": 1075.0, "time_steps_per_revolution": 120},
    "step0D": {"rpm": 1075.0, "time_steps_per_revolution": 120},
    "step1": {"rpm": 1075.0 * 0.980047894, "time_steps_per_revolution": 120 * 6},
    "step2": {
        "rpm": 1075.0 * 0.980047894 * 0.988510721,
        "time_steps_per_revolution": 120 * 16,
    },
    "step2B": {"rpm": 1076.228, "time_steps_per_revolution": 120 * 16},
    "step2C": {"rpm": 1070.902, "time_steps_per_revolution": 120 * 16},
    "step2D": {"rpm": 1063.677, "time_steps_per_revolution": 120 * 16},
    "step3": {"rpm": 1063.677, "time_steps_per_revolution": 120 * 32},
    "step4": {"rpm": 1063.677, "time_steps_per_revolution": 120 * 32 * 25/12},
}"""

"""RESULTS_DIR = Path("C:/git/flow360cases/DUC/POC2x2/results_postprocessing/forces")
STEP_CONFIG_OVERRIDES = {
    "step0": {"rpm": 104.92919462989909/(2*np.pi)*60, "time_steps_per_revolution": 120},
    "step1": {"rpm": 104.92919462989909/(2*np.pi)*60, "time_steps_per_revolution": 120 * 6},
    "step2": {"rpm": 104.92919462989909/(2*np.pi)*60, "time_steps_per_revolution": 120 * 16,},
    "step3": {"rpm": 104.92919462989909/(2*np.pi)*60, "time_steps_per_revolution": 120 * 32},
    "step4": {"rpm": 104.92919462989909/(2*np.pi)*60, "time_steps_per_revolution": 120 * 32 * 25/12},
}"""


OUTPUT_FILE = RESULTS_DIR / "thrust_torque_coefficients_over_revolutions.png"
AVERAGING_WINDOW_REVOLUTIONS = 3
AXIS_PERCENTILE_LOW = 2
AXIS_PERCENTILE_HIGH = 98
CQ_VERTICAL_OFFSET_FRACTION = 0.15
# Pusher: sea level, ISA +10 K; reference velocity is the propeller tip speed.
REFERENCE_DENSITY_KG_M3 = 1.1839124828745407
REFERENCE_SPEED_OF_SOUND_M_S = 346.1484522412313
PROPELLER_RADIUS_M = 2.120 / 2
REFERENCE_AREA_M2 = np.pi * PROPELLER_RADIUS_M**2
MOMENT_LENGTHS_M = (PROPELLER_RADIUS_M,) * 3
THRUST_COMPONENT = "-CFx"
TORQUE_COMPONENT = "-CMx"


def configure_case(case: str) -> None:
    """Select reference geometry, step timing, and signed force/moment axes."""
    global RESULTS_DIR, OUTPUT_FILE, STEP_CONFIG_OVERRIDES
    global REFERENCE_DENSITY_KG_M3, REFERENCE_SPEED_OF_SOUND_M_S
    global PROPELLER_RADIUS_M, REFERENCE_AREA_M2, MOMENT_LENGTHS_M
    global THRUST_COMPONENT, TORQUE_COMPONENT
    RESULTS_DIR = SCRIPT_DIR / case / "forces"
    OUTPUT_FILE = RESULTS_DIR / "thrust_torque_coefficients_over_revolutions.png"
    PROPELLER_RADIUS_M = 1.4
    REFERENCE_DENSITY_KG_M3 = 1.111635
    REFERENCE_SPEED_OF_SOUND_M_S = 349.237
    THRUST_COMPONENT, TORQUE_COMPONENT = "CFz", "CMz"
    if case == "Pusher":
        # DUC/Pusher/DUC_EVE_Pusher_noise.py: sea level, ISA +10 K,
        # radius 2.120/2 m; coefficient reference velocity is the tip speed.
        PROPELLER_RADIUS_M = 1.06
        REFERENCE_DENSITY_KG_M3 = 1.1839124828745407
        REFERENCE_SPEED_OF_SOUND_M_S = 346.1484522412313
        THRUST_COMPONENT, TORQUE_COMPONENT = "-CFx", "-CMx"
        STEP_CONFIG_OVERRIDES = {
            f"step{step}": {"rpm": 2186.0, "time_steps_per_revolution": count}
            for step, count in enumerate((54, 329, 878, 1756, 3660))
        }
    else:
        STEP_CONFIG_OVERRIDES = {
            f"step{step}": {"rpm": 1000.0, "time_steps_per_revolution": count}
            for step, count in enumerate((120, 720, 1920, 3840, 8000))
        }
    REFERENCE_AREA_M2 = np.pi * PROPELLER_RADIUS_M**2
    MOMENT_LENGTHS_M = (PROPELLER_RADIUS_M,) * 3


def _signed_component(value: str, prefix: str) -> tuple[str, int]:
    match = re.fullmatch(r"([+-]?)(c[fm])([xyz])", value.strip(), re.IGNORECASE)
    if match is None or match[2].lower() != prefix.lower():
        raise ValueError(f"Expected a signed {prefix} component, such as -{prefix}x; got {value!r}.")
    return prefix.upper() + match[3].lower(), -1 if match[1] == "-" else 1


@dataclass(frozen=True)
class RunConfig:
    rpm: float
    time_steps_per_revolution: float


def _run_label(path: Path) -> str:
    # Archive-style case filenames are identified by their step directory.
    for part in reversed(path.parts[:-1]):
        if part in STEP_CONFIG_OVERRIDES:
            return part
    name = re.sub(r"_(?:results_)?total_forces_v2\.csv$", "", path.name)
    return name.split("_case-", 1)[0]


def _run_config(label: str) -> RunConfig:
    if label not in STEP_CONFIG_OVERRIDES:
        raise KeyError(
            f"No rpm/time_steps_per_revolution preset is defined for run {label!r}. "
            "Add it to STEP_CONFIG_OVERRIDES."
        )
    return RunConfig(**STEP_CONFIG_OVERRIDES[label])


def _read_final_coefficients_by_physical_step(path: Path) -> dict[int, tuple[float, float]]:
    final_coefficients_by_step: dict[int, tuple[float, float]] = {}
    thrust_column, thrust_sign = _signed_component(THRUST_COMPONENT, "CF")
    torque_column, torque_sign = _signed_component(TORQUE_COMPONENT, "CM")

    with path.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        missing = {"physical_step", thrust_column, torque_column} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing force CSV columns in {path}: {sorted(missing)}")
        for row in reader:
            final_coefficients_by_step[int(row["physical_step"])] = (
                thrust_sign * float(row[thrust_column]),
                torque_sign * float(row[torque_column]),
            )

    if not final_coefficients_by_step:
        raise ValueError(f"No {thrust_column}/{torque_column} values found in {path}")
    return final_coefficients_by_step


def _load_runs(results_dir: Path | None = None) -> list[tuple[str, list[int], list[float], list[float]]]:
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    runs = []
    for path in sorted(results_dir.rglob("*total_forces_v2.csv")):
        final_coefficients_by_step = _read_final_coefficients_by_physical_step(path)
        steps = sorted(final_coefficients_by_step)
        thrust = [final_coefficients_by_step[step][0] for step in steps]
        torque = [final_coefficients_by_step[step][1] for step in steps]
        label = _run_label(path)
        _run_config(label)
        runs.append((label, steps, thrust, torque))

    if not runs:
        raise FileNotFoundError(f"No total-forces CSV files found in {results_dir}")
    return sorted(runs, key=lambda run: list(STEP_CONFIG_OVERRIDES).index(run[0]))


def _dimensional_loads(
    thrust: list[float],
    torque: list[float],
    cfg: RunConfig,
) -> tuple[list[float], list[float], list[float]]:
    """Convert the selected signed CFD coefficients to N, Nm, and kW."""
    reference_velocity = abs(2.0 * np.pi * cfg.rpm / 60.0) * PROPELLER_RADIUS_M
    dynamic_pressure = 0.5 * REFERENCE_DENSITY_KG_M3 * reference_velocity**2
    force_scale = dynamic_pressure * REFERENCE_AREA_M2
    moment_length = MOMENT_LENGTHS_M["xyz".index(TORQUE_COMPONENT[-1].lower())]
    thrust_n = [value * force_scale for value in thrust]
    torque_nm = [value * force_scale * moment_length for value in torque]
    power_kw = [value * (2.0 * np.pi * cfg.rpm / 60.0) / 1000.0 for value in torque_nm]
    return thrust_n, torque_nm, power_kw


def _propeller_coefficients(
    thrust: list[float],
    torque: list[float],
    cfg: RunConfig,
) -> tuple[list[float], list[float]]:
    thrust_n, torque_nm, _ = _dimensional_loads(thrust, torque, cfg)
    rev_per_second = cfg.rpm / 60.0
    diameter = 2.0 * PROPELLER_RADIUS_M
    thrust_denominator = REFERENCE_DENSITY_KG_M3 * rev_per_second**2 * diameter**4
    torque_denominator = REFERENCE_DENSITY_KG_M3 * rev_per_second**2 * diameter**5

    thrust_coefficient = [
        value / thrust_denominator for value in thrust_n
    ]
    torque_coefficient = [
        value / torque_denominator for value in torque_nm
    ]
    return thrust_coefficient, torque_coefficient


def _relative_revolutions(steps: list[int], cfg: RunConfig) -> list[float]:
    first_step = steps[0]
    return [(step - first_step) / cfg.time_steps_per_revolution for step in steps]


def _percentile_limits(values: list[float]) -> tuple[float, float]:
    lower, upper = np.percentile(values, [AXIS_PERCENTILE_LOW, AXIS_PERCENTILE_HIGH])
    if lower == upper:
        padding = 0.01 if lower == 0.0 else 0.08 * abs(lower)
        return lower - padding, upper + padding
    padding = 1 * (upper - lower)
    return lower - padding, upper + padding


def plot_thrust_and_torque_coefficients() -> Path:
    runs = _load_runs()
    processed_runs = []
    dimensional_runs = []
    revolution_offset = 0.0
    for label, steps, thrust, torque in runs:
        cfg = _run_config(label)
        dimensional_runs.append(_dimensional_loads(thrust, torque, cfg))
        thrust_coefficient, torque_coefficient = _propeller_coefficients(thrust, torque, cfg)
        local_revolutions = _relative_revolutions(steps, cfg)
        revolutions = [revolution_offset + value for value in local_revolutions]
        processed_runs.append((label, revolutions, thrust_coefficient, torque_coefficient, cfg))
        revolution_offset = revolutions[-1] + 1.0 / cfg.time_steps_per_revolution

    all_revolutions = [rev for _, revs, _, _, _ in processed_runs for rev in revs]
    all_thrust_coefficients = [value for _, _, values, _, _ in processed_runs for value in values]
    all_torque_coefficients = [value for _, _, _, values, _ in processed_runs for value in values]
    sample_widths = np.asarray([
        1.0 / cfg.time_steps_per_revolution
        for _, revs, _, _, cfg in processed_runs for _ in revs
    ])
    averaging_end = revolution_offset
    averaging_start = max(all_revolutions[0], averaging_end - AVERAGING_WINDOW_REVOLUTIONS)
    weights = np.maximum(
        0.0,
        np.minimum(np.asarray(all_revolutions) + sample_widths, averaging_end)
        - np.maximum(all_revolutions, averaging_start),
    )
    average_ct = float(np.average(all_thrust_coefficients, weights=weights))
    average_cq = float(np.average(all_torque_coefficients, weights=weights))
    average_loads = [
        float(np.average(
            [value for loads in dimensional_runs for value in loads[component]],
            weights=weights,
        ))
        for component in range(3)
    ]
    print(f"Averages over revolutions {averaging_start:.3f}-{averaging_end:.3f}:")
    print(f"  Thrust ({THRUST_COMPONENT}): {average_loads[0]:.2f} N")
    print(f"  Torque ({TORQUE_COMPONENT}): {average_loads[1]:.2f} Nm")
    print(f"  Power (torque * angular speed): {average_loads[2]:.2f} kW")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax_torque = ax.twinx()
    for label, revolutions, thrust_coefficient, torque_coefficient, _ in processed_runs:
        ax.plot(revolutions, thrust_coefficient, linewidth=1.0, label=f"{label} CT")
        ax_torque.plot(
            revolutions,
            torque_coefficient,
            linewidth=1.0,
            linestyle="--",
            label=f"{label} CQ",
        )

    ct_y_min, ct_y_max = _percentile_limits(all_thrust_coefficients)
    cq_y_min, cq_y_max = _percentile_limits(all_torque_coefficients)
    cq_axis_offset = CQ_VERTICAL_OFFSET_FRACTION * (cq_y_max - cq_y_min)
    ax.set_ylim(ct_y_min, ct_y_max)
    ax_torque.set_ylim(cq_y_min + cq_axis_offset, cq_y_max + cq_axis_offset)
    ax.axvspan(averaging_start, averaging_end, color="gray", alpha=0.18, label="averaging window")
    ax.hlines(
        average_ct,
        averaging_start,
        averaging_end,
        colors="black",
        linestyles="--",
        linewidth=1.5,
        label=f"window average CT = {average_ct:.6f}",
    )
    ax_torque.hlines(
        average_cq,
        averaging_start,
        averaging_end,
        colors="tab:red",
        linestyles=":",
        linewidth=1.5,
        label=f"window average CQ = {average_cq:.6f}",
    )

    for index, (label, revolutions, _, _, cfg) in enumerate(processed_runs):
        start = revolutions[0]
        end = revolutions[-1]
        section_width = end - start
        is_short_section = section_width < 0.25
        if index > 0:
            ax.axvline(start, color="red", linestyle="--", linewidth=1.0)
        ax.annotate(
            f"{label}\n{cfg.time_steps_per_revolution} steps/rev\n{cfg.rpm:.1f} rpm",
            xy=(
                (start + end) / 2,
                ct_y_min + (0.72 if is_short_section else 0.92) * (ct_y_max - ct_y_min),
            ),
            ha="center",
            va="center" if is_short_section else "top",
            rotation=90 if is_short_section else 0,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    ax.set_xlabel("Revolutions")
    ax.set_ylabel("Thrust coefficient CT")
    ax_torque.set_ylabel("Torque coefficient CQ")
    ax.set_title("Propeller thrust and torque coefficients over revolutions")
    ax.grid(True, alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    torque_lines, torque_labels = ax_torque.get_legend_handles_labels()
    ax.legend(lines + torque_lines, labels + torque_labels, loc="best")
    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.close(fig)
    return OUTPUT_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot signed propeller thrust and torque coefficients.")
    parser.add_argument("--case", choices=("POC2", "Pusher"), default="Pusher")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--thrust-component", help="Signed force component, e.g. --thrust-component=-cfx")
    parser.add_argument("--torque-component", help="Signed moment component, e.g. --torque-component=-cmx")
    args = parser.parse_args()
    # Keep the editable Pusher settings above when running the default case.
    if args.case != "Pusher":
        configure_case(args.case)
    global RESULTS_DIR, OUTPUT_FILE, THRUST_COMPONENT, TORQUE_COMPONENT
    if args.thrust_component is not None:
        THRUST_COMPONENT = args.thrust_component
    if args.torque_component is not None:
        TORQUE_COMPONENT = args.torque_component
    _signed_component(THRUST_COMPONENT, "CF")
    _signed_component(TORQUE_COMPONENT, "CM")
    if args.results_dir is not None:
        RESULTS_DIR = args.results_dir
        OUTPUT_FILE = RESULTS_DIR / "thrust_torque_coefficients_over_revolutions.png"
    if args.output_file is not None:
        OUTPUT_FILE = args.output_file
    print(f"{args.case}: thrust={THRUST_COMPONENT}, torque={TORQUE_COMPONENT}")
    output_file = plot_thrust_and_torque_coefficients()
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
