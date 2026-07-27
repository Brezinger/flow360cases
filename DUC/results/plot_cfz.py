from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("C:/git/flow360cases/DUC/results/forces").resolve()
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
}

"""RESULTS_DIR = Path("C:/git/flow360cases/DUC/POC2x2/results/forces")
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
REFERENCE_DENSITY_KG_M3 = 1.111635
REFERENCE_SPEED_OF_SOUND_M_S = 349.237
PROPELLER_RADIUS_M = 1.4
REFERENCE_AREA_M2 = np.pi * PROPELLER_RADIUS_M**2
MOMENT_LENGTH_Z_M = 1.4


@dataclass(frozen=True)
class RunConfig:
    rpm: float
    time_steps_per_revolution: float


def _run_label(path: Path) -> str:
    return path.name.split("_case-", 1)[0]


def _run_config(label: str) -> RunConfig:
    if label not in STEP_CONFIG_OVERRIDES:
        raise KeyError(
            f"No rpm/time_steps_per_revolution preset is defined for run {label!r}. "
            "Add it to STEP_CONFIG_OVERRIDES."
        )
    return RunConfig(**STEP_CONFIG_OVERRIDES[label])


def _read_final_coefficients_by_physical_step(path: Path) -> dict[int, tuple[float, float]]:
    final_coefficients_by_step: dict[int, tuple[float, float]] = {}

    with path.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            final_coefficients_by_step[int(row["physical_step"])] = (
                float(row["CFz"]),
                float(row["CMz"]),
            )

    if not final_coefficients_by_step:
        raise ValueError(f"No CFz/CMz values found in {path}")
    return final_coefficients_by_step


def _load_runs(results_dir: Path = RESULTS_DIR) -> list[tuple[str, list[int], list[float], list[float]]]:
    runs = []
    for path in sorted(results_dir.glob("*_results_total_forces_v2.csv")):
        final_coefficients_by_step = _read_final_coefficients_by_physical_step(path)
        steps = sorted(final_coefficients_by_step)
        cfz = [final_coefficients_by_step[step][0] for step in steps]
        cmz = [final_coefficients_by_step[step][1] for step in steps]
        runs.append((_run_label(path), steps, cfz, cmz))

    if not runs:
        raise FileNotFoundError(f"No total-forces CSV files found in {results_dir}")
    return runs


def _propeller_coefficients(
    cfz: list[float],
    cmz: list[float],
    cfg: RunConfig,
) -> tuple[list[float], list[float]]:
    reference_velocity = abs(2.0 * np.pi * cfg.rpm / 60.0) * PROPELLER_RADIUS_M
    dynamic_pressure = 0.5 * REFERENCE_DENSITY_KG_M3 * reference_velocity**2
    rev_per_second = cfg.rpm / 60.0
    diameter = 2.0 * PROPELLER_RADIUS_M
    thrust_denominator = REFERENCE_DENSITY_KG_M3 * rev_per_second**2 * diameter**4
    torque_denominator = REFERENCE_DENSITY_KG_M3 * rev_per_second**2 * diameter**5

    thrust_coefficient = [
        value * dynamic_pressure * REFERENCE_AREA_M2 / thrust_denominator
        for value in cfz
    ]
    torque_coefficient = [
        value * dynamic_pressure * REFERENCE_AREA_M2 * MOMENT_LENGTH_Z_M / torque_denominator
        for value in cmz
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
    revolution_offset = 0.0
    for label, steps, cfz, cmz in runs:
        cfg = _run_config(label)
        thrust_coefficient, torque_coefficient = _propeller_coefficients(cfz, cmz, cfg)
        local_revolutions = _relative_revolutions(steps, cfg)
        revolutions = [revolution_offset + value for value in local_revolutions]
        processed_runs.append((label, revolutions, thrust_coefficient, torque_coefficient, cfg))
        revolution_offset = revolutions[-1] + 1.0 / cfg.time_steps_per_revolution

    averaging_window_steps = int(processed_runs[-1][-1].time_steps_per_revolution * AVERAGING_WINDOW_REVOLUTIONS)
    all_revolutions = [rev for _, revs, _, _, _ in processed_runs for rev in revs]
    all_thrust_coefficients = [value for _, _, values, _, _ in processed_runs for value in values]
    all_torque_coefficients = [value for _, _, _, values, _ in processed_runs for value in values]
    averaging_window_steps = min(averaging_window_steps, len(all_thrust_coefficients))
    averaging_revolutions = all_revolutions[-averaging_window_steps:]
    averaging_ct = all_thrust_coefficients[-averaging_window_steps:]
    averaging_cq = all_torque_coefficients[-averaging_window_steps:]
    average_ct = sum(averaging_ct) / len(averaging_ct)
    average_cq = sum(averaging_cq) / len(averaging_cq)
    averaging_start = averaging_revolutions[0]
    averaging_end = averaging_revolutions[-1]

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
    ax.set_ylim(ct_y_min, ct_y_max)
    ax_torque.set_ylim(cq_y_min, cq_y_max)
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
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.close(fig)
    return OUTPUT_FILE


def main() -> None:
    output_file = plot_thrust_and_torque_coefficients()
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
