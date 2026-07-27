from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = RESULTS_DIR / "cfz_over_physical_steps.png"
AVERAGING_WINDOW_STEPS = 120 * 8


def _run_label(path: Path) -> str:
    return path.name.split("_case-", 1)[0]


def _read_final_cfz_by_physical_step(path: Path) -> dict[int, float]:
    final_cfz_by_step: dict[int, float] = {}

    with path.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            final_cfz_by_step[int(row["physical_step"])] = float(row["CFz"])

    if not final_cfz_by_step:
        raise ValueError(f"No CFz values found in {path}")
    return final_cfz_by_step


def _load_runs(results_dir: Path = RESULTS_DIR) -> list[tuple[str, list[int], list[float]]]:
    runs = []
    for path in sorted(results_dir.glob("*_results_total_forces_v2.csv")):
        final_cfz_by_step = _read_final_cfz_by_physical_step(path)
        steps = sorted(final_cfz_by_step)
        cfz = [final_cfz_by_step[step] for step in steps]
        runs.append((_run_label(path), steps, cfz))

    if not runs:
        raise FileNotFoundError(f"No total-forces CSV files found in {results_dir}")
    return runs


def plot_cfz_by_physical_step() -> Path:
    runs = _load_runs()
    all_steps = [step for _, steps, _ in runs for step in steps]
    all_cfz = [value for _, _, cfz in runs for value in cfz]
    if len(all_cfz) < AVERAGING_WINDOW_STEPS:
        raise ValueError(
            f"Only found {len(all_cfz)} physical steps, but averaging window needs "
            f"{AVERAGING_WINDOW_STEPS}."
        )
    averaging_steps = all_steps[-AVERAGING_WINDOW_STEPS:]
    averaging_cfz = all_cfz[-AVERAGING_WINDOW_STEPS:]
    average_cfz = sum(averaging_cfz) / len(averaging_cfz)
    averaging_start = averaging_steps[0]
    averaging_end = averaging_steps[-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, steps, cfz in runs:
        ax.plot(steps, cfz, linewidth=1.0, label=label)

    y_min = min(all_cfz)
    y_max = 0.06
    ax.set_ylim(y_min, y_max)
    y_text = y_min + 0.92 * (y_max - y_min)
    ax.axvspan(averaging_start, averaging_end, color="gray", alpha=0.18, label="averaging window")
    ax.hlines(
        average_cfz,
        averaging_start,
        averaging_end,
        colors="black",
        linestyles="--",
        linewidth=1.5,
        label=f"window average CFz = {average_cfz:.6f}",
    )

    for index, (label, steps, _) in enumerate(runs):
        start = steps[0]
        end = steps[-1]
        section_width = end - start
        is_short_section = section_width < 800
        if index > 0:
            ax.axvline(start, color="red", linestyle="--", linewidth=1.0)
        ax.annotate(
            label,
            xy=((start + end) / 2, y_min + (0.72 if is_short_section else 0.92) * (y_max - y_min)),
            ha="center",
            va="center" if is_short_section else "top",
            rotation=90 if is_short_section else 0,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    ax.set_xlabel("Physical step")
    ax.set_ylabel("CFz")
    ax.set_title("CFz over physical steps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.close(fig)
    return OUTPUT_FILE


def main() -> None:
    output_file = plot_cfz_by_physical_step()
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
