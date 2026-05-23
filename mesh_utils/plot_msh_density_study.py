from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_STUDY_DIR = Path(r"C:\WDIR\flow360\V3\mesh_density_study")
DEFAULT_PRINT_COLUMNS = ("CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
REQUIRED_HISTORY_COLUMNS = ("physical_step", "pseudo_step", "CL", "CD")


def load_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as file:
        reader = csv.reader(file)
        header = [item.strip() for item in next(reader) if item.strip()]
        missing_columns = set(REQUIRED_HISTORY_COLUMNS) - set(header)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path.name} is not a history CSV; missing {missing_text}")
        rows = []

        for row in reader:
            values = [item.strip() for item in row[: len(header)]]
            if not values or all(item == "" for item in values):
                continue
            rows.append([float(item) for item in values])

    return header, rows


def tail_stats(
    header: list[str], rows: list[list[float]], tail_fraction: float
) -> tuple[int, dict[str, tuple[float, float]]]:
    if not rows:
        raise ValueError("CSV has no data rows")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be between 0 and 1")

    tail_count = max(1, math.ceil(len(rows) * tail_fraction))
    tail_rows = rows[-tail_count:]
    stats = {}

    for column_index, column_name in enumerate(header):
        values = [row[column_index] for row in tail_rows]
        mean = statistics.fmean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[column_name] = (mean, std_dev)

    return tail_count, stats


def write_stats_csv(
    output_path: Path,
    rows: list[tuple[str, str, int, int, float, float]],
) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["case", "column", "total_rows", "tail_rows", "mean", "std_dev"]
        )
        writer.writerows(rows)


def growth_rate_from_name(path: Path) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", path.stem)
    if match is None:
        raise ValueError(f"Could not parse growth rate from {path.name}")
    return float(match.group())


def find_history_csvs(study_dir: Path) -> list[Path]:
    history_csvs = []
    for csv_path in sorted(study_dir.glob("*.csv")):
        with csv_path.open(newline="") as file:
            header = [item.strip() for item in next(csv.reader(file)) if item.strip()]
        if set(REQUIRED_HISTORY_COLUMNS).issubset(header):
            history_csvs.append(csv_path)
    return history_csvs


def plot_cl_cd_convergence(
    plot_rows: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    plot_rows = sorted(plot_rows, key=lambda row: float(row["growth_rate"]))
    growth_rates = [float(row["growth_rate"]) for row in plot_rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True)
    for axis, column, color in zip(axes, ("CL", "CD"), ("tab:blue", "tab:orange")):
        means = [float(row[f"{column}_mean"]) for row in plot_rows]
        std_devs = [float(row[f"{column}_std"]) for row in plot_rows]
        axis.errorbar(
            growth_rates,
            means,
            yerr=std_devs,
            marker="o",
            capsize=5,
            linewidth=1.8,
            color=color,
        )
        axis.set_ylabel(column)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wall-normal growth rate")
    axes[0].set_title("Mesh Density Study: Final 10% Mean and Standard Deviation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate mesh-density study statistics over the final rows."
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=DEFAULT_STUDY_DIR,
        help=f"Directory containing CSV files. Default: {DEFAULT_STUDY_DIR}",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.10,
        help="Fraction of final rows to use for statistics. Default: 0.10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Default: <study-dir>/last_10pct_stats.csv",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional plot path. Default: <study-dir>/cl_cd_convergence.png",
    )
    args = parser.parse_args()

    csv_paths = find_history_csvs(args.study_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.study_dir}")

    output_rows = []
    plot_rows = []
    for csv_path in csv_paths:
        header, data_rows = load_csv(csv_path)
        tail_count, stats = tail_stats(header, data_rows, args.tail_fraction)
        growth_rate = growth_rate_from_name(csv_path)

        print(
            f"\n{csv_path.name}: total_rows={len(data_rows)}, "
            f"tail_rows={tail_count}"
        )
        for column in DEFAULT_PRINT_COLUMNS:
            mean, std_dev = stats[column]
            print(f"  {column:>3}: mean={mean:.12g}, std={std_dev:.12g}")

        output_rows.extend(
            (
                csv_path.stem,
                column,
                len(data_rows),
                tail_count,
                mean,
                std_dev,
            )
            for column, (mean, std_dev) in stats.items()
        )
        plot_rows.append(
            {
                "case": csv_path.stem,
                "growth_rate": growth_rate,
                "CL_mean": stats["CL"][0],
                "CL_std": stats["CL"][1],
                "CD_mean": stats["CD"][0],
                "CD_std": stats["CD"][1],
            }
        )

    output_path = args.output or args.study_dir / "last_10pct_stats.csv"
    write_stats_csv(output_path, output_rows)
    print(f"\nWrote full statistics to {output_path}")

    plot_output_path = args.plot_output or args.study_dir / "cl_cd_convergence.png"
    plot_cl_cd_convergence(plot_rows, plot_output_path)
    print(f"Wrote CL/CD convergence plot to {plot_output_path}")


if __name__ == "__main__":
    main()
