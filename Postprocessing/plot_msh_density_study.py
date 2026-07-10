from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_STUDY_DIR = Path(r"C:\WDIR\flow360\V3\mesh_density_study")
DEFAULT_PRINT_COLUMNS = ("CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz")
REQUIRED_HISTORY_COLUMNS = ("physical_step", "pseudo_step", "CL", "CD")
DEFAULT_PLOT_LINES = ("CL", "CD")


@dataclass(frozen=True)
class StudyParameter:
    name: str
    value: float | str
    sort_key: float
    x_label: str
    filename_token: str
    is_categorical: bool = False


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


QUALITATIVE_MESH_LEVELS = {
    "finest": ("finest", 0.0),
    "fine": ("fine", 1.0),
    "finer": ("fine", 1.0),
    "medium": ("medium", 2.0),
    "med": ("medium", 2.0),
    "mid": ("medium", 2.0),
    "baseline": ("medium", 2.0),
    "coarse": ("coarse", 3.0),
    "coarser": ("coarse", 3.0),
    "coarsest": ("coarsest", 4.0),
}


def study_parameter_from_name(path: Path) -> StudyParameter:
    surface_edge_match = re.search(
        r"(?:surf(?:ace)?[_ -]*)?edge[_ -]*(?:max|length)?[_ -]*"
        r"([-+]?\d*\.?\d+)\s*mm\b",
        path.stem,
        flags=re.IGNORECASE,
    )
    if surface_edge_match is not None:
        return StudyParameter(
            name="surface_edge_length",
            value=float(surface_edge_match.group(1)),
            sort_key=float(surface_edge_match.group(1)),
            x_label="Maximum surface edge length [mm]",
            filename_token="surface_edge_length",
        )

    qualitative_mesh = qualitative_mesh_level_from_name(path)
    if qualitative_mesh is not None:
        label, sort_key = qualitative_mesh
        return StudyParameter(
            name="mesh_density",
            value=label,
            sort_key=sort_key,
            x_label="Mesh density",
            filename_token="mesh_density",
            is_categorical=True,
        )

    match = re.search(r"[-+]?\d*\.?\d+", path.stem)
    if match is None:
        raise ValueError(
            "Could not parse growth rate, surface edge length, or qualitative "
            f"mesh density from {path.name}"
        )
    growth_rate = float(match.group())
    return StudyParameter(
        name="growth_rate",
        value=growth_rate,
        sort_key=growth_rate,
        x_label="Wall-normal growth rate",
        filename_token="growth_rate",
    )


def qualitative_mesh_level_from_name(path: Path) -> tuple[str, float] | None:
    stem_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", path.stem.lower())
        if token and token != "mesh"
    ]
    if not stem_tokens:
        return None

    token_set = set(stem_tokens)
    for token, (label, sort_key) in QUALITATIVE_MESH_LEVELS.items():
        if token in token_set:
            if "very" in token_set and label in {"coarse", "fine"}:
                label = f"very {label}"
                sort_key += 0.5 if label.endswith("coarse") else -0.5
            return label, sort_key

    return None


def find_history_csvs(study_dir: Path) -> list[Path]:
    history_csvs = []
    for csv_path in sorted(study_dir.glob("*.csv")):
        with csv_path.open(newline="") as file:
            header = [item.strip() for item in next(csv.reader(file)) if item.strip()]
        if set(REQUIRED_HISTORY_COLUMNS).issubset(header):
            history_csvs.append(csv_path)
    return history_csvs


def parse_plot_lines(raw_value: str) -> list[str]:
    columns = [column.strip() for column in raw_value.split(",") if column.strip()]
    if not columns:
        raise ValueError("--plot-lines must contain at least one column name")
    return columns


def normalize_column_names(columns: list[str], header: list[str]) -> list[str]:
    by_lowercase = {column.lower(): column for column in header}
    normalized_columns = []
    for column in columns:
        normalized_column = by_lowercase.get(column.lower())
        if normalized_column is None:
            available_columns = ", ".join(header)
            raise ValueError(
                f"Column {column!r} from --plot-lines is not in the history CSV. "
                f"Available columns: {available_columns}"
            )
        normalized_columns.append(normalized_column)
    return normalized_columns


def plot_convergence(
    plot_rows: list[dict[str, float | str]],
    output_path: Path,
    plot_lines: list[str],
    x_key: str,
    x_label: str,
    is_categorical_x: bool,
) -> None:
    sort_key = f"{x_key}_sort_key"
    plot_rows = sorted(plot_rows, key=lambda row: float(row[sort_key]))
    if is_categorical_x:
        x_values = list(range(len(plot_rows)))
        x_tick_labels = [str(row[x_key]) for row in plot_rows]
    else:
        x_values = [float(row[x_key]) for row in plot_rows]
        x_tick_labels = None
    axes_count = len(plot_lines)

    fig_height = max(3.2, 2.8 * axes_count)
    fig, axes = plt.subplots(axes_count, 1, figsize=(7.0, fig_height), sharex=True)
    if axes_count == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (axis, column) in enumerate(zip(axes, plot_lines)):
        means = [float(row[f"{column}_mean"]) for row in plot_rows]
        std_devs = [float(row[f"{column}_std"]) for row in plot_rows]
        axis.errorbar(
            x_values,
            means,
            yerr=std_devs,
            marker="o",
            capsize=5,
            linewidth=1.8,
            color=colors[index % len(colors)],
        )
        axis.set_ylabel(column)
        set_x_axis_at_zero(axis, means, std_devs)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    if x_tick_labels is not None:
        axes[-1].set_xticks(x_values, x_tick_labels)
    axes[0].set_title("Mesh Density Study: Final Rows Mean and Standard Deviation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def set_x_axis_at_zero(
    axis: plt.Axes,
    means: list[float],
    std_devs: list[float],
) -> None:
    lower_data_limit = min(mean - std_dev for mean, std_dev in zip(means, std_devs))
    upper_data_limit = max(mean + std_dev for mean, std_dev in zip(means, std_devs))
    lower_limit = min(0.0, lower_data_limit)
    upper_limit = max(0.0, upper_data_limit)
    if math.isclose(lower_limit, upper_limit):
        padding = 1.0 if math.isclose(upper_limit, 0.0) else abs(upper_limit) * 0.1
    else:
        padding = 0.08 * (upper_limit - lower_limit)

    axis.set_ylim(lower_limit - padding, upper_limit + padding)
    axis.spines["bottom"].set_position(("data", 0.0))
    axis.spines["top"].set_visible(False)
    axis.xaxis.set_ticks_position("bottom")
    axis.xaxis.set_label_position("bottom")


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
        help="Optional plot path. Default: <study-dir>/<parameter>_convergence.png",
    )
    parser.add_argument(
        "--plot-lines",
        default=",".join(DEFAULT_PLOT_LINES),
        help='Comma-separated history columns to plot. Default: "CL,CD"',
    )
    args = parser.parse_args()

    csv_paths = find_history_csvs(args.study_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.study_dir}")

    requested_plot_lines = parse_plot_lines(args.plot_lines)
    output_rows = []
    plot_rows = []
    study_parameter_name = None
    study_parameter_x_label = None
    study_parameter_filename_token = None
    study_parameter_is_categorical = None
    plot_lines = None

    for csv_path in csv_paths:
        header, data_rows = load_csv(csv_path)
        if plot_lines is None:
            plot_lines = normalize_column_names(requested_plot_lines, header)
        else:
            normalize_column_names(plot_lines, header)

        tail_count, stats = tail_stats(header, data_rows, args.tail_fraction)
        study_parameter = study_parameter_from_name(csv_path)
        if study_parameter_name is None:
            study_parameter_name = study_parameter.name
            study_parameter_x_label = study_parameter.x_label
            study_parameter_filename_token = study_parameter.filename_token
            study_parameter_is_categorical = study_parameter.is_categorical
        elif study_parameter.name != study_parameter_name:
            raise ValueError(
                "Cannot mix mesh-density parameters in one plot: "
                f"found both {study_parameter_name!r} and {study_parameter.name!r}"
            )

        print(
            f"\n{csv_path.name}: total_rows={len(data_rows)}, "
            f"tail_rows={tail_count}, "
            f"{study_parameter.name}={study_parameter.value}"
        )
        for column in DEFAULT_PRINT_COLUMNS:
            if column in stats:
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
        plot_row: dict[str, float | str] = {
            "case": csv_path.stem,
            study_parameter.name: study_parameter.value,
            f"{study_parameter.name}_sort_key": study_parameter.sort_key,
        }
        for column in plot_lines:
            plot_row[f"{column}_mean"] = stats[column][0]
            plot_row[f"{column}_std"] = stats[column][1]
        plot_rows.append(plot_row)

    output_path = args.output or args.study_dir / "last_10pct_stats.csv"
    write_stats_csv(output_path, output_rows)
    print(f"\nWrote full statistics to {output_path}")

    if plot_lines is None or study_parameter_name is None:
        raise RuntimeError("No plot rows were collected")

    plot_output_path = (
        args.plot_output
        or args.study_dir / f"{study_parameter_filename_token}_convergence.png"
    )
    plot_convergence(
        plot_rows,
        plot_output_path,
        plot_lines,
        study_parameter_name,
        study_parameter_x_label,
        bool(study_parameter_is_categorical),
    )
    plotted_columns = ", ".join(plot_lines)
    print(f"Wrote {plotted_columns} convergence plot to {plot_output_path}")


if __name__ == "__main__":
    main()
