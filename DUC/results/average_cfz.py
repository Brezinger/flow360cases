from __future__ import annotations

import csv
from pathlib import Path


"""N_STEPS = 120 * 16
CSV_FILE = Path(__file__).with_name(
    "step2_case-950e2800-b54d-4826-b85f-56206efdc51f_results_total_forces_v2.csv"
)"""
N_STEPS = 120 * 4
CSV_FILE = Path(__file__).with_name(
    "step2B_case-a5d80787-d004-44d8-9892-3d1844516cb0_results_total_forces_v2.csv"
)


def average_last_cfz(csv_file: Path = CSV_FILE, n_steps: int = N_STEPS) -> float:
    latest_by_physical_step: dict[int, float] = {}

    with csv_file.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            physical_step = int(row["physical_step"])
            latest_by_physical_step[physical_step] = float(row["CFz"])

    values = list(latest_by_physical_step.values())[-n_steps:]
    if len(values) < n_steps:
        raise ValueError(f"Only found {len(values)} physical steps, but requested {n_steps}.")

    return sum(values) / len(values)


def main() -> None:
    cfz_average = average_last_cfz()
    print(f"Average CFz over last {N_STEPS} physical steps: {cfz_average:.12g}")


if __name__ == "__main__":
    main()
