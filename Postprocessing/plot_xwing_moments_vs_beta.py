"""Plot XWing rolling and yawing moments versus yaw angle.

The script reads the total-force histories for the U=35 m/s rectangular-wing
yaw study.  A steady coefficient is the mean of the final
``CONVERGED_FRACTION`` of finite Flow360 samples.  Cases whose result history
has not been downloaded yet are reported and omitted, so rerunning the script
after adding a case automatically adds its point to the plots.

Outputs are written to ``<flow360-root>/yaw_sensitivity``:

* ``xwing_moments_vs_beta.csv``
* ``xwing_cmx_vs_beta.png``
* ``xwing_cmz_vs_beta.png``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FLOW360_ROOT = Path(
    "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360"
)
CASE_ROOT = FLOW360_ROOT / "rectangular wing"
OUTPUT_DIR = FLOW360_ROOT / "yaw_sensitivity"
CONVERGED_FRACTION = 0.10
SHOW_PLOTS = True


@dataclass(frozen=True)
class YawCase:
    """One simulation contributing a point to the yaw-sensitivity curves."""

    beta_deg: float
    result_directory: Path

    @property
    def force_history_files(self) -> tuple[Path, ...]:
        return tuple(sorted(self.result_directory.glob("*_total_forces_v2.csv")))


CASES = (
    YawCase(-5.0, CASE_ROOT / "XWing2_2 fully_turbulent_SA U35 beta -5°"),
    YawCase(0.0, CASE_ROOT / "XWing2_2 fully_turbulent_SA U35"),
    YawCase(5.0, CASE_ROOT / "XWing2_2 fully_turbulent_SA U35 beta +5°"),
)


def _force_history_file(case: YawCase) -> Path | None:
    """Return the one force history for *case*, or warn if it is unavailable."""
    matches = case.force_history_files
    if not matches:
        print(
            f"Warning: beta={case.beta_deg:g}° is not available; expected one "
            f"'*_total_forces_v2.csv' in {case.result_directory}. Skipping it."
        )
        return None
    if len(matches) > 1:
        formatted = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Expected exactly one '*_total_forces_v2.csv' for beta="
            f"{case.beta_deg:g}°, found {len(matches)}: {formatted}"
        )
    return matches[0]


def calculate_converged_coefficients(case: YawCase) -> dict[str, object] | None:
    """Average the final converged fraction of CMx and CMz for one yaw case."""
    force_history_file = _force_history_file(case)
    if force_history_file is None:
        return None

    history = pd.read_csv(force_history_file, skipinitialspace=True)
    required_columns = {"CMx", "CMz"}
    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        raise ValueError(
            f"{force_history_file} is missing columns: {sorted(missing_columns)}"
        )

    coefficients = (
        history[["CMx", "CMz"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if coefficients.empty:
        raise ValueError(f"{force_history_file} contains no finite CMx/CMz samples.")

    sample_count = max(1, int(np.ceil(len(coefficients) * CONVERGED_FRACTION)))
    converged_samples = coefficients.tail(sample_count)
    return {
        "beta_deg": case.beta_deg,
        "cmx": float(converged_samples["CMx"].mean()),
        "cmz": float(converged_samples["CMz"].mean()),
        "converged_sample_count": sample_count,
        "total_finite_sample_count": len(coefficients),
        "force_history_file": str(force_history_file),
    }


def plot_moment_vs_beta(
    results: pd.DataFrame,
    coefficient_column: str,
    coefficient_label: str,
    title: str,
    output_filename: str,
) -> None:
    """Save one moment-coefficient curve over the available yaw angles."""
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.plot(
        results["beta_deg"],
        results[coefficient_column],
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
    )
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.axvline(0.0, color="0.25", linewidth=0.8)
    axis.set_xlabel(r"Yaw angle $\beta$ [deg]")
    axis.set_ylabel(coefficient_label)
    axis.set_title(title)
    axis.grid(True)
    figure.savefig(OUTPUT_DIR / output_filename, dpi=300)
    return figure


def main() -> None:
    if not 0.0 < CONVERGED_FRACTION <= 1.0:
        raise ValueError("CONVERGED_FRACTION must be greater than zero and at most one.")

    records = [record for case in CASES if (record := calculate_converged_coefficients(case))]
    if not records:
        raise FileNotFoundError("No yaw-case total-force histories are available.")

    results = pd.DataFrame(records).sort_values("beta_deg")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "xwing_moments_vs_beta.csv", index=False)

    print("Converged yaw-moment coefficients:")
    print(
        results[["beta_deg", "cmx", "cmz", "converged_sample_count"]]
        .to_string(index=False, float_format=lambda value: f"{value:.8f}")
    )

    figures = [
        plot_moment_vs_beta(
            results,
            "cmx",
            r"Rolling-moment coefficient $C_{Mx}$ [-]",
            "XWing rolling moment versus yaw angle",
            "xwing_cmx_vs_beta.png",
        ),
        plot_moment_vs_beta(
            results,
            "cmz",
            r"Yawing-moment coefficient $C_{Mz}$ [-]",
            "XWing yawing moment versus yaw angle",
            "xwing_cmz_vs_beta.png",
        ),
    ]
    if SHOW_PLOTS and "agg" not in plt.get_backend().lower():
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
