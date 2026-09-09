"""Plot XWing pitching moment versus lift and estimate the neutral point.

The script reads the Flow360 total-force histories for the rectangular and
trapezoidal XWing cases.  It averages the final portion of each steady history,
fits total-aircraft ``CMy`` as a function of ``CL``, and estimates the neutral
point from the fitted slope.  Results are written to
``<flow360-root>/cm_vs_cl``.
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
OUTPUT_DIR = FLOW360_ROOT / "cm_vs_cl"
SHOW_PLOT = True
CONVERGED_FRACTION = 0.10

# These match the Flow360 ReferenceGeometry used to run the XWing cases.
MOMENT_REFERENCE_X_MM = 743.0
MEAN_AERODYNAMIC_CHORD_MM = 108.0


@dataclass(frozen=True)
class CaseDefinition:
    """One XWing simulation contributing a point to a Cm(CL) curve."""

    wing_type: str
    airspeed_m_s: float
    result_directory: Path

    @property
    def force_history_files(self) -> tuple[Path, ...]:
        return tuple(sorted(self.result_directory.glob("*_total_forces_v2.csv")))


CASES = (
    CaseDefinition(
        "Rectangular wing",
        24.5,
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U24.5_AOA10",
    ),
    CaseDefinition(
        "Rectangular wing",
        35.0,
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U35",
    ),
    CaseDefinition(
        "Rectangular wing",
        39.5,
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U39.5_AOA-1.6",
    ),
    CaseDefinition(
        "Trapezoidal wing",
        24.5,
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2 fully_turbulent_SA U24.5_AOA10",
    ),
    CaseDefinition(
        "Trapezoidal wing",
        35.0,
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2_fully_turbulent_SA U35",
    ),
    CaseDefinition(
        "Trapezoidal wing",
        39.5,
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2 fully_turbulent_SA U39.5_AOA-1.6",
    ),
)


def _force_history_file(case: CaseDefinition) -> Path:
    """Return the sole total-force history associated with a case."""
    matches = case.force_history_files
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*_total_forces_v2.csv' in {case.result_directory}, "
            f"found {len(matches)}."
        )
    return matches[0]


def calculate_converged_coefficients(case: CaseDefinition) -> dict[str, object]:
    """Average CL and CMy over the final converged portion of one force history."""
    force_history_file = _force_history_file(case)
    history = pd.read_csv(force_history_file, skipinitialspace=True)
    required_columns = {"CL", "CMy"}
    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        raise ValueError(
            f"{force_history_file} is missing columns: {sorted(missing_columns)}"
        )

    coefficients = history[["CL", "CMy"]].apply(pd.to_numeric, errors="coerce").dropna()
    if coefficients.empty:
        raise ValueError(f"{force_history_file} contains no finite CL/CMy samples.")

    sample_count = max(1, int(np.ceil(len(coefficients) * CONVERGED_FRACTION)))
    converged_samples = coefficients.tail(sample_count)
    return {
        "wing_type": case.wing_type,
        "airspeed_m_s": case.airspeed_m_s,
        "cl": float(converged_samples["CL"].mean()),
        "cm": float(converged_samples["CMy"].mean()),
        "converged_sample_count": sample_count,
        "force_history_file": str(force_history_file),
    }


def fit_neutral_point(group: pd.DataFrame) -> dict[str, float]:
    """Fit Cm(CL) and derive the neutral point relative to the moment reference."""
    cl = group["cl"].to_numpy(dtype=float)
    cm = group["cm"].to_numpy(dtype=float)
    if len(cl) < 2 or np.unique(cl).size < 2:
        wing_type = group["wing_type"].iloc[0]
        raise ValueError(f"{wing_type} needs at least two distinct CL values for a fit.")

    cm_slope, cm_intercept = np.polyfit(cl, cm, deg=1)
    fitted_cm = cm_slope * cl + cm_intercept
    residual_sum_squares = float(np.sum((cm - fitted_cm) ** 2))
    total_sum_squares = float(np.sum((cm - cm.mean()) ** 2))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if not np.isclose(total_sum_squares, 0.0)
        else float("nan")
    )

    neutral_point_x_mm = MOMENT_REFERENCE_X_MM - MEAN_AERODYNAMIC_CHORD_MM * cm_slope
    static_margin_pct_mac = (
        (neutral_point_x_mm - MOMENT_REFERENCE_X_MM)
        / MEAN_AERODYNAMIC_CHORD_MM
        * 100.0
    )
    return {
        "cm_slope_per_cl": float(cm_slope),
        "cm_intercept": float(cm_intercept),
        "r_squared": r_squared,
        "neutral_point_x_mm": float(neutral_point_x_mm),
        "neutral_point_x_m": float(neutral_point_x_mm / 1000.0),
        "static_margin_pct_mac": float(static_margin_pct_mac),
    }


def main() -> None:
    if not 0.0 < CONVERGED_FRACTION <= 1.0:
        raise ValueError("CONVERGED_FRACTION must be greater than zero and at most one.")

    records = [calculate_converged_coefficients(case) for case in CASES]
    results = pd.DataFrame(records).sort_values(["wing_type", "airspeed_m_s"])

    fit_by_wing_type = {
        wing_type: fit_neutral_point(group)
        for wing_type, group in results.groupby("wing_type", sort=False)
    }
    for wing_type, fit in fit_by_wing_type.items():
        for column, value in fit.items():
            results.loc[results["wing_type"] == wing_type, column] = value
        print(
            f"{wing_type}: dCm/dCL={fit['cm_slope_per_cl']:.5f}, "
            f"x_NP={fit['neutral_point_x_m']:.4f} m, "
            f"static margin={fit['static_margin_pct_mac']:.2f}% MAC, "
            f"R^2={fit['r_squared']:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "xwing_cm_vs_cl.csv", index=False)

    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for wing_type, group in results.groupby("wing_type", sort=False):
        fit = fit_by_wing_type[wing_type]
        scatter = axis.scatter(group["cl"], group["cm"], s=45, label=wing_type)
        cl_fit = np.linspace(group["cl"].min(), group["cl"].max(), 100)
        axis.plot(
            cl_fit,
            fit["cm_slope_per_cl"] * cl_fit + fit["cm_intercept"],
            color=scatter.get_facecolor()[0],
            linewidth=1.8,
            label=(
                f"{wing_type} fit "
                f"(x_NP={fit['neutral_point_x_m']:.3f} m"
                #f", SM={fit['static_margin_pct_mac']:.1f}% MAC
                f")"
            ),
        )

    axis.set_xlabel(r"Lift coefficient $C_L$")
    axis.set_ylabel(r"Pitching-moment coefficient $C_m$ ($CMy$)")
    axis.set_title("XWing pitching moment versus lift coefficient")
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.grid(True)
    axis.legend(fontsize="small")
    figure.savefig(OUTPUT_DIR / "xwing_cm_vs_cl.png", dpi=300)
    if SHOW_PLOT and "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
