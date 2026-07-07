from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# User inputs
# -----------------------------
"""data_dir = Path(
    "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360/"
    "rectangular wing/XWing2_2 fully_turbulent_SA U24.5_AOA10"
)"""
data_dir = Path(
    "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360/"
    "rectangular wing/XWing2_2 fully_turbulent_SA U39.5_AOA-1.6"
)
wing_files = [data_dir / f"wing{index}_data.csv" for index in range(1, 5)]
wing_lift_force_direction_deg = {
    "wing1_data": 125.0,
    "wing2_data": 55.0,
    "wing3_data": 125.0,
    "wing4_data": 55.0,
}

dy = 25.0                      # strip width in same length unit as Points
S_ref = 0.283098946506276 * 1e6  # reference area, set correctly
b_ref = 658.54                 # reference span for Cmx, set correctly

# moment reference point
x_ref = 0.0
y_ref = 0.0
z_ref = 0.0


def process_wing(
    filepath: Path,
    lift_force_direction_deg: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = pd.read_csv(filepath)

    area = df["Area"].to_numpy()
    cp = df["Cp"].to_numpy()

    nx = df["Normals:0"].to_numpy()
    ny = df["Normals:1"].to_numpy()
    nz = df["Normals:2"].to_numpy()

    y = df["Points:1"].to_numpy() - y_ref
    z = df["Points:2"].to_numpy() - z_ref

    gamma = np.deg2rad(lift_force_direction_deg)
    force_direction = np.array([0.0, np.cos(gamma), np.sin(gamma)])
    ex, ey, ez = force_direction

    # Scalar pressure force in the inclined wing-local lift direction, normalized by q_inf.
    n_dot_ef = nx * ex + ny * ey + nz * ez
    dFn_q = -cp * area * n_dot_ef

    # Global pressure-force components, normalized by q_inf.
    dFy_q = -cp * area * ny
    dFz_q = -cp * area * nz

    # Rolling moment around x-axis: Mx = y Fz - z Fy.
    dMx_q = y * dFz_q - z * dFy_q

    df["dFn_q"] = dFn_q
    df["dCFn"] = dFn_q / S_ref
    df["dMx_q"] = dMx_q
    df["dCmx"] = dMx_q / (S_ref * b_ref)

    cmx_total = float(df["dCmx"].sum())

    y_min = df["Points:1"].min()
    df["strip"] = np.floor((df["Points:1"] - y_min) / dy).astype(int)

    strip = (
        df.groupby("strip")
        .agg(
            y_mid=("Points:1", "mean"),
            area=("Area", "sum"),
            Fn_q=("dFn_q", "sum"),
            CFn=("dCFn", "sum"),
            Mx_q=("dMx_q", "sum"),
            Cmx=("dCmx", "sum"),
        )
        .reset_index()
    )

    strip["dFn_dy_q"] = strip["Fn_q"] / dy
    strip["dCFn_dy"] = strip["CFn"] / dy
    strip["dCmx_dy"] = strip["Cmx"] / dy

    return df, strip, cmx_total


def main() -> None:
    wing_results: list[tuple[str, pd.DataFrame, float]] = []

    for wing_file in wing_files:
        lift_direction = wing_lift_force_direction_deg[wing_file.stem]
        _, strip, cmx_total = process_wing(wing_file, lift_direction)
        wing_results.append((wing_file.stem, strip, cmx_total))
        print(
            f"{wing_file.stem}: lift direction = {lift_direction:.1f} deg, "
            f"Cmx = {cmx_total:.6f}"
        )

    cmx_sum = sum(cmx_total for _, _, cmx_total in wing_results)
    print(f"Sum Cmx = {cmx_sum:.6f}")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    wing_colors = {
        wing_name: color_cycle[index % len(color_cycle)]
        for index, (wing_name, _, _) in enumerate(wing_results)
    }

    fig_cmx = plt.figure()
    for wing_name, strip, _ in wing_results:
        plt.plot(
            strip["y_mid"],
            strip["dCmx_dy"],
            marker="o",
            label=_wing_label(wing_name),
            color=wing_colors[wing_name],
        )
    plt.xlabel("y")
    plt.ylabel(r"$dC_{mx}/dy$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_cmx.savefig(data_dir / "wing_dCmx_dy.png", dpi=300)


    fig_cfn = plt.figure()
    for wing_name, strip, _ in wing_results:
        plt.plot(
            strip["y_mid"],
            strip["dCFn_dy"],
            marker="o",
            label=_wing_label(wing_name),
            color=wing_colors[wing_name],
        )
    plt.xlabel("y")
    plt.ylabel(r"$dC_{F_n}/dy$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig_cfn.savefig(data_dir / "wing_dCFn_dy.png", dpi=300)

    fig_cmx_bars = plt.figure()
    row_positions = {
        "wing1_data": 2.15,
        "wing2_data": 1.85,
        "sum_1_2": 1.0,
        "sum_3_4": 0.7,
        "wing3_data": -0.15,
        "wing4_data": -0.45,
    }
    max_abs_cmx = max(abs(cmx_total) for _, _, cmx_total in wing_results)
    cmx_by_wing = {
        wing_name: cmx_total
        for wing_name, _, cmx_total in wing_results
    }
    pair_sum_results = [
        (
            "sum_1_2",
            "wing1 + wing2",
            cmx_by_wing["wing1_data"] + cmx_by_wing["wing2_data"],
            wing_colors["wing1_data"],
        ),
        (
            "sum_3_4",
            "wing3 + wing4",
            cmx_by_wing["wing3_data"] + cmx_by_wing["wing4_data"],
            wing_colors["wing3_data"],
        ),
    ]
    min_bar_value = min(0.0, *(diff_value for _, _, diff_value, _ in pair_sum_results))
    max_bar_value = max(
        max_abs_cmx,
        *(diff_value for _, _, diff_value, _ in pair_sum_results),
    )
    max_abs_for_offset = max(abs(min_bar_value), abs(max_bar_value))
    label_offset = 0.02 * max_abs_for_offset if max_abs_for_offset > 0.0 else 0.01

    for wing_name, _, cmx_total in wing_results:
        abs_cmx = abs(cmx_total)
        plt.barh(
            row_positions[wing_name],
            abs_cmx,
            height=0.22,
            color=wing_colors[wing_name],
            label=_wing_label(wing_name),
        )
        plt.text(
            abs_cmx + label_offset,
            row_positions[wing_name],
            f"{cmx_total:.6f}",
            va="center",
        )

    for diff_name, label, diff_value, color in pair_sum_results:
        plt.barh(
            row_positions[diff_name],
            diff_value,
            height=0.22,
            color=color,
            label=label,
        )
        plt.text(
            diff_value + label_offset if diff_value >= 0.0 else diff_value - label_offset,
            row_positions[diff_name],
            f"{diff_value:.6f}",
            va="center",
            ha="left" if diff_value >= 0.0 else "right",
        )

    plt.yticks(
        [2.0, 0.85, -0.3],
        ["wings 1 + 2", "pair sums", "wings 3 + 4"],
    )
    plt.xlabel(r"$C_{mx}$ pair sum / $|C_{mx}|$ wing magnitude")
    plt.xlim(min_bar_value - 8.0 * label_offset, max_bar_value + 8.0 * label_offset)
    plt.axvline(0.0, color="0.25", linewidth=0.8)
    plt.grid(True, axis="x")
    plt.legend()
    plt.tight_layout()
    fig_cmx_bars.savefig(data_dir / "wing_cmx_barchart.png", dpi=300)
    plt.show()


def _wing_label(wing_name: str) -> str:
    return wing_name.removesuffix("_data")


if __name__ == "__main__":
    main()
