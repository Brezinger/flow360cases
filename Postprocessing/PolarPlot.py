#!/usr/bin/env python3
"""
Plot aerodynamic polars (like XFOIL) from a CSV file.

Supports two sweep modes:

1) Alpha sweeps (sweep_type = "alpha")
   - Left:  CL vs CD
   - Right: CL vs alpha (left axis, solid)
            CMy vs alpha (right axis, dashed)

2) Beta sweeps (sweep_type = "beta")
   - Left:  CY vs CD  (CY taken from CFy column)
   - Right: CY vs beta (left axis, solid)
            CMz vs beta (right axis, dashed)

Additionally:
- Computes Reynolds number based on:
    * specified reference length l_ref [m]
    * ISA conditions at 500 m MSL
- Includes U_inf and Re in the legend for each speed.

Hard-coded configuration in main().
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------- Atmosphere / Reynolds helpers ----------------- #

def isa_500m():
    """
    International Standard Atmosphere at 500 m MSL.

    Returns:
        rho : float
            Air density [kg/m^3]
        mu : float
            Dynamic viscosity [Pa·s]
    """
    # Constants
    T0 = 288.15       # sea level temperature [K]
    p0 = 101325.0     # sea level pressure [Pa]
    L = 0.0065        # temperature lapse rate [K/m]
    R = 287.058       # gas constant for air [J/(kg·K)]
    g0 = 9.80665      # gravity [m/s^2]
    h = 500.0         # altitude [m]

    # Temperature at altitude
    T = T0 - L * h

    # Pressure at altitude (standard troposphere formula)
    p = p0 * (T / T0) ** (g0 / (R * L))

    # Density
    rho = p / (R * T)

    # Dynamic viscosity via Sutherland's law
    # Parameters for air
    mu0 = 1.7894e-5   # reference viscosity [Pa·s] at T0suth
    T0suth = 288.15   # reference temperature [K]
    S = 110.4         # Sutherland temperature [K]

    mu = mu0 * (T / T0suth) ** 1.5 * (T0suth + S) / (T + S)

    return rho, mu


def reynolds_number(U, l_ref, rho, mu):
    """
    Compute Reynolds number Re = rho * U * l_ref / mu.
    """
    return rho * U * l_ref / mu


# --------------------- Data and plotting --------------------- #

def load_data(csv_file: str) -> pd.DataFrame:
    """Load CSV and ensure required columns exist."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {
        "U_inf",
        "alpha_deg",
        "beta_deg",
        "elev_deflection_deg",
        "CL",
        "CD",
        "CFy",
        "CMy",
        "CMz",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        print(f"CSV file missing: {', '.join(sorted(missing))}", file=sys.stderr)
        sys.exit(1)

    return df


def plot_polars(df, speeds, flaps, sweep_type="alpha", l_ref=1.0):
    """
    Create two plots:

    sweep_type="alpha":
        Left:  CL vs CD
        Right: CL vs alpha (solid, left axis)
               CMy vs alpha (dashed, right axis)

    sweep_type="beta":
        Left:  CY vs CD
        Right: CY vs beta (solid, left axis)
               CMz vs beta (dashed, right axis)

    l_ref:
        Reference length [m] used for Reynolds number computation.
    """
    if sweep_type not in ("alpha", "beta"):
        raise ValueError("sweep_type must be 'alpha' or 'beta'")

    # ISA atmosphere at 500 m
    rho, mu = isa_500m()

    # Precompute Re for each speed
    re_by_speed = {}
    for U in speeds:
        Re = reynolds_number(U, l_ref, rho, mu)
        re_by_speed[U] = Re

    # Choose columns and labels depending on sweep type
    if sweep_type == "alpha":
        sweep_col = "alpha_deg"
        sweep_label = r"$\alpha$ [deg]"
        primary_col = "CL"
        primary_label = r"$C_L$ [-]"
        moment_col = "CMy"
        moment_label = r"$C_{My}$ [-]"
        left_ylabel = r"$C_L$ [-]"
        left_title = "Lift Polar (CL vs CD)"
        right_title = "Lift & Moment Curves (vs α)"
        style_main_label = r"$C_L$"
        style_moment_label = r"$C_{My}$"
    else:  # beta
        sweep_col = "beta_deg"
        sweep_label = r"$\beta$ [deg]"
        primary_col = "CFy"     # interpret as CY
        primary_label = r"$C_Y$ [-]"
        moment_col = "CMz"
        moment_label = r"$C_{Mz}$ [-]"
        left_ylabel = r"$C_Y$ [-]"
        left_title = "Side-force Polar (CY vs CD)"
        right_title = "Side-force & Yawing Moment (vs β)"
        style_main_label = r"$C_Y$"
        style_moment_label = r"$C_{Mz}$"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_left = axes[0]
    ax_primary = axes[1]
    ax_moment = ax_primary.twinx()

    num_plotted = 0
    legend_handles = []
    legend_labels = []

    for U in speeds:
        for flap in flaps:
            mask = (df["U_inf"] == U) & (df["elev_deflection_deg"] == flap)
            sub = df.loc[mask].copy()
            if sub.empty:
                continue

            # Sort by sweep variable for nice curves
            sub.sort_values(sweep_col, inplace=True)

            Re = re_by_speed[U]
            # Legend label with U, Re, flap
            label = f"U={U:g} m/s, Re={Re/1e6:.2f}e6, δ={flap:g}°"

            # ---- Left: primary vs CD ----
            h_main, = ax_left.plot(
                sub["CD"],
                sub[primary_col],
                marker="o",
                linestyle="-",
                label=label,
            )

            color = h_main.get_color()

            # ---- Right: primary vs sweep_var (solid) ----
            ax_primary.plot(
                sub[sweep_col],
                sub[primary_col],
                marker="o",
                linestyle="-",
                color=color,
            )

            # ---- Right: moment vs sweep_var (dashed) ----
            ax_moment.plot(
                sub[sweep_col],
                sub[moment_col],
                marker="s",
                linestyle="--",
                color=color,
            )

            legend_handles.append(h_main)
            legend_labels.append(label)
            num_plotted += 1


    if num_plotted == 0:
        print("No matching U_inf / flap combination found.")
        sys.exit(1)

    # ---------- Axes formatting ----------
    # Left
    ax_left.set_xlabel(r"$C_D$ [-]")
    ax_left.set_ylabel(left_ylabel)
    ax_left.set_title(left_title)
    ax_left.grid(True, linestyle="--", alpha=0.5)

    # Right
    ax_primary.set_xlabel(sweep_label)
    ax_primary.set_ylabel(primary_label)
    ax_primary.set_title(right_title)
    ax_primary.grid(True, linestyle="--", alpha=0.5)

    ax_moment.set_ylabel(moment_label)

    # ---------- Legend for color (U, Re, flap combinations) ----------
    legend = ax_left.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        frameon=True,
    )
    legend.set_zorder(99)

    # ---------- Legend for line styles (in right plot) ----------
    style_handles = [
        Line2D(
            [0],
            [0],
            color="0.3",
            linestyle="-",
            marker="o",
            label=style_main_label,
        ),
        Line2D(
            [0],
            [0],
            color="0.3",
            linestyle="--",
            marker="s",
            label=style_moment_label,
        ),
    ]
    legend_linestyle = ax_moment.legend(
        style_handles,
        [style_main_label, style_moment_label],
        title="Line style",
        loc="lower right",
        frameon=True,
    )

    legend_linestyle.set_zorder(10)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig



def main():
    # ------------------------------------------------------------
    # HARD-CODE YOUR SETTINGS HERE
    # ------------------------------------------------------------

    study = "beta" # "flap", "Airspeeds", "beta"

    if study == "Airspeeds":
        csv_path = "C:/WDIR/flow360/AMDC/GBT_parametric_study.csv"  # <-- change to your filename
        sweep_type = "alpha"

        # flap study
        speeds = [100, 150, 200, 250]   # or [100], or [100,150], etc.
        flaps = [0.0]  # default like "flap 0" in XFOIL
    elif study == "flap":
        csv_path = "C:/WDIR/flow360/AMDC/GBT_parametric_study.csv"  # <-- change to your filename
        sweep_type = "alpha"

        # flap study
        speeds = [100]  # or [100], or [100,150], etc.
        flaps = [0.0, 2.5, 5, 7.5, 10]  # default like "flap 0" in XFOIL
    elif study == "beta":
        sweep_type = "beta"
        csv_path = "C:/WDIR/flow360/AMDC/GBT_asym_parametric_study_extended.csv"  # <-- change to your filename"""
        speeds = [100, 150, 200, 250]
        flaps = [0.0]

    # Reference length for Reynolds number [m]
    l_ref = 0.756

    # Output file (set to None to not save)
    output_file = None
    # output_file = "polars_alpha.png"

    # -----------------------------------------

    df = load_data(csv_path)

    # for symmetric cases (beta = 0): double all force coefficients. Set cmx, cmz to zero
    df.loc[df.beta_deg == 0, ['CL', 'CD', 'CMy', 'CFx', 'CFz']] *=2
    df.loc[df.beta_deg == 0, ['CMx', 'CMz', 'CFy']] = 0

    # Print Re table for info
    rho, mu = isa_500m()
    print(f"ISA at 500 m: rho = {rho:.4f} kg/m^3, mu = {mu:.2e} Pa·s")
    print("Reynolds numbers:")
    for U in speeds:
        Re = reynolds_number(U, l_ref, rho, mu)
        print(f"  U = {U:6.1f} m/s -> Re = {Re: .3e}")

    fig = plot_polars(df, speeds, flaps, sweep_type=sweep_type, l_ref=l_ref)

    if output_file:
        print(f"Saving to {output_file}")
        fig.savefig(output_file, dpi=300)

    plt.show()

    pass


if __name__ == "__main__":
    main()
