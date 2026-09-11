"""Create spanwise lift and rolling-moment distributions from Flow360 surface results.

How to prepare and run the analysis
-----------------------------------
1. In Flow360, download the surface-result data into the case result directory and
   extract the downloaded ``.tar.gz`` archive. The extracted directory must contain
   ``surfaces.vtu``.
2. Copy ``Extract_wings_stabs.pvsm`` to that same result directory.
3. To create the eight PatchID selection CSV files automatically, start ParaView 5.13.1
   and open ``Tools -> Python Script Editor``. Paste and run the following launcher,
   replacing ``<result-directory>`` with the directory from steps 1 and 2::

       import runpy
       import sys

       sys.argv = [
           "export_wing_stab_patch_ids.py",
           "--data-dir",
           r"C:\\path\\to\\<result-directory>",
       ]

       runpy.run_path(
           r"C:\\git\\flow360cases\\Postprocessing\\export_wing_stab_patch_ids.py",
           run_name="__main__",
       )

   The launcher runs ``export_wing_stab_patch_ids.py`` with ParaView's embedded Python.
   It loads ``Extract_wings_stabs.pvsm``, reconnects it to ``surfaces.vtu``, and exports
   ``wing1_data.csv`` through ``wing4_data.csv`` plus ``stab1_data.csv`` through
   ``stab4_data.csv``. The state file must contain filters named ``Extract wing1`` through
   ``Extract wing4`` and ``Extract stab1`` through ``Extract stab4``.
4. Select one result label or a list of result labels below, then run this script.
   A list overlays the selected cases in every figure and writes the comparison plots
   to ``<flow360-root>/comparison``.

The script reads aerodynamic cell data from ``surfaces.vtu`` and combines it with the
eight ``*_data.csv`` PatchID selections to calculate and plot spanwise distributions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


# -----------------------------
# User inputs
# -----------------------------
# result = "XWing 2.2 rect 24.5"
# result = "XWing 2.2 rect 24.5 fine"
# result = "XWing 2.2 rect 35"
# result = "XWing 2.2 rect 39"
# result = "XWing 2.2 rect 39 twisted"
# result =  "XWing 2.2 trap 24.5"
result =  "XWing 2.2 trap 35"
# result = "XWing 2.2 trap 39"
#result = ["XWing 2.2 rect 39", "XWing 2.2 rect 39 twisted"]

show_plots = True
mirror_one_sided_spanwise_results = True

FLOW360_ROOT = Path("C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360")
COMPARISON_OUTPUT_DIR = FLOW360_ROOT / "comparison"
DY = 25.0  # Strip width in model length units.


@dataclass(frozen=True)
class ResultConfiguration:
    label: str
    data_dir: Path
    reference_area_mm2: float
    reference_span_mm: float


RESULT_CONFIGURATIONS = {
    "XWing 2.2 rect 24.5": ResultConfiguration(
        "XWing 2.2 rect 24.5",
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U24.5_AOA10",
        0.2831e6,
        1312.0,
    ),
    "XWing 2.2 rect 24.5 fine": ResultConfiguration(
        "XWing 2.2 rect 24.5 fine",
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U24.5_AOA10_fine mesh",
        0.2831e6,
        1312.0,
    ),
    "XWing 2.2 rect 35": ResultConfiguration(
        "XWing 2.2 rect 35",
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U35",
        0.2831e6,
        1312.0,
    ),
    "XWing 2.2 rect 39": ResultConfiguration(
        "XWing 2.2 rect 39",
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U39.5_AOA-1.6",
        0.2831e6,
        1312.0,
    ),
    "XWing 2.2 rect 39 twisted": ResultConfiguration(
        "XWing 2.2 rect 39 incidence wing3 +0.5°",
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U39.5_twisted",
        0.2831e6,
        1312.0,
    ),
    "XWing 2.2 trap 24.5": ResultConfiguration(
        "XWing 2.2 trap 24.5",
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2 fully_turbulent_SA U24.5_AOA10",
        0.277649e6,
        1346.0,
    ),
    "XWing 2.2 trap 35": ResultConfiguration(
        "XWing 2.2 trap 35",
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2_fully_turbulent_SA U35",
        0.277649e6,
        1346.0,
    ),
    "XWing 2.2 trap 39": ResultConfiguration(
        "XWing 2.2 trap 39",
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2 fully_turbulent_SA U39.5_AOA-1.6",
        0.277649e6,
        1346.0,
    ),
}

i_wing_offset = [1, 2, 3, 4]
wing_tipward_angle_deg = 35.0

P_mot = 353.7    # Motor shaft power
rpm_mot = 7711 # Motor RPM
motor_torque_sign = 1.0  # +1 means motor torque adds positive Cmx.
rho_inf_kg_m3 = 1.225
length_unit_m = 1.0e-3


surface_names = [
    *(f"wing{index}" for index in range(1, 5)),
    *(f"stab{index}" for index in range(1, 5)),
]
surface_lift_force_direction_deg = {
    "wing1": 125.0,
    "wing2": 55.0,
    "wing3": 125.0,
    "wing4": 55.0,
    "stab1": 125.0,
    "stab2": 55.0,
    "stab3": 125.0,
    "stab4": 55.0,
}

SURFACE_COLORS = {
    "wing1": "red", "stab1": "red",
    "wing2": "green", "stab2": "green",
    "wing3": "blue", "stab3": "blue",
    "wing4": "black", "stab4": "black",
}

# moment reference point
x_ref = 743
y_ref = 0.0
z_ref = 0.0


@dataclass(frozen=True)
class WingOffsetResult:
    wing_names: tuple[str, ...]
    selected_fz_q: float
    aerodynamic_cmx: float
    motor_torque_nm: float
    motor_torque_cmx: float
    target_cmx: float
    required_delta_cmx: float
    offset_y: float
    offset_y_m: float
    selected_mean_y: float
    tipward_offset: float
    tipward_offset_m: float


@dataclass(frozen=True)
class AnalysisResult:
    configuration: ResultConfiguration
    surface_results: tuple[tuple[str, pd.DataFrame, pd.DataFrame, float], ...]
    component_sums: dict[str, float]
    aircraft_cl: float
    wing_offset: WingOffsetResult


def process_surface_dataframe(
    df: pd.DataFrame,
    lift_force_direction_deg: float,
    reference_area_mm2: float,
    reference_span_mm: float,
    force_mirror_spanwise_results: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    area = df["Area"].to_numpy()
    cp = df["Cp"].to_numpy()

    nx = df["Normals:0"].to_numpy()
    ny = df["Normals:1"].to_numpy()
    nz = df["Normals:2"].to_numpy()
    cfx = _optional_column(df, "CfVec:0")
    cfy = _optional_column(df, "CfVec:1")
    cfz = _optional_column(df, "CfVec:2")

    y = df["Points:1"].to_numpy() - y_ref
    z = df["Points:2"].to_numpy() - z_ref

    gamma = np.deg2rad(lift_force_direction_deg)
    force_direction = np.array([0.0, np.cos(gamma), np.sin(gamma)])
    ex, ey, ez = force_direction

    # Surface force vector normalized by q_inf: pressure plus skin friction.
    dFx_q = area * (-cp * nx + cfx)
    dFy_q = area * (-cp * ny + cfy)
    dFz_q = area * (-cp * nz + cfz)

    # Scalar force in the inclined surface-local normal direction.
    dFn_q = dFx_q * ex + dFy_q * ey + dFz_q * ez

    # Rolling moment around x-axis: Mx = y Fz - z Fy.
    dMx_q = y * dFz_q - z * dFy_q

    df["dFx_q"] = dFx_q
    df["dFy_q"] = dFy_q
    df["dFz_q"] = dFz_q
    df["dFn_q"] = dFn_q
    df["dMx_q"] = dMx_q
    df["dCmx"] = dMx_q / (reference_area_mm2 * reference_span_mm)
    df["dA_xy"] = area * np.abs(nz)

    cmx_total = float(df["dCmx"].sum())

    y_min = df["Points:1"].min()
    df["strip"] = np.floor((df["Points:1"] - y_min) / DY).astype(int)

    strip = (
        df.groupby("strip")
        .agg(
            y_mid=("Points:1", "mean"),
            area=("Area", "sum"),
            Fn_q=("dFn_q", "sum"),
            Mx_q=("dMx_q", "sum"),
            Cmx=("dCmx", "sum"),
            area_xy=("dA_xy", "sum"),
        )
        .reset_index()
    )

    strip["dCmx_dy"] = strip["Cmx"] / DY
    strip["cl_local"] = np.where(
        strip["area_xy"] > 0.0,
        strip["Fn_q"] / strip["area_xy"],
        np.nan,
    )
    if mirror_one_sided_spanwise_results:
        strip = mirror_one_sided_spanwise_distribution(
            strip,
            force_mirror=force_mirror_spanwise_results,
        )

    return df, strip, cmx_total


def mirror_one_sided_spanwise_distribution(
    strip: pd.DataFrame,
    tolerance: float = 1.0e-9,
    force_mirror: bool = False,
) -> pd.DataFrame:
    """Mirror a spanwise distribution about y=0.

    By default, only one-sided distributions are mirrored. Set ``force_mirror`` for
    surfaces that cross y=0 but still require a symmetric visual representation.
    """
    if strip.empty:
        return strip.assign(is_mirrored=False)

    original = strip.copy()
    original["is_mirrored"] = False

    y_mid = original["y_mid"].to_numpy()
    has_negative_y = np.any(y_mid < -tolerance)
    has_positive_y = np.any(y_mid > tolerance)
    if has_negative_y and has_positive_y and not force_mirror:
        return original

    rows_to_mirror = original[np.abs(original["y_mid"]) > tolerance]
    if rows_to_mirror.empty:
        return original

    mirrored = rows_to_mirror.copy()
    mirrored["is_mirrored"] = True
    mirrored["y_mid"] = -mirrored["y_mid"]
    mirrored["strip"] = -mirrored["strip"] - 1
    for column in ("Mx_q", "Cmx", "dCmx_dy"):
        if column in mirrored:
            mirrored[column] = -mirrored[column]

    return (
        pd.concat([original, mirrored], ignore_index=True)
        .sort_values("y_mid")
        .reset_index(drop=True)
    )


def load_vtu_cell_center_dataframe(filepath: Path) -> pd.DataFrame:
    mesh = pv.read(str(filepath)).extract_surface(algorithm="dataset_surface")
    mesh = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        inplace=False,
    )
    mesh = mesh.point_data_to_cell_data(pass_point_data=False)
    mesh = mesh.compute_cell_sizes(length=False, volume=False)
    centers = mesh.cell_centers().points

    data = {
        "PatchID": mesh.cell_data["PatchID"],
        "Cp": mesh.cell_data["Cp"],
        "Area": mesh.cell_data["Area"],
        "Points:0": centers[:, 0],
        "Points:1": centers[:, 1],
        "Points:2": centers[:, 2],
    }

    normals = mesh.cell_data["Normals"]
    for index in range(3):
        data[f"Normals:{index}"] = normals[:, index]

    if "CfVec" in mesh.cell_data:
        cf_vec = mesh.cell_data["CfVec"]
        for index in range(3):
            data[f"CfVec:{index}"] = cf_vec[:, index]

    return pd.DataFrame(data)


def patch_ids_from_selection_file(filepath: Path) -> np.ndarray:
    df = pd.read_csv(filepath, usecols=["PatchID"])
    return np.sort(df["PatchID"].dropna().astype(int).unique())


def process_vtu_surface(
    vtu_cell_df: pd.DataFrame,
    patch_id_file: Path,
    lift_force_direction_deg: float,
    reference_area_mm2: float,
    reference_span_mm: float,
    force_mirror_spanwise_results: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    surface_df = select_vtu_cells_from_selection_file(vtu_cell_df, patch_id_file)
    if surface_df.empty:
        raise ValueError(f"No VTU cells matched PatchID values from {patch_id_file}.")
    return process_surface_dataframe(
        surface_df,
        lift_force_direction_deg,
        reference_area_mm2,
        reference_span_mm,
        force_mirror_spanwise_results=force_mirror_spanwise_results,
    )


def select_vtu_cells_from_selection_file(
    vtu_cell_df: pd.DataFrame,
    selection_file: Path,
    coordinate_tolerance: float = 1.0e-2,
) -> pd.DataFrame:
    selection_df = pd.read_csv(selection_file)
    point_columns = ["Points:0", "Points:1", "Points:2"]
    if set(point_columns).issubset(selection_df.columns):
        matched_indices: list[int] = []
        for patch_id, patch_selection_df in selection_df.groupby("PatchID"):
            patch_vtu_df = vtu_cell_df[vtu_cell_df["PatchID"] == int(patch_id)]
            if patch_vtu_df.empty:
                continue

            tree = cKDTree(patch_vtu_df[point_columns].to_numpy())
            distances, local_indices = tree.query(
                patch_selection_df[point_columns].to_numpy(),
                k=1,
            )
            if np.any(distances > coordinate_tolerance):
                max_distance = float(distances.max())
                raise ValueError(
                    f"Selection file {selection_file} has cell centers that do not "
                    f"match PatchID {patch_id} within tolerance "
                    f"{coordinate_tolerance}. Maximum distance: {max_distance}."
                )
            matched_indices.extend(patch_vtu_df.index.to_numpy()[local_indices])

        return vtu_cell_df.loc[np.unique(matched_indices)].copy()

    patch_ids = np.sort(selection_df["PatchID"].dropna().astype(int).unique())
    return vtu_cell_df[vtu_cell_df["PatchID"].isin(patch_ids)].copy()


def _wing_names_from_indices(wing_indices: list[int]) -> tuple[str, ...]:
    wing_names = tuple(f"wing{index}" for index in wing_indices)
    missing_wings = [wing_name for wing_name in wing_names if wing_name not in surface_names]
    if missing_wings:
        raise ValueError(f"Unknown wings requested for y-offset: {missing_wings}")
    return wing_names


def _freestream_speed_m_s_from_result(result_label: str, path: Path) -> float:
    for text in (result_label, path.name):
        speed = _regex_group(text, r"U([-+]?\d+(?:\.\d+)?)")
        if speed is not None:
            return float(speed)
    raise ValueError(
        "Could not infer freestream speed from result label or data directory. "
        "Add a U-value to the case name or pass an explicit speed in the code."
    )


def _motor_torque_nm(power_w: float, rpm: float) -> float:
    if rpm <= 0.0:
        raise ValueError(f"Motor RPM must be positive, got {rpm}.")
    angular_speed_rad_s = 2.0 * np.pi * rpm / 60.0
    return power_w / angular_speed_rad_s


def _moment_coefficient_from_torque_nm(
    torque_nm: float,
    q_inf_pa: float,
    s_ref: float,
    b_ref: float,
    length_unit_to_m: float,
) -> float:
    s_ref_m2 = s_ref * length_unit_to_m**2
    b_ref_m = b_ref * length_unit_to_m
    denominator = q_inf_pa * s_ref_m2 * b_ref_m
    if np.isclose(denominator, 0.0):
        raise ValueError("Cannot convert motor torque to Cmx with zero q*S_ref*b_ref.")
    return torque_nm / denominator


def _rolling_moment_nm(
    cmx: float,
    freestream_speed_m_s: float,
    reference_area_mm2: float,
    reference_span_mm: float,
) -> float:
    """Convert aerodynamic Cmx to its signed dimensional rolling moment."""
    dynamic_pressure_pa = 0.5 * rho_inf_kg_m3 * freestream_speed_m_s**2
    reference_area_m2 = reference_area_mm2 * length_unit_m**2
    reference_span_m = reference_span_mm * length_unit_m
    return cmx * dynamic_pressure_pa * reference_area_m2 * reference_span_m


def calculate_required_wing_y_offset(
    surface_results: list[tuple[str, pd.DataFrame, pd.DataFrame, float]],
    wing_indices: list[int],
    aerodynamic_cmx: float,
    power_w: float,
    rpm: float,
    torque_sign: float,
    rho_kg_m3: float,
    freestream_speed_m_s: float,
    reference_area_mm2: float,
    reference_span_mm: float,
    length_unit_to_m: float,
    wing_tipward_angle_deg: float,
) -> WingOffsetResult:
    wing_names = _wing_names_from_indices(wing_indices)
    df_by_surface = {
        surface_name: surface_df
        for surface_name, surface_df, _, _ in surface_results
    }
    selected_fz_q = float(
        sum(df_by_surface[wing_name]["dFz_q"].sum() for wing_name in wing_names)
    )
    if np.isclose(selected_fz_q, 0.0):
        raise ValueError(
            f"Selected wings {wing_names} have near-zero summed dFz_q; cannot calculate a y-offset."
        )
    selected_mean_y = float(
        np.mean(
            np.concatenate(
                [
                    df_by_surface[wing_name]["Points:1"].to_numpy()
                    for wing_name in wing_names
                ]
            )
        )
    )

    q_inf_pa = 0.5 * rho_kg_m3 * freestream_speed_m_s**2
    signed_motor_torque_nm = torque_sign * _motor_torque_nm(power_w, rpm)
    motor_torque_cmx = _moment_coefficient_from_torque_nm(
        signed_motor_torque_nm,
        q_inf_pa,
        reference_area_mm2,
        reference_span_mm,
        length_unit_to_m,
    )
    target_cmx = aerodynamic_cmx + motor_torque_cmx
    required_delta_cmx = -target_cmx
    offset_y = required_delta_cmx * reference_area_mm2 * reference_span_mm / selected_fz_q
    tipward_sign = np.sign(selected_mean_y) if not np.isclose(selected_mean_y, 0.0) else np.nan
    y_projection = np.cos(np.deg2rad(wing_tipward_angle_deg))
    if np.isclose(y_projection, 0.0):
        raise ValueError("wing_tipward_angle_deg must not be 90 degrees modulo 180 degrees.")
    tipward_offset = offset_y / tipward_sign / y_projection

    return WingOffsetResult(
        wing_names=wing_names,
        selected_fz_q=selected_fz_q,
        aerodynamic_cmx=aerodynamic_cmx,
        motor_torque_nm=signed_motor_torque_nm,
        motor_torque_cmx=motor_torque_cmx,
        target_cmx=target_cmx,
        required_delta_cmx=required_delta_cmx,
        offset_y=offset_y,
        offset_y_m=offset_y * length_unit_to_m,
        selected_mean_y=selected_mean_y,
        tipward_offset=tipward_offset,
        tipward_offset_m=tipward_offset * length_unit_to_m,
    )


def print_required_wing_offset(
    result_label: str,
    wing_offset: WingOffsetResult,
    freestream_speed_m_s: float,
) -> None:
    """Print the required wing translation that balances roll and motor torque."""
    print(
        f"{result_label} - required wing y-offset to counter aircraft Cmx plus motor torque:\n"
        f"  selected wings = {', '.join(wing_offset.wing_names)}\n"
        f"  freestream speed = {freestream_speed_m_s:.3f} m/s, "
        f"rho = {rho_inf_kg_m3:.3f} kg/m^3\n"
        f"  selected sum(Fz/q) = {wing_offset.selected_fz_q:.6e}\n"
        f"  aerodynamic Cmx = {wing_offset.aerodynamic_cmx:.6f}\n"
        f"  motor torque = {wing_offset.motor_torque_nm:.6f} N m\n"
        f"  motor torque Cmx = {wing_offset.motor_torque_cmx:.6f}\n"
        f"  target Cmx = {wing_offset.target_cmx:.6f}\n"
        f"  required delta Cmx = {wing_offset.required_delta_cmx:.6f}\n"
        f"  selected mean y = {wing_offset.selected_mean_y:.3f} model units\n"
        f"  required y-offset = {wing_offset.offset_y:.3f} model units "
        f"({wing_offset.offset_y_m:.6f} m)\n"
        f"  wing tipward angle = {wing_tipward_angle_deg:.1f} deg\n"
        f"  required tipward offset = {wing_offset.tipward_offset:.3f} model units "
        f"({wing_offset.tipward_offset_m:.6f} m)"
    )


def _legacy_single_result_main() -> None:
    plt.close("all")

    surface_results: list[tuple[str, pd.DataFrame, pd.DataFrame, float]] = []
    plot_title = _plot_title_from_data_dir(data_dir)

    missing_patch_files = [
        patch_id_file
        for patch_id_file in surface_patch_id_files.values()
        if not patch_id_file.is_file()
    ]
    if missing_patch_files:
        missing_names = ", ".join(path.name for path in missing_patch_files)
        raise FileNotFoundError(f"Missing patch-id selection files: {missing_names}")

    vtu_cell_df = load_vtu_cell_center_dataframe(surface_vtu_file)

    for surface_name in surface_names:
        lift_direction = surface_lift_force_direction_deg[surface_name]
        surface_df, strip, cmx_total = process_vtu_surface(
            vtu_cell_df,
            surface_patch_id_files[surface_name],
            lift_direction,
            force_mirror_spanwise_results=surface_name.startswith("wing"),
        )
        surface_results.append((surface_name, surface_df, strip, cmx_total))
        print(
            f"{surface_name}: lift direction = {lift_direction:.1f} deg, "
            f"Cmx = {cmx_total:.6f}"
        )

    cmx_sum = sum(cmx_total for _, _, _, cmx_total in surface_results)
    print(f"Wing + stab Cmx sum = {cmx_sum:.6f}")
    component_sums: dict[str, float] = {}
    for prefix in ("wing", "stab"):
        prefix_sum = sum(
            cmx_total
            for surface_name, _, _, cmx_total in surface_results
            if surface_name.startswith(prefix)
        )
        component_sums[prefix] = prefix_sum
        print(f"{prefix} Cmx sum = {prefix_sum:.6f}")

    aircraft_df, _, aircraft_cmx_total = process_surface_dataframe(vtu_cell_df.copy(), 0.0)
    aircraft_cl = float(aircraft_df["dFz_q"].sum() / S_ref)
    fuselage_cmx = aircraft_cmx_total - component_sums["wing"] - component_sums["stab"]
    component_sums["fuselage"] = fuselage_cmx
    component_sums["aircraft"] = aircraft_cmx_total
    print(f"aircraft total Cmx = {aircraft_cmx_total:.6f}")
    print(f"aircraft global CL = {aircraft_cl:.6f}")
    print(f"fuselage Cmx = {fuselage_cmx:.6f}")

    freestream_speed_m_s = _freestream_speed_m_s_from_result(result, data_dir)
    wing_offset_result = calculate_required_wing_y_offset(
        surface_results,
        i_wing_offset,
        aircraft_cmx_total,
        P_mot,
        rpm_mot,
        motor_torque_sign,
        rho_inf_kg_m3,
        freestream_speed_m_s,
        length_unit_m,
        wing_tipward_angle_deg,
    )
    print_required_wing_offset(result, wing_offset_result, freestream_speed_m_s)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    surface_colors = {
        surface_name: color_cycle[index % len(color_cycle)]
        for index, (surface_name, _, _, _) in enumerate(surface_results)
    }

    fig_cmx = plt.figure(figsize=(9.0, 5.5), constrained_layout=True)
    for surface_name, _, strip, _ in surface_results:
        original_strip = strip[~strip["is_mirrored"]]
        mirrored_strip = strip[strip["is_mirrored"]]
        plt.plot(
            original_strip["y_mid"],
            original_strip["dCmx_dy"].abs(),
            marker="o",
            label=_surface_label(surface_name),
            color=surface_colors[surface_name],
        )
        if not mirrored_strip.empty:
            plt.plot(
                mirrored_strip["y_mid"],
                mirrored_strip["dCmx_dy"].abs(),
                marker="o",
                color=surface_colors[surface_name],
                alpha=0.5,
                label="_nolegend_",
            )
    plt.xlabel("y")
    plt.ylabel(r"$|dC_{mx}/dy|$")
    plt.title(f"{plot_title} - rolling moment distribution")
    plt.grid(True)
    plt.legend()
    fig_cmx.savefig(data_dir / "wing_stab_dCmx_dy.png", dpi=300)
    if not show_plots:
        plt.close(fig_cmx)

    fig_cl = plt.figure(figsize=(9.0, 5.5), constrained_layout=True)
    cl_normalization = aircraft_cl if not np.isclose(aircraft_cl, 0.0) else np.nan
    for surface_name, _, strip, _ in surface_results:
        original_strip = strip[~strip["is_mirrored"]]
        mirrored_strip = strip[strip["is_mirrored"]]
        plt.plot(
            original_strip["y_mid"],
            original_strip["cl_local"] / cl_normalization,
            marker="o",
            label=_surface_label(surface_name),
            color=surface_colors[surface_name],
        )
        if not mirrored_strip.empty:
            plt.plot(
                mirrored_strip["y_mid"],
                mirrored_strip["cl_local"] / cl_normalization,
                marker="o",
                color=surface_colors[surface_name],
                alpha=0.5,
                label="_nolegend_",
            )
    plt.xlabel("y")
    plt.ylabel(r"$c_l / C_L$")
    plt.title(f"{plot_title} - normalized local strip lift coefficient")
    plt.grid(True)
    plt.legend()
    fig_cl.savefig(data_dir / "wing_stab_cl_local_normalized.png", dpi=300)
    if not show_plots:
        plt.close(fig_cl)

    fig_cmx_bars = plt.figure(figsize=(10.0, 7.0), constrained_layout=True)
    row_positions = {
        "wing1": 5.15,
        "wing2": 4.85,
        "wing_sum_1_2": 4.0,
        "wing_sum_3_4": 3.7,
        "wing3": 2.85,
        "wing4": 2.55,
        "stab1": 1.15,
        "stab2": 0.85,
        "stab_sum_1_2": 0.0,
        "stab_sum_3_4": -0.3,
        "stab3": -1.15,
        "stab4": -1.45,
    }
    max_abs_cmx = max(abs(cmx_total) for _, _, _, cmx_total in surface_results)
    cmx_by_surface = {
        surface_name: cmx_total
        for surface_name, _, _, cmx_total in surface_results
    }
    pair_sum_results = [
        (
            "wing_sum_1_2",
            "wing1 + wing2",
            cmx_by_surface["wing1"] + cmx_by_surface["wing2"],
            surface_colors["wing1"],
        ),
        (
            "wing_sum_3_4",
            "wing3 + wing4",
            cmx_by_surface["wing3"] + cmx_by_surface["wing4"],
            surface_colors["wing3"],
        ),
        (
            "stab_sum_1_2",
            "stab1 + stab2",
            cmx_by_surface["stab1"] + cmx_by_surface["stab2"],
            surface_colors["stab1"],
        ),
        (
            "stab_sum_3_4",
            "stab3 + stab4",
            cmx_by_surface["stab3"] + cmx_by_surface["stab4"],
            surface_colors["stab3"],
        ),
    ]
    min_bar_value = min(0.0, *(sum_value for _, _, sum_value, _ in pair_sum_results))
    max_bar_value = max(
        max_abs_cmx,
        *(sum_value for _, _, sum_value, _ in pair_sum_results),
    )
    max_abs_for_offset = max(abs(min_bar_value), abs(max_bar_value))
    label_offset = 0.02 * max_abs_for_offset if max_abs_for_offset > 0.0 else 0.01

    for surface_name, _, _, cmx_total in surface_results:
        abs_cmx = abs(cmx_total)
        plt.barh(
            row_positions[surface_name],
            abs_cmx,
            height=0.22,
            color=surface_colors[surface_name],
            label=_surface_label(surface_name),
        )
        plt.text(
            abs_cmx + label_offset,
            row_positions[surface_name],
            f"{cmx_total:.6f}",
            va="center",
        )

    for sum_name, label, sum_value, color in pair_sum_results:
        plt.barh(
            row_positions[sum_name],
            sum_value,
            height=0.22,
            color=color,
            label=label,
        )
        plt.text(
            sum_value + label_offset if sum_value >= 0.0 else sum_value - label_offset,
            row_positions[sum_name],
            f"{sum_value:.6f}",
            va="center",
            ha="left" if sum_value >= 0.0 else "right",
        )

    plt.yticks(
        [5.0, 3.85, 2.7, 1.0, -0.15, -1.3],
        [
            "wings 1 + 2",
            "wing pair sums",
            "wings 3 + 4",
            "stabs 1 + 2",
            "stab pair sums",
            "stabs 3 + 4",
        ],
    )
    plt.xlabel(r"$C_{mx}$ pair sum / $|C_{mx}|$ surface magnitude")
    plt.title(f"{plot_title} - surface rolling moment contributions")
    plt.xlim(min_bar_value - 8.0 * label_offset, max_bar_value + 8.0 * label_offset)
    plt.axvline(0.0, color="0.25", linewidth=0.8)
    plt.grid(True, axis="x")
    plt.legend()
    fig_cmx_bars.savefig(data_dir / "wing_stab_cmx_barchart.png", dpi=300)
    if not show_plots:
        plt.close(fig_cmx_bars)

    fig_aircraft_bars = plt.figure(figsize=(9.0, 5.0), constrained_layout=True)
    aircraft_bar_values = [
        ("aircraft", "aircraft total", component_sums["aircraft"]),
        ("wing", "wings", component_sums["wing"]),
        ("stab", "stabilizer", component_sums["stab"]),
        ("fuselage", "fuselage", component_sums["fuselage"]),
    ]
    aircraft_bar_colors = {
        "aircraft": "0.25",
        "wing": surface_colors["wing1"],
        "stab": surface_colors["stab1"],
        "fuselage": "0.55",
    }
    aircraft_positions = {
        "aircraft": 3.0,
        "wing": 2.0,
        "stab": 1.0,
        "fuselage": 0.0,
    }
    min_aircraft_bar = min(0.0, *(value for _, _, value in aircraft_bar_values))
    max_aircraft_bar = max(0.0, *(value for _, _, value in aircraft_bar_values))
    aircraft_label_offset = (
        0.02 * max(abs(min_aircraft_bar), abs(max_aircraft_bar))
        if max(abs(min_aircraft_bar), abs(max_aircraft_bar)) > 0.0
        else 0.01
    )

    for key, label, value in aircraft_bar_values:
        plt.barh(
            aircraft_positions[key],
            value,
            height=0.32,
            color=aircraft_bar_colors[key],
            label=label,
        )
        plt.text(
            value + aircraft_label_offset if value >= 0.0 else value - aircraft_label_offset,
            aircraft_positions[key],
            f"{value:.6f}",
            va="center",
            ha="left" if value >= 0.0 else "right",
        )

    plt.yticks(
        [aircraft_positions[key] for key, _, _ in aircraft_bar_values],
        [label for _, label, _ in aircraft_bar_values],
    )
    plt.xlabel(r"$C_{mx}$")
    plt.title(f"{plot_title} - aircraft rolling moment contributions")
    plt.xlim(
        min_aircraft_bar - 8.0 * aircraft_label_offset,
        max_aircraft_bar + 8.0 * aircraft_label_offset,
    )
    plt.axvline(0.0, color="0.25", linewidth=0.8)
    plt.grid(True, axis="x")
    fig_aircraft_bars.savefig(data_dir / "aircraft_cmx_contributions.png", dpi=300)
    if show_plots:
        plt.show()
    plt.close("all")


def _selected_configurations() -> tuple[ResultConfiguration, ...]:
    labels = [result] if isinstance(result, str) else list(result)
    if not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("result must be a result label or a non-empty list of result labels.")
    if len(set(labels)) != len(labels):
        raise ValueError("result must not contain duplicate result labels.")
    unknown = [label for label in labels if label not in RESULT_CONFIGURATIONS]
    if unknown:
        raise ValueError(f"Unknown result labels: {unknown}")
    return tuple(RESULT_CONFIGURATIONS[label] for label in labels)


def _analyse_configuration(configuration: ResultConfiguration) -> AnalysisResult:
    surface_file = configuration.data_dir / "surfaces.vtu"
    selection_files = {
        name: configuration.data_dir / f"{name}_data.csv" for name in surface_names
    }
    required_files = [surface_file, *selection_files.values()]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing inputs for {configuration.label}:\n{formatted}")

    cells = load_vtu_cell_center_dataframe(surface_file)
    surface_results = []
    for name in surface_names:
        surface_results.append(
            (
                name,
                *process_vtu_surface(
                    cells,
                    selection_files[name],
                    surface_lift_force_direction_deg[name],
                    configuration.reference_area_mm2,
                    configuration.reference_span_mm,
                    force_mirror_spanwise_results=name.startswith("wing"),
                ),
            )
        )
    component_sums = {
        prefix: sum(cmx for name, _, _, cmx in surface_results if name.startswith(prefix))
        for prefix in ("wing", "stab")
    }
    aircraft_df, _, aircraft_cmx = process_surface_dataframe(
        cells.copy(), 0.0, configuration.reference_area_mm2, configuration.reference_span_mm
    )
    component_sums["aircraft"] = aircraft_cmx
    component_sums["fuselage"] = aircraft_cmx - component_sums["wing"] - component_sums["stab"]
    aircraft_cl = float(aircraft_df["dFz_q"].sum() / configuration.reference_area_mm2)
    freestream_speed_m_s = _freestream_speed_m_s_from_result(
        configuration.label, configuration.data_dir
    )
    rolling_moment_nm = _rolling_moment_nm(
        aircraft_cmx,
        freestream_speed_m_s,
        configuration.reference_area_mm2,
        configuration.reference_span_mm,
    )
    wing_offset = calculate_required_wing_y_offset(
        surface_results,
        i_wing_offset,
        aircraft_cmx,
        P_mot,
        rpm_mot,
        motor_torque_sign,
        rho_inf_kg_m3,
        freestream_speed_m_s,
        configuration.reference_area_mm2,
        configuration.reference_span_mm,
        length_unit_m,
        wing_tipward_angle_deg,
    )
    print(
        f"{configuration.label}: CL={aircraft_cl:.6f}, Cmx={aircraft_cmx:.6f}, "
        f"Mx={rolling_moment_nm:.6f} N m, "
        f"fuselage Cmx={component_sums['fuselage']:.6f}"
    )
    print_required_wing_offset(configuration.label, wing_offset, freestream_speed_m_s)
    return AnalysisResult(configuration, tuple(surface_results), component_sums, aircraft_cl, wing_offset)


def _offsets(count: int, spacing: float) -> np.ndarray:
    return (np.arange(count) - (count - 1) / 2.0) * spacing


def _plot_spanwise_results(
    analyses: tuple[AnalysisResult, ...],
    output_dir: Path,
    value_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    markers = ("o", "s", "^", "D", "P", "X")
    for result_index, analysis in enumerate(analyses):
        for surface_name, _, strip, _ in analysis.surface_results:
            original = strip[~strip["is_mirrored"]]
            mirrored = strip[strip["is_mirrored"]]
            values = original[value_column]
            if value_column == "cl_local":
                values = values / analysis.aircraft_cl
            elif value_column == "dCmx_dy":
                values = values.abs()
            kwargs = {
                "color": SURFACE_COLORS[surface_name],
                "linestyle": "-" if surface_name.startswith("wing") else "--",
                "marker": markers[result_index % len(markers)],
            }
            axis.plot(
                original["y_mid"], values,
                label=f"{analysis.configuration.label} - {surface_name}", **kwargs,
            )
            if not mirrored.empty:
                mirrored_values = mirrored[value_column]
                if value_column == "cl_local":
                    mirrored_values = mirrored_values / analysis.aircraft_cl
                elif value_column == "dCmx_dy":
                    mirrored_values = mirrored_values.abs()
                axis.plot(mirrored["y_mid"], mirrored_values, alpha=0.45, label="_nolegend_", **kwargs)
    axis.set_xlabel("y")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    axis.legend(fontsize="x-small", ncols=2)
    figure.savefig(output_dir / filename, dpi=300)


def _plot_bar_results(
    analyses: tuple[AnalysisResult, ...], output_dir: Path, title: str
) -> None:
    row_positions = {"wing1": 5.15, "wing2": 4.85, "wing_sum_1_2": 4.0, "wing_sum_3_4": 3.7, "wing3": 2.85, "wing4": 2.55, "stab1": 1.15, "stab2": 0.85, "stab_sum_1_2": 0.0, "stab_sum_3_4": -0.3, "stab3": -1.15, "stab4": -1.45}
    hatches = ("", "//", "xx", "..", "++", "\\\\")
    offsets = _offsets(len(analyses), 0.18)
    figure, axis = plt.subplots(figsize=(12.0, 8.0), constrained_layout=True)
    values = []
    for index, analysis in enumerate(analyses):
        cmx = {name: value for name, _, _, value in analysis.surface_results}
        pair_sums = {
            "wing_sum_1_2": cmx["wing1"] + cmx["wing2"],
            "wing_sum_3_4": cmx["wing3"] + cmx["wing4"],
            "stab_sum_1_2": cmx["stab1"] + cmx["stab2"],
            "stab_sum_3_4": cmx["stab3"] + cmx["stab4"],
        }
        values.extend([*cmx.values(), *pair_sums.values()])
        for name, value in cmx.items():
            axis.barh(row_positions[name] + offsets[index], abs(value), height=0.15, color=SURFACE_COLORS[name], hatch=hatches[index % len(hatches)], label=f"{analysis.configuration.label} - {name}")
        for name, value in pair_sums.items():
            color_name = "wing1" if name.endswith("1_2") else "wing3"
            axis.barh(row_positions[name] + offsets[index], value, height=0.15, color=SURFACE_COLORS[color_name], hatch=hatches[index % len(hatches)])
    limit = max((abs(value) for value in values), default=0.01) * 1.1
    axis.set_yticks([5.0, 3.85, 2.7, 1.0, -0.15, -1.3], ["wings 1 + 2", "wing pair sums", "wings 3 + 4", "stabs 1 + 2", "stab pair sums", "stabs 3 + 4"])
    axis.set_xlabel(r"$C_{mx}$ pair sum / $|C_{mx}|$ surface magnitude")
    axis.set_title(f"{title} - surface rolling moment contributions")
    axis.set_xlim(-limit, limit)
    axis.axvline(0.0, color="0.25", linewidth=0.8)
    axis.grid(True, axis="x")
    axis.legend(fontsize="x-small", ncols=2)
    figure.savefig(output_dir / "wing_stab_cmx_barchart.png", dpi=300)


def _plot_aircraft_bars(
    analyses: tuple[AnalysisResult, ...], output_dir: Path, title: str
) -> None:
    positions = {"aircraft": 3.0, "wing": 2.0, "stab": 1.0, "fuselage": 0.0}
    colors = {
        "aircraft": "0.25",
        "wing": "#1f77b4",
        "stab": "#9467bd",
        "fuselage": "0.55",
    }
    hatches = ("", "//", "xx", "..", "++", "\\\\")
    offsets = _offsets(len(analyses), 0.18)
    figure, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    values = []
    for index, analysis in enumerate(analyses):
        for key, position in positions.items():
            value = analysis.component_sums[key]
            values.append(value)
            axis.barh(position + offsets[index], value, height=0.15, color=colors[key], hatch=hatches[index % len(hatches)], label=f"{analysis.configuration.label} - {key}")
    limit = max((abs(value) for value in values), default=0.01) * 1.1
    axis.set_yticks(list(positions.values()), ["aircraft total", "wings", "stabilizer", "fuselage"])
    axis.set_xlabel(r"$C_{mx}$")
    axis.set_title(f"{title} - aircraft rolling moment contributions")
    axis.set_xlim(-limit, limit)
    axis.axvline(0.0, color="0.25", linewidth=0.8)
    axis.grid(True, axis="x")
    axis.legend(fontsize="x-small", ncols=2)
    figure.savefig(output_dir / "aircraft_cmx_contributions.png", dpi=300)


def main() -> None:
    plt.close("all")
    analyses = tuple(_analyse_configuration(config) for config in _selected_configurations())
    output_dir = (
        analyses[0].configuration.data_dir
        if len(analyses) == 1
        else COMPARISON_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    title = " / ".join(analysis.configuration.label for analysis in analyses)
    _plot_spanwise_results(analyses, output_dir, "dCmx_dy", r"$|dC_{mx}/dy|$", f"{title} - rolling moment distribution", "wing_stab_dCmx_dy.png")
    _plot_spanwise_results(analyses, output_dir, "cl_local", r"$c_l / C_L$", f"{title} - normalized local strip lift coefficient", "wing_stab_cl_local_normalized.png")
    _plot_bar_results(analyses, output_dir, title)
    _plot_aircraft_bars(analyses, output_dir, title)
    if show_plots:
        plt.show()
    plt.close("all")


def _surface_label(surface_name: str) -> str:
    return surface_name


def _optional_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column in df:
        return df[column].to_numpy()
    return np.zeros(len(df))


def _plot_title_from_data_dir(path: Path) -> str:
    wing_type = _wing_type_from_path(path.parent.name)
    case_name = path.name
    airspeed = _regex_group(case_name, r"U([-+]?\d+(?:\.\d+)?)(?=_|$)")
    aoa = _regex_group(case_name, r"AOA([-+]?\d+(?:\.\d+)?)(?=_|$)")

    title_parts = [wing_type]
    if airspeed is not None:
        title_parts.append(f"U={airspeed}")
    if aoa is not None:
        title_parts.append(f"AOA={aoa} deg")
    return ", ".join(title_parts)


def _wing_type_from_path(path_part: str) -> str:
    lower_path_part = path_part.lower()
    if "rectangular" in lower_path_part:
        return "rectangular wing"
    if "trapezoidal" in lower_path_part:
        return "trapezoidal wing"
    return path_part


def _regex_group(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


if __name__ == "__main__":
    main()
