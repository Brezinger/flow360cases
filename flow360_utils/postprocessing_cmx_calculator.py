from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree


# -----------------------------
# User inputs
# -----------------------------
variant = "trap 24.5"
variant = "trap 39"
variant = "rect 24.5"
variant = "rect 39"

if variant == "rect 24.5":
    data_dir = Path(
        "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360/"
        "rectangular wing/XWing2_2 fully_turbulent_SA U24.5_AOA10"
    )
elif variant == "rect 39":
    data_dir = Path(
        "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360/"
        "rectangular wing/XWing2_2 fully_turbulent_SA U39.5_AOA-1.6"
    )
elif variant == "trap 24.5":
    data_dir = Path(
        "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360/"
        "trapezoidal wing/XWing2_2 fully_turbulent_SA U24.5_AOA10"
    )
elif variant == "trap 39":
    data_dir = Path(
        "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360/"
        "trapezoidal wing/XWing2_2 fully_turbulent_SA U39.5_AOA-1.6"
    )
else:
    raise ValueError(f"Unknown variant {variant}")

if "trap" in variant:
    dy = 25.0                      # strip width in same length unit as Points
    S_ref = 0.277649 * 1e6  # reference area, set correctly
    b_ref = 1346                 # reference span for Cmx, for trap wing
elif "rect" in variant:
    dy = 25.0  # strip width in same length unit as Points
    S_ref = 0.2831 * 1e6  # reference area, set correctly
    b_ref = 1312  # reference span for Cmx, for trap wing


data_files = [
    *(data_dir / f"wing{index}_data.csv" for index in range(1, 5)),
    *(data_dir / f"stab{index}_data.csv" for index in range(1, 5)),
]
total_aircraft_data_file = data_dir / "aircraft_data.csv"
surface_vtu_file = data_dir / "surfaces.vtu"
use_vtu_surface_integration = True
surface_patch_id_files = {
    surface_file.stem: _patch_id_file
    for surface_file in data_files
    if (_patch_id_file := data_dir / f"{surface_file.stem.removesuffix('_data')}_test.csv").is_file()
}
surface_lift_force_direction_deg = {
    "wing1_data": 125.0,
    "wing2_data": 55.0,
    "wing3_data": 125.0,
    "wing4_data": 55.0,
    "stab1_data": 125.0,
    "stab2_data": 55.0,
    "stab3_data": 125.0,
    "stab4_data": 55.0,
}



# moment reference point
x_ref = 743
y_ref = 0.0
z_ref = 0.0


def process_surface(
    filepath: Path,
    lift_force_direction_deg: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = pd.read_csv(filepath)
    return process_surface_dataframe(df, lift_force_direction_deg)


def process_surface_dataframe(
    df: pd.DataFrame,
    lift_force_direction_deg: float,
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
    df["dCmx"] = dMx_q / (S_ref * b_ref)
    df["dA_xy"] = area * np.abs(nz)

    cmx_total = float(df["dCmx"].sum())

    y_min = df["Points:1"].min()
    df["strip"] = np.floor((df["Points:1"] - y_min) / dy).astype(int)

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

    strip["dCmx_dy"] = strip["Cmx"] / dy
    strip["cl_local"] = np.where(
        strip["area_xy"] > 0.0,
        strip["Fn_q"] / strip["area_xy"],
        np.nan,
    )

    return df, strip, cmx_total


def load_vtu_cell_center_dataframe(filepath: Path) -> pd.DataFrame:
    mesh = pv.read(str(filepath)).extract_surface()
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
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    surface_df = select_vtu_cells_from_selection_file(vtu_cell_df, patch_id_file)
    if surface_df.empty:
        raise ValueError(f"No VTU cells matched PatchID values from {patch_id_file}.")
    return process_surface_dataframe(surface_df, lift_force_direction_deg)


def select_vtu_cells_from_selection_file(
    vtu_cell_df: pd.DataFrame,
    selection_file: Path,
    coordinate_tolerance: float = 1.0e-2,
) -> pd.DataFrame:
    selection_df = pd.read_csv(selection_file)
    if {"Points:0", "Points:1", "Points:2"}.issubset(selection_df.columns):
        matched_indices = []
        point_columns = ["Points:0", "Points:1", "Points:2"]
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


def main() -> None:
    surface_results: list[tuple[str, pd.DataFrame, float]] = []
    plot_title = _plot_title_from_data_dir(data_dir)
    vtu_cell_df = None

    if use_vtu_surface_integration:
        missing_patch_files = [
            data_file.stem
            for data_file in data_files
            if data_file.stem not in surface_patch_id_files
        ]
        if missing_patch_files:
            print(
                "VTU integration disabled because patch-id selection files are missing for: "
                + ", ".join(missing_patch_files)
            )
        else:
            vtu_cell_df = load_vtu_cell_center_dataframe(surface_vtu_file)

    for data_file in data_files:
        lift_direction = surface_lift_force_direction_deg[data_file.stem]
        if vtu_cell_df is not None:
            _, strip, cmx_total = process_vtu_surface(
                vtu_cell_df,
                surface_patch_id_files[data_file.stem],
                lift_direction,
            )
        else:
            _, strip, cmx_total = process_surface(data_file, lift_direction)
        surface_results.append((data_file.stem, strip, cmx_total))
        print(
            f"{data_file.stem}: lift direction = {lift_direction:.1f} deg, "
            f"Cmx = {cmx_total:.6f}"
        )

    cmx_sum = sum(cmx_total for _, _, cmx_total in surface_results)
    print(f"Wing + stab Cmx sum = {cmx_sum:.6f}")
    component_sums: dict[str, float] = {}
    for prefix in ("wing", "stab"):
        prefix_sum = sum(
            cmx_total
            for surface_name, _, cmx_total in surface_results
            if surface_name.startswith(prefix)
        )
        component_sums[prefix] = prefix_sum
        print(f"{prefix} Cmx sum = {prefix_sum:.6f}")

    if vtu_cell_df is not None:
        aircraft_df, _, aircraft_cmx_total = process_surface_dataframe(vtu_cell_df.copy(), 0.0)
    else:
        aircraft_df, _, aircraft_cmx_total = process_surface(total_aircraft_data_file, 0.0)
    aircraft_cl = float(aircraft_df["dFz_q"].sum() / S_ref)
    fuselage_cmx = aircraft_cmx_total - component_sums["wing"] - component_sums["stab"]
    component_sums["fuselage"] = fuselage_cmx
    component_sums["aircraft"] = aircraft_cmx_total
    print(f"aircraft total Cmx = {aircraft_cmx_total:.6f}")
    print(f"aircraft global CL = {aircraft_cl:.6f}")
    print(f"fuselage Cmx = {fuselage_cmx:.6f}")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    surface_colors = {
        surface_name: color_cycle[index % len(color_cycle)]
        for index, (surface_name, _, _) in enumerate(surface_results)
    }

    fig_cmx = plt.figure(figsize=(9.0, 5.5), constrained_layout=True)
    for surface_name, strip, _ in surface_results:
        plt.plot(
            strip["y_mid"].abs(),
            strip["dCmx_dy"].abs(),
            marker="o",
            label=_surface_label(surface_name),
            color=surface_colors[surface_name],
        )
    plt.xlabel(r"$|y|$")
    plt.ylabel(r"$|dC_{mx}/dy|$")
    plt.title(f"{plot_title} - rolling moment distribution")
    plt.grid(True)
    plt.legend()
    fig_cmx.savefig(data_dir / "wing_stab_dCmx_dy.png", dpi=300)


    fig_cl = plt.figure(figsize=(9.0, 5.5), constrained_layout=True)
    cl_normalization = aircraft_cl if not np.isclose(aircraft_cl, 0.0) else np.nan
    for surface_name, strip, _ in surface_results:
        plt.plot(
            abs(strip["y_mid"]),
            strip["cl_local"] / cl_normalization,
            marker="o",
            label=_surface_label(surface_name),
            color=surface_colors[surface_name],
        )
    plt.xlabel("|y|")
    plt.ylabel(r"$c_l / C_L$")
    plt.title(f"{plot_title} - normalized local strip lift coefficient")
    plt.grid(True)
    plt.legend()
    fig_cl.savefig(data_dir / "wing_stab_cl_local_normalized.png", dpi=300)

    fig_cmx_bars = plt.figure(figsize=(10.0, 7.0), constrained_layout=True)
    row_positions = {
        "wing1_data": 5.15,
        "wing2_data": 4.85,
        "wing_sum_1_2": 4.0,
        "wing_sum_3_4": 3.7,
        "wing3_data": 2.85,
        "wing4_data": 2.55,
        "stab1_data": 1.15,
        "stab2_data": 0.85,
        "stab_sum_1_2": 0.0,
        "stab_sum_3_4": -0.3,
        "stab3_data": -1.15,
        "stab4_data": -1.45,
    }
    max_abs_cmx = max(abs(cmx_total) for _, _, cmx_total in surface_results)
    cmx_by_surface = {
        surface_name: cmx_total
        for surface_name, _, cmx_total in surface_results
    }
    pair_sum_results = [
        (
            "wing_sum_1_2",
            "wing1 + wing2",
            cmx_by_surface["wing1_data"] + cmx_by_surface["wing2_data"],
            surface_colors["wing1_data"],
        ),
        (
            "wing_sum_3_4",
            "wing3 + wing4",
            cmx_by_surface["wing3_data"] + cmx_by_surface["wing4_data"],
            surface_colors["wing3_data"],
        ),
        (
            "stab_sum_1_2",
            "stab1 + stab2",
            cmx_by_surface["stab1_data"] + cmx_by_surface["stab2_data"],
            surface_colors["stab1_data"],
        ),
        (
            "stab_sum_3_4",
            "stab3 + stab4",
            cmx_by_surface["stab3_data"] + cmx_by_surface["stab4_data"],
            surface_colors["stab3_data"],
        ),
    ]
    min_bar_value = min(0.0, *(sum_value for _, _, sum_value, _ in pair_sum_results))
    max_bar_value = max(
        max_abs_cmx,
        *(sum_value for _, _, sum_value, _ in pair_sum_results),
    )
    max_abs_for_offset = max(abs(min_bar_value), abs(max_bar_value))
    label_offset = 0.02 * max_abs_for_offset if max_abs_for_offset > 0.0 else 0.01

    for surface_name, _, cmx_total in surface_results:
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

    fig_aircraft_bars = plt.figure(figsize=(9.0, 5.0), constrained_layout=True)
    aircraft_bar_values = [
        ("aircraft", "aircraft total", component_sums["aircraft"]),
        ("wing", "wings", component_sums["wing"]),
        ("stab", "stabilizer", component_sums["stab"]),
        ("fuselage", "fuselage", component_sums["fuselage"]),
    ]
    aircraft_bar_colors = {
        "aircraft": "0.25",
        "wing": surface_colors["wing1_data"],
        "stab": surface_colors["stab1_data"],
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
    plt.show()


def _surface_label(surface_name: str) -> str:
    return surface_name.removesuffix("_data")


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
