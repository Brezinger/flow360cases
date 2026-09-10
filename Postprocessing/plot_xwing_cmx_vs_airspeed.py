"""Compare full-aircraft aerodynamic Cmx over airspeed for XWing wing variants.

The script reads the Flow360 ``surfaces.vtu`` results for the available rectangular
and trapezoidal XWing cases. It computes the aerodynamic rolling-moment coefficient
directly from pressure and skin-friction forces, then writes a comparison plot and a
CSV table to ``<flow360-root>/cmx_vs_airspeed``.

Set ``PLOT_COMPONENT_CMX`` to ``True`` to additionally plot the wings, stabilizer,
and fuselage contributions.  This requires the ``wing1_data.csv`` through
``wing4_data.csv`` and ``stab1_data.csv`` through ``stab4_data.csv`` PatchID
selection files in every case directory.  They can be created with
``export_wing_stab_patch_ids.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv


FLOW360_ROOT = Path(
    "C:/Nextcloud/Freigaben/XWing2_CAD+structure/XWing2_2/flow360"
)
OUTPUT_DIR = FLOW360_ROOT / "cmx_vs_airspeed"
SHOW_PLOT = True
PLOT_COMPONENT_CMX = True

WING_SELECTION_FILENAMES = tuple(f"wing{index}_data.csv" for index in range(1, 5))
STABILIZER_SELECTION_FILENAMES = tuple(
    f"stab{index}_data.csv" for index in range(1, 5)
)


@dataclass(frozen=True)
class CaseDefinition:
    wing_type: str
    airspeed_m_s: float
    result_directory: Path
    reference_area_mm2: float
    reference_span_mm: float

    @property
    def surface_file(self) -> Path:
        return self.result_directory / "surfaces.vtu"


CASES = (
    CaseDefinition(
        "Rectangular wing",
        24.5,
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U24.5_AOA10",
        0.2831e6,
        1312.0,
    ),
    CaseDefinition(
        "Rectangular wing",
        35.0,
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U35",
        0.2831e6,
        1312.0,
    ),
    CaseDefinition(
        "Rectangular wing",
        39.5,
        FLOW360_ROOT / "rectangular wing" / "XWing2_2 fully_turbulent_SA U39.5_AOA-1.6",
        0.2831e6,
        1312.0,
    ),
    CaseDefinition(
        "Trapezoidal wing",
        24.5,
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2 fully_turbulent_SA U24.5_AOA10",
        0.277649e6,
        1346.0,
    ),
    CaseDefinition(
        "Trapezoidal wing",
        35.0,
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2_fully_turbulent_SA U35",
        0.277649e6,
        1346.0,
    ),
    CaseDefinition(
        "Trapezoidal wing",
        39.5,
        FLOW360_ROOT / "trapezoidal wing" / "XWing2_2 fully_turbulent_SA U39.5_AOA-1.6",
        0.277649e6,
        1346.0,
    ),
)


def _patch_ids_from_files(selection_files: tuple[Path, ...]) -> np.ndarray:
    """Return the unique PatchIDs selected by the supplied ParaView exports."""
    return np.unique(
        np.concatenate(
            [pd.read_csv(path, usecols=["PatchID"])["PatchID"].dropna().to_numpy() for path in selection_files]
        ).astype(int)
    )


def calculate_cmx_contributions(case: CaseDefinition) -> dict[str, float]:
    """Return total Cmx and, when requested, Cmx for each aircraft component."""
    mesh = pv.read(str(case.surface_file)).extract_surface(algorithm="dataset_surface")
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    mesh = mesh.point_data_to_cell_data(pass_point_data=False)
    mesh = mesh.compute_cell_sizes(length=False, volume=False)
    centers = mesh.cell_centers().points

    required_arrays = {"Cp", "Area", "Normals"}
    if PLOT_COMPONENT_CMX:
        required_arrays.add("PatchID")
    missing = required_arrays - set(mesh.cell_data.keys())
    if missing:
        raise ValueError(f"{case.surface_file} is missing cell arrays: {sorted(missing)}")

    cp = mesh.cell_data["Cp"]
    area = mesh.cell_data["Area"]
    normals = mesh.cell_data["Normals"]
    cf_vec = mesh.cell_data.get("CfVec", np.zeros((mesh.n_cells, 3)))

    d_fy_q = area * (-cp * normals[:, 1] + cf_vec[:, 1])
    d_fz_q = area * (-cp * normals[:, 2] + cf_vec[:, 2])
    d_mx_q = centers[:, 1] * d_fz_q - centers[:, 2] * d_fy_q
    reference_moment = case.reference_area_mm2 * case.reference_span_mm
    contributions = {"aircraft": float(d_mx_q.sum() / reference_moment)}

    if not PLOT_COMPONENT_CMX:
        return contributions

    wing_files = tuple(case.result_directory / name for name in WING_SELECTION_FILENAMES)
    stabilizer_files = tuple(
        case.result_directory / name for name in STABILIZER_SELECTION_FILENAMES
    )
    missing_selection_files = [
        path for path in (*wing_files, *stabilizer_files) if not path.is_file()
    ]
    if missing_selection_files:
        formatted = "\n".join(f"  - {path}" for path in missing_selection_files)
        raise FileNotFoundError(
            f"Missing component PatchID selection files for {case.surface_file}:\n{formatted}"
        )

    wing_patch_ids = _patch_ids_from_files(wing_files)
    stabilizer_patch_ids = _patch_ids_from_files(stabilizer_files)
    overlap = np.intersect1d(wing_patch_ids, stabilizer_patch_ids)
    if overlap.size:
        raise ValueError(
            f"Wing and stabilizer PatchID selections overlap in {case.result_directory}: "
            f"{overlap.tolist()}"
        )

    patch_ids = mesh.cell_data["PatchID"]
    wing_cmx = float(d_mx_q[np.isin(patch_ids, wing_patch_ids)].sum() / reference_moment)
    stabilizer_cmx = float(
        d_mx_q[np.isin(patch_ids, stabilizer_patch_ids)].sum() / reference_moment
    )
    contributions["wings"] = wing_cmx
    contributions["stabilizer"] = stabilizer_cmx
    contributions["fuselage"] = contributions["aircraft"] - wing_cmx - stabilizer_cmx
    return contributions


def main() -> None:
    missing_files = [case.surface_file for case in CASES if not case.surface_file.is_file()]
    if missing_files:
        formatted = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing Flow360 surface results:\n{formatted}")

    records = []
    for case in CASES:
        contributions = calculate_cmx_contributions(case)
        record = {
            "wing_type": case.wing_type,
            "airspeed_m_s": case.airspeed_m_s,
            "aerodynamic_cmx": contributions["aircraft"],
            "surface_file": str(case.surface_file),
        }
        if PLOT_COMPONENT_CMX:
            record.update(
                {
                    f"{component}_cmx": value
                    for component, value in contributions.items()
                    if component != "aircraft"
                }
            )
        records.append(record)
        print(
            f"{case.wing_type}, U={case.airspeed_m_s:g} m/s: "
            f"Cmx={contributions['aircraft']:.6f}"
        )

    results = pd.DataFrame(records).sort_values(["wing_type", "airspeed_m_s"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "xwing_cmx_vs_airspeed.csv", index=False)

    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    component_styles = {
        "wings": "--",
        "stabilizer": ":",
        "fuselage": "-.",
    }
    for wing_type, group in results.groupby("wing_type", sort=False):
        axis.plot(
            group["airspeed_m_s"],
            group["aerodynamic_cmx"],
            marker="o",
            linewidth=1.8,
            label=wing_type,
        )
        if PLOT_COMPONENT_CMX:
            colour = axis.lines[-1].get_color()
            for component, linestyle in component_styles.items():
                axis.plot(
                    group["airspeed_m_s"],
                    group[f"{component}_cmx"],
                    color=colour,
                    linestyle=linestyle,
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    label=f"{wing_type} — {component}",
                )
    axis.set_xlabel("Freestream speed U∞ [m/s]")
    axis.set_ylabel(r"Aerodynamic $C_{mx}$")
    axis.set_title(
        "XWing rolling-moment contributions over airspeed"
        if PLOT_COMPONENT_CMX
        else "XWing rolling moment over airspeed"
    )
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.grid(True)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "xwing_cmx_vs_airspeed.png", dpi=300)
    if SHOW_PLOT and "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
