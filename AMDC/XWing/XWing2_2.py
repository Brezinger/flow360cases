import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import flow360 as fl
from flow360 import u

from flow360_utils.flow360_tools import (
    calculate_flat_plate_turbulent_y1,
    calculate_standard_atmosphere_density,
    calculate_target_lift_coefficient,
    geometric_spacing_values,
    make_refinement_cylinders_along_axis,
    make_segment_boxes,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_input_file(filename: str, working_dir: Path) -> Path:
    candidates = [
        Path(filename),
        working_dir / filename,
        SCRIPT_DIR / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find {filename!r}. Checked current directory, "
        f"{working_dir}, and {SCRIPT_DIR}."
    )


def _read_te_points(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path, sep=r"\s+", engine="python")
    required_columns = {"X", "Y", "Z"}
    missing_columns = required_columns - set(points.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    return points


def _rotate_points_about_x(points: pd.DataFrame, angle_deg: float) -> pd.DataFrame:
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    rotated = points.copy()
    y = points["Y"].to_numpy(dtype=float)
    z = points["Z"].to_numpy(dtype=float)
    rotated["Y"] = y * cos_angle - z * sin_angle
    rotated["Z"] = y * sin_angle + z * cos_angle
    return rotated


def _write_te_points(points: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(path, sep=" ", index=False, float_format="%.10g")


def prepare_trailing_edge_files(working_dir: Path, wing1_file="TE_XWing2_2_wing1.dat", wing5_file="TE_XWing2_2_wing5.dat") -> list[str]:
    """Create wing 2-4 and 6-8 TE coordinate files from the measured source files."""
    source_wing1 = _resolve_input_file(wing1_file, working_dir)
    source_wing5 = _resolve_input_file(wing5_file, working_dir)

    source_by_wing = {
        1: _read_te_points(source_wing1),
        5: _read_te_points(source_wing5),
    }
    rotations_deg = {
        1: 0.0,
        2: 70.0,
        3: 180.0,
        4: 250.0,
        5: 0.0,
        6: 70.0,
        7: 180.0,
        8: 250.0,
    }

    te_files = []
    generated_dir = working_dir / "generated_te"
    for wing_index in range(1, 9):
        source_points = source_by_wing[1 if wing_index <= 4 else 5]
        angle_deg = rotations_deg[wing_index]
        points = (
            source_points.copy()
            if angle_deg == 0.0
            else _rotate_points_about_x(source_points, angle_deg)
        )
        output_path = generated_dir / f"TE_XWing2_2_wing{wing_index}.dat"
        _write_te_points(points, output_path)
        te_files.append(str(output_path))

    return te_files


def _first_and_last_vertices(vertices_coords_file: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    points = _read_te_points(Path(vertices_coords_file))
    coords = points[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(coords) < 2:
        raise ValueError(f"{vertices_coords_file} must contain at least two points.")
    return tuple(coords[0]), tuple(coords[-1])


def define_and_run(
    project_cgns_file_name,
    name=None,
    symm_face="symm face",
    U_inf=24.5,
    alpha_deg=10.0,
    half_model=False,
    y1_fac=1.0,
    surf_mesh_lvl=0,
    enable_volume_refinements=True,
    enable_alpha_controller=True,
    target_lift_coefficient=None,
    alpha_controller_kp=0.2,
    alpha_controller_ki=0.002,
    alpha_controller_start_pseudo_step=50,
    alpha_controller_initial_alpha_deg=None,
    aircraft_mass=13.6,
    wing_area=0.277649964016683,
    wing_span=1.31708,
    wake_refinement_files=None,
    flow360folder=None,
    results_path=None,
    generate_vol_mesh=True,
    boundary_layer_growth_rate=1.2,
    target_yplus=1.0,
    n_timesteps=1000,
    run_flag=False,
):
    if alpha_controller_initial_alpha_deg is None:
        alpha_controller_initial_alpha_deg = alpha_deg

    async_flag = False
    altitude = 0
    surf_mesh_refine_factor = 2 ** (surf_mesh_lvl / 2)

    mac = 0.108
    moment_center_x = 0.743
    standard_atmosphere_density = calculate_standard_atmosphere_density(altitude)

    if target_lift_coefficient is None:
        target_lift_coefficient = calculate_target_lift_coefficient(
            aircraft_mass=aircraft_mass,
            wing_area=wing_area,
            freestream_velocity=U_inf,
            density=standard_atmosphere_density,
        )

    wake_refinement_angle_deg = 15
    wake_refinement_length = 700
    wake_refinement_delta_x = 100
    wake_refinement_box_overlap = 5
    wake_refinement_spanwise_overlap = 20
    wake_refinement_initial_spacing = 5 * 3 ** (1 / 2)
    wake_refinement_spacing_growth_rate = 1.2
    wake_refinement_max_spacing = None

    winglet_tip_refinement_axis = (1, 0, 0)
    winglet_tip_refinement_length = 1000
    winglet_tip_refinement_n_cylinders = 10
    winglet_tip_refinement_initial_diameter = 20
    winglet_tip_refinement_growth_angle_deg = 5
    winglet_tip_refinement_cylinder_overlap = 5
    winglet_tip_refinement_initial_spacing = surf_mesh_refine_factor * 30 * 3 ** (1 / 2)
    winglet_tip_refinement_spacing_growth_rate = 1.2
    winglet_tip_refinement_max_spacing = None

    ns_solver_tolerance = 1.0e-10
    turb_solver_tolerance = 1.0e-8
    if enable_alpha_controller and alpha_controller_start_pseudo_step >= n_timesteps:
        raise ValueError(
            "alpha_controller_start_pseudo_step must be smaller than n_timesteps "
            f"({n_timesteps}) for the alpha controller to activate."
        )

    _, flat_plate_y1 = calculate_flat_plate_turbulent_y1(
        freestream_velocity=U_inf,
        desired_yplus=target_yplus,
    )
    global_y1 = y1_fac * flat_plate_y1 * u.m

    surf_output_requests = ["Cp", "Cf", "yPlus", "CfVec", "mutRatio"]
    vol_output_requests = ["primitiveVars", "Cp", "qcriterion", "mut", "T", "vorticity"]

    sim_name = f"{name} " if name is not None else ""
    sim_name += "U{0:g}_AOA{1:g}".format(U_inf, alpha_deg)
    if surf_mesh_lvl != 0:
        sim_name += "_mshlvl{0:d}".format(surf_mesh_lvl)

    project = fl.Project.from_surface_mesh(
        project_cgns_file_name,
        name=sim_name,
        folder=flow360folder,
        length_unit="mm",
        run_async=async_flag,
    )
    surface_mesh = project.surface_mesh

    wall_surfaces = [
        surface_mesh[boundary_name]
        for boundary_name in surface_mesh.boundary_names
        if not half_model or boundary_name != symm_face
    ]

    condition = fl.AerospaceCondition(
        velocity_magnitude=U_inf * u.m / u.s,
        alpha=alpha_deg * u.deg,
        beta=0 * u.deg,
    )

    far_field_zone = fl.AutomatedFarfield()

    mesh_defaults = fl.MeshingDefaults(
        surface_edge_growth_rate=1.2,
        surface_max_edge_length=surf_mesh_refine_factor * 80 * u.mm,
        curvature_resolution_angle=surf_mesh_refine_factor * 5 * u.deg,
        boundary_layer_growth_rate=boundary_layer_growth_rate,
        boundary_layer_first_layer_thickness=global_y1,
    )

    refinements = []
    if enable_volume_refinements:
        if wake_refinement_files is None:
            raise ValueError("wake_refinement_files must be provided.")

        wake_refinement_rows = []
        for file in wake_refinement_files:
            h_box = 0
            for i_row, x in enumerate(np.arange(0, wake_refinement_length, wake_refinement_delta_x)):
                h_box += wake_refinement_delta_x * 2 * np.sin(np.deg2rad(wake_refinement_angle_deg / 2))
                boxes = make_segment_boxes(
                    vertices_coords_file=file,
                    x_size=wake_refinement_delta_x + wake_refinement_box_overlap,
                    z_size=h_box + wake_refinement_box_overlap,
                    center_offset=(x + wake_refinement_delta_x / 2, 0, -0.5),
                    name_prefix=f"wakebox_{Path(file).stem}_row{i_row:0d}",
                    segment_overlap=wake_refinement_spanwise_overlap,
                )
                wake_refinement_rows.append((i_row, boxes))

            for vertex_kind, start_center in zip(("root", "tip"), _first_and_last_vertices(file)):
                cylinders = make_refinement_cylinders_along_axis(
                    name_prefix=f"{Path(file).stem}_{vertex_kind}_vortex_cylinder",
                    start_center=start_center,
                    axis=winglet_tip_refinement_axis,
                    total_length=winglet_tip_refinement_length,
                    n_cylinders=winglet_tip_refinement_n_cylinders,
                    initial_diameter=winglet_tip_refinement_initial_diameter,
                    growth_angle_deg=winglet_tip_refinement_growth_angle_deg,
                    cylinder_overlap=winglet_tip_refinement_cylinder_overlap,
                )
                spacings = geometric_spacing_values(
                    initial_spacing=winglet_tip_refinement_initial_spacing,
                    growth_rate=winglet_tip_refinement_spacing_growth_rate,
                    n_values=len(cylinders),
                    max_spacing=winglet_tip_refinement_max_spacing,
                )
                for idx, (cylinder, spacing) in enumerate(zip(cylinders, spacings), start=1):
                    refinements.append(
                        fl.UniformRefinement(
                            name=f"{Path(file).stem}_{vertex_kind}_vortex_refinement_{idx:03d}",
                            entities=[cylinder],
                            spacing=spacing * u.mm,
                        )
                    )

        wake_spacings = geometric_spacing_values(
            initial_spacing=wake_refinement_initial_spacing,
            growth_rate=wake_refinement_spacing_growth_rate,
            n_values=len(wake_refinement_rows),
            max_spacing=wake_refinement_max_spacing,
        )
        for row_order, ((_, boxes), spacing) in enumerate(zip(wake_refinement_rows, wake_spacings), start=1):
            refinements.append(
                fl.UniformRefinement(
                    name=f"wake_refinement_row_{row_order:03d}",
                    entities=boxes,
                    spacing=spacing * u.mm,
                )
            )

    mesh_params = fl.MeshingParams(
        defaults=mesh_defaults,
        volume_zones=[far_field_zone],
        refinements=refinements if refinements else None,
    )

    moment_ref_lengths = (wing_span, mac, wing_span)
    ref_geometry = fl.ReferenceGeometry(
        moment_center=(moment_center_x, 0, 0) * u.m,
        moment_length=moment_ref_lengths * u.m,
        area=wing_area * u.m ** 2,
    )

    navier_stokes_solver = fl.NavierStokesSolver(
        absolute_tolerance=ns_solver_tolerance,
        kappa_MUSCL=0.33,
        linear_solver=fl.KrylovLinearSolver(),
    )
    turbulence_solver = fl.SpalartAllmaras(absolute_tolerance=turb_solver_tolerance)

    fl_models = [
        fl.Wall(surfaces=wall_surfaces, use_wall_function=None),
        fl.Freestream(surfaces=[far_field_zone.farfield]),
        fl.Fluid(
            navier_stokes_solver=navier_stokes_solver,
            turbulence_model_solver=turbulence_solver,
        ),
    ]

    if half_model:
        fl_models.append(fl.SymmetryPlane(surfaces=[far_field_zone.symmetry_planes]))

    user_defined_dynamics = []
    if enable_alpha_controller:
        alpha_controller = fl.UserDefinedDynamic(
            name="alphaController",
            input_vars=["CL"],
            constants={
                "CLTarget": target_lift_coefficient,
                "Kp": alpha_controller_kp,
                "Ki": alpha_controller_ki,
                "StartPseudoStep": alpha_controller_start_pseudo_step,
            },
            output_vars={
                "alphaAngle": "if (pseudoStep > StartPseudoStep) state[0]; else alphaAngle;",
            },
            state_vars_initial_value=[str(alpha_controller_initial_alpha_deg), "0.0"],
            update_law=[
                "if (pseudoStep > StartPseudoStep) state[0] + Kp * (CLTarget - CL) + Ki * state[1]; else state[0];",
                "if (pseudoStep > StartPseudoStep) state[1] + (CLTarget - CL); else state[1];",
            ],
            input_boundary_patches=wall_surfaces,
        )
        user_defined_dynamics.append(alpha_controller)

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            meshing=mesh_params,
            reference_geometry=ref_geometry,
            operating_condition=condition,
            time_stepping=fl.Steady(max_steps=n_timesteps),
            models=fl_models,
            outputs=[
                fl.SurfaceOutput(
                    surfaces=wall_surfaces,
                    output_fields=surf_output_requests,
                    write_single_file=True,
                ),
                fl.VolumeOutput(
                    name="VolumeOutput",
                    output_format="paraview",
                    output_fields=vol_output_requests,
                ),
            ],
        )
        if user_defined_dynamics:
            params.user_defined_dynamics = user_defined_dynamics

    if run_flag and not generate_vol_mesh:
        raise ValueError("generate_vol_mesh must be True when run_flag is True.")

    if generate_vol_mesh:
        project.generate_volume_mesh(
            params,
            use_beta_mesher=True,
            name="VolumeMesh",
            run_async=False,
            use_geometry_AI=False,
            raise_on_error=True,
        )

    if not run_flag:
        return project.id

    project.run_case(params=params, name="XWing2_2_case_" + sim_name, raise_on_error=True)
    case = project.case
    case.wait()

    results = case.results
    results.download(
        surface=True,
        volume=True,
        total_forces=True,
        nonlinear_residuals=True,
        destination=os.path.join(results_path, case.name),
    )

    return case.results.total_forces.as_dataframe()


def main():
    working_dir = Path(r"C:\WDIR\flow360\AMDC\XWing2_2")
    working_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(working_dir)

    wing_version = "rectangular"
    #wing_version = "trapezoidal"

    if wing_version == "rectangular":
        project_cgns_file_name = (
            r"C:/git/flow360cases/AMDC/XWing/2026-06-29_AMDC-simplified-XWingV22+rectangularwings_getrennt.cgns"
        )
        sim_name = "XWing2_2 rect fully_turbulent_SA"
        wing_area = 0.2831
        wing_span = 1.312
        wing1_TE_file = "TE_XWing2_2_rect_wing1.dat"
    elif wing_version == "trapezoidal":
        project_cgns_file_name = (
            r"C:/git/flow360cases/AMDC/XWing/2026-05-12_AMDC-simplified-XWingV22_getrennt_manual_V2.cgns"
        )
        sim_name = "XWing2_2 trap fully_turbulent_SA"
        wing_area = 0.277649964016683
        wing_span = 1.346
        wing1_TE_file = "TE_XWing2_2_wing1.dat"


    generate_vol_mesh = True
    run = False
    enable_volume_refinements = True
    enable_alpha_controller = True
    boundary_layer_growth_rate = 1.2
    target_yplus = 0.67
    n_timesteps = 600

    study_name = "AMDC XWing2_2"

    half_model = False
    aircraft_mass = 13.6
    U_inf_range = [24.5]
    alpha_deg_range = [10.0]
    #U_inf_range = [39.5]
    #alpha_deg_range = [-1.6]

    wake_refinement_files = prepare_trailing_edge_files(working_dir, wing1_file=wing1_TE_file)

    operating_points = []
    standard_atmosphere_density = calculate_standard_atmosphere_density(0)


    for U_inf, alpha_deg in product(U_inf_range, alpha_deg_range):
        operating_points.append(
            {
                "U_inf": U_inf,
                "target_lift_coefficient": calculate_target_lift_coefficient(
                    aircraft_mass=aircraft_mass,
                    wing_area=wing_area,
                    freestream_velocity=U_inf,
                    density=standard_atmosphere_density,
                ),
                "alpha_deg": alpha_deg,
            }
        )

    cols = ["U_inf", "target_lift_coefficient", "alpha_deg", "CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"]
    df_results = pd.DataFrame(operating_points).reindex(columns=cols)

    curr_folder = fl.Folder.create(study_name).submit()

    for i, row in df_results.iterrows():
        res = define_and_run(
            project_cgns_file_name=project_cgns_file_name,
            name=sim_name,
            U_inf=row["U_inf"],
            target_lift_coefficient=row["target_lift_coefficient"],
            alpha_deg=row["alpha_deg"],
            half_model=half_model,
            enable_volume_refinements=enable_volume_refinements,
            enable_alpha_controller=enable_alpha_controller,
            alpha_controller_initial_alpha_deg=row["alpha_deg"],
            aircraft_mass=aircraft_mass,
            wing_area=wing_area,
            wing_span=wing_span,
            wake_refinement_files=wake_refinement_files,
            flow360folder=curr_folder,
            results_path=str(working_dir),
            generate_vol_mesh=generate_vol_mesh,
            boundary_layer_growth_rate=boundary_layer_growth_rate,
            target_yplus=target_yplus,
            n_timesteps=n_timesteps,
            run_flag=run,
        )

        if run:
            df_results.loc[i, "CL"] = res["CL"].tail(15).mean()
            df_results.loc[i, "CD"] = res["CD"].tail(15).mean()
            df_results.loc[i, "CFx"] = res["CFx"].tail(15).mean()
            df_results.loc[i, "CFy"] = res["CFy"].tail(15).mean()
            df_results.loc[i, "CFz"] = res["CFz"].tail(15).mean()
            df_results.loc[i, "CMx"] = res["CMx"].tail(15).mean()
            df_results.loc[i, "CMy"] = res["CMy"].tail(15).mean()
            df_results.loc[i, "CMz"] = res["CMz"].tail(15).mean()

    if run:
        df_results.to_csv(working_dir / "XWing2_2_results.csv", index=False)


if __name__ == "__main__":
    main()
