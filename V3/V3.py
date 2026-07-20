import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import flow360 as fl
from flow360 import u

from flow360_utils.flow360_tools import (
    calc_ncrit_from_fsti,
    calculate_flat_plate_turbulent_y1,
    calculate_freestream_velocity_for_target_lift_coefficient,
    calculate_standard_atmosphere_density,
    calculate_target_lift_coefficient,
    geometric_spacing_values,
    make_refinement_cylinders_along_axis,
    make_segment_boxes,
    read_highest_yz_curvature_vertex,
    read_last_vertex,
)


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_ALTITUDE = 0
DEFAULT_AIRCRAFT_MASS = 600.0
DEFAULT_WING_AREA = 10.84
DEFAULT_WINGSPAN = 18.0
DEFAULT_MAC = 0.6356086024985311
DEFAULT_MOMENT_CENTER_X = 0.386666666666667
DEFAULT_TARGET_YPLUS = 0.67
DEFAULT_TURBULENCE_INTENSITY_PERC = 0.05


def _resolve_input_file(filename: str, working_dir=None) -> Path:
    candidates = [Path(filename)]
    if working_dir is not None:
        candidates.append(working_dir / filename)
    candidates.append(SCRIPT_DIR / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {filename!r}. Checked: {checked_paths}.")


def _resolve_input_files(filenames, working_dir=None):
    if filenames is None:
        return None
    return [str(_resolve_input_file(filename, working_dir)) for filename in filenames]


def define_and_run(project_cgns_file_name=None, name=None,
                   symm_face="body00001_face00003", U_inf=None, alpha_deg=0., half_model=True,
                   y1_fac=1., surf_mesh_lvl=0, enable_volume_refinements=True,
                   enable_alpha_controller=False, target_lift_coefficient=None,
                   alpha_controller_kp=0.2, alpha_controller_ki=0.002,
                   alpha_controller_start_pseudo_step=50,
                   alpha_controller_initial_alpha_deg=None,
                   aircraft_mass=DEFAULT_AIRCRAFT_MASS,
                   wing_area=DEFAULT_WING_AREA,
                   wingspan=DEFAULT_WINGSPAN,
                   mac=DEFAULT_MAC,
                   moment_center_x=DEFAULT_MOMENT_CENTER_X,
                   altitude=DEFAULT_ALTITUDE,
                   target_yplus=DEFAULT_TARGET_YPLUS,
                   turbulence_intensity_perc=DEFAULT_TURBULENCE_INTENSITY_PERC,
                   turbulator_location_files=None, wake_refinement_files=None,
                   flow360folder=None, results_path=None,
                   generate_vol_mesh=True,
                   boundary_layer_growth_rate=1.1,
                   n_timesteps=2000,
                   run_flag=False):
    """
    Define a V3 Flow360 setup and either prepare the next mesh step or run the case.

    :param project_cgns_file_name: CGNS surface mesh file used to create the project
    :param name:                   Simulation name prefix
    :param symm_face:              Symmetry face name
    :param U_inf:               Free stream velocity. Mutually exclusive with target_lift_coefficient
    :param alpha_deg:           Angle of attack in degrees
    :param half_model:          Half model flag. True for half-model, False for full model
    :param y1_fac:              first layer thickness scaling factor
    :param surf_mesh_lvl:       Mesh refinement level
    :param enable_volume_refinements: Enable wake and winglet volumetric refinement zones
    :param enable_alpha_controller: Enable a PI controller that adjusts alpha to reach target_lift_coefficient
    :param target_lift_coefficient: Target CL. Mutually exclusive with U_inf
    :param alpha_controller_kp:  Proportional gain for the alpha controller
    :param alpha_controller_ki:  Integral gain for the alpha controller
    :param alpha_controller_start_pseudo_step: Pseudo-step after which alpha control starts
    :param alpha_controller_initial_alpha_deg: Initial alpha angle in degrees for the controller state
    :param aircraft_mass:       Aircraft mass in kg used to calculate target CL when not specified
    :param wing_area:           Full aircraft wing area in square meters
    :param wingspan:            aircraft wingspan in meters
    :param mac:                 Mean aerodynamic chord in meters
    :param moment_center_x:     Moment reference x-coordinate in meters
    :param altitude:            Standard atmosphere altitude in meters
    :param target_yplus:        Target y-plus value for wall-resolved meshing
    :param turbulence_intensity_perc: Freestream turbulence intensity in percent
    :param turbulator_location_files: Geometry-specific turbulator coordinate files
    :param wake_refinement_files: Geometry-specific wake refinement coordinate files
    :param flow360folder:       Flow360 folder to put the case in
    :param results_path:        Path to store results
    :param generate_vol_mesh:   Generate the volume mesh before returning or running the case
    :param boundary_layer_growth_rate: Boundary layer growth rate for volume meshing
    :param n_timesteps:         Maximum steady solver pseudo-steps
    :param run_flag:            Flag, determines, if simulation is only set-up (False) or also run (True)
    :return:
    """
    if alpha_controller_initial_alpha_deg is None:
        alpha_controller_initial_alpha_deg = alpha_deg

    async_flag = False
    surf_mesh_refine_factor = 2 ** (surf_mesh_lvl / 2)  # Surface mesh size multiplier
    standard_atmosphere_density = calculate_standard_atmosphere_density(altitude)
    has_airspeed = U_inf is not None
    has_target_lift_coefficient = target_lift_coefficient is not None
    if has_airspeed == has_target_lift_coefficient:
        raise ValueError("Specify exactly one of U_inf or target_lift_coefficient.")

    if has_airspeed:
        target_lift_coefficient = calculate_target_lift_coefficient(
            aircraft_mass=aircraft_mass,
            wing_area=wing_area,
            freestream_velocity=U_inf,
            density=standard_atmosphere_density,
        )
    else:
        U_inf = calculate_freestream_velocity_for_target_lift_coefficient(
            aircraft_mass=aircraft_mass,
            wing_area=wing_area,
            target_lift_coefficient=target_lift_coefficient,
            density=standard_atmosphere_density,
        )

    # turbulator definition
    turbulator_box_x_size = 4
    turbulator_box_z_size = 1.6
    turbulator_box_x_offset = 0
    turbulator_box_y_offset = 0
    turbulator_box_z_offset = 0

    # wake volumetric refinement
    wake_refinement_angle_deg = 15
    wake_refinement_length = 1000
    wake_refinement_delta_x = 100
    wake_refinement_box_overlap = 5
    wake_refinement_spanwise_overlap = 20
    wake_refinement_initial_spacing = surf_mesh_refine_factor * 20 * 3 ** (1 / 2)
    wake_refinement_spacing_growth_rate = 1.2
    wake_refinement_max_spacing = None

    # winglet tip cylindrical volume refinement: 10 adjacent cylinders over 2 m total length
    winglet_tip_refinement_axis = (1, 0, 0)
    winglet_tip_refinement_length = 2000
    winglet_tip_refinement_n_cylinders = 10
    winglet_tip_refinement_initial_diameter = 100
    winglet_tip_refinement_growth_angle_deg = 5
    winglet_tip_refinement_cylinder_overlap = 5
    winglet_tip_refinement_initial_spacing = surf_mesh_refine_factor * 30 * 3 ** (1 / 2)
    winglet_tip_refinement_spacing_growth_rate = 1.2
    winglet_tip_refinement_max_spacing = None

    # winglet radius cylindrical volume refinement: start at max y-z trailing-edge curvature
    winglet_radius_refinement_fallback_start_center = (1267.0, 8997.5, 1357.4)
    winglet_radius_refinement_axis = (1, 0, 0)
    winglet_radius_refinement_length = 2000
    winglet_radius_refinement_n_cylinders = 10
    winglet_radius_refinement_initial_diameter = 100
    winglet_radius_refinement_growth_angle_deg = 5
    winglet_radius_refinement_cylinder_overlap = 5
    winglet_radius_refinement_initial_spacing = surf_mesh_refine_factor * 30 * 3 ** (1 / 2)
    winglet_radius_refinement_spacing_growth_rate = 1.2
    winglet_radius_refinement_max_spacing = None

    ns_solver_tolerance = 1.0e-10
    turb_solver_tolerance = 1.0e-8
    if enable_alpha_controller and alpha_controller_start_pseudo_step >= n_timesteps:
        raise ValueError(
            "alpha_controller_start_pseudo_step must be smaller than n_timesteps "
            f"({n_timesteps}) for the alpha controller to activate."
        )

    # First layer volumetric mesh thicknesses
    _, flat_plate_y1 = calculate_flat_plate_turbulent_y1(
        freestream_velocity=U_inf,
        desired_yplus=target_yplus,
    )
    global_y1 = y1_fac * flat_plate_y1 * u.m  # First layer thickness for boundary layer meshing (wall-resolved)

    # solver parameters
    surf_output_requests = ["Cp", "Cf", "yPlus", "CfVec", "solutionTransition"]
    vol_output_requests = ["primitiveVars", "Cp", "qcriterion", "mut", "solutionTransition", "T", "vorticity"]

    if name is not None:
        sim_name = name + " "
    else:
        sim_name = ""
    if has_airspeed:
        sim_name += "U{0:g}_AOA{1:g}".format(U_inf, alpha_deg)
    else:
        sim_name += "CL{0:g}".format(target_lift_coefficient)

    if surf_mesh_lvl != 0:
        sim_name += "_mshlvl{0:d}".format(surf_mesh_lvl)

    ###############################
    # Preface: Create a new project
    ###############################

    # Initialize project
    if project_cgns_file_name is None:
        raise ValueError("project_cgns_file_name must be specified.")

    working_dir = Path.cwd()
    project_cgns_file_name = str(_resolve_input_file(project_cgns_file_name, working_dir))
    project = fl.Project.from_surface_mesh(project_cgns_file_name, name=sim_name,
                                           folder=flow360folder, length_unit="mm", run_async=async_flag)
    surface_mesh = project.surface_mesh

    if turbulator_location_files is None:
        raise ValueError("turbulator_location_files must be provided.")
    if wake_refinement_files is None:
        raise ValueError("wake_refinement_files must be provided.")

    turbulator_location_files = _resolve_input_files(turbulator_location_files, working_dir)
    wake_refinement_files = _resolve_input_files(wake_refinement_files, working_dir)

    wall_surfaces = [
        surface_mesh[boundary_name]
        for boundary_name in surface_mesh.boundary_names
        if not half_model or boundary_name != symm_face
    ]

    ################################
    # 1) Define operating conditions
    ################################
    condition = fl.AerospaceCondition(velocity_magnitude=U_inf * u.m / u.s, alpha=alpha_deg * u.deg, beta=0 * u.deg)

    ################################
    # 2) Define mesh
    ################################

    # 3a) Farfield
    far_field_zone = fl.AutomatedFarfield()

    # 3b) Mesh parameters
    mesh_defaults = fl.MeshingDefaults(surface_edge_growth_rate=1.2,
                                       surface_max_edge_length=surf_mesh_refine_factor * 80 * u.mm,
                                       curvature_resolution_angle=surf_mesh_refine_factor * 5 * u.deg,
                                       boundary_layer_growth_rate=boundary_layer_growth_rate,
                                       boundary_layer_first_layer_thickness=global_y1)

    # 3c) Rotation region
    # None..

    # 3d) Mesh refinements
    refinements = []

    turbulator_boxes = list()
    for file in turbulator_location_files:
        boxes = make_segment_boxes(
            vertices_coords_file=file,
            x_size=turbulator_box_x_size,
            z_size=turbulator_box_z_size,
            center_offset=(turbulator_box_x_offset, turbulator_box_y_offset, turbulator_box_z_offset),
            name_prefix=file.rstrip(".dat"),
        )
        turbulator_boxes.append(boxes)

    if enable_volume_refinements:
        wake_refinement_rows = list()
        for file in wake_refinement_files:
            h_box = 0
            for i_row, x in enumerate(np.arange(0, wake_refinement_length, wake_refinement_delta_x)):
                h_box += wake_refinement_delta_x * 2 * np.sin(np.deg2rad(wake_refinement_angle_deg / 2))
                boxes = make_segment_boxes(
                    vertices_coords_file=file,
                    x_size=wake_refinement_delta_x + wake_refinement_box_overlap,
                    z_size=h_box + wake_refinement_box_overlap,
                    center_offset=(x + wake_refinement_delta_x / 2, 0, -0.5),
                    name_prefix="wakebox_row{0:0d}".format(i_row),
                    segment_overlap=wake_refinement_spanwise_overlap,
                )
                wake_refinement_rows.append((i_row, boxes))

        # Volume refinement at winglet tip: staggered cylinders with 5 deg diameter growth
        winglet_tip_refinement_start_center = read_last_vertex(wake_refinement_files[-1])
        winglet_tip_cylinders = make_refinement_cylinders_along_axis(
            name_prefix="winglet_tip_cylinder",
            start_center=winglet_tip_refinement_start_center,
            axis=winglet_tip_refinement_axis,
            total_length=winglet_tip_refinement_length,
            n_cylinders=winglet_tip_refinement_n_cylinders,
            initial_diameter=winglet_tip_refinement_initial_diameter,
            growth_angle_deg=winglet_tip_refinement_growth_angle_deg,
            cylinder_overlap=winglet_tip_refinement_cylinder_overlap,
        )
        winglet_tip_spacings = geometric_spacing_values(
            initial_spacing=winglet_tip_refinement_initial_spacing,
            growth_rate=winglet_tip_refinement_spacing_growth_rate,
            n_values=len(winglet_tip_cylinders),
            max_spacing=winglet_tip_refinement_max_spacing,
        )
        for idx, (cylinder, spacing) in enumerate(zip(winglet_tip_cylinders, winglet_tip_spacings), start=1):
            refinements.append(
                fl.UniformRefinement(
                    name=f"winglet_tip_refinement_{idx:03d}",
                    entities=[cylinder],
                    spacing=spacing * u.mm,
                )
            )

        # Volume refinement at winglet radius: start at highest y-z curvature on the trailing edge
        winglet_radius_refinement_start_center = read_highest_yz_curvature_vertex(
            wake_refinement_files[-1],
            fallback_vertex=winglet_radius_refinement_fallback_start_center,
        )
        winglet_radius_cylinders = make_refinement_cylinders_along_axis(
            name_prefix="winglet_radius_cylinder",
            start_center=winglet_radius_refinement_start_center,
            axis=winglet_radius_refinement_axis,
            total_length=winglet_radius_refinement_length,
            n_cylinders=winglet_radius_refinement_n_cylinders,
            initial_diameter=winglet_radius_refinement_initial_diameter,
            growth_angle_deg=winglet_radius_refinement_growth_angle_deg,
            cylinder_overlap=winglet_radius_refinement_cylinder_overlap,
        )
        winglet_radius_spacings = geometric_spacing_values(
            initial_spacing=winglet_radius_refinement_initial_spacing,
            growth_rate=winglet_radius_refinement_spacing_growth_rate,
            n_values=len(winglet_radius_cylinders),
            max_spacing=winglet_radius_refinement_max_spacing,
        )
        for idx, (cylinder, spacing) in enumerate(zip(winglet_radius_cylinders, winglet_radius_spacings), start=1):
            refinements.append(
                fl.UniformRefinement(
                    name=f"winglet_radius_refinement_{idx:03d}",
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

    # make mesh parameters
    mesh_params = fl.MeshingParams(defaults=mesh_defaults,
                                   volume_zones=[far_field_zone],
                                   refinements=refinements if len(refinements) > 0 else None)

    ###########################
    # 4) Flow solver parameters
    ###########################

    moment_ref_lengths = (wingspan, mac, wingspan)

    ref_geometry = fl.ReferenceGeometry(moment_center=(moment_center_x, 0, 0) * u.m,
                                        moment_length=moment_ref_lengths * u.m,
                                        area=wing_area / 2 * u.m ** 2)

    ncrit = calc_ncrit_from_fsti(turbulence_intensity_perc)

    navier_stokes_solver = fl.NavierStokesSolver(absolute_tolerance=ns_solver_tolerance,
                                                 kappa_MUSCL=0.33,
                                                 linear_solver=fl.KrylovLinearSolver())
    transition_solver = fl.TransitionModelSolver(N_crit=ncrit,
                                                 trip_regions=turbulator_boxes if len(turbulator_boxes) > 0 else None)
    turbulence_solver = fl.SpalartAllmaras(absolute_tolerance=turb_solver_tolerance)

    fl_models = [fl.Wall(surfaces=wall_surfaces, use_wall_function=None),
                 fl.Freestream(surfaces=[far_field_zone.farfield]),
                 fl.Fluid(navier_stokes_solver=navier_stokes_solver,
                          transition_model_solver=transition_solver,
                          turbulence_model_solver=turbulence_solver)]

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
        # Set up the main simulation parameters
        params = fl.SimulationParams(meshing=mesh_params,
                                     reference_geometry=ref_geometry,
                                     operating_condition=condition,
                                     time_stepping=fl.Steady(max_steps=n_timesteps),
                                     models=fl_models,
                                     outputs=[
                                         fl.SurfaceOutput(surfaces=wall_surfaces, output_fields=surf_output_requests,
                                                          write_single_file=True),
                                         fl.VolumeOutput(name="VolumeOutput", output_format="paraview",
                                                         output_fields=vol_output_requests)]

                                     )
        if user_defined_dynamics:
            params.user_defined_dynamics = user_defined_dynamics

    ###############################
    # 5) Generate mesh and run case
    ###############################
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

    project.run_case(params=params, name="V3_case_" + sim_name)

    case = project.case
    case.wait()

    results = case.results
    results.download(surface=True, volume=True, total_forces=True, nonlinear_residuals=True,
                     destination=os.path.join(results_path, case.name))

    total_forces = case.results.total_forces.as_dataframe()

    return total_forces


def main():
    mshlvl = 0
    generate_vol_mesh = True
    run = False
    n_test_cases = None
    enable_volume_refinements = True
    boundary_layer_growth_rate = 1.097
    enable_alpha_controller = True
    aircraft_mass = DEFAULT_AIRCRAFT_MASS
    alpha_controller_kp = 0.2
    alpha_controller_ki = 0.002
    alpha_controller_start_pseudo_step = 50

    results_dir = "C:/WDIR/flow360"

    # variant = "FlapletV2 WK-1"
    # variant = "Original WK-1"
    # variant = "FlapletV2 WK+2"
    # variant = "Original WK+2"
    # variant = "Original WK+2_15"
    variant = "FlapletV2 WK+2_15"

    sim_name = "V3 " + variant

    default_boundary_layer_growth_rate = 1.1
    if boundary_layer_growth_rate != default_boundary_layer_growth_rate:
        sim_name += f"_VolGR={boundary_layer_growth_rate:g}"

    study_name = "V3 Flaplets"
    half_model = True
    altitude = DEFAULT_ALTITUDE
    wing_area = DEFAULT_WING_AREA
    wingspan = DEFAULT_WINGSPAN
    mac = DEFAULT_MAC
    moment_center_x = DEFAULT_MOMENT_CENTER_X
    U_inf_range = None

    variant_configs = {
        "Original WK-1": {
            "project_cgns_file_name": "Ventus_Original_WK-1_G.cgns",
            "symm_face": "symm face",
            "turbulator_location_files": ["Turbulator_wing_lower_original_WK-1.dat"],
            "wake_refinement_files": ["TE_upper_VentusOrig_WK-1.dat"],
            #"target_lift_coefficient_range": [0.3, 0.5],
            # "alpha_deg_range": [3.0],
            #"target_lift_coefficient_range": [0.336899319, 0.381049961, 0.434484293],
            #"alpha_deg_range": [1.1, 1.5, 2.0],
            "target_lift_coefficient_range": [0.381049961, 0.434484293],
            "alpha_deg_range": [1.5, 2.0],
            "n_timesteps": 700,
        },
        "Original WK+2": {
            "project_cgns_file_name": "Ventus_Original_WK+2.cgns",
            "symm_face": "symm face",
            "turbulator_location_files": [],
            "wake_refinement_files": ["TE_upper_VentusOrig_WK+2.dat"],
            "target_lift_coefficient_range": [1.2],
            "alpha_deg_range": [1.85],
            "n_timesteps": 1000,
        },
        "Original WK+2_15": {
            "project_cgns_file_name": "Ventus_Original_WK+2_15.cgns",
            "symm_face": "symm face",
            "turbulator_location_files": [],
            "wake_refinement_files": ["TE_upper_VentusOrig_WK+2_15.dat"],
            "target_lift_coefficient_range": 1 / np.linspace((1/0.8)**0.5, (1/1.3)**0.5, 3)**2,
            "alpha_deg_range": [-0.87, 0, 1, 2.45, 4.4],
            "n_timesteps": 1000,
        },
        "FlapletV2 WK-1": {
            "project_cgns_file_name": "Ventus3_FlapletV2_WK-1_B.cgns",
            "symm_face": "symm face",
            "turbulator_location_files": [
                "Turbulator_wing_lower_FlapletV2_WK-1.dat",
                "Turbulator_Flaplet_upper_WK-1.dat",
            ],
            "wake_refinement_files": ["TE_upper_Flaplet_WK-1.dat"],
            #"target_lift_coefficient_range": [0.5],
            #"alpha_deg_range": [3.0],
            #"target_lift_coefficient_range": [0.381049961, 0.434484293],
            #"alpha_deg_range": [1.5, 2.0],
            "target_lift_coefficient_range": [0.434484293],
            "alpha_deg_range": [2.0],
            "n_timesteps": 700,
        },
        "FlapletV2 WK+2": {
            "project_cgns_file_name": "Ventus3_FlapletV2_WK+2_B_15.cgns",
            "symm_face": "symm face",
            "turbulator_location_files": [
                "Turbulator_Flaplet_upper_WK-1.dat",
            ],
            "wake_refinement_files": ["TE_upper_Flaplet_WK+2.dat"],
            "target_lift_coefficient_range": [1.2],
            "alpha_deg_range": [1.85],
            "n_timesteps": 1000,
        },
        "FlapletV2 WK+2_15": {
            "project_cgns_file_name": "Ventus_FlapletV2_WK+2_15.cgns",
            "symm_face": "symm face",
            "turbulator_location_files": [
                "Turbulator_Flaplet_upper_WK-1.dat",
            ],
            "wake_refinement_files": ["TE_upper_Flaplet_WK+2_15.dat"],
            "target_lift_coefficient_range": 1 / np.linspace((1 / 0.8) ** 0.5, (1 / 1.3) ** 0.5, 3) ** 2,
            "alpha_deg_range": [-0.87, 1, 4.4],
            "n_timesteps": 1000,
        },
    }

    if variant not in variant_configs:
        raise ValueError(
            f"Invalid variant {variant!r}. Available variants: {list(variant_configs)}"
        )

    variant_config = variant_configs[variant]
    proj_cgns_file = variant_config["project_cgns_file_name"]
    symm_face = variant_config["symm_face"]
    turbulator_location_files = variant_config["turbulator_location_files"]
    wake_refinement_files = variant_config["wake_refinement_files"]
    target_lift_coefficient_range = variant_config["target_lift_coefficient_range"]
    alpha_deg_range = variant_config["alpha_deg_range"]
    n_timesteps = variant_config["n_timesteps"]

    has_airspeed_range = U_inf_range is not None
    has_target_lift_coefficient_range = target_lift_coefficient_range is not None
    if has_airspeed_range == has_target_lift_coefficient_range:
        raise ValueError("Specify exactly one of U_inf_range or target_lift_coefficient_range.")

    standard_atmosphere_density = calculate_standard_atmosphere_density(altitude)
    operating_points = []
    if has_airspeed_range:
        for U_inf, alpha_deg in product(U_inf_range, alpha_deg_range):
            operating_points.append({
                "U_inf": U_inf,
                "target_lift_coefficient": calculate_target_lift_coefficient(
                    aircraft_mass=aircraft_mass,
                    wing_area=wing_area,
                    freestream_velocity=U_inf,
                    density=standard_atmosphere_density,
                ),
                "alpha_deg": alpha_deg,
                "input_mode": "airspeed",
            })
    else:
        if enable_alpha_controller:
            for target_cl, alpha_deg in zip(target_lift_coefficient_range, alpha_deg_range):
                operating_points.append({
                    "U_inf": calculate_freestream_velocity_for_target_lift_coefficient(
                        aircraft_mass=aircraft_mass,
                        wing_area=wing_area,
                        target_lift_coefficient=target_cl,
                        density=standard_atmosphere_density,
                    ),
                    "target_lift_coefficient": target_cl,
                    "alpha_deg": alpha_deg,
                    "input_mode": "target_lift_coefficient",
                })
        else:
            for target_cl, alpha_deg in product(target_lift_coefficient_range, alpha_deg_range):
                operating_points.append({
                    "U_inf": calculate_freestream_velocity_for_target_lift_coefficient(
                        aircraft_mass=aircraft_mass,
                        wing_area=wing_area,
                        target_lift_coefficient=target_cl,
                        density=standard_atmosphere_density,
                    ),
                    "target_lift_coefficient": target_cl,
                    "alpha_deg": alpha_deg,
                    "input_mode": "target_lift_coefficient",
                })
    # initialize results DataFrame
    cols = ["U_inf", "target_lift_coefficient", "alpha_deg", "CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"]
    df_results = pd.DataFrame(operating_points).reindex(columns=cols + ["input_mode"])

    if n_test_cases is not None:
        df_results = df_results.iloc[:n_test_cases]

    folder_toplvl = fl.Folder.create(study_name).submit()
    curr_folder = fl.Folder.create(variant, parent_folder=folder_toplvl).submit()

    for i, row in df_results.iterrows():
        U_inf = row["U_inf"]
        target_cl = row["target_lift_coefficient"]
        alpha_deg = row["alpha_deg"]
        input_mode = row["input_mode"]

        res = define_and_run(
            project_cgns_file_name=proj_cgns_file,
            name=sim_name,
            symm_face=symm_face,
            U_inf=U_inf if input_mode == "airspeed" else None,
            target_lift_coefficient=target_cl if input_mode == "target_lift_coefficient" else None,
            alpha_deg=alpha_deg,
            half_model=half_model,
            surf_mesh_lvl=mshlvl,
            enable_volume_refinements=enable_volume_refinements,
            enable_alpha_controller=enable_alpha_controller,
            alpha_controller_kp=alpha_controller_kp,
            alpha_controller_ki=alpha_controller_ki,
            alpha_controller_start_pseudo_step=alpha_controller_start_pseudo_step,
            alpha_controller_initial_alpha_deg=alpha_deg,
            aircraft_mass=aircraft_mass,
            wing_area=wing_area,
            wingspan=wingspan,
            mac=mac,
            moment_center_x=moment_center_x,
            altitude=altitude,
            turbulator_location_files=turbulator_location_files,
            wake_refinement_files=wake_refinement_files,
            flow360folder=curr_folder,
            results_path=results_dir,
            generate_vol_mesh=generate_vol_mesh,
            boundary_layer_growth_rate=boundary_layer_growth_rate,
            n_timesteps=n_timesteps,
            run_flag=run,
        )

        if run:
            CL_avg = res["CL"].tail(15).mean()
            CD_avg = res["CD"].tail(15).mean()
            CFx = res["CFx"].tail(15).mean()
            CFy = res["CFy"].tail(15).mean()
            CFz = res["CFz"].tail(15).mean()
            CMx_avg = res["CMx"].tail(15).mean()
            CMy_avg = res["CMy"].tail(15).mean()
            CMz_avg = res["CMz"].tail(15).mean()
            df_results.loc[i, "CL"] = CL_avg
            df_results.loc[i, "CD"] = CD_avg
            df_results.loc[i, "CMx"] = CMx_avg
            df_results.loc[i, "CMy"] = CMy_avg
            df_results.loc[i, "CMz"] = CMz_avg
            df_results.loc[i, "CFx"] = CFx
            df_results.loc[i, "CFy"] = CFy
            df_results.loc[i, "CFz"] = CFz

    if run:
        df_results.drop(columns=["input_mode"]).to_csv(study_name + ".csv", index=False)


if __name__ == "__main__":
    main()
