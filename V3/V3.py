# Import necessary modules from the Flow360 library
import os
from itertools import product
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

import flow360 as fl
from flow360 import u


def calculate_flat_plate_turbulent_y1(
        freestream_velocity=55.0,
        density=1.225,
        dynamic_viscosity=1.82e-5,
        boundary_layer_length=0.745,
        desired_yplus=0.67):
    """
    Estimate first-cell wall distance from turbulent flat-plate boundary layer theory.

    Units:
        freestream_velocity: m/s
        density: kg/m3
        dynamic_viscosity: kg/(m s)
        boundary_layer_length: m
        desired_yplus: nondimensional

    Returns:
        reynolds_number: nondimensional
        y1: m
    """
    reynolds_number = density * freestream_velocity * boundary_layer_length / dynamic_viscosity
    skin_friction_coefficient = 0.0576 / reynolds_number**0.2
    wall_shear_stress = 0.5 * density * freestream_velocity**2 * skin_friction_coefficient
    friction_velocity = np.sqrt(wall_shear_stress / density)
    y1 = desired_yplus * dynamic_viscosity / (density * friction_velocity)

    return reynolds_number, y1


def calculate_standard_atmosphere_density(altitude=0.0):
    """
    Calculate ISA density in the troposphere.

    Units:
        altitude: m

    Returns:
        density: kg/m3
    """
    sea_level_temperature = 288.15
    sea_level_pressure = 101325.0
    temperature_lapse_rate = 0.0065
    gas_constant_air = 287.05287
    gravity = 9.80665

    temperature = sea_level_temperature - temperature_lapse_rate * altitude
    pressure = sea_level_pressure * (
        temperature / sea_level_temperature
    ) ** (gravity / (gas_constant_air * temperature_lapse_rate))

    return pressure / (gas_constant_air * temperature)


def calculate_target_lift_coefficient(aircraft_mass, wing_area, freestream_velocity, density):
    gravity = 9.80665
    dynamic_pressure = 0.5 * density * freestream_velocity**2

    return aircraft_mass * gravity / (dynamic_pressure * wing_area)


def calculate_freestream_velocity_for_target_lift_coefficient(
        aircraft_mass, wing_area, target_lift_coefficient, density):
    gravity = 9.80665

    if target_lift_coefficient <= 0:
        raise ValueError("target_lift_coefficient must be positive to calculate freestream velocity.")

    return np.sqrt(aircraft_mass * gravity / (0.5 * density * wing_area * target_lift_coefficient))


def calc_ncrit_from_fsti(turbulence_intensity_perc=0.05):
    """
    calculates critical N factor from freestream turbulence intensity according to paper from Djeddi, Coder et al.
    "Adjoint-Based Uncertainty Quantiﬁcation and Calibration of RANS-Based Transition Modeling" DOI: 10.2514/6.2021-3036
    :param turbulence_intensity_perc: Turbulence intensity in percent
    :return: NCrit: critical amplification factor
    """
    a0 = 9.0064
    a1 = -4.4958
    a2 = -1.4208
    a3 = 1.5920
    a4 = -0.3532

    tau = 2.5 * np.tanh(turbulence_intensity_perc/2.5)
    Ncrit = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4

    return Ncrit


def _resolve_existing_path(filename):
    """Resolve a data file from cwd first, then relative to this script."""
    if filename is None:
        return None

    candidate_paths = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for candidate_path in candidate_paths:
        if os.path.isfile(candidate_path):
            return candidate_path

    raise FileNotFoundError(
        f"Could not find turbulator location file '{filename}' in the current working directory "
        f"or next to {__file__}."
    )


def _axis_perpendicular_to(segment_axis, preferred_axis):
    """Return preferred_axis projected onto the plane normal to segment_axis."""
    segment_axis = np.asarray(segment_axis, dtype=float)
    preferred_axis = np.asarray(preferred_axis, dtype=float)

    projected_axis = preferred_axis - np.dot(preferred_axis, segment_axis) * segment_axis
    projected_norm = np.linalg.norm(projected_axis)
    if projected_norm < 1e-12:
        return None

    return projected_axis / projected_norm


def make_boxes(vertices_coords_file, x_size, z_size, center_offset=(0, 0, 0),
               name_prefix="turbulator_box", segment_overlap=0):
    vertices_coords_path = _resolve_existing_path(vertices_coords_file)
    points = pd.read_csv(vertices_coords_path, sep=r"\s+", engine="python")

    required_columns = {"X", "Y", "Z"}
    missing_columns = required_columns - set(points.columns)
    if missing_columns:
        raise ValueError(
            f"Turbulator location file '{vertices_coords_path}' is missing columns: "
            f"{sorted(missing_columns)}"
        )

    coords = points[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(coords) < 2:
        raise ValueError(
            f"Turbulator location file '{vertices_coords_path}' must contain at least two points."
        )
    center_offset = np.asarray(center_offset, dtype=float)
    if center_offset.shape != (3,):
        raise ValueError("center_offset must contain exactly three values: x, y, z.")

    boxes = []
    for idx, (start, end) in enumerate(zip(coords[:-1], coords[1:]), start=1):
        segment = end - start
        segment_length = np.linalg.norm(segment)
        if segment_length <= 0:
            raise ValueError(
                f"Turbulator segment {idx} in '{vertices_coords_path}' has zero length."
            )

        y_axis = segment / segment_length
        x_axis = _axis_perpendicular_to(y_axis, (1, 0, 0))
        if x_axis is None:
            x_axis = _axis_perpendicular_to(y_axis, (0, 0, 1))

        boxes.append(
            fl.Box.from_principal_axes(
                name=f"{name_prefix}_{idx:03d}",
                axes=[tuple(x_axis), tuple(y_axis)],
                center=((start + end) / 2 + center_offset) * u.mm,
                size=(x_size, segment_length + segment_overlap, z_size) * u.mm,
            )
        )

    return boxes


def make_winglet_tip_refinement_cylinders(
        name_prefix, start_center, axis, total_length, n_cylinders, initial_diameter,
        growth_angle_deg, cylinder_overlap=0):
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        raise ValueError("Cylinder axis must have non-zero length.")
    axis = axis / axis_norm

    if n_cylinders <= 0:
        raise ValueError("n_cylinders must be positive.")

    cylinder_length = total_length / n_cylinders
    diameter_growth_per_cylinder = 2 * cylinder_length * np.tan(np.deg2rad(growth_angle_deg))
    start_center = np.asarray(start_center, dtype=float)

    cylinders = []
    for idx in range(n_cylinders):
        center = start_center + axis * (idx + 0.5) * cylinder_length
        diameter = initial_diameter + idx * diameter_growth_per_cylinder
        cylinders.append(
            fl.Cylinder(
                name=f"{name_prefix}_{idx + 1:03d}",
                center=center * u.mm,
                axis=tuple(axis),
                outer_radius=diameter / 2 * u.mm,
                height=(cylinder_length + cylinder_overlap) * u.mm,
            )
        )

    return cylinders


def geometric_spacing_values(initial_spacing, growth_rate, n_values, max_spacing=None):
    if n_values <= 0:
        return []

    spacings = [initial_spacing * growth_rate**idx for idx in range(n_values)]
    if max_spacing is not None:
        spacings = [min(spacing, max_spacing) for spacing in spacings]

    return spacings


def read_last_vertex(vertices_coords_file):
    vertices_coords_path = _resolve_existing_path(vertices_coords_file)
    points = pd.read_csv(vertices_coords_path, sep=r"\s+", engine="python")

    required_columns = {"X", "Y", "Z"}
    missing_columns = required_columns - set(points.columns)
    if missing_columns:
        raise ValueError(
            f"Vertex coordinate file '{vertices_coords_path}' is missing columns: "
            f"{sorted(missing_columns)}"
        )
    if points.empty:
        raise ValueError(f"Vertex coordinate file '{vertices_coords_path}' does not contain any points.")

    return tuple(points[["X", "Y", "Z"]].iloc[-1].to_numpy(dtype=float))


def read_highest_yz_curvature_vertex(vertices_coords_file, fallback_vertex=None):
    vertices_coords_path = _resolve_existing_path(vertices_coords_file)
    points = pd.read_csv(vertices_coords_path, sep=r"\s+", engine="python")

    required_columns = {"X", "Y", "Z"}
    missing_columns = required_columns - set(points.columns)
    if missing_columns:
        raise ValueError(
            f"Vertex coordinate file '{vertices_coords_path}' is missing columns: "
            f"{sorted(missing_columns)}"
        )

    coords = points[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(coords) < 3:
        if fallback_vertex is not None:
            return fallback_vertex
        raise ValueError(
            f"Vertex coordinate file '{vertices_coords_path}' must contain at least three points "
            "for curvature detection."
        )

    yz_coords = coords[:, [1, 2]]
    curvatures = np.full(len(coords), -np.inf)
    for idx in range(1, len(coords) - 1):
        previous_vector = yz_coords[idx] - yz_coords[idx - 1]
        next_vector = yz_coords[idx + 1] - yz_coords[idx]
        chord_vector = yz_coords[idx + 1] - yz_coords[idx - 1]

        previous_length = np.linalg.norm(previous_vector)
        next_length = np.linalg.norm(next_vector)
        chord_length = np.linalg.norm(chord_vector)
        if min(previous_length, next_length, chord_length) <= 0:
            continue

        twice_triangle_area = abs(np.cross(previous_vector, next_vector))
        curvatures[idx] = 2 * twice_triangle_area / (
            previous_length * next_length * chord_length
        )

    if not np.any(np.isfinite(curvatures)):
        if fallback_vertex is not None:
            return fallback_vertex
        raise ValueError(
            f"Could not determine finite y-z curvature from '{vertices_coords_path}'."
        )

    return tuple(coords[int(np.nanargmax(curvatures))])




def define_and_run(project_cgns_file_name=None, project_step_file_name=None, project_id=None, name=None,
                   symm_face = "body00001_face00003", U_inf=None, alpha_deg=0., half_model=True,
                   y1_fac=1., surf_mesh_lvl=0, enable_volume_refinements=True,
                   enable_alpha_controller=False, target_lift_coefficient=None,
                   alpha_controller_kp=0.2, alpha_controller_ki=0.002,
                   alpha_controller_start_pseudo_step=50,
                   alpha_controller_initial_alpha_deg=None,
                   aircraft_mass=600.0,
                   LE_edge_list=None, TE_edge_list=None,
                   turbulator_location_files=None, wake_refinement_files=None,
                   flow360folder=None, results_path=None,
                   generate_surf_mesh=True, generate_vol_mesh=True,
                   boundary_layer_growth_rate=1.1,
                   n_timesteps=2000,
                   run_flag = False):
    """
    Define a V3 Flow360 setup and either prepare the next mesh step or run the case.

    The project source is selected in this order:
    - project_cgns_file_name: create a project from an existing CGNS surface mesh
      and build surface models from the mesh boundaries.
    - project_step_file_name: create a project from CAD geometry and generate the
      surface mesh from geometry.
    - project_id: load an existing cloud project and use its geometry workflow.

    :param project_cgns_file_name: CGNS surface mesh file used to create the project
    :param project_step_file_name: CAD geometry file used to create the project
    :param project_id:             Existing Flow360 cloud project ID
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
    :param LE_edge_list:        Geometry-specific leading-edge IDs for CAD-based meshing
    :param TE_edge_list:        Geometry-specific trailing-edge IDs for CAD-based meshing
    :param turbulator_location_files: Geometry-specific turbulator coordinate files
    :param wake_refinement_files: Geometry-specific wake refinement coordinate files
    :param flow360folder:       Flow360 folder to put the case in
    :param results_path:        Path to store results
    :param generate_surf_mesh:  Generate the surface mesh when geometry-based meshing is used
    :param generate_vol_mesh:   Generate the volume mesh before returning or running the case
    :param boundary_layer_growth_rate: Boundary layer growth rate for volume meshing
    :param n_timesteps:         Maximum steady solver pseudo-steps
    :param run_flag:            Flag, determines, if simulation is only set-up (False) or also run (True)
    :return:
    """
    if alpha_controller_initial_alpha_deg is None:
        alpha_controller_initial_alpha_deg = alpha_deg

    #  #  #
    # global flags
    async_flag = False
    # global parameters
    altitude = 0
    # mesh parameters
    surf_mesh_refine_factor = 2**(surf_mesh_lvl/2)       # Surface mesh size multiplier
    target_yplus_wall_modeled = 0.67    # Target y-plus value for wall-resolved meshing

    turbulence_intensity_perc = 0.05

    mac = 0.6356086024985311  # mean aerodynamic chord
    wingspan = 18  # half-span
    moment_center_x = 0.386666666666667  # x reference location for moments
    wing_area = 10.84
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
    wake_refinement_initial_spacing = surf_mesh_refine_factor * 20 * 3**(1/2)
    wake_refinement_spacing_growth_rate = 1.2
    wake_refinement_max_spacing = None

    # winglet tip cylindrical volume refinement: 10 adjacent cylinders over 2 m total length
    winglet_tip_refinement_axis = (1, 0, 0)
    winglet_tip_refinement_length = 2000
    winglet_tip_refinement_n_cylinders = 10
    winglet_tip_refinement_initial_diameter = 100
    winglet_tip_refinement_growth_angle_deg = 5
    winglet_tip_refinement_cylinder_overlap = 5
    winglet_tip_refinement_initial_spacing = surf_mesh_refine_factor * 30 * 3**(1/2)
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
    winglet_radius_refinement_initial_spacing = surf_mesh_refine_factor * 30 * 3**(1/2)
    winglet_radius_refinement_spacing_growth_rate = 1.2
    winglet_radius_refinement_max_spacing = None

    ns_solver_tolerance = 1.e-7            # Navier-Stokes and turbulence model solver tolerance
    turb_solver_tolerance = 1.e-6          # turbulence model solver tolerance
    if enable_alpha_controller and alpha_controller_start_pseudo_step >= n_timesteps:
        raise ValueError(
            "alpha_controller_start_pseudo_step must be smaller than n_timesteps "
            f"({n_timesteps}) for the alpha controller to activate."
        )

    # First layer volumetric mesh thicknesses
    _, flat_plate_y1 = calculate_flat_plate_turbulent_y1(
        freestream_velocity=U_inf,
        desired_yplus=target_yplus_wall_modeled,
    )
    global_y1 = y1_fac * flat_plate_y1 * u.m    # First layer thickness for boundary layer meshing (wall-resolved)

    # solver parameters
    surf_output_requests = ["Cp", "Cf", "yPlus", "CfVec", "mutRatio", "solutionTransition"]
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
    project_from_surface_mesh = project_cgns_file_name is not None
    if project_from_surface_mesh:
        project = fl.Project.from_surface_mesh(project_cgns_file_name, name=sim_name,
                                               folder=flow360folder, length_unit="mm", run_async=async_flag)
        surface_mesh = project.surface_mesh
    elif project_step_file_name is not None:
        project = fl.Project.from_geometry(project_step_file_name, name=sim_name,
                                           folder=flow360folder, length_unit="mm", run_async=async_flag)
        surface_mesh = None
    elif project_id is not None:
        project = fl.Project.from_cloud(project_id)
        surface_mesh = None
    else:
        raise ValueError("Either project_cgns_file_name, project_step_file_name, or project_id must be specified")

    if turbulator_location_files is None:
        raise ValueError("turbulator_location_files must be provided.")
    if wake_refinement_files is None:
        raise ValueError("wake_refinement_files must be provided.")

    if project_from_surface_mesh:
        wall_surfaces = [
            surface_mesh[boundary_name]
            for boundary_name in surface_mesh.boundary_names
            if boundary_name != symm_face
        ]
    else:
        if LE_edge_list is None or TE_edge_list is None:
            raise ValueError(
                "LE_edge_list and TE_edge_list must be provided for geometry-based projects."
            )

        geo = project.geometry  # Access the geometry of the project

        # Display available groupings in the geometry (helpful for identifying group names)
        #geo.show_available_groupings(verbose_mode=True)
        #####################################################################################
        # Group edges and faces
        geo.group_faces_by_tag("faceId")
        geo.group_edges_by_tag("edgeId")

        LE_edges = [geo["body00001_edge{0:05d}".format(i)] for i in LE_edge_list]
        TE_edges = [geo["body00001_edge{0:05d}".format(i)] for i in TE_edge_list]

        wall_surfaces = [geo[face] for face in geo.entity_info.all_face_ids if face != symm_face]

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
    if not project_from_surface_mesh:
        surf_msh_data = {'refinement name': ["LE", "TE"],
                         "geo_item": [LE_edges, TE_edges],
                         "mesh size": [surf_mesh_refine_factor * 0.5 * u.mm,  # Leading edge refinement
                                       surf_mesh_refine_factor * 0.2 * u.mm,  # Trailing edge refinement
                         ]}
        df_mesh_refinement = pd.DataFrame(surf_msh_data)
        # Height-based edge refinement
        for idx, data in df_mesh_refinement.iterrows():
            edge_refinement = fl.SurfaceEdgeRefinement(name=data.iloc[0],
                edges=[data.iloc[1]],
                method=fl.HeightBasedRefinement(value=data.iloc[2]),
            )
            refinements.append(edge_refinement)

    turbulator_boxes = list()
    for file in turbulator_location_files:
        boxes = make_boxes(
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
                h_box += wake_refinement_delta_x * 2*np.sin(np.deg2rad(wake_refinement_angle_deg/2))
                boxes = make_boxes(
                    vertices_coords_file=file,
                    x_size=wake_refinement_delta_x + wake_refinement_box_overlap,
                    z_size=h_box + wake_refinement_box_overlap,
                    center_offset=(x + wake_refinement_delta_x/2, 0, -0.5),
                    name_prefix="wakebox_row{0:0d}".format(i_row),
                    segment_overlap=wake_refinement_spanwise_overlap,
                )
                wake_refinement_rows.append((i_row, boxes))

        # Volume refinement at winglet tip: staggered cylinders with 5 deg diameter growth
        winglet_tip_refinement_start_center = read_last_vertex(wake_refinement_files[-1])
        winglet_tip_cylinders = make_winglet_tip_refinement_cylinders(
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
        winglet_radius_cylinders = make_winglet_tip_refinement_cylinders(
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
                                   refinements=refinements)


    ###########################
    # 4) Flow solver parameters
    ###########################

    moment_ref_lengths = (wingspan/2, mac, wingspan/2)

    ref_geometry = fl.ReferenceGeometry(moment_center=(moment_center_x, 1.e-6, 0) * u.m,
                                        moment_length=moment_ref_lengths * u.mm,
                                        area= wing_area/2 * u.m**2)

    ncrit = calc_ncrit_from_fsti(turbulence_intensity_perc)

    navier_stokes_solver = fl.NavierStokesSolver(absolute_tolerance=ns_solver_tolerance, linear_solver=fl.KrylovLinearSolver())
    transition_solver = fl.TransitionModelSolver(N_crit=ncrit, trip_regions=turbulator_boxes)
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
                                     outputs=[fl.SurfaceOutput(surfaces=wall_surfaces, output_fields=surf_output_requests, write_single_file=True),
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

    if not run_flag:
        if not project_from_surface_mesh and generate_surf_mesh:
            project.generate_surface_mesh(
                params=params,
                name="SurfaceMesh",
                run_async=False,
                draft_only=False,
            )
        if generate_vol_mesh:
            project.generate_volume_mesh(
                params,
                use_beta_mesher=True,
                name="VolumeMesh",
                run_async=False,
                use_geometry_AI=False,
                raise_on_error=True,
            )
        return project.id

    if not project_from_surface_mesh and generate_surf_mesh:
        project.generate_surface_mesh(
            params=params,
            name="SurfaceMesh",
            run_async=False,
            draft_only=False,
        )
    if generate_vol_mesh:
        project.generate_volume_mesh(
            params,
            use_beta_mesher=True,
            name="VolumeMesh",
            run_async=False,
            use_geometry_AI=False,
            raise_on_error=True,
        )
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
    generate_surf_mesh = False
    generate_vol_mesh = True
    run = False
    n_test_cases = None
    enable_volume_refinements = True
    boundary_layer_growth_rate = 1.1
    enable_alpha_controller = True
    aircraft_mass = 600.0
    alpha_controller_kp = 0.2
    alpha_controller_ki = 0.002
    alpha_controller_start_pseudo_step = 50

    results_dir = "C:/WDIR/flow360"

    #variant = "FlapletV2 WKS"
    #variant = "Original WKS"
    variant = "FlapletV2 WK+2"

    sim_name = "V3 " + variant

    default_boundary_layer_growth_rate = 1.1
    if boundary_layer_growth_rate != default_boundary_layer_growth_rate:
        sim_name += f"_VolGR={boundary_layer_growth_rate:g}"

    study_name = "V3 Flaplets"
    half_model = True
    altitude = 0
    wing_area = 10.84
    U_inf_range = None


    variant_configs = {
        "Original WKS": {
            "project_step_file_name": None,
            "project_cgns_file_name": "Ventus_Original_WKS_G.cgns",
            "project_id": None,
            "symm_face": "symm face",
            "LE_edge_list": [],
            "TE_edge_list": [],
            "turbulator_location_files": ["Turbulator_wing_lower_original_WKS.dat"],
            "wake_refinement_files": ["TE_upper_VentusOrig_WKS.dat"],
            "target_lift_coefficient_range": [0.5],
            "alpha_deg_range": [3.0],
            "n_timesteps": 2000,
        },
        "Original WK+2": {
            "project_step_file_name": None,
            "project_cgns_file_name": None,
            "project_id": None,
            "symm_face": "symm face",
            "LE_edge_list": [],
            "TE_edge_list": [],
            "turbulator_location_files": [],
            "wake_refinement_files": ["TE_upper_VentusOrig_WK+2.dat"],
            "target_lift_coefficient_range": [1.2],
            "alpha_deg_range": [1.85],
            "n_timesteps": 2000,
        },
        "FlapletV2 WKS": {
            "project_step_file_name": None,
            "project_cgns_file_name": "Ventus3_FlapletV2_WKS_B.cgns",
            "project_id": None,
            "symm_face": "symm face",
            "LE_edge_list": [],
            "TE_edge_list": [],
            "turbulator_location_files": [
                "Turbulator_wing_lower_FlapletV2_WKS.dat",
                "Turbulator_Flaplet_upper_WKS.dat",
            ],
            "wake_refinement_files": ["TE_upper_Flaplet_WKS.dat"],
            "target_lift_coefficient_range": [0.5],
            "alpha_deg_range": [3.0],
            "n_timesteps": 2000,
        },
        "FlapletV2 WK+2": {
            "project_step_file_name": None,
            "project_cgns_file_name": "Ventus3_FlapletV2_WK+2_B.cgns",
            "project_id": None,
            "symm_face": "symm face",
            "LE_edge_list": [],
            "TE_edge_list": [],
            "turbulator_location_files": [
                "Turbulator_Flaplet_upper_WKS.dat",
            ],
            "wake_refinement_files": ["TE_upper_Flaplet_WK+2.dat"],
            "target_lift_coefficient_range": [1.2],
            "alpha_deg_range": [1.85],
            "n_timesteps": 2000,
        },
    }

    if variant not in variant_configs:
        raise ValueError(
            f"Invalid variant {variant!r}. Available variants: {list(variant_configs)}"
        )

    variant_config = variant_configs[variant]
    proj_step_file = variant_config["project_step_file_name"]
    proj_cgns_file = variant_config["project_cgns_file_name"]
    proj_id = variant_config["project_id"]
    symm_face = variant_config["symm_face"]
    LE_edge_list = variant_config["LE_edge_list"]
    TE_edge_list = variant_config["TE_edge_list"]
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
            project_step_file_name=proj_step_file,
            project_id=proj_id,
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
            LE_edge_list=LE_edge_list,
            TE_edge_list=TE_edge_list,
            turbulator_location_files=turbulator_location_files,
            wake_refinement_files=wake_refinement_files,
            flow360folder=curr_folder,
            results_path=results_dir,
            generate_surf_mesh=generate_surf_mesh,
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
