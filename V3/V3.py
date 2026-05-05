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


def make_boxes(vertices_coords_file, x_size, z_size, center_offset=(0, 0, 0), name_prefix="turbulator_box"):
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
                size=(x_size, segment_length, z_size) * u.mm,
            )
        )

    return boxes


def make_winglet_tip_refinement_cylinders(
        name_prefix, start_center, axis, total_length, n_cylinders, initial_diameter,
        growth_angle_deg):
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
                height=cylinder_length * u.mm,
            )
        )

    return cylinders


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




def define_and_run(project_step_file_name=None, project_id=None, name=None, U_inf = 55, alpha_deg=0., half_model=True,
                   y1_fac=1., surf_mesh_lvl=0, flow360folder=None, results_path=None, run_flag = False):
    """

    :param U_inf:               Free stream velocity
    :param alpha_deg:           Angle of attack in degrees
    :param half_model:          Half model flag. True for half-model, False for full model
    :param y1_fac:              first layer thickness scaling factor
    :param surf_mesh_lvl:       Mesh refinement level
    :param flow360folder:       Flow360 folder to put the case in
    :param results_path:        Path to store results
    :param run_flag:            Flag, determines, if simulation is only set-up (False) or also run (True)
    :return:
    """
    #  #  #
    # global flags
    async_flag = False
    # global parameters
    altitude = 1500
    # mesh parameters
    surf_mesh_refine_factor = 2**(surf_mesh_lvl/2)       # Surface mesh size multiplier
    target_yplus_wall_modeled = 0.67    # Target y-plus value for wall-resolved meshing

    turbulence_intensity_perc = 0.05

    mac = 0.6356086024985311  # mean aerodynamic chord
    wingspan = 18  # half-span
    moment_center_x = 0.386666666666667  # x reference location for moments
    wing_area = 10.84

    # For V3 original WKS
    LE_edge_list = [13, 24, 35, 46, 56, 78, 108, 172, 192, 215]
    TE_edge_list = [4, 16, 27, 38, 49, 62, 87, 129, 130, 131, 132, 133, 134, 191, 190, 189, 188, 187, 186, 185, 184,
                    183, 182, 181, 180, 179, 212, 216, 7, 20, 31, 42, 53, 66, 93, 146, 145, 144, 195, 196, 197, 198,
                    199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 217, 218]

    # symmetry face
    symm_face = "body00001_face00003"

    # turbulator definition
    turbulator_location_files = ["Turbulator_wing_lower_WKS.dat"] #, "Turbulator_Flaplet_upper_WKS.dat"]
    turbulator_box_x_size = 4
    turbulator_box_z_size = 1.6
    turbulator_box_x_offset = 0
    turbulator_box_y_offset = 0
    turbulator_box_z_offset = 0

    # wake volumetric refinement
    wake_refinement_files = ["TE_upper_VentusOrig_WKS.dat"]
    wake_refinement_angle_deg = 15
    wake_refinement_length = 1000
    wake_refinement_delta_x = 100

    # winglet tip cylindrical volume refinement: 10 adjacent cylinders over 2 m total length
    winglet_tip_refinement_axis = (1, 0, 0)
    winglet_tip_refinement_length = 2000
    winglet_tip_refinement_n_cylinders = 10
    winglet_tip_refinement_initial_diameter = 100
    winglet_tip_refinement_growth_angle_deg = 5
    winglet_tip_refinement_spacing = surf_mesh_refine_factor * 1 * u.mm

    # winglet radius cylindrical volume refinement: start at max y-z trailing-edge curvature
    winglet_radius_refinement_fallback_start_center = (1267.0, 8997.5, 1357.4)
    winglet_radius_refinement_axis = (1, 0, 0)
    winglet_radius_refinement_length = 2000
    winglet_radius_refinement_n_cylinders = 10
    winglet_radius_refinement_initial_diameter = 100
    winglet_radius_refinement_growth_angle_deg = 5
    winglet_radius_refinement_spacing = surf_mesh_refine_factor * 1 * u.mm

    ns_solver_tolerance = 1.e-7            # Navier-Stokes and turbulence model solver tolerance
    turb_solver_tolerance = 1.e-6          # turbulence model solver tolerance
    n_timesteps = 150

    # First layer volumetric mesh thicknesses
    _, flat_plate_y1 = calculate_flat_plate_turbulent_y1(
        freestream_velocity=U_inf,
        desired_yplus=target_yplus_wall_modeled,
    )
    global_y1 = y1_fac * flat_plate_y1 * u.m    # First layer thickness for boundary layer meshing (wall-resolved)

    # solver parameters
    surf_output_requests = ["Cp", "Cf", "yPlus", "CfVec", "mutRatio", "solutionTransition"]
    vol_output_requests = ["primitiveVars", "Cp", "Cpt", "qcriterion", "mutRatio", "solutionTransition"]

    if name is not None:
        sim_name = name + " "
    else:
        sim_name = ""
    sim_name += "U{0:d}_AOA{1:d}".format(int(U_inf), int(alpha_deg))

    if surf_mesh_lvl != 0:
        sim_name += "_mshlvl{0:d}".format(surf_mesh_lvl)


    ###############################
    # Preface: Create a new project
    ###############################


    alpha = np.deg2rad(alpha_deg)

    # Initialize project
    if project_step_file_name is not None:
        project = fl.Project.from_geometry(project_step_file_name, name=sim_name,
                                           folder=flow360folder, length_unit="mm", run_async=async_flag)
    elif project_id is not None:
        project = fl.Project.from_cloud(project_id)
    else:
        raise ValueError("Either parameter project_step_file_name or project_id must be specified")
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
                                       boundary_layer_growth_rate=1.3,
                                       boundary_layer_first_layer_thickness=global_y1)

    # 3c) Rotation region
    # None..

    # 3d) Mesh refinements
    surf_msh_data = {'refinement name': ["LE", "TE"],
                     "geo_item": [LE_edges, TE_edges],
                     "mesh size": [surf_mesh_refine_factor * 0.5 * u.mm,  # Leading edge refinement
                                   surf_mesh_refine_factor * 0.2 * u.mm,  # Trailing edge refinement
                     ]}
    df_mesh_refinement = pd.DataFrame(surf_msh_data)
    # Height-based edge refinement
    refinements = []
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

    wake_refinement_boxes = list()
    for file in wake_refinement_files:
        h_box = 0
        for i_row, x in enumerate(np.arange(0, wake_refinement_length, wake_refinement_delta_x)):
            h_box += wake_refinement_delta_x * 2*np.sin(np.deg2rad(wake_refinement_angle_deg/2))
            boxes = make_boxes(
                vertices_coords_file=file,
                x_size=wake_refinement_delta_x,
                z_size=h_box,
                center_offset=(x + wake_refinement_delta_x/2, 0, -0.5),
                name_prefix="wakebox_row{0:0d}".format(i_row)
            )
            wake_refinement_boxes.append(boxes)


    """# first layer refinements
    fuse_nose_ref = fl.BoundaryLayer(faces=[geo["fuseNoseFace"], ], first_layer_thickness=nose_y1)
    refinements.append(fuse_nose_ref)
    fin_ref = fl.BoundaryLayer(faces=[geo["tailFin"], geo["antenna"]], first_layer_thickness=tail_fin_y1)
    refinements.append(fin_ref)"""

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
    )
    winglet_tip_refinement = fl.UniformRefinement(
        name="winglet_tip_refinement",
        entities=winglet_tip_cylinders,
        spacing=winglet_tip_refinement_spacing,
    )
    refinements.append(winglet_tip_refinement)

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
    )
    winglet_radius_refinement = fl.UniformRefinement(
        name="winglet_radius_refinement",
        entities=winglet_radius_cylinders,
        spacing=winglet_radius_refinement_spacing,
    )
    refinements.append(winglet_radius_refinement)


    # make wake refinement
    wake_box_ref = fl.UniformRefinement(name="wake_refinement", entities=wake_refinement_boxes,
                                        spacing=surf_mesh_refine_factor * 1 * u.mm)
    refinements.append(wake_box_ref)

    # antenna volumetric refinement
    l_ant_cyl = 380
    antenna_cylinder = fl.Cylinder(name="antenna_tip_cylinder", center=(367 + l_ant_cyl / 2 * np.cos(alpha),
                                                                           76,
                                                                           l_ant_cyl / 2 * np.sin(alpha)) * fl.u.mm,
                                   axis=(np.cos(alpha), 0, np.sin(alpha)),
                                   outer_radius=20 * fl.u.mm,
                                   height=(l_ant_cyl + 30) * fl.u.mm)
    antenna_refinement = fl.UniformRefinement(name="antenna_tip_refinement", entities=[antenna_cylinder],
                                              spacing=surf_mesh_refine_factor * 1.1 * 3 ** (1 / 2) * u.mm)
    refinements.append(antenna_refinement)

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
    #turbulence_solver = fl.KOmegaSST(absolute_tolerance=turb_solver_tolerance)
    turbulence_solver = fl.SpalartAllmaras(absolute_tolerance=turb_solver_tolerance)

    fl_models = [fl.Wall(surfaces=wall_surfaces, use_wall_function=False),
                         fl.Freestream(surfaces=[far_field_zone.farfield]),
                         fl.Fluid(navier_stokes_solver=navier_stokes_solver,
                                  transition_model_solver=transition_solver,
                                  turbulence_model_solver=turbulence_solver)]

    if half_model:
        fl_models.append(fl.SymmetryPlane(surfaces=[far_field_zone.symmetry_planes]))

    with fl.SI_unit_system:
        # Set up the main simulation parameters
        params = fl.SimulationParams(meshing=mesh_params,
                                     reference_geometry=ref_geometry,
                                     operating_condition=condition,
                                     time_stepping=fl.Steady(max_steps=n_timesteps),
                                     models=fl_models,
                                     outputs=[fl.SurfaceOutput(surfaces=wall_surfaces, output_fields=surf_output_requests),
                                              fl.VolumeOutput(name="VolumeOutput", output_format="paraview",
                                                              output_fields=vol_output_requests)]

                                     )

    ###############################
    # 5) Generate mesh and run case
    ###############################
    # Step 5: Run the simulation case with the specified parameters
    if not run_flag:
        project.generate_surface_mesh(params=params, name='SurfaceMesh', run_async=False, draft_only=False)
        #project.generate_volume_mesh(params, name='VolumeMesh', run_async=False, use_geometry_AI=False,
        # raise_on_error=True)
        return project.id
    else:
        project.run_case(params=params, name="V3_case_" + sim_name)

        case = project.case
        case.wait()

        results = case.results
        results.download(surface=True, volume=True, total_forces=True, nonlinear_residuals=True,
                         destination=os.path.join(results_path, case.name))

        total_forces = case.results.total_forces.as_dataframe()

        return total_forces


def main():
    mshlvl = 1
    run=False
    n_test_cases = None

    results_dir = "F:/WDIR/flow360"

    proj_step_file = None
    proj_id = None

    # proj_step_file = "Ventus_Original_WKS.stp"
    #proj_step_file = "Ventus_Original_WK2.stp"
    #proj_step_file = "Ventus3_FlapletV2_WKS.stp"
    # proj_step_file = "Ventus3_FlapletV2_WK2.stp"

    proj_id = "prj-c6babf81-d21a-4715-a36c-b9190be22783" # orig WKS
    # proj_id = "" # orig WK+2
    # proj_id = "" # Flaplet WKS
    # proj_id = "" # Flaplet WK+2

    if proj_step_file is not None:
        sim_name = "V3 " + proj_step_file.lstrip("Ventus3_").rstrip(".stp").replace("_", " ")
    else:
        match proj_id:
            case "prj-c6babf81-d21a-4715-a36c-b9190be22783":
                sim_name = "V3 Original WKS"
            case "prj-0f7b4be0-5921-4245-8c96-08a4f1597752":
                sim_name = "V3 Original WK2"
            case "prj-9ee6a802-17af-481f-887a-ba992fce2db3":
                sim_name = "V3 Flaplet WKS"
            case "prj-f5032aa0-570f-44ab-ba5b-47f7fbde11fa":
                sim_name = "V3 Flaplet WK2"
            case _:
                raise ValueError(f"Invalid project ID: {proj_id}")

    half_model = True
    U_inf_range = [55]
    alpha_deg_range = [0.75, ]
    study_name = "V3 Flaplets"

    # initialize results DataFrame
    cols=['U_inf', 'alpha_deg', "CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"]
    df_results = pd.DataFrame(columns=cols)
    df_results[cols[:2]] = list(product(U_inf_range, alpha_deg_range))

    if n_test_cases is not None:
        df_results = df_results.iloc[:n_test_cases]   # limit number of runs for testing

    # create folders for airspeeds and AOA
    # create folder in ROOT level
    folder_toplvl = fl.Folder.create(study_name).submit()
    folders = []
    for U in U_inf_range:
        # create folder inside the above folder
        folder_U = fl.Folder.create("U_inf {0:d}".format(int(U)), parent_folder=folder_toplvl).submit()
        folders.append(folder_U)

    for i, row in df_results.iterrows():
        U_inf = row['U_inf']
        alpha_deg = row['alpha_deg']

        curr_folder = folders[list(U_inf_range).index(U_inf)]
        res = define_and_run(project_step_file_name=proj_step_file, project_id=proj_id, name=sim_name,
                             U_inf=U_inf, alpha_deg=alpha_deg, half_model=half_model,
                             surf_mesh_lvl=mshlvl, flow360folder=curr_folder, results_path=results_dir, run_flag=run)

        if run:
            # extract CL, CD, CMY from results as moving average over last 20 timesteps
            CL_avg = res['CL'].tail(15).mean()
            CD_avg = res['CD'].tail(15).mean()
            CFx = res['CFx'].tail(15).mean()
            CFy = res['CFy'].tail(15).mean()
            CFz = res['CFz'].tail(15).mean()
            CMx_avg = res['CMx'].tail(15).mean()
            CMy_avg = res['CMy'].tail(15).mean()
            CMz_avg = res['CMz'].tail(15).mean()
            df_results.loc[i, 'CL'] = CL_avg
            df_results.loc[i, 'CD'] = CD_avg
            df_results.loc[i, 'CMx'] = CMx_avg
            df_results.loc[i, 'CMy'] = CMy_avg
            df_results.loc[i, 'CMz'] = CMz_avg
            df_results.loc[i, 'CFx'] = CFx
            df_results.loc[i, 'CFy'] = CFy
            df_results.loc[i, 'CFz'] = CFz

    # write df_results to csv
    if run:
        df_results.to_csv(study_name + ".csv", index=False)
    pass


if __name__ == "__main__":
    main()
