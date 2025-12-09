# Import necessary modules from the Flow360 library
import numpy as np
from scipy.spatial.transform import Rotation as R

import flow360 as fl
from flow360 import u


def main():
    # global flags
    async_flag = False

    # global parameters
    solver_tolerance = 1.e-6            # Navier-Stokes and turbulence model solver tolerance
    elev_deflection_deg = 0  # Elevator deflection in degrees
    U_inf = 270                         # Free stream velocity
    alpha_deg = 0                       # Angle of attack in degrees
    beta_deg = 0                        # Angle of sideslip in degrees
    altitude = 500                      # Sideslip angle in degrees
    # mesh parameters

    global_y1 = 2.85e-6 * u.m           # First layer thickness for boundary layer meshing (wall-resolved)

    # solver parameters
    n_timesteps = 1000
    surf_output_requests = ["Cp", "Cf", "yPlus", "CfVec"]
    vol_output_requests = ["primitiveVars", "qcriterion", "mut", "T", "vorticity"]




    ###############################
    # Preface: Create a new project
    ###############################

    alpha = np.deg2rad(alpha_deg)

    vol_box_mwe_folder = fl.Folder.create("Vol_Box_MWE").submit()

    # This initializes a project with the specified geometry and assigns it a name.
    project = fl.Project.from_geometry("C:/git/flow360cases/examples/Xwing.csm", name="Xwing Vol Box MWE",
                                       folder=vol_box_mwe_folder, length_unit="m", run_async=async_flag)
    geo = project.geometry  # Access the geometry of the project
    geo.group_faces_by_tag("faceName")

    ################################
    # 1) Define operating conditions
    ################################
    condition = fl.AerospaceCondition(velocity_magnitude=U_inf * u.m / u.s, alpha=alpha_deg * u.deg, beta=beta_deg * u.deg,
                                      thermal_state=fl.ThermalState.from_standard_atmosphere(altitude=altitude * u.m))


    ################################
    # 2) Define mesh
    ################################

    # 3a) Farfield
    far_field_zone = fl.AutomatedFarfield()

    # 3b) Mesh parameters
    mesh_defaults = fl.MeshingDefaults(surface_edge_growth_rate=1.2,
                                       surface_max_edge_length=7 * u.mm,
                                       curvature_resolution_angle=5 * u.deg,
                                       boundary_layer_growth_rate=1.2,
                                       boundary_layer_first_layer_thickness=global_y1)

    # 3c) Rotation region
    # None..

    # 3d) Mesh refinements
    refinements = []

    # make tail fin refinement
    chord = 3530  # chord length of the wing
    t_rel = 0.1  # relative thickness of the wing
    h_box = 5 * chord * t_rel
    l_box = 4 * chord
    b_box = 7000.0  # spanwise box size
    r_box = (1315 + 8000) / 2
    box_list = []
    angles_deg = [-20, 20, 160, 200]
    for i, angle_deg in enumerate(angles_deg):
        angle = np.deg2rad(angle_deg)

        first_axis = np.array([1, 0, 0])
        second_axis = np.array([0, 0, 1])

        rot = R.from_euler('xy', [angle_deg, -alpha_deg], degrees=True)

        # fin box
        box_list.append(fl.Box.from_principal_axes(name="fin_box{0:d}".format(i),
                                             axes=[tuple(rot.apply(first_axis)), tuple(rot.apply(second_axis))],
                                             center=(-chord*1.05 + l_box/2,
                                                     r_box * np.cos(angle),
                                                     r_box * np.sin(angle) + 55 * np.sin(alpha)) * u.mm,
                                             size=(l_box*1.05, h_box, b_box) * u.mm)
        )

    wake_box_ref = fl.UniformRefinement(name="fin_box_refinement", entities=box_list,
                                        spacing=1.1 * 3 ** (1 / 2) * u.mm)
    refinements.append(wake_box_ref)

    # make mesh parameters
    mesh_params = fl.MeshingParams(defaults=mesh_defaults,
                                   volume_zones=[far_field_zone],
                                   refinements=refinements)

    ###########################
    # 4) Flow solver parameters
    ###########################
    moment_ref_lengths = (1, 1, 1)
    ref_geometry = fl.ReferenceGeometry(moment_center=(0, 0, 0) * u.m,
                                        moment_length=moment_ref_lengths * u.m,
                                        area=10 * u.m**2)

    wall_surfaces = [geo["mainBody"]]

    navier_stokes_solver = fl.NavierStokesSolver(absolute_tolerance=solver_tolerance)
    turbulence_solver = fl.SpalartAllmaras(absolute_tolerance=solver_tolerance)

    with fl.SI_unit_system:
        # Set up the main simulation parameters
        params = fl.SimulationParams(meshing=mesh_params,
                                     reference_geometry=ref_geometry,
                                     operating_condition=condition,
                                     time_stepping=fl.Steady(max_steps=n_timesteps),
                                     models=[fl.Wall(surfaces=wall_surfaces),
                                             fl.Freestream(surfaces=[far_field_zone.farfield]),
                                             # Define what sort of physical model of a fluid we will use
                                             fl.Fluid(navier_stokes_solver=navier_stokes_solver,
                                                      turbulence_model_solver=turbulence_solver,
                                                     ),],
                                     outputs=[fl.SurfaceOutput(surfaces=wall_surfaces, output_fields=surf_output_requests),
                                              fl.VolumeOutput(name="VolumeOutput", output_format="paraview",
                                                              output_fields=vol_output_requests)]

                                     )

    ###############################
    # 5) Generate mesh and run case
    ###############################
    #project.model_construct(params=params)

    # Step 5: Run the simulation case with the specified parameters
    project.generate_surface_mesh(params=params, name='SurfaceMesh', run_async=False)

    print("done")

if __name__ == "__main__":
    main()