# Import necessary modules from the Flow360 library
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from matplotlib.pyplot import show

import flow360 as fl
from flow360 import u

def mod_csm_file(path, elev_deflection_deg=10, half_model=True):
    # modify csm file to set elevator deflection
    with open(path, "r") as f:
        csm_data = f.readlines()
        csm_data[2] = "DESPMTR elevDefl {0:d}\n".format(elev_deflection_deg)
        csm_data[3] = "DESPMTR halfModel {0:d}\n".format(half_model)
    with open(path, "w") as f:
        f.writelines(csm_data)

def main():
    # global flags
    async_flag = False
    run_flag = True

    # global parameters
    half_model = True                   # True for wall-modeled, False for wall-resolved
    elev_deflection_deg = 0             # Elevator deflection in degrees
    wall_func_flag = False
    solver_tolerance = 1.e-6            # Navier-Stokes and turbulence model solver tolerance
    U_inf = 120                         # Free stream velocity
    alpha_deg = 0                       # Angle of attack in degrees
    beta_deg = 0                        # Angle of sideslip in degrees
    altitude = 500                      # Sideslip angle in degrees
    # mesh parameters
    surf_mesh_lvl = 2
    surf_mesh_refine_factor = 2**(surf_mesh_lvl/2)       # Surface mesh size multiplier
    target_yplus_wall_modeled = 30.0    # Target y-plus value for wall-resolved meshing
    global_y1 = 2.85e-6 * u.m           # First layer thickness for boundary layer meshing (wall-resolved)
    nose_y1 = 2e-6 * u.m
    tail_fin_y1 = 2.5e-6 * u.m
    antenna_y1 = 5e-6 * u.m

    # solver parameters
    n_timesteps = 1000
    surf_output_requests = ["Cp", "Cf", "yPlus", "CfVec"]
    vol_output_requests = ["primitiveVars", "qcriterion", "mut", "T", "vorticity"]

    # global fixed parameters
    l_fuse = 756.51
    r_fuse = 45

    mod_csm_file(path="C:/git/flow360cases/AMDC_GBT/GBT.csm", elev_deflection_deg=elev_deflection_deg,
                 half_model=half_model)

    ###############################
    # Preface: Create a new project
    ###############################

    if wall_func_flag:
        global_y1 *= target_yplus_wall_modeled
        nose_y1 *= target_yplus_wall_modeled
        tail_fin_y1 *= target_yplus_wall_modeled
        antenna_y1 *= target_yplus_wall_modeled

    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)

    gbt_folder = fl.Folder.create("GBT U{0:d}_AOA{1:d}_delta{2:.1f}_mshlvl{3:d}".format(U_inf, alpha_deg, elev_deflection_deg, surf_mesh_lvl)).submit()
    #gbt_folder = fl.Folder.get("folder-ee329f80-9142-496f-938c-15b650fdebc2")

    # This initializes a project with the specified geometry and assigns it a name.
    project = fl.Project.from_geometry("C:/git/flow360cases/AMDC_GBT/GBT.csm", name="GBT U{0:d}_AOA{1:d}_delta{2:.1f}_mshlvl{3:d}".format(U_inf, alpha_deg, elev_deflection_deg, surf_mesh_lvl),
                                       folder=gbt_folder, length_unit="mm", run_async=async_flag)
    geo = project.geometry  # Access the geometry of the project

    # Display available groupings in the geometry (helpful for identifying group names)
    #geo.show_available_groupings(verbose_mode=True)
    #####################################################################################
    # Group faces by a specific tag for easier reference in defining `Surface` objects
    geo.group_faces_by_tag("faceName")
    geo.group_edges_by_tag("edgeName")

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
                                       surface_max_edge_length=surf_mesh_refine_factor * 7 * u.mm,
                                       curvature_resolution_angle=surf_mesh_refine_factor * 5 * u.deg,
                                       boundary_layer_growth_rate=1.2,
                                       boundary_layer_first_layer_thickness=global_y1)

    # 3c) Rotation region
    # None..

    # 3d) Mesh refinements

    surf_msh_data = {'refinement name': ["LE", "TE", "fin_root_tip", "fuse_aniso_edges", "fuse_rear_refinement"],
                     "geo_item": [geo["leadingEdge"], geo["trailingEdge"], geo["finEndEdges"], geo["anisoEdges"], geo["fuseRearEdge"]],
                     "mesh size": [surf_mesh_refine_factor * 0.055 * u.mm,  # Leading edge refinement
                              surf_mesh_refine_factor * 0.085 * u.mm,  # Trailing edge refinement
                              surf_mesh_refine_factor * 0.06 * u.mm,  # Vertical fin root/tip refinement
                              surf_mesh_refine_factor * 2. * u.mm,  # Fuselage anisotropic edge refinement
                              surf_mesh_refine_factor * 0.5 * u.mm,  # Fuselage rear refinement
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

    tail_fin_ref = fl.SurfaceRefinement(name="tail_fin",
                                        faces=[geo["tailFin"]],
                                        max_edge_length=surf_mesh_refine_factor * 1.1 * u.mm
                                        )
    refinements.append(tail_fin_ref)

    fuse_nose_ref = fl.SurfaceRefinement(name="fuse_nose",
                                        faces=[geo["fuseNoseFace"]],
                                        max_edge_length=surf_mesh_refine_factor * 2.5 * u.mm
                                        )
    refinements.append(fuse_nose_ref)

    # first layer refinements
    fuse_nose_ref = fl.BoundaryLayer(faces=[geo["fuseNoseFace"], ], first_layer_thickness=nose_y1)
    refinements.append(fuse_nose_ref)
    fin_ref = fl.BoundaryLayer(faces=[geo["tailFin"], geo["antenna"]], first_layer_thickness=tail_fin_y1)
    refinements.append(fin_ref)

    # Volume refinement at tail
    tail_cyl = fl.Cylinder(name="tail_cyl_refinement", center=(l_fuse*3/2, 0, 0) * fl.u.mm,
                           axis=(1, 0, 0),
                           outer_radius=2*r_fuse * fl.u.mm,
                           height=3*l_fuse * fl.u.mm,
    )

    fuse_center_wake_refinement = fl.UniformRefinement(name="fuse_center_wake_refinement", entities=[tail_cyl],
        spacing=surf_mesh_refine_factor * 7 * 3**(1/2) * u.mm  # Finer spacing for wake resolution
    )
    refinements.append(fuse_center_wake_refinement)

    if alpha_deg !=0:
        i_box = 0
        l_tot = 3*l_fuse
        h_box = 4*r_fuse
        while True:
            x0 = i_box * 4 * r_fuse / np.tan(alpha)
            l_box = l_tot - x0
            if l_box <= 0:
                break
            x_center = x0 + l_box / 2
            z_center = h_box * i_box + h_box / 2

            # Volume refinement box to capture the wake at non-zero AoA
            wake_box = fl.Box.from_principal_axes(name="skew_wake_box_{0:d}".format(i_box),  axes=[(1, 0, 0), (0, 0, 1)],
                                       center=(x_center, 0, z_center) * fl.u.mm,
                                       size=(l_box, h_box, h_box) * fl.u.mm,
                                       )
            wake_box_ref = fl.UniformRefinement(name="fuse_wake_box_refinement_{0:d}".format(i_box), entities=[wake_box],
                                 spacing=surf_mesh_refine_factor * 7 * 3 ** (1 / 2) * u.mm)  # Finer spacing for wake resolution
            refinements.append(wake_box_ref)
            i_box += 1

    # make tail fin refinement
    h_box = 115 #5 * 55 * 0.12  # set to 115 to account for flow360 bug
    l_box = 4 * 55
    b_box = 115
    r_box = (45 + 125) / 2
    box_list = []
    for i in range(2*(1 + int(not half_model))):
        angle_deg = -45. + i * 90.
        angle = np.deg2rad(angle_deg)

        first_axis = np.array([1, 0, 0])
        second_axis = np.array([0, 0, 1])

        rot = R.from_euler('xy', [angle_deg, -alpha_deg], degrees=True)

        # fin box
        box_list.append(fl.Box.from_principal_axes(name="fin_box{0:d}".format(i),
                                             axes=[tuple(rot.apply(first_axis)), tuple(rot.apply(second_axis))],
                                             center=(690.5 + 55 + 55 * np.cos(alpha),
                                                     r_box * np.cos(angle),
                                                     r_box * np.sin(angle) + 55 * np.sin(alpha)) * u.mm,
                                             size=(l_box*1.05, h_box, b_box) * u.mm)
        )

    wake_box_ref = fl.UniformRefinement(name="fin_box_refinement", entities=box_list,
                                        spacing=surf_mesh_refine_factor * 1.1 * 3 ** (1 / 2) * u.mm)
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
    l_tot = 756.5         # total length of the GBT model
    b_fin_half_diag = 124 # diagonal half span of the tail fin
    b_fin_y = b_fin_half_diag * 2 / np.sqrt(2)
    moment_ref_lengths = (b_fin_y, l_tot, b_fin_y)
    moment_center_x = 416. # x reference location for moments
    ref_geometry = fl.ReferenceGeometry(moment_center=(moment_center_x, 0, 0) * u.mm,
                                        moment_length=moment_ref_lengths * u.mm,
                                        area=45**2 * np.pi * u.mm**2)

    wall_surfaces = [geo["mainBody"], geo["fuseNoseFace"], geo["antenna"], geo["tailFin"]]

    navier_stokes_solver = fl.NavierStokesSolver(absolute_tolerance=solver_tolerance)
    turbulence_solver = fl.KOmegaSST(absolute_tolerance=solver_tolerance)
    #turbulence_solver = fl.SpalartAllmaras(absolute_tolerance=solver_tolerance)


    with fl.SI_unit_system:
        # Set up the main simulation parameters
        params = fl.SimulationParams(meshing=mesh_params,
                                     reference_geometry=ref_geometry,
                                     operating_condition=condition,
                                     time_stepping=fl.Steady(max_steps=n_timesteps),
                                     models=[fl.Wall(surfaces=wall_surfaces, use_wall_function=wall_func_flag),
                                             fl.Freestream(surfaces=[far_field_zone.farfield]),
                                             # Define what sort of physical model of a fluid we will use
                                             fl.Fluid(navier_stokes_solver=navier_stokes_solver,
                                                      turbulence_model_solver=turbulence_solver,
                                                     ),
                                             fl.SymmetryPlane(surfaces=[far_field_zone.symmetry_planes])],
                                     outputs=[fl.SurfaceOutput(surfaces=wall_surfaces, output_fields=surf_output_requests),
                                              fl.VolumeOutput(name="VolumeOutput", output_format="paraview",
                                                              output_fields=vol_output_requests)]

                                     )

    ###############################
    # 5) Generate mesh and run case
    ###############################
    #project.model_construct(params=params)

    # Step 5: Run the simulation case with the specified parameters
    if not run_flag:
        project.generate_surface_mesh(params=params, name='SurfaceMesh', run_async=False)
        #project.generate_volume_mesh(params, name='VolumeMesh', run_async=False, use_geometry_AI=False,
        # raise_on_error=True)
    else:
        project.run_case(params=params, name="GBT case U{0:d}_AOA{1:d}_delta{2:.1f}_mshlvl{3:d}_total_forces.csv".format(U_inf, alpha_deg, elev_deflection_deg, surf_mesh_lvl))

        case = project.case
        case.wait()

        total_forces = case.results.total_forces.as_dataframe()
        total_forces.plot("pseudo_step", ["CL", "CD"], ylim=[-5, 15])
        total_forces.to_csv("U{0:d}_AOA{1:d}_delta{2:.1f}_mshlvl{3:d}_total_forces.csv".format(U_inf, alpha_deg, elev_deflection_deg, surf_mesh_lvl))
        show()

    print("done")

if __name__ == "__main__":
    main()