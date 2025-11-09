# Import necessary modules from the Flow360 library
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from matplotlib.pyplot import show

import flow360 as fl
from flow360 import u

# global flags
async_flag = False

# global parameters
half_model = True
wall_model_flag = False  # True for wall-modeled, False for wall-resolved
elev_deflection_deg = 0  # Elevator deflection in degrees
U_inf = 120            # Free stream velocity
alpha_deg = 10         # Angle of attack in degrees
beta_deg = 0           # Angle of sideslip in degrees
altitude = 500# Sideslip angle in degrees
# mesh parameters
surf_mesh_refine_factor = 1.0  # Surface mesh size multiplier
target_yplus_wall_modeled = 30.0  # Target y-plus value for wall-resolved meshing
global_y1 = 2.85e-6 * u.m  # First layer thickness for boundary layer meshing (wall-resolved)
nose_y1 = 2e-6 * u.m
tail_fin_y1 = 2.5e-6 * u.m
antenna_y1 = 5e-6 * u.m

# solver parameters
n_timesteps = 1000
surf_output_requests = ["Cp", "Cf", "yPlus", "CfVec"]
vol_output_requests = ["primitiveVars", "qcriterion", "mut", "T", "vorticity"]
wall_func_flag = True

# global fixed parameters
l_fuse = 756.51
r_fuse = 45

###############################
# Preface: Create a new project
###############################

if wall_model_flag:
    global_y1 *= target_yplus_wall_modeled
    nose_y1 *= target_yplus_wall_modeled
    tail_fin_y1 *= target_yplus_wall_modeled
    antenna_y1 *= target_yplus_wall_modeled

gbt_folder = fl.Folder.create("GBT").submit()
#gbt_folder = fl.Folder.get("folder-ee329f80-9142-496f-938c-15b650fdebc2")

# This initializes a project with the specified geometry and assigns it a name.
project = fl.Project.from_geometry("C:/git/flow360cases/AMDC_GBT/GBT.csm", name="GBT Python RevB",
                                   folder=gbt_folder, length_unit="mm", run_async=async_flag)
geo = project.geometry  # Access the geometry of the project

# Display available groupings in the geometry (helpful for identifying group names)
geo.show_available_groupings(verbose_mode=True)
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

msh_data = {'refinement name': ["LE", "TE", "fin_root_tip", "fuse_aniso_edges", "fuse_rear_refinement"],
            "geo_item": [geo["leadingEdge"], geo["trailingEdge"], geo["finEndEdges"], geo["anisoEdges"], geo["fuseRearEdge"]],
            "mesh size": [surf_mesh_refine_factor * 0.055 * u.mm,  # Leading edge refinement
                          surf_mesh_refine_factor * 0.085 * u.mm,  # Trailing edge refinement
                          surf_mesh_refine_factor * 0.06 * u.mm,  # Vertical fin root/tip refinement
                          surf_mesh_refine_factor * 2. * u.mm,  # Fuselage anisotropic edge refinement
                          surf_mesh_refine_factor * 0.5 * u.mm,  # Fuselage rear refinement
                          ]}
df_mesh_refinement = pd.DataFrame(msh_data)
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
        x0 = i_box * 4 * r_fuse / np.tan(np.deg2rad(alpha_deg))
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
h_box = 5 * 55 * 0.12
l_box = 1.5 * 55
l_box2 = 2 * 55
b_box = 115
r_box = (45 + 125) / 2
for i in range(2*(1 + int(not half_model))):
    angle_deg = -45. + i * 90.
    angle = np.deg2rad(angle_deg)

    first_axis = np.array([1, 0, 0])
    second_axis = np.array([0, 0, 1])

    r_1 = R.from_euler('x', angle_deg, degrees=True)

    fin_box = fl.Box.from_principal_axes(name="fin_box_front{0:d}".format(i),
                                         axes=[tuple(r_1.apply(first_axis)), tuple(r_1.apply(second_axis))],
                                         center=(690.5 + l_box / 2, r_box * np.cos(angle), r_box * np.sin(angle)) * u.mm,
                                         size=(l_box, b_box, h_box) * u.mm)
    #wake_box_ref = fl.UniformRefinement(name="fin_box_refinement_front{0:d}".format(i), entities=[fin_box,],
    #                                    spacing=surf_mesh_refine_factor * 1.1 * 3 ** (1 / 2) * u.mm)
    #refinements.append(wake_box_ref)
    # second, larger fin box

    r_2 = R.from_euler('xy', [angle_deg, -alpha_deg], degrees=True)

    fin_box2 = fl.Box.from_principal_axes(name="fin_box_rear{0:d}".format(i),
                                         axes=[tuple(r_2.apply(first_axis)), tuple(r_2.apply(second_axis))],
                                         center=(690.5 + 55 + 55 * np.cos(np.deg2rad(alpha_deg)),
                                                 r_box * np.cos(angle),
                                                 r_box * np.sin(angle) + 55 * np.sin(np.deg2rad(alpha_deg))) * u.mm,
                                         size=(l_box2, h_box, b_box) * u.mm)
    wake_box_ref = fl.UniformRefinement(name="fin_box_refinement{0:d}".format(i), entities=[fin_box, fin_box2],
                                        spacing=surf_mesh_refine_factor * 1.1 * 3 ** (1 / 2) * u.mm)
    refinements.append(wake_box_ref)

# make mesh parameters
mesh_params = fl.MeshingParams(defaults=mesh_defaults,
                               volume_zones=[far_field_zone],
                               refinements=refinements)


###########################
# 4) Flow solver parameters
###########################
ref_geometry = fl.ReferenceGeometry(moment_center=(416., 0, 0) * u.mm, moment_length=(756.5, 90, 90) * u.mm,
                                    area=45**2 * np.pi * u.mm**2)

wall_surfaces = [geo["mainBody"], geo["fuseNoseFace"], geo["antenna"], geo["tailFin"]]

with fl.SI_unit_system:
    # Set up the main simulation parameters
    params = fl.SimulationParams(meshing=mesh_params,
                                 reference_geometry=ref_geometry,
                                 operating_condition=condition,
                                 time_stepping=fl.Steady(max_steps=n_timesteps),
                                 models=[fl.Wall(surfaces=wall_surfaces, use_wall_function=wall_func_flag),
                                         fl.Freestream(surfaces=[far_field_zone.farfield]),
                                         fl.SymmetryPlane(surfaces=[far_field_zone.symmetry_planes])],
                                 outputs=[fl.SurfaceOutput(surfaces=wall_surfaces, output_fields=surf_output_requests),
                                          fl.VolumeOutput(name="VolumeOutput", output_format="paraview",
                                                          output_fields=vol_output_requests)]

                                 )

###############################
# 5) Generate mesh and run case
###############################
project.model_construct(params=params)


project.generate_surface_mesh(params=params, name='SurfaceMesh', run_async=False)

#project.generate_volume_mesh(params, name='VolumeMesh', run_async=False, use_geometry_AI=False, raise_on_error=True)

# Step 5: Run the simulation case with the specified parameters
project.run_case(params=params, name="GBT Case")

case = project.case
case.wait()

total_forces = case.results.total_forces.as_dataframe()
total_forces.plot("pseudo_step", ["CL", "CD"], ylim=[-5, 15])
show()

print("done")
