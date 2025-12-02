import flow360 as fl
import numpy as np
from scipy.integrate import simpson
from flow360.user_config import UserConfig
#UserConfig.set_profile(
#    "auto_test_1"
#)

# How to use:
# Since to simulate a turning vehicle we want to use SRF with a specified center and
# axis of rotation. In flow360 this is currently not fully supported but there is
# a quick workaround. We need to create a volume mesh in a separate project and
# reuse this mesh in a new project where the center and axis of rotation can now
# be specified. Here are the steps to run this script:

# STEP 1:
# start with creating the mesh only by setting 'create_mesh=True'
# If you want to use AD set 'use_AD=True'
# Note that you can add as many refinements as you want
# Note that mesh with actuator disk is different than mesh without actuator disk because of refinements
# so two separate meshes need to be created depending on the case.

# STEP 2:
# create project from generated volume mesh in webUI (I'll send screenshot in slack)

# STEP 3:
# use project ID from new project created in STEP 2
project_id = ''

# STEP 4:
# you can now set 'create_mesh=False' and keep 'use_AD' consistent with the used volume mesh
# Adapt boundary and zone names defined in lines -- for the imported mesh.
# You can now run as many cases with this script and the new project id like sweeps,
# I did an example of sweeps for different tangential velocity and turning radius.

# ADDITIONAL NOTES
# This script assumes that the mesh coordinate origin is in the front of the vehicle so the
# center of rotation is set considering the CG location and the turning radius. This is defined in
# farfield.center

# Geometrical parameters
x_prop = 4940/1e3 # Location of propeller blades in meter
R = 0.2032 # Propeller radius in meters
R0 = 0.1  # Propeller hub radius in meters
B = 5 # Number of blades
cg_coords = [x_prop/2, 0, 0] # CG coordinates in meter

# Operation parameters
thrust = 4900 # Propeller thrust in Newtons
torque = 0 # Propeller torque in Newton meter
omega = 100*2*np.pi/60 # Rotation speed in rad/s
rho = 1000 # Density in kg/m3

# Numerical parameters
AD_height = R*0.1 # Thickness of the AD
n_axial = 50 # Number of points in the axial direction
axial_spacing = AD_height/n_axial
v_tip = omega*R # Tip speed velocity only used as reference velocity

# Sweep parameters
v_sweep = [1.3] # Tangential velocity of vehicle CG in m/s
R_sweep = [0] # Turn radius in meter
beta_sweep = [0] # Drift angle in rad

# Parameter to control use of actuator disk
use_AD = True

# Parameter to generate mesh
create_mesh = False

def createParams(project, R_turn, v_freestream, beta, use_AD, imported_vm=False):
    # Define center and axis of rotation
    x_shift = R_turn*np.sin(beta*np.pi/180)
    y_shift = R_turn*np.cos(beta*np.pi/180)
    center = (cg_coords[0] + x_shift, cg_coords[1] + y_shift, cg_coords[2])
    axis = (0, 0, -1)
    print(f'Center of rotation: {center}')
    print(f'Axis of rotation: {axis}')

    # Define AD loading distributions
    # Betz-Prandtl distribution
    # Radial discretization
    n_points = 500
    r = np.linspace(R0, R, n_points)  # Avoid r=0
    xi = r / R

    # Step 1: Compute axial velocity through disk
    v_ax = 0.5 * (np.sqrt((2 * thrust) / (rho * np.pi * R**2) + v_freestream**2) + v_freestream)

    # Step 2: Wake advance ratio
    lambda_w = v_ax / (omega * R)

    # Step 3: Compute Prandtl tip-loss function
    f_tip = (B / 2) * (1 - xi) * np.sqrt(1 + 1 / lambda_w**2)
    f_hub = (B / 2) * (xi - R0/R) * np.sqrt(1 + 1 / lambda_w**2)
    F_tip = (2 / np.pi) * np.arccos(np.exp(-f_tip))
    F_hub = (2 / np.pi) * np.arccos(np.exp(-f_hub))
    F = F_tip*F_hub

    # Step 4: Non-dimensional circulation (Gamma_1)
    Gamma_1 = (F * xi**2) / (lambda_w**2 + xi**2)

    # Step 5: Compute T1 (Eq. 11)
    integrand_T1 = rho * omega * r * Gamma_1
    T1 = B * simpson(integrand_T1, r)

    # Step 6: Scaling factor K
    K = thrust / T1

    # Step 7: Final circulation distribution
    Gamma = K * Gamma_1

    # Step 8: Compute local body forces per unit area
    f_ax = (rho * omega * Gamma) / (2 * np.pi)
    f_cir = (rho * v_ax * Gamma) / (2 * np.pi * r)

    # Scale torque to match target torque
    integrand_Q = f_cir * r * 2 * np.pi * r
    Q1 = B * simpson(integrand_Q, r)
    K2 = torque / Q1
    f_cir = K2 * f_cir

    # Step 9: Compute local body forces per unit area
    delta_p_x = B * f_ax
    delta_p_theta = B * f_cir

    with fl.SI_unit_system:

        cylinder_AD=fl.Cylinder(
                name='actuator_disk_cylinder',
                center=(x_prop,0,0),
                axis=(-1,0,0),
                outer_radius=R,
                inner_radius=R0,
                height=R*0.1
            )

        # Define geometry or volume mesh zones
        if imported_vm:
            vm = project.volume_mesh
            # Adapt this names for your case
            walls = vm['farfield/body00001*'] # Names of wall patches
            freestream = vm['farfield/farfield'] # Names of farfield patches
            farfield = vm['farfield'] # Name of farfield zone

            # Define the center of rotation and axis of rotation for SRF
            farfield.center = center
            farfield.axis = axis
        else:
            # Only used to generate initial volume mesh
            geo = project.geometry
            geo.group_faces_by_tag("faceId")
            fine = [geo[f'body00001_face000{num}'] for num in [10,40]]
            coarse = [face for face in geo['*'] if face not in fine]
            walls = geo['*']
            farfield = fl.AutomatedFarfield()
            freestream = farfield.farfield

            # You can add additional refinements
            cylinder1 = fl.Cylinder(
                    name='cylinder_refinement_general',
                    center=(2500,0,0)*fl.u.mm,
                    axis=(1,0,0),
                    outer_radius=500*fl.u.mm,
                    height=6000*fl.u.mm
                )
            refinements=[
                fl.UniformRefinement(
                    entities=[cylinder1],
                    spacing=30*fl.u.mm
                ),
                fl.SurfaceRefinement(
                    faces=fine,
                    max_edge_length=2*fl.u.mm
                )
            ]
            
            # Actuator disk configuration
            if use_AD:
                h=0.5
                x_center=x_prop-0.1+h/2
                cylinder2=fl.Cylinder(
                    name='cylinder_refinement',
                    center=(x_center,0,0),
                    axis=(-1,0,0),
                    outer_radius=R*1.2,
                    height=h
                )
                actuator_disk_refinement=fl.AxisymmetricRefinement(
                    spacing_axial=axial_spacing,
                    spacing_circumferential=axial_spacing*3,
                    spacing_radial=axial_spacing*3,
                    entities=cylinder_AD
                )
                uniform_ref=fl.UniformRefinement(
                    entities=[cylinder2],
                    spacing=5*fl.u.mm
                )
                refinements+=[actuator_disk_refinement,uniform_ref]

        # Define models
        models=[
                fl.Wall(
                    surfaces=walls
                ),
                fl.Freestream(
                    surfaces=freestream
                ),
                fl.Fluid(
                    turbulence_model_solver=fl.SpalartAllmaras(),
                    navier_stokes_solver=fl.NavierStokesSolver()
                )]
        
        if use_AD:
            AD=fl.ActuatorDisk(
                volumes=cylinder_AD,
                force_per_area=fl.ForcePerArea(
                    radius=r,
                    thrust=delta_p_x,
                    circumferential=delta_p_theta
                )
            )
            models.append(AD)
        
        # When the mesh is imported and ready to be used, no meshing configuration is used
        # and we define the SRF model to be used with the generated mesh
        v_mag = 0
        meshing_settings = None
        if imported_vm and R_turn != 0:
            SRF = fl.Rotation(
                name='turnRotation',
                volumes=[farfield],
                spec=fl.AngularVelocity(v_freestream/R_turn * fl.u.rad / fl.u.s)
            )
            models.append(SRF)
        elif R_turn == 0:
            v_mag = v_freestream
        else:
            meshing_settings = fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    surface_edge_growth_rate=1.2,
                    surface_max_edge_length=50*fl.u.mm,
                    curvature_resolution_angle=5 * fl.u.deg,
                    boundary_layer_growth_rate=1.2,
                    boundary_layer_first_layer_thickness=1e-6
                ),
                volume_zones=[farfield],
                refinements=refinements
            )

        # Define variables
        velocity_dim = fl.UserVariable(name='velocity_dim', value=fl.solution.velocity).in_units('SI_unit_system')
        pressure_dim = fl.UserVariable(name='pressure_dim', value=fl.solution.pressure).in_units('SI_unit_system')

        # Define outputs
        vol_fields = ["Cp","VelocityRelative",velocity_dim,pressure_dim]
        surf_fields = ["Cp", "Cf", "yPlus", "CfVec"]

        # Define params
        params=fl.SimulationParams(
            meshing=meshing_settings,
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0,0,0),
                moment_length=(1,1,1),
                area=1
            ),
            operating_condition=fl.LiquidOperatingCondition(
                velocity_magnitude=v_mag * fl.u.m/fl.u.s,
                reference_velocity_magnitude=v_tip * fl.u.m/fl.u.s,
                alpha=0.0 * fl.u.deg,
                beta=0.0 * fl.u.deg,
                material=fl.Water(name="Water")
            ),
            time_stepping=fl.Steady(
                CFL=fl.AdaptiveCFL(),
                max_steps=5000
            ),
            models=models,
            outputs=[
                fl.SurfaceOutput(
                    surfaces=walls,
                    output_format="both",
                    output_fields=surf_fields
                ),
                fl.SliceOutput(
                    slices=[
                        fl.Slice(
                            name='z-slice',
                            normal=(0,0,1),
                            origin=cg_coords
                        ),
                        fl.Slice(
                            name='y-slice',
                            normal=(0,1,0),
                            origin=cg_coords
                        ),
                        fl.Slice(
                            name='x-slice',
                            normal=(1,0,0),
                            origin=cg_coords
                        )
                    ],
                    output_format="both",
                    output_fields=vol_fields
                ),
                fl.VolumeOutput(
                    output_format='both',
                    output_fields=vol_fields
                ),
                fl.ProbeOutput(
                    probe_points=[
                        fl.Point(
                            name='origin',
                            location=(0,0,0)
                        ),
                        fl.Point(
                            name='+z1',
                            location=(cg_coords[0],cg_coords[1],cg_coords[2]+50)
                        ),
                        fl.Point(
                            name='-z1',
                            location=(cg_coords[0],cg_coords[1],cg_coords[2]-50)
                        ),
                        fl.Point(
                            name='+z2',
                            location=(cg_coords[0],cg_coords[1],cg_coords[2]+100)
                        ),
                        fl.Point(
                            name='-z2',
                            location=(cg_coords[0],cg_coords[1],cg_coords[2]-100)
                        ),
                    ],
                    output_fields=vol_fields
                )
            ]
        )
    return params

if create_mesh:
    fuselage = 'geometry.STEP'
    project = fl.Project.from_geometry(fuselage, name='project_only_used_to_mesh', length_unit='mm')
    params = createParams(project, 10.0, 10.0, 0.0, use_AD, imported_vm=False)
    mesh = project.generate_volume_mesh(params=params, use_beta_mesher=True, name=f'volume_mesh_turning_vehicle_use_AD_{use_AD}')
else:
    project = fl.Project.from_cloud(project_id=project_id)
    for v in v_sweep:
        for R_turn in R_sweep:
            for beta in beta_sweep:
                params = createParams(project, R_turn, v, beta, use_AD, imported_vm=True)
                project.run_case(params=params,use_beta_mesher=True,name=f'R_{R_turn}_v_{v}_beta_{beta}_AD_{use_AD}')