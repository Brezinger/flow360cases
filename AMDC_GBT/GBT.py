"""
Sample Flow 360 API scripts.
Requires a mesh that you are ready to upload and run cases on.
"""

import os

import flow360 as fl
from flow360.log import log

# Variables we want to export in our volume solution files. Many more are available
vol_fields = ["Mach", "Cp", "mut", "mutRatio", "primitiveVars", "qcriterion"]

# Variables we want to export in our surface solution files. Many more are available
surf_fields = ["Cp", "yPlus", "Cf", "CfVec", "primitiveVars", "wallDistance"]


######################################################################################################################
def upload_mesh(file_path, project_name):
    """
    Given a file path and name of the project, this function creates a project and uploads a mesh.
    """
    # length_unit should be 'm', 'mm', 'cm', 'inch' or 'ft'
    project = fl.Project.from_volume_mesh(file_path, length_unit="m", name=project_name)
    log.info(f"The project id is {project.id}")

    return project


######################################################################################################################
def make_params(mesh_object):
    """
    Create the params object that contains all the run parameters.
    Needs the mesh_object to get the list of surfaces.
    """
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            # Dimensions can  be either in inches, or m or mm or many other units
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m, moment_length=1 * fl.u.m, area=1 * fl.u.m * fl.u.m
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=100 * fl.u.m / fl.u.s, alpha=0 * fl.u.deg
            ),
            time_stepping=fl.Steady(max_steps=5000, CFL=fl.AdaptiveCFL()),
            models=[
                # These boundary names can be taken from the vm.boundary_names printout
                fl.Wall(
                    surfaces=[
                        mesh_object["fluid/leftWing"],
                        mesh_object["fluid/rightWing"],
                        mesh_object["fluid/fuselage"],
                    ],
                ),
                fl.Freestream(surfaces=mesh_object["fluid/farfield"]),  # For far field boundaries
                # Define what sort of physical model of a fluid we will use
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(),
                    turbulence_model_solver=fl.SpalartAllmaras(),
                ),
            ],
            outputs=[
                fl.VolumeOutput(output_format="tecplot", output_fields=vol_fields),
                # This mesh_object['*'] will select all the boundaries in the mesh and export the surface results.
                # Regular expressions can be used to filter for certain boundaries
                fl.SurfaceOutput(
                    surfaces=[mesh_object["*"]], output_fields=surf_fields, output_format="tecplot"
                ),
            ],
        )
    return params


######################################################################################################################
def launch_sweep(params, project):
    """
    Launch a sweep of cases.
    """

    # for example let's vary alpha:
    alphas = [-10, -5, 0, 5, 10, 11, 12]

    for alpha_angle in alphas:
        # modify the alpha
        params.operating_condition.alpha = alpha_angle * fl.u.deg

        # launch the case
        project.run_case(params=params, name=f"{alpha_angle}_case ")
        log.info(f"The case ID is: {project.case.id} with {alpha_angle=} ")


######################################################################################################################
def main():
    """
    Main function that drives the mesh upload and case launching functions.
    """

    # if you want to upload a new mesh and create a new project
    mesh_file_path = os.path.join(os.getcwd(), "mesh_name.cgns")  # mesh could also be ugrid format
    project_name = "project_name"
    project = upload_mesh(mesh_file_path, project_name)

    # Or as an alternative, if you want to run from an existing project:
    # project = fl.Project.from_cloud(
    #     'prj-XXXXXXXXXX')  # where prj-XXXXXXXXXX is an ID that can be saved from a previously created project or read off the WEBUI

    vm = project.volume_mesh  # get the volume mesh entity associated with that project.
    log.info(f"The volume mesh contains the following boundaries:{vm.boundary_names}")
    log.info(f"The volume mesh ID is: {vm.id}")

    params = make_params(vm)  # define the run params used to launch the run

    # launch_sweep(params, project)  # if you want to launch a sweep

    # or if you want to simply launch the case\
    project.run_case(params=params, name=f"case_name")
    log.info(f"case id is {project.case.id}")


######################################################################################################################

if __name__ == "__main__":
    main()


# Import necessary modules from the Flow360 library
from matplotlib.pyplot import show

import flow360 as fl
from flow360.examples import Airplane




# Step 1: Create a new project from a predefined geometry file in the Airplane example
# This initializes a project with the specified geometry and assigns it a name.
project = fl.Project.from_geometry(
    Airplane.geometry,
    name="Python Project (Geometry, from file)",
)
geo = project.geometry  # Access the geometry of the project

# Step 2: Display available groupings in the geometry (helpful for identifying group names)
geo.show_available_groupings(verbose_mode=True)

# Step 3: Group faces by a specific tag for easier reference in defining `Surface` objects
geo.group_faces_by_tag("groupName")

# Step 4: Define simulation parameters within a specific unit system
with fl.SI_unit_system:
    # Define an automated far-field boundary condition for the simulation
    far_field_zone = fl.AutomatedFarfield()

    # Set up the main simulation parameters
    params = fl.SimulationParams(
        # Meshing parameters, including boundary layer and maximum edge length
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001,  # Boundary layer thickness
                surface_max_edge_length=1,  # Maximum edge length on surfaces
            ),
            volume_zones=[far_field_zone],  # Apply the automated far-field boundary condition
        ),
        # Reference geometry parameters for the simulation (e.g., center of pressure)
        reference_geometry=fl.ReferenceGeometry(),
        # Operating conditions: setting speed and angle of attack for the simulation
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=100,  # Velocity of 100 m/s
            alpha=5 * fl.u.deg,  # Angle of attack of 5 degrees
        ),
        # Time-stepping configuration: specifying steady-state with a maximum step limit
        time_stepping=fl.Steady(max_steps=1000),
        # Define models for the simulation, such as walls and freestream conditions
        models=[
            fl.Wall(
                surfaces=[geo["*"]],  # Apply wall boundary conditions to all surfaces in geometry
            ),
            fl.Freestream(
                surfaces=[
                    far_field_zone.farfield
                ],  # Apply freestream boundary to the far-field zone
            ),
        ],
        # Define output parameters for the simulation
        outputs=[
            fl.SurfaceOutput(
                surfaces=geo["*"],  # Select all surfaces for output
                output_fields=["Cp", "Cf", "yPlus", "CfVec"],  # Output fields for post-processing
            )
        ],
    )

project.generate_volume_mesh(params, name='VolumeMesh', run_async=True, use_geometry_AI=True, raise_on_error=True)

# Step 5: Run the simulation case with the specified parameters
#project.run_case(params=params, name="Case of Simple Airplane from Python")

print("done")
