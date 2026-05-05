import flow360 as fl
from flow360.examples.XV15_csm import XV15_CSM

import numpy as np
import matplotlib.pyplot as plt

XV15_CSM.get_files()
project = fl.Project.from_geometry(
    XV15_CSM.geometry, length_unit="inch", name="MRF tutorial from Python"
)
geometry = project.geometry

with fl.SI_unit_system:

    farfield = fl.AutomatedFarfield()
    R = 3.81
    MRF_cylinder = fl.Cylinder(
        center=(0, 0, 0),
        outer_radius=190 * fl.u.inch,
        height=40 * fl.u.inch,
        axis=(0, 0, -1),
        name="RotatingCylinder",
    )

    Wake_Cylinder = fl.Cylinder(
        center=(0, 0, -200) * fl.u.inch,
        outer_radius=210 * fl.u.inch,
        height=500 * fl.u.inch,
        axis=(0, 0, -1),
        name="WakeCylinder",
    )

RotatingZone = fl.RotationVolume(
    name="RotatingZone",
    spacing_axial=1 * fl.u.inch,
    spacing_circumferential=2 * fl.u.inch,
    spacing_radial=2 * fl.u.inch,
    entities=MRF_cylinder,
    enclosed_entities=geometry["*"],
)

meshing = fl.MeshingParams(
    defaults=fl.MeshingDefaults(
        surface_max_edge_length=1 * fl.u.inch,
        boundary_layer_first_layer_thickness=1.5e-6 * fl.u.m,
        curvature_resolution_angle=8 * fl.u.deg,
        boundary_layer_growth_rate=1.2,
    ),
    volume_zones=[farfield, RotatingZone],
    refinements=[fl.UniformRefinement(entities=Wake_Cylinder, spacing=30 * fl.u.mm)],
)

omega = 589 * fl.u.rpm

MRF = fl.Rotation(name="MRF", volumes=MRF_cylinder, spec=fl.AngularVelocity(omega))

time_stepping = fl.Steady(
    max_steps=3500, CFL=fl.AdaptiveCFL(convergence_limiting_factor=0.75)
)

with fl.SI_unit_system:

    params = fl.SimulationParams(
        operating_condition=fl.AerospaceCondition.from_mach(
            mach=0.12,
            alpha=-90 * fl.u.deg,
        ),
        reference_geometry=fl.ReferenceGeometry(
            moment_center=(0, 0, 0), moment_length=(R, R, R), area=np.pi * R**2
        ),
        time_stepping=time_stepping,
        meshing=meshing,
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    low_mach_preconditioner=True, absolute_tolerance=1e-10
                ),
                turbulence_model_solver=fl.SpalartAllmaras(),
            ),
            fl.Wall(surfaces=geometry["*"]),
            fl.Freestream(surfaces=farfield.farfield),
            MRF,
        ],
        outputs=[
            fl.SurfaceOutput(
                surfaces=geometry["*"],
                output_fields=[
                    "primitiveVars",
                    "Cp",
                    "Cf",
                    "CfVec",
                ],
            ),
            fl.VolumeOutput(
                output_fields=[
                    "Mach",
                    "primitiveVars",
                    "qcriterion",
                    "Cp",
                ]
            ),
        ],
    )

project.generate_surface_mesh(params=params, name='SurfaceMesh', run_async=False)
