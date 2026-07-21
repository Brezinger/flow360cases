from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

import flow360 as fl
from flow360 import u
from flow360.component.simulation.folder import ROOT_FOLDER


GEOMETRY_STEP_FILE = Path(__file__).resolve().with_name(
    "EM_002_DS_12_02-EVE100_CCW_LIFTER4B_ID20025v2_tmc50p_h60wide_20260708_w_spacer.STEP"
)
GEOMETRY_CSM_FILE = GEOMETRY_STEP_FILE.with_suffix(".csm")
GEOMETRY_FILE = GEOMETRY_STEP_FILE.with_suffix(".egads")


@dataclass(frozen=True)
class OldCaseSetup:
    """Constants copied from the old POC2x2 Flow360 JSON setup."""

    name: str = "DUC_EVE_4blade_noise_from_geometry"
    geometry_length_unit: str = "m"
    flow360_folder_path: tuple[str, ...] = ("DUC", "EVE Lifter 4 blade prop")

    propeller_radius: float = 1.4
    rpm: float = 1075.0
    time_steps_per_revolution: int = 120
    moment_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moment_length: tuple[float, float, float] = (1.4, 1.4, 1.4)

    alpha_deg: float = -90.0
    beta_deg: float = 0.0
    mach: float = 0.451443
    altitude_ft: float = 2460.0
    temperature_offset_deg_c: float = 20.0

    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    physical_steps: int = 600
    max_pseudo_steps: int = 35

    wall_roughness_height: float = 1.0e-5

    # Mesh-generation placeholders. The old case JSON was produced from a
    # supplied volume mesh, so it does not define these. Keep these conservative
    # until the CAD import tags and generated mesh are checked.
    rotation_volume_radius: float = 1.55
    rotation_volume_height: float = 0.20
    rotation_volume_spacing: float = 0.002
    surface_max_edge_length: float = 0.0049
    curvature_resolution_angle_deg: float = 5.0
    boundary_layer_first_layer_thickness: float = 1.0e-5
    boundary_layer_growth_rate: float = 1.2
    surface_edge_growth_rate: float = 1.2

    @property
    def ref_area(self) -> float:
        return math.pi * self.propeller_radius**2

    @property
    def omega_rad_s(self) -> float:
        return -2.0 * math.pi * self.rpm / 60.0

    @property
    def tip_speed_m_s(self) -> float:
        return abs(self.omega_rad_s) * self.propeller_radius

    @property
    def time_step_size_s(self) -> float:
        return 60.0 / (self.rpm * self.time_steps_per_revolution)


CONFIG = OldCaseSetup()


# Update these after the first geometry upload if Flow360 exposes different names.
FACE_GROUP_ALIASES = {
    "blade1": ["zone_r1/blade1", "blade1"],
    "blade2": ["zone_r1/blade2", "blade2"],
    "blade3": ["zone_r1/blade3", "blade3"],
    "blade4": ["zone_r1/blade4", "blade4"],
    "hub": ["hub"],
}


def _write_geometry_csm_file(csm_file: Path, step_file: Path) -> None:
    if not step_file.exists():
        raise FileNotFoundError(f"Could not find STEP file for CSM import: {step_file}")

    csm_text = f"""import $/{step_file.name} -1
group -1
store $body00001
mark
   restore $body00001
      select face 21
         attribute faceName $hub
         attribute groupName $hub
   restore $body00001
      select face 33
         attribute faceName $hub
         attribute groupName $hub
   restore $body00001
      select face 66
         attribute faceName $hub
         attribute groupName $hub
   restore $body00001
      select face 47
         attribute faceName $hub
         attribute groupName $hub
dump $/{step_file.with_suffix(".egads").name} 0 1 0
"""
    csm_file.write_text(csm_text, encoding="utf-8")


def _entities_by_possible_names(geometry, possible_names: list[str]):
    entities = []
    for name in possible_names:
        try:
            entities.append(geometry[name])
        except Exception:
            continue
    return entities


def _available_surface_group_names(geometry) -> list[str]:
    registry = getattr(geometry, "internal_registry", None)
    if registry is None:
        return []
    names = []
    for entity_list in registry.internal_registry.values():
        for entity in entity_list:
            if getattr(entity, "private_attribute_entity_type_name", None) == "Surface":
                names.append(getattr(entity, "name", repr(entity)))
    return sorted(set(names))


def _wall_entities(geometry) -> dict[str, list]:
    return {
        wall_name: _entities_by_possible_names(geometry, possible_names)
        for wall_name, possible_names in FACE_GROUP_ALIASES.items()
    }


def _all_walls(wall_entities: dict[str, list]):
    walls = []
    for entities in wall_entities.values():
        walls.extend(entities)
    return walls


def _rotation_volume(cfg: OldCaseSetup):
    return fl.Cylinder(
        name="zone_r1",
        center=cfg.rotation_center * u.m,
        axis=cfg.rotation_axis,
        height=cfg.rotation_volume_height * u.m,
        outer_radius=cfg.rotation_volume_radius * u.m,
    )


def _make_project(geometry_file: Path, cfg: OldCaseSetup, flow360_folder):
    _write_geometry_csm_file(GEOMETRY_CSM_FILE, GEOMETRY_STEP_FILE)
    if geometry_file.suffix.lower() == ".egads" and not geometry_file.exists():
        raise FileNotFoundError(
            f"Could not find generated EGADS geometry: {geometry_file}. "
            f"Open or run the CSM file first to create it: {GEOMETRY_CSM_FILE}"
        )

    project = fl.Project.from_geometry(
        str(geometry_file),
        name=cfg.name,
        folder=flow360_folder,
        length_unit=cfg.geometry_length_unit,
        run_async=False,
    )
    geometry = project.geometry

    # Same convention as GBT.py. These calls are harmless if the imported CAD
    # does not have these tags; in that case FACE_GROUP_ALIASES must be updated from
    # geometry.show_available_groupings(verbose_mode=True).
    try:
        geometry.group_faces_by_tag("faceName")
    except Exception:
        pass
    try:
        geometry.group_edges_by_tag("edgeName")
    except Exception:
        pass

    return project


def _find_subfolder(folder_tree: dict, name: str) -> dict | None:
    matches = [folder for folder in folder_tree["subfolders"] if folder["name"] == name]
    if len(matches) > 1:
        raise ValueError(f"Found multiple Flow360 folders named {name!r} under {folder_tree['name']!r}.")
    return matches[0] if matches else None


def _get_or_create_flow360_folder(folder_path: tuple[str, ...]):
    folder = fl.Folder(ROOT_FOLDER)
    folder_tree = folder.get_folder_tree()

    for folder_name in folder_path:
        subfolder_tree = _find_subfolder(folder_tree, folder_name)
        if subfolder_tree is None:
            folder = fl.Folder.create(folder_name, parent_folder=folder).submit()
            folder_tree = {"name": folder_name, "id": folder.id, "subfolders": []}
        else:
            folder = fl.Folder(subfolder_tree["id"])
            folder_tree = subfolder_tree

    return folder


def _make_meshing_params(rotation_volume, cfg: OldCaseSetup):
    farfield = fl.AutomatedFarfield(domain_type="full_body", relative_size=50.0)

    return farfield, fl.MeshingParams(
        defaults=fl.MeshingDefaults(
            surface_edge_growth_rate=cfg.surface_edge_growth_rate,
            surface_max_edge_length=cfg.surface_max_edge_length * u.m,
            curvature_resolution_angle=cfg.curvature_resolution_angle_deg * u.deg,
            boundary_layer_growth_rate=cfg.boundary_layer_growth_rate,
            boundary_layer_first_layer_thickness=cfg.boundary_layer_first_layer_thickness * u.m,
        ),
        volume_zones=[
            farfield,
            fl.RotationVolume(
                name="zone_r1",
                entities=[rotation_volume],
                spacing_axial=cfg.rotation_volume_spacing * u.m,
                spacing_radial=cfg.rotation_volume_spacing * u.m,
                spacing_circumferential=cfg.rotation_volume_spacing * u.m,
            ),
        ],
        refinements=[],
    )


def _make_operating_condition(cfg: OldCaseSetup):
    thermal_state = _make_thermal_state(cfg)
    return fl.AerospaceCondition.from_mach(
        mach=cfg.mach,
        reference_mach=_tip_mach(cfg, thermal_state),
        alpha=cfg.alpha_deg * u.deg,
        beta=cfg.beta_deg * u.deg,
        thermal_state=thermal_state,
    )


def _make_thermal_state(cfg: OldCaseSetup):
    return fl.ThermalState.from_standard_atmosphere(
        altitude=cfg.altitude_ft * u.ft,
        temperature_offset=cfg.temperature_offset_deg_c * u.delta_degC,
    )


def _tip_mach(cfg: OldCaseSetup, thermal_state) -> float:
    return cfg.tip_speed_m_s / thermal_state.speed_of_sound.to("m/s").value


def _make_fluid_model(cfg: OldCaseSetup):
    return fl.Fluid(
        initial_condition=fl.NavierStokesInitialCondition(
            rho="rho",
            u="u",
            v="v",
            w="w",
            p="p",
        ),
        navier_stokes_solver=fl.NavierStokesSolver(
            absolute_tolerance=1.0e-11,
            relative_tolerance=0.01,
            order_of_accuracy=1,
            equation_evaluation_frequency=1,
            linear_solver=fl.LinearSolver(max_iterations=25),
            CFL_multiplier=1.0,
            kappa_MUSCL=0.33,
            limit_velocity=False,
            limit_pressure_density=False,
            update_jacobian_frequency=1,
            max_force_jac_update_physical_steps=0,
            riemann_solver=fl.RoeFlux(
                numerical_dissipation_factor=1.0,
                low_mach_preconditioner=False,
            ),
        ),
        turbulence_model_solver=fl.NoneSolver(),
        transition_model_solver=fl.NoneSolver(),
    )


def _make_wall_model(name: str, surfaces: list, cfg: OldCaseSetup):
    return fl.Wall(
        name=name,
        surfaces=surfaces,
        use_wall_function=True,
        heat_spec=fl.HeatFlux(0.0 * u.W / u.m**2),
        roughness_height=cfg.wall_roughness_height * u.m,
    )


def _make_models(geometry, farfield, rotation_volume, cfg: OldCaseSetup):
    wall_entities = _wall_entities(geometry)
    missing = [name for name, entities in wall_entities.items() if not entities]
    if missing:
        raise ValueError(
            f"Could not resolve wall face groups {missing}. "
            f"Available face groups: {_available_surface_group_names(geometry)}. "
            "Update FACE_GROUP_ALIASES."
        )

    models = [
        _make_fluid_model(cfg),
        fl.Rotation(
            name="zone_r1",
            volumes=[rotation_volume],
            spec=fl.AngularVelocity(cfg.omega_rad_s * u.rad / u.s),
            rotating_reference_frame_model=False,
        ),
    ]
    for wall_name, surfaces in wall_entities.items():
        models.append(_make_wall_model(f"zone_r1/{wall_name}", surfaces, cfg))
    models.append(fl.Freestream(name="Freestream", surfaces=[farfield.farfield]))
    return models, _all_walls(wall_entities)


def _make_outputs(wall_surfaces: list):
    return [
        fl.VolumeOutput(
            name="VolumeOutput",
            output_format="paraview",
            frequency=-1,
            output_fields=[
                "primitiveVars",
                "vorticity",
                "residualNavierStokes",
                "Cp",
                "Mach",
                "qcriterion",
                "mut",
                "mutRatio",
                "gradW",
                "T",
            ],
        ),
        fl.SurfaceOutput(
            name="SurfaceOutput",
            surfaces=wall_surfaces,
            output_format="paraview",
            frequency=-1,
            output_fields=[
                "primitiveVars",
                "Cp",
                "Cf",
                "CfVec",
                "yPlus",
                "nodeForcesPerUnitArea",
                "nodeNormals",
            ],
            write_single_file=False,
        ),
    ]


def build_params(project, cfg: OldCaseSetup = CONFIG):
    geometry = project.geometry
    rotation_volume = _rotation_volume(cfg)
    farfield, mesh_params = _make_meshing_params(rotation_volume, cfg)
    models, wall_surfaces = _make_models(geometry, farfield, rotation_volume, cfg)

    with fl.SI_unit_system:
        return fl.SimulationParams(
            version="25.10",
            unit_system=fl.SI_unit_system,
            meshing=mesh_params,
            reference_geometry=fl.ReferenceGeometry(
                moment_center=cfg.moment_center * u.m,
                moment_length=cfg.moment_length * u.m,
                area=cfg.ref_area * u.m**2,
            ),
            operating_condition=_make_operating_condition(cfg),
            models=models,
            time_stepping=fl.Unsteady(
                max_pseudo_steps=cfg.max_pseudo_steps,
                steps=cfg.physical_steps,
                step_size=cfg.time_step_size_s * u.s,
                CFL=fl.AdaptiveCFL(
                    min=0.1,
                    max=1.0e6,
                    max_relative_change=50.0,
                    convergence_limiting_factor=1.0,
                ),
                order_of_accuracy=2,
            ),
            outputs=_make_outputs(wall_surfaces),
        )


def define_and_run(
    *,
    geometry_file: Path = GEOMETRY_FILE,
    cfg: OldCaseSetup = CONFIG,
    flow360_folder=None,
    generate_surface_mesh: bool = True,
    generate_volume_mesh: bool = False,
    run_case: bool = False,
    results_path: str | None = None,
):
    project = _make_project(geometry_file, cfg, flow360_folder)
    params = build_params(project, cfg)

    if generate_surface_mesh:
        project.generate_surface_mesh(
            params=params,
            name="SurfaceMesh",
            use_beta_mesher=False,
            use_geometry_AI=False,
            run_async=False,
            raise_on_error=True,
        )

    if generate_volume_mesh:
        project.generate_volume_mesh(
            params=params,
            name="VolumeMesh",
            use_beta_mesher=False,
            use_geometry_AI=False,
            run_async=False,
            raise_on_error=True,
        )

    if not run_case:
        return project.id

    project.run_case(
        params=params,
        name="DUC_EVE_4blade_noise_case",
        use_beta_mesher=False,
        use_geometry_AI=False,
        run_async=False,
        raise_on_error=True,
    )
    case = project.case
    case.wait()

    if results_path is not None:
        case.results.download(
            surface=True,
            volume=True,
            total_forces=True,
            nonlinear_residuals=True,
            destination=os.path.join(results_path, case.name),
        )

    return case.results.total_forces.as_dataframe()


def main():
    generate_surface_mesh = False
    generate_volume_mesh = False
    run_case = False

    folder = _get_or_create_flow360_folder(CONFIG.flow360_folder_path)

    project_id = define_and_run(
        flow360_folder=folder,
        generate_surface_mesh=generate_surface_mesh,
        generate_volume_mesh=generate_volume_mesh,
        run_case=run_case,
    )
    print(f"Project id: {project_id}")


if __name__ == "__main__":
    main()
