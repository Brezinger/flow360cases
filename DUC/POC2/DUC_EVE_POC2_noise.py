from __future__ import annotations

import json
import os
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

os.environ["FLOW360_DISABLE_SYSTEM_CERTS"] = "1"

import flow360 as fl
from flow360 import u
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import CaseInterfaceV2
from flow360.component.simulation.folder import ROOT_FOLDER
from flow360.environment import current_environment


AEROACOUSTIC_SOURCE_FILE = Path(__file__).resolve().with_name(
    "Q40LT_step4_simulation.json"
)
DEFAULT_SURFACE_MESH_FILE = Path(__file__).resolve().with_name("POC2.cgns")


@dataclass(frozen=True)
class VolumeCylinderRefinementSpec:
    name: str
    center_z: float
    z_min: float
    z_max: float
    inner_radius: float
    outer_radius: float
    spacing: float


@dataclass(frozen=True)
class BladeVortexBoxRefinementSpec:
    name: str
    center: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], tuple[float, float, float]]
    length_radius_ratio: float
    tangential_width_radius_ratio: float
    spacing: float


@dataclass(frozen=True)
class CaseSetup:
    """Constants copied from the old POC2x2 Flow360 JSON setup."""

    name: str = "DUC POC2"
    surface_mesh_length_unit: str = "mm"
    flow360_folder_path: tuple[str, ...] = ("DUC", "POC2")

    propeller_radius: float = 1.4

    moment_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moment_length: tuple[float, float, float] = (1.4, 1.4, 1.4)

    alpha_deg: float = -90.0
    beta_deg: float = 0.0
    altitude_ft: float = 2460.0
    temperature_offset_deg_c: float = 20.0

    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_center: tuple[float, float, float] = (0.0, 0.0, 0.0184375)

    # ---------------------------------------
    # consecutive refined time stepping setup
    # ---------------------------------------
    # step 0
    """case_step: int = 0
    time_steps_per_revolution: int = 120
    parent_case_id: str | None = None
    physical_steps: int = 600
    max_pseudo_steps: int = 35
    rpm: float = 1000.0
    rotation_direction: int = -1
    wall_roughness_height: float = 1.0e-5
    navier_stokes_relative_tolerance: float = 0.01
    navier_stokes_order_of_accuracy: int = 1
    navier_stokes_linear_solver_max_iterations: int = 25
    numerical_dissipation_factor: float = 1.
    low_mach_preconditioner: bool = False
    adaptive_cfl_min: float = 0.1
    adaptive_cfl_max: float = 1.0e6
    adaptive_cfl_max_relative_change: float = 50.0
    adaptive_cfl_convergence_limiting_factor: float = 1.0
    include_aeroacoustic_output: bool = False"""
    # ------
    # step 1
    """case_step: int = 1
    parent_case_id: str | None = "case-8ab2e55e-63c6-4167-9679-1b77eac32325"
    time_steps_per_revolution: int = 120 * 6
    physical_steps: int = 3600
    max_pseudo_steps: int = 15
    rpm: float = 1000.0
    rotation_direction: int = -1
    wall_roughness_height: float = 1.0e-5
    navier_stokes_relative_tolerance: float = 0.0175
    navier_stokes_order_of_accuracy: int = 2
    navier_stokes_linear_solver_max_iterations: int = 25
    numerical_dissipation_factor: float = 0.7
    low_mach_preconditioner: bool = True
    adaptive_cfl_min: float = 0.1
    adaptive_cfl_max: float = 1.0e6
    adaptive_cfl_max_relative_change: float = 50.0
    adaptive_cfl_convergence_limiting_factor: float = 1.0
    include_aeroacoustic_output: bool = False"""
    # ------
    # step 2
    """case_step: int = 2
    parent_case_id: str | None = "case-2cd84f26-a687-4209-8f74-7f2fc69c1b5a"
    time_steps_per_revolution: int = 120 * 16
    num_revolutions: int = 2
    physical_steps: int = num_revolutions * time_steps_per_revolution
    max_pseudo_steps: int = 12
    rpm: float = 1000.0
    rotation_direction: int = -1
    wall_roughness_height: float = 1.0e-5
    navier_stokes_relative_tolerance: float = 0.0175
    navier_stokes_order_of_accuracy: int = 2
    navier_stokes_linear_solver_max_iterations: int = 25
    numerical_dissipation_factor: float = 0.5
    low_mach_preconditioner: bool = True
    adaptive_cfl_min: float = 0.1
    adaptive_cfl_max: float = 1.0e6
    adaptive_cfl_max_relative_change: float = 50.0
    adaptive_cfl_convergence_limiting_factor: float = 1.0
    include_aeroacoustic_output: bool = False"""
    # ------
    # step 3
    """case_step: int = 3
    parent_case_id: str | None = "case-beb13e25-7f8b-4891-9f9d-53f9d8462d9c"
    time_steps_per_revolution: int = 120 * 32
    num_revolutions: int = 2 # 3 in original simulation for step 3
    physical_steps: int = num_revolutions * time_steps_per_revolution
    max_pseudo_steps: int = 8
    rpm: float = 1000.0
    rotation_direction: int = -1
    wall_roughness_height: float = 1.0e-5
    navier_stokes_relative_tolerance: float = 0.0175
    navier_stokes_order_of_accuracy: int = 2
    navier_stokes_linear_solver_max_iterations: int = 25
    numerical_dissipation_factor: float = 0.2
    low_mach_preconditioner: bool = True
    adaptive_cfl_min: float = 0.1
    adaptive_cfl_max: float = 1e2
    adaptive_cfl_max_relative_change: float = 50.0
    adaptive_cfl_convergence_limiting_factor: float = 0.5
    include_aeroacoustic_output: bool = True
    aeroacoustic_solver_start_time_s: float = 0.0
    aeroacoustic_force_clean_start: bool = False"""
    # ------
    # step 4
    case_step: int = 4
    parent_case_id: str | None = "case-4fcc3b53-bb3e-43fc-97c1-7047b13067e3"
    time_steps_per_revolution: int = int(120 * 32 * 25 / 12)
    num_revolutions: int = 3 # 3 in original simulation for step 3
    physical_steps: int = num_revolutions * time_steps_per_revolution
    max_pseudo_steps: int = 10
    rpm: float = 1000.0
    rotation_direction: int = -1
    wall_roughness_height: float = 0.
    navier_stokes_relative_tolerance: float = 0.0175
    navier_stokes_order_of_accuracy: int = 2
    navier_stokes_linear_solver_max_iterations: int = 20
    numerical_dissipation_factor: float = 0.2
    low_mach_preconditioner: bool = True
    adaptive_cfl_min: float = 0.1
    adaptive_cfl_max: float = 1e2
    adaptive_cfl_max_relative_change: float = 50.0
    adaptive_cfl_convergence_limiting_factor: float = 0.5
    include_aeroacoustic_output: bool = True
    aeroacoustic_solver_start_time_s: float = 0.0
    aeroacoustic_force_clean_start: bool = True

    include_aeroacoustic_observer_group_2: bool = True


    # Volume-mesh settings for the Gmsh surface mesh.
    rotation_volume_radius: float = 1.5
    rotation_volume_height: float = 0.423125
    octree_base_spacing: float = 0.00295898 * 1.234
    rotation_volume_spacing: float = octree_base_spacing * 2
    farfield_relative_size: float = 50
    boundary_layer_first_layer_thickness: float = 3.3e-5 * 1.5
    boundary_layer_growth_rate: float = 1.16
    tip_wake_inner_radius: float = 1.1
    tip_wake_z_lower: float = -0.1
    tip_wake_spacing: float = octree_base_spacing
    blade_vortex_spacing: float = octree_base_spacing

    def __post_init__(self) -> None:
        if type(self.rotation_direction) is not int or self.rotation_direction not in (-1, 1):
            raise ValueError("rotation_direction must be +1 or -1")

    @property
    def ref_area(self) -> float:
        return math.pi * self.propeller_radius**2

    @property
    def omega_rad_s(self) -> float:
        return self.rotation_direction * 2.0 * math.pi * self.rpm / 60.0

    @property
    def tip_speed_m_s(self) -> float:
        return abs(self.omega_rad_s) * self.propeller_radius

    @property
    def time_step_size_s(self) -> float:
        return 60.0 / (self.rpm * self.time_steps_per_revolution)


CONFIG = CaseSetup()


VOLUME_CYLINDER_REFINEMENTS: tuple[VolumeCylinderRefinementSpec, ...] = (
    VolumeCylinderRefinementSpec(
        name="intermediate_cylinder",
        center_z=0.0366215,
        z_min=-0.36621,
        z_max=0.439453,
        inner_radius=0.0,
        outer_radius=1.68457,
        spacing=CONFIG.octree_base_spacing * 2**3,
    ),
    VolumeCylinderRefinementSpec(
        name="intermediate_annulus",
        center_z=-3.016105,
        z_min=-6.26221,
        z_max=0.23,
        inner_radius=1.06201,
        outer_radius=1.9043,
        spacing=CONFIG.octree_base_spacing * 2**3,
    ),
    VolumeCylinderRefinementSpec(
        name="coarse_cylinder",
        center_z=0.073242,
        z_min=-0.732422,
        z_max=0.878906,
        inner_radius=0.0,
        outer_radius=1.9043,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    VolumeCylinderRefinementSpec(
        name="short_coarse_annulus",
        center_z=-2.966309,
        z_min=-6.66504,
        z_max=0.732422,
        inner_radius=0.585938,
        outer_radius=2.05078,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    VolumeCylinderRefinementSpec(
        name="coarse_annulus",
        center_z=-3.9916985,
        z_min=-8.42285,
        z_max=0.439453,
        inner_radius=0.732422,
        outer_radius=2.27051,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    VolumeCylinderRefinementSpec(
        name="outer_short_wide_cylinder",
        center_z=-4.82941,
        z_min=-10.83,
        z_max=1.17118,
        inner_radius=0.0,
        outer_radius=2.65,
        spacing=CONFIG.octree_base_spacing * 2**5,
    ),
    VolumeCylinderRefinementSpec(
        name="outermost_cylinder",
        center_z=-3.808595,
        z_min=-9.22852,
        z_max=1.61133,
        inner_radius=0.0,
        outer_radius=3.06,
        spacing=CONFIG.octree_base_spacing * 2**5,
    ),
)


# Span and tangential widths as multiples of the original 1.4 m blade radius.
BLADE_VORTEX_REFINEMENTS: tuple[BladeVortexBoxRefinementSpec, ...] = (
    BladeVortexBoxRefinementSpec(
        name="Blade1+3Refinement",
        center=(0.0, 0.0, 0.13),
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        length_radius_ratio=2.97 / 1.4,
        tangential_width_radius_ratio=0.34 / 1.4,
        spacing=CONFIG.blade_vortex_spacing,
    ),
    BladeVortexBoxRefinementSpec(
        name="Blade2+4Refinement",
        center=(0.0, 0.0, 0.003),
        axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        length_radius_ratio=2.97 / 1.4,
        tangential_width_radius_ratio=0.34 / 1.4,
        spacing=CONFIG.blade_vortex_spacing,
    ),
)


BLADE_IDS = (1, 2, 3, 4)


SURFACE_MESH_WALL_PREFIXES = {
    "hub": ["hub_"],
    **{f"blade{blade_id}": [f"blade{blade_id}"] for blade_id in BLADE_IDS},
}


def _available_surface_group_names(surface_mesh) -> list[str]:
    return sorted(getattr(surface_mesh, "boundary_names", []))


def _wall_entities(mesh) -> dict[str, list]:
    boundary_names = getattr(mesh, "boundary_names", [])
    wall_entities = {
        wall_name: [
            mesh[boundary_name]
            for boundary_name in boundary_names
            if any(
                boundary_name.startswith(prefix)
                or boundary_name.rsplit("/", 1)[-1].startswith(prefix)
                for prefix in prefixes
            )
        ]
        for wall_name, prefixes in SURFACE_MESH_WALL_PREFIXES.items()
    }
    if any(wall_entities.values()):
        return wall_entities

    wall_boundary_names = [
        name for name in boundary_names
        if "farfield" not in name.lower() and "interface" not in name.lower()
    ]
    if not wall_boundary_names:
        wall_boundary_names = list(boundary_names)
    return {"wall": [mesh[name] for name in wall_boundary_names]}


def _all_walls(wall_entities: dict[str, list]):
    walls = []
    for entities in wall_entities.values():
        walls.extend(entities)
    return walls


def _project_root_type_name(project) -> str | None:
    root_item_type = getattr(getattr(project, "metadata", None), "root_item_type", None)
    return getattr(root_item_type, "name", None)


def _rotation_volume(cfg: CaseSetup):
    return fl.Cylinder(
        name="zone_r1",
        center=cfg.rotation_center * u.m,
        axis=cfg.rotation_axis,
        height=cfg.rotation_volume_height * u.m,
        outer_radius=cfg.rotation_volume_radius * u.m,
    )


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


def _make_meshing_defaults(cfg: CaseSetup):
    return fl.MeshingDefaults(
        boundary_layer_growth_rate=cfg.boundary_layer_growth_rate,
        boundary_layer_first_layer_thickness=cfg.boundary_layer_first_layer_thickness * u.m,
        octree_spacing=fl.OctreeSpacing(base_spacing=cfg.octree_base_spacing * u.m),
    )


def _make_volume_cylinder_refinements(cfg: CaseSetup):
    refinements = []
    for spec in VOLUME_CYLINDER_REFINEMENTS:
        cylinder = fl.Cylinder(
            name=f"{spec.name}_volume",
            center=(0.0, 0.0, spec.center_z) * u.m,
            axis=cfg.rotation_axis,
            height=(spec.z_max - spec.z_min) * u.m,
            inner_radius=spec.inner_radius * u.m,
            outer_radius=spec.outer_radius * u.m,
        )
        refinements.append(
            fl.UniformRefinement(
                name=f"{spec.name}_refinement",
                entities=[cylinder],
                spacing=spec.spacing * u.m,
            )
        )
    return refinements


def _make_tip_wake_refinement(cfg: CaseSetup):
    rotation_z_center = cfg.rotation_center[2]
    z_upper = rotation_z_center + 0.5 * cfg.rotation_volume_height
    z_lower = cfg.tip_wake_z_lower
    cylinder = fl.Cylinder(
        name="tip_wake_annulus_volume",
        center=(cfg.rotation_center[0], cfg.rotation_center[1], 0.5 * (z_lower + z_upper)) * u.m,
        axis=cfg.rotation_axis,
        height=(z_upper - z_lower) * u.m,
        inner_radius=cfg.tip_wake_inner_radius * u.m,
        outer_radius=cfg.rotation_volume_radius * u.m,
    )
    return fl.UniformRefinement(
        name="tip_wake_annulus_refinement",
        entities=[cylinder],
        spacing=cfg.tip_wake_spacing * u.m,
    )


def _make_blade_vortex_box_refinements(cfg: CaseSetup):
    refinements = []
    upper_z = cfg.rotation_center[2] + 0.5 * cfg.rotation_volume_height
    blade_vortex_z_size = 2.0 * (upper_z - BLADE_VORTEX_REFINEMENTS[0].center[2])
    if blade_vortex_z_size <= 0.0:
        raise ValueError(
            "Blade-vortex refinement center is above the rotation-volume upper boundary. "
            f"center_z={BLADE_VORTEX_REFINEMENTS[0].center[2]}, upper_z={upper_z}"
        )
    for spec in BLADE_VORTEX_REFINEMENTS:
        box = fl.Box.from_principal_axes(
            name=f"{spec.name}_volume",
            center=spec.center * u.m,
            axes=spec.axes,
            size=(
                spec.length_radius_ratio * cfg.propeller_radius,
                spec.tangential_width_radius_ratio * cfg.propeller_radius,
                blade_vortex_z_size,
            ) * u.m,
        )
        refinements.append(
            fl.UniformRefinement(
                name=spec.name,
                entities=[box],
                spacing=spec.spacing * u.m,
            )
        )
    return refinements


def _make_meshing_params(
    rotation_volume,
    cfg: CaseSetup,
    *,
    include_volume_refinements: bool = True,
    enclosed_entities: list | None = None,
):
    farfield = fl.AutomatedFarfield(
        name="farfield",
        relative_size=cfg.farfield_relative_size,
    )
    refinements = []
    if include_volume_refinements:
        refinements.extend(
            [
                *_make_volume_cylinder_refinements(cfg),
                _make_tip_wake_refinement(cfg),
                *_make_blade_vortex_box_refinements(cfg),
            ]
        )
    return farfield, fl.MeshingParams(
        defaults=_make_meshing_defaults(cfg),
        volume_zones=[
            farfield,
            fl.RotationVolume(
                name="zone_r1",
                entities=[rotation_volume],
                enclosed_entities=enclosed_entities,
                spacing_axial=cfg.rotation_volume_spacing * u.m,
                spacing_radial=cfg.rotation_volume_spacing * u.m,
                spacing_circumferential=cfg.rotation_volume_spacing * u.m,
            ),
        ],
        refinements=refinements,
    )


def _make_operating_condition(cfg: CaseSetup):
    thermal_state = _make_thermal_state(cfg)
    return fl.AerospaceCondition.from_mach(
        mach=0 / thermal_state.speed_of_sound.to("m/s").value,
        reference_mach=_tip_mach(cfg, thermal_state),
        alpha=cfg.alpha_deg * u.deg,
        beta=cfg.beta_deg * u.deg,
        thermal_state=thermal_state,
    )


def _make_thermal_state(cfg: CaseSetup):
    return fl.ThermalState.from_standard_atmosphere(
        altitude=cfg.altitude_ft * u.ft,
        temperature_offset=cfg.temperature_offset_deg_c * u.delta_degC,
    )


def _tip_mach(cfg: CaseSetup, thermal_state) -> float:
    return cfg.tip_speed_m_s / thermal_state.speed_of_sound.to("m/s").value


def _reference_velocity(thermal_state, reference_mach: float):
    return reference_mach * thermal_state.speed_of_sound


def _make_fluid_model(cfg: CaseSetup):
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
            relative_tolerance=cfg.navier_stokes_relative_tolerance,
            order_of_accuracy=cfg.navier_stokes_order_of_accuracy,
            equation_evaluation_frequency=1,
            linear_solver=fl.LinearSolver(max_iterations=cfg.navier_stokes_linear_solver_max_iterations),
            CFL_multiplier=1.0,
            kappa_MUSCL=0.33,
            limit_velocity=False,
            limit_pressure_density=False,
            update_jacobian_frequency=1,
            max_force_jac_update_physical_steps=0,
            riemann_solver=fl.RoeFlux(
                numerical_dissipation_factor=cfg.numerical_dissipation_factor,
                low_mach_preconditioner=cfg.low_mach_preconditioner,
            ),
        ),
        turbulence_model_solver=fl.NoneSolver(),
        transition_model_solver=fl.NoneSolver(),
    )


def _make_wall_model(name: str, surfaces: list, cfg: CaseSetup):
    return fl.Wall(
        name=name,
        surfaces=surfaces,
        use_wall_function=fl.WallFunction(),
        heat_spec=fl.HeatFlux(0.0 * u.W / u.m**2),
        roughness_height=cfg.wall_roughness_height * u.m,
    )


def _make_models(geometry, farfield, rotation_volume, cfg: CaseSetup, wall_entities=None):
    if wall_entities is None:
        wall_entities = _wall_entities(geometry)
    missing = [name for name, entities in wall_entities.items() if not entities]
    if missing:
        raise ValueError(
            f"Could not resolve wall boundaries {missing}. "
            f"Available boundaries: {_available_surface_group_names(geometry)}."
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


def _volume_mesh_rotation_volume(volume_mesh, cfg: CaseSetup):
    preferred_zone_name = "zone_r1"
    zone_names = getattr(volume_mesh, "zone_names", [])
    if preferred_zone_name in zone_names:
        return volume_mesh[preferred_zone_name]
    if len(zone_names) == 1:
        return volume_mesh[zone_names[0]]
    raise ValueError(
        f"Could not identify rotation volume {preferred_zone_name!r}. "
        f"Available volume zones: {zone_names}"
    )


def _volume_mesh_farfield_surfaces(volume_mesh) -> list:
    boundary_names = getattr(volume_mesh, "boundary_names", [])
    farfield_names = [
        boundary_name
        for boundary_name in boundary_names
        if "farfield" in boundary_name.lower()
    ]
    if not farfield_names:
        raise ValueError(
            f"Could not identify farfield boundaries in parent volume mesh. "
            f"Available boundaries: {boundary_names}"
        )
    return [volume_mesh[boundary_name] for boundary_name in farfield_names]


def _make_models_from_volume_mesh(volume_mesh, cfg: CaseSetup):
    wall_entities = _wall_entities(volume_mesh)
    missing = [name for name, entities in wall_entities.items() if not entities]
    if missing:
        raise ValueError(
            f"Could not resolve wall boundary groups {missing}. "
            f"Available boundaries: {getattr(volume_mesh, 'boundary_names', [])}. "
            "Update SURFACE_MESH_WALL_PREFIXES."
        )

    rotation_volume = _volume_mesh_rotation_volume(volume_mesh, cfg)
    rotation_name = getattr(rotation_volume, "name", "zone_r1")
    models = [
        _make_fluid_model(cfg),
        fl.Rotation(
            name=rotation_name,
            volumes=[rotation_volume],
            spec=fl.AngularVelocity(cfg.omega_rad_s * u.rad / u.s),
            rotating_reference_frame_model=False,
        ),
    ]
    for wall_name, surfaces in wall_entities.items():
        models.append(_make_wall_model(f"{rotation_name}/{wall_name}", surfaces, cfg))
    models.append(
        fl.Freestream(
            name="Freestream",
            surfaces=_volume_mesh_farfield_surfaces(volume_mesh),
        )
    )
    return models, _all_walls(wall_entities)


def _make_aeroacoustic_output(cfg: CaseSetup, source_file: Path = AEROACOUSTIC_SOURCE_FILE):
    with source_file.open(encoding="utf-8") as file:
        source = json.load(file)

    if "aeroacousticOutput" in source:
        aeroacoustic_output = source["aeroacousticOutput"]
        observers = [
            fl.Observer(
                position=position * u.m,
                group_name=f"observer_{index:03d}",
            )
            for index, position in enumerate(aeroacoustic_output["observers"], start=1)
        ]
        write_per_surface_output = aeroacoustic_output["writePerSurfaceOutput"]
        patch_type = aeroacoustic_output["patchType"]
    else:
        aeroacoustic_outputs = [
            output for output in source.get("outputs", [])
            if output.get("output_type") == "AeroAcousticOutput"
        ]
        if len(aeroacoustic_outputs) != 1:
            raise ValueError(
                f"Expected exactly one AeroAcousticOutput in {source_file}, "
                f"found {len(aeroacoustic_outputs)}"
            )
        aeroacoustic_output = aeroacoustic_outputs[0]
        observers = []
        included_groups = {"1"}
        if cfg.include_aeroacoustic_observer_group_2:
            included_groups.add("2")
        for observer in aeroacoustic_output["observers"]:
            if observer["group_name"] not in included_groups:
                continue
            position = observer["position"]
            units = position["units"]
            try:
                position_unit = getattr(u, units)
            except AttributeError as exc:
                raise ValueError(
                    f"Unsupported observer position unit {units!r} in {source_file}"
                ) from exc
            observers.append(
                fl.Observer(
                    position=position["value"] * position_unit,
                    group_name=observer["group_name"],
                )
            )
        write_per_surface_output = aeroacoustic_output["write_per_surface_output"]
        patch_type = aeroacoustic_output["patch_type"]

    return fl.AeroAcousticOutput(
        name="AeroAcousticOutput",
        observers=observers,
        write_per_surface_output=write_per_surface_output,
        patch_type=patch_type,
        aeroacoustic_solver_start_time=cfg.aeroacoustic_solver_start_time_s * u.s,
        force_clean_start=cfg.aeroacoustic_force_clean_start,
    )


def _make_outputs(wall_surfaces: list, cfg: CaseSetup):
    outputs = [
        fl.VolumeOutput(
            name="VolumeOutput",
            output_format=["paraview"],
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
            output_format=["paraview"],
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

    if cfg.include_aeroacoustic_output:
        outputs.append(_make_aeroacoustic_output(cfg))

    return outputs


def build_params(
    project,
    cfg: CaseSetup = CONFIG,
    *,
    include_volume_refinements: bool = True,
):
    root_type_name = _project_root_type_name(project)
    if root_type_name == "VOLUME_MESH":
        mesh_params = None
        models, wall_surfaces = _make_models_from_volume_mesh(project.volume_mesh, cfg)
    elif root_type_name == "SURFACE_MESH":
        surface_mesh = project.surface_mesh
        wall_entities = _wall_entities(surface_mesh)
        wall_surfaces = _all_walls(wall_entities)
        rotation_volume = _rotation_volume(cfg)
        farfield, mesh_params = _make_meshing_params(
            rotation_volume,
            cfg,
            include_volume_refinements=include_volume_refinements,
            enclosed_entities=wall_surfaces,
        )
        models, wall_surfaces = _make_models(surface_mesh, farfield, rotation_volume, cfg, wall_entities)
    else:
        raise ValueError(f"Expected a surface or volume mesh project, got {root_type_name!r}")

    with fl.SI_unit_system:
        params_kwargs = {
            "version": "25.10.6",
            "unit_system": fl.SI_unit_system,
            "reference_geometry": fl.ReferenceGeometry(
                moment_center=cfg.moment_center * u.m,
                moment_length=cfg.moment_length * u.m,
                area=cfg.ref_area * u.m**2,
            ),
            "operating_condition": _make_operating_condition(cfg),
            "models": models,
            "time_stepping": fl.Unsteady(
                max_pseudo_steps=cfg.max_pseudo_steps,
                steps=cfg.physical_steps,
                step_size=cfg.time_step_size_s * u.s,
                CFL=fl.AdaptiveCFL(
                    min=cfg.adaptive_cfl_min,
                    max=cfg.adaptive_cfl_max,
                    max_relative_change=cfg.adaptive_cfl_max_relative_change,
                    convergence_limiting_factor=cfg.adaptive_cfl_convergence_limiting_factor,
                ),
                order_of_accuracy=2,
            ),
            "outputs": _make_outputs(wall_surfaces, cfg),
        }
        if mesh_params is not None:
            params_kwargs["meshing"] = mesh_params
        return fl.SimulationParams(**params_kwargs)


def _make_project_from_local_surface_mesh(surface_mesh_file: Path, cfg: CaseSetup, flow360_folder):
    surface_mesh_file = Path(surface_mesh_file)
    if not surface_mesh_file.is_file():
        raise FileNotFoundError(f"Could not find Gmsh CGNS surface mesh: {surface_mesh_file}")
    if surface_mesh_file.suffix.lower() != ".cgns":
        raise ValueError(f"Expected a CGNS surface mesh, got {surface_mesh_file}")

    print(f"Uploading surface mesh: {surface_mesh_file}")
    return fl.Project.from_surface_mesh(
        str(surface_mesh_file),
        name=cfg.name + "_simulation_from_surface_mesh",
        folder=flow360_folder,
        length_unit=cfg.surface_mesh_length_unit,
        run_async=False,
    )


def _fork_from_case(cfg: CaseSetup):
    if not cfg.parent_case_id:
        return None
    parent_case_id = _parent_case_full_id(cfg)
    case = fl.Case(parent_case_id.removeprefix("case-"))
    # The SDK Case constructor validates bare UUIDs, but the v2 draft API
    # expects the full asset id when selecting a Case as the draft source.
    case._id = parent_case_id
    return case


def _case_run_name(cfg: CaseSetup, fallback: str) -> str:
    if cfg.parent_case_id and cfg.case_step in (1, 2, 3, 4):
        return f"DUC_POC2_noise_step{cfg.case_step}"
    return fallback


def _parent_case_full_id(cfg: CaseSetup) -> str:
    if not cfg.parent_case_id:
        raise ValueError("Cannot resolve a parent case id without cfg.parent_case_id.")
    parent_case_id = cfg.parent_case_id.strip()
    return parent_case_id if parent_case_id.startswith("case-") else f"case-{parent_case_id}"


def _parent_case_project_id(cfg: CaseSetup) -> str:
    parent_case_id = _parent_case_full_id(cfg)
    case_info = RestApi(CaseInterfaceV2.endpoint, environment_provider=current_environment).get(
        path=f"{CaseInterfaceV2.endpoint}/{parent_case_id}"
    )
    project_id = case_info.get("projectId")
    if not project_id:
        raise ValueError(f"Parent case {parent_case_id} does not expose a projectId in v2 metadata.")
    return project_id


def _make_project_from_parent_case(cfg: CaseSetup):
    project_id = _parent_case_project_id(cfg)
    print(f"Loading parent project {project_id} from parent case {_parent_case_full_id(cfg)}.")
    return fl.Project.from_cloud(project_id)


def define_and_run(
    *,
    surface_mesh_file: Path = DEFAULT_SURFACE_MESH_FILE,
    cfg: CaseSetup = CONFIG,
    flow360_folder=None,
    submit_draft_only: bool = True,
    generate_volume_mesh: bool = False,
    run_case: bool = False,
    use_beta_mesher: bool = True,
    bypass_length_scale_warning: bool = True,
    results_path: str | None = None,
):
    surface_mesh_file = Path(surface_mesh_file)
    fork_from = _fork_from_case(cfg)
    project = (
        _make_project_from_parent_case(cfg)
        if fork_from is not None
        else _make_project_from_local_surface_mesh(surface_mesh_file, cfg, flow360_folder)
    )
    params = build_params(
        project,
        cfg,
        include_volume_refinements=generate_volume_mesh,
    )

    warning_context = (
        fl.warning_bypass("potential_length_scale_mismatch")
        if bypass_length_scale_warning
        else nullcontext()
    )

    with warning_context:
        if submit_draft_only and not generate_volume_mesh and not run_case:
            draft = project.run_case(
                params=params,
                name=_case_run_name(cfg, "DUC_POC2_noise_setup"),
                use_beta_mesher=use_beta_mesher,
                use_geometry_AI=False,
                fork_from=fork_from,
                run_async=False,
                raise_on_error=True,
                draft_only=True,
            )
            return draft.id

        if generate_volume_mesh:
            project.generate_volume_mesh(
                params=params,
                name="VolumeMesh",
                use_beta_mesher=use_beta_mesher,
                use_geometry_AI=False,
                run_async=False,
                raise_on_error=True,
            )

        if not run_case:
            return project.id
        project.run_case(
            params=params,
            name=_case_run_name(cfg, "DUC_POC2_noise_case"),
            use_beta_mesher=use_beta_mesher,
            use_geometry_AI=False,
            fork_from=fork_from,
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
    # Continue from the configured parent case.
    use_beta_mesher = True
    generate_volume_mesh = False
    run_case = False

    folder = _get_or_create_flow360_folder(CONFIG.flow360_folder_path)

    thermal_state = _make_thermal_state(CONFIG)
    reference_mach = _tip_mach(CONFIG, thermal_state)
    reference_velocity = _reference_velocity(thermal_state, reference_mach)
    dynamic_pressure = 0.5 * thermal_state.density * reference_velocity**2
    print(
        "Reference thermal state:\n"
        f"  Temperature: {thermal_state.temperature.to('K').value:.2f} K "
        f"({thermal_state.temperature.to('degC').value:.2f} degC)\n"
        f"  Density: {thermal_state.density.to('kg/m**3').value:.6f} kg/m^3\n"
        f"  Speed of sound: {thermal_state.speed_of_sound.to('m/s').value:.3f} m/s\n"
        f"  Reference Mach: {reference_mach:.6f}\n"
        f"  Reference velocity: {reference_velocity.to('m/s').value:.3f} m/s\n"
        f"  Dynamic pressure: {dynamic_pressure.to('Pa').value:.2f} Pa "
        f"({dynamic_pressure.to('kPa').value:.3f} kPa)\n"
        f"  RPM: {CONFIG.rpm:.3f}\n"
        f"  Rotation direction: {CONFIG.rotation_direction:+d}\n"
        f"  Omega: {CONFIG.omega_rad_s:.6f} rad/s\n"
        f"  Physical steps: {CONFIG.physical_steps}\n"
        f"  Time step size: {CONFIG.time_step_size_s:.9f} s\n"
        f"  Parent case id: {CONFIG.parent_case_id or 'None'}\n"

    )

    project_id = define_and_run(
        flow360_folder=folder,
        submit_draft_only=True,
        generate_volume_mesh=generate_volume_mesh,
        run_case=run_case,
        use_beta_mesher=use_beta_mesher,
        bypass_length_scale_warning=True,
    )
    print(f"Project or draft id: {project_id}")


if __name__ == "__main__":
    main()
