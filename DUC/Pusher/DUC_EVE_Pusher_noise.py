from __future__ import annotations

import json
import os
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

os.environ["FLOW360_DISABLE_SYSTEM_CERTS"] = "1"

import flow360 as fl
from flow360 import u
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import CaseInterfaceV2
from flow360.component.simulation.folder import ROOT_FOLDER
from flow360.environment import current_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from flow360_utils.flow360_tools import cgns_surfaces_inside_rotation_region


SURFACE_MESH_FILE = Path(__file__).resolve().with_name("Pusher.cgns")
AEROACOUSTIC_SOURCE_FILE = Path(__file__).resolve().with_name(
    "Q40LT_step4_simulation.json"
)

@dataclass(frozen=True)
class AxisymmetricVolumeRefinementSpec:
    name: str
    axial_min_offset: float
    axial_max_offset: float
    upstream_radius_ratio: float
    spacing: float


@dataclass(frozen=True)
class BladeVortexBoxRefinementSpec:
    name: str
    axial_offset: float
    span_axis_index: int
    length_radius_ratio: float
    tangential_width_radius_ratio: float
    height: float | None  # Metres; set explicitly for each refinement.
    spacing: float


@dataclass(frozen=True)
class StepSettings:
    step_number: int
    parent_case_id: str | None
    time_steps_per_revolution: int
    num_revolutions: int
    max_pseudo_steps: int
    numerical_dissipation_factor: float
    rpm: float = 2186.0
    rotation_direction: int = 1
    wall_roughness_height: float = 1.0e-5
    navier_stokes_relative_tolerance: float = 0.0175
    navier_stokes_order_of_accuracy: int = 2
    navier_stokes_linear_solver_max_iterations: int = 25
    low_mach_preconditioner: bool = True
    adaptive_cfl_min: float = 0.1
    adaptive_cfl_max: float = 1.0e6
    adaptive_cfl_max_relative_change: float = 50.0
    adaptive_cfl_convergence_limiting_factor: float = 1.0
    include_aeroacoustic_output: bool = False
    aeroacoustic_solver_start_time_s: float = 0.0
    aeroacoustic_force_clean_start: bool = False
    make_render_output: bool = False

    @property
    def physical_steps(self) -> int:
        return self.num_revolutions * self.time_steps_per_revolution


# Change this one value to select the active simulation stage.
step_selector = 3

final_timesteps_per_rev = 3660

# Parent case IDs are inherited from the original POC2 setup; update each
# entry to the preceding Pusher case before using that stage to fork a run.
STEP_SETTINGS: dict[int, StepSettings] = {
    0: StepSettings(
        step_number=0,
        parent_case_id=None,
        time_steps_per_revolution=int(final_timesteps_per_rev * 12/25 * 1/2 * 6/16 * 1/6),
        num_revolutions=5,
        max_pseudo_steps=35,
        numerical_dissipation_factor=1.0,
        navier_stokes_relative_tolerance=0.01,
        navier_stokes_order_of_accuracy=1,
        low_mach_preconditioner=False,
    ),
    1: StepSettings(
        step_number=1,
        parent_case_id="case-62098822-d756-4abe-b1f3-d3fa440bb775",
        time_steps_per_revolution=int(final_timesteps_per_rev * 12/25 * 1/2 * 6/16),
        num_revolutions=5,
        max_pseudo_steps=15,
        numerical_dissipation_factor=0.7,
    ),
    2: StepSettings(
        step_number=2,
        parent_case_id="case-b23e2038-baf4-4a72-a52a-429faf23f56f",
        time_steps_per_revolution=int(final_timesteps_per_rev * 12/25 * 1/2),
        num_revolutions=2,
        max_pseudo_steps=12,
        numerical_dissipation_factor=0.5,
        include_aeroacoustic_output=True,
    ),
    3: StepSettings(
        step_number=3,
        parent_case_id="case-259bfe57-12d2-401f-866e-5dea5dffcf78",
        time_steps_per_revolution=int(final_timesteps_per_rev * 12/25),
        num_revolutions=2,
        max_pseudo_steps=8,
        numerical_dissipation_factor=0.2,
        adaptive_cfl_max=1.0e2,
        adaptive_cfl_convergence_limiting_factor=0.5,
        include_aeroacoustic_output=True,
        make_render_output = True,
    ),
    4: StepSettings(
        step_number=4,
        parent_case_id="",
        time_steps_per_revolution=final_timesteps_per_rev,
        num_revolutions=3,
        max_pseudo_steps=10,
        numerical_dissipation_factor=0.2,
        wall_roughness_height=0.0,
        navier_stokes_linear_solver_max_iterations=20,
        adaptive_cfl_max=1.0e2,
        adaptive_cfl_convergence_limiting_factor=0.5,
        include_aeroacoustic_output=True,
        aeroacoustic_force_clean_start=True,
        make_render_output = True,
    ),
}


def select_step(selector: int) -> StepSettings:
    try:
        return STEP_SETTINGS[selector]
    except KeyError as exc:
        raise ValueError(f"step_selector must be one of {sorted(STEP_SETTINGS)}; got {selector}") from exc


@dataclass(frozen=True)
class CaseSetup:
    """Constants copied from the old POC2x2 Flow360 JSON setup."""

    name: str = "DUC EVE Pusher"
    surface_mesh_length_unit: str = "mm"
    flow360_folder_path: tuple[str, ...] = ("DUC", "Pusher")

    propeller_radius: float = 2.120/2

    moment_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moment_length: tuple[float, float, float] = (propeller_radius, propeller_radius, propeller_radius)

    alpha_deg: float = 3.5
    beta_deg: float = 0.0
    true_airspeed_knots: float = 91.6
    altitude_ft: float = 0.0
    temperature_offset_deg_c: float = 10.0

    #rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    #rotation_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    rotation_center: tuple[float, float, float] = (0.36138462, 0.0, 0.0)

    step: StepSettings = field(default_factory=lambda: select_step(step_selector))

    include_aeroacoustic_observer_group_1: bool = False
    include_aeroacoustic_observer_group_2: bool = True
    aeroacoustic_centerline_arc_radius_ft: float = 1000.0
    aeroacoustic_centerline_arc_angles_deg: tuple[float, ...] = (20.0, 45.0, 90.0, 135.0, 160.0)
    # Lower centerline semicircle. Angles start along rotation_axis (upstream).
    aeroacoustic_centerline_arc_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


    # Volume-mesh settings for the Gmsh surface mesh.
    # Dimensionless radial ratios from the original 2.8 m propeller setup.
    rotation_volume_radius_ratio: float = 1.0714285714285714
    rotation_volume_height: float = 0.931
    octree_base_spacing: float = 0.00295898 * 1.234
    rotation_volume_spacing: float = octree_base_spacing * 2
    farfield_relative_size: float = 50
    boundary_layer_first_layer_thickness: float = 2.62e-5
    boundary_layer_growth_rate: float = 1.16
    tip_wake_inner_radius_ratio: float = 0.7857142857142857
    tip_wake_spacing: float = octree_base_spacing
    blade_vortex_spacing: float = octree_base_spacing
    # Ideal far-wake radius ratio for 4914 N at 91.6 kn, from the notebook.
    far_wake_radius_ratio: float = 0.9209301158839447

    def __post_init__(self) -> None:
        if type(self.step.rotation_direction) is not int or self.step.rotation_direction not in (-1, 1):
            raise ValueError("rotation_direction must be +1 or -1")
        if not 0.0 < self.far_wake_radius_ratio <= 1.0:
            raise ValueError("far_wake_radius_ratio must be in (0, 1]")

    @property
    def ref_area(self) -> float:
        return math.pi * self.propeller_radius**2

    @property
    def rotation_volume_radius(self) -> float:
        return self.rotation_volume_radius_ratio * self.propeller_radius

    @property
    def tip_wake_inner_radius(self) -> float:
        return self.tip_wake_inner_radius_ratio * self.propeller_radius

    @property
    def omega_rad_s(self) -> float:
        return self.step.rotation_direction * 2.0 * math.pi * self.step.rpm / 60.0

    @property
    def tip_speed_m_s(self) -> float:
        return abs(self.omega_rad_s) * self.propeller_radius

    @property
    def freestream_speed_m_s(self) -> float:
        return self.true_airspeed_knots * (1852.0 / 3600.0)

    @property
    def time_step_size_s(self) -> float:
        return 60.0 / (self.step.rpm * self.step.time_steps_per_revolution)


CONFIG = CaseSetup()


# Axial offsets are measured from the rotation center along the upstream
# freestream axis (opposite the Flow360 freestream velocity direction).
# The minimum offset is downstream; its radius is scaled by far_wake_radius_ratio.
# Historic refinement names are retained even though every body is now solid.
VOLUME_AXISYMMETRIC_REFINEMENTS: tuple[AxisymmetricVolumeRefinementSpec, ...] = (
    AxisymmetricVolumeRefinementSpec(
        name="intermediate_cylinder",
        axial_min_offset=-1.1903105,
        axial_max_offset=0.4210155,
        upstream_radius_ratio=1.2032642857142857,
        spacing=CONFIG.octree_base_spacing * 2**3,
    ),
    AxisymmetricVolumeRefinementSpec(
        name="intermediate_annulus",
        axial_min_offset=-6.2806475,
        axial_max_offset=0.2115625,
        upstream_radius_ratio=0.9 * 1.3602142857142858,
        spacing=CONFIG.octree_base_spacing * 2**3,
    ),
    AxisymmetricVolumeRefinementSpec(
        name="coarse_cylinder",
        axial_min_offset=-0.7508595,
        axial_max_offset=0.8604685,
        upstream_radius_ratio=1.3602142857142858,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    AxisymmetricVolumeRefinementSpec(
        name="short_coarse_annulus",
        axial_min_offset=-6.6834775,
        axial_max_offset=0.7139845,
        upstream_radius_ratio=0.9 * 1.4648428571428572,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    AxisymmetricVolumeRefinementSpec(
        name="coarse_annulus",
        axial_min_offset=-8.4412875,
        axial_max_offset=0.4210155,
        upstream_radius_ratio=0.9 * 1.6217928571428573,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    AxisymmetricVolumeRefinementSpec(
        name="outer_short_wide_cylinder",
        axial_min_offset=-10.8484375,
        axial_max_offset=1.1527425,
        upstream_radius_ratio=1.8928571428571428,
        spacing=CONFIG.octree_base_spacing * 2**5,
    ),
    AxisymmetricVolumeRefinementSpec(
        name="outermost_cylinder",
        axial_min_offset=-9.2469575,
        axial_max_offset=1.5928925,
        upstream_radius_ratio=2.185714285714286,
        spacing=CONFIG.octree_base_spacing * 2**5,
    ),
)


# Span and tangential widths retain the ratios from POC2's 1.4 m blade radius.
BLADE_VORTEX_REFINEMENTS: tuple[BladeVortexBoxRefinementSpec, ...] = (
    BladeVortexBoxRefinementSpec(
        name="Blade1+3Refinement",
        axial_offset=0.36138462,
        span_axis_index=0,
        length_radius_ratio=2.97 / 1.4,
        tangential_width_radius_ratio=0.5 / 1.4,
        height=0.15,
        spacing=CONFIG.blade_vortex_spacing,
    ),
    BladeVortexBoxRefinementSpec(
        name="Blade2+4Refinement",
        axial_offset=0.36138462,
        span_axis_index=1,
        length_radius_ratio=2.97 / 1.4,
        tangential_width_radius_ratio=0.5 / 1.4,
        height=0.15,
        spacing=CONFIG.blade_vortex_spacing,
    ),
)


BLADE_IDS = (1, 2, 3, 4)


SURFACE_MESH_WALL_PREFIXES = {
    "hub": ["hub_"],
    **{f"blade{blade_id}": [f"blade{blade_id}"] for blade_id in BLADE_IDS},
}

def _available_surface_group_names(geometry) -> list[str]:
    return sorted(getattr(geometry, "boundary_names", []))


def _wall_entities(geometry) -> dict[str, list]:
    boundary_names = getattr(geometry, "boundary_names", [])
    wall_entities = {
        wall_name: [
            geometry[boundary_name]
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
        assigned_names = {
            entity.name for entities in wall_entities.values() for entity in entities
        }
        other_walls = [
            geometry[name] for name in boundary_names
            if name not in assigned_names
            and "farfield" not in name.lower()
            and "interface" not in name.lower()
        ]
        if other_walls:
            wall_entities["other"] = other_walls
        return wall_entities

    wall_boundary_names = [
        boundary_name
        for boundary_name in boundary_names
        if "farfield" not in boundary_name.lower()
        and "interface" not in boundary_name.lower()
    ]
    if not wall_boundary_names:
        wall_boundary_names = list(boundary_names)
    return {"wall": [geometry[name] for name in wall_boundary_names]}


def _all_walls(wall_entities: dict[str, list]):
    walls = []
    for entities in wall_entities.values():
        walls.extend(entities)
    return walls


def _project_root_type_name(project) -> str | None:
    root_item_type = getattr(getattr(project, "metadata", None), "root_item_type", None)
    return getattr(root_item_type, "name", None)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotation_frame(cfg: CaseSetup) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    magnitude = math.sqrt(sum(component**2 for component in cfg.rotation_axis))
    if magnitude == 0.0:
        raise ValueError("rotation_axis must be nonzero")
    axis = tuple(component / magnitude for component in cfg.rotation_axis)
    # Keep the original XY box orientation for a +Z propeller axis.
    reference = (0.0, 1.0, 0.0) if abs(axis[2]) > 0.9 else (0.0, 0.0, 1.0)
    radial_u = _cross(reference, axis)
    radial_magnitude = math.sqrt(sum(component**2 for component in radial_u))
    radial_u = tuple(component / radial_magnitude for component in radial_u)
    radial_v = _cross(axis, radial_u)
    return axis, radial_u, radial_v


def _freestream_upstream_axis(cfg: CaseSetup) -> tuple[float, float, float]:
    alpha = math.radians(cfg.alpha_deg)
    beta = math.radians(cfg.beta_deg)
    # Flow360 freestream: (cos(beta)*cos(alpha), -sin(beta), cos(beta)*sin(alpha)).
    return (
        -math.cos(beta) * math.cos(alpha),
        math.sin(beta),
        -math.cos(beta) * math.sin(alpha),
    )


def _axis_point(
    cfg: CaseSetup, axis: tuple[float, float, float], axial_offset: float
) -> tuple[float, float, float]:
    return tuple(
        center + axial_offset * direction
        for center, direction in zip(cfg.rotation_center, axis)
    )


def _rotation_volume(cfg: CaseSetup):
    axis, _, _ = _rotation_frame(cfg)
    return fl.Cylinder(
        name="zone_r1",
        center=cfg.rotation_center * u.m,
        axis=axis,
        height=cfg.rotation_volume_height * u.m,
        outer_radius=cfg.rotation_volume_radius * u.m,
    )


def _make_project_from_local_surface_mesh(
    surface_mesh_file: Path, cfg: CaseSetup, flow360_folder
):
    surface_mesh_file = Path(surface_mesh_file)
    if not surface_mesh_file.is_file():
        raise FileNotFoundError(f"Could not find Gmsh CGNS surface mesh: {surface_mesh_file}")
    if surface_mesh_file.suffix.lower() != ".cgns":
        raise ValueError(f"Expected a CGNS surface mesh, got {surface_mesh_file}")
    return fl.Project.from_surface_mesh(
        str(surface_mesh_file),
        name=cfg.name,
        folder=flow360_folder,
        length_unit=cfg.surface_mesh_length_unit,
        run_async=False,
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


def _make_axisymmetric_volume_refinements(cfg: CaseSetup):
    axis = _freestream_upstream_axis(cfg)
    refinements = []
    for spec in VOLUME_AXISYMMETRIC_REFINEMENTS:
        upstream_radius = spec.upstream_radius_ratio * cfg.propeller_radius
        downstream_radius = upstream_radius * cfg.far_wake_radius_ratio
        volume = fl.AxisymmetricBody(
            name=f"{spec.name}_volume",
            center=cfg.rotation_center * u.m,
            axis=axis,
            profile_curve=[
                (spec.axial_min_offset, 0.0) * u.m,
                (spec.axial_min_offset, downstream_radius) * u.m,
                (spec.axial_max_offset, upstream_radius) * u.m,
                (spec.axial_max_offset, 0.0) * u.m,
            ],
        )
        refinements.append(
            fl.UniformRefinement(
                name=f"{spec.name}_refinement",
                entities=[volume],
                spacing=spec.spacing * u.m,
            )
        )
    return refinements


def _make_tip_wake_refinement(cfg: CaseSetup):
    axis, _, _ = _rotation_frame(cfg)
    cylinder = fl.Cylinder(
        name="tip_wake_annulus_volume",
        center=cfg.rotation_center * u.m,
        axis=axis,
        height=cfg.rotation_volume_height * u.m,
        inner_radius=cfg.tip_wake_inner_radius * u.m,
        outer_radius=cfg.rotation_volume_radius * u.m,
    )
    return fl.UniformRefinement(
        name="tip_wake_annulus_refinement",
        entities=[cylinder],
        spacing=cfg.tip_wake_spacing * u.m,
    )


def _make_blade_vortex_box_refinements(cfg: CaseSetup):
    axis, radial_u, radial_v = _rotation_frame(cfg)
    radial_axes = (radial_u, radial_v)
    refinements = []
    for spec in BLADE_VORTEX_REFINEMENTS:
        if spec.height is None or spec.height <= 0.0:
            raise ValueError(f"Set a positive height in metres for {spec.name} before volume meshing.")
        box = fl.Box.from_principal_axes(
            name=f"{spec.name}_volume",
            center=_axis_point(cfg, axis, spec.axial_offset) * u.m,
            axes=(radial_axes[spec.span_axis_index], radial_axes[1 - spec.span_axis_index]),
            size=(
                spec.length_radius_ratio * cfg.propeller_radius,
                spec.tangential_width_radius_ratio * cfg.propeller_radius,
                spec.height,
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
                *_make_axisymmetric_volume_refinements(cfg),
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
        mach=cfg.freestream_speed_m_s / thermal_state.speed_of_sound.to("m/s").value,
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
            relative_tolerance=cfg.step.navier_stokes_relative_tolerance,
            order_of_accuracy=cfg.step.navier_stokes_order_of_accuracy,
            equation_evaluation_frequency=1,
            linear_solver=fl.LinearSolver(max_iterations=cfg.step.navier_stokes_linear_solver_max_iterations),
            CFL_multiplier=1.0,
            kappa_MUSCL=0.33,
            limit_velocity=False,
            limit_pressure_density=False,
            update_jacobian_frequency=1,
            max_force_jac_update_physical_steps=0,
            riemann_solver=fl.RoeFlux(
                numerical_dissipation_factor=cfg.step.numerical_dissipation_factor,
                low_mach_preconditioner=cfg.step.low_mach_preconditioner,
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
        roughness_height=cfg.step.wall_roughness_height * u.m,
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


def _make_centerline_arc_observers(cfg: CaseSetup):
    """Observers about the hub, from the upstream axis toward the lower arc."""
    axis, _, _ = _rotation_frame(cfg)
    direction = cfg.aeroacoustic_centerline_arc_direction
    projection = sum(a * b for a, b in zip(axis, direction))
    radial = tuple(d - projection * a for d, a in zip(direction, axis))
    magnitude = math.sqrt(sum(component**2 for component in radial))
    if magnitude <= 1e-12:
        raise ValueError("The observer arc direction must not be parallel to rotation_axis.")
    if cfg.aeroacoustic_centerline_arc_radius_ft <= 0:
        raise ValueError("The observer arc radius must be positive.")
    radial = tuple(component / magnitude for component in radial)
    radius_m = float((cfg.aeroacoustic_centerline_arc_radius_ft * u.ft).to("m").value)
    observers = []
    for angle_deg in cfg.aeroacoustic_centerline_arc_angles_deg:
        if not 0 <= angle_deg <= 180:
            raise ValueError("Centerline emission angles must lie between 0 and 180 degrees.")
        angle = math.radians(angle_deg)
        position = tuple(
            center + radius_m * (math.cos(angle) * axial + math.sin(angle) * transverse)
            for center, axial, transverse in zip(cfg.rotation_center, axis, radial)
        )
        observers.append(fl.Observer(position=position * u.m, group_name="centerline_arc"))
    return observers


def _make_aeroacoustic_output(
    cfg: CaseSetup,
    source_file: Path = AEROACOUSTIC_SOURCE_FILE,
):
    with source_file.open(encoding="utf-8") as file:
        source = json.load(file)

    observers = _make_centerline_arc_observers(cfg)

    if "aeroacousticOutput" in source:
        aeroacoustic_output = source["aeroacousticOutput"]
        # Legacy files have no group IDs; treat their observer list as group 1.
        observers.extend([
            fl.Observer(
                position=position * u.m,
                group_name=f"observer_{index:03d}",
            )
            for index, position in enumerate(aeroacoustic_output["observers"], start=1)
        ] if cfg.include_aeroacoustic_observer_group_1 else [])
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
        included_groups = set()
        if cfg.include_aeroacoustic_observer_group_1:
            included_groups.add("1")
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
                    position=(position["value"] * position_unit).to("m"),
                    group_name=observer["group_name"],
                )
            )

    return fl.AeroAcousticOutput(
        name="AeroAcousticOutput",
        observers=observers,
        write_per_surface_output=False,
        patch_type="solid",
        aeroacoustic_solver_start_time=cfg.step.aeroacoustic_solver_start_time_s * u.s,
        force_clean_start=cfg.step.aeroacoustic_force_clean_start,
    )


def _make_render_output(wall_surfaces: list, cfg: CaseSetup):
    # Target 2 seconds per revolution when played at 48 fps (96 frames/rev).
    # Flow360 requires a whole physical-step interval, so timing is approximate.
    render_frequency = max(1, round(cfg.step.time_steps_per_revolution / (2.0 * 48)))
    structural_surfaces = [
        surface for surface in wall_surfaces
        if not getattr(surface, "private_attribute_is_interface", False)
        and not any(
            excluded in str(getattr(surface, "full_name", surface.name)).lower()
            for excluded in ("interface", "inverted", "farfield")
        )
    ]
    if not structural_surfaces:
        raise ValueError("Render output requires at least one structural surface.")

    colormap = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]
    render_output_1_camera = fl.render_config.Camera(
        position=(-0.617042045143, -0.732898677141, 0.286563507047),
        look_at=(1163.34074092, -1877.94150843, 1146.90520487),
        pan_target=(1163.34074092, -1877.94150843, 1146.90520487),
        up=(0.183698898514, 0.219942098007, 0.958060639108),
        dimension=10499.0268044,
        dimension_dir="width",
    )
    return fl.RenderOutput(
        name="Render output 1",
        frequency=render_frequency,
        frequency_offset=0,
        camera=render_output_1_camera,
        groups=[
            fl.RenderOutputGroup(
                surfaces=structural_surfaces,
                material=fl.render_config.FieldMaterial(
                    output_field="CfVec", min=0.0, max=0.002, colormap=colormap,
                ),
            ),
            fl.RenderOutputGroup(
                isosurfaces=[
                    fl.Isosurface(name="q_criterion", field="qcriterion", iso_value=2.50e-8),
                ],
                material=fl.render_config.FieldMaterial(
                    output_field="Mach", min=0.0, max=0.5, colormap=colormap,
                ),
            ),
        ],
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
                "CfVec",
                "yPlus",
            ],
            write_single_file=True,
        ),
    ]

    if cfg.step.make_render_output:
        outputs.append(_make_render_output(wall_surfaces, cfg))

    if cfg.step.include_aeroacoustic_output:
        outputs.append(_make_aeroacoustic_output(cfg))

    return outputs


def build_params(
    project,
    cfg: CaseSetup = CONFIG,
    *,
    include_volume_refinements: bool = True,
    surface_mesh_file: Path = SURFACE_MESH_FILE,
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
        enclosed_surfaces = cgns_surfaces_inside_rotation_region(
            surface_mesh_file,
            wall_surfaces,
            rotation_volume,
            mesh_length_unit=cfg.surface_mesh_length_unit,
        )
        farfield, mesh_params = _make_meshing_params(
            rotation_volume,
            cfg,
            include_volume_refinements=include_volume_refinements,
            enclosed_entities=enclosed_surfaces,
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
                max_pseudo_steps=cfg.step.max_pseudo_steps,
                steps=cfg.step.physical_steps,
                step_size=cfg.time_step_size_s * u.s,
                CFL=fl.AdaptiveCFL(
                    min=cfg.step.adaptive_cfl_min,
                    max=cfg.step.adaptive_cfl_max,
                    max_relative_change=cfg.step.adaptive_cfl_max_relative_change,
                    convergence_limiting_factor=cfg.step.adaptive_cfl_convergence_limiting_factor,
                ),
                order_of_accuracy=2,
            ),
            "outputs": _make_outputs(wall_surfaces, cfg),
        }
        if mesh_params is not None:
            params_kwargs["meshing"] = mesh_params
        return fl.SimulationParams(**params_kwargs)


def _fork_from_case(cfg: CaseSetup):
    if not cfg.step.parent_case_id:
        return None
    parent_case_id = _parent_case_full_id(cfg)
    case = fl.Case(parent_case_id.removeprefix("case-"))
    # The SDK Case constructor validates bare UUIDs, but the v2 draft API
    # expects the full asset id when selecting a Case as the draft source.
    case._id = parent_case_id
    return case


def _case_run_name(cfg: CaseSetup, fallback: str) -> str:
    if cfg.step.parent_case_id:
        return f"DUC_EVE_Pusher_noise_step{cfg.step.step_number}"
    return fallback


def _parent_case_full_id(cfg: CaseSetup) -> str:
    if not cfg.step.parent_case_id:
        raise ValueError("Cannot resolve a parent case id without cfg.step.parent_case_id.")
    parent_case_id = cfg.step.parent_case_id.strip()
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
    surface_mesh_file: Path = SURFACE_MESH_FILE,
    cfg: CaseSetup = CONFIG,
    flow360_folder=None,
    submit_draft_only: bool = True,
    generate_volume_mesh: bool = True,
    run_case: bool = False,
    use_beta_mesher: bool = True,
    bypass_length_scale_warning: bool = True,
    results_path: str | None = None,
):
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
        surface_mesh_file=surface_mesh_file,
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
                name=_case_run_name(cfg, "DUC_EVE_Pusher_noise_setup"),
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
            name=_case_run_name(cfg, "DUC_EVE_Pusher_noise_case"),
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
    # Continue from the existing Flow360 case; meshing is handled separately.
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
        f"  Freestream speed: {CONFIG.freestream_speed_m_s:.3f} m/s "
        f"({CONFIG.true_airspeed_knots:.1f} kn)\n"
        f"  Freestream Mach: "
        f"{CONFIG.freestream_speed_m_s / thermal_state.speed_of_sound.to('m/s').value:.6f}\n"
        f"  Reference Mach: {reference_mach:.6f}\n"
        f"  Reference velocity: {reference_velocity.to('m/s').value:.3f} m/s\n"
        f"  Dynamic pressure: {dynamic_pressure.to('Pa').value:.2f} Pa "
        f"({dynamic_pressure.to('kPa').value:.3f} kPa)\n"
        f"  Selected step: {CONFIG.step.step_number}\n"
        f"  RPM: {CONFIG.step.rpm:.3f}\n"
        f"  Rotation direction: {CONFIG.step.rotation_direction:+d}\n"
        f"  Omega: {CONFIG.omega_rad_s:.6f} rad/s\n"
        f"  Physical steps: {CONFIG.step.physical_steps}\n"
        f"  Time step size: {CONFIG.time_step_size_s:.9f} s\n"
        f"  Parent case id: {CONFIG.step.parent_case_id or 'None'}\n"

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
