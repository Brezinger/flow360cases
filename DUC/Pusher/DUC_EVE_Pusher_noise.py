from __future__ import annotations

import json
import os
import math
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


GEOMETRY_STEP_FILE = None
GEOMETRY_CSM_FILE = None
GEOMETRY_FILE = None
AEROACOUSTIC_SOURCE_FILE = Path(__file__).resolve().with_name(
    "Q40LT_step4_simulation.json"
)

@dataclass(frozen=True)
class VolumeCylinderRefinementSpec:
    name: str
    axial_min_offset: float
    axial_max_offset: float
    inner_radius_ratio: float
    outer_radius_ratio: float
    spacing: float


@dataclass(frozen=True)
class BladeVortexBoxRefinementSpec:
    name: str
    axial_offset: float
    span_axis_index: int
    length: float
    tangential_width: float
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
    rotation_direction: int = -1
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

    @property
    def physical_steps(self) -> int:
        return self.num_revolutions * self.time_steps_per_revolution


# Change this one value to select the active simulation stage.
step_selector = 0

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
        parent_case_id="case-8ab2e55e-63c6-4167-9679-1b77eac32325",
        time_steps_per_revolution=int(final_timesteps_per_rev * 12/25 * 1/2 * 6/16),
        num_revolutions=5,
        max_pseudo_steps=15,
        numerical_dissipation_factor=0.7,
    ),
    2: StepSettings(
        step_number=2,
        parent_case_id="case-2cd84f26-a687-4209-8f74-7f2fc69c1b5a",
        time_steps_per_revolution=int(final_timesteps_per_rev * 12/25 * 1/2),
        num_revolutions=2,
        max_pseudo_steps=12,
        numerical_dissipation_factor=0.5,
    ),
    3: StepSettings(
        step_number=3,
        parent_case_id="case-beb13e25-7f8b-4891-9f9d-53f9d8462d9c",
        time_steps_per_revolution=int(final_timesteps_per_rev * 12/25),
        num_revolutions=2,
        max_pseudo_steps=8,
        numerical_dissipation_factor=0.2,
        adaptive_cfl_max=1.0e2,
        adaptive_cfl_convergence_limiting_factor=0.5,
        include_aeroacoustic_output=True,
    ),
    4: StepSettings(
        step_number=4,
        parent_case_id="case-4fcc3b53-bb3e-43fc-97c1-7047b13067e3",
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
    geometry_length_unit: str = "mm"
    flow360_folder_path: tuple[str, ...] = ("DUC", "POC2")

    propeller_radius: float = 2.120/2

    moment_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moment_length: tuple[float, float, float] = (propeller_radius, propeller_radius, propeller_radius)

    alpha_deg: float = 3.5
    beta_deg: float = 0.0
    altitude_ft: float = 0.0
    temperature_offset_deg_c: float = 10.0

    #rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    #rotation_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_center: tuple[float, float, float] = (0.0, 0.0, 0.0184375)

    step: StepSettings = field(default_factory=lambda: select_step(step_selector))

    include_aeroacoustic_observer_group_2: bool = False


    # Mesh-generation placeholders. The old case JSON was produced from a
    # supplied volume mesh, so it does not define these. Keep these conservative
    # until the CAD import tags and generated mesh are checked.
    # Dimensionless radial ratios from the original 2.8 m propeller setup.
    rotation_volume_radius_ratio: float = 1.0714285714285714
    rotation_volume_height: float = 0.423125
    octree_base_spacing: float = 0.00295898 * 1.234
    rotation_volume_spacing: float = octree_base_spacing * 2
    farfield_relative_size: float = 50
    surface_max_edge_length: float = 9.88e-3
    curvature_resolution_angle_deg: float = 5.0
    boundary_layer_first_layer_thickness: float = 3.3e-5 * 1.5
    boundary_layer_growth_rate: float = 1.16
    surface_edge_growth_rate: float = 1.12
    blade_inner_max_edge_length: float = 9.88e-3
    hub_max_edge_length: float = 5.02e-3
    blade_main_max_edge_length: float = 3.02e-3
    blade_te_max_edge_length: float = 1.42e-3
    trailing_edge_normal_spacing: float = 8.9e-5
    hub_edge_spacing: float = 0.992e-3
    tip_wake_inner_radius_ratio: float = 0.7857142857142857
    tip_wake_axial_lower_offset: float = -0.1184375
    tip_wake_spacing: float = octree_base_spacing
    blade_vortex_spacing: float = octree_base_spacing

    def __post_init__(self) -> None:
        if type(self.step.rotation_direction) is not int or self.step.rotation_direction not in (-1, 1):
            raise ValueError("rotation_direction must be +1 or -1")

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
    def time_step_size_s(self) -> float:
        return 60.0 / (self.step.rpm * self.step.time_steps_per_revolution)


CONFIG = CaseSetup()


# Axial offsets are measured from the rotation center. These values preserve
# the original positions when the rotation center is (0, 0, 0.0184375) m.
# Cylinder radial dimensions are multiples of the blade radius.
VOLUME_CYLINDER_REFINEMENTS: tuple[VolumeCylinderRefinementSpec, ...] = (
    VolumeCylinderRefinementSpec(
        name="intermediate_cylinder",
        axial_min_offset=-0.3846475,
        axial_max_offset=0.4210155,
        inner_radius_ratio=0.0,
        outer_radius_ratio=1.2032642857142857,
        spacing=CONFIG.octree_base_spacing * 2**3,
    ),
    VolumeCylinderRefinementSpec(
        name="intermediate_annulus",
        axial_min_offset=-6.2806475,
        axial_max_offset=0.2115625,
        inner_radius_ratio=0.7585785714285714,
        outer_radius_ratio=1.3602142857142858,
        spacing=CONFIG.octree_base_spacing * 2**3,
    ),
    VolumeCylinderRefinementSpec(
        name="coarse_cylinder",
        axial_min_offset=-0.7508595,
        axial_max_offset=0.8604685,
        inner_radius_ratio=0.0,
        outer_radius_ratio=1.3602142857142858,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    VolumeCylinderRefinementSpec(
        name="short_coarse_annulus",
        axial_min_offset=-6.6834775,
        axial_max_offset=0.7139845,
        inner_radius_ratio=0.4185271428571429,
        outer_radius_ratio=1.4648428571428572,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    VolumeCylinderRefinementSpec(
        name="coarse_annulus",
        axial_min_offset=-8.4412875,
        axial_max_offset=0.4210155,
        inner_radius_ratio=0.5231585714285714,
        outer_radius_ratio=1.6217928571428573,
        spacing=CONFIG.octree_base_spacing * 2**4,
    ),
    VolumeCylinderRefinementSpec(
        name="outer_short_wide_cylinder",
        axial_min_offset=-10.8484375,
        axial_max_offset=1.1527425,
        inner_radius_ratio=0.0,
        outer_radius_ratio=1.8928571428571428,
        spacing=CONFIG.octree_base_spacing * 2**5,
    ),
    VolumeCylinderRefinementSpec(
        name="outermost_cylinder",
        axial_min_offset=-9.2469575,
        axial_max_offset=1.5928925,
        inner_radius_ratio=0.0,
        outer_radius_ratio=2.185714285714286,
        spacing=CONFIG.octree_base_spacing * 2**5,
    ),
)


BLADE_VORTEX_REFINEMENTS: tuple[BladeVortexBoxRefinementSpec, ...] = (
    BladeVortexBoxRefinementSpec(
        name="Blade1+3Refinement",
        axial_offset=0.1115625,
        span_axis_index=0,
        length=2.97,
        tangential_width=0.34,
        spacing=CONFIG.blade_vortex_spacing,
    ),
    BladeVortexBoxRefinementSpec(
        name="Blade2+4Refinement",
        axial_offset=-0.0154375,
        span_axis_index=1,
        length=2.97,
        tangential_width=0.34,
        spacing=CONFIG.blade_vortex_spacing,
    ),
)


BLADE_IDS = (1, 2, 3, 4)


def _blade_group_names(suffix: str) -> list[str]:
    return [f"blade{blade_id}{suffix}" for blade_id in BLADE_IDS]


# Update these after the first geometry upload if Flow360 exposes different names.
FACE_GROUP_ALIASES = {
    "bladeInner": _blade_group_names("Inner"),
    "hub": ["hub"],
    "bladeMain": _blade_group_names("main"),
    "bladeTE": _blade_group_names("TE"),
}

CAD_WALL_GROUP_ALIASES = {
    "hub": ["hub"],
    **{
        f"blade{blade_id}": [
            f"blade{blade_id}Inner",
            f"blade{blade_id}main",
            f"blade{blade_id}TE",
        ]
        for blade_id in BLADE_IDS
    },
}

SURFACE_MESH_WALL_PREFIXES = {
    "hub": ["hub_"],
    **{f"blade{blade_id}": [f"blade{blade_id}"] for blade_id in BLADE_IDS},
}

EDGE_GROUP_ALIASES = {
    "leading_edge": [
        "leadingEdges",
        "leadingEdge",
        "leading_edge",
        *_blade_group_names("LE"),
        *_blade_group_names("LeadingEdges"),
    ],
    "trailing_edge": [
        "trailingEdges",
        "trailingEdge",
        "trailing_edge",
        "bladeTrailingEdge",
        *_blade_group_names("TEEdges"),
        *_blade_group_names("TrailingEdges"),
    ],
    "hub_edges": ["hubEdges", "hub_edges"],
}

CSM_FACE_GROUPS = {
    "hub": [20, 37, 1, 56, 50, 26],
    "blade1": [18, 13, 12, 8, 9, 19, 14, 22, 6],
}


def _write_geometry_csm_file(csm_file: Path, step_file: Path) -> None:
    if not step_file.exists():
        raise FileNotFoundError(f"Could not find STEP file for CSM import: {step_file}")

    lines = [
        f"import $/{step_file.name} -1",
        "",
    ]
    for group_name, face_ids in CSM_FACE_GROUPS.items():
        variable_name = f"{group_name}Faces"
        face_id_values = "; ".join(str(face_id) for face_id in face_ids)
        lines.extend(
            [
                f"# {group_name}",
                f"dimension {variable_name} {len(face_ids)} 1",
                f"set {variable_name} \"{face_id_values}\"",
                f"patbeg i {variable_name}.size",
                f"   select face {variable_name}[i]",
                f"   attribute faceName ${group_name}",
                f"   attribute groupName ${group_name}",
                "patend",
                "",
            ]
        )
    lines.append(f"dump $/{step_file.with_suffix('.egads').name} 0 1 0")
    csm_text = "\n".join(lines) + "\n"
    csm_file.write_text(csm_text, encoding="utf-8")


def _matches_entity_type(entity, entity_type_name: str | None) -> bool:
    if entity_type_name is None:
        return True
    actual_type_name = getattr(entity, "private_attribute_entity_type_name", None)
    return actual_type_name is None or actual_type_name == entity_type_name


def _entities_by_possible_names(
    geometry,
    possible_names: list[str],
    *,
    entity_type_name: str | None = None,
):
    entities = []
    for name in possible_names:
        try:
            matched = geometry[name]
        except Exception:
            registry = getattr(geometry, "internal_registry", None)
            if registry is None:
                continue
            matched = registry.find_by_naming_pattern(name)
            if not matched:
                continue
        if isinstance(matched, list):
            entities.extend(entity for entity in matched if _matches_entity_type(entity, entity_type_name))
        else:
            if _matches_entity_type(matched, entity_type_name):
                entities.append(matched)
    return entities


def _available_surface_group_names(geometry) -> list[str]:
    boundary_names = getattr(geometry, "boundary_names", None)
    if boundary_names is not None:
        return sorted(boundary_names)
    return _available_entity_group_names(geometry, "Surface")


def _available_edge_group_names(geometry) -> list[str]:
    return _available_entity_group_names(geometry, "Edge")


def _available_entity_group_names(geometry, entity_type_name: str) -> list[str]:
    registry = getattr(geometry, "internal_registry", None)
    if registry is None:
        return []
    names = []
    for entity_list in registry.internal_registry.values():
        for entity in entity_list:
            if getattr(entity, "private_attribute_entity_type_name", None) == entity_type_name:
                names.append(getattr(entity, "name", repr(entity)))
    return sorted(set(names))


def _wall_entities(geometry) -> dict[str, list]:
    boundary_names = getattr(geometry, "boundary_names", None)
    if boundary_names is not None:
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
            return wall_entities

        wall_boundary_names = [
            boundary_name
            for boundary_name in boundary_names
            if "farfield" not in boundary_name.lower() and "interface" not in boundary_name.lower()
        ]
        if not wall_boundary_names:
            wall_boundary_names = list(boundary_names)
        return {"wall": [geometry[boundary_name] for boundary_name in wall_boundary_names]}

    return {
        wall_name: _entities_by_possible_names(
            geometry,
            possible_names,
            entity_type_name="Surface",
        )
        for wall_name, possible_names in CAD_WALL_GROUP_ALIASES.items()
    }


def _surface_refinement_entities(geometry) -> dict[str, list]:
    return {
        zone_name: _entities_by_possible_names(
            geometry,
            possible_names,
            entity_type_name="Surface",
        )
        for zone_name, possible_names in FACE_GROUP_ALIASES.items()
    }


def _edge_entities(geometry) -> dict[str, list]:
    return {
        edge_name: _entities_by_possible_names(
            geometry,
            possible_names,
            entity_type_name="Edge",
        )
        for edge_name, possible_names in EDGE_GROUP_ALIASES.items()
    }


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


def _make_project(geometry_file: Path, cfg: CaseSetup, flow360_folder, use_beta_mesher):
    if not GEOMETRY_CSM_FILE.exists():
        raise FileNotFoundError(
            f"Could not find CSM file with faceName/edgeName tags: {GEOMETRY_CSM_FILE}"
        )
    if geometry_file.suffix.lower() == ".egads" and not geometry_file.exists():
        raise FileNotFoundError(
            f"Could not find generated EGADS geometry: {geometry_file}. "
            f"Open or run the CSM file first to create it: {GEOMETRY_CSM_FILE}"
        )

    if use_beta_mesher:
        proj_name = cfg.name + "_beta_mesher"
    else:
        proj_name = cfg.name

    project = fl.Project.from_geometry(
        str(geometry_file),
        name=proj_name,
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


def _make_surface_edge_refinements(geometry, cfg: CaseSetup):
    geometry.group_edges_by_tag("edgeName")
    edge_entities = _edge_entities(geometry)
    required_edge_groups = ("trailing_edge", "hub_edges")
    missing = [name for name in required_edge_groups if not edge_entities[name]]
    if missing:
        raise ValueError(
            f"Could not resolve edge groups {missing}. "
            f"Available edge groups: {_available_edge_group_names(geometry)}. "
            "Tag the CAD edges with edgeName/groupName='trailingEdges' and 'hubEdges' in the CSM."
        )

    return [
        fl.SurfaceEdgeRefinement(
            name="TrailingEdgeSurfaceEdgeRefinement",
            edges=edge_entities["trailing_edge"],
            method=fl.HeightBasedRefinement(value=cfg.trailing_edge_normal_spacing * u.m),
        ),
        fl.SurfaceEdgeRefinement(
            name="HubEdgesSurfaceEdgeRefinement",
            edges=edge_entities["hub_edges"],
            method=fl.HeightBasedRefinement(value=cfg.hub_edge_spacing * u.m),
        ),
    ]


def _make_surface_refinements(geometry, cfg: CaseSetup):
    geometry.group_faces_by_tag("faceName")
    face_entities = _surface_refinement_entities(geometry)
    missing = [name for name, entities in face_entities.items() if not entities]
    if missing:
        raise ValueError(
            f"Could not resolve faceName groups {missing}. "
            f"Available faceName groups: {_available_surface_group_names(geometry)}. "
            "Check the CSM faceName attributes."
        )

    zone_sizes = {
        "bladeInner": cfg.blade_inner_max_edge_length,
        "hub": cfg.hub_max_edge_length,
        "bladeMain": cfg.blade_main_max_edge_length,
        "bladeTE": cfg.blade_te_max_edge_length,
    }
    return [
        fl.SurfaceRefinement(
            name=f"{zone_name}SurfaceRefinement",
            faces=face_entities[zone_name],
            max_edge_length=max_edge_length * u.m,
        )
        for zone_name, max_edge_length in zone_sizes.items()
    ]


def _make_meshing_defaults(cfg: CaseSetup):
    return fl.MeshingDefaults(
        surface_edge_growth_rate=cfg.surface_edge_growth_rate,
        surface_max_edge_length=cfg.surface_max_edge_length * u.m,
        curvature_resolution_angle=cfg.curvature_resolution_angle_deg * u.deg,
        boundary_layer_growth_rate=cfg.boundary_layer_growth_rate,
        boundary_layer_first_layer_thickness=cfg.boundary_layer_first_layer_thickness * u.m,
        octree_spacing=fl.OctreeSpacing(base_spacing=cfg.octree_base_spacing * u.m),
    )


def _make_volume_cylinder_refinements(cfg: CaseSetup):
    axis, _, _ = _rotation_frame(cfg)
    refinements = []
    for spec in VOLUME_CYLINDER_REFINEMENTS:
        axial_center = 0.5 * (spec.axial_min_offset + spec.axial_max_offset)
        cylinder = fl.Cylinder(
            name=f"{spec.name}_volume",
            center=_axis_point(cfg, axis, axial_center) * u.m,
            axis=axis,
            height=(spec.axial_max_offset - spec.axial_min_offset) * u.m,
            inner_radius=spec.inner_radius_ratio * cfg.propeller_radius * u.m,
            outer_radius=spec.outer_radius_ratio * cfg.propeller_radius * u.m,
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
    axis, _, _ = _rotation_frame(cfg)
    axial_upper = 0.5 * cfg.rotation_volume_height
    axial_lower = cfg.tip_wake_axial_lower_offset
    cylinder = fl.Cylinder(
        name="tip_wake_annulus_volume",
        center=_axis_point(cfg, axis, 0.5 * (axial_lower + axial_upper)) * u.m,
        axis=axis,
        height=(axial_upper - axial_lower) * u.m,
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
    axial_upper = 0.5 * cfg.rotation_volume_height
    blade_vortex_axial_size = 2.0 * (axial_upper - BLADE_VORTEX_REFINEMENTS[0].axial_offset)
    if blade_vortex_axial_size <= 0.0:
        raise ValueError(
            "Blade-vortex refinement center is above the rotation-volume upper boundary. "
            f"axial_offset={BLADE_VORTEX_REFINEMENTS[0].axial_offset}, upper_offset={axial_upper}"
        )
    for spec in BLADE_VORTEX_REFINEMENTS:
        box = fl.Box.from_principal_axes(
            name=f"{spec.name}_volume",
            center=_axis_point(cfg, axis, spec.axial_offset) * u.m,
            axes=(radial_axes[spec.span_axis_index], radial_axes[1 - spec.span_axis_index]),
            size=(spec.length, spec.tangential_width, blade_vortex_axial_size) * u.m,
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
    geometry,
    cfg: CaseSetup,
    *,
    use_beta_mesher: bool,
    include_surface_refinements: bool = True,
    include_volume_refinements: bool = True,
    enclosed_entities: list | None = None,
):
    farfield = fl.AutomatedFarfield(
        name="farfield",
        relative_size=cfg.farfield_relative_size,
    )
    refinements = []
    if include_surface_refinements:
        refinements.extend(_make_surface_refinements(geometry, cfg))
    if include_volume_refinements:
        refinements.extend(
            [
                *_make_volume_cylinder_refinements(cfg),
                _make_tip_wake_refinement(cfg),
                *_make_blade_vortex_box_refinements(cfg),
            ]
        )
    if include_surface_refinements and not use_beta_mesher:
        refinements.extend(_make_surface_edge_refinements(geometry, cfg))

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
    if hasattr(geometry, "group_faces_by_tag"):
        geometry.group_faces_by_tag("faceName")
    if wall_entities is None:
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
        aeroacoustic_solver_start_time=cfg.step.aeroacoustic_solver_start_time_s * u.s,
        force_clean_start=cfg.step.aeroacoustic_force_clean_start,
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

    if cfg.step.include_aeroacoustic_output:
        outputs.append(_make_aeroacoustic_output(cfg))

    return outputs


def build_params(
    project,
    cfg: CaseSetup = CONFIG,
    *,
    use_beta_mesher: bool = False,
    include_volume_refinements: bool = True,
):
    root_type_name = _project_root_type_name(project)
    if root_type_name == "VOLUME_MESH":
        mesh_params = None
        models, wall_surfaces = _make_models_from_volume_mesh(project.volume_mesh, cfg)
    else:
        use_existing_surface_mesh = root_type_name == "SURFACE_MESH"
        geometry = project.surface_mesh if use_existing_surface_mesh else project.geometry
        if hasattr(geometry, "group_faces_by_tag"):
            geometry.group_faces_by_tag("faceName")
        wall_entities = _wall_entities(geometry)
        wall_surfaces = _all_walls(wall_entities)
        rotation_volume = _rotation_volume(cfg)
        farfield, mesh_params = _make_meshing_params(
            rotation_volume,
            geometry,
            cfg,
            use_beta_mesher=use_beta_mesher,
            include_surface_refinements=not use_existing_surface_mesh,
            include_volume_refinements=include_volume_refinements,
            enclosed_entities=wall_surfaces,
        )
        models, wall_surfaces = _make_models(geometry, farfield, rotation_volume, cfg, wall_entities)

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
    geometry_file: Path = GEOMETRY_FILE,
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
        else _make_project(geometry_file, cfg, flow360_folder, use_beta_mesher)
    )
    params = build_params(
        project,
        cfg,
        use_beta_mesher=use_beta_mesher,
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
