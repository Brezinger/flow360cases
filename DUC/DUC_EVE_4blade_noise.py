from __future__ import annotations

import json
import os
import math
import re
import struct
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import flow360 as fl
from flow360 import u
from flow360.component.simulation.folder import ROOT_FOLDER


GEOMETRY_STEP_FILE = Path(__file__).resolve().with_name(
    "EM_002_DS_12_02-EVE100_CCW_LIFTER4B_ID20025v2_tmc50p_h60wide_20260708_w_spacer.STEP"
)
GEOMETRY_CSM_FILE = GEOMETRY_STEP_FILE.with_suffix(".csm")
GEOMETRY_FILE = GEOMETRY_STEP_FILE.with_suffix(".egads")
AEROACOUSTIC_SOURCE_FILE = Path(__file__).resolve().parent / (
    "POC2x2/case-ab4d94eb-4311-4a4e-946d-b5756958c604_flow360.json"
)
LEGACY_SURFACE_MESH_ASSETS_DIR = Path(__file__).resolve().parent / "legacy_SM_assets"
DEFAULT_SURFACE_MESH_FILE = (
    LEGACY_SURFACE_MESH_ASSETS_DIR
    / "sm-ab0c68d1-f434-4ab4-839d-84573ec6df80_surfaceMesh.lb8.ugrid"
)
DEFAULT_SURFACE_MESH_LOG_FILE = (
    LEGACY_SURFACE_MESH_ASSETS_DIR
    / "sm-ab0c68d1-f434-4ab4-839d-84573ec6df80_logs_flow360_surface_mesh.user.log"
)
MAPBC_WALL_BC_CODE = 4000
UGRID_WELD_TOLERANCE = 1.0e-7


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
    length: float
    tangential_width: float
    spacing: float


@dataclass(frozen=True)
class CaseSetup:
    """Constants copied from the old POC2x2 Flow360 JSON setup."""

    name: str = "DUC_EVE_4blade_noise_from_geometry"
    geometry_length_unit: str = "mm"
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
    rotation_volume_radius: float = 1.5
    rotation_volume_height: float = 0.38625
    rotation_volume_spacing: float = 0.006
    farfield_relative_size: float = 107
    surface_max_edge_length: float = 0.00866
    curvature_resolution_angle_deg: float = 5.0
    boundary_layer_first_layer_thickness: float = 3.3e-5 * 1.5
    boundary_layer_growth_rate: float = 1.16
    surface_edge_growth_rate: float = 1.12
    blade_inner_max_edge_length: float = 8.66e-3
    hub_max_edge_length: float = 4.4e-3
    blade_main_max_edge_length: float = 2.65e-3
    blade_te_max_edge_length: float = 1.25e-3
    trailing_edge_normal_spacing: float = 8.9e-5
    hub_edge_spacing: float = 0.87e-3
    tip_wake_inner_radius: float = 1.1
    tip_wake_z_lower: float = -0.1
    tip_wake_spacing: float = 0.003
    blade_vortex_spacing: float = 0.003

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


CONFIG = CaseSetup()


VOLUME_CYLINDER_REFINEMENTS: tuple[VolumeCylinderRefinementSpec, ...] = (
    VolumeCylinderRefinementSpec(
        name="intermediate_cylinder",
        center_z=0.0366215,
        z_min=-0.36621,
        z_max=0.439453,
        inner_radius=0.0,
        outer_radius=1.68457,
        spacing=0.03171385028658623,
    ),
    VolumeCylinderRefinementSpec(
        name="intermediate_annulus",
        center_z=-3.0944839,
        z_min=-6.26221,
        z_max=0.0732422,
        inner_radius=1.06201,
        outer_radius=1.9043,
        spacing=0.03171385028658623,
    ),
    VolumeCylinderRefinementSpec(
        name="coarse_cylinder",
        center_z=0.073242,
        z_min=-0.732422,
        z_max=0.878906,
        inner_radius=0.0,
        outer_radius=1.9043,
        spacing=0.06342770057317168,
    ),
    VolumeCylinderRefinementSpec(
        name="short_coarse_annulus",
        center_z=-2.966309,
        z_min=-6.66504,
        z_max=0.732422,
        inner_radius=0.585938,
        outer_radius=2.05078,
        spacing=0.06342770057317168,
    ),
    VolumeCylinderRefinementSpec(
        name="coarse_annulus",
        center_z=-3.9916985,
        z_min=-8.42285,
        z_max=0.439453,
        inner_radius=0.732422,
        outer_radius=2.27051,
        spacing=0.06342770057317168,
    ),
    VolumeCylinderRefinementSpec(
        name="outer_short_wide_cylinder",
        center_z=-4.82941,
        z_min=-10.83,
        z_max=1.17118,
        inner_radius=0.0,
        outer_radius=2.65,
        spacing=0.12685540114634414,
    ),
    VolumeCylinderRefinementSpec(
        name="outermost_cylinder",
        center_z=-3.808595,
        z_min=-9.22852,
        z_max=1.61133,
        inner_radius=0.0,
        outer_radius=3.06,
        spacing=0.12685540114634414,
    ),
)


BLADE_VORTEX_REFINEMENTS: tuple[BladeVortexBoxRefinementSpec, ...] = (
    BladeVortexBoxRefinementSpec(
        name="Blade1+3Refinement",
        center=(0.0, 0.0, 0.1),
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        length=2.97,
        tangential_width=0.34,
        spacing=CONFIG.blade_vortex_spacing,
    ),
    BladeVortexBoxRefinementSpec(
        name="Blade2+4Refinement",
        center=(0.0, 0.0, 0.003),
        axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
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
        return {
            wall_name: [
                geometry[boundary_name]
                for boundary_name in boundary_names
                if any(boundary_name.startswith(prefix) for prefix in prefixes)
            ]
            for wall_name, prefixes in SURFACE_MESH_WALL_PREFIXES.items()
        }

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


def _rotation_volume(cfg: CaseSetup):
    return fl.Cylinder(
        name="zone_r1",
        center=cfg.rotation_center * u.m,
        axis=cfg.rotation_axis,
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
            size=(spec.length, spec.tangential_width, blade_vortex_z_size) * u.m,
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
):
    farfield = fl.AutomatedFarfield(
        name="farfield",
        relative_size=cfg.farfield_relative_size,
    )
    refinements = []
    if include_surface_refinements:
        refinements.extend(_make_surface_refinements(geometry, cfg))
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
        mach=cfg.mach,
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


def _make_wall_model(name: str, surfaces: list, cfg: CaseSetup):
    return fl.Wall(
        name=name,
        surfaces=surfaces,
        use_wall_function=fl.WallFunction(),
        heat_spec=fl.HeatFlux(0.0 * u.W / u.m**2),
        roughness_height=cfg.wall_roughness_height * u.m,
    )


def _make_models(geometry, farfield, rotation_volume, cfg: CaseSetup):
    if hasattr(geometry, "group_faces_by_tag"):
        geometry.group_faces_by_tag("faceName")
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


def _make_aeroacoustic_output(source_file: Path = AEROACOUSTIC_SOURCE_FILE):
    with source_file.open(encoding="utf-8") as file:
        aeroacoustic_output = json.load(file)["aeroacousticOutput"]

    observers = [
        fl.Observer(
            position=observer_position * u.m,
            group_name=f"observer_{index:03d}",
        )
        for index, observer_position in enumerate(aeroacoustic_output["observers"], start=1)
    ]

    return fl.AeroAcousticOutput(
        name="AeroAcousticOutput",
        observers=observers,
        write_per_surface_output=aeroacoustic_output["writePerSurfaceOutput"],
        patch_type=aeroacoustic_output["patchType"],
    )


def _make_outputs(wall_surfaces: list):
    return [
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
        _make_aeroacoustic_output(),
    ]


def build_params(
    project,
    cfg: CaseSetup = CONFIG,
    *,
    use_beta_mesher: bool = False,
    use_surface_mesh: bool = False,
):
    geometry = project.surface_mesh if use_surface_mesh else project.geometry
    rotation_volume = _rotation_volume(cfg)
    farfield, mesh_params = _make_meshing_params(
        rotation_volume,
        geometry,
        cfg,
        use_beta_mesher=use_beta_mesher,
        include_surface_refinements=not use_surface_mesh,
    )
    models, wall_surfaces = _make_models(geometry, farfield, rotation_volume, cfg)

    with fl.SI_unit_system:
        return fl.SimulationParams(
            version="25.10.0",
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


def _mapbc_file_for_ugrid(surface_mesh_file: Path) -> Path | None:
    file_name = surface_mesh_file.name
    if file_name.endswith(".lb8.ugrid"):
        return surface_mesh_file.with_name(file_name.removesuffix(".lb8.ugrid") + ".mapbc")
    if file_name.endswith(".b8.ugrid"):
        return surface_mesh_file.with_name(file_name.removesuffix(".b8.ugrid") + ".mapbc")
    if file_name.endswith(".ugrid"):
        return surface_mesh_file.with_suffix(".mapbc")
    return None


def _surface_mesh_log_for_ugrid(surface_mesh_file: Path) -> Path:
    if DEFAULT_SURFACE_MESH_LOG_FILE.exists() and surface_mesh_file.parent == DEFAULT_SURFACE_MESH_LOG_FILE.parent:
        return DEFAULT_SURFACE_MESH_LOG_FILE

    candidates = sorted(
        surface_mesh_file.parent.glob("*surface_mesh.user.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a Flow360 surface mesh log next to {surface_mesh_file}. "
            "The log is required to reconstruct boundary patch names for the MAPBC file."
        )
    return candidates[0]


def _parse_surface_mesh_log_patches(log_file: Path) -> list[tuple[int, str, int, int]]:
    pattern = re.compile(
        r"Boundary patch (\d+) \(name: ([^)]+)\) contains (\d+) triangles, (\d+) quads"
    )
    patches = []
    for line in log_file.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            patches.append(
                (
                    int(match.group(1)),
                    match.group(2),
                    int(match.group(3)),
                    int(match.group(4)),
                )
            )
    if not patches:
        raise ValueError(f"Could not find boundary patch rows in Flow360 surface mesh log: {log_file}")
    return patches


def _ugrid_endianness(surface_mesh_file: Path) -> str:
    file_name = surface_mesh_file.name
    if ".lb8." in file_name:
        return "<"
    if ".b8." in file_name:
        return ">"
    raise ValueError(
        f"Cannot infer binary UGRID endianness from {surface_mesh_file.name}. "
        "Expected .lb8.ugrid or .b8.ugrid."
    )


def _is_binary_ugrid(surface_mesh_file: Path) -> bool:
    file_name = surface_mesh_file.name
    return file_name.endswith(".lb8.ugrid") or file_name.endswith(".b8.ugrid")


def _welded_ugrid_file_for(surface_mesh_file: Path) -> Path:
    file_name = surface_mesh_file.name
    if file_name.endswith(".lb8.ugrid"):
        return surface_mesh_file.with_name(file_name.removesuffix(".lb8.ugrid") + "_welded.lb8.ugrid")
    if file_name.endswith(".b8.ugrid"):
        return surface_mesh_file.with_name(file_name.removesuffix(".b8.ugrid") + "_welded.b8.ugrid")
    raise ValueError(f"Cannot create welded file name for non-binary UGRID: {surface_mesh_file}")


def _read_surface_ugrid(surface_mesh_file: Path):
    endian = _ugrid_endianness(surface_mesh_file)
    int_dtype = np.dtype(endian + "i4")
    float_dtype = np.dtype(endian + "f8")
    with surface_mesh_file.open("rb") as file:
        header = np.fromfile(file, dtype=int_dtype, count=7)
        if header.size != 7:
            raise ValueError(f"Could not read UGRID header from {surface_mesh_file}")
        n_nodes, n_tris, n_quads, n_tets, n_pyrs, n_prisms, n_hexes = header.tolist()
        if any(count < 0 for count in header):
            raise ValueError(f"Invalid UGRID header in {surface_mesh_file}: {tuple(header.tolist())}")
        if n_tets or n_pyrs or n_prisms or n_hexes:
            raise ValueError(
                f"Expected a surface-only UGRID, but {surface_mesh_file} contains volume cells: "
                f"tets={n_tets}, pyramids={n_pyrs}, prisms={n_prisms}, hexes={n_hexes}"
            )

        coords = np.fromfile(file, dtype=float_dtype, count=n_nodes * 3).reshape((n_nodes, 3))
        tris = np.fromfile(file, dtype=int_dtype, count=n_tris * 3).reshape((n_tris, 3))
        quads = np.fromfile(file, dtype=int_dtype, count=n_quads * 4).reshape((n_quads, 4))
        tri_ids = np.fromfile(file, dtype=int_dtype, count=n_tris)
        quad_ids = np.fromfile(file, dtype=int_dtype, count=n_quads)

    return endian, coords, tris, quads, tri_ids, quad_ids


def _write_surface_ugrid(
    surface_mesh_file: Path,
    endian: str,
    coords: np.ndarray,
    tris: np.ndarray,
    quads: np.ndarray,
    tri_ids: np.ndarray,
    quad_ids: np.ndarray,
) -> None:
    int_dtype = np.dtype(endian + "i4")
    float_dtype = np.dtype(endian + "f8")
    header = np.array([len(coords), len(tris), len(quads), 0, 0, 0, 0], dtype=int_dtype)
    with surface_mesh_file.open("wb") as file:
        header.tofile(file)
        coords.astype(float_dtype, copy=False).tofile(file)
        tris.astype(int_dtype, copy=False).tofile(file)
        quads.astype(int_dtype, copy=False).tofile(file)
        tri_ids.astype(int_dtype, copy=False).tofile(file)
        quad_ids.astype(int_dtype, copy=False).tofile(file)


def _surface_edge_multiplicity_counts(tris: np.ndarray, quads: np.ndarray) -> Counter[int]:
    edge_blocks = []
    if len(tris):
        edge_blocks.extend(
            [
                tris[:, [0, 1]],
                tris[:, [1, 2]],
                tris[:, [2, 0]],
            ]
        )
    if len(quads):
        edge_blocks.extend(
            [
                quads[:, [0, 1]],
                quads[:, [1, 2]],
                quads[:, [2, 3]],
                quads[:, [3, 0]],
            ]
        )
    if not edge_blocks:
        return Counter()

    edges = np.sort(np.vstack(edge_blocks), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    return Counter(int(count) for count in edge_counts)


def _parse_csm_face_name_tags(csm_file: Path = GEOMETRY_CSM_FILE) -> dict[int, str]:
    set_pattern = re.compile(r'^\s*set\s+(\w+)\s+"([^"]+)"')
    patbeg_pattern = re.compile(r"^\s*patbeg\s+\w+\s+(\w+)\.size")
    face_name_pattern = re.compile(r"^\s*attribute\s+faceName\s+\$(\w+)")
    face_sets: dict[str, list[int]] = {}
    face_name_by_face_id: dict[int, str] = {}
    active_set_name: str | None = None

    for line in csm_file.read_text(encoding="utf-8").splitlines():
        set_match = set_pattern.match(line)
        if set_match:
            face_sets[set_match.group(1)] = [
                int(value.strip())
                for value in set_match.group(2).split(";")
                if value.strip()
            ]
            continue

        patbeg_match = patbeg_pattern.match(line)
        if patbeg_match:
            active_set_name = patbeg_match.group(1)
            continue

        face_name_match = face_name_pattern.match(line)
        if face_name_match and active_set_name is not None:
            for face_id in face_sets.get(active_set_name, []):
                existing_name = face_name_by_face_id.get(face_id)
                face_name = face_name_match.group(1)
                if existing_name is not None and existing_name != face_name:
                    raise ValueError(
                        f"Face {face_id} has multiple CSM faceName tags: "
                        f"{existing_name!r} and {face_name!r}"
                    )
                face_name_by_face_id[face_id] = face_name
            continue

        if line.strip() == "patend":
            active_set_name = None

    if not face_name_by_face_id:
        raise ValueError(f"Could not find faceName groups in CSM file: {csm_file}")
    return face_name_by_face_id


def _mapbc_group_name(original_name: str, face_name_by_face_id: dict[int, str]) -> str:
    face_match = re.fullmatch(r"body\d+_face0*(\d+)", original_name)
    if face_match is None:
        return original_name
    face_id = int(face_match.group(1))
    group_name = face_name_by_face_id.get(face_id)
    if group_name is None:
        return original_name
    return f"{group_name}_face{face_id:05d}"


def write_grouped_mapbc(
    source_mapbc_file: Path,
    target_mapbc_file: Path,
    *,
    csm_file: Path = GEOMETRY_CSM_FILE,
) -> Path:
    face_name_by_face_id = _parse_csm_face_name_tags(csm_file)
    lines = source_mapbc_file.read_text(encoding="ascii").splitlines()
    if not lines:
        raise ValueError(f"MAPBC file is empty: {source_mapbc_file}")

    grouped_lines = [lines[0]]
    unmapped_body_faces = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid MAPBC row in {source_mapbc_file}: {line!r}")
        patch_id, bc_code, original_name = parts[:3]
        grouped_name = _mapbc_group_name(original_name, face_name_by_face_id)
        if grouped_name == original_name and re.fullmatch(r"body\d+_face0*(\d+)", original_name):
            unmapped_body_faces.append(original_name)
        grouped_lines.append(f"{patch_id} {bc_code} {grouped_name}")

    if unmapped_body_faces:
        raise ValueError(
            f"Could not map {len(unmapped_body_faces)} MAPBC body faces through {csm_file}: "
            f"{unmapped_body_faces[:10]}"
        )

    target_mapbc_file.write_text("\n".join(grouped_lines) + "\n", encoding="ascii", newline="\n")
    return target_mapbc_file


def _copy_mapbc_for_welded_ugrid(source_ugrid_file: Path, welded_ugrid_file: Path) -> None:
    source_mapbc = _mapbc_file_for_ugrid(source_ugrid_file)
    welded_mapbc = _mapbc_file_for_ugrid(welded_ugrid_file)
    if source_mapbc is None or welded_mapbc is None:
        return
    if not source_mapbc.exists():
        _ensure_mapbc_for_ugrid(source_ugrid_file)
    write_grouped_mapbc(source_mapbc, welded_mapbc)


def _drop_degenerate_faces(connectivity: np.ndarray, boundary_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if not len(connectivity):
        return connectivity, boundary_ids, 0
    unique_vertex_count = np.apply_along_axis(lambda row: len(set(row.tolist())), 1, connectivity)
    keep = unique_vertex_count == connectivity.shape[1]
    return connectivity[keep], boundary_ids[keep], int((~keep).sum())


def weld_ugrid_surface_mesh(
    surface_mesh_file: Path = DEFAULT_SURFACE_MESH_FILE,
    *,
    tolerance: float = UGRID_WELD_TOLERANCE,
    force: bool = False,
) -> Path:
    surface_mesh_file = Path(surface_mesh_file)
    if not _is_binary_ugrid(surface_mesh_file):
        return surface_mesh_file

    welded_file = _welded_ugrid_file_for(surface_mesh_file)
    if (
        welded_file.exists()
        and not force
        and welded_file.stat().st_mtime >= surface_mesh_file.stat().st_mtime
    ):
        _copy_mapbc_for_welded_ugrid(surface_mesh_file, welded_file)
        return welded_file

    _ensure_mapbc_for_ugrid(surface_mesh_file)
    endian, coords, tris, quads, tri_ids, quad_ids = _read_surface_ugrid(surface_mesh_file)
    original_edge_counts = _surface_edge_multiplicity_counts(tris, quads)

    coordinate_keys = np.round(coords / tolerance).astype(np.int64)
    _, unique_indices, inverse = np.unique(
        coordinate_keys,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    welded_coords = coords[unique_indices]
    welded_tris = (inverse[tris - 1] + 1).astype(np.int32, copy=False)
    welded_quads = (inverse[quads - 1] + 1).astype(np.int32, copy=False)
    welded_tris, welded_tri_ids, dropped_tris = _drop_degenerate_faces(welded_tris, tri_ids)
    welded_quads, welded_quad_ids, dropped_quads = _drop_degenerate_faces(welded_quads, quad_ids)
    welded_edge_counts = _surface_edge_multiplicity_counts(welded_tris, welded_quads)

    _write_surface_ugrid(
        welded_file,
        endian,
        welded_coords,
        welded_tris,
        welded_quads,
        welded_tri_ids,
        welded_quad_ids,
    )

    _copy_mapbc_for_welded_ugrid(surface_mesh_file, welded_file)

    print(
        f"Wrote welded UGRID {welded_file}: "
        f"nodes {len(coords)} -> {len(welded_coords)}, "
        f"triangles {len(tris)} -> {len(welded_tris)}, "
        f"quads {len(quads)} -> {len(welded_quads)}, "
        f"dropped degenerate faces={dropped_tris + dropped_quads}, "
        f"open edges {original_edge_counts.get(1, 0)} -> {welded_edge_counts.get(1, 0)}, "
        f"non-manifold edges after welding="
        f"{sum(count for multiplicity, count in welded_edge_counts.items() if multiplicity != 2)}."
    )
    return welded_file


def _read_ugrid_boundary_id_counts(surface_mesh_file: Path) -> tuple[tuple[int, ...], Counter[int]]:
    endian = _ugrid_endianness(surface_mesh_file)
    with surface_mesh_file.open("rb") as file:
        header = struct.unpack(endian + "7i", file.read(28))
        n_nodes, n_tris, n_quads, n_tets, n_pyrs, n_prisms, n_hexes = header
        if any(count < 0 for count in header):
            raise ValueError(f"Invalid UGRID header in {surface_mesh_file}: {header}")
        file.seek(28 + n_nodes * 3 * 8 + n_tris * 3 * 4 + n_quads * 4 * 4)
        tri_ids = struct.unpack(endian + f"{n_tris}i", file.read(n_tris * 4)) if n_tris else ()
        quad_ids = struct.unpack(endian + f"{n_quads}i", file.read(n_quads * 4)) if n_quads else ()

    counts: Counter[int] = Counter(tri_ids)
    counts.update(quad_ids)
    return header, counts


def generate_mapbc_from_surface_mesh_log(
    surface_mesh_file: Path = DEFAULT_SURFACE_MESH_FILE,
    log_file: Path | None = None,
    *,
    wall_bc_code: int = MAPBC_WALL_BC_CODE,
) -> Path:
    surface_mesh_file = Path(surface_mesh_file)
    mapbc_file = _mapbc_file_for_ugrid(surface_mesh_file)
    if mapbc_file is None:
        raise ValueError(f"MAPBC generation is only needed for UGRID files, got: {surface_mesh_file}")
    if log_file is None:
        log_file = _surface_mesh_log_for_ugrid(surface_mesh_file)

    patches = _parse_surface_mesh_log_patches(Path(log_file))
    header, id_counts = _read_ugrid_boundary_id_counts(surface_mesh_file)
    n_nodes, n_tris, n_quads, *_ = header
    expected_surface_faces = n_tris + n_quads
    log_surface_faces = sum(tri_count + quad_count for _, _, tri_count, quad_count in patches)
    if log_surface_faces != expected_surface_faces:
        raise ValueError(
            f"Patch face count from {log_file} ({log_surface_faces}) does not match "
            f"UGRID header for {surface_mesh_file} ({expected_surface_faces})."
        )

    mismatches = []
    for patch_id, name, tri_count, quad_count in patches:
        expected_count = tri_count + quad_count
        actual_count = id_counts[patch_id]
        if actual_count != expected_count:
            mismatches.append((patch_id, name, expected_count, actual_count))
    if mismatches:
        raise ValueError(
            "UGRID boundary IDs do not match Flow360 surface mesh log patch counts: "
            f"{mismatches[:10]}"
        )
    if len(id_counts) != len(patches):
        raise ValueError(
            f"UGRID contains {len(id_counts)} boundary IDs but log contains {len(patches)} patches."
        )

    with mapbc_file.open("w", encoding="ascii", newline="\n") as file:
        file.write(f"{len(patches)}\n")
        for patch_id, name, _, _ in patches:
            file.write(f"{patch_id} {wall_bc_code} {name}\n")

    print(
        f"Wrote {mapbc_file} from {log_file} "
        f"({n_nodes} nodes, {expected_surface_faces} surface faces, {len(patches)} patches)."
    )
    return mapbc_file


def _ensure_mapbc_for_ugrid(surface_mesh_file: Path) -> None:
    mapbc_file = _mapbc_file_for_ugrid(surface_mesh_file)
    if mapbc_file is not None and not mapbc_file.exists():
        generate_mapbc_from_surface_mesh_log(surface_mesh_file)


def _make_project_from_local_surface_mesh(surface_mesh_file: Path, cfg: CaseSetup, flow360_folder):
    surface_mesh_file = Path(surface_mesh_file)
    if not surface_mesh_file.exists():
        raise FileNotFoundError(
            f"Could not find reusable surface mesh file: {surface_mesh_file}. "
            "Provide an existing CGNS/UGRID surface mesh at this path."
        )

    surface_mesh_file = weld_ugrid_surface_mesh(surface_mesh_file)
    _ensure_mapbc_for_ugrid(surface_mesh_file)
    mapbc_file = _mapbc_file_for_ugrid(surface_mesh_file)
    if mapbc_file is not None and not mapbc_file.exists():
        raise FileNotFoundError(
            f"Could not find matching MAPBC file for UGRID surface mesh: {mapbc_file}. "
            "Flow360 requires the MAPBC file in the same directory with the same prefix "
            "to preserve boundary names."
        )

    return fl.Project.from_surface_mesh(
        str(surface_mesh_file),
        name=cfg.name + "_simulation_from_surface_mesh",
        folder=flow360_folder,
        length_unit=cfg.geometry_length_unit,
        run_async=False,
    )

def define_and_run_from_surface_mesh(
    *,
    surface_mesh_file: Path = DEFAULT_SURFACE_MESH_FILE,
    cfg: CaseSetup = CONFIG,
    flow360_folder=None,
    submit_draft_only: bool = True,
    generate_volume_mesh: bool = False,
    run_case: bool = False,
    bypass_length_scale_warning: bool = True,
    results_path: str | None = None,
):
    surface_mesh_file = Path(surface_mesh_file)
    project = _make_project_from_local_surface_mesh(surface_mesh_file, cfg, flow360_folder)
    params = build_params(project, cfg, use_beta_mesher=True, use_surface_mesh=True)

    warning_context = (
        fl.warning_bypass("potential_length_scale_mismatch")
        if bypass_length_scale_warning
        else nullcontext()
    )

    with warning_context:
        if submit_draft_only and not generate_volume_mesh and not run_case:
            draft = project.run_case(
                params=params,
                name="DUC_EVE_4blade_noise_setup",
                use_beta_mesher=True,
                use_geometry_AI=False,
                run_async=False,
                raise_on_error=True,
                draft_only=True,
            )
            return draft.id

        if generate_volume_mesh:
            project.generate_volume_mesh(
                params=params,
                name="VolumeMesh",
                use_beta_mesher=True,
                use_geometry_AI=False,
                run_async=False,
                raise_on_error=True,
            )

        if not run_case:
            return project.id
        project.run_case(
            params=params,
            name="DUC_EVE_4blade_noise_case",
            use_beta_mesher=True,
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


def define_and_run(
    *,
    geometry_file: Path = GEOMETRY_FILE,
    surface_mesh_file: Path | None = None,
    cfg: CaseSetup = CONFIG,
    flow360_folder=None,
    submit_draft_only: bool = True,
    generate_volume_mesh: bool = True,
    run_case: bool = False,
    use_beta_mesher: bool = True,
    bypass_length_scale_warning: bool = True,
    results_path: str | None = None,
):
    if surface_mesh_file is not None:
        return define_and_run_from_surface_mesh(
            surface_mesh_file=surface_mesh_file,
            cfg=cfg,
            flow360_folder=flow360_folder,
            submit_draft_only=submit_draft_only,
            generate_volume_mesh=generate_volume_mesh,
            run_case=run_case,
            bypass_length_scale_warning=bypass_length_scale_warning,
            results_path=results_path,
        )

    project = _make_project(geometry_file, cfg, flow360_folder, use_beta_mesher)
    params = build_params(project, cfg, use_beta_mesher=use_beta_mesher)

    warning_context = (
        fl.warning_bypass("potential_length_scale_mismatch")
        if bypass_length_scale_warning
        else nullcontext()
    )

    with warning_context:
        if submit_draft_only and not generate_volume_mesh and not run_case:
            draft = project.run_case(
                params=params,
                name="DUC_EVE_4blade_noise_setup",
                use_beta_mesher=use_beta_mesher,
                use_geometry_AI=False,
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
            name="DUC_EVE_4blade_noise_case",
            use_beta_mesher=use_beta_mesher,
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
    surface_mesh_file = DEFAULT_SURFACE_MESH_FILE
    use_beta_mesher = True
    generate_volume_mesh = True
    run_case = False

    folder = _get_or_create_flow360_folder(CONFIG.flow360_folder_path)

    project_id = define_and_run(
        flow360_folder=folder,
        surface_mesh_file=surface_mesh_file,
        submit_draft_only=True,
        generate_volume_mesh=generate_volume_mesh,
        run_case=run_case,
        use_beta_mesher=use_beta_mesher,
        bypass_length_scale_warning=True,
    )
    print(f"Project or draft id: {project_id}")


if __name__ == "__main__":
    main()
