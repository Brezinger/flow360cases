import os

import numpy as np
import pandas as pd

import flow360 as fl
from flow360 import u


def calculate_flat_plate_turbulent_y1(
        freestream_velocity=55.0,
        density=1.225,
        dynamic_viscosity=1.82e-5,
        boundary_layer_length=0.745,
        desired_yplus=0.67):
    """
    Estimate first-cell wall distance from turbulent flat-plate boundary layer theory.

    Units:
        freestream_velocity: m/s
        density: kg/m3
        dynamic_viscosity: kg/(m s)
        boundary_layer_length: m
        desired_yplus: nondimensional

    Returns:
        reynolds_number: nondimensional
        y1: m
    """
    reynolds_number = density * freestream_velocity * boundary_layer_length / dynamic_viscosity
    skin_friction_coefficient = 0.0576 / reynolds_number ** 0.2
    wall_shear_stress = 0.5 * density * freestream_velocity ** 2 * skin_friction_coefficient
    friction_velocity = np.sqrt(wall_shear_stress / density)
    y1 = desired_yplus * dynamic_viscosity / (density * friction_velocity)

    return reynolds_number, y1


def calculate_standard_atmosphere_density(altitude=0.0):
    """
    Calculate ISA density in the troposphere.

    Units:
        altitude: m

    Returns:
        density: kg/m3
    """
    sea_level_temperature = 288.15
    sea_level_pressure = 101325.0
    temperature_lapse_rate = 0.0065
    gas_constant_air = 287.05287
    gravity = 9.80665

    temperature = sea_level_temperature - temperature_lapse_rate * altitude
    pressure = sea_level_pressure * (
            temperature / sea_level_temperature
    ) ** (gravity / (gas_constant_air * temperature_lapse_rate))

    return pressure / (gas_constant_air * temperature)


def calculate_target_lift_coefficient(aircraft_mass, wing_area, freestream_velocity, density):
    gravity = 9.80665
    dynamic_pressure = 0.5 * density * freestream_velocity ** 2

    return aircraft_mass * gravity / (dynamic_pressure * wing_area)


def calculate_freestream_velocity_for_target_lift_coefficient(
        aircraft_mass, wing_area, target_lift_coefficient, density):
    gravity = 9.80665

    if target_lift_coefficient <= 0:
        raise ValueError("target_lift_coefficient must be positive to calculate freestream velocity.")

    return np.sqrt(aircraft_mass * gravity / (0.5 * density * wing_area * target_lift_coefficient))


def calc_ncrit_from_fsti(turbulence_intensity_perc=0.05):
    """
    Calculates critical N factor from freestream turbulence intensity.

    Based on Djeddi, Coder et al.,
    "Adjoint-Based Uncertainty Quantification and Calibration of RANS-Based
    Transition Modeling", DOI: 10.2514/6.2021-3036.
    """
    a0 = 9.0064
    a1 = -4.4958
    a2 = -1.4208
    a3 = 1.5920
    a4 = -0.3532

    tau = 2.5 * np.tanh(turbulence_intensity_perc / 2.5)
    ncrit = a0 + a1 * tau + a2 * tau ** 2 + a3 * tau ** 3 + a4 * tau ** 4

    return ncrit


def resolve_existing_path(filename, search_root=None):
    """Resolve a data file from cwd first, then relative to search_root."""
    if filename is None:
        return None

    if search_root is None:
        search_root = os.path.dirname(__file__)

    candidate_paths = [
        filename,
        os.path.join(search_root, filename),
    ]
    for candidate_path in candidate_paths:
        if os.path.isfile(candidate_path):
            return candidate_path

    raise FileNotFoundError(
        f"Could not find file '{filename}' in the current working directory "
        f"or next to {search_root}."
    )


def axis_perpendicular_to(segment_axis, preferred_axis):
    """Return preferred_axis projected onto the plane normal to segment_axis."""
    segment_axis = np.asarray(segment_axis, dtype=float)
    preferred_axis = np.asarray(preferred_axis, dtype=float)

    projected_axis = preferred_axis - np.dot(preferred_axis, segment_axis) * segment_axis
    projected_norm = np.linalg.norm(projected_axis)
    if projected_norm < 1e-12:
        return None

    return projected_axis / projected_norm


def make_segment_boxes(vertices_coords_file, x_size, z_size, center_offset=(0, 0, 0),
                       name_prefix="segment_box", segment_overlap=0,
                       search_root=None):
    vertices_coords_path = resolve_existing_path(vertices_coords_file, search_root)
    points = pd.read_csv(vertices_coords_path, sep=r"\s+", engine="python")

    required_columns = {"X", "Y", "Z"}
    missing_columns = required_columns - set(points.columns)
    if missing_columns:
        raise ValueError(
            f"Coordinate file '{vertices_coords_path}' is missing columns: "
            f"{sorted(missing_columns)}"
        )

    coords = points[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(coords) < 2:
        raise ValueError(
            f"Coordinate file '{vertices_coords_path}' must contain at least two points."
        )
    center_offset = np.asarray(center_offset, dtype=float)
    if center_offset.shape != (3,):
        raise ValueError("center_offset must contain exactly three values: x, y, z.")

    boxes = []
    for idx, (start, end) in enumerate(zip(coords[:-1], coords[1:]), start=1):
        segment = end - start
        segment_length = np.linalg.norm(segment)
        if segment_length <= 0:
            raise ValueError(
                f"Segment {idx} in '{vertices_coords_path}' has zero length."
            )

        y_axis = segment / segment_length
        x_axis = axis_perpendicular_to(y_axis, (1, 0, 0))
        if x_axis is None:
            x_axis = axis_perpendicular_to(y_axis, (0, 0, 1))

        boxes.append(
            fl.Box.from_principal_axes(
                name=f"{name_prefix}_{idx:03d}",
                axes=[tuple(x_axis), tuple(y_axis)],
                center=((start + end) / 2 + center_offset) * u.mm,
                size=(x_size, segment_length + segment_overlap, z_size) * u.mm,
            )
        )

    return boxes


def make_refinement_cylinders_along_axis(
        name_prefix, start_center, axis, total_length, n_cylinders, initial_diameter,
        growth_angle_deg, cylinder_overlap=0):
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        raise ValueError("Cylinder axis must have non-zero length.")
    axis = axis / axis_norm

    if n_cylinders <= 0:
        raise ValueError("n_cylinders must be positive.")

    cylinder_length = total_length / n_cylinders
    diameter_growth_per_cylinder = 2 * cylinder_length * np.tan(np.deg2rad(growth_angle_deg))
    start_center = np.asarray(start_center, dtype=float)

    cylinders = []
    for idx in range(n_cylinders):
        center = start_center + axis * (idx + 0.5) * cylinder_length
        diameter = initial_diameter + idx * diameter_growth_per_cylinder
        cylinders.append(
            fl.Cylinder(
                name=f"{name_prefix}_{idx + 1:03d}",
                center=center * u.mm,
                axis=tuple(axis),
                outer_radius=diameter / 2 * u.mm,
                height=(cylinder_length + cylinder_overlap) * u.mm,
            )
        )

    return cylinders


def geometric_spacing_values(initial_spacing, growth_rate, n_values, max_spacing=None):
    if n_values <= 0:
        return []

    spacings = [initial_spacing * growth_rate ** idx for idx in range(n_values)]
    if max_spacing is not None:
        spacings = [min(spacing, max_spacing) for spacing in spacings]

    return spacings


def read_last_vertex(vertices_coords_file, search_root=None):
    vertices_coords_path = resolve_existing_path(vertices_coords_file, search_root)
    points = pd.read_csv(vertices_coords_path, sep=r"\s+", engine="python")

    required_columns = {"X", "Y", "Z"}
    missing_columns = required_columns - set(points.columns)
    if missing_columns:
        raise ValueError(
            f"Vertex coordinate file '{vertices_coords_path}' is missing columns: "
            f"{sorted(missing_columns)}"
        )
    if points.empty:
        raise ValueError(f"Vertex coordinate file '{vertices_coords_path}' does not contain any points.")

    return tuple(points[["X", "Y", "Z"]].iloc[-1].to_numpy(dtype=float))


def read_highest_yz_curvature_vertex(vertices_coords_file, fallback_vertex=None, search_root=None):
    vertices_coords_path = resolve_existing_path(vertices_coords_file, search_root)
    points = pd.read_csv(vertices_coords_path, sep=r"\s+", engine="python")

    required_columns = {"X", "Y", "Z"}
    missing_columns = required_columns - set(points.columns)
    if missing_columns:
        raise ValueError(
            f"Vertex coordinate file '{vertices_coords_path}' is missing columns: "
            f"{sorted(missing_columns)}"
        )

    coords = points[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(coords) < 3:
        if fallback_vertex is not None:
            return fallback_vertex
        raise ValueError(
            f"Vertex coordinate file '{vertices_coords_path}' must contain at least three points "
            "for curvature detection."
        )

    yz_coords = coords[:, [1, 2]]
    curvatures = np.full(len(coords), -np.inf)
    for idx in range(1, len(coords) - 1):
        previous_vector = yz_coords[idx] - yz_coords[idx - 1]
        next_vector = yz_coords[idx + 1] - yz_coords[idx]
        chord_vector = yz_coords[idx + 1] - yz_coords[idx - 1]

        previous_length = np.linalg.norm(previous_vector)
        next_length = np.linalg.norm(next_vector)
        chord_length = np.linalg.norm(chord_vector)
        if min(previous_length, next_length, chord_length) <= 0:
            continue

        twice_triangle_area = abs(np.cross(previous_vector, next_vector))
        curvatures[idx] = 2 * twice_triangle_area / (
                previous_length * next_length * chord_length
        )

    if not np.any(np.isfinite(curvatures)):
        if fallback_vertex is not None:
            return fallback_vertex
        raise ValueError(
            f"Could not determine finite y-z curvature from '{vertices_coords_path}'."
        )

    return tuple(coords[int(np.nanargmax(curvatures))])
