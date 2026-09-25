# `gmsh_surf_mesh.py` surface-mesh guide

`gmsh_surf_mesh.py` imports a STEP model into Gmsh and creates a two-dimensional
surface mesh.  Its configuration is a JSON file that describes geometry handling,
mesh sizing, structured regions, unstructured regions, and boundary layers.  It is
intended for CAD models whose Gmsh entity tags have been identified beforehand.

[`../DUC/msh_def_POC2.json`](../DUC/msh_def_POC2.json) is the primary working
example.  It configures four automatically traced blade zones, local size limits,
and a boundary layer.

## Running the tool

Run the script from a Python environment that has the Gmsh Python module installed:

```powershell
python .\gmsh_surf_mesh.py --mesh-def ..\DUC\msh_def_POC2.json --step path\to\model.step --out path\to\surface.cgns --no-show
```

`--step` overrides the `step_file` in the JSON file.  If neither is usable, mesh
generation stops before importing geometry.  When `--out` is omitted, the output
defaults to the STEP path with a `.cgns` suffix.  Gmsh chooses the writer from the
output extension.

| Option | Effect |
| --- | --- |
| `--mesh-def FILE` | JSON mesh definition.  Defaults to the script's configured default definition. |
| `--step FILE` | STEP file to import; overrides `step_file`. |
| `--out FILE` | Mesh output path. |
| `--no-write` | Generate the mesh without writing a file. |
| `--format msh2|msh4` | Selects the MSH version when the output suffix is `.msh`; default is `msh2`. |
| `--no-recombine` | Leaves structured surfaces unrecombined.  Recombination is enabled by default. |
| `--no-show` | Does not open the Gmsh GUI after generation. |
| `--no-mesh` | Imports and configures the geometry, then opens Gmsh unless `--no-show` is given; it does not create elements. |

The script enables Gmsh terminal output, saves all entities, displays CAD vertices
as visible spherical points, and sets the Gmsh thread limit to one fewer than the
available physical cores (with a minimum of one).  If generation fails while the
GUI is enabled, it clears mesh elements and opens the imported geometry for
inspection before re-raising the error.

## Definition-file structure

The root object must contain exactly these sections:

```json
{
  "compound_curves_at_degree_2_vertices": true,
  "geometry definition": { "...": "..." },
  "mesh definition": { "...": "..." }
}
```

`compound_curves_at_degree_2_vertices` is optional.  When true, degree-two CAD
curve chains are compounded before transfinite constraints are applied.  The two
required sections are flattened internally, so the options below appear under
their respective sections in the JSON file.

### Geometry definition

| Key | Purpose |
| --- | --- |
| `step_file` | STEP path.  A relative path is resolved relative to the JSON file. |
| `geometry_preprocessing` | Optional transforms applied to the imported highest-dimensional entities before meshing. |
| `geometry_healing` | Optional OpenCASCADE healing before meshing. |

`geometry_preprocessing` accepts `enabled`, `origin`, `scale`, `rotation_deg` (or
`rotation_rad`), and `translation`.  Coordinates can be three-item arrays; scale
and rotations may also be scalar or `x`/`y`/`z` objects.  Scaling occurs first,
then rotations about X, Y, and Z, then translation.

`geometry_healing` accepts `enabled`, `tolerance`, `fix_degenerated`,
`fix_small_edges`, `fix_small_faces`, `sew_faces`, and `make_solids`.  The POC2
definition enables healing with a tolerance of `0.25`.

### Mesh definition

`mesh_zones` is required and must be a non-empty list.  The remaining top-level
mesh options are optional:

| Key | Purpose |
| --- | --- |
| `min element size` / `min_element_size` | Global Gmsh minimum element size. |
| `max element size` / `max_element_size` | Global Gmsh maximum element size. |
| `surface_meshing_algorithm` | Optional global 2D meshing algorithm. |
| `surface_size_limits` | Per-surface maximum sizes. |
| `anisotropic_curve_refinements` | One or more `AttractorAnisoCurve` background sizing fields. |
| `boundary_layers` | Boundary-layer fields applied along selected curves and surfaces. |
| `surface_meshing_algorithms` | Per-surface unstructured meshing algorithm. |
| `unstructured_surfaces` | Surface tags deliberately excluded from transfinite/recombined meshing. |

#### Global and surface size limits

`min element size` and `max element size` set `Mesh.MeshSizeMin` and
`Mesh.MeshSizeMax`.  `surface_size_limits` adds maximum-size caps only on the
selected surfaces:

```json
{
  "surface_size_limits": [
    { "surfaces": [7, 95], "max_element_size": 5.0 },
    { "surfaces": [8, 94], "max_element_size": 10.0 }
  ]
}
```

If a surface appears more than once, the smallest configured maximum wins.  In
POC2, all other surfaces are still subject to the global `max_element_size` of
`10.0`.

#### Anisotropic curve refinement

`anisotropic_curve_refinements` creates one Gmsh `AttractorAnisoCurve` field per
entry.  Each field refines independently normal and tangent to the nearest listed curve;
the normal and tangent sizes transition from their minimum values at `dist_min` to
their maximum values at `dist_max`.

```json
{
  "anisotropic_curve_refinements": [
    {
      "curves": [407, 416, 420],
      "sampling": 1000,
      "size_min_normal": 0.1,
      "size_min_tangent": 1.0,
      "size_max_normal": 0.3,
      "size_max_tangent": 1.0,
      "dist_min": 1.0,
      "dist_max": 5.0
    },
    {
      "curves": [429, 108, 110, 428, 96, 792, 218, 427, 171, 106, 105],
      "sampling": 1000,
      "size_min_normal": 1.0,
      "size_min_tangent": 3.0,
      "size_max_normal": 3.0,
      "size_max_tangent": 3.0,
      "dist_min": 1.0,
      "dist_max": 20.0
    }
  ]
}
```

All listed curves must exist in the imported model.  `sampling` must be a positive
integer; every size and distance must be positive; and `dist_max` must be at least
`dist_min`.  Multiple refinements are combined with a Gmsh `MinAniso` field.  It
intersects their directional metrics, preserving normal and tangential sizing where
their regions overlap.  Per-surface size limits run afterward and can further reduce
that size.  The script generates the curve mesh before installing these fields, so
an unconstrained refinement curve receives uniform transfinite spacing from
`size_min_tangent`; explicit transfinite definitions take precedence.  The
anisotropic fields then control the surface mesh.  The earlier singular
`anisotropic_curve_refinement` key remains supported for one field, but it cannot
be used together with the plural key.

#### Boundary layers

Each boundary-layer item requires adjacent curve and surface tags plus its layer
parameters:

```json
{
  "boundary_layers": [
    {
      "curves": [429, 108],
      "surfaces": [8],
      "size": 1.0,
      "thickness": 20.0,
      "n_layers": 10,
      "ratio": 1.2,
      "quads": true,
      "size_far": 3.0
    }
  ]
}
```

For each listed surface, the script creates a Gmsh `BoundaryLayer` field using
only the listed curves that are adjacent to that surface.  Curves shared with
other surfaces receive an `ExcludedSurfacesList`, which keeps that field on its
intended surface.  `quads` is optional and defaults to `false`; all other fields
shown above are required.  Curve and surface tags are validated against the
imported model.

#### Surface algorithms and unstructured exceptions

Use `surface_meshing_algorithms` to select an algorithm on a surface:

```json
{
  "surface_meshing_algorithms": [
    { "surfaces": [95], "algorithm": "bamg" },
    { "surfaces": [96], "algorithm": "frontal-delaunay" }
  ]
}
```

Accepted names are `meshadapt`, `automatic`, `initialmeshonly`, `delaunay`,
`frontal-delaunay`, `bamg`, `frontal-delaunay for quads`, `packing of
parallelograms`, and `quasi-structured quad`; their numeric Gmsh codes are also
accepted.  These surfaces are treated as unstructured, so they are not given
transfinite constraints or structured recombination.  `unstructured_surfaces`
provides the same exclusion without selecting a specific algorithm.

Set `surface_meshing_algorithm` to one of the same algorithm names (or its Gmsh
numeric code) to select the global default.  A `surface_meshing_algorithms` entry
overrides that default for its listed surfaces.  The script normally uses one
fewer than the available physical cores for meshing.  When BAMG is selected globally
or on any surface, the script retains that limit for general, 1D, and 3D meshing but
limits 2D meshing to one thread.

## Structured mesh zones

A mesh zone has a `name` and `curve_definition` of either `automatic` or `manual`.
It produces curve sequences that define the transfinite constraints for a region.

### Automatic curve discovery

Automatic zones trace the CAD topology from a known start point.  The principal
keys are:

| Key | Purpose |
| --- | --- |
| `start_pt` | Gmsh point tag from which tracing begins. |
| `circumferential_direction` | Non-zero three-component direction used to trace the cross-section loop. |
| `longitudinal_direction` | Non-zero three-component direction used to trace spanwise paths. |
| `has_blunt_te` | Enables blunt trailing-edge handling. |
| `n_subcurvs_per_compound_curve` | Number of CAD subcurves in each circumferential compound group. |
| `circumferential_transfinite_def` | Distribution for the traced circumferential groups. |
| `longitudinal_transfinite_def` | Distribution for the traced spanwise sections. |

POC2 defines four automatic zones (`blade1` through `blade4`) with different
start points and directions.  They share the same seven compound-group layout;
the root blade uses larger circumferential point counts than the other blades.

The circumferential definition accepts scalar values or lists whose length matches
the number of groups.  Its usual keys are `type`, `Parameter`,
`invert_direction`, and `n_pts`.  `type` can be `Progression` or `Bump`.
`n_pts` includes both curve end points.  `invert_direction` reverses a
progression's density direction for the relevant group.

The longitudinal definition may provide a `sections` list.  Shared values such as
`default type` and `Parameter` are inherited by each section.  A section can use
either of these sizing modes:

```json
{ "type": "Progression", "n_pts": 21, "Parameter": 1.1 }
```

```json
{
  "type": "Progression",
  "mesh size mode": "ele size",
  "target ele size 1": 3.0,
  "target ele size 2": 1.0
}
```

The first form directly supplies the number of points.  The second derives a
point count and progression coefficient from the curve length and requested sizes.
For `Bump`, the two target sizes describe the corner and middle sizes.  In POC2,
the longitudinal sections use element-size mode to tighten the mesh through the
bend and tip regions.

The script creates transfinite surfaces automatically from successfully traced
automatic zones.  It also fills missing boundary-curve constraints and attempts
to make opposite edges have compatible divisions.  If two protected explicit
counts conflict, the affected surface is converted to unstructured meshing rather
than silently changing those counts.

### Manual zones

Manual zones place `circumferential_curve_sequences`,
`longitudinal_curve_sequences`, `explicit_curve_sequences`, and/or
`transfinite_surfaces` inside the zone.  Curve-sequence entries use `curve_ids`,
`invert_direction`, and the same distribution fields described above.  A
`curve_ids` item may be one curve tag or a list of contiguous curve tags treated
as one compound edge.

Manual `transfinite_surfaces` entries identify an `id`, optionally an
`Arrangement` (default `Left`), and optional four `boundary points` that set the
transfinite corner order.  Manual zones must define at least one curve sequence or
one transfinite surface.

## Surface generation and exported names

After curve and surface constraints are applied, the tool recombines every
structured surface unless `--no-recombine` is used.  It then generates a 2D mesh.
No volume mesh is created.

Before writing, it replaces any surface physical groups with export groups:

- Surfaces whose bounding box lies on the `y = 0` plane are grouped as `symm face`.
- Automatically named structured surfaces are grouped as
  `struct_surf_S##_C##`; blunt trailing-edge surfaces use `struct_bluntTE_S##`.
- Other surfaces retain their Gmsh entity name, or use `surface_#####` when no
  entity name is present.

Duplicate physical names receive numeric suffixes.

## Relation to `POC2.geo`

When a definition uses BAMG, the script automatically applies the one-thread 2D
safeguard while retaining the normal core-count-minus-one limit for 1D meshing.
The POC2 example combines its two anisotropic refinements with BAMG on surface `95`;
it does not use a boundary layer.

## Practical workflow

1. Import the STEP file in Gmsh and record the required point, curve, and surface
   tags.
2. Start with `--no-mesh` to verify import, preprocessing, healing, and the
   configured entities.
3. Define global limits and any boundary layers or local surface caps.
4. Configure each structured region as an automatic zone when its topology is
   regular, otherwise as a manual zone.
5. Generate with `--no-show` for batch runs; inspect the exported physical names
   and mesh quality in Gmsh or the downstream solver.
