from __future__ import annotations

import argparse
import csv
import re
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent
DEFAULT_SURFACE_RESOLVED_DIR = RESULTS_DIR / "aeroacoustics" / "step3" / "surface_resolved"
GROUPS = ("blade1", "blade2", "blade3", "blade4", "hub")
REQUIRED_BLADE_SECTIONS = ("Inner", "main", "TE")
EXPECTED_BLADE_SECTION_COUNTS = {
    "Inner": 5,
    "main": 2,
    "TE": 8,
}
EXPECTED_HUB_FACE_COUNT = 6

INDIVIDUAL_SURFACE_PATTERN = re.compile(
    r"^(?P<case_prefix>.+?_results_surface_zone_r1_)"
    r"(?P<group>blade(?P<blade_index>[1-4])(?P<section>Inner|main|TE)|hub)"
    r"_face(?P<face_index>\d+)"
    r"(?P<suffix>_acoustics_v\d+\.csv)$",
    re.IGNORECASE,
)
AGGREGATE_SURFACE_PATTERN = re.compile(
    r"^.+?_results_surface_zone_r1_(?:blade[1-4]|hub)_acoustics_v\d+\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SurfaceFile:
    path: Path
    case_prefix: str
    group: str
    section: str | None
    suffix: str


def _parse_individual_surface_file(path: Path) -> SurfaceFile | None:
    match = INDIVIDUAL_SURFACE_PATTERN.match(path.name)
    if match is None:
        return None

    group = match.group("group")
    if group.lower().startswith("blade"):
        group = f"blade{match.group('blade_index')}"
        section = match.group("section")
    else:
        group = "hub"
        section = None

    return SurfaceFile(
        path=path,
        case_prefix=match.group("case_prefix"),
        group=group,
        section=section,
        suffix=match.group("suffix"),
    )


def _discover_surface_files(surface_resolved_dir: Path) -> tuple[dict[str, list[SurfaceFile]], list[Path]]:
    groups = {group: [] for group in GROUPS}
    aggregate_files = []

    for path in sorted(surface_resolved_dir.glob("*_acoustics_v*.csv")):
        parsed_file = _parse_individual_surface_file(path)
        if parsed_file is not None:
            groups[parsed_file.group].append(parsed_file)
        elif AGGREGATE_SURFACE_PATTERN.match(path.name):
            aggregate_files.append(path)

    return groups, aggregate_files


def _validate_surface_groups(groups: dict[str, list[SurfaceFile]], aggregate_files: list[Path]) -> tuple[str, str]:
    individual_files = [surface_file for files in groups.values() for surface_file in files]
    if not individual_files:
        if aggregate_files:
            raise RuntimeError(
                "Surface-resolved acoustics already appear to be bunched; no individual face files were found."
            )
        raise FileNotFoundError("No individual surface-resolved acoustics CSV files were found.")

    case_prefixes = {surface_file.case_prefix for surface_file in individual_files}
    suffixes = {surface_file.suffix for surface_file in individual_files}
    if len(case_prefixes) != 1:
        raise ValueError(f"Expected one case prefix, found: {sorted(case_prefixes)}")
    if len(suffixes) != 1:
        raise ValueError(f"Expected one acoustics CSV suffix/version, found: {sorted(suffixes)}")

    missing_groups = [group for group in GROUPS if not groups[group]]
    if missing_groups:
        raise ValueError(f"Missing required surface groups: {missing_groups}")

    for blade in ("blade1", "blade2", "blade3", "blade4"):
        section_counts = {
            section: sum(1 for surface_file in groups[blade] if surface_file.section == section)
            for section in REQUIRED_BLADE_SECTIONS
        }
        if section_counts != EXPECTED_BLADE_SECTION_COUNTS:
            raise ValueError(
                f"{blade} has unexpected section counts {section_counts}; "
                f"expected {EXPECTED_BLADE_SECTION_COUNTS}."
            )

    hub_count = len(groups["hub"])
    if hub_count != EXPECTED_HUB_FACE_COUNT:
        raise ValueError(f"hub has {hub_count} files; expected {EXPECTED_HUB_FACE_COUNT}.")

    return next(iter(case_prefixes)), next(iter(suffixes))


def _validate_header(header: list[str]) -> tuple[int, int, list[int]]:
    if len(header) < 3:
        raise ValueError("CSV header is too short.")
    normalized_header = [column.strip() for column in header]
    try:
        time_index = normalized_header.index("time")
        physical_step_index = normalized_header.index("physical_step")
    except ValueError as exc:
        raise ValueError("CSV must contain 'time' and 'physical_step' columns.") from exc

    summed_indices = [
        index
        for index, column in enumerate(header)
        if index not in {time_index, physical_step_index}
    ]
    if not summed_indices:
        raise ValueError("No acoustic observer columns found to sum.")
    return time_index, physical_step_index, summed_indices


def _same_coordinate(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) <= 1.0e-12
    except ValueError:
        return False


def _sum_surface_group(surface_files: list[SurfaceFile], output_file: Path) -> int:
    temp_output_file = output_file.with_name(f"{output_file.name}.tmp")
    if temp_output_file.exists():
        temp_output_file.unlink()

    with ExitStack() as stack:
        file_handles = [
            stack.enter_context(surface_file.path.open(newline=""))
            for surface_file in surface_files
        ]
        readers = [csv.reader(file_handle) for file_handle in file_handles]
        headers = [next(reader) for reader in readers]
        reference_header = headers[0]
        for surface_file, header in zip(surface_files[1:], headers[1:]):
            if header != reference_header:
                raise ValueError(f"Header mismatch in {surface_file.path}")

        time_index, physical_step_index, summed_indices = _validate_header(reference_header)
        output_handle = stack.enter_context(temp_output_file.open("w", newline=""))
        writer = csv.writer(output_handle)
        writer.writerow(reference_header)

        row_count = 0
        for row_group in zip_longest(*readers):
            if any(row is None for row in row_group):
                raise ValueError(f"Row-count mismatch while summing {output_file.name}")
            reference_row = row_group[0]
            output_row = list(reference_row)

            for row in row_group[1:]:
                if not _same_coordinate(reference_row[time_index], row[time_index]):
                    raise ValueError(f"Time mismatch while summing {output_file.name} at row {row_count + 2}")
                if reference_row[physical_step_index] != row[physical_step_index]:
                    raise ValueError(
                        f"physical_step mismatch while summing {output_file.name} at row {row_count + 2}"
                    )

            for index in summed_indices:
                output_row[index] = f"{sum(float(row[index]) for row in row_group):.16e}"
            writer.writerow(output_row)
            row_count += 1

    temp_output_file.replace(output_file)
    return row_count


def bunch_surface_resolved_acoustics(
    surface_resolved_dir: Path = DEFAULT_SURFACE_RESOLVED_DIR,
    delete_originals: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    surface_resolved_dir = Path(surface_resolved_dir)
    groups, aggregate_files = _discover_surface_files(surface_resolved_dir)
    case_prefix, suffix = _validate_surface_groups(groups, aggregate_files)

    output_files = {
        group: surface_resolved_dir / f"{case_prefix}{group}{suffix}"
        for group in GROUPS
    }
    existing_outputs = [path for path in output_files.values() if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Bunched output files already exist. Re-run with --overwrite to replace them: "
            f"{existing_outputs}"
        )

    written_files = []
    try:
        for group in GROUPS:
            row_count = _sum_surface_group(groups[group], output_files[group])
            written_files.append(output_files[group])
            print(f"Wrote {output_files[group]} from {len(groups[group])} files and {row_count} rows.")
    except Exception:
        for path in written_files:
            path.unlink(missing_ok=True)
        for temp_file in surface_resolved_dir.glob("*.tmp"):
            temp_file.unlink(missing_ok=True)
        raise

    if delete_originals:
        for surface_file in [surface_file for files in groups.values() for surface_file in files]:
            surface_file.path.unlink()
        print(f"Deleted {sum(len(files) for files in groups.values())} original individual surface files.")

    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sum Flow360 individual surface-resolved aeroacoustic CSVs into blade/hub CSVs."
    )
    parser.add_argument("--surface-resolved-dir", type=Path, default=DEFAULT_SURFACE_RESOLVED_DIR)
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep the individual face CSV files after successful aggregation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing bunched blade/hub CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bunch_surface_resolved_acoustics(
        surface_resolved_dir=args.surface_resolved_dir,
        delete_originals=not args.keep_originals,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
