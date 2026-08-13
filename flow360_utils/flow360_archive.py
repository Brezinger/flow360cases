"""Archive Flow360 case result files and optionally clean up cloud projects.

The script mirrors the first-level Flow360 folder/project layout locally:

    target/<flow360 folders...>/<project name>/<run name>/<downloaded files>

Deletion is configured before any downloads start, but is executed only after the
download pass completes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT_FOLDER_ID = "ROOT.FLOW360"
REQUESTED_CASE_FILES = (
    "simulation.json",
    "nonlinear_residual_v2.csv",
    "linear_residual_v2.csv",
    "surface_forces_v2.csv",
    "total_forces_v2.csv",
    "total_acoustics_v3.csv",
    "surfaces.tar.gz",
    "volumes.tar.gz",
)
ACOUSTICS_SUFFIX = "_acoustics_v3.csv"
CONTROLLER_SUFFIX = "Controller_v2.csv"
WILDCARD_RESULT_SUFFIXES = (
    ACOUSTICS_SUFFIX,
    CONTROLLER_SUFFIX,
)
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class ProjectRef:
    """Flow360 project plus its folder path and top-level cleanup group."""

    id: str
    name: str
    folder_path: tuple[str, ...]
    top_level_name: str
    top_level_id: str | None
    top_level_type: str


@dataclass(frozen=True)
class TopLevelEntry:
    """A selectable first-tier Flow360 entity."""

    name: str
    id: str
    type: str


@dataclass
class DownloadRecord:
    """One attempted download entry for the manifest."""

    project_id: str
    project_name: str
    case_id: str
    case_name: str
    remote_file: str
    local_file: str | None
    status: str
    message: str = ""


@dataclass
class ArchivePlan:
    """The discovered Flow360 workspace subset to download and optionally delete."""

    projects: list[ProjectRef] = field(default_factory=list)
    top_level_entries: list[TopLevelEntry] = field(default_factory=list)


class DownloadFailedError(RuntimeError):
    """Raised when a monitored file download fails after all attempts."""

    def __init__(self, message: str, watchdog_kills: int):
        super().__init__(message)
        self.watchdog_kills = watchdog_kills


class ProgressFileCallback:
    """Boto3 progress callback that exposes byte progress to the parent process."""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.total = 0
        self.downloaded = 0
        self._lock = threading.Lock()
        self._last_write_time = 0.0

    def __call__(self, bytes_in_chunk: int) -> None:
        with self._lock:
            self.downloaded += bytes_in_chunk
            now = time.time()
            if now - self._last_write_time < 0.5 and self.downloaded < self.total:
                return
            self._last_write_time = now
            progress = {
                "downloaded": self.downloaded,
                "total": self.total,
                "updated_at": now,
            }
            progress_file = self.progress_file.with_name(
                f"{self.progress_file.name}.{time.monotonic_ns()}.json"
            )
            progress_file.write_text(json.dumps(progress), encoding="utf-8")


def _import_flow360():
    try:
        import flow360 as fl  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import flow360. Run this script with the repository venv, for example:\n"
            r"  .\venv\Scripts\python.exe flow360_utils\flow360_archive.py C:\WDIR\flow360_archive"
        ) from exc

    return fl


def run_single_file_download(argv: list[str]) -> int:
    """Internal child-process entry point for one monitored file download."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_download-one", action="store_true")
    parser.add_argument("case_id")
    parser.add_argument("remote_file")
    parser.add_argument("local_file")
    parser.add_argument("progress_file")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    fl = _import_flow360()
    local_file = Path(args.local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    progress_callback = ProgressFileCallback(Path(args.progress_file))
    case = fl.Case.from_cloud(args.case_id)
    case._download_file(  # pylint: disable=protected-access
        args.remote_file,
        to_file=str(local_file),
        overwrite=args.overwrite,
        progress_callback=progress_callback,
    )
    return 0


def sanitize_path_part(value: str, fallback: str = "unnamed") -> str:
    """Convert a Flow360 name into one local filesystem-safe path component."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned[:180]


def basename(remote_file: str) -> str:
    """Return a remote file basename for both slash conventions."""
    return remote_file.replace("\\", "/").rsplit("/", 1)[-1]


def _record_id(record: dict) -> str | None:
    return record.get("id") or record.get("projectId") or record.get("project_id")


def _record_name(record: dict) -> str:
    return record.get("name") or record.get("projectName") or record.get("project_name") or "unnamed"


def _is_project_record(record: dict) -> bool:
    record_type = str(record.get("type") or record.get("itemType") or "").lower()
    return record_type == "project" or bool(record.get("projectId") or record.get("project_id"))


def _project_ref_from_item(
    item: dict,
    folder_path: tuple[str, ...],
    top_level_name: str,
    top_level_id: str | None,
    top_level_type: str,
) -> ProjectRef | None:
    project_id = _record_id(item)
    if project_id is None:
        content_info = item.get("contentInfo")
        if isinstance(content_info, dict):
            project_id = _record_id(content_info)
    if project_id is None:
        return None

    return ProjectRef(
        id=project_id,
        name=_record_name(item),
        folder_path=folder_path,
        top_level_name=top_level_name,
        top_level_id=top_level_id,
        top_level_type=top_level_type,
    )


def discover_workspace(fl) -> ArchivePlan:
    """Discover all first-tier entries and all projects under the Flow360 root."""
    root_folder = fl.Folder(ROOT_FOLDER_ID)
    folder_tree = root_folder.get_folder_tree()
    projects: list[ProjectRef] = []
    top_level_entries: list[TopLevelEntry] = []
    seen_project_ids: set[str] = set()

    def append_project(project_ref: ProjectRef | None) -> None:
        if project_ref is None or project_ref.id in seen_project_ids:
            return
        seen_project_ids.add(project_ref.id)
        projects.append(project_ref)

    for item in root_folder.get_projects(exclude_subfolders=True):
        project_ref = _project_ref_from_item(
            item=item,
            folder_path=(),
            top_level_name=_record_name(item),
            top_level_id=_record_id(item),
            top_level_type="project",
        )
        if project_ref is not None:
            top_level_entries.append(
                TopLevelEntry(name=project_ref.name, id=project_ref.id, type="project")
            )
            append_project(project_ref)

    def walk_folder(node: dict, path: tuple[str, ...], top_node: dict):
        folder = fl.Folder(node["id"])
        for item in folder.get_projects(exclude_subfolders=True):
            append_project(
                _project_ref_from_item(
                    item=item,
                    folder_path=path,
                    top_level_name=top_node["name"],
                    top_level_id=top_node["id"],
                    top_level_type="folder",
                )
            )

        for child in node.get("subfolders", []):
            walk_folder(child, path + (child["name"],), top_node)

    for child in folder_tree.get("subfolders", []):
        top_level_entries.append(TopLevelEntry(name=child["name"], id=child["id"], type="folder"))
        walk_folder(child, (child["name"],), child)

    return ArchivePlan(projects=projects, top_level_entries=top_level_entries)


def unique_path(path: Path) -> Path:
    """Return a non-existing sibling path if the requested local file already exists."""
    if not path.exists():
        return path
    stem = path.name
    suffixes = "".join(path.suffixes)
    if suffixes:
        stem = path.name[: -len(suffixes)]
    for idx in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{idx:03d}{suffixes}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique local filename for {path}")


def is_wildcard_result_file(remote_file: str) -> bool:
    """Return True for requested Flow360 result artifacts matched by suffix."""
    remote_basename = basename(remote_file).lower()
    return any(remote_basename.endswith(suffix.lower()) for suffix in WILDCARD_RESULT_SUFFIXES)


def wanted_remote_files(case) -> list[str]:
    """Return the requested remote files available for this case."""
    available = case.get_download_file_list()
    by_basename: dict[str, str] = {}
    wildcard_files: list[str] = []

    for entry in available:
        remote_file = entry.get("fileName") if isinstance(entry, dict) else str(entry)
        if not remote_file:
            continue
        remote_basename = basename(remote_file)
        by_basename.setdefault(remote_basename, remote_file)
        if is_wildcard_result_file(remote_file):
            wildcard_files.append(remote_file)

    selected: list[str] = []
    for requested_file in REQUESTED_CASE_FILES:
        remote_file = by_basename.get(requested_file)
        if remote_file is not None:
            selected.append(remote_file)
    selected.extend(wildcard_files)

    return sorted(dict.fromkeys(selected), key=lambda name: (basename(name).lower(), name))


def missing_requested_files(case) -> list[str]:
    """Return requested filenames missing from the case file list."""
    available_basenames = {
        basename(entry.get("fileName") if isinstance(entry, dict) else str(entry))
        for entry in case.get_download_file_list()
    }
    missing = [file_name for file_name in REQUESTED_CASE_FILES if file_name not in available_basenames]
    for suffix in WILDCARD_RESULT_SUFFIXES:
        if not any(name.lower().endswith(suffix.lower()) for name in available_basenames):
            missing.append(f"*{suffix}")
    return missing


def case_destination(target_dir: Path, project_ref: ProjectRef, case_name: str) -> Path:
    """Build the local destination directory for one Flow360 case/run."""
    path_parts = [sanitize_path_part(part) for part in project_ref.folder_path]
    path_parts.append(sanitize_path_part(project_ref.name, fallback=project_ref.id))
    path_parts.append(sanitize_path_part(case_name, fallback="run"))
    return target_dir.joinpath(*path_parts)


def _read_progress(progress_file: Path) -> tuple[int, int]:
    progress_files = sorted(
        progress_file.parent.glob(f"{progress_file.name}.*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not progress_files:
        return 0, 0
    for candidate in progress_files[:5]:
        try:
            progress = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return int(progress.get("downloaded", 0)), int(progress.get("total", 0))
    return 0, 0


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _remove_progress_files(progress_file: Path) -> None:
    for path in progress_file.parent.glob(f"{progress_file.name}.*.json"):
        _remove_if_exists(path)
    _remove_if_exists(progress_file)


def download_file_with_watchdog(
    case,
    remote_file: str,
    local_file: Path,
    overwrite: bool,
    max_retries: int,
    stall_timeout_seconds: float,
    check_interval_seconds: float,
) -> tuple[str, int]:
    """Download one file, retrying if no byte progress is reported for too long."""
    attempts = max_retries + 1
    last_error = ""
    watchdog_kills = 0
    progress_file = local_file.with_name(local_file.name + ".download_progress.json")

    for attempt in range(1, attempts + 1):
        if attempt > 1:
            print(f"      retry {attempt - 1}/{max_retries}")
        if attempt > 1:
            _remove_if_exists(local_file)
        _remove_progress_files(progress_file)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_download-one",
            case.id,
            remote_file,
            str(local_file),
            str(progress_file),
        ]
        if overwrite:
            command.append("--overwrite")

        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        last_downloaded = -1
        last_progress_time = time.monotonic()

        while process.poll() is None:
            time.sleep(check_interval_seconds)
            downloaded, total = _read_progress(progress_file)
            if downloaded > last_downloaded:
                last_downloaded = downloaded
                last_progress_time = time.monotonic()
                if total:
                    percent = downloaded / total * 100
                    print(f"      progress: {percent:5.1f}% ({downloaded}/{total} bytes)")
                else:
                    print(f"      progress: {downloaded} bytes")
                continue

            if time.monotonic() - last_progress_time >= stall_timeout_seconds:
                last_error = (
                    f"stalled for {stall_timeout_seconds:g}s at "
                    f"{max(last_downloaded, 0)} bytes"
                )
                print(f"      {last_error}; restarting")
                watchdog_kills += 1
                process.terminate()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=10)
                _remove_if_exists(local_file)
                _remove_progress_files(progress_file)
                break
        else:
            return_code = process.returncode
            if return_code == 0:
                _remove_progress_files(progress_file)
                return str(local_file), watchdog_kills
            _, stderr = process.communicate(timeout=10)
            last_error = f"download process exited with code {return_code}"
            if stderr.strip():
                last_error += f": {stderr.strip().splitlines()[-1]}"
            print(f"      {last_error}")

    raise DownloadFailedError(
        f"Failed after {attempts} attempts: {last_error}",
        watchdog_kills=watchdog_kills,
    )


def download_project_cases(
    fl,
    project_ref: ProjectRef,
    target_dir: Path,
    overwrite: bool,
    restart: bool,
    max_retries: int,
    stall_timeout_seconds: float,
    check_interval_seconds: float,
) -> list[DownloadRecord]:
    """Download requested artifacts for every case in one project."""
    records: list[DownloadRecord] = []
    print(f"\nProject: {project_ref.name} ({project_ref.id})")
    try:
        project = fl.Project.from_cloud(project_ref.id)
        case_ids = project.get_case_ids()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"  ERROR: could not load project: {exc}")
        records.append(
            DownloadRecord(
                project_id=project_ref.id,
                project_name=project_ref.name,
                case_id="",
                case_name="",
                remote_file="",
                local_file=None,
                status="error",
                message=f"Could not load project: {exc}",
            )
        )
        return records

    if not case_ids:
        print("  No cases found.")
        return records

    for case_id in case_ids:
        try:
            case = fl.Case.from_cloud(case_id)
            run_name = case.name
            destination = case_destination(target_dir, project_ref, run_name)
            destination.mkdir(parents=True, exist_ok=True)
            selected_files = wanted_remote_files(case)
            missing_files = missing_requested_files(case)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  Case {case_id}: ERROR during case inspection: {exc}")
            records.append(
                DownloadRecord(
                    project_id=project_ref.id,
                    project_name=project_ref.name,
                    case_id=case_id,
                    case_name="",
                    remote_file="",
                    local_file=None,
                    status="error",
                    message=f"Could not inspect case: {exc}",
                )
            )
            continue

        print(f"  Run: {run_name} ({case_id})")
        for missing_file in missing_files:
            print(f"    missing: {missing_file}")
            records.append(
                DownloadRecord(
                    project_id=project_ref.id,
                    project_name=project_ref.name,
                    case_id=case_id,
                    case_name=run_name,
                    remote_file=missing_file,
                    local_file=None,
                    status="missing",
                )
            )

        for remote_file in selected_files:
            local_name = basename(remote_file)
            local_file = destination / local_name
            if restart and local_file.exists():
                print(f"    skipped existing: {local_name}")
                records.append(
                    DownloadRecord(
                        project_id=project_ref.id,
                        project_name=project_ref.name,
                        case_id=case_id,
                        case_name=run_name,
                        remote_file=remote_file,
                        local_file=str(local_file),
                        status="skipped_existing",
                        message="restart mode",
                    )
                )
                continue
            if not overwrite:
                local_file = unique_path(local_file)
            try:
                downloaded_file, watchdog_kills = download_file_with_watchdog(
                    case=case,
                    remote_file=remote_file,
                    local_file=local_file,
                    overwrite=overwrite,
                    max_retries=max_retries,
                    stall_timeout_seconds=stall_timeout_seconds,
                    check_interval_seconds=check_interval_seconds,
                )
                print(f"    downloaded: {local_name}")
                message = (
                    f"watchdog_restarts={watchdog_kills}" if watchdog_kills else ""
                )
                records.append(
                    DownloadRecord(
                        project_id=project_ref.id,
                        project_name=project_ref.name,
                        case_id=case_id,
                        case_name=run_name,
                        remote_file=remote_file,
                        local_file=str(downloaded_file),
                        status="downloaded",
                        message=message,
                    )
                )
            except DownloadFailedError as exc:
                print(f"    ERROR downloading {local_name}: {exc}")
                status = "watchdog_failed" if exc.watchdog_kills else "error"
                records.append(
                    DownloadRecord(
                        project_id=project_ref.id,
                        project_name=project_ref.name,
                        case_id=case_id,
                        case_name=run_name,
                        remote_file=remote_file,
                        local_file=None,
                        status=status,
                        message=f"{exc}; watchdog_kills={exc.watchdog_kills}",
                    )
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"    ERROR downloading {local_name}: {exc}")
                records.append(
                    DownloadRecord(
                        project_id=project_ref.id,
                        project_name=project_ref.name,
                        case_id=case_id,
                        case_name=run_name,
                        remote_file=remote_file,
                        local_file=None,
                        status="error",
                        message=str(exc),
                    )
                )

    return records


def write_manifest(target_dir: Path, records: list[DownloadRecord]) -> Path:
    """Write a CSV manifest of all download attempts."""
    manifest_path = target_dir / "flow360_archive_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "project_id",
                "project_name",
                "case_id",
                "case_name",
                "remote_file",
                "local_file",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
    return manifest_path


def write_run_metadata(
    target_dir: Path,
    selected_delete_names: set[str],
    delete_mode: str,
    records: list[DownloadRecord],
) -> Path:
    """Write a compact JSON summary for later audit."""
    metadata_path = target_dir / "flow360_archive_run.json"
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "delete_mode": delete_mode,
        "selected_top_level_names_for_delete": sorted(selected_delete_names),
        "status_counts": status_counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def print_watchdog_summary(records: list[DownloadRecord]) -> None:
    """Print stalled-download outcomes, especially files that never completed."""
    restarted = [
        record
        for record in records
        if record.status == "downloaded" and "watchdog_restarts=" in record.message
    ]
    failed = [record for record in records if record.status == "watchdog_failed"]

    print("\nWatchdog summary:")
    print(f"  Restarted and completed: {len(restarted)}")
    print(f"  Watchdog-killed and not completed: {len(failed)}")
    for record in failed:
        print(
            "  FAILED: "
            f"{record.project_name} / {record.case_name} / {basename(record.remote_file)} "
            f"({record.case_id}) - {record.message}"
        )


def parse_name_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def prompt_yes_no(question: str, default: bool = False) -> bool:
    """Prompt for a yes/no answer."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_delete_policy(
    top_level_entries: list[TopLevelEntry],
    delete_mode: str | None = None,
) -> tuple[set[str], str]:
    """Interactively choose top-level entries to delete after download."""
    if not sys.stdin.isatty():
        if delete_mode is None:
            return set(), "none"
        raise SystemExit(
            "--delete-mode requires an interactive terminal unless --delete-top-level "
            "or --keep-top-level is also provided."
        )

    print("\nFirst-tier Flow360 entities:")
    for entry in sorted(top_level_entries, key=lambda item: (item.type, item.name.lower())):
        print(f"  [{entry.type}] {entry.name}")

    if delete_mode is None and not prompt_yes_no("\nDelete anything from Flow360 after the download pass?"):
        return set(), "none"

    mode = delete_mode
    if mode is None:
        mode = input(
            "Delete mode: 'projects' deletes projects under selected top-level entries; "
            "'top-level' deletes the selected root folders/projects themselves [projects]: "
        ).strip().lower()
        if mode not in {"projects", "top-level"}:
            mode = "projects"

    selected = {
        entry.name
        for entry in sorted(top_level_entries, key=lambda item: (item.type, item.name.lower()))
        if prompt_yes_no(f"Delete [{entry.type}] {entry.name}?")
    }
    if not selected:
        print("No names selected for deletion. Deletion disabled.")
        return set(), "none"

    return selected, mode


def validate_selected_names(selected_names: set[str], top_level_entries: Iterable[TopLevelEntry]) -> None:
    entries = list(top_level_entries)
    available = {entry.name for entry in entries}
    unknown = selected_names - available
    if unknown:
        raise SystemExit(
            "Unknown first-tier Flow360 names selected for deletion: "
            + ", ".join(sorted(unknown))
            + "\nAvailable names: "
            + ", ".join(sorted(available))
        )
    duplicate_names = {
        entry.name
        for entry in entries
        if sum(candidate.name == entry.name for candidate in entries) > 1
    }
    ambiguous = selected_names & duplicate_names
    if ambiguous:
        raise SystemExit(
            "These first-tier names are ambiguous because more than one root entity has that name: "
            + ", ".join(sorted(ambiguous))
        )


def names_to_delete_from_keep_list(keep_names: set[str], entries: Iterable[TopLevelEntry]) -> set[str]:
    """Return all first-tier names except the explicit keep-list."""
    entries = list(entries)
    validate_selected_names(keep_names, entries)
    return {entry.name for entry in entries if entry.name not in keep_names}


def confirm_delete(selected_names: set[str], delete_mode: str, records: list[DownloadRecord]) -> bool:
    """Require a second confirmation after downloads before deleting cloud resources."""
    if not selected_names or delete_mode == "none":
        return False
    if any(record.status in {"error", "watchdog_failed"} for record in records):
        print("\nDownload errors were recorded. Server deletion is skipped.")
        return False

    print("\nDeletion summary:")
    print(f"  Mode: {delete_mode}")
    print(f"  First-tier names: {', '.join(sorted(selected_names))}")
    confirmation = input("Type DELETE to permanently delete the selected Flow360 resources: ").strip()
    return confirmation == "DELETE"


def delete_selected_resources(
    fl,
    plan: ArchivePlan,
    selected_names: set[str],
    delete_mode: str,
) -> None:
    """Delete selected cloud resources according to the startup policy."""
    if delete_mode == "projects":
        project_ids = sorted(
            {
                project.id
                for project in plan.projects
                if project.top_level_name in selected_names
            }
        )
        for project_id in project_ids:
            print(f"Deleting project {project_id}")
            fl.Project.from_cloud(project_id).delete()
        return

    if delete_mode == "top-level":
        entries_by_name = {entry.name: entry for entry in plan.top_level_entries}
        for name in sorted(selected_names):
            entry = entries_by_name[name]
            print(f"Deleting {entry.type} {name} ({entry.id})")
            if entry.type == "folder":
                fl.Folder(entry.id).delete()
            elif entry.type == "project":
                fl.Project.from_cloud(entry.id).delete()
        return

    raise ValueError(f"Unknown delete mode: {delete_mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected files from every Flow360 case into a local folder tree, "
            "then optionally delete selected first-tier Flow360 entities."
        )
    )
    parser.add_argument("target_dir", help="Local directory where the Flow360 archive will be created.")
    parser.add_argument(
        "--delete-top-level",
        default=None,
        help=(
            "Comma-separated first-tier Flow360 folder/project names to delete after download. "
            "If omitted in an interactive terminal, the script prompts."
        ),
    )
    parser.add_argument(
        "--keep-top-level",
        default=None,
        help=(
            "Comma-separated first-tier Flow360 folder/project names to keep after download. "
            "Every other first-tier entity is selected for deletion. Mutually exclusive with "
            "--delete-top-level."
        ),
    )
    parser.add_argument(
        "--delete-mode",
        choices=("none", "projects", "top-level"),
        default=None,
        help=(
            "'projects' deletes projects under selected first-tier names. "
            "'top-level' deletes selected root folders/projects themselves."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local files with the same name instead of creating numbered siblings.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Resume an interrupted archive run by skipping requested files that already exist "
            "at their normal local destination."
        ),
    )
    parser.add_argument(
        "--download-stall-timeout",
        type=float,
        default=60.0,
        help="Seconds without byte progress before a file download is restarted. Default: 60.",
    )
    parser.add_argument(
        "--download-watchdog-interval",
        type=float,
        default=10.0,
        help="Seconds between watchdog checks during file downloads. Default: 10.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=5,
        help="Number of retry attempts after a failed or stalled file download. Default: 5.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--_download-one":
        return run_single_file_download(argv)

    args = build_parser().parse_args(argv)
    if args.delete_top_level is not None and args.keep_top_level is not None:
        raise SystemExit("--delete-top-level and --keep-top-level are mutually exclusive.")
    if args.download_stall_timeout <= 0:
        raise SystemExit("--download-stall-timeout must be greater than 0.")
    if args.download_watchdog_interval <= 0:
        raise SystemExit("--download-watchdog-interval must be greater than 0.")
    if args.download_retries < 0:
        raise SystemExit("--download-retries must be greater than or equal to 0.")

    fl = _import_flow360()
    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering Flow360 workspace...")
    plan = discover_workspace(fl)
    print(f"Discovered {len(plan.projects)} projects.")

    if args.delete_top_level is not None:
        selected_delete_names = parse_name_list(args.delete_top_level)
        delete_mode = args.delete_mode or ("projects" if selected_delete_names else "none")
    elif args.keep_top_level is not None:
        selected_delete_names = names_to_delete_from_keep_list(
            parse_name_list(args.keep_top_level),
            plan.top_level_entries,
        )
        delete_mode = args.delete_mode or ("projects" if selected_delete_names else "none")
    elif args.delete_mode == "none":
        selected_delete_names = set()
        delete_mode = "none"
    elif args.delete_mode in {"projects", "top-level"}:
        selected_delete_names, delete_mode = prompt_delete_policy(
            plan.top_level_entries,
            args.delete_mode,
        )
    else:
        selected_delete_names, delete_mode = prompt_delete_policy(plan.top_level_entries)

    validate_selected_names(selected_delete_names, plan.top_level_entries)

    records: list[DownloadRecord] = []
    for project_ref in plan.projects:
        records.extend(
            download_project_cases(
                fl=fl,
                project_ref=project_ref,
                target_dir=target_dir,
                overwrite=args.overwrite,
                restart=args.restart,
                max_retries=args.download_retries,
                stall_timeout_seconds=args.download_stall_timeout,
                check_interval_seconds=args.download_watchdog_interval,
            )
        )

    manifest_path = write_manifest(target_dir, records)
    metadata_path = write_run_metadata(target_dir, selected_delete_names, delete_mode, records)
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Wrote run metadata: {metadata_path}")
    print_watchdog_summary(records)

    if confirm_delete(selected_delete_names, delete_mode, records):
        delete_selected_resources(fl, plan, selected_delete_names, delete_mode)
        print("Deletion pass completed.")
    elif selected_delete_names:
        print("Deletion pass skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
