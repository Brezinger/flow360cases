"""Export XWing wing and stabilizer PatchID selections from a ParaView state file.

Run this script with ParaView's Python interpreter, not with a regular Python
installation. From PowerShell:

    pvpython export_wing_stab_patch_ids.py --data-dir "C:/path/to/result"

To run it from ParaView's Tools -> Python Script Editor, do not paste this
script unchanged: it requires the ``--data-dir`` command-line argument.
Paste and run this launcher instead, updating the result-directory path:

    import sys

    sys.argv = [
        "export_wing_stab_patch_ids.py",
        "--data-dir",
        r"C:\\path\\to\\result",
    ]
    exec(
        open(
            r"C:\\git\\flow360cases\\Postprocessing\\export_wing_stab_patch_ids.py",
            encoding="utf-8",
        ).read(),
        {"__name__": "__main__"},
    )

The result directory must contain ``surfaces.vtu`` and
``Extract_wings_stabs.pvsm``. The state file must register its eight
``Extract Cells by Region`` filters as ``Extract wing1`` ... ``Extract wing4``
and ``Extract stab1`` ... ``Extract stab4``. The script writes
``<surface_name>_data.csv`` files containing only cell-data ``PatchID`` values.
These are the selection files required by ``cl_cm_spanwise_distr_analysis.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SURFACE_NAMES = (
    *(f"wing{index}" for index in range(1, 5)),
    *(f"stab{index}" for index in range(1, 5)),
)
FILTER_NAME_BY_SURFACE = {
    surface_name: f"Extract {surface_name}"
    for surface_name in SURFACE_NAMES
}
DEFAULT_STATE_FILENAME = "Extract_wings_stabs.pvsm"
DEFAULT_SURFACE_FILENAME = "surfaces.vtu"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PatchID-only CSV selections from the XWing ParaView state."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Result directory containing surfaces.vtu and the ParaView state file.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "Optional state-file path. Defaults to "
            "<data-dir>/Extract_wings_stabs.pvsm."
        ),
    )
    return parser.parse_args()


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_surface_filters(sources: dict[tuple[str, str], object]) -> dict[str, object]:
    """Map state-file pipeline registrations to the expected eight surface names."""
    available_names = [registration[0] for registration in sources]
    filters: dict[str, object] = {}

    for surface_name, filter_name in FILTER_NAME_BY_SURFACE.items():
        normalised_target = _normalise_name(filter_name)
        matches = [
            proxy
            for registration, proxy in sources.items()
            if _normalise_name(registration[0]) == normalised_target
        ]
        if len(matches) != 1:
            available = ", ".join(sorted(available_names))
            raise RuntimeError(
                f"Could not uniquely find the {filter_name!r} Extract Cells by Region "
                f"filter in the loaded state. Available pipeline names: {available}"
            )
        filters[surface_name] = matches[0]

    return filters


def main() -> None:
    args = _parse_arguments()
    data_dir = args.data_dir.resolve()
    state_file = (args.state_file or data_dir / DEFAULT_STATE_FILENAME).resolve()
    surface_file = data_dir / DEFAULT_SURFACE_FILENAME

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {data_dir}")
    if not state_file.is_file():
        raise FileNotFoundError(f"ParaView state file does not exist: {state_file}")
    if not surface_file.is_file():
        raise FileNotFoundError(f"Flow360 surface file does not exist: {surface_file}")

    try:
        from paraview.simple import (  # type: ignore[import-not-found]
            Delete,
            GetSources,
            LoadState,
            SaveData,
            UpdatePipeline,
        )
    except ImportError as error:
        raise RuntimeError(
            "ParaView Python was not found. Run this file with pvpython, for example: "
            "pvpython export_wing_stab_patch_ids.py --data-dir <result-directory>"
        ) from error

    LoadState(
        str(state_file),
        data_directory=str(data_dir),
        restrict_to_data_directory=True,
    )
    filters = _find_surface_filters(GetSources())

    for surface_name, surface_filter in filters.items():
        UpdatePipeline(proxy=surface_filter)
        output_file = data_dir / f"{surface_name}_data.csv"
        writer = SaveData(
            str(output_file),
            proxy=surface_filter,
            FieldAssociation="Cell Data",
            ChooseArraysToWrite=1,
            CellDataArrays=["PatchID"],
            PointDataArrays=[],
            FieldDataArrays=[],
            AddMetaData=0,
            AddTime=0,
            AddTimeStep=0,
        )
        # SaveData usually returns a writer proxy; delete it when it does so to
        # keep the state pipeline unchanged while exporting all eight selections.
        if writer is not None:
            Delete(writer)
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
