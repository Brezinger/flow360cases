from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TRANSFINITE_RE = re.compile(
    r"""
    ^\s*Transfinite\s+
    (?P<kind>Curve|Line|Surface)\s*
    \{(?P<ids>[^}]*)\}
    (?P<tail>.*)$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class GeoStatement:
    text: str
    leading_comments: tuple[str, ...]
    index: int


def cleanup_geo_text(geo_text: str) -> str:
    """Remove overwritten Transfinite Curve/Line/Surface definitions."""
    statements = _parse_geo_statements(geo_text)
    latest_statement_for_entity = _latest_statement_indices(statements)

    kept_statements: list[GeoStatement] = []
    for statement in statements:
        rewritten_statement = _rewrite_transfinite_statement(
            statement,
            latest_statement_for_entity,
        )
        if rewritten_statement is None:
            continue
        if rewritten_statement == statement.text:
            kept_statements.append(statement)
            continue

        kept_statements.append(
            GeoStatement(
                text=rewritten_statement,
                leading_comments=statement.leading_comments,
                index=statement.index,
            )
        )

    return _format_geo_statements(kept_statements)


def cleanup_geo_file(
    input_file: str | Path,
    output_file: str | Path | None = None,
    *,
    backup: bool = True,
) -> Path:
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file is not None else input_path

    cleaned_text = cleanup_geo_text(input_path.read_text(encoding="utf-8"))

    if output_path == input_path and backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    output_path.write_text(cleaned_text, encoding="utf-8")
    return output_path


def _parse_geo_statements(geo_text: str) -> list[GeoStatement]:
    statements: list[GeoStatement] = []
    pending_comments: list[str] = []
    statement_lines: list[str] = []
    statement_index = 0

    for line in geo_text.splitlines():
        stripped = line.strip()
        if not statement_lines and stripped.startswith("//"):
            pending_comments.append(line)
            continue

        statement_lines.append(line)
        if ";" not in line:
            continue

        statement_text = "\n".join(statement_lines).rsplit(";", 1)[0].strip()
        if statement_text:
            statements.append(
                GeoStatement(
                    text=statement_text,
                    leading_comments=tuple(pending_comments),
                    index=statement_index,
                )
            )
            statement_index += 1

        pending_comments = []
        statement_lines = []

    trailing_text = "\n".join(statement_lines).strip()
    if trailing_text:
        statements.append(
            GeoStatement(
                text=trailing_text,
                leading_comments=tuple(pending_comments),
                index=statement_index,
            )
        )

    return statements


def _latest_statement_indices(statements: list[GeoStatement]) -> dict[tuple[str, int], int]:
    latest: dict[tuple[str, int], int] = {}
    for statement in statements:
        for entity_key in _transfinite_entity_keys(statement.text):
            latest[entity_key] = statement.index
    return latest


def _rewrite_transfinite_statement(
    statement: GeoStatement,
    latest_statement_for_entity: dict[tuple[str, int], int],
) -> str | None:
    match = _TRANSFINITE_RE.match(statement.text)
    if not match:
        return statement.text

    kind = _normalized_kind(match.group("kind"))
    entity_ids = _parse_int_list(match.group("ids"))
    kept_ids = [
        entity_id
        for entity_id in entity_ids
        if latest_statement_for_entity[(kind, abs(entity_id))] == statement.index
    ]
    if not kept_ids:
        return None
    if kept_ids == entity_ids:
        return statement.text

    return (
        f"Transfinite {match.group('kind')} "
        f"{{{_format_int_list(kept_ids)}}}{match.group('tail')}"
    )


def _transfinite_entity_keys(statement: str) -> list[tuple[str, int]]:
    match = _TRANSFINITE_RE.match(statement)
    if not match:
        return []

    kind = _normalized_kind(match.group("kind"))

    return [(kind, abs(entity_id)) for entity_id in _parse_int_list(match.group("ids"))]


def _normalized_kind(value: str) -> str:
    kind = value.lower()
    return "curve" if kind == "line" else kind


def _parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for raw_item in re.split(r"[,\s]+", value.strip()):
        if not raw_item:
            continue
        result.append(int(raw_item))
    return result


def _format_int_list(values: Iterable[int]) -> str:
    return ", ".join(str(value) for value in values)


def _format_geo_statements(statements: list[GeoStatement]) -> str:
    lines: list[str] = []
    for statement in statements:
        lines.extend(statement.leading_comments)
        lines.append(f"{statement.text};")
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove overwritten Transfinite Curve/Line/Surface definitions from a Gmsh .geo file."
        )
    )
    parser.add_argument("input_file", type=Path, help="Input .geo file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .geo file. Defaults to modifying input_file in place.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file when modifying the input in place.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    output_path = cleanup_geo_file(
        args.input_file,
        args.output,
        backup=not args.no_backup,
    )
    print(f"Wrote cleaned .geo file to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
