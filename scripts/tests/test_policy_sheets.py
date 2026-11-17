"""The recorded policy sheets under ``crates/policy_controller/tests/classic/sheets``.

The policy controller's replay runs the rows of a sheet it has a module for and reads no row of
the others, so the table in the sheets' README is the only statement of what each sheet is. This
holds the table to the files: every sheet has a row and every row a sheet and a recorder, and a
row's commit, row counts and sha256 are the file's. The recording table's test filter for a sheet
is a tail of the module path the layout places its recorder at, and the recorder's parent module
declares it: a filter that names no test, or a file no parent declares, runs none and writes no
sheet. A sheet of picks states its sequence rows and its drawn rows; the pair sheet picks nothing
and states its pair rows, each an outcome on both pair wires; the overload sheet picks nothing
either and states its verdict rows, each a verdict after every step.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHEETS = ROOT / "crates" / "policy_controller" / "tests" / "classic" / "sheets"
GATEWAY = ROOT / "model_gateway" / "src"
ROW = re.compile(
    r"^\| `(?P<sheet>[a-z_]+\.json)` \| [^|]+ \| `(?P<commit>[0-9a-f]{40})` "
    r"\| (?:(?P<rows>\d+): (?P<sequence>\d+) sequence, (?P<drawn>\d+) drawn"
    r"|(?P<pairs>\d+) pair rows|(?P<verdicts>\d+) verdict rows) "
    r"\| `(?P<sha256>[0-9a-f]{64})` \| `(?P<recorder>[a-z_]+\.rs)` \|$"
)
# A line of the README's layout: a recorder's path under the gateway's sources, then its sheet.
PLACED = re.compile(r"^  (?P<path>[a-z_/]+\.rs) +(?P<sheet>[a-z_]+\.json)$")
# A row of the README's recording table: a sheet, then the test filter that runs its recorder.
TESTED = re.compile(r"^\| `(?P<sheet>[a-z_]+\.json)` \| `(?P<test>[a-z_:]+)` \|$")


def readme() -> list[str]:
    return (SHEETS / "README.md").read_text().splitlines()


def once(names: list[str], where: str) -> None:
    assert len(set(names)) == len(names), f"{where} names a sheet twice: {sorted(names)}"


def table() -> dict[str, dict[str, str]]:
    rows = [row.groupdict() for line in readme() if (row := ROW.match(line))]
    once([row["sheet"] for row in rows], "the table")
    return {row["sheet"]: row for row in rows}


def layout() -> dict[str, str]:
    lines = [at.groupdict() for line in readme() if (at := PLACED.match(line))]
    once([at["sheet"] for at in lines], "the layout")
    return {at["sheet"]: at["path"] for at in lines}


def test_every_sheet_has_a_row_and_every_row_a_sheet_and_a_recorder():
    rows = table()
    assert sorted(rows) == sorted(path.name for path in SHEETS.glob("*.json"))
    placed = layout()
    assert sorted(placed) == sorted(rows), "the layout and the table name different sheets"
    for name, row in rows.items():
        recorder = GATEWAY / placed[name]
        assert recorder.name == row["recorder"], f"{name}: the layout says {placed[name]}"
        assert recorder.is_file(), f"{name}: no recorder {placed[name]}"


def test_the_recording_table_runs_the_recorder_the_layout_places():
    placed = layout()
    tested = [at.groupdict() for line in readme() if (at := TESTED.match(line))]
    once([at["sheet"] for at in tested], "the recording table")
    assert sorted(at["sheet"] for at in tested) == sorted(placed), (
        "the recording table and the layout name different sheets"
    )
    for at in tested:
        module = "::" + placed[at["sheet"]].removesuffix(".rs").replace("/", "::")
        assert module.endswith("::" + at["test"]), (
            f"{at['sheet']}: the recording table runs {at['test']}, the layout says {module}"
        )
        # The recorder is compiled where its parent module declares it, in `<parent>.rs` or in
        # the directory's `mod.rs`: a file neither declares is in no test binary.
        path = GATEWAY / placed[at["sheet"]]
        parents = (path.parent.with_suffix(".rs"), path.parent / "mod.rs")
        assert any(
            parent.is_file() and re.search(rf"^mod {path.stem};$", parent.read_text(), re.M)
            for parent in parents
        ), f"{at['sheet']}: no parent module declares mod {path.stem}"


def test_a_row_states_its_sheets_commit_row_counts_and_sha256():
    for name, row in table().items():
        data = (SHEETS / name).read_bytes()
        sheet = json.loads(data)
        names = [recorded["name"] for recorded in sheet["rows"]]
        outcomes = [recorded["outcome"] for recorded in sheet["rows"]]
        assert len(set(names)) == len(names), f"{name}: a row name repeats"
        assert sheet["recorded_from"] == row["commit"], name
        if row["verdicts"] is not None:
            assert all(set(outcome) == {"verdicts"} for outcome in outcomes), (
                f"{name}: a row that is no verdict row"
            )
            steps = [len(recorded["steps"]) for recorded in sheet["rows"]]
            assert [len(outcome["verdicts"]) for outcome in outcomes] == steps, (
                f"{name}: a step with no verdict"
            )
            assert len(names) == int(row["verdicts"]), name
        elif row["pairs"] is None:
            kinds = [set(outcome) for outcome in outcomes]
            assert all(kind in ({"sequence"}, {"counts"}) for kind in kinds), (
                f"{name}: a row that is no pick"
            )
            drawn = kinds.count({"counts"})
            stated = (int(row["rows"]), int(row["sequence"]), int(row["drawn"]))
            assert (len(names), len(names) - drawn, drawn) == stated, name
        else:
            assert all(set(outcome) == {"grpc", "http"} for outcome in outcomes), (
                f"{name}: a row that is no pair row"
            )
            assert len(names) == int(row["pairs"]), name
        assert hashlib.sha256(data).hexdigest() == row["sha256"], name
