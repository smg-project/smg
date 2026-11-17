"""The recorded policy sheets under ``crates/policy_controller/tests/classic/sheets``.

The policy controller's replay runs the rows of a sheet it has a module for and reads no row of
the others, so the table in the sheets' README is the only statement of what each sheet is. This
holds the table to the files: every sheet has a row and every row a sheet and a recorder, and a
row's commit, row counts and sha256 are the file's. A sheet of picks states its sequence rows and
its drawn rows; the pair sheet picks nothing and states its pair rows, each an outcome on both
pair wires; the overload sheet picks nothing either and states its verdict rows, each a verdict
after every step.
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


def readme() -> list[str]:
    return (SHEETS / "README.md").read_text().splitlines()


def table() -> dict[str, dict[str, str]]:
    return {row["sheet"]: row.groupdict() for line in readme() if (row := ROW.match(line))}


def test_every_sheet_has_a_row_and_every_row_a_sheet_and_a_recorder():
    rows = table()
    assert sorted(rows) == sorted(path.name for path in SHEETS.glob("*.json"))
    placed = {at["sheet"]: at["path"] for line in readme() if (at := PLACED.match(line))}
    assert sorted(placed) == sorted(rows), "the layout and the table name different sheets"
    for name, row in rows.items():
        recorder = GATEWAY / placed[name]
        assert recorder.name == row["recorder"], f"{name}: the layout says {placed[name]}"
        assert recorder.is_file(), f"{name}: no recorder {placed[name]}"


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
            kinds = [next(iter(outcome)) for outcome in outcomes]
            assert set(kinds) <= {"sequence", "counts"}, f"{name}: a row that is no pick"
            drawn = kinds.count("counts")
            stated = (int(row["rows"]), int(row["sequence"]), int(row["drawn"]))
            assert (len(names), len(names) - drawn, drawn) == stated, name
        else:
            assert all(set(outcome) == {"grpc", "http"} for outcome in outcomes), (
                f"{name}: a row that is no pair row"
            )
            assert len(names) == int(row["pairs"]), name
        assert hashlib.sha256(data).hexdigest() == row["sha256"], name
