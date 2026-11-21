"""The recorded policy sheets under ``crates/policy_controller/tests/classic/sheets``.

The policy controller's replay runs the rows of a sheet it has a module for and reads no row of
the others, so the table in the sheets' README is the only statement of what each sheet is. This
holds the table to the files: every sheet has a row and every row a sheet and a recorder, and a
row's commit, row counts and sha256 are the file's. The recording table's test filter for a sheet
is a tail of the module path the layout places its recorder at, and the recorder's parent module
declares it: a filter that names no test, or a file no parent declares, runs none and writes no
sheet. A sheet of picks states its sequence rows and its drawn rows; the pair sheet picks nothing
and states its pair rows, each an outcome on both pair wires; the overload sheet picks nothing
either and states its verdict rows, each a verdict after every step; a sheet of one hard demand
picks nothing and states its demand rows, each the gateway's answer to every request of a pool.

A recorder stands under the source root its layout line is beneath: the gateway's, or the worker
crate's for a recorder that is a child of code the crate holds. The README names the package of
every root it places a recorder under, since a test filter run in another package runs none.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHEETS = ROOT / "crates" / "policy_controller" / "tests" / "classic" / "sheets"
ROW = re.compile(
    r"^\| `(?P<sheet>[a-z_]+\.json)` \| [^|]+ \| `(?P<commit>[0-9a-f]{40})` "
    r"\| (?:(?P<rows>\d+): (?P<sequence>\d+) sequence, (?P<drawn>\d+) drawn"
    r"|(?P<pairs>\d+) pair rows|(?P<verdicts>\d+) verdict rows|(?P<demands>\d+) demand rows) "
    r"\| `(?P<sha256>[0-9a-f]{64})` \| `(?P<recorder>[a-z_]+\.rs)` \|$"
)
# A line of the README's layout that opens a source root: the root, then what stands under it.
UNDER = re.compile(r"^(?P<root>[a-z_/]+/src/) {2,}\S")
# A line of the README's layout: a recorder's path under the root above it, then its sheet.
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


def layout() -> dict[str, tuple[pathlib.Path, str]]:
    """Each sheet's recorder: the source root its layout line stands under, and its path there."""
    placed: list[tuple[str, tuple[pathlib.Path, str]]] = []
    root: pathlib.Path | None = None
    for line in readme():
        if under := UNDER.match(line):
            root = ROOT / under["root"]
        elif at := PLACED.match(line):
            assert root is not None, f"the layout places {at['path']} under no source root"
            placed.append((at["sheet"], (root, at["path"])))
    once([sheet for sheet, _ in placed], "the layout")
    return dict(placed)


def test_every_sheet_has_a_row_and_every_row_a_sheet_and_a_recorder():
    rows = table()
    assert sorted(rows) == sorted(path.name for path in SHEETS.glob("*.json"))
    placed = layout()
    assert sorted(placed) == sorted(rows), "the layout and the table name different sheets"
    for name, row in rows.items():
        root, path = placed[name]
        recorder = root / path
        assert recorder.name == row["recorder"], f"{name}: the layout says {path}"
        assert recorder.is_file(), f"{name}: no recorder {recorder.relative_to(ROOT)}"


def test_the_recording_table_runs_the_recorder_the_layout_places():
    placed = layout()
    tested = [at.groupdict() for line in readme() if (at := TESTED.match(line))]
    once([at["sheet"] for at in tested], "the recording table")
    assert sorted(at["sheet"] for at in tested) == sorted(placed), (
        "the recording table and the layout name different sheets"
    )
    for at in tested:
        root, placed_at = placed[at["sheet"]]
        module = "::" + placed_at.removesuffix(".rs").replace("/", "::")
        assert module.endswith("::" + at["test"]), (
            f"{at['sheet']}: the recording table runs {at['test']}, the layout says {module}"
        )
        # The recorder is compiled where its parent module declares it, in `<parent>.rs` or in
        # the directory's `mod.rs`: a file neither declares is in no test binary.
        path = root / placed_at
        parents = (path.parent.with_suffix(".rs"), path.parent / "mod.rs")
        assert any(
            parent.is_file() and re.search(rf"^mod {path.stem};$", parent.read_text(), re.M)
            for parent in parents
        ), f"{at['sheet']}: no parent module declares mod {path.stem}"


def test_the_readme_names_the_package_of_every_root_it_places_a_recorder_under():
    text = "\n".join(readme())
    for root in sorted({root for root, _ in layout().values()}):
        manifest = root.parent / "Cargo.toml"
        package = re.search(r'^name = "([a-z-]+)"$', manifest.read_text(), re.M)
        assert package, f"no package name in {manifest.relative_to(ROOT)}"
        assert re.search(rf"-p {package[1]}(?![a-z-])", text), (
            f"the README runs no recorder with -p {package[1]}"
        )


def test_a_row_states_its_sheets_commit_row_counts_and_sha256():
    for name, row in table().items():
        data = (SHEETS / name).read_bytes()
        sheet = json.loads(data)
        names = [recorded["name"] for recorded in sheet["rows"]]
        # A demand row holds answers and no outcome.
        demands = row["demands"] is not None
        outcomes = [] if demands else [recorded["outcome"] for recorded in sheet["rows"]]
        assert len(set(names)) == len(names), f"{name}: a row name repeats"
        assert sheet["recorded_from"] == row["commit"], name
        if demands:
            for recorded in sheet["rows"]:
                # A row that lists no request holds the answer to the one request its sheet asks.
                asked = len(recorded["requests"]) if "requests" in recorded else 1
                answers = recorded["answers"]
                assert len(answers) == asked, (
                    f"{name}: {recorded['name']}: {len(answers)} answers to {asked} requests"
                )
                assert all(set(answer) in ({"placed"}, {"refused"}) for answer in answers), (
                    f"{name}: {recorded['name']}: an answer that places nothing and refuses nothing"
                )
            assert len(names) == int(row["demands"]), name
        elif row["verdicts"] is not None:
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
