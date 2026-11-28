"""The recorded policy sheets under ``crates/grand_central/tests/classic/sheets``.

Grand Central's replay runs the rows of a sheet it has a module for and reads no row of
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
crate's for a recorder that is a child of code the crate holds. Its file is the one its parent
module's declaration names: the file of the module's name in the folder the parent keeps its
children in, or the file the declaration's path attribute says. The README names the package of
every root it places a recorder under, since a test filter run in another package runs none.

A declaration is read as text, and no Rust is parsed. The check knows a declaration and a path
attribute by its patterns alone (`DECLARED`, `ELSEWHERE`): one they do not read is none to it. One
on a line of its own in a block comment or a string, or one under a condition, reads as live. A
module whose declaration is not read is taken at its own path, and one whose path attribute is not
read by its name, as one with none. A file may stand there that the build does not read. Nothing
here holds that the tree compiles: the build does.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHEETS = ROOT / "crates" / "grand_central" / "tests" / "classic" / "sheets"
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
# A module's `mod` line, under any visibility. The line ends at its semicolon: a comment behind
# it leaves the declaration unread, and so does a name written as a raw identifier.
MOD = r"(?:pub(?:\([^)]+\))? )?mod {stem};$"
# A parent's declaration of a module, from its line's start: one behind a line comment's slashes
# is not read.
DECLARED = "^" + MOD
# The same for a module kept in another file: the path attribute alone on its line, then the
# `mod` line. A line between the two that opens with `#[` or `///` is skipped whole. Any other
# line there breaks the reading, and the module then reads as one with no path attribute. An
# attribute that takes more lines breaks it too, unless each of its lines opens so. The path is
# the text between two plain quotes, with no backslash and no line break in it. An empty path,
# one in a raw string and one with an escape leave the attribute unread.
ELSEWHERE = r'^#\[path = "(?P<at>[^"\\\n]+)"\]\n(?:(?:#\[|///).*\n)*' + MOD


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


Kept = tuple[pathlib.Path, pathlib.Path]


def named(folder: pathlib.Path, name: str) -> Kept:
    """A module's file by its name in `folder`, and the folder its children stand in.

    `<name>.rs`, or the `mod.rs` in the folder of that name. Both keep their children there.
    """
    inside = folder / name / "mod.rs"
    return (inside if inside.is_file() else folder / f"{name}.rs"), folder / name


def parent(root: pathlib.Path, path: pathlib.Path) -> Kept:
    """The file that declares the module at `path`, and the folder that file keeps its children in.

    The crate's library root, `lib.rs` under the source root, for a child of that root: a
    recording runs the library's tests. For any other module, the file `kept` gives for its
    parent module. A parent written inline in another module's file is taken at its own path,
    as any module whose declaration is not read. Where no file stands there, its child reads
    as one that no file declares.
    """
    if path.parent == root:
        return root / "lib.rs", root
    return kept(root, path.parent.with_suffix(".rs"))


def kept(root: pathlib.Path, path: pathlib.Path) -> Kept:
    """The file of the module at `path` by its parent's declaration, and the folder of its children.

    A path attribute says where the file is, and the file it names keeps its children beside
    itself, as a `mod.rs` does. A declaration with none names the module's file by its name, in
    the parent's folder of children. A module whose declaration is not read is taken at its own
    path under the source root, whatever file stands there.

    The answer is the text's, not the compiler's. Where a parent declares the module more than
    once, a declaration with a path attribute is read first, and no condition on either is read.
    """
    declaring, children = parent(root, path)
    text = declaring.read_text() if declaring.is_file() else ""
    if elsewhere := re.search(ELSEWHERE.format(stem=path.stem), text, re.M):
        # Asked for from the declaring file's folder with no `..` folded by text: a path through
        # a folder that does not stand names no file, whatever it normalizes to. A file that
        # stands is resolved, and a link with it: the compiler looks for a linked file's children
        # beside the link, and this looks beside its target.
        file = declaring.parent / elsewhere["at"]
        file = file.resolve() if file.is_file() else file
        return file, file.parent
    if re.search(DECLARED.format(stem=path.stem), text, re.M):
        return named(children, path.stem)
    return named(path.parent, path.stem)


def declares(root: pathlib.Path, path: pathlib.Path) -> bool:
    """Whether the module at `path` is declared in the file `parent` names for it."""
    declaring, _ = parent(root, path)
    if not declaring.is_file():
        return False
    return bool(re.search(DECLARED.format(stem=path.stem), declaring.read_text(), re.M))


def shown(path: pathlib.Path) -> pathlib.Path:
    """A path as a refusal names it, with no `..` folded.

    From the checkout's root where its text begins there, and whole where it does not.
    """
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def test_every_sheet_has_a_row_and_every_row_a_sheet_and_a_recorder():
    rows = table()
    assert sorted(rows) == sorted(path.name for path in SHEETS.glob("*.json"))
    placed = layout()
    assert sorted(placed) == sorted(rows), "the layout and the table name different sheets"
    for name, row in rows.items():
        root, path = placed[name]
        assert (root / path).name == row["recorder"], f"{name}: the layout says {path}"
        recorder, _ = kept(root, root / path)
        assert recorder.is_file(), f"{name}: no recorder {shown(recorder)}"


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
        # The recorder is compiled where its parent module declares it, in the file `parent`
        # names: a file that one does not declare is in no test binary.
        path = root / placed_at
        assert declares(root, path), f"{at['sheet']}: no parent module declares mod {path.stem}"


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


def test_a_recorder_is_read_where_its_parents_declaration_names_it(tmp_path):
    root = tmp_path.resolve() / "package" / "src"
    unit = root.parent / "tests" / "unit" / "side_v2"
    (root / "side" / "folded").mkdir(parents=True)
    (unit / "kept").mkdir(parents=True)
    (root / "lib.rs").write_text("pub mod side;\npub(crate) mod v2;\n")
    (root / "main.rs").write_text("mod binary;\n")
    (root / "side" / "mod.rs").write_text(
        "mod beside;\n"
        '#[cfg(test)]\n#[path = "../../tests/unit/side_v2/moved.rs"]\nmod moved;\n'
        '#[path = "../../tests/unit/side_v2/kept/mod.rs"]\n#[cfg(test)] // in tests alone\n'
        "/// Kept.\npub(crate) mod kept;\n"
        '#[path = "gone.rs"]\nmod stale;\n'
        '#[path = "absent/../through.rs"]\nmod through;\n'
        '#[path = "long.rs"]\n#[cfg(all(\n    test,\n))]\nmod wrapped;\n'
        '// mod commented;\n// #[path = "gone.rs"]\nmod plain;\nmod trailing; // by hand\n'
        '#[path = "gone.rs"] // by hand\nmod noted;\n'
        '#[cfg(not(test))]\nmod twice;\n#[cfg(test)]\n#[path = "other.rs"]\nmod twice;\n'
        '#[cfg(not(test))]\n#[path = "first.rs"]\nmod both;\n#[cfg(test)]\nmod both;\n'
        '#[path = "back\\\\slash.rs"]\nmod escaped;\n#[path = "two\nlines.rs"]\nmod split;\n'
    )
    (unit / "kept" / "mod.rs").write_text("mod child;\n")
    (unit / "moved.rs").write_text("pub(super) mod child;\n")
    (unit / "child.rs").write_text("mod leaf;\n")
    (root / "v2.rs").write_text("pub(in crate::v2) mod leaf;\n")

    def file_of(*module: str) -> pathlib.Path:
        return kept(root, root.joinpath(*module))[0]

    # A declaration with no path attribute names the module's file in its parent's folder of
    # children: beside a `mod.rs`, and in the folder of another parent's name.
    assert file_of("side", "beside.rs") == root / "side" / "beside.rs"
    assert file_of("v2", "leaf.rs") == root / "v2" / "leaf.rs"
    # A path attribute names the file, whatever stands at the module's own path.
    (root / "side" / "moved.rs").write_text("")
    assert file_of("side", "moved.rs") == unit / "moved.rs"
    # The file a path attribute names declares its children, and they stand beside it whatever
    # its name. A child's own children stand in the folder of the child's name again. The
    # declarations of `kept` and of `moved`'s child are read under their visibility.
    assert file_of("side", "kept.rs") == unit / "kept" / "mod.rs"
    child = root / "side" / "kept" / "child.rs"
    assert parent(root, child) == (unit / "kept" / "mod.rs", unit / "kept")
    assert file_of("side", "kept", "child.rs") == unit / "kept" / "child.rs"
    assert file_of("side", "moved", "child.rs") == unit / "child.rs"
    assert file_of("side", "moved", "child", "leaf.rs") == unit / "child" / "leaf.rs"
    # A path attribute that names no file names it all the same, whatever stands at the module's
    # own path. A path through a folder that does not stand names no file either.
    (root / "side" / "stale.rs").write_text("")
    assert file_of("side", "stale.rs") == root / "side" / "gone.rs"
    (root / "side" / "through.rs").write_text("")
    assert not file_of("side", "through.rs").is_file()
    # An attribute behind a path attribute breaks the reading where a line of it opens with
    # neither `#[` nor `///`: its module reads as one with no path attribute.
    assert file_of("side", "wrapped.rs") == root / "side" / "wrapped.rs"
    # A line comment's slashes ahead of a declaration or of a path attribute leave it unread, and
    # so does a comment behind a `mod` line's semicolon or behind a path attribute's bracket.
    assert not declares(root, root / "side" / "commented.rs")
    assert file_of("side", "plain.rs") == root / "side" / "plain.rs"
    assert not declares(root, root / "side" / "trailing.rs")
    assert file_of("side", "noted.rs") == root / "side" / "noted.rs"
    # A path with an escape or a line break in it leaves its attribute unread.
    assert file_of("side", "escaped.rs") == root / "side" / "escaped.rs"
    assert file_of("side", "split.rs") == root / "side" / "split.rs"
    # Of two declarations of one module, the one with a path attribute is read, whichever stands
    # first and whatever condition each is under.
    assert file_of("side", "twice.rs") == root / "side" / "other.rs"
    assert file_of("side", "both.rs") == root / "side" / "first.rs"
    # A module no parent declares keeps its own path, where no file stands: under a parent kept
    # elsewhere too, and under one whose own file does not stand. Its own path is the `mod.rs`
    # in the folder of its name where that file stands.
    assert file_of("side", "absent.rs") == root / "side" / "absent.rs"
    assert file_of("side", "moved", "absent.rs") == root / "side" / "moved" / "absent.rs"
    assert file_of("nowhere", "rows.rs") == root / "nowhere" / "rows.rs"
    (root / "side" / "folded" / "mod.rs").write_text("")
    assert file_of("side", "folded.rs") == root / "side" / "folded" / "mod.rs"
    # A parent declares a module with or without a path attribute, under any visibility. The
    # library's root declares a child of the source root, and the binary's root does not. The
    # answer is `True` or `False`.
    assert declares(root, root / "side" / "kept.rs") is True
    assert declares(root, root / "side" / "moved" / "child.rs")
    assert declares(root, root / "v2" / "leaf.rs")
    assert declares(root, root / "v2.rs")
    assert not declares(root, root / "binary.rs")
    assert declares(root, root / "side" / "absent.rs") is False
    # A module whose parent's file does not stand is declared by none.
    assert declares(root, root / "nowhere" / "rows.rs") is False


def test_both_checks_take_a_layout_whose_recorder_is_kept_elsewhere(tmp_path, monkeypatch):
    top = tmp_path.resolve()
    root = top / "package" / "src"
    unit = top / "package" / "tests" / "unit" / "side"
    (top / "sheets").mkdir()
    (root / "flat").mkdir(parents=True)
    (unit / "moved").mkdir(parents=True)
    (top / "sheets" / "one.json").write_text("{}")
    (top / "sheets" / "two.json").write_text("{}")
    cells = f"| a sheet | `{'0' * 40}` | 1 pair rows | `{'0' * 64}` |"
    page = (
        f"| `one.json` {cells} `rows.rs` |\n"
        f"| `two.json` {cells} `flat.rs` |\n"
        "package/src/   the package\n"
        "  side/moved/rows.rs   one.json\n"
        "  flat.rs   two.json\n"
        "| `one.json` | `moved::rows` |\n"
        "| `two.json` | `flat` |\n"
    )
    (top / "sheets" / "README.md").write_text(page)
    # One recorder stands by its name beside a parent kept under `tests/unit`. The other is the
    # `mod.rs` in the folder of its name, and the library's root declares it.
    (root / "lib.rs").write_text("mod side;\npub mod flat;\n")
    (root / "side.rs").write_text(
        '#[cfg(test)]\n#[path = "../tests/unit/side/moved/mod.rs"]\nmod moved;\n'
    )
    (unit / "moved" / "mod.rs").write_text("pub(crate) mod rows;\n")
    (unit / "moved" / "rows.rs").write_text("")
    (root / "flat" / "mod.rs").write_text("")
    # The two cases above read the layout of the folder these names point at.
    monkeypatch.setitem(globals(), "ROOT", top)
    monkeypatch.setitem(globals(), "SHEETS", top / "sheets")
    first = test_every_sheet_has_a_row_and_every_row_a_sheet_and_a_recorder
    second = test_the_recording_table_runs_the_recorder_the_layout_places
    first()
    second()

    def refuses(case, line: str) -> None:
        with pytest.raises(AssertionError, match=f"(?m)^{line}$"):
            case()

    # A recorder whose path attribute names no file is refused, whatever stands beside its
    # parent, and the refusal names the file with the attribute's `..` unfolded.
    (unit / "moved" / "mod.rs").write_text('#[path = "../gone/rows.rs"]\nmod rows;\n')
    gone = "one.json: no recorder package/tests/unit/side/moved/../gone/rows.rs"
    refuses(first, re.escape(gone))
    # A folder that the attribute names is no recorder's file.
    (unit / "moved" / "mod.rs").write_text('#[path = "../moved"]\nmod rows;\n')
    refuses(first, re.escape("one.json: no recorder package/tests/unit/side/moved/../moved"))
    # The attribute may write a whole path that does not begin at the checkout's root: the refusal
    # names it whole, its `..` too. No path through `/dev/null` names a file: it is no folder.
    (unit / "moved" / "mod.rs").write_text('#[path = "/dev/null/../rows.rs"]\nmod rows;\n')
    refuses(first, re.escape("one.json: no recorder /dev/null/../rows.rs"))
    # So is a recorder that its parent's file does not declare, whatever a file left at the
    # parent's own path declares.
    (root / "side").mkdir()
    (root / "side" / "moved.rs").write_text("mod rows;\n")
    (unit / "moved" / "mod.rs").write_text("")
    refuses(second, r"one\.json: no parent module declares mod rows")
    # So is a row whose recorder is not the file its layout line names.
    (top / "sheets" / "README.md").write_text(page.replace("`rows.rs` |", "`other.rs` |"))
    refuses(first, r"one\.json: the layout says side/moved/rows\.rs")
