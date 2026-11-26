"""The layout rule of a package that keeps its unit tests under tests/unit.

Each case builds a small package in a scratch folder and asks the check for its
faults. The last cases hold the repository's own tree and the hook's entry.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_unit_test_layout as layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POINTER = '#[cfg(test)]\n#[path = "{path}"]\nmod {name};\n'
UNNAMED = "no pointer under src and no module of a named file names it: its cases run nowhere"


def pointer(path, name="tests"):
    return POINTER.format(path=path, name=name)


def faults(root, files):
    """The faults of the package `pkg` that holds `files`, without the package's name."""
    files = {"Cargo.toml": '[package]\nname = "pkg"\n', **files}
    for name, text in files.items():
        path = root / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return [fault.removeprefix("pkg/") for fault in layout.check(root)]


VIEW = {
    "src/lib.rs": "mod view;\n",
    "src/view.rs": "pub fn width() -> usize {\n    4\n}\n\n"
    + pointer("../tests/unit/view/tests.rs"),
    "tests/unit/view/tests.rs": "#[test]\nfn width_is_four() {}\n",
}


def test_a_pointer_at_the_rules_place_passes(tmp_path):
    assert faults(tmp_path, VIEW) == []


def test_a_package_without_the_unit_folder_is_not_bound(tmp_path):
    files = {"src/lib.rs": "#[cfg(test)]\nmod tests {\n    #[test]\n    fn holds() {}\n}\n"}
    assert faults(tmp_path, files) == []


def test_a_case_and_a_test_modules_body_under_src_are_faults(tmp_path):
    files = dict(VIEW)
    files["src/lib.rs"] = (
        "mod view;\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn inside() {}\n}\n"
        '#[tokio::test(flavor = "multi_thread")]\nasync fn loose() {}\n#[test]\nfn second() {}\n'
    )
    assert faults(tmp_path, files) == [
        "src/lib.rs:2: a test module's body under src: it stands in a file under tests/unit",
        "src/lib.rs:7: 2 cases, the first here, under src: a case stands in its module's file under tests/unit",
    ]


def test_an_attribute_is_read_to_its_closing_bracket(tmp_path):
    wrapped = '#[expect(\n    clippy::too_many_lines,\n    reason = "one long [table]"\n)]\n'
    body = dict(VIEW)
    body["src/view.rs"] = f"#[cfg(test)]\n{wrapped}mod tests {{\n}}\n"
    found = faults(tmp_path / "body", body)
    assert found[0].startswith("src/view.rs:1: a test module's body under src"), found
    gate = dict(VIEW)
    gate["src/view.rs"] = '#[cfg(all(\n    test,\n    feature = "in-process"\n))]\nmod tests {\n}\n'
    found = faults(tmp_path / "gate", gate)
    assert found[0].startswith("src/view.rs:1: a test module's body under src"), found
    behind = dict(VIEW)
    behind["src/view.rs"] = (
        f'#[cfg(test)]\n{wrapped}#[path = "../tests/unit/view/tests.rs"]\nmod tests;\n'
    )
    assert faults(tmp_path / "behind", behind) == []
    elsewhere = dict(VIEW)
    elsewhere["src/view.rs"] = (
        f'#[cfg(test)]\n{wrapped}#[path = "../unit/view/tests.rs"]\nmod tests;\n'
    )
    found = faults(tmp_path / "elsewhere", elsewhere)
    assert 'the rule\'s path is "../tests/unit/view/tests.rs"' in found[0], found


@pytest.mark.parametrize(
    "gate",
    [
        'all(not(test), feature = "in-process")',
        'any(test, feature = "test-util")',
        'all(feature = "runtime", any(test, feature = "test-util"))',
        "not(test)",
        'feature = "test"',
    ],
)
def test_a_gate_that_holds_without_test_is_product_codes(tmp_path, gate):
    files = dict(VIEW)
    files["src/lib.rs"] = (
        f"mod view;\n#[cfg({gate})]\nmod support {{\n    pub fn helper() {{}}\n}}\n"
    )
    assert faults(tmp_path, files) == []


@pytest.mark.parametrize(
    "gate",
    [
        "test",
        'all(test, not(feature = "wasm"))',
        'all(feature = "a", all(test, feature = "b"))',
        "any(test)",
    ],
)
def test_a_gate_that_cannot_hold_without_test_is_a_test_gate(tmp_path, gate):
    files = dict(VIEW)
    files["src/lib.rs"] = f"mod view;\n#[cfg({gate})]\nmod tests {{\n}}\n"
    assert faults(tmp_path, files) == [
        "src/lib.rs:2: a test module's body under src: it stands in a file under tests/unit"
    ]


@pytest.mark.parametrize(
    "source,string,file",
    [
        ("src/lib.rs", "../tests/unit/tests.rs", "tests/unit/tests.rs"),
        ("src/main.rs", "../tests/unit/main/tests.rs", "tests/unit/main/tests.rs"),
        (
            "src/bin/replay.rs",
            "../../tests/unit/bin/replay/tests.rs",
            "tests/unit/bin/replay/tests.rs",
        ),
        (
            "src/engine/view.rs",
            "../../tests/unit/engine/view/tests/mod.rs",
            "tests/unit/engine/view/tests/mod.rs",
        ),
        (
            "src/document/operator/mod.rs",
            "../../../tests/unit/document/operator/tests.rs",
            "tests/unit/document/operator/tests.rs",
        ),
    ],
)
def test_the_rules_place_is_counted_from_the_source_file(tmp_path, source, string, file):
    parents = {
        "src/engine/view.rs": {"src/lib.rs": "mod engine;\n", "src/engine/mod.rs": "mod view;\n"},
        "src/document/operator/mod.rs": {
            "src/lib.rs": "mod document;\n",
            "src/document/mod.rs": "mod operator;\n",
        },
    }
    files = {**parents.get(source, {}), source: pointer(string), file: ""}
    assert faults(tmp_path, files) == []


def test_a_root_that_is_not_the_library_takes_a_folder_of_its_own_name(tmp_path):
    files = {"src/main.rs": pointer("../tests/unit/tests.rs"), "tests/unit/tests.rs": ""}
    assert faults(tmp_path, files) == [
        'src/main.rs:1: `mod tests` names "../tests/unit/tests.rs": the rule\'s path is "../tests/unit/main/tests.rs"',
        f"tests/unit/tests.rs: {UNNAMED}",
    ]


def test_a_test_module_inside_an_inline_module_is_a_fault_that_names_the_cure(tmp_path):
    inside = (
        "src/metrics.rs:2: a test module inside the inline module `handles`: that module "
        "becomes a file of its own, and the pointer stands at that file's first level"
    )
    files = {
        "src/lib.rs": "mod metrics;\n",
        "src/metrics.rs": "pub mod handles {\n"
        '    #[cfg(test)]\n    #[path = "../../../tests/unit/metrics/handles/tests.rs"]\n    mod tests;\n}\n',
        "tests/unit/metrics/handles/tests.rs": "",
    }
    lost = f"tests/unit/metrics/handles/tests.rs: {UNNAMED}"
    assert faults(tmp_path / "pointer", files) == [inside, lost]
    # A body there is that one fault, and so is each case and module it holds.
    files["src/metrics.rs"] = (
        "pub mod handles {\n    #[cfg(test)]\n    mod tests {\n        #[test]\n"
        "        fn holds() {}\n        #[cfg(test)]\n        mod deeper {}\n    }\n}\n"
    )
    assert faults(tmp_path / "body", files) == [inside, lost]
    # The cure: the inline module in a file of its own. The test file's place stays.
    files["src/metrics.rs"] = "pub mod handles;\n"
    files["src/metrics/handles.rs"] = pointer("../../tests/unit/metrics/handles/tests.rs")
    assert faults(tmp_path / "lifted", files) == []


def test_a_test_module_without_a_path_attribute_is_named_with_the_rules_path(tmp_path):
    files = dict(VIEW)
    files["src/view.rs"] = "#[cfg(test)]\nmod tests;\n"
    files["src/view/tests.rs"] = ""
    assert faults(tmp_path, files) == [
        'src/view.rs:1: `mod tests` holds no path attribute: the rule\'s is #[path = "../tests/unit/view/tests.rs"]',
        f"tests/unit/view/tests.rs: {UNNAMED}",
    ]


def test_a_pointer_names_a_file_that_stands(tmp_path):
    files = dict(VIEW)
    del files["tests/unit/view/tests.rs"]
    files["tests/unit/other/tests.rs"] = ""
    found = faults(tmp_path, files)
    assert (
        found[0] == "src/view.rs:5: `mod tests` names tests/unit/view/tests.rs, which is no file"
    ), found


def test_a_test_file_that_nothing_names_is_a_fault(tmp_path):
    taken_out = dict(VIEW)
    taken_out["src/view.rs"] = "pub fn width() -> usize {\n    4\n}\n"
    lost = f"tests/unit/view/tests.rs: {UNNAMED}"
    assert faults(tmp_path / "pointer", taken_out) == [lost]
    deleted = dict(VIEW)
    del deleted["src/view.rs"]
    deleted["src/lib.rs"] = ""
    assert faults(tmp_path / "source", deleted) == [lost]
    # A line that lost its test gate is no pointer.
    ungated = dict(VIEW)
    ungated["src/view.rs"] = '#[path = "../tests/unit/view/tests.rs"]\nmod tests;\n'
    assert faults(tmp_path / "gate", ungated) == [lost]


def test_a_source_file_that_no_module_names_compiles_no_pointer(tmp_path):
    files = dict(VIEW)
    files["src/lib.rs"] = "// `mod view;` was taken out\n"
    found = faults(tmp_path, files)
    assert found[0].startswith(
        "src/view.rs:5: no `mod` line of the package names this source file"
    ), found
    assert found[1].startswith("tests/unit/view/tests.rs: no pointer under src"), found


def test_the_walk_follows_a_child_to_the_place_where_the_compiler_looks(tmp_path):
    files = {
        "src/lib.rs": "mod kept;\nmod pointed;\n",
        "src/kept.rs": pointer("../tests/unit/kept/tests/mod.rs"),
        "src/pointed.rs": pointer("../tests/unit/pointed/tests.rs"),
        # A pointer's file owns its folder, whatever its name: its child stands beside it.
        "tests/unit/pointed/tests.rs": "mod beside; // a comment behind the line\n",
        "tests/unit/pointed/beside.rs": "",
        "tests/unit/pointed/tests/beside.rs": "",
        # A body adds its name to the folder, and gives it back where it closes.
        "tests/unit/kept/tests/mod.rs": "mod child;\nmod inner {\n    fn helper() {\n    }\n    mod deep;\n}\n"
        "mod after;\nmod folder;\n"
        'mod r#type;\n#[path = "elsewhere/named.rs"]\nmod by_path;\n',
        "tests/unit/kept/tests/child.rs": "mod below;\n",
        "tests/unit/kept/tests/child/below.rs": "",
        "tests/unit/kept/tests/inner/deep.rs": "",
        "tests/unit/kept/tests/after.rs": "",
        "tests/unit/kept/tests/inner/after.rs": "",
        "tests/unit/kept/tests/folder/mod.rs": "mod leaf;\n",
        "tests/unit/kept/tests/folder/leaf.rs": "",
        "tests/unit/kept/tests/type.rs": "",
        "tests/unit/kept/tests/elsewhere/named.rs": "mod beside_a_path;\n",
        "tests/unit/kept/tests/elsewhere/beside_a_path.rs": "",
        # A file that nothing names gives its own child no name.
        "tests/unit/kept/tests/lost.rs": "mod kid;\n",
        "tests/unit/kept/tests/lost/kid.rs": "",
    }
    assert sorted(fault.split(":")[0] for fault in faults(tmp_path, files)) == [
        "tests/unit/kept/tests/inner/after.rs",
        "tests/unit/kept/tests/lost.rs",
        "tests/unit/kept/tests/lost/kid.rs",
        "tests/unit/pointed/tests/beside.rs",
    ]


def test_a_comment_and_a_string_are_not_read_as_code(tmp_path):
    files = dict(VIEW)
    files["src/view.rs"] = (
        "/// ```\n/// #[test]\n/// fn in_a_doc() {}\n/// ```\n"
        'pub const TEXT: &str = "\n#[cfg(test)]\nmod tests {\n#[test]\n";\n'
        'pub const RAW: &str = r#"\n#[test]\nmod ghost;\n"#;\n'
        "/*\n#[cfg(test)]\nmod tests {\n*/\n"
        "pub const QUOTE: char = '\"';\n"
        # A bracket in a comment behind the gate closes no attribute.
        '#[cfg(test)] // [\n#[path = "../tests/unit/view/tests.rs"]\nmod tests;\n'
    )
    files["tests/unit/view/tests.rs"] = (
        'const TEXT: &str = "\nmod ghost;\n";\n/* mod ghost;\nmod ghost { */\n// mod ghost;\n'
        "fn lifetime<'a>(text: &'a str) -> &'a str {\n    text\n}\n"
    )
    # The file that only a comment and a string name is compiled by nobody.
    files["tests/unit/view/ghost.rs"] = ""
    assert faults(tmp_path, files) == [f"tests/unit/view/ghost.rs: {UNNAMED}"]


def test_a_case_in_the_rules_of_a_macro_is_no_case_under_src(tmp_path):
    files = dict(VIEW)
    files["src/lib.rs"] = (
        "mod view;\n#[macro_export]\nmacro_rules! cases {\n    ($name:ident) => {\n"
        "        #[test]\n        fn $name() {}\n    };\n}\n"
    )
    assert faults(tmp_path, files) == []


def test_a_pointer_outside_the_unit_folder_needs_a_test_program_that_loads_its_file(tmp_path):
    files = dict(VIEW)
    files["src/lib.rs"] = (
        'mod view;\n#[cfg(test)]\n#[path = "../tests/common/mod.rs"]\nmod fixtures;\n'
    )
    files["tests/common/mod.rs"] = "pub fn fixture() {}\n"
    alone = faults(tmp_path / "alone", files)
    assert alone == [
        "src/lib.rs:2: `mod fixtures` names tests/common/mod.rs: a pointer outside tests/unit stands only "
        "where a test program of the package loads the same file"
    ]
    files["tests/parity.rs"] = "mod common;\n"
    assert faults(tmp_path / "shared", files) == []


def test_a_main_file_under_the_unit_folder_is_a_test_program_of_its_own(tmp_path):
    files = dict(VIEW)
    files["tests/unit/main.rs"] = ""
    found = faults(tmp_path, files)
    assert "tests/unit/main.rs: cargo reads this file as a test program of its own" in found, found


def test_the_script_prints_each_fault_and_ends_with_one(tmp_path, capsys):
    files = dict(VIEW)
    files["tests/unit/lost.rs"] = ""
    faults(tmp_path, files)
    assert layout.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("pkg/tests/unit/lost.rs: no pointer under src"), out
    assert out[-1].startswith("1 fault of the unit-test layout"), out
    (tmp_path / "pkg" / "tests" / "unit" / "lost.rs").unlink()
    assert layout.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_the_repositorys_own_tree_holds_the_layout():
    assert layout.check(ROOT) == []


def test_the_hook_set_runs_the_script_and_the_hosted_job_does_not_skip_it():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [hook for repo in config["repos"] if repo["repo"] == "local" for hook in repo["hooks"]]
    hook = next(hook for hook in hooks if hook["id"] == "unit-test-layout")
    assert hook["entry"] == "python3 scripts/check_unit_test_layout.py"
    assert hook["language"] == "system"
    assert hook["pass_filenames"] is False
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "pr-test-rust.yml").read_text())
    steps = workflow["jobs"]["pre-commit"]["steps"]
    step = next(step for step in steps if step.get("name") == "Run pre-commit checks")
    assert "unit-test-layout" not in step["env"]["SKIP"].split(",")
    assert "--all-files" in step["run"]
