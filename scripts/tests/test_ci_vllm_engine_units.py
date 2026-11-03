"""Check individual JUnit outcomes without importing an engine."""

import ast
import copy
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_check_vllm_engine_units.py"
SPEC = importlib.util.spec_from_file_location("vllm_engine_unit_check", SCRIPT)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
REQUIRED = json.loads(SCRIPT.with_name("ci_vllm_required_cases.json").read_text())
ROOT = SCRIPT.parents[1]


def definition_exists(root, case_id):
    classname, test_name = case_id.split("::", 1)
    if test_name.endswith("]"):
        test_name = test_name.split("[", 1)[0]
    parts = classname.split(".")
    for split in range(len(parts), 0, -1):
        source = root.joinpath(*parts[:split]).with_suffix(".py")
        if source.is_file():
            nodes = ast.parse(source.read_text()).body
            break
    else:
        return False
    for name in parts[split:]:
        classes = [node for node in nodes if isinstance(node, ast.ClassDef) and node.name == name]
        if len(classes) != 1:
            return False
        nodes = classes[0].body
    functions = [
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name
    ]
    return len(functions) == 1


@pytest.mark.parametrize("case_id", REQUIRED)
def test_manifest_resolves_to_real_definitions_without_importing_vllm(case_id):
    # A module-level vLLM skip can hide definitions from CPU collection.
    # Parse source to check required test renames without loading an engine.
    assert definition_exists(ROOT, case_id), f"Required vLLM case no longer exists: {case_id}"


@pytest.mark.parametrize(
    "source,case_id,expected",
    [
        ("def test_case(): pass", "sample::test_case", True),
        ("async def test_case(): pass", "sample::test_case", True),
        ("class Cases:\n def test_case(self): pass", "sample.Cases::test_case", True),
        ("def test_renamed(): pass", "sample::test_case", False),
        ("def test_case(): pass", "sample.Cases::test_case", False),
        ("class Renamed:\n def test_case(self): pass", "sample.Cases::test_case", False),
        ("def test_case(): pass", "missing::test_case", False),
    ],
)
def test_definition_lookup_detects_missing_module_class_or_test(
    tmp_path, source, case_id, expected
):
    (tmp_path / "sample.py").write_text(source)
    assert definition_exists(tmp_path, case_id) is expected


@pytest.fixture
def junit(tmp_path):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for name in REQUIRED:
        classname, name = name.split("::", 1)
        ET.SubElement(suite, "testcase", classname=classname, name=name)
    unrelated = ET.SubElement(suite, "testcase", classname="other", name="optional")
    ET.SubElement(unrelated, "skipped")
    return root, suite, tmp_path / "units.xml"


def test_required_passes_allow_unrelated_skips(junit):
    root, _, path = junit
    ET.ElementTree(root).write(path)
    CHECK.require_vllm(path)


@pytest.mark.parametrize(
    "outcome", ["missing", "duplicate", "skipped", "failure", "error", "rerunFailure", "rerunError"]
)
def test_required_case_must_pass_once(junit, outcome):
    root, suite, path = junit
    case = suite[0]
    if outcome == "missing":
        suite.remove(case)
    elif outcome == "duplicate":
        suite.append(copy.deepcopy(case))
    else:
        ET.SubElement(case, outcome)
    ET.ElementTree(root).write(path)
    with pytest.raises(SystemExit, match=REQUIRED[0]):
        CHECK.require_vllm(path)


def test_missing_report_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        CHECK.require_vllm(tmp_path / "absent.xml")


def _gated_tree(tmp_path, source, required):
    tests = tmp_path / "grpc_servicer" / "tests"
    tests.mkdir(parents=True)
    module = tests / "test_sample.py"
    module.write_text(source)
    manifest = tmp_path / "required.json"
    manifest.write_text(json.dumps(required))
    return module, manifest


def test_manifest_matches_every_vllm_gated_source_case():
    assert CHECK.gated_cases(ROOT) == set(REQUIRED)


def test_inventory_detects_an_omitted_whole_gated_module(tmp_path):
    required = ["grpc_servicer.tests.test_other::test_present"]
    module, manifest = _gated_tree(
        tmp_path,
        'import pytest\npytest.importorskip("vllm")\ndef test_missing(): pass\n',
        required,
    )
    module.with_name("test_other.py").write_text(
        'import pytest\ndef test_present():\n pytest.importorskip("vllm")\n'
    )
    with pytest.raises(SystemExit, match="missing=.*test_sample::test_missing"):
        CHECK.load_required(tmp_path, manifest)


def test_parameter_ids_expand_to_junit_names_and_resolve_bare_definitions(tmp_path):
    required = [
        "grpc_servicer.tests.test_sample::test_case[mover]",
        "grpc_servicer.tests.test_sample::test_case[second-p2p-child]",
        "grpc_servicer.tests.test_sample::test_case[empty-engine-id]",
        "grpc_servicer.tests.test_sample::test_case[malformed-sibling]",
    ]
    _, manifest = _gated_tree(
        tmp_path,
        'import pytest\npytest.importorskip("vllm")\n'
        '@pytest.mark.parametrize("value", [1, 2, 3, 4], '
        'ids=["mover", "second-p2p-child", "empty-engine-id", "malformed-sibling"])\n'
        "def test_case(value): pass\n",
        required,
    )
    assert CHECK.gated_cases(tmp_path) == set(required)
    assert CHECK.load_required(tmp_path, manifest) == required
    assert all(definition_exists(tmp_path, case) for case in required)


def test_a_new_function_gated_definition_requires_a_manifest_update(tmp_path):
    required = ["grpc_servicer.tests.test_sample.TestCases::test_existing"]
    module, manifest = _gated_tree(
        tmp_path,
        "import pytest\nclass TestCases:\n def test_existing(self):\n"
        '  pytest.importorskip("vllm")\n',
        required,
    )
    assert CHECK.load_required(tmp_path, manifest) == required
    extra = ' def test_new(self):\n  pytest.importorskip("vllm")\n'
    module.write_text(module.read_text() + extra)
    with pytest.raises(SystemExit, match="missing=.*TestCases::test_new"):
        CHECK.load_required(tmp_path, manifest)


def test_a_module_skip_cannot_hide_the_required_rejection_notice_cases(junit):
    root, suite, path = junit
    module = "grpc_servicer.tests.test_vllm_pd_rejection_notice"
    removed = []
    for case in list(suite):
        if case.get("classname") == module:
            removed.append(case)
            suite.remove(case)
    assert len(removed) == 12
    assert sum("[" in case.get("name", "") for case in removed) == 4
    skipped = ET.SubElement(suite, "testcase", classname=module, name="module import")
    ET.SubElement(skipped, "skipped", message="torch import failed")
    ET.ElementTree(root).write(path)
    with pytest.raises(SystemExit, match="missing, duplicated or not passing") as caught:
        CHECK.require_vllm(path)
    assert all(case.get("name") in str(caught.value) for case in removed)


def test_success_reports_the_actual_manifest_length(tmp_path, monkeypatch, capsys):
    required = [
        "grpc_servicer.tests.test_sample::test_one",
        "grpc_servicer.tests.test_sample::test_two",
        "grpc_servicer.tests.test_sample::test_three",
    ]
    _, manifest = _gated_tree(
        tmp_path,
        'import pytest\npytest.importorskip("vllm")\n'
        "def test_one(): pass\ndef test_two(): pass\ndef test_three(): pass\n",
        required,
    )
    root = ET.Element("testsuite")
    for case in required:
        classname, name = case.split("::", 1)
        ET.SubElement(root, "testcase", classname=classname, name=name)
    path = tmp_path / "units.xml"
    ET.ElementTree(root).write(path)
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)
    monkeypatch.setattr(CHECK, "MANIFEST", manifest)
    CHECK.require_vllm(path)
    assert capsys.readouterr().out == (
        "All 3 required installed-vLLM cases executed once and passed\n"
    )


@pytest.mark.parametrize(
    "parameters",
    [
        '"value", VALUES, ids=["one"]',
        '"value", [1], ids=IDS',
        '"value", [1]',
        '"value", [1, 2], ids=["same", "same"]',
        '"value", [1, 2], ids=["one"]',
        '"value", [pytest.param(1, id="override")], ids=["one"]',
    ],
)
def test_unenumerable_or_duplicate_parameter_ids_fail_closed(tmp_path, parameters):
    _gated_tree(
        tmp_path,
        'import pytest\npytest.importorskip("vllm")\n'
        f"@pytest.mark.parametrize({parameters})\ndef test_case(value): pass\n",
        [],
    )
    with pytest.raises(SystemExit, match="Cannot enumerate required vLLM cases"):
        CHECK.gated_cases(tmp_path)


def test_dynamic_collection_fails_closed(tmp_path):
    _gated_tree(
        tmp_path,
        'import pytest\npytest.importorskip("vllm")\n'
        "def pytest_generate_tests(metafunc): pass\ndef test_case(value): pass\n",
        [],
    )
    with pytest.raises(SystemExit, match="dynamic collection"):
        CHECK.gated_cases(tmp_path)


def test_duplicate_required_manifest_ids_fail_closed(tmp_path):
    case = "grpc_servicer.tests.test_sample::test_case"
    _, manifest = _gated_tree(
        tmp_path,
        'import pytest\npytest.importorskip("vllm")\ndef test_case(): pass\n',
        [case, case],
    )
    with pytest.raises(SystemExit, match="distinct nonempty case IDs"):
        CHECK.load_required(tmp_path, manifest)
