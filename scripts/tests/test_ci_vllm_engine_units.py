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
    # Several modules importorskip vLLM at import time; CPU collection would
    # silently miss renamed tests. Parse their source without loading an engine.
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
