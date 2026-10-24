"""Check individual JUnit outcomes without importing an engine."""

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
