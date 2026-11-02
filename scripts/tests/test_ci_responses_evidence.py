"""Exercise the Responses evidence CLI using complete synthetic reference cases."""

import copy
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
VALIDATOR = SCRIPTS / "ci_responses_evidence.py"
ORIGINAL_PATH = SCRIPTS / "ci_responses_original128.json"
ORIGINAL_FINGERPRINT = "3a62d0478235567d544b0108fb653403ffd4f34c2b8ccd8b640f3a706522ff6a"
EXTRA = "responses/test_current_extra.py::test_current_extra"


@pytest.fixture
def evidence(tmp_path):
    original = json.loads(ORIGINAL_PATH.read_text())
    assert len(original) == len(set(original)) == 128
    assert sha256(("\n".join(sorted(original)) + "\n").encode()).hexdigest() == (
        ORIGINAL_FINGERPRINT
    )
    source = tmp_path / "source"
    (source / "e2e_test").mkdir(parents=True)
    directory = tmp_path / "evidence"
    directory.mkdir()
    return {
        "directory": directory,
        "source": source,
        "original": original,
        "selection": {
            "nodeids": [nodeid.removeprefix("e2e_test/") for nodeid in original],
            "source_root": str(source),
            "rootpath": str(source / "e2e_test"),
            "inipath": "None",
        },
    }


def complete_records(evidence):
    """Create all three phases and one JUnit record for every selected identity."""
    selected = evidence["selection"]["nodeids"]
    suite = ET.Element("testsuite", tests=str(len(selected)))
    for nodeid in selected:
        module, *parts = nodeid.split("::")
        classname = ".".join([module.removesuffix(".py").replace("/", "."), *parts[:-1]])
        ET.SubElement(suite, "testcase", classname=classname, name=parts[-1])
    evidence["xml"] = suite
    evidence["outcomes"] = {
        "pytest_exitstatus": 0,
        "reports": [
            {"nodeid": nodeid, "when": phase, "outcome": "passed"}
            for nodeid in selected
            for phase in ("setup", "call", "teardown")
        ],
    }


def run_validator(evidence):
    directory = evidence["directory"]
    xml_path = directory / "responses.xml"
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(VALIDATOR),
            "verify",
            str(directory),
            str(xml_path),
            str(ORIGINAL_PATH),
        ],
        cwd=evidence["source"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def verify(evidence, error=None):
    directory = evidence["directory"]
    for name, value in (
        ("selected.json", evidence["selection"]),
        ("outcomes.json", evidence["outcomes"]),
    ):
        (directory / name).write_text(json.dumps(value) + "\n")
    xml_path = directory / "responses.xml"
    ET.ElementTree(evidence["xml"]).write(xml_path, encoding="utf-8", xml_declaration=True)
    child = run_validator(evidence)
    report = json.loads((directory / "validation.json").read_text())
    assert json.loads(child.stdout) == report
    assert child.returncode == (0 if error is None else 1), child.stdout + child.stderr
    assert report["business_verified"] is (error is None)
    if error is None:
        assert report["errors"] == []
        assert report["original_required"] == 128
        assert report["final_selected"] == len(evidence["selection"]["nodeids"])
        assert report["junit_records"] == report["junit_unique"] == report["final_selected"]
        assert report["phase_records"] == 3 * report["final_selected"]
    else:
        assert any(
            message == error or message.startswith(error + ":") for message in report["errors"]
        )
    return report


@pytest.mark.parametrize("layout", ["repo-root", "e2e-root", "current-extra"])
def test_complete_original128_accepts_both_roots_and_all_current_extras(evidence, layout):
    if layout == "repo-root":
        evidence["selection"]["nodeids"] = list(evidence["original"])
        evidence["selection"]["rootpath"] = str(evidence["source"])
    elif layout == "current-extra":
        evidence["selection"]["nodeids"].append(EXTRA)
    complete_records(evidence)
    verify(evidence)


@pytest.mark.parametrize(
    "scenario,error",
    [
        ("wrong-prefix", "Original128 identities are missing from final selection"),
        ("outside-source-root", "Evidence validation failed: ValueError"),
        ("duplicate-selected", "Final selection must contain >=128 unique identities"),
        ("skip-record", "Actual JUnit contains skipped/failed/error records"),
        ("error-record", "Actual JUnit contains skipped/failed/error records"),
        ("missing-original-same-count", "Original128 identities are missing from final selection"),
        ("raw-JUnit-mismatch", "Actual JUnit identities differ from complete final selection"),
        ("raw-phase-mismatch", "Phase-report identities differ from complete final selection"),
        ("duplicate-JUnit", "Actual JUnit must contain exactly one testcase per selected identity"),
        ("failed-call-then-retry-pass", "A selected case has a non-passing phase report"),
    ],
)
def test_invalid_complete_evidence_is_rejected(evidence, scenario, error):
    selection = evidence["selection"]
    if scenario == "wrong-prefix":
        selection["nodeids"] = list(evidence["original"])
    elif scenario == "outside-source-root":
        selection["rootpath"] = str(evidence["source"].parent / "outside")
    elif scenario == "duplicate-selected":
        selection["nodeids"].append(selection["nodeids"][0])
    elif scenario == "missing-original-same-count":
        selection["nodeids"][-1] = EXTRA
    complete_records(evidence)
    cases = list(evidence["xml"])
    if scenario in ("skip-record", "error-record"):
        ET.SubElement(cases[0], "skipped" if scenario == "skip-record" else "error")
    elif scenario == "raw-JUnit-mismatch":
        for case in cases:
            case.set("classname", "e2e_test." + case.get("classname"))
    elif scenario == "raw-phase-mismatch":
        for report in evidence["outcomes"]["reports"]:
            report["nodeid"] = "e2e_test/" + report["nodeid"]
    elif scenario == "duplicate-JUnit":
        evidence["xml"].append(copy.deepcopy(cases[-1]))
    elif scenario == "failed-call-then-retry-pass":
        failed_attempt = copy.deepcopy(evidence["outcomes"]["reports"][:3])
        failed_attempt[1]["outcome"] = "failed"
        evidence["outcomes"]["reports"] = failed_attempt + evidence["outcomes"]["reports"]
    verify(evidence, error)


@pytest.mark.parametrize(
    "tag", ["failure", "rerunFailure", "rerunError", "flakyFailure", "flakyError"]
)
def test_junit_failure_or_retry_markers_are_not_a_clean_pass(evidence, tag):
    complete_records(evidence)
    ET.SubElement(evidence["xml"][0], tag)
    verify(evidence, "Actual JUnit contains skipped/failed/error records")


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_any_failed_phase_rejects_even_when_exit_and_junit_claim_pass(evidence, phase):
    complete_records(evidence)
    report = next(item for item in evidence["outcomes"]["reports"] if item["when"] == phase)
    report["outcome"] = "failed"
    verify(evidence, "A selected case has a non-passing phase report")


def test_nonzero_actual_exit_rejects_all_passing_records(evidence):
    complete_records(evidence)
    evidence["outcomes"]["pytest_exitstatus"] = 1
    verify(evidence, "Actual pytest exit status is nonzero")


def test_missing_call_phase_is_not_accepted(evidence):
    complete_records(evidence)
    del evidence["outcomes"]["reports"][1]
    verify(evidence, "A selected case lacks final passing setup/call/teardown")


def test_unexpected_phase_is_not_accepted(evidence):
    complete_records(evidence)
    evidence["outcomes"]["reports"][0]["when"] = "rerun"
    verify(evidence, "A selected case has an unexpected phase report")


def test_repeated_passing_phase_does_not_replace_missing_teardown(evidence):
    complete_records(evidence)
    evidence["outcomes"]["reports"][2]["when"] = "setup"
    verify(evidence, "A selected case has duplicate phase reports")


def test_current_extra_requires_its_own_junit_and_phase_evidence(evidence):
    complete_records(evidence)
    evidence["selection"]["nodeids"].append(EXTRA)
    verify(evidence, "Actual JUnit identities differ from complete final selection")


def test_missing_evidence_reports_its_path(evidence):
    complete_records(evidence)
    verify(evidence)
    directory = evidence["directory"]
    (directory / "validation.json").unlink()
    missing = directory / "selected.json"
    missing.unlink()
    child = run_validator(evidence)
    report = json.loads(child.stdout)
    assert child.returncode == 1 and not report["business_verified"]
    assert any(str(missing) in message for message in report["errors"])
    assert "Traceback" not in child.stderr


def test_failed_phases_identify_each_case(evidence):
    complete_records(evidence)
    selected = evidence["selection"]["nodeids"][:2]
    for item in evidence["outcomes"]["reports"]:
        if item["nodeid"] in selected and item["when"] == "call":
            item["outcome"] = "failed"
    report = verify(evidence, "A selected case has a non-passing phase report")
    failures = [message for message in report["errors"] if "non-passing phase report" in message]
    assert len(failures) == 2
    for nodeid in selected:
        assert any(nodeid in message and "call=failed" in message for message in failures)


def test_existing_validation_report_fails_clearly_without_overwriting(evidence):
    complete_records(evidence)
    verify(evidence)
    output = evidence["directory"] / "validation.json"
    retained = output.read_bytes()
    child = run_validator(evidence)
    report = json.loads(child.stdout)
    assert child.returncode == 1 and not report["business_verified"]
    assert any(
        message.startswith("Cannot write validation report:") for message in report["errors"]
    )
    assert output.read_bytes() == retained
    assert "Traceback" not in child.stderr


def test_missing_reference_case_names_the_id_and_manifest(evidence):
    missing = evidence["selection"]["nodeids"][-1]
    evidence["selection"]["nodeids"][-1] = EXTRA
    complete_records(evidence)
    report = verify(evidence, "Original128 identities are missing from final selection")
    assert any(missing in message and str(ORIGINAL_PATH) in message for message in report["errors"])
