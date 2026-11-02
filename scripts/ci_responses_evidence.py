"""Validate pytest phase reports and JUnit XML against 128 reference Responses cases."""

import json
import os
import sys
import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path

import pytest

ORIGINAL_SHA256 = "3a62d0478235567d544b0108fb653403ffd4f34c2b8ccd8b640f3a706522ff6a"
DIRECTORY = Path(os.environ.get("RESPONSES_EVIDENCE_DIR", "e2e-logs/responses"))
REPORTS = []


def save(name, value):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    save(
        "selected.json",
        {
            "nodeids": [item.nodeid for item in session.items],
            "rootpath": str(session.config.rootpath),
            "source_root": str(session.config.invocation_params.dir.resolve()),
            "inipath": str(session.config.inipath),
            "evidence_scope": "final session.items after all selection; no fixture or outcome changes",
        },
    )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    REPORTS.append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome})


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    save("outcomes.json", {"pytest_exitstatus": int(exitstatus), "reports": REPORTS})


def junit_identity(nodeid):
    path, *parts = nodeid.split("::")
    if not path.endswith(".py") or not parts:
        raise ValueError(f"Unexpected pytest identity: {nodeid!r}")
    module = path[:-3].replace("/", ".")
    return (".".join([module, *parts[:-1]]), parts[-1])


def verify(directory, xml_path, original_path):
    global DIRECTORY
    DIRECTORY = directory
    errors = []
    result = {
        "business_verified": False,
        "scope": "Validate pytest phases and JUnit XML against the 128 reference Responses cases",
    }
    try:
        for path in [
            directory / "selected.json",
            directory / "outcomes.json",
            xml_path,
            original_path,
        ]:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or symlinked evidence: {path}")
        selection = json.loads((directory / "selected.json").read_text())
        selected = selection["nodeids"]
        if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
            raise ValueError(f"{directory / 'selected.json'}: nodeids must be a list of strings")
        source_root, rootpath = Path(selection["source_root"]), Path(selection["rootpath"])
        if not source_root.is_absolute() or not rootpath.is_absolute():
            raise ValueError("Collection source root and pytest rootpath must be absolute")
        root_prefix = rootpath.resolve().relative_to(source_root.resolve())
        original_comparison = []
        for nodeid in selected:
            path, *parts = nodeid.split("::")
            if Path(path).is_absolute() or any(part in ("", ".", "..") for part in path.split("/")):
                raise ValueError(
                    f"{directory / 'selected.json'}: unexpected raw pytest path: {nodeid!r}"
                )
            original_comparison.append("::".join([(root_prefix / path).as_posix(), *parts]))
        result.update(
            source_root=str(source_root),
            rootpath=str(rootpath),
            original_reference_comparison_nodeids=original_comparison,
            identity_scope=(
                "Only historical Original128 subset uses source-root-relative paths; "
                "selected/JUnit/phases retain raw identities"
            ),
        )
        actual = json.loads((directory / "outcomes.json").read_text())
        reports = actual["reports"]
        if not isinstance(reports, list):
            raise ValueError(f"{directory / 'outcomes.json'}: reports must be a list of objects")
        for index, report in enumerate(reports):
            if not isinstance(report, dict):
                raise ValueError(
                    f"{directory / 'outcomes.json'}: reports[{index}] must be an object"
                )
            for field in ("nodeid", "when", "outcome"):
                if not isinstance(report.get(field), str):
                    raise ValueError(
                        f"{directory / 'outcomes.json'}: reports[{index}].{field} must be a string"
                    )
        original = json.loads(original_path.read_text())
        if not isinstance(original, list) or any(not isinstance(item, str) for item in original):
            raise ValueError(f"{original_path}: reference identities must be a list of strings")
        if len(original) != 128 or len(set(original)) != 128:
            errors.append("Original128 identity data is not exact and unique")
        if sha256(("\n".join(sorted(original)) + "\n").encode()).hexdigest() != ORIGINAL_SHA256:
            errors.append("Original128 identity fingerprint mismatch")
        if len(selected) < 128 or len(selected) != len(set(selected)):
            errors.append("Final selection must contain >=128 unique identities")
        missing = sorted(set(original) - set(original_comparison))
        if missing:
            errors.append(
                "Original128 identities are missing from final selection: "
                f"{missing}; reference list: {original_path}"
            )
        wanted = {junit_identity(nodeid) for nodeid in selected}
        if len(wanted) != len(selected):
            errors.append("Final selection has ambiguous JUnit identities")
        cases = list(ET.parse(xml_path).getroot().iter("testcase"))
        identities = {(case.get("classname"), case.get("name")) for case in cases}
        if identities != wanted:
            errors.append("Actual JUnit identities differ from complete final selection")
        if len(cases) != len(selected) or len(identities) != len(cases):
            errors.append("Actual JUnit must contain exactly one testcase per selected identity")
        bad_tags = {
            "skipped",
            "failure",
            "error",
            "rerunFailure",
            "rerunError",
            "flakyFailure",
            "flakyError",
        }
        if any(node.tag in bad_tags for case in cases for node in case.iter()):
            errors.append("Actual JUnit contains skipped/failed/error records")
        if {report["nodeid"] for report in reports} != set(selected):
            errors.append("Phase-report identities differ from complete final selection")
        if actual["pytest_exitstatus"] != 0:
            errors.append("Actual pytest exit status is nonzero")
        if len(reports) != 3 * len(selected):
            errors.append("Expected exactly three phase reports per selected identity")
        for nodeid in selected:
            phases = {}
            for report in reports:
                if report["nodeid"] == nodeid:
                    phase, outcome = report["when"], report["outcome"]
                    if phase not in ("setup", "call", "teardown"):
                        errors.append(
                            f"A selected case has an unexpected phase report: {nodeid} ({phase})"
                        )
                    if phase in phases:
                        errors.append(
                            f"A selected case has duplicate phase reports: {nodeid} ({phase})"
                        )
                    else:
                        phases[phase] = outcome
                    if outcome != "passed":
                        errors.append(
                            "A selected case has a non-passing phase report: "
                            f"{nodeid} ({phase}={outcome})"
                        )
                    if outcome == "skipped":
                        errors.append(f"An actually selected case skipped: {nodeid} ({phase})")
            if any(phases.get(phase) != "passed" for phase in ["setup", "call", "teardown"]):
                errors.append("A selected case lacks final passing setup/call/teardown: " + nodeid)
        result.update(
            original_required=128,
            final_selected=len(selected),
            junit_records=len(cases),
            junit_unique=len(identities),
            phase_records=len(reports),
            pytest_exitstatus=actual["pytest_exitstatus"],
        )
    except (OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
        errors.append(f"Evidence validation failed: {type(error).__name__}: {error}")
    result["errors"] = sorted(set(errors))
    result["business_verified"] = not errors
    try:
        save("validation.json", result)
    except OSError as error:
        result["business_verified"] = False
        result["errors"].append(f"Cannot write validation report: {type(error).__name__}: {error}")
    print(json.dumps(result))
    return 0 if result["business_verified"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "verify":
        raise SystemExit("usage: ci_responses_evidence.py verify DIRECTORY JUNIT ORIGINAL128_JSON")
    raise SystemExit(verify(*map(Path, sys.argv[2:])))
