"""Check passive original SG-ZMQ evidence; never execute or replace a test."""
import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from junit_join import join_unit_junit

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def final_chat_attempts(observed, expected):
    """Retain all original rerun reports, and join only each final attempt to XML."""
    normalized = dict(observed, reports={})
    reruns = {}
    for item in observed["selected"]:
        history = observed["reports"][item["pytest_nodeid"]]
        attempts = []
        for report in history:
            if report["when"] == "setup":
                attempts.append([])
            assert attempts and not report["xfail"]
            attempts[-1].append(report)
        assert 1 <= len(attempts) <= 3, "original reruns2 limit"
        for attempt in attempts:
            assert [x["when"] for x in attempt] in (["setup", "call", "teardown"], ["setup", "teardown"])
        for attempt in attempts[:-1]:
            assert any(x["outcome"] == "rerun" for x in attempt), "unexplained extra attempt"
        final = attempts[-1]
        skipped = [x for x in final if x["outcome"] == "skipped"]
        reason = expected[item["nodeid"]]["skip_reason"]
        if reason is None:
            assert not skipped, "new runtime skip: " + item["nodeid"]
        else:
            assert len(skipped) == 1 and skipped[0]["junit_skip"]["attributes"] == {
                "type": "pytest.skip", "message": reason}, "original skip reason changed"
        normalized["reports"][item["pytest_nodeid"]] = final
        reruns[item["nodeid"]] = len(attempts) - 1
    return normalized, reruns



def final_chat_xml(observed, suites, expected):
    """Derive a final-attempt view only after exact raw-history/XML closure."""
    normalized, reruns = final_chat_attempts(observed, expected)
    assert suites.tag == "testsuites" and len(suites) == 1 and suites[0].tag == "testsuite"
    suite = suites[0]
    assert not list(suites.iter("failure")) and not list(suites.iter("error"))
    cases = list(suite.findall("testcase"))
    assert len(cases) == sum(count + 1 for count in reruns.values()), "raw attempt/XML count"
    assert len(list(suites.iter("testcase"))) == len(cases), "nested/unknown testcase"
    for attribute in ("tests", "errors", "failures", "skipped"):
        assert suite.attrib[attribute].isdecimal(), "invalid suite count"
    all_reports = [report for history in observed["reports"].values() for report in history]
    # Pytest counts passed calls, including a call before a teardown rerun.
    passed_calls = sum(x["when"] == "call" and x["outcome"] == "passed" for x in all_reports)
    skipped_reports = sum(x["outcome"] == "skipped" for x in all_reports)
    assert int(suite.attrib["tests"]) == passed_calls + skipped_reports, "raw suite tests/history"
    assert int(suite.attrib["skipped"]) == skipped_reports, "raw suite skips/history"
    assert int(suite.attrib["errors"]) == int(suite.attrib["failures"]) == 0
    cursor, final_indices, audit = 0, [], []
    for item in observed["selected"]:
        attempts = []
        for report in observed["reports"][item["pytest_nodeid"]]:
            if report["when"] == "setup":
                attempts.append([])
            attempts[-1].append(report)
        assert len(attempts) == reruns[item["nodeid"]] + 1
        for index, attempt in enumerate(attempts):
            assert all(x["outcome"] in {"passed", "skipped", "rerun"} and not x["xfail"] for x in attempt)
            if index < len(attempts) - 1:
                assert sum(x["outcome"] == "rerun" for x in attempt) == 1, "ambiguous intermediate attempt"
                assert all(x["longrepr"] for x in attempt if x["outcome"] == "rerun")
            else:
                assert not any(x["outcome"] == "rerun" for x in attempt)
                final_indices.append(cursor)
            case = cases[cursor]
            assert {k: case.attrib[k] for k in ("classname", "name")} == item["junit_address"], "raw XML address/order"
            assert all(child.tag in {"properties", "system-out", "system-err", "skipped"} for child in case), "unknown XML outcome"
            skipped = [x for x in attempt if x["outcome"] == "skipped"]
            actual_skip = case.findall("skipped")
            assert len(skipped) == len(actual_skip) <= 1, "raw attempt/XML skip"
            if skipped:
                proof = skipped[0]["junit_skip"]
                assert proof and not proof["truncated"]
                assert expected[item["nodeid"]]["skip_reason"] == proof["attributes"]["message"]
                assert actual_skip[0].attrib == proof["attributes"] and (actual_skip[0].text or "") == proof["text"]
            audit.append({"nodeid": item["nodeid"], "attempt": index,
                "outcomes": [x["outcome"] for x in attempt], "XML_index": cursor,
                "XML_element_sha256": hashlib.sha256(ET.tostring(case)).hexdigest(),
                "final": index == len(attempts) - 1})
            cursor += 1
    assert cursor == len(cases)
    derived = copy.deepcopy(suites)
    final_set = set(final_indices)
    for index, case in enumerate(list(derived[0].findall("testcase"))):
        if index not in final_set:
            derived[0].remove(case)
    derived[0].set("tests", str(len(final_indices)))
    derived[0].set("skipped", str(sum(bool(x.findall("skipped")) for x in derived[0].findall("testcase"))))
    return normalized, reruns, derived, {"raw_testcase_elements": len(cases),
        "raw_suite_tests": int(suite.attrib["tests"]), "final_testcase_elements": len(final_indices),
        "raw_attempt_XML_bijection": True, "derived_final_view_only": True, "attempts": audit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("units", "chat"))
    args = parser.parse_args()
    assert __debug__ and os.environ.get("SMG_SGLANG_ZMQ_ORIGINAL") == "1"
    root = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    directory = Path(os.environ["SG_ZMQ_PROOF_DIRECTORY"])
    assert directory.resolve(strict=True) == directory
    assert directory.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve(strict=True))
    result = {"result": "FAIL", "phase": args.phase, "GPU_row_accepted": False}
    try:
        contract = json.loads((HERE / "contract.json").read_text())
        for relative, expected in contract["source_files"].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
        observed = json.loads((directory / (args.phase + "-reports.json")).read_text())
        assert observed["exit_code"] == 0 and observed["phase"] == args.phase
        assert observed["source"] == str(root)
        record = directory / (args.phase + "-pytest-asyncio-RECORD.csv")
        assert hashlib.sha256(record.read_bytes()).hexdigest() == observed["pytest_asyncio"]["RECORD_text_sha256"]
        assert len(observed["selected"]) == len({x["nodeid"] for x in observed["selected"]})
        assert set(observed["reports"]) == {x["pytest_nodeid"] for x in observed["selected"]}
        required = set()
        if args.phase == "units":
            for definition in contract["unit_definitions"]:
                prefix = definition["module"] + "::" + definition["name"]
                matches = [x["nodeid"] for x in observed["selected"] if x["nodeid"].split("[", 1)[0] == prefix]
                assert matches, "missing original SG definition: " + prefix
                required.update(matches)
            value, reruns = observed, None
            xml = Path(os.environ["RUNNER_TEMP"]) / "sglang-zmq-units.xml"
        else:
            units = json.loads((directory / "units-reports.json").read_text())
            assert observed["pytest_asyncio"] == units["pytest_asyncio"], "async plugin changed between original commands"
            expected = {x["id"]: x for x in contract["chat_cases"]}
            assert [x["nodeid"] for x in observed["selected"]] == list(expected)
            for item in observed["selected"]:
                view = {k: item[k] for k in ("nodeid", "model", "params", "skip_markers")}
                assert digest(view) == expected[item["nodeid"]]["sha256"], "original case/model/params/skip changed"
            assert dict(Counter(x["model"] for x in observed["selected"])) == contract["model_counts"]
            assert observed["collection_reports"] == []
            value, reruns = observed, None
            required = {x for x in expected if expected[x]["skip_reason"] is None}
            xml = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "e2e.xml"
        xml_root = ET.parse(xml).getroot()
        if args.phase == "chat":
            value, reruns, xml_root, rerun_XML = final_chat_xml(observed, xml_root, expected)
            result.update(rerun_XML=rerun_XML, raw_XML_sha256=hashlib.sha256(xml.read_bytes()).hexdigest())
        joined = join_unit_junit(value, xml_root, required)
        result.update(result="PASS_ORIGINAL_SG_ZMQ_" + args.phase.upper() + "_EVIDENCE_ONLY",
            selected=len(observed["selected"]), required_instances=len(required),
            required_source_definitions=len(contract["unit_definitions"]) if args.phase == "units" else None,
            original_chat_cases=contract["original_chat_cases"] if args.phase == "chat" else None,
            reruns=reruns, **joined)
    except BaseException as error:
        result["first_failure"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        raise
    finally:
        with (directory / (args.phase + "-checked.json")).open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
