"""Check passive original SG-ZMQ evidence; never execute or replace a test."""
import argparse
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
            value, reruns = final_chat_attempts(observed, expected)
            required = {x for x in expected if expected[x]["skip_reason"] is None}
            xml = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "e2e.xml"
        joined = join_unit_junit(value, ET.parse(xml).getroot(), required)
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
