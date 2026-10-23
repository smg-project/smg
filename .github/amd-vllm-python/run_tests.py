"""Run the complete original Python-servicer chat row with strict terminal accounting."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from python_proof import PythonProof
from reuse import require
from selection import capture

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostic-results"
CASES = Path(__file__).with_name("cases.json")


class Reports:
    def __init__(self, path, recipe=None):
        self.path = path
        self.items = {}
        self.recipe = recipe

    def pytest_collection_finish(self, session):
        if self.recipe is not None:
            capture(session.items, self.recipe, self.path.parent / "actual-selection.json")

    def pytest_runtest_logreport(self, report):
        item = {
            "phase": report.when,
            "outcome": report.outcome,
            "xfail": hasattr(report, "wasxfail"),
            "detail": str(report.longrepr) if report.longrepr else "",
        }
        self.items.setdefault(report.nodeid, []).append(item)
        with self.path.open("a") as output:
            output.write(json.dumps({"nodeid": report.nodeid, **item}) + "\n")

    def verify(self, recipe, junit):
        require(set(self.items) == set(recipe["cases"]), "missing/unexpected terminal case IDs")
        for nodeid, reports in self.items.items():
            require(not any(r["xfail"] for r in reports), "xfail is not acceptance")
            phases = {r["phase"]: r for r in reports}
            require(len(phases) == len(reports), "duplicate phase/rerun")
            allowed = recipe["allowed_skips"].get(nodeid)
            if phases.get("setup", {}).get("outcome") == "skipped":
                require(allowed and allowed in phases["setup"]["detail"], "unexpected skip")
                require(set(phases) == {"setup", "teardown"}, "skipped case lifecycle")
                require(phases["teardown"]["outcome"] == "passed", "skip teardown failure")
            else:
                require(set(phases) == {"setup", "call", "teardown"}, "incomplete case lifecycle")
                require(all(r["outcome"] == "passed" for r in reports), "case failure")
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        require(len(cases) == len(recipe["cases"]), "JUnit missing/duplicate cases")
        expected = {}
        for nodeid in recipe["cases"]:
            module, cls, name = nodeid.split("::", 2)
            key = (module.removesuffix(".py").replace("/", ".") + "." + cls, name)
            expected[key] = self.items[nodeid][0]["outcome"] == "skipped"
        actual = {(c.get("classname"), c.get("name")) for c in cases}
        require(actual == set(expected), "JUnit case identity")
        require(
            all(
                (c.find("skipped") is not None) == expected[(c.get("classname"), c.get("name"))]
                for c in cases
            ),
            "JUnit skip status differs from terminal reports",
        )
        require(
            not any(c.find("failure") is not None or c.find("error") is not None for c in cases),
            "JUnit failure",
        )


def main():
    require(os.environ["E2E_ENGINE"] == "vllm", "unexpected engine")
    require(not os.environ.get("E2E_VLLM_SERVICER_IMPL"), "wrong servicer lane")
    require(os.environ.get("SPT_NOENV") == "1", "process marker preservation missing")
    require(not os.environ.get("E2E_MM_PROCESSING"), "original processing lane changed")
    require(not os.environ.get("E2E_CONNECTION_MODE"), "original connection lane changed")
    recipe = json.loads(CASES.read_text())["chat"]
    os.environ["E2E_MIN_SELECTED"] = str(recipe["min_selected"])
    output = OUT / "chat"
    output.mkdir()
    reports = Reports(output / "case-reports.jsonl", recipe)
    junit = output / "result.xml"
    proof = PythonProof(output, recipe)
    result = pytest.main(
        [
            recipe["test_dir"],
            "--ignore=e2e_test/benchmarks",
            "--ignore=e2e_test/bindings_go",
            "--reruns",
            "0",
            "-s",
            "-vv",
            "--junitxml=" + str(junit),
        ],
        plugins=[reports, proof],
    )
    reports.verify(recipe, junit)
    proof.verify()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
