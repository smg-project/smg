"""Run every original worker-side multimodal case and verify all terminal phases."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from reuse import require

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostic-results"
CASES = Path(__file__).with_name("cases.json")


class Reports:
    def __init__(self, path):
        self.path = path
        self.items = {}

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
    require(os.environ["E2E_MM_PROCESSING"] == "worker", "wrong processing location")
    require(os.environ["E2E_VLLM_MM_PROCESSOR"] == "inprocess", "wrong media processor")
    recipe = json.loads(CASES.read_text())["multimodal"]
    os.environ["E2E_MIN_SELECTED"] = str(recipe["min_selected"])
    output = OUT / "multimodal"
    output.mkdir()
    reports = Reports(output / "case-reports.jsonl")
    junit = output / "result.xml"
    result = pytest.main(
        [
            recipe["test_dir"],
            "--ignore=e2e_test/benchmarks",
            "--ignore=e2e_test/bindings_go",
            "--reruns", "0",
            "-s", "-vv",
            "--junitxml=" + str(junit),
        ],
        plugins=[reports],
    )
    reports.verify(recipe, junit)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
