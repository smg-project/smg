"""Run the original router suite using recorded cached inputs and terminal reports."""

import hashlib
import importlib
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from reuse import require

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostic-results"
CASES = Path(__file__).with_name("cases.json")
DATA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"


def cached_models(specs, cache):
    records = []
    for name, resolved in specs.items():
        directory = cache / ("models--" + name.replace("/", "--"))
        revision = (directory / "refs/main").read_text().strip()
        require(re.fullmatch(r"[0-9a-f]{40}", revision), "unknown model revision")
        snapshot = directory / "snapshots" / revision
        require(resolved == name or Path(resolved).resolve() == snapshot.resolve(), "model path")
        config = snapshot / "config.json"
        index = snapshot / "model.safetensors.index.json"
        names = (
            set(json.loads(index.read_text())["weight_map"].values())
            if index.is_file()
            else {"model.safetensors"}
        )
        require(names and all(Path(n).name == n for n in names), "unsafe/empty model shard list")
        for path in [config, *(snapshot / name for name in names)]:
            require(path.is_file() and path.stat().st_size > 0, "missing cached model file")
            require(path.resolve().is_relative_to(cache.resolve()), "model file escapes cache")
        records.append(
            {
                "model": name,
                "snapshot": revision,
                "shards": sorted(names),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "full_weight_hashes_verified": False,
            }
        )
    return records


def cache_dataset(module):
    require(module.MMLU_DATASET_URL == DATA_URL, "MMLU source URL changed")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        data = response.read(64 * 1024 * 1024 + 1)
    require(0 < len(data) <= 64 * 1024 * 1024, "MMLU dataset size")
    path = OUT / "mmlu.csv"
    path.write_bytes(data)
    (OUT / "mmlu-input.json").write_text(
        json.dumps(
            {"url": DATA_URL, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    )
    # MMLUEval's existing filename input accepts this exact downloaded CSV.
    module.MMLU_DATASET_URL = str(path)


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
        expected = set()
        for nodeid in recipe["cases"]:
            module, cls, name = nodeid.split("::", 2)
            expected.add((module.removesuffix(".py").replace("/", ".") + "." + cls, name))
        actual = {(c.get("classname"), c.get("name")) for c in cases}
        require(actual == expected, "JUnit case identity")
        require(
            not any(c.find("failure") is not None or c.find("error") is not None for c in cases),
            "JUnit failure",
        )


def main():
    engine = os.environ["E2E_ENGINE"]
    recipe = json.loads(CASES.read_text())[engine]
    from infra.model_specs import get_model_spec

    models = cached_models(
        {name: get_model_spec(name)["model"] for name in recipe["models"]},
        Path(os.environ["HF_HOME"]) / "hub",
    )
    (OUT / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    cache_dataset(importlib.import_module("infra.run_eval"))
    reports = Reports(OUT / "case-reports.jsonl")
    junit = OUT / "router.xml"
    result = pytest.main(
        [
            "e2e_test/router",
            "--ignore=e2e_test/benchmarks",
            "--ignore=e2e_test/bindings_go",
            "-p",
            "no:rerunfailures",
            "-s",
            "-vv",
            "--junitxml=" + str(junit),
        ],
        plugins=[reports],
    )
    reports.verify(recipe, junit)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
