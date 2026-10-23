"""Observe original gateway requests; add no generation or parser override."""

import hashlib
import json
import math
import os
import re
from pathlib import Path

import httpx
import pytest
from reuse import require

NATIVE_HASHES = {}


def counters(text):
    values = {"outputs": 0.0, "clean_outputs": 0.0, "fallback": 0.0, "lines": []}
    for line in text.splitlines():
        for key, metric in (
            ("outputs", "smg_symphony_outputs_total"),
            ("fallback", "smg_symphony_fallback_total"),
        ):
            match = re.fullmatch(re.escape(metric) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
            if match:
                value = float(match[1])
                require(math.isfinite(value) and value >= 0, "invalid parser counter")
                values[key] += value
                if key == "outputs" and re.search(r'(?:\{|,)outcome="clean"(?:,|\})', line):
                    values["clean_outputs"] += value
                values["lines"].append(line)
    return values


def gateway_identity(gateway, expected):
    require(gateway.process is not None and gateway.process.poll() is None, "gateway not alive")
    pid = gateway.process.pid
    root = Path("/proc") / str(pid)
    environment = (root / "environ").read_bytes().split(b"\0")
    marker = "E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]
    require(marker.encode() in environment, "gateway not owned by this job")
    argv = [value.decode() for value in (root / "cmdline").read_bytes().split(b"\0") if value]
    launch = list(gateway.process.args)
    require(argv == launch or argv == ["smg::router"], "actual gateway argv/title differs")
    require(
        launch.count("--parser") == 1 and launch[launch.index("--parser") + 1] == "symphony",
        "harness requested parser differs",
    )
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    parent_pid = int(fields[1])
    require(parent_pid == os.getpid(), "gateway is not a child of this pytest harness")
    start = fields[19]
    native = set()
    for line in (root / "maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "x" in fields[1] and Path(fields[5]).name.startswith("smg_rs"):
            native.add(Path(fields[5]).resolve())
    require(len(native) == 1, "missing or ambiguous loaded gateway native extension")
    path = native.pop()
    info = path.stat()
    key = (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if key not in NATIVE_HASHES:
        with path.open("rb") as stream:
            NATIVE_HASHES[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    digest = NATIVE_HASHES[key]
    require(digest == expected["sha256"], "loaded gateway native bytes differ from producer")
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "start_ticks": start,
        "argv": argv,
        "launch_argv_intent": launch,
        "native": str(path),
        "sha256": digest,
        "marker_visible_in_proc": True,
    }


class ParserProof:
    def __init__(self, output, recipe):
        self.output, self.recipe = output, recipe
        self.records, self.errors = [], []
        self.native = json.loads(Path(__file__).with_name("inputs.json").read_text())[
            "parser_native"
        ]

    def save(self):
        (self.output / "parser-proof.json").write_text(
            json.dumps({"records": self.records, "errors": self.errors}, indent=2) + "\n"
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        backend = item.funcargs.get("setup_backend")
        record = None
        if backend:
            try:
                from fixtures.markers import model_id_for_engine, resolve_class_marker

                model = model_id_for_engine(resolve_class_marker(item, "model"), "vllm")
                gateway = backend[3]
                response = httpx.get(gateway.metrics_url + "/metrics", timeout=3)
                response.raise_for_status()
                record = {
                    "nodeid": item.nodeid,
                    "model": model,
                    "gateway": gateway_identity(gateway, self.native),
                    "before": counters(response.text),
                    "metrics_url": gateway.metrics_url,
                }
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "stage": "before", "error": str(error)})
        yield
        if record is not None:
            try:
                response = httpx.get(record["metrics_url"] + "/metrics", timeout=3)
                response.raise_for_status()
                require(
                    gateway_identity(gateway, self.native) == record["gateway"],
                    "gateway process identity changed during original case",
                )
                record["after"] = counters(response.text)
                record["delta"] = {
                    key: record["after"][key] - record["before"][key]
                    for key in ("outputs", "clean_outputs", "fallback")
                }
                require(all(value >= 0 for value in record["delta"].values()), "counter reset")
                self.records.append(record)
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "stage": "after", "error": str(error)})
        self.save()

    def verify(self):
        self.save()
        require(not self.errors, "incomplete actual parser observations: " + repr(self.errors))
        require(
            {row["model"] for row in self.records} == set(self.recipe["models"]),
            "actual parser gateway witness missing for a model",
        )
        tool_records = [
            row
            for row in self.records
            if row["nodeid"] in self.recipe["tool_cases"].get(row["model"], [])
        ]
        require(
            any(row["delta"]["clean_outputs"] > 0 for row in tool_records),
            "no actual clean Symphony tool output",
        )
        for model in self.recipe["tool_cases"]:
            observed = [row for row in tool_records if row["model"] == model]
            require(
                any(
                    row["delta"]["clean_outputs"] > 0 or row["delta"]["fallback"] > 0
                    for row in observed
                ),
                "applicable tool model has no observed table or fallback path: " + model,
            )
