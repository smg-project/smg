"""Passive observations for the private original SG-ZMQ unit and chat commands."""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import sys
import _pytest.junitxml as junitxml

selected, reports, collection = [], {}, []
prefix = phase = directory = root = None
asyncio = None


def pytest_configure(config):
    global prefix, phase, directory, root, asyncio
    assert __debug__ and os.environ.get("SMG_SGLANG_ZMQ_ORIGINAL") == "1"
    assert os.environ["E2E_ENGINE"] == "sglang" and os.environ["E2E_CONNECTION_MODE"] == "zmq"
    phase = os.environ["SG_ZMQ_PROOF_PHASE"]
    assert phase in {"units", "chat"}
    root = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    directory = Path(os.environ["SG_ZMQ_PROOF_DIRECTORY"])
    assert directory.is_absolute() and directory.resolve(strict=True) == directory
    assert directory.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve(strict=True))
    prefix = config.getoption("junitprefix")
    asyncio = asyncio_proof()


def asyncio_proof():
    """Record the original workflow's installed and active async test plugin."""
    dist = metadata.distribution("pytest-asyncio")
    raw = dist.read_text("RECORD")
    assert raw and Path(dist._path).resolve().is_relative_to(Path(sys.prefix).resolve())
    with (directory / (phase + "-pytest-asyncio-RECORD.csv")).open("x") as stream:
        stream.write(raw)
    rows = list(csv.reader(io.StringIO(raw)))
    found = [row for row in rows if row[0] == "pytest_asyncio/plugin.py"]
    assert len(found) == 1
    name, encoded, size = found[0]
    path = Path(dist.locate_file(name)).resolve(strict=True)
    active = Path(sys.modules["pytest_asyncio.plugin"].__file__).resolve(strict=True)
    content = path.read_bytes()
    digest = hashlib.sha256(content).digest()
    assert active == path and size.isdecimal() and len(content) == int(size)
    assert encoded == "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return {"version": dist.version, "metadata_origin": str(dist._path), "active_origin": str(active),
        "plugin_sha256": digest.hex(), "RECORD_text_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def junit_address(nodeid):
    names = junitxml.mangle_test_address(nodeid)
    classes = names[:-1]
    if prefix:
        classes.insert(0, prefix)
    return {"classname": ".".join(classes), "name": junitxml.bin_xml_escape(names[-1])}


def junit_skip(report, collecting=False):
    if collecting:
        raw = str(report.longrepr)
        return {"attributes": {"message": "collection skipped"},
            "text": junitxml.bin_xml_escape(raw[:16384]), "truncated": len(raw) > 16384}
    if hasattr(report, "wasxfail"):
        reason = report.wasxfail
        if reason.startswith("reason: "):
            reason = reason[8:]
        return {"attributes": {"type": "pytest.xfail", "message": junitxml.bin_xml_escape(reason)},
            "text": "", "truncated": False}
    assert isinstance(report.longrepr, tuple)
    filename, lineno, reason = report.longrepr
    if reason.startswith("Skipped: "):
        reason = reason[9:]
    text = f"{filename}:{lineno}: {reason}"
    return {"attributes": {"type": "pytest.skip", "message": junitxml.bin_xml_escape(reason)},
        "text": junitxml.bin_xml_escape(text[:16384]), "truncated": len(text) > 16384}


def pytest_collection_finish(session):
    base = root / "e2e_test" if phase == "chat" else root
    for item in session.items:
        nodeid = item.path.resolve().relative_to(base).as_posix() + "::" + item.nodeid.split("::", 1)[1]
        value = {"nodeid": nodeid, "pytest_nodeid": item.nodeid,
            "params": {k: repr(v) for k, v in getattr(getattr(item, "callspec", None), "params", {}).items()},
            "junit_address": junit_address(item.nodeid)}
        if phase == "chat":
            from fixtures.markers import model_id_for_engine, resolve_class_marker
            value["model"] = model_id_for_engine(resolve_class_marker(item, "model"), "sglang")
            value["skip_markers"] = [{"name": mark.name, "args": [repr(v) for v in mark.args],
                "reason": mark.kwargs.get("reason")} for name in ("skip", "skipif")
                for mark in item.iter_markers(name)]
        selected.append(value)


def pytest_collectreport(report):
    if report.skipped or report.failed:
        collection.append({"nodeid": report.nodeid, "outcome": report.outcome,
            "longrepr": str(report.longrepr)[:16384], "junit_address": junit_address(report.nodeid),
            "junit_skip": junit_skip(report, collecting=True) if report.skipped else None})


def pytest_runtest_logreport(report):
    reports.setdefault(report.nodeid, []).append({"when": report.when, "outcome": report.outcome,
        "xfail": bool(getattr(report, "wasxfail", False)),
        "junit_skip": junit_skip(report) if report.skipped else None,
        "longrepr": str(report.longrepr)[:16384] if report.outcome != "passed" else None})


def pytest_sessionfinish(session, exitstatus):
    with (directory / (phase + "-reports.json")).open("x") as stream:
        json.dump({"selected": selected, "reports": reports, "collection_reports": collection,
            "exit_code": int(exitstatus), "phase": phase,
            "junit_writer": {"path": str(Path(junitxml.__file__).resolve()),
                "sha256": hashlib.sha256(Path(junitxml.__file__).read_bytes()).hexdigest(), "prefix": prefix},
            "pytest_asyncio": asyncio, "source": str(root),
            "scope": "Passive original command observations; no selection or outcome changes."}, stream, indent=2)
        stream.write("\n")
