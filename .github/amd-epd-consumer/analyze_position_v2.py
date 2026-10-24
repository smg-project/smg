"""Offline v2-only diagnostic table. Requires a separately bound future-run contract."""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from position_analysis_v2 import analyze, compare, expected_sources, require

HERE = Path(__file__).resolve().parent


def read(path, cap):
    require(not path.is_symlink(), "symlink input")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, "analysis input byte cap")
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", choices=("position_manifest.json", "mrope_manifest.json"),
                        default="position_manifest.json")
    args = parser.parse_args()
    require(not args.output.exists(), "analysis refuses overwrite")
    contract = json.loads(read(args.contract, 65536))
    require(contract["version"] == 2 and type(contract["run_id"]) is int and contract["run_id"] > 0
            and type(contract["run_attempt"]) is int and contract["run_attempt"] > 0
            and re.fullmatch(r"[0-9a-f]{40}", contract["admitted_head"])
            and contract["gpu_job_name"] == "amd-sglang-epd / run", "future run identity not bound")
    original_ids = contract["original_case_ids"]
    require(len(original_ids) == len(set(original_ids)) == 8, "original eight case contract")
    manifest_bytes = read(HERE / args.manifest, 16384)
    require(sha(manifest_bytes) == contract["position_manifest_sha256"], "source manifest binding")
    manifest = json.loads(manifest_bytes)
    root = args.capture.resolve()
    capture = json.loads(read(root / "actual-runtime-artifact-capture.json", 1024**2))
    run = json.loads(read(root / "run.json", 1024**2))
    jobs = json.loads(read(root / "jobs.json", 1024**2))["jobs"]
    require(run["status"] == "completed" and run["id"] == capture["run"] == contract["run_id"]
            and run["head_sha"] == capture["head"] == contract["admitted_head"]
            and run["run_attempt"] == contract["run_attempt"], "actual run differs")
    gpu = [job for job in jobs if job["name"] == contract["gpu_job_name"]]
    require(len(gpu) == 1 and gpu[0]["id"] == capture["job_id"], "unique GPU job differs")
    require(gpu[0]["runner_name"] == contract["runner_name"], "actual runner differs")
    marker = f"e2e-{run['id']}-{run['run_attempt']}-{gpu[0]['runner_name']}"
    payloads, seen, total_bytes = [], set(), 0
    require(len(capture["archives"]) <= 16, "archive count cap")
    for archive in capture["archives"]:
        zipped = root / "artifacts" / (str(archive["id"]) + ".zip")
        raw_zip = read(zipped, 128 * 1024**2)
        require(len(raw_zip) == archive["archive_bytes"] and sha(raw_zip) == archive["archive_sha256"],
                "captured archive size/hash differs")
        require(len(archive["files"]) <= 512, "payload count cap")
        directory = root / "artifacts" / str(archive["id"])
        require(directory.is_dir() and not directory.is_symlink(), "unsafe artifact directory")
        for member in archive["files"]:
            path = directory / member["name"]
            require(path.resolve().is_relative_to(directory.resolve()), "unsafe payload path")
            raw = read(path, 64 * 1024**2)
            require(len(raw) == member["bytes"] and sha(raw) == member["sha256"], "captured payload differs")
            key = (member["name"], member["sha256"])
            if key not in seen:
                total_bytes += len(raw)
                require(total_bytes <= 256 * 1024**2, "total retained payload byte cap")
                seen.add(key)
                payloads.append((path, raw))
    processes, overlays, records, rejected_lines, seen_records = defaultdict(list), {}, [], [], set()
    scanned = 0
    xml = []
    for path, raw in payloads:
        if path.parent.name == "workers" and path.suffix == ".jsonl":
            require(len(raw) <= 1024**2, "launcher file cap")
            for line in raw.splitlines():
                require(len(line) <= 16384, "launcher record cap")
                row = json.loads(line)
                if row["kind"] == "process":
                    require(row["pid"] == int(path.stem), "launcher filename differs")
                    processes[row["pid"]].append(row)
        elif path.name.startswith("position-overlay-") and path.suffix == ".json":
            row = json.loads(raw)
            pid = int(path.stem.rsplit("-", 1)[1])
            require(pid not in overlays or overlays[pid] == row, "ambiguous overlay")
            overlays[pid] = row
        elif path.name.startswith("worker-") and path.suffix == ".log":
            scanned += len(raw)
            require(scanned <= 256 * 1024**2, "total worker log cap")
            for number, line in enumerate(raw.decode(errors="replace").splitlines(), 1):
                match = re.search(r'\{"kind"\s*:\s*"smg_epd_position_', line)
                if not match:
                    continue
                evidence = {"path": str(path), "line": number}
                try:
                    require(len(line.encode()) <= 16384, "worker log line cap")
                    row, end = json.JSONDecoder().raw_decode(line[match.start():])
                    encoded = line[match.start():match.start() + end].encode()
                    require(len(encoded) + 1 <= 4096, "diagnostic JSON line cap")
                    key = sha(json.dumps(row, sort_keys=True).encode())
                    if key not in seen_records:
                        seen_records.add(key)
                        records.append((row, dict(evidence, JSON_sha256=sha(encoded), JSON_bytes=len(encoded) + 1)))
                except (ValueError, KeyError, TypeError) as error:
                    rejected_lines.append(dict(evidence, reason=str(error)))
        elif path.name == "e2e.xml":
            require(len(raw) <= 8 * 1024**2 and b"<!DOCTYPE" not in raw.upper(), "JUnit safety cap")
            cases = []
            for element in ET.fromstring(raw).iter("testcase"):
                module, classname = element.attrib["classname"].rsplit(".", 1)
                nodeid = (module.replace(".", "/") + ".py::" + classname + "::" + element.attrib["name"]).removeprefix("e2e_test/")
                cases.append({"nodeid": nodeid, "time": element.get("time"), "child_tags": [c.tag for c in element]})
            ids = {r["nodeid"] for r in cases}
            xml.append({"path": str(path), "sha256": sha(raw), "entries": cases,
                        "missing_original_IDs": sorted(set(original_ids) - ids),
                        "unexpected_IDs": sorted(ids - set(original_ids)),
                        "blank_duplicate_is_not_pass": True})
    require(records, "no position records captured")
    result = analyze(records, processes, overlays, manifest, sha(manifest_bytes), marker)
    tables, unmatched = compare(result["validated_events"])
    output = {"result": "OFFLINE_V2_DIAGNOSTIC_ONLY_RUNTIME_ACCEPTANCE_HOLD", "accepted": None,
              "runtime_accepted": False, "coverage_increment": 0, "run": run["id"], "head": run["head_sha"],
              "manifest_sha256": sha(manifest_bytes), "expected_sources": expected_sources(manifest),
              **result, "malformed_records": rejected_lines, "validated_event_count": len(result["validated_events"]),
              "RID_tables": tables, "unmatched_receiver_events": unmatched, "JUnit": xml,
              "limits": ["v1 records cannot acquire v2 identity retrospectively", "ancestry is a bounded live proc snapshot",
                         "current sys.modules file/hash is not a Python bytecode attestation", "first eight RID sample is not all16 requests",
                         "no GPU position values/KV usage/embedding contents/transfer ACK/HBM evidence",
                         "wheel/pip/constraints/native/cleanup require a separate complete actual audit"]}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
