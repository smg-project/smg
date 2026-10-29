"""Future normally installed native identity delta; no install, model or runner.
The caller is the separately reviewed CPU600 payload, after its normal install
and whole-pip/RECORD/source proof. This source preview cannot import a candidate.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys
import traceback


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--pins-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(sha(args.pins) == args.pins_sha256, "identity pins SHA256")
    pins = json.loads(args.pins.read_text())
    # Fail before native/package import, evidence creation or any installation.
    need(pins["PREVIEW_ONLY"] is False, "source preview is not runnable")
    need(pins["source_base"] == "9134d03216a53077668f373e2d9ed2546da57635", "compiled source base")
    producer = pins["producer"]
    need(producer["head"] == "9134d03216a53077668f373e2d9ed2546da57635", "actual producer HEAD")
    need(producer["run_id"] == 40795986051 and producer["job_id"] == 123192466952, "actual producer run/job")
    for name in ("build_source_evidence", "artifact_integrity_data"):
        row = producer[name]
        need(isinstance(row, dict) and re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")), "unbound producer proof: " + name)
    archives = pins["archives"]
    need(len(archives) == 2 and {r["name"] for r in archives} == {"smg-wheel", "python-client-types"}, "two installed current-product input archives")
    for row in archives:
        need(type(row["artifact_id"]) is int and row["artifact_id"] > 0 and type(row["bytes"]) is int and row["bytes"] > 0, "unbound archive")
        need(re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or "") and row["expires_at"], "unbound archive digest/expiry")
    payloads = pins["payloads"]
    need(len(payloads) == 2 and {r["role"] for r in payloads} == {"smg_wheel", "client_types"}, "two installed current-product payloads")
    for row in payloads:
        need(row["member"] and type(row["bytes"]) is int and row["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or ""), "unbound payload")
    expected = pins["installed_native"]
    need(re.fullmatch(r"[0-9a-f]{64}", expected["sha256"] or "") and type(expected["bytes"]) is int and expected["bytes"] > 0, "unbound native")
    need(expected["prefix"] and pins["installation_source"]["head"] and pins["installation_source"]["tar_sha256"], "unbound installation source/prefix")
    need(Path("/dev/kfd").is_char_device() and any(p.is_char_device() for p in Path("/dev/dri").glob("renderD*")) and not list(Path("/dev").glob("nvidia*")), "real AMD devices required")
    out = args.output
    out.mkdir(parents=False, exist_ok=False)
    report = {"passed": False, "source_base": pins["source_base"],
              "actual_producer_head": producer["head"], "actual_producer_run_id": producer["run_id"],
              "actual_producer_job_id": producer["job_id"], "scope": "NORMAL_NATIVE_IDENTITY_WITH_AMD_DEVICES_NOT_ENGINE_ACCEPTANCE",
              "GPU": False, "coverage_increment": 0}
    def save():
        (out / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        prefix = Path(expected["prefix"]).resolve(strict=True)
        need(Path(sys.prefix).resolve(strict=True) == prefix, "normal interpreter prefix")
        dist = metadata.distribution("smg")
        native = Path(dist.locate_file("smg/smg_rs.abi3.so")).resolve(strict=True)
        need(native.is_relative_to(prefix), "native outside normal prefix")
        report["distribution"] = {"name": "smg", "version": dist.version,
                                  "metadata_sha256": sha(Path(dist._path) / "METADATA"),
                                  "RECORD_sha256": sha(Path(dist._path) / "RECORD")}
        report["native"] = {"path": str(native), "sha256": sha(native), "bytes": native.stat().st_size}
        need(report["native"]["sha256"] == expected["sha256"] and report["native"]["bytes"] == expected["bytes"], "new native exact identity")
        module = importlib.import_module("smg.smg_rs")
        need(Path(module.__file__).resolve(strict=True) == native, "actual imported native origin")
        report["native_maps"] = [line for line in Path("/proc/self/maps").read_text().splitlines() if line.endswith(str(native))]
        need(report["native_maps"], "exact native not mapped")
        report["symbols"] = {name: hasattr(module, name) for name in ("Router", "BackendType", "PolicyType", "get_verbose_version_string")}
        need(all(report["symbols"].values()), "required public native API absent")
        # Exactly one actual exported native call; retain the complete return unchanged.
        verbose = module.get_verbose_version_string()
        need(type(verbose) is str, "native verbose version is not a string")
        raw = verbose.encode("utf-8")
        (out / "native-version-verbatim.txt").write_bytes(raw)
        report["native_version_verbatim"] = verbose
        report["native_version_utf8_sha256"] = hashlib.sha256(raw).hexdigest()
        save()  # Also preserve an old/missing label as a real failing observation.
        labels = {}
        for key in ("Build Time", "Build Mode", "Platform", "Git Branch", "Git Commit", "Git Status"):
            found = re.findall(r"^" + re.escape(key) + r": (.*)$", verbose, re.MULTILINE)
            need(len(found) == 1 and found[0] and found[0] != "unknown", "missing/ambiguous native label: " + key)
            labels[key] = found[0]
        report["native_version_fields"] = labels
        # build.rs uses git rev-parse --short HEAD. Compare its unmodified value
        # to actual producer HEAD; do not substitute source_base or a binary hash.
        commit = labels["Git Commit"]
        need(re.fullmatch(r"[0-9a-f]{7,40}", commit) and producer["head"].startswith(commit), "old/wrong baked producer Git label")
        need(labels["Git Status"] in ("clean", "dirty"), "unknown Git status")
        need(sha(native) == expected["sha256"] and native.stat().st_size == expected["bytes"], "native changed during observation")
        report["passed"] = True
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
