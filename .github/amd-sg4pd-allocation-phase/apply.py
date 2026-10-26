"""Default-off exact-image SG4 PD phase/CU overlays; no engine imports."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

HERE = Path(__file__).resolve().parent
FLAG = "SMG_SG4_PD_ALLOCATION_PHASE_DIAGNOSTIC"
CU_FLAG = "SMG_SG4_PD_CU_EXISTING_RUNTIME"
sha = lambda data: hashlib.sha256(data).hexdigest()


def read(path):
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > 512 * 1024:
        raise RuntimeError("bounded regular source required")
    return path.read_bytes(), stat.S_IMODE(status.st_mode)


def atomic(path, data, mode):
    temporary = path.with_name(path.name + f".sg4pd-phase-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def package_record(plan, target):
    distribution = importlib.metadata.distribution(plan["distribution"])
    if distribution.version != plan["distribution_version"]:
        raise RuntimeError("AITER distribution version changed")
    spec = importlib.util.find_spec("aiter")  # Top-level lookup only; no AITER import.
    if spec is None or spec.origin is None or Path(spec.origin).resolve().parent != Path(plan["package_root"]):
        raise RuntimeError("AITER package origin changed")
    rows = {str(Path(distribution.locate_file(item)).resolve()): item.hash for item in distribution.files or ()}
    metadata = {name: distribution.read_text(name) for name in ("METADATA", "RECORD", "direct_url.json")}
    if any(value is None for value in metadata.values()):
        raise RuntimeError("AITER distribution metadata missing")
    if json.loads(metadata["direct_url.json"]) != plan["direct_url"]:
        raise RuntimeError("AITER editable source changed")
    if sum(len(value.encode()) for value in metadata.values()) > 512 * 1024:
        raise RuntimeError("bounded AITER metadata required")
    return {"distribution": plan["distribution"], "version": distribution.version,
            "raw_metadata": metadata,
            "metadata_sha256": {name: sha(value.encode()) for name, value in metadata.items()},
            "source_record_hash": str(rows.get(str(target.resolve()))),
            "interpretation": "Intentional source overlay; editable source may be outside RECORD. Metadata/RECORD are not rewritten or claimed to validate changed source."}


def main():
    flags = {key: os.environ.get(key, "0") or "0" for key in (FLAG, CU_FLAG)}
    if all(value == "0" for value in flags.values()):
        if len(sys.argv) > 1 and sys.argv[1] == "final":
            raise RuntimeError("same-job overlay flags disappeared before final verification")
        return  # No source, helper or evidence access in the default path.
    if any(value not in ("0", "1") for value in flags.values()):
        raise RuntimeError("invalid explicit SG4 phase/CU opt-in")
    phase_enabled, cu_enabled = flags[FLAG] == "1", flags[CU_FLAG] == "1"
    cu_source_enabled = cu_enabled or phase_enabled
    tier = os.environ.get("E2E_GPU_TIER")
    supported = tier == "4" or (tier == "2" and os.environ.get("SMG_AMD_TS_PD2_REFERENCE") == "1")
    if os.environ.get("E2E_ENGINE") != "sglang" or not supported:
        raise RuntimeError("only complete SG4 PD or explicit TS PD2 SGLang reference is supported")
    if any(os.environ.get(key) == "1" for key in (
        "SMG_SGLANG_PD_PROOF", "SMG_SG2_STARTUP_DIAGNOSTIC", "SMG_SG2_CU_EXISTING_RUNTIME",
        "SMG_SG2_PREFILL_STREAM_BUDGET", "SMG_SG2_ALLOCATION_FAILURE_DIAGNOSTIC",
    )):
        raise RuntimeError("legacy SG2 plugin and startup behavior must remain off")
    marker = os.environ.get("E2E_JOB_MARKER", "")
    expected = "e2e-{GITHUB_RUN_ID}-{GITHUB_RUN_ATTEMPT}-{RUNNER_NAME}".format_map(os.environ)
    if marker != expected or not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
        raise RuntimeError("exact current AMD job marker required")
    mode, directory = sys.argv[1], Path(sys.argv[2])
    if mode not in ("apply", "final") or not directory.is_absolute():
        raise RuntimeError("apply/final and absolute same-job evidence directory required")
    parent = Path(os.environ["RUNNER_TEMP"])
    basename = "sg4pd-allocation-phase-{GITHUB_RUN_ID}-{GITHUB_RUN_ATTEMPT}".format_map(os.environ)
    if directory.parent != parent or directory.name != basename or parent.is_symlink():
        raise RuntimeError("unique same-job RUNNER_TEMP evidence directory required")
    if mode == "final" and json.loads((directory / "apply.json").read_text())["enabled_flags"] != flags:
        raise RuntimeError("same-job overlay flags changed")
    plan = json.loads((HERE / "overlay.json").read_text())
    if os.environ.get("SMG_SGLANG_ROCM_IMAGE") != plan["image"]:
        raise RuntimeError("exact4420 image required")
    # Preflight all requested source states before creating evidence or writing any.
    if phase_enabled:
        target, helper_target = Path(plan["target"]), Path(plan["helper_target"])
        raw, target_mode = read(target)
        helper = (HERE / "allocation_failure.py").read_bytes()
        if sha(helper) != plan["helper_sha256"]:
            raise RuntimeError("diagnostic helper source differs")
        if sha(raw) == plan["before_sha256"] and mode == "apply":
            before = plan["before"].encode()
            if raw.count(before) != 1:
                raise RuntimeError("exact one original tensor/.to expression required")
            raw = raw.replace(before, plan["after"].encode())
        if sha(raw) != plan["after_sha256"]:
            raise RuntimeError("unknown or unapplied engine source")
        if helper_target.exists() or helper_target.is_symlink():
            accepted_helpers = {plan["helper_sha256"]}
            if mode == "apply":
                accepted_helpers.add(plan["previous_helper_sha256"])
                accepted_helpers.add(plan["previous_runtime_helper_sha256"])
            if sha(read(helper_target)[0]) not in accepted_helpers:
                raise RuntimeError("refuse unrelated installed diagnostic helper")
        elif mode == "final":
            raise RuntimeError("diagnostic helper missing at final verification")
        scheduler_plan = plan["runtime_scheduler"]
        scheduler_target = Path(scheduler_plan["target"])
        scheduler_raw, scheduler_mode = read(scheduler_target)
        if sha(scheduler_raw) == scheduler_plan["before_sha256"] and mode == "apply":
            before = scheduler_plan["before"].encode()
            if scheduler_raw.count(before) != 1:
                raise RuntimeError("exact one original stream alias expression required")
            scheduler_raw = scheduler_raw.replace(before, scheduler_plan["after"].encode())
        if sha(scheduler_raw) != scheduler_plan["after_sha256"]:
            raise RuntimeError("unknown or unapplied current4420 scheduler source")
        future_plan = plan["runtime_future_map"]
        future_target = Path(future_plan["target"])
        future_raw, future_mode = read(future_target)
        if sha(future_raw) == future_plan["before_sha256"] and mode == "apply":
            before = future_plan["before"].encode()
            if future_raw.count(before) != 1:
                raise RuntimeError("exact one original FutureMap second allocation required")
            future_raw = future_raw.replace(before, future_plan["after"].encode())
        if sha(future_raw) != future_plan["after_sha256"]:
            raise RuntimeError("unknown or unapplied current4420 FutureMap source")
    if cu_source_enabled:
        cu_plan = json.loads((HERE / "cu-overlay.json").read_text())
        if phase_enabled:
            cu_plan = {**cu_plan, **cu_plan["diagnostic"]}
        if cu_plan["image"] != plan["image"] or cu_plan["flag"] != CU_FLAG:
            raise RuntimeError("CU source plan identity differs")
        cu_target = Path(cu_plan["target"])
        cu_raw, cu_mode = read(cu_target)
        record = package_record(cu_plan, cu_target)
        if phase_enabled and sha(cu_raw) == cu_plan["current_applied_preimage_sha256"] and mode == "apply":
            before = cu_plan["current_applied_before"].encode()
            if cu_raw.count(before) != 1:
                raise RuntimeError("exact one current CU function required")
            cu_raw = cu_raw.replace(before, cu_plan["after"].encode())
        if sha(cu_raw) == cu_plan["before_sha256"] and mode == "apply":
            before = cu_plan["before"].encode()
            if cu_raw.count(before) != 1:
                raise RuntimeError("exact one original CU read required")
            cu_raw = cu_raw.replace(before, cu_plan["after"].encode())
        if sha(cu_raw) != cu_plan["after_sha256"]:
            raise RuntimeError("unknown or unapplied current4420 AITER source")
        if mode == "final" and record != json.loads((directory / "cu-record-before.json").read_text()):
            raise RuntimeError("AITER metadata/RECORD changed after overlay")
    if mode == "apply":
        directory.mkdir(mode=0o700)
        (directory / "sglang-startup").mkdir(mode=0o700)
        if phase_enabled:
            atomic(helper_target, helper, 0o644)
            atomic(target, raw, target_mode)
            atomic(scheduler_target, scheduler_raw, scheduler_mode)
            atomic(future_target, future_raw, future_mode)
        if cu_source_enabled:
            (directory / "cu-record-before.json").write_text(json.dumps(record, indent=2) + "\n")
            atomic(cu_target, cu_raw, cu_mode)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("same-job evidence directory missing or replaced")
    if phase_enabled and (sha(read(target)[0]) != plan["after_sha256"] or sha(read(helper_target)[0]) != plan["helper_sha256"]):
        raise RuntimeError("installed engine or diagnostic source changed")
    if phase_enabled and sha(read(scheduler_target)[0]) != scheduler_plan["after_sha256"]:
        raise RuntimeError("installed scheduler diagnostic source changed")
    if phase_enabled and sha(read(future_target)[0]) != future_plan["after_sha256"]:
        raise RuntimeError("installed FutureMap diagnostic source changed")
    if cu_source_enabled and (sha(read(cu_target)[0]) != cu_plan["after_sha256"] or package_record(cu_plan, cu_target) != record):
        raise RuntimeError("installed AITER source/metadata changed")
    (directory / (mode + ".json")).write_text(json.dumps({
        "job_marker": marker, "enabled_flags": flags,
        "overlay_manifest_sha256": sha((HERE / "overlay.json").read_bytes()),
        "engine_source_sha256": plan["after_sha256"] if phase_enabled else None,
        "helper_sha256": plan["helper_sha256"] if phase_enabled else None,
        "scheduler_source_sha256": scheduler_plan["after_sha256"] if phase_enabled else None,
        "futuremap_source_sha256": future_plan["after_sha256"] if phase_enabled else None,
        "CU_runtime_observation_enabled": phase_enabled,
        "CU_source_sha256": cu_plan["after_sha256"] if cu_source_enabled else None,
        "CU_overlay_manifest_sha256": sha((HERE / "cu-overlay.json").read_bytes()) if cu_source_enabled else None,
        "CU_package_record": record if cu_source_enabled else None,
        "RECORD_rewritten": False,
        "actual_GPU_operation_or_cause_verified": False, "coverage_increment": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
