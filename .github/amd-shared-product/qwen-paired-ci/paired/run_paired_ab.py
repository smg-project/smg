"""NEW2100 paired Qwen diagnostic in a root-prepared, nonce-owned4420 container.

Not a publisher or original6/231 test. No pip install, Docker, SSH, API, or
CONTROL writes. Root must first admit the window and bind actual container,
environment preparation, source, model snapshot and resource evidence.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

import owned_processes as proc

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SOURCE = "1473e2d4fae7084d0f744055ba29a1a4b9679ae3"
NATIVE = "b7d5ccb1d5e4c06cf93fc9f77ad405951b9d3def7038113f5435a0d57a170fd6"
PUG_URL = "https://raw.githubusercontent.com/smg-project/smg/main/e2e_test/fixtures/images/pug.jpg"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def check_pin(item):
    path = Path(item["path"])
    require(path.is_file() and not path.is_symlink(), "pin not a regular direct file")
    require(("bytes" not in item or path.stat().st_size == item["bytes"]) and
            sha(path) == item["sha256"], "pin bytes differ: " + str(path))


def check_model_pin(item, snapshot):
    path = Path(item["path"])
    # HF snapshot metadata legitimately symlinks into its own repository blobs.
    require(path.absolute().is_relative_to(snapshot.absolute()), "logical metadata path outside snapshot")
    resolved = path.resolve()
    require(str(resolved) == item["resolved_path"] and resolved.is_relative_to(snapshot.parent.parent.resolve()),
            "model metadata resolved path differs/escapes repository")
    check_pin({**item, "path": str(resolved)})


def check_renewed_delivery(bound, inputs):
    expected, actual = inputs["artifact_delivery"], bound["artifact_delivery"]
    require(actual["policy"] == expected["policy"] and
            actual["copy_owner"] == expected["copy_owner"] and
            actual["origin_artifacts"] == expected["origin_artifacts"] and
            actual["origin_evidence"] == expected["origin_evidence"] and
            actual["four_artifacts"] == inputs["four_artifacts"] and
            actual["origin_verified_step_completed_at"] == expected["origin_verified_step_completed_at"] and
            actual["six_payloads_verified"] is True and actual["origin17_preserved"] is True and
            actual["new_build_executed"] is False and actual["GPU_credit"] == 0,
            "exact renewed delivery, distinct from original17 provenance")
    for name in ["copy_run_api", "copy_job_api", "copy_origin_proof"]:
        check_pin(bound[name])
    run, job, proof = [read_json(bound[name]["path"]) for name in
                       ["copy_run_api", "copy_job_api", "copy_origin_proof"]]
    owner = expected["copy_owner"]
    require(run["id"] == owner["run_id"] and run["head_sha"] == owner["head_sha"] and
            run["head_branch"] == owner["branch"] and run["run_attempt"] == 1 and
            run["actor"]["login"] == owner["actor"] and run["event"] == "workflow_dispatch" and
            run["path"] == owner["workflow"] and run["status"] == "completed" and run["conclusion"] == "success",
            "actual successful renewed-copy run")
    require(job["id"] == owner["job_id"] and job["run_id"] == run["id"] and
            job["head_sha"] == run["head_sha"] and job["run_attempt"] == 1 and
            job["name"] == "renew-d5-artifacts-only" and job["runner_id"] == owner["runner_id"] and
            job["labels"] == ["ubuntu-latest"] and job["status"] == "completed" and job["conclusion"] == "success",
            "actual hosted copy job; not a newbuild")
    require([s["name"] for s in job["steps"]] == expected["copy_step_names"] and
            all(s["status"] == "completed" and s["conclusion"] == "success" for s in job["steps"]) and
            job["steps"][2]["completed_at"] == actual["origin_verified_step_completed_at"],
            "copy validation/upload steps")
    require(proof["passed"] is True and proof["original_producer"]["run_id"] == inputs["producer_run"] and
            proof["original_producer"]["head_sha"] == inputs["producer_head"] and
            proof["original_producer"]["artifacts"] == expected["origin_artifacts"] and
            proof["original_producer"]["native"] == inputs["gateway_native"] and
            proof["copy_job"]["run_id"] == run["id"] and proof["copy_job"]["job_id"] == job["id"] and
            proof["copy_job"]["head_sha"] == run["head_sha"] and
            proof["build_executed"] is False and proof["runtime_accepted"] is False and proof["coverage_increment"] == 0,
            "pinned copy-origin proof; no execution credit transfer")


def main(approval_path, output):
    started = time.monotonic()
    approval, inputs = read_json(approval_path), read_json(HERE / "inputs.json")
    require(approval["root_manual_admission"] is True, "source-only; root admission required")
    require(approval["window_seconds"] == 2100 and approval["NEW_window"] is True, "NEW2100 only")
    require(approval["GPU_jobs"] == 1 and approval["GPU_devices_simultaneous"] == 1, "one owned GPU only")
    require(approval["driver_sha256"] == sha(__file__) and approval["inputs_sha256"] == sha(HERE / "inputs.json"), "packet admission hashes")
    for name, expected in inputs["helpers"].items():
        require(sha(HERE / name) == expected, "helper source differs")
    require(sha(approval["independent_peer_path"]) == approval["independent_peer_sha256"], "independent peer bytes")
    peer = read_json(approval["independent_peer_path"])
    require(peer["result"].startswith("PASS") and peer["source_parent"] == SOURCE and
            peer["driver_sha256"] == approval["driver_sha256"] and peer["inputs_sha256"] == approval["inputs_sha256"] and
            peer["envelope_sha256"] == approval["envelope_sha256"], "exact nonauthor source peer required")
    issued = datetime.fromisoformat(approval["issued_at_utc"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    require(0 <= (now - issued).total_seconds() < 180, "admission freshness")
    # Includes the real CI environment preparation, not a reset when this runner starts.
    window = datetime.fromisoformat(approval["window_started_at_utc"].replace("Z", "+00:00"))
    spent = (now - window).total_seconds()
    require(0 <= spent < 180, "setup180 budget exhausted before entry")
    deadline, work_deadline = started + 2100 - spent, started + 1980 - spent
    require(approval["source_parent"] == SOURCE and inputs["source_parent"] == SOURCE, "source cohort")
    for name in ["container_proof", "preparation_proof", "source_binding_proof", "model_snapshot_proof", "resource_and_network_proof"]:
        check_pin(approval[name])
    container = read_json(approval["container_proof"]["path"])
    require(container["owner"] == "root" and container["root_approved"] is True, "container ownership")
    require(container["image"] == inputs["image"] and container["GPU_devices_selected"] == 1, "exact image/GPU")
    require(container["hostname"] == socket.gethostname(), "must run in bound owned container")
    require(container["CID"] and container["foreign_jobs_present"] is False, "container/foreign job scope")
    prepared = read_json(approval["preparation_proof"]["path"])
    require(prepared["actual_success"] is True and prepared["source_parent"] == SOURCE, "real CI preparation missing")
    require(prepared["whole_pip"] == 0 and prepared["native_sha256"] == NATIVE, "prepared pip/native")
    require(prepared["original_CI_setup_used"] is True and prepared["same_for_both_lanes"] is True, "do not invent an installation")
    bound = read_json(approval["source_binding_proof"]["path"])
    require(bound["source_parent"] == SOURCE and bound["consumer19"] == inputs["consumer19"] and
            bound["compiled37"] == inputs["compiled37"] and bound["producer14"] == inputs["producer14"], "19/37/14 proof")
    require(bound["producer_run"] == 40218847194 and bound["producer_head"] == inputs["producer_head"] and
            bound["four_artifacts"] == inputs["four_artifacts"] and bound["native_sha256"] == NATIVE, "actual1473/d5/b7 provenance")
    model = read_json(approval["model_snapshot_proof"]["path"])
    snapshot = Path(model["snapshot"])
    require(model["model_id"] == MODEL and snapshot.is_absolute() and snapshot.is_dir(), "same original model snapshot")
    require(snapshot.name == model["revision"] and len(model["revision"]) == 40 and
            all(c in "0123456789abcdef" for c in model["revision"]), "immutable snapshot directory")
    require(model["all_declared_weight_headers_valid"] is True, "complete model metadata prerequisite")
    require(len(model["config_and_tokenizer_files"]) >= 4, "model metadata manifest incomplete")
    model_value = model["fixture_resolved_model_path"]
    require(isinstance(model_value, str) and (model_value == MODEL or Path(model_value).is_absolute()),
            "actual original fixture model path required")
    def check_fixture_reference():
        if model_value != MODEL:
            require(Path(model_value).resolve() == snapshot.resolve(), "fixture path/snapshot join")
        else:
            ref = model["fixture_HF_ref"]
            require(Path(ref["path"]).resolve() == snapshot.parent.parent / "refs/main", "actual original HF cache reference")
            check_pin(ref)
            require(Path(ref["path"]).read_text().strip() == model["revision"], "HF model reference changed snapshot")
    check_fixture_reference()
    request_model_value = model_value
    engine_sources = prepared["engine_source_files"]
    require(isinstance(engine_sources, list) and engine_sources, "actual normal4420 engine source evidence required")
    require(all(any(item["path"].endswith(suffix) for item in engine_sources)
                for suffix in inputs["required_engine_source_suffixes"]), "engine source manifest incomplete")
    require(any(item["path"].endswith("/sglang/srt/utils/common.py") and
                item["sha256"] == prepared["native_download_common_sha256"] for item in engine_sources),
            "native URL helper must join the same actual engine manifest")
    require(prepared["old_private_MM_or_MRoPE_hooks_installed"] is False,
            "current1473 normal cohort must not acquire old private probes")
    installed_sources = inputs["installed_source_files"] + engine_sources
    for item in model["config_and_tokenizer_files"]:
        check_model_pin(item, snapshot)
    resources = read_json(approval["resource_and_network_proof"]["path"])
    for proof in [container, prepared, bound, model, resources]:
        require(proof["window_id"] == approval["window_id"] and proof["CID"] == container["CID"], "old/different window or container proof")
    for proof in [container, resources, bound]:
        checked = datetime.fromisoformat(proof["checked_at_utc"].replace("Z", "+00:00"))
        require(0 <= (datetime.now(timezone.utc) - checked).total_seconds() < 180, "fresh root prerequisite proof required")
    # Reuse the accepted exact d5 main-push build-leg policy. The parent
    # pipeline was cancelled; never turn that fact into a success prerequisite.
    context = bound["producer_context"]
    require(context["producer_context_verified"] is True and
            context["producer_policy"] == inputs["producer_policy"] and
            context["source_base"] == inputs["producer_head"] and
            context["actual_head"] == inputs["producer_head"] and
            context["producer_build_leg_only"] is True and
            context["producer_parent_pipeline_accepted"] is False and
            context["producer_parent_status"] == bound["actual_producer_status"] == "completed" and
            context["producer_parent_conclusion"] == bound["actual_producer_conclusion"] == "cancelled",
            "exact d5 build-leg-only context; cancelled parent is not pipeline acceptance")
    for name in ["producer_run_api", "producer_build_job_api"]:
        check_pin(bound[name])
    producer_run = read_json(bound["producer_run_api"]["path"])
    producer_job = read_json(bound["producer_build_job_api"]["path"])
    require(producer_run["id"] == inputs["producer_run"] and
            producer_run["head_sha"] == inputs["producer_head"] and
            producer_run["run_attempt"] == 1 and producer_run["event"] == "push" and
            producer_run["head_branch"] == "main" and
            producer_run["path"] == ".github/workflows/pr-test-rust.yml" and
            producer_run["status"] == "completed" and producer_run["conclusion"] == "cancelled",
            "actual d5 parent run identity/state")
    require(producer_job["id"] == inputs["producer_build_job_id"] and
            producer_job["run_id"] == producer_run["id"] and
            producer_job["head_sha"] == producer_run["head_sha"] and
            producer_job["run_attempt"] == producer_run["run_attempt"] and
            producer_job["name"] == "build-wheel" and producer_job["status"] == "completed" and
            producer_job["conclusion"] == "success", "actual d5 successful build-leg identity")
    steps = producer_job["steps"]
    names = [step["name"] for step in steps]
    require(len(names) == len(set(names)) == 17 and set(names) == set(inputs["producer_build_step_names"]) and
            all(step["status"] == "completed" and step["conclusion"] == "success" for step in steps),
            "exact d5 build-leg all17 named steps must succeed")
    if "artifact_delivery" in inputs:
        check_renewed_delivery(bound, inputs)
    require(all(datetime.fromisoformat(item["expires_at"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
                for item in inputs["four_artifacts"]), "expired delivered artifact copies")
    require(resources["root_approved"] is True and resources["NEW_runner_seconds"] == 2100 and
            resources["no_retired_allowance_reused"] is True, "new root accounting required")
    require(resources["one_selected_GPU"] is True and resources["ordinary_jobs_preserved"] is True,
            "resource ownership prerequisite")
    require(resources["original_raw_image_HTTPS_read_allowed"] is True and resources["external_model_API_calls"] == 0,
            "original image URL needs approved real read-only fetch; no paid model API")
    source = Path(approval["source_worktree"])
    require(str(source) == prepared["source_worktree"], "prepared source path differs")
    for item in installed_sources:
        check_pin({**item, "path": str(source / item["source_relative"])} if "source_relative" in item else item)
    require(str(Path(sys.executable).resolve()) == approval["python_resolved"], "actual prepared interpreter")
    output = output.resolve()
    require(str(output) == approval["output"] and not output.exists(), "exact unused own output")
    require(output.parent.is_dir(), "output parent must already be root-owned stage")
    output.mkdir(mode=0o700)
    nonce = "smg-qwen-ab-" + uuid.uuid4().hex
    report = {"result": "IN_PROGRESS", "started_at_utc": utc(), "source_parent": SOURCE,
              "nonce": nonce, "window_seconds": 2100, "already_spent_seconds": spent,
              "container": {k: container[k] for k in ["CID", "hostname", "image", "owner", "GPU_devices_selected"]},
              "lanes": [], "operations": [], "passed": False, "first_request_failure": None, "root_cause": "UNDETERMINED",
              "coverage_increment": 0, "original6_or_231_accepted": False}
    (output / "owner.json").write_text(json.dumps({"nonce": nonce, "window": approval["window_id"]}))
    environment = {k: v for k, v in os.environ.items() if k.lower() not in
                   {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    # Current1473 has no old private MM/MRoPE/sampling probes. Do not enable them.
    for key in ["SMG_QWEN_MROPE_TRACE", "SMG_QWEN_MM_TRACE", "SMG_QWEN_MM_MODEL", "SMG_SGLANG_SERVICER_IMPL"]:
        environment.pop(key, None)
    environment.update({proc.KEY: nonce, "PYTHONDONTWRITEBYTECODE": "1",
                        "SMG_SGLANG_SERVICER_WARMUP": "fast", "SPT_NOENV": "1",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    masks = {k: environment.get(k) for k in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]}
    require(masks == approval["actual_GPU_masks"], "exact root-selected logical GPU masks")
    require(any(masks.values()) and all(value is None or (bool(value) and len(value.split(",")) == 1)
                                       for value in masks.values()), "consistent single-device masks required")
    children, logs = [], []
    python = sys.executable
    fixtures = source / "e2e_test/fixtures/images"

    def alive_and_bounded():
        require(time.monotonic() < work_deadline, "absolute1980s work deadline")
        require(all(not f.exists() or f.stat().st_size <= 32 * 1024**2 for f in logs), "launcher log32MiB cap")

    def launch(label, argv):
        alive_and_bounded()
        path = output / (label + ".log")
        log = path.open("xb")
        try:
            child = subprocess.Popen(argv, cwd=source, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        children.append(child)
        logs.append(path)
        snapshot = proc.launcher_snapshot(child, nonce)
        child.smg_identity = snapshot["process"] or snapshot["stat"]
        record = {"label": label, "started_at_utc": utc(), **snapshot, "log": str(path)}
        report["operations"].append(record)
        (output / (label + "-launcher.json")).write_text(json.dumps(record, indent=2) + "\n")
        require(snapshot["terminal_reaped"] or snapshot["live_identity_verified"], "own launcher identity")
        return child

    def operation(label, argv, limit, until=None):
        nonlocal setup_used
        alive_and_bounded()
        begin = time.monotonic()
        if until is None and "-request-" not in label:
            limit = min(limit, 180 - setup_used)
            require(limit > 2, "combined preparation/URL/validation180s budget exhausted")
        end = min(work_deadline, until or work_deadline, time.monotonic() + limit) - 2
        child = launch(label, argv)
        while child.poll() is None and time.monotonic() < end:
            alive_and_bounded()
            time.sleep(min(0.1, max(0, end - time.monotonic())))
        timed_out = child.poll() is None
        if timed_out:
            row = proc.identity(child.pid)
            if row:
                proc.signal_group(row, signal.SIGKILL, nonce)
            child.wait(timeout=2)
        if until is None and "-request-" not in label:
            setup_used += time.monotonic() - begin
        result = {"label": label, "exit": child.wait(),
                  "hard_timeout": timed_out or time.monotonic() - begin > limit,
                  "elapsed_seconds": time.monotonic() - begin,
                  "finished_at_utc": utc(), "setup_spent_seconds": setup_used}
        result["passed"] = result["exit"] == 0 and not result["hard_timeout"]
        report["operations"].append(result)
        return result

    def observe(label, mode, endpoint, cap, until=None, case_index=None):
        dest = output / label
        args = [python, str(HERE / "http_observe.py"), mode, "--endpoint", endpoint,
                "--out", str(dest), "--request-id", nonce + ":" + label, "--fixtures", str(fixtures)]
        if mode == "URL-native":
            args += ["--installed-common-sha", prepared["native_download_common_sha256"]]
        if mode == "request":
            args += ["--model-value", request_model_value, "--case-index", str(case_index)]
        if mode == "metrics":
            args += ["--metrics-source-sha", inputs["metrics_source_sha256"]]
        result = operation(label, args, cap, until)
        result["observation"] = read_json(dest / "result.json") if (dest / "result.json").exists() else None
        # A forcibly stopped response cannot be turned into a successful oracle.
        result["passed"] = not result["hard_timeout"] and result["exit"] == 0 and bool(result["observation"] and result["observation"]["passed"])
        (output / (label + "-parent.json")).write_text(json.dumps(result, indent=2))
        return result

    def stop_current():
        result = proc.cleanup(nonce, min(deadline, time.monotonic() + 50), children)
        report["operations"].append({"label": "owned_cleanup", "result": result, "at_utc": utc()})
        require(result["passed"], "owned descendants remain; do not start next engine")

    def interrupted(signum, frame):
        raise TimeoutError("signal/absolute work deadline")

    previous = {sig: signal.signal(sig, interrupted) for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGALRM]}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, work_deadline - time.monotonic()))
    setup_used = spent + time.monotonic() - started
    try:
        require(operation("runtime-before", [python, str(HERE / "runtime_check.py"), "--inputs", str(HERE / "inputs.json"),
                    "--out", str(output / "runtime-before.json")], 15)["passed"], "actual runtime before")
        require(operation("whole-pip-before", [python, "-m", "pip", "check"], 30)["passed"], "whole pip before")
        help_result = operation("actual-SGLang-CLI", [python, "-m", "sglang.launch_server", "--help"], 30)
        help_text = (output / "actual-SGLang-CLI.log").read_text(errors="replace")
        require(help_result["passed"] and all(flag in help_text for flag in
                ["--model-path", "--host", "--port", "--tp-size", "--log-level", "--grpc-mode"]), "actual installed CLI flags")
        request_hashes = {}
        matrix = read_json(HERE / "six-request-matrix.json")["requests"]
        require([row["order"] for row in matrix] == list(range(1, 7)), "original six-request order")
        for lane, engine_port, gateway_port in [("A-nativeHTTP", 31000, None), ("B-SMG-gRPC", 31001, 31002)]:
            require(not proc.owned(nonce), "previous lane processes still live")
            URL_before = [observe(lane + "-URL-before-" + name, "URL-native", PUG_URL.replace("pug.jpg", name), 15)
                          for name in ["dog.jpg", "pug.jpg"]]
            require(all(row["passed"] for row in URL_before), "original URL byte precondition failed")
            lane_row = {"name": lane, "startup_begin_utc": utc(), "observations": [],
                        "private_tensor_or_sampling_traces_available": False}
            if gateway_port:
                environment["SMG_SGLANG_SERVICER_IMPL"] = "rust"
            else:
                environment.pop("SMG_SGLANG_SERVICER_IMPL", None)
            report["lanes"].append(lane_row)
            startup_end = min(work_deadline, time.monotonic() + 600)
            #600 includes model start, real health-generated warmup, gateway and metadata.
            engine_args = [python, "-m", "sglang.launch_server", "--model-path", model_value,
                           "--host", "127.0.0.1", "--port", str(engine_port),
                           "--tp-size", "1", "--log-level", "warning"]
            if gateway_port:
                engine_args.append("--grpc-mode")
            engine = launch(lane + "-engine", engine_args)
            gateway = None
            while True:
                require(engine.poll() is None, "engine exited before real health")
                require(time.monotonic() < startup_end, "600s combined startup/warmup limit")
                engine_ready = observe(lane + "-engine-health-" + str(len(report["operations"])),
                    "grpc-health" if gateway_port else "health",
                    "127.0.0.1:" + str(engine_port) if gateway_port else f"http://127.0.0.1:{engine_port}/health",
                    15, startup_end)
                if engine_ready["passed"]:
                    break
                time.sleep(min(0.5, max(0, startup_end - time.monotonic())))
            if gateway_port:
                gateway = launch(lane + "-gateway", [python, "-m", "smg.launch_router", "--host", "127.0.0.1",
                    "--port", str(gateway_port), "--prometheus-host", "127.0.0.1", "--prometheus-port", "31003",
                    "--policy", "round_robin", "--log-level", "warn", "--model-path", model_value,
                    "--worker-urls", "grpc://127.0.0.1:" + str(engine_port)])
            endpoint = f"http://127.0.0.1:{gateway_port or engine_port}"
            while True:
                require(engine.poll() is None and (gateway is None or gateway.poll() is None), "real launcher exited during readiness")
                require(time.monotonic() < startup_end, "600s combined startup/warmup limit")
                if gateway is None:
                    break
                ready = observe(lane + "-ready-" + str(len(report["operations"])), "health",
                                endpoint + ("/readiness" if gateway_port else "/health"), 15, startup_end)
                if ready["passed"]:
                    break
                time.sleep(min(0.5, max(0, startup_end - time.monotonic())))
            if gateway_port:
                meta = observe(lane + "-metadata", "grpc", "127.0.0.1:" + str(engine_port), 10, startup_end)
                require(meta["passed"], "real GetModelInfo/GetServerInfo")
                model_info, server_info = meta["observation"]["model"], meta["observation"]["server"]["server_args"]
                registry = observe(lane + "-registry", "json", endpoint + "/workers", 10, startup_end)
                require(registry["passed"], "real worker registry")
                workers = registry["observation"]["value"]["workers"]
                require(any(row.get("url") == "grpc://127.0.0.1:" + str(engine_port) and row.get("is_healthy") is True
                            for row in workers), "expected owned healthy worker URL missing")
                lane_row["gateway_native"] = proc.mapped_native(gateway.pid, NATIVE)
            else:
                model_result = observe(lane + "-model-info", "json", endpoint + "/model_info", 10, startup_end)
                server_result = observe(lane + "-server-info", "json", endpoint + "/server_info", 10, startup_end)
                require(model_result["passed"] and server_result["passed"], "real native metadata")
                model_info, server_info = model_result["observation"]["value"], server_result["observation"]["value"]
            require(model_info["model_path"] == model_value, "loaded original fixture model differs")
            require(model_info["tokenizer_path"] == model_value or
                    (model_value != MODEL and Path(model_info["tokenizer_path"]).resolve() == snapshot.resolve()),
                    "resolved tokenizer differs from same pinned original snapshot")
            models_result = observe(lane + "-advertised-models", "json", endpoint + "/v1/models", 10, startup_end)
            require(models_result["passed"], "actual /v1/models discovery")
            advertised = models_result["observation"]["value"]["data"]
            ids = {row["id"] for row in advertised if isinstance(row, dict) and isinstance(row.get("id"), str)}
            candidates = [model_value, model_info.get("served_model_name"), model_info.get("model_path")]
            request_model_value = next((item for item in candidates if item in ids), None)
            require(request_model_value is not None, "no actually advertised original model identity; refuse artificial404")
            lane_row["advertised_models"] = advertised
            lane_row["actual_request_model"] = request_model_value
            lane_row["original_fixture_model_value"] = model_value
            # No dtype/tokenizer/served-name/template override. These are actual
            # metadata, not a claim that server_args dtype=auto is a weight dtype.
            lane_row.update(model_info=model_info, server_info=server_info, startup_ready_utc=utc(),
                            configured_overrides={"dtype": None, "tokenizer_path": None,
                                                  "served_model_name": None, "chat_template": None},
                            observed_defaults={"dtype_server_args": server_info.get("dtype"),
                                               "tokenizer_path": model_info.get("tokenizer_path"),
                                               "served_model_name": model_info.get("served_model_name"),
                                               "chat_template": server_info.get("chat_template"),
                                               "default_sampling_params_json": model_info.get("default_sampling_params_json")},
                            owned_processes=[proc.public(row) for row in proc.owned(nonce)])
            for case in matrix:
                index = case["order"]
                require(engine.poll() is None and (gateway is None or gateway.poll() is None), "owned process exited before original request")
                result = observe(lane + "-request-" + str(index), "request", endpoint + "/v1/chat/completions", 90,
                                 case_index=index)
                observed = result["observation"]
                if observed:
                    require(isinstance(observed.get("request_semantics_sha256"), str), "missing actual request semantics")
                    if index not in request_hashes:
                        request_hashes[index] = observed["request_semantics_sha256"]
                    require(observed["request_semantics_sha256"] == request_hashes[index], "A/B original request messages/parameters changed")
                if index == 2:
                    if gateway_port:
                        metrics = observe(lane + "-request-2-metrics", "metrics", "http://127.0.0.1:31003/metrics", 10)
                        result["processing_location"] = metrics
                        result["passed"] = bool(result["passed"] and metrics["passed"])
                    else:
                        result["processing_location"] = {"scope": "nativeHTTP has no SMG gateway metric; original oracle not applicable"}
                result["original_nodeid"] = case["nodeid"]
                lane_row["observations"].append(result)
                lane_row["owned_after_" + str(index)] = [proc.public(row) for row in proc.owned(nonce)]
                if not result["passed"] and report["first_request_failure"] is None:
                    report["first_request_failure"] = {"lane": lane, "order": index, "nodeid": case["nodeid"],
                                                       "result": result, "actual_response_and_chunk_dir": str(output / (lane + "-request-" + str(index)))}
                # No automatic retry. Continue only the remaining explicitly
                # ordered original requests under the unchanged absolute deadline.
            URL_after = [observe(lane + "-URL-after-" + name, "URL-native", PUG_URL.replace("pug.jpg", name), 15)
                         for name in ["dog.jpg", "pug.jpg"]]
            lane_row["URL_bytes_before_after_passed"] = all(row["passed"] for row in URL_before + URL_after)
            require(all(row["passed"] for row in URL_after), "URL bytes changed during lane")
            check_fixture_reference()
            for item in model["config_and_tokenizer_files"]:
                check_model_pin(item, snapshot)
            for item in installed_sources:
                check_pin({**item, "path": str(source / item["source_relative"])} if "source_relative" in item else item)
            lane_row["passed"] = all(row["passed"] for row in lane_row["observations"])
            stop_current()
        after_pip = operation("whole-pip-after", [python, "-m", "pip", "check"], 30)
        require(after_pip["passed"], "whole pip after")
        require(operation("runtime-after", [python, str(HERE / "runtime_check.py"), "--inputs", str(HERE / "inputs.json"),
                    "--out", str(output / "runtime-after.json")], 15)["passed"], "actual runtime after")
        require(read_json(output / "runtime-before.json") == read_json(output / "runtime-after.json"), "runtime continuity")
        report["oracles_passed"] = len(report["lanes"]) == 2 and all(row["passed"] for row in report["lanes"])
        report["diagnostic_data_complete"] = all(len(row["observations"]) == 6 for row in report["lanes"])
        report["passed"] = report["oracles_passed"] and report["diagnostic_data_complete"]
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2048],
                           "traceback": traceback.format_exc()[:65536]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            report["final_cleanup"] = proc.cleanup(nonce, deadline, children)
        except BaseException as error:
            report["final_cleanup"] = {"passed": False, "error": str(error)[:1024]}
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(finished_at_utc=utc(), elapsed_runner_seconds=time.monotonic() - started,
                      total_window_seconds=spent + time.monotonic() - started)
        report["passed"] = bool(report["passed"] and report["final_cleanup"]["passed"] and time.monotonic() <= deadline)
        report["result"] = "PAIRED_DIAGNOSTIC_ORACLES_PASS_NO_COVERAGE" if report["passed"] else "PAIRED_DIAGNOSTIC_FAILURE_PRESERVED"
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.approval, args.output))
