"""Copy already-built exact d5 artifacts; never build, install, or run a payload."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HELPERS = HERE.parent
sys.path.insert(0, str(HELPERS))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(descriptor):
    rel = Path(descriptor["path"])
    need(not rel.is_absolute() and ".." not in rel.parts, "bound descriptor path")
    p = (HERE / rel).resolve(strict=True)
    need(p.is_relative_to(HERE) and p.is_file() and not (HERE / rel).is_symlink(), "bound descriptor file")
    need(sha(p) == descriptor["sha256"], "bound descriptor digest")
    return json.loads(p.read_text())


class Deadline(BaseException):
    pass


def expired(_signum, _frame):
    raise Deadline()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    admission = json.loads((HERE / "admission.json").read_text())
    if admission["PREVIEW_ONLY"] or not admission["root_manual_upload_only"]:
        raise SystemExit("SOURCE ONLY: root CPU-only source binding and NEW900 required")
    need(admission["NEW_CPU_seconds"] == 900 and admission["GPU_seconds"] == 0, "CPU-only finite grant")
    nonce = os.environ["SMG_RENEWAL_NONCE"]
    need(len(nonce) == 32 and all(c in "0123456789abcdef" for c in nonce), "actual root nonce")
    need(admission["admission_nonce"] == nonce, "root nonce binding")
    need(os.environ["GITHUB_RUN_ATTEMPT"] == "1", "only attempt1")
    need(admission["approved_actor"] == os.environ["GITHUB_ACTOR"], "approved actor")
    need(admission["approved_branch"] == os.environ["GITHUB_REF_NAME"], "approved isolated branch")
    budget, peer = bound(admission["CPU_budget"]), bound(admission["source_peer"])
    need(budget["root_admitted"] is True and budget["NEW_CPU_seconds"] == 900 and budget["GPU_seconds"] == 0, "actual CPU budget")
    need(budget["admission_nonce"] == nonce and budget["source_parent"] == admission["source_parent"], "CPU budget source/nonce")
    need(peer["result"].startswith("PASS") and peer["findings"] == [], "independent source peer")
    need(peer["actual_head"] == admission["source_parent"], "actual disabled source peer")
    need(peer["leaf_sha256"] == sha(Path(__file__)) and peer["workflow_sha256"] == sha(ROOT / ".github/workflows/pr-test-rust.yml"), "reviewed code bytes")
    for name, digest in peer["helper_sha256"].items():
        need(name in {"source.py", "relay.py", "artifact_io.py", "frozen_producer.py", "wheel_record.py"} and sha(HELPERS / name) == digest, "reviewed helper bytes")
    need(len(peer["helper_sha256"]) == 5, "complete helper pins")
    import artifact_io as io
    import frozen_producer
    import relay
    import source
    import wheel_record
    need(io.git_output("rev-parse", "HEAD").strip() == os.environ["GITHUB_SHA"], "exact checkout")
    need(io.git_output("rev-parse", "HEAD^").strip() == admission["source_parent"], "direct CPU-only data child")
    changed = set(io.git_output("diff", "--name-only", "HEAD^", "HEAD").splitlines())
    allowed = {".github/amd-shared-product/artifact-renewal/admission.json", ".github/amd-shared-product/artifact-renewal/review/source-peer.json", ".github/amd-shared-product/artifact-renewal/review/root-CPU900-budget.json"}
    need(changed <= allowed and changed, "only actual CPU admission data may differ")
    shared = json.loads((HELPERS / "inputs.json").read_text())
    old_admission = json.loads((HELPERS / "qwen-paired-ci/admission.json").read_text())
    need(all(shared[k] is False for k in ("ready", "run_budget_admitted", "artifacts_pinned", "runtime_cohort_accepted", "consumer_gpu_ready")), "GPU source authorities must stay false")
    need(all(old_admission[k] is False for k in ("ready", "root_manual_admission", "run_budget_admitted", "artifacts_pinned")), "Qwen admission authorities must stay false")
    need(shared["budget"]["admitted"] is False and shared["budget"]["reserved_runner_seconds"] == old_admission["reservation_seconds"] == 0, "no GPU reservation")
    need(shared["producer_policy"] == frozen_producer.MAIN_PUSH_POLICY and shared["producer_sha"] == frozen_producer.MAIN_PUSH_SHA and shared["run_id"] == frozen_producer.MAIN_PUSH_RUN, "exact original d5 producer")
    paired = json.loads((HELPERS / "qwen-paired-ci/paired/inputs.json").read_text())
    need(shared["artifacts"] == paired["four_artifacts"] and shared["gateway_native"] == paired["gateway_native"], "exact four origins/native")
    run_id = int(os.environ["GITHUB_RUN_ID"])
    temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    out = args.out.resolve()
    need(out.parent == temp and out.name == "smg-renewal-" + str(run_id) + "-1", "own runner temp output")
    out.mkdir(mode=0o700)
    evidence = out / "evidence"
    evidence.mkdir(mode=0o700)
    report = {"passed": False, "scope": "copy-only; no build/install/test/model/GPU", "admission_nonce": nonce, "stage": "start", "original_producer": {}, "copy_job": {}, "payloads": [], "all_source_GPU_authorities_false": True, "build_executed": False, "tests_executed": False, "runtime_accepted": False, "coverage_increment": 0}
    def save(name, value):
        raw = json.dumps(value, indent=2) + "\n"
        need(len(raw.encode()) <= 2 * 1024**2, "bounded evidence")
        (evidence / name).write_text(raw)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 240)
    try:
        report["stage"] = "current-job-source"
        current = relay.get("actions/runs/" + str(run_id))
        copy_pages = evidence / "copy-attempt-pages"
        copy_pages.mkdir()
        current_jobs = relay.get_attempt_jobs({"run_id": run_id, "run_attempt": 1}, copy_pages)
        need(current["id"] == run_id and current["head_sha"] == os.environ["GITHUB_SHA"] and current["run_attempt"] == 1 and current["event"] == "workflow_dispatch" and current["path"] == source.WORKFLOW and current["head_branch"] == admission["approved_branch"], "copy run identity")
        need(current["actor"]["login"] == current["triggering_actor"]["login"] == admission["approved_actor"], "copy actor identity")
        need(current_jobs["total_count"] == len(current_jobs["jobs"]) == 1, "single CPU-only job")
        job = current_jobs["jobs"][0]
        need(job["name"] == "renew-d5-artifacts-only" and job["status"] == "in_progress" and job["head_sha"] == current["head_sha"] and job["run_id"] == run_id and job["run_attempt"] == 1, "copy job identity")
        need(job.get("runner_id") and job["runner_id"] not in (331464, 331467) and not any("mi300" in label.lower() or "gpu" in label.lower() for label in job.get("labels", [])), "hosted CPU runner only")
        save("copy-run-api.json", current)
        save("copy-attempt-jobs-api.json", current_jobs)
        report["copy_job"] = {"run_id": run_id, "run_attempt": 1, "head_sha": current["head_sha"], "job_id": job["id"], "runner_id": job["runner_id"], "actor": admission["approved_actor"], "workflow": source.WORKFLOW, "only_upload_copy_job_not_producer_build": True}
        source.verify(shared)
        report["source_maps"] = {"consumer19": shared["source_trees"], "compiled37": shared["compiled_source_trees"], "producer14": shared["producer_source_trees"]}
        report["stage"] = "original-producer"
        origin_run = relay.get("actions/runs/" + str(shared["run_id"]))
        need(origin_run["status"] == "completed" and origin_run["conclusion"] == "cancelled", "actual old parent cancelled, not pipeline success")
        origin_pages = evidence / "origin-attempt-pages"
        origin_pages.mkdir()
        origin_jobs = relay.get_attempt_jobs(shared, origin_pages)
        save("producer-run.json", origin_run)
        save("producer-attempt-jobs.json", origin_jobs)
        records = []
        for page in range(1, 11):
            batch = relay.get("actions/runs/" + str(shared["run_id"]) + "/artifacts?per_page=100&page=" + str(page))
            save("origin-artifacts-page-" + str(page) + ".json", batch)
            records.extend(batch["artifacts"])
            if len(batch["artifacts"]) < 100:
                break
        else:
            raise ValueError("origin artifact pagination cap")
        for artifact in shared["artifacts"]:
            report["stage"] = "origin-archive-" + artifact["name"]
            meta = relay.get("actions/artifacts/" + str(artifact["id"]))
            relay.metadata(meta, artifact, shared)
            save("origin-artifact-" + str(artifact["id"]) + ".json", meta)
            files = io.unpack(io.archive_bytes(artifact), artifact)
            destination = out / artifact["name"]
            destination.mkdir(mode=0o755)
            for name, payload in files.items():
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(payload)
                need(path.stat().st_size == artifact["files"][name]["bytes"] and sha(path) == artifact["files"][name]["sha256"], "saved original payload bytes")
                report["payloads"].append({"role": artifact["name"], "name": name, **artifact["files"][name]})
                if name.endswith(".whl"):
                    save("wheel-RECORD.json", wheel_record.verify(payload, shared["gateway_native"]))
        need(len(report["payloads"]) == 6 and (evidence / "wheel-RECORD.json").is_file(), "all six original payloads")
        # This private projection describes verified archive pins, never an admission.
        # The original source inputs stay byte-identical with all five GPU flags false.
        verification = {**shared, "artifacts_pinned": True}
        context = frozen_producer.verify_completed(origin_run, origin_jobs["jobs"], verification, records, datetime.datetime.now(datetime.timezone.utc))
        need(context["producer_build_leg_only"] is True and context["producer_parent_pipeline_accepted"] is False, "origin build-leg scope only")
        save("origin-frozen-producer-context.json", context)
        report["original_producer"] = {"run_id": shared["run_id"], "attempt": 1, "head_sha": shared["producer_sha"], "job_id": shared["build_job_id"], "all17_build_steps_success": True, "parent_pipeline_status": origin_run["status"], "parent_pipeline_conclusion": origin_run["conclusion"], "parent_pipeline_success_accepted": False, "artifacts": shared["artifacts"], "native": shared["gateway_native"]}
        need(not io.git_output("status", "--porcelain", "--untracked-files=all"), "source unchanged after verification")
        report["stage"] = "all-original-payloads-verified-ready-for-copy-upload"
        report["passed"] = True
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "HTTP_status": getattr(error, "code", None), "frames": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name} for frame in traceback.extract_tb(error.__traceback__)[-8:]]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save("origin-copy-proof.json", report)
    print(json.dumps({"passed": report["passed"], "stage": report["stage"], "proof": str(evidence / "origin-copy-proof.json"), "products_to_upload": 4 if report["passed"] else 0, "build_executed": False, "GPU_charge": 0}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
