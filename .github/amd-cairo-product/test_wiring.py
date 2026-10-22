"""Temporary workflow scope and immutable complete product-helper wiring."""

import hashlib
import json
from pathlib import Path

import budget
import proof
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_dispatch_only_single_job_has_no_device_or_model_mounts():
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-test-rust.yml").read_text())
    assert workflow.get("on", workflow.get(True)) == {"workflow_dispatch": None}
    assert list(workflow["jobs"]) == ["cairo-product-setup"]
    job = workflow["jobs"]["cairo-product-setup"]
    assert job["timeout-minutes"] == 5
    assert job["container"] == {"image": budget.IMAGE}
    assert job["env"]["SMG_QA_IMAGE"] == budget.IMAGE
    assert all(
        job["env"][name] == ""
        for name in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
    )
    assert not any(name in job for name in ["services", "strategy"])
    assert workflow["concurrency"]["cancel-in-progress"] is False
    steps = job["steps"]
    core = next(step for step in steps if step.get("run", "").endswith("/budget.py"))
    assert core["timeout-minutes"] == 4 and core["shell"] == "bash"
    final = next(step for step in steps if step.get("run", "").endswith("devices-after"))
    assert final["if"] == "always()"
    upload = next(
        step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()" and upload["with"]["path"] == "cairo-results/"


def test_exact_helper_is_sourced_once_and_is_not_rewritten():
    helper = ROOT / "scripts/ci_setup_vllm_rocm_image.sh"
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == proof.HELPER_SHA
    run = Path(__file__).with_name("run.sh").read_text()
    assert run.count("source scripts/ci_setup_vllm_rocm_image.sh") == 1
    assert not any(command in run for command in ["apt-get", "pip install", "pycairo-1.29.2"])
    inputs = json.loads(budget.INPUTS.read_text())
    assert inputs["ready"] is False
    assert inputs["automatic_retry"] is False
    assert inputs["GPU_device_mapping"] is False and inputs["models_mount"] is False
