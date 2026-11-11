"""Exercise the actual planner without dispatching a workflow or using hardware."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/nightly-pd.yml"
READ_IMAGE = "example.invalid/moriio-read@sha256:" + "a" * 64
DEFAULT_AMD_OPTIONS = (
    "--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host "
    "--shm-size=32g --volume /models:/models --env HF_HOME=/models --env HF_TOKEN"
)


def plan(
    tmp_path,
    engines="",
    enabled="false",
    options="",
    test_filter="",
    base_options="",
    event_name="workflow_dispatch",
    schedule="",
    read_image=READ_IMAGE,
):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    script = next(
        step["run"] for step in workflow["jobs"]["plan"]["steps"] if step.get("id") == "plan"
    )
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env=os.environ
        | {
            "ENGINES": engines,
            "EVENT_NAME": event_name,
            "SCHEDULE": schedule,
            "AMD_ENABLED": enabled,
            "AMD_CONTAINER_OPTIONS": base_options,
            "AMD_RDMA_OPTIONS": options,
            "AMD_MORIIO_READ_IMAGE": read_image,
            "TEST_FILTER": test_filter,
            "GITHUB_OUTPUT": str(output),
        },
        cwd=WORKFLOW.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    values = {}
    if output.exists():
        for line in output.read_text().splitlines():
            key, value = line.split("=", 1)
            values[key] = json.loads(value)
    return result, values


@pytest.mark.parametrize("enabled,count", [("false", 5), ("true", 7)])
def test_default_matrix_keeps_read_rdma_opt_in(tmp_path, enabled, count):
    result, values = plan(tmp_path, enabled=enabled)
    assert result.returncode == 0, result.stderr
    assert len(values["matrix"]) == count
    assert not any(row.get("moriio_mode") == "read" for row in values["matrix"])
    assert "amd_read_rdma_options_json" not in values


@pytest.mark.parametrize(
    "options",
    [
        "--device=/dev/infiniband",
        "--device /dev/infiniband",
        "--device=/dev/infiniband:/dev/infiniband:rw",
        "--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0",
        "--device /dev/infiniband/rdma_cm --device /dev/infiniband/uverbs12",
        "--device=/dev/infiniband --env 'EXAMPLE=spaces and quotes'",
    ],
)
def test_explicit_read_uses_full_suite_and_captures_options_once(tmp_path, options):
    result, values = plan(tmp_path, engines="vllm-amd-moriio-read", options=options)
    assert result.returncode == 0, result.stderr
    assert values == {
        "amd_read_rdma_options_json": f"{DEFAULT_AMD_OPTIONS} {options}",
        "matrix": [
            {
                "engine": "vllm",
                "kv_backend": "moriio",
                "moriio_mode": "read",
                "moriio_backend": "rdma",
                "vendor": "amd",
                "container_image": READ_IMAGE,
            }
        ],
    }


@pytest.mark.parametrize(
    "base_options",
    [
        "--device=/dev/kfd --device=/dev/dri --volume /custom:/models --env HF_TOKEN",
        "--device /dev/kfd --device /dev/dri --volume '/cache with spaces:/models'",
    ],
)
def test_rdma_extras_preserve_custom_amd_base_options(tmp_path, base_options):
    extras = "--device=/dev/infiniband --ulimit memlock=-1 --cap-add=IPC_LOCK"
    result, values = plan(
        tmp_path,
        engines="vllm-amd-moriio-read",
        options=extras,
        base_options=base_options,
    )
    assert result.returncode == 0, result.stderr
    assert values["amd_read_rdma_options_json"] == f"{base_options} {extras}"


@pytest.mark.parametrize(
    "options",
    [
        "",
        "--device=/dev/kfd --device=/dev/dri",
        "--device=/dev/infiniband/rdma_cm",
        "--device=/dev/infiniband/uverbs0",
        "--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbsX",
        "--env EXAMPLE=--device=/dev/infiniband",
        "--volume /dev/infiniband:/dev/infiniband",
        "--device=/dev/infiniband-other",
        "--device",
        "'unterminated",
    ],
)
def test_missing_or_malformed_device_declarations_fail_before_matrix(tmp_path, options):
    result, values = plan(tmp_path, engines="vllm-amd-moriio-read", options=options)
    assert result.returncode != 0
    assert "matrix" not in values


@pytest.mark.parametrize("test_filter", ["1p1d", "not AssembledAtRuntime", " "])
def test_read_rejects_any_nonempty_filter(tmp_path, test_filter):
    result, values = plan(
        tmp_path,
        engines="vllm-amd-moriio-read",
        options="--device=/dev/infiniband",
        test_filter=test_filter,
    )
    assert result.returncode != 0
    assert "leave filter empty" in result.stderr
    assert not values


def test_write_dispatch_needs_no_read_device_options(tmp_path):
    result, values = plan(tmp_path, engines="vllm-amd", test_filter="1p1d")
    assert result.returncode == 0, result.stderr
    assert [row["kv_backend"] for row in values["matrix"]] == ["nixl", "moriio"]
    assert values["matrix"][1]["moriio_mode"] == "write"
    assert "amd_read_rdma_options_json" not in values


@pytest.mark.parametrize(
    "schedule,enabled,expected_vendors",
    [
        ("45 9 * * 0", "true", ["nvidia"] * 5),
        ("35 22 * * 0", "true", ["amd"] * 2),
        ("35 22 * * 0", "false", []),
        ("45 9 * * 1-6", "true", ["nvidia"] * 5 + ["amd"] * 2),
    ],
)
def test_read_addition_preserves_shared_node_schedule_split(
    tmp_path,
    schedule,
    enabled,
    expected_vendors,
):
    result, values = plan(tmp_path, enabled=enabled, event_name="schedule", schedule=schedule)
    assert result.returncode == 0, result.stderr
    assert [row.get("vendor", "nvidia") for row in values["matrix"]] == expected_vendors
    assert not any(row.get("moriio_mode") == "read" for row in values["matrix"])
    assert "amd_read_rdma_options_json" not in values


@pytest.mark.parametrize(
    "image",
    [
        "",
        "vllm/vllm-openai-rocm:v0.31.0",
        "bad image@sha256:" + "a" * 64,
        "vllm/vllm-openai-rocm@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f",
    ],
)
def test_read_requires_distinct_digest_image_before_gpu_matrix(tmp_path, image):
    result, values = plan(
        tmp_path,
        engines="vllm-amd-moriio-read",
        options="--device=/dev/infiniband",
        read_image=image,
    )
    assert result.returncode != 0
    assert "matrix" not in values


def test_write_ignores_unset_read_image_and_keeps_old_image_path(tmp_path):
    result, values = plan(tmp_path, engines="vllm-amd", read_image="")
    assert result.returncode == 0
    assert all("container_image" not in row for row in values["matrix"])
