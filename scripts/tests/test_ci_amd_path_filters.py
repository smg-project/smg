"""Actual lane decisions must include changes to their AMD setup and hard gates."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/pr-test-rust.yml"
AREAS = (
    "chat-completions",
    "chat-symphony",
    "completions",
    "embeddings",
    "realtime",
    "agentic",
    "vendor",
    "go-bindings",
)


def lane_decisions(path, tmp_path, monkeypatch):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["detect-changes"]["steps"]
    outputs = tmp_path / "outputs"
    for key, value in {
        "PULL_REQUEST": "true",
        "FORCE_FULL": "false",
        "CHANGED_FILES": json.dumps([path]),
        "DOCS": "false",
        "NON_DOCS": "true",
        "TESTS": "false",
        "SYMPHONY": "false",
        "OUTSIDE_SYMPHONY": "true",
        "GITHUB_OUTPUT": str(outputs),
    }.items():
        monkeypatch.setenv(key, value)
    shell = next(step["run"] for step in steps if step.get("id") == "lanes")
    script = shell.split("python3 - <<'EOF'\n", 1)[1].rsplit("\nEOF", 1)[0]
    exec(compile(script, str(WORKFLOW), "exec"), {})
    decisions = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    return {area for area in AREAS if decisions[f"skip-{area}"] == "false"}


@pytest.mark.parametrize(
    "path,areas",
    [
        (
            "scripts/ci_setup_sglang_rocm_image.sh",
            "chat-completions chat-symphony completions embeddings agentic go-bindings",
        ),
        (
            "scripts/ci_setup_vllm_rocm_image.sh",
            "chat-completions chat-symphony completions embeddings realtime",
        ),
        ("scripts/ci_check_e2e_dependencies.py", " ".join(AREAS)),
        ("scripts/ci_check_vllm_engine_units.py", " ".join(AREAS)),
        ("scripts/ci_vllm_required_cases.json", " ".join(AREAS)),
        ("scripts/ci_download_model.sh", " ".join(AREAS)),
    ],
)
def test_amd_setup_and_gate_changes_select_their_lanes(tmp_path, monkeypatch, path, areas):
    assert lane_decisions(path, tmp_path, monkeypatch) == set(areas.split())


@pytest.mark.parametrize(
    "path",
    ["docs/example.md", "scripts/tau2/run_ab.py", ".github/workflows/ci-tooling-tests.yml"],
)
def test_unrelated_changes_do_not_select_gpu_lanes(tmp_path, monkeypatch, path):
    assert lane_decisions(path, tmp_path, monkeypatch) == set()


def test_unclassified_paths_keep_the_full_coverage_fallback(tmp_path, monkeypatch):
    assert lane_decisions("new-subsystem/source.py", tmp_path, monkeypatch) == set(AREAS)
