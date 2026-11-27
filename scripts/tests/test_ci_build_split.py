"""Contract tests for splitting `build-wheel` and `build-go-ffi` in CI.

Verify that:
- `scripts/ci_build_wheel.sh` builds only the Python wheel (no Go FFI build).
- `bindings/golang/Makefile` keeps `release` as the default profile while honoring
  `CARGO_PROFILE`, `BUILD_MODE`, and `CARGO_BUILD_DIR` overrides.
- `pr-test-rust.yml` runs `build-go-ffi` in parallel with `build-wheel` using
  `--profile ci`, dedicated rust-cache key, and workspace `target/`, gates only
  Go consumers (and `finish`) on `build-go-ffi`, and keeps artifact download
  paths intact.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "pr-test-rust.yml"
BUILD_WHEEL_SCRIPT = ROOT / "scripts" / "ci_build_wheel.sh"
GOLANG_MAKEFILE = ROOT / "bindings" / "golang" / "Makefile"
SETUP_RUST_ACTION = ROOT / ".github" / "actions" / "setup-rust" / "action.yml"
DOWNLOAD_ARTIFACTS_ACTION = ROOT / ".github" / "actions" / "download-build-artifacts" / "action.yml"


def _load_workflow_jobs() -> dict:
    """Return the parsed `jobs` mapping from `pr-test-rust.yml`."""
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def _as_list(needs: str | list[str] | None) -> list[str]:
    """Normalize a GitHub Actions `needs` field into a list of job names."""
    if needs is None:
        return []
    return [needs] if isinstance(needs, str) else list(needs)


def _dry_run_make(*args: str) -> str:
    """Run `make -n` in `bindings/golang` with the given arguments and return stdout."""
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    result = subprocess.run(
        ["make", "-C", str(GOLANG_MAKEFILE.parent), "-n", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_ci_build_wheel_script_builds_only_python_wheel() -> None:
    """`scripts/ci_build_wheel.sh` builds the Python wheel without invoking Go FFI."""
    script = BUILD_WHEEL_SCRIPT.read_text()
    assert "maturin build --profile ci" in script
    assert "bindings/golang" not in script
    assert "make build" not in script
    assert "libsmg_go" not in script


@pytest.mark.parametrize(
    "make_args,expected_flag,expected_dir",
    [
        (["build"], "--profile release", "/bindings/golang/target/release/libsmg_go."),
        (["build", "CARGO_PROFILE=ci"], "--profile ci", "/bindings/golang/target/ci/libsmg_go."),
        (["build", "BUILD_MODE=ci"], "--profile ci", "/bindings/golang/target/ci/libsmg_go."),
        (
            ["build", "CARGO_PROFILE=ci", "CARGO_BUILD_DIR=/tmp/smg-target"],
            "--profile ci",
            "/tmp/smg-target/ci/libsmg_go.",
        ),
        (
            ["build", "CARGO_PROFILE=dev"],
            "--profile dev",
            "/bindings/golang/target/debug/libsmg_go.",
        ),
        (
            ["build", "BUILD_MODE=debug"],
            "--profile dev",
            "/bindings/golang/target/debug/libsmg_go.",
        ),
    ],
)
def test_golang_makefile_profile_and_target_dir_handling(
    make_args: list[str], expected_flag: str, expected_dir: str
) -> None:
    """`bindings/golang/Makefile` defaults to `release` and honors profile/dir overrides."""
    output = _dry_run_make(*make_args)
    assert f"cargo build {expected_flag} --manifest-path Cargo.toml" in output
    assert expected_dir in output


def test_build_wheel_and_build_go_ffi_run_in_parallel() -> None:
    """`build-wheel` and `build-go-ffi` are independent parallel jobs with separate artifacts."""
    jobs = _load_workflow_jobs()
    build_wheel = jobs["build-wheel"]
    build_go_ffi = jobs["build-go-ffi"]

    assert _as_list(build_wheel.get("needs")) == []
    assert _as_list(build_go_ffi.get("needs")) == []

    wheel_uploads = [
        step.get("with", {}).get("name")
        for step in build_wheel["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert "smg-wheel" in wheel_uploads
    assert "go-ffi-library" not in wheel_uploads

    setup_rust = next(
        step for step in build_go_ffi["steps"] if step.get("uses") == "./.github/actions/setup-rust"
    )
    assert setup_rust["with"]["cache-key"] == "go-ffi"

    build_step = next(
        step for step in build_go_ffi["steps"] if step.get("name") == "Build Go FFI library"
    )
    assert "CARGO_PROFILE=ci" in build_step["run"]
    assert 'CARGO_BUILD_DIR="$PWD/target"' in build_step["run"]

    upload_step = next(
        step
        for step in build_go_ffi["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    )
    assert upload_step["with"]["name"] == "go-ffi-library"
    assert upload_step["with"]["path"] == "target/ci/libsmg_go.*"


def test_only_go_consumers_and_finish_depend_on_build_go_ffi() -> None:
    """Only Go jobs and `finish` depend on `build-go-ffi`; non-Go jobs are unblocked."""
    jobs = _load_workflow_jobs()

    assert _as_list(jobs["go-unit-tests"].get("needs")) == ["build-go-ffi"]
    assert set(_as_list(jobs["go-bindings-e2e"].get("needs"))) == {
        "build-wheel",
        "build-go-ffi",
        "detect-changes",
    }
    assert "needs.build-go-ffi.result == 'success'" in jobs["go-bindings-e2e"]["if"]
    assert set(_as_list(jobs["go-bindings-benchmark"].get("needs"))) == {
        "build-wheel",
        "build-go-ffi",
    }

    dependents = {
        name for name, job in jobs.items() if "build-go-ffi" in _as_list(job.get("needs"))
    }
    assert dependents == {"go-unit-tests", "go-bindings-e2e", "go-bindings-benchmark", "finish"}


def test_finish_gates_on_build_go_ffi() -> None:
    """`finish` includes `build-go-ffi` in both `needs` and its `Check CI result` step."""
    jobs = _load_workflow_jobs()
    finish = jobs["finish"]
    assert "build-go-ffi" in _as_list(finish.get("needs"))
    check_step = next(step for step in finish["steps"] if step.get("name") == "Check CI result")
    assert "needs.build-go-ffi.result" in check_step["run"]


def test_actions_keep_cache_clean_and_download_path_compatible() -> None:
    """`setup-rust` drops `bindings/golang/target` and `download-build-artifacts` keeps paths."""
    setup_rust = yaml.safe_load(SETUP_RUST_ACTION.read_text())
    rust_cache_step = next(
        step
        for step in setup_rust["runs"]["steps"]
        if str(step.get("uses", "")).startswith("Swatinem/rust-cache")
    )
    assert "cache-directories" not in rust_cache_step["with"]
    assert "bindings/golang/target" not in SETUP_RUST_ACTION.read_text()

    download_action = yaml.safe_load(DOWNLOAD_ARTIFACTS_ACTION.read_text())
    go_step = next(
        step
        for step in download_action["runs"]["steps"]
        if step.get("name") == "Download Go FFI library"
    )
    assert go_step["with"]["name"] == "go-ffi-library"
    assert go_step["with"]["path"] == "bindings/golang/target/release/"
