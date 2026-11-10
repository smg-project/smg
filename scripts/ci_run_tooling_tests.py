"""Run the bounded CPU tooling suites: the baseline, then the suite of each component
and regression pair, every one of which the tree must hold."""

import subprocess
import sys
from pathlib import Path

BASELINE = (
    "scripts/tests/test_ci_prepared_backend_env.py",
    "scripts/tests/test_ci_gpu_runner_image.py",
    "scripts/tests/test_ci_tooling_tests.py",
    "scripts/tests/test_ci_amd_path_filters.py",
    "scripts/tests/test_ci_tokenspeed_scheduler_pin.py",
    "e2e_test/infra/test_epd_cmd_builders.py",
)
COMPONENTS = (
    ("e2e_test/infra/cloud_credentials.py", "e2e_test/infra/test_cloud_credentials.py"),
    ("scripts/ci_responses_evidence.py", "scripts/tests/test_ci_responses_evidence.py"),
    ("e2e_test/infra/mcp_readiness.py", "e2e_test/infra/test_mcp_readiness.py"),
    ("scripts/release/release.sh", "scripts/tests/test_release_tools.py"),
    ("scripts/branch-name-check.sh", "scripts/tests/test_branch_name_check.py"),
    ("scripts/ci_version_only.py", "scripts/tests/test_ci_version_only.py"),
    ("scripts/bfcl/arm_probe.py", "scripts/tests/test_bfcl_arm_probe.py"),
    ("scripts/codex_review_post.py", "scripts/tests/test_codex_review_post.py"),
    ("scripts/codex_review_skills.py", "scripts/tests/test_codex_review_skills.py"),
    (".github/workflows/codex-code-review.yml", "scripts/tests/test_codex_review_workflow.py"),
    (".github/workflows/claude-code-review.yml", "scripts/tests/test_claude_review_workflow.py"),
    (
        "scripts/k8s-runner-resources/runner-values-cpu.yaml",
        "scripts/tests/test_runner_values_policy.py",
    ),
)
# Regression suites of scripts that predate them. A pair is named in the change
# that brings the suite, and both paths must exist: a suite the tree lacks is an
# error, not a notice, so a deleted or renamed suite is noticed at once.
REGRESSIONS = (
    ("e2e_test/infra/worker.py", "e2e_test/infra/test_worker_grpc_health.py"),
    ("e2e_test/infra/gateway.py", "e2e_test/infra/test_cloud_mcp_readiness.py"),
    (
        "e2e_test/benchmarks/nightly_summarize.py",
        "scripts/tests/test_nightly_summary_errors.py",
    ),
    (
        ".github/workflows/e2e-gpu-job.yml",
        "scripts/tests/test_ci_final_environment_scope.py",
    ),
    (
        ".github/workflows/e2e-gpu-job.yml",
        "scripts/tests/test_ci_e2e_log_pump_lifetime.py",
    ),
    (
        "scripts/ci_verify_sglang_rocm_cohort.py",
        "scripts/tests/test_ci_engine_http_dependencies.py",
    ),
    ("scripts/bfcl/run_ab.py", "scripts/tests/test_bfcl_run_ab.py"),
    (".github/workflows/nightly-bfcl.yml", "scripts/tests/test_bfcl_score_caller.py"),
)


def test_paths(root):
    tests = list(BASELINE)
    for path in tests:
        if not (root / path).is_file():
            raise ValueError(f"Required CPU suite is missing: {path}")
    for source, suite in COMPONENTS + REGRESSIONS:
        for path in (source, suite):
            if not (root / path).is_file():
                raise ValueError(f"CPU test component is incomplete: {path}")
        tests.append(suite)
    return tests


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    tests = test_paths(root)
    print("CPU suites selected:\n" + "\n".join(tests), flush=True)
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--noconftest",
                "--junitxml=ci-tooling-results.xml",
                *tests,
            ],
            cwd=root,
        )
    )
