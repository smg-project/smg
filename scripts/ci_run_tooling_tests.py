"""Run bounded CPU tooling suites available at this PR's stack layer."""

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
    ("scripts/ci_check_e2e_dependencies.py", "scripts/tests/test_ci_e2e_dependencies.py"),
    ("scripts/ci_check_vllm_engine_units.py", "scripts/tests/test_ci_vllm_engine_units.py"),
)
# These scripts predate their regression suites. Earlier stack layers must not
# claim to have run the later suites, or be required to contain later fixes.
REGRESSIONS = (
    ("scripts/ci_download_model.sh", "scripts/tests/test_ci_download_model.py"),
    ("scripts/ci_install_e2e_deps.sh", "scripts/test_grpc_ci_dependencies.py"),
    ("e2e_test/infra/worker.py", "e2e_test/infra/test_vllm_model_name.py"),
    ("e2e_test/responses/conftest.py", "e2e_test/fixtures/test_responses_startup.py"),
)


def test_paths(root):
    tests = list(BASELINE)
    for source, suite in COMPONENTS + REGRESSIONS:
        available = (root / suite).is_file()
        if (source, suite) in COMPONENTS and (root / source).is_file():
            available = True
        if available:
            for path in (source, suite):
                if not (root / path).is_file():
                    raise ValueError(f"CPU test component is incomplete: {path}")
            tests.append(suite)
        else:
            print(f"Suite not present at this stack layer (not tested): {suite}", flush=True)
    for path in tests:
        if not (root / path).is_file():
            raise ValueError(f"Required CPU suite is missing: {path}")
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
