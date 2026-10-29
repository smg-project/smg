"""Use real pip and a local wheel with an unmet dependency; no network or engine."""

import json
import os
import subprocess
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_real_pip_dependency_failure_is_preserved_at_both_stages(tmp_path):
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    env = {**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    for key in ["PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT"]:
        env.pop(key, None)

    def audit(directory, stage, expected):
        process = subprocess.run(
            [
                str(python),
                str(ROOT / "scripts/ci_check_e2e_dependencies.py"),
                str(directory),
                stage,
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert process.returncode == expected, process.stdout + process.stderr
        status = json.loads((directory / f"{stage}-status.json").read_text())
        assert status["pip_check_exit_code"] == expected
        assert status["python"] == str(python)
        return (directory / f"{stage}-pipcheck.log").read_text()

    for stage in ["installed", "final"]:
        assert "No broken requirements found." in audit(tmp_path / "clean", stage, 0)

    wheel = tmp_path / "smg_ci_dependency_probe-0-py3-none-any.whl"
    info = "smg_ci_dependency_probe-0.dist-info/"
    with zipfile.ZipFile(wheel, "w") as output:
        output.writestr(
            info + "METADATA",
            "Metadata-Version: 2.1\nName: smg-ci-dependency-probe\nVersion: 0\n"
            "Requires-Dist: smg-ci-absent-for-dependency-control==0\n",
        )
        output.writestr(
            info + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        output.writestr(info + "RECORD", "")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for stage in ["installed", "final"]:
        assert "smg-ci-absent-for-dependency-control" in audit(tmp_path / "broken", stage, 1)
        packages = json.loads((tmp_path / "broken" / f"{stage}-packages.json").read_text())
        assert any(p["name"] == "smg-ci-dependency-probe" and p["requires"] for p in packages)
