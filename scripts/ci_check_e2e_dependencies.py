"""Preserve and enforce the actual interpreter's complete pip dependency check."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    directory, stage = Path(sys.argv[1]), sys.argv[2]
    if stage not in {"installed", "final"}:
        raise ValueError("Expected installed or final stage")
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (directory / f"{stage}-pipcheck.log").write_text(result.stdout)
    (directory / f"{stage}-status.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "prefix": sys.prefix,
                "stage": stage,
                "pip_check_exit_code": result.returncode,
                "run": os.environ.get("GITHUB_RUN_ID"),
                "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "source": os.environ.get("GITHUB_SHA"),
                "engine": os.environ.get("E2E_ENGINE"),
                "runner": os.environ.get("RUNNER_NAME"),
                "constraint": os.environ.get("PIP_CONSTRAINT"),
            },
            indent=2,
        )
        + "\n"
    )
    packages = [
        {
            "name": package.metadata["Name"],
            "version": package.version,
            "location": str(package.locate_file("")),
            "requires": package.requires or [],
        }
        for package in importlib.metadata.distributions()
    ]
    (directory / f"{stage}-packages.json").write_text(json.dumps(packages, indent=2) + "\n")
    print(result.stdout, end="")
    if os.environ.get("SMG_TS_PYTHON173_LANE") == "python":
        origins = subprocess.run([sys.executable, str(Path(__file__).with_name("ci_check_python173_origins.py")), stage], check=False)
        if origins.returncode:
            return result.returncode or origins.returncode
    if os.environ.get("SMG_CI_RUNTIME_STATE_DIR"):
        audit = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("ci_runtime_constraints.py")),
                "verify",
                os.environ["SMG_CI_RUNTIME_STATE_DIR"],
                stage,
                str(directory),
            ],
            check=False,
        )
        return result.returncode or audit.returncode
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
