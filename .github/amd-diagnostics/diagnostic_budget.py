"""One absolute job budget for artifact reuse, installation, and requests."""

import json
import os
import subprocess
import time
from pathlib import Path

import reuse_foundation as reuse

COMMAND = ["bash", ".github/amd-diagnostics/run_qwen_diagnostic.sh"]


def main():
    budget = reuse.consumer_budget()
    path = Path(os.environ["RUNNER_TEMP"]) / f"qwen-budget-{budget['consumer_job']}.json"
    path.write_text(json.dumps(budget, indent=2) + "\n")
    env = dict(os.environ, QWEN_DIAG_BUDGET_FILE=str(path))
    try:
        remaining = int(1000 * (budget["diagnostic_deadline_epoch"] - time.time())) / 1000
        reuse.require(remaining > 0, "diagnostic budget elapsed before execution")
        return subprocess.run(
            ["timeout", "--signal=INT", "--kill-after=15s", f"{remaining}s", *COMMAND],
            cwd=reuse.ROOT,
            env=env,
            check=False,
        ).returncode
    finally:
        out = reuse.ROOT / "diagnostic-results"
        out.mkdir(exist_ok=True)
        (out / "job-budget.json").write_bytes(path.read_bytes())


if __name__ == "__main__":
    raise SystemExit(main())
