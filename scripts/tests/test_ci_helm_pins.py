"""Every workflow that sets up helm pins the same literal version. The chart
tests in the Python unit test job render with helm; the chart workflows lint and
release with it. One pin kept in several files drifts, and a chart that renders
with one helm and lints with another passes or fails for the wrong reason, so a
bump in one place has to be a bump in all of them. `latest` (the action's default)
and an expression are no pin: they resolve differently on different days or files."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted(
    path
    for suffix in ("yml", "yaml")
    for path in (ROOT / ".github" / "workflows").glob(f"*.{suffix}")
)
PIN = re.compile(r"v\d+\.\d+\.\d+")


def setup_helm_steps():
    """``(workflow, job, version)`` for every ``azure/setup-helm`` step."""
    steps = []
    for workflow in WORKFLOWS:
        jobs = yaml.safe_load(workflow.read_text()).get("jobs") or {}
        for job_id, job in jobs.items():
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).lower().startswith("azure/setup-helm@"):
                    version = (step.get("with") or {}).get("version")
                    steps.append((workflow.name, job_id, version))
    return steps


def test_every_setup_helm_step_pins_the_same_helm_version():
    steps = setup_helm_steps()
    assert len(steps) >= 2, "the chart workflows set up helm"
    unpinned = [step for step in steps if not PIN.fullmatch(str(step[2] or ""))]
    assert not unpinned, f"setup-helm steps without a literal version pin: {unpinned}"
    assert len({version for _, _, version in steps}) == 1, f"the helm pins differ: {steps}"
