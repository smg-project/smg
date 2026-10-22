"""Explicit temporary probe preparation; the default path skips before setup."""

import os
import re
import time
import uuid
from pathlib import Path

import pytest

from .pd_probe_contract import validate_prefix
from .pd_probe_prefix import read_prefix
from .pd_scheduler_compat import require_installed_contract

PROBE = "TestPDSchedulerObserverProbe"


def explicit_context(request):
    if request.config.getoption("keyword").strip() != PROBE:
        pytest.skip("temporary observer probe requires exact explicit selection")
    from pytest_rerunfailures import get_reruns_count

    selected = [
        item for item in request.session.items if item.getparent(pytest.Class) is request.node
    ]
    if len(selected) != 1:
        raise ValueError("probe requires exactly one selected class item")
    reruns = get_reruns_count(selected[0])
    if type(reruns) is not int or reruns != 0:
        raise ValueError("unknown or nonzero effective probe rerun policy")
    env = os.environ
    run, attempt = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_RUN_ATTEMPT", "")
    if (
        env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in (run, attempt))
        or re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")) is None
        or not env.get("RUNNER_NAME")
        or env.get("E2E_JOB_MARKER") != f"e2e-{run}-{attempt}-{env.get('RUNNER_NAME', '')}"
        or env.get("GITHUB_REPOSITORY") != "smg-project/smg"
        or not env.get("E2E_LOG_DIR")
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", env.get("E2E_JOB_MARKER", "")) is None
        or env.get("E2E_RUNTIME") != "vllm"
        or env.get("E2E_GPU_TIER") != "4"
        or env.get("E2E_VLLM_KV_BACKEND") not in ("nixl", "moriio")
        or env.get("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0") != "0"
        or request.config.getoption("force_reruns", default=None) is not None
        or request.config.getoption("reruns_mode", default="immediate") == "append"
    ):
        raise ValueError("invalid explicit probe identity/lane/retry policy")
    import torch

    if not torch.version.hip:
        raise ValueError("probe requires a ROCm torch build; no GPU query performed")
    contract = require_installed_contract()
    directory = Path(env["E2E_LOG_DIR"]).resolve() / "pd-observer-probe" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    context = {"run": run, "attempt": attempt, "trace_dir": directory}
    request.node.add_marker(
        pytest.mark.workers(parallel_start=True, scheduler_observation=context), append=False
    )
    return context, contract["root"]


def observe_request(directory, workers, identity, state, expected_root):
    deadline = time.monotonic() + 30
    while True:
        try:
            files = read_prefix(directory)
            state["files"] = files
            validate_prefix(files, workers, identity, expected_root)
            return files
        except (ValueError, OSError, KeyError, TypeError, AttributeError, RecursionError) as exc:
            if time.monotonic() >= deadline:
                raise ValueError("unknown startup/admission prefix after 30 seconds") from exc
            time.sleep(0.2)
