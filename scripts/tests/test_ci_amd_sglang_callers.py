"""Configuration and admission controls for the opt-in SGLang callers."""

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CALLER = yaml.safe_load((ROOT / ".github/workflows/pr-test-rust.yml").read_text())
CALLEE = yaml.safe_load((ROOT / ".github/workflows/e2e-gpu-job.yml").read_text())
TARGETS = {
    "e2e-1gpu-chat-zmq-amd": (
        "SMG_RUN_AMD_SGLANG_ZMQ",
        {"test_dirs": "e2e_test/chat_completions", "connection_mode": "zmq", "min_selected": 209},
    ),
    "e2e-2gpu-pd-amd": (
        "SMG_RUN_AMD_SGLANG_PD",
        {"test_dirs": "e2e_test/router", "kv_backend": "mooncake", "min_selected": 15},
    ),
}
assert "e2e-1gpu-chat-zmq-amd" in CALLER["jobs"]
PRESENT = [name for name in TARGETS if name in CALLER["jobs"]]


def evaluate_gate(expression, flags, event, build, detect, cancelled, skip):
    expression = expression.replace("always()", "True").replace("cancelled()", repr(cancelled))
    expression = expression.replace("github.event_name", repr(event))
    expression = expression.replace("needs.build-wheel.result", repr(build))
    expression = expression.replace("needs.detect-changes.result", repr(detect))
    expression = expression.replace(
        "needs.detect-changes.outputs.skip-chat-completions", repr(skip)
    )
    expression = re.sub(r"vars\.([A-Z0-9_]+)", lambda m: repr(flags.get(m[1], "")), expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    tree = ast.parse("(" + expression.strip() + ")", mode="eval")
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Constant,
    )
    assert all(isinstance(node, allowed) for node in ast.walk(tree))
    return eval(compile(tree, "<actual caller gate>", "eval"), {"__builtins__": {}})


@pytest.mark.parametrize("name", PRESENT)
@pytest.mark.parametrize(
    (
        "global_on",
        "dedicated",
        "pr_on",
        "event",
        "build",
        "detect",
        "cancelled",
        "skip",
        "expected",
    ),
    [
        (False, False, False, "push", "success", "success", False, "false", False),
        (True, False, False, "push", "success", "success", False, "false", False),
        (False, True, False, "push", "success", "success", False, "false", False),
        (True, True, False, "push", "success", "success", False, "false", True),
        (True, True, False, "push", "failure", "success", False, "false", False),
        (True, True, False, "pull_request", "success", "success", False, "false", False),
        (True, True, True, "pull_request", "success", "success", False, "false", True),
        (True, True, True, "pull_request", "success", "failure", False, "false", False),
        (True, True, True, "push", "success", "success", True, "false", False),
        (True, True, True, "push", "success", "success", False, "true", False),
    ],
)
def test_actual_caller_gate_stays_off_until_all_applicable_optins_are_present(
    name, global_on, dedicated, pr_on, event, build, detect, cancelled, skip, expected
):
    flags = {}
    for key, enabled in (
        ("SMG_RUN_AMD_LEGS", global_on),
        (TARGETS[name][0], dedicated),
        ("SMG_RUN_AMD_ON_PRS", pr_on),
    ):
        if enabled:
            flags[key] = "true"
    assert (
        evaluate_gate(CALLER["jobs"][name]["if"], flags, event, build, detect, cancelled, skip)
        is expected
    )


@pytest.mark.parametrize("name", PRESENT)
def test_full_selection_and_real_callee_interface_are_preserved(name):
    values = CALLER["jobs"][name]["with"]
    inputs = CALLEE.get("on", CALLEE.get(True))["workflow_call"]["inputs"]
    assert set(values) <= set(inputs)
    assert values["engine"] == "sglang"
    assert values["gpu_vendor"] == "amd"
    assert values["final_environment_check"] is True
    assert "test_filter" not in values
    assert all(values[key] == value for key, value in TARGETS[name][1].items())
    assert values["container_image"] == (
        "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:"
        "4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
    )


@pytest.mark.parametrize("name", PRESENT)
def test_caller_failure_or_cancellation_reaches_finish(name):
    finish = CALLER["jobs"]["finish"]
    assert finish["needs"].count(name) == 1
    check = next(step["run"] for step in finish["steps"] if step.get("name") == "Check CI result")
    assert check.count("needs." + name + ".result") == 1
    assert "bad='^(failure|cancelled)$'" in check
