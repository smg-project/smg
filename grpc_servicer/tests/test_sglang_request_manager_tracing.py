"""The Python SGLang servicer starts each request's trace from the Router's
context (smg-lab#182). The request manager imports SGLang at module level, so
this reads its source: the wiring the tokenizer manager has behind SGLang's
own frontend must be there, under the engine's tracing switch."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "request_manager.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _calls(node: ast.AST) -> list[str]:
    names = []
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            func = call.func
            names.append(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    return names


def _tests_enable_trace(test: ast.expr) -> bool:
    """Whether an `if` test is `self.enable_trace`, alone or as one operand of an `and`."""
    if isinstance(test, ast.Attribute):
        return test.attr == "enable_trace"
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return any(_tests_enable_trace(value) for value in test.values)
    return False


def _guarded_by_enable_trace(node: ast.AST, call_name: str) -> bool:
    """Whether every `call_name` call in `node` sits under an `if self.enable_trace`
    (and there is at least one): a call added outside the switch fails this."""
    total = _calls(node).count(call_name)
    guarded = 0
    for branch in ast.walk(node):
        if not isinstance(branch, ast.If):
            continue
        if _tests_enable_trace(branch.test):
            guarded += _calls(ast.Module(body=branch.body, type_ignores=[])).count(call_name)
    return total > 0 and guarded == total


def test_the_request_state_starts_the_trace_from_the_calls_metadata():
    tree = ast.parse(SOURCE.read_text())
    init = _function(_class(tree, "GrpcRequestManager"), "_req_stats_init")
    calls = _calls(init)
    # The trace context comes from the call's metadata through the shared helper.
    assert "trace_headers" in calls and "invocation_metadata" in calls
    assert _guarded_by_enable_trace(init, "init_trace_ctx")
    # The tokenize slice closes at once, and the scheduler gets the stats on
    # the request, as behind the tokenizer manager.
    assert _guarded_by_enable_trace(init, "set_tokenize_finish_time")
    assigned = [
        target.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
        if target.value.id == "obj"
    ]
    assert "time_stats" in assigned


def test_the_manager_reads_the_engines_tracing_switch_once():
    tree = ast.parse(SOURCE.read_text())
    init = _function(_class(tree, "GrpcRequestManager"), "__init__")
    assigned = [
        target.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    ]
    assert "enable_trace" in assigned
    switch = _function(tree, "_tracing_enabled")
    assert "get_observability" in _calls(switch)


def test_request_tracing_initializes_as_sglangs_own_frontend_does():
    tree = ast.parse(SOURCE.read_text())
    init = _function(tree, "init_request_tracing")
    calls = _calls(init)
    assert "process_tracing_init" in calls and "trace_set_thread_info" in calls
    assert "_tracing_enabled" in calls


def test_the_abort_paths_end_the_requests_root_span_under_the_switch():
    tree = ast.parse(SOURCE.read_text())
    manager = _class(tree, "GrpcRequestManager")
    for name in ("abort_request", "_handle_abort_req"):
        assert _guarded_by_enable_trace(_function(manager, name), "set_finished_time"), name


def test_request_tracing_survives_an_sglang_without_it_or_a_failing_setup():
    tree = ast.parse(SOURCE.read_text())
    init = _function(tree, "init_request_tracing")
    handled = {
        handler.type.id
        for node in ast.walk(init)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name)
    }
    assert {"ImportError", "RuntimeError"} <= handled, handled
    # process_tracing_init runs inside a try that handles RuntimeError
    guarded = any(
        "process_tracing_init" in _calls(ast.Module(body=node.body, type_ignores=[]))
        for node in ast.walk(init)
        if isinstance(node, ast.Try)
        and any(isinstance(h.type, ast.Name) and h.type.id == "RuntimeError" for h in node.handlers)
    )
    assert guarded
