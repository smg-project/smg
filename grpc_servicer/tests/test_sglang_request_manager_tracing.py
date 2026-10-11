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


def _guarded_by_enable_trace(node: ast.AST, call_name: str) -> bool:
    """Whether every `call_name` call sits under an `if self.enable_trace`."""
    found = False
    for branch in ast.walk(node):
        if not isinstance(branch, ast.If):
            continue
        test = branch.test
        if isinstance(test, ast.Attribute) and test.attr == "enable_trace":
            if call_name in _calls(ast.Module(body=branch.body, type_ignores=[])):
                found = True
    return found


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
