"""Require the installed-vLLM cases to execute once and pass in engine unit JUnit."""

import ast
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("ci_vllm_required_cases.json")


def _pytest_call(node, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == name
    )


def _vllm_skip(node):
    return (
        _pytest_call(node, "importorskip")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "vllm"
    )


def _scope_skips_vllm(nodes, seen):
    pending = list(nodes)
    found = False
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if _vllm_skip(node):
            seen.add(id(node))
            found = True
        pending.extend(ast.iter_child_nodes(node))
    return found


def _parameter_marks(node):
    return [
        mark
        for mark in node.decorator_list
        if isinstance(mark, ast.Call)
        and isinstance(mark.func, ast.Attribute)
        and mark.func.attr == "parametrize"
    ]


def _case_names(node, location):
    marks = _parameter_marks(node)
    if not marks:
        return [node.name]
    error = (
        f"Cannot enumerate required vLLM cases at {location}: "
        "use one literal parametrize with explicit distinct ASCII ids"
    )
    if len(marks) != 1:
        raise SystemExit(error)
    mark = marks[0]
    if len(mark.args) != 2 or not isinstance(mark.args[1], (ast.List, ast.Tuple)):
        raise SystemExit(error)
    names = mark.args[0]
    if not isinstance(names, ast.Constant) or not isinstance(names.value, str):
        raise SystemExit(error)
    names = [name.strip() for name in names.value.split(",")]
    if not all(name.isidentifier() for name in names) or len(set(names)) != len(names):
        raise SystemExit(error)
    if any(key.arg != "ids" for key in mark.keywords) or len(mark.keywords) != 1:
        raise SystemExit(error)
    ids = mark.keywords[0].value
    if not isinstance(ids, (ast.List, ast.Tuple)):
        raise SystemExit(error)
    labels = [item.value if isinstance(item, ast.Constant) else None for item in ids.elts]
    if not labels or len(labels) != len(mark.args[1].elts):
        raise SystemExit(error)
    if any(
        not isinstance(label, str)
        or not label
        or not label.isascii()
        or not all(char.isalnum() or char in "_.-" for char in label)
        for label in labels
    ):
        raise SystemExit(error)
    if len(set(labels)) != len(labels) or any(
        _pytest_call(value, "param") for value in ast.walk(mark.args[1])
    ):
        raise SystemExit(error)
    return [f"{node.name}[{label}]" for label in labels]


def gated_cases(root):
    """Inventory literal module/function vLLM gates without importing an engine."""
    cases = set()
    for source in sorted((root / "grpc_servicer" / "tests").glob("test_*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        module = ".".join(source.relative_to(root).with_suffix("").parts)
        if not any(_vllm_skip(node) for node in ast.walk(tree)):
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "pytest_generate_tests"
            for node in ast.walk(tree)
        ):
            raise SystemExit(
                f"Cannot enumerate required vLLM cases in {source}: dynamic collection"
            )

        seen = set()

        def collect(nodes, classes=(), inherited=False):
            scoped = _scope_skips_vllm(nodes, seen)
            inherited = inherited or scoped
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("Test"):
                        if _parameter_marks(node):
                            raise SystemExit(
                                "Cannot enumerate required vLLM class parameters: "
                                f"{source}:{node.name}"
                            )
                        collect(node.body, (*classes, node.name), inherited)
                    elif any(_vllm_skip(child) for child in ast.walk(node)):
                        raise SystemExit(
                            "Cannot enumerate vLLM gate outside a Test class: "
                            f"{source}:{node.name}"
                        )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    gated = _scope_skips_vllm(node.body, seen)
                    if node.name.startswith("test_") and (inherited or gated):
                        classname = ".".join((module, *classes))
                        for name in _case_names(node, f"{source}:{node.name}"):
                            case = f"{classname}::{name}"
                            if case in cases:
                                raise SystemExit(f"Duplicate required vLLM definition: {case}")
                            cases.add(case)
                    elif gated:
                        raise SystemExit(
                            f"Cannot enumerate vLLM gate outside a test: {source}:{node.name}"
                        )
                elif (inherited or any(_vllm_skip(child) for child in ast.walk(node))) and any(
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and child.name.startswith(("test_", "Test"))
                    for child in ast.walk(node)
                ):
                    raise SystemExit(
                        f"Cannot enumerate conditional vLLM test definitions in {source}"
                    )

        collect(tree.body)
        if any(_vllm_skip(node) and id(node) not in seen for node in ast.walk(tree)):
            raise SystemExit(f"Cannot enumerate a nested or conditional vLLM gate in {source}")
    return cases


def load_required(root, manifest):
    required = json.loads(manifest.read_text())
    if (
        not isinstance(required, list)
        or not required
        or any(
            not isinstance(case, str)
            or case.count("::") != 1
            or not all(case.split("::"))
            for case in required
        )
        or len(set(required)) != len(required)
    ):
        raise SystemExit("Required vLLM test manifest must contain distinct nonempty case IDs")
    discovered = gated_cases(root)
    missing = sorted(discovered - set(required))
    stale = sorted(set(required) - discovered)
    if missing or stale:
        raise SystemExit(
            "Required vLLM manifest differs from gated source: "
            f"missing={missing!r}, stale={stale!r}"
        )
    return required


def require_vllm(junit):
    required = load_required(ROOT, MANIFEST)
    cases = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        cases.setdefault(name, []).append(case)
    bad = []
    for name in required:
        found = cases.get(name, [])
        if len(found) != 1 or any(
            child.tag in {"failure", "error", "skipped", "rerunFailure", "rerunError"}
            for case in found
            for child in case
        ):
            bad.append(name)
    if bad:
        raise SystemExit("Required vLLM cases missing, duplicated or not passing: " + repr(bad))
    print(f"All {len(required)} required installed-vLLM cases executed once and passed")


if __name__ == "__main__":
    require_vllm(Path(sys.argv[1]))
