"""CPU Source contracts and real canary lifecycle/reporting with inert SDK calls."""

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "numeric_canary", SCRIPTS / "ci_sglang_mxfp4_numeric.py"
)
numeric = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(numeric)


def added_validator_source():
    """Read the complete added module-level guard hunk, without running the SDK."""
    lines = (SCRIPTS / "patches/sglang-0.5.21-gptoss-gfx942.patch").read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if line.startswith("+_GPTOSS_REVISION ="))
    start = max(i for i in range(marker) if lines[i].startswith("@@"))
    end = next(i for i in range(marker + 1, len(lines)) if lines[i].startswith("@@"))
    return "\n".join(line[1:] for line in lines[start + 1 : end] if line.startswith("+"))


def actual_lifecycle_and_report(source):
    """Keep real main's lifecycle, PASS assignment, exception handler and JSON writer."""
    main = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    outer = next(n for n in main.body if isinstance(n, ast.Try))
    lifecycle = next(
        n
        for n in ast.walk(outer)
        if isinstance(n, ast.With)
        and isinstance(n.items[0].context_expr, ast.Call)
        and isinstance(n.items[0].context_expr.func, ast.Name)
        and n.items[0].context_expr.func.id == "distributed_numeric_scope"
    )
    passed = next(
        n
        for n in outer.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Constant)
        and n.value.value == "original_packed_slice_actual_generic_triton_numeric_pass"
    )
    selected = ast.Try(
        body=[copy.deepcopy(lifecycle), copy.deepcopy(passed)],
        handlers=copy.deepcopy(outer.handlers),
        orelse=[],
        finalbody=copy.deepcopy(outer.finalbody),
    )
    return compile(
        ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
        "actual-canary-report",
        "exec",
    )


class NumericCriterionFailure(RuntimeError):
    pass


class CleanupFailure(RuntimeError):
    pass


class NumericSourceTests(unittest.TestCase):
    def test_patched_postimages_match_observer(self):
        tree = ast.parse((SCRIPTS / "ci_sglang_mxfp4_fallback.py").read_text())
        files = ast.literal_eval(
            next(
                n.value
                for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "FILES" for t in n.targets)
            )
        )
        pins = json.loads((SCRIPTS / "ci_sglang_mxfp4_numeric_inputs.json").read_text())
        self.assertEqual(len(pins["sdk_files"]), 12)
        for name, (_, postimage) in files.items():
            self.assertEqual(pins["sdk_files"][name], postimage)

    def test_actual_resolver_has_exact_two_nodes(self):
        tree = ast.parse((ROOT / "e2e_test/infra/model_specs.py").read_text())
        nodes = [
            n
            for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == "_resolve_model_path")
            or (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "ROUTER_LOCAL_MODEL_PATH" for t in n.targets
                )
            )
        ]
        self.assertEqual(len(nodes), 2)

    def test_pinned_guard_has_four_nodes_and_shared_blob_policy_matches(self):
        tree = ast.parse(added_validator_source())
        names = {
            "_GPTOSS_REVISION",
            "_GPTOSS_METADATA_SHA256",
            "_original_shared_blob",
            "_validate_gptoss_local_checkpoint",
        }
        nodes = [
            n
            for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
            )
        ]
        self.assertEqual(len(nodes), 4)
        shared = next(
            n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "_original_shared_blob"
        )
        self.assertEqual(
            ast.dump(shared),
            ast.dump(ast.parse(inspect.getsource(numeric._original_shared_blob)).body[0]),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "srt/layers/quantization/mxfp4.py"
            target.parent.mkdir(parents=True)
            target.write_text(ast.unparse(ast.Module(body=nodes, type_ignores=[])))
            pins = {
                "sdk_files": {
                    "srt/layers/quantization/mxfp4.py": hashlib.sha256(
                        target.read_bytes()
                    ).hexdigest()
                }
            }
            self.assertTrue(callable(numeric.identity_validator(root, pins)))
            target.write_text("_GPTOSS_REVISION = 'changed'\n")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                numeric.identity_validator(root, pins)
            pins["sdk_files"]["srt/layers/quantization/mxfp4.py"] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "contract changed"):
                numeric.identity_validator(root, pins)

    def test_lifecycle_preserves_primary_and_cleanup_failures_in_real_report(self):
        code = actual_lifecycle_and_report(inspect.getsource(numeric))
        cases = [
            (False, (), False, None),
            (True, (), False, "NumericCriterionFailure"),
            (True, ("model", "environment"), False, "NumericCriterionFailure"),
            (False, ("model",), False, "RuntimeError"),
            (False, ("environment",), False, "RuntimeError"),
            (True, (), True, "NumericCriterionFailure"),
            (False, (), True, "RuntimeError"),
        ]
        original_tempdir = tempfile.TemporaryDirectory
        for primary, broken, broken_temp, expected in cases:
            with self.subTest(primary=primary, broken=broken, broken_temp=broken_temp):
                with original_tempdir() as directory:
                    output = Path(directory).resolve()
                    events = []
                    rendezvous = []

                    def initialize(**kwargs):
                        rendezvous.append(
                            Path(kwargs["distributed_init_method"].removeprefix("file://")).parent
                        )
                        self.assertTrue(rendezvous[0].is_dir())

                    def execute(*args):
                        if primary:
                            raise NumericCriterionFailure(
                                "Fixed predeclared numeric criterion failed"
                            )

                    def destroy(stage):
                        self.assertTrue(rendezvous[0].is_dir())
                        events.append(stage)
                        if stage in broken:
                            raise CleanupFailure(stage)

                    def make_tempdir(*args, **kwargs):
                        context = original_tempdir(*args, **kwargs)
                        real_cleanup = context.cleanup

                        def cleanup():
                            events.append("rendezvous")
                            real_cleanup()
                            if broken_temp:
                                raise CleanupFailure("rendezvous")

                        context.cleanup = cleanup
                        return context

                    record = {"status": "failed_or_incomplete"}
                    namespace = {
                        "distributed_numeric_scope": numeric.distributed_numeric_scope,
                        "record": record,
                        "args": SimpleNamespace(output=output),
                        "group_started": False,
                        "torch": SimpleNamespace(
                            distributed=SimpleNamespace(is_initialized=lambda: True)
                        ),
                        "tensors": None,
                        "json": json,
                        "Path": Path,
                        "init_distributed_environment": initialize,
                        "initialize_model_parallel": lambda **kwargs: None,
                        "execute": execute,
                        "destroy_model_parallel": lambda: destroy("model"),
                        "destroy_distributed_environment": lambda: destroy("environment"),
                    }
                    caught = None
                    with patch.object(numeric.tempfile, "TemporaryDirectory", make_tempdir):
                        try:
                            exec(code, namespace)
                        except Exception as error:
                            caught = error
                    report = json.loads((output / "result.json").read_text())
                    self.assertEqual(events, ["model", "environment", "rendezvous"])
                    self.assertFalse(rendezvous[0].exists())
                    if expected is None:
                        self.assertIsNone(caught)
                        self.assertEqual(
                            report["status"],
                            "original_packed_slice_actual_generic_triton_numeric_pass",
                        )
                        self.assertNotIn("failure", report)
                    else:
                        self.assertEqual(type(caught).__name__, expected)
                        self.assertEqual(report["failure"]["type"], expected)
                        self.assertEqual(report["status"], "failed_or_incomplete")
                    stages = [item["stage"] for item in report.get("teardown_failures", [])]
                    self.assertEqual(
                        stages,
                        [
                            *(["destroy_model_parallel"] if "model" in broken else []),
                            *(
                                ["destroy_distributed_environment"]
                                if "environment" in broken
                                else []
                            ),
                            *(["rendezvous"] if broken_temp else []),
                        ],
                    )

    def test_unknown_distributed_state_records_failure_and_attempts_cleanup(self):
        record, events = {}, []
        with tempfile.TemporaryDirectory() as directory:

            def unknown():
                raise CleanupFailure("state unavailable")

            with self.assertRaisesRegex(RuntimeError, "teardown failed"):
                with numeric.distributed_numeric_scope(
                    record,
                    Path(directory),
                    unknown,
                    [
                        ("first", lambda: events.append("first")),
                        ("second", lambda: events.append("second")),
                    ],
                ):
                    pass
        self.assertEqual(events, ["first", "second"])
        self.assertEqual(record["teardown_failures"][0]["stage"], "distributed_state")

    def test_tooling_selects_this_cpu_suite_once(self):
        spec = importlib.util.spec_from_file_location(
            "numeric_tooling_selector", SCRIPTS / "ci_run_tooling_tests.py"
        )
        selector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selector)
        self.assertEqual(
            selector.test_paths(ROOT).count("scripts/tests/test_ci_sglang_mxfp4_numeric.py"), 1
        )


if __name__ == "__main__":
    unittest.main()
