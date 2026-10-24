"""Unexecuted CPU controls: real files/logger and exact AST; no engine or device."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "aiter", "sglang", "vllm", "mooncake", "smg"}:
            raise ImportError("CPU startup controls prohibit engine import")


sys.meta_path.insert(0, NoEngine())
sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError(event))
                 if event in {"subprocess.Popen", "socket.connect", "socket.bind", "socket.getaddrinfo"} else None)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trace = load("startup_CPU_fixture", HERE / "startup_trace.py")
applier = load("startup_patch_CPU_fixture", HERE / "startup_patch.py")
CONTRACT = json.loads((HERE / "startup-overlay.json").read_text())


class StartupControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, SMG_SG2_STARTUP_DIAGNOSTIC="1", CU_NUM="0",
                              AMD_DEPENDENCY_DIRECTORY=str(self.root), E2E_JOB_MARKER="e2e-123-1-amd-mi300x-2")
        self.env.start()
        self.addCleanup(self.env.stop)
        (self.root / "sglang-startup").mkdir()
        trace._pid, trace._memory_count, trace._cu_count, trace._dropped = os.getpid(), 0, 0, False

    def rows(self):
        return [json.loads(line) for f in (self.root / "sglang-startup").glob("*.jsonl") for line in f.read_text().splitlines()]

    def function(self, patched, result):
        item = next(v for k, v in CONTRACT["files"].items() if k.endswith("chip_info.py"))
        text = Path(item["actual_capture_path"]).read_text()
        if patched:
            for change in item["changes"]:
                text = text.replace(change["before"], change["after"])
        fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "get_cu_num_custom_op")
        fn.decorator_list = []
        calls = []
        def run(*args, **kwargs):
            calls.append((args, kwargs))
            if isinstance(result, Exception):
                raise result
            return result
        scope = {"os": os, "re": re, "subprocess": types.SimpleNamespace(run=run),
                 "executable_path": lambda name: __file__, "_smg_sg2_cu": trace.cu,
                 "__name__": __name__, "__file__": __file__}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "actual_AST_only", "exec"), scope)
        return scope[fn.name], calls

    def test_default_off_has_no_file_or_capability_query(self):
        os.environ["SMG_SG2_STARTUP_DIAGNOSTIC"] = "0"
        trace.memory("off", __name__, __file__, {}, {})
        trace.cu(__name__, __file__, {})
        self.assertFalse(list((self.root / "sglang-startup").iterdir()))

    def test_actual_CU_old_new_single_call_and_original_assertion(self):
        for values, rc in [([304] * 8, 0), ([304], 17), ([], 0), ([304, 80], 0)]:
            for amended in [False, True]:
                with self.subTest(values=values, rc=rc, patched=amended):
                    result = types.SimpleNamespace(returncode=rc, stdout="".join(f"Agent {i}\nDevice Type: GPU\nCompute Unit: {v}\n" for i, v in enumerate(values)), stderr="")
                    fn, calls = self.function(amended, result)
                    if len(set(values)) == 1:
                        self.assertEqual(fn(), 304)
                    else:
                        with self.assertRaises(AssertionError):
                            fn()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0][1], {"capture_output": True, "text": True, "check": False})

    def test_subprocess_failure_retains_original_error(self):
        fn, calls = self.function(True, OSError("fixture failure"))
        with self.assertRaisesRegex(RuntimeError, "Get GPU Compute Unit from rocminfo failed fixture failure"):
            fn()
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.rows()[0]["exception_type"], "OSError")

    def test_real_warning_gate_memory_cap_and_raw_bounds(self):
        cuda = types.SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 0,
                                     mem_get_info=lambda index: (4, 8), memory_allocated=lambda index: 1,
                                     memory_reserved=lambda index: 2)
        with self.assertLogs("smg.sg2.startup", logging.WARNING) as output:
            for i in range(14):
                trace.memory(str(i), __name__, __file__, {"mem_fraction_static": .9}, {"torch": types.SimpleNamespace(cuda=cuda)})
            trace.cu(__name__, __file__, {"result": types.SimpleNamespace(returncode=17, stdout="a" * 200000, stderr="b" * 200000), "gpu_compute_units": []})
        self.assertEqual(len(output.records), 13)
        self.assertEqual(len(self.rows()), 13)
        self.assertTrue(all(len(line.encode()) <= 4096 for f in (self.root / "sglang-startup").glob("*.jsonl") for line in f.read_text().splitlines()))
        self.assertEqual([f.stat().st_size for f in (self.root / "sglang-startup").glob("*.txt")], [131072, 131072])

    def test_logging_failure_does_not_replace_original_return_or_assertion(self):
        with patch.object(trace, "_emit", side_effect=OSError("fixture evidence failure")):
            fn, _ = self.function(True, types.SimpleNamespace(returncode=0, stdout="Agent 0\nDevice Type: GPU\nCompute Unit: 304\n", stderr=""))
            self.assertEqual(fn(), 304)
            fn, _ = self.function(True, types.SimpleNamespace(returncode=0, stdout="", stderr=""))
            with self.assertRaises(AssertionError):
                fn()

    def fixture_applier(self):
        contract = json.loads(json.dumps(CONTRACT))
        contract["files"] = {str(self.root / f"source-{i}.py"): item for i, item in enumerate(contract["files"].values())}
        for path, item in contract["files"].items():
            Path(path).write_bytes(Path(item["actual_capture_path"]).read_bytes())
        contract["helper"]["path"] = str(self.root / "installed-helper.py")
        (self.root / "startup-overlay.json").write_text(json.dumps(contract))
        (self.root / "startup_trace.py").write_bytes((HERE / "startup_trace.py").read_bytes())
        for name, value in {"HERE": self.root, "TARGETS": set(contract["files"]), "HELPER": Path(contract["helper"]["path"])}.items():
            guard = patch.object(applier, name, value)
            guard.start()
            self.addCleanup(guard.stop)
        return contract

    def test_actual_source_file_apply_idempotence_and_tamper_red(self):
        contract = self.fixture_applier()
        applier.apply(self.root)
        applier.apply(self.root)
        for path, item in contract["files"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), item["after_sha256"])
        Path(next(iter(contract["files"]))).write_text("unknown")
        with self.assertRaises(RuntimeError):
            applier.apply(self.root)

    def test_artifact_manifest_binds_output_and_rejects_symlink(self):
        self.fixture_applier()
        applier.apply(self.root)
        trace.memory("artifact", __name__, __file__, {}, {})
        applier.verify(self.root)
        self.assertTrue(json.loads((self.root / "sglang-startup-overlay-final.json").read_text())["artifact_files"])
        (self.root / "sglang-startup/1-1.jsonl").symlink_to(HERE / "startup_trace.py")
        with self.assertRaises(RuntimeError):
            applier.verify(self.root)

    def test_original_three_source_AST_is_preserved(self):
        class Strip(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "smg_sg2_startup_trace" else node
            def visit_Expr(self, node):
                return None if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id.startswith("_smg_sg2_") else node
        for item in CONTRACT["files"].values():
            before = Path(item["actual_capture_path"]).read_text()
            after = before
            for change in item["changes"]:
                self.assertEqual(after.count(change["before"]), 1)
                after = after.replace(change["before"], change["after"])
            self.assertEqual(hashlib.sha256(after.encode()).hexdigest(), item["after_sha256"])
            self.assertEqual(ast.dump(Strip().visit(ast.parse(before))), ast.dump(Strip().visit(ast.parse(after))))


if __name__ == "__main__":
    unittest.main()
