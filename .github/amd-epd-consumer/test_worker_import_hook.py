"""Exercise the actual diagnostic hook; no engine or RPC implementation is loaded."""

import builtins
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

SOURCE = Path(
    os.environ.get("WORKER_PROOF_UNDER_TEST", Path(__file__).with_name("worker_proof.py"))
)
SPEC = importlib.util.spec_from_file_location("worker_proof_under_test", SOURCE)
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


class WorkerImportHookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original_import = builtins.__import__
        self.original_path = sys.path[:]
        sys.path.insert(0, str(self.root))
        importlib.invalidate_caches()

    def tearDown(self):
        builtins.__import__ = self.original_import
        sys.path[:] = self.original_path
        for name in list(sys.modules):
            if name == "smg_grpc_proto" or name.startswith("smg_grpc_proto."):
                del sys.modules[name]
        self.temporary.cleanup()

    def observe(self, expected):
        PROOF.observe(self.root, expected, "encode", "Qwen/Qwen3.5-9B", ["-m", "fixture"])

    def fixture(self):
        package = self.root / "smg_grpc_proto"
        package.mkdir()
        sources = {package / "__init__.py": "", package / "child.py": "value = 42\n"}
        for path, text in sources.items():
            path.write_text(text)
        return {
            str(path): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources
        }

    def test_dynamic_exec_accepts_none_package(self):
        self.observe({})
        module = types.ModuleType("inductor_generated_fixture")
        self.assertIsNone(module.__package__)
        exec("from ctypes import c_void_p\nvalue = c_void_p(42).value", module.__dict__)
        self.assertEqual(module.value, 42)
        records = self.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["kind"], "process")
        self.assertEqual(records[0]["job_marker"], os.environ["E2E_JOB_MARKER"])

    def test_normal_prefix_and_relative_import_keep_exact_origin_gate(self):
        expected = self.fixture()
        self.observe(expected)
        package = __import__("smg_grpc_proto")
        namespace = {"__package__": package.__name__}
        exec("from .child import value", namespace)
        self.assertEqual(namespace["value"], 42)
        imports = {r["module"]: r for r in self.records() if r["kind"] == "import"}
        self.assertEqual(set(imports), {"smg_grpc_proto", "smg_grpc_proto.child"})
        for record in imports.values():
            self.assertEqual(record["sha256"], expected[record["path"]]["sha256"])
            self.assertEqual(Path(record["path"]).parent, self.root / "smg_grpc_proto")

    def test_wrong_loaded_origin_is_rejected(self):
        self.fixture()
        self.observe({})
        with self.assertRaisesRegex(ValueError, "actual worker imported unexpected source"):
            __import__("smg_grpc_proto")
        self.assertFalse(any(r["kind"] == "import" for r in self.records()))

    def records(self):
        return [
            json.loads(line)
            for line in (self.root / "workers" / f"{os.getpid()}.jsonl").read_text().splitlines()
        ]


if __name__ == "__main__":
    unittest.main(verbosity=2)
