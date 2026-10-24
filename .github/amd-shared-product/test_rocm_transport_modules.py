"""CPU-only checks of the real transport-discovery function, extracted by AST."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


class TransportModules(unittest.TestCase):
    def discover(self, available, *packages):
        path = Path(os.environ.get("SMG_TRANSPORT_SOURCE", ""))
        if not os.environ.get("SMG_TRANSPORT_SOURCE"):
            path = Path(__file__).resolve().parents[2] / "e2e_test/router/test_pd_topologies.py"
        module = ast.parse(path.read_text())
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_vllm_transport_installed"
        )
        queries = []

        def find_spec(name):
            queries.append(name)
            return object() if name in available else None

        namespace = {"importlib": SimpleNamespace(util=SimpleNamespace(find_spec=find_spec))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[function.name](*packages), queries

    def test_original_nixl_and_mooncake(self):
        self.assertTrue(self.discover({"nixl", "mooncake"}, "nixl", "mooncake")[0])

    def test_rocm_nixl_and_mooncake(self):
        found, queries = self.discover({"nixl_rocm", "mooncake"}, "nixl", "mooncake")
        self.assertTrue(found)
        self.assertEqual(queries, ["nixl", "nixl_rocm", "mooncake"])

    def test_mooncake_still_required(self):
        self.assertFalse(self.discover({"nixl", "nixl_rocm"}, "nixl", "mooncake")[0])

    def test_nixl_still_required(self):
        self.assertFalse(self.discover({"mooncake"}, "nixl", "mooncake")[0])

    def test_no_transport_modules(self):
        self.assertFalse(self.discover(set(), "nixl", "mooncake")[0])

    def test_unrelated_backend_keeps_its_exact_module(self):
        self.assertEqual(self.discover({"ucx"}, "ucx"), (True, ["ucx"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
