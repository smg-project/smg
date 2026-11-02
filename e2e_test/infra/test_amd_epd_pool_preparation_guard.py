"""Exercise the actual pool entrypoint without importing GPU infrastructure.

Only the real acquire method and enum definitions are compiled at test time.
Eviction, model lookup and port preparation are hostile or recording sentinels;
these controls do not start workers or repeat the command-builder suite.
"""

from __future__ import annotations

import ast
import logging
import textwrap
import threading
import unittest
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace


def _one(nodes, node_type, name):
    matches = [node for node in nodes if isinstance(node, node_type) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def _load_acquire():
    infra = Path(__file__).resolve().parent
    constants_source = (infra / "constants.py").read_text()
    constants = ast.parse(constants_source, filename=str(infra / "constants.py"))
    enum_source = [
        ast.get_source_segment(constants_source, _one(constants.body, ast.ClassDef, name))
        for name in ("ConnectionMode", "WorkerType")
    ]
    pool_source = (infra / "worker_pool.py").read_text()
    pool_tree = ast.parse(pool_source, filename=str(infra / "worker_pool.py"))
    pool_class = _one(pool_tree.body, ast.ClassDef, "WorkerPool")
    acquire = _one(pool_class.body, ast.FunctionDef, "acquire")
    method_source = ast.get_source_segment(pool_source, acquire)
    namespace = {
        "StrEnum": StrEnum,
        "DEFAULT_STARTUP_TIMEOUT": 1,
        "logger": logging.getLogger(__name__),
    }
    source = "from __future__ import annotations\n" + "\n\n".join(enum_source)
    source += "\n\n" + textwrap.dedent(method_source) + "\n"
    exec(compile(source, str(infra / "worker_pool.py"), "exec"), namespace)
    return namespace


def _load_start_workers():
    namespace = _load_acquire()
    path = Path(__file__).resolve().parent / "worker.py"
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    entrypoint = _one(tree.body, ast.FunctionDef, "start_workers")
    selected = ast.get_source_segment(source, entrypoint)
    exec(compile("from __future__ import annotations\n" + selected, str(path), "exec"), namespace)
    return namespace


class PoolPreparationGuardCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = _load_acquire()
        cls.mode = cls.scope["ConnectionMode"]
        cls.role = cls.scope["WorkerType"]

    def _pool(self, cached):
        return SimpleNamespace(
            _lock=threading.Lock(),
            _closed=False,
            _key=("old-cache",) if cached else None,
            _workers=[object()] if cached else [],
            _gateway=object() if cached else None,
            _gateway_key=("old-gateway",) if cached else None,
        )

    def _invalid_pairs(self):
        return [(engine, mode) for engine in ("sglang", "vllm") for mode in self.mode] + [
            ("trtllm", self.mode.GRPC),
            ("mlx", self.mode.GRPC),
            ("mock", self.mode.GRPC),
            ("tokenspeed", self.mode.HTTP),
            ("tokenspeed", self.mode.ZMQ),
        ]

    def _assert_invalid_untouched(self, cached):
        for engine, mode in self._invalid_pairs():
            with self.subTest(engine=engine, mode=mode, cached=cached):
                pool = self._pool(cached)
                before = (pool._key, pool._workers, tuple(pool._workers))
                gateway = (pool._gateway, pool._gateway_key)
                events = []

                def forbidden_evict():
                    events.append("eviction")
                    pool._key = None
                    pool._workers.clear()
                    raise AssertionError("unsupported encode reached cache mutation")

                def model_lookup():
                    events.append("model_lookup")

                def port_prepare():
                    events.append("port_prepare")
                    raise AssertionError("unsupported encode reached model/port preparation")

                def forbidden_start(**kwargs):
                    events.append("start_workers")
                    model_lookup()
                    port_prepare()

                pool._evict_locked = forbidden_evict
                self.scope["start_workers"] = forbidden_start
                with self.assertRaisesRegex(
                    ValueError, "^EPD encode workers currently require TokenSpeed over gRPC$"
                ):
                    self.scope["acquire"](
                        pool,
                        model_id="never-look-up-this-model",
                        engine=engine,
                        mode=mode,
                        worker_type=self.role.ENCODE,
                    )
                self.assertEqual(events, [])
                self.assertIs(pool._key, before[0])
                self.assertIs(pool._workers, before[1])
                self.assertEqual(tuple(pool._workers), before[2])
                self.assertIs(pool._gateway, gateway[0])
                self.assertIs(pool._gateway_key, gateway[1])

    def test_PG01_invalid_encode_preserves_cached_workers(self):
        self._assert_invalid_untouched(cached=True)

    def test_PG02_invalid_encode_precedes_model_and_port_preparation(self):
        self._assert_invalid_untouched(cached=False)

    def test_PG03_tokenspeed_grpc_encode_retains_uncached_route(self):
        pool = self._pool(cached=True)
        events = []
        workers = [object()]
        extra_args = ["--example"]
        extra_env = {"EXAMPLE": "1"}

        def evict():
            events.append("eviction")
            pool._key = None
            pool._workers = []

        def start(**kwargs):
            events.append(("start_workers", kwargs))
            return workers

        pool._evict_locked = evict
        self.scope["start_workers"] = start
        expected = dict(
            model_id="encode-model",
            engine="tokenspeed",
            mode=self.mode.GRPC,
            count=2,
            worker_type=self.role.ENCODE,
            timeout=7,
            log_dir="private-logs",
            gpu_offset=3,
            gpus=4,
            extra_engine_args=extra_args,
            wait_ready=False,
            tp=2,
            kv_backend="mooncake",
            extra_env=extra_env,
        )
        self.assertIs(self.scope["acquire"](pool, **expected), workers)
        self.assertEqual(events, ["eviction", ("start_workers", expected)])

    def test_PG04_other_roles_preserve_cache_and_uncached_routes(self):
        cases = [
            ("vllm", self.mode.HTTP, self.role.REGULAR),
            ("vllm", self.mode.GRPC, self.role.REGULAR),
            ("trtllm", self.mode.GRPC, self.role.REGULAR),
            ("mlx", self.mode.HTTP, self.role.REGULAR),
            ("mock", self.mode.HTTP, self.role.REGULAR),
            ("sglang", self.mode.ZMQ, self.role.REGULAR),
            ("sglang", self.mode.GRPC, self.role.PREFILL),
            ("vllm", self.mode.HTTP, self.role.DECODE),
            ("tokenspeed", self.mode.GRPC, self.role.PREFILL),
        ]
        for engine, mode, role in cases:
            with self.subTest(engine=engine, mode=mode, role=role):
                pool = self._pool(cached=True)
                events = []
                worker = SimpleNamespace(is_alive=lambda: True)
                pool._workers = [worker]
                pool._key = (engine, "control-model", mode, role, 1, None, None)
                workers = [object()]

                def evict():
                    events.append("eviction")
                    pool._key = None
                    pool._workers = []

                def start(**kwargs):
                    events.append(("start_workers", kwargs["worker_type"]))
                    return workers

                pool._evict_locked = evict
                self.scope["start_workers"] = start
                result = self.scope["acquire"](
                    pool, model_id="control-model", engine=engine, mode=mode, worker_type=role
                )
                if role == self.role.REGULAR and mode != self.mode.ZMQ:
                    self.assertEqual(result, [worker])
                    self.assertEqual(events, [])
                else:
                    self.assertIs(result, workers)
                    self.assertEqual(events, ["eviction", ("start_workers", role)])

    def test_PG05_direct_invalid_encode_precedes_all_worker_preparation(self):
        scope = _load_start_workers()
        mode, role = scope["ConnectionMode"], scope["WorkerType"]
        cases = [(engine, engine, mode(wire.value)) for engine, wire in self._invalid_pairs()] + [
            (None, "vllm", mode.GRPC),
            (None, "tokenspeed", mode.HTTP),
        ]
        for engine, resolved, wire in cases:
            with self.subTest(engine=engine, resolved=resolved, mode=wire):
                events = []
                scope["get_gpu_offset"] = lambda: 0

                def runtime():
                    events.append("resolve_engine")
                    return resolved

                def forbidden(stage):
                    def fail(*args, **kwargs):
                        events.append(stage)
                        raise AssertionError(f"invalid direct encode reached {stage}")

                    return fail

                scope["get_runtime"] = runtime
                scope["get_model_spec"] = forbidden("model_lookup")
                scope["worker_startup_timeout"] = forbidden("timeout_resolution")
                scope["get_engine_port"] = forbidden("port_reservation")
                scope["Worker"] = forbidden("Worker_construction")
                with self.assertRaisesRegex(
                    ValueError, "^EPD encode workers currently require TokenSpeed over gRPC$"
                ):
                    scope["start_workers"](
                        "never-look-up-this-model",
                        engine=engine,
                        mode=wire,
                        worker_type=role.ENCODE,
                        log_dir="private-logs",
                    )
                self.assertEqual(events, ["resolve_engine"] if engine is None else [])

    def test_PG06_direct_supported_requests_reach_original_model_lookup(self):
        scope = _load_start_workers()
        mode, role = scope["ConnectionMode"], scope["WorkerType"]
        cases = [
            ("tokenspeed", "tokenspeed", mode.GRPC, role.ENCODE),
            (None, "tokenspeed", mode.GRPC, role.ENCODE),
            ("sglang", "sglang", mode.GRPC, role.REGULAR),
            ("vllm", "vllm", mode.HTTP, role.REGULAR),
            ("vllm", "vllm", mode.ZMQ, role.REGULAR),
            ("sglang", "sglang", mode.HTTP, role.PREFILL),
            ("vllm", "vllm", mode.GRPC, role.DECODE),
            ("trtllm", "trtllm", mode.GRPC, role.REGULAR),
            ("mlx", "mlx", mode.HTTP, role.REGULAR),
            ("mock", "mock", mode.HTTP, role.REGULAR),
        ]
        for engine, resolved, wire, worker_role in cases:
            with self.subTest(engine=engine, resolved=resolved, mode=wire, role=worker_role):
                events = []
                scope["get_gpu_offset"] = lambda: 0

                def runtime():
                    events.append("resolve_engine")
                    return resolved

                def model_lookup(model_id):
                    events.append(("model_lookup", model_id))
                    raise LookupError("stop at original model lookup sentinel")

                scope["get_runtime"] = runtime
                scope["get_model_spec"] = model_lookup
                with self.assertRaisesRegex(
                    LookupError, "^stop at original model lookup sentinel$"
                ):
                    scope["start_workers"](
                        "supported-control-model",
                        engine=engine,
                        mode=wire,
                        worker_type=worker_role,
                        log_dir="private-logs",
                    )
                expected = ["resolve_engine"] if engine is None else []
                self.assertEqual(events, expected + [("model_lookup", "supported-control-model")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
