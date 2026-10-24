"""Prepared CPU controls only. Requires explicit saved actual-source mapping; no engine import."""

import ast
import hashlib
import importlib.util
import io
import json
import logging
import os
import shutil
import tempfile
import types
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

import position_overlay as overlay

HERE = Path(__file__).resolve().parent


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.root, self.assets, self.out = root / "sglang", root / "assets", root / "proof"
        self.assets.mkdir()
        self.out.mkdir()
        self.manifest = json.loads((HERE / "position_manifest.json").read_text())
        actual = json.loads(Path(os.environ["EPD_POSITION_ACTUAL_SOURCES"]).read_text())
        for row in self.manifest["modules"]:
            source = Path(actual[row["module"]])
            raw = source.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["original_sha256"])
            target = self.root / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        for name in ("position_manifest.json", "position_trace.py"):
            shutil.copyfile(HERE / name, self.assets / name)
        shutil.copytree(HERE / "position_patches", self.assets / "position_patches")
        self.manifest["root"] = str(self.root)
        (self.assets / "position_manifest.json").write_text(json.dumps(self.manifest))
        self.addCleanup(patch.stopall)
        patch.object(overlay, "HERE", self.assets).start()
        patch.object(overlay, "ROOT", self.root).start()
        patch.dict(os.environ, SMG_EPD_POSITION_DIAGNOSTIC="1",
                   SMG_EPD_POSITION_MODEL="Qwen/Qwen3.5-9B").start()

    def helper(self):
        overlay.apply(self.out)
        path = self.root / self.manifest["helper"]["path"]
        spec = importlib.util.spec_from_file_location("private_position_control", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.stream = io.StringIO()
        logger = module.LOGGER
        old = logger.handlers[:], logger.propagate, logger.level
        self.addCleanup(lambda: (setattr(logger, "handlers", old[0]),
                                setattr(logger, "propagate", old[1]), logger.setLevel(old[2])))
        logger.handlers = [logging.StreamHandler(self.stream)]
        logger.propagate = False
        return module

    def request(self, rid="cpu-final"):
        return types.SimpleNamespace(rid=rid, bootstrap_room=42, input_ids=array("q", [1, 2]),
                                     origin_input_ids=array("q", [1, 2]),
                                     origin_input_ids_unpadded=array("q", [1, 2]), mm_inputs=None)

    def emit(self, helper, req, stage="receiver_before"):
        row = self.manifest["modules"][0]
        helper.emit(stage, "prefill", req, row["module"], str(self.root / row["path"]))

    def records(self):
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]


class OverlayControls(Fixture):
    def test_actual_sources_old_new_tamper_and_origin(self):
        overlay.apply(self.out)
        overlay.apply(self.out)
        for row in self.manifest["modules"]:
            self.assertEqual(hashlib.sha256((self.root / row["path"]).read_bytes()).hexdigest(),
                             row["patched_sha256"])
        expected = overlay.expected_engine()
        self.assertEqual(len(expected), 4)
        helper = self.helper()
        self.emit(helper, self.request())
        module = self.manifest["modules"][0]["module"]
        self.assertEqual(self.records()[0]["origin"], expected[module] | {"module": module})
        bad = self.root / self.manifest["modules"][1]["path"]
        bad.write_bytes(bad.read_bytes() + b"\n# tamper\n")
        with self.assertRaisesRegex(ValueError, "overlay source identity"):
            overlay.apply(self.out)
        receiver = self.root / self.manifest["modules"][0]["path"]
        receiver.write_bytes(receiver.read_bytes() + b"\n# tamper\n")
        helper.ORIGINS.clear()
        self.emit(helper, self.request("bad-origin"))
        self.assertEqual(self.records()[-1]["kind"], "smg_epd_position_drop")


class LoggerControls(Fixture):
    def test_warning_env_off_cpu_scalar_and_gpu_never_read(self):
        helper = self.helper()
        req = self.request()
        req.input_text = "SECRET-PROMPT"
        with patch.dict(os.environ, SMG_EPD_POSITION_DIAGNOSTIC="0"):
            self.emit(helper, req)
        self.assertEqual(self.records(), [])

        class CPU:
            device, shape, dtype = types.SimpleNamespace(type="cpu"), (1,), "int64"
            def item(self):
                return 7

        class GPU:
            device, shape, dtype = types.SimpleNamespace(type="cuda"), (1,), "int64"
            def item(self):
                raise AssertionError("GPU item is forbidden")
            cpu = numpy = tolist = item

        req.mm_inputs = types.SimpleNamespace(mrope_positions=GPU(), mrope_position_delta=CPU())
        self.emit(helper, req)
        first = self.records()[0]
        self.assertEqual(first["mrope_delta"]["scalar"], 7)
        self.assertEqual(first["input_ids"]["status"], "CPU_FULL")
        req.mm_inputs.mrope_position_delta = GPU()
        self.emit(helper, req, "receiver_after")
        self.assertEqual(self.records()[1]["mrope_delta"]["scalar_status"],
                         "NOT_READ_GPU_OR_NONSCALAR")
        self.assertNotIn("SECRET-PROMPT", self.stream.getvalue())
        self.assertNotIn("scalar", self.records()[1]["mrope_delta"])


class BoundControls(Fixture):
    def test_poll_dedup_request_id_and_byte_caps(self):
        helper = self.helper()
        for index in range(12):
            req = self.request(str(index))
            for _ in range(20):
                self.emit(helper, req)
        records = self.records()
        self.assertEqual(sum(row["kind"] == "smg_epd_position_v1" for row in records), 8)
        self.assertEqual(sum(row["kind"] == "smg_epd_position_drop" for row in records), 1)
        self.assertTrue(all(len(line.encode()) <= 4096 for line in self.stream.getvalue().splitlines()))
        self.assertLessEqual(len(self.stream.getvalue().encode()), 256 * 1024)
        self.assertEqual(helper.ids([1] * 16385)["status"], "OVER_CAP_NOT_HASHED")
        helper.REQUESTS.clear()
        helper.USED, helper.DROPPED = 256 * 1024 - 511, False
        self.stream.seek(0)
        self.stream.truncate()
        self.emit(helper, self.request("byte-cap"))
        self.assertEqual([r["kind"] for r in self.records()], ["smg_epd_position_drop"])


class ReceiverFlowControls(Fixture):
    def test_actual_receiver_rebuild_and_original_exception_survive_logger_failure(self):
        helper = self.helper()
        row = self.manifest["modules"][0]
        path = self.root / row["path"]
        source = ast.parse(path.read_text())
        method = next(node for node in ast.walk(source)
                      if isinstance(node, ast.FunctionDef) and node.name == "_finish_assemble")
        namespace = {"__name__": row["module"], "__file__": str(path),
                     "_smg_epd_trace": helper.emit, "array": array,
                     "_select_mm_processor_prompt": lambda req, processor: req.input_ids,
                     "WaitingMMRequestStatus": types.SimpleNamespace(SUCCESS=1)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        mm = types.SimpleNamespace(input_ids=[1, 8, 8, 2], mrope_positions=None,
                                   mrope_position_delta=7)
        processor = types.SimpleNamespace(get_validated_mm_data=lambda *a, **k: mm)
        req = self.request()
        owner = types.SimpleNamespace(mm_processor=processor, recv_req=req,
                                      recv_embedding_data=types.SimpleNamespace(get_mm_extra_meta=lambda: {}),
                                      _bind_pool_slot_to_mm_inputs=lambda value: None)
        namespace["_finish_assemble"](owner, None)
        self.assertEqual(list(req.input_ids), [1, 8, 8, 2])
        before, after = self.records()
        self.assertEqual((before["input_ids"]["length"], after["input_ids"]["length"]), (2, 4))
        self.assertEqual(after["mrope_delta"]["scalar"], 7)

        class BrokenHandler(logging.Handler):
            def emit(self, record):
                raise RuntimeError("logging unavailable")

        def failed_processor(*args, **kwargs):
            raise ValueError("ORIGINAL-COMPUTE-ERROR")

        helper.LOGGER.handlers = [BrokenHandler()]
        processor.get_validated_mm_data = failed_processor
        owner.recv_req = self.request("original-error")
        with self.assertRaisesRegex(ValueError, "ORIGINAL-COMPUTE-ERROR"):
            namespace["_finish_assemble"](owner, None)


if __name__ == "__main__":
    unittest.main()
