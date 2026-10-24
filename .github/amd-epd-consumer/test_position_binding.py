"""Unexecuted CPU proposal: real proc/Popen, saved source files, no engine imports."""

import builtins
import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import position_analysis_v2 as analysis
import worker_proof
from test_position_diagnostic import Fixture


class ChildBindingControls(Fixture):
    def evidence(self, row):
        return {"path": "CPU-STDERR-FIXTURE", "line": 1,
                "JSON_bytes": len(json.dumps(row, separators=(",", ":")).encode()) + 1}

    def bundle(self):
        helper = self.helper()
        self.emit(helper, self.request())
        identity, event = self.records(include_identity=True)
        anchor = identity["launcher"]
        processes = {anchor["pid"]: [{"kind": "process", "pid": anchor["pid"],
            "launcher_pid": anchor["pid"], "start_ticks": anchor["start_ticks"],
            "position_anchor": anchor, "role": "prefill", "model_id": self.manifest["model_id"],
            "job_marker": anchor["job_marker"], "original_launcher": ["-m", "sglang.launch_server"]}]}
        overlays = {anchor["pid"]: json.loads((self.out / f"position-overlay-{anchor['pid']}.json").read_text())}
        manifest_sha = hashlib.sha256((self.assets / "position_manifest.json").read_bytes()).hexdigest()
        return helper, identity, event, processes, overlays, manifest_sha

    def test_first_identity_once_actual_proc_and_actual_loaded_module(self):
        helper, identity, event, processes, overlays, manifest_sha = self.bundle()
        self.assertEqual(identity["pid"], os.getpid())
        self.assertEqual(identity["start_ticks"], helper.process(os.getpid())["start_ticks"])
        self.assertEqual(identity["parent_pid"], os.getppid())
        self.assertEqual(identity["helper"]["path"], str(self.root / self.manifest["helper"]["path"]))
        self.emit(helper, self.request(), "receiver_after")
        records = self.records(include_identity=True)
        self.assertEqual(sum(r["kind"] == "smg_epd_position_identity_v2" for r in records), 1)
        result = analysis.analyze([(r, self.evidence(r)) for r in records], processes, overlays,
                                  self.manifest, manifest_sha, identity["job_marker"])
        self.assertEqual(len(result["validated_events"]), 2)
        self.assertEqual(result["rejected_records"], [])
        self.assertEqual(event["identity_sha256"], analysis.digest(identity))

    def test_real_child_to_actual_observer_launcher_without_launcher_import_substitution(self):
        self.helper()
        original_import = builtins.__import__
        try:
            with patch.object(worker_proof, "expected_engine", return_value={}):
                worker_proof.observe(self.out, {}, "prefill", self.manifest["model_id"],
                                     ["-m", "sglang.launch_server"])
        finally:
            builtins.__import__ = original_import
        helper_path = self.root / self.manifest["helper"]["path"]
        row = self.manifest["modules"][0]
        program = """
import importlib.util,json,sys,types
cfg=json.loads(sys.argv[1])
spec=importlib.util.spec_from_file_location(cfg['helper_name'],cfg['helper_path'])
helper=importlib.util.module_from_spec(spec)
sys.modules[cfg['helper_name']]=helper
spec.loader.exec_module(helper)
emitter=types.ModuleType(cfg['module']);emitter.__file__=cfg['source']
sys.modules[cfg['module']]=emitter
req=types.SimpleNamespace(rid='child-cpu',bootstrap_room=42,input_ids=[1,2])
helper.emit('receiver_before','prefill',req,cfg['module'],cfg['source'])
"""
        config = {"helper_name": self.manifest["helper"]["module"], "helper_path": str(helper_path),
                  "module": row["module"], "source": str(self.root / row["path"])}
        completed = subprocess.run([sys.executable, "-c", program, json.dumps(config)],
                                   cwd=self.temp.name, env=os.environ.copy(), capture_output=True,
                                   text=True, timeout=5, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        identity, event = [json.loads(line) for line in completed.stderr.splitlines()]
        self.assertNotEqual(identity["pid"], os.getpid())
        self.assertEqual(identity["ancestry"][0]["parent_pid"], os.getpid())
        self.assertEqual(identity["ancestry"][-1]["pid"], os.getpid())
        processes = {os.getpid(): [json.loads(line) for line in
                                  (self.out / "workers" / f"{os.getpid()}.jsonl").read_text().splitlines()
                                  if json.loads(line)["kind"] == "process"]}
        overlay = json.loads((self.out / f"position-overlay-{os.getpid()}.json").read_text())
        manifest_sha = hashlib.sha256((self.assets / "position_manifest.json").read_bytes()).hexdigest()
        result = analysis.analyze([(r, self.evidence(r)) for r in (identity, event)], processes,
                                  {os.getpid(): overlay}, self.manifest, manifest_sha,
                                  identity["job_marker"])
        self.assertEqual(len(result["validated_events"]), 1)
        self.assertEqual(result["rejected_records"], [])

    def test_missing_ancestry_and_reused_launcher_pid_emit_only_bounded_drop(self):
        for field, value in (("pid", 2147483647), ("start_ticks", "0")):
            with self.subTest(field=field):
                helper = self.helper()
                anchor = json.loads(os.environ["SMG_EPD_POSITION_ANCHOR"])
                anchor[field] = value
                with patch.dict(os.environ, SMG_EPD_POSITION_ANCHOR=json.dumps(anchor)):
                    self.emit(helper, self.request())
                    self.emit(helper, self.request("again"))
                self.assertEqual([r["kind"] for r in self.records(True)], ["smg_epd_position_drop_v2"])
                self.assertLessEqual(len(self.stream.getvalue().encode()), 512)

    def test_missing_loaded_module_or_wrong_hash_cannot_emit_attributed_event(self):
        for mode in ("missing-emitter", "wrong-emitter-path", "tampered-helper"):
            with self.subTest(mode=mode):
                helper = self.helper()
                row = self.manifest["modules"][0]
                if mode == "missing-emitter":
                    sys.modules.pop(row["module"])
                elif mode == "wrong-emitter-path":
                    sys.modules[row["module"]].__file__ = str(self.root / "wrong.py")
                else:
                    path = self.root / self.manifest["helper"]["path"]
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# wrong hash\n")
                    self.addCleanup(path.write_bytes, original)
                self.emit(helper, self.request())
                self.assertEqual([r["kind"] for r in self.records(True)], ["smg_epd_position_drop_v2"])

    def test_offline_identity_refuses_missing_parent_reuse_source_and_hash(self):
        _, identity, _, processes, overlays, manifest_sha = self.bundle()
        mutations = [lambda r: r["ancestry"].clear(),
                     lambda r: r["launcher"].update(start_ticks="0"),
                     lambda r: r["helper"].update(path="/wrong.py"),
                     lambda r: r["helper"].update(sha256="0" * 64),
                     lambda r: r["first_origin"].update(sha256="0" * 64),
                     lambda r: r.update(job_marker="foreign-job")]
        for mutate in mutations:
            changed = copy.deepcopy(identity)
            mutate(changed)
            with self.subTest(identity=changed), self.assertRaises((ValueError, KeyError, TypeError)):
                analysis.validate_identity(changed, processes, overlays, self.manifest,
                                           manifest_sha, identity["job_marker"])

    def test_offline_event_requires_identity_generation_and_exact_module(self):
        _, identity, event, processes, overlays, manifest_sha = self.bundle()
        mutations = [lambda r: r.update(start_ticks="0"),
                     lambda r: r.update(identity_sha256="0" * 64),
                     lambda r: r["origin"].update(path="/wrong.py"),
                     lambda r: r["origin"].update(sha256="0" * 64),
                     lambda r: r.update(kind="smg_epd_position_v1")]
        for mutate in mutations:
            changed = copy.deepcopy(event)
            mutate(changed)
            records = [(r, self.evidence(r)) for r in (identity, changed)]
            result = analysis.analyze(records, processes, overlays, self.manifest,
                                      manifest_sha, identity["job_marker"])
            self.assertEqual(result["validated_events"], [])
            self.assertTrue(result["rejected_records"])
        result = analysis.analyze([(event, self.evidence(event))], processes, overlays,
                                  self.manifest, manifest_sha, identity["job_marker"])
        self.assertEqual(result["validated_events"], [])
        self.assertTrue(result["rejected_records"])

    def test_offline_multiple_identities_and_excess_bytes_fail_closed(self):
        _, identity, event, processes, overlays, manifest_sha = self.bundle()
        duplicate = copy.deepcopy(identity)
        duplicate["start_ticks"] = "0"
        records = [(r, self.evidence(r)) for r in (identity, duplicate, event)]
        result = analysis.analyze(records, processes, overlays, self.manifest,
                                  manifest_sha, identity["job_marker"])
        self.assertEqual(result["validated_events"], [])
        records = [(identity, self.evidence(identity)), (event, {"JSON_bytes": 256 * 1024})]
        result = analysis.analyze(records, processes, overlays, self.manifest,
                                  manifest_sha, identity["job_marker"])
        self.assertEqual(result["validated_events"], [])


if __name__ == "__main__":
    unittest.main()
