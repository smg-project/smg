"""Fault controls for the compatibility check's orchestration, without RPCs.

The full checker module is loaded. Subprocesses and readiness are replaced by
owned fakes; generated metadata is inert private text. These controls cover
error propagation, cleanup and guard selection, not the existing 44 real RPCs.
Run this file normally and with python -O; unittest assertions remain active.
"""

from __future__ import annotations

import importlib.util
import io
import json
import signal
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "check_grpc_compatibility.py"


class FakeProcess:
    def __init__(self, runtime, returncode=0):
        self.stdout = mock.Mock()
        self.stdout.readline.return_value = json.dumps({"grpc": runtime, "port": 23001})
        self.terminate = mock.Mock()
        self.wait = mock.Mock(return_value=returncode)
        self.kill = mock.Mock()


class CompatibilityOrchestrationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("compatibility_under_test", SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        self.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.checker)
        self.assertEqual(Path(self.checker.__file__).resolve(), SCRIPT)
        self.temp = tempfile.TemporaryDirectory(prefix="grpc-orchestration-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.peers = [("old-python", str(root / "old")), ("new-python", str(root / "new"))]
        self.guard = root / "new/smg_grpc_proto/generated/common_pb2_grpc.py"
        self.guard.parent.mkdir(parents=True)
        self.write_guard("1.81.1")
        self.fingerprints = [{"common.proto": "same"}, {"common.proto": "same"}]
        self.processes = [FakeProcess("1.78.0"), FakeProcess("1.81.1")]
        self.refused = SimpleNamespace(
            returncode=1, stderr="RuntimeError: generated code requires grpcio>=1.81.1."
        )
        self.client_error = None
        self.client_calls = 0
        self.fingerprint_calls = 0
        self.spawn_calls = 0
        self.readiness = True
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def write_guard(self, minimum):
        self.guard.write_text(
            f"GRPC_GENERATED_VERSION = {minimum!r}\n"
            "raise RuntimeError('this metadata must never be imported by its reader')\n"
        )

    def check_output(self, args, **kwargs):
        if args[2] == "fingerprint":
            value = self.fingerprints[self.fingerprint_calls]
            self.fingerprint_calls += 1
            return json.dumps(value)
        self.assertEqual(args[2], "client")
        self.client_calls += 1
        if self.client_error is not None:
            raise self.client_error
        return json.dumps({"grpc": "fake-client", "checks": ["orchestration-fixture"]})

    def popen(self, args, **kwargs):
        self.assertEqual(args[2], "server")
        process = self.processes[self.spawn_calls]
        self.spawn_calls += 1
        return process

    def select(self, readers, writers, errors, timeout):
        self.assertEqual(timeout, 10)
        return (readers if self.readiness else [], [], [])

    def run_check(self):
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(self.stdout))
            stack.enter_context(redirect_stderr(self.stderr))
            self.output_mock = stack.enter_context(
                mock.patch.object(self.checker.subprocess, "check_output", self.check_output)
            )
            self.popen_mock = stack.enter_context(
                mock.patch.object(self.checker.subprocess, "Popen", self.popen)
            )
            self.select_mock = stack.enter_context(
                mock.patch.object(self.checker.select, "select", side_effect=self.select)
            )
            self.refusal_mock = stack.enter_context(
                mock.patch.object(self.checker.subprocess, "run", return_value=self.refused)
            )
            self.checker.check(self.peers)

    def test_normal_shutdown_checks_both_servers_and_preserves_four_pairs(self):
        self.run_check()
        output = json.loads(self.stdout.getvalue())
        self.assertEqual(len(output["pairs"]), 4)
        self.assertTrue(output["old_gencode_rejected"])
        self.assertEqual(self.client_calls, 4)
        for process in self.processes:
            process.terminate.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=5)
            process.kill.assert_not_called()
        self.assertEqual(self.refusal_mock.call_args.kwargs["env"]["PYTHONPATH"], self.peers[1][1])

    def test_generated_minimum_does_not_follow_independently_upgraded_runtime(self):
        self.processes[1] = FakeProcess("1.90.0")
        self.run_check()
        self.assertTrue(json.loads(self.stdout.getvalue())["old_gencode_rejected"])

    def test_new_generated_minimum_uses_its_own_guard(self):
        self.write_guard("1.82.0")
        self.refused.stderr = "RuntimeError: generated code requires grpcio>=1.82.0."
        self.run_check()

    def test_other_rejection_diagnostic_is_not_compatibility_acceptance(self):
        self.refused.stderr = "ModuleNotFoundError: unrelated dependency"
        with self.assertRaisesRegex(AssertionError, "unrelated dependency"):
            self.run_check()

    def test_minimum_version_prefix_is_not_a_matching_guard(self):
        self.refused.stderr = "RuntimeError: generated code requires grpcio>=1.81.10."
        with self.assertRaisesRegex(AssertionError, "1.81.10"):
            self.run_check()

    def test_successful_import_is_rejected_even_with_guard_text_on_stderr(self):
        self.refused.returncode = 0
        with self.assertRaisesRegex(AssertionError, "exit status 0"):
            self.run_check()

    def test_empty_rejection_stderr_is_not_a_guard(self):
        self.refused.stderr = ""
        with self.assertRaisesRegex(AssertionError, "Expected incompatible-gencode refusal"):
            self.run_check()

    def test_missing_generated_minimum_does_not_fall_back_to_runtime(self):
        self.guard.write_text("GRPC_VERSION = '1.81.1'\n")
        with self.assertRaisesRegex(RuntimeError, "Expected one generated gRPC minimum"):
            self.run_check()
        self.refusal_mock.assert_not_called()

    def test_invalid_generated_minimum_does_not_fall_back_to_runtime(self):
        self.guard.write_text("GRPC_GENERATED_VERSION = 181\n")
        with self.assertRaisesRegex(RuntimeError, "Expected one generated gRPC minimum"):
            self.run_check()
        self.refusal_mock.assert_not_called()

    def test_duplicate_generated_minimum_is_rejected(self):
        self.guard.write_text(
            "GRPC_GENERATED_VERSION = '1.78.0'\nGRPC_GENERATED_VERSION = '1.81.1'\n"
        )
        with self.assertRaisesRegex(RuntimeError, "Expected one generated gRPC minimum"):
            self.run_check()

    def test_nonliteral_generated_minimum_is_not_executed(self):
        self.guard.write_text("GRPC_GENERATED_VERSION = dangerous_function()\n")
        with self.assertRaises(ValueError):
            self.run_check()
        self.refusal_mock.assert_not_called()

    def test_nonzero_server_exit_after_clients_is_not_hidden(self):
        self.processes[0].wait.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "exited with status 7"):
            self.run_check()
        self.assertEqual(self.client_calls, 2)
        self.assertEqual(self.stdout.getvalue(), "")
        self.refusal_mock.assert_not_called()

    def test_negative_signal_exit_after_clients_is_not_hidden(self):
        self.processes[1].wait.return_value = -signal.SIGSEGV
        with self.assertRaisesRegex(RuntimeError, "exited with status -11"):
            self.run_check()
        self.assertEqual(self.client_calls, 4)
        self.assertEqual(self.stdout.getvalue(), "")

    def test_sigterm_exit_code_is_not_a_clean_handled_shutdown(self):
        self.processes[0].wait.return_value = -signal.SIGTERM
        with self.assertRaisesRegex(RuntimeError, "exited with status -15"):
            self.run_check()

    def test_shutdown_timeout_kills_reaps_owned_child_and_preserves_timeout(self):
        timeout = subprocess.TimeoutExpired(["owned-fake-server"], 5)
        process = self.processes[0]
        process.wait.side_effect = [timeout, -signal.SIGKILL]
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            self.run_check()
        self.assertIs(raised.exception, timeout)
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=5), mock.call(timeout=1)])
        self.assertEqual(self.stdout.getvalue(), "")

    def test_settlement_error_does_not_replace_original_shutdown_timeout(self):
        timeout = subprocess.TimeoutExpired(["owned-fake-server"], 5)
        settlement = subprocess.TimeoutExpired(["owned-fake-server"], 1)
        self.processes[0].wait.side_effect = [timeout, settlement]
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            self.run_check()
        self.assertIs(raised.exception, timeout)
        self.assertIn("Server kill/reap also failed", self.stderr.getvalue())
        self.processes[0].kill.assert_called_once_with()

    def test_client_primary_error_survives_nonzero_shutdown(self):
        primary = subprocess.CalledProcessError(9, ["fixture-client"])
        self.client_error = primary
        self.processes[0].wait.return_value = 7
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            self.run_check()
        self.assertIs(raised.exception, primary)
        self.assertIn("exited with status 7", self.stderr.getvalue())

    def test_client_primary_error_survives_shutdown_timeout_and_reaping(self):
        primary = subprocess.CalledProcessError(9, ["fixture-client"])
        self.client_error = primary
        self.processes[0].wait.side_effect = [subprocess.TimeoutExpired(["owned-server"], 5), 0]
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            self.run_check()
        self.assertIs(raised.exception, primary)
        self.assertIn("Server cleanup also failed", self.stderr.getvalue())
        self.processes[0].kill.assert_called_once_with()

    def test_startup_json_primary_error_survives_nonzero_shutdown(self):
        self.processes[0].stdout.readline.return_value = "not JSON"
        self.processes[0].wait.return_value = 7
        with self.assertRaises(json.JSONDecodeError):
            self.run_check()
        self.assertIn("exited with status 7", self.stderr.getvalue())
        self.assertEqual(self.client_calls, 0)

    def test_startup_select_gate_runs_before_readline_even_when_optimized(self):
        self.readiness = False
        with self.assertRaisesRegex(AssertionError, "Server startup timeout"):
            self.run_check()
        self.select_mock.assert_called_once()
        self.processes[0].stdout.readline.assert_not_called()
        self.assertEqual(self.client_calls, 0)
        self.processes[0].terminate.assert_called_once_with()
        self.processes[0].wait.assert_called_once_with(timeout=5)

    def test_changed_descriptor_is_named_and_stops_before_server_launch(self):
        self.fingerprints[1]["common.proto"] = "different"
        with self.assertRaisesRegex(AssertionError, "common.proto"):
            self.run_check()
        self.assertEqual(self.spawn_calls, 0)

    def test_missing_descriptor_is_named(self):
        self.fingerprints[1] = {}
        with self.assertRaisesRegex(AssertionError, "common.proto"):
            self.run_check()
        self.assertEqual(self.spawn_calls, 0)

    def test_extra_descriptor_is_named(self):
        self.fingerprints[1]["extra.proto"] = "new"
        with self.assertRaisesRegex(AssertionError, "extra.proto"):
            self.run_check()
        self.assertEqual(self.spawn_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
