import ast
import copy
import errno
import importlib.util
import inspect
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "publication", ROOT / "scripts/ci_moriio_image_publication.py"
)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def exercise_bake_processes(source, directory):
    """Execute the real command closure/call sites; mock only the process boundary.

    Buildx's context read is implicit inside cwd; the only extra grant is the
    exact recipe directory. No Docker or network process runs in this test.
    """
    tree = ast.parse(textwrap.dedent(source))
    publish = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "publish")
    command = next(
        n for n in publish.body if isinstance(n, ast.FunctionDef) and n.name == "command"
    )
    bake_assignment = next(
        n
        for n in ast.walk(publish)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "bake" for t in n.targets)
    )
    phases = {"bake_plan", "full_build", "publish_cached_target"}
    calls = [
        n
        for n in ast.walk(publish)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "command"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value in phases
    ]
    assert len(calls) == 3
    calls.sort(key=lambda node: node.lineno)
    output = Path(directory)
    context = output / "source"
    context.mkdir()
    observed = []

    def fake_popen(argv, **kwargs):
        if "bake" in argv:
            grants = {
                Path(a.removeprefix("--allow=fs.read="))
                for a in argv
                if a.startswith("--allow=fs.read=")
            }
            assert grants == {p.RECIPE}, "Do not widen the recipe read grant"
            cwd = Path(kwargs.get("cwd") or p.ROOT)
            if not context.is_relative_to(cwd) and not any(
                context.is_relative_to(g) for g in grants
            ):
                raise PermissionError(
                    "Owned source context is outside cwd and precise recipe grant"
                )
            assert cwd == context
        else:
            assert kwargs.get("cwd") is None, "Preflight/cleanup cannot require the exported source"
        observed.append((argv, kwargs.get("cwd")))
        process = mock.Mock(pid=2**30, returncode=0, stdin=None, stdout=None, stderr=None)
        process.communicate.return_value = (b"{}", None)
        return process

    namespace = dict(
        p.__dict__,
        deadline=time.monotonic() + 30,
        output=output,
        env={"CONTEXT": str(context)},
        event=lambda *_args, **_kwargs: None,
        state={"events": []},
        docker=["docker"],
        builder="owned-test-builder",
        image="candidate",
        metadata=output / "metadata.json",
    )
    control = next(
        n
        for n in publish.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "client_control" for t in n.targets)
    )
    module = ast.Module(body=[control, command, bake_assignment], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<actual publisher command boundary>", "exec"),
        namespace,
    )
    with (
        mock.patch.object(p.subprocess, "Popen", side_effect=fake_popen),
        mock.patch.object(p.os, "killpg", side_effect=ProcessLookupError) as kill,
    ):
        for call in calls:
            eval(compile(ast.Expression(call), "<actual bake invocation>", "eval"), namespace)
        # The source may not exist during early diagnosis or final cleanup.
        shutil.rmtree(context)
        namespace["command"]("metadata_without_source", ["docker", "info"])
        namespace["command"]("cleanup_without_source", ["docker", "inspect", "owned"])
        assert all(call.args[1] == 0 for call in kill.call_args_list)
    return observed


class PublicationControls(unittest.TestCase):
    def test_bake_context_is_readable_from_all_three_actual_process_invocations(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = exercise_bake_processes(inspect.getsource(p.publish), directory)
            self.assertEqual(len(calls), 5)
            self.assertEqual([cwd for _, cwd in calls[:3]], [Path(directory) / "source"] * 3)
            self.assertEqual([cwd for _, cwd in calls[3:]], [None, None])

    def test_actual_recipe_manifest_and_material_pins(self):
        self.assertEqual(len(p.check_recipe()), 64)
        self.assertEqual(len(p.CONTEXTS), 3)

    def test_pr_cannot_reach_output_or_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            with mock.patch.dict(
                os.environ,
                {"GITHUB_REPOSITORY": "smg-project/smg", "GITHUB_EVENT_NAME": "pull_request"},
            ):
                with (
                    mock.patch.object(p.subprocess, "Popen") as start,
                    self.assertRaises(AssertionError),
                ):
                    p.publish(Path(directory), out)
                start.assert_not_called()
                self.assertFalse(out.exists())

    def test_foreign_repository_cannot_publish(self):
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "foreign/smg"}):
            with (
                mock.patch.object(p.subprocess, "Popen") as start,
                self.assertRaises(AssertionError),
            ):
                p.publish(Path("/unused"), Path("/unused"))
            start.assert_not_called()

    def test_wrong_main_ref_cannot_publish(self):
        with mock.patch.dict(
            os.environ,
            {
                "GITHUB_REPOSITORY": "smg-project/smg",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/proposal",
            },
        ):
            with (
                mock.patch.object(p.subprocess, "Popen") as start,
                self.assertRaises(AssertionError),
            ):
                p.publish(Path("/unused"), Path("/unused"))
            start.assert_not_called()

    def v2(self, directory, swap="0", memory=None, cpu=None):
        root = Path(directory)
        leaf = root / "owned"
        leaf.mkdir()
        (leaf / "cpu.max").write_text(cpu or f"{p.QUOTA} {p.PERIOD}")
        (leaf / "memory.max").write_text(memory or str(p.MEMORY))
        (leaf / "memory.swap.max").write_text(swap)
        member = Path(f"/proc/{os.getpid()}/cgroup").read_text()
        return member, {"unified": (leaf, root)}

    def test_v2_exact_kernel_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(p, "cgroup_paths", return_value=self.v2(directory)):
                result = p.limits(os.getpid())
                self.assertEqual(result["controllers"][0]["swap"], "0")

    def test_v2_extra_swap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(p, "cgroup_paths", return_value=self.v2(directory, swap="1")):
                with self.assertRaises(AssertionError):
                    p.limits(os.getpid())

    def test_worker_ancestor_capacity_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                p, "cgroup_paths", return_value=self.v2(directory, memory=str(p.MEMORY - 1))
            ):
                with self.assertRaises(AssertionError):
                    p.limits(os.getpid(), capacity=True)

    def test_v1_memory_memsw_total(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "owned"
            leaf.mkdir()
            for name, value in {
                "cpu.cfs_quota_us": p.QUOTA,
                "cpu.cfs_period_us": p.PERIOD,
                "memory.limit_in_bytes": p.MEMORY,
                "memory.memsw.limit_in_bytes": p.MEMORY,
            }.items():
                (leaf / name).write_text(str(value))
            member = Path(f"/proc/{os.getpid()}/cgroup").read_text()
            with mock.patch.object(
                p,
                "cgroup_paths",
                return_value=(member, {"cpu": (leaf, root), "memory": (leaf, root)}),
            ):
                p.limits(os.getpid())
                (leaf / "memory.memsw.limit_in_bytes").write_text(str(p.MEMORY + 1))
                with self.assertRaises(AssertionError):
                    p.limits(os.getpid())

    def test_redirect_does_not_forward_registry_bearer_to_cdn(self):
        request = urllib.request.Request(
            "https://ghcr.io/v2/repo/blobs/sha256:a", headers={"Authorization": "Bearer fake"}
        )
        redirected = p.RegistryRedirect().redirect_request(
            request, None, 307, "", {}, "https://cdn.example/blob?fake=1"
        )
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_redirect_http_is_rejected(self):
        request = urllib.request.Request("https://ghcr.io/v2/repo/manifests/a")
        with self.assertRaises(AssertionError):
            p.RegistryRedirect().redirect_request(
                request, None, 307, "", {}, "http://cdn.example/blob"
            )

    def test_registry_raw_config_digest_not_version_string(self):
        config = json.dumps(
            {"os": "linux", "architecture": "amd64", "rootfs": {"diff_ids": ["sha256:layer"]}}
        ).encode()
        manifest = json.dumps(
            {
                "config": {"digest": "sha256:" + p.digest(config)},
                "layers": [{"digest": "sha256:layer"}],
            }
        ).encode()
        token = mock.MagicMock()
        token.__enter__.return_value = io.BytesIO(b'{"token":"fake"}')
        with (
            mock.patch.object(p, "registry_open", return_value=token),
            mock.patch.object(p, "registry_get", side_effect=[manifest, config]),
        ):
            result = p.image_identity(
                "ghcr.io/smg-project/smg", "sha256:" + p.digest(manifest), p.time.monotonic() + 30
            )
            self.assertEqual(result["config_sha256"], p.digest(config))
        with (
            mock.patch.object(p, "registry_open", return_value=token),
            mock.patch.object(p, "registry_get", side_effect=[manifest, config + b"\n"]),
        ):
            token.__enter__.return_value = io.BytesIO(b'{"token":"fake"}')
            with self.assertRaises(AssertionError):
                p.image_identity(
                    "ghcr.io/smg-project/smg",
                    "sha256:" + p.digest(manifest),
                    p.time.monotonic() + 30,
                )

    def test_expired_registry_deadline_stops_before_network(self):
        with (
            mock.patch.object(p.urllib.request, "build_opener") as opener,
            self.assertRaises(AssertionError),
        ):
            p.registry_open(urllib.request.Request("https://ghcr.io/token"), p.time.monotonic() - 1)
        opener.assert_not_called()


def actual_client_namespace(directory, hook=None):
    """Compile the exact production closures; only external faults are injected."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(p.publish)))
    publish = tree.body[0]
    nodes = [
        node
        for node in publish.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in {"persist", "event", "command", "interrupted"}
        )
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "client_control" for t in node.targets)
        )
    ]
    output = Path(directory)
    namespace = dict(
        p.__dict__,
        deadline=time.monotonic() + 15,
        output=output,
        record=output / "result.json",
        state={"events": []},
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            **(
                {"LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"]}
                if "LD_LIBRARY_PATH" in os.environ
                else {}
            ),
        },
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "<actual publisher client closures>",
            "exec",
        ),
        namespace,
    )
    actual_event = namespace["event"]

    def event(phase, **extra):
        if hook:
            hook(phase, extra, namespace)
        actual_event(phase, **extra)

    namespace["event"] = event
    return namespace


class OwnedClientLifecycle(unittest.TestCase):
    def run_owned(
        self, directory, namespace, code, *, during_popen=None, after_communicate=None, until=None
    ):
        """Every child is ours; even an old/failing implementation is safely reaped."""
        born = []
        popen = subprocess.Popen
        handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
        }
        for sig in handlers:
            signal.signal(sig, namespace["interrupted"])

        def start(*args, **kwargs):
            child = popen(*args, **kwargs)
            born.append(child)
            if after_communicate:
                communicate = child.communicate

                def completed(*args, **kwargs):
                    result = communicate(*args, **kwargs)
                    after_communicate()
                    return result

                child.communicate = completed
            if during_popen:
                during_popen()
            return child

        try:
            with mock.patch.object(p.subprocess, "Popen", side_effect=start):
                return namespace["command"](
                    "client",
                    [sys.executable, "-I", "-S", "-B", "-c", code],
                    until=until,
                    capture=True,
                )
        finally:
            for sig, old in handlers.items():
                signal.signal(sig, old)
            observations = []
            for child in born:
                live = child.poll() is None
                observations.append(
                    {"PID": child.pid, "live_after_command": live, "returncode": child.returncode}
                )
                # Cleanup belongs to this test, and cannot certify the production path.
                if live:
                    self.assertEqual(os.getpgid(child.pid), child.pid)
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
                with self.assertRaises(ProcessLookupError):
                    os.killpg(child.pid, 0)
                if child.stdout:
                    child.stdout.close()
                if child.stderr:
                    child.stderr.close()
            print(json.dumps({"owned_client_observations": observations}))
            self.assertTrue(all(not item["live_after_command"] for item in observations))

    def assert_closed(self, namespace):
        client = namespace["state"]["clients"][-1]
        self.assertTrue(client["reaped"])
        self.assertIs(client["PG_gone"], True)
        self.assertTrue(client["pipes_closed"])
        self.assertFalse(client["cleanup_unknown"])

    def test_running_event_failure_reaps_actual_client(self):
        def fail(phase, _extra, _namespace):
            if phase.endswith("_running"):
                raise OSError("injected durable HANDLE event failure")

        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory, fail)
            with self.assertRaisesRegex(OSError, "HANDLE"):
                self.run_owned(directory, ns, "import time; time.sleep(30)")
            self.assert_closed(ns)

    def test_signals_between_actual_popen_and_handle_are_deferred_then_reaped(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as directory:
                ns = actual_client_namespace(directory)
                with self.assertRaisesRegex(RuntimeError, "cancellation pending"):
                    self.run_owned(
                        directory,
                        ns,
                        "import time; time.sleep(30)",
                        during_popen=lambda: os.kill(os.getpid(), sig),
                    )
                self.assert_closed(ns)
                self.assertEqual(ns["client_control"]["pending_signal"], sig)

    def test_completed_communicate_delivers_pending_before_direct_consumer(self):
        publish = ast.parse(textwrap.dedent(inspect.getsource(p.publish))).body[0]
        direct_consumer = next(
            node
            for node in ast.walk(publish)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "export_source"
        )
        for sig in (signal.SIGTERM, signal.SIGALRM):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as directory:
                ns = actual_client_namespace(directory)
                ns["engine_checkout"] = Path(directory)
                consumer = mock.Mock()
                ns["export_source"] = consumer
                with self.assertRaisesRegex(RuntimeError, "cancellation pending"):
                    self.run_owned(
                        directory,
                        ns,
                        "print('completed')",
                        after_communicate=lambda: os.kill(os.getpid(), sig),
                    )
                    exec(
                        compile(
                            ast.fix_missing_locations(
                                ast.Module(body=[direct_consumer], type_ignores=[])
                            ),
                            "<actual next direct consumer>",
                            "exec",
                        ),
                        ns,
                    )
                consumer.assert_not_called()
                self.assert_closed(ns)
                self.assertTrue(ns["state"]["clients"][-1]["pending_cancel_delivered"])

    def test_pending_before_admission_never_spawns(self):
        def cancel(phase, _extra, ns):
            if phase.endswith("_submitted"):
                ns["interrupted"](signal.SIGTERM, None)

        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory, cancel)
            with mock.patch.object(p.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "cancellation pending"):
                    ns["command"]("client", ["not-executed"])
                spawn.assert_not_called()
            self.assertFalse(ns["state"]["clients"][-1]["Popen_attempted"])
            self.assert_closed(ns)

    def test_healthy_actual_child_capture_and_unblocked_exec_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            result = self.run_owned(
                directory,
                ns,
                "import json,signal; print(json.dumps(sorted(int(s) for s in "
                "signal.pthread_sigmask(signal.SIG_BLOCK, []))))",
            )
            blocked = json.loads(result)
            self.assertTrue(
                all(sig not in blocked for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM))
            )
            self.assert_closed(ns)

    def test_deadline_timeout_reaps_actual_client(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            before = time.monotonic()
            with self.assertRaisesRegex(AssertionError, "deadline expired"):
                self.run_owned(directory, ns, "import time; time.sleep(30)", until=before + 3.15)
            self.assertLess(time.monotonic() - before, 3)
            self.assert_closed(ns)

    def test_nonzero_actual_child_is_failure_and_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            with self.assertRaisesRegex(AssertionError, "failed"):
                self.run_owned(directory, ns, "raise SystemExit(7)")
            self.assertEqual(ns["state"]["clients"][-1]["returncode"], 7)
            self.assert_closed(ns)

    def test_cleanup_phase_runs_bounded_owned_child_despite_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            ns["client_control"]["cleaning"] = True
            ns["interrupted"](signal.SIGTERM, None)
            self.assertEqual(self.run_owned(directory, ns, "print('cleanup')"), b"cleanup\n")
            self.assert_closed(ns)

    def test_outside_command_signals_still_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            with self.assertRaisesRegex(RuntimeError, "publisher interrupted"):
                ns["interrupted"](signal.SIGTERM, None)

    def test_actual_export_source_direct_git_still_interrupted(self):
        # Real export_source -> check_output/run/communicate, only the external
        # Git executable is replaced by our harmless sleeping Python child.
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            born = []
            popen = subprocess.Popen
            previous = signal.getsignal(signal.SIGALRM)

            def harmless_git(*args, **kwargs):
                self.assertEqual(args[0][0], "git")
                kwargs["start_new_session"] = True
                child = popen(
                    [sys.executable, "-I", "-S", "-B", "-c", "import time; time.sleep(30)"],
                    **kwargs,
                )
                born.append(child)
                return child

            signal.signal(signal.SIGALRM, ns["interrupted"])
            started = time.monotonic()
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.05)
                with mock.patch.object(p.subprocess, "Popen", side_effect=harmless_git):
                    with self.assertRaisesRegex(RuntimeError, "publisher interrupted"):
                        p.export_source(Path(directory), Path(directory) / "source")
                self.assertLess(time.monotonic() - started, 2)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
                live = [child for child in born if child.poll() is None]
                for child in born:
                    if child.poll() is None:
                        self.assertEqual(os.getpgid(child.pid), child.pid)
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
                    with self.assertRaises(ProcessLookupError):
                        os.killpg(child.pid, 0)
                self.assertEqual(live, [])

    def test_constructor_without_process_object_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            with mock.patch.object(p.subprocess, "Popen", side_effect=OSError("constructor")):
                with self.assertRaisesRegex(OSError, "constructor"):
                    ns["command"]("client", ["not-executed"])
            client = ns["state"]["clients"][-1]
            self.assertTrue(client["Popen_attempted"])
            self.assertFalse(client["reaped"])
            self.assertIsNone(client["PG_gone"])
            self.assertTrue(client["cleanup_unknown"])
            self.assertTrue(ns["state"]["owned_client_cleanup_unknown"])

    def test_pre_spawn_event_error_has_closed_no_spawn_receipt(self):
        def fail(phase, _extra, _namespace):
            if phase.endswith("_submitted"):
                raise OSError("pre-spawn event")

        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory, fail)
            with mock.patch.object(p.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(OSError, "pre-spawn"):
                    ns["command"]("client", ["not-executed"])
                spawn.assert_not_called()
            self.assert_closed(ns)

    def test_returned_receipt_failure_preserves_closed_actual_child(self):
        def fail(phase, _extra, _namespace):
            if phase.endswith("_returned"):
                raise OSError("returned receipt")

        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory, fail)
            with self.assertRaisesRegex(OSError, "returned receipt"):
                self.run_owned(directory, ns, "print('finite')")
            self.assert_closed(ns)


def exercise_owned_cleanup_fault(directory, fault, *, foreign=False):
    """Run actual publisher teardown and real harmless clients, never Docker.

    Only the productive try-body is replaced by a fault after exact ownership
    binding. Command/identity/except/finally and final outcome stay production.
    """
    publish = ast.parse(textwrap.dedent(inspect.getsource(p.publish))).body[0]
    nodes = [
        copy.deepcopy(node)
        for node in publish.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name
            in {"persist", "event", "command", "inspect", "verify_owned", "interrupted"}
        )
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "client_control" for t in node.targets)
        )
    ]
    operation = next(node for node in publish.body if isinstance(node, ast.Try))
    operation = copy.deepcopy(operation)
    operation.body = ast.parse("write_json(record, state)").body
    tail = publish.body[
        publish.body.index(next(n for n in publish.body if isinstance(n, ast.Try))) + 1 :
    ]
    output = Path(directory)
    state = {"events": [], "public_pull_verified": True, "owned_builder_container_id": "a" * 64}
    namespace = dict(
        p.__dict__,
        output=output,
        record=output / "result.json",
        state=state,
        start=time.monotonic(),
        deadline=time.monotonic() + 15,
        builder="smg-moriio-123-1",
        name="buildx_buildkit_smg-moriio-123-10",
        owned_id="a" * 64,
        creation_submitted=True,
        docker=["docker"],
        failure=None,
        env={
            "PATH": "/usr/bin:/bin",
            **(
                {"LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"]}
                if "LD_LIBRARY_PATH" in os.environ
                else {}
            ),
        },
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "<exact publisher cleanup closures>",
            "exec",
        ),
        namespace,
    )
    actual_write = p.write_json
    calls, children = [], []
    interrupted = False
    actual_open = Path.open

    def record_write(path, value):
        nonlocal interrupted
        if path == namespace["record"]:
            if fault == "persistent":
                raise OSError(errno.ENOSPC, "original persistent receipt fault")
            if fault == "interrupt" and not interrupted:
                interrupted = True

                def partial(value, stream, **kwargs):
                    stream.write('{"partial":')
                    os.kill(os.getpid(), signal.SIGTERM)
                    raise AssertionError("SIGTERM handler did not stop the direct write")

                with mock.patch.object(p.json, "dump", side_effect=partial):
                    return actual_write(path, value)
        return actual_write(path, value)

    def open_with_failed_log(path, *args, **kwargs):
        if fault == "persistent" and path.suffix == ".log":
            raise OSError(errno.ENOSPC, "persistent owned log fault")
        return actual_open(path, *args, **kwargs)

    actual_popen = subprocess.Popen

    def harmless_client(argv, **kwargs):
        calls.append(list(argv))
        assert argv[0] == "docker"
        if argv[1] == "inspect":
            stopped = any(a[1] == "stop" for a in calls)
            payload = json.dumps(
                [
                    {
                        "Id": "a" * 64,
                        "Name": "/" + namespace["name"],
                        "Config": {"Image": "foreign" if foreign else p.BUILDKIT},
                        "State": {"Running": not stopped, "Pid": 0 if stopped else 123},
                    }
                ]
            )
        elif argv[1] == "stop":
            assert argv == ["docker", "stop", "--time", "5", "a" * 64]
            payload = "stopped"
        elif argv[1:3] == ["buildx", "rm"]:
            assert argv == ["docker", "buildx", "rm", namespace["builder"]]
            payload = "removed"
        elif argv[1] == "ps":
            assert "id=" + "a" * 64 in argv
            payload = ""
        else:
            raise AssertionError("No productive or unknown Docker command is allowed")
        child = actual_popen(
            [sys.executable, "-I", "-S", "-B", "-c", "print(" + repr(payload) + ")"], **kwargs
        )
        children.append(child)
        return child

    error = None
    handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
    }
    for sig in handlers:
        signal.signal(sig, namespace["interrupted"])
    try:
        with (
            mock.patch.object(p, "write_json", side_effect=record_write),
            mock.patch.object(Path, "open", open_with_failed_log),
            mock.patch.object(p.subprocess, "Popen", side_effect=harmless_client),
        ):
            namespace["write_json"] = p.write_json
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[operation, *tail], type_ignores=[])),
                    "<exact publisher finally and outcome>",
                    "exec",
                ),
                namespace,
            )
    except BaseException as exc:
        error = exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError("test-owned client group survived")
    print(
        json.dumps(
            {
                "cleanup_fault": fault,
                "foreign": foreign,
                "actual_Docker_argv_mapped_to_harmless_clients": calls,
                "children_reaped": all(c.returncode is not None for c in children),
                "error": type(error).__name__ if error else None,
                "message": str(error),
                "state": state,
            }
        )
    )
    return namespace, calls, error


class ReceiptFaultCleanup(unittest.TestCase):
    def test_interrupted_record_retains_original_error_and_cleans_exact_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            ns, calls, error = exercise_owned_cleanup_fault(directory, "interrupt")
            self.assertIsInstance(error, RuntimeError)
            self.assertIn("publisher interrupted " + str(signal.SIGTERM), str(error))
            self.assertTrue(ns["state"].get("cleanup_verified_absent"))
            self.assertIn(["docker", "stop", "--time", "5", "a" * 64], calls)
            self.assertFalse(list(Path(directory).glob("result*.new")))

    def test_persistent_receipt_and_log_fault_still_stops_and_removes_owned_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            ns, calls, error = exercise_owned_cleanup_fault(directory, "persistent")
            self.assertIsInstance(error, OSError)
            self.assertIn("original persistent receipt fault", str(error))
            self.assertTrue(ns["state"].get("cleanup_verified_absent"))
            self.assertIn(["docker", "buildx", "rm", "smg-moriio-123-1"], calls)
            self.assertTrue(ns["state"].get("receipt_failures"))
            self.assertTrue(
                all(c["reaped"] and c["PG_gone"] is True for c in ns["state"]["clients"])
            )

    def test_failed_receipt_does_not_authorize_foreign_builder_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            ns, calls, error = exercise_owned_cleanup_fault(directory, "persistent", foreign=True)
            self.assertIsInstance(error, OSError)
            self.assertTrue(ns["state"].get("cleanup_failure"))
            self.assertFalse(ns["state"].get("cleanup_verified_absent", False))
            self.assertTrue(any(a[1] == "inspect" for a in calls))
            self.assertFalse(any(a[1] in {"stop", "buildx"} for a in calls))

    def test_receipt_failure_refuses_later_productive_command(self):
        with tempfile.TemporaryDirectory() as directory:
            ns = actual_client_namespace(directory)
            with mock.patch.object(p, "write_json", side_effect=OSError("writer fault")):
                ns["write_json"] = p.write_json
                with self.assertRaises(OSError):
                    ns["event"]("prior_receipt")
            with mock.patch.object(p.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "receipt"):
                    ns["command"]("full_build", ["not-executed"])
                spawn.assert_not_called()

    def test_healthy_cleanup_and_old_unrelated_temp_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "result.new"
            stale.write_text("unrelated old file")
            ns, calls, error = exercise_owned_cleanup_fault(directory, "healthy")
            self.assertIsNone(error)
            self.assertTrue(ns["state"]["cleanup_verified_absent"])
            self.assertEqual(stale.read_text(), "unrelated old file")
            self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
