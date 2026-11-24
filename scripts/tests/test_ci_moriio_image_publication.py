import ast
import importlib.util
import inspect
import io
import json
import os
import shutil
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
        process = mock.Mock(pid=os.getpid(), returncode=0)
        process.communicate.return_value = (b"{}", None)
        return process

    namespace = dict(
        p.__dict__,
        deadline=time.monotonic() + 30,
        output=output,
        env={"CONTEXT": str(context)},
        event=lambda *_args, **_kwargs: None,
        docker=["docker"],
        builder="owned-test-builder",
        image="candidate",
        metadata=output / "metadata.json",
    )
    module = ast.Module(body=[command, bake_assignment], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<actual publisher command boundary>", "exec"),
        namespace,
    )
    with (
        mock.patch.object(p.subprocess, "Popen", side_effect=fake_popen),
        mock.patch.object(p.os, "killpg") as kill,
    ):
        for call in calls:
            eval(compile(ast.Expression(call), "<actual bake invocation>", "eval"), namespace)
        # The source may not exist during early diagnosis or final cleanup.
        shutil.rmtree(context)
        namespace["command"]("metadata_without_source", ["docker", "info"])
        namespace["command"]("cleanup_without_source", ["docker", "inspect", "owned"])
        kill.assert_not_called()
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


if __name__ == "__main__":
    unittest.main()
