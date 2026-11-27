"""Portable stdlib fixtures for the owned interpreter and native path APIs.

No pip, build backend, SGLang, Torch, native extension or model is executed.
The existing qualification counterexamples are retained in the author evidence;
these tests make their current invariants available to ordinary CPU CI.
"""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import py_compile
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root, name, value="fresh"):
    source = root / name / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = " + repr(value) + "\n")
    folder = root / (name + "-1.0.dist-info")
    folder.mkdir()
    metadata = folder / "METADATA"
    metadata.write_text("Metadata-Version: 2.1\nName: " + name + "\nVersion: 1.0\n")
    (folder / "RECORD").write_text(
        name + "/__init__.py,,\n" + folder.name + "/METADATA,,\n" + folder.name + "/RECORD,,\n"
    )
    return {
        "module": name,
        "distribution": name,
        "version": "1.0",
        "package_root": str(root),
        "module_path": str(source),
        "metadata_path": str(metadata),
        "source_sha256": digest(source),
        "metadata_sha256": digest(metadata),
    }


def child(python, code, **kwargs):
    return subprocess.run(
        [str(python), "-I", "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
        **kwargs,
    )


@pytest.fixture
def cohort(tmp_path):
    api = load("ci_sglang_cohort")
    owned, borrowed = tmp_path / "owned", tmp_path / "borrowed"
    owned.mkdir()
    borrowed.mkdir()
    packages = [package(owned, "ownedfixture"), package(borrowed, "borrowedfixture")]
    sentinel = tmp_path / "foreign-startup"
    startup = "from pathlib import Path; Path(" + repr(str(sentinel)) + ").touch()\n"
    (borrowed / "hostile.pth").write_text(startup)
    (borrowed / "sitecustomize.py").write_text(startup)
    hook = tmp_path / "fixture_hook.py"
    hook.write_text(
        "import json,sys,subprocess\nfrom pathlib import Path\n"
        "result=subprocess.run([sys.executable,'-c',"
        "'import sys,json; print(json.dumps([sys.flags.isolated,sys.flags.dont_write_bytecode]))'],"
        "capture_output=True,text=True,check=True)\n"
        "(Path(sys.argv[2])/'trace.json').write_text(json.dumps("
        "[sys.flags.isolated,sys.flags.dont_write_bytecode,json.loads(result.stdout)]))\n"
    )
    prefix = tmp_path / "interpreter"
    python = api.create(
        prefix,
        [owned],
        packages,
        borrowed_sites=[borrowed],
        isolated_children=True,
        backend_hook={"path": str(hook), "sha256": digest(hook)},
    )
    manifest = next(prefix.glob("lib/python*/site-packages/smg-cohort.json"))
    return types.SimpleNamespace(
        api=api,
        owned=owned,
        borrowed=borrowed,
        packages=packages,
        sentinel=sentinel,
        hook=hook,
        prefix=prefix,
        python=python,
        manifest=manifest,
        root=tmp_path,
    )


def test_fresh_module_metadata_projection_and_startup_exclusion(cohort):
    result = child(
        cohort.python,
        "import ownedfixture,borrowedfixture,importlib.metadata as md,sys,json,sitecustomize; "
        "from ci_sglang_cohort import verify_loaded; verify_loaded(sitecustomize.SMG_COHORT); "
        "assert [ownedfixture.VALUE,borrowedfixture.VALUE]==['fresh','fresh']; "
        "names=sorted(d.metadata['Name'] for p in sys.path for d in md.distributions(path=[p])); "
        "assert names==['borrowedfixture','ownedfixture']; "
        "print(json.dumps([sys.flags.isolated,sys.flags.dont_write_bytecode]))",
        env={**os.environ, "PYTHONPATH": str(cohort.borrowed)},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [1, 1]
    assert not cohort.sentinel.exists()


@pytest.mark.parametrize("kind", ["source", "version", "metadata", "cache", "launcher", "core"])
def test_bootstrap_rejects_tampering_before_the_consumer(cohort, kind):
    spec = json.loads(cohort.manifest.read_text())
    if kind == "source":
        Path(cohort.packages[0]["module_path"]).write_text("VALUE='tampered'\n")
    elif kind == "version":
        spec["packages"][0]["version"] = "2.0"
    elif kind == "metadata":
        other = package(cohort.borrowed, "ownedfixture")
        spec["packages"][0].update(
            metadata_path=other["metadata_path"],
            metadata_sha256=other["metadata_sha256"],
            package_root=str(cohort.root),
        )
    elif kind == "cache":
        (cohort.prefix / "source-cache/untrusted").touch()
    elif kind == "launcher":
        cohort.python.write_text(cohort.python.read_text() + "# changed\n")
    else:
        spec["interpreter_core_sha256"] = "0" * 64
    cohort.manifest.write_text(json.dumps(spec))
    result = child(cohort.python, "print('CONSUMER_EXECUTED')")
    assert result.returncode == 78, result.stderr
    assert "Cohort bootstrap refused" in result.stderr
    assert "CONSUMER_EXECUTED" not in result.stdout
    assert not cohort.sentinel.exists()


def test_a_preloaded_module_cannot_be_requalified_by_matching_metadata(cohort):
    result = child(
        cohort.python,
        "import ownedfixture; from ci_sglang_cohort import activate; "
        "activate(" + repr(str(cohort.manifest)) + ")",
    )
    assert result.returncode == 1
    assert "Protected module was preloaded" in result.stderr


def test_source_cache_redirect_ignores_valid_old_pyc_for_owned_and_borrowed(cohort):
    for entry in cohort.packages:
        source = Path(entry["module_path"])
        source.write_text("VALUE = 'stale'\n")
        stamp = source.stat()
        py_compile.compile(str(source), doraise=True)
        source.write_text("VALUE = 'fresh'\n")
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert digest(source) == entry["source_sha256"]
    result = child(
        cohort.python,
        "import ownedfixture,borrowedfixture; "
        "assert [ownedfixture.VALUE,borrowedfixture.VALUE]==['fresh','fresh']",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["normal", "changed_hook", "outside_control", "undeclared"])
def test_actual_pip_hook_argv_shape_keeps_child_and_grandchild_isolated(cohort, mode):
    control = cohort.root / "control"
    control.mkdir()
    if mode == "changed_hook":
        cohort.hook.write_text(cohort.hook.read_text() + "# changed\n")
    elif mode == "outside_control":
        control = cohort.root.parent
    elif mode == "undeclared":
        spec = json.loads(cohort.manifest.read_text())
        spec["backend_hook"] = None
        cohort.manifest.write_text(json.dumps(spec))
    # Captured pip calls its selected interpreter without adding -I or -B.
    result = subprocess.run(
        [str(cohort.python), str(cohort.hook), "build_wheel", str(control)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if mode == "normal":
        assert result.returncode == 0, result.stderr
        assert json.loads((control / "trace.json").read_text()) == [1, 1, [1, 1]]
    else:
        assert result.returncode == 78, result.stderr
        assert not (control / "trace.json").exists()


def test_owned_launcher_does_not_source_bash_env(cohort):
    startup = cohort.root / "bash-env"
    startup.write_text("touch " + str(cohort.sentinel) + "\n")
    result = child(cohort.python, "pass", env={**os.environ, "BASH_ENV": str(startup)})
    assert result.returncode == 0, result.stderr
    assert not cohort.sentinel.exists()


def test_namespace_packages_keep_owned_and_borrowed_parts(cohort):
    for root, name in ((cohort.owned, "ownedpart"), (cohort.borrowed, "borrowedpart")):
        namespace = root / "namespacefixture"
        namespace.mkdir()
        (namespace / (name + ".py")).write_text("VALUE='fixture'\n")
    result = child(
        cohort.python,
        "import namespacefixture.ownedpart,namespacefixture.borrowedpart; "
        "assert namespacefixture.ownedpart.VALUE==namespacefixture.borrowedpart.VALUE=='fixture'",
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def private_pip(cohort, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    api = load("ci_sglang_private_pip")
    target = cohort.root / "native"
    site = target / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    site.mkdir(parents=True)
    constraint = cohort.root / "torch-constraint.txt"
    constraint.write_text("torch==fixture\n")
    monkeypatch.setattr(api, "CONSTRAINT", constraint)
    monkeypatch.setattr(api, "CONSTRAINT_SHA", digest(constraint))
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraint))
    monkeypatch.setenv("PIP_TARGET", "/foreign-target")
    monkeypatch.setenv("PYTHONPATH", "/foreign-python")
    for name in ("home", "tmp", "cache", "build-tracker"):
        (cohort.root / name).mkdir()
    spec = json.loads(cohort.manifest.read_text())
    spec.update(image=api.IMAGE, install_prefix=str(target))
    spec["import_roots"].append(str(site))
    cohort.manifest.write_text(json.dumps(spec))
    return api, cohort, target


def test_private_install_command_preserves_full_dependencies_and_owns_write_inputs(private_pip):
    api, cohort, target = private_pip
    command, env = api.install_command(cohort.manifest, ["-e", "fixture-project"])
    assert command == [
        str(cohort.python),
        "-m",
        "pip",
        "install",
        "--no-user",
        "--prefix",
        str(target),
        "--no-compile",
        "--no-build-isolation",
        "-e",
        "fixture-project",
    ]
    assert "--no-deps" not in command and "--ignore-installed" not in command
    assert "PIP_TARGET" not in env and "PYTHONPATH" not in env
    assert env["TMPDIR"] == str(cohort.root / "tmp")
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["_PIP_USE_IMPORTLIB_METADATA"] == "1"


@pytest.mark.parametrize("arguments", [["--target", "/foreign"], ["--pre", "/foreign"], ["-e"]])
def test_private_install_rejects_destination_and_unknown_or_incomplete_options(
    private_pip, arguments
):
    api, cohort, _ = private_pip
    with pytest.raises(ValueError):
        api.install_command(cohort.manifest, arguments)


def test_private_install_checks_original_constraint_before_any_process(private_pip):
    api, cohort, _ = private_pip
    api.CONSTRAINT.write_text("changed\n")
    with pytest.raises(ValueError, match="Torch constraint Source differs"):
        api.install_command(cohort.manifest, ["fixture-project"])


def test_strict_record_console_and_installed_pyc_paths_are_independent(tmp_path, monkeypatch):
    native = load("ci_verify_sglang_rocm_cohort")
    root = tmp_path / "native/lib/python/site-packages"
    root.mkdir(parents=True)
    console = root.parents[2] / "bin/fixture-console"
    console.parent.mkdir()
    console.write_bytes(b"fixture console\n")
    folder = root / "fixture.dist-info"
    folder.mkdir()
    name = "../../../bin/fixture-console"
    encoded = (
        base64.urlsafe_b64encode(hashlib.sha256(console.read_bytes()).digest()).decode().rstrip("=")
    )
    data = io.StringIO()
    writer = csv.writer(data)
    writer.writerow([name, "sha256=" + encoded, str(console.stat().st_size)])
    writer.writerow([folder.name + "/RECORD", "", ""])
    (folder / "RECORD").write_text(data.getvalue())
    with pytest.raises(RuntimeError, match="Unexpected console path"):
        native.verify_record(root, folder, (name,))
    native.verify_record(root, folder, (name,), console_dir=console.parent)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "execution-cache"))
    assert native._installed_pyc_path(root / "fixture.py").is_relative_to(root)
    console.write_bytes(b"tampered console\n")
    with pytest.raises(RuntimeError, match="Installed RECORD mismatch"):
        native.verify_record(root, folder, (name,), console_dir=console.parent)


@pytest.mark.parametrize(
    "mode", ["normal", "foreign_startup", "changed_manifest", "changed_metadata", "console_alias"]
)
def test_native_layout_requires_the_active_bootstrap_and_one_owned_scheme(
    cohort, monkeypatch, mode
):
    api = load("ci_sglang_native_paths")
    root = (
        cohort.root
        / f"native/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    root.mkdir(parents=True)
    entries = {name: package(root, name) for name in ("tvm_ffi", "tilelang", "torch_c_dlpack_ext")}
    console = root.parents[2] / "bin"
    console.mkdir()
    boot = types.ModuleType("ci_sglang_cohort")
    boot.__file__ = str(cohort.manifest.with_name("ci_sglang_cohort.py"))
    boot.SMG_MANIFEST_SHA256 = digest(cohort.manifest)
    boot.SMG_COHORT = entries
    monkeypatch.setattr(sys, "prefix", str(cohort.prefix))
    monkeypatch.setitem(sys.modules, "ci_sglang_cohort", boot)
    monkeypatch.setitem(sys.modules, "sitecustomize", boot)
    if mode == "foreign_startup":
        monkeypatch.setitem(sys.modules, "sitecustomize", types.ModuleType("foreign"))
    elif mode == "changed_manifest":
        cohort.manifest.write_text(cohort.manifest.read_text() + " ")
    elif mode == "changed_metadata":
        Path(entries["tilelang"]["metadata_path"]).write_text("changed\n")
    elif mode == "console_alias":
        console.rmdir()
        console.symlink_to(cohort.owned, target_is_directory=True)
    if mode == "normal":
        assert api.current_layout() == {
            "prefix": cohort.prefix,
            "site": root,
            "bin": console,
            "scope": cohort.root,
        }
    else:
        with pytest.raises(ValueError):
            api.current_layout()


def test_original_sdk_pre_observer_rejects_changed_or_aliased_fixture_bytes(tmp_path, monkeypatch):
    observer = load("ci_sglang_mxfp4_fallback")
    source = tmp_path / "fixture.py"
    source.write_bytes(b"original\n")
    monkeypatch.setattr(observer, "FILES", {source.name: (digest(source), "0" * 64)})
    monkeypatch.setenv(observer.ROW_ENV, "0")
    assert observer.verify_installed(tmp_path)["state"] == "original"
    source.write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="mixed"):
        observer.verify_installed(tmp_path)
    source.unlink()
    source.symlink_to(tmp_path / "outside.py")
    with pytest.raises(RuntimeError, match="Invalid SGLang SDK source"):
        observer.source_hashes(tmp_path)
