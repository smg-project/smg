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
        py_compile.compile(
            str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP
        )
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
    project = cohort.root / "fixture-project"
    project.mkdir()
    (project / "editablefixture.py").write_text("VALUE='editable'\n")
    spec.update(
        image=api.IMAGE,
        install_prefix=str(target),
        editable_projects={str(project): [str(project)]},
    )
    spec["import_roots"].append(str(project))
    spec["import_roots"].append(str(site))
    cohort.manifest.write_text(json.dumps(spec))
    return api, cohort, target


def test_private_install_command_preserves_full_dependencies_and_owns_write_inputs(private_pip):
    api, cohort, target = private_pip
    command, env = api.install_command(
        cohort.manifest, ["-e", str(cohort.root / "fixture-project")]
    )
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
        str(cohort.root / "fixture-project"),
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


@pytest.mark.parametrize(
    "name", ["PIP_PYTHON", "PIP_NO_DEPS", "PIP_DRY_RUN", "PIP_REQUIREMENT", "PIP_EDITABLE"]
)
def test_inherited_pip_behavior_cannot_override_complete_arguments(private_pip, monkeypatch, name):
    api, cohort, _ = private_pip
    monkeypatch.setenv(name, "untrusted")
    monkeypatch.setenv("PIP_INDEX_URL", "https://fixture.invalid/simple")
    command, env = api.install_command(cohort.manifest, ["fixture-package", "--no-deps"])
    assert name not in env
    assert "--no-deps" in command
    assert env["PIP_INDEX_URL"] == "https://fixture.invalid/simple"
    assert env["PIP_CONSTRAINT"] == str(api.CONSTRAINT)


def test_editable_needs_declared_mapping_and_fresh_child_can_import_it(private_pip):
    api, cohort, _ = private_pip
    project = cohort.root / "fixture-project"
    api.install_command(cohort.manifest, ["-e", str(project)])
    result = child(
        cohort.python, "import editablefixture; assert editablefixture.VALUE=='editable'"
    )
    assert result.returncode == 0, result.stderr
    spec = json.loads(cohort.manifest.read_text())
    spec["editable_projects"] = {}
    cohort.manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="explicit cohort import mapping"):
        api.install_command(cohort.manifest, ["--editable=" + str(project)])


def test_transaction_scheme_uses_declared_cohort_minor(private_pip):
    api, cohort, target = private_pip
    spec = json.loads(cohort.manifest.read_text())
    minor = sys.version_info.minor + 1
    site = target / f"lib/python3.{minor}/site-packages"
    site.mkdir(parents=True)
    spec["python_version"] = [3, minor]
    spec["import_roots"].append(str(site))
    cohort.manifest.write_text(json.dumps(spec))
    # Command construction observes the declared scheme; executing the foreign
    # version manifest would be rejected by the actual interpreter bootstrap.
    api.install_command(cohort.manifest, ["fixture-package"])


def test_create_failure_removes_only_its_reserved_prefix(tmp_path, monkeypatch):
    api = load("ci_sglang_cohort")
    prefix = tmp_path / "new-interpreter"

    def fail(_self, path):
        (path / "bin").mkdir()
        (path / "bin/python").write_text("partial launcher")
        raise OSError("fixture setup failure")

    monkeypatch.setattr(api.venv.EnvBuilder, "create", fail)
    with pytest.raises(OSError, match="fixture setup failure"):
        api.create(prefix, [], [])
    assert not prefix.exists()


def test_exclusive_reservation_refuses_a_creator_that_lost_the_race(tmp_path, monkeypatch):
    api = load("ci_sglang_cohort")
    prefix = tmp_path / "interpreter"
    prefix.mkdir()
    marker = prefix / "other-owner"
    marker.touch()
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if p == prefix else exists(p))
    with pytest.raises(FileExistsError):
        api.create(prefix, [], [])
    assert marker.is_file()


def test_stdlib_stays_before_unprotected_owned_module(cohort):
    (cohort.owned / "colorsys.py").write_text("raise RuntimeError('owned stdlib shadow')\n")
    result = child(cohort.python, "import colorsys; assert colorsys.rgb_to_hsv(1,0,0)==(0.0,1.0,1)")
    assert result.returncode == 0, result.stderr


def test_distribution_cannot_redirect_record_to_another_metadata_file(cohort):
    entry = cohort.packages[0]
    original = Path(entry["metadata_path"]).parent
    nested = cohort.owned / "pins" / original.name
    nested.parent.mkdir()
    original.rename(nested)
    decoy = cohort.owned / "decoy.dist-info"
    decoy.mkdir()
    (decoy / "METADATA").write_text("Name: ownedfixture\nVersion: 1.0\nUnverified: yes\n")
    (decoy / "RECORD").write_text(str((nested / "METADATA").relative_to(cohort.owned)) + ",,\n")
    spec = json.loads(cohort.manifest.read_text())
    spec["packages"][0]["metadata_path"] = str(nested / "METADATA")
    cohort.manifest.write_text(json.dumps(spec))
    result = child(cohort.python, "print('CONSUMER_EXECUTED')")
    assert result.returncode == 78
    assert "CONSUMER_EXECUTED" not in result.stdout


def test_start_hashes_exactly_the_manifest_bytes_it_validated(cohort, monkeypatch):
    api = cohort.api
    before = cohort.manifest.read_bytes()
    monkeypatch.setattr(api, "__file__", str(cohort.manifest.with_name("ci_sglang_cohort.py")))
    monkeypatch.setitem(sys.modules, "ci_sglang_cohort", api)
    # Record the original presence/value even when no startup module existed.
    monkeypatch.setitem(sys.modules, "sitecustomize", object())
    monkeypatch.delitem(sys.modules, "sitecustomize")

    def validate(path, *, manifest_bytes=None):
        assert manifest_bytes == before
        path.write_bytes(before + b" ")
        return {}

    def exit_refused(code):
        raise RuntimeError("refused startup " + str(code))

    monkeypatch.setattr(api, "activate", validate)
    monkeypatch.setattr(api, "os", types.SimpleNamespace(write=lambda *_: None, _exit=exit_refused))
    api.start()
    assert api.SMG_MANIFEST_SHA256 == hashlib.sha256(before).hexdigest()
    with pytest.raises(RuntimeError, match="refused startup 78"):
        api.start()


def test_sdk_root_matches_its_approved_provider_not_the_native_site(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    native = load("ci_verify_sglang_rocm_cohort")
    import ci_sglang_mxfp4_fallback as observer

    sdk = tmp_path / "sdk-site/sglang"
    sdk.mkdir(parents=True)
    boot = types.SimpleNamespace(SMG_COHORT={"sglang": {"module_path": str(sdk / "__init__.py")}})
    monkeypatch.setitem(sys.modules, "ci_sglang_cohort", boot)
    monkeypatch.setattr(observer, "sdk_root", lambda: sdk)
    assert native.sdk_source_root(private_cohort=True) == sdk
    monkeypatch.setattr(observer, "sdk_root", lambda: tmp_path / "foreign-sdk")
    with pytest.raises(RuntimeError, match="SDK origin differs"):
        native.sdk_source_root(private_cohort=True)


@pytest.mark.parametrize("mode", ["new_distribution", "version_replacement", "stale_same_version"])
def test_owned_transactions_refuse_mixed_or_stale_replacements(private_pip, monkeypatch, mode):
    api, cohort, target = private_pip
    site = target / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"

    def dist(name, version, extra=False):
        folder = site / f"{name}-{version}.dist-info"
        folder.mkdir(exist_ok=True)
        (folder / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
        paths = [f"{folder.name}/METADATA", f"{folder.name}/RECORD", f"{name}.py"]
        (site / f"{name}.py").write_text("# harmless fixture\n")
        if extra:
            (site / "old_only.py").touch()
            paths.append("old_only.py")
        (folder / "RECORD").write_text("".join(p + ",,\n" for p in paths))

    dist("fixture", "1.0", extra=True)

    def transaction(*args, **kwargs):
        if mode == "new_distribution":
            dist("another", "1.0")
        elif mode == "version_replacement":
            dist("fixture", "2.0")
        else:
            dist("fixture", "1.0")

    monkeypatch.setattr(api, "subprocess", types.SimpleNamespace(run=transaction))
    if mode == "new_distribution":
        api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)
        assert not (cohort.root / ".private-pip-transaction").exists()
        api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)
        assert not (cohort.root / ".private-pip-transaction").exists()
    else:
        with pytest.raises(ValueError, match="duplicate metadata|stale files"):
            api.install(cohort.manifest, ["fixture-input", "--upgrade"], cwd=cohort.root)
        assert (cohort.root / ".private-pip-transaction").is_file()
        # ci_retry must not turn the second attempt into success after a new
        # RECORD has forgotten the stale file retained by the failed attempt.
        monkeypatch.setattr(api, "subprocess", types.SimpleNamespace(run=lambda *a, **k: None))
        with pytest.raises(FileExistsError):
            api.install(cohort.manifest, ["fixture-input"], cwd=cohort.root)


def test_computed_site_cannot_be_inside_an_ancestor_target_interpreter(private_pip):
    api, cohort, _ = private_pip
    prefix = cohort.root / "lib"
    site = prefix / f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    # The venv itself uses lib/lib/pythonX.Y; the target scheme below would
    # still write inside the interpreter even though target is its ancestor.
    python = cohort.api.create(prefix, [cohort.owned], [cohort.packages[0]], isolated_children=True)
    site.mkdir(parents=True)
    manifest = next(prefix.glob("lib/python*/site-packages/smg-cohort.json"))
    spec = json.loads(manifest.read_text())
    spec.update(image=api.IMAGE, install_prefix=str(cohort.root))
    spec["import_roots"].append(str(site))
    manifest.write_text(json.dumps(spec))
    assert python.is_file()
    with pytest.raises(ValueError, match="Install site must be outside"):
        api.install_command(manifest, ["fixture-input"])


def test_shared_observer_does_not_expose_a_dead_specialized_install_cli(tmp_path):
    report = tmp_path / "must-not-be-created.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPTS / "ci_sglang_mxfp4_fallback.py"),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "no install CLI" in result.stderr
    assert not report.exists()
