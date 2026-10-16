"""The servicer switch verifies its hook at start-up
(``launcher_switch.require_launcher_hook``, run from the plugin's parse step):
a launcher the switch cannot bind is refused with one line before the engine
is built, not by the Python servicer after the model has loaded. No engine:
the launchers are stub modules and stub packages on disk."""

from __future__ import annotations

import argparse
import logging
import sys
import types
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.vllm import launcher_switch, plugin, rust

LAUNCHER = "vllm.entrypoints.launchers.grpc_server"
OLD_LAUNCHER = "vllm.entrypoints.grpc_server"
SERVE_GRPC = "async def serve_grpc(args):\n    return 'python ran'\n"


@pytest.fixture
def vllm_package(monkeypatch):
    """A stub ``vllm`` package (version 0.29.1) in place of any installed one,
    with no launcher loaded; what an import under test adds under it is
    removed again."""
    present = {name for name in sys.modules if name == "vllm" or name.startswith("vllm.")}
    for name in present:
        monkeypatch.delitem(sys.modules, name)
    package = types.ModuleType("vllm")
    package.__version__ = "0.29.1"
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", package)
    yield package
    for name in [name for name in sys.modules if name == "vllm" or name.startswith("vllm.")]:
        if name not in present:
            sys.modules.pop(name, None)


def _loaded_launcher(monkeypatch, name=LAUNCHER, **attrs):
    """A launcher module already in ``sys.modules`` (the deprecated shim's
    import, or a launcher that imported this package after its body)."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _launcher_tree(tmp_path, package, source, *, layout="launchers"):
    """A ``vllm`` tree on disk the stub package imports from: the launcher at
    the 0.30+ path, the 0.28/0.29 path, or a path the switch does not know."""
    root = tmp_path / "vllm"
    entrypoints = root / "entrypoints"
    entrypoints.mkdir(parents=True)
    (entrypoints / "__init__.py").write_text("")
    if layout == "launchers":
        (entrypoints / "launchers").mkdir()
        (entrypoints / "launchers" / "__init__.py").write_text("")
        (entrypoints / "launchers" / "grpc_server.py").write_text(source)
    elif layout == "old":
        (entrypoints / "grpc_server.py").write_text(source)
    else:
        (entrypoints / "grpc").mkdir()
        (entrypoints / "grpc" / "__init__.py").write_text("")
        (entrypoints / "grpc" / "server.py").write_text(source)
    package.__path__ = [str(root)]
    return root


async def _run_grpc_server(args):
    return "python ran"


def test_a_bound_launcher_passes(vllm_package, monkeypatch):
    module = _loaded_launcher(monkeypatch)
    assert launcher_switch.install_launcher_switch() == [LAUNCHER]
    module.__dict__["serve_grpc"] = _run_grpc_server  # the body's own `async def`
    hook = launcher_switch.require_launcher_hook()
    assert (hook.state, hook.module, hook.bindable) == ("bound", LAUNCHER, True)
    assert getattr(module.__dict__["serve_grpc"], "__smg_servicer_switch__", False)


def test_a_launcher_without_the_hook_is_refused_by_name(vllm_package, monkeypatch):
    """A release whose launcher defines no serve_grpc: one line naming the
    installed version, the module, what it lacks, the launchers the switch
    binds, and the way out."""
    _loaded_launcher(monkeypatch, run_grpc_server=_run_grpc_server)
    assert launcher_switch.launcher_hook() == launcher_switch.LauncherHook("missing", LAUNCHER)
    with pytest.raises(rust_lifecycle.ServicerConfigError) as raised:
        launcher_switch.require_launcher_hook()
    line = str(raised.value)
    assert "\n" not in line
    assert line.startswith("--servicer-impl rust (SMG_VLLM_SERVICER_IMPL=rust) refused")
    assert f"vLLM 0.29.1: its gRPC launcher {LAUNCHER} defines no serve_grpc" in line
    assert "vllm.entrypoints.grpc_server (0.28 and 0.29)" in line
    assert "run as __main__ by `python -m`" in line
    assert "select --servicer-impl python (or unset SMG_VLLM_SERVICER_IMPL)" in line


@pytest.mark.parametrize("name", [LAUNCHER, OLD_LAUNCHER])
def test_a_launcher_running_as_main_is_bound_in_its_namespace_and_never_re_imported(
    vllm_package, monkeypatch, tmp_path, name
):
    """`python -m` on the launcher module itself (the 0.30+ module, or the
    0.28/0.29 one): the check runs inside its main(), after the body defined
    serve_grpc, and the switch is bound in the __main__ namespace; importing
    the module under its dotted name would only make a second copy. A body
    that defines no serve_grpc is refused by name."""
    layout = "launchers" if name == LAUNCHER else "old"
    _launcher_tree(tmp_path, vllm_package, SERVE_GRPC, layout=layout)
    main = SimpleNamespace(__spec__=SimpleNamespace(name=name), serve_grpc=_run_grpc_server)
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert launcher_switch.require_launcher_hook() == launcher_switch.LauncherHook("bound", name)
    assert getattr(main.serve_grpc, "__smg_servicer_switch__", False)
    assert name not in sys.modules
    bare = SimpleNamespace(__spec__=SimpleNamespace(name=name))
    monkeypatch.setitem(sys.modules, "__main__", bare)
    with pytest.raises(rust_lifecycle.ServicerConfigError) as raised:
        launcher_switch.require_launcher_hook()
    assert f"its gRPC launcher {name} defines no serve_grpc" in str(raised.value)
    assert name not in sys.modules


def test_the_deprecated_shim_as_main_passes_through_the_launcher_it_imports(
    vllm_package, monkeypatch
):
    """0.30+: `python -m vllm.entrypoints.grpc_server` runs the shim as
    __main__, and the launcher it imports under its own name is bound."""
    module = _loaded_launcher(monkeypatch)
    launcher_switch.install_launcher_switch()
    module.__dict__["serve_grpc"] = _run_grpc_server
    monkeypatch.setitem(
        sys.modules, "__main__", SimpleNamespace(__spec__=SimpleNamespace(name=OLD_LAUNCHER))
    )
    assert launcher_switch.require_launcher_hook().state == "bound"


@pytest.mark.parametrize(
    ("layout", "name"), [("launchers", LAUNCHER), ("old", OLD_LAUNCHER)], ids=["0.30+", "0.28"]
)
def test_a_launcher_not_loaded_yet_is_imported_and_bound(
    vllm_package, monkeypatch, tmp_path, layout, name
):
    """`vllm serve --grpc` parses before it imports the launcher: the check
    imports it under the first known name the installation has, and the
    finder binds the switch as it loads."""
    _launcher_tree(tmp_path, vllm_package, SERVE_GRPC, layout=layout)
    launcher_switch.install_launcher_switch()
    hook = launcher_switch.require_launcher_hook()
    assert (hook.state, hook.module) == ("bound", name)
    assert getattr(sys.modules[name].__dict__["serve_grpc"], "__smg_servicer_switch__", False)
    # Without the import, nothing is loaded and nothing is said.
    assert launcher_switch.launcher_hook().state == "bound"


def test_a_launcher_at_a_path_the_switch_does_not_know_is_refused(
    vllm_package, monkeypatch, tmp_path
):
    _launcher_tree(tmp_path, vllm_package, SERVE_GRPC, layout="moved")
    with pytest.raises(rust_lifecycle.ServicerConfigError) as raised:
        launcher_switch.require_launcher_hook()
    assert f"it has no gRPC launcher module under {LAUNCHER} or {OLD_LAUNCHER}" in str(raised.value)
    # Nothing of the launcher is loaded under the known names.
    assert launcher_switch.launcher_hook() == launcher_switch.LauncherHook("absent")
    assert LAUNCHER not in sys.modules and OLD_LAUNCHER not in sys.modules


def test_a_launcher_at_an_unknown_path_passes_when_its_source_carries_the_check(
    vllm_package, monkeypatch, tmp_path
):
    """The documented alternative to the binding: a launcher that carries the
    check in its own source passes the parse-time check as it passes the
    preflight, wherever it lives under ``entrypoints/``."""
    carrying = f"# consults the flag itself: {rust.HOOK_SYMBOL}\n" + SERVE_GRPC
    _launcher_tree(tmp_path, vllm_package, carrying, layout="moved")
    hook = launcher_switch.require_launcher_hook()
    assert (hook.state, hook.bindable) == ("source", True)
    assert LAUNCHER not in sys.modules and OLD_LAUNCHER not in sys.modules


def test_a_loaded_launcher_without_the_hook_passes_when_its_source_carries_the_check(
    vllm_package, monkeypatch, tmp_path
):
    """A known-path launcher loaded without ``serve_grpc`` (renamed, or run as
    ``__main__`` with the check in its body) is not refused when its file
    carries the check."""
    module = _loaded_launcher(monkeypatch, run_grpc_server=_run_grpc_server)
    path = tmp_path / "grpc_server.py"
    path.write_text(f"# consults the flag itself: {rust.HOOK_SYMBOL}\n")
    module.__file__ = str(path)
    assert launcher_switch.launcher_hook() == launcher_switch.LauncherHook("missing", LAUNCHER)
    hook = launcher_switch.require_launcher_hook()
    assert (hook.state, hook.module) == ("source", LAUNCHER)


def test_a_newer_name_without_the_hook_is_passed_over_for_an_older_one_that_binds(
    vllm_package, monkeypatch, tmp_path
):
    """A tree that answers the newer name with a module that is not the
    launcher: the older name binds, and the verdict is its."""
    root = _launcher_tree(tmp_path, vllm_package, "def main():\n    pass\n", layout="launchers")
    (root / "entrypoints" / "grpc_server.py").write_text(SERVE_GRPC)
    launcher_switch.install_launcher_switch()
    hook = launcher_switch.require_launcher_hook()
    assert (hook.state, hook.module) == ("bound", OLD_LAUNCHER)


def test_a_newer_name_without_the_hook_keeps_its_verdict_when_the_older_import_fails(
    vllm_package, monkeypatch, tmp_path
):
    """The 0.30+ shape this check exists for: the launcher renamed its
    coroutine and the deprecated name is a shim whose import fails on it; the
    shim's failure does not turn the missing hook into a pass."""
    root = _launcher_tree(tmp_path, vllm_package, "def main():\n    pass\n", layout="launchers")
    (root / "entrypoints" / "grpc_server.py").write_text(
        "raise ImportError('cannot import serve_grpc from the launcher')\n"
    )
    launcher_switch.install_launcher_switch()
    with pytest.raises(rust_lifecycle.ServicerConfigError) as raised:
        launcher_switch.require_launcher_hook()
    assert f"its gRPC launcher {LAUNCHER} defines no serve_grpc" in str(raised.value)


def test_a_launcher_whose_import_fails_on_its_own_dependency_is_left_to_report_it(
    vllm_package, monkeypatch, tmp_path
):
    """The launcher's own `ImportError("gRPC mode requires ...") from e` is
    not the switch's verdict: no refusal, the CLI's import raises it next."""
    _launcher_tree(
        tmp_path,
        vllm_package,
        "try:\n"
        "    import no_such_dependency_for_this_test\n"
        "except ImportError as e:\n"
        "    raise ImportError('gRPC mode requires smg-grpc-servicer') from e\n" + SERVE_GRPC,
    )
    hook = launcher_switch.require_launcher_hook()
    assert (hook.state, hook.module, hook.bindable) == ("unknown", LAUNCHER, True)


def test_a_launcher_still_importing_is_not_judged(vllm_package, monkeypatch):
    """The launcher imports this package mid-body, before its serve_grpc
    exists: the hooked module binds it after the body, so nothing is said."""
    _loaded_launcher(monkeypatch, __spec__=SimpleNamespace(_initializing=True))
    hook = launcher_switch.require_launcher_hook(import_launcher=False)
    assert (hook.state, hook.bindable) == ("pending", True)


def test_the_version_comes_from_the_package_then_the_distribution(vllm_package, monkeypatch):
    assert launcher_switch.installed_engine_version() == "0.29.1"
    del vllm_package.__version__
    monkeypatch.setattr(
        launcher_switch.importlib.metadata, "version", lambda name: {"vllm": "0.31.0"}[name]
    )
    assert launcher_switch.installed_engine_version() == "0.31.0"

    def not_found(name):
        raise launcher_switch.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(launcher_switch.importlib.metadata, "version", not_found)
    assert launcher_switch.installed_engine_version() is None
    _loaded_launcher(monkeypatch)
    assert "vLLM (version unknown)" in launcher_switch.refusal_line(launcher_switch.launcher_hook())


def test_the_plugins_parse_step_refuses_before_the_engine(vllm_package, monkeypatch, caplog):
    """The check runs where the launcher parses its arguments, for a parser
    that serves gRPC: a Rust selection (flag or environment) on a launcher
    the switch cannot bind is one ERROR line and exit code 2 there; Python
    passes, and a parser of another command selects nothing."""
    _loaded_launcher(monkeypatch, run_grpc_server=_run_grpc_server)
    monkeypatch.delenv(rust.SERVICER_IMPL_ENV, raising=False)
    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("--grpc", action="store_true")
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as raised:
        plugin.check_servicer_impl(parser, argparse.Namespace(grpc=True, servicer_impl="rust"))
    assert raised.value.code == rust_lifecycle.CONFIG_ERROR_EXIT_CODE
    assert "servicer configuration error: --servicer-impl rust" in caplog.text
    assert f"{LAUNCHER} defines no serve_grpc" in caplog.text
    # The environment form is the same selection; the stock launcher's own
    # parser has no --grpc (gRPC is all it serves).
    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "rust")
    with pytest.raises(SystemExit):
        plugin.check_servicer_impl(parser, argparse.Namespace(grpc=True, servicer_impl=None))
    with pytest.raises(SystemExit):
        plugin.check_servicer_impl(parser, argparse.Namespace(servicer_impl=None))
    # `vllm serve` without --grpc serves no gRPC: nothing is selected or checked.
    assert (
        plugin.check_servicer_impl(parser, argparse.Namespace(grpc=False, servicer_impl="rust"))
        is None
    )
    # Python, chosen by the flag over the exported rust, needs no launcher.
    assert (
        plugin.check_servicer_impl(parser, argparse.Namespace(grpc=True, servicer_impl="python"))
        == "python"
    )
    # A parser of another command selects nothing, whatever the environment says.
    other = argparse.ArgumentParser(prog="vllm bench")
    other.add_argument("--model")
    assert plugin.check_servicer_impl(other, argparse.Namespace(servicer_impl="rust")) is None
    # A value the switch cannot use is the same exit, at the parse.
    with pytest.raises(SystemExit) as raised:
        plugin.check_servicer_impl(parser, argparse.Namespace(grpc=True, servicer_impl="go"))
    assert raised.value.code == rust_lifecycle.CONFIG_ERROR_EXIT_CODE
    assert "must be one of" in caplog.text


def test_the_plugins_parse_step_passes_a_bound_launcher(vllm_package, monkeypatch):
    module = _loaded_launcher(monkeypatch)
    launcher_switch.install_launcher_switch()
    module.__dict__["serve_grpc"] = _run_grpc_server
    monkeypatch.delenv(rust.SERVICER_IMPL_ENV, raising=False)
    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("--grpc", action="store_true")
    assert (
        plugin.check_servicer_impl(parser, argparse.Namespace(grpc=True, servicer_impl="rust"))
        == "rust"
    )


def test_the_flags_help_names_the_launchers_the_switch_binds():
    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("--grpc", action="store_true")
    assert plugin.add_servicer_impl_argument(parser)
    # argparse wraps the help at the terminal width: compare on one line.
    help_text = " ".join(parser.format_help().split())
    assert f"vLLM {launcher_switch.SUPPORTED_VLLM_RELEASES} checked" in help_text
    assert "python -m" in help_text and "refused at this parse" in help_text


def test_the_preflight_counts_the_import_only_at_the_paths_the_switch_binds(tmp_path):
    """`smg serve` and the e2e harness ask `upstream_hook_installed` before
    starting workers: a launcher that imports this package from a path the
    switch does not bind by name is not a hook; the two known paths are."""
    importing = "from smg_grpc_servicer.vllm.servicer import VllmEngineServicer\n" + SERVE_GRPC
    root = tmp_path / "vllm"
    moved = root / "entrypoints" / "grpc" / "server.py"
    moved.parent.mkdir(parents=True)
    moved.write_text(importing)
    assert rust.upstream_hook_installed(vllm_root=str(root)) is False
    # 0.28/0.29: the launcher is vllm/entrypoints/grpc_server.py.
    old = root / "entrypoints" / "grpc_server.py"
    old.write_text(importing)
    assert rust.upstream_hook_installed(vllm_root=str(root)) is True
    # 0.30+: that file is a shim over vllm/entrypoints/launchers/grpc_server.py.
    old.write_text("from vllm.entrypoints.launchers.grpc_server import main, serve_grpc\n")
    assert rust.upstream_hook_installed(vllm_root=str(root)) is False
    launcher = root / "entrypoints" / "launchers" / "grpc_server.py"
    launcher.parent.mkdir()
    launcher.write_text(importing)
    assert rust.upstream_hook_installed(vllm_root=str(root)) is True
    # A launcher anywhere that carries the check itself still counts.
    launcher.unlink()
    moved.write_text("from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust\n")
    assert rust.upstream_hook_installed(vllm_root=str(root)) is True
