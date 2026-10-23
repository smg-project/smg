"""The servicer switch on a launcher running as ``__main__`` (``python -m`` on
the launcher module itself, which is what ``python -m vllm.entrypoints.grpc_server``
is on vLLM 0.28 and 0.29), the launcher table, and the CPU smoke that runs each
upstream launcher layout through to the servicer construction."""

from __future__ import annotations

import argparse
import asyncio
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
from smg_grpc_servicer.vllm import launcher_switch, plugin, rust

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS_DIR = Path(__file__).resolve().parent / "launchers"
NEW, OLD = launcher_switch.LAUNCHER_MODULES
SWITCHED = "__smg_servicer_switch__"


def _main_module(spec_name: str | None, **names) -> types.ModuleType:
    """A ``__main__`` as runpy makes it for ``python -m``: a plain module whose
    spec names the module run; ``None`` for a file path."""
    main = types.ModuleType("__main__")
    main.__spec__ = None if spec_name is None else importlib.machinery.ModuleSpec(spec_name, None)
    main.__dict__.update(names)
    return main


async def serve_grpc(args):
    """The launcher's own coroutine, as its body defines it."""
    return "python ran"


@pytest.fixture
def no_loaded_launcher(monkeypatch):
    """No launcher module imported under its name in this process."""
    for name in launcher_switch.LAUNCHER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize("name", [OLD, NEW], ids=["0.28-0.29", "0.30+"])
def test_a_launcher_running_as_main_is_bound_in_its_namespace(
    monkeypatch, no_loaded_launcher, name
):
    """The launcher's main() reads serve_grpc from its globals: the entry in
    the __main__ dict is rebound, the module object itself left as it is."""
    main = _main_module(name, serve_grpc=serve_grpc)
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert launcher_switch.launcher_main() is main
    assert launcher_switch.install_launcher_switch() == ["__main__"]
    switched = main.__dict__["serve_grpc"]
    assert switched is not serve_grpc and switched.__wrapped__ is serve_grpc
    assert getattr(switched, SWITCHED, False) and switched.__name__ == "serve_grpc"
    assert type(main) is types.ModuleType
    assert asyncio.run(switched(argparse.Namespace(servicer_impl="python"))) == "python ran"
    calls = []

    async def fake_serve_rust(args):
        calls.append(args)
        return 7

    monkeypatch.setattr(rust, "serve_rust", fake_serve_rust)
    with pytest.raises(SystemExit) as raised:
        asyncio.run(switched(argparse.Namespace(servicer_impl="rust")))
    assert raised.value.code == 7 and len(calls) == 1
    # Idempotent, and stable across re-installs.
    assert launcher_switch.install_launcher_switch() == ["__main__"]
    assert main.__dict__["serve_grpc"] is switched


def test_a_main_launcher_mid_body_is_bound_by_the_plugins_parse_step(
    monkeypatch, no_loaded_launcher
):
    """The servicer import runs before the body defined serve_grpc, so there
    is nothing to bind then; the launcher's own parse, inside its main(), runs
    the plugin's parse step, which binds it before main() reads the name."""
    main = _main_module(OLD)
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert launcher_switch.install_launcher_switch() == []

    # vLLM's parser class, as the plugin wraps it; the media flags' handoff is
    # not this test's (it writes to the environment).
    class FlexibleArgumentParser(argparse.ArgumentParser):
        pass

    for name in ("vllm", "vllm.utils", "vllm.utils.argparse_utils"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.utils.argparse_utils"].FlexibleArgumentParser = FlexibleArgumentParser
    monkeypatch.setattr(plugin, "add_mm_arguments_to_grpc_parser", lambda parser: [])
    monkeypatch.setattr(plugin, "handoff_mm_flags", lambda parser, parsed: None)
    # vLLM loaded the plugins during the launcher's own imports: still mid-body.
    plugin.register()
    main.__dict__["serve_grpc"] = serve_grpc  # the body goes on
    assert not getattr(main.__dict__["serve_grpc"], SWITCHED, False)
    parser = FlexibleArgumentParser(prog="grpc_server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--model")
    parser.parse_args(["--model", "m"])  # the launcher's own parse, inside its main()
    assert getattr(main.__dict__["serve_grpc"], SWITCHED, False)


def test_a_main_that_is_not_the_launcher_is_left_alone(monkeypatch, no_loaded_launcher):
    """The vllm console script, the shim, a file path: not the launcher as
    __main__, nothing in their namespace is touched."""
    for spec_name in ("vllm.entrypoints.cli.main", None):
        main = _main_module(spec_name, serve_grpc=serve_grpc)
        monkeypatch.setitem(sys.modules, "__main__", main)
        assert launcher_switch.launcher_main() is None
        assert launcher_switch.install_launcher_switch() == []
        assert main.__dict__["serve_grpc"] is serve_grpc


def test_the_launcher_table_names_both_layouts_and_their_releases():
    assert launcher_switch.LAUNCHER_MODULES == (NEW, OLD)
    assert [launcher.floor for launcher in launcher_switch.LAUNCHERS] == [(0, 30), (0, 28)]
    assert launcher_switch.HOOK_ATTRIBUTE == "serve_grpc"
    for_release = launcher_switch.launcher_for_release
    assert for_release("0.28.0").name == OLD
    assert for_release("v0.29.1").name == OLD
    # By the release's layout; a main-branch build may already carry the next one.
    assert for_release("0.29.1rc1.dev347+gabcdef0").name == OLD
    assert for_release("0.30.0").name == NEW
    assert for_release("0.31.0").name == NEW
    assert for_release("1.0").name == NEW
    assert for_release("0.27.0") is None
    assert for_release(None) is None
    assert for_release("latest") is None
    line = launcher_switch.supported_launchers()
    assert NEW in line and OLD in line and launcher_switch.SUPPORTED_VLLM_RELEASES in line
    assert "python -m" in line


def test_the_python_guard_names_the_entries_the_switch_binds():
    with pytest.raises(RuntimeError) as raised:
        rust.require_python_impl(environ={rust.SERVICER_IMPL_ENV: "rust"})
    message = str(raised.value)
    assert "`python -m` on the launcher module" in message
    assert launcher_switch.SUPPORTED_VLLM_RELEASES in message
    assert "VLLM_PLUGINS" in message


def test_the_flag_joins_the_stock_launchers_own_parser(monkeypatch):
    """The stock gRPC launcher's parser defines no --grpc; it serves gRPC all
    the same (the process's __main__ is the launcher module), so the plugin's
    flag joins it as the --mm-* flags do. Other commands' parsers stay bare."""
    for name in launcher_switch.LAUNCHER_MODULES:
        monkeypatch.setitem(sys.modules, "__main__", _main_module(name))
        parser = argparse.ArgumentParser(prog="grpc_server")
        parser.add_argument("--port", type=int, default=50051)
        assert plugin.add_servicer_impl_argument(parser) is True
        assert plugin.add_servicer_impl_argument(parser) is False  # once
        assert parser.parse_args(["--servicer-impl", "rust"]).servicer_impl == "rust"
    monkeypatch.setitem(sys.modules, "__main__", _main_module("vllm.entrypoints.cli.main"))
    serve = argparse.ArgumentParser(prog="vllm serve")
    serve.add_argument("--grpc", action="store_true")
    assert plugin.add_servicer_impl_argument(serve) is True
    bench = argparse.ArgumentParser(prog="vllm bench")
    bench.add_argument("--model")
    assert plugin.add_servicer_impl_argument(bench) is False


# ---------------------------------------------------------------------------
# The CPU smoke: each upstream launcher, through to the servicer construction
# ---------------------------------------------------------------------------


def _smoke():
    name = "launcher_switch_smoke"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve the module's annotations through it
    spec.loader.exec_module(module)
    return module


def _run(layout: str, version: str, selections: tuple[str, ...]):
    smoke = _smoke()
    results = smoke.run_matrix(
        launcher_tree=LAUNCHERS_DIR / layout,
        servicer_root=ROOT,
        proto_path=os.environ.get("SMG_GRPC_PROTO_PATH"),
        version=version,
        selections=selections,
        timeout=120,
    )
    failed = [result for result in results if not result.passed]
    assert not failed, "\n".join(
        f"{result.case.label} ({result.case.impl}): {result.detail}\n{result.stderr[-2000:]}"
        for result in failed
    )
    return smoke, results


LAYOUTS = [
    pytest.param("v0_28", "0.28.0", [OLD], id="0.28"),
    pytest.param("v0_30", "0.30.0", [NEW, OLD], id="0.30"),
    pytest.param("v0_31", "0.31.0", [NEW, OLD], id="0.31"),
]


@pytest.mark.parametrize(("layout", "version", "launchers"), LAYOUTS)
def test_each_upstream_launcher_runs_through_to_the_servicer_on_every_entry(
    layout, version, launchers
):
    """The vendored upstream launcher of each release (tests/launchers/README.md)
    runs in a fresh interpreter, as `python -m` on every module the release has
    and as the CLI's import after its parse, with Rust selected by the variable
    (what a deployment sets) and with the Python default: Rust hands over to
    serve_rust before any engine exists; Python builds the stub engine and
    constructs the servicer on it. The 0.28 `python -m` case is the launcher
    as __main__."""
    smoke, results = _run(layout, version, ("env",))
    assert smoke.launcher_modules_in_tree(LAUNCHERS_DIR / layout) == launchers
    covered = {(result.case.form, result.case.launcher, result.case.impl) for result in results}
    expected = {("module", launcher, impl) for launcher in launchers for impl in ("rust", "python")}
    expected |= {("cli", launchers[0], impl) for impl in ("rust", "python")}
    assert covered == expected


def test_the_flag_selects_rust_on_the_launcher_run_as_main():
    """`--servicer-impl rust` on the 0.28 launcher's own command line: the
    plugin's flag, parsed by the launcher as __main__, selects Rust."""
    smoke, results = _run("v0_28", "0.28.0", ("flag",))
    flagged = [result for result in results if result.case.selection == "flag"]
    assert {result.case.form for result in flagged} == {"module", "cli"}
    for result in flagged:
        assert '"servicer_impl": "rust"' in result.stdout
