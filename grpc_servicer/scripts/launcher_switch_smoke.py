#!/usr/bin/env python3
"""CPU smoke for the vLLM servicer switch: run vLLM's gRPC launcher the way
its documented entry points run it, through to the servicer construction,
against a stub engine, and check that the switch engaged.

For every launcher module the installation has (``vllm.entrypoints.launchers
.grpc_server`` from vLLM 0.30, ``vllm.entrypoints.grpc_server``: the launcher
itself on 0.28 and 0.29, a shim importing the new module from 0.30) and every
entry form (``python -m <module>``, which runs the module as ``__main__``, and
``vllm serve <model> --grpc``, which imports ``serve_grpc`` after its parse),
a fresh interpreter runs the launcher with the Rust implementation selected
(the environment variable, as a deployment sets it, or ``--servicer-impl
rust``) and with the default Python one:

* Rust: the launcher must hand the process to ``serve_rust`` before any
  engine is built. ``serve_rust`` is a stand-in that prints a marker and
  returns :data:`EXIT_RUST`, which the switch raises as the exit code.
* Python: the launcher must build its engine (a stub) and construct the
  Python servicer on it; the constructor runs the real guard
  (``require_python_impl``), prints a marker and stops the process with
  :data:`EXIT_SERVICER`.

Two sources of launcher:

``--launcher-tree DIR``
    A ``vllm/`` tree holding only the launcher files (``vllm/entrypoints/
    grpc_server.py``, ``vllm/entrypoints/launchers/grpc_server.py``), as
    extracted from a wheel or as vendored under ``tests/launchers/``. The
    rest of vLLM is stubbed: the modules the launcher imports with the
    behaviour the launcher needs (an ``AsyncEngineArgs.add_cli_args`` that
    loads the general plugins as vLLM's does, a ``FlexibleArgumentParser``),
    every other ``vllm`` name a placeholder. This package's modules are the
    real ones (``--servicer-root`` names the checkout; the installed package
    otherwise), so the switch installs itself from the launcher's own import
    of them exactly as in production. Needs no engine, no torch.
``--installed``
    The installed vLLM: its real launcher, CLI, parser and plugin loading,
    with only the engine stubbed (``AsyncEngineArgs.create_engine_config``,
    ``AsyncLLM.from_vllm_config``). A CPU-only container with the engine
    image is enough; no model is read.

Exit status 0 when every case passed; the table says which did not.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

MARK_RUST = "launcher-switch-smoke: serve_rust reached"
MARK_ENGINE = "launcher-switch-smoke: engine built"
MARK_SERVICER = "launcher-switch-smoke: Python servicer constructed"
EXIT_RUST = 7  # what the stand-in serve_rust returns, raised by the switch
EXIT_SERVICER = 42  # the process stops at the Python servicer's construction
LAUNCHER_FILES = (
    ("vllm.entrypoints.launchers.grpc_server", "vllm/entrypoints/launchers/grpc_server.py"),
    ("vllm.entrypoints.grpc_server", "vllm/entrypoints/grpc_server.py"),
)
MODEL = "smoke/model"
_PLACEHOLDER_TOP_LEVEL = ("torch", "transformers", "grpc_health", "grpc_reflection")


@dataclasses.dataclass(frozen=True)
class Case:
    """One launch: the entry form, the launcher it goes through, the
    implementation selected and how."""

    form: str  # "module" (python -m <launcher>) or "cli" (vllm serve --grpc)
    launcher: str  # the module python -m runs, or the one the CLI imports
    impl: str  # "rust" or "python"
    selection: str  # "env" (the variable), "flag" (--servicer-impl), "default"

    @property
    def label(self) -> str:
        how = {"env": "SMG_VLLM_SERVICER_IMPL=rust", "flag": "--servicer-impl rust"}.get(
            self.selection, "default"
        )
        entry = f"python -m {self.launcher}" if self.form == "module" else "vllm serve --grpc"
        return f"{entry} [{how}]"


@dataclasses.dataclass(frozen=True)
class Result:
    case: Case
    passed: bool
    detail: str
    returncode: int
    stdout: str
    stderr: str


def launcher_modules_in_tree(tree: Path) -> list[str]:
    """The launcher modules a tree has, newest layout first."""
    return [name for name, relative in LAUNCHER_FILES if (tree / relative).is_file()]


def launcher_modules_installed() -> list[str]:
    """The launcher modules the installed vLLM has, newest layout first."""
    found = []
    for name, _relative in LAUNCHER_FILES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            found.append(name)
    return found


def cases_for(launchers: list[str], selections: tuple[str, ...]) -> list[Case]:
    """The matrix: ``python -m`` on every launcher module present, and the
    CLI's import of the newest one (what ``vllm serve --grpc`` imports on that
    release), each with the Rust selections asked for and the Python default."""
    cases = []
    for launcher in launchers:
        for selection in selections:
            cases.append(Case("module", launcher, "rust", selection))
        cases.append(Case("module", launcher, "python", "default"))
    if launchers:
        for selection in selections:
            cases.append(Case("cli", launchers[0], "rust", selection))
        cases.append(Case("cli", launchers[0], "python", "default"))
    return cases


# ---------------------------------------------------------------------------
# The stub vLLM around a launcher tree
# ---------------------------------------------------------------------------

_STUB_FILES = {
    "vllm/_smoke.py": '''
"""Stand-ins for the launcher-switch smoke."""
import logging

MARK_ENGINE = %(mark_engine)r


class _PlaceholderMeta(type):
    def __getattr__(cls, name):
        return cls


class Placeholder(metaclass=_PlaceholderMeta):
    """Any vLLM (or torch, transformers) name a module imports and the
    smoke does not model: a class whose attributes, calls and instances are
    placeholders too, so module bodies that only name them import."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Placeholder()

    def __getattr__(self, name):
        return Placeholder


class StubConfig:
    """The engine config the stub arguments build; nothing reads it before
    the servicer construction."""


class StubEngine:
    """The engine the Python servicer is constructed on; nothing reads it
    before the construction is recorded."""

    vllm_config = StubConfig()

    def shutdown(self):
        pass


def engine_built():
    print(MARK_ENGINE, flush=True)
    return StubEngine()
''',
    "vllm/__init__.py": '''
"""Stub vllm package for the launcher-switch smoke: the real launcher files
under entrypoints/, the modules they import stubbed, every other name a
placeholder (vllm._smoke)."""
__version__ = %(version)r


def __getattr__(name):
    from vllm._smoke import Placeholder

    return Placeholder
''',
    "vllm/version.py": "__version__ = %(version)r\n",
    "vllm/envs.py": """
VLLM_LOG_STATS_INTERVAL = 10.0
VLLM_PLUGINS = None


def __getattr__(name):
    return None
""",
    "vllm/logger.py": """
import logging


def init_logger(name):
    return logging.getLogger(name)


def configure_logging_from_args(args):
    pass
""",
    "vllm/plugins/__init__.py": '''
"""vLLM's general-plugin loading, as the launcher's parser build runs it:
once per process, the `vllm.general_plugins` entry points, filtered by
VLLM_PLUGINS when set. A checkout of this package that is not installed as a
distribution has no entry point; its plugin is then loaded by import, which
is what an editable install registers."""
import importlib
import logging
import os
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)
plugins_loaded = False


def load_plugins_by_group(group):
    allowed = os.environ.get("VLLM_PLUGINS")
    allowed_plugins = None if allowed is None else [p for p in allowed.split(",") if p]
    discovered = {plugin.name: plugin.load for plugin in entry_points(group=group)}
    if group == "vllm.general_plugins" and "smg-servicer" not in discovered:
        discovered["smg-servicer"] = lambda: importlib.import_module(
            "smg_grpc_servicer.vllm.plugin"
        ).register
    plugins = {}
    for name, load in discovered.items():
        if allowed_plugins is not None and name not in allowed_plugins:
            logger.info("plugin %%s skipped by VLLM_PLUGINS", name)
            continue
        plugins[name] = load()
    return plugins


def load_general_plugins():
    global plugins_loaded
    if plugins_loaded:
        return
    plugins_loaded = True
    for name, func in load_plugins_by_group("vllm.general_plugins").items():
        func()
        logger.info("plugin %%s loaded", name)
''',
    "vllm/usage/__init__.py": "",
    "vllm/usage/usage_lib.py": """
import enum


class UsageContext(enum.Enum):
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
""",
    "vllm/utils/__init__.py": "",
    "vllm/utils/argparse_utils.py": '''
import argparse


class FlexibleArgumentParser(argparse.ArgumentParser):
    """vLLM's parser class, as far as the launcher and the plugin use it."""
''',
    "vllm/engine/__init__.py": "",
    "vllm/engine/arg_utils.py": '''
"""The engine arguments, as far as the launcher uses them: the parser build
loads the general plugins first, as vLLM's does."""
import dataclasses

from vllm.plugins import load_general_plugins


@dataclasses.dataclass
class AsyncEngineArgs:
    model: str = "facebook/opt-125m"

    @staticmethod
    def add_cli_args(parser):
        load_general_plugins()
        parser.add_argument("--model", default="facebook/opt-125m")
        parser.add_argument("--served-model-name", nargs="+", default=None)
        parser.add_argument("--trust-remote-code", action="store_true")
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--enable-log-requests", action="store_true")
        parser.add_argument("--disable-log-stats", action="store_true")
        return parser

    @classmethod
    def from_cli_args(cls, args):
        return cls(model=args.model)

    def create_engine_config(self, usage_context=None, **kwargs):
        from vllm._smoke import StubConfig

        return StubConfig()
''',
    "vllm/v1/__init__.py": "",
    "vllm/v1/engine/__init__.py": "",
    "vllm/v1/engine/async_llm.py": """
from vllm._smoke import engine_built


class AsyncLLM:
    @classmethod
    def from_vllm_config(cls, vllm_config=None, **kwargs):
        return engine_built()
""",
    "vllm/entrypoints/__init__.py": "",
    "vllm/entrypoints/serve/__init__.py": "",
    "vllm/entrypoints/serve/utils/__init__.py": "",
    "vllm/entrypoints/serve/utils/api_utils.py": """
def log_version_and_model(logger, version, model):
    logger.info("vLLM API server version %%s, model %%s", version, model)
""",
}


def build_stub_tree(tree: Path, root: Path, version: str) -> list[str]:
    """Assemble the stub ``vllm`` package under ``root`` around the launcher
    files of ``tree``; returns the launcher modules the tree had."""
    launchers = launcher_modules_in_tree(tree)
    if not launchers:
        raise SystemExit(f"{tree}: no launcher file under vllm/entrypoints/ (see LAUNCHER_FILES)")
    values = {"version": version, "mark_engine": MARK_ENGINE}
    for relative, template in _STUB_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(template % values).lstrip("\n"))
    for name, relative in LAUNCHER_FILES:
        if name in launchers:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.parent.name == "launchers":
                (destination.parent / "__init__.py").write_text("")
            shutil.copyfile(tree / relative, destination)
    return launchers


# ---------------------------------------------------------------------------
# The child: one launch, in a fresh interpreter
# ---------------------------------------------------------------------------


class _PlaceholderModule(types.ModuleType):
    """A module of names the smoke does not model (``vllm.outputs``,
    ``torch``, ...): every attribute is the placeholder class."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        from vllm._smoke import Placeholder

        return Placeholder


class _PlaceholderLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = _PlaceholderModule(spec.name)
        module.__path__ = []  # a package: its submodules are placeholders too
        return module

    def exec_module(self, module):
        pass


class _PlaceholderFinder(importlib.abc.MetaPathFinder):
    """Placeholders for what a stub tree lacks: ``vllm.*`` modules the tree
    has no file for, and the heavy third-party packages the servicer modules
    import when they are not installed."""

    def __init__(self, top_level: tuple[str, ...]) -> None:
        self.top_level = top_level

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".", 1)[0]
        if top == "vllm":
            if fullname == "vllm":
                return None
            if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
                return None  # the tree has it
        elif top not in self.top_level:
            return None
        return importlib.machinery.ModuleSpec(fullname, _PlaceholderLoader(), is_package=True)


class _PatchingLoader(importlib.abc.Loader):
    """The module's own loader, plus a patch applied right after its body ran."""

    def __init__(self, inner, patch):
        self.inner = inner
        self.patch = patch

    def create_module(self, spec):
        create = getattr(self.inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.patch(module)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _PatchingFinder(importlib.abc.MetaPathFinder):
    """Patches applied to named modules as they load, so the import order
    stays the launcher's own (nothing is imported ahead of it)."""

    def __init__(self, patches) -> None:
        self.patches = patches

    def find_spec(self, fullname, path=None, target=None):
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and not isinstance(spec.loader, _PatchingLoader):
                    spec.loader = _PatchingLoader(spec.loader, patch)
                return spec
        return None


def _patch_rust(module) -> None:
    async def serve_rust(args):
        facts = {
            "model": getattr(args, "model", None),
            "port": getattr(args, "port", None),
            "servicer_impl": getattr(args, "servicer_impl", None),
        }
        print(f"{MARK_RUST} {json.dumps(facts, sort_keys=True)}", flush=True)
        return EXIT_RUST

    module.serve_rust = serve_rust


def _patch_servicer(module) -> None:
    from smg_grpc_servicer.vllm.rust import require_python_impl

    def __init__(self, async_llm, start_time, *args, **kwargs):
        require_python_impl()  # the real guard: what fires when the switch never ran
        print(f"{MARK_SERVICER} on {type(async_llm).__name__}", flush=True)
        raise SystemExit(EXIT_SERVICER)

    module.VllmEngineServicer.__init__ = __init__


def _patch_installed_engine_args(module) -> None:
    def create_engine_config(self, usage_context=None, **kwargs):
        return types.SimpleNamespace(smoke="stub engine config")

    module.AsyncEngineArgs.create_engine_config = create_engine_config


def _patch_installed_async_llm(module) -> None:
    def from_vllm_config(cls, vllm_config=None, **kwargs):
        print(MARK_ENGINE, flush=True)
        return types.SimpleNamespace(vllm_config=vllm_config, shutdown=lambda: None)

    module.AsyncLLM.from_vllm_config = classmethod(from_vllm_config)


def _stub_uvloop_if_missing() -> None:
    if importlib.util.find_spec("uvloop") is not None:
        return
    import asyncio

    uvloop = types.ModuleType("uvloop")
    uvloop.run = asyncio.run
    sys.modules["uvloop"] = uvloop


def _child(config: dict) -> None:
    for path in reversed(config["paths"]):
        sys.path.insert(0, path)
    patches = {
        "smg_grpc_servicer.vllm.rust": _patch_rust,
        "smg_grpc_servicer.vllm.servicer": _patch_servicer,
    }
    if config["installed"]:
        patches["vllm.engine.arg_utils"] = _patch_installed_engine_args
        patches["vllm.v1.engine.async_llm"] = _patch_installed_async_llm
    else:
        missing = tuple(
            name for name in _PLACEHOLDER_TOP_LEVEL if importlib.util.find_spec(name) is None
        )
        sys.meta_path.insert(0, _PlaceholderFinder(missing))
    sys.meta_path.insert(0, _PatchingFinder(patches))
    _stub_uvloop_if_missing()
    os.environ.pop("SMG_VLLM_SERVICER_IMPL", None)
    os.environ.update(config["env"])
    sys.argv = [config["argv0"], *config["argv"]]
    if config["form"] == "module":
        runpy.run_module(config["launcher"], run_name="__main__", alter_sys=True)
    elif config["installed"]:
        from vllm.entrypoints.cli.main import main

        main()
    else:
        _emulated_vllm_serve(config)


def _emulated_vllm_serve(config: dict) -> None:
    """``vllm serve <model> --grpc`` with the stub tree: the parser the CLI
    builds (``AsyncEngineArgs.add_cli_args`` loads the plugins), its parse,
    then the three lines of the serve command's ``--grpc`` branch, identical
    from 0.28 through 0.31 but for the launcher's name."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(prog="vllm serve")
    parser.add_argument("model_tag", nargs="?")
    parser.add_argument("--grpc", action="store_true")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=50051)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args(config["argv"])
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    if getattr(args, "grpc", False):
        launcher = importlib.import_module(config["launcher"])
        serve_grpc = launcher.serve_grpc
        import uvloop

        uvloop.run(serve_grpc(args))


# ---------------------------------------------------------------------------
# The parent: the matrix, judged
# ---------------------------------------------------------------------------


def _argv(case: Case) -> tuple[str, list[str]]:
    flags = ["--port", "0"]
    if case.selection == "flag":
        flags += ["--servicer-impl", case.impl]
    if case.form == "module":
        return f"python -m {case.launcher}", ["--model", MODEL, *flags]
    return "vllm", ["serve", MODEL, "--grpc", *flags]


def run_case(case: Case, *, paths: list[str], installed: bool, timeout: float) -> Result:
    env = {"SMG_VLLM_SERVICER_IMPL": case.impl} if case.selection == "env" else {}
    argv0, argv = _argv(case)
    if case.form == "cli" and not installed:
        argv0 = "vllm serve"
        argv = argv[1:]  # the subcommand's own arguments
    config = {
        "paths": paths,
        "installed": installed,
        "form": case.form,
        "launcher": case.launcher,
        "env": env,
        "argv0": argv0,
        "argv": argv,
    }
    proc = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--child", json.dumps(config)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **env},
    )
    return _judge(case, proc)


def _judge(case: Case, proc: subprocess.CompletedProcess) -> Result:
    out = proc.stdout
    rust, engine, servicer = MARK_RUST in out, MARK_ENGINE in out, MARK_SERVICER in out
    if case.impl == "rust":
        expected = proc.returncode == EXIT_RUST and rust and not engine and not servicer
        detail = (
            "serve_rust reached before any engine, exit code the servicer's"
            if expected
            else f"exit {proc.returncode}; serve_rust={rust} engine={engine} servicer={servicer}"
        )
    else:
        expected = proc.returncode == EXIT_SERVICER and engine and servicer and not rust
        detail = (
            "engine built, Python servicer constructed on it"
            if expected
            else f"exit {proc.returncode}; engine={engine} servicer={servicer} serve_rust={rust}"
        )
    return Result(case, expected, detail, proc.returncode, proc.stdout, proc.stderr)


def run_matrix(
    *,
    launcher_tree: Path | None,
    servicer_root: Path | None,
    proto_path: str | None,
    version: str,
    selections: tuple[str, ...],
    timeout: float,
) -> list[Result]:
    """Every case for the launcher source given; the stub tree, when one is
    built, lives for the run."""
    paths = []
    if servicer_root is not None:
        paths.append(str(servicer_root))
    if proto_path:
        paths.append(proto_path)
    if launcher_tree is None:
        launchers = launcher_modules_installed()
        if not launchers:
            raise SystemExit("no vLLM gRPC launcher module is importable here (is vLLM installed?)")
        cases = cases_for(launchers, selections)
        return [run_case(case, paths=paths, installed=True, timeout=timeout) for case in cases]
    with tempfile.TemporaryDirectory(prefix="launcher-switch-smoke-") as stub_dir:
        root = Path(stub_dir) / "site"
        launchers = build_stub_tree(launcher_tree, root, version)
        cases = cases_for(launchers, selections)
        return [
            run_case(case, paths=[str(root), *paths], installed=False, timeout=timeout)
            for case in cases
        ]


def format_table(results: list[Result]) -> str:
    width = max(len(result.case.label) for result in results)
    lines = [f"{'case'.ljust(width)}  impl    result  detail", "-" * (width + 40)]
    for result in results:
        lines.append(
            f"{result.case.label.ljust(width)}  {result.case.impl:<7} "
            f"{'PASS' if result.passed else 'FAIL':<7} {result.detail}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--child"]:  # one launch, in the interpreter the parent started
        _child(json.loads(argv[1]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--launcher-tree",
        type=Path,
        help="a vllm/ tree holding the launcher files (the rest of vLLM is stubbed)",
    )
    source.add_argument(
        "--installed", action="store_true", help="the installed vLLM, with the engine stubbed"
    )
    parser.add_argument(
        "--servicer-root",
        type=Path,
        help="the checkout's grpc_servicer/ directory to test (default: the installed package)",
    )
    parser.add_argument(
        "--proto-path",
        default=os.environ.get("SMG_GRPC_PROTO_PATH"),
        help="generated smg_grpc_proto stubs to put on the path ($SMG_GRPC_PROTO_PATH)",
    )
    parser.add_argument("--version", default="0.0.0", help="the __version__ the stub vllm reports")
    parser.add_argument(
        "--selection",
        action="append",
        choices=("env", "flag"),
        help="how Rust is selected (repeatable; default: env and flag)",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per case")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="print each failing child's output"
    )
    args = parser.parse_args(argv)
    results = run_matrix(
        launcher_tree=args.launcher_tree,
        servicer_root=args.servicer_root,
        proto_path=args.proto_path,
        version=args.version,
        selections=tuple(args.selection or ("env", "flag")),
        timeout=args.timeout,
    )
    print(format_table(results))
    failed = [result for result in results if not result.passed]
    if args.verbose:
        for result in failed:
            print(f"\n--- {result.case.label} ({result.case.impl}) stdout:\n{result.stdout}")
            print(f"--- stderr:\n{result.stderr[-4000:]}")
    print(f"\n{len(results) - len(failed)} of {len(results)} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
