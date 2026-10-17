"""The servicer switch, installed into vLLM's gRPC launcher by the launcher's
own import of this package.

vLLM's launcher (one of :data:`LAUNCHERS`, behind ``vllm serve --grpc`` and
``python -m`` on the launcher module) imports this package's servicer
classes at module level, then defines ``serve_grpc``, which builds an
AsyncLLM before it instantiates them. The Rust path has to take the process
before that engine exists, so the switch must run ahead of ``serve_grpc``;
the launcher's import of this package is the first moment this package runs
inside it, and :func:`install_launcher_switch` uses that moment.

The launcher module is mid-import then (``serve_grpc`` not yet defined), so
its class is swapped for a ``ModuleType`` subclass whose first attribute
access after the import (the CLI's ``from ... import serve_grpc``, the
deprecated shim's ``import main, serve_grpc``) rebinds ``serve_grpc`` in the
module dict to :func:`with_servicer_switch`'s wrapper: the three lines an
in-source hook would carry, with the launcher's own ``args``. The launcher's
``main()`` reads the rebound name from its globals, so every documented entry
runs the switch. A launcher imported after this package (something imported
the servicer first, or a launcher that imports it lazily) is caught by a
meta-path finder that binds the switch as the launcher loads.

What the switch binds is specific to the launcher's shape: the module under
one of :data:`LAUNCHER_MODULES` (imported under its name, or run as
``__main__`` by ``python -m``), defining :data:`HOOK_ATTRIBUTE`. A launcher
of another shape (a release that moved the module or renamed the
coroutine) is never bound. Rather than let such a launcher build the
engine and fail in the Python servicer's guard once the model has loaded,
:func:`require_launcher_hook` checks the hook as soon as the Rust
implementation is selected, from the launcher's own argument parse (the
plugin's parse step, :mod:`smg_grpc_servicer.vllm.plugin`), and refuses with
one line naming the installed vLLM and the launchers the switch binds.

``python -m`` on the launcher module itself runs it as ``__main__``, where it
is never in ``sys.modules`` under its name: on 0.28 and 0.29 that is what
``python -m vllm.entrypoints.grpc_server`` does (from 0.30 the same command
runs a shim whose import of the launcher the finder catches). That launcher
is bound in its ``__main__`` namespace instead, from the two moments this
package runs inside its ``main()`` after the body defined ``serve_grpc``:
vLLM loads this package's plugin while the launcher builds its parser
(``AsyncEngineArgs.add_cli_args``), and the plugin's parse step runs on the
launcher's own parse; both call :func:`install_launcher_switch`, and
``main()`` reads the rebound name right after. A launcher run as a file path
(no module spec), or with the plugin left out of ``VLLM_PLUGINS``, is the one
form nothing reaches; there the Python servicer refuses the flag loudly
(:func:`smg_grpc_servicer.vllm.rust.require_python_impl`).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.abc
import importlib.metadata
import re
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

from smg_grpc_servicer.vllm import SERVICER_IMPL_ENV


@dataclasses.dataclass(frozen=True)
class LauncherModule:
    """One layout of vLLM's gRPC launcher: the module that defines the
    coroutine the switch binds (and runs ``main()`` under ``python -m``), and
    the releases that ship it there."""

    name: str
    """The module's dotted name."""
    releases: str
    """The releases whose launcher this is, as the docs name them."""
    floor: tuple[int, int]
    """The first release (major, minor) with this layout."""


# The launchers the switch binds, newest layout first. From 0.30 the old
# name is a shim that imports ``main`` and ``serve_grpc`` from the new module,
# so the deprecated ``python -m vllm.entrypoints.grpc_server`` keeps working
# there (the shim is ``__main__``, the import it does is bound as it loads). A
# main-branch build between two releases carries the next release's layout
# under the earlier release's version, so the switch binds by what is loaded,
# never by the version; the releases here are for the docs and the messages
# (:func:`launcher_for_release`).
LAUNCHERS = (
    LauncherModule("vllm.entrypoints.launchers.grpc_server", "0.30 and later", (0, 30)),
    LauncherModule("vllm.entrypoints.grpc_server", "0.28 and 0.29", (0, 28)),
)
LAUNCHER_MODULES = tuple(launcher.name for launcher in LAUNCHERS)
# The launcher's coroutine the switch binds over.
HOOK_ATTRIBUTE = "serve_grpc"
# The releases the binding is verified on, every layout and entry form, by
# the CPU smoke (``grpc_servicer/scripts/launcher_switch_smoke.py``): the
# table's floor through the newest release checked. The hook points are the
# same in all of them: the servicer classes imported at module level, then
# ``async def serve_grpc(args)``, then a ``main()`` that builds the parser
# with ``AsyncEngineArgs.add_cli_args`` (which loads this package's plugin),
# parses, and runs ``serve_grpc(args)``.
SUPPORTED_VLLM_RELEASES = "0.28 through 0.31"
_SWITCH_MARK = "__smg_servicer_switch__"
_RELEASE = re.compile(r"^\s*v?(\d+)\.(\d+)")


def launcher_for_release(version: str | None) -> LauncherModule | None:
    """The launcher a vLLM *release* of this version ships, by the table:
    the newest layout whose floor the version reaches; ``None`` below the
    table's floor or for a version that does not parse. A main-branch build
    (``0.29.1rc1.devN+g...``) may already carry the next release's layout:
    this is what the release notes say, not what is installed."""
    match = _RELEASE.match(version or "")
    if match is None:
        return None
    release = (int(match.group(1)), int(match.group(2)))
    for launcher in LAUNCHERS:
        if release >= launcher.floor:
            return launcher
    return None


def supported_launchers() -> str:
    """The launchers the switch binds, for a message: each module with its
    releases, and the releases the binding is verified on."""
    layouts = ", ".join(f"{launcher.name} ({launcher.releases})" for launcher in LAUNCHERS)
    return (
        f"the switch binds the {HOOK_ATTRIBUTE} of {layouts}, imported under that name or run as "
        f"__main__ by `python -m` (vLLM {SUPPORTED_VLLM_RELEASES} checked)"
    )


def with_servicer_switch(
    serve_grpc: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """``serve_grpc`` behind the flag: Rust takes the process (and its exit
    code), Python runs the launcher's own coroutine."""
    if getattr(serve_grpc, _SWITCH_MARK, False):
        return serve_grpc

    @functools.wraps(serve_grpc)
    async def serve_grpc_with_switch(args: Any, *rest: Any, **kwargs: Any) -> Any:
        from smg_grpc_servicer.rust_lifecycle import ServicerConfigError, configuration_exit
        from smg_grpc_servicer.vllm.rust import (
            configure_logging,
            require_python_media_mode,
            resolve_servicer_impl,
            serve_rust,
        )

        try:
            if resolve_servicer_impl(args) == "rust":
                raise SystemExit(await serve_rust(args))
            # The Python servicer's media mode is known here, from the flags
            # and the environment: one it cannot serve is refused now, before
            # the launcher's own coroutine builds the engine and loads the
            # model (the servicer's own refusal would come after that load).
            require_python_media_mode(args)
        except ServicerConfigError as error:
            # A knob the servicer cannot use: one named ERROR line and exit
            # code 2, not the launcher's traceback.
            configure_logging()
            raise configuration_exit(error) from None
        return await serve_grpc(args, *rest, **kwargs)

    setattr(serve_grpc_with_switch, _SWITCH_MARK, True)
    return serve_grpc_with_switch


def _rebind(namespace: dict[str, Any]) -> None:
    serve_grpc = namespace.get(HOOK_ATTRIBUTE)
    if callable(serve_grpc) and not getattr(serve_grpc, _SWITCH_MARK, False):
        namespace[HOOK_ATTRIBUTE] = with_servicer_switch(serve_grpc)


def _bound(namespace: dict[str, Any]) -> bool:
    return bool(getattr(namespace.get(HOOK_ATTRIBUTE), _SWITCH_MARK, False))


class _SwitchedLauncher(types.ModuleType):
    """The launcher module once hooked: any attribute access binds the switch
    over a ``serve_grpc`` the module body defined since."""

    def __getattribute__(self, name: str) -> Any:
        if name != "__dict__":
            _rebind(object.__getattribute__(self, "__dict__"))
        return super().__getattribute__(name)


class _SwitchingLoader(importlib.abc.Loader):
    """The launcher's own loader, plus the switch bound right after the module
    body ran (so the name is rebound before anything can call it)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: types.ModuleType) -> None:
        self.inner.exec_module(module)
        _rebind(vars(module))

    def __getattr__(self, name: str) -> Any:  # get_code, get_source, is_package, ...
        return getattr(self.inner, name)


class _LauncherFinder(importlib.abc.MetaPathFinder):
    """For a launcher imported after this package: the real spec from the
    other finders, with the loader wrapped in :class:`_SwitchingLoader`."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in LAUNCHER_MODULES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and not isinstance(spec.loader, _SwitchingLoader):
                    spec.loader = _SwitchingLoader(spec.loader)
                return spec
        return None


def launcher_main() -> Any | None:
    """The launcher when it is this process's ``__main__``: ``python -m`` on
    the launcher module itself, which runpy names in ``__main__.__spec__``
    (the real module on 0.28 and 0.29, where the deprecated command is not a
    shim; the 0.30+ module run by its own name). ``None`` for any other
    ``__main__``: the ``vllm`` console script, the shim, a file path."""
    main = sys.modules.get("__main__")
    name = getattr(getattr(main, "__spec__", None), "name", None)
    return main if name in LAUNCHER_MODULES else None


def install_launcher_switch() -> list[str]:
    """Hook every launcher module that is importing or imported, and arrange
    for one imported later (this package imported first, or a launcher that
    imports it lazily) to be hooked as it loads; bind a launcher running as
    ``__main__`` in its namespace once its body defined ``serve_grpc``.
    Idempotent; returns the names hooked now (``"__main__"`` for the
    launcher run by ``python -m``). A module another extension already
    replaced with its own class is left alone."""
    if not any(isinstance(finder, _LauncherFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LauncherFinder())
    hooked: list[str] = []
    for name in LAUNCHER_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if type(module) is _SwitchedLauncher:
            hooked.append(name)
            continue
        if type(module) is not types.ModuleType:
            continue
        _rebind(vars(module))
        module.__class__ = _SwitchedLauncher
        hooked.append(name)
    # The launcher as __main__: its main() reads serve_grpc from this dict
    # (not through the module object), so the dict is rebound, from the calls
    # that run inside main(): the plugin's load and its parse step. Mid-body
    # (the servicer import) there is nothing to bind yet; nothing is left
    # hooked, the next call binds.
    main = launcher_main()
    if main is not None:
        namespace = vars(main)
        _rebind(namespace)
        if _bound(namespace):
            hooked.append("__main__")
    return hooked


# ---------------------------------------------------------------------------
# The hook, verified before the engine is built
# ---------------------------------------------------------------------------

# What the switch finds of the launcher (`LauncherHook.state`).
BOUND = "bound"  # the switch is bound over the launcher's serve_grpc
MISSING = "missing"  # a launcher module is loaded and defines no serve_grpc
ABSENT = "absent"  # no launcher module is loaded, or importable under the known names
PENDING = "pending"  # a launcher module is still importing (it is importing this package)
UNKNOWN = "unknown"  # the launcher's import failed on something its own import reports
SOURCE = "source"  # the launcher carries the check in its own source (no binding needed)


@dataclasses.dataclass(frozen=True)
class LauncherHook:
    """What the switch found of the launcher: the ``state`` (one of the
    constants above) and the module it describes, when there is one."""

    state: str
    module: str | None = None

    @property
    def bindable(self) -> bool:
        """Whether the switch binds, or may still bind, on this launcher: the
        hook is bound, the launcher is mid-import (its ``serve_grpc`` comes
        after this package's import, and the hooked module binds it), its
        import failed on something its own import reports, or the launcher
        carries the check in its own source and needs no binding."""
        return self.state in (BOUND, PENDING, UNKNOWN, SOURCE)


def launcher_hook() -> LauncherHook:
    """What the switch finds of the launcher in this process: the known
    modules in ``sys.modules`` (bound, loaded without the hook, or still
    importing), else a launcher running as ``__main__`` (bound in its own
    namespace once its body defined the coroutine; this runs inside its
    ``main()``, after the body, so a namespace without it has none to bind),
    else nothing loaded."""
    missing: str | None = None
    pending = False
    for name in LAUNCHER_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if getattr(getattr(module, "__spec__", None), "_initializing", False):
            pending = True
            continue
        namespace = vars(module)
        _rebind(namespace)
        hook = namespace.get(HOOK_ATTRIBUTE)
        if callable(hook) and getattr(hook, _SWITCH_MARK, False):
            return LauncherHook(BOUND, name)
        missing = missing or name
    if missing is not None:
        return LauncherHook(MISSING, missing)
    if pending:
        return LauncherHook(PENDING)
    main = launcher_main()
    if main is not None:
        namespace = vars(main)
        _rebind(namespace)
        state = BOUND if _bound(namespace) else MISSING
        return LauncherHook(state, main.__spec__.name)
    return LauncherHook(ABSENT)


def _import_launcher() -> LauncherHook:
    """Import the launcher under the known names the installed vLLM has, newest
    first (what ``vllm serve --grpc`` does next; the finder binds the switch as
    it loads) and report what the switch finds of it. A name the installation
    lacks is skipped; a name whose module defines no hook is passed over for
    the next known name (a tree may answer the newer name with a module that
    is not the launcher) and judged only when no other name binds; an import
    that fails on anything else is left to the launcher's own import to
    report."""
    found: LauncherHook | None = None
    for name in LAUNCHER_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            absent = error.name or ""
            if absent and (name == absent or name.startswith(absent + ".")):
                continue
            # An earlier name's verdict stands: the CLI imports that one, and
            # a later name's failure (a shim importing the unbound newer
            # module) must not pass the launcher the check would refuse.
            return found or LauncherHook(UNKNOWN, name)
        except Exception:  # noqa: BLE001 - the launcher's own import reports its failure
            return found or LauncherHook(UNKNOWN, name)
        hook = launcher_hook()
        if hook.bindable:
            return hook
        found = hook
    return found or LauncherHook(ABSENT)


def require_launcher_hook(import_launcher: bool = True) -> LauncherHook:
    """The hook as the switch finds it, once the Rust implementation is
    selected: bound (or still binding), or a :class:`ServicerConfigError`
    naming what the installed vLLM lacks and the launchers the switch binds.
    A launcher not loaded yet is imported first under its known names when
    ``import_launcher`` is set (``vllm serve --grpc`` imports it right after
    its parse); a launcher running as ``__main__`` is never re-imported."""
    from smg_grpc_servicer.rust_lifecycle import ServicerConfigError

    hook = launcher_hook()
    if hook.state == ABSENT and import_launcher:
        hook = _import_launcher()
    if hook.bindable:
        return hook
    if _carries_the_check(hook):
        # The documented alternative to the binding: the launcher's own
        # source consults this package's flag, as the preflight
        # (``rust.upstream_hook_installed``) accepts it; the two agree.
        return LauncherHook(SOURCE, hook.module)
    raise ServicerConfigError(refusal_line(hook))


def _carries_the_check(hook: LauncherHook) -> bool:
    """Whether the launcher carries the check in its own source
    (``rust.HOOK_SYMBOL``): the file of a loaded launcher without the hook
    (the named module, or the one running as ``__main__``), or any launcher
    under ``entrypoints/`` of the installed vLLM when none is loaded or
    importable under the known names, the preflight's own scan."""
    from smg_grpc_servicer.vllm import rust

    if hook.state == MISSING and hook.module:
        module = sys.modules.get(hook.module)
        main = launcher_main()
        if module is None and main is not None and main.__spec__.name == hook.module:
            module = main
        path = getattr(module, "__file__", None)
        source = rust._read_source(path) if path else None
        return source is not None and rust.HOOK_SYMBOL in source
    if hook.state == ABSENT:
        package = sys.modules.get("vllm")
        roots = list(getattr(package, "__path__", []) or [])
        return any(rust.upstream_hook_installed(root) for root in roots)
    return False


def refusal_line(hook: LauncherHook) -> str:
    """The one line a launcher the switch cannot bind is refused with: the
    installed vLLM and what was found of its launcher, the launchers the
    switch binds, and the way out."""
    version = installed_engine_version()
    installed = f"vLLM {version}" if version else "vLLM (version unknown)"
    way_out = "install a vLLM release the switch binds"
    if hook.state == MISSING:
        found = f"its gRPC launcher {hook.module} defines no {HOOK_ATTRIBUTE}"
    else:
        found = f"it has no gRPC launcher module under {' or '.join(LAUNCHER_MODULES)}"
    return (
        f"--servicer-impl rust ({SERVICER_IMPL_ENV}=rust) refused: the servicer switch cannot "
        f"bind on the installed {installed}: {found}; {supported_launchers()}; {way_out}, or "
        f"select --servicer-impl python (or unset {SERVICER_IMPL_ENV}) for the Python servicer"
    )


def installed_engine_version() -> str | None:
    """The installed vLLM's version: the imported package's, else the
    distribution's; ``None`` when neither is known."""
    version = getattr(sys.modules.get("vllm"), "__version__", None)
    if not version:
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            version = None
    return str(version) if version else None
