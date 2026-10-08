"""The servicer switch, installed into vLLM's gRPC launcher by the launcher's
own import of this package.

vLLM's launcher (``vllm/entrypoints/launchers/grpc_server.py``, behind
``vllm serve --grpc`` and ``python -m vllm.entrypoints.grpc_server``) imports
this package's servicer classes at module level, then defines ``serve_grpc``,
which builds an AsyncLLM before it instantiates them. The Rust path has to
take the process before that engine exists, so the switch must run ahead of
``serve_grpc``; the launcher's import of this package is the first moment this
package runs inside it, and :func:`install_launcher_switch` uses that moment.

The launcher module is mid-import then (``serve_grpc`` not yet defined), so
its class is swapped for a ``ModuleType`` subclass whose first attribute
access after the import (the CLI's ``from ... import serve_grpc``, the
deprecated shim's ``import main, serve_grpc``) rebinds ``serve_grpc`` in the
module dict to :func:`with_servicer_switch`'s wrapper: the three lines an
in-source hook would carry, with the launcher's own ``args``. The launcher's
``main()`` reads the rebound name from its globals, so every documented entry
runs the switch. A launcher imported after this package (something imported
the servicer first, or a launcher that imports it lazily) is caught by a
meta-path finder that binds the switch as the launcher loads. A launcher file
executed directly as ``__main__`` is the one form neither reaches; there the
Python servicer refuses the flag loudly
(:func:`smg_grpc_servicer.vllm.rust.require_python_impl`).
"""

from __future__ import annotations

import functools
import importlib.abc
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

LAUNCHER_MODULES = (
    "vllm.entrypoints.launchers.grpc_server",
    "vllm.entrypoints.grpc_server",
)
_SWITCH_MARK = "__smg_servicer_switch__"


def with_servicer_switch(
    serve_grpc: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """``serve_grpc`` behind the flag: Rust takes the process (and its exit
    code), Python runs the launcher's own coroutine."""
    if getattr(serve_grpc, _SWITCH_MARK, False):
        return serve_grpc

    @functools.wraps(serve_grpc)
    async def serve_grpc_with_switch(args: Any, *rest: Any, **kwargs: Any) -> Any:
        from smg_grpc_servicer.vllm.rust import resolve_servicer_impl, serve_rust

        if resolve_servicer_impl(args) == "rust":
            raise SystemExit(await serve_rust(args))
        return await serve_grpc(args, *rest, **kwargs)

    setattr(serve_grpc_with_switch, _SWITCH_MARK, True)
    return serve_grpc_with_switch


def _rebind(namespace: dict[str, Any]) -> None:
    serve_grpc = namespace.get("serve_grpc")
    if callable(serve_grpc) and not getattr(serve_grpc, _SWITCH_MARK, False):
        namespace["serve_grpc"] = with_servicer_switch(serve_grpc)


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


def install_launcher_switch() -> list[str]:
    """Hook every launcher module that is importing or imported, and arrange
    for one imported later (this package imported first, or a launcher that
    imports it lazily) to be hooked as it loads. Idempotent; returns the names
    hooked now. A module another extension already replaced with its own
    class is left alone."""
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
    return hooked
