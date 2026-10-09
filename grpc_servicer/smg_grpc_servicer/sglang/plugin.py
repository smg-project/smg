"""smg's SGLang plugin: ``--servicer-impl`` on SGLang's server parser,
defined by this package.

SGLang's launchers (``sglang serve``, ``python -m sglang.launch_server``)
load the ``sglang.srt.plugins`` entry points before they parse the command
line, and a plugin may wrap any function by its dotted path. Which
implementation serves the gRPC contract is this package's decision, so the
flag is this package's too: :func:`register` hooks ``ServerArgs.add_cli_args``
so every parser SGLang builds from it grows ``--servicer-impl {python,rust}``
as it is built, and :func:`smg_grpc_servicer.sglang.rust.resolve_servicer_impl`
reads the parsed value ahead of the environment. ``ServerArgs`` keeps only
its own fields, so the value travels as the launcher's flag in this process
(:func:`parsed_flag`) and in ``SMG_SGLANG_SERVICER_IMPL``, which the flag
overrides. A ``SGLANG_PLUGINS`` setting that leaves this plugin out only
removes the flag; the environment variable works regardless.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

# Loaded in every SGLang process: nothing heavier than the package init here.
from smg_grpc_servicer.sglang import SERVICER_IMPL_ENV

SERVICER_IMPL_FLAG = "--servicer-impl"
SERVICER_IMPL_CHOICES = ("python", "rust")
# SGLang's parser builder; the flag joins each parser after it ran.
ADD_CLI_ARGS = "sglang.srt.server_args.ServerArgs.add_cli_args"


class _State:
    """Per-process plugin state: the value the flag parsed, if it did."""

    def __init__(self) -> None:
        self.parsed: str | None = None


_state = _State()


def parsed_flag() -> str | None:
    """The ``--servicer-impl`` value parsed in this process, or ``None``."""
    return _state.parsed


class _ServicerImplAction(argparse._StoreAction):  # noqa: SLF001 — SGLang's --config merger admits store actions only
    """A store action that also keeps the choice as this process's launcher
    flag and in the environment: SGLang's ``ServerArgs`` keeps only its own
    fields, so the namespace value does not survive the parse, and the
    environment is what the headless scheduler child inherits. It derives
    from argparse's store action because SGLang's ``--config`` merger takes
    only store and store_true options from the YAML file."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        super().__call__(parser, namespace, values, option_string)
        _state.parsed = values
        os.environ[SERVICER_IMPL_ENV] = values


def add_servicer_impl_argument(parser: argparse.ArgumentParser) -> bool:
    """Add ``--servicer-impl`` to a parser that lacks it; returns whether it
    was added. A new parser is a new parse (SGLang builds one per
    ``prepare_server_args``): the choice an earlier parser in this process
    remembered is dropped, so a parse without the flag falls through to the
    environment."""
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if SERVICER_IMPL_FLAG in actions:
        return False
    _state.parsed = None
    parser.add_argument(
        SERVICER_IMPL_FLAG,
        dest="servicer_impl",
        action=_ServicerImplAction,
        choices=list(SERVICER_IMPL_CHOICES),
        default=None,
        help=(
            "Which implementation serves the gRPC contract under --grpc-mode: the Python "
            "servicer (default) or the Rust one (smg.servicer.SglangGrpcServer, with the "
            f"scheduler headless). Unset falls back to ${SERVICER_IMPL_ENV}."
        ),
    )
    return True


def after_add_cli_args(result: Any, *args: Any, **kwargs: Any) -> None:
    """AFTER ``ServerArgs.add_cli_args``: the flag joins the parser SGLang
    just filled, so it is there for ``--help`` and for the parse. Leaves the
    hooked method's result alone."""
    for candidate in (*args, *kwargs.values()):
        if isinstance(candidate, argparse.ArgumentParser):
            add_servicer_impl_argument(candidate)
            break
    return None


def register() -> None:
    """The ``sglang.srt.plugins`` entry point: one hook on the parser
    builder, nothing else."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(ADD_CLI_ARGS, after_add_cli_args, HookType.AFTER)
