"""smg's vLLM general plugin: ``--servicer-impl`` and the ``--mm-*`` flags on
vLLM's gRPC server parsers, defined by this package.

vLLM loads ``vllm.general_plugins`` entry points while it builds the engine
CLI (``AsyncEngineArgs.add_cli_args``), for plugins that extend the parser.
Which implementation serves the gRPC contract is this package's decision, so
the flag is this package's too: :func:`register` makes every
``FlexibleArgumentParser`` that defines ``--grpc`` (the ``vllm serve``
parser) grow ``--servicer-impl {python,rust}`` as it parses, and
:func:`smg_grpc_servicer.vllm.rust.resolve_servicer_impl` reads it from the
parsed namespace ahead of the environment. The worker-side media settings
(``--mm-processor`` and the other flags :class:`MmSettings.from_args` reads)
join every parser that serves gRPC and lacks them: the ``vllm serve`` parser,
and the stock ``python -m vllm.entrypoints.grpc_server`` launcher's, which
defines none of its own (its process's ``__main__`` is the launcher module).
Their parsed values are carried into the environment as well, for a launcher
that builds the servicer without its namespace. Parsers of other commands
are untouched. A ``VLLM_PLUGINS`` setting that leaves this plugin out only
removes the flags; the environment variables work regardless.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from typing import Any

# Loaded in every vLLM process: nothing heavier than the package init here.
from smg_grpc_servicer.vllm import SERVICER_IMPL_ENV
from smg_grpc_servicer.vllm.launcher_switch import LAUNCHER_MODULES, install_launcher_switch

SERVICER_IMPL_FLAG = "--servicer-impl"
SERVICER_IMPL_CHOICES = ("python", "rust")
_FLAG_MARK = "_smg_servicer_impl_flag"


def add_servicer_impl_argument(parser: argparse.ArgumentParser) -> bool:
    """Add ``--servicer-impl`` to a parser that serves gRPC (defines
    ``--grpc``) and lacks it; returns whether it was added."""
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if "--grpc" not in actions or SERVICER_IMPL_FLAG in actions:
        return False
    parser.add_argument(
        SERVICER_IMPL_FLAG,
        dest="servicer_impl",
        choices=list(SERVICER_IMPL_CHOICES),
        default=None,
        help=(
            "Which implementation serves the gRPC contract under --grpc: the Python "
            "servicer (default) or the Rust one (smg.servicer.VllmGrpcServer, with the "
            f"engine headless). Unset falls back to ${SERVICER_IMPL_ENV}."
        ),
    )
    return True


def serves_grpc(parser: argparse.ArgumentParser) -> bool:
    """Whether a parser is a gRPC launcher's: it defines ``--grpc`` (``vllm
    serve``), or it is the stock gRPC launcher's own parser, which has no
    ``--grpc`` because gRPC is all it serves: the process's ``__main__`` is
    the launcher module, and the parser carries the listener's ``--port``."""
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if "--grpc" in actions:
        return True
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(main_spec, "name", None) in LAUNCHER_MODULES and "--port" in actions


def add_mm_arguments_to_grpc_parser(parser: argparse.ArgumentParser) -> list[str]:
    """Add the ``--mm-*`` flags to a gRPC launcher's parser that lacks them;
    returns the flags added (none for other parsers)."""
    if not serves_grpc(parser):
        return []
    from smg_grpc_servicer.vllm.mm_processor import add_mm_arguments

    return add_mm_arguments(parser)


def handoff_mm_flags(parser: argparse.ArgumentParser, parsed: Any) -> Any | None:
    """After a parse by a parser that serves gRPC: its `--mm-*` values go to
    the servicer this process builds and into the environment, whether this
    parse defined the flags or the parser had them already (a `vllm serve`
    with its own; a second parse of the same parser). Returns the settings
    kept, None for a parser of another command."""
    if not serves_grpc(parser):
        return None
    from smg_grpc_servicer.vllm.mm_processor import carry_mm_flags

    return carry_mm_flags(parsed)


def register() -> None:
    """The ``vllm.general_plugins`` entry point. Idempotent, and loaded in
    every vLLM process: it wraps the parser class's parse step and installs
    the launcher switch's import hook, nothing else."""
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if getattr(FlexibleArgumentParser, _FLAG_MARK, False):
        return
    original = FlexibleArgumentParser.parse_known_args

    @functools.wraps(original)
    def parse_known_args(self: Any, args: Any = None, namespace: Any = None) -> Any:
        # The option joins right before parsing, after the launcher finished
        # building the parser, so it is there for `--help` and for the
        # subcommand parse `vllm serve` dispatches to.
        add_servicer_impl_argument(self)
        add_mm_arguments_to_grpc_parser(self)
        parsed, extras = original(self, args, namespace)
        impl = getattr(parsed, "servicer_impl", None)
        if impl:
            # The Python servicer's guard reads only the environment: carry the
            # flag there, so a switch that never bound still fails loudly.
            os.environ[SERVICER_IMPL_ENV] = impl
        # Upstream's launcher builds the Python servicer without its
        # namespace: a gRPC parser's parsed settings are kept for it in this
        # process (so a flag stays `source=flag`), and the set values go into
        # the environment too; on every gRPC parse, the flags' origin aside.
        handoff_mm_flags(self, parsed)
        return parsed, extras

    FlexibleArgumentParser.parse_known_args = parse_known_args  # type: ignore[method-assign]
    setattr(FlexibleArgumentParser, _FLAG_MARK, True)
    # The launcher is imported after the parse; the finder binds the switch as it loads.
    install_launcher_switch()
