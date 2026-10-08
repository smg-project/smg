"""smg's vLLM general plugin: ``--servicer-impl`` on vLLM's gRPC server
parser, defined by this package.

vLLM loads ``vllm.general_plugins`` entry points while it builds the engine
CLI (``AsyncEngineArgs.add_cli_args``), for plugins that extend the parser.
Which implementation serves the gRPC contract is this package's decision, so
the flag is this package's too: :func:`register` makes every
``FlexibleArgumentParser`` that defines ``--grpc`` (the ``vllm serve``
parser) grow ``--servicer-impl {python,rust}`` as it parses, and
:func:`smg_grpc_servicer.vllm.rust.resolve_servicer_impl` reads it from the
parsed namespace ahead of the environment. Parsers without ``--grpc`` are
untouched. A ``VLLM_PLUGINS`` setting that leaves this plugin out only
removes the flag; ``SMG_VLLM_SERVICER_IMPL`` works regardless.
"""

from __future__ import annotations

import argparse
import functools
import os
from typing import Any

# Loaded in every vLLM process: nothing heavier than the package init here.
from smg_grpc_servicer.vllm import SERVICER_IMPL_ENV
from smg_grpc_servicer.vllm.launcher_switch import install_launcher_switch

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
        parsed, extras = original(self, args, namespace)
        impl = getattr(parsed, "servicer_impl", None)
        if impl:
            # The Python servicer's guard reads only the environment: carry the
            # flag there, so a switch that never bound still fails loudly.
            os.environ[SERVICER_IMPL_ENV] = impl
        return parsed, extras

    FlexibleArgumentParser.parse_known_args = parse_known_args  # type: ignore[method-assign]
    setattr(FlexibleArgumentParser, _FLAG_MARK, True)
    # The launcher is imported after the parse; the finder binds the switch as it loads.
    install_launcher_switch()
