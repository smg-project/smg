"""smg's vLLM general plugin: ``--servicer-impl`` and the ``--mm-*`` flags on
vLLM's gRPC server parsers, defined by this package.

vLLM loads ``vllm.general_plugins`` entry points while it builds the engine
CLI (``AsyncEngineArgs.add_cli_args``), for plugins that extend the parser.
Which implementation serves the gRPC contract is this package's decision, so
the flag is this package's too: :func:`register` makes every
``FlexibleArgumentParser`` that serves gRPC (the ``vllm serve`` parser, which
defines ``--grpc``, and the stock gRPC launcher's own) grow
``--servicer-impl {python,rust}`` as it parses, and
:func:`smg_grpc_servicer.vllm.rust.resolve_servicer_impl` reads it from the
parsed namespace ahead of the environment. The plugin's load and its parse
step are also the two moments inside the launcher's ``main()`` where a
launcher running as ``__main__`` (``python -m`` on the launcher module) is
bound to the servicer switch (:mod:`smg_grpc_servicer.vllm.launcher_switch`).
The worker-side media settings
(``--mm-processor`` and the other flags :class:`MmSettings.from_args` reads)
join every parser that serves gRPC and lacks them: the ``vllm serve`` parser,
and the stock ``python -m vllm.entrypoints.grpc_server`` launcher's, which
defines none of its own (its process's ``__main__`` is the launcher module).
Their parsed values are carried into the environment as well, for a launcher
that builds the servicer without its namespace. Parsers of other commands
are untouched. A ``VLLM_PLUGINS`` setting that leaves this plugin out only
removes the flags; the environment variables work regardless.

The parse step is also where a launch that builds an engine for gRPC gets
its CUDA-graph capture cap derived from the admission step when the operator
set none (:mod:`smg_grpc_servicer.vllm.capture_cap`): the value joins the
parsed namespace the launcher builds the engine from, for both servicer
implementations and for the engine-only ranks of a multi-node engine.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from typing import Any

# Loaded in every vLLM process: nothing heavier than the package init here.
from smg_grpc_servicer.vllm import SERVICER_IMPL_ENV
from smg_grpc_servicer.vllm.launcher_switch import (
    LAUNCHER_MODULES,
    SUPPORTED_VLLM_RELEASES,
    install_launcher_switch,
)

SERVICER_IMPL_FLAG = "--servicer-impl"
SERVICER_IMPL_CHOICES = ("python", "rust")
_FLAG_MARK = "_smg_servicer_impl_flag"


def add_servicer_impl_argument(parser: argparse.ArgumentParser) -> bool:
    """Add ``--servicer-impl`` to a parser that serves gRPC (``vllm serve``'s,
    which defines ``--grpc``, or the stock gRPC launcher's own, see
    :func:`serves_grpc`) and lacks it; returns whether it was added."""
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if SERVICER_IMPL_FLAG in actions or not serves_grpc(parser):
        return False
    parser.add_argument(
        SERVICER_IMPL_FLAG,
        dest="servicer_impl",
        choices=list(SERVICER_IMPL_CHOICES),
        default=None,
        help=(
            "Which implementation serves the gRPC contract under --grpc: the Python "
            "servicer (default) or the Rust one (smg.servicer.VllmGrpcServer, with the "
            f"engine headless). Unset falls back to ${SERVICER_IMPL_ENV}. The Rust one needs "
            "a launcher the switch binds (`vllm serve <model> --grpc`, or `python -m` on the "
            f"launcher module; vLLM {SUPPORTED_VLLM_RELEASES} checked) or one that carries "
            "the check in its own source; any other launcher is refused at this parse, "
            "before the engine is built."
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


def launches_engine(parser: argparse.ArgumentParser, parsed: Any) -> bool:
    """Whether a parse builds an engine this package serves: the stock gRPC
    launcher's (gRPC is all it serves), ``vllm serve --grpc``, and ``vllm
    serve --headless`` (the engine-only ranks of a multi-node engine, started
    with the leader's flags). An HTTP ``vllm serve`` is not this package's
    launch and keeps the engine's own defaults."""
    if not serves_grpc(parser):
        return False
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if "--grpc" not in actions:
        return True
    return bool(getattr(parsed, "grpc", False) or getattr(parsed, "headless", False))


def handoff_capture_cap(parser: argparse.ArgumentParser, parsed: Any) -> Any | None:
    """After a parse that launches an engine for gRPC: the CUDA-graph capture
    cap derived from the admission step joins the namespace where the
    operator set none, and one line says what applies. A knob value the
    derivation does not take is a usage error (exit code 2, the variable and
    the value named), as a bad flag value would be. Returns the decision,
    ``None`` for any other parse."""
    if not launches_engine(parser, parsed):
        return None
    from smg_grpc_servicer.vllm.capture_cap import apply_capture_cap, log_capture_cap

    try:
        cap = apply_capture_cap(parsed)
    except ValueError as error:
        parser.error(str(error))
    log_capture_cap(cap)
    return cap


def check_servicer_impl(parser: argparse.ArgumentParser, parsed: Any) -> str | None:
    """After a parse that serves gRPC (a gRPC launcher's parser, with
    ``--grpc`` given where the parser has the flag): the implementation the
    parse selected (the flag, else the environment), with a Rust selection
    verified against the launcher here, in the launcher's own argument parse
    and before it builds the engine: a value the switch cannot use, or a
    launcher the switch cannot bind (``launcher_switch.require_launcher_hook``),
    is one ERROR line and exit code 2 now, not a Python servicer that refuses
    once the model has loaded. ``None`` for a parse that serves no gRPC
    (another command's, ``vllm serve`` without ``--grpc``), which selects
    nothing."""
    if not serves_grpc(parser) or getattr(parsed, "grpc", None) is False:
        return None
    from smg_grpc_servicer.rust_lifecycle import ServicerConfigError, configuration_exit
    from smg_grpc_servicer.vllm.launcher_switch import require_launcher_hook
    from smg_grpc_servicer.vllm.rust import configure_logging, resolve_servicer_impl

    try:
        impl = resolve_servicer_impl(parsed)
        if impl == "rust":
            require_launcher_hook()
    except ServicerConfigError as error:
        configure_logging()
        raise configuration_exit(error) from None
    return impl


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
        # The launcher's own parse runs inside its main(), after its body
        # defined serve_grpc: a launcher running as __main__ (python -m on the
        # launcher module) is bound here when register() ran before that
        # definition existed (vLLM loaded the plugins during the launcher's
        # own imports). Idempotent and cheap on every other parse.
        install_launcher_switch()
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
        # A Rust selection is checked against the launcher first, before
        # anything else is derived or logged for a launch that is not going
        # ahead: a launcher the switch cannot bind is refused here, not by the
        # Python servicer after the model loaded, and the refusal is the only
        # line such a launch prints.
        check_servicer_impl(self, parsed)
        # Upstream's launcher builds the Python servicer without its
        # namespace: a gRPC parser's parsed settings are kept for it in this
        # process (so a flag stays `source=flag`), and the set values go into
        # the environment too; on every gRPC parse, the flags' origin aside.
        handoff_mm_flags(self, parsed)
        # A launch that builds an engine for gRPC: the capture cap the
        # engine would default below the admission step is derived here.
        handoff_capture_cap(self, parsed)
        return parsed, extras

    FlexibleArgumentParser.parse_known_args = parse_known_args  # type: ignore[method-assign]
    setattr(FlexibleArgumentParser, _FLAG_MARK, True)
    # `vllm serve` imports the launcher after the parse; the finder binds the
    # switch as it loads. A launcher running as __main__ has defined
    # serve_grpc by the time vLLM loads the plugins from its main(), and is
    # bound in its namespace now.
    install_launcher_switch()
