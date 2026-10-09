"""CLI entrypoint for the TokenSpeed gRPC server.

Usage::

    python -m smg_grpc_servicer.tokenspeed --model <model> --host 127.0.0.1 --port 50051

``--servicer-impl {python,rust}`` is this package's flag; every other flag is
TokenSpeed's :class:`ServerArgs`, parsed by ``prepare_server_args`` so there
is no flag drift vs the HTTP frontend. ``--servicer-impl rust`` (or
``SMG_TOKENSPEED_SERVICER_IMPL=rust``, the fallback the flag overrides) serves
the same contract from Rust (see :mod:`smg_grpc_servicer.tokenspeed.rust`).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from tokenspeed.runtime.utils.server_args import prepare_server_args

from smg_grpc_servicer.tokenspeed.rust import (
    add_servicer_impl_argument,
    serve_rust,
    servicer_impl_source,
)
from smg_grpc_servicer.tokenspeed.server import serve_grpc

try:
    import uvloop
except ImportError:  # uvloop is optional — fall back to the default loop.
    uvloop = None

logger = logging.getLogger("smg_grpc_servicer.tokenspeed")


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Our flags first; everything else is TokenSpeed's ``ServerArgs``.

    A ``--help`` goes on to TokenSpeed's parser, which prints its own help
    and exits; this package's flags are printed ahead of it."""
    parser = argparse.ArgumentParser(
        prog="python -m smg_grpc_servicer.tokenspeed",
        description="smg's own flags; every other flag is TokenSpeed's ServerArgs, listed below.",
        add_help=False,
        allow_abbrev=False,
    )
    add_servicer_impl_argument(parser)
    ours, rest = parser.parse_known_args(argv)
    if any(arg in ("-h", "--help") for arg in rest):
        parser.print_help()
        print()
    return ours, rest


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    ours, rest = split_args(argv)
    server_args = prepare_server_args(rest)
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    # One flag selects the Rust request path (--servicer-impl, else the
    # environment); the entrypoint and every ServerArgs flag stay the same,
    # and the Router cannot tell them apart.
    impl, origin = servicer_impl_source(ours)
    logger.info("Servicer implementation: %s (source=%s)", impl, origin)
    if impl == "rust":
        raise SystemExit(asyncio.run(serve_rust(server_args)))
    asyncio.run(serve_grpc(server_args))


if __name__ == "__main__":
    main()
