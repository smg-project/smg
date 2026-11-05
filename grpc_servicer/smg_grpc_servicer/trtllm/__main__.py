"""CLI entrypoint for the TensorRT-LLM gRPC server.

Usage::

    python -m smg_grpc_servicer.trtllm --model <model> --host 127.0.0.1 --port 50051 \\
        --llm-args '{"tensor_parallel_size": 2}' --servicer-impl rust

``--servicer-impl {python,rust}``, ``--metrics-port`` and ``--metrics-host``
are this package's flags; the LLM API's keyword arguments ride ``--llm-args``
(JSON text, or ``@path`` to a JSON or YAML file, the shape of the engine's own
``--extra_llm_api_options``). ``python`` hands the process to the engine's own
servicer (``tensorrt_llm.grpc.smg``); ``rust`` (or
``SMG_TRTLLM_SERVICER_IMPL=rust``, the fallback the flag overrides) serves the
same contract from Rust (see :mod:`smg_grpc_servicer.trtllm.rust`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from smg_grpc_servicer.rust_lifecycle import (
    ServicerConfigError,
    check_metrics_port_needs_rust,
    configuration_exit,
)
from smg_grpc_servicer.trtllm.rust import (
    METRICS_PORT_ENV,
    ServeArgs,
    add_metrics_arguments,
    add_servicer_impl_argument,
    serve_rust,
    servicer_impl_source,
)

logger = logging.getLogger("smg_grpc_servicer.trtllm")


def load_llm_args(text: str | None) -> dict[str, Any]:
    """``--llm-args``: JSON text, or ``@path`` to a JSON or YAML file."""
    if not text:
        return {}
    if text.startswith("@"):
        path = Path(text[1:])
        raw = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml

            loaded = yaml.safe_load(raw)
        else:
            loaded = json.loads(raw)
    else:
        loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ServicerConfigError("--llm-args must be a JSON/YAML object")
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smg_grpc_servicer.trtllm",
        description="The TensorRT-LLM gRPC servicer; --llm-args carries the LLM API's keywords.",
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True, help="The checkpoint (path or repository id).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--served-model-name", dest="served_model_name", default=None)
    parser.add_argument(
        "--llm-args",
        dest="llm_args",
        default=None,
        help="The LLM API's keyword arguments: JSON text, or @path to a JSON/YAML file.",
    )
    add_servicer_impl_argument(parser)
    add_metrics_arguments(parser)
    return parser


def serve_args(parsed: argparse.Namespace) -> ServeArgs:
    llm_args = load_llm_args(parsed.llm_args)
    llm_args.setdefault("model", parsed.model)
    llm_args.setdefault("backend", "pytorch")
    return ServeArgs(
        model=parsed.model,
        host=parsed.host,
        port=parsed.port,
        llm_args=llm_args,
        served_model_name=parsed.served_model_name,
        servicer_impl=getattr(parsed, "servicer_impl", None),
        metrics_port=getattr(parsed, "metrics_port", None),
        metrics_host=getattr(parsed, "metrics_host", None),
    )


def serve_python(args: ServeArgs) -> None:
    """The engine's own servicer, from its tree."""
    from tensorrt_llm.grpc.smg.server import launch_smg_server

    launch_smg_server(
        args.host, args.port, dict(args.llm_args), served_model_name=args.served_model_name
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    parsed = build_parser().parse_args(argv)
    try:
        args = serve_args(parsed)
        impl, origin = servicer_impl_source(args)
        logger.info("Servicer implementation: %s (source=%s)", impl, origin)
        if impl == "rust":
            raise SystemExit(asyncio.run(serve_rust(args)))
        # A metrics port under the Python servicer is a port nobody serves.
        check_metrics_port_needs_rust(impl, args, METRICS_PORT_ENV)
    except ServicerConfigError as error:
        raise configuration_exit(error, logger) from None
    serve_python(args)


if __name__ == "__main__":
    main()
