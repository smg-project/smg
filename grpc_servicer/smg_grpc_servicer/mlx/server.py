"""
MLX gRPC Server

Standalone gRPC server entrypoint for MLX inference.
CLI: python -m smg_grpc_servicer.mlx.server --model <path> --port 50051
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from concurrent import futures

import grpc
import mlx.core as mx
import mlx.nn as nn
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from huggingface_hub import snapshot_download
from mlx_lm import load
from smg_grpc_proto import mlx_engine_pb2, mlx_engine_pb2_grpc

from smg_grpc_servicer.mlx.health_servicer import MlxHealthServicer
from smg_grpc_servicer.mlx.servicer import MlxEngineServicer

logger = logging.getLogger(__name__)


def _eval_all_module_arrays(module: nn.Module) -> None:
    """Force-eval every mx.array in the module tree, including private attrs.

    mlx-lm's load_model() calls mx.eval(model.parameters()), but
    nn.Module.parameters() silently skips underscore-prefixed attributes
    (e.g. YarnRoPE._freqs, which is computed lazily from mx.arange() and
    arithmetic operations at module-construction time).  Those unevaluated
    arrays are scheduled on the main thread's default stream (Stream gpu, 1).

    The generation loop runs inside ``with mx.stream(generation_stream):``
    (Stream gpu, 0) on a background thread.  When its forward pass tries to
    materialize KV-cache arrays that transitively depend on an unevaluated
    _freqs, MLX cannot find Stream(gpu, 1) on that thread and raises::

        RuntimeError: There is no Stream(gpu, 1) in current thread.

    Calling this function right after load() materialises every lazy array
    on the main thread before the generation thread starts.
    """
    arrays: list[mx.array] = []

    def _collect_value(v: object) -> None:
        if isinstance(v, mx.array):
            arrays.append(v)
        elif isinstance(v, nn.Module):
            # m.items() (MLX's dict-like API) returns ALL module attributes,
            # including underscore-prefixed ones like _freqs. vars()/.__dict__
            # is NOT the right API — MLX stores module state internally, so
            # vars(m) only returns ['_no_grad', '_training'].
            for _, child in v.items():
                _collect_value(child)
        elif isinstance(v, dict):
            for child in v.values():
                _collect_value(child)
        elif isinstance(v, (list, tuple)):
            for child in v:
                _collect_value(child)

    _collect_value(module)
    if arrays:
        mx.eval(arrays)


def parse_args():
    parser = argparse.ArgumentParser(description="MLX gRPC inference server")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace repo ID")
    parser.add_argument("--port", type=int, default=50051, help="gRPC listen port")
    parser.add_argument("--host", default="0.0.0.0", help="gRPC listen address")
    parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Max concurrent prefill requests"
    )
    parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Max concurrent generation requests"
    )
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path")
    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer via mlx-lm."""
    logger.info("Loading model: %s", args.model)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    # Force-eval non-parameter arrays missed by mlx-lm's mx.eval(model.parameters()).
    # Needed for models with YaRN/other RoPE variants whose _freqs arrays are
    # excluded from parameters() due to underscore prefix. See _eval_all_module_arrays.
    _eval_all_module_arrays(model)
    logger.info("Model loaded successfully")

    model_dir = args.model
    if not os.path.isdir(model_dir):
        model_dir = snapshot_download(
            args.model,
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "special_tokens*",
                "merges.txt",
                "vocab.json",
                "added_tokens.json",
                # Chat template sidecars (Gemma 4, Llama 3.1+, newer models).
                "chat_template.json",
                "chat_template.jinja",
                # tiktoken-style tokenizer artifacts — must stay in sync
                # with MlxEngineServicer._TOKENIZER_FILES / _SUFFIXES.
                "tiktoken.model",
                "*.tiktoken",
            ],
        )

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)

    eos = model_config.get("eos_token_id")
    if isinstance(eos, int):
        eos_token_ids = [eos]
    elif isinstance(eos, list):
        eos_token_ids = eos
    else:
        eos_token_ids = list(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else []

    return model, tokenizer, model_dir, model_config, eos_token_ids


async def serve_grpc(args):
    """Start the MLX gRPC server."""
    start_time = time.time()

    model, tokenizer, model_dir, model_config, eos_token_ids = load_model(args)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    health_servicer = MlxHealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Construct the servicer WITHOUT a BatchGenerator. The BatchGenerator
    # (and its thread-local mlx stream) is built on the generation thread
    # below, so all mlx state lives on one thread — same model as
    # mlx-lm.server. This avoids the cross-thread "no Stream(gpu, 1) in
    # current thread" RuntimeError we saw when an mx.async_eval
    # continuation tried to look up the stream context on a thread that
    # never bound it.
    servicer = MlxEngineServicer(
        model=model,
        completion_batch_size=args.completion_batch_size,
        prefill_batch_size=args.prefill_batch_size,
        model_path=args.model,
        model_dir=model_dir,
        model_config=model_config,
        eos_token_ids=eos_token_ids,
        start_time=start_time,
    )
    mlx_engine_pb2_grpc.add_MlxEngineServicer_to_server(servicer, server)

    SERVICE_NAMES = (
        mlx_engine_pb2.DESCRIPTOR.services_by_name["MlxEngine"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    listen_addr = f"{args.host}:{args.port}"
    bound_port = server.add_insecure_port(listen_addr)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server to {listen_addr}")

    # The gen thread does construction → warmup → enters main loop. Wait
    # for it to signal ready before flipping the health check to SERVING,
    # otherwise a Generate RPC could slip into the window where the gen
    # thread hasn't constructed BatchGenerator yet and block forever on
    # _pending. wait_ready() returns False if BatchGenerator construction
    # raised on the gen thread — fail startup loudly in that case rather
    # than advertising a healthy server with a dead gen thread that hangs
    # every Generate RPC on _pending.
    servicer.start_generation_loop()
    loop = asyncio.get_running_loop()
    ready = await loop.run_in_executor(None, servicer.wait_ready)
    if not ready:
        servicer.stop_generation_loop()
        raise RuntimeError("MLX generation thread failed to become ready — see preceding logs")

    await server.start()
    health_servicer.set_serving()
    logger.info("gRPC server listening on %s — model: %s", listen_addr, args.model)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down...")
        health_servicer.set_not_serving()
        # Stop accepting new RPCs first so in-flight requests can still
        # drain against the running generation thread. Stopping the gen
        # loop first would leave new/in-flight RPCs stranded.
        await server.stop(5.0)
        servicer.stop_generation_loop()
        logger.info("Server stopped")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(serve_grpc(args))


if __name__ == "__main__":
    main()
