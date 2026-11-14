"""Explicit managed P/D entry point; ordinary SGLang launch never selects this path."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid

from .plugin import EVENTS_ENV, INSTANCE_ENV, ROLE_ENV, WORKER_ENV


class ManagedBootstrap:
    """Own only native encoder registration; no frontend encode dispatcher or processor."""

    def __init__(self, args):
        self.args = args
        self.server = None

    def start(self):
        """Start the native receiver bootstrap before native scheduler ranks register."""
        from sglang.srt.disaggregation.encoder.receiver import EncoderBootstrapServer

        self.server = EncoderBootstrapServer(
            host=self.args.host,
            port=self.args.encoder_bootstrap_port,
            urls=[],
        )

    def shutdown(self):
        """Release the native bootstrap server with the scheduler process group."""
        if self.server is not None:
            self.server.close()


def launcher(config):
    """Construct a separate lifecycle without changing the default SGLang launcher."""
    from smg_grpc_servicer.sglang.rust import SglangLauncher, launch_scheduler_ranks

    class ManagedLauncher(SglangLauncher):
        ipc_prefix = "sglang-managed-servicer"
        warmup_env = ""

        def server_extras(self, args, tokenizer_dir):
            """Managed startup never constructs the legacy encode-dispatch bridge."""
            if args.disaggregation_mode != config["role"]:
                raise ValueError("managed role must equal native disaggregation mode")
            if not args.language_only:
                raise ValueError("managed P/D requires --language-only")
            if os.environ.get("SMG_SGLANG_MM_PROCESSOR"):
                raise ValueError("managed P/D refuses legacy media preprocessing")
            return {"sglang_managed_epd": json.dumps(config)}

        def launch(self, args, handshake_port, facts):
            """Use unchanged native TP/DP spawning and P's native receiver bootstrap."""
            bootstrap = ManagedBootstrap(args) if config["role"] == "prefill" else None
            return launch_scheduler_ranks(
                args, handshake_port=handshake_port, encode_bridge=bootstrap
            )

    return ManagedLauncher()


def main():
    """Load finite limits explicitly and assign a new process fence on every launch."""
    from sglang.srt.server_args import prepare_server_args

    from .config import complete

    args = prepare_server_args(sys.argv[1:])
    config = complete({}, args, args.disaggregation_mode)
    if config["role"] not in ("prefill", "decode"):
        raise ValueError("managed P/D requires native --disaggregation-mode prefill or decode")
    config["generation"] = uuid.uuid4().hex
    os.environ[ROLE_ENV] = config["role"]
    os.environ[WORKER_ENV] = config["worker_id"]
    os.environ[INSTANCE_ENV] = config["generation"]
    # The temporary directory is kept until the native ranks and Rust servicer have exited.
    with tempfile.TemporaryDirectory(prefix="smg-managed-") as directory:
        endpoint = f"ipc://{directory}/metadata.sock" if config["role"] == "prefill" else None
        config["metadata_endpoint"] = endpoint
        if endpoint:
            os.environ[EVENTS_ENV] = endpoint
        from sglang.srt.plugins import load_plugins

        load_plugins()
        raise SystemExit(asyncio.run(launcher(config).serve(args)))


if __name__ == "__main__":
    main()
