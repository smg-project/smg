"""Launch SGLang scheduler ranks headless behind SMG.

    python -m smg_grpc_servicer.sglang.headless \\
        --zmq-handshake-address tcp://127.0.0.1:<port> [--zmq-engine-index N] \\
        --model-path <model> [any SGLang server args]

No HTTP server, no tokenizer manager, no detokenizer: the ranks are spawned
the way this package's gRPC server spawns them, and the rank that owns
request I/O dials SMG once its model is loaded (see :mod:`zmq_plugin`, which
SGLang loads in each rank because this package registers it as an SGLang
plugin). This process only supervises: it exits when a rank exits or on
SIGTERM / SIGINT, taking the ranks with it.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Any

from smg_grpc_servicer.sglang.zmq_msgpack import MAX_ENGINE_INDEX
from smg_grpc_servicer.sglang.zmq_plugin import ENGINE_INDEX_ENV, HANDSHAKE_ENV

logger = logging.getLogger(__name__)


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Our flags first; everything else is SGLang's ``ServerArgs``."""
    parser = argparse.ArgumentParser(
        prog="python -m smg_grpc_servicer.sglang.headless", add_help=False
    )
    parser.add_argument(
        "--zmq-handshake-address",
        required=True,
        help="SMG's handshake socket, tcp://host:port (what SMG derives from the worker's ipc:// URL).",
    )
    parser.add_argument(
        "--zmq-engine-index",
        type=int,
        default=0,
        help="Engine index this launch registers with (plus each rank's DP rank).",
    )
    ours, rest = parser.parse_known_args(argv)
    if not ours.zmq_handshake_address.startswith("tcp://"):
        parser.error("--zmq-handshake-address must be tcp://host:port")
    if not 0 <= ours.zmq_engine_index <= MAX_ENGINE_INDEX:
        parser.error(f"--zmq-engine-index must be in 0..={MAX_ENGINE_INDEX}")
    return ours, rest


class Interrupted(Exception):
    """A stop signal arrived while the ranks were still starting."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def make_signal_handler(stop: threading.Event, faulted: threading.Event, started: threading.Event):
    """SIGTERM/SIGINT stop the supervise loop; SIGQUIT is SGLang reporting a
    crashed rank. While the ranks are still starting, the main thread sits in
    a blocking pipe read that Python retries after a handler returns (PEP
    475), so the handler raises instead and ``launch`` cleans up at once.
    Only the first signal raises: a second one, landing while ``launch`` is
    already cleaning up, would otherwise unwind the cleanup itself."""

    def on_signal(signum, _frame):
        if signum == signal.SIGQUIT:
            logger.error("headless scheduler: a scheduler rank failed; stopping")
            faulted.set()
        else:
            logger.info("headless scheduler: received signal %d; stopping", signum)
        first = not stop.is_set()
        stop.set()
        if first and not started.is_set():
            raise Interrupted(signum)

    return on_signal


def launch(ours: argparse.Namespace, server_argv: list[str]) -> int:
    from sglang.srt.server_args import prepare_server_args

    return run(prepare_server_args(server_argv), ours.zmq_handshake_address, ours.zmq_engine_index)


def run(server_args: Any, handshake_address: str, engine_index: int = 0) -> int:
    """Run the scheduler ranks headless on prepared server args until they
    exit or a signal arrives; returns the process exit code. The Rust servicer
    calls this in a spawned child with the args SGLang's entrypoint parsed."""
    os.environ[HANDSHAKE_ENV] = handshake_address
    os.environ[ENGINE_INDEX_ENV] = str(engine_index)

    from sglang.srt.entrypoints.engine import Engine, _set_envs_and_config
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.plugins import load_plugins
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import configure_logger, kill_process_tree

    from smg_grpc_servicer.sglang.scheduler_launcher import terminate_scheduler_processes

    configure_logger(server_args)
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    load_plugins()
    server_args.check_server_args()
    # This process fills the tokenizer manager's role for argument resolution
    # only; the ranks publish their own records when they start.
    publish(server_args, role="tokenizer")
    port_args = PortArgs.init_new(server_args)

    stop = threading.Event()
    faulted = threading.Event()  # a rank raised (SGLang signals its parent with SIGQUIT)
    started = threading.Event()
    on_signal = make_signal_handler(stop, faulted, started)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGQUIT, on_signal)

    procs: list = []
    exit_code = 0
    try:
        # SGLang's own spawner: it knows the running version's rank layout and
        # `run_scheduler_process` signature, which change between releases.
        init, procs = Engine._launch_scheduler_processes(
            server_args, port_args, run_scheduler_process
        )
        procs = list(procs or [])
        init.wait_for_ready()
        started.set()
        logger.info(
            "headless scheduler ready: %d rank process(es); SMG's handshake is %s (engine index %d)",
            len(procs),
            handshake_address,
            engine_index,
        )
        while not stop.is_set():
            for proc in procs:
                if proc.exitcode is not None:
                    logger.error(
                        "headless scheduler: rank process %d exited with %s",
                        proc.pid,
                        proc.exitcode,
                    )
                    exit_code = 1 if proc.exitcode != 0 else exit_code
                    stop.set()
                    break
            time.sleep(0.5)
    except Interrupted as interrupted:
        exit_code = 128 + interrupted.signum
    finally:
        terminate_scheduler_processes(procs)
        kill_process_tree(os.getpid(), include_parent=False)
    return 1 if faulted.is_set() else exit_code


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
    ours, rest = split_args(sys.argv[1:] if argv is None else argv)
    raise SystemExit(launch(ours, rest))


if __name__ == "__main__":
    main()
