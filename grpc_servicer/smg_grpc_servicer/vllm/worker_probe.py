"""The multi-node worker pod's liveness probe for the vLLM servicer.

An engine whose tensor-parallel ranks span pods runs its EngineCore and the
gRPC servicer on a leader pod and the other ranks on worker pods started with
``vllm serve --headless --node-rank <k> --master-addr <leader>``. Every rank
joins the process group through the leader's TCP store (``--master-port``)
and keeps that connection for its lifetime; when the leader's engine goes
away, the ranks on the worker pods wait on a dead store, which is what this
probe detects::

    python3 -I -m smg_grpc_servicer.vllm.worker_probe [--leader HOST] [--port PORT]

Without arguments the leader is read off the worker's own command line
(``--master-addr`` and ``--master-port`` in ``/proc/1/cmdline``), the port
falling back to the engine's own default. The contract and the manifest are
:mod:`smg_grpc_servicer.worker_probe`'s.
"""

from __future__ import annotations

from collections.abc import Sequence

from smg_grpc_servicer import worker_probe

#: vLLM's ``--master-port`` default.
DEFAULT_MASTER_PORT = 29501


def master_from_cmdline(cmdline: bytes) -> tuple[str | None, int | None]:
    """``--master-addr`` and ``--master-port`` as the engine's command line
    carries them (``--flag value`` or ``--flag=value``, with ``-`` or ``_``
    in the flag's name, inside a shell wrapper's one argument too); ``None``
    for a flag that is absent or unreadable."""
    values = worker_probe.flag_values(cmdline, "--master-addr", "--master-port")
    address = values.get("--master-addr") or None
    try:
        port = int(values["--master-port"]) if "--master-port" in values else None
    except ValueError:
        port = None
    return address, port


RENDEZVOUS = worker_probe.Rendezvous(
    module="smg_grpc_servicer.vllm.worker_probe",
    leader_flag="--master-addr",
    port_flag="--master-port",
    leader_from_cmdline=master_from_cmdline,
    default_port=DEFAULT_MASTER_PORT,
)


def main(argv: Sequence[str] | None = None) -> int:
    return worker_probe.main(RENDEZVOUS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
