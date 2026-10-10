"""The multi-node worker pod's liveness probe for the SGLang servicer and
launcher.

An engine whose tensor-parallel ranks span pods is launched on every pod with
the same engine flags, ``--nnodes N --dist-init-addr <leader>:<port>`` among
them, and the pod's own ``--node-rank <k>``: the gRPC servicer (or the
headless launcher) on the leader pod, the scheduler ranks alone on the
workers. Every rank joins the process group through the TCP store at
``--dist-init-addr``, which rank 0 on the leader pod hosts, and keeps that
connection for its lifetime; when the leader's engine goes away, the ranks on
the worker pods wait on a dead store, which is what this probe detects::

    python3 -I -m smg_grpc_servicer.sglang.worker_probe [--leader HOST] [--port PORT]

Without arguments both are read off ``--dist-init-addr`` on the worker's own
command line (``/proc/1/cmdline``). The contract and the manifest are
:mod:`smg_grpc_servicer.worker_probe`'s.
"""

from __future__ import annotations

from collections.abc import Sequence

from smg_grpc_servicer import worker_probe

#: SGLang's rendezvous flag: ``host:port`` of the leader's store.
DIST_INIT_ADDR = "--dist-init-addr"


def leader_from_cmdline(cmdline: bytes) -> tuple[str | None, int | None]:
    """The leader's host and store port as ``--dist-init-addr host:port`` on
    the engine's command line carries them (an IPv6 literal in brackets, the
    ``=`` and ``_`` spellings, inside a shell wrapper's one argument too);
    ``None`` for a part that is absent or unreadable."""
    value = worker_probe.flag_values(cmdline, DIST_INIT_ADDR).get(DIST_INIT_ADDR)
    return worker_probe.split_host_port(value) if value else (None, None)


RENDEZVOUS = worker_probe.Rendezvous(
    module="smg_grpc_servicer.sglang.worker_probe",
    leader_flag=DIST_INIT_ADDR,
    port_flag=DIST_INIT_ADDR,
    leader_from_cmdline=leader_from_cmdline,
)


def main(argv: Sequence[str] | None = None) -> int:
    return worker_probe.main(RENDEZVOUS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
