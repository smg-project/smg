"""The multi-node worker pod's liveness probe for the TokenSpeed servicer.

An engine whose tensor-parallel ranks span pods is launched on every pod with
the same engine flags, ``--nnodes N --dist-init-addr <head>:<port>`` among
them, and the pod's own ``--node-rank <k>``: the launcher
(``python -m smg_grpc_servicer.tokenspeed``, the process ``ts serve`` spawns
for its gRPC worker) on the leader pod, the scheduler ranks alone on the
workers. Every rank joins the process group through the TCP store at
``--dist-init-addr``, which rank 0 on the leader pod hosts, and keeps that
connection for its lifetime; when the leader's engine goes away, the ranks on
the worker pods wait on a dead store, which is what this probe detects::

    python3 -I -m smg_grpc_servicer.tokenspeed.worker_probe [--leader HOST] [--port PORT]

Without arguments both are read off ``--dist-init-addr`` on the worker's own
command line (``/proc/1/cmdline``); when TokenSpeed derives that flag from a
multi-node Slurm step's environment instead, the manifest passes ``--leader``
and ``--port``. The contract and the manifest are
:mod:`smg_grpc_servicer.worker_probe`'s.
"""

from __future__ import annotations

from collections.abc import Sequence

from smg_grpc_servicer import worker_probe

#: TokenSpeed's rendezvous flag: ``host:port`` of the head node's store.
DIST_INIT_ADDR = "--dist-init-addr"


RENDEZVOUS = worker_probe.Rendezvous.from_host_port_flag(
    module="smg_grpc_servicer.tokenspeed.worker_probe",
    flag=DIST_INIT_ADDR,
)

#: The leader's host and store port as ``--dist-init-addr host:port`` on the
#: engine's command line carries them; ``None`` for a part that is absent or
#: unreadable. The engine's own ``--port`` is its gRPC port, not the store's.
leader_from_cmdline = RENDEZVOUS.leader_from_cmdline


def main(argv: Sequence[str] | None = None) -> int:
    return worker_probe.main(RENDEZVOUS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
