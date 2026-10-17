# Mesh Module

The mesh lets several gateways run as one: they find each other, keep a
shared view of who is in the mesh, and replicate the state that routing
needs across the members. It is a library crate (`smg-mesh`) embedded in the
gateway; it serves one gRPC listener of its own and nothing over HTTP.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [What is synchronized](#what-is-synchronized)
- [Observability](#observability)
- [Example: two gateways on one host](#example-two-gateways-on-one-host)
- [Stopping a member](#stopping-a-member)
- [See Also](#see-also)

## Overview

Two protocols run over one listener (`--mesh-port`, default `39527`), both
RPCs of the `Gossip` service in `proto/gossip.proto`:

- **Membership** (`PingServer`, unary): SWIM-style probing. Every second a
  node probes one random member with a `Ping` that carries its whole
  membership table; the answer is merged into the local table. A member
  that answers neither a direct probe nor an indirect one (a `PingReq`
  relayed through up to three other live members) is marked `Suspected`, on
  the next failure `Down`, and the new status is broadcast.
- **State** (`SyncStream`, bidirectional streaming): one long-lived stream per
  pair of members. Every second each end sends the round's new entries of the
  stream namespaces and the CRDT operations the peer has not acknowledged
  yet; CRDT batches are acknowledged per key, so a lost batch is resent on a
  later round.

Nodes are `Init`, `Alive`, `Suspected`, `Down` or `Leaving` (`NodeStatus`
in `proto/gossip.proto`). There is no leader and no consensus: the stores are
conflict-free replicated data types (`crdt_kv`), so every member converges to
the same values without coordination.

## Architecture

```
model_gateway (model_gateway/src/mesh): adapters that bridge MeshKV
namespaces to the worker registry, the rate limiter and the prefix tree
        │
        ▼
service.rs        MeshServerBuilder / MeshServer / MeshServerHandler:
                  the cluster state, the listener, start and shutdown
        │
        ├── gossip_controller.rs   the 1 Hz event loop: probes, the init-peer
        │                          dial, outbound sync streams, retry backoff,
        │                          the round's RoundBatch for every sender
        ├── gossip_service.rs      the inbound side: PingServer and the
        │                          accepted sync streams
        ├── transport/             what both ends of a stream share
        │   ├── peer_stream.rs     one sender round, one inbound frame handler
        │   ├── sync_stream.rs     stream batches, heartbeats
        │   ├── crdt_batch.rs      CRDT batches and their acks
        │   ├── chunking.rs        values larger than one frame
        │   └── limits.rs          10 MiB frames, 60 s stream idle timeout
        ├── kv.rs                  MeshKV: CRDT and stream namespaces
        ├── crdt_kv/               merge strategies (last writer wins,
        │                          epoch-max-wins), the operation log
        ├── mtls.rs                the peers' TLS material
        └── metrics.rs             the router_mesh_* series
```

Which end dials the sync stream is decided by name: after a successful probe
of an `Alive` member, the node whose name sorts lower opens the stream
(`GossipController::should_initiate_connection`). The accepting end learns
the dialer's name from the first frame it receives. A stream that carries no
frame for 60 s (`STREAM_IDLE_TIMEOUT`) is closed and dialed again on a later
round. A peer that cannot be reached is retried with exponential backoff:
1 s doubling to at most 60 s (`RetryManager`).

## Configuration

The gateway configures the mesh from its flags (`model_gateway/src/main.rs`,
"Mesh Server"); the mesh has no configuration file of its own.

| Flag | Meaning |
|---|---|
| `--enable-mesh` | Start the mesh. Off by default. |
| `--mesh-server-name <name>` | This node's name in the mesh, the `peer` of its metrics and log lines. Default: `Mesh_` and four random characters. |
| `--mesh-host <ip>` | Bind address of the mesh listener; an IP literal, IPv6 with or without brackets. Default `0.0.0.0`. |
| `--mesh-port <port>` | The listener's port, default `39527`; must differ from `--port`, `--prometheus-port`, `--health-check-port` and `--admin-port`, and cannot be `0`. |
| `--mesh-advertise-host <ip>` | The address peers dial for this node; required when `--mesh-host` is unspecified (`0.0.0.0`, `::`). Advertised with `--mesh-port`. |
| `--mesh-peer-urls <ip:port> ...` | Where to join: the first entry is dialed once, in the node's first round, when its membership table holds nobody else; the rest of the mesh is learned from that peer. If that one dial fails (the peer is not up yet) it is not retried and the node stays alone, so start the peer first. Further entries are accepted and not used. |
| `--mesh-tls-ca-cert`, `--mesh-tls-cert`, `--mesh-tls-key` | Mutual TLS between members: all three or none. The listener then serves TLS with the node certificate and requires a peer certificate from the CA; every dial presents this node's certificate. The node certificate carries the node's IP as a SAN. |

A member whose mesh port is held by another process, or whose TLS material
cannot be used, does not start: the gateway refuses the start with
`Failed to bind the mesh listener on <addr> (--mesh-port): ...` or
`Failed to start the mesh server (<flags>): ...` instead of serving outside
the mesh (`model_gateway/src/server.rs`).

On Kubernetes the members can also find each other without
`--mesh-peer-urls`: `--router-selector key=value` (with Kubernetes worker
discovery on) watches the gateway Pods matching the selector and writes
them into the membership table; a Pod's mesh port is read from its
`sglang.ai/mesh-port` annotation (`router_mesh_port_annotation` in the
configuration file) and falls back to this gateway's own `--mesh-port` when
the annotation is missing or invalid (`model_gateway/src/mesh_discovery`).
The Helm chart's `deploy/helm/smg/examples/with-mesh-ha.yaml` is a two-replica
example.

## What is synchronized

`MeshKV` (`kv.rs`) offers two kinds of namespace, each selected by a key
prefix. CRDT namespaces hold durable, mergeable state; stream namespaces
carry ephemeral traffic that the application regenerates if a round is lost.
The gateway wires these (`model_gateway/src/mesh/wiring.rs`):

| Prefix | Kind, merge | Content | Gateway adapter |
|---|---|---|---|
| `worker:` | CRDT, last writer wins | `WorkerState` of every worker a member registered locally (id, model, URL, health, load, spec, owner); a tombstone when it is removed. Peers import them into their worker registry. | `adapters/worker_sync.rs` |
| `rl:` | CRDT, epoch-max-wins | One rate-limit shard per member and counter (`rl:<counter>:<node>`): a window epoch and a count; totals are the sum of the shards at the newest epoch. | `adapters/rate_limit_sync.rs` |
| `config:` | CRDT, last writer wins | Built into every `MeshKV` for the embedding application; the gateway writes nothing there today. | — |
| `td:` | stream, broadcast | Per-model deltas of the cache-aware prefix tree, one entry per model and round. | `adapters/tree_sync.rs` |
| `tree:req:`, `tree:page:` | stream, targeted | Repair requests for a prefix a member does not know, and the pages answering them, each sent to one peer. | `adapters/tree_sync.rs` |

## Observability

There is no HTTP API for the mesh: the gateway serves no route under
`/mesh`, and the former `/ha/*` management routes were removed
(`model_gateway/src/server.rs`). What an operator can read is the gateway's
log and the `router_mesh_*` series on the metrics listener
(`--prometheus-port`, default `29000`).

### Log lines

With `--log-level info` the mesh reports (source file in parentheses):

| Level | Line | Meaning |
|---|---|---|
| info | `Round <n> Status:{...}` | Once a second, the whole membership table as this node sees it: every member's name, address, status and version (`gossip_controller.rs`, the event loop). |
| info | `Connecting to peer <name> at <addr>` / `Successfully connected to peer <name>` | A probe and its success. The configured join peer is called `init_peer` until its name is learned from its answer. |
| warn | `Error connecting to peer <name> (attempt <n>): <error>. Next retry in <delay>` | A failed probe and the backoff. |
| info | `Cluster state updated. Current nodes: <n>` | A received probe changed the membership table (`gossip_service.rs`). |
| info | `Starting sync_stream connection to peer <name> at address <addr>` / `Sync stream connection to <name> established` | This node dialed a sync stream. |
| info | `Sync stream connection to <name> has finished, removing` | That stream ended; it is dialed again on a later round. |
| warn | `sync_stream idle timeout (60s) — closing` | No frame for 60 s on a stream (`transport/peer_stream.rs`). |
| info | `Mesh listener <addr> is prepared to serve TLS with the node certificate` | At start, with the TLS flags. |
| error | `Mesh server failed: <error>` | The mesh server's task ended while the gateway runs (`model_gateway/src/server.rs`). |
| info | `Mesh router discovery started` | Kubernetes discovery of the members is running. |

### Metrics

`metrics.rs` records these series (the gateway calls `init_mesh_metrics` for
their descriptions):

| Series | Type, labels | Recorded by |
|---|---|---|
| `router_mesh_peer_connections` | gauge, `peer` | The end that accepted the sync stream from `peer`: `1` while the stream is open, `0` after it closed. For a pair of members the series appears on the accepting member only (the dialing end records none). |
| `router_mesh_peer_reconnects_total` | counter, `peer` | The accepting end, once per closed stream from `peer`. |
| `router_mesh_peer_ack_total` | counter, `peer`, `status` (`success`, `failure`) | The accepting end, per ACK frame from `peer`. |
| `router_mesh_peer_nack_total` | counter, `peer` | The accepting end, per NACK frame from `peer`. |
| `router_mesh_sync_round_duration_seconds` | histogram, `peer` | The dialing end, the time of each sender round towards `peer`. |

`router_mesh_store_cardinality`, `router_mesh_store_hash`,
`router_rl_drift_ratio` and `router_lb_drift_ratio` are described in
`metrics.rs` but nothing records them, so they do not appear in a scrape.

## Example: two gateways on one host

```bash
smg --host 127.0.0.1 --port 29740 --prometheus-host 127.0.0.1 --prometheus-port 29741 \
    --enable-mesh --mesh-server-name A --mesh-host 127.0.0.1 --mesh-port 29742
smg --host 127.0.0.1 --port 29744 --prometheus-host 127.0.0.1 --prometheus-port 29745 \
    --enable-mesh --mesh-server-name B --mesh-host 127.0.0.1 --mesh-port 29746 \
    --mesh-peer-urls 127.0.0.1:29742
```

A goes first: B dials it once, in B's first round. Within a few rounds B's
log shows `Successfully connected to peer init_peer` and A's `Cluster state
updated. Current nodes: 2`; each `Round <n> Status` line then lists both
members with `status: Alive`. `A` sorts before `B`, so A dials the sync
stream and B accepts it:

```bash
curl -s http://127.0.0.1:29745/metrics | grep '^router_mesh_peer_connections'
# router_mesh_peer_connections{peer="A"} 1
```

The same pair with the three `--mesh-tls-*` flags on both sides forms the
mesh over mutual TLS; a dialer without a certificate from the CA is refused by
the listener.

## Stopping a member

A gateway told to stop (SIGTERM) leaves the mesh first: during its settle
window it broadcasts `Leaving` to the live members
(`MeshServerHandler::graceful_shutdown` in `service.rs`, bounded by the
window), so they see the member leave within seconds. A member that dies
without that is noticed through the probes, `Suspected` after the first
failed probe round and `Down` after the next. Under mesh mTLS the broadcast
reaches the members only once the status broadcasts dial with the node's TLS
identity (a change of the mesh crate of its own); until then they find out
through the probes as well.

Only the owner of a `worker:` key tombstones it. Without `--mesh-server-name`
a restarted gateway comes back under a fresh random name, so the old name
stays `Down` in the membership table and its keys are never tombstoned. The
Helm chart avoids this: with `router.mesh.enabled` it runs the gateways as a
StatefulSet and passes the Pod's name, which survives restarts, as
`--mesh-server-name`. Once a key's owner has been `Down` or `Leaving` for
20 s (`DEAD_OWNER_GRACE` in `adapters/worker_sync.rs`), the other members
ignore the key and remove the worker they had imported from it. A member that
should reclaim its keys when it returns needs a stable `--mesh-server-name`.

## See Also

- The High Availability page of the documentation site, linked from the
  repository README.
- `proto/gossip.proto`: the wire format.
- `crdt_kv/`: the merge strategies and the operation log.
