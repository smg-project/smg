# radix-index

A shared radix membership index for cache-aware routing: gateways ask
"which worker already holds the longest prefix of this request?" without
each gateway building and syncing its own tree.

The data structure is SMG `kv_index`'s block-quantized positional
index (a direct import — whether the service should instead get a
ground-up generic radix tree is an open evaluation). This crate wraps
it in a keyspace-partitioned engine, a gRPC surface, and the
client/bridge pieces that feed and query it.

## Interface: two verbs

- **Publish** (client-streamed `Update`s, acked): a holder's block-hash
  chain changed. Two feeds share the verb:
  - **Event feed** — a bridge subscribes to engine KV events
    (`SubscribeKvEvents`) and forwards Stored/Removed/Cleared batches,
    sequenced per holder with epoch bumps on gap or backend loss.
    Eviction is *observed*, so index state tracks engine truth.
  - **Placement feed** — a gateway publishes "this request's chain now
    (probably) resides on that worker" after each completed request
    (`seq=0`, content-idempotent). No engine cooperation needed — this
    is the path for engines/modes with no KV event stream. Inferred
    state is bounded by idle TTL + per-holder capacity with tail-first
    (prefix-closed) eviction.
  The first sequenced or Removed-bearing update marks a holder
  *event-fed*: placements for it are ignored from then on, so the
  precise feed always wins.
- **Subscribe** (bidirectional query stream): content-hash chain in,
  per-holder matched-block counts out. The gateway enforces its own
  deadline (2 ms in SMG) and falls back to its local policy on a miss —
  answers are advisory, never load-bearing for correctness.

`Pull` streams the whole state as synthetic `Update`s; a starting
replica bootstraps from a sibling with it before declaring ready.

## Replication: copy, don't agree

No consensus. Writes are per-holder sequenced (event feed) or
content-idempotent (placement feed), so replicas converge by applying
the same updates in any interleaving. A replica relays each accepted
`Publish` to its `--peers` best-effort; a wedged peer drops relayed
updates rather than wedging ingest, and TTL plus re-placement plus
bootstrap bound the divergence. Gateways stay stupid: one endpoint, no
fan-out, reconnect on failure.

## Keyspaces

State is partitioned by `(model, symbol_kind, block_size)`. TOKENS at
the engine page size is what SMG uses today; BYTES exists for text-mode
(HTTP) feeds that hash normalized bytes instead of token ids.

## Binaries

- `radix-index-service` — the server. Flags:

  | flag | default | meaning |
  |---|---|---|
  | `--bind` | `127.0.0.1` | listen address (set `0.0.0.0` in k8s) |
  | `--port` | `40000` | gRPC port |
  | `--metrics-port` | off | admin plane: `/metrics`, `/healthz`, `/readyz` |
  | `--peers` | none | sibling replicas to relay Publishes to (comma-separated URLs) |
  | `--bootstrap-from` | none | sibling to Pull state from before serving |
  | `--inferred-ttl-secs` | `180` | idle TTL for placement-fed holders |
  | `--event-ttl-secs` | `1800` | liveness backstop for EVENT-fed holders: silence past this soft-retires the holder (a lost departure signal must not leak it); `0` disables |
  | `--default-capacity-blocks` | unbounded | RUNAWAY PROTECTION for holders that never sent `Added` — the index truncates only past 2x this value. Leave unbounded or set well above worker KV size: the placement feed carries no removal signal, so an index that races the worker's own eviction under-matches (measured); idle TTL is the freshness bound. |
  | `--sweep-interval-secs` | `5` | idle-sweep cadence |
  | `--apply-delay-stored-ms` / `--apply-delay-removed-ms` | `0` | staleness injection (experiments only) |

  Stops gracefully on SIGTERM/ctrl-c. `/readyz` answers 503 until the
  bootstrap pull completes — point the k8s readiness probe at it.

- `radix-index-bridge` — per-fleet event bridge: engine KV event
  streams in, index Updates out. Flags: `--workers` (comma-separated
  worker URLs), `--index` (service URL), `--model`, `--block-size`
  (MUST match the gateway's `--kv-indexer-block-size` — the keyspace
  key includes it, and a mismatch silently splits the fleet's state
  into two keyspaces; both default to the same shared value).
- `radix-index-bench` — apply/query throughput and memory-per-entry
  microbench.

A reference StatefulSet lives in [`deploy/statefulset.yaml`](deploy/statefulset.yaml).

## Gateway side (SMG)

`--kv-indexer-url` + `--kv-indexer-block-size` on the gateway enable
the remote path: a routing-time overlap prefetch (2 ms deadline,
fast-fail while disconnected) feeding the cache-aware policy, and a
placement publish of the prompt⊕output chain after each completed
request. Flag off = every code path byte-identical to local behavior.

## Metrics

`/metrics` (Prometheus text): `radix_index_{keyspaces,holders,event_fed_holders,dropped_holders,blocks}`
gauges and `radix_index_{applies,queries,relay_dropped}_total` counters.
