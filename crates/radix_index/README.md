# radix-index

Published on crates.io as **`smg-radix-index`** (import path `radix_index`).
Depends on `smg-radix-tree` and `smg-grpc-client` (the event bridge), so it
publishes in tier 3 of the release workflow.

A shared radix membership index for cache-aware routing: gateways ask
"which worker already holds the longest prefix of this request?" without
each gateway building and syncing its own tree.

The data structure is [`smg-radix-tree`](../radix_tree): a chain-native
block-quantized positional index owned by this stack rather than shared
with the gateway's `kv_index` (the wire hash scheme is pinned against
`kv_index` by golden vectors in `src/wire_hash.rs`, so the two agree by
proof, not by import). This crate wraps it in a keyspace-partitioned
engine, a gRPC surface, and the client/bridge pieces that feed and
query it.

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
    state is bounded by idle TTL + per-holder capacity, cut
    recency-ordered: whole chains go least-recently-published first, so
    a cut drops cold prompts entire instead of shaving the freshest
    long-prompt tails off every chain and freeing no chain slot.
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
the same updates in any interleaving. Three mechanisms carry the state
across, cheapest first: a replica relays each accepted `Publish` to its
`--peers` best-effort; a starting replica `Pull`s a sibling's whole
state before going ready; and `--anti-entropy-secs` periodically
compares per-holder digests with each peer and re-pulls the holders
that disagree. The third exists because the first is lossy on purpose —
a wedged peer drops relayed updates rather than wedging ingest, and
once those deltas are gone nothing in the stream brings the two
replicas back together, so an event-fed holder would stay diverged
until its next epoch bump. A placement-fed holder whose digests merely
disagree at the same watermark is deliberately left alone and bounded
by TTL instead: it is inferred state, and copying one replica's guesses
onto another is not convergence. Gateways stay stupid: one endpoint,
no fan-out, reconnect on failure.

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
  | `--anti-entropy-secs` | `15` | divergence backstop: compare per-holder digests with each peer this often and re-pull the ones that disagree. `0` disables it (so does an empty `--peers`) — note the asymmetry with `--sweep-interval-secs` below, where `0` aborts startup instead |
  | `--inferred-ttl-secs` | `180` | idle TTL for placement-fed holders |
  | `--event-ttl-secs` | `1800` | liveness backstop for EVENT-fed holders: silence past this soft-retires the holder (a lost departure signal must not leak it); `0` disables |
  | `--default-capacity-blocks` | unbounded | RUNAWAY PROTECTION for holders that never sent `Added` — nothing is cut until the holder passes 2x this value, and the cut then takes it back to 1x. The hysteresis is load-bearing: cutting back to the bound re-cuts on every subsequent apply, which pins a replica at 100% CPU behind the keyspace write lock. The cut is recency-ordered (whole chains, least-recently-published first), not depth-ordered, so it drops cold prompts rather than the freshest tails. Leave unbounded or set well above worker KV size: the placement feed carries no removal signal, so an index that races the worker's own eviction under-matches (measured); idle TTL is the freshness bound. |
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
- `radix-index-loadbench` — multi-publisher write-scaling bench against
  a live service (placement or `--events` feed, optional `--digest`),
  reporting apply rate and query-p99 isolation under load.
- `radix-index-dump` — Pull one replica's whole state and print, as
  JSON, a per-holder block count and an order-independent digest (a
  commutative fold, so it compares sets regardless of stream order).
  Flags: `--connect` (service URL, default `http://127.0.0.1:40000`).
  Two converged replicas print identical digests, so diffing two dumps
  is how a fault drill proves a partition actually healed rather than
  merely stopped erroring.

All five binaries validate their flags strictly (`--flag value` or
`--flag=value`; an unknown flag or unparsable value aborts startup).

A reference StatefulSet lives in [`deploy/statefulset.yaml`](deploy/statefulset.yaml).

## Gateway side (SMG)

`--kv-indexer-url` + `--kv-indexer-block-size` on the gateway enable
the remote path: a routing-time overlap prefetch (2 ms deadline,
fast-fail while disconnected) feeding the cache-aware policy, and a
placement publish of the prompt⊕output chain after each completed
request. Flag off = every code path byte-identical to local behavior.

## Metrics

`/metrics` (Prometheus text), in full:

- Gauges `radix_index_{keyspaces,holders,event_fed_holders,dropped_holders,blocks}`.
- Counters `radix_index_{applies,queries,relay_dropped}_total`.
- Counters `radix_index_anti_entropy_{rounds,holders_pulled,failures}_total`
  — the only outward sign that the divergence backstop is alive. A flat
  rounds counter means the replicas are running on best-effort relay
  alone, which no other signal reports. A failures counter climbing in
  step with rounds means a peer is unreachable and no pull is landing.
- Histogram `radix_index_capacity_cut_duration_seconds` — the capacity
  cut's cost. Cuts run under the keyspace write lock, so this is the
  stall queries queue behind, not background work. `_count` is how often
  the fleet is over capacity and `_sum` is how much time that spends
  holding the lock; the buckets give a tail that a since-boot maximum
  could not, because a maximum never comes back down once one slow cut
  has pinned it.
- Histograms `radix_index_{apply_batch,query}_duration_seconds`, ENGINE
  time only: transport, decode, queueing and the ack are all outside the
  clock, so a gateway-side query timeout never appears here. The apply
  histogram times a whole batch, so divide `_sum` by
  `radix_index_applies_total` for a per-update average.

## Lanes and intervals

A worker's cache is not always one contiguous prefix. Hybrid models keep
several independently evicted position sets per worker (full attention, a
sliding window that frees its old blocks, recurrent state saved at
checkpoints), and the reusable prefix is the largest position every set
accepts at that position. The index stays engine-neutral about that:

- A **lane** is one independently evicted set, addressed as its own holder
  named `worker#lane`, with the publisher's opaque description on
  `Added.metadata`; the index keeps it, carries it in snapshots and echoes
  it on every answer (`HolderScore.lane_meta`), and never parses it.
  Lifecycle control addressed to `worker` fans out to `worker#*`, because
  a publisher that split a worker into lanes never sees the gateway's
  add/drop for the bare worker name. Nothing in this repository publishes
  more than one holder per worker today: splitting an engine's event
  stream into lanes is the publisher's job, and none does it yet. The
  storage and answer shape land first so the wire stays additive when one
  arrives — read the lane fields as reserved, not as live.
- Answers carry every covered run of the query per holder
  (`HolderScore.intervals`) next to the contiguous depth (`matched_blocks`),
  from one `coverage` walk. The reuse rules (full attention = coverage from
  0, window = `W-1` tokens before the candidate, checkpoint = a block end at
  the candidate, candidates aligned to the lanes' blocks) are the gateway's
  routing policy, not this crate's: the index ships the covered runs and
  holds no opinion about which engine can reuse them.
- Snapshots (bootstrap, anti-entropy) ship a holder per run, path-prefixed
  with placeholder blocks the same snapshot removes, so a run that starts
  past position 0 lands at the same positions on the same lineage; replica
  digests are position-bound.
