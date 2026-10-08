# kv_index: the KV-event index behind cache-aware routing

The gateway's cache-aware routing keeps an index of which worker holds which prefix of which
prompt. The engines publish it: every scheduler step, vLLM and SGLang emit the block hashes they
stored, the hashes they evicted and the occasional full clear, over a ZMQ publisher that the
servicer relays into a gRPC stream the gateway subscribes to. A request is hashed into the same
per-block chain, the index answers how many leading blocks of that chain each worker holds, and
the policy credits the overlap when it picks a worker.

This document describes the index this crate provides, the relay that feeds it and the routing
changes around it. Benchmarks and how to run them: `benches/README.md`.

## The problem with one entry per block

The positional indexer (`src/event_tree.rs`, the default) keys one entry per (position, content
hash) with the set of holding workers, and a lookup probes one entry per request block. It is
exact (its jump search was not: the first commits of this series fixed it against the reference
indexer and made it verify every position), but exactness costs it one dependent memory access per
block per lookup, the per-entry worker set grows with every holder, and every store or removal is
a hash-map write per block under a shard lock. A fleet of engines sharing long prefixes makes all
three worse at once: the same chain is stored by many workers, requests are hundreds of blocks
long, and the engines evict and re-store at the rate they serve.

## The chain index

`ChainIndex` (`src/chain_index.rs`) stores the engines' block-hash chains run-length compressed.

**Chains and runs.** Every stored block names its parent, so each worker's blocks form chains and
the chains of a fleet form a trie keyed by block hash. The index stores that trie as runs: a run is
a maximal stretch of consecutive positions on one chain whose set of holding workers is the same at
every position. It stores one content hash and one engine hash per position (shared by every
worker that holds it) and, per run, a coverage bitset with one bit per worker slot plus a table of
partial holders, `(worker, cutoff)` entries for workers that hold a prefix of the run. A worker is
in the bitset or in the table, never both. A tail eviction lowers a cutoff or turns a full holder
into a partial one; a prefix join adds a partial entry; a decode that extends a prefix to the end
of the run promotes the worker back to the bitset. None of those split the run, so a popular
prompt prefix stays one run however many workers hold different lengths of it.

**Children at any offset.** A chain that diverges from a run becomes a child keyed by the offset
and the next block hash in the run's child table (an open-addressing table in the arena, so the
root with tens of thousands of children inserts in constant time). The run itself is not split by
the divergence. This is what keeps the index's shape stable under the engines' churn: a request's
decode blocks are private content stored under the prompt's last block, they diverge from the
prompt chain at the prompt's end, and they die when the engine evicts them; as children they come
and go without leaving a boundary in the prompt chain. Splits remain for the two cases that need
one: a hole (a worker evicted a block in the middle of a chain it otherwise keeps, so the coverage
genuinely changes at that position) and a store whose parent sits inside a run (a stale parent).

**Holes are exact.** Because coverage is uniform within a run, a worker that evicts a middle block
stops covering the piece that holds it and nothing else; a lookup stops there for that worker and
the chain matches up to the hole and no further, which is what the engine would serve. A later
store from the hole on heals it.

**Lookups write nothing.** A lookup walks from the root, compares the request's content hashes
against each run on the path (one compare loop per run, not one probe per block) and ANDs the alive
set with the run's coverage; a worker scores the position where it drops out, and its partial
cutoff is applied when it has one. Readers take no locks, allocate nothing but the result map (or
nothing at all through `score_into`) and store to no shared memory: every run header, hash array
and child table lives in an arena addressed by integer ids, and a run's window (hash array, base,
length, coverage, partial table, child entry) is read under a seqlock version that is checked
again after the reads, so a split, a growth, an unlink or a reuse of the run is atomic to a
reader. Stale reads before the version check are clamped and probed with bounds checks, never
indexed.

**Writers lock one run at a time.** A store walk descends through runs under their versions and
locks a run only when it has to change it: a join, a promotion, a split, an in-place append of the
worker's own leaf (one lock, no allocation) or a new child, which is claimed in the child table by
compare-and-swap and published with a release store, so a new chain costs the root no lock. A
walk whose plan is invalidated by a concurrent split gives up and restarts from the parent block.
A run is matched by the engine hash at the end of a window, with a bisection to the first
differing position on a mismatch; the content hash at the landing is checked, and a store that
fails it is placed by content, block by block. Dead run headers, hash arrays (reference-counted
across the runs a split leaves sharing one) and child tables return to lock-free free lists; a run
id carries a generation so a stale pointer to a reused id is recognised. Hash arrays come in
size classes a quarter of an octave apart with a best-fit search, so an array wastes at most a
quarter of its words to its class.

**Lanes.** Events of one worker apply in the order the engine sent them, so the unit of
scheduling is the worker. Each event lane keeps, for the workers it owns, a lane map
(`src/lane_map.rs`): an open-addressing table from engine hash to (run, offset) with 16-byte slots
and a tag byte per slot, Fibonacci home slot, linear probing over the tags, backward-shift deletion
so steady churn never accumulates tombstones, and the home slots of the next keys prefetched a few
keys ahead so one event's cache misses overlap. The lane pool (`src/lane_pool.rs`) gives every
worker a bounded FIFO queue, puts a worker with queued events on exactly one lane's ready list, and
lets any lane steal a whole ready worker from a lane busy with another one, or from a lane that has
stopped running once a worker's backlog behind it is `steal_after` deep; `enqueue` never blocks
and never drops (at the cap the event comes back to the producer, which holds its cursor). The
lanes are the caller's threads, so a harness keeps its pinning and accounting.

**Shards.** `ShardedChainIndex` (`src/sharded.rs`) holds one complete chain index per shard, each
written only by the lanes assigned to it; a worker id carries its shard, stores and removals go to
the worker's shard, and a lookup probes every shard's root and walks the shards that hold the
request's first block, reporting each worker once. The worker sets are disjoint, so the union of
the shards' scores is the single index's answer whatever the assignment. One shard is the chain
index behind one indirection, byte for byte. On a machine with several memory nodes, one shard per
node removes every cache line the event lanes of different nodes would otherwise write in common.

**Memory.** 16 bytes per distinct block on a chain (its content hash and its engine hash), shared
by every worker that holds it, plus a run header, the coverage words and a child table per
branching run, against the 16-byte lane-map slot each lane keeps per held block for removals. The
index holds exactly what the engines report and shrinks with their removals; nothing in it ages or
is pruned.

**Exactness.** `src/reference.rs` is a single-threaded reference indexer kept as literal as
possible: per worker, blocks by engine hash and by (position, content hash) with the prefix hashes
that reach them, stores placed after the parent, removals and clears by forgetting, a lookup that
scores the longest prefix matched position by position. Every indexer in this crate is a function
of the event stream and must equal it after any replay, on content and on every score; the tests
and benches in `tests/` and `benches/` check that on seeded corpora with holes, under concurrent
lanes with worker replacement, under churn, and, in the gateway, on recorded engine streams through
the gateway's own apply path. Guardrail tests also check that a burst of lookups leaves the index's
counters and content as they were and that a clear and a removal return every arena word but the
root's child table to a free list.

## The relay

The servicer relays the engine's publisher into `SubscribeKvEvents` through one relay per engine
(`crates/engine_servicer/src/kv_events.rs`, `kv_wire.rs`, `kv_history.rs`, `kv_state.rs`,
`engine_hash.rs`, `load_tracker.rs`; the Python servicers mirror the decoding and normalization in
`smg_grpc_servicer/kv_relay.py`).

- **Wire and normalization.** Both engines' layouts (tagged maps and the older tag-first arrays)
  decode into one model; a field that cannot be read costs that event, not the batch. Per stream,
  the normalizer keeps only local, engine-managed blocks of the main-attention cache group, maps
  the medium to a tier (device, host, disk, external), drops placeholders, unaligned and
  self-referencing stores, folds speculative-decoding bigram pages to their tokens, carries the
  cache namespace (LoRA name, cache salt) on every store and down the parent chain, and counts
  every drop by reason. Stores and removals are forwarded one for one; the gateway counts physical
  copies per worker, rank and tier, so vLLM's duplicate copies and SGLang's host copies evict a
  block only when no copy remains.
- **Hash check.** The relay can rehash every admitted store the engine's own way (vLLM's
  `sha256_cbor` chain and SGLang's per-page SHA-256 chain are reproduced) and count matches,
  mismatches and unverifiable blocks, so a worker with another algorithm, seed or page size shows
  as a rate instead of silent misses (`SMG_KV_EVENT_HASH_CHECK`).
- **History and resume.** The relay subscribes once, at the servicer's start, for the servicer's
  lifetime (`SMG_KV_EVENT_RELAY_START=lazy` defers it), keeps the last `SMG_KV_EVENT_HISTORY_BATCHES`
  batches within `SMG_KV_EVENT_HISTORY_BYTES` under the publisher's own sequence numbers, serves a
  subscriber's cursor from it, asks the engine's replay socket for sequences its own socket missed,
  and answers a cursor below the window, beyond the newest sequence or before its own start with
  `OUT_OF_RANGE`, which makes the gateway clear and resubscribe from zero.
- **Snapshot.** The relay keeps the engine's live blocks per rank, tier and hash, folded from the
  stream it relays. A subscription from zero whose history no longer starts at the publisher's
  first batch receives a state snapshot: chunks marked `KvSnapshotChunk`, chunk 0 beginning with
  a clear, stores parent-first with the original fields, consecutive sequence numbers ending at
  the relay's cursor, cut atomically under the lock that admits batches; live events continue from
  the next sequence with no gap and no duplicate. A fresh gateway in front of warm engines starts
  with their whole resident cache.
- **Slow joiner.** A ZMQ subscriber sees nothing published before it joined, so the relay asks the
  engine's replay for everything from zero at start, after a detected publisher restart, and when
  a subscription from zero finds it holding nothing; what the replay no longer has is marked in the
  snapshot and the gateway keeps the rank degraded until the next resync.
- **Publisher restarts** are read by rules that hold on every wire (the sequence goes backwards on
  the socket, the engine's startup clear arrives under a passed sequence, the counter is back at 0
  or 1 under a larger cursor); each ends live streams with `DATA_LOSS` and starts a new incarnation.
- **Pushed loads.** Each servicer installs a load source over its own `GetLoads` figures; every
  batch a subscriber receives carries the engine's load, and while the publisher is quiet the
  stream sends load-only batches on change and as a heartbeat (1 s, backing off to 5 s). The vLLM
  servicers derive queued uncached token-work, generation throughput and hit rate from the requests
  they forward, the fields the gateway's expected-wait score drains.

## Routing and recovery in the gateway

- **Two backends, one type.** `KvIndex` (`model_gateway/src/worker/kv_index_backend.rs`) holds the
  positional indexer or the chain index, selected by `--kv-index {positional,chain}`; the monitor
  and the policy see the operations they used before. An in-gateway exactness test replays the
  recorded engine streams and a synthetic hole corpus through the monitor's own apply path into
  both backends and the reference indexer.
- **Per-rank admission.** One cursor per (worker, dp_rank): contiguous batches apply, duplicates
  skip, a gap gets one replay request and is otherwise settled (a small one keeps the blocks and
  marks the rank degraded, a large one clears the worker), a publisher restart clears the worker's
  state, `OUT_OF_RANGE` and `DATA_LOSS` reset everything, a snapshot chunk applies as a resync.
- **Liveness beside the health check.** A worker that dies or restarts fails every stream on its
  connection at once, and one that falls silent fails them at the keepalive timeout (the channel
  profile pings every 30 s, answered within 10 s: faster pings draw a `too_many_pings` GOAWAY from
  the engines' grpc-core servers) or at the next poll's deadline; a liveness tracker turns a
  connection failure on the KV stream or the load poll into an `unreachable` veto after
  `--worker-stall-secs`, vetoes a worker that holds requests and a growing queue without producing a
  token for `--worker-wedge-secs` as `wedged` (the clock starts at the first dispatch and the bound
  stretches to the prefill still in flight), re-admits a worker on its first contact with a closed
  circuit breaker, and never touches the health status. A veto steers, it never refuses: when every
  candidate is vetoed, the request goes to the least-loaded ready worker among them.
- **Warm-up slice.** A worker that just became routable, or whose index is thin against the
  fleet's (`--worker-warmup-thin-ratio`), receives a share of cache-miss traffic until its index has
  grown, so an empty index does not starve it while every prompt has a holder elsewhere.
- **Completion reporting and the selection layer.** Every request's end reaches the policy that
  placed it through the worker's load guard, on every path; booked state is reconciled with the
  live in-flight count each poll. Worker selection runs through a cost-function selection layer
  (`model_gateway/src/policies/cost/`) whose default reproduces the existing cache-aware decision;
  event-driven requests are hashed under their cache namespace.
- **Protection by default.** Worker overload protection is on as steering: a worker at or above
  `--worker-overload-waiting-requests` or `--worker-overload-token-usage` is left out of selection
  while another is under them, and a fleet uniformly over them is routed to its least-loaded worker;
  shedding stays opt-in (`--worker-overload-shed`).

## Flags and settings

Gateway: `--kv-index {positional,chain}` (default `positional`; `run` is accepted as a deprecated
alias of `chain`); `--kv-indexer-ttl-secs`, `--kv-indexer-max-entries` (positional only);
`--worker-stall-secs` (2), `--worker-wedge-secs` (3); `--worker-warmup-secs` (60),
`--worker-warmup-share` (0.25), `--worker-warmup-blocks` (1024), `--worker-warmup-thin-ratio`
(0.5); `--worker-overload-protection` (on), `--disable-worker-overload-protection`,
`--worker-overload-waiting-requests` (8), `--worker-overload-token-usage` (0.8),
`--worker-overload-shed` (off); `--selection-policy` (`cache-aware-default`),
`--selection-accounting-ttl-ms` (0);
`--load-monitor-interval` (10; a `GetLoads` poll goes only to a worker whose KV-event stream pushed no load
record within the interval, the poll being the fallback for servicers that do not push).

Servicers: `SMG_KV_EVENT_HISTORY_BATCHES` (10,000), `SMG_KV_EVENT_HISTORY_BYTES` (256 MiB),
`SMG_KV_EVENT_RELAY_START` (`lazy` to subscribe at the first gateway),
`SMG_KV_EVENT_HASH_CHECK` (`sglang` or `vllm-sha256-cbor`).

Metrics added: `smg_kv_index_lookup_seconds{index}`, `smg_kv_event_apply_seconds{worker}`,
`smg_kv_event_blocks_total{worker,op}`, `smg_kv_event_parentless_stores_total{worker}`,
`smg_kv_index_blocks{worker}`, `smg_kv_index_memberships` and `smg_kv_index_entries` per model,
`smg_kv_index_runs_live`, `smg_kv_index_blocks_live`, `smg_kv_index_arena_bytes`,
`smg_kv_index_arena_free_bytes`, `smg_kv_index_slab_bytes`, `smg_kv_index_moved_hashes`, `smg_kv_index_engine_conflicts`;
`smg_kv_event_batches_total{disposition}`, `smg_kv_event_gaps_total{outcome}`,
`smg_kv_event_resyncs_total{reason}`, `smg_kv_event_lag_seconds`, `smg_kv_event_degraded_ranks`,
`smg_kv_event_subscriptions_total`; `smg_worker_stalled{reason}`,
`smg_worker_stall_transitions_total`, `smg_worker_overload_fallback_total`,
`smg_policy_inflight_reconciled_total{policy}`, `smg_cache_aware_policy_branch_total{branch}`.

## Testing without engines

`crates/mock_worker --engine realistic` is a vLLM-style engine: a pass scheduler with a token
budget, a block-level KV pool with reference counts, LRU and LIFO preemption, admission only when
the whole prompt fits, tail-first frees, KV events per pass on the gRPC stream and on the engines'
own ZMQ wires (`--kv-events-zmq-base-port`, `--kv-events-wire vllm|sglang`), an admin API with
fault hooks (drop, delay, publisher restart, pause) and engine truth, and timing from a
calibration file. The mock worker's replay binary (`replay`) replays a Mooncake-style trace through the
gateway and scores every decision against the fleet's arrival-time oracle and the engines' own
cached-token counts. The mock worker's README describes both binaries' flags.
