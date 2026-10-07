# kv_index benchmarks

| File | What it is |
| --- | --- |
| `throughput_bench.rs` | Criterion micro-benchmarks of the indexers (see its module doc). |
| `match_insert.rs` | Criterion benchmarks of single operations: a store that extends a shared chain, a store that diverges from it, a lookup over a chain many workers hold. |
| `churn.rs` | The churn bench: workers with block-LRU caches over a shared chain pool, as engines behave, with decode tails and restarts; records the index's shape over time and checks exactness against the reference indexer at every sample. |
| `mooncake_replay.rs` | Open-loop replay of an indexer corpus (an exported Mooncake-style trace schedule) against this crate's indexers: `cargo bench -p kv-index --bench mooncake_replay -- --help`. |

Build the replay once and run the binary directly, one process per trial:

```
cargo bench -p kv-index --bench mooncake_replay --no-run
target/release/deps/mooncake_replay-<hash> --help
```

## The replay's definition

The replay follows the open-loop indexer benchmark definition that KV routers are compared on: a
trace of requests and KV-cache events is replayed with its deadlines scaled into a window; lookups
run on query lanes and events on event lanes, sharded by worker; throughput is counted in block
operations (requested, stored and removed block hashes) over the time from the start of issue to
the last completion, drain included; latency is the lookup service time measured at each lane.

- Query lane = `worker_id % query_lanes`. Event lane = round robin over `(worker_id, dp_rank)` in
  order of first appearance. Event issuers own contiguous worker ranges (or, with
  `--issuer-by-lane`, the workers of a lane range); query lanes are sharded over the query issuers
  in contiguous ranges.
- Before the trial: `malloc_trim`, a quiescence sleep (`--pre-run-quiescence-ms`, default 5000),
  a pass over the corpus pages, thread pinning (`--issuer-cpus`, `--query-issuer-cpus`,
  `--backend-cpus`).
- Issue: each issuer sleeps to the absolute monotonic deadline and spins the last
  `--issuer-spin-us`; the queries of a deadline are published before its events.
- Validity: the generator is valid when nothing failed and the issue span is at most 1.01 × the
  window; `kept_up` additionally needs the last completion within 1.10 × the window. A sustained
  point (see Sustained throughput below) asks for more: a valid generator and at least 99% of the offered
  block ops achieved.
- Rates: `achieved_block_ops_per_sec = total_block_ops / (last_completion − start)`, drain
  included; `offered` divides by the window; `actual_issue` by the issue span. Percentiles are
  nearest-rank (p50, p99, p99.9, max).
- `--owned-payloads` (default on) charges the lanes what a harness that hands each lane an
  owned event charges every backend: each event arrives as an owned payload (40 bytes per block,
  allocated before the trial) that the lane converts into this crate's blocks and frees after the
  apply, and each lookup copies its hashes into this crate's hash type. With it off, lanes read the
  corpus slabs and copy nothing: the cheapest way to drive a backend, not a comparable number.
  `--payload-home issuer` builds those payloads on the issuing thread instead of the main thread,
  so on a multi-socket machine the payloads live on the issuer's socket.

Backends (`--backend`): `positional` (`PositionalIndexer`), `chain` (`ChainIndex`, or
`ShardedChainIndex` with `--shards N`; `run` is its deprecated spelling), `reference` (the
single-threaded exactness reference; small corpora only) and `null` (no indexer: the harness's own
ceiling on a layout). A new index plugs in by implementing `ReplayBackend`.

### Corpus format `SMGMCK01`, version 1

All integers are little-endian. One file holds one prepared schedule for one window; deadlines are
already scaled to that window and are rescaled linearly for another (`--benchmark-duration-ms`) or
for an offered rate (`--offered-block-ops-per-sec`, which sets the window from the corpus's
block-op total).

| Section | Layout |
| --- | --- |
| Magic | 8 bytes `SMGMCK01` |
| Header | u32 version (1), u32 block_size, u64 reference_window_ns, u64 trace_duplication_factor, u64 trace_length_factor, u64 inference_worker_duplication_factor, u64 logical_workers (max worker id + 1) |
| Totals | 7 × u64: requests, stored_events, removed_events, cleared_events, request_blocks, stored_blocks, removed_blocks |
| Trace path | u64 length, UTF-8 bytes (informational) |
| Query hashes | u64 count, count × u64 local block hash |
| Stored blocks | u64 count, count × (u64 block_hash, u64 tokens_hash) |
| Removed hashes | u64 count, count × u64 block_hash |
| Operations | u64 count, then per operation: u32 id, u64 deadline_ns, u64 worker_id, u8 kind, kind-specific fields |

| kind | Fields |
| --- | --- |
| 0 query | u64 start into the query hash slab, u32 length |
| 1 stored | u32 dp_rank, u64 event_id, u8 has_parent, u64 parent (0 when absent), u8 has_start_position, u32 start_position (0 when absent), u64 start into the stored-block slab, u32 length |
| 2 removed | u32 dp_rank, u64 event_id, u64 start into the removed-hash slab, u32 length |
| 3 cleared | u32 dp_rank, u64 event_id |

Operations appear in issue order (deadline, queries before events at equal deadlines, worker id,
trace order) and ids are their dense positions; the loader verifies both, and the totals. The
export is deterministic: two exports of the same arguments hash identically.

### Flags

| Flag | Meaning |
| --- | --- |
| `--backend`, `--jump-size`, `--max-workers`, `--shards`, `--lane-memory inherit\|local` | The index under test; `--shards` places each pinned event lane's workers on the shard of the lane's NUMA node; `--lane-memory local` makes a pinned lane prefer its own node for what it allocates. |
| `--benchmark-duration-ms` or `--offered-block-ops-per-sec` | The window, or the offered rate that sets it. |
| `--query-lanes`, `--event-lanes`, `--issuer-threads`, `--query-issuer-threads` | Lane and issuer counts. |
| `--issuer-cpus`, `--query-issuer-cpus`, `--backend-cpus`, `--pin-event-lanes`, `--issuer-by-lane` | Placement. Issuer CPUs inside the lane set are refused. |
| `--owned-payloads`, `--payload-home main\|issuer` | The owned-payload cost model above. |
| `--lane-scheduling owned\|stealing`, `--steal-after` | How event lanes share work: each lane applies its own workers only, or the event lanes of each shard form one lane pool that serves whole workers and takes a worker from a lane that is not running once its backlog is `--steal-after` events deep; the pools' counters go into the result. |
| `--queries on\|off`, `--lookups all-shards\|per-shard`, `--count-shard-heads` | Diagnostics: events only; one lane per shard merged by the last to finish; a count of lookups by how many shards hold the request's first block. |
| `--result-json-output` | The result record: rates, lookup percentiles, queue depths, per-lane CPU and completion times, the layout, and a provenance object (argv, binary and corpus hashes, trace parameters). |

The first line of every log states the issuer CPUs, the query-issuer CPUs and the lane CPU set.

## Sustained throughput

Sustained throughput is the highest offered rate at which a trial keeps up: generator valid and
at least 99% of the offered block operations achieved. A window-driven replay only says "keeps up
at this window", so the threshold is bracketed: start from a rate that keeps up and one that fails,
run several fresh processes at the geometric midpoint (every one of them must keep up for the point
to pass), and move the bracket until it is within 10%. `--offered-block-ops-per-sec` is the knob;
one process per trial. A published point is then a series of fresh-process trials at the kept-up
rate, each followed by a control trial of the same binary and configuration (so the pair shows the
noise floor an A/A comparison would show), summarised as medians with bootstrap confidence
intervals of achieved block ops/s and lookup p50/p99, with the trials that overlapped foreign load
on the measurement cores discarded and replaced. Capacity (the achieved rate when overloaded) is
compared only within one harness and layout. Every point is reported with the lookup percentiles
measured at that load.

## Exactness

Every indexer answers the same question as the single-threaded `ReferenceIndexer`
(`src/reference.rs`): after any replay, the set of (worker, position, block) and every lookup
score equal the reference's. That is checked three ways:

- `tests/exactness_positional.rs` and `tests/exactness_chain.rs` replay seeded corpora (new
  conversations, extensions, siblings diverging at any position, tail and middle evictions,
  clears, worker removal and arrival) into the positional and the chain index beside the
  reference, compare every lookup kind after every 256 events and the full block set at the end;
  `KV_INDEX_EXACTNESS_EVENTS`, `KV_INDEX_EXACTNESS_SEED` and `KV_INDEX_EXACTNESS_SHARDS` scale,
  reseed and shard them;
- `tests/concurrency_chain.rs` runs 16 event lanes and 4 readers with worker replacement and
  replays every lane's log into the reference at the end;
- the churn bench and `tests/churn_gate.rs` check the chain index against the reference at every
  sample while runs split and die.

In the replay, `--backend reference` runs the reference itself on a small corpus so a backend's
result record can be compared with it.

## Churn

`cargo bench -p kv-index --bench churn -- --help`: workers, chains, cache blocks, decode blocks
per request, eviction order (tail-first or by hash), restart schedule, request count and sample
interval. The series it prints per sample: lookup p50/p99, runs walked per lookup, runs and blocks
live, mean run length, splits by cause, mergeable adjacent pairs, memory, and the exactness
verdict. `--json <file>` writes the series and the run's figures as JSON; nothing is written
without it.
