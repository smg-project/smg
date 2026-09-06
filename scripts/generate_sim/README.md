# /generate scale simulation

Local-only reproduction of the sanitized production `/generate` workload
(design: `.claude/generate-scale-sim/01-design.md`): K SMG replicas in front of
a mock worker fleet (`crates/mock_worker --engine realistic`, SGLang-native
`/generate`), driven by the
open-loop `sim-loadgen` crate. No Kubernetes, no production access.

## Quick start

```sh
# Smoke run on a laptop (24 workers, 2 SMGs, 40 s):
python3 scripts/generate_sim/sim.py run \
    --profile scripts/generate_sim/profiles/smoke.json

# Laptop profile at production-like worker pressure (~450 req/s, 120 workers, 2.5 min):
python3 scripts/generate_sim/sim.py run \
    --profile scripts/generate_sim/profiles/local-small.json

# Reuse existing binaries, tweak a knob without editing the profile:
python3 scripts/generate_sim/sim.py run --profile ... --skip-build \
    --override loadgen.ingress=random --override smg_count=1

# Rebuild the report for a finished (or aborted) run:
python3 scripts/generate_sim/sim.py report --run-dir target/generate-sim/<run>

# Named comparisons (side-by-side compare.md):
python3 scripts/generate_sim/scenarios.py list
python3 scripts/generate_sim/scenarios.py compare \
    --scenario stable-key-vs-random-ingress \
    --profile scripts/generate_sim/profiles/local-small.json
```

A run builds `smg`, `mock-worker`, and `sim-loadgen` (release), launches the
fleet, registers every worker with every SMG (`POST /workers`,
`disable_health_check`, 64-way per SMG), gates on `GET /workers` reaching 99%,
warms up, drives the load, samples each SMG every 5 s (RSS/%CPU via `ps`, fds
via `lsof` or `/proc`, plus `/metrics` admission-queue and connection gauges),
then tears everything down and writes `report.json` / `report.md` into the run
dir (default `target/generate-sim/<profile>-<timestamp>/`, with per-process
logs under `logs/`).

## Port plan

| range | use |
|---|---|
| 9000 .. 9000+workers-1 | mock workers (template `workers_total` 10000: 9000–18999) |
| 30000 .. 30000+K-1 | SMG data ports |
| 39000 .. 39000+K-1 | SMG prometheus ports |
| 40000 .. 40000+replicas-1 | index replicas (profiles with `index_service`) |
| 40100 .. | index replica metrics / readiness |
| 40200 .. | severable peer proxies (`index_service.partitionable`) |
| 41000 .. 41000+K-1 | gateway mesh ports (`mesh_smgs`) |

Everything binds 127.0.0.1. Runs `pkill` leftover `smg`/`mock-worker`/
`sim-loadgen`/`radix-index-service`/`radix-index-bridge` processes started
from the same binary paths before and after, so ports are free across runs.

## Profiles

Plain JSON consumed by `sim.py`; `profiles/local-small.json` is the schema by
example. Every unknown production property is an explicit knob — never edit
the harness to change the workload:

- top level: `smg_count`, `workers_total`, `mock_processes`, `duration_secs`,
  `warmup_secs`, `sample_interval_secs`, `sample_fds`, `readiness_*`.
- `mock`: forwarded to `mock-worker` as `--key-with-hyphens value`
  (`decode_base_ms`, `decode_per_req_ms`, `prefill_tps`, `max_running`,
  `kv_tokens`, `block_size`, `prefix_cache`). Image bytes in request bodies
  are payload only: the engine counts prompt tokens, not images.
- `smg_flags`: the literal gateway argv tail — the design doc's
  production-equivalent cache_aware set. The harness adds only
  `--host/--port/--prometheus-*`; it never passes `--enable-igw`.
- `loadgen`: forwarded to `sim-loadgen` the same way (`session_rps`,
  `t2_ratio`, `think_secs`, `system_prefix_tokens`, `prompt_cdf`/`output_cdf`,
  `image_bytes`, `image_count`, `routing_key_reuse`, `ingress`,
  `turn2_ingress`, `tokens_hint`, `stream`, `http2`). Booleans are
  value-style (`--stream true`), matching mock-worker's `--prefix-cache`.
  The harness adds `--smg-urls`, `--duration-secs`, and `--out` (the run
  dir, where `requests.jsonl` and `summary.json` land).
- `index_service` (optional): launches the shared radix-index service and,
  by default, its event bridge — `replicas`, `bridge`, `inferred_ttl_secs`,
  `event_ttl_secs`, `default_capacity_blocks`, `sweep_interval_secs`,
  `apply_delay_stored_ms` / `apply_delay_removed_ms` (staleness injection),
  `partitionable` (peer traffic via severable proxies), `deferred_replicas`
  (listed as peers, launched by a drill). Needs the `smg-radix-index`
  package in the tree; the build step adds it only when this block is set.
- `mesh_smgs`: gateway-to-gateway TreeSync mesh (one mesh port per SMG).

## Fault drills

Top-level profile keys, each firing once mid-run on its own thread; every
drill records epoch-ms timestamps of what it did in `meta.json` (surfaced in
`report.md` under "Drills"), and a drill that fails records `<drill>_error`
instead of dying silently. Any key that looks like a drill but is not one of
these fails the run at start, so a leg can never silently measure nothing.

| key | shape | effect |
|---|---|---|
| `restart_smgs_at_secs` | seconds | kill and relaunch every SMG (sticky pins and placements lost) |
| `kill_index_replica` | `{at_secs, replica, relaunch_after_secs?}` | SIGKILL a replica; relaunch bootstrapping from a survivor |
| `flap_index_replica` | `{at_secs, replica, cycles, period_secs}` | kill + relaunch the replica `cycles` times |
| `hang_index_replica` | `{at_secs, replica, resume_after_secs?}` | SIGSTOP (TCP up, nothing drains), then SIGCONT |
| `start_deferred_replica` | `{at_secs, replica}` | launch a `deferred_replicas` member under load, bootstrapping from replica 0 |
| `partition_drill` | `{at_secs, heal_after_secs?}` | sever every inter-replica link (needs `partitionable`), then heal |
| `remove_workers_drill` | `{at_secs, count}` | deregister the last `count` workers from every gateway (DELETE /workers); listeners stay up |
| `add_workers_drill` | `{at_secs, count}` | start `count` NEW workers on fresh ports and register them with every gateway |
| `restart_one_smg_drill` | `{at_secs, smg}` | kill ONE gateway and relaunch it cold, re-registering the fleet with it |
| `rolling_replica_restart_drill` | `{at_secs, gap_secs}` | kill and relaunch every replica in turn, each bootstrapping from a live peer |
| `gateway_partition_drill` | `{at_secs, heal_after_secs?, scope: all\|half}` | sever the gateways' link to replica 0 (needs `gateway_proxy`; `half` = even-numbered gateways only) |

`index_service.anti_entropy_secs` is forwarded to the service (peer digest
exchange period; 0 disables it). Two more `index_service` keys support the audits: `gateway_proxy` routes every
gateway's `--kv-indexer-url` through a severable proxy (even gateways one,
odd gateways another), and `dump_on_exit` pulls every live replica's state
with `radix-index-dump` before teardown and records per-holder divergence
(`index_divergence` in `meta.json` / the `replicas converged at end` row).

Rows the compare table adds for index legs: gateway-side lookup latency
p50/p90/p99 (what the 2 ms deadline is measured against), service-side apply
and query engine time p50/p90/p99, index replica peak RSS and mean CPU,
prediction error bias / exact share / p50 / p90 / p95 / max, and the
follow-up cache ratio in the first and last full minute (drift).

Scenarios `eval-matrix` (regimes × gateways × concurrency), `io-shapes`,
`chaos` and `soak` are the side-by-side evaluation; `production` in those
legs is the shipping configuration (hash placement index + sticky routing
key, HTTP workers).

`failover_bins.py <scenario-run-dir>` bins follow-up cache ratios around the
kill instant of a `kill_index_replica` run.

Aggregate request rps = `session_rps × (1 + t2_ratio)`; the local profiles
compress time 10× (`decode_base_ms` 4.3, `prefill_tps` 80000 → ~8.9 s mean
lifetime; `prefill_chunk` 320 keeps one step's prefill under one decode
step, since the engine's step time is max(prefill chunk, decode) — a
larger chunk stretches every decode step while any prompt is prefilling) and bodies 10× (`image_bytes` 62000) together, per the design doc.

## Mock engine fidelity notes

Two properties of `crates/mock_worker`'s realistic engine decide whether a
cache-aware result means anything, and both are set by this harness:

- `prefill_chunk` 320 (see the compression note below): the engine's step
  time is max(prefill chunk, decode step), so a large chunk stretches every
  decode step while any prompt is prefilling.
- Reported usage is the running requests' pinned tokens, not physical KV
  occupancy (SGLang's definition: used = total − available − evictable). A
  warm radix cache keeps KV physically full; reporting that as usage tripped
  the gateway's `--worker-overload-token-usage 0.9` gate on every warm
  worker and made cache-aware routing avoid the workers holding the
  prefixes (same-worker follow-ups 0.87 → 0.15 over 100 s).

## Load generator transport

The profiles drive the gateways over h2 (`loadgen.http2: true`), one
multiplexed connection per client per gateway, so a high request rate does
not exhaust ephemeral ports the way HTTP/1.1 connection churn does (~16k on
macOS; 600+ sessions/s over h1 hit it). This needs the gateway fix in #2488:
before it, every streamed `/generate` response relayed by the gateway ended
with a zero-length non-terminal DATA frame, and h2 ≥ 0.4.16 clients close a
connection after 100 of those — a 60% transport-error rate under load that
had nothing to do with routing. Set `loadgen.http2: false` to measure
against a gateway without that fix.

## File descriptors / ulimit

`sim.py` raises `RLIMIT_NOFILE` to the hard limit before spawning children
(they inherit it). If the hard limit itself is low:

- macOS: `sudo launchctl limit maxfiles 65536 1048576`, then a new shell; the
  per-process cap is `kern.maxfilesperproc`.
- Linux: raise `nofile` in `/etc/security/limits.conf` or run under
  `prlimit --nofile=1048576`.

Budget: each mock process holds its listeners plus accepted upstream conns;
each SMG holds ~1 h2c conn per worker (`--upstream-http2`) plus client conns.
The full profile needs ≥200k fds system-wide; `local-*` fit default-raised
laptop limits. `lsof`-based fd sampling is slow at high fd counts — the full
profile sets `"sample_fds": false`.

## Full-profile host sizing

`profiles/full.template.json` carries generic round placeholders — copy it to
`profiles/full.local.json` (gitignored) and fill in your fleet's real worker
count, request rate, timing, and body sizes; never commit those values. A
full-scale run belongs on a large Linux host, not a laptop. Scaling rules of
thumb for K SMGs, W workers, R req/s, mean body B, mean lifetime L:

- Aggregate body ingest ≈ R × B, all loopback.
- Concurrent client h2 streams ≈ R × L — run the loadgen with `http2: true`
  and size `conns_per_origin`; h1 would need one socket per stream.
- SMG→worker connections ≈ K × W (one h2c conn per pair with
  `--upstream-http2`).
- Registration is K × W `POST /workers` calls (64-way per SMG, SMGs in
  parallel); allow a generous readiness timeout.
- Body memory depends on the routing regime: with the sticky override and a
  valid routing key, bodies STREAM (`REASON_PURE_FORWARD`) and are not
  buffered; without it (or without a key) the typed path buffers each body —
  the `body path streamed share` report row verifies which regime a run was
  actually in.
- Resource conclusions (CPU/RSS/fds) are only meaningful at full scale;
  reduced-scale reports carry an explicit warning banner.

## Policy A/B

`scenarios.py compare --scenario policy-ab` runs the identical fleet and
workload twice, once per gateway binary. Build binary B from another checkout
with an **isolated** `CARGO_TARGET_DIR` so the two builds never trample each
other's artifacts (and A stays warm):

```sh
cd /path/to/other-checkout
CARGO_TARGET_DIR=/tmp/smg-ab-b RUSTC_WRAPPER= cargo build --release -p smg

python3 scripts/generate_sim/scenarios.py compare --scenario policy-ab \
    --profile scripts/generate_sim/profiles/local-medium.json \
    --smg-bin-a "$PWD/target/release/smg" \
    --smg-bin-b /tmp/smg-ab-b/release/smg
```

`--only LEG` (repeatable) runs a subset of a scenario's legs by label, for
reruns after a fix; an unknown label is an error rather than an empty run.

`--smg-bin` also works on plain `sim.py run` for one-off runs against a
prebuilt gateway. Mock workers and the loadgen always come from this
checkout, so only the gateway varies.

## What the report contains

- `report.md` / `report.json`: loadgen summary (TTFT/E2E percentiles,
  cache-hit rates), per-worker imbalance from `requests.jsonl` (fleet CoV,
  max/mean, distinct workers; split per turn), turn-2 same-worker rate,
  per-SMG cache-aware branch counts (`hash_hit`/`hash_spill`/fallbacks,
  parsed from the scoped `RUST_LOG=warn,smg::policies::cache_aware=debug`
  logs — branches are debug-log-only, there is no branch metric), and per-SMG
  resource samples (RSS, CPU, fds, admission-queue depth, active connections,
  rejected/selection counters).
- `samples.jsonl`: the raw 5 s samples; `meta.json`: pids, registration and
  readiness counts, timings.
