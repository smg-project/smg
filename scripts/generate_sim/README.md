# /generate scale simulation

Local-only reproduction of the sanitized production `/generate` workload
(design: `.claude/generate-scale-sim/01-design.md`): K SMG replicas in front of
a mock worker fleet (`crates/mock_worker --engine sim`), driven by the
open-loop `sim-loadgen` crate. No Kubernetes, no production access.

## Quick start

```sh
# Smoke run on a laptop (~40 req/s, 240 workers, 2 min):
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
| 9000 .. 9000+workers-1 | mock workers (full profile: 9000–26999) |
| 30000 .. 30000+K-1 | SMG data ports |
| 39000 .. 39000+K-1 | SMG prometheus ports |

Everything binds 127.0.0.1. Runs `pkill` leftover `smg`/`mock-worker`/
`sim-loadgen` processes started from the same binary paths before and after,
so ports are free across runs.

## Profiles

Plain JSON consumed by `sim.py`; `profiles/local-small.json` is the schema by
example. Every unknown production property is an explicit knob — never edit
the harness to change the workload:

- top level: `smg_count`, `workers_total`, `mock_processes`, `duration_secs`,
  `warmup_secs`, `sample_interval_secs`, `sample_fds`, `readiness_*`.
- `mock`: forwarded to `mock-worker` as `--key-with-hyphens value`
  (`sim_itl_ms`, `prefill_tps`, `max_running`, `kv_tokens`, image flags, ...).
- `smg_flags`: the literal gateway argv tail — the design doc's
  production-equivalent cache_aware set. The harness adds only
  `--host/--port/--prometheus-*`; it never passes `--enable-igw`.
- `loadgen`: forwarded to `sim-loadgen` the same way (`session_rps`,
  `t2_ratio`, `think_secs`, `system_prefix_tokens`, `prompt_cdf`/`output_cdf`,
  `image_bytes`, `image_count`, `routing_key_reuse`, `ingress`,
  `turn2_ingress`, `tokens_hint`, `stream`, `http2`). Booleans are
  value-style (`--stream true`), matching mock-worker's `--prefix-cache`.
  The harness adds `--smg-urls`, `--duration-secs`, and `--out-dir` (the run
  dir, where `requests.jsonl` and `summary.json` land).

Aggregate request rps = `session_rps × (1 + t2_ratio)`; the local profiles
compress time 10× (`sim_itl_ms` 4.3, `prefill_tps` 80000 → ~8.9 s mean
lifetime) and bodies 10× (`image_bytes` 62000) together, per the design doc.

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
