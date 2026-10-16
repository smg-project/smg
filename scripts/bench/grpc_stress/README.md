# gRPC router stress bed (CPU only)

A repeatable stress test of the gateway's gRPC request path with no GPU and
no engine: the release gateway (`cache_aware`, KV events on) in front of
`mock-worker --engine realistic` gRPC engines, driven by an open-loop client,
with the gateway's CPU per request and per streamed frame, its memory before
and after each cell with the slope over the cell, its allocation sites, and
the client's latency percentiles and error counts as the output. It answers
the question "what does the router itself cost, and where does it clone,
allocate and copy" for eight traffic shapes.

## Pieces

| file | what |
|---|---|
| `bed.sh up\|down\|status` | the fleet (12 realistic gRPC engines with KV events, an admin API) and the gateway, registered and probed; the gateway can be pinned to a cpu list and runs in its own transient systemd user scope when available, so its cgroup memory can be sampled |
| `stress_load` (bin of `crates/mock_worker`) | the open-loop client: `--mode chat\|tools\|responses\|image`, `--stream`, `--rate`, deterministic prompts with shared heads, tenants, tool-heavy conversations, `previous_response_id` chains, inline images; TTFB/TTFT/e2e percentiles, frames and inter-frame gaps, status and error counts |
| `cell.sh <name> -- <client args>` | one cell: the sampler on the gateway, the client pinned elsewhere, optional `perf record -g` and heap-profile windows inside the sampled window, the metrics before and after |
| `sample.py` | once a second: the gateway's CPU time, RSS (anon/file), threads, fds, cgroup `memory.current`/anon, and every 5 s the allocator gauges and request counters from `/metrics` |
| `run_all.sh [family ...]` | the matrix below, each family on a bed of its own, cells interleaved and repeated |
| `report.py <run dir>` | the tables: CPU ms/request, CPU us/frame, cores, latency percentiles, anon RSS start/end and slope (MB/min), peak RSS, jemalloc live bytes, shed counts; and the mean over repeats |
| `perf_sites.py` | folds `perf script` output into top self symbols, the share of cycles in the allocator / memcpy / `Clone` impls, and the sites behind them (first frame outside the allocator or libc, with its caller) |
| `heap_sites.py` | diffs two jemalloc heap profiles (`POST /heap_profile` at the window's start and end, `prof_accum:true`) into allocation sites by bytes and by objects, unbiased for the sampling interval, symbolized with `addr2line` |
| `gen_images.py`, `gen_tenants.py` | the 4 MB noise PNGs of the image family and the 200-tenant rate-limit YAML |

## Families (`run_all.sh`)

| family | cells | what it stresses |
|---|---|---|
| `unary` | 1k / 3k / 5k req/s, 64-token prompts, 8 output tokens | the per-request path: parse, template, tokenize, select, dispatch, assemble |
| `stream` | 500 / 1,500 req/s x 64-token SSE streams | the per-frame path: chunk decode, SSE framing, inter-frame gaps |
| `idle`, `idle-default` | 10,000 streams opened against frozen engines, held 30 s, then released (`--max-concurrent-requests 20000`, and the default admission bound) | memory per open stream; shedding at the default bound |
| `long` | 32k and 128k-token prompts at 50 req/s | the clone and copy cost of big bodies, tokenizer throughput |
| `image-router`, `image-worker` | 4 MB inline PNGs at 100 req/s, media preprocessed by the gateway or passed to the worker (needs `LLAVA_TOKENIZER`, a `llava-hf` model directory) | media decode, pixel buffers, body copies |
| `tenants` | 1k req/s over 200 header-resolved tenants with per-tenant limits | tenant resolution and rate-limit accounting |
| `tools` | 500 req/s of 10-turn tool conversations with 8 schemas, unary and streaming, `--parser symphony` | template rendering of tools, tool-call parsing |
| `responses` | 500 req/s of `previous_response_id` chains of 10 (`store: true`, memory backend) | the response store |

Each cell is 120 s after a 10 s warm-up, repeated 3 times, interleaved with
its family's other cells; the second repeat carries a 30 s `perf record -g`
of the gateway. With `HEAP_LG_SAMPLE=<n>` every family runs once more (60 s cells; the idle
cells keep their 20 s opening window) on a gateway built with `--features jemalloc-profiling` and started with
profiling on, and `heap_sites.py` turns the two dumps into allocation sites.

## Running it

```bash
cargo build --release -p smg -p mock-worker --bin smg --bin mock-worker --bin stress_load
TOKENIZER=Qwen/Qwen3-0.6B GW_CORES=0-7 MOCK_CORES=8-23 GEN_CORES=24-39 \
  scripts/bench/grpc_stress/run_all.sh unary stream
python3 scripts/bench/grpc_stress/report.py target/grpc-stress
perf script -i target/grpc-stress/cells/stream-s500-r2/perf.data | scripts/bench/grpc_stress/perf_sites.py
```

For readable profiles build the gateway with frame pointers and line tables
(`RUSTFLAGS="-C force-frame-pointers=yes" CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`,
and `CFLAGS=-fno-omit-frame-pointer` for the allocator), as the nightly
workflow does. The numbers are comparable only between runs on the same host
with the same pinning; compare a change against `main` by running both
binaries through the same cells, interleaved.

Ports: gateway 29600, metrics 29601, admin 29609, workers 29610 and up
(override with `GW_PORT`, `PROM_PORT`, `ADMIN_PORT`, `MOCK_BASE`).
