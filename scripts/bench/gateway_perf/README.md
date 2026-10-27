# Gateway perf bed: the gateway's cost per request, base against head (CPU only)

What a change to the gateway costs or saves per request, measured on this machine with no
GPU and no engine: one release gateway in front of mock gRPC engines that answer from a
script, an open-loop client at a fixed rate, and the gateway process's own counters read
from outside it. The same bed and client as `scripts/bench/grpc_stress`; what this adds is
the A/B discipline (a fresh gateway per cell, an idle cell subtracted, base and head
interleaved, medians with their spread) and the counters that make a per-request number
exact: instructions and allocation calls do not move with the load of the machine, so two
runs of the same binary agree to a fraction of a percent, and a change of a few hundred
instructions or one allocation per request is visible.

## One command for a pull request

```bash
make perf-base                 # the base branch's gateway (origin/main; PERF_BASE_REV=<rev> for another)
make perf-alloc perf-cpu       # the head is built from the working tree; both runs print their tables
make perf-report PERF_OUT=target/gateway-perf/<stamp>/perf-cpu   # the Numbers table again, for the PR body
```

`perf-alloc` is the deterministic part (allocation calls and KiB per request, RSS) and takes
about a minute per path and side at the defaults; `perf-cpu` adds instructions and cycles
when `perf stat` can count them here (otherwise CPU time per request, which carries the
machine's load). The table each run prints is the `## Numbers` table of the pull-request
body: metric, before, after with the change, and the setup with the base's spread. Paste it.

## The targets

| target | what it measures | how |
|---|---|---|
| `make perf-build` | nothing: the release binaries. The gateway is built alone (`cargo build --release -p smg --bin smg`), then the mock engine and the client, because building the gateway together with another package changes its binary through feature unification; `perf-base` uses the same line | copies under `target/gateway-perf/head/` |
| `make perf-base` | the base revision's gateway binary, `target/gateway-perf/base/smg` | a detached worktree of `PERF_BASE_REV` built into its own target directory (a full release build; sccache shortens it) |
| `make perf-bed` | the bed comes up and answers: one request per path, the first response shown, the scripted answers' hit and miss counts | `gateway_perf.py bed smoke` (`bed up` / `bed down` keep one running for `grpc_stress/cell.sh` or by hand) |
| `make perf-alloc` | allocation calls per request, KiB requested per request (the allocator's size classes, small and large), allocations of 16 KiB and above per request, RSS and peak RSS after the window | jemalloc's exit statistics: the gateway is started with `_RJEM_MALLOC_CONF=stats_print:true,stats_print_opts:Jaxe` and prints them at its clean shutdown; no special build |
| `make perf-cpu` | user instructions per request, instructions (user + kernel), cycles, context switches, from `perf stat -p <gateway pid>` over the window; CPU time per request (utime + stime) always | hardware counters need `perf`, a kernel that allows them (`kernel.perf_event_paranoid` at 2 or below for a process of your own) and hardware that exposes them (a virtual machine often does not: the run then says so and keeps the CPU-time column) |
| `make perf-mem` | RSS after each of several windows on ONE gateway, the growth per 10,000 requests, the peak (`VmHWM`) after a burst window at twice the rate, the allocator's live bytes | `PERF_MEM_PATHS` (default `stream`, the path that holds the most per request), `--windows`, `--burst` |
| `make perf-live` | an open-loop ramp: offered rate against achieved rate, TTFT and end-to-end p50/p99, errors, gateway CPU per request and RSS per step, and the capacity | a fresh gateway per step; the capacity rule below; the base too when its binary exists |
| `make perf-report` | the tables of a run again (`PERF_OUT=<stamp>/perf-cpu`, the mode's directory) | `results.json` of that directory |

Variables: `PERF_OUT` (the run root, `target/gateway-perf/<stamp>`, one stamp per `make` invocation; each target writes under it, `<stamp>/perf-cpu`, `<stamp>/perf-alloc`, ...), `PERF_PATHS`,
`PERF_REQUESTS` (10,000 per cell), `PERF_REPEATS` (3 per side), `PERF_ARGS` (extra flags of
the driver, valid for the target's mode: the cell targets take the cell flags, `perf-bed` and `perf-live` the bed flags only, e.g. `--rate-scale 0.5 --gw-cores 0-7 --mock-cores 8-15 --gen-cores 16-23
--port-base 29700`), `TOKENIZER`, `PERF_GATES` (a `bench_gate` TOML for the verdict column).
`python3 scripts/bench/gateway_perf/gateway_perf.py <mode> --help` lists the rest.

## The paths

Every cell drives one path: 64 distinct deterministic prompts of 64 words (a shared head
from a pool of 16, a unique tail), the client's seed per path, and ONE scripted answer for
every request of the cell, handed to the mock engines before the cell starts (the worker's
admin queue) and served by the wire with no engine time: the numbers are the gateway's.

| path | request | answer | rate |
|---|---|---|---|
| `chat` | `POST /v1/chat/completions`, non-stream, `max_tokens` 8 | 8 tokens, `stop` | 500/s |
| `stream` | the same, `stream: true`, `max_tokens` 128 | 128 tokens, one per frame, `length` | 250/s |
| `tools` | chat with 4 function schemas, 2 earlier tool rounds in the messages | a JSON tool call the gateway parses into `tool_calls` | 500/s |
| `json` | chat with `response_format: json_schema` | a JSON object | 500/s |
| `messages` | `POST /v1/messages`, `max_tokens` 8 | 8 tokens | 500/s |

The gateway runs with `--parser symphony --tool-call-parser json` (`--gw-args` changes it),
`cache_aware` routing, 8 runtime threads and log level `warn`, as `grpc_stress/bed.sh` sets
it up; 4 mock engines (`--workers`).

## The tokenizer

The bed needs a tokenizer with a chat template. By default `synthetic_tokenizer.py` writes
one under the run directory, in the exact component shape of the common byte-level BPE
tokenizers (a BPE model over the 256 byte symbols, the word-splitting pattern, the ByteLevel
pre-tokenizer, post-processor and decoder, an NFC normalizer), so the gateway's own encoder
takes it the way it takes a production tokenizer (it logs `tokenizer encode path path=native`
at info level), with merges that make every word of the bed's prompts, schemas and answers
one token; anything else encodes to its bytes, so there is no unknown token and every text
decodes back to itself. Role markers are added special tokens; the chat template declares
tools, renders every role and the generation prompt. It needs no network and no third party,
which is what a CI job wants, and the per-request numbers match a production tokenizer's
within a few percent (the shape matters: a word-level tokenizer sends the gateway down its
generic encode path and reads six times the allocations). The numbers are still relative
(base against head on the same tokenizer); for absolute per-request costs with a production
template, pass `TOKENIZER=<tokenizer directory or hub id>`.

## Reading the numbers

- **Load-independent columns**: user instructions per request, allocation calls, KiB
  requested, allocations of 16 KiB and above. They are the judgement. Two cells of one binary
  agree to a fraction of a percent whatever else the machine runs; a difference beyond the
  base's spread is the change.
- **Load-dependent columns**: cycles, CPU time per request, kernel instructions, latencies.
  They carry the machine's load (the idle cell's rate is subtracted, the rest is not) and are
  reported with the 1-minute load average beside them in `cells.json`; compare them between
  cells run minutes apart at a similar load, and read a CPU-time delta under the spread as
  noise. In `cpu` mode the attached `perf stat` adds its own per-context-switch cost to the
  gateway's CPU time (a chat request read 530 us with the counters on and 320 us without, on
  the same binary and bed); the column is comparable between the two sides of one run, and
  `alloc` mode gives the absolute.
- **RSS**: after the window, from `/proc`; `VmHWM` is the peak. RSS moves with the allocator's
  decay and the request mix in flight at the moment of reading: a few percent is its band.
  `perf-mem` separates a slope (growth per 10,000 requests on one gateway) from the band.
- **The spread** in the setup column is the base's (max - min) / median over its repeats:
  the floor under which a change is not a change. The verdict table applies the rule of
  `scripts/bench_gate.py`: an id regresses only when its change is worse than the threshold
  AND every head repeat is worse than every base repeat (with no real change, three
  interleaved repeats per side fall in that order one time in twenty); a change beyond the
  threshold whose repeats overlap is `suspect`, never a failure. Without a gate file every id
  is `report only`.
- **Some paths have a per-process band**: the `tools` path's allocation count can differ by
  a few calls per request between two processes of ONE binary (a hash order decided at
  start); a change there is one that moves every head cell out of the base cells' range.
- **Warnings** at the end of a run (failed requests, script misses = answers served by the
  engine simulator instead of the script, missing jemalloc statistics) mean the cell did not
  measure what it should; read `cells.json` before using its row.

Measured spreads on one shared many-core machine (1-minute load 19-36), the same binary on
both sides, three interleaved repeats per side, 10,000 requests per cell, the gateway pinned
to 8 cores, over the five paths ((max - min) / median of the per-side repeats):

| column | identical-build spread | what the gate reads it as |
|---|---|---|
| user instructions per request | 0.08 - 0.17 % | the judgement; 2 % is a change |
| allocation calls per request | 0.08 - 0.22 % | the judgement; 2 % is a change (the tools path's per-process modes stay under it) |
| KiB requested per request | 0.01 - 0.02 % | the judgement; 2 % is a change |
| RSS after the window | 1.3 - 2.6 % | 5 % is a change |
| instructions per request, user + kernel | 0.5 - 3.2 % | the kernel share is the host's noise: report only |
| CPU time per request | 6.8 - 18.8 % | load-dependent: report only; read it with the load line under the table |
| cycles per request | 9 - 29 % | load-dependent: report only |

## The capacity rule of `perf-live`

A step holds when TTFT p99, end-to-end p99 and the error rate are within the SLO flags and
the achieved rate is within 95 % of the offered one. The capacity is the HIGHEST offered step
at which the SLO holds; the strict reading (the last holding step before the first failing
one) is printed beside it, with the failing steps below the hold named. Steps run on a fresh
gateway each, so a step never inherits the previous step's queue.

## Files

| file | what |
|---|---|
| `gateway_perf.py` | the driver: `cpu`, `alloc`, `mem`, `live`, `bed up\|down\|smoke`, `report`; standard library only |
| `synthetic_tokenizer.py` | the hub-free tokenizer directory |
| `build_base.sh` | the base revision's gateway by the head's cargo line |
| `../grpc_stress/bed.sh` | the bed itself (ports, pinning, registration, the probe) |
| `../../bench_gate.py` | the verdict rule and its table (`results.json` of a run is its input: `python3 scripts/bench_gate.py compare --gates <toml> <run>/results.json`) |
| `../../tests/test_gateway_perf_*.py` | the driver's arithmetic, the reports, the tokenizer |

A run directory holds `cells/<side>/<path>/<repeat>/` (the bed's logs, `summary.json` of the
client, `perf.csv`, `cell.json`), `cells.json`, `results.json`, `numbers.md`, `gate.md`
(`live.md` / `live.json` for a ramp).

Ports: a free block from 29700 (gateway, +1 metrics, +9 the mock admin API, +10.. the
engines) unless `--port-base` names one; pass `--gw-cores`, `--mock-cores` and `--gen-cores`
to keep the three parties on cores of their own when the machine is shared.
