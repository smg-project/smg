# gateway-perf: the gateway's cost per request, base against head (CPU only)

What a change to the gateway costs or saves per request, measured on this machine with no
GPU and no engine: one release gateway in front of mock gRPC engines that answer from a
script, an open-loop client at a fixed rate, and the gateway process's own counters read
from outside it. The same bed and client as `scripts/bench/grpc_stress`; what this crate adds
is the A/B discipline (a fresh gateway per cell, an idle cell subtracted, base and head
interleaved, medians with their spread) and the counters that make a per-request number
exact: instructions and allocation calls do not move with the load of the machine, so two
runs of the same binary agree to a fraction of a percent, and a change of a few hundred
instructions or one allocation per request is visible.

The crate drives what the repository already has and re-implements none of it: the bed is
`scripts/bench/grpc_stress/bed.sh`, the engines are `mock-worker` with their admin API, the
client is `stress_load` (both binaries of `crates/mock_worker`), and the verdict rule is
`scripts/bench_gate.py`'s (the crate applies the same rule natively and writes the
`results.json` that script reads).

## One command for a pull request

```bash
make perf-base                 # the base branch's gateway (origin/main; PERF_BASE_REV=<rev> for another)
make perf-alloc perf-cpu       # the head is built from the working tree; both runs print their tables
make perf-report PERF_OUT=target/gateway-perf/<stamp>/perf-cpu   # the Numbers table again, for the PR body
```

`perf-alloc` is the deterministic part (allocation calls and KiB per request; RSS is reported
beside them and moves with the allocator, see below) and takes about a minute per path and
side at the defaults; `perf-cpu` adds instructions and cycles when `perf stat` can count them
here (otherwise CPU time per request, which carries the machine's load). The table each run
prints is the `## Numbers` table of the pull-request body: metric, before, after with the
change, and the setup with the base's spread. Paste it.

## The targets

| target | what it measures | how |
|---|---|---|
| `make perf-build` | nothing: the release binaries. The gateway is built alone (`cargo build --release -p smg --bin smg`), then the mock engine, the client and the driver, because building the gateway together with another package changes its binary through feature unification; `perf-base` uses the same line | copies the gateway under `target/gateway-perf/head/` |
| `make perf-base` | the base revision's gateway binary, `target/gateway-perf/base/smg` | `gateway-perf base <rev> <out>`: a detached worktree of `PERF_BASE_REV` built into its own target directory (a full release build; sccache shortens it) |
| `make perf-bed` | the bed comes up and answers: one request per path, the first response shown, the scripted answers' hit and miss counts | `gateway-perf bed smoke` (`bed up` / `bed down` keep one running for `grpc_stress/cell.sh` or by hand) |
| `make perf-alloc` | allocation calls per request, KiB requested per request (the allocator's size classes, small and large), allocations of 16 KiB and above per request, RSS and peak RSS after the window | jemalloc's exit statistics: the gateway is started with `_RJEM_MALLOC_CONF=stats_print:true,stats_print_opts:Jaxe` and prints them at its clean shutdown; no special build |
| `make perf-cpu` | user instructions per request, instructions (user + kernel), cycles, context switches, from `perf stat -p <gateway pid>` over the window; CPU time per request (utime + stime) always | hardware counters need `perf`, a kernel that allows them (`kernel.perf_event_paranoid` at 2 or below for a process of your own) and hardware that exposes them (a virtual machine often does not: the run then says so and keeps the CPU-time column) |
| `make perf-mem` | RSS after each of several windows on ONE gateway, the growth per 10,000 requests, the peak (`VmHWM`) after a burst window at twice the rate, the allocator's live bytes | `PERF_MEM_PATHS` (default `stream`, the path that holds the most per request), `--windows`, `--burst` |
| `make perf-live` | an open-loop ramp: offered rate against achieved rate, TTFT and end-to-end p50/p99, errors, gateway CPU per request and RSS per step, and the capacity | a fresh gateway per step; the capacity rule below; the base too when its binary exists |
| `make perf-report` | the tables of a run again (`PERF_OUT=<stamp>/perf-cpu`, the mode's directory) | `results.json` of that directory |

Variables: `PERF_OUT` (the run root, `target/gateway-perf/<stamp>`, one stamp per `make`
invocation; each target writes under it, `<stamp>/perf-cpu`, `<stamp>/perf-alloc`, ...),
`PERF_PATHS`, `PERF_REQUESTS` (10,000 per cell), `PERF_REPEATS` (3 per side), `PERF_ARGS`
(extra flags of the driver, valid for the target's mode: the cell targets take the cell flags,
`perf-bed` and `perf-live` the bed flags only, e.g. `--rate-scale 0.5 --gw-cores 0-7
--mock-cores 8-15 --gen-cores 16-23 --port-base 29700`), `TOKENIZER`, `PERF_GATES` (a
`bench_gate` TOML for the verdict column). `target/release/gateway-perf <mode> --help` lists
the rest; the driver finds the bed script under `--repo` (the current directory by default).

## The paths

Every cell drives one path: 64 distinct deterministic prompts of 64 words (a shared head
from a pool of 16, a unique tail), the client's seed per path, and ONE scripted answer for
every request of the cell, handed to the mock engines before the cell starts (the worker's
admin queue) and served by the wire with no engine time, except the pause between frames the
spaced paths ask of the engine: the numbers are the gateway's.

| path | request | answer | rate |
|---|---|---|---|
| `chat` | `POST /v1/chat/completions`, non-stream, `max_tokens` 8 | 8 tokens, `stop` | 500/s |
| `stream` | the same, `stream: true`, `max_tokens` 128 | 128 tokens, one per frame, `length` | 250/s |
| `tools` | chat with 4 function schemas, 2 earlier tool rounds in the messages | a JSON tool call the gateway parses into `tool_calls` | 500/s |
| `json` | chat with `response_format: json_schema` | a JSON object | 500/s |
| `messages` | `POST /v1/messages`, `max_tokens` 8 | 8 tokens | 500/s |

Ten streamed shapes are opt-in (name them in `PERF_PATHS`; their rates keep the frames
per second near the `stream` path's, except where said):

| path | request | answer | rate |
|---|---|---|---|
| `messages_stream` | `POST /v1/messages`, `stream: true`, `max_tokens` 300 | 300 tokens, one per event, `length` | 100/s |
| `generate_stream` | `POST /generate`, `stream: true`, `max_new_tokens` 300 | 300 tokens, one per frame, the whole text so far in every frame, `length` | 100/s |
| `completion_stream` | `POST /v1/completions`, `stream: true`, `max_tokens` 300 | 300 tokens, one `text` delta per frame, `length` | 100/s |
| `generate_stream_long` | the same, `max_new_tokens` 2000 | 2,000 tokens, one per frame | 16/s |
| `generate_stream_spaced` | the same, `max_new_tokens` 300, the engine pausing 20 ms between frames | 300 tokens, one per frame 20 ms apart, the whole text so far in every frame | 100/s |
| `stream_spaced` | the `stream` request, the engine pausing 20 ms between frames | 128 tokens, one per frame 20 ms apart | 250/s |
| `completion_stream_spaced` | the `completion_stream` request, the engine pausing 20 ms between frames | 300 tokens, one `text` delta per frame 20 ms apart | 100/s |
| `messages_stream_spaced` | the `messages_stream` request, the engine pausing 20 ms between frames | 300 tokens, one per event 20 ms apart | 100/s |
| `generate_stream_spaced_long` | the `generate_stream` request with `max_new_tokens` 5000, the engine pausing 20 ms between frames | 5,000 tokens, one per frame 20 ms apart, the whole text so far in every frame | 6/s |
| `stream_logprobs` | the `stream` request with `logprobs: true, top_logprobs: 5` | 128 tokens, one per frame, the sampled logprob and 5 candidates each, `length` | 100/s |

The spaced paths read the stream loops as a decoding engine drives them, one frame at a time
(about 600 streams open at once at their rates), where the burst paths hand a whole answer over
at once. Their pause is engine time by design: a 300-frame answer takes about six seconds, so
they serve the cost cells (`cpu`, `alloc`, `mem`), and `live` on one of them needs
`--slo-e2e-p99-ms` above about 6,000 to hold a step.

`generate_stream_spaced_long` is the spaced generate path on a long answer: its text passes
the gateway's 16 KiB frame bound about two thirds of the way through, so each of the last 1,700
or so events fills a frame of its own with 20 ms between them, where `generate_stream_long`
fills its frames from a burst (its events stay under 10 KiB; two or three make a frame) and the
300-frame spaced path never reaches the bound. It
reads what a whole-text-per-event stream loop's frame buffer does between full frames at an
engine's pace: the allocation calls, the KiB and the allocations of 16 KiB and above per
request, and the RSS after the window, beside the instruction count. An answer takes about 100 s and about 600
streams are open at once at 6/s; 10,000 requests at that rate are about 28 minutes of window
per cell, so name it with a smaller count (`PERF_REQUESTS=5000` is about 15 minutes); its rate
stays a whole number at `--rate-scale 0.5`. Its answer fits the load client's 120 s drain with
about 15 s to spare (a longer or slower paced path needs the drain made a knob first; a cell whose
last requests the drain cuts off warns, as a cell with failed requests does), and `live` on it
names its rates beside its SLO, since the ramp's default steps start at 200/s, the burst paths'
rates: `gateway-perf live --path generate_stream_spaced_long --rates 3,6 --slo-e2e-p99-ms 110000`,
the SLO above the answer's own 105 s and under the drain (a step's sampled answers complete in
the client's drain after the step's seconds, so the default `--step-seconds` holds).

The logprobs path's rate is the gateway's load, not a frame-rate match: a logprobs answer costs
the gateway about 3.5 times the instructions and 13 times the allocation calls of the plain
128-token stream, so at 100/s the cell runs about 1.4 times the `stream` cell's gateway
instructions per second and about five times its allocation calls, where the `stream` rate would
run 3.5 and 13 times them; the cell is read against its own base spreads, not balanced with
`stream`.

The long path's cells are long at the Makefile's default `PERF_REQUESTS` (10,000 requests at
16/s is 625 s per cell, twice that at `--rate-scale 0.5`): pass a smaller count when naming it,
for example `PERF_REQUESTS=3000`; its rate stays a whole number at `--rate-scale 0.5`.

The gateway runs with `--parser symphony --tool-call-parser json` (`--gw-args` changes it),
`cache_aware` routing (`GW_POLICY` in the environment names another), 8 runtime threads and
log level `warn`, as `grpc_stress/bed.sh` sets it up; 4 mock engines (`--workers`).

## The fleets

Every path above runs on any fleet (`--fleet`, a bed flag of every mode), except on the HTTP
fleet the generate paths (that wire answers `/generate` in the chat shape): the driver refuses
them there, and each
fleet has a default path list (`chat,stream,tools,json` on HTTP). A run is one fleet, named in its
setup line, and the ids of `results.json` are the same on every fleet, so a gate file reads a
prefill-decode or an HTTP run as it reads a regular one. On the HTTP wire the scripted
text goes as one piece per word, the units that wire streams by, so a streamed answer frames one
token at a time there too, and a path that asks for logprobs scripts one value per piece.

| fleet | engines | the gateway path a cell reads |
|---|---|---|
| `grpc` (the default) | 4 regular gRPC engines | the gRPC regular path: one selection per request |
| `http` | 4 regular engines on the mock's HTTP wire (`--http-count`), registered with `connection_mode: http` and no runtime (the gateway reads it at registration) | the HTTP regular path: one selection per request under the request lease, the body forwarded as JSON, the answer relayed |
| `prefill-decode` | 1 prefill engine (`--prefill-workers`; `--worker-type prefill`, a bootstrap port the gateway probes at registration, a prompt-side answer of 1 ms from the realistic engine's admission) and 4 decode engines (`--workers`; `--worker-type decode`, `--max-running 8` and the `max_running_requests: 8` label), both registered with their roles behind one gateway in multi-router mode with `--pd-disaggregation`; the gateway's in-flight bound of 8 per prefill engine (`--prefill-max-inflight-requests-per-worker`) | the gRPC prefill-decode path: the prefill admission queue, the pair selection, the decode rooms claimed against the label's window, two dispatches per request; the scripted answer is the decode side's (the prefill side answers the prompt alone, from no script) |

The prefill-decode fleet's shape is the bed script's (`FLEET=prefill-decode`; `PREFILL_WINDOW`,
`DECODE_WINDOW`, `PREFILL_ENGINE_ARGS` and the ports in its header), and the driver passes its
environment through, so `DECODE_WINDOW=64 gateway-perf cpu --fleet prefill-decode ...` widens
the rooms for a paced path whose streams stay open longer than 32 rooms hold. `make perf-cpu
PERF_ARGS="--fleet prefill-decode"` runs the usual cells on it; a prefill-decode block takes
`10 + workers + 2 x prefill` ports (the second admin API at +8, the prefill engines and their
bootstrap ports after the decode engines).

## The tokenizer

The bed needs a tokenizer with a chat template. By default the driver writes one under the
run directory (`gateway-perf tokenizer <dir>` writes one anywhere), in the exact component
shape of the common byte-level BPE tokenizers (a BPE model over the 256 byte symbols, the
word-splitting pattern, the ByteLevel pre-tokenizer, post-processor and decoder, an NFC
normalizer), so the gateway's own encoder takes it the way it takes a production tokenizer
(it logs `tokenizer encode path path=native` at info level), with merges that make every
word of the bed's prompts, schemas and answers one token; anything else encodes to its bytes,
so there is no unknown token and every text decodes back to itself. Role markers are added
special tokens; the chat template declares tools, renders every role and the generation
prompt. It needs no network and no third party, which is what a CI job wants, and the
per-request numbers match a production tokenizer's within a few percent (the shape matters:
a word-level tokenizer sends the gateway down its generic encode path and reads six times the
allocations). The numbers are still relative (base against head on the same tokenizer); for
absolute per-request costs with a production template, pass `TOKENIZER=<tokenizer directory
or hub id>`.

## Reading the numbers

- **Load-independent columns**: user instructions per request, allocation calls, KiB
  requested, allocations of 16 KiB and above. They are the judgement. Two cells of one binary
  agree to a fraction of a percent whatever else the machine runs; a difference beyond the
  base's spread is the change.
- **Load-dependent columns**: cycles, CPU time per request, kernel instructions, latencies.
  They carry the machine's load (the idle cell's rate over the window is subtracted from the
  counters read over the window; the allocator's exit totals lose the idle cell's totals
  whole, which removes the start-up share and leaves the idle rate times the difference of
  the two cells' lives, the same on both sides) and are reported with the 1-minute load
  average beside them in `cells.json`; compare them between cells run minutes apart at a
  similar load, and read a CPU-time delta under the spread as noise. In `cpu` mode the
  attached `perf stat` adds its own per-context-switch cost to the gateway's CPU time (a chat
  request read 530 us with the counters on and 320 us without, on the same binary and bed);
  the column is comparable between the two sides of one run, and `alloc` mode gives the
  absolute.
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
  engine simulator instead of the script, unreadable script counters, missing jemalloc
  statistics) mean the cell did not measure what it should: the run exits 2 with its tables
  on disk; read `cells.json` before using its row.

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

## The pull-request job (advisory)

`.github/workflows/benchmark-gateway-perf.yml` runs `gateway-perf cpu` on every pull
request that touches `model_gateway/` or `crates/` (tests and benches excluded): the merge
commit's first parent against the pull request, built by the same cargo line (`perf-build`'s),
on the chat, stream and tools paths, 4,000 requests per cell at half the bed's rates, 3+3 interleaved cells, judged by
`.github/bench-gates/gateway-perf.toml` (allocation calls, KiB and user instructions at 2 %,
RSS at 5 %; CPU time and cycles report only). The job is ADVISORY: the tables go to the step
summary, to a comment on the pull request when the token may write one, and to the
`gateway-perf-<sha>` artifact (with `results.json`, the per-repeat values); its verdict never
fails the job. A hosted runner exposes no hardware counters, so the instruction ids are
absent there and the exact allocation counts carry the judgement; the artifact's
`results.json` is where the runner's own spread is read before any threshold turns blocking.

## The capacity rule of `perf-live`

A step holds when TTFT p99, end-to-end p99 and the error rate are within the SLO flags and
the achieved rate is within 95 % of the offered one. The capacity is the HIGHEST offered step
at which the SLO holds; the strict reading (the last holding step before the first failing
one) is printed beside it, with the failing steps below the hold named. Steps run on a fresh
gateway each, so a step never inherits the previous step's queue.

## The crate

| module | what |
|---|---|
| `main.rs` | the `gateway-perf` binary: `cpu`, `alloc`, `mem`, `live`, `bed up\|down\|smoke`, `report`, `tokenizer`, `base` |
| `paths.rs` | the paths: the client's flags, the scripted answer, the rate; the interleaving |
| `bed.rs` | the bed through `bed.sh` (ports, environment, up/down, bed.json) and the engines' script API (enqueue, hit/miss counts) |
| `http.rs` | a small HTTP/1.1 client over `std::net` for the loopback endpoints (no TLS, no runtime) |
| `counters.rs` | `/proc` CPU time and memory, the allocator gauges, `perf stat`, jemalloc's exit statistics |
| `cells.rs` | a cell, the per-request arithmetic, `results.json`, the outputs of a run, the `mem` soak, the bed smoke |
| `report.rs` | the Numbers table |
| `gate.rs` | the verdict rule and its table (`scripts/bench_gate.py`'s, with its gate file) |
| `live.rs` | the ramp and the capacity rule |
| `tokenizer.rs` | the hub-free tokenizer directory |
| `base.rs` | the base revision's gateway by the head's cargo line |

A run directory holds `cells/<side>/<path>/<repeat>/` (the bed's logs, `summary.json` of the
client, `perf.csv`, `cell.json`), `cells.json`, `results.json`, `numbers.md`, `gate.md`
(`live.md` / `live.json` for a ramp). `python3 scripts/bench_gate.py compare --gates <toml>
<run>/results.json` re-evaluates a run with the Python gate.

Ports: a free block from 29700 (gateway, +1 metrics, +9 the mock admin API, +10.. the
engines; on the prefill-decode fleet +8 the prefill fleet's admin API and, after the decode
engines, the prefill engines and their bootstrap ports) unless `--port-base` names one; pass
`--gw-cores`, `--mock-cores` and `--gen-cores` to keep the three parties on cores of their own
when the machine is shared.
