# mock-worker

Multi-port mock HTTP/gRPC inference workers for scale-testing the SMG gateway's
routing and async-runtime behavior. One process hosts many protocol-accurate
stand-ins for vLLM/SGLang engines.

Two modes:

- **canned** (default) — every response is fixed (no model); a single optional
  `--gen-ms` delay. Cheap enough to run thousands of idle workers for measuring
  the gateway's own CPU/registration/routing cost.
- **realistic** (`--engine realistic`) — each worker runs a continuous-batching
  **engine simulator** that behaves like a real LLM engine on CPU, so the whole
  gateway (including load- and cache-aware routing) can be exercised without
  GPUs. See [Realistic engine](#realistic-engine).

## What it implements

**HTTP** (vLLM/SGLang HTTP surface the gateway probes and routes to):
- `GET /health` → `200 OK` (gates registration + health promotion)
- `GET /v1/models` → one model, `owned_by: sglang` (backend/model detection)
- `POST /v1/chat/completions` · `/v1/completions` · `/generate` — non-stream JSON
  or SSE (`data: {chunk}\n\n … data: [DONE]\n\n`)
- `GET /v1/loads?include=core` → `WorkerLoadResponse` (load-aware policies)

**gRPC** (TokenSpeed scheduler — the gateway tokenizes, the worker speaks token
ids): `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `Generate` (streamed
chunks + complete), `GetLoads`, `Abort`, and (realistic mode only)
`SubscribeKvEvents`; other admin RPCs return `unimplemented`.

**ZMQ** (`--zmq-handshake <ipc://…> --zmq-count N [--zmq-start-index I]`):
mock vLLM `EngineCore` ranks on the `engine-zmq-client` wire. Unlike the HTTP
and gRPC workers, which bind and are dialed, each rank dials the frontend's
handshake address, registers, takes `EngineCoreRequest`s and pushes
`EngineCoreOutputs` with the engine's load as `scheduler_stats`; the admin API
sees it as worker `zmq:<index>`.

## Run (canned)

```bash
cargo run --release -p mock-worker -- \
  --http-base-port 9000 --http-count 2000 \
  --grpc-base-port 19000 --grpc-count 0 \
  --model mock-model --gen-ms 5
```

Each worker is one port. Register them against an IGW gateway with
`POST /workers` (`{"url":"http://127.0.0.1:9000"}`, or `grpc://…` with
`connection_mode`/`runtime`/`models` for gRPC). `mock-worker --help` lists
every flag with its default.

## Realistic engine

`--engine realistic` backs each worker with a continuous-batching simulator
([`src/engine.rs`](src/engine.rs)) built the way vLLM schedules, so routing
experiments against it transfer to engines:

- **pass loop** — every pass has a token budget (`--max-batched-tokens`, 8192)
  and a sequence cap (`--max-running`, 256). Running requests go first, each
  taking one decode token or a prefill chunk; then the queue is admitted FCFS
  while budget and KV room remain. With `--prefill-first true` (SGLang) a pass
  that contains prefill runs prefill only.
- **block-level KV pool** — `--kv-blocks`/`--kv-tokens` physical blocks of
  `--block-size` tokens, keyed by content hash with reference counts; idle
  cached blocks sit in an LRU and are evicted head-first when an allocation
  needs room; a waiting prompt is admitted only when the free and evictable
  blocks could hold all of it beyond its cached blocks (vLLM's
  `scheduler_reserve_full_isl`, a gate read at admission, head-of-line) while
  only the pass's chunk is allocated, the next chunks allocating as they run,
  and a running request that still cannot get a block for its
  next tokens preempts the most recently admitted request (LIFO), which
  recomputes later. A fully cached prompt recomputes its last block; a prefix
  hit on an idle block references it again.
- **KV events** (`SubscribeKvEvents`) — per request within a pass, `Removed`
  for the blocks evicted by its allocation, then one `Stored` per contiguous
  run of blocks it completed (parent-chained, with token ids). `Removed` fires
  only when the last copy of a hash leaves the pool, `Stored` only when a hash
  first appears; completion and preemption emit nothing. Every event of a pass
  becomes visible at the pass end, so a request arriving mid-pass cannot see
  that pass's blocks. A reset (`POST /admin/reset`) publishes
  `AllBlocksCleared`.
- **timing** — `--timing polynomial` (default) uses AISimulate's baseline:
  prefill `16.50142 + 1.518344e-2·T + 4.209989e-7·T²` ms over the uncached
  tokens `T` of the pass, decode `max(1, 5.74 + 54.01·u − 25.74·u²)` ms over
  the KV utilisation `u` of the decoding requests; a pass lasts prefill plus
  decode, and the first token of a prefill adds no decode time.
  `--timing linear` is the simpler model: prefill at `--prefill-tps` tokens/s,
  decode `base + per-request · batch` ms.
- **cached tokens** — a request sharing a prefix with cached blocks pays less
  prefill and reports `cached_tokens` (gRPC chunks, HTTP
  `usage.prompt_tokens_details.cached_tokens`).

| Flag | Default | Meaning |
|------|---------|---------|
| `--timing` | polynomial | `polynomial`, `linear`, or `fit:<path>` (a hardware calibration JSON, below) |
| `--prefill-poly a,b,c` | AISimulate | prefill ms = a + b·T + c·T² |
| `--decode-poly a,b,c` | AISimulate | decode ms = max(1, a + b·u + c·u²) |
| `--request-overhead-ms` | 0 | fixed per-request latency added to every event of a stream (TTFT and e2e grow by it, ITL does not) |
| `--prefill-tps` | 8000 | linear model: prefill tokens/s (selects `linear`) |
| `--decode-base-ms` | 6.0 | linear model: fixed decode-pass ms |
| `--decode-per-req-ms` | 0.35 | linear model: decode ms per running request |
| `--max-batched-tokens` | 8192 | token budget per pass |
| `--max-running` | 256 | sequences per pass |
| `--kv-tokens` / `--kv-blocks` | 524288 tokens | KV pool capacity |
| `--block-size` | 16 | cache block/page size (tokens); must match the worker's `kv_block_size` |
| `--prefix-cache` | true | prefix caching + KV events |
| `--prefill-first` | false | SGLang-style prefill-only passes |
| `--reserve-full-isl` | true | admit a waiting prompt only when the free and evictable blocks could hold all of it beyond its cached blocks, stopping at the first that does not fit (vLLM's `scheduler_reserve_full_isl`: a gate, only the pass's chunk is allocated at admission); `false` skips the gate |
| `--context-length` | 32768 | advertised context length |
| `--loads-like` | mock | `vllm`: report only what the vLLM servicer reports (running, waiting, `token_usage`, maxima), so the gateway's expected-wait routes as it does on a vLLM fleet |
| `--admin-port` | off | process-wide admin API (below) |
| `--kv-events-zmq-base-port` | off | ZMQ KV-event publishers on vLLM's or SGLang's wire (below) |

```bash
cargo run --release -p mock-worker -- \
  --engine realistic --grpc-base-port 19000 --grpc-count 8 --model mock-model --admin-port 19100
```

`--timing fit:<path>` replaces the uncalibrated polynomials with a hardware
calibration the GPU harness writes. Prefill comes either from a measured
point table (`prefill_table_ms: [[tokens, ms], ...]`, or the harness's
`prefill_points_ms: {"<tokens>": {"median_ms": ..}}`), interpolated linearly
and extrapolated with the last slope, together with a pass-total form for
batched prefills (`prefill_pass_ms: {"intercept_ms", "ms_per_token"}`, from a
concurrent sweep): a lone request costs its table value, a pass that prefills
several requests costs the pass-total form and never less than the table
value of its largest request; or, without a table, from the polynomial
`prefill_fit_ms` / `prefill_ms` (`{"a","b","c"}` or `[a,b,c]`). Decode is
`decode_fit_vs_utilisation_ms` / `decode_ms` (`{"d","e","f"}` or `[d,e,f]`, ms
over KV utilisation); capacity is `kv_capacity_tokens` or
`kv_capacity_blocks` with `block_size`; `request_overhead_ms` shifts every
event of a stream. Unknown keys are ignored, and `--block-size`,
`--kv-tokens`/`--kv-blocks` and `--request-overhead-ms` given explicitly win
over the file (a restricted pool is, for example, `--kv-blocks 12000
--block-size 16`); the decode fit's utilisation is still read against the
file's `kv_capacity_tokens`, so a smaller pool changes how much fits, not
how long a decode step takes. The defaults stay AISimulate's uncalibrated
baseline.

Agreement with hardware is the caller's problem: AISimulate's published
agreement for these polynomials (mean absolute percentage error 48.5% on TTFT,
28.9% on TPOT) was measured with prefix caching disabled, so cache-hit and
routing effects have no published validation. Treat the simulator as a relative
A/B harness for policies and validate absolute numbers on GPUs.

### KV-event publisher on the engines' ZMQ wire

`--kv-events-zmq-base-port <port>` gives every realistic engine the publisher
vLLM runs (`ZmqEventPublisher` in `vllm/distributed/kv_events.py`), so the
servicers' relays (`crates/engine_servicer`, the Python servicer) can be
exercised end to end without a GPU: worker `i` (gRPC workers by port offset,
then ZMQ ranks) publishes on `base + 2i` and answers replay on `base + 2i + 1`.

- PUB frames: `[topic, sequence as u64 big-endian, msgpack]`, the sequence
  counting from 0 per publisher (`--kv-events-topic`, empty by default as
  vLLM's).
- Payload: vLLM's `EventBatch`, `[ts, events, data_parallel_rank]`; events are
  tagged maps in vLLM's field order with its `omit_defaults`: `BlockStored`
  (`type`, `block_hashes`, `parent_block_hash`, `token_ids`, `block_size`,
  `lora_id`, `medium: "GPU"`, `lora_name`, `group_idx: 0`,
  `kv_cache_spec_kind: "full_attention"`), `BlockRemoved` (`type`,
  `block_hashes`, `medium`, `group_idx`), `AllBlocksCleared` (`type`). Block
  hashes are unsigned 64-bit integers (vLLM's int form of the same bits the
  gRPC stream carries signed).
- Replay (`--kv-events-replay`, on by default): a DEALER sends
  `[b"", start as 8 bytes big-endian]`; the ROUTER answers every buffered
  batch from `start` as `[b"", topic, seq, payload]` and then
  `[b"", b"", END, b""]` with END = eight 0xff bytes; the last
  `--kv-events-buffer-steps` (10000) batches are kept.

`--kv-events-wire sglang` speaks SGLang's publisher instead: signed 64-bit
hashes (SGLang takes the first eight digest bytes signed), `BlockStored` with
only `block_hashes`, `parent_block_hash`, `token_ids`, `block_size`, `lora_id`
(one per radix node: here one per contiguous run a request completed),
`BlockRemoved` with one node's hashes (here one per evicted block), a nil
`attn_dp_rank`, an `AllBlocksCleared` batch at startup as the scheduler
publishes, and replay replies without the topic frame (`[b"", seq, payload]`,
then `[b"", END, b""]`).

The unit tests decode both wires with the Rust relay's own normalizer
(`engine_servicer::kv_wire`), so a change on either side shows up here.

### Admin API

`--admin-port` serves the ground truth a routing benchmark needs and real
engines do not expose:

- `GET /admin/health` — `ok`;
- `GET /admin/fleet` — every engine (`grpc:<port>` / `http:<port>` /
  `zmq:<index>`) with cache size, load, cached blocks and preemptions;
- `GET /admin/requests?since=<seq>&limit=<n>` — admitted requests with the
  serving worker, prompt/cached tokens, queue wait and the **arrival-time
  oracle**: the most cached tokens any worker of the process held when the
  request arrived (the best a router could have obtained). Join on the
  gateway's response `id`. `injected_failures` beside the records counts, per
  worker, the requests the `fail` hook answered before admission (they left no
  record) and the streams it cut;
- `GET /admin/cache/{worker}` — the worker's cached block keys;
- `POST /admin/reset[/{worker}]` — clear caches and publish `AllBlocksCleared`
  (an engine restart, to the index).

**Tokenizer note (gRPC):** the gateway tokenizes prompts before routing, so it
needs a real tokenizer for the model. Register each worker with a tokenizer
label, e.g. `"labels":{"tokenizer_path":"gpt2"}`, and a `"kv_block_size":16`, and
do **not** pass `--disable-tokenizer-autoload`. (HTTP workers need no tokenizer
but cannot drive event-driven `cache_aware`, which requires token ids.)

### Fault hooks

All under the admin API; `{worker}` is a worker name (`grpc:<port>`,
`http:<port>`, `zmq:<index>`) or `all`. A hook applies to every KV-event
transport of the worker (gRPC `SubscribeKvEvents` and the ZMQ publisher alike)
and answers with the worker's hook state (`drop_pending`, `dropped_total`,
`delay_ms`, `admit_delay_ms`, `admit_per_sec`, `paused`, `generation`,
`restarts`, and the request fault's `fail_status`, `fail_pending`,
`fail_secs_left`, `fail_after_tokens`, `fail_stall_ms`, `failed_total`,
`cut_total`, `stalled_total`). The hooks need `--engine realistic`: the canned
path has no engine to hook.

| Hook | Effect |
|------|--------|
| `POST /admin/fault/{worker}/drop?batches=N` | the next N event batches are not published (lost on the wire; they stay in the replay buffer, so a gap replay recovers them) |
| `POST /admin/fault/{worker}/delay?ms=D` | every batch is published D ms after its pass ends (0 clears) |
| `POST /admin/fault/{worker}/admit-delay?ms=D` | every new gRPC request is held D ms before the engine sees it (0 clears): the gateway has dispatched it, the load record does not count it yet, as a backlog in transit between the two would behave |
| `POST /admin/fault/{worker}/admit-rate?per_sec=R` | at most R new gRPC requests per second enter the engine (0 clears); the rest wait their turn, unseen by the load record: a throttled input path whose backlog grows while the gateway keeps sending |
| `POST /admin/fault/{worker}/restart-publisher` | the publisher restarts: gRPC sequence numbers start over at 1 and the ZMQ sequence at 0, the replay buffers are emptied, the cache is kept (no `AllBlocksCleared`); `generation` increments |
| `POST /admin/fault/{worker}/pause` | the engine freezes after its current pass: no passes, no tokens, no events; requests queue (and count as waiting); health and `GetLoads` keep answering |
| `POST /admin/fault/{worker}/resume` | the engine runs again |
| `POST /admin/fault/{worker}/fail?status=S[&count=N\|&secs=T][&after_tokens=K][&stall_ms=M]` | the worker answers status S (400-599) instead of serving: the next N requests, every request for T seconds, or every request until cleared; an HTTP worker answers the status with `{"error": {"message": "injected", "type": "fault"}}`, a gRPC worker the status code the gateway maps back to it (429 `resource_exhausted`, 500 `internal`, 502/503 `unavailable`, 504 `deadline_exceeded`). Before admission, so the request never counts as served and leaves no record. With `after_tokens=K` the request is admitted and served for K tokens, then the stream is cut with S (the gRPC trailer; an HTTP body that ends without its finish frame): the non-retryable case, for a duplication check (a non-streaming HTTP request is answered with S after K tokens instead, which the gateway may retry: the duplicated-work case). On the gRPC worker 408 reads back at the gateway as 504 and 502 as 503 (both retryable), any other 4xx as 400. `stall_ms=M` holds the request M ms first; `status=0&stall_ms=M` stalls and then serves. `status=0` alone clears |
| `GET /admin/fault/{worker}` | the hooks' current state (pending drops, delay, restarts, paused, generation, the request fault and its counts) |
| `POST /admin/reset/{worker}` | (already there) clear the cache and publish `AllBlocksCleared` |

### Truth endpoints

| Endpoint | Answer |
|----------|--------|
| `POST /admin/truth/{worker}` with `{"token_ids": [...]}` | what the worker would serve from cache for that prompt right now: `cached_tokens`, `cached_blocks`, `block_size` (the engine's own prefix match, last-block rule included) |
| `GET /admin/truth` | per worker, over every admitted request: `requests`, `prompt_tokens`, `cached_tokens`, `oracle_tokens`, so a gateway's hit-rate claim can be checked against what the engines actually served; `injected_failures` and `cut_streams` count what the `fail` hook refused before admission and cut after output |

A circuit-breaker or retry drill arms `fail` on one worker (`status=503&count=N`
for exactly N failures inside the breaker's window, `secs=T` for a worker that
stays up but failing), sends requests through the gateway and reads
`failed_total` against the gateway's retry and breaker metrics; a retried
request that reached a worker shows in `/admin/requests` once per admission,
so `requests` there against `injected_failures` tells retried attempts from
duplicated work.

## Long prompts

A gRPC worker decodes and encodes messages of any size by default, as the
engine servicers do. `--grpc-max-message-bytes <n>` sets a limit; tonic's own
default, 4 MiB, refused the `Generate` of a million-token prompt (its ids
alone are about 2 MB as varints, and a gateway that sends the prompt text
alongside adds the text's bytes on top). What bounds a long prompt end to end
on the gateway's gRPC path: the gateway's HTTP body limit
(`--max-payload-size`, 512 MiB by default), the context length the worker
advertises (`--context-length`), and the worker's decode limit. The gateway
sends `Generate` with no size limit of its own and decodes each response
message up to tonic's 4 MiB default, which a streamed chunk never approaches.

## Capturing requests

`--capture PATH` appends every gRPC `Generate` request a worker receives to
`PATH`, one JSON object per line, so a test can check exactly what the gateway
put on the wire. Responses are unchanged, in canned and realistic mode alike.
HTTP and ZMQ workers do not capture.

```bash
cargo run --release -p mock-worker -- \
  --grpc-base-port 19000 --grpc-count 1 --model mock-model --capture generate.jsonl
```

Keys are the field names in
[`tokenspeed_scheduler.proto`](../grpc_client/proto/tokenspeed_scheduler.proto):

- `request_id`, `input_ids`, `original_text` and `stream`. The gateway sends a
  client `rid` as `request_id` (with a suffix under PD), so lines join to
  responses on it.
- Every `SamplingParams` scalar, and `logit_bias` as an object with sorted
  keys, so the same request always gives the same bytes. An optional the
  gateway left unset is `null`. A float is the shortest decimal that reads back
  as the same `f32`, so a request's `0.7` shows as `0.7`; NaN and infinities,
  which JSON cannot hold, are strings such as `"NaN"`.
- `constraint`: `{"kind": "regex" | "json_schema" | "ebnf_grammar" |
  "structural_tag", "value": <the string as sent>}`, or `null`.
- `return_logprob`, `logprob_start_len`, `top_logprobs_num` and
  `token_ids_logprob`.
- `has_custom_params`, `has_mm_inputs`, `has_encode_bootstrap_info`,
  `has_kv_bootstrap_info` and `has_data_parallel_rank`: those fields are
  recorded only as present or absent.

Each line is in the file before the worker sends the first frame of its
response, so killing the worker loses no line. A capture path that cannot be
opened fails at startup with exit code 2, before any worker starts. A new
capture file is readable by its owner only (mode 0o600); an existing file keeps
its mode.

## Scale-test rig (gateway CPU)

`scripts/scale_test.sh` launches an IGW gateway, starts a canned mock fleet,
REST-registers it, and samples the gateway PID's CPU + `/health` latency:

```bash
scripts/scale_test.sh --http 2000 --policy cache_aware --rps 500 --duration 30
scripts/scale_test.sh --grpc 1000 --policy least_load
```

## Policy A/B rig (no-GPU routing fidelity)

`scripts/sim_ab.sh` launches the gateway + a **realistic** mock fleet, drives the
same Poisson workload (`scripts/sim_load.py`, with a tunable shared-prefix
fraction) under each routing policy, and prints a side-by-side table of
TTFT / ITL / E2E / throughput:

```bash
# Full fidelity: gateway tokenizes (downloads gpt2) and routes on token ids,
# so both least_load and event-driven cache_aware engage.
scripts/sim_ab.sh --mode grpc --workers 8 --rps 120 --duration 30

# Offline-friendly: no tokenizer; drives least_load + latency + approximate cache_aware.
scripts/sim_ab.sh --mode http --workers 16 --policies "random least_load"
```

## Replaying a trace
The `replay` binary of this crate (`cargo run --release -p mock-worker --bin replay`)
replays a Mooncake-format trace (`{timestamp, input_length, output_length,
hash_ids}` per line, one `hash_id` per 512-token block) through the gateway and
scores routing quality end to end.

- Every `hash_id` becomes a deterministic text block (`--words-per-block`, 480
  words, about one token each), so rows that share ids share prompt prefixes
  after the gateway tokenizes them. Calibrate the knob against the real
  tokenizer once per model: replay a few hundred rows, fit
  `prompt_tokens = a + b * blocks` over the successful rows of `requests.csv`
  (`blocks = ceil(trace_input_length / 512)`; `a` is the chat template's
  overhead, `b` the tokens one block really produced), then set
  `--words-per-block` to `480 * 512 / b` so a block is 512 tokens again (a
  tokenizer with a ~150k vocabulary gives `b` near 481, i.e. 511 words).
- Requests are sent open-loop at `timestamp / --speedup` as streaming chat
  completions with `stream_options.include_usage`, recording TTFT, inter-token
  latencies (a thinking model's streamed `reasoning_content` counts as output
  for both, and separately as `reasoning_tokens` in the CSV), end-to-end
  latency, the serving worker (`system_fingerprint`, which the gateway sets
  from the worker's `weight_version` label), and the engine-reported
  `cached_tokens`.
- `--chat-template-kwargs '{"enable_thinking":false}'` sends the object as
  `chat_template_kwargs` in every request, the way to turn a model's default
  reasoning off without touching the trace.
- With `--admin <mock admin url>` each request is joined with the mock fleet's
  record of it (`GET /admin/requests`), adding the arrival-time oracle (the most
  cached tokens any worker held when it arrived) and the queue wait.
- `--connections pooled` (the default) reuses HTTP/1.1 keep-alive connections:
  fewer handshakes, and the TTFT percentiles stay comparable with every
  earlier run, but the client's pool can hand a request a connection whose
  previous response is still streaming, and that request then waits inside
  the client for the whole stream before the gateway reads it, which a TTFT
  column shows as one rare request per few tens of thousands (about one in
  thirty thousand streamed requests) taking a full stream's time on an
  otherwise idle gateway. `--connections fresh` is the opt-in for
  tail-sensitive runs (max TTFT, stall hunting): every request gets a
  connection of its own, closed by the server after the response
  (`Connection: close`, so the TIME_WAIT sits on the server's side and the
  replayer's ephemeral ports stay free at rate), at a measured cost of about
  +3 ms TTFT p50 and +75 ms p99 at 40 req/s of 4k-token streams. Either way
  the gateway's (and the admin API's) host name is resolved once at start and
  pinned, so no connection waits on a name lookup.

Output: `summary.json` (mean/p50/p90/p99 TTFT, per-request mean ITL (TPOT)
distribution, e2e latency, goodput at the SLO `--slo-ttft-ms` / `--slo-itl-ms`
(default 500 ms TTFT and 50 ms per-request mean ITL; a strict variant uses the
per-request p99), prefix reuse = cached / prompt tokens, oracle prefix reuse,
hit-over-oracle, per-worker request and uncached-token counts, balance) and
`requests.csv` with one row per request.

```bash
replay --trace mooncake_trace.jsonl --gateway http://127.0.0.1:31000 \
  --model mock-model --speedup 4 --limit 5000 --admin http://127.0.0.1:31002 --out out/
```

With `--gateway-log <file>` (the gateway run at `--log-level debug`) the
gateway's routing decisions are joined to the requests by request id (the
`x-request-id` header, which is the response id without its uuid tail) and
`t4.md` lists them per request: `phase, idx, worker, branch, prompt_tokens,
engine cached_tokens, implied overlap, agree`. The implied overlap is the
gateway's stated credit when its log carries one (`overlap_tokens=`,
`overlap_blocks=` or the tree path's `matched_ratio=`); otherwise `agree`
compares the branch's claim of an overlap (`event_hit`, `event_spill`) with
whether the engine served cached tokens. With `--admin` the summary also
carries `engine_truth_per_worker`, each engine's own account of the prompt,
cached and oracle tokens it served, and `fleet.csv` samples every worker's
load, cached blocks, preemptions and KV batches once a second for the whole
run.

The mock fleet it is meant for is `mock-worker --engine realistic --admin-port`
(the [Realistic engine](#realistic-engine) above; note the caveat that its timing
polynomials were validated against hardware only with prefix caching off).

### Long runs

For a run of hours, keep one gateway and one fleet up and start the replayer
window by window (`--skip` advancing by `--limit`, a `--label` per window,
`--admin` for the oracle join), driving the admin fault hooks on a cycle over
the workers in turn (lost batches, a publishing delay, a publisher restart, a
pause and resume, a cache reset) and sampling the gateway's `/metrics` once a
minute. The memory reading that matters is the gateway allocator's live bytes
at idle, hour over hour; its RSS shows how much the allocator keeps, not what
is live.
