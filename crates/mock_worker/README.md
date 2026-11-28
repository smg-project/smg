# mock-worker

Multi-port mock HTTP/gRPC inference workers for scale-testing the SMG gateway's
routing and async-runtime behavior. One process hosts many protocol-accurate
stand-ins for vLLM/SGLang engines.

Two modes:

- **canned** (default) — every response is fixed (no model), in the shape of
  its route; a single optional `--gen-ms` delay. Cheap enough to run thousands
  of idle workers for measuring the gateway's own CPU/registration/routing cost.
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
- `POST /v1/messages` — the same answers in the Messages shape: one JSON
  message, or the `message_start` … `message_stop` event stream (no `[DONE]`)
- `GET /v1/loads?include=core` → `WorkerLoadResponse` (load-aware policies)
- `POST /flush_cache` · `/start_profile` · `/stop_profile` — the gateway's admin
  fan-out: the flush clears the engine's cache (a canned worker has none) and
  answers 200; a profiling run starts with the body's options and ends with one
  trace file in its output directory (a run already in progress, or a stop with
  none, is a 409)

**gRPC** (one engine's scheduler service, `--grpc-wire tokenspeed|vllm|sglang`,
TokenSpeed by default — the gateway tokenizes, the worker speaks token ids):
`HealthCheck`, `GetModelInfo`, `GetServerInfo`, `Generate` (streamed chunks +
complete), `GetLoads`, `Abort`, and (realistic mode only) `SubscribeKvEvents`, and
the admin fan-out's `FlushCache` (every rank's cache cleared at once,
`success: true`), `StartProfile` and `StopProfile` (a run in progress refused
with `success: false`, the trace file written at the stop), the same on each
wire; the other RPCs (`GetTokenizer`, embeddings, LoRA) return `unimplemented`
(a replay answers `GetTokenizer` where its recording holds the fetch).
`GetLoads` and the pushed load record count a rank's resident tokens
(`num_resident_tokens`, the context tokens of the requests it runs) on the
TokenSpeed wire, whose load report has the figure; the other two wires send
none, and so does a canned worker, which runs no engine. The canned
and scripted answers, the realistic engine, the data-parallel ranks and the
prefill/decode roles are the same behind each wire; each shapes its own
messages (a role is `disaggregation_mode` in the server args on the TokenSpeed
and SGLang wires, with `disaggregation_transfer_backend` mooncake, and
`kv_role` behind `kv_connector` MooncakeConnector on the vLLM one, whose info
is a struct of named fields the admin API's info patch sets by name: the mock
moves no KV cache and names Mooncake as its transport on every wire). The
encode role (`--worker-type encode`) is the two wires' with an encoder. On the
vLLM wire the worker advertises `ec_role` ec_producer behind `ec_connector`
ECCPUConnector (a prefill worker on that wire advertises the consumer side),
reads as vision-capable with a worker-side media processor (`mm_processor`
mock, so the gateway forwards media references to it), and answers every
`Generate` that carries `ec_transfer_params_json` with the encoder's token-less
completion, the media identity (the prompt ids as sent, one hash per media
item) and one pull handle per hash naming its own address. On the TokenSpeed
wire the worker mounts the engine's encoder service beside the scheduler's:
its `Encode` takes a request's one media item with its rendezvous room and a
request id (refused without, as the engine's adapter refuses them), runs the
realistic engine's admission hooks, counts in the admin API's `encodes` and
answers accepted; it reports health under the encoder service's name, as the
engine does in that role; its `Generate` is refused as the wrong pool, and a
prefill worker on that wire refuses a request whose `encode_bootstrap_info`
does not name a rendezvous for each media item once. A request reaches that
`Encode` only when the gateway processes the media itself, which it does for
a model whose family it has a media spec for, matched by a model id that
names the family (`llava` in it) or by the `model_type` in the model
directory's `config.json`: with the default model id and a text model's
tokenizer a chat request that carries an image is refused with
`multimodal_not_supported`. A model directory of a vision family serves it:
every worker is started with `--model mock-llava --tokenizer <dir>` and
registered with `<dir>` as its tokenizer path, which the gateway reads itself
(the mock serves no tokenizer of its own: where the gateway cannot read `<dir>`, the
model's requests are answered with `tokenizer_load_failed` once the load has
failed), and `<dir>` holds a `config.json` of the llava family (`model_type`,
`image_token_index`, the vision tower's `patch_size` and `image_size`),
tokenizer files with the token `<image>`, a `preprocessor_config.json` of a
CLIP image processor, and a chat template whose prompt holds one `<image>`
for each image: a template that walks the content parts renders an image
part as `<image>`, and one that prints the content as a string is given the
tag by the gateway. A prompt with another count of the tag (a walk of the
parts with no clause for an image part, a literal `<image>` in the text) is
refused with `multimodal_prompt_contract_mismatch`; with the right count the
request is answered, the encode worker takes one `Encode` for each image, and
a prefill and a decode worker take the request. That holds with the gateway's
Grand Central off and its media processing at `auto` or `router` (the
defaults are off and `auto`); with `--mm-processing worker` the gateway does
not process the media and refuses the request with `multimodal_not_supported`.
The SGLang wire has no encode role. On the TokenSpeed and the SGLang wires a
request's media items are checked against the shape the engine's adapter
requires before the canned answer: a known modality, the TokenSpeed item's
content hash and, on the prefill and regular roles, its encoder input (or the
encode room that hands the item over; a decode worker takes the gateway's
pixel-free clone, metadata alone), the SGLang batch's pixel tensor (or its
feature hash) and a video's placeholder token id, every tensor read as the
engine reads it (bytes matching the shape at the dtype's width; the SGLang
reader refuses a dtype outside its six, the TokenSpeed reader passes one it
does not know; on the TokenSpeed wire an inline payload through the adapter's
reader, an SHM handle with the servicer's checks: that byte length, a bare
name and, for the scheduler's feature, a dtype of bfloat16, float16 or float32,
the encoder reading its feature generically, a remote handle on a feature as
it came, pulled by the servicer, and a remote handle on a side tensor refused
as the servicer's generic reader refuses it), one SGLang batch per modality
(image or
video, an empty list reading as image) beside a primary batch, placeholder spans
inside the prompt and disjoint across the request's items, and, when the item
names its placeholder token id, spans that cover exactly the prompt's tokens of
that id; a wrong item is refused with `InvalidArgument` naming the item and the
field, so a gateway regression in a wire's items reads as a red row on the mock.
A regular worker takes a request with media once the gateway lets it through:
on every wire the gateway asks the model's family for a media spec (the model
directory above), and on the vLLM wire it also asks the workers, refusing the
request with `multimodal_not_supported` while every worker that would take the
pixels reports `supports_vision` false, as a regular worker of the mock does
unless `--supports-vision true` or the admin API's info patch says otherwise
before the gateway reads the model info. The realistic engine keys a media
item's tokens by the item. The gateway writes the same placeholder id under
every image, so the cache reads the ids under each placeholder span as one id
of the item's hash (below 2^30): on the vLLM wire the i-th entry of a batch's
`mm_hashes` over the batch's i-th span, on the TokenSpeed wire an item's
`content_hash` with its modality over the item's spans, on the SGLang wire one
value over all the spans of a batch (its `feature_hash`, or a fold of its
pixel tensor where the request carries none, taken where the cache reads the
request, so a prefill-role or scripted request does not pay for it: one batch
sent both ways reads as two items, and the mock cuts no batch into items by a
model family's grid). Where the wire names the item's placeholder id, a token
of another id under a span stays as sent; a token two spans claim goes to the
first of them, in the prompt's order, that takes its id. A vLLM request's
media is read one way, as the engine's servicer reads it: where a batch
carries tensors (`pixel_values` or `model_specific_tensors`), the batches with
tensors key their spans and a batch without them keys nothing; where none
does, as on the decode side of a pair that holds the language model alone,
every batch's hashes, in order, salt the whole prompt, folded with the first
token into that token's key (an empty prompt has no token to carry them). The
prompt's token count stays, a script and `--capture` read the ids as sent, the
same image is the same prompt to the cache, and another image parts the prompt
at the image's first token (`cached_tokens`, `GET /admin/truth`).
`--output-seed prompt` reads the prompt as the cache does: the same image
gives the same output, another image another. The KV
events follow each wire's engine. On the vLLM wire a stored block carries the
ids as sent and the item is in the block's hash alone, so an index built from
the events reads a prompt by its ids, two images under one placeholder id as
one prompt, as it does of the engine's events. On the SGLang and TokenSpeed
wires, whose engines write the item's id under its tokens themselves, the
events carry the keyed ids (read in the engines' servicers and clients, not
compared with an engine's events).
A reference batch on the SGLang wire (media URLs without a pixel tensor, the
form the gateway hands a worker that processes media itself) is the primary
batch's alone, travels with no extra batch, and is accepted as it stands: the
worker's processor or an encoder would resolve it, and the mock has neither.
`--capture` records the TokenSpeed wire's requests only. Beside the scheduler
service the server hosts the standard `grpc.health.v1` service (`Check`,
`Watch`): the empty name and the served scheduler service are SERVING exactly
when `HealthCheck` answers healthy (`POST /admin/serving/{port}` flips both),
any other name is unknown; the gateway detects a worker's runtime by the
scheduler service that answers its health check, so a worker serves one.

**ZMQ** (`--zmq-handshake <ipc://…> --zmq-count N [--zmq-start-index I]`):
mock vLLM `EngineCore` ranks on the `engine-zmq-client` wire. Unlike the HTTP
and gRPC workers, which bind and are dialed, each rank dials the frontend's
handshake address, registers, takes `EngineCoreRequest`s and pushes
`EngineCoreOutputs` with the engine's load as `scheduler_stats`; the admin API
sees it as worker `zmq:<index>`. `--zmq-step-ms <ms>` has the ranks send an
engine's steps: one message per period carrying the next token of every
running request (a request that produced faster than the period waits its
turn, one output per step), instead of one message per token;
`GET /admin/zmq/zmq:<index>` reads what the rank pushed since it connected,
the messages that carried request outputs and the outputs in them
(cumulative; a reset does not clear them). A rank waits for the frontend to
bind its handshake endpoint in three rounds of 30 s, each round it misses
logged (a gateway binds the handshake of a worker it was started with at that
worker's first health probe, up to one check interval after its own start),
then gives up.

A worker that stops on its own (a port already in use, a rank whose frontend
never bound) is logged and the others keep serving; when the last one stops
the process exits with status 1, so a launcher sees a fleet that serves
nothing as the failed start it is instead of waiting on it. The admin API is
not a worker: it neither counts nor keeps the process up alone.

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
  while budget and KV room remain. A queued request whose client has gone (its
  stream closed with no abort) leaves the queue at the next pass, without a
  prefill, as a running one does. With `--prefill-first true` (SGLang) a pass
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
  decode (plus any restore from the shared KV tier, below), and the first
  token of a prefill adds no decode time.
  `--timing linear` is the simpler model: prefill at `--prefill-tps` tokens/s,
  decode `base + per-request · batch` ms.
- **cached tokens** — a request sharing a prefix with cached blocks pays less
  prefill and reports `cached_tokens` (gRPC chunks, HTTP
  `usage.prompt_tokens_details.cached_tokens`).
- **shared KV tier** — `--global-l3-blocks <n>` gives the process one LRU
  store of `n` blocks that every engine writes its completed blocks to, as
  engines do behind a 3FS- or Mooncake-style pool; a pass's blocks become
  restorable at its end, as its events become visible. An admitted prompt
  restores the run of blocks the tier holds right after its local prefix
  hit: those blocks take KV room but no token budget, count in
  `cached_tokens`, and add `--global-l3-ms-per-token` per restored token to
  the pass (a synchronous load). The tier is per process (a fleet split over
  several processes has one tier each), and a reset (`POST /admin/reset`)
  leaves it as it is: a restarted engine does not empty a shared store. A
  queued request's `num_waiting_uncached_tokens` leaves out what the tier
  would restore. The tier needs generating workers (a prefill or encode worker
  runs no passes, but a gRPC prefill worker under `--prefill-engine true`
  does), `--prefix-cache true` and one main-attention KV cache group
  (`full_attention`, `mla_attention` or `sink_full_attention`): the tier holds
  whole blocks of one group.
- **repeatable output** — `--output-seed prompt` derives synthetic output from
  the complete prompt token sequence, including its partial last block. Equal
  token inputs then produce equal output regardless of transport request ID,
  worker placement, cache warmth, or preemption. Repeated prompts can share
  decode blocks too. Request IDs still identify cancellation and records.
  The default `request-id` mode preserves existing stress workloads. This option
  applies to the realistic engine; canned and scripted responses are unchanged.
  Synthetic output remains in the existing token range 100–30099. Where
  `--tokenizer` names local tokenizer files, an ID of that range gives way to
  another when the tokenizer does not decode it to text on its own (an ID
  outside its vocabulary, a special token, a part of a character) or it is an
  end-of-sequence ID; the other IDs stay as drawn. A tokenizer that decodes
  no ID of the range leaves the output as drawn, as does a worker with no
  local tokenizer: a benchmark must then separately verify those IDs against
  its pinned tokenizer and special IDs. The worker logs a warning when local
  tokenizer files give no such table (they do not load, or decode no ID of
  the range); a `--tokenizer` that is no local path stays quiet. An HTTP
  worker sends text and no IDs, so it loads no tokenizer for this.

| Flag | Default | Meaning |
|------|---------|---------|
| `--timing` | polynomial | `polynomial`, `linear`, or `fit:<path>` (a hardware calibration JSON, below) |
| `--output-seed` | request-id | `prompt` selects output from normalized input tokens for cross-router/retry comparisons |
| `--prefill-poly a,b,c` | AISimulate | prefill ms = a + b·T + c·T² |
| `--decode-poly a,b,c` | AISimulate | decode ms = max(1, a + b·u + c·u²) |
| `--request-overhead-ms` | 0 | fixed per-request latency added to every event of a stream (TTFT and e2e grow by it, ITL does not) |
| `--prefill-tps` | 8000 | linear model: prefill tokens/s (selects `linear`) |
| `--decode-base-ms` | 6.0 | linear model: fixed decode-pass ms |
| `--decode-per-req-ms` | 0.35 | linear model: decode ms per running request |
| `--max-batched-tokens` | 8192 | token budget per pass |
| `--max-running` | 256 | sequences per pass |
| `--kv-tokens` / `--kv-blocks` | 524288 tokens | KV pool capacity |
| `--worker-kv-tokens` | none | `worker=tokens,...`: the named workers' pool capacity instead, at most the fleet's or a calibration's pool, whichever is larger (any size under `--timing linear`, whose decode reads no utilisation; workers named `grpc:<port>`, `http:<port>` or `zmq:<index>`; a data-parallel worker is named once and every rank takes the override). The pool's size is room, not speed: the worker's decode keeps the fleet's pool as its utilisation reference (a calibration's reference when one gave it), so it steps as fast as its peers on the same context. Generating workers only: a prefill or encode worker answers without running its engine, but a gRPC prefill worker under `--prefill-engine true` takes it |
| `--worker-pass-time-scale` | none | `worker=factor,...`: every pass of the named workers takes `factor` times the timing model's duration (0.001 to 1000; above 1 a slower engine, below 1 a faster one); the rates the worker reports follow, as a servicer measures them. Named and limited as `--worker-kv-tokens` |
| `--global-l3-blocks` | 0 (off) | capacity of the process-wide shared KV tier, in blocks (above) |
| `--global-l3-ms-per-token` | 0 | pass time per token restored from the shared tier |
| `--block-size` | 16 | cache block/page size (tokens); must match the worker's `kv_block_size` |
| `--prefix-cache` | true | prefix caching + KV events |
| `--grpc-kv-events` | true | serve the gRPC subscription to KV-cache events; `false` refuses it as an engine without the call does, so a gateway hears no KV event from the process over gRPC, while the engines keep their prefix cache and the load records stay served (the ZMQ publishers are not touched) |
| `--prefill-first` | false | SGLang-style prefill-only passes |
| `--reserve-full-isl` | true | admit a waiting prompt only when the free and evictable blocks could hold all of it beyond its cached blocks, stopping at the first that does not fit (vLLM's `scheduler_reserve_full_isl`: a gate, only the pass's chunk is allocated at admission); `false` skips the gate |
| `--context-length` | 32768 | advertised context length |
| `--output-prefix-ids` | none | the token ids every output starts with, before the synthetic ids: a stream shaped for a gateway that parses the engine's tokens (a channel header, say); the canned gRPC and ZMQ workers and the realistic engine shape their output with it (the canned HTTP worker does not read it) |
| `--loads-like` | mock | `vllm`: report only what the vLLM servicer reports (running, waiting, `token_usage`, maxima), so the gateway's expected-wait routes as it does on a vLLM fleet |
| `--supports-vision` | unset | what the gRPC workers' model info says of vision input from their start: the `supports_vision` entry of the info the admin API patches (`/admin/info/{port}`), seeded; unset, an encode worker on the vLLM wire says `true` and every other worker `false` |
| `--admin-port` | off | process-wide admin API (below) |
| `--drain-timeout-secs` | 5 | on SIGTERM or SIGINT: stop accepting, let the requests in flight finish for this long, then exit 0 either way |
| `--prefill-engine` | false | a gRPC worker in the prefill role runs each prompt on its engine as a request of no output, so the engine's prefix cache, its request records, its KV events and its load record see the prompt, and the prompt side's completion carries the engine's cached count once the prompt is prefilled; such a worker runs its engine's passes, so it takes `--worker-kv-tokens`, `--worker-pass-time-scale` and `--global-l3-blocks` as a generating worker does; `false` answers without the engine (below) |
| `--kv-events-zmq-base-port` | off | ZMQ KV-event publishers on vLLM's or SGLang's wire (below) |
| `--kv-cache-groups` | `full_attention` | the KV cache groups a realistic engine publishes, `kind[:window],...`, one to eight of `full_attention`, `mla_attention`, `sink_full_attention`, `sliding_window:<tokens>`, `sliding_window_mla:<tokens>` and `mamba` (at most one of them one of the three main-attention kinds, at any index, and a model without one starts with a window kind; a window kind with its window, the others without): every group's events carry its index, kind and window, a window group stores only the window tail ending at the prompt's aligned replay boundary and a Mamba group the one state block there (vLLM 0.31.0's default retention), a sparse store's `token_ids` on the vLLM wire spanning the whole step as the engine writes it, and an eviction is published once per group, every group's in one batch from the mock's one pool where the engine's per-group pools may stagger them; one group only on the SGLang wire. The gRPC stream carries every group's events unfiltered (a servicer's relay forwards every group, a sparse group's store as its hashes; the gateway indexes the main group and keeps the other groups' blocks in a per-worker view that bounds its overlap the way the engine does), and the mock's own prefix hit and `cached_tokens` are the engine's fixed point over its groups (a window group's tail ending at the hit or its leading run, a Mamba group's rightmost state block, iterated until nothing moves) |

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
  `lora_id`, `medium: "GPU"`, `lora_name`, `group_idx`,
  `kv_cache_spec_kind` and, for a window group, `kv_cache_spec_sliding_window`;
  one store per cache group of `--kv-cache-groups`, a sparse group's store
  carrying the step's whole `token_ids` span while its `block_hashes` name
  only the blocks the group retains, as the engine's `cache_full_blocks`
  writes a masked group), `BlockRemoved` (`type`,
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
- `GET /admin/fleet` — every worker of the process (`grpc:<port>` /
  `http:<port>` / `zmq:<index>`): a realistic engine with cache size, load,
  cached blocks and preemptions under `"engine": "realistic"` and, in a
  canned process, each listener with its name under `"engine": "canned"`
  and, on an HTTP or gRPC worker, `requests`: the generation requests that
  reached it since the process started, one it refused for its rendezvous or
  failed by a fault hook included, one its server could not read (a message
  the wire refuses, a body over the size limit) left out (a realistic
  process lists its engines alone, since a listener
  registers before its engine spawns; a listener joins the fleet once its
  bootstrap port, when it has one, is bound, so a worker that stopped at
  that bind is never listed); with the shared KV tier on, `global_l3` beside
  them: `blocks`, what the tier holds, and `restored_tokens`, every token
  restored from it (a preempted request's restores on re-admission included,
  which its record does not count);
- `GET /admin/requests?since=<seq>&limit=<n>` — admitted requests with the
  serving worker, prompt/cached tokens (`restored_tokens`: those of the cached
  tokens restored from the shared KV tier), queue wait and the **arrival-time
  oracle**: the most cached tokens any worker of the process held when the
  request arrived, with the shared tier on counting what the worker would
  restore after its own hit (the best a router could have obtained). Join on the
  gateway's response `id`. `injected_failures` beside the records counts, per
  worker, the requests the `fail` hook answered before admission (they left no
  record) and the streams it cut;
- `GET /admin/cache/{worker}` — the worker's cached block keys;
- `GET /admin/ops/{worker}` — what the gateway's admin fan-out reached on the
  worker: `flushes`, `profiles_started`, `profiles_stopped`, whether a run is
  `profiling`, the `last_trace` written, and beside them `requests`, the
  generation requests the worker received, in either engine mode (`all`:
  every worker);
- `POST /admin/reset[/{worker}]` — clear caches and publish `AllBlocksCleared`
  (an engine restart, to the index).

**Tokenizer note (gRPC):** the gateway tokenizes prompts before routing, so it
needs a real tokenizer for the model. Register each worker with a tokenizer
label, e.g. `"labels":{"tokenizer_path":"gpt2"}`, and a `"kv_block_size":16`, and
do **not** pass `--disable-tokenizer-autoload`. (HTTP workers need no tokenizer
but cannot drive event-driven `cache_aware`, which requires token ids.)

### Fault hooks

All under the admin API; `{worker}` is a worker name (`grpc:<port>`,
`http:<port>`, `zmq:<index>`) or `all`. The name of a gRPC worker with
data-parallel ranks addresses each of its ranks (engines of their own, named
`grpc:<port>:r<k>`), and a rank's name that rank alone. A hook applies to every KV-event
transport of the worker (gRPC `SubscribeKvEvents` and the ZMQ publisher alike)
and answers with the worker's hook state (`drop_pending`, `dropped_total`,
`delay_ms`, `admit_delay_ms`, `admit_per_sec`, `cold_shape_ms`, `load_lag_ms`, `load_frozen`, `paused`, `generation`,
`restarts`, and the request fault's `fail_status`, `fail_pending`,
`fail_secs_left`, `fail_after_tokens`, `fail_finish_error`, `fail_stall_ms`, `failed_total`,
`cut_total`, `stalled_total`, `load_fail_status`, `load_muted`). The hooks need `--engine realistic`: the canned
path has no engine to hook.

| Hook | Effect |
|------|--------|
| `POST /admin/fault/{worker}/drop?batches=N` | the next N event batches are not published (lost on the wire; they stay in the replay buffer, so a gap replay recovers them) |
| `POST /admin/fault/{worker}/delay?ms=D` | every batch is published D ms after its pass ends (0 clears) |
| `POST /admin/fault/{worker}/admit-delay?ms=D` | every new gRPC request is held D ms before the engine sees it (0 clears): the gateway has dispatched it, the load record does not count it yet, as a backlog in transit between the two would behave |
| `POST /admin/fault/{worker}/admit-rate?per_sec=R` | at most R new gRPC requests per second enter the engine (0 clears); the rest wait their turn, unseen by the load record: a throttled input path whose backlog grows while the gateway keeps sending |
| `POST /admin/fault/{worker}/cold-shape?ms=D` | the engine holds D ms before the pass that first runs a prompt longer than any it has run (0 clears): a kernel built for a new shape. It produces nothing meanwhile, its load record keeps flowing with its silence growing; paid once per new longest prompt, and the lengths run before the hook was set count as run |
| `POST /admin/fault/{worker}/load-lag?ms=D` | every load report of the worker (`GetLoads`, `/v1/loads`, the pushed load record, a ZMQ rank's scheduler stats) states the counts the engine published D ms before the report is read (0 clears; a minute at most, a larger D is taken as 60000): an engine whose counters trail its intake. The requests are served on time, and what a servicer counts itself (the pushed record's silence and pending prefill tokens) does not trail, and `GET /admin/fleet` keeps stating the engine's own counts. The report that stood when the lag was set is stated for the first D ms; a lag set over a set one keeps its history |
| `POST /admin/fault/{worker}/load-freeze?on=true` | every load report of the worker (`GetLoads`, `/v1/loads`, the pushed load record, a ZMQ rank's scheduler stats) states what a report read when the hook was set, until `on=false`: an engine that serves its requests while its counters stand still. The requests are served, what a servicer counts itself (the pushed record's silence and pending prefill tokens) moves on, and `GET /admin/fleet` keeps stating the engine's own counts. Set under a `load-lag`, the trailing report of that moment stands, and a `load-lag` set while the hook is on starts from the engine's own counts of that moment, which a report states once the hook is off; set again while it is set, the report that stands stays |
| `POST /admin/fault/{worker}/restart-publisher` | the publisher restarts: gRPC sequence numbers start over at 1 and the ZMQ sequence at 0, the replay buffers are emptied, the cache is kept (no `AllBlocksCleared`); `generation` increments |
| `POST /admin/fault/{worker}/pause` | the engine freezes after its current pass: no passes, no tokens, no events; requests queue (and count as waiting); health and `GetLoads` keep answering |
| `POST /admin/fault/{worker}/resume` | the engine runs again |
| `POST /admin/fault/{worker}/fail?status=S[&count=N\|&secs=T][&after_tokens=K][&stall_ms=M][&finish=error]` | the worker answers status S (400-599) instead of serving: the next N requests, every request for T seconds, or every request until cleared; an HTTP worker answers the status with `{"error": {"message": "injected", "type": "fault"}}`, a gRPC worker the status code the gateway maps back to it (429 `resource_exhausted`, 500 `internal`, 502/503 `unavailable`, 504 `deadline_exceeded`). Before admission, so the request never counts as served and leaves no record. With `after_tokens=K` the request is admitted and served for K tokens, then the stream is cut with S (the gRPC trailer; an HTTP body that ends without its finish frame): the non-retryable case, for a duplication check (a non-streaming HTTP request is answered with S after K tokens instead, which the gateway may retry: the duplicated-work case). On the gRPC worker 408 reads back at the gateway as 504 and 502 as 503 (both retryable), 499 as 502 (`cancelled`, retryable), any other 4xx as 400. `finish=error` ends the stream the way an engine that failed the request does, with no status needed: a gRPC worker sends a `Complete` whose finish reason is `error` after K tokens (0 when `after_tokens` is unset), the shape the gateway turns into a stream error for the client; an HTTP worker, whose wire has no such frame, cuts with 500 instead. `stall_ms=M` holds the request M ms first; `status=0&stall_ms=M` stalls and then serves. `status=0` alone clears |
| `POST /admin/fault/{worker}/load-fail?status=S` | every load poll of the worker is answered with status S (400-599) instead of a report, until `status=0`: `/v1/loads` answers S with the `fail` hook's body, `GetLoads` the status code the `fail` hook maps S to (501 `unimplemented`, a worker that does not implement the call; 502/503 `unavailable`; 422 the code alone, with no constraint trailer). A data-parallel rank under the hook has no row in the worker's answer, and a poll the hook leaves with no row is answered with S. The requests are served, and what the worker pushes (the load record, a ZMQ rank's scheduler stats) is no poll and keeps flowing: a worker whose load call fails or is missing, and once cleared one that reports again. A gateway's load monitor takes a `/v1/loads` answered 404, 405 or 501 as a route the worker does not have and asks it no more until the worker leaves the ready state, is replaced or is removed: 503 is the status for a poll that is to come back |
| `POST /admin/fault/{worker}/load-mute?on=true\|false` | while `on=true` the worker pushes no load report: a batch of its KV events goes out without a load record (its sequence still counts as the rank's last), no `load_only` batch of it is sent on an event stream or on a `loads_only` subscription, and a ZMQ rank's request outputs carry no scheduler stats (a rank that sends in steps, `--zmq-step-ms`, may still send the stats it took before the hook was set with the step under way). Its load polls (`/v1/loads`, `GetLoads`) are answered and its requests are served: a worker whose pushed reports stop, so that the last one a gateway holds ages. An HTTP worker pushes none and the hook changes nothing there. `on=false` and the worker reports as before, on the batches that carry a report: its next batch of events, its next `load_only` batch once the record moved from the last one sent or the rank's heartbeat is due, and a ZMQ rank's next request output that carries stats |
| `GET /admin/fault/{worker}` | the hooks' current state (pending drops, delay, restarts, paused, generation, the request fault and its counts) |
| `POST /admin/reset/{worker}` | (already there) clear the cache and publish `AllBlocksCleared` |

### Truth endpoints

| Endpoint | Answer |
|----------|--------|
| `POST /admin/truth/{worker}` with `{"token_ids": [...]}` | what the worker would serve from cache for that prompt right now: `cached_tokens`, `cached_blocks`, `restored_tokens`, `block_size` (the engine's own prefix match and, with the shared KV tier on, the run the tier holds after it, as `restored_tokens`; last-block rule included); the ids are read as sent, with no media item beside them, so of a prompt with media it reads only the blocks in front of the first item, and nothing of a prompt its hashes salt |
| `GET /admin/truth` | per worker, over every admitted request: `requests`, `prompt_tokens`, `cached_tokens`, `oracle_tokens`, `restored_tokens` (the shared tier's share of `cached_tokens`), so a gateway's hit-rate claim can be checked against what the engines actually served; `injected_failures` and `cut_streams` count what the `fail` hook refused before admission and cut after output |
| `GET /admin/zmq/{worker}` | a ZMQ rank's output counts since it connected: `batch_messages`, the messages that carried request outputs, and `outputs`, the outputs in them (one per message without a step period, several with one; a utility reply is neither); cumulative, `POST /admin/reset` does not clear them; a rank name, not `all` |

A circuit-breaker or retry drill arms `fail` on one worker (`status=503&count=N`
for exactly N failures inside the breaker's window, `secs=T` for a worker that
stays up but failing), sends requests through the gateway and reads
`failed_total` against the gateway's retry and breaker metrics; a retried
request that reached a worker shows in `/admin/requests` once per admission,
so `requests` there against `injected_failures` tells retried attempts from
duplicated work. `status=499` makes a gRPC worker answer `cancelled`, the code
a call gets when the worker's connection closes under it (a worker dying as
the request arrives); an HTTP worker answers the bare 499.

## Prefill/decode roles

`--worker-type prefill|decode` gives a process's HTTP workers alone, or its
gRPC workers and ZMQ ranks together, a role in a disaggregated fleet, so a
gateway in PD mode (`--prefill <url> <bootstrap port> --decode <url>`) runs
with no engine behind it. The
gateway detects the mock by the scheduler service it serves (or, on the HTTP
wire, by its model list) and runs that wire's PD protocol: both sides get the
same request, with the rendezvous the gateway mints (the wire's rendezvous
field on the gRPC wires, named below; `bootstrap_host`, `bootstrap_port` and
`bootstrap_room` in the request body on the HTTP wire, whose PD router is its
own code path), and the client reads the decode side's output.

A ZMQ rank declares the role in its handshake's ready response as the engines
do (`disaggregation_mode`, the request types it takes, and for a prefill the
`bootstrap_host` and `bootstrap_port` of the port it binds, numbered after
the gRPC workers: rank `I`, counting `--zmq-start-index`, binds
`base + grpc_count + I` of `--bootstrap-base-port <base>`), which is where a
gateway checks a worker registered for a role. A ZMQ prefill rank needs the
base port: its handshake is the only place the port reaches the gateway, so
the flags are refused without one. A prefill rank answers the prompt side
alone, as on the gRPC wires; the vLLM dialect's request carries no
rendezvous, so there is none to read. The encode role stays the gRPC wire's.

- A **prefill** worker answers every request with the prompt side's
  completion alone (`prompt_tokens`, no output, `stop`), ahead of the script
  and of either engine mode, and advertises `disaggregation_mode: prefill`
  (in its server args on the gRPC wires, in `/server_info` on the HTTP wire).
  With `--bootstrap-base-port <port>`, worker `i` also binds `port + i`,
  where a decode partner's engine would rendezvous for the KV handoff: it
  accepts and closes each connection (the gateway probes the port once the
  worker is registered) and advertises it as
  `disaggregation_bootstrap_port`.
  With `--prefill-engine true` a realistic gRPC prefill worker first runs
  the prompt on its engine as a request of no output, as a regular worker
  prefills it: the prefix cache holds the prompt, `/admin/requests` records
  it with its cached and oracle tokens, the KV events name its blocks and
  the load record counts it as pending prefill until its answer stream is
  read to the engine's end or dropped, as a regular worker's request. The
  answer is the same completion, sent once the engine has prefilled the
  prompt (the engine's prefill time in place of `--gen-ms`), with the
  engine's `cached_tokens`. An armed `finish=error` ends it with `error`
  whatever the engine's end; with none armed, a prompt the engine aborts
  ends with `abort`. The flag needs `--engine realistic` and gRPC workers
  with no ZMQ rank: an HTTP worker and a ZMQ rank in the prefill role
  answer without the engine.
- A **decode** worker generates as a regular worker (canned, realistic or
  scripted: the answers a test states go to the decode side) and advertises
  `disaggregation_mode: decode`.

- Either side refuses a request that carries no rendezvous, as the engines'
  disaggregated servers answer one: on the HTTP wire a 400 for a missing
  `bootstrap_room`, on the gRPC wires `InvalidArgument` for a missing
  `kv_bootstrap_info` (TokenSpeed), `disaggregated_params` (SGLang) or
  `kv_transfer_params_json` (vLLM, the typed `kv_transfer_params` also
  taken, and refused under its own name when it names no host or port), or
  for a named rendezvous without a host or a port (a room of 0 is one the
  gateway can mint, and is taken). On the
  vLLM wire the decode side alone refuses: the gateway names the pair to it,
  and the prefill request carries the connector's parameters only when a
  connector relays them. A prefill worker refuses one whose bootstrap port is
  not the port it bound: its decode partner would rendezvous elsewhere for
  the handoff this mock stands in for. On the HTTP wire a missing or null
  `bootstrap_port` reads as the engines' default, 8998 (what the gateway sends
  for a prefill worker registered without a port): the worker on that port
  takes it, one on another refuses it.

No KV moves between the two: the decode side has the prompt. One process
gives the role to its HTTP workers alone or to its gRPC workers and ZMQ ranks
together (an HTTP worker `i` and a gRPC worker `i` would bind the same
bootstrap port; the ZMQ ranks bind after the gRPC workers); the encode role
is the vLLM and the TokenSpeed gRPC wires'.

Every gRPC worker advertises the pairing protocol its environment names
(`SMG_PAIRING_PROTOCOL`) as `pairing_protocol`, as the engine servicers do:
a prefill and a decode that name the same one pair; name different ones for
the two to see the gateway refuse the pair.

In realistic mode (`--engine realistic`) a prefill worker, on either wire, honors the
[fault hooks](#fault-hooks) as any request: the stall, a status before
admission (`fail?status=S`, counted under `failed_total`), the admission
delay and rate (`pause` freezes the passes, which the prompt side never
takes). With no output to cut after, an armed cut (`after_tokens=K`) fires
as its status at once, and `finish=error` ends the prompt side with `error`
whatever the status (a 500 on `/v1/messages`, which has no `error` stop
reason); both count under `cut_total`, as on any worker. The
engine itself never runs the prompt: nothing is admitted and no record is
left. A gRPC worker under `--prefill-engine true` is the exception: after
these hooks its engine admits and records the prompt and prefills it in
its passes, so `pause` holds the prompt side's completion until `resume`.

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

## Scripted output

What the model says can be stated ahead of the request, on every wire (HTTP,
gRPC, ZMQ), ahead of either engine mode. Two sources feed one store:

- `--script PATH`: a JSONL of answers keyed by the prompt's token ids, for
  recorded corpora and bisects. One answer per line:

  ```json
  {"input_ids": [151644, 872, 198], "output_ids": [9707, 11, 1879], "output_text": ["Hello", ",", " world"],
   "finish_reason": "stop", "matched_stop": 151645, "cached_tokens": 0, "prompt_tokens": 3,
   "logprobs": [-0.1, -0.2, -0.3], "delay_ms": 0, "chunk_tokens": 1}
  ```

  `key` may stand in for `input_ids`: `<len>:<fold>`, the prompt's length and
  the FNV-style fold of its ids (`h = (h * 1099511628211) ^ id` over a `u64`
  from 0, wrapping). Everything but one of `output_ids` / `output_text` is
  optional: `finish_reason` defaults to `stop` (to `error` on a line that
  states a `status`, whose `code` is the number of a gRPC failure, 1 to 16:
  a line whose call ends well states none); `matched_stop` is a token id
  or a string; `prompt_tokens` defaults to the request's own count;
  `logprobs` holds the sampled token's logprob per output token and is sent
  when the request asks for logprobs; `token_id_logprobs` holds the logprob
  of each token id a scoring request asks for (`token_ids_logprob` on the
  SGLang wire), in the request's order, on an answer of no tokens
  (`"output_text": ""`), for a request that asks for logprobs, as the engine
  computes the row only then; `delay_ms` is the pause before every
  frame but the first; `chunk_tokens` the tokens per frame (1 by default).
- `POST /admin/script/{worker}` on the admin API, with one such answer or an
  array of them: the answers the worker's next requests get, in order
  (`all`: whichever worker gets the request). An answer with a `key` or
  `input_ids` is filed under its prompt for every worker instead. An answer
  with a `prompt_contains` text is filed for the first request whose prompt
  holds the text, served once, whatever order the requests arrive in: what
  a test states when its requests run concurrently, or when the gateway
  picks the worker. `GET /admin/script/{worker}` reads `queued`, `marked`
  (the answers filed under a text), `hits`, `misses` and the `keyed` count;
  `DELETE` drops the queue and the answers filed under a text; a name no
  worker of the process has is a 404. This is the form a test wants: it
  states the model's answer right before it sends the request, no tokenizer
  prediction needed.

For a request a worker looks in this order: its own queue, the shared queue,
the answers filed under a text its prompt holds (its own, then the shared
ones), the keyed answers under its prompt's key. Nothing found is a miss,
counted per worker, and the request goes to the canned or realistic engine as
if no script existed (a gRPC worker fails it instead once a recording is
loaded: [Recording an engine](#recording-an-engine)). A scripted answer is served by the wire, not by the
engine simulator: no load, no KV events, no request record. The prompt's text
is what an HTTP worker was sent (the Messages request's top-level `system`,
a string or text blocks, read ahead of the messages; a message's string
content or the `text` of its parts; a prompt string or the strings of a
prompt array); a gRPC worker
reads the rendered text the gateway sends beside the ids (`original_text`)
and decodes the ids with its `--tokenizer` only when that is empty, as a ZMQ
worker always does. Either reads the prompt only while answers filed under a
text wait for it, and a prompt it cannot read then fails the request with
the reason, the filed answers left where they are.

Per wire: a gRPC worker streams `chunk_tokens` ids per `Chunk` (the cuts of
`chunks` where the answer states them; cuts that hold none of an output,
an empty list among them, stream a `Chunk` of no ids each and leave the
output to the `Complete`) and ends with the `Complete`
(`output_ids`, `finish_reason`, `matched_stop`, `cached_tokens`,
`output_logprobs` when `return_logprob` is set), or with the `status` of an
answer that states one; a ZMQ rank
pushes one `EngineCoreOutputs` batch per frame, the prompt's prefill stats
on the first, the finish reason and `stop_reason` on the last; an HTTP worker
answers `/v1/chat/completions`, `/v1/completions`, `/generate` and `/v1/messages` with the
`output_text` pieces (one SSE frame per `chunk_tokens` pieces, the finish
frame with `matched_stop`, under `stream_options.include_usage` a usage
chunk of its own, then `[DONE]`; or one JSON body; on `/v1/messages` the pieces are
`content_block_delta` events between `message_start` and `message_stop`, the
`message_delta` carrying the stop reason, a text stop as `stop_sequence` (a
token-id stop is none) and the output usage, with no usage chunk and no
`[DONE]`; a scripted or armed `error` finish answers a 500 there, as the
Messages shape has no `error` stop reason, and the gateway retries a 500 on
another worker, so a scripted error on this route takes the next queued
answer with it, while the completion routes pass a scripted `error` through
as a 200 with its finish reason) and keys its prompt on its synthetic ids, one
per whitespace word of the prompt text (`http::synth_token_ids`). An answer
without `output_ids` is encoded on the token wires with the worker's
`--tokenizer`, which must then name local tokenizer files (a directory with
`tokenizer.json`, or the file): the mock never fetches a tokenizer, and an
answer it cannot encode (no such files, or per-piece `logprobs` that do not
align with the encoded ids) fails the request with the reason instead of a
canned answer; one without `output_text` has no frames on the HTTP wire.

```bash
cargo run --release -p mock-worker -- \
  --grpc-base-port 19000 --grpc-count 1 --http-base-port 9000 --http-count 1 \
  --admin-port 9100 --model m --script answers.jsonl
curl -s -X POST localhost:9100/admin/script/grpc:19000 -H 'content-type: application/json' \
  -d '{"output_ids": [9707, 11, 1879], "finish_reason": "stop", "logprobs": [-0.1, -0.2, -0.3]}'
```

### Recording an engine

A gRPC worker can stand between a gateway and a real engine and write down
what the engine says; the file replays later with no engine behind it.

- `--record-upstream URL --record PATH` (a pair): the gRPC workers of the
  process run no engine of their own. Every call that reaches one, of any
  service and method, goes to the engine at `URL` as the messages and the
  metadata it came in, and the engine's answer comes back as the messages it
  sent, with the trailers of a call that ended well behind the last. A
  request message the recorder cannot read (one over
  `--grpc-max-message-bytes`) ends the request it hands on (a call of these
  wires holds one request message, so the engine is handed none), and the
  caller reads the status of that message, as a worker answers it itself. A
  call the transport ends with an error, ahead of the engine's headers or
  behind them, keeps the code a caller of the engine reads (`UNAVAILABLE`
  for an engine the recorder cannot reach, the code of its reason for a
  stream the engine reset) and names `the recorder's upstream` ahead of the
  reason; every other status passes as it is: one the engine sent, one the
  client makes of an answer that ended with no gRPC status, and one it
  makes of an answer it cannot read: a message over
  `--grpc-max-message-bytes` (`OUT_OF_RANGE`), one cut short or with a
  compressed flag it cannot follow (`INTERNAL`), an encoding it does not
  take (`UNIMPLEMENTED`).
  The recorder never fails a call for its own files: a write that fails
  is logged and the call goes on, and one that fails part way leaves a part
  of a line, which `--script` refuses with its line number. The recorder
  holds the bytes of a request until its call is over and sets no bound of
  its own on the size of its two files. A recorder that cannot start (an
  address that does not parse or names no scheme, a file that cannot be
  opened) ends the process at its start, as a capture file does: `URL` is
  `http://HOST:PORT` or `grpc://HOST:PORT`, and a bare `HOST:PORT` is
  refused before a file is made. Each `Generate`
  exchange of a scheduler wire (vLLM, TokenSpeed, SGLang) is appended to
  `PATH` when the call is over, one line for each choice the request asked
  (a call that failed or was let go early keeps a line for each all the
  same, so the calls of one key hold the same count of lines; a call whose
  request the recorder did not hand on, the engine not reached or the
  request message not read, writes none, as does one whose request does
  not read as a request of its wire: a replay holds nothing of such a
  call, and where its request reaches the script it finds what other
  calls of its key left, or nothing), as lines
  `--script` reads: `input_ids`, `output_ids` (`"output_text": ""` for an
  answer of no tokens), `chunks` (the tokens of each frame in turn, as the
  engine cut a streamed answer; a frame may hold none, and where no frame
  held a token the output is the one of the completion: its cuts hold none
  of it, an empty list for a streamed call answered by its completion
  alone), `finish_reason`
  (`error` where the call ended with a status ahead of the choice's
  completion, `abort` when the caller let go ahead of it), `matched_stop`,
  `prompt_tokens`, `cached_tokens` (where no completion came, the two the
  choice's last frame stated), `completion_tokens` (the engine's
  count: one more than the output where it took a matched stop token off),
  `logprobs`, `token_id_logprobs` (on the SGLang wire, the one scores row
  of a completion where it scores the token ids the request asked),
  `status` (`{"code", "message", "trailers"}`, the gRPC status
  the call ended with, on each of its lines, with the trailers named
  `smg-...` by which a worker says what kind of failure it reports),
  and under `recorded` the facts of the exchange: the `wire`, the choice's
  `index`, what the request `asked` beyond its prompt, the fold of that
  (`variant`) and the count of output ids its completion carried
  (`complete_ids`). Each gRPC worker of the process keeps the answers of
  `GetModelInfo` and `GetServerInfo` once each, as the bytes of the message
  in hex, in the info file: `PATH` with its extension replaced by
  `info.jsonl` (`session.jsonl` gives `session.info.jsonl`, and two records
  that differ only in their extension share one info file); an info answer
  whose line the file took nothing of is written by the next answer of its
  kind. A `GetTokenizer` fetch that ends well is kept there too, once a
  path, as one line: its chunks one behind the other, each behind its
  length, two bytes of the file for a byte of the bundle (a fetch of 2 MiB
  is a line of about 4 MiB). One fetch waits for the file at a time: a
  fetch that a status ended is not kept, and neither is one that begins
  while another waits, even where it alone ends well; the next fetch that
  begins with none waiting is kept. A fetch whose chunks sum above 200 MiB,
  what a gateway takes of a bundle at most, passes through and is not kept.
  Each gRPC worker keeps a fetch of its own, so with `--grpc-count` above
  one the file may hold one line of the path a worker; a replay reads the
  first. Of a
  `GetServerInfo` answer the SGLang and TokenSpeed wires hand two free-form
  maps, the arguments the engine was started with and its scheduler's
  state, and a start argument may be a key: the info file keeps of each map
  the entries the gateway reads as labels and no other. A `PATH` holds one
  session: a later one appends to it, and a replay answers an info call
  with the first answer the file holds. Both files hold what the session
  carried (prompts, outputs, the engine's words in a status, the info
  answers), so a file the recorder creates is readable by its owner
  only, as a capture file is. A file that exists already keeps its mode and
  its owner, and a link is followed, as with a capture file: whoever made
  it chose who reads it, so a session that must stay private starts on a
  new path. What a request asked keeps a media reference
  as the fold of its modality and URL, never as the URL, a batch of the
  vLLM wire that states no hash as one item with no hash at its place, and
  pixels that
  state no hash as the identity the SGLang wire gives their batch (a fold
  of the tensor, eight bytes a step) under their modality; the
  engine's words in a status are kept as they came.
- `--script PATH` on such a file replays it. A recorded line is filed under
  its prompt and what its request asked (`<len>:<fold>/<ask>`): whether it
  streams, the limit, the count of choices, the sampling fields, the stops,
  the logprobs, the constraint and the media, and where the request states
  one, a further field that shapes the answer: `min_p`, the frequency,
  presence and repetition penalties, `min_new_tokens`, `ignore_eos`,
  `no_stop_trim`, `logit_bias` and the token ids it asks a score for. A
  request that states none of these keeps the key it has on every wire. The
  lines of one key are served in turn, in file order, and from the first
  again; a frame carries the tokens of the recorded cut, and the completion
  the count the line states. On another wire than the line's
  `recorded.wire` that count is never below the output's ids: a wire whose
  stream a gateway counts frame by frame may state the last frame's tokens
  alone, where the completion of the other two carries the total. A
  recorded `status` ends the call, its trailers
  with it: the frames the call's lines hold go out first (a line whose
  finish reason is `error` is a choice the engine left open, sent without a
  completion), and a call that delivered nothing is answered with the
  status alone. The status of a call the engine reset at the recording,
  ahead of its headers or behind them, replays with the code and the words
  its caller read then, `the recorder's upstream` ahead of the reason.
  Once a recorded line is held, a gRPC worker of the process fails a request
  that finds nothing with `INTERNAL` and the key it missed (and whether the
  prompt is recorded, asked otherwise) instead of going on to its engine: a
  recording is replayed whole, or the test learns what was not recorded. A
  record file with no line beside its info file (what a session with no
  `Generate` call leaves) is such a recording: every request fails so. The
  HTTP and ZMQ workers do not read what a request asks, so no recorded line
  serves them. Through the admin API a keyed answer replaces what its key
  held, a recorded line too: a recording is loaded with `--script`.
- With the info file beside `PATH` (read beside a file that holds a recorded
  line or no line at all, never beside hand-written answers), a replay
  answers `GetModelInfo` and
  `GetServerInfo` with the recorded answers on the wire that recorded them,
  as this crate's proto reads and writes them (a field the proto does not
  know is not handed on; a
  `--tokenizer` path other than the `--model` path in place of the engine's
  own, while one equal to it reads as none stated), so
  that a gateway shapes its requests as it did in front of the engine. On
  another wire the worker's own model info carries the recorded fields that
  shape a request (the context length, the vocabulary size, the vision flag,
  the model type, the end tokens, the sampling defaults); the server info
  there stays the worker's own. While an info file is loaded the recorded
  answers stand: `supports_vision` set at the start (`--supports-vision`)
  or through the admin API is not read on any wire, and neither are the
  server arguments set there on the wire of the recording. A replay
  answers `GetTokenizer` with the recorded chunks where the info file holds
  a fetch of its wire, unless it was started with a `--tokenizer` path
  other than the `--model` path: then, and with no fetch recorded, the call
  is `unimplemented` as on any mock. Each fetch a replay answers holds a
  copy of the recorded bundle until it is sent.
- One recording of text requests serves all three wires: the fields the
  wires state differently for one client request are brought to one form
  before the fold (a temperature left unstated reads 1.0, a `top_k` that is
  off reads 0, the seed is left out). The price: requests that differ only in
  their seed, or in stating a temperature of 1.0, share a key and are served
  in turn; so do requests that differ only in the special-token settings
  (the wires state them differently for one client request) or in a field
  that stands in no key: `logprob_start_len` and `custom_params` (two wires
  carry them), the adapter `lora_id` and `require_reasoning` (one wire
  carries them), and every other field one wire alone carries. A media
  item that states neither a hash nor a reference is one constant in the
  key on the vLLM and TokenSpeed wires. A media item is keyed as its wire
  names it (a reference, a hash),
  so a request with media replays on the wire that recorded it; another
  wire finds its line only where it names every item by the same text.
- The limits of a recorded line. A status that arrives as the first item
  of a stream is written like a refusal of the call and replays as one: a
  line cannot tell the two apart. A completion whose own finish reason is
  `error`, in a call that also ended with a status, replays without that
  completion: its line reads as a choice the engine left open. A line whose
  finish reason is `abort` (the caller of the recording let go ahead of the
  completion) replays to a caller that stays as the frames the engine had
  sent and a completion with that reason, which the engine never sent. A
  line keeps the counts of its completion, not the ones each frame of a
  stream stated: a replayed frame states as its completion count the tokens
  sent so far, whatever the engine's frame stated.
  `logit_bias`
  is keyed by token id on the vLLM wire and by text on the other two; the
  fold reads an id as its plain decimal, so a key a client wrote otherwise
  (`007`) has another key on the vLLM wire than on the other two. The top
  alternatives of a position are not kept: a request that asks for them
  reads at a replay the ones the worker makes for any scripted answer. The
  scores of a request that also generates (a row for each output token) are
  not kept either: such a request replays without them.
- A replay states the tokenizer a gateway reads from its worker:
  `--tokenizer` names the model's tokenizer files. The info file keeps the
  tokenizer path the engine stated, a folder of the engine's host where the
  engine reads local files, and on the wire of the recording a replay
  started without `--tokenizer`, or with one equal to `--model`, hands that
  path to the gateway: a recording that leaves its host is replayed with a
  `--tokenizer` other than `--model`.

```bash
# record: gateway -> this worker -> the engine
cargo run --release -p mock-worker -- --grpc-base-port 19000 --grpc-count 1 \
  --grpc-wire vllm --record-upstream grpc://engine-host:50051 --record session.jsonl
# replay: gateway -> this worker, no engine
cargo run --release -p mock-worker -- --grpc-base-port 19000 --grpc-count 1 \
  --grpc-wire vllm --model m --tokenizer ./tokenizer-dir --script session.jsonl
```

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
whether the engine served cached tokens. With `--admin`, both comparisons read
the engine's cached tokens less those it restored from the mock's shared KV
tier (the gateway's index sees the workers' own caches), while the table's
`engine cached_tokens` column shows them all. With `--admin` the summary also
carries `engine_truth_per_worker`, each engine's own account of the prompt,
cached and oracle tokens it served, and `fleet.csv` samples every worker's
load, cached blocks, preemptions and KV batches once a second for the whole
run.

The mock fleet it is meant for is `mock-worker --engine realistic --admin-port`
(the [Realistic engine](#realistic-engine) above; note the caveat that its timing
polynomials were validated against hardware only with prefix caching off).

`scripts/multiturn_trace.py` writes a synthetic trace in this format, shaped like
multi-turn chat or agent sessions: Poisson session arrivals, tenant system
prompts drawn by Zipf, and every turn of a session re-sending the conversation so
far with the previous answer and a new message appended (`--help` lists the
knobs). Its blocks are 64 tokens, not 512: replay it with `--words-per-block 64`
(about one token a word), and to calibrate against a real tokenizer as above,
count `blocks = trace_input_length / 64` and set `--words-per-block` to
`64 * 64 / b`.

```bash
python3 scripts/multiturn_trace.py --out trace.jsonl --duration-s 480 --sessions-per-s 10
replay --trace trace.jsonl --words-per-block 64 --max-output 200 --speedup 1 --limit 0 ...
```

### Long runs

For a run of hours, keep one gateway and one fleet up and start the replayer
window by window (`--skip` advancing by `--limit`, a `--label` per window,
`--admin` for the oracle join), driving the admin fault hooks on a cycle over
the workers in turn (lost batches, a publishing delay, a publisher restart, a
pause and resume, a cache reset) and sampling the gateway's `/metrics` once a
minute. The memory reading that matters is the gateway allocator's live bytes
at idle, hour over hour; its RSS shows how much the allocator keeps, not what
is live.
