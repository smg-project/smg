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

## Run (canned)

```bash
cargo run --release -p mock-worker -- \
  --http-base-port 9000 --http-count 2000 \
  --grpc-base-port 19000 --grpc-count 0 \
  --model mock-model --gen-ms 5
```

Each worker is one port. Register them against an IGW gateway with
`POST /workers` (`{"url":"http://127.0.0.1:9000"}`, or `grpc://…` with
`connection_mode`/`runtime`/`models` for gRPC).

## Realistic engine

`--engine realistic` backs each worker with a continuous-batching simulator
([`src/engine.rs`](src/engine.rs)) that reproduces the behaviors driving routing:

- **prefill latency scales with input length** — TTFT grows with the *uncached*
  prompt size, chunked across scheduler steps;
- **inter-token latency grows with batch size** — `ITL = base + slope · batch`,
  so a busy replica is slower per token;
- **finite KV capacity + queueing** — when KV is full, requests wait, producing
  the `num_waiting_uncached_tokens` signal `least_load` consumes;
- **prefix caching** — a request sharing a prefix with cached blocks pays less
  prefill, reports `cached_tokens`, and the worker emits the KV-cache events
  (`SubscribeKvEvents`) that drive event-driven `cache_aware` routing.

The cost model is parametric (defaults approximate one mid-size replica):

| Flag | Default | Meaning |
|------|---------|---------|
| `--prefill-tps` | 8000 | prefill throughput (tokens/s) |
| `--decode-base-ms` | 6.0 | fixed decode-step latency (ms) |
| `--decode-per-req-ms` | 0.35 | added decode latency per running request |
| `--prefill-chunk` | 2048 | max prompt tokens prefilled per step |
| `--max-running` | 256 | continuous-batching width |
| `--kv-tokens` | 524288 | KV cache capacity (tokens) |
| `--block-size` | 16 | cache block/page size (tokens) |
| `--prefix-cache` | true | enable prefix caching + KV events |

```bash
cargo run --release -p mock-worker -- \
  --engine realistic --grpc-base-port 19000 --grpc-count 8 --model mock-model
```

**Tokenizer note (gRPC):** the gateway tokenizes prompts before routing, so it
needs a real tokenizer for the model. Register each worker with a tokenizer
label, e.g. `"labels":{"tokenizer_path":"gpt2"}`, and a `"kv_block_size":16`, and
do **not** pass `--disable-tokenizer-autoload`. (HTTP workers need no tokenizer
but cannot drive event-driven `cache_aware`, which requires token ids.)

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
