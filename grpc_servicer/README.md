# smg-grpc-servicer

gRPC servicer implementations for LLM inference engines. Supports vLLM, MLX, TokenSpeed, and SGLang.

## Installation

For vLLM:

```bash
pip install smg-grpc-servicer[vllm]
```

For MLX:

```bash
pip install smg-grpc-servicer[mlx]
```

For TokenSpeed, install the TokenSpeed runtime first, then install the servicer bridge:

```bash
pip install smg-grpc-servicer
```

For SGLang:

```bash
pip install smg-grpc-servicer[sglang]
```

## Usage

### vLLM

```bash
vllm serve meta-llama/Llama-2-7b-hf --grpc
```

#### Worker-side multimodal processing (media refs)

By default the smg router fetches and preprocesses images itself and sends
pixel tensors. A vLLM gRPC worker can instead accept media references (URLs)
and run vLLM's own multimodal processor (or, on the Rust request path, smg's
own pipeline with `--mm-processor smg`; see below):

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --grpc --mm-processor inprocess \
    --allowed-media-domains example.com
```

The `--mm-*` flags come from a vLLM launcher that knows them (it hands an
`MmSettings` to `VllmEngineServicer`); an older launcher, or a flag left out,
falls back to the matching `SMG_VLLM_MM_*` variable, which logs a deprecation
line and goes away in the next minor release. The startup line
`VllmEngineServicer initialized (mm_processor=inprocess, source=flag)` and the
`mm_processor_source` label name where the value came from.

The worker then advertises `mm_processor=inprocess` and `mm_media_ref_schemes`
through `GetServerInfo`; a router with media-reference support forwards
`media_refs` only to workers that advertise, and a router without it ignores the
labels and keeps sending preprocessed tensors. vLLM's `--allowed-media-domains`,
`--allowed-local-media-path`, `--media-io-kwargs`, `--limit-mm-per-prompt` and
`VLLM_*_FETCH_TIMEOUT` govern fetching on the worker; without
`--allowed-media-domains` the worker fetches from any host the router forwards.
Related knobs: `--mm-max-inflight` (`SMG_VLLM_MM_MAX_INFLIGHT`, default 64)
bounds concurrent media jobs; `--mm-max-items` (`SMG_VLLM_MM_MAX_ITEMS`, unset
by default) overrides the model's per-modality reference limits;
`--mm-max-item-bytes` (`SMG_VLLM_MM_MAX_ITEM_BYTES`, default 32 MiB) caps inline
`data:` payloads; `SMG_VLLM_MM_MAX_VIDEO_FRAMES` (env only for now; default 0,
meaning vLLM's own `--media-io-kwargs` decide) caps the frames a video is
sampled to, so long clips stay bounded. All eight flag-backed settings are
validated when the servicer starts, whatever the processor mode, so a stale
unreadable value fails loudly.

On the router side, `--mm-processing` selects `auto` (default: forward when
the model's spec opts in and every registered worker of the model advertises
`mm_processor`), `router` (always preprocess) or `worker` (strict: 400 when a
request cannot be forwarded); the outcome is counted in
`smg_mm_processing_total{model,mode,reason}`, and the startup line
`multimodal processing mode` names the value and its source (`flag`, `env` or
`default`). `SMG_MM_PROCESSING` is the deprecated env fallback: it applies only
when the flag is absent, logs a deprecation line, and goes away in the next
minor release. Any other value stops the router at startup instead of quietly
reverting to `auto`. On the worker path the
router never expands placeholders, so routing decisions that weigh the prompt's token
count (cache-aware policies, load estimates) see one token per media item where
the worker will schedule the full placeholder run. The `E2E_MM_PROCESSING=worker`
e2e lanes run the multimodal suites in this mode, and
`crates/multimodal/scripts/check_worker_anchor_parity.py` checks that a spec's
anchor is the token vLLM expands.

To move fetching and processing out of the vLLM process, run the GPU-free
sidecar next to a private Redis and point the worker at it
(`pip install smg-grpc-servicer[vllm,vllm-redis]`):

```bash
python -m smg_grpc_servicer.vllm.mm_sidecar --model Qwen/Qwen3-VL-8B-Instruct \
    --redis-url redis://127.0.0.1:6379/0 --allowed-media-domains example.com
vllm serve Qwen/Qwen3-VL-8B-Instruct --grpc --mm-processor redis \
    --mm-redis-url redis://127.0.0.1:6379/0
```

The sidecar and the worker must agree on model, vLLM version, dtype, video
backend, media/processor kwargs and `--limit-mm-per-prompt` (pass the flag to
both processes; the sidecar's limit is the one that applies, the limit is
resolved per modality before hashing so equivalent spellings match, and the key
namespace is derived from all of these): the worker advertises
`mm_processor=redis` only while a sidecar with a matching fingerprint keeps its
`hello` key alive, and rejects results that disagree. Jobs and results travel over Redis lists under
`smg:mm:v1:{namespace}`; results carry full tensors keyed by a per-attempt job
id and expire after 120 s. Knobs: `--mm-sidecar-timeout-ms`
(`SMG_VLLM_MM_SIDECAR_TIMEOUT_MS`, 30000), `--mm-sidecar-max-queue`
(`SMG_VLLM_MM_SIDECAR_MAX_QUEUE`, 256, fail fast when the queue is deeper),
`--mm-sidecar-namespace` (`SMG_VLLM_MM_SIDECAR_NAMESPACE`, override the derived
namespace). The sidecar resolves `--redis-url` and `--namespace` the same way,
so the two processes cannot disagree on the namespace; the timeout travels
with each job as its deadline.
On the sidecar, `SMG_VLLM_MM_MAX_RESULT_BYTES` (default 512 MiB, lowered to
Redis's `proto-max-bulk-len` when that is smaller) caps an encoded result and
`SMG_VLLM_MM_MAX_VIDEO_FRAMES` caps video sampling as above. A result over the
cap is answered as a 400 `media_too_large` instead of being pushed, and a result
Redis refuses is reported to the worker at once; a sidecar timeout is not
retried by the router, since the worker already spent the whole budget on it.

#### Rust request path (`SMG_VLLM_SERVICER_IMPL=rust`)

The same `vllm.grpc.engine.VllmEngine` contract can be served from Rust, with
Python keeping only the lifecycle. The switch is a flag inside this package,
not a second server: upstream vLLM's gRPC entrypoint asks the package which
implementation to run before it builds an AsyncLLM, and hands the process to
`smg_grpc_servicer.vllm.serve_rust` when the answer is `rust`. That function
launches the engine headless (`vllm serve --headless`), which dials a
same-host ZMQ handshake, and serves the gRPC contract from
`smg.servicer.VllmGrpcServer` on a Rust-owned thread. The Router cannot tell
the two apart.

```bash
# Python (default): upstream's gRPC server, AsyncLLM in-process.
python -m vllm.entrypoints.grpc_server --model Qwen/Qwen3-0.6B --port 50051

# Rust request path, same entrypoint.
SMG_VLLM_SERVICER_IMPL=rust python -m vllm.entrypoints.grpc_server --model Qwen/Qwen3-0.6B --port 50051
```

The upstream hook is the first thing in its `serve_grpc`:

```python
from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust

if resolve_servicer_impl(args) == "rust":
    raise SystemExit(await serve_rust(args))
```

`smg serve --backend vllm --connection-mode grpc --servicer-impl rust` sets
the flag in each worker's environment, after checking that the installed
vLLM carries the hook (`smg_grpc_servicer.vllm.rust.upstream_hook_installed`);
the Python servicer itself refuses to start when the flag asks for Rust, so a
vLLM without the hook fails loudly instead of silently running Python. The
headless engine is launched from the parsed namespace through vLLM's own
`run_headless`, so both entrypoints above work unchanged.

Rust mode needs the `smg` wheel (for the binding) and serves the whole
contract the Python servicer serves: text generation, PD disaggregation
(`--kv-transfer-config`: connector params pass through both ways and
`GetServerInfo` carries the pairing identity), Router-preprocessed media
(inline and `/dev/shm` tensors), worker-side media processing (`media_refs`,
below), `Embed`, `FlushCache`, `GetTokenizer` (which answers
FAILED_PRECONDITION when the launcher could not resolve a local tokenizer
directory) and `SubscribeKvEvents` (`--kv-events-config` with the ZMQ
publisher). Tuning: `SMG_VLLM_SERVICER_HANDSHAKE_PORT` (default: a free port),
`SMG_VLLM_SERVICER_DRAIN_SECS` (default 5),
`SMG_VLLM_SERVICER_STARTUP_TIMEOUT_SECS` (default 1800: how long the servicer
waits for the engine's handshake; an engine's first start on a host
JIT-compiles and autotunes kernels, and a dead engine fails fast regardless),
`SMG_ZMQ_SOCKET_DIR`, `SMG_SERVICER_WORKER_THREADS` (default 4).

Worker-side media processing uses the same `--mm-processor` /
`SMG_VLLM_MM_PROCESSOR` setting as the Python servicer, with one more choice:

- `inprocess` (vLLM's MediaConnector and the engine's renderer) and `redis`
  (the sidecar) are the Python servicer's backends, so the media is fetched and
  processed by vLLM's own code on either servicer. The Rust server hands a
  request's `media_refs` to the Python bridge
  (`smg_grpc_servicer.vllm.rust_media`), which runs the processor and vLLM's
  input processor on the launcher's asyncio loop and returns the expanded
  prompt and `mm_features` as vLLM's own encoder writes them; Rust relays the
  encoded features to the engine untouched and sends the tensor frames
  straight from the memory Python lent it, without a copy.
- `smg` is smg's own media pipeline, the one the Router runs for
  `--mm-processing router`, run inside the Rust servicer with no Python on the
  request path: fetch, decode, preprocess, placeholder expansion, and the
  batches a Router-preprocessed request would carry, translated for the engine
  the same way. It serves the model families that pipeline supports
  (`crates/multimodal`), takes `http`, `https` and `data` references, and
  reads the model's `config.json` and preprocessor configs from the tokenizer
  directory the launcher resolved. An engine that normalizes pixels on device
  (vLLM's `mm_device_do_normalize`, on by default for the Qwen-VL family)
  takes raw `uint8` pixels, and the pipeline writes those for it; a model
  whose processor cannot emit raw pixels is refused at startup under that
  setting (start the engine with `--mm-device-do-normalize=false` or use
  `inprocess`). The Python servicer refuses `smg`: it has no such processor.

The in-flight cap, the saturation refusal, the advertised `mm_processor` /
`mm_media_ref_schemes` / `mm_processor_source` and the PD prefill leg's
`media_identity` behave as on the Python servicer whichever processor runs;
the engine-side contract does not change.

Known difference: under `--structured-outputs-config.backend auto` (the
default) vLLM's frontend validates each constraint with xgrammar and falls
back to guidance when xgrammar rejects it. Rust mode applies the same static
rules (JSON-schema features xgrammar lacks go to guidance, a `choice` becomes
the grammar xgrammar compiles), but it cannot run xgrammar's parser, so a regex
or grammar that only guidance accepts fails that request at the engine's
grammar compile instead of falling back. Pin `guidance` (or `xgrammar`)
explicitly when that matters; the engine keeps the backend of its first
structured request either way, as it does behind vLLM's own frontend.

### MLX

```bash
python -m smg_grpc_servicer.mlx --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051
```

### TokenSpeed

```bash
python -m smg_grpc_servicer.tokenspeed --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051
```

This is the process `ts serve` spawns for its gRPC worker. With
`SMG_TOKENSPEED_SERVICER_IMPL=rust` the same process serves the
`tokenspeed.grpc.scheduler.TokenSpeedScheduler` contract from Rust
(`smg.servicer.TokenSpeedGrpcServer`, which needs the `smg` wheel): the
launcher computes the model and server facts from TokenSpeed's own config,
runs the scheduler(s) headless in a spawned child over the msgpack ZMQ wire,
and supervises both. What that wire does not carry is reported, not emulated:
`FlushCache` and profiling answer UNIMPLEMENTED, ranked `top_logprobs` and
prompt logprobs are refused, and PD/EPD disaggregation stays with the Python
implementation. Tuning: `SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT` (default: a
free port), `SMG_TOKENSPEED_SERVICER_DRAIN_SECS` (default 5),
`SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS` (default 1800, as for vLLM
above), `SMG_ZMQ_SOCKET_DIR`, `SMG_SERVICER_WORKER_THREADS` (default 4).

### SGLang

```bash
sglang serve --model-path meta-llama/Llama-2-7b-hf --grpc-mode
```

#### KV-event recovery

To retain cache knowledge across a recoverable event gap, configure SGLang's
`--kv-events-config` with both a PUB endpoint and a replay endpoint, for example:

```json
{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","buffer_steps":10000}
```

The bridge subscribes to live events before requesting missed batches from the
replay endpoint, preserves publisher sequence numbers, and removes overlap at
handoff. Both subscriptions currently use DP rank 0; allocate non-overlapping
port ranges if multiple DP ranks publish events.

Without replay, or when history is expired, empty, malformed, or unavailable
(timeout: five seconds), the bridge reports `OUT_OF_RANGE` before streaming or
`DATA_LOSS` after streaming starts. SMG discards that worker's stale mappings and
resubscribes with zero. A zero cursor rebuilds knowledge from subsequent live
events; it is not a complete cache snapshot. An empty replay is conservatively
reset because it cannot distinguish an idle publisher from a restarted one.

## Architecture

```
smg-grpc-servicer[vllm]    ──optional dep──>  vllm       (lazy import)
smg-grpc-servicer[mlx]     ──optional dep──>  mlx-lm     (lazy import)
smg-grpc-servicer          ──external runtime──>  tokenspeed (lazy import)
smg-grpc-servicer[sglang]  ──optional dep──>  sglang     (lazy import)
smg-grpc-servicer          ──depends on────>  smg-grpc-proto  (hard dependency)
vllm                       ──optional──────>  smg-grpc-servicer (via vllm serve --grpc)
sglang                     ──optional──────>  smg-grpc-servicer (via --grpc-mode)
```

Backend dependencies are isolated via extras or runtime installs to avoid conflicts between vLLM, MLX, TokenSpeed, and SGLang.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local development setup, CI, and release workflows.
