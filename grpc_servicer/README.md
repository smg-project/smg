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
`MmSettings` to `VllmEngineServicer`), or from this package's general plugin,
which defines them on a gRPC launcher's parser that lacks them: `vllm serve
--grpc` and the stock `python -m vllm.entrypoints.grpc_server` (there the
values also reach a servicer built without the namespace, through the
environment). A flag left out falls back to the matching `SMG_VLLM_MM_*`
variable, which logs a deprecation line on a launcher that has the flag; a
launcher without the flags (no plugins loaded, a servicer built on its own)
takes the variable as its documented setting and logs nothing. The startup
line `VllmEngineServicer initialized (mm_processor=inprocess, source=flag)` and
the `mm_processor_source` label name where the value came from.

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
not a second server, and it needs no vLLM change: vLLM's gRPC launcher imports
this package's servicer classes before it defines `serve_grpc`, and that
import installs the switch over `serve_grpc` (`launcher_switch.py`), so the
launcher asks the package which implementation to run before it builds an
AsyncLLM and hands the process to `smg_grpc_servicer.vllm.serve_rust` when
the answer is `rust`. That function launches the engine headless (`vllm serve
--headless`), which dials a same-host ZMQ handshake, and serves the gRPC
contract from `smg.servicer.VllmGrpcServer` on a Rust-owned thread. The
Router cannot tell the two apart.

```bash
# Python (default): upstream's gRPC server, AsyncLLM in-process.
vllm serve Qwen/Qwen3-0.6B --grpc --port 50051

# Rust request path, same entrypoint. The flag is this package's: vLLM loads
# it as a general plugin while it builds the parser (smg_grpc_servicer/vllm/plugin.py).
vllm serve Qwen/Qwen3-0.6B --grpc --port 50051 --servicer-impl rust

# The environment form works for the import-based entrypoints too (`vllm serve --grpc`,
# the deprecated `python -m vllm.entrypoints.grpc_server`); a launcher file executed
# directly as __main__ is not switched and refuses the flag instead.
SMG_VLLM_SERVICER_IMPL=rust python -m vllm.entrypoints.grpc_server --model Qwen/Qwen3-0.6B --port 50051
```

What the switch runs ahead of `serve_grpc` is the check a launcher could also
carry in its own source:

```python
from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust

if resolve_servicer_impl(args) == "rust":
    raise SystemExit(await serve_rust(args))
```

`smg serve --backend vllm --connection-mode grpc --servicer-impl rust` sets
the flag in each worker's environment, after checking that the installed
vLLM's launcher will consult it (`smg_grpc_servicer.vllm.rust.upstream_hook_installed`:
it imports this package at module level, or carries the check itself); the
Python servicer refuses to start when the flag asks for Rust, so a launcher
that reaches it anyway fails loudly instead of silently running Python. The
headless engine is launched from the parsed namespace through vLLM's own
`run_headless`, so both entrypoints above work unchanged.

Rust mode needs the `smg` wheel (for the binding) and serves the whole
contract the Python servicer serves: text generation, PD disaggregation
(`--kv-transfer-config`: connector params pass through both ways and
`GetServerInfo` carries the pairing identity), Router-preprocessed media
(inline and `/dev/shm` tensors), worker-side media processing (`media_refs`,
below), `Embed`, `FlushCache`, `GetTokenizer` (which answers
FAILED_PRECONDITION when the launcher could not resolve a local tokenizer
directory) and `SubscribeKvEvents` (vLLM's ZMQ publisher, which Rust mode
turns on by itself when the launcher is given no `--kv-events-config`; an
explicit one is kept as given, and `SMG_VLLM_SERVICER_KV_EVENTS=0` leaves the
publisher off). Tuning: `SMG_VLLM_SERVICER_HANDSHAKE_PORT` (default: a free port),
`SMG_VLLM_SERVICER_DRAIN_SECS` (default 5),
`SMG_VLLM_SERVICER_STARTUP_TIMEOUT_SECS` (default 1800: how long the servicer
waits for a sign of life from the engine during its handshake; the launcher
reports the engine process alive each time it polls it, and the handshake's
own messages count too, so a healthy start that takes longer, a large
checkpoint streaming in for an hour, still completes, while an engine that
exits fails fast regardless), `SMG_VLLM_SERVICER_STARTUP_CEILING_SECS`
(default 14400: the most a start may take however alive the engine is; 0
lifts the ceiling), `SMG_ZMQ_SOCKET_DIR`, `SMG_SERVICER_WORKER_THREADS`
(default 4).

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
  (`crates/multimodal`), takes `http`, `https` and `data` references, reads
  the model's `config.json` and preprocessor configs from the tokenizer
  directory the launcher resolved, and refuses a request above the engine's
  own `--limit-mm-per-prompt` as the engine's server does (`--mm-max-items`
  and `SMG_*_MAX_COUNT` tighten those limits, never loosen them). An engine
  that normalizes pixels on device
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
implementation, as do the RL control-plane extras (the `rl.*` advertisement,
the live `weight_version` on generate responses, `is_paused`); the Rust server
redacts credentials from `server_args` the same way. `SubscribeKvEvents` relays
TokenSpeed's ZMQ KV-event publisher, which Rust mode turns on by itself when
the launcher is given no `--kv-events-config` (an explicit one is kept as
given; `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` leaves the publisher off). Tuning:
`SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT` (default: a
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
port ranges if multiple DP ranks publish events. The Rust servicer's relay
uses the same replay endpoint (the vLLM, SGLang and TokenSpeed launchers all
pass `replay_endpoint` from the engine's kv-events config) for gaps in flight
and for the batches published before its subscription joined the publisher;
without one those are counted as lost or unknown, never silently skipped.

Without replay, or when history is expired, empty, malformed, or unavailable
(timeout: five seconds), the bridge reports `OUT_OF_RANGE` before streaming or
`DATA_LOSS` after streaming starts. SMG discards that worker's stale mappings and
resubscribes with zero. A zero cursor rebuilds knowledge from subsequent live
events; it is not a complete cache snapshot: this bridge relays per call and
keeps no history or block record between calls, so it has nothing older to
serve. The Rust servicer's relay (`crates/engine_servicer`, the vLLM servicer
and `SMG_SGLANG_SERVICER_IMPL=rust`) does: it keeps a bounded history and the
engine's live blocks for the servicer's lifetime, serves the history to a
subscription from zero while it is complete, and a state snapshot
(`KvSnapshotChunk`) before live events once the window has rolled. An empty
replay is conservatively reset because it cannot distinguish an idle publisher
from a restarted one.

#### Rust request path (`SMG_SGLANG_SERVICER_IMPL=rust`)

`sglang.launch_server --grpc-mode` hands the process to this package's
`serve_grpc`. With `SMG_SGLANG_SERVICER_IMPL=rust` that entry serves the
`sglang.grpc.scheduler.SglangScheduler` contract from Rust
(`smg.servicer.SglangGrpcServer`, which needs the `smg` wheel) instead: the
scheduler runs headless in a spawned child over the msgpack ZMQ wire below,
the Rust server speaks the gRPC contract on top of it, and Python keeps the
lifecycle only. The Router cannot tell the two implementations apart: text
generation with sampled and prompt logprobs, reasoning-token counts, LoRA ids
and custom logit processors forwarded as the Python servicer forwards them,
`Embed`, `FlushCache` and profiling (the scheduler's own control requests,
carried by the wire's control call) all answer as the Python servicer does.
What the wire does not carry is reported, not emulated: multimodal inputs and
hidden states are refused, PD/EPD disaggregation stays with the Python
implementation (the worker refuses to start on the Rust path in those modes),
and LoRA loading answers UNIMPLEMENTED, as it does on the Python servicer.
`SubscribeKvEvents` relays SGLang's ZMQ KV-event publisher, which Rust mode
turns on by itself when the launcher is given no `--kv-events-config` (an
explicit one is kept as given; `SMG_SGLANG_SERVICER_KV_EVENTS=0` leaves the
publisher off). SGLang's HTTP sidecar (metrics and profiling endpoints) is not
started on this path.

#### Headless over ZMQ (no SGLang change)

SMG can drive SGLang's scheduler directly over ZMQ, the same same-host lane it
has for vLLM and TokenSpeed, without a gRPC server or SGLang's tokenizer
manager in between:

```bash
python -m smg_grpc_servicer.sglang.headless \
    --zmq-handshake-address tcp://127.0.0.1:<port> --zmq-engine-index 0 \
    --model-path Qwen/Qwen3-0.6B [any SGLang server args]
```

`smg serve --backend sglang --connection-mode zmq` runs exactly this for each
worker, with the handshake port SMG derives from the worker's `ipc://` URL.
The launcher spawns the scheduler ranks the way SGLang's engine does; the
scheduler keeps its tokenizer (SMG tokenizes prompts, but SGLang's grammar
backend for constrained decoding only exists alongside a tokenizer). Inside
each rank, SGLang loads
this package's plugin (`smg_grpc_servicer.sglang.zmq_plugin`, registered under
the `sglang.srt.plugins` entry point and inert unless the launcher's
`SMG_SGLANG_ZMQ_HANDSHAKE` is set), which hooks the scheduler's own ingress
and egress seams: the rank that owns request I/O dials SMG's handshake once
its model is loaded, registers its geometry, decodes SMG's requests (the
scheduler's native `TokenizedGenerateReqInput` or `TokenizedEmbeddingReqInput`,
normalized and verified as the tokenizer manager would, and control calls
that become the scheduler's own `FlushCacheReqInput` or `ProfileReq`) and
answers with slim positional structs: a per-step token batch (token ids,
finish reason with message, matched stop and abort status, counts including
reasoning tokens, sampled-token and prompt logprobs with ranked top logprobs
when asked, and a scheduler-load tail), an embedding batch, and a control
reply under the call id. Requests the scheduler cannot serve, `n > 1` for
example (SMG fans out itself), are answered with a terminal abort rather than
dropped. SGLang itself is unchanged; the plugin pins the wire of the SGLang
version it was tested with, and a struct change upstream shows up in
`grpc_servicer/tests/test_sglang_zmq_msgpack.py`.


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
