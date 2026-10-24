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

Without `--kv-events-config`, the servicer turns vLLM's ZMQ KV-event publisher
on by itself, under either implementation, before the engine is built, at a
port found free on the host rather than vLLM's fixed default (one per
data-parallel rank), so `SubscribeKvEvents` relays the events a cache-aware
router routes by and workers that share a host do not collide; an explicit
configuration is kept as given, its endpoint and off included, and
`SMG_VLLM_SERVICER_KV_EVENTS=0` leaves the publisher off.

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
variable. Legacy variables log a deprecation line on a launcher that has the
flag; the TurboJPEG policy and download timeout variables remain supported
fallbacks and never log a deprecation warning. A launcher without the flags
(no plugins loaded, a servicer built on its own) takes the variable as its
documented setting and logs nothing. The startup
line `VllmEngineServicer initialized (mm_processor=inprocess, source=flag)` and
the `mm_processor_source` label name where the value came from.

A model family whose reference image processor raises a short side below a
floor before its own resize (MiniMax-M3: 112 px) gets the same raise on the
worker: `inprocess` and the sidecar raise such an image to the floor, bicubic,
before the engine's processor sizes it, as the router's own pipeline does, so
a tiny image costs the same visual tokens whichever component preprocessed it
(video frames are not raised, as there). An engine started with
`--mm-processor-kwargs '{"do_resize": false}'` raises nothing, as the pipeline
raises nothing then.

The worker then advertises `mm_processor=inprocess` and `mm_media_ref_schemes`
through `GetServerInfo`; a router with media-reference support forwards
`media_refs` only to workers that advertise, and a router without it ignores the
labels and keeps sending preprocessed tensors. Whatever the processor mode, a
multimodal engine also advertises its resolved `--limit-mm-per-prompt` as
`mm_item_limits` (`image=8,video=2`), so a router that preprocesses media
itself holds requests to the engine's limits, and its encoder budget as
`mm_encoder_budget` (the `budget of N tokens` of the engine's encoder-cache
start-up line, derived at launch with vLLM's own helper): the most feature
tokens one media item may carry and still be scheduled, so the router refuses
a larger item (a long video sampled at a high rate) with a 400 that names the
item's size and the budget, instead of sending the engine an item it admits
and never schedules. vLLM's `--allowed-media-domains`,
`--allowed-local-media-path`, `--media-io-kwargs`, `--limit-mm-per-prompt` and
`VLLM_*_FETCH_TIMEOUT` govern fetching on the worker; without
`--allowed-media-domains` the worker fetches from any public host the router
forwards. A media URL that names or resolves to a loopback, link-local or
private address is refused on the worker as on the router, redirects included,
unless `SMG_MM_ALLOW_PRIVATE_MEDIA_HOSTS=1` is set for a deployment whose media
origin is internal.
Related knobs: `--mm-max-inflight` (`SMG_VLLM_MM_MAX_INFLIGHT`, default 64)
bounds concurrent media jobs; `--mm-max-items` (`SMG_VLLM_MM_MAX_ITEMS`, unset
by default) overrides the model's per-modality reference limits;
`--mm-max-item-bytes` (`SMG_VLLM_MM_MAX_ITEM_BYTES`, default 32 MiB) caps inline
`data:` payloads; `SMG_VLLM_MM_MAX_VIDEO_FRAMES` (env only for now; default 0,
meaning vLLM's own `--media-io-kwargs` decide) caps the frames a video is
sampled to, so long clips stay bounded. All nine flag-backed settings are
validated when the servicer starts, whatever the processor mode, so a stale
unreadable value fails loudly. Two layers bound an inline item: the router
refuses a `data:` item above its own per-kind input bound
(`SMG_IMAGE_MAX_INPUT_BYTES`, `SMG_VIDEO_MAX_INPUT_BYTES`,
`SMG_AUDIO_MAX_INPUT_BYTES`; default 256 MiB each) with a 400
`media_ref_too_large` before the worker sees it, and the worker refuses one
above `--mm-max-item-bytes` with a message that names the worker as the flag's
owner, which the router relays as a 400 `start_generation_failed`. With the
defaults a 50 MB item clears the router and stops at the worker's 32 MiB, so
that flag is the one to raise; the router's `--max-payload-size` (default
512 MiB) must also hold the whole request body.

Media URL download budgets can be set with `--mm-fetch-timeout-ms` (images
and audio, default **10000**) and `--mm-video-fetch-timeout-ms` (video,
default **30000**). Both take positive integer milliseconds and cover the
fetch from connection through the last byte, not decoding or inference.
Set them on the process that downloads the media:

- **Router preprocessing** (shared by vLLM, SGLang and TokenSpeed): the Rust
  and Python router CLIs accept both flags, with environment fallbacks
  `SMG_MM_FETCH_TIMEOUT_MS` and `SMG_MM_VIDEO_FETCH_TIMEOUT_MS`.
- **vLLM Rust worker**, `--servicer-impl rust --mm-processor smg`: the same
  flags use `SMG_VLLM_MM_FETCH_TIMEOUT_MS` and
  `SMG_VLLM_MM_VIDEO_FETCH_TIMEOUT_MS` as environment fallbacks.

An explicit flag takes precedence over its environment variable. Unset
values retain the defaults above. These settings do not change downloads
performed by vLLM's own `inprocess` or `redis` processors; configure those
through the engine's media loader settings. SGLang and TokenSpeed currently
use the Router's media pipeline, so set the budgets on the Router for them.

A fetch that fails is answered under `media_fetch_failed` by whose fault it
is, on the Router's pipeline and the vLLM Rust worker's alike: a reference the
request got wrong (a URL that cannot be parsed or resolved, a host that refuses
the connection or answers 4xx, a payload that is not the media it claims to
be) is a 400; a host that does not deliver within the budget, answers 5xx or
breaks the transfer is a 502, so a client knows whether to retry. Neither
counts against the worker's circuit breaker, and the Router does not replay
either on another worker, which would meet the same host.

On the router side, `--mm-processing` selects `auto` (default: forward when
every registered worker of the model advertises `mm_processor` and the model's
spec opts in, or every one of those workers runs the router's own pipeline,
`mm_processor=smg`, which expands any spec's image or video anchor itself), `router` (always
preprocess) or `worker` (strict: 400 when a request cannot be forwarded, and
`503 no_media_ref_capable_worker` for a model none of whose registered workers
advertises `mm_processor`, answered before any worker is chosen, except a model
with an encode worker among them: on the encode-prefill-decode pipeline media
references never travel, a request that would forward them is answered its own
400 `multimodal_not_supported`, and such a model is not named under
`models_without_media_processing` either; a family
without the opt-in forwards when every media-processing worker of the model
runs `smg`, and its requests are placed on `smg` workers only, so a worker on
the engine's own processor never receives an anchor it cannot expand; while a
model whose workers report a vision encoder has no worker advertising
`mm_processor`, the ready body of `/readiness` names the model under
`models_without_media_processing`, the pod staying ready for its text requests
and its other models, and each vision worker without a media processor is
warned about once at registration); the outcome is counted in
`smg_mm_processing_total{model,mode,reason}` (`pipeline_workers` and
`auto_pipeline` name the forwarded-by-the-pipeline case), and the startup line
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

Both pipelines (the router's own and `--mm-processor smg` on the worker)
decode JPEGs through libjpeg-turbo's TurboJPEG library when `libturbojpeg.so.0`
loads, so their pixels are the ones an engine that decodes with Pillow sees.
Without the library they fall back to a pure-Rust decoder whose pixels differ
by a few levels on up to a fifth of a JPEG's bytes (no JPEG of the parity
corpus decodes identically), and the only marks of that state are the start-up
line `libturbojpeg not found: JPEGs decode with the pure-Rust decoder ...` and
the gauge `smg_mm_turbojpeg_available` at 0. The TurboJPEG policy makes the
state fatal instead: `--mm-turbojpeg require` on the router (`SMG_MM_TURBOJPEG`
for a process without the flag, `mm_turbojpeg` in a config file) stops the
router at start-up, and `--mm-turbojpeg require` on the worker
(`SMG_VLLM_MM_TURBOJPEG`, a supported spelling beside the flag, not a
deprecated one) refuses to build smg's pipeline at launch, each with an error
naming the library and the gauge. The router checks the policy once at
start-up, in every routing mode, before anything binds; a worker whose
launcher was not given the setting reads `SMG_MM_TURBOJPEG` as the router
does. The default, `prefer`, keeps the fallback; an unreadable value stops
start-up as an unreadable `SMG_MM_PROCESSING` does.

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
# it as a general plugin while it builds the parser (smg_grpc_servicer/vllm/plugin.py);
# the stock launcher's own parser (`python -m` below) takes it too.
vllm serve Qwen/Qwen3-0.6B --grpc --port 50051 --servicer-impl rust

# The environment form, as a deployment sets it, works for every entry: `vllm serve --grpc`,
# and `python -m` on the launcher module (`vllm.entrypoints.grpc_server` is the launcher itself
# on vLLM 0.28 and 0.29, and from 0.30 a shim importing `vllm.entrypoints.launchers.grpc_server`).
SMG_VLLM_SERVICER_IMPL=rust python -m vllm.entrypoints.grpc_server --model Qwen/Qwen3-0.6B --port 50051
```

What the switch runs ahead of `serve_grpc` is the check a launcher could also
carry in its own source:

```python
from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust

if resolve_servicer_impl(args) == "rust":
    raise SystemExit(await serve_rust(args))
```

##### Metrics (`--metrics-port`)

The Rust servicer serves Prometheus metrics on a listener of its own when
asked: `--metrics-port <port>` (`--metrics-host` picks the interface; it
defaults to the gRPC `--host`), or the environment form
`SMG_VLLM_SERVICER_METRICS_PORT` / `SMG_VLLM_SERVICER_METRICS_HOST` that the
flags override. Without either, no endpoint. The families are the same on every
engine (`smg_servicer_*`, labels `model`, `rank`, `wire`): requests by finish
reason, refusals, errors, in-flight requests, prompt / cached / generated
tokens, and the time-to-first-token, inter-token, per-request-mean and
end-to-end latency histograms (vLLM's own bucket edges), plus the servicer's
lifecycle state and start time. The listener binds beside the gRPC one, before
the engine connects (a scrape during the engine's start answers
`smg_servicer_state{state="starting"} 1`; a readiness hold (admission open to
the warm-up's own requests only) reads `warming_up`, a drain `draining`), runs
on its own thread, and closes
with the servicer; a port equal to `--port` is refused before the engine
starts, and the flag needs `--servicer-impl rust` (the Python servicer exposes
no endpoint). A planned warm-up's own requests go through the request path
and are counted like any other, so the first scrapes include them.

```bash
vllm serve Qwen/Qwen3-0.6B --grpc --port 50051 --servicer-impl rust --metrics-port 29100
curl -s http://127.0.0.1:29100/metrics | grep smg_servicer_requests_total
```

##### Supported vLLM releases

The switch binds `serve_grpc` in the launcher module of the installed release
(`launcher_switch.LAUNCHERS`, newest first):

| vLLM | launcher module | what `python -m vllm.entrypoints.grpc_server` runs |
|---|---|---|
| 0.30 and later | `vllm.entrypoints.launchers.grpc_server` | a shim importing the launcher, which is bound as it loads |
| 0.28 and 0.29 | `vllm.entrypoints.grpc_server` | the launcher itself, as `__main__` |

Verified on 0.28 through 0.31 (`launcher_switch.SUPPORTED_VLLM_RELEASES`),
for every entry. `vllm serve <model> --grpc` imports the launcher under its
name after the parse, and the import-time hook (or the finder) binds it.
`python -m` on the launcher module itself runs it as `__main__`, where no
import hook reaches it, so it is bound in its own namespace from the two
moments this package runs inside the launcher's `main()` once `serve_grpc`
exists: vLLM loads the plugin while the launcher builds its parser
(`AsyncEngineArgs.add_cli_args`), and the plugin's parse step runs on the
launcher's own parse. A main-branch build between two releases carries the
next release's layout under the earlier version; the binding goes by what is
loaded, never by the version. A launcher run as a file path, or with the
plugin left out of `VLLM_PLUGINS`, is not reached; the Python servicer then
refuses the flag with one line.

The CPU smoke behind that table (`scripts/launcher_switch_smoke.py`) runs each
launcher through to the servicer construction against a stub engine, in a
fresh interpreter per case, and checks that Rust takes the process before any
engine exists and that the Python default still constructs the servicer:

```bash
# The vendored upstream launchers (tests/launchers/, one per layout); no vLLM needed.
pytest -q grpc_servicer/tests/test_vllm_launcher_switch_main.py
# A launcher tree extracted from a wheel (vllm/entrypoints/**/grpc_server.py), the rest stubbed.
python -I grpc_servicer/scripts/launcher_switch_smoke.py --launcher-tree <dir> --servicer-root grpc_servicer
# The installed vLLM (a CPU-only container with the engine image is enough): its real launcher,
# CLI, parser and plugin loading, with only the engine stubbed; no model is read.
python -I grpc_servicer/scripts/launcher_switch_smoke.py --installed
```

The switch verifies its hook at start-up: once the Rust implementation is
selected, the launcher's own argument parse (the plugin's parse step) checks
that the switch is bound over the launcher (the table above; a launcher run
as `__main__` is checked in its own namespace), importing the module under
its known names where the CLI has not yet, and a launcher the switch cannot
bind (one without `serve_grpc`, a release that moved the module) is refused
with one ERROR line naming the installed vLLM version and the launchers the
switch binds, exit code 2, before the engine is built, rather than by the
Python servicer's guard after the model has loaded.

`smg serve --backend vllm --connection-mode grpc --servicer-impl rust` sets
the flag in each worker's environment, after checking that the installed
vLLM's launcher will consult it (`smg_grpc_servicer.vllm.rust.upstream_hook_installed`:
a launcher at a module path the switch binds imports this package at module
level, or a launcher anywhere carries the check itself); the
Python servicer refuses to start when the flag asks for Rust, so a launcher
that reaches it anyway fails loudly instead of silently running Python. The
headless engine is launched from the parsed namespace through vLLM's own
`run_headless`, so both entrypoints above work unchanged.

Rust mode needs the `smg` wheel (for the binding) and serves the whole
contract the Python servicer serves: text generation, PD disaggregation
(`--kv-transfer-config`: connector params pass through both ways and
`GetServerInfo` carries the pairing identity), Router-preprocessed media
(inline and `/dev/shm` tensors), worker-side media processing (`media_refs`,
below), `Embed`, `FlushCache`, `StartProfile` and `StopProfile` (the engine's own
torch profiler as its launch configured it, started and stopped on every rank;
a start that sets per-request options is refused, since the engine takes its
profiler's directory and options at launch; the same on the Python servicer),
`GetTokenizer` (which answers
FAILED_PRECONDITION when the launcher could not resolve a local tokenizer
directory) and `SubscribeKvEvents` (vLLM's ZMQ publisher, which the servicer
turns on by itself, on this path as on the Python one, when the launcher is
given no `--kv-events-config`, at a port found free on the host rather than
vLLM's fixed default, so two engines on one host do not collide; an explicit
one is kept as given, its endpoint included, and
`SMG_VLLM_SERVICER_KV_EVENTS=0` leaves the publisher off). Tuning:
`SMG_VLLM_SERVICER_HANDSHAKE_PORT` (default: a free port),
`SMG_VLLM_SERVICER_DRAIN_SECS` (default 5),
`SMG_VLLM_SERVICER_STARTUP_TIMEOUT_SECS` (default 1800: how long the servicer
waits for a sign of life from the engine during its handshake; the launcher
reports the engine process alive each time it polls it, and the handshake's
own messages count too, so a healthy start that takes longer, a large
checkpoint streaming in for an hour, still completes, while an engine that
exits fails fast regardless), `SMG_VLLM_SERVICER_STARTUP_CEILING_SECS`
(default 14400: the most a start may take however alive the engine is; 0
lifts the ceiling), `SMG_VLLM_SERVICER_SILENCE_TIMEOUT_SECS` (default 300: how
long the connected engine may hold requests in flight, in total since its last
output, with nothing pending prefill, before the servicer treats it as dead,
fails them and closes the link for the launcher to restart the pair; a hard
engine death sends nothing over the wire, so silence is the only sign of it),
`SMG_VLLM_SERVICER_PREFILL_TOKENS_PER_SEC` (default 2000: the prompt tokens
pending prefill, those of the requests the engine has accepted and produced
nothing for yet, extend that bound by the time they take at this rate, 500 s
per million tokens; a wire that says nothing until a prompt's first token is
silent through a whole prefill, and a batch of long prompts prefills for
longer than any fixed bound; a low rate is a bound on how slowly a healthy
engine may prefill, not an estimate of how fast it does; 0 allows nothing),
`SMG_ZMQ_SOCKET_DIR`, `SMG_SERVICER_WORKER_THREADS`
(default 1, for the one runtime `server.rs` builds for the vLLM, SGLang and
TokenSpeed Rust servicers alike: one runtime worker carries about 100k output
tokens/s on one connection with its lock uncontended; raise it to 2-4 when one
engine serves more than about 512 concurrent streams and their inter-token p99
matters more than the servicer's CPU).

##### Prefill/decode over NIXL on one host: restarting a leg in place

A prefill and a decode leg on one host (or one NVLink domain) move KV over
UCX's CUDA IPC: the importing side opens the peer's allocations and keeps the
mappings in a process-wide cache that lives as long as the importer does, and
an imported mapping keeps the exporter's allocation alive after the exporter
died. A leg that crashes therefore cannot restart on its own GPU until its
peers let go: its startup memory check sees the dead process's memory as still
in use ("Free memory on device ... is less than desired GPU memory
utilization") and the pod crash-loops until the peers restart too. Two of the
engine's own settings bound the hold, and they work only together:

- `UCX_CUDA_IPC_CACHE=n` in the engine's environment: a mapping ends with its
  transfer instead of staying cached.
- `kv_connector_extra_config.engine_ttl` (seconds) in `--kv-transfer-config`:
  a peer idle longer than this is dropped the next time the connector runs
  its sweep, which it does when it handshakes a new peer and when it sends a
  heartbeat; an hour by the connector's default. A restarted leg counts as a
  new peer only for the peers that do not hold its memory: the leg whose
  memory a peer still holds fails its startup memory check before it
  handshakes anyone, so its own restart cannot run the sweep that would
  release it. The heartbeat goes out on a known peer's requests as well, at
  most once per sixth of the connector's lease (five seconds at its default)
  while any request is waiting for its KV, so a busy decode runs the sweep
  every few seconds; the peer a request reads from is stamped active before
  the sweep, so a pair's own traffic never drops the pair. A single pair
  whose prefill died sends no heartbeat either; its
  decode lets go of the dead leg's memory only when another peer turns up or
  a live peer's requests run the sweep. Dropping the peer releases its
  transfer state, the transfers that were in flight when it died included;
  with the cache off their mappings go with them, with the cache on the
  mappings outlive the peer either way.

Neither is free, which is why the servicer does not set them for you. With
the cache off, every request's transfer opens and closes each of the peer's
allocations it touches (about one per layer, the way the engine allocates its
KV cache). Measured on one host, two tensor-parallel-2 legs over CUDA IPC,
prompts of about 5100 tokens, 40 sequential requests per arm: the first token
came 2 % later at the median (885 to 903 ms) and 5 % later at the 90th
percentile (900 to 943 ms), with one three-second outlier in forty, in the
cache-off arm. A short TTL costs a quiet pair nothing by itself, since the
sweep runs only when a new peer turns up or a heartbeat goes out on a
request; when either happens, a healthy pair that had been idle longer than
the TTL (other than the pair whose own request ran the sweep) is dropped
with the dead ones and runs its handshake again on its next request (the
peer's metadata plus a descriptor list over every one of its KV blocks:
seconds on a large cache). Legs on different hosts are not
affected: over TCP or RDMA nothing of the peer's memory is imported, and a
restart passes the memory check at once. The repository's end-to-end harness
runs its NIXL pairs with the cache off and a TTL of 3 s because it restarts
legs on purpose; a production pair on one host wants a TTL of a few minutes,
longer than any gap a busy pair leaves, shorter than the restart back-off a
crashed leg would otherwise sit through.

##### Planned warm-up before SERVING (`SMG_VLLM_SERVICER_WARMUP`)

The engine selects and compiles kernels lazily: the first scheduler step at a
token count it has not seen (a new prefill length class, a new running-batch
size, a count off the alignment its compiler specialised on) pays a kernel
search or a compile of up to several seconds, and every request in flight
during that step waits. The engine's own start warms a fixed set of shapes
(its graph-capture sizes, one tuning pass at the admission step's full token
count), so the shapes a traffic ramp reaches first were paid for by the ramp's
requests, on every engine of a fleet at once. Both servicers now run a planned
warm-up over the engine before they report SERVING: a decode staircase (as
many concurrent requests as the running window holds, each asking for one more
token than the last, so every running-batch size runs once), a prefill ladder
(one request per 128-token length class up to `--max-num-batched-tokens`, the
graph-capture sizes, lengths one above the tuners' bucket grid), a second
pass over the bucket grid whose timings show what the first pass paid, and,
where the engine's module asks for it (TokenSpeed, below), every remaining
prompt length up to the step, last, in an order whose every prefix is a
uniform spread of the range, so a budget cut leaves an even coverage. The
longest prompt is what the engine's scheduler admits with one output token:
the context less the positions the engine keeps back (none on vLLM, where a
prompt and an output budget may fill the context; two on TokenSpeed, or its
speculative draft window when wider; six on SGLang), never the context
itself. Every request carries its own token ids, so prefix caching never
hides a prefill, and the engine's prefix cache is flushed at the end. One INFO line per shape
with its time (a chunk remainder's at DEBUG, with an INFO progress line every
500 remainders and one at their end); readiness (the gRPC health service, so the startup and
readiness probes) stays NOT_SERVING until the pass ran, and flips SERVING
whatever the pass returned (a failed request is a logged line, never a worker
kept out of rotation). `SMG_VLLM_SERVICER_WARMUP` = `full` (the default:
every 128-token class; about a minute on a 16k step), `fast` (the tuners'
bucket grid only: seconds), `off`, a budget in seconds (`120`) or
`<level>:<seconds>` (`fast:30`); the budget (default 300 s; 900 s for
TokenSpeed, whose plan holds the chunk remainders, below) bounds the pass:
it ends between shapes, the remaining ones named in the log, and each
request's timeout is clamped to what is left of it (one second at least), so
a shape that hangs cannot hold readiness past the budget. On the Rust path the
pass runs over the servicer's own gRPC port between the engine's handshake and
SERVING, once the KV-event relay's subscription has reached the engine's
publisher (the warm-up's batches are the engine's first, and a subscription
that landed after them would leave the relay's history starting late for the
engine's life; the relay dials the publisher every second from the servicer's
start, so the subscription is normally in place before the handshake, and the
wait is bounded at 30 s with a logged fall-through); the Python servicer runs
it over its in-process engine with its health service holding NOT_SERVING
meanwhile. Under data parallelism every
rank is a scheduler of its own: the plan runs the staircase once per rank and
every prefill shape on every rank at once, each request pinned to its rank
(`data_parallel_rank`), so the log's step count is per rank.

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
  directory the launcher resolved, lays the engine's `--mm-processor-kwargs`
  over those preprocessor configs as the engine's own processor would take
  them (less `device`, which only says where that processor would run; a key
  the pipeline has no field for is refused at launch), refuses a request
  above the engine's own `--limit-mm-per-prompt` as the engine's server does
  (`--mm-max-items` and `SMG_*_MAX_COUNT` tighten those limits, never loosen
  them), refuses a media item whose features exceed the engine's encoder
  budget (`mm_encoder_budget` above) before it reaches the engine, as the
  engine's own front end does, with the item's feature tokens and the budget
  in the same message, and samples a
  video to the engine's own frame budget (`--media-io-kwargs`
  `video.num_frames`, else its loader's default) where the family's processor
  takes the loader's frames, so a clip costs the same tokens as on the engine's
  own server; the budget is read under the same `SMG_VLLM_MM_MAX_VIDEO_FRAMES`
  cap the in-process and Redis processors apply, so an unbounded count stays
  bounded in `smg` mode too (a cap below the engine's count then samples fewer
  frames than the engine's own server would, as intended), and a loader rule
  the pipeline cannot follow (`video_backend` in the engine's media kwargs,
  else `VLLM_VIDEO_LOADER_BACKEND`, other than `opencv`, or a `video.fps` above
  zero, which thins frames by duration) is refused at launch for a family that
  samples the way the loader does (Gemma 4) and serves video at all; the
  families with a rate-based sampler of their own never followed the loader and
  start as before. An engine that normalizes pixels on
  device
  (vLLM's `mm_device_do_normalize`, on by default for the Qwen-VL family)
  takes raw `uint8` pixels, and the pipeline writes those for it; a model
  whose processor cannot emit raw pixels is refused at startup under that
  setting (start the engine with `--mm-device-do-normalize=false` or use
  `inprocess`). The Python servicer refuses `smg`: it has no such processor, and
  the launcher's switch refuses it before the engine is built and the model
  loaded (one configuration-error line, exit code 2), not after.

An audio reference (a family whose audio the engine's own processor takes)
reaches `inprocess` and `redis` on the Rust servicer as it does on the Python
servicer; the `smg` pipeline takes image and video references and refuses an
audio one before any work, naming the modes that take it.

The in-flight cap, the saturation refusal, the advertised `mm_processor` /
`mm_media_ref_schemes` / `mm_processor_source` and the PD prefill side's
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

A constraint the engine's backend cannot enforce is refused before the engine
with INVALID_ARGUMENT (a 400 at a router), the same way on every Rust servicer:
a JSON-schema feature xgrammar drops (`multipleOf`, `uniqueItems`, a string
`format` it lacks, ...) wherever xgrammar is the backend, whether the schema is
a `response_format`'s or inside a forced tool call's structural tag; every kind
of constraint on an engine with no grammar backend; a kind the backend does not
compile. The vLLM servicer reads the backend off `--structured-outputs-config`,
the SGLang and TokenSpeed servicers off their engine's `--grammar-backend` (see
their sections); a backend the servicer does not know is left to the engine.

A constraint the engine admitted and then could not continue the request under
(a sampled token the grammar rejects, as speculative decoding can produce) is
one status on every servicer that can tell it apart, Rust and Python:
`FAILED_PRECONDITION` with the trailer `smg-constraint-failed: engine`, the
message naming the constraint and keeping the engine's own words. Behind the
servicers a router answers the client a 422 under the code `constraint_failed`,
replays nothing (the same tokens fail the same way, each replay a prefill), and
samples no circuit breaker against the worker; a router that predates the
trailer answers a 400 for the code and retries nothing. The status is raised
only on the engine's own attribution, never inferred from the request carrying
a constraint: an engine aborts or fails requests for other reasons too (a pause
or an abort-all, a decode side's failed KV handoff, a capacity abort), and
those keep the status they have on any request. Per engine:

- SGLang: the scheduler's grammar paths end such a request with a bare
  `FINISH_ABORT()` (message `Aborted`, no status, no `err_type`), the finish an
  abort-all gives every running request too. This package's SGLang plugin
  (`smg_grpc_servicer.sglang.zmq_plugin`, loaded in every scheduler rank)
  hooks those paths (`SchedulerBatchResultProcessor._accept_grammar_tokens`,
  `_apply_prefill_grammar`, and the PD decode side's `process_prebuilt`) and
  stamps the finish they set with `err_type` `GrammarRejectedToken`; the ZMQ
  wire carries it in the slim batch's `finished_err_types` column, the Python
  servicer reads it off the finish dict. The hooks register under the ZMQ
  handshake and under `SMG_SGLANG_ATTRIBUTE_GRAMMAR_ABORTS=1`, which the Python
  servicer sets for the ranks it spawns; a release without one of the paths
  keeps that path's aborts bare, which the servicers answer as before.
- vLLM: the engine finishes a grammar rejection, a grammar compile error, a
  failed KV load, an encoder-cache miss and a pre-processing fault alike with
  `error`, and a pause or shutdown with `abort`; nothing on its wire names the
  cause, so an `error` stays the engine's own report (retried, charged and
  probed as such) and an `abort` keeps its shape.
- TokenSpeed: the engine's abort codes name transfer, numerical and capacity
  failures and the pause's abort, none the grammar (a rejected token is dropped
  on the speculative path); the slim wire carries the finish type alone. Its
  finishes keep their shapes.

Both vLLM servicers also advertise the engine's reasoning parser
(`engine_reasoning_parser`; `reasoning_parser` is the per-model parser override
a router reads off a worker's labels) and where its grammar starts on a prompt
that ends inside the model's thinking block (`constrained_decoding_start`): `after_reasoning` when
a reasoning parser is set and `enable_in_reasoning` is off, since vLLM then
holds a request's grammar back until the parser has seen the thought's end,
`first_token` otherwise. A router shapes a forced tool call's grammar by it as
the engine's own server would: the bare calls behind the parser's gate, so the
call is forced right there; the thought and then the calls from the first
token.

#### CUDA-graph capture cap

The engine replays a captured CUDA graph for a step that fits one of its
capture sizes and runs a larger step eagerly, kernel by kernel from Python.
Its default for the largest size (`max_cudagraph_capture_size`) is twice
`--max-num-seqs`, at most 512 (1024 on one newer GPU platform): with a window
of 32 that is 64 tokens, so a step that admits a prefill chunk beside the
decodes runs eagerly, and an engine whose eager step is expensive (a
sparse-attention or MoE model: hundreds of launches per rank) is host-bound at
every admission. A gRPC launch (`vllm serve --grpc`, the stock launcher, and
`vllm serve --headless` for the engine-only ranks of a multi-node engine)
therefore derives the cap from the admission step when the operator set none:
`--max-num-batched-tokens` (unset: unbounded, the engine's own step-size
defaults are 2048 or more), at most 512, applied only where it raises the
engine's default, for both servicer implementations, and logged once at start
(`CUDA-graph capture cap: ...`). The engine's rule is kept: the value is the
step size itself or a value of the engine's capture grid (1, 2, 4, multiples
of 8 to 248, multiples of 16 from 256), which the engine neither clamps nor
truncates.

| setting | engine default | launch default |
|---|---|---|
| `--max-num-seqs 32 --max-num-batched-tokens 16384` | 64 | 512 |
| `--max-num-seqs 32 --max-num-batched-tokens 300` | 64 | 300 |
| `--max-num-seqs 32`, step size unset | 64 | 512 |
| `--max-num-seqs 32 --max-num-batched-tokens 48` | 48 | the engine's |
| `--max-num-seqs 256` | 512 | the engine's |
| `--max-num-seqs` above 256 | 512, or twice the window up to 1024 on one platform | the engine's |
| `--max-num-seqs` unset | 512 or 1024 by platform (the engine's window defaults) | the engine's |
| `--max-num-seqs` unset, `--max-num-batched-tokens 16384`, `SMG_VLLM_CUDAGRAPH_CAPTURE_CAP=2048` | 512 or 1024 by platform | 2048 |

Kept as given: an explicit `--max-cudagraph-capture-size`,
`--cudagraph-capture-sizes` or the same keys of `--compilation-config`;
`--enforce-eager` and a `cudagraph_mode` of `NONE` (no graphs); a
`--speculative-config` (the engine's default adds the uniform-decode sizes an
explicit cap drops); an HTTP `vllm serve`, which is not this package's launch.

`SMG_VLLM_CUDAGRAPH_CAPTURE_CAP` is the knob: `auto` (the default), `engine`
(the opt-out: the engine's own default, as before this derivation), or a
positive integer, the ceiling in place of 512, rounded down to the capture
grid. A ceiling applies only where it does not lower the engine's default on
any platform (twice the window, at most 512, or at most 1024 on one newer GPU
platform the launch cannot tell apart, each as the engine's list ends it):
with `--max-num-seqs` left to the engine or set to 512 or more, a ceiling of
1024 or more applies (it raises the 512 default and, on the platform whose
default is already 1024, matches it); for a window between 257 and 511, a
ceiling of twice the window (on the capture grid) or more applies; a smaller
one is kept out, the line saying why. Each captured size costs capture time at start and
graph memory per rank: 11 sizes at the engine's 64, 51 at 512, 83 at 1024,
147 at 2048.

##### Per recipe: what a ceiling above 512 buys, and what it costs

The ceiling of 512 is the engine's own default ceiling, so the launch default
never spends more on graphs than the engine does for itself at a large window,
and it is the right cap where the admitted step stays under it. Under
open-loop arrivals with chunked prefill and prefix caching, the step that
decides capacity holds up to `--max-num-seqs` decode tokens plus the prefill
chunks admitted beside them, so on a large running window that step is above
512 and runs eagerly at every arrival. Measured on three recipes (the same
build and recipe on both arms, one engine per arm, open-loop Poisson ramps of
150 s per step; SLO-1 = TTFT p99 under 1 s at goodput >= 0.9, SLO-2 = TTFT
p99 under 400 ms and TPOT p99 under 25 ms at goodput >= 0.95; capacity = the
highest offered rate at which the SLO holds, in requests per second per
engine; the costs are per engine against the 512 arm: KV-cache tokens,
container memory at the ramp's end, start-up time):

| recipe | 512 (the launch default) | 1024 | 2048 |
|---|---|---|---|
| `--max-num-seqs 32 --max-num-batched-tokens 16384`, a 1M-context model (the engine's own 64); 8k prompts, 4k shared prefix | SLO-1 12, SLO-2 11 | SLO-1 12, SLO-2 8; -0.5 % KV, +2.0 GiB, +45 s | SLO-1 12, SLO-2 10; -1.5 % KV, +6.1 GiB, +93 s |
| `--max-num-seqs 128 --max-num-batched-tokens 16384`, a sparse-attention MoE (the engine's own 256); 4k prompts, 1k shared prefix | SLO-1 28-34, SLO-2 20-22; collapses from 36 | SLO-1 40 (the ramp's top), SLO-2 24; -0.3 % KV, +1.7 GiB, +38 s | SLO-1 38, SLO-2 28; -0.9 % KV, +4.9 GiB, +126 s |
| `--max-num-seqs 256 --max-num-batched-tokens 16384`, a 3B-active MoE (the engine's own 512: the derivation changes nothing); 8k prompts, 4k shared prefix | SLO-1 11-12, SLO-2 5-6 | SLO-1 13-14, SLO-2 7-8; -0.15 % KV, +1.5 GiB, +27 s | SLO-1 20 (the ramp's top), SLO-2 14; -0.5 % KV, +4.7 GiB, +60 s |

The rows are same-build pairs on the engine release before the shipped one.
On the shipped release the 128-sequence recipe was run at 1024 only: SLO-1 24
req/s (22.9 achieved, TTFT p99 394 ms, TPOT p99 38 ms), no SLO-2, a collapse
from 28 with the running window full and the decode step near 75 ms at 127
running. That release's step at a full window bounds the bed below the
previous release's 512 knee, and without a 512 arm on it the ceiling's own
effect on the shipped release is not separated from the release's step time
(that arm is owed); the rows above are the measured basis of the rule, not a
reading of the shipped release.

The cap that captures a recipe's gain follows its running window (about eight
times `--max-num-seqs` on these three), not one ceiling: with a window of 128,
1024 buys the whole of 2048's gain at a third of the memory; with 256 the gain
needs 2048; with 32 nothing above the launch default buys anything (on the
128-window recipe, 8k prompts at a 94 % cache hit held the ramp's top on 512
and 2048 alike: that shape's knee lay above the ramp, which is why the wide
shape is the one in the table). The launch default therefore stays at the
memory-neutral 512, and a recipe whose class gains sets its own cap: the
engine's own `--max-cudagraph-capture-size N` among the recipe's flags (it
travels with the command line to every rank of a multi-node engine), or
`SMG_VLLM_CUDAGRAPH_CAPTURE_CAP=N` in the environment of every rank's process,
a ceiling the derivation still bounds by the admission step and applies where
it raises the engine's default on every platform. Graph memory per captured size was
0.011-0.021 GiB per rank on these three models, which is how the KV-cache
cost above scales with the capture list.

#### Multi-node engines: the worker pod's liveness

An engine whose tensor-parallel ranks span pods runs its EngineCore and the
gRPC servicer on a leader pod (`--nnodes N --node-rank 0 --master-addr <its
own address> --master-port <port>`) and the other ranks on worker pods
(`vllm serve <model> --headless --nnodes N --node-rank <k> --master-addr
<the leader>`, the same engine flags). Every rank joins the process group
through the leader's TCP store at `--master-port` and keeps that connection
for its lifetime. When the leader's engine goes away (the pod restarts, or
the servicer gives up on a start), the ranks on the worker pods are not
told: their store connection is dead and they wait on it, the next leader
waits for ranks that never join, and a worker probe that only asks whether
the engine process exists (`grep vllm /proc/1/cmdline`) keeps those orphans
running until someone deletes the pod. A leader restart must restart its
workers: ranks cannot re-join a store they were not started against.

`python -m smg_grpc_servicer.vllm.worker_probe` is the worker pod's probe.
It passes while at least one TCP connection from the pod to the leader's
store port is established and fails otherwise, so the kubelet restarts the
worker and its fresh ranks join the new leader within `periodSeconds x
failureThreshold`. Without arguments it reads `--master-addr` and
`--master-port` off the worker's own command line (`/proc/1/cmdline`);
`--leader <name>` names the leader by its stable DNS name instead,
re-resolved on every run, so that a recreated leader pod at a new address
fails the probe even while the stale connections to the old one linger, and
`--port` names the store port. The ranks connect to the store early in
their start, before the model loads, so the same command serves as the
startup probe, with a budget that covers waiting for the leader. Run it
with `python3 -I`: the probe itself needs only the standard library, but
the engine's `PYTHONPATH` may carry site customizations that import the
engine, and an interpreter start on a node saturated by the weight stream
takes seconds; `-I` leaves both out, and `timeoutSeconds` of 30 or more
keeps a slow start from counting as a failure:

```yaml
# the worker pod's container; LEADER_HOST is the leader's stable DNS name
startupProbe:
  exec: {command: [/bin/sh, -c, 'python3 -I -m smg_grpc_servicer.vllm.worker_probe --leader "$LEADER_HOST"']}
  periodSeconds: 30
  timeoutSeconds: 30
  failureThreshold: 480
livenessProbe:
  exec: {command: [/bin/sh, -c, 'python3 -I -m smg_grpc_servicer.vllm.worker_probe --leader "$LEADER_HOST"']}
  periodSeconds: 30
  timeoutSeconds: 30
  failureThreshold: 3
```

### MLX

```bash
python -m smg_grpc_servicer.mlx --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051
```

### TokenSpeed

```bash
# Python (default)
python -m smg_grpc_servicer.tokenspeed --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051

# Rust request path, same entrypoint
python -m smg_grpc_servicer.tokenspeed --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051 --servicer-impl rust

# The environment form, overridden by the flag when both are given
SMG_TOKENSPEED_SERVICER_IMPL=rust python -m smg_grpc_servicer.tokenspeed --model meta-llama/Llama-2-7b-hf --port 50051
```

Without `--kv-events-config`, the servicer turns TokenSpeed's ZMQ KV-event
publisher on by itself, under either implementation, before the engine is
built, at a port found free on the host rather than TokenSpeed's fixed default,
so `SubscribeKvEvents` relays the events a cache-aware router routes by and
workers that share a host do not collide; an explicit configuration is kept as
given, its endpoint and off included, and `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0`
leaves the publisher off. An encode worker (`--disaggregation-mode encode`) has
no KV cache and is left as given. A hybrid model (Gated DeltaNet, Mamba or KDA
layers) keeps its recurrent state in State groups of TokenSpeed's cache, which
span no tokens, and TokenSpeed's KV-event drain aborts the engine on such a
group: for a model whose layer labels the engine's own cache recipes map to a
State group, the servicer leaves the publisher off and logs why once
(`KV event publishing left off: ...`); `--kv-events-config` turns it on anyway.

This is the process `ts serve` spawns for its gRPC worker; `--servicer-impl`
is this package's flag (listed first by `--help`, ahead of TokenSpeed's own
flags), and `SMG_TOKENSPEED_SERVICER_IMPL` is the fallback the flag overrides
(flag, then the environment, then `python`; the launcher logs which one
decided). With `--servicer-impl rust` the same process serves the
`tokenspeed.grpc.scheduler.TokenSpeedScheduler` contract from Rust
(`smg.servicer.TokenSpeedGrpcServer`, which needs the `smg` wheel): the
launcher computes the model and server facts from TokenSpeed's own config,
runs the scheduler(s) headless in a spawned child over the msgpack ZMQ wire,
and supervises both. What that wire does not carry is reported, not emulated:
`FlushCache` and profiling ride the wire's control call on a scheduler whose
handshake declares its control reply (the control-wire revision) and answer
UNIMPLEMENTED, naming the gap, on one that does not; ranked `top_logprobs` and
prompt logprobs are refused, and the RL control-plane extras (the `rl.*`
advertisement, the live `weight_version` on generate responses, `is_paused`)
stay with the Python implementation; the Rust server redacts credentials from
`server_args` the same way. Prefill/decode disaggregation is served on this
path: a request's `kv_bootstrap_info` rides the wire in the scheduler's own
bootstrap slots as the Python servicer forwards it, a prefill launch starts
the engine's KV bootstrap server in the launcher process before the ranks (as
the engine's own frontend does), and under attention data parallelism the
servicer places a disaggregated request on the rank of its room's residue,
the engine's own dispatch rule, and an aggregated one on its
`data_parallel_rank` pin. The EPD prefill side is served the same way (each
item's `encode_bootstrap_info` entry rides on the item); the EPD encode role
(`--disaggregation-mode encode`) is refused at launch under `--servicer-impl
rust` and stays with the Python implementation, whose encode servicer feeds
the encode worker over the engine's in-process channel. `SubscribeKvEvents` relays
TokenSpeed's ZMQ KV-event publisher, which the servicer turns on by itself, on
this path as on the Python one, when the launcher is given no
`--kv-events-config`, at a port found free on the host rather than
TokenSpeed's fixed default (an explicit one is kept as given, its endpoint
included; `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` leaves the publisher off; a
hybrid model's State groups leave it off, as above).
Tuning:
`SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT` (default: a
free port), `SMG_TOKENSPEED_SERVICER_DRAIN_SECS` (default 5),
`SMG_TOKENSPEED_SERVICER_WARMUP` (the planned warm-up before SERVING, as for
vLLM above: `full` by default, `fast`, `off`, a budget in seconds or
`<level>:<seconds>`; planned from `--max-num-seqs` and the chunk size or
`--max-prefill-tokens`; at the `full` level, on an engine of more than one
process (its parallel layout off the server arguments; a one-process engine
plans the classes alone, like the other engines, since the hang below needs
a collective waiting on a peer), this engine's plan also visits
every prompt length up to the chunk size the class ladder leaves out, last,
in an order whose every prefix is a uniform spread (`fast` plans the classes
alone and leaves the hazard below open, which the start log says): a chunked prefill's last chunk is any count
below the chunk size, a prefill chunk reaches two cuBLAS GEMMs here (the
compressor's fp32 router projection above 32 tokens, the grouped output
projection's batched GEMM) whose kernels the library picks per token count
and loads on first use, and such a first use during serving, with the
overlapped previous chunk's TP collective in flight, has deadlocked the
forward; at a 4,096-token chunk that is 4,042 one-token requests after the
ladder's 54, about 0.17 s each on a four-GPU node (692 s measured for the
whole pass, inside this engine's 900 s default budget; with no chunk size set
the engine's default step of 8,192 tokens gives about 8,140 remainders, about
1,380 s, which the budget cuts about halfway, evenly: set a chunk size or a
larger budget to cover every count); the Python servicer runs the same plan
over its own port after its first warm-up request, and both skip it on a
disaggregation role; under attention data parallelism the servicer routes a
request by load, not by rank, so each prefill shape, the chunk remainders
included, warms one rank and the log says so: there the remainder hazard
stays open on the ranks a count did not reach until the wire can pin a rank),
`SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS` (default 1800) and
`SMG_TOKENSPEED_SERVICER_STARTUP_CEILING_SECS` (default 14400; the two startup
bounds as for vLLM above: the launcher reports the scheduler process alive each
time it polls it, so a slow but living start completes, under the ceiling, and
0 lifts the ceiling), `SMG_TOKENSPEED_SERVICER_SILENCE_TIMEOUT_SECS` (default
300) and `SMG_TOKENSPEED_SERVICER_PREFILL_TOKENS_PER_SEC` (default 2000; the
connected scheduler's silence bound and its prefill allowance as for vLLM
above: this wire carries token batches only, so a batch of long prompts is
silent for as long as its prefill takes, and the allowance, 500 s per million
pending prompt tokens at the default rate, is what keeps that from reading as
a dead scheduler), `SMG_ZMQ_SOCKET_DIR`, `SMG_SERVICER_WORKER_THREADS`
(default 1, for the one runtime `server.rs` builds for the vLLM, SGLang and
TokenSpeed Rust servicers alike: one runtime worker carries about 100k output
tokens/s on one connection with its lock uncontended; raise it to 2-4 when one
engine serves more than about 512 concurrent streams and their inter-token p99
matters more than the servicer's CPU).

Both TokenSpeed servicers advertise, as `server_args` keys, the engine's
reasoning parser (`engine_reasoning_parser`; `reasoning_parser` is the
per-model parser override a router reads off a worker's labels) and where its
grammar starts on a prompt that ends inside the model's thinking block
(`constrained_decoding_start`): always `first_token`, since this engine's grammar
manager has no reasoning gate and starts every grammar at the first generated
token (`--reasoning-parser` only has a JSON-schema constraint rewritten into a
structural tag that lets the thought precede the JSON). A router shapes a
forced tool call's grammar by it as the engine's own server would: the thought
and then the calls, so the call is not forced into the thought.

They also carry the grammar backend the engine runs as the `grammar_backend`
key, resolved: `--grammar-backend` as given (`xgrammar` or `none`), else the
checkpoint's default once the engine has read it (`none` for most checkpoints;
the key is absent until it has). The Rust servicer refuses a constraint that
backend cannot enforce before the scheduler, as the vLLM servicer does: every
kind under `none` (the scheduler itself would admit the request and abort it
with a 400 that its slim wire reports as a bare `abort`, an empty answer to the
caller), and a JSON-schema feature xgrammar drops under `xgrammar`; with no key
it judges nothing ahead of the engine.

Both TokenSpeed servicers also report the engine's constrained-decoding policy,
when it declares one, as `GetServerInfo.constrained_decoding` (a field of its
own, not a `server_args` key): `always` for a model that takes a grammar on
every turn, `requested` for one only when the request asks. The servicers read
it as `constrained_decoding` off TokenSpeed's `ServerArgs`, else its
`ModelConfig`. A router turns the field into the worker's `constrained_decoding`
label, and its gRPC request building follows the selected worker's policy (under
PD the decode worker's). `requested`, the default, constrains what a request asks
for. `always`, on an engine that starts the grammar at the first token, gives
every turn the grammar of the model's response template table where that table
spells whole turns; otherwise it constrains the calls a tool request allows under
`auto`.

#### Metrics (`--metrics-port`)

With `--servicer-impl rust`, `--metrics-port <port>` (and `--metrics-host`,
defaulting to the gRPC `--host`) serves the Rust servicer's Prometheus metrics
on a listener of its own; `SMG_TOKENSPEED_SERVICER_METRICS_PORT` /
`SMG_TOKENSPEED_SERVICER_METRICS_HOST` are the environment forms the flags
override, and without either there is no endpoint. The families and labels are
the ones every engine's Rust servicer exposes (see the vLLM section above):
requests by finish reason, refusals, errors, in-flight requests, prompt /
cached / generated tokens, the latency histograms, the lifecycle state. A port
equal to `--port` is refused before the engine starts, and the flag is refused
under the Python servicer, which exposes no endpoint.

```bash
python -m smg_grpc_servicer.tokenspeed --model <model> --port 50051 --servicer-impl rust --metrics-port 29100
```

#### Multi-node engines: the worker pod's liveness

An engine whose tensor-parallel ranks span pods is launched on every pod with
the same flags, `--nnodes N --dist-init-addr <the leader>:<port>` among them,
and the pod's own `--node-rank <k>`: the launcher above on the leader pod
(`--node-rank 0`, where the gRPC servicer runs), the scheduler ranks alone on
the worker pods. Every rank joins the process group through the TCP store at
`--dist-init-addr`, which rank 0 on the leader pod hosts, and keeps that
connection for its lifetime, so the contract of the vLLM section above holds
here too: a leader restart must restart its workers, and
`python -m smg_grpc_servicer.tokenspeed.worker_probe` is the worker pod's
probe. It passes while at least one TCP connection from the pod to the
leader's store port is established and fails otherwise; without arguments it
reads the leader and the port off `--dist-init-addr` on the worker's own
command line (`/proc/1/cmdline`), `--leader <name>` names the leader by its
stable DNS name instead, re-resolved on every run, and `--port` names the
store port (both are needed when TokenSpeed derives the flag from a multi-node
Slurm step's environment rather than the command line). Run it with
`python3 -I` and a `timeoutSeconds` of 30 or more, as above:

```yaml
# the worker pod's container; LEADER_HOST is the leader's stable DNS name
startupProbe:
  exec: {command: [/bin/sh, -c, 'python3 -I -m smg_grpc_servicer.tokenspeed.worker_probe --leader "$LEADER_HOST"']}
  periodSeconds: 30
  timeoutSeconds: 30
  failureThreshold: 480
livenessProbe:
  exec: {command: [/bin/sh, -c, 'python3 -I -m smg_grpc_servicer.tokenspeed.worker_probe --leader "$LEADER_HOST"']}
  periodSeconds: 30
  timeoutSeconds: 30
  failureThreshold: 3
```

### SGLang

```bash
sglang serve --model-path meta-llama/Llama-2-7b-hf --grpc-mode
```

Without `--kv-events-config`, the servicer turns SGLang's ZMQ KV-event
publisher on by itself, under either implementation, before the schedulers are
started, at a port found free on the host rather than SGLang's fixed default,
so `SubscribeKvEvents` relays the events a cache-aware router routes by and
workers that share a host do not collide; an explicit configuration is kept as
given, its endpoint and off included, and `SMG_SGLANG_SERVICER_KV_EVENTS=0`
leaves the publisher off.

Both SGLang servicers advertise the engine's `--limit-mm-data-per-request` as
the flat `mm_item_limits` label (`image=1,video=1`) in `GetServerInfo`'s
`server_args`, so a router that preprocesses media itself holds requests to
the engine's limits: the precomputed inputs it sends bypass the tokenizer
manager's own check. Without the flag the engine has no such limit and the
label is absent.

They also advertise, as `server_args` keys, the engine's reasoning parser
(`engine_reasoning_parser`; `reasoning_parser` is the per-model parser override
a router reads off a worker's labels) and where its grammar starts on a prompt
that ends inside the model's thinking block (`constrained_decoding_start`):
`after_reasoning` with `--reasoning-parser`, since the scheduler then wraps its
grammar backend in a reasoner that holds a request's grammar back until the
parser's thought-end sequence has been generated (for a request that asks for
reasoning, which a router does on a thinking prompt), `first_token` without
one. A router shapes a forced tool call's grammar by it as the engine's own
server would: the bare calls behind the parser's gate, so the call is forced
right there; the thought and then the calls from the first token.

They also carry the grammar backend the scheduler runs as the `grammar_backend`
key, resolved: `--grammar-backend` as given (`xgrammar`, `outlines`,
`llguidance` or `none`), `xgrammar` when it was not given, and `none` under
`--skip-tokenizer-init`, since the scheduler builds no grammar backend without
its tokenizer. The Rust servicer refuses a constraint that backend cannot
enforce before the scheduler, as the vLLM servicer does: every kind under
`none` (the scheduler itself would admit the request and abort it with a 400),
an EBNF grammar or a structural tag under `outlines`, which compiles JSON
schemas and regular expressions only, and a JSON-schema feature xgrammar drops
under `xgrammar`; a backend it does not know (a plugin of the scheduler's) is
left to the engine.

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
without one those are counted as lost or unknown, never silently skipped. The
relay dials the publisher every second from the servicer's start, so its
subscription lands within a second of the engine binding the socket, and the
launcher starts its planned warm-up only after that, so the warm-up's batches
(the engine's first) are the first the relay's history records.

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

#### Rust request path (`--servicer-impl rust`)

`sglang serve --grpc-mode` (`python -m sglang.launch_server --grpc-mode`)
hands the process to this package's `serve_grpc`. With `--servicer-impl rust`
that entry serves the `sglang.grpc.scheduler.SglangScheduler` contract from
Rust (`smg.servicer.SglangGrpcServer`, which needs the `smg` wheel) instead:
the scheduler runs headless in a spawned child over the msgpack ZMQ wire
below, the Rust server speaks the gRPC contract on top of it, and Python keeps
the lifecycle only. The flag is this package's, as it is for vLLM: SGLang
loads it as an `sglang.srt.plugins` entry point while it builds its parser
(`smg_grpc_servicer/sglang/plugin.py`), and `SMG_SGLANG_SERVICER_IMPL` is the
fallback the flag overrides (flag, then the environment, then `python`; the
servicer logs which one decided). A `SGLANG_PLUGINS` whitelist without
`smg-servicer` removes the flag, not the environment form.

```bash
# Python (default)
sglang serve --model-path Qwen/Qwen3-0.6B --grpc-mode --port 30000

# Rust request path, same entrypoint
sglang serve --model-path Qwen/Qwen3-0.6B --grpc-mode --port 30000 --servicer-impl rust

# The environment form, overridden by the flag when both are given
SMG_SGLANG_SERVICER_IMPL=rust sglang serve --model-path Qwen/Qwen3-0.6B --grpc-mode --port 30000
```

`python -m sglang.launch_server --help` lists the flag; `sglang serve --help`
prints SGLang's help before it loads plugins, so the flag is missing from that
listing but accepted. The Router cannot tell the two implementations apart: text
generation with sampled and prompt logprobs, reasoning-token counts, LoRA ids
and custom logit processors forwarded as the Python servicer forwards them,
`Embed`, `FlushCache` and profiling (the scheduler's own control requests,
carried by the wire's control call) all answer as the Python servicer does.
Hidden states (`return_hidden_states`) are served on this path, and on this
path only: the final hidden layer's vector per position a forward ran on (the
prompt positions the prefill computed, then each generated token fed back:
(prompt - cached prefix) + generated - 1 rows, with absolute positions anchored
from the engine's counters), the prefill block on the first chunk, a decode
step's rows on each later one and the whole list on the completion, within the
scheduler's `--return-hidden-states-mode` cap, which both servicers keep ahead
of the scheduler with the engine's own words; the Python servicer then refuses
the request by name (`INVALID_ARGUMENT`, pointing at `--servicer-impl rust`),
where it once accepted the flag and returned nothing.
Prefill/decode disaggregation (`--disaggregation-mode prefill|decode`) works as
on the Python servicer: a request's bootstrap fields (host, port, room) ride
the scheduler's own request struct over the wire, the prefill side's bootstrap
server starts in the servicer process before the ranks, and `GetServerInfo`
carries the server args the Router pairs on. Multimodal inputs reach the
scheduler as its own processor output: the preprocessed tensors the Router's
media pipeline sends ride the wire as SGLang's own tensor encoding, and the
scheduler hashes, pads and positions them as it does behind the Python
servicer; `supports_vision` is advertised as SGLang's model config decides it.
The language side of encode/prefill disaggregation (`--language-only`) works as
behind SGLang's own frontend: the servicer process runs SGLang's own encoder
bootstrap server and receiver (`--encoder-bootstrap-port`, `--encoder-urls`,
the receiver's transport by `SGLANG_ENCODER_MM_RECEIVER_MODE`), a request's
image references (`mm_inputs.image_urls`, with the prompt tokenized but not
expanded and no tensors) go to the encoders through them before the request
reaches the scheduler, the scheduler's request carries what the dispatch set
(its wait flag, the per-encoder assignment, the encoder list) and the prompt
text the scheduler rebuilds from once the embeddings arrive, and a dispatch
that fails afterwards reaches the scheduler as its own failure notice. A
request of references on any other worker, or on a language-only worker with
no encoder registered, is refused (`FAILED_PRECONDITION`) rather than answered
as text. A request the Router pins to a data-parallel rank (`--dp-size` above one
behind a dp-aware Router) goes to that rank's scheduler, as the engine's
data-parallel controller sends it behind SGLang's own frontend; a pin past the
connected ranks is refused as the caller's error. On the prefill side of a disaggregated pair a request runs on
the rank of its room's residue instead (`bootstrap_room % dp_size`: SGLang's
prefill side follows the room and its KV sender fails a request placed
elsewhere, and the decode side looks for the KV on that rank), the Router's pin
yielding to it with a warning; the decode side keeps the pin. A prefill leg of
several plain data-parallel replicas (`--dp-size N` without attention DP) is
refused at launch with the engine's reason: every replica reports attention-DP
rank 0 to the engine's KV sender, which fails each request whose room's residue
is not 0; attention DP (`--tp-size N --attn-dp-size N`) is the multi-rank
prefill shape. The planned
warm-up pins each of its requests to a rank (on a prefill or decode leg with a
stand-in room of that rank's residue), so every rank is warm before the Router
sends anything. What the wire does not carry is reported, not emulated: an `--encoder-only` launch is refused (SGLang's entrypoint runs
that role on its own encoder server, never on a servicer), a `--language-only`
launch on the `zmq_to_tokenizer` transfer backend is refused (the frontend
itself receives and processes the embeddings there, which neither servicer
runs), and LoRA loading answers UNIMPLEMENTED, as it does on the Python
servicer.
`SubscribeKvEvents` relays SGLang's ZMQ KV-event publisher, which the servicer
turns on by itself, on this path as on the Python one, when the launcher is
given no `--kv-events-config`, at a port found free on the host rather than
SGLang's fixed default (an explicit one is kept as given, its endpoint
included; `SMG_SGLANG_SERVICER_KV_EVENTS=0` leaves the publisher off). SGLang's
HTTP sidecar (metrics and profiling endpoints) is not started on this path.
Tuning: `SMG_SGLANG_SERVICER_HANDSHAKE_PORT` (default: a
free port), `SMG_SGLANG_SERVICER_DRAIN_SECS` (default 5),
`SMG_SGLANG_SERVICER_WARMUP` (the planned warm-up before SERVING, as for vLLM
above: `full` by default, `fast`, `off`, a budget in seconds or
`<level>:<seconds>`; planned from `--max-running-requests` and
`--chunked-prefill-size` or `--max-prefill-tokens`, the engine's defaults
where unset; the Python servicer runs the same plan over its own port after
its first warm-up request and skips it on a disaggregation role; the Rust
path plans a prefill or decode leg as a single engine, its requests carrying
the engine's stand-in bootstrap so the leg warms up without its peer, and the
language side of an encode-disaggregated deployment (`--language-only`) plans
as a single engine, so the first forward of its fresh scheduler is paid before
SERVING rather than by its first request, as the engine's own frontend warms
that role on the text path, and where its
frontend dispatches to the encoders the plan ends its shapes with one image
through them (the prompt the model's chat template renders around the
engine's own minimal warm-up image; a template without an image placeholder
leaves the leg out, a leg the encoders cannot serve yet is a logged step), so
the scheduler's first media forward is paid before SERVING too; under
data parallelism the servicer routes a request by load, not by rank, so each
prefill shape warms one rank and the log says so),
`SMG_SGLANG_SERVICER_STARTUP_TIMEOUT_SECS` (default 1800) and
`SMG_SGLANG_SERVICER_STARTUP_CEILING_SECS` (default 14400; the two startup
bounds as for vLLM above: the launcher reports the scheduler process alive each
time it polls it, so a slow but living start completes, under the ceiling, and
0 lifts the ceiling), `SMG_SGLANG_SERVICER_SILENCE_TIMEOUT_SECS` (default 300)
and `SMG_SGLANG_SERVICER_PREFILL_TOKENS_PER_SEC` (default 2000; the connected
scheduler's silence bound and its prefill allowance as for TokenSpeed above:
this wire too carries token batches only), `SMG_ZMQ_SOCKET_DIR`,
`SMG_SERVICER_WORKER_THREADS`.

#### Metrics (`--metrics-port`)

With `--servicer-impl rust`, `--metrics-port <port>` (and `--metrics-host`,
defaulting to the gRPC `--host`) serves the Rust servicer's Prometheus metrics
on a listener of its own; `SMG_SGLANG_SERVICER_METRICS_PORT` /
`SMG_SGLANG_SERVICER_METRICS_HOST` are the environment forms the flags
override, and without either there is no endpoint. The flags are this package's
(the same plugin that adds `--servicer-impl`); a `SGLANG_PLUGINS` whitelist
without `smg-servicer` removes them, not the environment form. The families
and labels are the ones every engine's Rust servicer exposes (see the vLLM
section above); the scheduler matches string stops itself, so this servicer
has no `smg_servicer_string_stops_total` series. A port equal to `--port` is
refused before the scheduler starts, and the flag is refused under the Python
servicer, which exposes no endpoint (SGLang's own HTTP sidecar is not started
on the Rust path either).

```bash
sglang serve --model-path Qwen/Qwen3-0.6B --grpc-mode --port 30000 --servicer-impl rust --metrics-port 29100
```

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
SMG's wait for the scheduler's HELLO, and for the READY and the registration
that follow it, is bounded per stage by the router's
`--zmq-startup-timeout-secs` (default 3600 s; `--router-zmq-startup-timeout-secs`
from `smg serve`): the scheduler dials in only after its model is loaded, so the
bound has to cover that load.
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
dropped. SGLang itself is unchanged. The plugin declares the field order of
the structs it decodes in its ready response, and the Rust servicer checks the
prefixes it encodes against it once, at the handshake: a release that moves a
field inside them fails the engine link with the struct and the field named,
instead of refusing every request. The sampling parameters go out as their API
prefix (through `custom_params`), so a release that adds a field after it
decodes them unchanged; a struct change upstream otherwise shows up in
`grpc_servicer/tests/test_sglang_zmq_msgpack.py`.


#### Multi-node engines: the worker pod's liveness

An engine whose tensor-parallel ranks span pods is launched on every pod with
the same flags, `--nnodes N --dist-init-addr <the leader>:<port>` among them,
and the pod's own `--node-rank <k>`: the gRPC servicer (`sglang serve
--grpc-mode`) or the headless launcher above on the leader pod (`--node-rank
0`), the scheduler ranks alone on the worker pods. Every rank joins the
process group through the TCP store at `--dist-init-addr`, which rank 0 on
the leader pod hosts, and keeps that connection for its lifetime, so the
contract of the vLLM section above holds here too: a leader restart must
restart its workers, and `python -m smg_grpc_servicer.sglang.worker_probe` is
the worker pod's probe. It passes while at least one TCP connection from the
pod to the leader's store port is established and fails otherwise; without
arguments it reads the leader and the port off `--dist-init-addr` on the
worker's own command line (`/proc/1/cmdline`), `--leader <name>` names the
leader by its stable DNS name instead, re-resolved on every run, and `--port`
names the store port. Run it with `python3 -I` and a `timeoutSeconds` of 30
or more, as above:

```yaml
# the worker pod's container; LEADER_HOST is the leader's stable DNS name
startupProbe:
  exec: {command: [/bin/sh, -c, 'python3 -I -m smg_grpc_servicer.sglang.worker_probe --leader "$LEADER_HOST"']}
  periodSeconds: 30
  timeoutSeconds: 30
  failureThreshold: 480
livenessProbe:
  exec: {command: [/bin/sh, -c, 'python3 -I -m smg_grpc_servicer.sglang.worker_probe --leader "$LEADER_HOST"']}
  periodSeconds: 30
  timeoutSeconds: 30
  failureThreshold: 3
```


### TensorRT-LLM

The engine's own Python servicer lives in the TensorRT-LLM tree
(`tensorrt_llm.grpc.smg`, what `trtllm-serve <model> --grpc --grpc-protocol smg`
runs, in-process over the LLM API). This package adds the Rust request path
for it, behind one switch.

#### Rust request path (`SMG_TRTLLM_SERVICER_IMPL=rust`)

With `SMG_TRTLLM_SERVICER_IMPL=rust`, the engine's gRPC entrypoint hands the
process to this package's `serve_rust_from_engine` before an LLM or a Python
gRPC server exists, and `python -m smg_grpc_servicer.trtllm --servicer-impl
rust` reaches the same path: the `trtllm.TrtllmService` contract is served by
`smg.servicer.TrtllmGrpcServer` (the `smg` wheel) on a Rust-owned thread, and
the engine runs behind the wire shim beside its LLM API
(`tensorrt_llm.grpc.smg.shim` in the engine's tree, or the same package
installed beside it as `trtllm_smg_shim`; the launcher finds it by import),
started from this process as a spawned child that dials the servicer's
same-host msgpack ZMQ handshake. The shim declares the field order of the
structs it decodes in its ready response and the Rust servicer checks the
prefixes it encodes against it once, at connect time. Python keeps the
lifecycle only; the Router cannot tell the two implementations apart.

```bash
# Python (default): the engine's own servicer
trtllm-serve org/model --grpc --grpc-protocol smg --port 50051

# Rust request path, same entrypoint
SMG_TRTLLM_SERVICER_IMPL=rust trtllm-serve org/model --grpc --grpc-protocol smg --port 50051

# This package's launcher; --llm-args carries the LLM API's keywords (JSON text, or @file)
python -m smg_grpc_servicer.trtllm --model org/model --port 50051 --servicer-impl rust \
    --llm-args '{"tensor_parallel_size": 2}'
```

Text generation with sampled, ranked and prompt logprobs, string and token
stops matched by the engine (`matched_stop` from its `stop_reason`), images as
raw bytes decoded by the engine's own loader, guided decoding (the five kinds,
refused before the engine when its `guided_decoding_backend` cannot enforce
them), LoRA by task id and the disaggregated parameters pass through as the
Python servicer forwards them; `num_return_sequences > 1` fans out to one
engine request per sample with derived seeds. What the wire does not carry is
refused by name instead of dropped: beam search, the logits-returning flags,
KV-cache retention, prompt-tuning and lookahead configs. `Embed` and
`SubscribeKvEvents` answer UNIMPLEMENTED, as the Python servicer answers them;
`GetTokenizer` serves the local tokenizer directory. The launcher's knobs are
the shared ones under the `SMG_TRTLLM_SERVICER_*` names (handshake port, drain,
start-up and silence bounds, prefill allowance); `--metrics-port` (and
`--metrics-host`, `SMG_TRTLLM_SERVICER_METRICS_PORT/_HOST`) serves the Rust
servicer's Prometheus metrics, the families every engine's Rust servicer
exposes (see the vLLM section above), with `wire="trtllm"`.

## Architecture

```
smg-grpc-servicer[vllm]    ──optional dep──>  vllm       (lazy import)
smg-grpc-servicer[mlx]     ──optional dep──>  mlx-lm     (lazy import)
smg-grpc-servicer          ──external runtime──>  tokenspeed (lazy import)
smg-grpc-servicer[sglang]  ──optional dep──>  sglang     (lazy import)
smg-grpc-servicer          ──external runtime──>  tensorrt_llm (the shim beside its LLM API, lazy)
smg-grpc-servicer          ──depends on────>  smg-grpc-proto  (hard dependency)
vllm                       ──optional──────>  smg-grpc-servicer (via vllm serve --grpc)
sglang                     ──optional──────>  smg-grpc-servicer (via --grpc-mode)
```

Backend dependencies are isolated via extras or runtime installs to avoid conflicts between vLLM, MLX, TokenSpeed, SGLang, and TensorRT-LLM.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local development setup, CI, and release workflows.
