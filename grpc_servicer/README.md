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
and run vLLM's own multimodal processor:

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

### MLX

```bash
python -m smg_grpc_servicer.mlx --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051
```

### TokenSpeed

```bash
python -m smg_grpc_servicer.tokenspeed --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --port 50051
```

### SGLang

```bash
sglang serve --model-path meta-llama/Llama-2-7b-hf --grpc-mode
```

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
