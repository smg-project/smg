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
SMG_VLLM_MM_PROCESSOR=inprocess vllm serve Qwen/Qwen3-VL-8B-Instruct --grpc \
    --allowed-media-domains example.com
```

The worker then advertises `mm_processor=inprocess` and `mm_media_ref_schemes`
through `GetServerInfo`; a router with media-reference support forwards
`media_refs` only to workers that advertise, and a router without it ignores the
labels and keeps sending preprocessed tensors. vLLM's `--allowed-media-domains`,
`--allowed-local-media-path`, `--media-io-kwargs`, `--limit-mm-per-prompt` and
`VLLM_*_FETCH_TIMEOUT` govern fetching on the worker; without
`--allowed-media-domains` the worker fetches from any host the router forwards.
Related knobs: `SMG_VLLM_MM_MAX_INFLIGHT` (default 64) bounds concurrent media
jobs; `SMG_VLLM_MM_MAX_ITEMS` (default 16) caps references per request;
`SMG_VLLM_MM_MAX_ITEM_BYTES` (default 32 MiB) caps inline `data:` payloads.

On the router side, `SMG_MM_PROCESSING` selects `auto` (default: forward when
the model's spec opts in and every registered worker of the model advertises
`mm_processor`), `router` (always preprocess) or `worker` (strict: 400 when a
request cannot be forwarded); the outcome is counted in
`smg_mm_processing_total{model,mode,reason}`. On the worker path the router
never expands placeholders, so routing decisions that weigh the prompt's token
count (cache-aware policies, load estimates) see one token per media item where
the worker will schedule the full placeholder run. The `E2E_MM_PROCESSING=worker`
e2e lanes run the multimodal suites in this mode, and
`crates/multimodal/scripts/check_worker_anchor_parity.py` checks that a spec's
anchor is the token vLLM expands.

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
