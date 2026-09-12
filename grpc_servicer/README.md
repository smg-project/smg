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

#### DP load reporting

`GetLoads` reports cached scheduler statistics for each managed global DP rank;
the optional `dp_rank` filter includes rank zero. Keep vLLM stats logging enabled:
missing snapshots are omitted rather than reported as idle. The gateway scopes
each virtual DP worker's load to its own rank, not the backend-wide average.

`max_total_num_tokens` uses vLLM's profiled **per-engine**
`kv_cache_size_tokens`. `num_used_tokens` is the rounded-up occupied KV-token
equivalent (`token_usage * capacity`), not pending prompt tokens. Older vLLM
versions without this capacity field retain ratio-only reporting (absolute
fields remain zero). Deferred/skipped waiting requests are included in queue
depth. Optional sections other than core are still best-effort; no extra
scheduler RPC is issued.

Rollout requires updating both the gateway binary and `smg-grpc-servicer`
inside the vLLM image. With DP4, verify four distinct ranks in unfiltered
`GetLoads` and one matching rank per `@0`–`@3` worker in the gateway's
`/get_loads`. This fixes telemetry attribution; throughput/latency improvements
require workload testing.

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
