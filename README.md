<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/smg-project/smg/main/assets/images/logomark-white.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/smg-project/smg/main/assets/images/logomark-black.svg">
    <img alt="SMG Logo" src="https://raw.githubusercontent.com/smg-project/smg/main/assets/images/logomark.svg" width="80">
  </picture>
</p>

<h1 align="center">Shepherd Model Gateway</h1>

<p align="center">
  <a href="https://github.com/smg-project/smg/releases/latest"><img src="https://img.shields.io/github/v/release/smg-project/smg?logo=github&label=Release" alt="Release"></a>
  <a href="https://hub.docker.com/r/lightseekorg/smg"><img src="https://img.shields.io/docker/v/lightseekorg/smg?logo=docker&label=Docker" alt="Docker"></a>
  <a href="https://pypi.org/project/smg/"><img src="https://img.shields.io/pypi/v/smg?logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://lightseek.org/smg/"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Docs"></a>
  <a href="https://discord.lightseek.org"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://slack.lightseek.org"><img src="https://img.shields.io/badge/Slack-Join%20Us-4A154B?logo=slack&logoColor=white" alt="Slack"></a>
  <a href="https://deepwiki.com/smg-project/smg"><img src="https://img.shields.io/badge/Ask-DeepWiki-blue.svg" alt="Ask DeepWiki"></a>
  <a href="https://pytorch.org/blog/lightseek-smg/"><img src="https://img.shields.io/badge/PyTorch-Technical%20Blog-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch Blog"></a>
</p>

Engine-agnostic, high-performance model-routing gateway for large-scale LLM deployments. SMG centralizes worker lifecycle management, balances traffic across self-hosted engines and cloud providers, and gives you enterprise-grade control over multi-tenancy, chat-history storage, MCP tooling, and observability — behind one unified endpoint.

<p align="center">
  <img src="https://raw.githubusercontent.com/smg-project/smg/main/assets/images/architecture.svg" alt="SMG architecture: clients flow through the gateway layer and router layer to gRPC workers, HTTP workers, and external APIs" width="100%">
</p>

## Why SMG?

|                                 |                                                                                                                                                                  |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **🚀 Maximize GPU Utilization** | Cache-aware routing tracks each worker's KV-cache state in radix trees to reuse prefixes across SGLang, vLLM, TensorRT-LLM, TokenSpeed, and MLX — with load modeling that accounts for queued token work and KV pressure. |
| **🔌 One API, Any Backend**     | Route to self-hosted engines over HTTP or gRPC, or to OpenAI, Anthropic, Gemini, and xAI — plus any OpenAI-compatible endpoint — through a single unified gateway. |
| **⚡ Built for Speed**           | Native Rust with streaming gRPC pipelines, cached tokenization with zero-copy cache hits, prefill/decode disaggregation (including a separate encode stage for vision), and DP-aware routing for data-parallel engines. |
| **🔒 Enterprise Control**       | Priority admission scheduling with preemption and per-tenant controls, API-key auth with OIDC on the control plane, WebAssembly plugins for custom logic, and chat history that never leaves your infrastructure. |
| **📊 Full Observability**       | 90+ Prometheus metrics, OpenTelemetry tracing with W3C trace context propagated into the engines over both HTTP and gRPC, and structured JSON logs with request correlation. |

**API Coverage:** OpenAI Chat Completions, Completions, Embeddings, Rerank, and Classify; Responses and Conversations APIs for agents; Anthropic Messages; Gemini Interactions; Realtime over WebSocket and WebRTC; audio transcription; tokenize/detokenize; and MCP tool execution with approval policies in the Responses and Messages APIs.

## Quick Start

**Install** — pick your preferred method:

```bash
# Docker
docker pull lightseekorg/smg:latest

# Kubernetes (Helm)
helm install smg oci://ghcr.io/smg-project/charts/smg

# Python
pip install smg

# Rust (needs protoc)
cargo install smg
```

**Run** — point SMG at your inference workers:

```bash
# Single worker
smg launch --worker-urls http://localhost:8000

# Multiple workers with cache-aware routing
smg launch --worker-urls http://gpu1:8000 http://gpu2:8000 --policy cache_aware

# With high availability mesh
smg launch --worker-urls http://gpu1:8000 --enable-mesh \
  --mesh-advertise-host 10.0.0.1 --mesh-peer-urls 10.0.0.2:39527
```

**Use** — send requests to the gateway:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

That's it. SMG is now load-balancing requests across your workers.

## Supported Backends

| | |
|:--|:--|
| **Self-Hosted Engines** | vLLM · SGLang · TokenSpeed · TensorRT-LLM · MLX (Apple Silicon) · any OpenAI-compatible server (e.g. Ollama) |
| **Cloud Providers** | OpenAI · Anthropic · Google Gemini · xAI · OCI Generative AI · AWS Bedrock · Azure OpenAI · any OpenAI-compatible provider (Groq, Together, …) |

## Features

| Feature | Description |
|---------|-------------|
| **[10 Routing Policies](https://lightseek.org/smg/concepts/routing/load-balancing)** | cache_aware, least_load, power_of_two, consistent_hashing, prefix_hash, bucket, round_robin, random, manual, passthrough |
| **[gRPC Pipeline](https://lightseek.org/smg/concepts/architecture/grpc-pipeline)** | Native streaming gRPC to the engines with prefill/decode and encode disaggregation and DP-aware routing |
| **[Kubernetes Discovery](https://lightseek.org/smg/getting-started/service-discovery)** | Native pod watchers with label selectors, per-role prefill/decode/encode selectors, and router peer discovery |
| **[Model Parsers](https://lightseek.org/smg/getting-started/tokenization-and-parsing)** | 21 tool-call parsers and 16 reasoning parsers with automatic model detection — DeepSeek, Qwen, Kimi, GLM, Llama, Mistral, Command, Nemotron, and more |
| **[MCP Integration](https://lightseek.org/smg/concepts/extensibility/mcp)** | Tool discovery and execution over stdio, SSE, and streamable HTTP, with approval policies and audit logging |
| **[High Availability](https://lightseek.org/smg/concepts/architecture/high-availability)** | Mesh networking with SWIM gossip and CRDT-replicated state for multi-node deployments |
| **[Chat History](https://lightseek.org/smg/concepts/data/chat-history)** | Pluggable storage with schema migrations: PostgreSQL, Oracle, Redis, or in-memory |
| **[WASM Plugins](https://lightseek.org/smg/concepts/extensibility/wasm-plugins)** | Extend request and response handling with custom WebAssembly middleware |
| **[Resilience](https://lightseek.org/smg/concepts/reliability/index)** | Circuit breakers, retries with backoff and jitter, rate limiting, and priority admission scheduling |

## Documentation

Full documentation lives at **[lightseek.org/smg](https://lightseek.org/smg/)**.

| | |
|:--|:--|
| [Getting Started](https://lightseek.org/smg/getting-started) | Installation and first steps |
| [Architecture](https://lightseek.org/smg/concepts/architecture/overview) | How SMG works |
| [Configuration](https://lightseek.org/smg/reference/configuration) | CLI reference and options |
| [API Reference](https://lightseek.org/smg/reference/api/openai) | OpenAI-compatible endpoints |
| [Kubernetes Setup](https://lightseek.org/smg/getting-started/service-discovery) | In-cluster discovery and production setup |

## Contributing

We welcome contributions! See the [Contributing Guide](https://lightseek.org/smg/contributing) for details.

- [Development Setup](https://lightseek.org/smg/contributing/development)
- [Code Style](https://lightseek.org/smg/contributing/code-style)
