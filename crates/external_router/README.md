# smg-external-router

Routers for third-party providers behind the SMG gateway, and the contract they
are built on. The gateway depends on this crate; this crate never depends on the
gateway.

## What is in here

| Module | Role |
|---|---|
| `router` | `ExternalRouter`, the trait a provider router implements; `ExternalRouterSpec`, how the gateway knows a router; `known`, the identity table of the built-in routers |
| `context` | `ExternalContext`, everything a router may borrow from the gateway |
| `worker` | `WorkerSource` and `ExternalWorker`, the gateway's workers as a router sees them; `SelectWorkerRequest` |
| `openai`, `anthropic`, `gemini` | the built-in routers, one Cargo feature each |
| `error`, `sse`, `openai_bridge`, `mcp_utils`, `header_utils`, `retry`, `persistence_utils`, `realtime`, `metrics`, `tenant`, `sglang_fields` | protocol-level glue shared by every router family; the gateway re-exports these at their old paths |

## The contract

A provider router implements `ExternalRouter`. It has the same method names as
the gateway's router trait for the endpoints a provider can serve: chat,
Responses, Messages, Interactions, the realtime entry points, health and server
info. Every method defaults to 501, so a router implements only what its
provider offers.

The gateway hands each router an `ExternalContext`: the HTTP client, request
timeout, retry policy, the MCP orchestrator and format registry, response and
conversation storage, the realtime registry, WebRTC settings, and `workers`, a
`WorkerSource`. The source selects a worker for a request, reports per-model
retry policy and fleet stats, and lists the provider-backed workers. A selected
worker is an `ExternalWorker` handle: URL, API key, model, health, provider for
a model, outcome recording, its HTTP client, and a load hold for long-lived
sessions.

Selection carries the selecting router (`SelectWorkerRequest::router`), so in a
mixed-provider registry a router only reaches the workers it takes and a
caller's credentials never reach another provider's worker.

## How the gateway mounts a router

`router::known` lists the built-in routers whether or not this build carries
them: router id, backend name, label, the gateway Cargo feature that compiles it
in, the providers it takes (`serves`), whether it is the fallback for external
workers that name no provider, a `compiled` flag, and a constructor. On a build
without the router the constructor answers with the feature to enable.

- `builtin_routers()` is the compiled subset; IGW mode mounts all of it.
- `spec_for_backend(name)` serves `--backend <name>`; a known but absent router
  fails at startup naming its feature.
- `spec_for_provider(provider)` is the one resolution rule shared by dispatch
  and admission: a router that names a provider beats the fallback, so a
  provider-specific router can coexist with the OpenAI-compatible fallback that
  takes every custom provider.

## Features

| Crate feature | Gateway feature | Router |
|---|---|---|
| `openai` | `provider-openai` | OpenAI-compatible: OpenAI, xAI, custom providers, and the fallback |
| `anthropic` | `provider-anthropic` | Anthropic Messages |
| `gemini` | `provider-gemini` | Gemini Interactions |

The gateway's `providers` feature turns on all three and is in its default set.
A self-hosted build takes none and compiles only the glue:

```sh
cargo build -p smg --no-default-features --features grpc-client,jemalloc-stats
```

## Adding a provider

1. Add a module with a type implementing `ExternalRouter`, taking
   `&ExternalContext` in its constructor and selecting workers through
   `ctx.workers` with `router: Some(known::YOURS)`.
2. Add a crate feature and gate the module on it.
3. Add the identity to `router::known`: id, backend, label, feature, `serves`,
   `fallback: false`, `compiled: cfg!(feature = "...")`, and the two `build`
   functions (one per cfg).
4. Add the entry to `known::all()`.
5. In the gateway, forward a `provider-<name>` feature to the crate feature.

Admission, startup and dispatch then know the router without further edits.

## Checks

```sh
cargo clippy -p smg-external-router --features openai,anthropic,gemini --all-targets -- -D warnings
cargo clippy -p smg-external-router --all-targets -- -D warnings
cargo test -p smg-external-router --features openai,anthropic,gemini
```
