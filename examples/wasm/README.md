# WASM Guest Examples for Shepherd Model Gateway

This directory contains example WASM middleware components demonstrating how to implement custom middleware for Shepherd Model Gateway (SMG) using the WebAssembly Component Model. One example, `wasm-guest-policy`, is a routing policy for the policy controller instead of middleware.

## Examples Overview

### [wasm-guest-auth](./wasm-guest-auth/)

API key authentication middleware that validates API keys for requests to `/api` and `/v1` paths.

**Features:**
- Validates API keys from `Authorization` header or `x-api-key` header
- Returns `401 Unauthorized` for missing or invalid keys
- Attach point: `OnRequest` only

**Use case:** Protect API endpoints with API key authentication.

### [wasm-guest-logging](./wasm-guest-logging/)

Request tracking and status code conversion middleware.

**Features:**
- Adds tracking headers (`x-request-id`, `x-wasm-processed`, `x-processed-at`, `x-api-route`)
- Converts `500` errors to `503` for better client handling
- Attach points: `OnRequest` and `OnResponse`

**Use case:** Request tracing and error status code conversion.

### [wasm-guest-ratelimit](./wasm-guest-ratelimit/)

Rate limiting middleware with configurable limits.

**Features:**
- Rate limiting per identifier (API Key, IP, or Request ID)
- Default: 60 requests per minute
- Returns `429 Too Many Requests` when limit exceeded
- Attach point: `OnRequest` only

**Note:** This is a simplified demonstration with per-instance state. For production, use router-level rate limiting with shared state.

**Use case:** Protect against request flooding and abuse.

### [wasm-guest-storage-hook](./wasm-guest-storage-hook/)

Storage hook demonstrating multi-tenancy and audit trail via extra columns.

**Features:**
- Extracts `tenant_id` from request context and writes it as an extra column
- Adds `CREATED_BY` / `STORED_BY` audit columns from `user_id` context
- Rejects `StoreResponse` when `tenant_id` is missing (validation example)
- Read/delete operations pass through unchanged

**Use case:** Multi-tenant storage isolation and audit logging without forking backend code.

### [wasm-guest-storage-hook-passthrough](./wasm-guest-storage-hook-passthrough/)

Minimal storage hook that never rejects. Used as an E2E test fixture.

**Features:**
- Always continues without rejection
- Adds `HOOK_ACTIVE=true` marker on write operations
- Passes through extra columns unchanged in `after` hook

**Use case:** Verifying the storage hook pipeline is active in integration/E2E tests.

### [wasm-guest-policy](./wasm-guest-policy/)

A routing policy for the policy controller's plugin host: a filter, a scorer and a picker in one component. Not gateway middleware: it targets a different host, with a different world and a different build target.

**Features:**
- Keeps the healthy workers that have capacity, scores the least loaded highest, with a bonus for the session's worker
- Reads its configuration in `init` and counts completed dispatches in `observe`
- Builds for `wasm32-unknown-unknown`, not `wasm32-wasip2`: the policy host refuses a component that imports any `wasi:` interface

**Use case:** A starting point for a custom routing policy.

## Quick Start

Each example includes its own README with detailed build and deployment instructions. See individual example directories for:

- Build instructions
- Deployment configuration
- Customization options
- Testing examples

## Common Prerequisites

The middleware and storage hook examples require (the policy example has its own target and needs no running gateway; see its README):

- Rust toolchain (latest stable)
- `wasm32-wasip2` target: `rustup target add wasm32-wasip2`
- `wasm-tools`: `cargo install wasm-tools`
- SMG running with WASM enabled (`--enable-wasm`)

**Storage hook examples** additionally require the storage hook WIT interface at `crates/wasm/src/interface/storage/`.

## Building All Examples

```bash
cd examples/wasm
for example in wasm-guest-auth wasm-guest-logging wasm-guest-ratelimit wasm-guest-storage-hook wasm-guest-storage-hook-passthrough; do
  echo "Building $example..."
  cd $example && ./build.sh && cd ..
done
```

## Deploying Multiple Modules

You can deploy all three modules together:

```bash
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [
      {
        "name": "auth-middleware",
        "file_path": "/path/to/wasm_guest_auth.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}]
      },
      {
        "name": "logging-middleware",
        "file_path": "/path/to/wasm_guest_logging.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}, {"Middleware": "OnResponse"}]
      },
      {
        "name": "ratelimit-middleware",
        "file_path": "/path/to/wasm_guest_ratelimit.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}]
      }
    ]
  }'
```

Modules execute in the order they are deployed. If a module returns `Reject`, subsequent modules won't execute.
