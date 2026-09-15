# Coupling contract with model_gateway (M1, M2)

The RL crate may touch the gateway only through the surfaces below. Any PR
that adds a surface must update this file.

| # | Surface | model_gateway file | Notes |
|---|---|---|---|
| (a) | `RlWorkerView` read-only registry view | `src/rl_adapter.rs`, `src/lib.rs` | `RegistryRlView` over `WorkerRegistry::{get_all,get,get_id_by_url}`; hands the RL crate each HTTP worker's negotiated client through `Worker::{http_client_handle_if_initialized,http_client}`, the same client the gateway's admin ops use, so control calls inherit the worker's HTTP version, TLS identity and roots, and pool tuning; `lib.rs` gains `pub mod rl_adapter;` |
| (d) | `AppContext.rl: Option<Arc<RlState>>` | `src/app_context.rs` | built in `AppContextBuilder::build()` when `router_config.rl.enabled` |
| (d) | route mount | `src/server.rs` `build_app` | `nest("/v1/rl", smg_rl::router(..))` under `apply_control_plane_auth` |
| (d) | metrics HELP registration | `src/observability/metrics.rs` | `smg_rl::init_rl_metrics()` |
| (d) | config + flags | `src/config/{types,builder,validation}.rs`, `src/main.rs`, `bindings/python/src/smg/router_args.py`, `bindings/python/src/lib.rs` | `RouterConfig.rl`, three CLI flags |
| (d) | manifests | `model_gateway/Cargo.toml`, `bindings/python/Cargo.toml` | `smg-rl` pulled in via `workspace = true` (router crate and the pyo3 bindings crate) |
| (a) | `RlWorkerView::inflight(base_url)` and `VersionEvictionSink` | `src/rl_adapter.rs` | inflight sums `Worker::load()` over the ranks; `VersionEvictionSink::on_version_changed` runs under the RL table's write mutex, so the gateway's `RegistryEvictionSink` only resets policy caches and never writes back into the table; it holds `Weak<WorkerRegistry>`/`Weak<PolicyRegistry>` handles (a strong pair would close an ownership cycle through the candidate filter and leak both registries) and calls `PolicyRegistry::reset_worker_cache`, which covers the model policy, the PD/EPD leg policies, and the default-policy fallback |
| (a) | registry event subscription | `src/rl_adapter.rs` | seeds on `Registered`, reseeds on `Replaced` only when the discovered label changed, drops on the last `Removed` for a base URL; a `Replaced` that changes a worker's `model_id` without changing its label calls `seed`, which is insert-if-absent, so the table entry keeps the *old* `model_id` until the worker is removed and re-registered |
| (a) | `CacheAwarePolicy::reset_worker_cache`, `PolicyRegistry::reset_worker_cache` | `src/policies/cache_aware.rs`, `src/policies/registry.rs` | tree purge + root re-seed, no load-scorer effect |
| (b) | `CandidateFilter` trait and `PolicyRegistry::{candidate_filter, set_candidate_filter}` | `src/policies/mod.rs`, `src/policies/registry.rs`, `src/app_context.rs` | set in `AppContextBuilder::build()` when `rl` is `Some`; EPD per-item encode assignment and external-provider routing call their own selection paths and bypass the candidate filter by design. An `x-smg-target-worker` index names a worker in the caller's slice, never in the narrowed subset, so when the RL filter dropped the named worker the registry refuses the request (the retryable 503) instead of letting the policy re-read the index against the survivors |
| (c) | `RoutedWorker` + `stamp_routed_worker` | `crates/external_router/src/header_utils.rs`, seven HTTP sites, `routers/grpc/pipeline.rs` | header and extension in one call; the seven HTTP sites include the Responses-API streaming path in `crates/external_router/src/openai/responses/streaming.rs`; gRPC stamps the eight `pipeline.rs` dispatch sites, while the gRPC Responses-API helpers (typed, non-streaming returns) stay unstamped in M2 |
| (c) | RL middleware | `src/rl_adapter.rs`, `src/server.rs` `build_app` | header validation, version stamp, mixed flag; added inside `build_app`, so `tests/common/test_app.rs` inherits it automatically |
| (c) | `dispatch_metadata.rs` reads the table first | `src/routers/grpc/common/stages/dispatch_metadata.rs` (+ the pipeline holding `Option<Arc<RlState>>`) | label fallback unchanged |
| (c) | CORS expose list | `src/server.rs` `create_cors_layer` | three RL headers plus `x-request-id` |
| (d) | `--rl-version-policy` | `src/config/{types,validation}.rs`, `src/main.rs`, Python bindings | |

Test-only files that the mount also touches, none of them a new surface:
`model_gateway/tests/rl_control_plane_test.rs` (gateway-level `/v1/rl` tests),
`model_gateway/tests/common/mock_worker.rs` (engine-native RL routes on the
mock), and the three `#[cfg(test)]` `AppContext { .. }` literals in
`src/service_discovery.rs`, `src/workflow/steps/local/drain_workers.rs`, and
`src/workflow/steps/local/update_worker_properties.rs`, which gain `rl: None`
because the struct grew a field. The gateway-level test relies on
`TestRouterConfig` disabling health checks, so the mock stopped mid-test
stays registered and the fan-out still targets it.

Wire types are not a gateway coupling: they live in `crates/protocols/src/rl.rs`
(`openai_protocol::rl`) next to the `/workers` types, and
`clients/openapi-gen/src/main.rs` registers the `/v1/rl/*` paths.

## Why the proxy is separate

`crates/rl/src/proxy.rs` re-implements header selection and a bounded body
read rather than calling the data-plane proxy in `routers/http/router.rs`.
The mechanism overlaps; the policy is deliberately different and is what
makes this a control plane:

| Concern | Data-plane proxy | RL proxy |
|---|---|---|
| Retry | `RetryExecutor` per router config | none: a refit is not idempotent |
| Streaming | SSE relayed through a bounded channel | none: control routes answer once |
| Breaker and load accounting | `WorkerLoadGuard`, breaker outcome recorded | none: an engine that is paused or refitting must not trip inference routing |
| Over-cap body | reject with 502 `upstream_response_too_large` | keep the first 1 MiB and flag `body_truncated` |
| Deadline | `request_timeout_secs` on the worker client | `--rl-control-timeout-secs` applied per request on the same client |
| Failure of one target | one request, one status | 207 with every outcome and `failed[]` |
| Forwarded request headers | the router allow-list | `x-request-id`, `traceparent`, `tracestate` only; caller `authorization` never forwarded |

The workspace already compiles with `lto = "fat"` and `codegen-units = 1`,
so the duplicated loop costs nothing at runtime. The shared mechanism (the
header allow-list and one bounded reader for the four readers now in the
tree) is tracked in #2489.

Not touched: the worker trait, the health machine, `WorkerStatus::Draining`;
EPD per-item encode assignment and external-provider routing bypass the
candidate filter by design.
