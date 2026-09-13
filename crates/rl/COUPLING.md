# Coupling contract with model_gateway (M1)

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

Not touched: policies, routers, worker trait, response pipeline (M2).
