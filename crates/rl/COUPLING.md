# Coupling contract with model_gateway (M1)

The RL crate may touch the gateway only through the surfaces below. Any PR
that adds a surface must update this file.

| # | Surface | model_gateway file | Notes |
|---|---|---|---|
| (a) | `RlWorkerView` read-only registry view | `src/rl_adapter.rs` | `RegistryRlView` over `WorkerRegistry::{get_all,get,get_id_by_url}` |
| (d) | `AppContext.rl: Option<Arc<RlState>>` | `src/app_context.rs` | built in `AppContextBuilder::build()` when `router_config.rl.enabled` |
| (d) | route mount | `src/server.rs` `build_app` | `nest("/v1/rl", smg_rl::router(..))` under `apply_control_plane_auth` |
| (d) | metrics HELP registration | `src/observability/metrics.rs` | `smg_rl::init_rl_metrics()` |
| (d) | config + flags | `src/config/{types,builder,validation}.rs`, `src/main.rs`, `bindings/python/src/smg/router_args.py`, `bindings/python/src/lib.rs` | `RouterConfig.rl`, three CLI flags |

Test-only files that the mount also touches, none of them a new surface:
`model_gateway/tests/rl_control_plane_test.rs` (gateway-level `/v1/rl` tests),
`model_gateway/tests/common/mock_worker.rs` (engine-native RL routes on the
mock), and the three `#[cfg(test)]` `AppContext { .. }` literals in
`src/service_discovery.rs`, `src/workflow/steps/local/drain_workers.rs`, and
`src/workflow/steps/local/update_worker_properties.rs`, which gain `rl: None`
because the struct grew a field.

Not touched: policies, routers, worker trait, response pipeline (M2).
