//! RL control plane for SMG: worker discovery, verbatim passthrough of
//! engine-native RL routes, and label-selected fan-out. Compiled into the
//! gateway but inert unless `--enable-rl`.

pub mod capability;
pub mod config;
pub mod control;
pub mod discovery;
pub mod error;
pub mod fanout;
pub mod metrics;
pub mod observe;
pub mod path;
pub mod policy;
pub mod proxy;
pub mod selector;
pub mod stamp;
pub mod state;
pub mod table;
#[cfg(test)]
pub(crate) mod testing;
pub mod version;
pub mod view;

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};
pub use config::RlConfig;
pub use error::RlError;
pub use metrics::init_rl_metrics;
pub use policy::{VersionPolicy, VersionPolicyError, VERSION_POLICY_HEADER};
pub use state::RlState;
pub use table::{ControlState, FilterReason, RlTable, VersionEvictionSink, VersionSource};
pub use version::{Version, VersionError};
pub use view::{RlWorkerInfo, RlWorkerView};

/// Build the `/v1/rl` router. `with_state` returns `Router<S>` for any `S`,
/// so the gateway can nest this under its own state type.
pub fn router<S: Clone + Send + Sync + 'static>(state: Arc<RlState>) -> Router<S> {
    Router::new()
        .route("/workers", get(discovery::list_workers))
        .route("/workers/{id}", get(discovery::get_worker))
        .route(
            "/workers/{id}/engine/{*path}",
            get(proxy::proxy_handler).post(proxy::proxy_handler),
        )
        .route(
            "/engine/{*path}",
            get(fanout::fanout_handler).post(fanout::fanout_handler),
        )
        .route("/workers/{id}/version", post(control::set_worker_version))
        .route("/workers/{id}/state", post(control::set_worker_state))
        .route("/version", post(control::set_fleet_version))
        .route("/state", post(control::set_fleet_state))
        .with_state(state)
}
