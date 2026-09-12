//! RL control plane for SMG: worker discovery, verbatim passthrough of
//! engine-native RL routes, and label-selected fan-out. Compiled into the
//! gateway but inert unless `--enable-rl`.

pub mod capability;
pub mod config;
pub mod discovery;
pub mod error;
pub mod fanout;
pub mod metrics;
pub mod path;
pub mod proxy;
pub mod selector;
pub mod state;
#[cfg(test)]
pub(crate) mod testing;
pub mod view;

use std::sync::Arc;

use axum::{routing::get, Router};
pub use config::RlConfig;
pub use error::RlError;
pub use metrics::init_rl_metrics;
pub use state::RlState;
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
        .with_state(state)
}
