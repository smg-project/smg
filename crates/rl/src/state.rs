//! Process-wide state for the RL control plane: the registry view, the
//! control-plane HTTP client, and the configuration.

use std::{sync::Arc, time::Duration};

use crate::{config::RlConfig, error::RlError, view::RlWorkerView};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const POOL_MAX_IDLE_PER_HOST: usize = 8;
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);

pub struct RlState {
    pub(crate) view: Arc<dyn RlWorkerView>,
    #[expect(dead_code, reason = "used by Task 7's proxy handler")]
    pub(crate) client: reqwest::Client,
    #[expect(dead_code, reason = "used by Task 8's fan-out handler")]
    pub(crate) config: RlConfig,
}

impl RlState {
    /// Build the state and its dedicated control-plane client. The client is
    /// separate from the gateway's data-plane and worker clients on purpose:
    /// a minutes-long refit must not share a pool, breaker, or counter with
    /// inference traffic.
    pub fn new(
        view: Arc<dyn RlWorkerView>,
        config: RlConfig,
        upstream_http2: bool,
    ) -> Result<Self, RlError> {
        let mut builder = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.control_timeout_secs))
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(POOL_MAX_IDLE_PER_HOST)
            .tcp_nodelay(true)
            .tcp_keepalive(Some(TCP_KEEPALIVE));
        if upstream_http2 {
            builder = builder.http2_prior_knowledge();
        }
        let client = builder
            .build()
            .map_err(|e| RlError::Client(e.to_string()))?;
        Ok(Self {
            view,
            client,
            config,
        })
    }
}
