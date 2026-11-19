//! Shared worker-directed HTTP clients: the worker crate's, at the paths they
//! had here.

#[cfg(test)]
pub(crate) use smg_worker::connection::http_client::no_root_store;
pub(crate) use smg_worker::connection::http_client::{
    build_client, build_with, connect_timeout_secs_below, DEFAULT_CONNECT_TIMEOUT_SECS,
};
pub use smg_worker::connection::http_client::{
    LazyHttpClient, WorkerHttpClientCache, WorkerHttpClientSettings,
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::HttpPoolConfig;

    use super::*;
    use crate::{config::RouterConfig, worker::BasicWorkerBuilder};

    #[test]
    fn entry_lives_exactly_as_long_as_worker_handles() {
        let cache =
            WorkerHttpClientCache::new(RouterConfig::default().worker_http_client_settings());
        let handle = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let probe = Arc::downgrade(&handle);
        let worker_a = BasicWorkerBuilder::new("http://a:1")
            .http_client(handle.clone())
            .build();
        let worker_b = BasicWorkerBuilder::new("http://b:1")
            .http_client(handle)
            .build();

        drop(worker_a);
        assert!(
            probe.upgrade().is_some(),
            "entry survives while another worker holds it"
        );

        drop(worker_b);
        assert!(
            probe.upgrade().is_none(),
            "last worker drop releases the entry"
        );

        // The dead entry cannot be upgraded, so the next get rebuilds.
        let rebuilt = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        assert!(probe.upgrade().is_none());
        drop(rebuilt);
    }
}
