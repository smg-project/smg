//! Shared worker-directed HTTP clients.
//!
//! One `reqwest::Client` per distinct effective connection config instead of
//! one per worker: a uniform fleet shares a single connector, pool, and DNS
//! cache regardless of worker count.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError, Weak},
    time::Duration,
};

use openai_protocol::worker::HttpPoolConfig;
use tracing::debug;

use crate::config::RouterConfig;

/// Default pool settings for worker-directed HTTP clients.
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 8;
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 50;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// HTTP/2 prior-knowledge tuning for `--upstream-http2`, shared by the
/// dispatch client and the worker client cache.
///
/// Multiplex everything to a worker over one HTTP/2 connection. The default
/// 64KB flow-control windows would let concurrent token streams throttle each
/// other, so start large and let the adaptive window take over; h2 PING
/// keepalives replace idle-connection churn and detect dead peers under
/// long-lived streams.
pub(crate) fn apply_upstream_http2(builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    builder
        .http2_prior_knowledge()
        .http2_initial_stream_window_size(2 * 1024 * 1024)
        .http2_initial_connection_window_size(16 * 1024 * 1024)
        .http2_adaptive_window(true)
        .http2_keep_alive_interval(Duration::from_secs(30))
        .http2_keep_alive_timeout(Duration::from_secs(20))
        .http2_keep_alive_while_idle(true)
}

/// The `HttpPoolConfig` settings reqwest only honors client-wide, with
/// defaults applied.
///
/// Client-level (buckets the cache): connect timeout, pool sizing, pool idle
/// timeout. Router TLS identity/roots and the h2c mode are also client-level
/// but process-constant, so they apply to every entry instead of keying it.
/// Request-level (never buckets): total timeout — every worker-client call
/// site sets `RequestBuilder::timeout`, which overrides the client default.
#[derive(Debug, PartialEq, Eq, Hash)]
struct ClientKey {
    connect_timeout_secs: u64,
    pool_max_idle_per_host: usize,
    pool_idle_timeout_secs: u64,
}

impl ClientKey {
    fn from_pool_config(pool_config: &HttpPoolConfig) -> Self {
        Self {
            connect_timeout_secs: pool_config
                .connect_timeout_secs
                .unwrap_or(DEFAULT_CONNECT_TIMEOUT_SECS),
            pool_max_idle_per_host: pool_config
                .pool_max_idle_per_host
                .unwrap_or(DEFAULT_POOL_MAX_IDLE_PER_HOST),
            pool_idle_timeout_secs: pool_config
                .pool_idle_timeout_secs
                .unwrap_or(DEFAULT_POOL_IDLE_TIMEOUT_SECS),
        }
    }
}

/// Cache of worker-directed HTTP clients, keyed by effective client-level
/// config. Entries are weak: pool settings can be worker-supplied, so a
/// client lives exactly as long as some worker holds its handle, dead entries
/// are pruned on the next build, and cardinality is bounded by live distinct
/// configs — a churning fleet cannot grow the map.
pub struct WorkerHttpClientCache {
    client_identity: Option<Vec<u8>>,
    ca_certificates: Vec<Vec<u8>>,
    upstream_http2: bool,
    request_timeout_secs: u64,
    clients: Mutex<HashMap<ClientKey, Weak<reqwest::Client>>>,
}

impl WorkerHttpClientCache {
    pub fn new(router_config: &RouterConfig) -> Self {
        Self {
            client_identity: router_config.client_identity.clone(),
            ca_certificates: router_config.ca_certificates.clone(),
            upstream_http2: router_config.upstream_http2,
            request_timeout_secs: router_config.request_timeout_secs,
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// The shared client for a worker's effective pool config, rebuilt when
    /// no live worker holds it anymore.
    pub fn get(&self, pool_config: &HttpPoolConfig) -> Result<Arc<reqwest::Client>, String> {
        let key = ClientKey::from_pool_config(pool_config);
        let mut clients = self.clients.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(client) = clients.get(&key).and_then(Weak::upgrade) {
            return Ok(client);
        }
        let client = Arc::new(self.build(&key)?);
        clients.retain(|_, entry| entry.strong_count() > 0);
        debug!(?key, "built shared worker HTTP client");
        clients.insert(key, Arc::downgrade(&client));
        Ok(client)
    }

    /// Live + not-yet-pruned dead entries (test observation of bounds).
    #[cfg(test)]
    fn cached_len(&self) -> usize {
        self.clients
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    fn build(&self, key: &ClientKey) -> Result<reqwest::Client, String> {
        let has_tls = self.client_identity.is_some() || !self.ca_certificates.is_empty();

        let mut builder = reqwest::Client::builder()
            .pool_max_idle_per_host(key.pool_max_idle_per_host)
            .pool_idle_timeout(Some(Duration::from_secs(key.pool_idle_timeout_secs)))
            .timeout(Duration::from_secs(self.request_timeout_secs))
            .connect_timeout(Duration::from_secs(key.connect_timeout_secs))
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)));

        if self.upstream_http2 {
            builder = apply_upstream_http2(builder);
        }

        if has_tls {
            builder = builder.use_rustls_tls();
        }

        if let Some(identity_pem) = &self.client_identity {
            let identity = reqwest::Identity::from_pem(identity_pem)
                .map_err(|e| format!("Failed to create client identity: {e}"))?;
            builder = builder.identity(identity);
        }

        for ca_cert in &self.ca_certificates {
            let cert = reqwest::Certificate::from_pem(ca_cert)
                .map_err(|e| format!("Failed to add CA certificate: {e}"))?;
            builder = builder.add_root_certificate(cert);
        }

        builder
            .build()
            .map_err(|e| format!("Failed to create worker HTTP client: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(config: RouterConfig) -> WorkerHttpClientCache {
        WorkerHttpClientCache::new(&config)
    }

    /// Loopback echo server; axum::serve accepts HTTP/1.1 and prior-knowledge
    /// h2c on the same listener, mirroring a dual-protocol engine.
    async fn spawn_echo_server() -> String {
        let app = axum::Router::new()
            .route("/probe", axum::routing::get(|| async { "ok" }))
            .route(
                "/hang",
                axum::routing::get(|| async {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    "late"
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind echo server");
        let addr = listener.local_addr().expect("echo server address");
        #[expect(
            clippy::disallowed_methods,
            reason = "test server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("echo serve");
        });
        format!("http://{addr}/probe")
    }

    #[test]
    fn same_effective_config_shares_one_client() {
        let cache = cache(RouterConfig::default());
        let a = cache.get(&HttpPoolConfig::default()).expect("client");
        // Explicit values equal to the defaults are the same effective config.
        let b = cache
            .get(&HttpPoolConfig {
                connect_timeout_secs: Some(DEFAULT_CONNECT_TIMEOUT_SECS),
                ..Default::default()
            })
            .expect("client");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn client_level_override_gets_its_own_client() {
        let cache = cache(RouterConfig::default());
        let default = cache.get(&HttpPoolConfig::default()).expect("client");
        let overridden = cache
            .get(&HttpPoolConfig {
                connect_timeout_secs: Some(3),
                ..Default::default()
            })
            .expect("client");
        assert!(!Arc::ptr_eq(&default, &overridden));
        // The override bucket is itself cached.
        let again = cache
            .get(&HttpPoolConfig {
                connect_timeout_secs: Some(3),
                ..Default::default()
            })
            .expect("client");
        assert!(Arc::ptr_eq(&overridden, &again));
    }

    #[test]
    fn entry_lives_exactly_as_long_as_worker_handles() {
        use crate::worker::BasicWorkerBuilder;

        let cache = cache(RouterConfig::default());
        let handle = cache.get(&HttpPoolConfig::default()).expect("client");
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
        let rebuilt = cache.get(&HttpPoolConfig::default()).expect("client");
        assert!(probe.upgrade().is_none());
        assert_eq!(cache.cached_len(), 1);
        drop(rebuilt);
    }

    #[test]
    fn dead_entries_are_pruned_on_insert() {
        let cache = cache(RouterConfig::default());
        let dropped = cache.get(&HttpPoolConfig::default()).expect("client");
        drop(dropped);

        let _live = cache
            .get(&HttpPoolConfig {
                connect_timeout_secs: Some(3),
                ..Default::default()
            })
            .expect("client");
        assert_eq!(cache.cached_len(), 1);
    }

    #[test]
    fn request_level_timeout_override_never_buckets() {
        let cache = cache(RouterConfig::default());
        let a = cache.get(&HttpPoolConfig::default()).expect("client");
        let b = cache
            .get(&HttpPoolConfig {
                timeout_secs: Some(600),
                ..Default::default()
            })
            .expect("client");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn per_request_timeout_overrides_client_default() {
        let url = spawn_echo_server().await;
        // A zero client-level total timeout fails every request that relies
        // on the client default. Aim it at the hanging route: a fast local
        // response can otherwise win the race against a zero-duration timer.
        let hang_url = url.replace("/probe", "/hang");
        let client = cache(RouterConfig {
            request_timeout_secs: 0,
            ..RouterConfig::default()
        })
        .get(&HttpPoolConfig::default())
        .expect("client");
        let err = client
            .get(&hang_url)
            .send()
            .await
            .expect_err("default applies");
        assert!(err.is_timeout());
        // ...while a per-request timeout replaces it entirely.
        let resp = client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .expect("per-request timeout wins");
        assert_eq!(resp.text().await.expect("body"), "ok");
    }

    #[tokio::test]
    async fn upstream_http2_worker_client_speaks_h2c_prior_knowledge() {
        let url = spawn_echo_server().await;
        let client = cache(RouterConfig {
            upstream_http2: true,
            ..RouterConfig::default()
        })
        .get(&HttpPoolConfig::default())
        .expect("client");
        let resp = client.get(&url).send().await.expect("h2c request");
        assert_eq!(resp.version(), http::Version::HTTP_2);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }

    #[tokio::test]
    async fn default_worker_client_stays_http1() {
        let url = spawn_echo_server().await;
        let client = cache(RouterConfig::default())
            .get(&HttpPoolConfig::default())
            .expect("client");
        let resp = client.get(&url).send().await.expect("h1 request");
        assert_eq!(resp.version(), http::Version::HTTP_11);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }
}
