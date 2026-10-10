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
use tracing::{debug, warn};

use crate::config::RouterConfig;

/// Default pool settings for worker-directed HTTP clients.
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 500;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// HTTP/2 tuning for `--upstream-http2`, applied to every worker client under
/// the flag; only the prior-knowledge bucket forces h2 on cleartext.
///
/// Multiplex everything to a worker over one HTTP/2 connection. The default
/// 64KB flow-control windows would let concurrent token streams throttle each
/// other, so start large and let the adaptive window take over; h2 PING
/// keepalives replace idle-connection churn and detect dead peers under
/// long-lived streams.
fn tune_http2(builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    builder
        .http2_initial_stream_window_size(2 * 1024 * 1024)
        .http2_initial_connection_window_size(16 * 1024 * 1024)
        .http2_adaptive_window(true)
        .http2_keep_alive_interval(Duration::from_secs(30))
        .http2_keep_alive_timeout(Duration::from_secs(20))
        .http2_keep_alive_while_idle(true)
}

/// The `HttpPoolConfig` settings reqwest only honors client-wide, with router
/// defaults applied, plus the HTTP version spoken to the worker.
///
/// Router TLS identity/roots and the h2 tuning are also client-level but
/// process-constant, so they apply to every entry instead of keying it.
#[derive(Debug, PartialEq, Eq, Hash)]
struct ClientKey {
    connect_timeout_secs: u64,
    pool_max_idle_per_host: usize,
    /// `0` keeps idle connections indefinitely.
    pool_idle_timeout_secs: u64,
    /// Client default; a call site's `RequestBuilder::timeout` overrides it.
    timeout_secs: u64,
    /// HTTP/2 prior knowledge (h2c on cleartext, ALPN pinned to h2 on TLS).
    http2: bool,
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
    pool_idle_timeout_secs: u64,
    clients: Mutex<HashMap<ClientKey, Weak<reqwest::Client>>>,
}

impl WorkerHttpClientCache {
    pub fn new(router_config: &RouterConfig) -> Self {
        Self {
            client_identity: router_config.client_identity.clone(),
            ca_certificates: router_config.ca_certificates.clone(),
            upstream_http2: router_config.upstream_http2,
            request_timeout_secs: router_config.request_timeout_secs,
            pool_idle_timeout_secs: router_config.upstream_pool_idle_timeout_secs,
            clients: Mutex::new(HashMap::new()),
        }
    }

    /// The shared client for a worker's effective pool config and HTTP
    /// version, rebuilt when no live worker holds it anymore.
    pub fn get(
        &self,
        pool_config: &HttpPoolConfig,
        http2: bool,
    ) -> Result<Arc<reqwest::Client>, String> {
        let key = self.key(pool_config, http2);
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

    fn key(&self, pool_config: &HttpPoolConfig, http2: bool) -> ClientKey {
        ClientKey {
            connect_timeout_secs: pool_config
                .connect_timeout_secs
                .unwrap_or(DEFAULT_CONNECT_TIMEOUT_SECS),
            pool_max_idle_per_host: pool_config
                .pool_max_idle_per_host
                .unwrap_or(DEFAULT_POOL_MAX_IDLE_PER_HOST),
            pool_idle_timeout_secs: pool_config
                .pool_idle_timeout_secs
                .unwrap_or(self.pool_idle_timeout_secs),
            timeout_secs: pool_config
                .timeout_secs
                .unwrap_or(self.request_timeout_secs),
            http2,
        }
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

        // Idle pooled connections must expire before the backend server's
        // keep-alive closes them (vLLM/SGLang default: 5s), or checkout races
        // the server's FIN and non-idempotent sends fail.
        let pool_idle_timeout = match key.pool_idle_timeout_secs {
            0 => None,
            secs => Some(Duration::from_secs(secs)),
        };
        let identity = self
            .client_identity
            .as_deref()
            .map(reqwest::Identity::from_pem)
            .transpose()
            .map_err(|e| format!("Failed to create client identity: {e}"))?;
        let ca_certificates = self
            .ca_certificates
            .iter()
            .map(|pem| reqwest::Certificate::from_pem(pem))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to add CA certificate: {e}"))?;

        let make_builder = || {
            let mut builder = reqwest::Client::builder()
                .pool_max_idle_per_host(key.pool_max_idle_per_host)
                .pool_idle_timeout(pool_idle_timeout)
                .timeout(Duration::from_secs(key.timeout_secs))
                .connect_timeout(Duration::from_secs(key.connect_timeout_secs))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)));

            if self.upstream_http2 {
                builder = tune_http2(builder);
            }
            if key.http2 {
                builder = builder.http2_prior_knowledge();
            }

            if has_tls {
                builder = builder.use_rustls_tls();
            }
            if let Some(identity) = &identity {
                builder = builder.identity(identity.clone());
            }
            for cert in &ca_certificates {
                builder = builder.add_root_certificate(cert.clone());
            }
            builder
        };
        build_client(make_builder, has_tls, "worker HTTP client")
    }
}

/// Build a `reqwest` client, tolerating a host without a native CA root store
/// when the gateway has no TLS configuration of its own.
///
/// `reqwest`'s rustls backend loads the platform's root certificates while the
/// client is built and refuses to build when it finds none (a minimal
/// container image without the `ca-certificates` package, for example). A
/// gateway whose workers are plaintext does not need them: with no client
/// identity or CA bundle configured, the client is rebuilt without the
/// platform store (HTTPS upstreams then fail at the handshake with an unknown
/// issuer, which the warning explains) instead of refusing to start. With TLS
/// configured the error is returned with its full cause chain, which
/// `reqwest`'s top-level message ("builder error") omits.
pub(crate) fn build_client(
    make_builder: impl Fn() -> reqwest::ClientBuilder,
    tls_configured: bool,
    purpose: &str,
) -> Result<reqwest::Client, String> {
    let error = match make_builder().build() {
        Ok(client) => return Ok(client),
        Err(error) => error,
    };
    if tls_configured {
        return Err(format!("Failed to create {purpose}: {}", describe(&error)));
    }
    match make_builder().tls_certs_only(std::iter::empty()).build() {
        Ok(client) => {
            warn!(
                "Failed to create {purpose} with the platform's root certificates ({}); no TLS \
                 client identity or CA bundle is configured, so it runs without a root store: \
                 plaintext upstreams work, HTTPS upstreams fail until the ca-certificates \
                 package is installed or SSL_CERT_FILE/SSL_CERT_DIR names a PEM bundle",
                describe(&error)
            );
            Ok(client)
        }
        Err(retry_error) => Err(format!(
            "Failed to create {purpose}: {}",
            describe(&retry_error)
        )),
    }
}

/// `reqwest::Error` displays its kind alone ("builder error"); the cause is in
/// its source chain. Spell the chain out and, when it is the missing platform
/// root store, say what to do about it.
fn describe(error: &reqwest::Error) -> String {
    let mut text = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    if text.contains("No CA certificates were loaded") {
        text.push_str(
            " (no native CA root store: install the ca-certificates package or point \
             SSL_CERT_FILE/SSL_CERT_DIR at a PEM bundle)",
        );
    }
    text
}

/// Test support: run one `#[test]` of this binary in a child process whose
/// certificate environment names no root certificate at all. This only removes
/// native roots on platforms using the file-based verifier; Apple, Windows and
/// Android use their system trust APIs instead and ignore these variables.
#[cfg(test)]
pub(crate) mod no_root_store {
    use std::process::Command;

    /// Names the case a child-body test runs; unset in the parent process,
    /// where the child-body tests return at once.
    pub(crate) const CASE: &str = "SMG_TEST_NO_ROOT_STORE_CASE";

    /// The libtest name of `test` in `module` (a `module_path!()`, which
    /// carries the crate name libtest leaves out).
    pub(crate) fn test_name(module: &str, test: &str) -> String {
        let module = module.strip_prefix("smg::").unwrap_or(module);
        format!("{module}::{test}")
    }

    /// Run `test` in a child process with `SSL_CERT_FILE` naming an empty
    /// bundle and `SSL_CERT_DIR` an empty directory, so the platform
    /// verifier finds no root certificate, and assert that it passes.
    pub(crate) fn run(test: &str, case: &str) {
        let dir =
            std::env::temp_dir().join(format!("smg-no-root-store-{}-{case}", std::process::id()));
        let empty_dir = dir.join("certs");
        std::fs::create_dir_all(&empty_dir).expect("temp dir");
        let empty_bundle = dir.join("empty.pem");
        std::fs::write(&empty_bundle, b"").expect("empty bundle");

        let output = Command::new(std::env::current_exe().expect("test binary"))
            .args([test, "--exact", "--test-threads=1"])
            .env(CASE, case)
            .env("SSL_CERT_FILE", &empty_bundle)
            .env("SSL_CERT_DIR", &empty_dir)
            .output()
            .expect("run the child test");
        assert!(
            output.status.success(),
            "child test {test} ({case}) failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(config: RouterConfig) -> WorkerHttpClientCache {
        WorkerHttpClientCache::new(&config)
    }

    fn http2_cache() -> WorkerHttpClientCache {
        cache(RouterConfig {
            upstream_http2: true,
            ..RouterConfig::default()
        })
    }

    /// A throwaway self-signed certificate with its key, as a client identity.
    fn self_signed_identity_pem() -> Vec<u8> {
        use openssl::{
            asn1::Asn1Time,
            hash::MessageDigest,
            pkey::PKey,
            rsa::Rsa,
            x509::{X509NameBuilder, X509},
        };

        let key = PKey::from_rsa(Rsa::generate(2048).unwrap()).unwrap();
        let mut name = X509NameBuilder::new().unwrap();
        name.append_entry_by_text("CN", "test-client").unwrap();
        let name = name.build();
        let mut builder = X509::builder().unwrap();
        builder.set_version(2).unwrap();
        builder.set_subject_name(&name).unwrap();
        builder.set_issuer_name(&name).unwrap();
        builder.set_pubkey(&key).unwrap();
        builder
            .set_not_before(&Asn1Time::days_from_now(0).unwrap())
            .unwrap();
        builder
            .set_not_after(&Asn1Time::days_from_now(1).unwrap())
            .unwrap();
        builder.sign(&key, MessageDigest::sha256()).unwrap();
        let mut pem = builder.build().to_pem().unwrap();
        pem.extend(key.private_key_to_pem_pkcs8().unwrap());
        pem
    }

    /// Child-process body of the two root-store tests below (see
    /// `no_root_store`); a no-op unless the case variable is set.
    #[test]
    fn child_builds_without_native_roots() {
        let Ok(case) = std::env::var(no_root_store::CASE) else {
            return;
        };
        match case.as_str() {
            "plaintext" => {
                cache(RouterConfig::default())
                    .get(&HttpPoolConfig::default(), false)
                    .expect("a plaintext worker client needs no root store");
            }
            "tls" => {
                let error = cache(RouterConfig {
                    client_identity: Some(self_signed_identity_pem()),
                    ..RouterConfig::default()
                })
                .get(&HttpPoolConfig::default(), false)
                .expect_err("a client identity needs a root store");
                assert!(
                    error.contains("No CA certificates were loaded"),
                    "cause missing: {error}"
                );
                assert!(error.contains("SSL_CERT_FILE"), "hint missing: {error}");
            }
            other => panic!("unknown case {other}"),
        }
    }

    #[test]
    #[cfg_attr(
        any(target_vendor = "apple", windows, target_os = "android"),
        ignore = "the native verifier does not use SSL_CERT_FILE/SSL_CERT_DIR"
    )]
    fn plaintext_worker_clients_build_without_a_native_root_store() {
        no_root_store::run(
            &no_root_store::test_name(module_path!(), "child_builds_without_native_roots"),
            "plaintext",
        );
    }

    #[test]
    #[cfg_attr(
        any(target_vendor = "apple", windows, target_os = "android"),
        ignore = "the native verifier does not use SSL_CERT_FILE/SSL_CERT_DIR"
    )]
    fn a_missing_root_store_is_named_when_tls_is_configured() {
        no_root_store::run(
            &no_root_store::test_name(module_path!(), "child_builds_without_native_roots"),
            "tls",
        );
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
        let a = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        // Explicit values equal to the defaults are the same effective config.
        let b = cache
            .get(
                &HttpPoolConfig {
                    connect_timeout_secs: Some(DEFAULT_CONNECT_TIMEOUT_SECS),
                    ..Default::default()
                },
                false,
            )
            .expect("client");
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn client_level_override_gets_its_own_client() {
        let cache = cache(RouterConfig::default());
        let default = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let overridden = cache
            .get(
                &HttpPoolConfig {
                    connect_timeout_secs: Some(3),
                    ..Default::default()
                },
                false,
            )
            .expect("client");
        assert!(!Arc::ptr_eq(&default, &overridden));
        // The override bucket is itself cached.
        let again = cache
            .get(
                &HttpPoolConfig {
                    connect_timeout_secs: Some(3),
                    ..Default::default()
                },
                false,
            )
            .expect("client");
        assert!(Arc::ptr_eq(&overridden, &again));
    }

    #[test]
    fn pool_defaults_follow_router_upstream_settings() {
        let cache = cache(RouterConfig {
            upstream_pool_idle_timeout_secs: 7,
            request_timeout_secs: 90,
            ..RouterConfig::default()
        });
        let default = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let explicit = cache
            .get(
                &HttpPoolConfig {
                    pool_idle_timeout_secs: Some(7),
                    timeout_secs: Some(90),
                    pool_max_idle_per_host: Some(DEFAULT_POOL_MAX_IDLE_PER_HOST),
                    ..Default::default()
                },
                false,
            )
            .expect("client");
        assert!(Arc::ptr_eq(&default, &explicit));
        let other_timeout = cache
            .get(
                &HttpPoolConfig {
                    timeout_secs: Some(91),
                    ..Default::default()
                },
                false,
            )
            .expect("client");
        assert!(!Arc::ptr_eq(&default, &other_timeout));
    }

    #[test]
    fn http_version_buckets_are_distinct_clients() {
        let cache = http2_cache();
        let h1 = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let h2 = cache.get(&HttpPoolConfig::default(), true).expect("client");
        assert!(!Arc::ptr_eq(&h1, &h2));
        let h2_again = cache.get(&HttpPoolConfig::default(), true).expect("client");
        assert!(Arc::ptr_eq(&h2, &h2_again));
    }

    #[test]
    fn entry_lives_exactly_as_long_as_worker_handles() {
        use crate::worker::BasicWorkerBuilder;

        let cache = cache(RouterConfig::default());
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
        assert_eq!(cache.cached_len(), 1);
        drop(rebuilt);
    }

    #[test]
    fn dead_entries_are_pruned_on_insert() {
        let cache = cache(RouterConfig::default());
        let dropped = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        drop(dropped);

        let _live = cache
            .get(
                &HttpPoolConfig {
                    connect_timeout_secs: Some(3),
                    ..Default::default()
                },
                false,
            )
            .expect("client");
        assert_eq!(cache.cached_len(), 1);
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
        .get(&HttpPoolConfig::default(), false)
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
    async fn http2_bucket_speaks_h2c_prior_knowledge() {
        let url = spawn_echo_server().await;
        let client = http2_cache()
            .get(&HttpPoolConfig::default(), true)
            .expect("client");
        let resp = client.get(&url).send().await.expect("h2c request");
        assert_eq!(resp.version(), http::Version::HTTP_2);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }

    #[tokio::test]
    async fn http1_bucket_stays_http1_under_upstream_http2() {
        let url = spawn_echo_server().await;
        let client = http2_cache()
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let resp = client.get(&url).send().await.expect("h1 request");
        assert_eq!(resp.version(), http::Version::HTTP_11);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }

    #[tokio::test]
    async fn default_worker_client_stays_http1() {
        let url = spawn_echo_server().await;
        let client = cache(RouterConfig::default())
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let resp = client.get(&url).send().await.expect("h1 request");
        assert_eq!(resp.version(), http::Version::HTTP_11);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }
}
