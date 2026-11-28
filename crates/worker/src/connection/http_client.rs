//! Shared worker-directed HTTP clients.
//!
//! One `reqwest::Client` per distinct effective connection config instead of
//! one per worker: a uniform fleet shares a single connector, pool, and DNS
//! cache regardless of worker count.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, PoisonError, Weak},
    time::Duration,
};

use tracing::{debug, warn};

use crate::{
    connection::client_factory::{http_client, http_client_builder},
    worker::HttpPoolConfig,
};

/// Default pool settings for worker-directed HTTP clients.
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 500;
/// The connect timeout every gateway HTTP client starts from (the worker
/// clients and the shared external-provider client), clamped below the
/// request timeout by [`connect_timeout_secs_below`].
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// The connect timeout a client may use under `request_timeout_secs`: the
/// requested `connect_timeout_secs`, clamped to one second below the request
/// timeout. A total timeout that runs out while the connect is still pending
/// reads as the request's own (`retry::is_request_timeout`: terminal for the
/// retry layer), so a host that has gone dark must fail as a connect timeout
/// first, which the retry layer routes around; with the defaults (10 s
/// against 1800 s) nothing changes. Whole seconds cannot go below one: a
/// request timeout of 1 s (or 0) keeps a 1 s connect timeout, and the two
/// timers race there. Every client the gateway builds with a request timeout
/// takes its connect timeout from here (the worker clients through
/// [`WorkerHttpClientCache`], the shared client for external providers in
/// the gateway's `AppContext::with_client`), so the two cannot drift.
pub const fn connect_timeout_secs_below(
    connect_timeout_secs: u64,
    request_timeout_secs: u64,
) -> u64 {
    let below = request_timeout_secs.saturating_sub(1);
    let floor = if below < 1 { 1 } else { below };
    if connect_timeout_secs < floor {
        connect_timeout_secs
    } else {
        floor
    }
}

/// Longest idle timeout a pool is given: thirty years, which no process
/// outlives. The pool's idle sweeper adds the timeout to the clock, and a sum
/// the clock cannot hold panics the sweeper's task.
const MAX_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 365 * 24 * 60 * 60);

/// The idle timeout a pool is built with for a configured number of seconds:
/// zero keeps idle connections indefinitely, and a timeout longer than thirty
/// years is thirty years. Every client the gateway builds with an idle
/// timeout takes it from here (the worker clients through
/// [`WorkerHttpClientCache`], the shared client for external providers in
/// the gateway's `AppContext::with_client`), so the bound stands in one place.
pub fn pool_idle_timeout(secs: u64) -> Option<Duration> {
    (secs != 0).then(|| Duration::from_secs(secs).min(MAX_POOL_IDLE_TIMEOUT))
}

/// HTTP/2 tuning for the gateway's `--upstream-http2`, applied to every worker
/// client under the flag; only the prior-knowledge bucket forces h2 on
/// cleartext.
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

/// The `HttpPoolConfig` settings reqwest only honors client-wide, with the
/// host's defaults applied, plus the HTTP version spoken to the worker.
///
/// The host's TLS identity/roots and the h2 tuning are also client-level but
/// process-constant, so they apply to every entry instead of keying it.
#[derive(Debug, PartialEq, Eq, Hash)]
struct ClientKey {
    /// Below `timeout_secs` where whole seconds allow it
    /// ([`connect_timeout_secs_below`]): the connect phase times out as a
    /// connect failure, not as the request's own timeout, except under a 1 s
    /// request timeout, where the two timers race.
    connect_timeout_secs: u64,
    pool_max_idle_per_host: usize,
    /// `0` keeps idle connections indefinitely.
    pool_idle_timeout_secs: u64,
    /// Client default; a call site's `RequestBuilder::timeout` overrides it.
    timeout_secs: u64,
    /// HTTP/2 prior knowledge (h2c on cleartext, ALPN pinned to h2 on TLS).
    http2: bool,
}

/// What the host fixes for every worker-directed client, filled from its own
/// configuration. A worker's pool settings may replace the two timeouts.
pub struct WorkerHttpClientSettings {
    /// PEM client certificate and key presented to workers, when set.
    pub client_identity: Option<Vec<u8>>,
    /// PEM CA certificates trusted for worker connections.
    pub ca_certificates: Vec<Vec<u8>>,
    /// Whether every client is tuned for HTTP/2 upstreams.
    pub upstream_http2: bool,
    /// Total request timeout of a client, in seconds.
    pub request_timeout_secs: u64,
    /// Idle timeout of a pooled connection, in seconds; zero keeps idle
    /// connections indefinitely.
    pub pool_idle_timeout_secs: u64,
}

/// Cache of worker-directed HTTP clients, keyed by effective client-level
/// config. Entries are weak: pool settings can be worker-supplied, so a
/// client lives exactly as long as some worker holds its handle, dead entries
/// are pruned on the next build, and cardinality is bounded by live distinct
/// configs — a churning fleet cannot grow the map.
pub struct WorkerHttpClientCache {
    settings: WorkerHttpClientSettings,
    clients: Mutex<HashMap<ClientKey, Weak<reqwest::Client>>>,
}

impl WorkerHttpClientCache {
    pub fn new(settings: WorkerHttpClientSettings) -> Self {
        Self {
            settings,
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
        let timeout_secs = pool_config
            .timeout_secs
            .unwrap_or(self.settings.request_timeout_secs);
        let connect_timeout_secs = connect_timeout_secs_below(
            pool_config
                .connect_timeout_secs
                .unwrap_or(DEFAULT_CONNECT_TIMEOUT_SECS),
            timeout_secs,
        );
        ClientKey {
            connect_timeout_secs,
            pool_max_idle_per_host: pool_config
                .pool_max_idle_per_host
                .unwrap_or(DEFAULT_POOL_MAX_IDLE_PER_HOST),
            pool_idle_timeout_secs: pool_config
                .pool_idle_timeout_secs
                .unwrap_or(self.settings.pool_idle_timeout_secs),
            timeout_secs,
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
        let has_tls =
            self.settings.client_identity.is_some() || !self.settings.ca_certificates.is_empty();

        // Idle pooled connections must expire before the backend server's
        // keep-alive closes them (a common engine default: 5s), or checkout races
        // the server's FIN and non-idempotent sends fail.
        let pool_idle_timeout = pool_idle_timeout(key.pool_idle_timeout_secs);
        let identity = self
            .settings
            .client_identity
            .as_deref()
            .map(reqwest::Identity::from_pem)
            .transpose()
            .map_err(|e| format!("Failed to create client identity: {e}"))?;
        let ca_certificates = self
            .settings
            .ca_certificates
            .iter()
            .map(|pem| reqwest::Certificate::from_pem(pem))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to add CA certificate: {e}"))?;

        let make_builder = || {
            let mut builder = http_client_builder()
                .pool_max_idle_per_host(key.pool_max_idle_per_host)
                .pool_idle_timeout(pool_idle_timeout)
                .timeout(Duration::from_secs(key.timeout_secs))
                .connect_timeout(Duration::from_secs(key.connect_timeout_secs))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)));

            if self.settings.upstream_http2 {
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
pub fn build_client(
    make_builder: impl Fn() -> reqwest::ClientBuilder,
    tls_configured: bool,
    purpose: &str,
) -> Result<reqwest::Client, String> {
    build_with(
        make_builder,
        tls_configured,
        purpose,
        reqwest::ClientBuilder::build,
    )
}

/// [`build_client`] for a client that `build` finishes from the builder (a
/// library that completes the client it fetches with), with the same
/// tolerance of a host without a root store: `build` is tried on the
/// builder, then, with no TLS configured, on the builder without the
/// platform store. An error of `build`'s own (not the client's) fails the
/// same way both times and is reported as it is.
pub fn build_with<T, E: std::error::Error + 'static>(
    make_builder: impl Fn() -> reqwest::ClientBuilder,
    tls_configured: bool,
    purpose: &str,
    build: impl Fn(reqwest::ClientBuilder) -> Result<T, E>,
) -> Result<T, String> {
    let error = match build(make_builder()) {
        Ok(client) => return Ok(client),
        Err(error) => error,
    };
    if tls_configured {
        return Err(format!("Failed to create {purpose}: {}", describe(&error)));
    }
    match build(make_builder().tls_certs_only(std::iter::empty())) {
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
fn describe(error: &(dyn std::error::Error + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
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

/// A worker's HTTP client handle, materialized on first use.
///
/// Registration hands in a shared client from the worker client cache. A
/// worker built without one whose connection mode never speaks HTTP (ZMQ:
/// local health check, admin ops rejected up front) would otherwise pay for a
/// connector and idle pool it can never use, so the fallback client is built
/// only when a caller actually asks for it.
pub struct LazyHttpClient {
    cell: OnceLock<Arc<reqwest::Client>>,
}

impl LazyHttpClient {
    /// Wrap an already-built client (the registration paths hand one in).
    /// The strong handle is what keeps the client's cache entry alive.
    pub fn ready(client: Arc<reqwest::Client>) -> Self {
        let cell = OnceLock::new();
        let _ = cell.set(client);
        Self { cell }
    }

    /// Defer construction until [`Self::client`] is first called.
    pub fn deferred() -> Self {
        Self {
            cell: OnceLock::new(),
        }
    }

    /// Whether the client is still unbuilt (test-only observation of laziness).
    #[cfg(any(test, feature = "test-util"))]
    pub fn cell_is_empty(&self) -> bool {
        self.cell.get().is_none()
    }

    /// The client, building the default one on first use. The fallback sets
    /// no total timeout: dispatch deadlines come from the host's settings via
    /// the client cache, and every health/admin call site sets its own.
    pub fn client(&self) -> &reqwest::Client {
        self.init()
    }

    /// A strong handle to the client if one was materialized, for a
    /// replacement worker to adopt. Never forces the lazy cell: a worker
    /// that never spoke HTTP hands its replacement a still-deferred slot.
    pub fn handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
        self.cell.get().map(Arc::clone)
    }

    fn init(&self) -> &Arc<reqwest::Client> {
        self.cell.get_or_init(|| Arc::new(http_client()))
    }
}

/// Test support: run one `#[test]` of this binary in a child process whose
/// certificate environment names no root certificate at all. This only removes
/// native roots on platforms using the file-based verifier; Apple, Windows and
/// Android use their system trust APIs instead and ignore these variables.
#[cfg(any(test, feature = "test-util"))]
pub mod no_root_store {
    use std::process::Command;

    /// Names the case a child-body test runs; unset in the parent process,
    /// where the child-body tests return at once.
    pub const CASE: &str = "SMG_TEST_NO_ROOT_STORE_CASE";

    /// The libtest name of `test` in `module` (a `module_path!()`, which
    /// carries the crate name libtest leaves out).
    pub fn test_name(module: &str, test: &str) -> String {
        let module = module.split_once("::").map_or(module, |(_, rest)| rest);
        format!("{module}::{test}")
    }

    /// Run `test` in a child process with `SSL_CERT_FILE` naming an empty
    /// bundle and `SSL_CERT_DIR` an empty directory, so the platform
    /// verifier finds no root certificate, and assert that it ran and passed.
    #[expect(
        clippy::expect_used,
        reason = "test support: a setup that fails must fail the test that called it"
    )]
    pub fn run(test: &str, case: &str) {
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
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child test {test} ({case}) failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        // A name that matches no test also exits with success: the child
        // must have run the one case it was sent for.
        assert!(
            stdout.contains("test result: ok. 1 passed;"),
            "child test {test} ({case}) ran no test:\n{stdout}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host's settings with no TLS, HTTP/1 and the gateway's default
    /// timeouts; a case replaces the fields it is about.
    fn settings() -> WorkerHttpClientSettings {
        WorkerHttpClientSettings {
            client_identity: None,
            ca_certificates: Vec::new(),
            upstream_http2: false,
            request_timeout_secs: 1800,
            pool_idle_timeout_secs: 3,
        }
    }

    fn cache(settings: WorkerHttpClientSettings) -> WorkerHttpClientCache {
        WorkerHttpClientCache::new(settings)
    }

    fn http2_cache() -> WorkerHttpClientCache {
        cache(WorkerHttpClientSettings {
            upstream_http2: true,
            ..settings()
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
                cache(settings())
                    .get(&HttpPoolConfig::default(), false)
                    .expect("a plaintext worker client needs no root store");
            }
            "tls" => {
                let error = cache(WorkerHttpClientSettings {
                    client_identity: Some(self_signed_identity_pem()),
                    ..settings()
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

    /// The connect timeout never reaches the request timeout where whole
    /// seconds allow it: a request timeout of 2 s clamps the default 10 s
    /// connect timeout to 1 s, a worker's own 30 s request timeout leaves it
    /// at 10 s, and an explicit connect timeout above the request timeout is
    /// clamped the same way, so a host that has gone dark fails as a connect
    /// timeout (routed around) before the request's own timeout (terminal)
    /// can fire; a 1 s request timeout keeps the 1 s floor (the two timers
    /// race there), as the helper's doc says.
    #[test]
    fn connect_timeout_stays_below_the_request_timeout() {
        assert_eq!(connect_timeout_secs_below(10, 1800), 10);
        assert_eq!(connect_timeout_secs_below(10, 2), 1);
        assert_eq!(connect_timeout_secs_below(3, 2), 1);
        assert_eq!(connect_timeout_secs_below(10, 1), 1, "the floor");
        assert_eq!(connect_timeout_secs_below(10, 0), 1, "the floor");
        assert_eq!(connect_timeout_secs_below(10, 11), 10);
        assert_eq!(connect_timeout_secs_below(10, 10), 9);
        let floor = cache(WorkerHttpClientSettings {
            request_timeout_secs: 1,
            ..settings()
        });
        assert_eq!(
            floor
                .key(
                    &HttpPoolConfig {
                        timeout_secs: Some(1),
                        ..Default::default()
                    },
                    false
                )
                .connect_timeout_secs,
            1
        );
        let short = cache(WorkerHttpClientSettings {
            request_timeout_secs: 2,
            ..settings()
        });
        assert_eq!(
            short
                .key(&HttpPoolConfig::default(), false)
                .connect_timeout_secs,
            1
        );
        assert_eq!(
            short
                .key(
                    &HttpPoolConfig {
                        timeout_secs: Some(30),
                        ..Default::default()
                    },
                    false
                )
                .connect_timeout_secs,
            DEFAULT_CONNECT_TIMEOUT_SECS
        );
        assert_eq!(
            short
                .key(
                    &HttpPoolConfig {
                        connect_timeout_secs: Some(3),
                        ..Default::default()
                    },
                    false
                )
                .connect_timeout_secs,
            1
        );
        let long = cache(WorkerHttpClientSettings {
            request_timeout_secs: 1800,
            ..settings()
        });
        let key = long.key(&HttpPoolConfig::default(), false);
        assert_eq!(
            (key.connect_timeout_secs, key.timeout_secs),
            (DEFAULT_CONNECT_TIMEOUT_SECS, 1800)
        );
    }

    #[test]
    fn pool_idle_timeout_keeps_zero_unbounded_and_caps_the_rest() {
        assert_eq!(pool_idle_timeout(0), None, "zero keeps idle connections");
        assert_eq!(pool_idle_timeout(5), Some(Duration::from_secs(5)));
        assert_eq!(pool_idle_timeout(u64::MAX), Some(MAX_POOL_IDLE_TIMEOUT));
    }

    #[test]
    fn same_effective_config_shares_one_client() {
        let cache = cache(settings());
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
        let cache = cache(settings());
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
        let cache = cache(WorkerHttpClientSettings {
            pool_idle_timeout_secs: 7,
            request_timeout_secs: 90,
            ..settings()
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
        let cache = cache(settings());
        let handle = cache
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let probe = Arc::downgrade(&handle);
        let worker_a = LazyHttpClient::ready(handle.clone());
        let worker_b = LazyHttpClient::ready(handle);

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
        let cache = cache(settings());
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

    /// The pool's idle sweeper is a task the client starts when its first
    /// connection goes back to the pool, and it adds the idle timeout to the
    /// clock. The largest number of seconds a worker's pool settings can
    /// carry is capped where the client is built, so the sum holds and the
    /// sweeper runs; before the cap the sweeper panicked on it. The runtime
    /// catches a task's panic, so the test counts panics in the panic hook: a
    /// current-thread runtime polls the sweeper on this thread.
    #[tokio::test]
    async fn the_largest_idle_timeout_leaves_the_pools_sweeper_running() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static PANICS_ON_THIS_THREAD: AtomicUsize = AtomicUsize::new(0);
        let url = spawn_echo_server().await;
        let this_thread = std::thread::current().id();
        let previous = Arc::new(std::panic::take_hook());
        let forward = Arc::clone(&previous);
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().id() == this_thread {
                PANICS_ON_THIS_THREAD.fetch_add(1, Ordering::SeqCst);
            }
            forward(info);
        }));

        let client = cache(settings())
            .get(
                &HttpPoolConfig {
                    pool_idle_timeout_secs: Some(u64::MAX),
                    ..HttpPoolConfig::default()
                },
                false,
            )
            .expect("client");
        let resp = client.get(&url).send().await.expect("response");
        assert_eq!(resp.text().await.expect("body"), "ok");
        // The connection is idle now: give the sweeper it started its first poll.
        tokio::time::sleep(Duration::from_millis(100)).await;

        std::panic::set_hook(Box::new(move |info| previous(info)));
        assert_eq!(
            PANICS_ON_THIS_THREAD.load(Ordering::SeqCst),
            0,
            "a task of this client panicked"
        );
    }

    #[tokio::test]
    async fn per_request_timeout_overrides_client_default() {
        let url = spawn_echo_server().await;
        // A zero client-level total timeout fails every request that relies
        // on the client default. Aim it at the hanging route: a fast local
        // response can otherwise win the race against a zero-duration timer.
        let hang_url = url.replace("/probe", "/hang");
        let client = cache(WorkerHttpClientSettings {
            request_timeout_secs: 0,
            ..settings()
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
        let client = cache(settings())
            .get(&HttpPoolConfig::default(), false)
            .expect("client");
        let resp = client.get(&url).send().await.expect("h1 request");
        assert_eq!(resp.version(), http::Version::HTTP_11);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }
}
