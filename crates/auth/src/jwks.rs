//! JWKS (JSON Web Key Set) fetching and caching.
//!
//! Handles:
//! - OIDC discovery to find JWKS endpoint
//! - Fetching and parsing JWKS
//! - Caching with TTL and automatic refresh
//!
//! Security features:
//! - SSRF protection: only HTTPS URLs allowed, private IPs blocked
//! - Response size limits to prevent DoS
//! - Redirect prevention
//! - Rate-limited refresh on an unknown `kid`

use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    time::{Duration, Instant},
};

use jsonwebtoken::jwk::{Jwk, JwkSet};
use parking_lot::{Mutex, RwLock};
use rustls::crypto::{ring, CryptoProvider};
use tracing::{debug, info, warn};
use url::Url;

/// Maximum allowed JWKS response size (1 MB)
const MAX_JWKS_RESPONSE_SIZE: u64 = 1024 * 1024;

/// Minimum time between two fetches of the key set caused by a token whose `kid`
/// is not in the cached set. That refresh is how a rotated key gets picked up,
/// but the `kid` comes from an unauthenticated caller, so without a floor every
/// made-up `kid` costs the issuer one request.
const UNKNOWN_KID_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// Error types for JWKS operations.
#[derive(Debug, thiserror::Error)]
pub enum JwksError {
    #[error("Failed to fetch OIDC discovery document: {0}")]
    DiscoveryFetch(String),

    #[error("Failed to parse OIDC discovery document: {0}")]
    DiscoveryParse(String),

    #[error("OIDC discovery document is for the issuer {discovered:?}, not the configured issuer {configured:?} (OpenID Connect Discovery 1.0 requires the two to be identical)")]
    IssuerMismatch {
        configured: String,
        discovered: String,
    },

    #[error("JWKS URI not found in discovery document")]
    JwksUriNotFound,

    #[error("Failed to fetch JWKS: {0}")]
    JwksFetch(String),

    #[error("Failed to parse JWKS: {0}")]
    JwksParse(String),

    #[error("Key not found for kid: {0}")]
    KeyNotFound(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("SSRF protection: {0}")]
    SsrfBlocked(String),

    #[error("Response too large: {0} bytes (max: {1})")]
    ResponseTooLarge(u64, u64),

    #[error("Failed to create HTTP client: {0}")]
    HttpClientError(String),
}

/// Check if an IP address is private/internal (SSRF protection).
fn is_private_ip(ip: &IpAddr) -> bool {
    // IPv4-mapped IPv6 reaches the same destination as its embedded IPv4.
    match &ip.to_canonical() {
        IpAddr::V4(ipv4) => {
            ipv4.is_private()                           // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                || ipv4.is_loopback()                   // 127.0.0.0/8
                || ipv4.is_link_local()                 // 169.254.0.0/16
                || ipv4.is_broadcast()                  // 255.255.255.255
                || ipv4.is_unspecified()                // 0.0.0.0
                || is_shared_address(*ipv4)              // 100.64.0.0/10 (CGNAT)
                || is_documentation_v4(*ipv4)            // 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24
                || is_cloud_metadata(*ipv4) // 169.254.169.254
        }
        IpAddr::V6(ipv6) => {
            ipv6.is_loopback()                          // ::1
                || ipv6.is_unspecified()                // ::
                || is_unique_local(ipv6)                // fc00::/7
                || is_link_local_v6(ipv6) // fe80::/10
        }
    }
}

/// Check for CGNAT shared address space (100.64.0.0/10)
fn is_shared_address(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    octets[0] == 100 && (octets[1] & 0xC0) == 64
}

/// Check for IPv4 documentation addresses
fn is_documentation_v4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    // 192.0.2.0/24 (TEST-NET-1)
    (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
    // 198.51.100.0/24 (TEST-NET-2)
    || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
    // 203.0.113.0/24 (TEST-NET-3)
    || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
}

/// Check for cloud metadata endpoint (169.254.169.254)
fn is_cloud_metadata(ip: Ipv4Addr) -> bool {
    ip.octets() == [169, 254, 169, 254]
}

/// Check for IPv6 unique local addresses (fc00::/7)
fn is_unique_local(ip: &Ipv6Addr) -> bool {
    let segments = ip.segments();
    (segments[0] & 0xfe00) == 0xfc00
}

/// Check for IPv6 link-local addresses (fe80::/10)
fn is_link_local_v6(ip: &Ipv6Addr) -> bool {
    let segments = ip.segments();
    (segments[0] & 0xffc0) == 0xfe80
}

/// Validate a URL for SSRF protection.
/// Returns the validated URL or an error.
///
/// For testing purposes, HTTP is allowed for localhost/127.0.0.1 only.
/// In production, only HTTPS should be used.
pub(crate) fn validate_url(url_str: &str) -> Result<Url, JwksError> {
    let url = Url::parse(url_str)
        .map_err(|e| JwksError::InvalidUrl(format!("Failed to parse URL: {e}")))?;

    // Check if this is a localhost URL (allowed for testing with HTTP)
    let is_localhost = match url.host_str() {
        Some("localhost") | Some("127.0.0.1") => true,
        Some(host) => host == "::1" || host == "[::1]",
        None => false,
    };

    // Only allow HTTPS (except for localhost in tests)
    match url.scheme() {
        "https" => {}
        "http" => {
            if !is_localhost {
                return Err(JwksError::SsrfBlocked(
                    "Only HTTPS URLs are allowed for JWKS endpoints".to_string(),
                ));
            }
            // HTTP localhost is allowed - skip further checks and return early
            return Ok(url);
        }
        scheme => {
            return Err(JwksError::SsrfBlocked(format!(
                "Invalid URL scheme '{scheme}'. Only HTTPS is allowed"
            )));
        }
    }

    // For HTTPS URLs, check for private/internal addresses
    if let Some(host) = url.host() {
        match host {
            url::Host::Ipv4(ip) => {
                if is_private_ip(&IpAddr::V4(ip)) {
                    return Err(JwksError::SsrfBlocked(format!(
                        "Private/internal IP addresses are not allowed: {ip}"
                    )));
                }
            }
            url::Host::Ipv6(ip) => {
                if is_private_ip(&IpAddr::V6(ip)) {
                    return Err(JwksError::SsrfBlocked(format!(
                        "Private/internal IP addresses are not allowed: {ip}"
                    )));
                }
            }
            url::Host::Domain(domain) => {
                // Block common internal hostnames (except localhost which was handled above)
                let lower = domain.to_lowercase();
                if lower == "metadata"
                    || lower == "metadata.google.internal"
                    || lower.ends_with(".internal")
                    || lower.ends_with(".local")
                {
                    return Err(JwksError::SsrfBlocked(format!(
                        "Internal hostnames are not allowed: {domain}"
                    )));
                }
            }
        }
    }

    Ok(url)
}

/// OIDC discovery document (subset of fields we need).
#[derive(Debug, serde::Deserialize)]
struct OidcDiscovery {
    jwks_uri: String,
    /// The issuer the document is for: identical to the issuer the document
    /// was requested for, or the document is refused (Discovery 1.0, section 4.3).
    issuer: String,
}

/// Cached JWKS with expiration tracking.
struct CachedJwks {
    jwks: JwkSet,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedJwks {
    fn is_expired(&self) -> bool {
        self.fetched_at.elapsed() > self.ttl
    }
}

/// JWKS provider with caching and automatic refresh.
pub(crate) struct JwksProvider {
    /// HTTP client for fetching JWKS
    client: reqwest::Client,
    /// JWKS endpoint URL (validated)
    jwks_uri: String,
    /// Cached JWKS
    cache: RwLock<Option<CachedJwks>>,
    /// Cache TTL
    ttl: Duration,
    /// When the key set was last requested, whether or not the fetch succeeded;
    /// gates the refresh on an unknown `kid`.
    last_fetch_attempt: Mutex<Option<Instant>>,
}

/// Installs `ring` as the process-level TLS crypto provider unless one is
/// installed already. The HTTP client takes the process provider (reqwest's
/// `rustls-no-provider` feature) and panics at build time when there is
/// none, so the client factory installs it before building the client; a
/// provider the embedding process installed earlier stays.
fn install_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // An error here means another thread installed one in between,
        // which is the outcome wanted.
        let _ = ring::default_provider().install_default();
    }
}

/// The provider's HTTP client: a 10 s timeout and no redirects (SSRF).
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
fn http_client() -> Result<reqwest::Client, JwksError> {
    install_crypto_provider();
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none()) // Prevent SSRF via redirects
        .build()
        .map_err(|e| JwksError::HttpClientError(e.to_string()))
}

/// The response body, bounded by [`MAX_JWKS_RESPONSE_SIZE`] as it is read: a
/// declared length over the limit is refused before the read, and a body that
/// declares no length (a chunked response) is refused the moment the bytes
/// read pass the limit, so no fetch holds more than the limit in memory.
/// `fetch_error` wraps a transport failure in the caller's variant.
async fn bounded_body(
    mut response: reqwest::Response,
    fetch_error: fn(String) -> JwksError,
) -> Result<Vec<u8>, JwksError> {
    if let Some(content_length) = response.content_length() {
        if content_length > MAX_JWKS_RESPONSE_SIZE {
            return Err(JwksError::ResponseTooLarge(
                content_length,
                MAX_JWKS_RESPONSE_SIZE,
            ));
        }
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| fetch_error(e.to_string()))?
    {
        let read = body.len() as u64 + chunk.len() as u64;
        if read > MAX_JWKS_RESPONSE_SIZE {
            return Err(JwksError::ResponseTooLarge(read, MAX_JWKS_RESPONSE_SIZE));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

impl JwksProvider {
    /// Create a new JWKS provider with explicit JWKS URI.
    /// The URL is validated for SSRF protection.
    pub fn new(jwks_uri: impl Into<String>, ttl: Duration) -> Result<Self, JwksError> {
        let jwks_uri = jwks_uri.into();

        // Validate URL for SSRF protection
        validate_url(&jwks_uri)?;

        let client = http_client()?;

        Ok(Self {
            client,
            jwks_uri,
            cache: RwLock::new(None),
            ttl,
            last_fetch_attempt: Mutex::new(None),
        })
    }

    /// Create a new JWKS provider using OIDC discovery.
    /// Both issuer and discovered JWKS URI are validated for SSRF protection,
    /// and the document is taken only when its `issuer` is identical to the
    /// configured issuer (OpenID Connect Discovery 1.0, section 4.3).
    pub async fn from_issuer(issuer: &str, ttl: Duration) -> Result<Self, JwksError> {
        // The discovery URL is the issuer without its terminating slash plus the
        // well-known path (Discovery 1.0, section 4.1); the document's issuer is
        // compared with the issuer as configured.
        let discovery_url = format!(
            "{}/.well-known/openid-configuration",
            issuer.trim_end_matches('/')
        );

        // Validate discovery URL for SSRF protection
        validate_url(&discovery_url)?;

        let client = http_client()?;

        info!("Fetching OIDC discovery from: {}", discovery_url);

        let response = client
            .get(&discovery_url)
            .send()
            .await
            .map_err(|e| JwksError::DiscoveryFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(JwksError::DiscoveryFetch(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let body = bounded_body(response, JwksError::DiscoveryFetch).await?;
        let discovery: OidcDiscovery =
            serde_json::from_slice(&body).map_err(|e| JwksError::DiscoveryParse(e.to_string()))?;

        // The document's issuer must be identical to the issuer it was requested
        // for (Discovery 1.0, section 4.3): a document for another issuer names
        // another issuer's key set. The comparison is exact, as the one of a
        // token's `iss` claim against the configured issuer is, so a document
        // whose issuer differs only in a terminating slash is refused here as
        // its tokens would be.
        if discovery.issuer != issuer {
            return Err(JwksError::IssuerMismatch {
                configured: issuer.to_string(),
                discovered: discovery.issuer,
            });
        }

        // Validate discovered JWKS URI for SSRF protection
        validate_url(&discovery.jwks_uri)?;

        info!("Discovered JWKS URI: {}", discovery.jwks_uri);

        Ok(Self {
            client,
            jwks_uri: discovery.jwks_uri,
            cache: RwLock::new(None),
            ttl,
            last_fetch_attempt: Mutex::new(None),
        })
    }

    /// Fetch JWKS from the endpoint.
    /// Response size is limited to prevent DoS attacks.
    async fn fetch_jwks(&self) -> Result<JwkSet, JwksError> {
        debug!("Fetching JWKS from: {}", self.jwks_uri);
        *self.last_fetch_attempt.lock() = Some(Instant::now());

        let response = self
            .client
            .get(&self.jwks_uri)
            .send()
            .await
            .map_err(|e| JwksError::JwksFetch(e.to_string()))?;

        if !response.status().is_success() {
            return Err(JwksError::JwksFetch(format!("HTTP {}", response.status())));
        }

        let bytes = bounded_body(response, JwksError::JwksFetch).await?;
        let jwks: JwkSet =
            serde_json::from_slice(&bytes).map_err(|e| JwksError::JwksParse(e.to_string()))?;

        debug!("Fetched JWKS with {} keys", jwks.keys.len());
        Ok(jwks)
    }

    /// Fetch the key set and make it the cached one: the step behind an expiry
    /// refresh and an unknown-`kid` refresh alike.
    async fn refresh(&self) -> Result<JwkSet, JwksError> {
        let jwks = self.fetch_jwks().await?;
        *self.cache.write() = Some(CachedJwks {
            jwks: jwks.clone(),
            fetched_at: Instant::now(),
            ttl: self.ttl,
        });
        Ok(jwks)
    }

    /// Get the cached JWKS, refreshing if expired or not present.
    pub async fn get_jwks(&self) -> Result<JwkSet, JwksError> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.as_ref() {
                if !cached.is_expired() {
                    return Ok(cached.jwks.clone());
                }
            }
        }

        // Cache miss or expired.
        self.refresh().await
    }

    /// Get a specific key by kid (key ID).
    pub async fn get_key(&self, kid: &str) -> Result<Jwk, JwksError> {
        let jwks = self.get_jwks().await?;

        // First try to find by kid
        if let Some(key) = jwks.find(kid) {
            return Ok(key.clone());
        }

        // Key not found - try refreshing the cache in case keys were rotated,
        // at most once per interval: inside it the kid is simply unknown.
        if !self.unknown_kid_refresh_due() {
            debug!(
                "Key {} not found in cached JWKS; last fetch less than {:?} ago, not refreshing",
                kid, UNKNOWN_KID_REFRESH_INTERVAL
            );
            return Err(JwksError::KeyNotFound(kid.to_string()));
        }
        warn!("Key {} not found in cached JWKS, refreshing...", kid);
        let jwks = self.refresh().await?;
        jwks.find(kid)
            .cloned()
            .ok_or_else(|| JwksError::KeyNotFound(kid.to_string()))
    }

    /// Whether a refresh for an unknown `kid` may go out now: only when no fetch
    /// was attempted within [`UNKNOWN_KID_REFRESH_INTERVAL`]. Claims the slot, so
    /// concurrent misses inside the window do not each fetch.
    fn unknown_kid_refresh_due(&self) -> bool {
        let mut last_attempt = self.last_fetch_attempt.lock();
        if last_attempt.is_some_and(|at| at.elapsed() < UNKNOWN_KID_REFRESH_INTERVAL) {
            return false;
        }
        *last_attempt = Some(Instant::now());
        true
    }
}

/// A loopback OIDC discovery endpoint for the tests of the provider and of
/// the validator built on it.
#[cfg(test)]
pub(crate) mod test_support {
    use axum::{routing::get, Json, Router};
    use serde_json::json;
    use tokio::net::TcpListener;

    /// An OIDC discovery endpoint over loopback. Returns the server's origin,
    /// the issuer a document served there is requested for; the document names
    /// `document_issuer(origin)` as its issuer and the origin's `/jwks.json` as
    /// its key set.
    #[expect(
        clippy::disallowed_methods,
        reason = "the server lives exactly as long as the test"
    )]
    pub(crate) async fn discovery_server(document_issuer: impl FnOnce(&str) -> String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let document = json!({
            "issuer": document_issuer(&origin),
            "jwks_uri": format!("{origin}/jwks.json"),
        });
        let app = Router::new().route(
            "/.well-known/openid-configuration",
            get(move || {
                let document = document.clone();
                async move { Json(document) }
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        origin
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
    use serde_json::{json, Value};
    use tokio::net::TcpListener;

    use super::{test_support::discovery_server, *};

    #[test]
    fn test_cached_jwks_expiration() {
        let jwks = JwkSet { keys: vec![] };
        let cached = CachedJwks {
            jwks,
            fetched_at: Instant::now() - Duration::from_secs(100),
            ttl: Duration::from_secs(60),
        };
        assert!(cached.is_expired());

        let jwks = JwkSet { keys: vec![] };
        let cached = CachedJwks {
            jwks,
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(60),
        };
        assert!(!cached.is_expired());
    }

    #[test]
    fn test_validate_url_https_required() {
        // Valid HTTPS URL
        assert!(validate_url("https://example.com/.well-known/jwks.json").is_ok());

        // HTTP not allowed for non-localhost
        assert!(validate_url("http://example.com/.well-known/jwks.json").is_err());

        // HTTP allowed for localhost (testing)
        assert!(validate_url("http://localhost:8080/jwks").is_ok());
        assert!(validate_url("http://127.0.0.1:8080/jwks").is_ok());

        // Invalid schemes
        assert!(validate_url("ftp://example.com/jwks").is_err());
        assert!(validate_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_validate_url_blocks_private_ips() {
        // Private IPv4 ranges
        assert!(validate_url("https://10.0.0.1/jwks").is_err());
        assert!(validate_url("https://172.16.0.1/jwks").is_err());
        assert!(validate_url("https://192.168.1.1/jwks").is_err());

        // Loopback
        assert!(validate_url("https://127.0.0.1/jwks").is_err());

        // Link-local
        assert!(validate_url("https://169.254.1.1/jwks").is_err());

        // Cloud metadata endpoint
        assert!(validate_url("https://169.254.169.254/jwks").is_err());

        // CGNAT
        assert!(validate_url("https://100.64.0.1/jwks").is_err());
    }

    #[test]
    fn ipv4_mapped_ipv6_jwks_urls_obey_ipv4_address_policy() {
        for (host, blocked) in [
            ("::ffff:127.0.0.1", true),
            ("::ffff:7f00:1", true),
            ("::ffff:10.0.0.1", true),
            ("::ffff:172.16.0.1", true),
            ("::ffff:192.168.1.1", true),
            ("::ffff:169.254.169.254", true),
            ("::ffff:169.254.1.1", true),
            ("::ffff:100.64.0.1", true),
            ("::ffff:192.0.2.1", true),
            ("::ffff:198.51.100.1", true),
            ("::ffff:203.0.113.1", true),
            ("::ffff:0.0.0.0", true),
            ("::ffff:255.255.255.255", true),
            ("::ffff:8.8.8.8", false),
            ("::ffff:1.1.1.1", false),
            ("::1", true),
            ("::", true),
            ("fc00::1", true),
            ("fe80::1", true),
            ("2001:4860:4860::8888", false),
        ] {
            let result = validate_url(&format!("https://[{host}]/jwks"));
            if blocked {
                assert!(
                    matches!(result, Err(JwksError::SsrfBlocked(_))),
                    "private address {host} must be blocked, got {result:?}"
                );
            } else {
                assert!(result.is_ok(), "public address {host} must remain allowed");
            }
        }
    }

    #[test]
    fn test_validate_url_blocks_internal_hostnames() {
        assert!(validate_url("https://metadata/jwks").is_err());
        assert!(validate_url("https://metadata.google.internal/jwks").is_err());
        assert!(validate_url("https://internal.example.internal/jwks").is_err());
        assert!(validate_url("https://printer.local/jwks").is_err());
    }

    #[test]
    fn test_is_private_ip() {
        use std::net::Ipv4Addr;

        // Private ranges
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))));
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1))));
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));

        // Loopback
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));

        // Cloud metadata
        assert!(is_private_ip(&IpAddr::V4(Ipv4Addr::new(
            169, 254, 169, 254
        ))));

        // Public IPs should not be blocked
        assert!(!is_private_ip(&IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
        assert!(!is_private_ip(&IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
    }

    /// A JWKS endpoint over loopback that counts its hits and serves whatever
    /// key set (or failure) it currently holds.
    #[derive(Clone)]
    struct JwksServer {
        url: String,
        hits: Arc<AtomicUsize>,
        keys: Arc<RwLock<Vec<&'static str>>>,
        failing: Arc<RwLock<bool>>,
    }

    impl JwksServer {
        #[expect(
            clippy::disallowed_methods,
            reason = "the server lives exactly as long as the test"
        )]
        async fn start(kids: &[&'static str]) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let server = Self {
                url: format!(
                    "http://127.0.0.1:{}/jwks.json",
                    listener.local_addr().unwrap().port()
                ),
                hits: Arc::default(),
                keys: Arc::new(RwLock::new(kids.to_vec())),
                failing: Arc::default(),
            };
            let app = Router::new()
                .route(
                    "/jwks.json",
                    get(|State(server): State<Self>| async move { server.respond() }),
                )
                .with_state(server.clone());
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            server
        }

        fn respond(&self) -> (StatusCode, Json<Value>) {
            self.hits.fetch_add(1, Ordering::SeqCst);
            if *self.failing.read() {
                return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({})));
            }
            let keys: Vec<Value> = self
                .keys
                .read()
                .iter()
                .map(|kid| {
                    json!({ "kty": "RSA", "use": "sig", "alg": "RS256", "kid": kid,
                            "n": "sXchDaQebHnPiGvyDOAT4saGEUetSyo9MKLOoWFsueri23bOdgWp4Dy1Wl", "e": "AQAB" })
                })
                .collect();
            (StatusCode::OK, Json(json!({ "keys": keys })))
        }

        fn hits(&self) -> usize {
            self.hits.load(Ordering::SeqCst)
        }

        fn provider(&self) -> JwksProvider {
            JwksProvider::new(self.url.clone(), Duration::from_secs(3600)).unwrap()
        }
    }

    #[tokio::test]
    async fn an_unknown_kid_does_not_refetch_a_recently_fetched_key_set() {
        let server = JwksServer::start(&["k1"]).await;
        let provider = server.provider();

        assert_eq!(
            provider
                .get_key("k1")
                .await
                .unwrap()
                .common
                .key_id
                .as_deref(),
            Some("k1")
        );
        assert_eq!(server.hits(), 1);

        for kid in ["unknown-1", "unknown-2", "unknown-1"] {
            let err = provider.get_key(kid).await.unwrap_err();
            assert!(
                matches!(&err, JwksError::KeyNotFound(k) if k == kid),
                "{err:?}"
            );
        }
        assert_eq!(
            server.hits(),
            1,
            "misses inside the interval must not fetch"
        );
    }

    #[tokio::test]
    async fn a_rotated_key_is_fetched_once_the_interval_has_passed() {
        let server = JwksServer::start(&["k1"]).await;
        let provider = server.provider();
        provider.get_key("k1").await.unwrap();
        *server.keys.write() = vec!["k2"];

        // Inside the interval the new key is still unknown ...
        assert!(matches!(
            provider.get_key("k2").await,
            Err(JwksError::KeyNotFound(_))
        ));
        assert_eq!(server.hits(), 1);

        // ... and once the interval has passed, one refetch picks it up.
        *provider.last_fetch_attempt.lock() = Some(Instant::now() - UNKNOWN_KID_REFRESH_INTERVAL);
        assert_eq!(
            provider
                .get_key("k2")
                .await
                .unwrap()
                .common
                .key_id
                .as_deref(),
            Some("k2")
        );
        assert_eq!(server.hits(), 2);

        // That refetch opened a new interval: the next miss does not fetch.
        assert!(matches!(
            provider.get_key("k3").await,
            Err(JwksError::KeyNotFound(_))
        ));
        assert_eq!(server.hits(), 2);
    }

    #[tokio::test]
    async fn a_failed_refetch_counts_against_the_interval() {
        let server = JwksServer::start(&["k1"]).await;
        let provider = server.provider();
        provider.get_key("k1").await.unwrap();
        *server.failing.write() = true;
        *provider.last_fetch_attempt.lock() = Some(Instant::now() - UNKNOWN_KID_REFRESH_INTERVAL);

        assert!(matches!(
            provider.get_key("k2").await,
            Err(JwksError::JwksFetch(_))
        ));
        assert_eq!(server.hits(), 2);

        // The issuer is failing: further misses do not retry it until the interval has passed.
        assert!(matches!(
            provider.get_key("k2").await,
            Err(JwksError::KeyNotFound(_))
        ));
        assert_eq!(server.hits(), 2);
    }

    /// The expiry refresh and the unknown-`kid` refresh are one step: each
    /// fetches once and leaves the fetched set cached, fresh.
    #[tokio::test]
    async fn an_expiry_refresh_and_an_unknown_kid_refresh_cache_the_fetched_set_alike() {
        let server = JwksServer::start(&["k1"]).await;
        let provider = server.provider();
        let cached_kids = |provider: &JwksProvider| -> Vec<String> {
            let cache = provider.cache.read();
            let cached = cache.as_ref().unwrap();
            assert!(!cached.is_expired());
            cached
                .jwks
                .keys
                .iter()
                .filter_map(|key| key.common.key_id.clone())
                .collect()
        };
        provider.get_key("k1").await.unwrap();
        assert_eq!(cached_kids(&provider), ["k1"]);

        // The unknown-kid refresh, once the interval has passed.
        *server.keys.write() = vec!["k2"];
        *provider.last_fetch_attempt.lock() = Some(Instant::now() - UNKNOWN_KID_REFRESH_INTERVAL);
        provider.get_key("k2").await.unwrap();
        assert_eq!(cached_kids(&provider), ["k2"]);
        assert_eq!(server.hits(), 2);

        // The expiry refresh.
        *server.keys.write() = vec!["k3"];
        provider.cache.write().as_mut().unwrap().fetched_at =
            Instant::now() - TTL - Duration::from_secs(1);
        assert_eq!(provider.get_jwks().await.unwrap().keys.len(), 1);
        assert_eq!(cached_kids(&provider), ["k3"]);
        assert_eq!(server.hits(), 3);
    }

    const TTL: Duration = Duration::from_secs(3600);

    /// A loopback HTTP/1.1 server that answers one request with a chunked
    /// body (no declared length) of `chunks` times 64 KiB of JSON-looking
    /// text, then hangs up. Returns its origin. A refusal that comes within
    /// one chunk past the limit shows the body was stopped as it was read; a
    /// read-then-check would report the whole body.
    #[expect(
        clippy::disallowed_methods,
        reason = "the server lives exactly as long as the test"
    )]
    async fn chunked_server(chunks: usize) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = vec![0u8; 4096];
            let _ = socket.read(&mut head).await;
            let _ = socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n",
                )
                .await;
            let chunk = vec![b'['; 64 * 1024];
            for _ in 0..chunks {
                // The client may hang up once it has refused the body: a write
                // error ends the server, as it should.
                if socket
                    .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                    .await
                    .is_err()
                    || socket.write_all(&chunk).await.is_err()
                    || socket.write_all(b"\r\n").await.is_err()
                {
                    return;
                }
            }
            let _ = socket.write_all(b"0\r\n\r\n").await;
        });
        origin
    }

    #[tokio::test]
    async fn a_chunked_key_set_past_the_limit_is_refused_as_it_is_read() {
        let origin = chunked_server(20).await; // 1.25 MiB over a 1 MiB limit
        let provider = JwksProvider::new(format!("{origin}/jwks.json"), TTL).unwrap();
        let err = provider.get_jwks().await.unwrap_err();
        assert!(
            matches!(err, JwksError::ResponseTooLarge(read, max) if read > max && read <= max + 64 * 1024 && max == MAX_JWKS_RESPONSE_SIZE),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn a_chunked_discovery_document_past_the_limit_is_refused_as_it_is_read() {
        let origin = chunked_server(20).await;
        let Err(err) = JwksProvider::from_issuer(&origin, TTL).await else {
            panic!("a chunked discovery document past the limit must be refused");
        };
        assert!(
            matches!(err, JwksError::ResponseTooLarge(read, max) if read > max && read <= max + 64 * 1024 && max == MAX_JWKS_RESPONSE_SIZE),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn a_discovery_document_for_another_issuer_is_refused() {
        let origin = discovery_server(|origin| format!("{origin}/another-tenant")).await;
        let other = format!("{origin}/another-tenant");

        let Err(err) = JwksProvider::from_issuer(&origin, TTL).await else {
            panic!("a discovery document for another issuer must be refused");
        };

        assert!(
            matches!(
                &err,
                JwksError::IssuerMismatch { configured, discovered }
                    if configured == &origin && discovered == &other
            ),
            "{err:?}"
        );
        // The error names both issuers.
        let message = err.to_string();
        assert!(
            message.contains(&format!("{other:?}")) && message.contains(&format!("{origin:?}")),
            "{message}"
        );
    }

    #[tokio::test]
    async fn a_discovery_document_for_the_configured_issuer_is_accepted() {
        let origin = discovery_server(str::to_string).await;

        let provider = JwksProvider::from_issuer(&origin, TTL).await.unwrap();

        assert_eq!(provider.jwks_uri, format!("{origin}/jwks.json"));
    }

    #[tokio::test]
    async fn an_issuer_with_a_terminating_slash_is_compared_as_configured() {
        // The discovery URL carries one slash before the well-known path, and the
        // document must name the issuer with its slash, as the tokens' `iss` does.
        let origin = discovery_server(|origin| format!("{origin}/")).await;

        let provider = JwksProvider::from_issuer(&format!("{origin}/"), TTL)
            .await
            .unwrap();
        assert_eq!(provider.jwks_uri, format!("{origin}/jwks.json"));

        // The same document is not for the issuer without the slash.
        let Err(err) = JwksProvider::from_issuer(&origin, TTL).await else {
            panic!("a document naming the issuer with a slash is not for the issuer without it");
        };
        assert!(matches!(err, JwksError::IssuerMismatch { .. }), "{err:?}");
    }
}
