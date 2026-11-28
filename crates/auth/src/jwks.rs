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
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    sync::Arc,
    time::{Duration, Instant},
};

use jsonwebtoken::{
    jwk::{AlgorithmParameters, Jwk, JwkSet},
    Algorithm, DecodingKey,
};
use parking_lot::{Mutex, RwLock};
use rustls::crypto::{ring, CryptoProvider};
use tracing::{debug, info, warn};
use url::Url;

use crate::jwt::JwtValidatorError;

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

/// The cached key set with expiration tracking: its keys by `kid`, each with
/// its parsed form, built once here so a token's check borrows its key instead
/// of copying the set, the key and the key's parsed form per token.
struct CachedJwks {
    keys: HashMap<String, Arc<SigningKey>>,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedJwks {
    fn new(jwks: JwkSet, ttl: Duration) -> Self {
        let mut keys = HashMap::with_capacity(jwks.keys.len());
        for jwk in jwks.keys {
            let Some(kid) = jwk.common.key_id.clone() else {
                continue;
            };
            // The first key of a `kid` is the one a lookup finds, as before.
            keys.entry(kid)
                .or_insert_with(|| Arc::new(SigningKey::new(jwk)));
        }
        Self {
            keys,
            fetched_at: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.fetched_at.elapsed() > self.ttl
    }
}

/// A key of the cached set as a token's check reads it: the JWK as fetched
/// and, for a key the validator can use, its parsed form, built once when the
/// set is fetched and dropped with it, so no token parses the key again.
#[derive(Debug)]
pub(crate) struct SigningKey {
    pub(crate) jwk: Jwk,
    pub(crate) parsed: Option<ParsedKey>,
}

/// The parsed form of a JWK: the algorithm its tokens must carry and the
/// decoding key built from its components.
#[derive(Debug)]
pub(crate) struct ParsedKey {
    pub(crate) algorithm: Algorithm,
    pub(crate) decoding_key: DecodingKey,
}

impl SigningKey {
    fn new(jwk: Jwk) -> Self {
        let parsed = match (key_algorithm(&jwk), decoding_key(&jwk)) {
            (Ok(algorithm), Ok(decoding_key)) => Some(ParsedKey {
                algorithm,
                decoding_key,
            }),
            // A key this validator cannot use: a token's check reports why.
            _ => None,
        };
        Self { jwk, parsed }
    }
}

/// The algorithm a JWK's tokens must carry.
pub(crate) fn key_algorithm(jwk: &Jwk) -> Result<Algorithm, JwtValidatorError> {
    // First check if algorithm is explicitly specified
    if let Some(alg) = &jwk.common.key_algorithm {
        return Ok(match alg {
            jsonwebtoken::jwk::KeyAlgorithm::RS256 => Algorithm::RS256,
            jsonwebtoken::jwk::KeyAlgorithm::RS384 => Algorithm::RS384,
            jsonwebtoken::jwk::KeyAlgorithm::RS512 => Algorithm::RS512,
            jsonwebtoken::jwk::KeyAlgorithm::ES256 => Algorithm::ES256,
            jsonwebtoken::jwk::KeyAlgorithm::ES384 => Algorithm::ES384,
            other => {
                return Err(JwtValidatorError::ValidationFailed(format!(
                    "Unsupported key algorithm: {other:?}"
                )))
            }
        });
    }

    // Infer from key type
    match &jwk.algorithm {
        AlgorithmParameters::RSA(_) => Ok(Algorithm::RS256), // Default RSA to RS256
        AlgorithmParameters::EllipticCurve(ec) => {
            use jsonwebtoken::jwk::EllipticCurve;
            match ec.curve {
                EllipticCurve::P256 => Ok(Algorithm::ES256),
                EllipticCurve::P384 => Ok(Algorithm::ES384),
                // Other curves not supported for ECDSA
                _ => Err(JwtValidatorError::ValidationFailed(format!(
                    "Unsupported EC curve: {:?}",
                    ec.curve
                ))),
            }
        }
        _ => Err(JwtValidatorError::ValidationFailed(
            "Cannot determine algorithm from key".to_string(),
        )),
    }
}

/// The decoding key built from a JWK's components.
pub(crate) fn decoding_key(jwk: &Jwk) -> Result<DecodingKey, JwtValidatorError> {
    match &jwk.algorithm {
        AlgorithmParameters::RSA(rsa) => Ok(DecodingKey::from_rsa_components(&rsa.n, &rsa.e)?),
        AlgorithmParameters::EllipticCurve(ec) => {
            Ok(DecodingKey::from_ec_components(&ec.x, &ec.y)?)
        }
        AlgorithmParameters::OctetKey(_) => {
            Err(JwtValidatorError::UnsupportedAlgorithm(Algorithm::HS256))
        }
        AlgorithmParameters::OctetKeyPair(_) => {
            Err(JwtValidatorError::UnsupportedAlgorithm(Algorithm::EdDSA))
        }
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

/// The provider's HTTP client: a 10 s timeout, no redirects (SSRF) and the
/// provider's request headers.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
fn http_client() -> Result<reqwest::Client, JwksError> {
    install_crypto_provider();
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none()) // Prevent SSRF via redirects
        .default_headers(default_request_headers())
        .build()
        .map_err(|e| JwksError::HttpClientError(e.to_string()))
}

/// The headers the provider's client sends with every request: it names
/// `identity` as the one content coding it accepts. The client is built
/// without decoders, so a coded discovery document or key set would fail to
/// parse; a server that honours the header never sends one (RFC 9110,
/// section 12.5.3).
fn default_request_headers() -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::ACCEPT_ENCODING,
        reqwest::header::HeaderValue::from_static("identity"),
    );
    headers
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
    async fn refresh(&self) -> Result<(), JwksError> {
        let jwks = self.fetch_jwks().await?;
        *self.cache.write() = Some(CachedJwks::new(jwks, self.ttl));
        Ok(())
    }

    /// Fetch the key set when none is cached or the cached one has expired.
    async fn ensure_fresh(&self) -> Result<(), JwksError> {
        if self
            .cache
            .read()
            .as_ref()
            .is_some_and(|cached| !cached.is_expired())
        {
            return Ok(());
        }
        self.refresh().await
    }

    /// The cached set's key of this `kid`: a shared entry, nothing copied.
    fn cached_key(&self, kid: &str) -> Option<Arc<SigningKey>> {
        self.cache
            .read()
            .as_ref()
            .and_then(|cached| cached.keys.get(kid))
            .cloned()
    }

    /// Get a specific key by kid (key ID): the cached set's entry, with the
    /// key's parsed form, shared with every other check of a token it signed.
    pub async fn get_key(&self, kid: &str) -> Result<Arc<SigningKey>, JwksError> {
        self.ensure_fresh().await?;
        if let Some(key) = self.cached_key(kid) {
            return Ok(key);
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
        self.refresh().await?;
        self.cached_key(kid)
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
#[path = "../tests/unit/jwks/test_support.rs"]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "../tests/unit/jwks/tests.rs"]
mod tests;
