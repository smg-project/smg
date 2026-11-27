use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use super::{test_support::discovery_server, *};
use crate::jwt::test_support::rsa_components;

#[test]
fn test_cached_jwks_expiration() {
    let mut cached = CachedJwks::new(JwkSet { keys: vec![] }, Duration::from_secs(60));
    cached.fetched_at = Instant::now() - Duration::from_secs(100);
    assert!(cached.is_expired());

    let cached = CachedJwks::new(JwkSet { keys: vec![] }, Duration::from_secs(60));
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
        // One real key under every served `kid`: the provider parses each
        // entry at the fetch, so the set must carry parseable components.
        let (n, e) = rsa_components();
        let keys: Vec<Value> = self
            .keys
            .read()
            .iter()
            .map(|kid| {
                json!({ "kty": "RSA", "use": "sig", "alg": "RS256", "kid": kid,
                        "n": n, "e": e })
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
            .jwk
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

/// A lookup hands out the cached set's own entry, parsed once at the fetch:
/// two lookups of a `kid` share it, and a token's check borrows the
/// algorithm and the decoding key instead of building them.
#[tokio::test]
async fn a_lookup_shares_the_key_parsed_once_at_the_fetch() {
    let server = JwksServer::start(&["k1"]).await;
    let provider = server.provider();

    let first = provider.get_key("k1").await.unwrap();
    let again = provider.get_key("k1").await.unwrap();

    assert!(Arc::ptr_eq(&first, &again), "two lookups share one entry");
    assert_eq!(first.jwk.common.key_id.as_deref(), Some("k1"));
    let parsed = first.parsed.as_ref().expect("an RSA key parses");
    assert_eq!(parsed.algorithm, Algorithm::RS256);
    assert_eq!(server.hits(), 1);
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
            .jwk
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
        let mut kids: Vec<String> = cached.keys.keys().cloned().collect();
        kids.sort();
        kids
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
    provider.get_key("k3").await.unwrap();
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
    // The first lookup fetches the set; the refusal is its error.
    let err = provider.get_key("k1").await.unwrap_err();
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

/// The provider's client names `identity` as the content coding it
/// accepts, so a key-set server that compresses every request naming no
/// coding answers this one in the clear and the key is read. Without the
/// header the server sends a gzip body, which the client, built without
/// decoders, cannot parse.
#[tokio::test]
async fn the_client_names_identity_so_a_server_compressing_unasked_answers_in_the_clear() {
    use axum::{
        http::{
            header::{ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE},
            HeaderMap,
        },
        response::{IntoResponse, Response},
    };
    // `{"keys":[]}` under gzip: what the server sends a request that names no coding.
    const GZIPPED_KEY_SET: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0xab, 0x56, 0xca, 0x4e, 0xad,
        0x2c, 0x56, 0xb2, 0x8a, 0x8e, 0xad, 0x05, 0x00, 0xef, 0x67, 0x88, 0x2a, 0x0b, 0x00, 0x00,
        0x00,
    ];
    fn respond(headers: HeaderMap) -> Response {
        let names_identity = headers
            .get(ACCEPT_ENCODING)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(',')
                    .any(|coding| coding.trim().eq_ignore_ascii_case("identity"))
            });
        if names_identity {
            let (n, e) = rsa_components();
            return Json(json!({ "keys": [
                { "kty": "RSA", "use": "sig", "alg": "RS256", "kid": "k1", "n": n, "e": e }
            ] }))
            .into_response();
        }
        (
            [
                (CONTENT_TYPE, "application/json"),
                (CONTENT_ENCODING, "gzip"),
            ],
            GZIPPED_KEY_SET,
        )
            .into_response()
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://127.0.0.1:{}/jwks.json",
        listener.local_addr().unwrap().port()
    );
    let app = Router::new().route(
        "/jwks.json",
        get(|headers: HeaderMap| async move { respond(headers) }),
    );
    #[expect(
        clippy::disallowed_methods,
        reason = "the server lives exactly as long as the test"
    )]
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let provider = JwksProvider::new(url, Duration::from_secs(3600)).unwrap();
    let key = provider.get_key("k1").await.unwrap();
    assert_eq!(key.jwk.common.key_id.as_deref(), Some("k1"));
}
