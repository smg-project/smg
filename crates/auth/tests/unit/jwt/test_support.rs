use std::{
    sync::LazyLock,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{routing::get, Json, Router};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use rsa::{
    pkcs8::{EncodePrivateKey, LineEnding},
    rand_core::OsRng,
    traits::PublicKeyParts,
    RsaPrivateKey,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;

use super::JwtValidator;
use crate::config::{JwtConfig, Role};

pub(crate) const ISSUER: &str = "https://issuer.example.com";
const AUDIENCE: &str = "smg";
const KID: &str = "test-key-1";

/// One RSA key pair for the module: the private half signs the test tokens,
/// the public components are what the JWKS endpoint serves.
struct TestKeyPair {
    private_key_pem: String,
    jwk: Value,
}

static TEST_KEYS: LazyLock<TestKeyPair> = LazyLock::new(|| {
    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
    let jwk = json!({
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": KID,
        "n": URL_SAFE_NO_PAD.encode(private_key.n().to_bytes_be()),
        "e": URL_SAFE_NO_PAD.encode(private_key.e().to_bytes_be()),
    });
    let private_key_pem = private_key
        .to_pkcs8_pem(LineEnding::LF)
        .unwrap()
        .to_string();
    TestKeyPair {
        private_key_pem,
        jwk,
    }
});

/// The module's RSA public components (`n`, `e`, base64url), for a key set
/// a test serves under its own `kid`s: a key the provider can parse.
pub(crate) fn rsa_components() -> (String, String) {
    let jwk = &TEST_KEYS.jwk;
    let field = |name: &str| jwk[name].as_str().unwrap_or_default().to_string();
    (field("n"), field("e"))
}

/// A validator with the default leeway, reading the test key set over loopback.
pub(crate) async fn validator() -> JwtValidator {
    validator_over(vec![TEST_KEYS.jwk.clone()]).await
}

/// A validator with the default leeway, reading `keys` as its key set over
/// loopback.
#[expect(
    clippy::disallowed_methods,
    reason = "the JWKS server lives exactly as long as the test"
)]
pub(crate) async fn validator_over(keys: Vec<Value>) -> JwtValidator {
    let jwks = json!({ "keys": keys });
    let app = Router::new().route(
        "/jwks.json",
        get(move || {
            let jwks = jwks.clone();
            async move { Json(jwks) }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let jwks_uri = format!(
        "http://127.0.0.1:{}/jwks.json",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let config = JwtConfig::new(ISSUER, AUDIENCE)
        .with_jwks_uri(jwks_uri)
        .with_role_mapping("admin", Role::Admin);
    JwtValidator::from_config(config).await.unwrap()
}

pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// An admin token for `alice`, good for an hour, with the given `nbf`.
pub(crate) fn token(nbf: Option<u64>) -> String {
    let now = now();
    let mut claims = json!({
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "alice",
        "iat": now,
        "exp": now + 3600,
        "roles": ["admin"],
    });
    if let Some(nbf) = nbf {
        claims["nbf"] = json!(nbf);
    }
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(KID.to_string());
    let key = EncodingKey::from_rsa_pem(TEST_KEYS.private_key_pem.as_bytes()).unwrap();
    encode(&header, &claims, &key).unwrap()
}
