//! What a bearer token's check allocates once the key set is cached, pinned
//! by shape: a check costs the same on a set of one key and on a set of
//! four, because it borrows its key from the cached set instead of copying
//! the set, the key and the key's parsed form per token. The counts are
//! printed for the Numbers table. A separate executable, so it can declare
//! the counting allocator (`common`).

#![expect(
    clippy::unwrap_used,
    clippy::print_stderr,
    reason = "a test that stops on a broken setup and prints the counts it pins"
)]

mod common;

use axum::{routing::get, Json, Router};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use common::counting_alloc::{allocations_during, Counting};
use jsonwebtoken::{encode, get_current_timestamp, Algorithm, EncodingKey, Header};
use rsa::{
    pkcs8::{EncodePrivateKey, LineEnding},
    rand_core::OsRng,
    traits::PublicKeyParts,
    RsaPrivateKey,
};
use serde_json::{json, Value};
use smg_auth::{JwtConfig, JwtValidator};
use tokio::{net::TcpListener, runtime::Runtime};

#[global_allocator]
static GLOBAL: Counting = Counting;

const ISSUER: &str = "https://issuer.example.com";
const AUDIENCE: &str = "smg";

/// One RSA key pair: the private half signs the tokens, the public
/// components are what the key sets serve, under one `kid` or four.
struct KeyPair {
    private_key_pem: String,
    n: String,
    e: String,
}

fn key_pair() -> KeyPair {
    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
    KeyPair {
        private_key_pem: private_key
            .to_pkcs8_pem(LineEnding::LF)
            .unwrap()
            .to_string(),
        n: URL_SAFE_NO_PAD.encode(private_key.n().to_bytes_be()),
        e: URL_SAFE_NO_PAD.encode(private_key.e().to_bytes_be()),
    }
}

fn jwk(pair: &KeyPair, kid: &str) -> Value {
    json!({ "kty": "RSA", "use": "sig", "alg": "RS256", "kid": kid, "n": pair.n, "e": pair.e })
}

/// A key set served over loopback for as long as the test runs; its URI.
#[expect(
    clippy::disallowed_methods,
    reason = "the server lives exactly as long as the test"
)]
async fn serve(keys: Vec<Value>) -> String {
    let jwks = json!({ "keys": keys });
    let app = Router::new().route(
        "/jwks.json",
        get(move || {
            let jwks = jwks.clone();
            async move { Json(jwks) }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let uri = format!(
        "http://127.0.0.1:{}/jwks.json",
        listener.local_addr().unwrap().port()
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    uri
}

/// A token for `alice`, good for an hour, signed under `kid`.
fn token(pair: &KeyPair, kid: &str) -> String {
    let now = get_current_timestamp();
    let claims = json!({
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "alice",
        "iat": now,
        "exp": now + 3600,
        "roles": ["admin"],
    });
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.to_string());
    let key = EncodingKey::from_rsa_pem(pair.private_key_pem.as_bytes()).unwrap();
    encode(&header, &claims, &key).unwrap()
}

async fn validator(jwks_uri: String) -> JwtValidator {
    let config = JwtConfig::new(ISSUER, AUDIENCE).with_jwks_uri(jwks_uri);
    JwtValidator::from_config(config).await.unwrap()
}

/// The allocation calls of one check of `token` by `validator`, on the
/// test's thread: the runtime's threads only serve the key sets, and a check
/// on a cached set awaits nothing that yields, so the whole check runs here.
fn calls_per_check(rt: &Runtime, validator: &JwtValidator, token: &str) -> usize {
    let (validated, calls) = allocations_during(|| rt.block_on(validator.validate(token)).unwrap());
    assert_eq!(validated.subject, "alice");
    calls
}

/// A check reads its key through the cached set's entry: the set's size
/// changes nothing a token's check allocates, and a key is never parsed or
/// copied per token.
#[test]
fn a_token_check_costs_the_same_on_a_set_of_one_key_and_of_four() {
    let rt = Runtime::new().unwrap();
    let pair = key_pair();
    let one = rt.block_on(serve(vec![jwk(&pair, "k1")]));
    let four = rt.block_on(serve(
        ["k1", "k2", "k3", "k4"]
            .iter()
            .map(|kid| jwk(&pair, kid))
            .collect(),
    ));
    let on_one = rt.block_on(validator(one));
    let on_four = rt.block_on(validator(four));
    let token = token(&pair, "k1");

    // Warm: the first check fetches and caches the set; whatever a first
    // check builds once is not the pin's.
    for validator in [&on_one, &on_four] {
        assert_eq!(
            rt.block_on(validator.validate(&token)).unwrap().subject,
            "alice"
        );
    }
    let ((), runtime_own) = allocations_during(|| rt.block_on(async {}));
    let one_calls = calls_per_check(&rt, &on_one, &token);
    let four_calls = calls_per_check(&rt, &on_four, &token);
    eprintln!(
        "allocation calls per token check on a cached set: one key {one_calls}, four keys {four_calls} (the runtime's own block_on: {runtime_own})"
    );
    assert_eq!(
        four_calls, one_calls,
        "a token's check must not grow with the cached key set: the set is borrowed per token, never copied"
    );
}
