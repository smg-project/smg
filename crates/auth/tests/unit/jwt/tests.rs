use jsonwebtoken::{encode, errors::ErrorKind, EncodingKey, Header};
use serde_json::json;

use super::{
    test_support::{now, token, validator, validator_over, ISSUER},
    *,
};
use crate::jwks::test_support::discovery_server;

#[test]
fn test_audience_contains() {
    let single = Audience::Single("api://test".to_string());
    assert!(single.contains("api://test"));
    assert!(!single.contains("other"));

    let multiple = Audience::Multiple(vec!["api://test".to_string(), "api://other".to_string()]);
    assert!(multiple.contains("api://test"));
    assert!(multiple.contains("api://other"));
    assert!(!multiple.contains("unknown"));

    let none = Audience::None;
    assert!(!none.contains("anything"));
}

#[tokio::test]
async fn a_token_that_is_not_valid_yet_is_rejected() {
    let validator = validator().await;

    let err = validator
        .validate(&token(Some(now() + 60)))
        .await
        .unwrap_err();

    assert!(
        matches!(
            &err,
            JwtValidatorError::DecodeFailed(e)
                if matches!(e.kind(), ErrorKind::ImmatureSignature)
        ),
        "expected ImmatureSignature, got {err:?}"
    );
}

#[tokio::test]
async fn nbf_in_the_past_or_within_the_leeway_is_accepted() {
    let validator = validator().await;
    assert_eq!(validator.config().leeway_secs, 30);

    for nbf in [Some(now() - 1), Some(now() + 10), None] {
        let validated = validator
            .validate(&token(nbf))
            .await
            .unwrap_or_else(|e| panic!("nbf {nbf:?} must be accepted: {e}"));
        assert_eq!(validated.subject, "alice");
        assert_eq!(validated.role, Role::Admin);
    }
}

#[tokio::test]
async fn a_token_of_another_algorithm_naming_an_unusable_key_reads_the_mismatch() {
    // An octet key declaring RS256: its algorithm converts, its material
    // does not, so the cached set keeps it without a parsed form. A token
    // of another algorithm naming it reads the mismatch (and its warning),
    // not the conversion's error: the check keeps its place before the
    // key's conversion.
    let validator = validator_over(vec![json!({
        "kty": "oct", "alg": "RS256", "kid": "oct-1", "k": "c2VjcmV0",
    })])
    .await;
    let mut header = Header::new(Algorithm::HS256);
    header.kid = Some("oct-1".to_string());
    let claims = json!({ "iss": ISSUER, "aud": "smg", "sub": "alice", "exp": now() + 3600 });
    let token = encode(&header, &claims, &EncodingKey::from_secret(b"secret")).unwrap();

    let err = validator.validate(&token).await.unwrap_err();

    assert!(
        matches!(
            &err,
            JwtValidatorError::AlgorithmMismatch {
                token_alg: Algorithm::HS256,
                key_alg: Algorithm::RS256,
            }
        ),
        "expected the mismatch, got {err:?}"
    );
}

#[tokio::test]
async fn no_validator_is_built_on_a_discovery_document_for_another_issuer() {
    // The document at the configured issuer's discovery endpoint names another
    // issuer: the key set provider refuses it, so no validator comes to be.
    let origin = discovery_server(|_| ISSUER.to_string()).await;

    let Err(err) = JwtValidator::from_config(JwtConfig::new(&origin, "smg")).await else {
        panic!("no validator may be built on a discovery document for another issuer");
    };

    assert!(
        matches!(
            &err,
            JwtValidatorError::KeyError(JwksError::IssuerMismatch { configured, discovered })
                if configured == &origin && discovered == ISSUER
        ),
        "{err:?}"
    );
}
