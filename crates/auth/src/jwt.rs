//! JWT validation for control plane authentication.
//!
//! Supports:
//! - RS256, RS384, RS512 (RSA)
//! - ES256, ES384 (ECDSA)
//! - Audience and issuer validation
//! - Expiry and not-before validation with a shared leeway
//! - Role extraction from claims
//!
//! Security features:
//! - Algorithm verification (token alg must match key algorithm)
//! - Optional JTI tracking for replay protection
//! - SSRF protection via JWKS provider

use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};

use jsonwebtoken::{
    decode, decode_header,
    jwk::{AlgorithmParameters, Jwk},
    Algorithm, DecodingKey, TokenData, Validation,
};
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::{
    config::{JwtConfig, Role},
    jwks::{JwksError, JwksProvider},
};

/// Default size for JTI cache (number of tokens to track)
const DEFAULT_JTI_CACHE_SIZE: usize = 10_000;

/// Error types for JWT validation.
#[derive(Debug, thiserror::Error)]
pub enum JwtValidatorError {
    #[error("Invalid token format")]
    InvalidFormat,

    #[error("Token header missing 'kid' claim")]
    MissingKid,

    #[error("Failed to get signing key: {0}")]
    KeyError(#[from] JwksError),

    #[error("Unsupported algorithm: {0:?}")]
    UnsupportedAlgorithm(Algorithm),

    #[error("Token validation failed: {0}")]
    ValidationFailed(String),

    #[error("Failed to decode token: {0}")]
    DecodeFailed(#[from] jsonwebtoken::errors::Error),

    #[error("Failed to extract role from claims")]
    RoleExtractionFailed,

    #[error("No role mapping found for: {0}")]
    NoRoleMapping(String),

    #[error("Algorithm mismatch: token uses {token_alg:?} but key requires {key_alg:?}")]
    AlgorithmMismatch {
        token_alg: Algorithm,
        key_alg: Algorithm,
    },

    #[error("Token replay detected: JTI '{0}' has already been used")]
    TokenReplay(String),
}

/// Standard JWT claims we extract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StandardClaims {
    /// Subject (user ID)
    pub sub: Option<String>,

    /// Issuer
    pub iss: Option<String>,

    /// Audience (can be string or array)
    #[serde(default)]
    pub aud: Audience,

    /// Expiration time
    pub exp: Option<u64>,

    /// Issued at
    pub iat: Option<u64>,

    /// Not before
    pub nbf: Option<u64>,

    /// JWT ID
    pub jti: Option<String>,

    /// Email (common claim)
    pub email: Option<String>,

    /// Name (common claim)
    pub name: Option<String>,

    /// Preferred username (OIDC claim)
    pub preferred_username: Option<String>,

    /// All other claims
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Audience claim can be a single string or an array.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum Audience {
    Single(String),
    Multiple(Vec<String>),
    #[default]
    None,
}

impl Audience {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "public API reserved for production use; currently only exercised by tests"
        )
    )]
    pub fn contains(&self, aud: &str) -> bool {
        match self {
            Audience::Single(s) => s == aud,
            Audience::Multiple(v) => v.iter().any(|s| s == aud),
            Audience::None => false,
        }
    }
}

/// Validated token with extracted claims and role.
#[derive(Debug, Clone)]
pub struct ValidatedToken {
    /// Subject (user ID)
    pub subject: String,

    /// Issuer
    pub issuer: String,

    /// Assigned role
    pub role: Role,

    /// Email if present
    pub email: Option<String>,

    /// Display name if present
    pub name: Option<String>,
}

/// JTI (JWT ID) cache entry with expiration tracking.
struct JtiCacheEntry {
    /// When the token expires (for cleanup)
    expires_at: Instant,
}

/// JWT validator with JWKS integration.
pub struct JwtValidator {
    /// JWKS provider for key fetching
    jwks_provider: Arc<JwksProvider>,

    /// JWT configuration
    config: JwtConfig,

    /// Pre-configured validation settings
    validation: Validation,

    /// JTI cache for replay protection (optional)
    /// Maps JTI -> expiration time
    jti_cache: Option<Mutex<LruCache<String, JtiCacheEntry>>>,

    /// Whether to enable JTI replay protection
    enable_jti_check: bool,
}

impl JwtValidator {
    /// Create a new JWT validator with explicit JWKS URI.
    #[expect(dead_code)]
    pub(crate) fn new(config: JwtConfig, jwks_provider: Arc<JwksProvider>) -> Self {
        Self::new_with_options(config, jwks_provider, false)
    }

    /// Create a new JWT validator with optional JTI replay protection.
    pub(crate) fn new_with_options(
        config: JwtConfig,
        jwks_provider: Arc<JwksProvider>,
        enable_jti_check: bool,
    ) -> Self {
        let mut validation = Validation::default();

        // Set audience
        validation.set_audience(&[&config.audience]);

        // Set issuer
        validation.set_issuer(&[&config.issuer]);

        // Set leeway for clock skew
        validation.leeway = config.leeway_secs;

        // Enforce `nbf`: jsonwebtoken leaves it off by default, which let a token
        // minted for a future window through. Same leeway as `exp`; a token
        // without the claim is still accepted.
        validation.validate_nbf = true;

        validation.algorithms = vec![
            Algorithm::RS256,
            Algorithm::RS384,
            Algorithm::RS512,
            Algorithm::ES256,
            Algorithm::ES384,
        ];

        let jti_cache = if enable_jti_check {
            let cache_size = NonZeroUsize::new(DEFAULT_JTI_CACHE_SIZE).unwrap_or(NonZeroUsize::MIN);
            Some(Mutex::new(LruCache::new(cache_size)))
        } else {
            None
        };

        Self {
            jwks_provider,
            config,
            validation,
            jti_cache,
            enable_jti_check,
        }
    }

    /// Create a new JWT validator using OIDC discovery.
    pub async fn from_config(config: JwtConfig) -> Result<Self, JwtValidatorError> {
        Self::from_config_with_options(config, false).await
    }

    /// Create a new JWT validator with optional JTI replay protection.
    pub async fn from_config_with_options(
        config: JwtConfig,
        enable_jti_check: bool,
    ) -> Result<Self, JwtValidatorError> {
        let ttl = Duration::from_secs(config.jwks_cache_ttl_secs);

        let jwks_provider = if let Some(jwks_uri) = &config.jwks_uri {
            Arc::new(JwksProvider::new(jwks_uri.clone(), ttl)?)
        } else {
            Arc::new(JwksProvider::from_issuer(&config.issuer, ttl).await?)
        };

        Ok(Self::new_with_options(
            config,
            jwks_provider,
            enable_jti_check,
        ))
    }

    /// Validate a JWT token and extract claims.
    pub async fn validate(&self, token: &str) -> Result<ValidatedToken, JwtValidatorError> {
        // Decode header to get kid and algorithm
        let header = decode_header(token)?;
        let kid = header.kid.ok_or(JwtValidatorError::MissingKid)?;
        let token_algorithm = header.alg;

        debug!(
            "Validating JWT with kid: {}, alg: {:?}",
            kid, token_algorithm
        );

        // Get the signing key
        let jwk = self.jwks_provider.get_key(&kid).await?;

        // Determine algorithm from JWK
        let key_algorithm = Self::jwk_to_algorithm(&jwk)?;

        // SECURITY: Verify token algorithm matches key algorithm
        // This prevents algorithm confusion attacks
        if token_algorithm != key_algorithm {
            warn!(
                "Algorithm mismatch: token uses {:?} but key requires {:?}",
                token_algorithm, key_algorithm
            );
            return Err(JwtValidatorError::AlgorithmMismatch {
                token_alg: token_algorithm,
                key_alg: key_algorithm,
            });
        }

        // Create decoding key from JWK
        let decoding_key = Self::jwk_to_decoding_key(&jwk)?;

        // Create validation with specific algorithm
        let mut validation = self.validation.clone();
        validation.algorithms = vec![key_algorithm];

        // Decode and validate token
        let token_data: TokenData<StandardClaims> = decode(token, &decoding_key, &validation)?;

        let claims = token_data.claims;

        // Check JTI for replay protection if enabled
        if self.enable_jti_check {
            if let Some(jti) = &claims.jti {
                self.check_jti_replay(jti, &claims)?;
            }
        }

        // Extract subject
        let subject = claims
            .sub
            .clone()
            .or_else(|| claims.email.clone())
            .or_else(|| claims.preferred_username.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Extract issuer
        let issuer = claims
            .iss
            .clone()
            .unwrap_or_else(|| self.config.issuer.clone());

        // Extract role
        let role = self.extract_role(&claims);

        debug!(
            "JWT validated: subject={}, issuer={}, role={:?}",
            subject, issuer, role
        );

        Ok(ValidatedToken {
            subject,
            issuer,
            role,
            email: claims.email.clone(),
            name: claims.name.clone(),
        })
    }

    /// Check if a JTI has been used before (replay protection).
    fn check_jti_replay(
        &self,
        jti: &str,
        claims: &StandardClaims,
    ) -> Result<(), JwtValidatorError> {
        let Some(cache) = &self.jti_cache else {
            return Ok(());
        };

        let mut cache = cache.lock();

        let now = Instant::now();

        // Check if JTI exists and is still valid
        if let Some(entry) = cache.get(jti) {
            if entry.expires_at > now {
                // Token is still valid but JTI was already used
                return Err(JwtValidatorError::TokenReplay(jti.to_string()));
            }
            // Entry expired, remove it
            cache.pop(jti);
        }

        // Calculate expiration time from claims
        let expires_at = if let Some(exp) = claims.exp {
            // exp is Unix timestamp
            let exp_duration = Duration::from_secs(exp);
            let now_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();

            if exp_duration > now_unix {
                now + (exp_duration - now_unix)
            } else {
                // Already expired, but we still track it briefly
                now + Duration::from_secs(60)
            }
        } else {
            // No exp claim, use default TTL
            now + Duration::from_secs(3600)
        };

        // Record this JTI
        cache.put(jti.to_string(), JtiCacheEntry { expires_at });

        Ok(())
    }

    /// Extract role from claims using configured role claim and mapping.
    fn extract_role(&self, claims: &StandardClaims) -> Role {
        // Try to get the role claim value
        let role_value = claims.extra.get(&self.config.role_claim);

        let role_strings: Vec<String> = match role_value {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None => {
                // Try alternate claim names
                let alternates = ["role", "roles", "groups", "group"];
                let mut found = Vec::new();
                for alt in alternates {
                    if let Some(v) = claims.extra.get(alt) {
                        match v {
                            serde_json::Value::String(s) => found.push(s.clone()),
                            serde_json::Value::Array(arr) => found
                                .extend(arr.iter().filter_map(|v| v.as_str().map(String::from))),
                            _ => {}
                        }
                    }
                }
                found
            }
            _ => Vec::new(),
        };

        // If no role mapping configured, check for direct "admin" or "user" values
        if self.config.role_mapping.is_empty() {
            for role_str in &role_strings {
                if let Ok(role) = role_str.parse::<Role>() {
                    return role;
                }
            }
            // Default to User if no explicit role found
            warn!("No role found in JWT claims, defaulting to User");
            return Role::User;
        }

        // Use role mapping
        for role_str in &role_strings {
            if let Some(role) = self.config.role_mapping.get(role_str) {
                return *role;
            }
        }

        // Check if any mapped role is admin - if so, we need explicit mapping
        // Otherwise default to User for safety
        warn!(
            "No matching role mapping found for {:?}, defaulting to User",
            role_strings
        );
        Role::User
    }

    /// Convert a JWK to a DecodingKey.
    fn jwk_to_decoding_key(jwk: &Jwk) -> Result<DecodingKey, JwtValidatorError> {
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

    /// Determine the algorithm from a JWK.
    fn jwk_to_algorithm(jwk: &Jwk) -> Result<Algorithm, JwtValidatorError> {
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

    /// Get a reference to the JWKS provider.
    #[expect(dead_code)]
    pub(crate) fn jwks_provider(&self) -> &Arc<JwksProvider> {
        &self.jwks_provider
    }

    /// Get a reference to the JWT config.
    pub fn config(&self) -> &JwtConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::LazyLock,
        time::{SystemTime, UNIX_EPOCH},
    };

    use axum::{routing::get, Json, Router};
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use jsonwebtoken::{encode, errors::ErrorKind, EncodingKey, Header};
    use rsa::{
        pkcs8::{EncodePrivateKey, LineEnding},
        rand_core::OsRng,
        traits::PublicKeyParts,
        RsaPrivateKey,
    };
    use serde_json::json;
    use tokio::net::TcpListener;

    use super::*;

    #[test]
    fn test_audience_contains() {
        let single = Audience::Single("api://test".to_string());
        assert!(single.contains("api://test"));
        assert!(!single.contains("other"));

        let multiple =
            Audience::Multiple(vec!["api://test".to_string(), "api://other".to_string()]);
        assert!(multiple.contains("api://test"));
        assert!(multiple.contains("api://other"));
        assert!(!multiple.contains("unknown"));

        let none = Audience::None;
        assert!(!none.contains("anything"));
    }

    const ISSUER: &str = "https://issuer.example.com";
    const AUDIENCE: &str = "smg";
    const KID: &str = "test-key-1";

    /// One RSA key pair for the module: the private half signs the test tokens,
    /// the public components are what the JWKS endpoint serves.
    struct TestKeyPair {
        private_key_pem: String,
        jwk: serde_json::Value,
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

    /// A validator with the default leeway, reading the test key set over loopback.
    #[expect(
        clippy::disallowed_methods,
        reason = "the JWKS server lives exactly as long as the test"
    )]
    async fn validator() -> JwtValidator {
        let jwks = json!({ "keys": [TEST_KEYS.jwk.clone()] });
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

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// An admin token good for an hour, with the given `nbf`.
    fn token(nbf: Option<u64>) -> String {
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
}
