//! Control plane authentication middleware.
//!
//! Provides middleware for authenticating and authorizing access to control plane APIs.
//! Supports both JWT/OIDC tokens and API keys.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tracing::{debug, error, info, warn};

use crate::{
    audit::{AuditContext, AuditLogger},
    config::{ControlPlaneAuthConfig, Role},
    jwt::JwtValidator,
    RequestId,
};

/// Authenticated principal information.
#[derive(Debug, Clone)]
pub struct Principal {
    /// Subject identifier (user ID, email, or API key ID)
    pub id: String,

    /// Display name if available
    pub name: Option<String>,

    /// Authentication method used
    pub auth_method: AuthMethod,

    /// Assigned role
    pub role: Role,
}

/// Authentication method used to authenticate the principal.
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// JWT/OIDC token from external IDP
    Jwt { issuer: String },
    /// API key for service accounts
    ApiKey { key_id: String },
}

impl std::fmt::Display for AuthMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthMethod::Jwt { issuer } => write!(f, "jwt:{issuer}"),
            AuthMethod::ApiKey { key_id } => write!(f, "api_key:{key_id}"),
        }
    }
}

/// Extension trait for extracting Principal from request extensions.
pub trait PrincipalExt {
    fn principal(&self) -> Option<&Principal>;
}

impl<B> PrincipalExt for http::Request<B> {
    fn principal(&self) -> Option<&Principal> {
        self.extensions().get::<Principal>()
    }
}

/// State for the control plane authentication middleware.
#[derive(Clone)]
pub struct ControlPlaneAuthState {
    /// Authentication configuration
    pub config: ControlPlaneAuthConfig,

    /// JWT validator (if JWT auth is configured)
    pub jwt_validator: Option<Arc<JwtValidator>>,

    /// Audit logger
    pub audit_logger: AuditLogger,
}

impl ControlPlaneAuthState {
    /// Create a new control plane auth state.
    pub fn new(config: ControlPlaneAuthConfig, jwt_validator: Option<Arc<JwtValidator>>) -> Self {
        let audit_logger = AuditLogger::new(config.audit_enabled);
        Self {
            config,
            jwt_validator,
            audit_logger,
        }
    }

    /// Create from config, initializing JWT validator if needed.
    pub async fn from_config(
        config: ControlPlaneAuthConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let jwt_validator = if let Some(jwt_config) = &config.jwt {
            Some(Arc::new(
                JwtValidator::from_config(jwt_config.clone()).await?,
            ))
        } else {
            None
        };

        Ok(Self::new(config, jwt_validator))
    }

    /// Try to initialize control plane auth from config.
    ///
    /// Returns `None` only when authentication is absent or disabled.
    /// If JWT initialization fails, configured authentication remains required:
    /// JWT credentials are unavailable, but explicit control-plane API keys
    /// still work. JWT initialization is not retried; restart after correcting
    /// the configuration or restoring the identity provider.
    pub async fn try_init(config: Option<&ControlPlaneAuthConfig>) -> Option<Self> {
        let config = config.filter(|c| c.is_enabled())?;

        info!("Initializing control plane authentication...");
        match Self::from_config(config.clone()).await {
            Ok(state) => {
                if config.has_jwt() {
                    info!("Control plane JWT/OIDC authentication enabled");
                }
                if config.has_api_keys() {
                    info!(
                        "Control plane API key authentication enabled ({} keys)",
                        config.api_keys.len()
                    );
                }
                if config.audit_enabled {
                    info!("Control plane audit logging enabled");
                }
                Some(state)
            }
            Err(e) => {
                error!(
                    "Failed to initialize JWT authentication: {}. Control-plane authentication remains required; configured control-plane API keys remain available. Restart to retry JWT initialization.",
                    e
                );
                Some(Self::new(config.clone(), None))
            }
        }
    }

    /// Check if authentication is required.
    pub fn is_auth_required(&self) -> bool {
        self.config.is_enabled()
    }
}

/// Whether `role` may perform `method` on a control plane route: `admin`
/// every method, `user` the read-only ones (GET and HEAD).
fn role_allows(role: Role, method: &str) -> bool {
    role.is_admin() || matches!(method, "GET" | "HEAD")
}

/// Check the role against the request method and log a denial.
/// Returns Some(Response) if denied, None if allowed.
fn check_role(
    principal_id: &str,
    auth_method: &str,
    role: Role,
    method: &str,
    path: &str,
    request_id: Option<&str>,
    audit_logger: &AuditLogger,
) -> Option<Response> {
    if role_allows(role, method) {
        return None;
    }

    warn!(
        "{} {} has role {:?} but admin is required for {} {}",
        auth_method, principal_id, role, method, path
    );
    let ctx = AuditContext::new(principal_id, auth_method, role, method, path, request_id);
    audit_logger.log_denied(&ctx, "Admin role required for this control plane operation");

    Some(
        (
            StatusCode::FORBIDDEN,
            "Admin role required for this control plane operation",
        )
            .into_response(),
    )
}

/// Log successful authentication.
fn log_auth_success(
    principal: &Principal,
    auth_method: &str,
    method: &str,
    path: &str,
    request_id: Option<&str>,
    audit_logger: &AuditLogger,
) {
    debug!(
        "{} authentication successful for {} with role {:?}",
        auth_method, principal.id, principal.role
    );
    let ctx = AuditContext::new(
        &principal.id,
        auth_method,
        principal.role,
        method,
        path,
        request_id,
    );
    audit_logger.log_success(&ctx, None);
}

/// The token of a `Bearer` credential in an `Authorization` header value.
///
/// The scheme is matched case-insensitively (RFC 7235: authentication scheme
/// names are case-insensitive), followed by one or more spaces and a
/// non-empty token. Returns `None` for any other scheme or an empty token.
pub fn bearer_token(header_value: &str) -> Option<&str> {
    let (scheme, rest) = header_value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = rest.trim_start_matches(' ');
    (!token.is_empty()).then_some(token)
}

/// Control plane authentication middleware.
///
/// This middleware:
/// 1. Extracts the Bearer token from the Authorization header
/// 2. Attempts JWT validation first (if configured)
/// 3. Falls back to API key validation (if configured)
/// 4. Checks the principal's role against the request: `admin` for every
///    method, `user` for the read-only ones (GET, HEAD)
/// 5. Logs audit events for control plane access
///
/// Returns 401 Unauthorized if authentication fails.
/// Returns 403 Forbidden if the principal's role does not allow the method.
pub async fn control_plane_auth_middleware(
    State(auth_state): State<ControlPlaneAuthState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // If no authentication is configured, allow through (backward compatibility)
    if !auth_state.is_auth_required() {
        return next.run(request).await;
    }

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let request_id = request.extensions().get::<RequestId>().map(|r| r.0.clone());

    // Extract Bearer token from Authorization header
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(bearer_token);

    let Some(token) = token else {
        debug!("Missing or invalid Authorization header for control plane API");
        auth_state.audit_logger.log_auth_failure(
            &method,
            &path,
            "Missing or invalid Authorization header",
            request_id.as_deref(),
        );
        return (
            StatusCode::UNAUTHORIZED,
            [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
            "Missing or invalid Authorization header",
        )
            .into_response();
    };

    // Try JWT validation first
    if let Some(jwt_validator) = &auth_state.jwt_validator {
        match jwt_validator.validate(token).await {
            Ok(validated_token) => {
                if let Some(resp) = check_role(
                    &validated_token.subject,
                    "jwt",
                    validated_token.role,
                    &method,
                    &path,
                    request_id.as_deref(),
                    &auth_state.audit_logger,
                ) {
                    return resp;
                }

                let principal = Principal {
                    id: validated_token.subject.clone(),
                    name: validated_token.name.clone(),
                    auth_method: AuthMethod::Jwt {
                        issuer: validated_token.issuer.clone(),
                    },
                    role: validated_token.role,
                };

                log_auth_success(
                    &principal,
                    "jwt",
                    &method,
                    &path,
                    request_id.as_deref(),
                    &auth_state.audit_logger,
                );
                request.extensions_mut().insert(principal);
                return next.run(request).await;
            }
            Err(e) => {
                // 3 dot-separated parts => almost certainly a JWT; fail fast instead of trying API key.
                if token.split('.').count() == 3 {
                    warn!("Invalid JWT provided: {}. Not falling back to API key.", e);
                    auth_state.audit_logger.log_auth_failure(
                        &method,
                        &path,
                        &format!("Invalid JWT: {e}"),
                        request_id.as_deref(),
                    );
                    return (
                        StatusCode::UNAUTHORIZED,
                        [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
                        format!("Invalid JWT: {e}"),
                    )
                        .into_response();
                }
                debug!("JWT validation failed: {}, trying API key", e);
            }
        }
    }

    // Try API key validation
    if let Some(api_key_entry) = auth_state.config.find_api_key(token) {
        if let Some(resp) = check_role(
            &api_key_entry.id,
            "api_key",
            api_key_entry.role,
            &method,
            &path,
            request_id.as_deref(),
            &auth_state.audit_logger,
        ) {
            return resp;
        }

        let principal = Principal {
            id: api_key_entry.id.clone(),
            name: Some(api_key_entry.name.clone()),
            auth_method: AuthMethod::ApiKey {
                key_id: api_key_entry.id.clone(),
            },
            role: api_key_entry.role,
        };

        log_auth_success(
            &principal,
            "api_key",
            &method,
            &path,
            request_id.as_deref(),
            &auth_state.audit_logger,
        );
        request.extensions_mut().insert(principal);
        return next.run(request).await;
    }

    // Authentication failed
    debug!("Control plane authentication failed: invalid token");
    auth_state.audit_logger.log_auth_failure(
        &method,
        &path,
        "Invalid token",
        request_id.as_deref(),
    );

    (
        StatusCode::UNAUTHORIZED,
        [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
        "Invalid authentication token",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn failed_jwt_initialization_retains_required_authentication() {
        use crate::config::{ApiKeyEntry, JwtConfig};

        for with_api_key in [false, true] {
            let config = ControlPlaneAuthConfig {
                jwt: Some(
                    JwtConfig::new("https://issuer.example.com", "audience")
                        .with_jwks_uri("https://127.0.0.1/jwks"),
                ),
                api_keys: if with_api_key {
                    vec![ApiKeyEntry::new(
                        "admin",
                        "Admin",
                        "control-plane-key",
                        Role::Admin,
                    )]
                } else {
                    vec![]
                },
                audit_enabled: false,
            };
            let state = ControlPlaneAuthState::try_init(Some(&config)).await;
            assert!(
                state.is_some(),
                "failed JWT setup must retain the configured auth boundary"
            );
            let state = state.unwrap();
            assert!(state.is_auth_required());
            assert!(state.jwt_validator.is_none());
            assert_eq!(state.config.has_api_keys(), with_api_key);
        }
    }

    #[tokio::test]
    async fn optional_auth_initialization_preserves_disabled_and_api_key_only_modes() {
        use crate::config::ApiKeyEntry;

        assert!(ControlPlaneAuthState::try_init(None).await.is_none());
        assert!(
            ControlPlaneAuthState::try_init(Some(&ControlPlaneAuthConfig::default()))
                .await
                .is_none()
        );
        let config = ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![ApiKeyEntry::new(
                "admin",
                "Admin",
                "control-plane-key",
                Role::Admin,
            )],
            audit_enabled: false,
        };
        let state = ControlPlaneAuthState::try_init(Some(&config))
            .await
            .unwrap();
        assert!(state.is_auth_required());
        assert!(state.config.find_api_key("control-plane-key").is_some());
    }

    #[test]
    fn test_auth_method_display() {
        let jwt = AuthMethod::Jwt {
            issuer: "https://example.com".to_string(),
        };
        assert_eq!(jwt.to_string(), "jwt:https://example.com");

        let api_key = AuthMethod::ApiKey {
            key_id: "key-123".to_string(),
        };
        assert_eq!(api_key.to_string(), "api_key:key-123");
    }

    #[test]
    fn bearer_token_matches_the_scheme_case_insensitively() {
        assert_eq!(bearer_token("Bearer k1"), Some("k1"));
        assert_eq!(bearer_token("bearer k1"), Some("k1"));
        assert_eq!(bearer_token("BEARER k1"), Some("k1"));
        assert_eq!(bearer_token("Bearer   k1"), Some("k1"));
    }

    #[test]
    fn bearer_token_rejects_other_schemes_and_empty_tokens() {
        assert_eq!(bearer_token("Basic k1"), None);
        assert_eq!(bearer_token("Bearerk1"), None);
        assert_eq!(bearer_token("Bearer"), None);
        assert_eq!(bearer_token("Bearer "), None);
        assert_eq!(bearer_token("k1"), None);
    }

    #[test]
    fn admin_may_use_every_method() {
        for method in ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"] {
            assert!(role_allows(Role::Admin, method), "{method}");
        }
    }

    #[test]
    fn user_may_only_read() {
        assert!(role_allows(Role::User, "GET"));
        assert!(role_allows(Role::User, "HEAD"));
        for method in ["POST", "PUT", "PATCH", "DELETE"] {
            assert!(!role_allows(Role::User, method), "{method}");
        }
    }

    #[test]
    fn test_control_plane_auth_state_no_config() {
        let config = ControlPlaneAuthConfig::default();
        let state = ControlPlaneAuthState::new(config, None);
        assert!(!state.is_auth_required());
    }

    #[test]
    fn test_control_plane_auth_state_with_api_keys() {
        use crate::config::ApiKeyEntry;

        let config = ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![ApiKeyEntry::new("test", "Test Key", "secret", Role::Admin)],
            audit_enabled: true,
        };
        let state = ControlPlaneAuthState::new(config, None);
        assert!(state.is_auth_required());
    }
}
