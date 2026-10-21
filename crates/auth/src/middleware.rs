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

impl AuthMethod {
    /// The method's name in the audit lines and logs: `jwt` or `api_key`.
    fn kind(&self) -> &'static str {
        match self {
            AuthMethod::Jwt { .. } => "jwt",
            AuthMethod::ApiKey { .. } => "api_key",
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

/// The request's audit facts, held apart from the request so they survive
/// its move into the handler.
struct Attempt {
    method: String,
    path: String,
    request_id: Option<String>,
}

impl Attempt {
    fn of(request: &Request<Body>) -> Self {
        Self {
            method: request.method().to_string(),
            path: request.uri().path().to_string(),
            request_id: request.extensions().get::<RequestId>().map(|r| r.0.clone()),
        }
    }

    /// Refuse the request: the audit line with `reason`, then a 401 whose
    /// challenge names this listener's realm and whose body says `detail`.
    fn unauthorized(&self, audit_logger: &AuditLogger, reason: &str, detail: &str) -> Response {
        audit_logger.log_auth_failure(&self.method, &self.path, reason, self.request_id.as_deref());
        (
            StatusCode::UNAUTHORIZED,
            [("WWW-Authenticate", "Bearer realm=\"control-plane\"")],
            detail.to_string(),
        )
            .into_response()
    }

    /// Admit `principal`, whichever credential proved it: the role check (a
    /// 403 for a method the role may not use), the audit line, the extension
    /// the handlers read, then the request itself.
    async fn admit(
        &self,
        audit_logger: &AuditLogger,
        principal: Principal,
        mut request: Request<Body>,
        next: Next,
    ) -> Response {
        let kind = principal.auth_method.kind();
        if let Some(resp) = check_role(
            &principal.id,
            kind,
            principal.role,
            &self.method,
            &self.path,
            self.request_id.as_deref(),
            audit_logger,
        ) {
            return resp;
        }
        log_auth_success(
            &principal,
            kind,
            &self.method,
            &self.path,
            self.request_id.as_deref(),
            audit_logger,
        );
        request.extensions_mut().insert(principal);
        next.run(request).await
    }
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
    request: Request<Body>,
    next: Next,
) -> Response {
    // If no authentication is configured, allow through (backward compatibility)
    if !auth_state.is_auth_required() {
        return next.run(request).await;
    }

    let attempt = Attempt::of(&request);
    let audit_logger = &auth_state.audit_logger;

    // Extract Bearer token from Authorization header
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(bearer_token);

    let Some(token) = token else {
        debug!("Missing or invalid Authorization header for control plane API");
        let reason = "Missing or invalid Authorization header";
        return attempt.unauthorized(audit_logger, reason, reason);
    };

    // Try JWT validation first
    if let Some(jwt_validator) = &auth_state.jwt_validator {
        match jwt_validator.validate(token).await {
            Ok(validated_token) => {
                let principal = Principal {
                    id: validated_token.subject,
                    name: validated_token.name,
                    auth_method: AuthMethod::Jwt {
                        issuer: validated_token.issuer,
                    },
                    role: validated_token.role,
                };
                return attempt.admit(audit_logger, principal, request, next).await;
            }
            Err(e) => {
                // 3 dot-separated parts => almost certainly a JWT; fail fast instead of trying API key.
                if token.split('.').count() == 3 {
                    warn!("Invalid JWT provided: {}. Not falling back to API key.", e);
                    let reason = format!("Invalid JWT: {e}");
                    return attempt.unauthorized(audit_logger, &reason, &reason);
                }
                debug!("JWT validation failed: {}, trying API key", e);
            }
        }
    }

    // Try API key validation
    if let Some(api_key_entry) = auth_state.config.find_api_key(token) {
        let principal = Principal {
            id: api_key_entry.id.clone(),
            name: Some(api_key_entry.name.clone()),
            auth_method: AuthMethod::ApiKey {
                key_id: api_key_entry.id.clone(),
            },
            role: api_key_entry.role,
        };
        return attempt.admit(audit_logger, principal, request, next).await;
    }

    // Authentication failed
    debug!("Control plane authentication failed: invalid token");
    attempt.unauthorized(
        audit_logger,
        "Invalid token",
        "Invalid authentication token",
    )
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

    /// The handler the middleware admits to answers with the principal it
    /// finds in the request, so the test sees what reached it.
    async fn whoami(request: Request<Body>) -> String {
        request
            .principal()
            .map(|p| format!("{} via {} as {:?}", p.id, p.auth_method, p.role))
            .unwrap_or_else(|| "nobody".to_string())
    }

    async fn call(
        app: &axum::Router,
        method: &str,
        key: Option<&str>,
    ) -> (StatusCode, Option<String>, String) {
        use tower::ServiceExt;

        let mut request = Request::builder().method(method).uri("/thing");
        if let Some(key) = key {
            request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        let response = app
            .clone()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let challenge = response
            .headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        (status, challenge, String::from_utf8(body.to_vec()).unwrap())
    }

    /// The API-key credential through the one admission path: a missing or
    /// unknown token is refused with the realm's challenge, a role short of
    /// the method is forbidden, and an admitted caller reaches the handler as
    /// the principal its key names.
    #[tokio::test]
    async fn an_api_key_caller_is_refused_forbidden_or_admitted_through_one_path() {
        use axum::{middleware::from_fn_with_state, routing::get, Router};

        use crate::config::ApiKeyEntry;

        let state = ControlPlaneAuthState::new(
            ControlPlaneAuthConfig {
                jwt: None,
                api_keys: vec![ApiKeyEntry::new(
                    "reader",
                    "Reader",
                    "reader-key",
                    Role::User,
                )],
                audit_enabled: false,
            },
            None,
        );
        let app = Router::new()
            .route("/thing", get(whoami).post(whoami))
            .layer(from_fn_with_state(state, control_plane_auth_middleware));
        let challenge = Some("Bearer realm=\"control-plane\"".to_string());

        assert_eq!(
            call(&app, "GET", None).await,
            (
                StatusCode::UNAUTHORIZED,
                challenge.clone(),
                "Missing or invalid Authorization header".to_string()
            )
        );
        assert_eq!(
            call(&app, "GET", Some("not-a-key")).await,
            (
                StatusCode::UNAUTHORIZED,
                challenge,
                "Invalid authentication token".to_string()
            )
        );
        let (status, _, body) = call(&app, "POST", Some("reader-key")).await;
        assert_eq!(
            (status, body),
            (
                StatusCode::FORBIDDEN,
                "Admin role required for this control plane operation".to_string()
            )
        );
        let (status, _, body) = call(&app, "GET", Some("reader-key")).await;
        assert_eq!(
            (status, body),
            (
                StatusCode::OK,
                "reader via api_key:reader as User".to_string()
            )
        );
    }

    /// The JWT credential through the same path: a signed token is admitted
    /// as the principal its claims name, and a token that is a JWT in shape
    /// but does not verify is refused at once with the challenge and the
    /// reason, with no fall-back to the API keys.
    #[tokio::test]
    async fn a_jwt_caller_is_admitted_or_refused_through_the_same_path() {
        use axum::{middleware::from_fn_with_state, routing::get, Router};

        use crate::{
            config::ApiKeyEntry,
            jwt::test_support::{token, validator, ISSUER},
        };

        let validator = validator().await;
        let state = ControlPlaneAuthState::new(
            ControlPlaneAuthConfig {
                jwt: Some(validator.config().clone()),
                api_keys: vec![ApiKeyEntry::new("ops", "Ops", "ops-key", Role::Admin)],
                audit_enabled: false,
            },
            Some(Arc::new(validator)),
        );
        let app = Router::new()
            .route("/thing", get(whoami))
            .layer(from_fn_with_state(state, control_plane_auth_middleware));

        let signed = token(None);
        let (status, _, body) = call(&app, "GET", Some(&signed)).await;
        assert_eq!(
            (status, body),
            (StatusCode::OK, format!("alice via jwt:{ISSUER} as Admin"))
        );

        let mut tampered = signed.clone();
        let last = tampered.pop().expect("a signed token has a signature");
        tampered.push(if last == 'A' { 'B' } else { 'A' });
        let (status, challenge, body) = call(&app, "GET", Some(&tampered)).await;
        assert_eq!(
            (status, challenge),
            (
                StatusCode::UNAUTHORIZED,
                Some("Bearer realm=\"control-plane\"".to_string())
            )
        );
        assert!(body.starts_with("Invalid JWT: "), "{body}");

        // A token that is not a JWT in shape still reaches the API keys.
        let (status, _, body) = call(&app, "GET", Some("ops-key")).await;
        assert_eq!(
            (status, body),
            (StatusCode::OK, "ops via api_key:ops as Admin".to_string())
        );
    }
}
