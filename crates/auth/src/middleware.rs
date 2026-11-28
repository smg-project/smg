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
use tracing::{debug, error, info, instrument::WithSubscriber, warn, Dispatch, Instrument};

use crate::{
    audit::{AuditContext, AuditLogger, AuditResource},
    config::{ApiKeySet, ControlPlaneAuthConfig, Role},
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
    /// Authentication configuration, as the start read it.
    pub config: ControlPlaneAuthConfig,

    /// The API keys in force: the configuration's at start, the key file's
    /// after a reload.
    pub keys: ApiKeySet,

    /// JWT validator (if JWT auth is configured)
    pub jwt_validator: Option<Arc<JwtValidator>>,

    /// Audit logger
    pub audit_logger: AuditLogger,

    /// The paths whose reads the `admin` role alone may make: a view that
    /// shows what the `user` role's reads do not. The gate refuses a user
    /// there like any other control-plane 403, denial line included.
    pub admin_only_reads: &'static [&'static str],

    /// What a recorded mutation's task takes for its lifetime, when the
    /// gateway hands one out: its in-flight count, so a shutdown drains the
    /// task of a client that left as it drains a waiting one.
    pub mutation_hold: Option<Arc<dyn Fn() -> MutationHold + Send + Sync>>,
}

/// A hold a recorded mutation's task owns until its outcome line is written:
/// whatever the gateway's shutdown drain waits on, released by its drop.
pub type MutationHold = Box<dyn std::any::Any + Send>;

impl ControlPlaneAuthState {
    /// Create a new control plane auth state.
    pub fn new(config: ControlPlaneAuthConfig, jwt_validator: Option<Arc<JwtValidator>>) -> Self {
        let audit_logger = AuditLogger::new(config.audit_enabled);
        let keys = ApiKeySet::new(config.api_keys.clone());
        Self {
            config,
            keys,
            jwt_validator,
            audit_logger,
            admin_only_reads: &[],
            mutation_hold: None,
        }
    }

    /// The same state with `paths` as the reads the `admin` role alone may
    /// make; the gateway names them where it attaches the layer.
    #[must_use]
    pub fn with_admin_only_reads(mut self, paths: &'static [&'static str]) -> Self {
        self.admin_only_reads = paths;
        self
    }

    /// The same state handing every recorded mutation's task a hold from
    /// `take` for its lifetime; the gateway passes its in-flight count where
    /// it attaches the layer, so the shutdown drain waits for a mutation
    /// whose client left until its outcome line is written.
    #[must_use]
    pub fn with_mutation_hold(
        mut self,
        take: impl Fn() -> MutationHold + Send + Sync + 'static,
    ) -> Self {
        self.mutation_hold = Some(Arc::new(take));
        self
    }

    /// Whether a read of `path` needs the `admin` role.
    fn is_admin_only_read(&self, path: &str) -> bool {
        self.admin_only_reads.contains(&path)
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

/// Whether `method` reads: the `user` role may use it, and no outcome line
/// follows its handler.
fn is_read(method: &str) -> bool {
    matches!(method, "GET" | "HEAD")
}

/// Whether `role` may perform `method` on a control plane route: `admin`
/// every method, `user` the read-only ones (GET and HEAD) unless the route
/// is one of the admin-only reads (`admin_only`).
fn role_allows(role: Role, method: &str, admin_only: bool) -> bool {
    role.is_admin() || (is_read(method) && !admin_only)
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
    /// The path is one of the reads the `admin` role alone may make.
    admin_only: bool,
    request_id: Option<String>,
}

impl Attempt {
    fn of(request: &Request<Body>, auth_state: &ControlPlaneAuthState) -> Self {
        let path = request.uri().path();
        Self {
            method: request.method().to_string(),
            admin_only: auth_state.is_admin_only_read(path),
            path: path.to_string(),
            request_id: request.extensions().get::<RequestId>().map(|r| r.0.clone()),
        }
    }

    /// The role check: `None` when `role` may make this request, else
    /// the 403 with the denial line written.
    fn refuse_by_role(
        &self,
        principal_id: &str,
        auth_method: &str,
        role: Role,
        audit_logger: &AuditLogger,
    ) -> Option<Response> {
        let (method, path, admin_only) =
            (self.method.as_str(), self.path.as_str(), self.admin_only);
        let request_id = self.request_id.as_deref();
        if role_allows(role, method, admin_only) {
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
    /// 403 for a method the role may not use, or for a read kept to the
    /// `admin` role), the admission line, the extension the handlers read,
    /// the request itself and, for a request that is not a read, the outcome
    /// line with the status the handler answered and the resource it named.
    ///
    /// A recorded mutation and its outcome line run as one spawned task, so a
    /// client that leaves before the answer drops the wait and not the
    /// record. The record owns the hold `take_hold` gives (the gateway's
    /// in-flight count) and releases it after the line, so a shutdown drains
    /// the task as it drains a waiting client's request. A task that ends
    /// without an answer is recorded and answered as a 500.
    async fn admit(
        &self,
        audit_logger: &AuditLogger,
        take_hold: Option<&(dyn Fn() -> MutationHold + Send + Sync)>,
        principal: Principal,
        mut request: Request<Body>,
        next: Next,
    ) -> Response {
        let kind = principal.auth_method.kind();
        if let Some(resp) = self.refuse_by_role(&principal.id, kind, principal.role, audit_logger) {
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
        let outcome = (audit_logger.is_enabled() && !is_read(&self.method))
            .then(|| (principal.id.clone(), principal.role));
        request.extensions_mut().insert(principal);
        let Some((principal_id, role)) = outcome else {
            return next.run(request).await;
        };
        let record = OutcomeRecord::new(
            OutcomeFacts {
                principal_id,
                kind,
                role,
                method: self.method.clone(),
                path: self.path.clone(),
                request_id: self.request_id.clone(),
                audit_logger: audit_logger.clone(),
            },
            take_hold.map(|take| take()),
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "the task is the admitted request's own work and ends with the handler's answer; it holds the gateway's in-flight count, so a shutdown drains it before the runtime drops it"
        )]
        let mutation = tokio::spawn(
            async move {
                let response = next.run(request).await;
                let resource = response.extensions().get::<AuditResource>();
                record.answered(response.status(), resource.map(|r| r.0.as_str()));
                response
            }
            .instrument(tracing::Span::current())
            .with_subscriber(tracing::dispatcher::get_default(Dispatch::clone)),
        );
        match mutation.await {
            Ok(response) => response,
            Err(error) => {
                error!(
                    "{} {} ended without an answer: {}",
                    self.method, self.path, error
                );
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        }
    }
}

/// What the outcome line of a recorded mutation says besides the status and
/// the resource.
struct OutcomeFacts {
    principal_id: String,
    kind: &'static str,
    role: Role,
    method: String,
    path: String,
    request_id: Option<String>,
    audit_logger: AuditLogger,
}

impl OutcomeFacts {
    fn write(&self, status: StatusCode, resource: Option<&str>) {
        let ctx = AuditContext::new(
            &self.principal_id,
            self.kind,
            self.role,
            &self.method,
            &self.path,
            self.request_id.as_deref(),
        );
        self.audit_logger.log_outcome(&ctx, status, resource);
    }
}

/// The outcome line the task that runs a recorded mutation owes: written with
/// the handler's answer, or at 500 with no resource when the handler
/// panicked (the record is dropped on the unwind, whichever frame drops it).
/// A record the runtime drops with the mutation in flight writes nothing: no
/// client was answered a 500 and the handler's blocking work may still
/// complete, so the admission line stands alone. The task owns it, so a
/// client that leaves cannot skip it.
struct OutcomeRecord {
    facts: OutcomeFacts,
    written: bool,
    /// The gateway's hold on its in-flight count, kept for its drop: a field
    /// drops after `Drop::drop` has run, so the hold goes after the line,
    /// whichever way the record goes.
    _hold: Option<MutationHold>,
}

impl OutcomeRecord {
    fn new(facts: OutcomeFacts, hold: Option<MutationHold>) -> Self {
        Self {
            facts,
            written: false,
            _hold: hold,
        }
    }

    fn answered(mut self, status: StatusCode, resource: Option<&str>) {
        self.written = true;
        self.facts.write(status, resource);
    }
}

impl Drop for OutcomeRecord {
    fn drop(&mut self) {
        if !self.written && std::thread::panicking() {
            self.facts.write(StatusCode::INTERNAL_SERVER_ERROR, None);
        }
    }
}

/// Control plane authentication middleware.
///
/// This middleware:
/// 1. Extracts the Bearer token from the Authorization header
/// 2. Attempts JWT validation first (if configured)
/// 3. Falls back to API key validation (if configured)
/// 4. Checks the principal's role against the request: `admin` for every
///    method, `user` for the read-only ones (GET, HEAD) that are not kept
///    to the `admin` role ([`ControlPlaneAuthState::with_admin_only_reads`])
/// 5. Writes the admission audit line and, after the handler of a request
///    that is not a read, the outcome line with its status and resource
///
/// Returns 401 Unauthorized if authentication fails.
/// Returns 403 Forbidden if the principal's role does not allow the method,
/// or the read is one kept to the `admin` role.
pub async fn control_plane_auth_middleware(
    State(auth_state): State<ControlPlaneAuthState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // If no authentication is configured, allow through (backward compatibility)
    if !auth_state.is_auth_required() {
        return next.run(request).await;
    }

    let attempt = Attempt::of(&request, &auth_state);
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
                return attempt
                    .admit(
                        audit_logger,
                        auth_state.mutation_hold.as_deref(),
                        principal,
                        request,
                        next,
                    )
                    .await;
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
    if let Some(api_key_entry) = auth_state.keys.find(token) {
        let principal = Principal {
            id: api_key_entry.id.clone(),
            name: Some(api_key_entry.name.clone()),
            auth_method: AuthMethod::ApiKey {
                key_id: api_key_entry.id.clone(),
            },
            role: api_key_entry.role,
        };
        return attempt
            .admit(
                audit_logger,
                auth_state.mutation_hold.as_deref(),
                principal,
                request,
                next,
            )
            .await;
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
#[path = "../tests/unit/middleware/tests.rs"]
mod tests;
