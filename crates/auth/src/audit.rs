//! Audit logging for control plane operations.
//!
//! Provides structured audit events for security monitoring and compliance.
//!
//! Security features:
//! - Input sanitization to prevent log injection attacks
//! - Structured logging for safe parsing
//!
//! Two lines per control-plane call, both on target `smg::audit` and both
//! silenced by one switch. `control_plane_audit` is written before the
//! handler, when the request is admitted or refused. `control_plane_mutation`
//! is written after the handler of a request that is not a read (GET, HEAD)
//! has answered: it carries the status and the resource the handler named
//! through [`AuditResource`].

use axum::response::Response;
use chrono::{DateTime, Utc};
use http::StatusCode;
use serde::Serialize;
use tracing::{info, span, Level};

use crate::config::Role;

/// Maximum length for sanitized strings to prevent log flooding
const MAX_SANITIZED_LENGTH: usize = 1024;

/// Sanitize a string for safe logging.
/// Removes/escapes control characters and newlines that could be used for log injection.
fn sanitize_for_log(input: &str) -> String {
    let mut result = String::with_capacity(input.len().min(MAX_SANITIZED_LENGTH));

    for ch in input.chars().take(MAX_SANITIZED_LENGTH) {
        match ch {
            // Replace newlines and carriage returns with escaped versions
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            // Replace tabs with escaped version
            '\t' => result.push_str("\\t"),
            // Escape backslashes
            '\\' => result.push_str("\\\\"),
            // Remove other control characters (ASCII 0-31, 127)
            c if c.is_control() => {
                result.push_str(&format!("\\x{:02x}", c as u32));
            }
            // Keep printable characters as-is
            c => result.push(c),
        }
    }

    // Indicate truncation if input was too long
    if input.len() > MAX_SANITIZED_LENGTH {
        result.push_str("...[TRUNCATED]");
    }

    result
}

/// Outcome of an audited operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AuditOutcome {
    /// Operation completed successfully
    Success,
    /// Operation denied (authorization failure)
    Denied,
    /// The handler answered an error status (a mutation's outcome line)
    Failed,
}

impl std::fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditOutcome::Success => write!(f, "success"),
            AuditOutcome::Denied => write!(f, "denied"),
            AuditOutcome::Failed => write!(f, "failed"),
        }
    }
}

/// Audit event for control plane operations.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Principal who performed the action (subject ID or API key ID)
    pub principal: String,

    /// Authentication method used (jwt, api_key)
    pub auth_method: String,

    /// Role of the principal
    pub role: Role,

    /// HTTP method
    pub method: String,

    /// Request path
    pub path: String,

    /// Resource being accessed (e.g., worker ID, wasm module ID)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,

    /// Operation outcome
    pub outcome: AuditOutcome,

    /// The status the handler answered, on a mutation's outcome line
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,

    /// Request ID for correlation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Additional details or error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Context for audit logging containing request and principal information.
#[derive(Debug, Clone)]
pub struct AuditContext<'a> {
    /// Principal who performed the action
    pub principal: &'a str,
    /// Authentication method used (jwt, api_key)
    pub auth_method: &'a str,
    /// Role of the principal
    pub role: Role,
    /// HTTP method
    pub method: &'a str,
    /// Request path
    pub path: &'a str,
    /// Request ID for correlation
    pub request_id: Option<&'a str>,
}

impl<'a> AuditContext<'a> {
    /// Create a new audit context.
    pub fn new(
        principal: &'a str,
        auth_method: &'a str,
        role: Role,
        method: &'a str,
        path: &'a str,
        request_id: Option<&'a str>,
    ) -> Self {
        Self {
            principal,
            auth_method,
            role,
            method,
            path,
            request_id,
        }
    }
}

/// The resource a control-plane handler acted on, for the outcome line of a
/// mutation: a worker id, a tokenizer id, a module uuid. A handler returns
/// its response through [`AuditResource::name`]; a response without one is
/// audited by its path alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditResource(pub String);

impl AuditResource {
    /// `response` with the resource it acted on in its extensions.
    pub fn name(mut response: Response, resource: impl Into<String>) -> Response {
        response.extensions_mut().insert(Self(resource.into()));
        response
    }
}

impl AuditEvent {
    /// Create a new audit event from context.
    fn from_context(
        ctx: &AuditContext<'_>,
        outcome: AuditOutcome,
        resource: Option<&str>,
        details: Option<&str>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            principal: sanitize_for_log(ctx.principal),
            auth_method: sanitize_for_log(ctx.auth_method),
            role: ctx.role,
            method: sanitize_for_log(ctx.method),
            path: sanitize_for_log(ctx.path),
            resource: resource.map(sanitize_for_log),
            outcome,
            status: None,
            request_id: ctx.request_id.map(sanitize_for_log),
            details: details.map(sanitize_for_log),
        }
    }

    /// Create an audit event for unauthenticated requests.
    fn unauthenticated(method: &str, path: &str, reason: &str, request_id: Option<&str>) -> Self {
        Self {
            timestamp: Utc::now(),
            principal: "unauthenticated".to_string(),
            auth_method: "none".to_string(),
            role: Role::User,
            method: sanitize_for_log(method),
            path: sanitize_for_log(path),
            resource: None,
            outcome: AuditOutcome::Denied,
            status: None,
            request_id: request_id.map(sanitize_for_log),
            details: Some(sanitize_for_log(reason)),
        }
    }
}

/// Audit logger that emits structured audit events.
#[derive(Clone, Default)]
pub struct AuditLogger {
    enabled: bool,
}

impl AuditLogger {
    /// Create a new audit logger.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Check if audit logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log an audit event: the admission line, `control_plane_audit`.
    pub fn log(&self, event: &AuditEvent) {
        self.emit(event, "control_plane_audit");
    }

    /// Log the outcome of a mutation once its handler has answered: the
    /// `control_plane_mutation` line with the status, `success` for a 2xx
    /// and `failed` otherwise, and the resource when the handler named one.
    pub fn log_outcome(&self, ctx: &AuditContext<'_>, status: StatusCode, resource: Option<&str>) {
        let outcome = if status.is_success() {
            AuditOutcome::Success
        } else {
            AuditOutcome::Failed
        };
        let mut event = AuditEvent::from_context(ctx, outcome, resource, None);
        event.status = Some(status.as_u16());
        self.emit(&event, "control_plane_mutation");
    }

    /// Write `event` as the line `name`, when enabled.
    fn emit(&self, event: &AuditEvent, name: &str) {
        if !self.enabled {
            return;
        }

        // Create a span for structured logging
        let _span = span!(
            Level::INFO,
            "audit",
            principal = %event.principal,
            auth_method = %event.auth_method,
            role = %event.role,
            method = %event.method,
            path = %event.path,
            outcome = %event.outcome,
        )
        .entered();

        // Log the event. The optional fields are recorded as plain string
        // values and left out when absent: formatted with `?` they would be
        // Rust `Debug` text (`Some("req-1")`, `None`) inside the JSON log.
        info!(
            target: "smg::audit",
            timestamp = %event.timestamp.to_rfc3339(),
            principal = %event.principal,
            auth_method = %event.auth_method,
            role = %event.role,
            method = %event.method,
            path = %event.path,
            resource = event.resource.as_deref(),
            outcome = %event.outcome,
            status = event.status,
            request_id = event.request_id.as_deref(),
            details = event.details.as_deref(),
            "{name}"
        );
    }

    /// Log a successful operation.
    pub fn log_success(&self, ctx: &AuditContext<'_>, resource: Option<&str>) {
        self.log(&AuditEvent::from_context(
            ctx,
            AuditOutcome::Success,
            resource,
            None,
        ));
    }

    /// Log a denied operation (authorization failure).
    pub fn log_denied(&self, ctx: &AuditContext<'_>, reason: &str) {
        self.log(&AuditEvent::from_context(
            ctx,
            AuditOutcome::Denied,
            None,
            Some(reason),
        ));
    }

    /// Log an authentication failure (before principal is known).
    pub fn log_auth_failure(
        &self,
        method: &str,
        path: &str,
        reason: &str,
        request_id: Option<&str>,
    ) {
        self.log(&AuditEvent::unauthenticated(
            method, path, reason, request_id,
        ));
    }
}

#[cfg(test)]
#[path = "../tests/unit/audit/tests.rs"]
pub(crate) mod tests;
