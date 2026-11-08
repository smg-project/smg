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
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_audit_event_from_context() {
        let ctx = AuditContext::new(
            "user@example.com",
            "jwt",
            Role::Admin,
            "POST",
            "/workers",
            Some("req-abc"),
        );
        let event = AuditEvent::from_context(&ctx, AuditOutcome::Success, Some("worker-123"), None);

        assert_eq!(event.principal, "user@example.com");
        assert_eq!(event.auth_method, "jwt");
        assert_eq!(event.role, Role::Admin);
        assert_eq!(event.method, "POST");
        assert_eq!(event.path, "/workers");
        assert_eq!(event.resource, Some("worker-123".to_string()));
        assert_eq!(event.outcome, AuditOutcome::Success);
        assert_eq!(event.request_id, Some("req-abc".to_string()));
    }

    #[test]
    fn test_audit_outcome_display() {
        assert_eq!(AuditOutcome::Success.to_string(), "success");
        assert_eq!(AuditOutcome::Denied.to_string(), "denied");
    }

    /// A subscriber that keeps, per audit line, how each field was recorded:
    /// as a string value (`record_str`) or as `Debug` text (`record_debug`).
    /// Lines on other targets (the middleware's own debug and warn lines)
    /// are not kept.
    #[derive(Default)]
    pub(crate) struct FieldCapture {
        pub(crate) events: std::sync::Mutex<Vec<Vec<(String, String)>>>,
    }

    impl tracing::Subscriber for FieldCapture {
        fn register_callsite(
            &self,
            _: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            tracing::subscriber::Interest::sometimes()
        }

        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            metadata.target() == "smg::audit"
        }

        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}

        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            struct Visitor<'a>(&'a mut Vec<(String, String)>);

            impl tracing::field::Visit for Visitor<'_> {
                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    self.0
                        .push((field.name().to_string(), format!("str:{value}")));
                }

                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    self.0
                        .push((field.name().to_string(), format!("debug:{value:?}")));
                }
            }

            let mut fields = Vec::new();
            event.record(&mut Visitor(&mut fields));
            self.events.lock().unwrap().push(fields);
        }

        fn enter(&self, _: &span::Id) {}

        fn exit(&self, _: &span::Id) {}
    }

    /// The audit lines `log` wrote, each as its recorded fields in order.
    pub(crate) fn recorded_events(log: impl FnOnce()) -> Vec<Vec<(String, String)>> {
        let capture = std::sync::Arc::new(FieldCapture::default());
        tracing::subscriber::with_default(capture.clone(), log);
        let events = capture.events.lock().unwrap();
        events.clone()
    }

    /// The value of `name` in one line's fields, without the `str:` or
    /// `debug:` prefix that says how it was recorded.
    pub(crate) fn field<'a>(line: &'a [(String, String)], name: &str) -> Option<&'a str> {
        line.iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value.split_once(':').map_or(value.as_str(), |(_, v)| v))
    }

    /// The optional fields of every line `log` wrote, flattened.
    fn recorded_fields(log: impl FnOnce()) -> Vec<(String, String)> {
        recorded_events(log)
            .into_iter()
            .flatten()
            .filter(|(name, _)| {
                ["resource", "request_id", "details", "status"].contains(&name.as_str())
            })
            .collect()
    }

    fn mutation_context() -> AuditContext<'static> {
        AuditContext::new(
            "admin",
            "api_key",
            Role::Admin,
            "POST",
            "/workers",
            Some("req-7"),
        )
    }

    #[test]
    fn the_outcome_line_carries_the_status_and_the_resource_as_plain_values() {
        let logger = AuditLogger::new(true);

        let events = recorded_events(|| {
            logger.log_outcome(
                &mutation_context(),
                StatusCode::ACCEPTED,
                Some("worker-123"),
            );
        });

        assert_eq!(events.len(), 1, "one line per outcome: {events:?}");
        let line = &events[0];
        assert_eq!(
            field(line, "message"),
            Some("control_plane_mutation"),
            "{line:?}"
        );
        assert_eq!(field(line, "status"), Some("202"), "{line:?}");
        assert_eq!(field(line, "outcome"), Some("success"), "{line:?}");
        assert_eq!(field(line, "resource"), Some("worker-123"), "{line:?}");
        assert_eq!(field(line, "request_id"), Some("req-7"), "{line:?}");
        assert_eq!(field(line, "principal"), Some("admin"), "{line:?}");
        assert!(
            !line.iter().any(|(_, value)| value.contains("Some(")),
            "Debug text in the outcome line: {line:?}"
        );
    }

    #[test]
    fn a_failed_mutation_is_logged_as_failed_with_its_status_and_no_resource() {
        let logger = AuditLogger::new(true);

        let events = recorded_events(|| {
            logger.log_outcome(&mutation_context(), StatusCode::BAD_REQUEST, None);
        });

        let line = &events[0];
        assert_eq!(field(line, "status"), Some("400"), "{line:?}");
        assert_eq!(field(line, "outcome"), Some("failed"), "{line:?}");
        assert_eq!(field(line, "resource"), None, "{line:?}");
    }

    #[test]
    fn the_admission_line_carries_no_status() {
        let logger = AuditLogger::new(true);

        let events = recorded_events(|| logger.log_success(&mutation_context(), None));

        let line = &events[0];
        assert_eq!(
            field(line, "message"),
            Some("control_plane_audit"),
            "{line:?}"
        );
        assert_eq!(field(line, "status"), None, "{line:?}");
    }

    #[test]
    fn a_disabled_logger_writes_no_outcome_line() {
        let logger = AuditLogger::new(false);

        let events = recorded_events(|| {
            logger.log_outcome(
                &mutation_context(),
                StatusCode::ACCEPTED,
                Some("worker-123"),
            );
        });

        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn optional_fields_are_logged_as_plain_values() {
        let logger = AuditLogger::new(true);
        let ctx = AuditContext::new(
            "user@example.com",
            "jwt",
            Role::Admin,
            "POST",
            "/workers",
            Some("req-abc"),
        );

        let fields = recorded_fields(|| logger.log_success(&ctx, Some("worker-123")));

        assert!(
            fields.contains(&("resource".to_string(), "str:worker-123".to_string())),
            "resource is not a plain string: {fields:?}"
        );
        assert!(
            fields.contains(&("request_id".to_string(), "str:req-abc".to_string())),
            "request_id is not a plain string: {fields:?}"
        );
        assert!(
            !fields.iter().any(|(_, value)| value.contains("Some(")),
            "Debug text in the audit line: {fields:?}"
        );
    }

    #[test]
    fn absent_optional_fields_are_left_out() {
        let logger = AuditLogger::new(true);

        let fields =
            recorded_fields(|| logger.log_auth_failure("GET", "/workers", "Invalid token", None));

        assert!(
            fields.contains(&("details".to_string(), "str:Invalid token".to_string())),
            "details is not a plain string: {fields:?}"
        );
        assert!(
            !fields
                .iter()
                .any(|(name, _)| name == "resource" || name == "request_id"),
            "absent fields were logged (as `None`): {fields:?}"
        );
    }

    #[test]
    fn test_sanitize_for_log_normal_input() {
        assert_eq!(sanitize_for_log("normal string"), "normal string");
        assert_eq!(sanitize_for_log("user@example.com"), "user@example.com");
        assert_eq!(sanitize_for_log("/api/v1/workers"), "/api/v1/workers");
    }

    #[test]
    fn test_sanitize_for_log_newlines() {
        assert_eq!(sanitize_for_log("line1\nline2"), "line1\\nline2");
        assert_eq!(sanitize_for_log("line1\r\nline2"), "line1\\r\\nline2");
    }

    #[test]
    fn test_sanitize_for_log_control_chars() {
        assert_eq!(sanitize_for_log("test\x00null"), "test\\x00null");
        assert_eq!(sanitize_for_log("test\x1bescape"), "test\\x1bescape");
    }

    #[test]
    fn test_sanitize_for_log_backslashes() {
        assert_eq!(sanitize_for_log("path\\to\\file"), "path\\\\to\\\\file");
    }

    #[test]
    fn test_sanitize_for_log_truncation() {
        let long_string = "a".repeat(2000);
        let sanitized = sanitize_for_log(&long_string);
        assert!(sanitized.len() < 2000);
        assert!(sanitized.ends_with("...[TRUNCATED]"));
    }

    #[test]
    fn test_audit_event_sanitizes_inputs() {
        let ctx = AuditContext::new(
            "user\ninjected",
            "jwt",
            Role::User,
            "GET",
            "/workers\r\nfake_log_entry",
            None,
        );
        let event = AuditEvent::from_context(
            &ctx,
            AuditOutcome::Denied,
            None,
            Some("error\x00with\x1bnull"),
        );

        // Verify sanitization was applied
        assert!(!event.principal.contains('\n'));
        assert!(event.principal.contains("\\n"));
        assert!(!event.path.contains('\r'));
        assert!(event.path.contains("\\r"));
        assert!(event.details.as_ref().unwrap().contains("\\x00"));
    }
}
