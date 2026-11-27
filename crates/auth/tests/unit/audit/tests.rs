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

            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
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
