use super::*;

#[test]
fn test_audit_log_record() {
    let log = AuditLog::new();
    let tenant = TenantId::new("test");

    log.record_decision(
        &QualifiedToolName::new("server", "tool"),
        &tenant,
        "req-1",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );

    assert_eq!(log.len(), 1);
    let recent = log.recent(10);
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].tool_name.as_ref(), "tool");
}

#[test]
fn test_audit_log_max_entries() {
    let log = AuditLog::with_capacity(5);
    let tenant = TenantId::new("test");

    for i in 0..10 {
        log.record_decision(
            &QualifiedToolName::new("server", format!("tool-{i}")),
            &tenant,
            &format!("req-{i}"),
            DecisionResult::Approved,
            DecisionSource::PolicyEngine,
        );
    }

    assert_eq!(log.len(), 5);
    let recent = log.recent(10);
    assert_eq!(recent[0].tool_name.as_ref(), "tool-9");
    assert_eq!(recent[4].tool_name.as_ref(), "tool-5");
}

#[test]
fn test_filter_by_tenant() {
    let log = AuditLog::new();
    let tenant1 = TenantId::new("tenant1");
    let tenant2 = TenantId::new("tenant2");
    let name = QualifiedToolName::new("server", "tool");

    log.record_decision(
        &name,
        &tenant1,
        "r1",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );
    log.record_decision(
        &name,
        &tenant2,
        "r2",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );
    log.record_decision(
        &name,
        &tenant1,
        "r3",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );

    let t1_entries = log.for_tenant(&tenant1, 10);
    assert_eq!(t1_entries.len(), 2);
}

#[test]
fn test_for_request_reverse_order_no_limit() {
    let log = AuditLog::new();
    let tenant = TenantId::new("test");
    let name = QualifiedToolName::new("server", "tool");

    log.record_decision(
        &name,
        &tenant,
        "req-1",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );
    log.record_decision(
        &name,
        &tenant,
        "req-2",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );
    log.record_decision(
        &name,
        &tenant,
        "req-1",
        DecisionResult::Denied {
            reason: "policy violation".into(),
        },
        DecisionSource::RuleMatch,
    );
    log.record_decision(
        &name,
        &tenant,
        "req-1",
        DecisionResult::Pending,
        DecisionSource::UserInteractive,
    );

    let entries = log.for_request("req-1", None);
    assert_eq!(entries.len(), 3);
    // Newest first (reverse chronological)
    assert!(matches!(entries[0].result, DecisionResult::Pending));
    assert!(matches!(entries[1].result, DecisionResult::Denied { .. }));
    assert!(matches!(entries[2].result, DecisionResult::Approved));
}

#[test]
fn test_for_request_with_limit() {
    let log = AuditLog::new();
    let tenant = TenantId::new("test");
    let name = QualifiedToolName::new("server", "tool");

    log.record_decision(
        &name,
        &tenant,
        "req-1",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );
    log.record_decision(
        &name,
        &tenant,
        "req-1",
        DecisionResult::Denied {
            reason: "denied".into(),
        },
        DecisionSource::RuleMatch,
    );
    log.record_decision(
        &name,
        &tenant,
        "req-1",
        DecisionResult::Pending,
        DecisionSource::UserInteractive,
    );

    // Limit to 1 returns only the newest entry
    let entries = log.for_request("req-1", Some(1));
    assert_eq!(entries.len(), 1);
    assert!(matches!(entries[0].result, DecisionResult::Pending));

    // Limit to 2 returns the two newest entries
    let entries = log.for_request("req-1", Some(2));
    assert_eq!(entries.len(), 2);
    assert!(matches!(entries[0].result, DecisionResult::Pending));
    assert!(matches!(entries[1].result, DecisionResult::Denied { .. }));
}

#[test]
fn test_for_request_with_zero_limit() {
    let log = AuditLog::new();
    let tenant = TenantId::new("test");
    let name = QualifiedToolName::new("server", "tool");

    log.record_decision(
        &name,
        &tenant,
        "req-1",
        DecisionResult::Approved,
        DecisionSource::PolicyEngine,
    );

    let entries = log.for_request("req-1", Some(0));
    assert!(entries.is_empty());
}
