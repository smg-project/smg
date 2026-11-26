use super::*;
use crate::approval::{audit::AuditLog, policy::PolicyEngine};

fn test_handler() -> SmgClientHandler {
    let audit_log = Arc::new(AuditLog::new());
    let policy_engine = Arc::new(PolicyEngine::new(audit_log.clone()));
    let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

    SmgClientHandler::new("test-server", approval_manager)
}

#[test]
fn test_handler_creation() {
    let handler = test_handler();
    assert_eq!(handler.server_key(), "test-server");
    assert!(handler.request_context().is_none());
}

#[test]
fn test_request_context() {
    let handler = test_handler();

    let ctx = HandlerRequestContext::new(
        "req-1",
        ApprovalMode::PolicyOnly,
        TenantContext::new("tenant-1"),
        HashMap::new(),
    );

    handler.set_request_context(ctx.clone());
    assert!(handler.request_context().is_some());

    let retrieved = handler.request_context().unwrap();
    assert_eq!(retrieved.request_id, "req-1");

    handler.clear_request_context();
    assert!(handler.request_context().is_none());
}

#[test]
fn test_client_info() {
    let handler = test_handler();
    let info = handler.get_info();
    assert_eq!(info.client_info.name, "smg");
}

#[test]
fn test_with_refresh_channel() {
    let (tx, _rx) = mpsc::channel(10);
    let handler = test_handler().with_refresh_channel(tx);
    assert!(handler.refresh_tx.is_some());
}
