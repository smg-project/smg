use super::*;

fn test_manager() -> ApprovalManager {
    let audit_log = Arc::new(AuditLog::new());
    let policy_engine = Arc::new(PolicyEngine::new(audit_log.clone()));
    ApprovalManager::new(policy_engine, audit_log)
}

/// The approval parameters of the request `req-1` for `tool_name` on
/// `server` (the elicitation `elicit-1`), with `message` for the user,
/// under `hints` and `tenant`.
fn approval_params<'a>(
    tool_name: &'a str,
    message: &'a str,
    hints: &'a ToolAnnotations,
    tenant: &'a TenantContext,
) -> ApprovalParams<'a> {
    ApprovalParams {
        request_id: "req-1",
        server_key: "server",
        elicitation_id: "elicit-1",
        tool_name,
        hints,
        message,
        tenant_ctx: tenant,
    }
}

#[tokio::test]
async fn test_policy_only_mode() {
    let manager = test_manager();
    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new().with_read_only(true);

    let params = approval_params("read_tool", "Allow read?", &hints, &tenant);

    let outcome = manager
        .handle_approval(ApprovalMode::PolicyOnly, params)
        .await
        .unwrap();

    match outcome {
        ApprovalOutcome::Decided(decision) => assert!(decision.is_allowed()),
        ApprovalOutcome::Pending { .. } => panic!("Expected decided outcome"),
    }
}

#[tokio::test]
async fn test_interactive_mode_pending() {
    let manager = test_manager();
    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new().with_destructive(true);

    let params = approval_params("delete_tool", "Allow delete?", &hints, &tenant);

    let outcome = manager
        .handle_approval(ApprovalMode::Interactive, params)
        .await
        .unwrap();

    match outcome {
        ApprovalOutcome::Pending {
            key,
            approval_request,
            ..
        } => {
            assert_eq!(key.request_id, "req-1");
            assert_eq!(approval_request.tool_name, "delete_tool");
            assert!(manager.has_pending(&key));
        }
        ApprovalOutcome::Decided(_) => panic!("Expected pending outcome"),
    }
}

#[tokio::test]
async fn test_interactive_resolve() {
    let manager = test_manager();
    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new();

    let params = approval_params("tool", "Allow?", &hints, &tenant);

    let outcome = manager
        .handle_approval(ApprovalMode::Interactive, params)
        .await
        .unwrap();

    let rx = match outcome {
        ApprovalOutcome::Pending { rx, .. } => rx,
        ApprovalOutcome::Decided(_) => panic!("Expected pending"),
    };

    manager
        .resolve("req-1", "server", "elicit-1", true, None, &tenant)
        .await
        .unwrap();

    let decision = rx.await.unwrap();
    assert!(decision.is_approved());
    assert_eq!(manager.pending_count(), 0);
}

#[tokio::test]
async fn test_resolve_not_found() {
    let manager = test_manager();
    let tenant = TenantContext::new("test");

    let result = manager
        .resolve("nonexistent", "server", "elicit", true, None, &tenant)
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_duplicate_pending() {
    let manager = test_manager();
    let tenant = TenantContext::new("test");
    let hints = ToolAnnotations::new();

    let params = approval_params("tool", "Allow?", &hints, &tenant);

    manager
        .handle_approval(ApprovalMode::Interactive, params)
        .await
        .unwrap();

    let params2 = approval_params("tool", "Allow?", &hints, &tenant);

    let result = manager
        .handle_approval(ApprovalMode::Interactive, params2)
        .await;
    assert_eq!(
        result.unwrap_err().to_string(),
        "Approval error: Approval already pending: elicit-1"
    );
}

#[test]
fn test_evict_expired() {
    let manager = test_manager().with_timeout(Duration::from_millis(1));
    manager.evict_expired();
    assert_eq!(manager.pending_count(), 0);
}
