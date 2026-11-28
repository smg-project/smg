use super::*;

#[test]
fn test_call_metrics() {
    let metrics = McpMetrics::new();
    let tool = QualifiedToolName::new("server", "tool");

    metrics.record_call_start(&tool);
    assert_eq!(metrics.snapshot().total_calls, 1);
    assert_eq!(metrics.snapshot().active_executions, 1);

    metrics.record_call_end(&tool, true, 100);
    assert_eq!(metrics.snapshot().successful_calls, 1);
    assert_eq!(metrics.snapshot().active_executions, 0);

    metrics.record_call_start(&tool);
    metrics.record_call_end(&tool, false, 50);
    assert_eq!(metrics.snapshot().failed_calls, 1);
}

#[test]
fn test_approval_metrics() {
    let metrics = McpMetrics::new();

    metrics.record_approval_requested();
    metrics.record_approval_granted();
    metrics.record_approval_requested();
    metrics.record_approval_denied();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.approvals_requested, 2);
    assert_eq!(snapshot.approvals_granted, 1);
    assert_eq!(snapshot.approvals_denied, 1);
}

#[test]
fn test_connection_metrics() {
    let metrics = McpMetrics::new();

    metrics.record_connection_opened();
    metrics.record_connection_opened();
    assert_eq!(metrics.snapshot().active_connections, 2);

    metrics.record_connection_closed();
    assert_eq!(metrics.snapshot().active_connections, 1);

    metrics.record_connection_error();
    assert_eq!(metrics.snapshot().connection_errors, 1);
}

#[test]
fn test_latency_stats() {
    let metrics = McpMetrics::new();
    let tool = QualifiedToolName::new("server", "tool");

    metrics.record_call_start(&tool);
    metrics.record_call_end(&tool, true, 100);

    metrics.record_call_start(&tool);
    metrics.record_call_end(&tool, true, 200);

    metrics.record_call_start(&tool);
    metrics.record_call_end(&tool, true, 150);

    let latency = metrics.tool_latency(&tool).unwrap();
    assert_eq!(latency.count, 3);
    assert_eq!(latency.avg_ms, 150); // (100 + 200 + 150) / 3
    assert_eq!(latency.min_ms, 100);
    assert_eq!(latency.max_ms, 200);
}

#[test]
fn test_success_rate() {
    let metrics = McpMetrics::new();
    let tool = QualifiedToolName::new("server", "tool");

    // 3 successful, 1 failed = 75%
    for _ in 0..3 {
        metrics.record_call_start(&tool);
        metrics.record_call_end(&tool, true, 100);
    }
    metrics.record_call_start(&tool);
    metrics.record_call_end(&tool, false, 100);

    let snapshot = metrics.snapshot();
    assert!((snapshot.success_rate() - 75.0).abs() < 0.1);
}

#[test]
fn test_reset() {
    let metrics = McpMetrics::new();
    let tool = QualifiedToolName::new("server", "tool");

    metrics.record_call_start(&tool);
    metrics.record_call_end(&tool, true, 100);
    metrics.record_approval_requested();

    metrics.reset();

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.total_calls, 0);
    assert_eq!(snapshot.approvals_requested, 0);
}

#[test]
fn test_all_tool_latencies() {
    let metrics = McpMetrics::new();
    let tool1 = QualifiedToolName::new("server1", "tool1");
    let tool2 = QualifiedToolName::new("server2", "tool2");

    metrics.record_call_start(&tool1);
    metrics.record_call_end(&tool1, true, 100);

    metrics.record_call_start(&tool2);
    metrics.record_call_end(&tool2, true, 200);

    let latencies = metrics.all_tool_latencies();
    assert_eq!(latencies.len(), 2);
}
