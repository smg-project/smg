//! MCP metrics for monitoring operations.

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

use crate::inventory::QualifiedToolName;

/// Metrics for MCP operations.
pub struct McpMetrics {
    // Call metrics
    total_calls: AtomicU64,
    successful_calls: AtomicU64,
    failed_calls: AtomicU64,

    // Approval metrics
    approvals_requested: AtomicU64,
    approvals_granted: AtomicU64,
    approvals_denied: AtomicU64,

    // Connection metrics
    connection_errors: AtomicU64,
    active_connections: AtomicU64,

    // Execution metrics
    active_executions: AtomicU64,

    // Per-tool latency tracking
    tool_latencies: DashMap<QualifiedToolName, LatencyStats>,
}

impl McpMetrics {
    /// Create a new metrics instance.
    pub fn new() -> Self {
        Self {
            total_calls: AtomicU64::new(0),
            successful_calls: AtomicU64::new(0),
            failed_calls: AtomicU64::new(0),
            approvals_requested: AtomicU64::new(0),
            approvals_granted: AtomicU64::new(0),
            approvals_denied: AtomicU64::new(0),
            connection_errors: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            active_executions: AtomicU64::new(0),
            tool_latencies: DashMap::new(),
        }
    }

    /// Record the start of a tool call.
    pub fn record_call_start(&self, _tool: &QualifiedToolName) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.active_executions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the end of a tool call.
    pub fn record_call_end(&self, tool: &QualifiedToolName, success: bool, duration_ms: u64) {
        self.active_executions.fetch_sub(1, Ordering::Relaxed);

        if success {
            self.successful_calls.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_calls.fetch_add(1, Ordering::Relaxed);
        }

        // Record latency
        self.tool_latencies
            .entry(tool.clone())
            .or_insert_with(LatencyStats::new)
            .record(duration_ms);
    }

    /// Record an approval request.
    pub fn record_approval_requested(&self) {
        self.approvals_requested.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an approval granted.
    pub fn record_approval_granted(&self) {
        self.approvals_granted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an approval denied.
    pub fn record_approval_denied(&self) {
        self.approvals_denied.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a connection error.
    pub fn record_connection_error(&self) {
        self.connection_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a connection opened.
    pub fn record_connection_opened(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a connection closed.
    pub fn record_connection_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_calls: self.total_calls.load(Ordering::Relaxed),
            successful_calls: self.successful_calls.load(Ordering::Relaxed),
            failed_calls: self.failed_calls.load(Ordering::Relaxed),
            approvals_requested: self.approvals_requested.load(Ordering::Relaxed),
            approvals_granted: self.approvals_granted.load(Ordering::Relaxed),
            approvals_denied: self.approvals_denied.load(Ordering::Relaxed),
            connection_errors: self.connection_errors.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            active_executions: self.active_executions.load(Ordering::Relaxed),
        }
    }

    /// Get latency stats for a specific tool.
    pub fn tool_latency(&self, tool: &QualifiedToolName) -> Option<LatencySnapshot> {
        self.tool_latencies.get(tool).map(|stats| stats.snapshot())
    }

    /// Get latency stats for all tools.
    pub fn all_tool_latencies(&self) -> Vec<(QualifiedToolName, LatencySnapshot)> {
        self.tool_latencies
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().snapshot()))
            .collect()
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.total_calls.store(0, Ordering::Relaxed);
        self.successful_calls.store(0, Ordering::Relaxed);
        self.failed_calls.store(0, Ordering::Relaxed);
        self.approvals_requested.store(0, Ordering::Relaxed);
        self.approvals_granted.store(0, Ordering::Relaxed);
        self.approvals_denied.store(0, Ordering::Relaxed);
        self.connection_errors.store(0, Ordering::Relaxed);
        // Don't reset active_connections or active_executions
        self.tool_latencies.clear();
    }
}

impl Default for McpMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-tool latency statistics.
pub struct LatencyStats {
    count: AtomicU64,
    total_ms: AtomicU64,
    min_ms: AtomicU64,
    max_ms: AtomicU64,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ms: AtomicU64::new(0),
            min_ms: AtomicU64::new(u64::MAX),
            max_ms: AtomicU64::new(0),
        }
    }

    fn record(&self, ms: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ms.fetch_add(ms, Ordering::Relaxed);

        // The extremes: relaxed ordering is fine for approximate stats.
        self.min_ms.fetch_min(ms, Ordering::Relaxed);
        self.max_ms.fetch_max(ms, Ordering::Relaxed);
    }

    fn snapshot(&self) -> LatencySnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let total = self.total_ms.load(Ordering::Relaxed);
        let min = self.min_ms.load(Ordering::Relaxed);
        let max = self.max_ms.load(Ordering::Relaxed);

        LatencySnapshot {
            count,
            avg_ms: total.checked_div(count).unwrap_or(0),
            min_ms: if min == u64::MAX { 0 } else { min },
            max_ms: max,
        }
    }
}

/// Snapshot of overall metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub approvals_requested: u64,
    pub approvals_granted: u64,
    pub approvals_denied: u64,
    pub connection_errors: u64,
    /// As the orchestrator records it: connects of configured servers, less
    /// one at shutdown for each that connected; a reconnect adds one more. A
    /// server a request names by `server_url` is pooled and not counted here.
    pub active_connections: u64,
    pub active_executions: u64,
}

impl MetricsSnapshot {
    /// Calculate success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let completed = self.successful_calls + self.failed_calls;
        if completed == 0 {
            100.0
        } else {
            (self.successful_calls as f64 / completed as f64) * 100.0
        }
    }

    /// Calculate approval rate as a percentage.
    pub fn approval_rate(&self) -> f64 {
        let total = self.approvals_granted + self.approvals_denied;
        if total == 0 {
            100.0
        } else {
            (self.approvals_granted as f64 / total as f64) * 100.0
        }
    }
}

/// Snapshot of latency statistics for a tool.
#[derive(Debug, Clone)]
pub struct LatencySnapshot {
    pub count: u64,
    pub avg_ms: u64,
    pub min_ms: u64,
    pub max_ms: u64,
}

#[cfg(test)]
mod tests {
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
}
