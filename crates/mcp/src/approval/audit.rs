//! Audit logging for MCP approval decisions.

use std::{collections::VecDeque, sync::Arc};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{inventory::QualifiedToolName, tenant::TenantId};

const DEFAULT_MAX_ENTRIES: usize = 10000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionSource {
    UserInteractive,
    PolicyEngine,
    ExplicitToolPolicy,
    ServerPolicy,
    RuleMatch,
    AnnotationDefault,
    GlobalDefault,
    Timeout,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionResult {
    Approved,
    Denied { reason: String },
    Pending,
    TimedOut,
}

impl DecisionResult {
    pub fn is_approved(&self) -> bool {
        matches!(self, DecisionResult::Approved)
    }

    pub fn is_final(&self) -> bool {
        !matches!(self, DecisionResult::Pending)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Arc<str>,
    pub timestamp: DateTime<Utc>,
    pub tenant_id: TenantId,
    pub request_id: Arc<str>,
    pub server_key: Arc<str>,
    pub tool_name: Arc<str>,
    pub result: DecisionResult,
    pub source: DecisionSource,
}

impl AuditEntry {
    pub fn new(
        tenant_id: TenantId,
        request_id: Arc<str>,
        server_key: Arc<str>,
        tool_name: Arc<str>,
        result: DecisionResult,
        source: DecisionSource,
    ) -> Self {
        Self {
            id: Arc::from(uuid::Uuid::now_v7().to_string()),
            timestamp: Utc::now(),
            tenant_id,
            request_id,
            server_key,
            tool_name,
            result,
            source,
        }
    }

    pub fn qualified_name(&self) -> QualifiedToolName {
        QualifiedToolName::new(self.server_key.as_ref(), self.tool_name.as_ref())
    }
}

#[derive(Debug)]
pub struct AuditLog {
    entries: RwLock<VecDeque<AuditEntry>>,
    max_entries: usize,
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditLog {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_ENTRIES)
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(VecDeque::with_capacity(max_entries)),
            max_entries,
        }
    }

    pub fn record(&self, entry: AuditEntry) {
        let mut entries = self.entries.write();
        if entries.len() >= self.max_entries {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    pub fn record_decision(
        &self,
        qualified_name: &QualifiedToolName,
        tenant_id: &TenantId,
        request_id: &str,
        result: DecisionResult,
        source: DecisionSource,
    ) {
        self.record(AuditEntry::new(
            tenant_id.clone(),
            Arc::from(request_id),
            Arc::from(qualified_name.server_key()),
            Arc::from(qualified_name.tool_name()),
            result,
            source,
        ));
    }

    pub fn recent(&self, limit: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read();
        entries.iter().rev().take(limit).cloned().collect()
    }

    pub fn for_tenant(&self, tenant_id: &TenantId, limit: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read();
        entries
            .iter()
            .rev()
            .filter(|e| &e.tenant_id == tenant_id)
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn for_request(&self, request_id: &str, limit: Option<usize>) -> Vec<AuditEntry> {
        let entries = self.entries.read();
        entries
            .iter()
            .rev()
            .filter(|e| e.request_id.as_ref() == request_id)
            .take(limit.unwrap_or(usize::MAX))
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/approval/audit/tests.rs"]
mod tests;
