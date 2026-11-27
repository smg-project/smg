//! Tenant context for multi-tenant MCP operations.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Unique identifier for a tenant.
///
/// Uses `Arc<str>` internally for cheap cloning in hot paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantId(Arc<str>);

impl TenantId {
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TenantId {
    fn default() -> Self {
        Self(Arc::from("default"))
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        Self(Arc::from(s))
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(Arc::from(s))
    }
}

impl Serialize for TenantId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TenantId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(Arc::from(s)))
    }
}

/// Unique identifier for a session within a tenant.
///
/// Uses `Arc<str>` internally for cheap cloning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(Arc<str>);

impl SessionId {
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self(Arc::from(uuid::Uuid::now_v7().to_string()))
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for SessionId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(Arc::from(s)))
    }
}

#[derive(Debug, Clone, Default)]
pub struct TenantContext {
    pub tenant_id: TenantId,
    pub session_id: SessionId,
}

impl TenantContext {
    pub fn new(tenant_id: impl Into<TenantId>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn with_session(mut self, session_id: impl AsRef<str>) -> Self {
        self.session_id = SessionId::new(session_id);
        self
    }
}

#[cfg(test)]
#[path = "../tests/unit/tenant/tests.rs"]
mod tests;
