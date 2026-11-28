//! Tool annotations for approval decisions.
//!
//! We maintain [`ToolAnnotations`] separate from [`rmcp::model::ToolAnnotations`] because:
//! - rmcp uses `Option<bool>` requiring unwrapping everywhere
//! - We use `bool` with conservative defaults (destructive=true, read_only=false)

use rmcp::model::ToolAnnotations as RmcpToolAnnotations;
use serde::{Deserialize, Serialize};

/// Tool behavior hints for approval decisions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    pub read_only: bool,
    pub destructive: bool,
    pub idempotent: bool,
    pub open_world: bool,
}

impl ToolAnnotations {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert from rmcp's optional annotations with conservative defaults.
    pub fn from_rmcp(rmcp: &RmcpToolAnnotations) -> Self {
        Self {
            read_only: rmcp.read_only_hint.unwrap_or(false),
            destructive: rmcp.destructive_hint.unwrap_or(true),
            idempotent: rmcp.idempotent_hint.unwrap_or(false),
            open_world: rmcp.open_world_hint.unwrap_or(true),
        }
    }

    pub fn from_rmcp_option(rmcp: Option<&RmcpToolAnnotations>) -> Self {
        rmcp.map(Self::from_rmcp).unwrap_or_default()
    }

    #[must_use]
    pub fn with_read_only(mut self, v: bool) -> Self {
        self.read_only = v;
        self
    }

    #[must_use]
    pub fn with_destructive(mut self, v: bool) -> Self {
        self.destructive = v;
        self
    }

    #[must_use]
    pub fn with_open_world(mut self, v: bool) -> Self {
        self.open_world = v;
        self
    }

    #[must_use]
    pub fn should_require_approval(&self) -> bool {
        self.destructive && !self.read_only
    }
}

/// Annotation types for pattern matching in policy rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnnotationType {
    Destructive,
    ReadOnly,
    Idempotent,
    OpenWorld,
}

impl AnnotationType {
    pub fn matches(&self, annotations: &ToolAnnotations) -> bool {
        match self {
            AnnotationType::Destructive => annotations.destructive,
            AnnotationType::ReadOnly => annotations.read_only,
            AnnotationType::Idempotent => annotations.idempotent,
            AnnotationType::OpenWorld => annotations.open_world,
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/annotations/tests.rs"]
mod tests;
