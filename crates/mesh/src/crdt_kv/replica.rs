use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Replica Identity - Globally Unique Node Identity
// ============================================================================

/// Replica ID, using UUID to ensure global uniqueness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ReplicaId(Uuid);

impl ReplicaId {
    /// Greatest possible replica id. As a watermark tie-breaker it makes a
    /// `(timestamp, MAX)` entry suppress every op at or below that timestamp —
    /// the timestamp-only semantics legacy acks (no `replica_id`) speak.
    pub const MAX: Self = Self(Uuid::max());

    /// Generate a new replica ID
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// Parse replica ID from string
    pub fn from_string(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

impl Default for ReplicaId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ReplicaId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================
// Lamport Clock - Causal Ordering Guarantee
// ============================================================================

/// Lamport logical clock, used to establish causal ordering of operations
#[derive(Debug)]
pub struct LamportClock {
    counter: AtomicU64,
}

impl Clone for LamportClock {
    fn clone(&self) -> Self {
        Self {
            counter: AtomicU64::new(self.counter.load(Ordering::Acquire)),
        }
    }
}

impl LamportClock {
    /// Create a new Lamport clock
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }

    /// Increment and return the new timestamp: an update from a clock at zero.
    pub fn tick(&self) -> u64 {
        self.update(0)
    }

    /// Update clock to max(local, remote) + 1
    pub fn update(&self, remote_timestamp: u64) -> u64 {
        let mut current = self.counter.load(Ordering::Acquire);
        loop {
            let new_value = current.max(remote_timestamp).saturating_add(1);
            match self.counter.compare_exchange(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return new_value,
                Err(actual) => current = actual,
            }
        }
    }
}

impl Default for LamportClock {
    fn default() -> Self {
        Self::new()
    }
}
