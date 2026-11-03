//! Compact serializable snapshot of a radix tree.
//!
//! The structure-preserving format shares common prefixes. Current mesh
//! repair uses paginated entries instead; this format supports saving and
//! restoring whole string trees.
//!
//! Wire format: flattened pre-order node list.  Each node stores its
//! edge label and tenant list.  Children are implicitly ordered by
//! `child_count` (parent before children in the list).

use serde::{Deserialize, Serialize};

/// Compact serializable snapshot of a radix tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TreeSnapshot {
    /// Flattened tree nodes in pre-order (parent before children).
    pub nodes: Vec<SnapshotNode>,
}

/// A single node in the serialized tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotNode {
    /// Edge label — the text on this edge from parent.
    /// For string trees: UTF-8 string of the shared prefix segment.
    /// For token trees: will use a separate type.
    pub edge: String,
    /// Tenants (workers) that have this prefix cached,
    /// with their access stamp: seconds in the high 32 bits, sequence in
    /// the low 32 bits. A zero high half carries no access time.
    pub tenants: Vec<(String, u64)>,
    /// Number of children (for tree reconstruction from flat list).
    pub child_count: u32,
}

impl TreeSnapshot {
    /// Create an empty snapshot.
    pub fn empty() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Serialize to bincode bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<bincode::ErrorKind>> {
        bincode::serialize(self)
    }

    /// Deserialize from bincode bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<bincode::ErrorKind>> {
        bincode::deserialize(bytes)
    }

    /// Number of nodes in the snapshot.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total serialized edge bytes (approximate wire size without overhead).
    pub fn total_edge_bytes(&self) -> usize {
        self.nodes.iter().map(|n| n.edge.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_round_trip() {
        let snapshot = TreeSnapshot {
            nodes: vec![
                SnapshotNode {
                    edge: String::new(),
                    tenants: vec![("worker-1".to_string(), 100)],
                    child_count: 2,
                },
                SnapshotNode {
                    edge: "Hello ".to_string(),
                    tenants: vec![("worker-1".to_string(), 100)],
                    child_count: 1,
                },
                SnapshotNode {
                    edge: "world".to_string(),
                    tenants: vec![("worker-1".to_string(), 100)],
                    child_count: 0,
                },
                SnapshotNode {
                    edge: "Goodbye".to_string(),
                    tenants: vec![("worker-2".to_string(), 200)],
                    child_count: 0,
                },
            ],
        };

        let bytes = snapshot.to_bytes().unwrap();
        let restored = TreeSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snapshot, restored);
    }

    #[test]
    fn test_empty_snapshot() {
        let snapshot = TreeSnapshot::empty();
        assert_eq!(snapshot.node_count(), 0);
        assert_eq!(snapshot.total_edge_bytes(), 0);

        let bytes = snapshot.to_bytes().unwrap();
        let restored = TreeSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snapshot, restored);
    }
}
