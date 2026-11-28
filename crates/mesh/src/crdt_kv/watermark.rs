use std::collections::HashMap;

use super::{operation::Operation, replica::ReplicaId};

/// A CRDT write's identity and order: `(Lamport timestamp, replica_id)`. This
/// is exactly the key LWW uses to pick a winner (`max_by_key((timestamp,
/// replica_id))`), so a watermark built from it advances in lockstep with the
/// merge — including the replica tie-break when two writes share a timestamp.
pub type CrdtVersion = (u64, ReplicaId);

/// Per-peer CRDT send watermark, keyed by KEY (not by author replica). Tracks
/// the op-id `(timestamp, replica_id)` a peer has acknowledged for each key, so
/// the sender can skip re-broadcasting keys the peer already has.
///
/// Keying by key — not by replica — keeps this correct on our compacted,
/// multi-author op-log: each key is tracked independently, so a dropped or late
/// op only delays that one key (resent next round) and can never be hidden
/// behind a higher version on a different key.
///
/// The op-id, not just the timestamp, is the watermark value: equal timestamps
/// from different replicas are ordered by `replica_id` (the LWW tie-break), so
/// a same-timestamp winner change must still be sent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CrdtWatermark {
    versions: HashMap<String, CrdtVersion>,
}

impl CrdtWatermark {
    /// Empty watermark — every op is allowed (a fresh peer gets the full set).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a set of ops, keeping the max op-id per key. Used by the
    /// receiver to ack the versions it just merged. Clones a key only when
    /// first seen, so allocation is O(distinct keys), not O(ops).
    pub fn from_ops(ops: &[Operation]) -> Self {
        let mut versions: HashMap<String, CrdtVersion> = HashMap::new();
        for op in ops {
            let id = (op.timestamp(), op.replica_id());
            match versions.get_mut(op.key()) {
                Some(existing) => *existing = (*existing).max(id),
                None => {
                    versions.insert(op.key().to_owned(), id);
                }
            }
        }
        Self { versions }
    }

    /// True if `op` is a newer write than what the peer has acked for its key,
    /// by `(timestamp, replica_id)` order. An unacked key always sends.
    pub fn allows(&self, op: &Operation) -> bool {
        match self.get(op.key()) {
            Some(acked) => (op.timestamp(), op.replica_id()) > acked,
            None => true,
        }
    }

    /// The acked op-id for `key`, if the peer has acked it.
    pub fn get(&self, key: &str) -> Option<CrdtVersion> {
        self.versions.get(key).copied()
    }

    /// Advance toward `other`, taking the per-key maximum op-id. Monotone,
    /// idempotent, and commutative, so out-of-order or duplicate acks are
    /// self-correcting.
    pub fn merge_max(&mut self, other: &CrdtWatermark) {
        for (key, &id) in &other.versions {
            match self.versions.get_mut(key) {
                Some(existing) => *existing = (*existing).max(id),
                None => {
                    self.versions.insert(key.clone(), id);
                }
            }
        }
    }

    /// Advance from owned `(key, op-id)` pairs in place, taking the key only
    /// when inserting a new entry. Avoids cloning keys / a temp map when
    /// merging a received ack on the hot path.
    pub fn merge_max_owned<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (String, CrdtVersion)>,
    {
        for (key, id) in iter {
            match self.versions.get_mut(&key) {
                Some(existing) => *existing = (*existing).max(id),
                None => {
                    self.versions.insert(key, id);
                }
            }
        }
    }

    /// Iterate `(key, op-id)` pairs — used by the wire codec.
    pub fn iter(&self) -> impl Iterator<Item = (&str, CrdtVersion)> {
        self.versions.iter().map(|(k, &id)| (k.as_str(), id))
    }

    /// True if no key has been acked yet.
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

impl FromIterator<(String, CrdtVersion)> for CrdtWatermark {
    /// Build from `(key, op-id)` pairs, keeping the max op-id per key. Used by
    /// the wire codec to rebuild a watermark from a received ack.
    fn from_iter<I: IntoIterator<Item = (String, CrdtVersion)>>(iter: I) -> Self {
        let mut watermark = Self::new();
        watermark.merge_max_owned(iter);
        watermark
    }
}

#[cfg(test)]
#[path = "../../tests/unit/crdt_kv/watermark/tests.rs"]
mod tests;
