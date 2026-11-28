// ============================================================================
// CRDT OR-Map - High-Performance Transparent CRDT KV Storage
// ============================================================================

mod crdt;
mod engine;
mod epoch_max_wins;
mod kv_store;
mod merge_strategy;
mod operation;
mod replica;
mod watermark;

// Export core types
pub use crdt::CrdtOrMap;
pub use epoch_max_wins::{decode, encode, EpochCount, EPOCH_MAX_WINS_ENCODED_LEN};
pub use merge_strategy::MergeStrategy;
pub use operation::{CrdtChange, Operation, OperationLog};
pub use replica::ReplicaId;
pub use watermark::CrdtWatermark;

#[cfg(test)]
#[path = "../../tests/unit/crdt_kv/tests.rs"]
mod tests;
