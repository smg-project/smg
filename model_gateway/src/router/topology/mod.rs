//! Topology‑aware routing module for SMG.
//!
//! This module provides a high‑performance, lock‑free routing policy
//! that uses prefix hashing and weighted Rendezvous hashing.

/// Module for extracting conversation prefixes from JSON.

pub mod extractor;
/// Weighted Rendezvous (HRW) hashing router.
pub mod consistent_hash;
pub mod prefix_cache;
/// Error types for the topology module.
pub mod error;
/// LRU prefix cache with reverse indexing.

pub use extractor::extract_conversation_prefix;
pub use consistent_hash::{RendezvousRouter, WorkerNode};
pub use prefix_cache::PrefixCache;
pub use error::TopologyError;

use std::sync::Arc;
use xxhash_rust::xxh64::xxh64;

pub struct TopologyRouter {
    prefix_cache: PrefixCache,
    hrw_router: RendezvousRouter,
}

impl TopologyRouter {
    pub fn new(workers: Vec<WorkerNode>, cache_capacity: usize) -> Self {
        Self {
            prefix_cache: PrefixCache::new(cache_capacity),
            hrw_router: RendezvousRouter::new(workers),
        }
    }

    pub fn route(&self, json_bytes: &[u8]) -> Result<&WorkerNode, TopologyError> {
        let prefix = extract_conversation_prefix(json_bytes)?;
        let prefix_hash = xxh64(prefix.as_bytes(), 0);

        if let Some(worker_id) = self.prefix_cache.lookup(prefix_hash) {
            for worker in self.hrw_router.workers() {
                if worker.id == *worker_id {
                    return Ok(worker);
                }
            }
            self.prefix_cache.remove_worker(&worker_id);
        }

        let worker = self.hrw_router.route(prefix_hash)?;
        self.prefix_cache.insert(prefix_hash, Arc::from(worker.id.as_str()));
        Ok(worker)
    }

    pub fn register_worker_cache(&self, prefix: &str, worker_id: &str) {
        let hash = xxh64(prefix.as_bytes(), 0);
        self.prefix_cache.insert(hash, Arc::from(worker_id));
    }

    pub fn remove_worker(&self, worker_id: &str) {
        self.prefix_cache.remove_worker(worker_id);
    }

    
    pub fn set_worker_load(&self, worker_id: &str, load: f32) {
        for worker in self.hrw_router.workers() {
            if worker.id == worker_id {
                worker.set_load(load);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_single_turn() {
        let json = br#"{"messages": [{"role": "user", "content": "Hello"}]}"#;
        let workers = vec![
            WorkerNode::new("w1".into(), "1".into(), 0.0),
        ];
        let router = TopologyRouter::new(workers, 10);
        let result = router.route(json);
        assert!(result.is_ok());
    }

    #[test]
    fn route_with_prefix() {
        let json = br#"{
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ]
        }"#;
        let workers = vec![
            WorkerNode::new("w1".into(), "1".into(), 0.0),
        ];
        let router = TopologyRouter::new(workers, 10);
        let result = router.route(json);
        assert!(result.is_ok());
    }
}