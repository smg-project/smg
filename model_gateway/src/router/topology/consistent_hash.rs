// model_gateway/src/router/topology/consistent_hash.rs

use std::sync::atomic::{AtomicU32, Ordering};
use xxhash_rust::xxh64::Xxh64;
use crate::router::topology::error::TopologyError;

/// A worker node in the Rendezvous hashing ring.
#[derive(Debug)]
pub struct WorkerNode {
    /// Unique identifier for the worker.
    pub id: String,
    /// Network address of the worker.
    pub address: String,
    /// Current load of the worker (0.0 to 1.0), stored as atomic bits.
    load_raw: AtomicU32,
}

impl WorkerNode {
    /// Creates a new worker node with the given ID, address, and initial load.
    pub fn new(id: String, address: String, load: f32) -> Self {
        Self {
            id,
            address,
            load_raw: AtomicU32::new(load.to_bits()),
        }
    }

    /// Returns the current load of this worker (0.0 to 1.0).
    pub fn load(&self) -> f32 {
        f32::from_bits(self.load_raw.load(Ordering::Relaxed))
    }

    /// Updates the load of this worker.
    pub fn set_load(&self, load: f32) {
        self.load_raw.store(load.to_bits(), Ordering::Relaxed);
    }
}

impl Clone for WorkerNode {
    fn clone(&self) -> Self {
        Self::new(self.id.clone(), self.address.clone(), self.load())
    }
}

/// A Rendezvous (HRW) hashing router that distributes keys to workers.
pub struct RendezvousRouter {
    workers: Vec<WorkerNode>,
}

impl RendezvousRouter {
    /// Creates a new router with the given list of workers.
    pub fn new(workers: Vec<WorkerNode>) -> Self {
        Self { workers }
    }

    /// Returns a reference to the list of workers.
    pub fn workers(&self) -> &[WorkerNode] {
        &self.workers
    }

    /// Returns a mutable reference to the list of workers.
    pub fn workers_mut(&mut self) -> &mut [WorkerNode] {
        &mut self.workers
    }

    #[inline]
    fn hash_to_uniform(hash: u64) -> f64 {
        const SHIFT: u64 = 64 - 53;
        let normalized = (hash >> SHIFT) as f64;
        (normalized + 0.5) / 9007199254740992.0
    }

    /// Selects the worker with the highest weighted Rendezvous score for a given key.
    ///
    /// # Errors
    /// Returns `TopologyError::EmptyRing` if no workers are available.
    pub fn route(&self, primary_hash: u64) -> Result<&WorkerNode, TopologyError> {
        if self.workers.is_empty() {
            return Err(TopologyError::EmptyRing);
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_worker = None;
        let key_bytes = primary_hash.to_le_bytes();

        for worker in &self.workers {
            let mut hasher = Xxh64::new(0);
            hasher.update(&key_bytes);
            hasher.update(worker.id.as_bytes());
            let hash = hasher.digest();

            let u = Self::hash_to_uniform(hash);
            let weight = (1.0 - worker.load() as f64).max(0.001);
            let score = -weight / u.ln();

            if score > best_score {
                best_score = score;
                best_worker = Some(worker);
            }
        }

        best_worker.ok_or(TopologyError::EmptyRing)
    }

    /// Replaces the worker list with a new set of workers.
    pub fn update_workers(&mut self, workers: Vec<WorkerNode>) {
        self.workers = workers;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_routing() {
        let workers = vec![
            WorkerNode::new("a".into(), "1".into(), 0.0),
            WorkerNode::new("b".into(), "2".into(), 0.0),
        ];
        let router = RendezvousRouter::new(workers);
        let k = 0x1234;
        assert_eq!(router.route(k).unwrap().id, router.route(k).unwrap().id);
    }

    #[test]
    fn load_aware_routing() {
        let workers = vec![
            WorkerNode::new("a".into(), "1".into(), 0.9),
            WorkerNode::new("b".into(), "2".into(), 0.1),
        ];
        let router = RendezvousRouter::new(workers);
        let selected = router.route(0x1234).unwrap();
        assert_eq!(selected.id, "b");
    }

    #[test]
    fn set_load_works() {
        let worker = WorkerNode::new("a".into(), "1".into(), 0.5);
        assert_eq!(worker.load(), 0.5);
        worker.set_load(0.8);
        assert_eq!(worker.load(), 0.8);
    }

    #[test]
    fn empty_workers_returns_error() {
        let router: RendezvousRouter = RendezvousRouter::new(Vec::new());
        assert!(matches!(
            router.route(0x1234),
            Err(TopologyError::EmptyRing)
        ));
    }
}