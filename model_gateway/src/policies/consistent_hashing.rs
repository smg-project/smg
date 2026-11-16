//! Consistent hashing routing policy with header-based routing support
//!
//! Supports two routing mechanisms via HTTP headers:
//! - `X-SMG-Target-Worker`: Direct routing by worker index (0-based) into the
//!   workers the caller hands the policy. Placement hands it the workers
//!   available when the request arrives, so the index counts those and an
//!   index past them is a miss, not another worker. The one exception is a
//!   pool with every worker vetoed, which placement's all-vetoed fallback
//!   serves without reading the header.
//! - `X-SMG-Routing-Key`: Consistent hash routing for session affinity
//!
//! ## Consistent Hashing
//!
//! Uses a pre-computed hash ring from WorkerRegistry where:
//! 1. Each worker is placed at a fixed position based on hash(worker_url)
//! 2. Keys are hashed to the ring, then walk clockwise to find first healthy worker
//! 3. When workers scale up/down, only keys in the affected range redistribute (~1/N keys move)
//!
//! The ring is built once when workers are added/removed, not per-request.
//! This ensures O(log n) lookup performance.
//!
//! Complexity per request: O(log n) binary search + O(k) walk where k =
//! consecutive unhealthy workers, each probed node read for eligibility and
//! found in the pool by one scan of its URLs; nothing is allocated for the
//! translation.

use std::{cell::Cell, sync::Arc};

use rand::RngExt as _;

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::{
    observability::metrics::Metrics,
    policies::hash_ring::HashRing,
    routers::common::header_utils::{extract_routing_key_hint, extract_target_worker},
    worker::Worker,
};

/// Execution branch for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    NoHealthyWorkers,
    TargetWorkerHit,
    TargetWorkerMiss,
    RoutingKeyHit,
    ImplicitKeyHit,
    RandomFallback,
}

impl Branch {
    #[inline]
    const fn as_str(self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::TargetWorkerHit => "target_worker_hit",
            Self::TargetWorkerMiss => "target_worker_miss",
            Self::RoutingKeyHit => "routing_key_hit",
            Self::ImplicitKeyHit => "implicit_key_hit",
            Self::RandomFallback => "random_fallback",
        }
    }
}

#[derive(Debug, Default)]
pub struct ConsistentHashingPolicy;

impl ConsistentHashingPolicy {
    pub fn new() -> Self {
        Self
    }

    /// Use consistent hashing to find a worker for the given key.
    /// Uses pre-computed ring from SelectWorkerInfo if available.
    ///
    /// The ring returns a worker URL, which we then map to an index in the workers array.
    /// This correctly handles filtered worker arrays since we match by URL, not by index.
    ///
    /// Complexity: O(log n) ring lookup + O(k) walk, each probed node read for
    /// eligibility and found in the pool by a scan of its URLs (the last
    /// eligible worker at that URL, as the map this replaced answered).
    fn find_by_consistent_hash(
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        key: &str,
    ) -> Option<usize> {
        // A lifted overload veto keeps an owner over the thresholds.
        let eligible = |worker: &Arc<dyn Worker>| {
            if info.overload_lifted {
                worker.routing_state().eligible_ignoring_overload()
            } else {
                worker.is_healthy_and_eligible()
            }
        };

        // Use pre-computed ring if available
        if let Some(ref ring) = info.hash_ring {
            let eligible_at = |url: &str| {
                workers
                    .iter()
                    .rposition(|worker| worker.url() == url && eligible(worker))
            };
            // The probe that accepts the ring's answer keeps the position it
            // found: one scan and one eligibility read per probed node, and
            // the answer cannot disagree with the probe that admitted it.
            let found = Cell::new(None);
            ring.find_healthy_url(key, |url| {
                found.set(eligible_at(url));
                found.get().is_some()
            })?;
            return found.get();
        }

        // Fallback: no ring provided, use simple modulo (less optimal but functional)
        // This shouldn't happen in normal operation as WorkerSelectionStage provides the ring
        let healthy_indices: Vec<usize> = workers
            .iter()
            .enumerate()
            .filter(|(_, worker)| eligible(worker))
            .map(|(idx, _)| idx)
            .collect();
        if healthy_indices.is_empty() {
            return None;
        }

        // The ring's position hash, so the fallback spells a key's position
        // as the ring does.
        let idx = (HashRing::hash_position(key) as usize) % healthy_indices.len();
        Some(healthy_indices[idx])
    }

    /// The explicit session key the request hashes on. Explicit routing-key
    /// headers outrank the rid-derived session key (populated only under the
    /// routing-key override). Both header sides apply the hint caps: an
    /// over-cap or non-UTF-8 key must not influence placement on any path.
    fn session_key<'a>(info: &SelectWorkerInfo<'a>) -> Option<&'a str> {
        info.routing_key
            .or_else(|| extract_routing_key_hint(info.headers))
            .or(info.rid_key)
    }

    #[expect(
        clippy::unused_self,
        reason = "consistent with LoadBalancingPolicy trait pattern; may use self in future"
    )]
    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        if workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        let target_worker = extract_target_worker(info.headers);
        let routing_key = Self::session_key(info);

        // Priority 1: X-SMG-Target-Worker - direct routing by worker index
        // O(1) parse + O(1) bounds check + O(1) health check
        if let Some(idx_str) = target_worker {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx < workers.len() && workers[idx].is_healthy_and_eligible() {
                    return (Some(idx), Branch::TargetWorkerHit);
                }
            }
            return (None, Branch::TargetWorkerMiss);
        }

        // Priority 2: X-SMG-Routing-Key - consistent hash routing (O(log n))
        if let Some(key) = routing_key {
            return match Self::find_by_consistent_hash(workers, info, key) {
                Some(idx) => (Some(idx), Branch::RoutingKeyHit),
                None => (None, Branch::NoHealthyWorkers),
            };
        }

        // Priority 3: Implicit routing key from stable headers (session affinity).
        // Only headers that identify a client or a session qualify. A credential
        // (`authorization`) identifies a tenant or a service account, not a
        // session: every client sharing an API key would hash to one worker.
        let implicit_key = info.headers.and_then(|h| {
            h.get("x-forwarded-for")
                .or_else(|| h.get("cookie"))
                .and_then(|v| v.to_str().ok())
                .filter(|s| !s.is_empty())
        });

        if let Some(key) = implicit_key {
            return match Self::find_by_consistent_hash(workers, info, key) {
                Some(idx) => (Some(idx), Branch::ImplicitKeyHit),
                None => (None, Branch::NoHealthyWorkers),
            };
        }

        // Fallback: random selection (truly anonymous client)
        let healthy_count = workers
            .iter()
            .filter(|w| w.is_healthy_and_eligible())
            .count();
        if healthy_count == 0 {
            return (None, Branch::NoHealthyWorkers);
        }

        let random_healthy_idx = rand::rng().random_range(0..healthy_count);
        let idx = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy_and_eligible())
            .nth(random_healthy_idx)
            .map(|(i, _)| i);

        (idx, Branch::RandomFallback)
    }
}

impl LoadBalancingPolicy for ConsistentHashingPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_consistent_hashing_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "consistent_hashing"
    }

    /// A session key places the request on its ring owner. An explicit
    /// target worker outranks it, and the implicit client key
    /// (`x-forwarded-for`, `cookie`) identifies a client rather than a
    /// session, so neither counts.
    fn routes_by_session_key(&self, info: &SelectWorkerInfo) -> bool {
        extract_target_worker(info.headers).is_none() && Self::session_key(info).is_some()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests;
