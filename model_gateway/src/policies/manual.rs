//! Manual routing policy based on routing key header
//!
//! This policy provides sticky session routing where each unique routing key
//! is consistently mapped to the same worker. Unlike consistent hashing,
//! this policy:
//! - Does NOT redistribute any sessions when workers are added
//! - Only remaps sessions when their assigned worker becomes unhealthy
//! - Maintains up to 2 candidate workers per routing key for fast failover
//!
//! Use this when you need stronger stickiness guarantees than consistent hashing,
//! for example with stateful chat sessions where context is stored on the worker.
//!
//! ## Key sources
//! - `X-SMG-Routing-Key`: The preferred routing key for sticky session routing
//! - `SelectWorkerInfo::rid_key`: the fallback session key derived from the
//!   request body's `rid` (populated under `--routing-key-override`)

use std::{
    borrow::Borrow,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::{mapref::entry::Entry, DashMap};
use rand::RngExt;
use tracing::info;

use super::{
    get_healthy_worker_indices, utils::PeriodicTask, LoadBalancingPolicy, SelectWorkerInfo,
    WorkerRole,
};
use crate::{
    config::ManualAssignmentMode, observability::metrics::Metrics,
    routers::common::header_utils::extract_routing_key_hint, worker::Worker,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionBranch {
    NoHealthyWorkers,
    OccupiedHit,
    OccupiedMiss,
    Vacant,
    NoRoutingId,
    CapRespill,
}

impl ExecutionBranch {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::OccupiedHit => "occupied_hit",
            Self::OccupiedMiss => "occupied_miss",
            Self::Vacant => "vacant",
            Self::NoRoutingId => "no_routing_id",
            Self::CapRespill => "cap_respill",
        }
    }
}

/// Result of a non-mutating pin probe for a sticky key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PinState {
    /// A healthy pinned worker exists at this index.
    Pinned(usize),
    /// An entry exists but none of its candidates are healthy.
    Stale,
    /// No entry for this key.
    Vacant,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RoutingId(String);

impl RoutingId {
    fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// A lookup borrows the request's key: `RoutingId` hashes and compares as
/// its string does, so the map is read by `&str` and copies the key only
/// when it inserts one.
impl Borrow<str> for RoutingId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

const MAX_CANDIDATE_WORKERS: usize = 2;

#[derive(Debug, Clone)]
pub struct ManualConfig {
    pub eviction_interval_secs: u64,
    pub max_idle_secs: u64,
    pub assignment_mode: ManualAssignmentMode,
}

impl Default for ManualConfig {
    fn default() -> Self {
        Self {
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: ManualAssignmentMode::Random,
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    candi_worker_urls: Vec<String>,
    last_access: Instant,
}

impl Node {
    fn push_bounded(&mut self, url: String) {
        while self.candi_worker_urls.len() >= MAX_CANDIDATE_WORKERS {
            self.candi_worker_urls.remove(0);
        }
        self.candi_worker_urls.push(url);
    }
}

#[derive(Debug)]
pub struct ManualPolicy {
    routing_map: Arc<DashMap<RoutingId, Node>>,
    /// The map's entry count, kept where the map changes (a key it did not
    /// hold, the idle eviction) and published to the size gauge from there:
    /// `DashMap::len` locks and walks every shard, so a selection reads
    /// nothing about the size.
    entries: Arc<AtomicUsize>,
    assignment_mode: ManualAssignmentMode,
    _eviction_task: Option<PeriodicTask>,
}

impl Default for ManualPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ManualPolicy {
    /// The map goes with the policy (the registry drops a model's policy
    /// with its last worker), and its entries leave the size gauge with it:
    /// the eviction task is stopped and joined first, so no pass sits
    /// between its `retain` and its decrement when the count is read.
    fn drop(&mut self) {
        self._eviction_task.take();
        let remaining = self.entries.swap(0, Ordering::Relaxed);
        if remaining > 0 {
            Metrics::manual_policy_cache_entries_evicted(remaining);
        }
    }
}

impl ManualPolicy {
    pub fn new() -> Self {
        Self::with_config(ManualConfig::default())
    }

    pub fn with_config(config: ManualConfig) -> Self {
        let routing_map = Arc::new(DashMap::<RoutingId, Node>::new());
        let entries = Arc::new(AtomicUsize::new(0));

        let eviction_task = if config.eviction_interval_secs > 0 && config.max_idle_secs > 0 {
            let routing_map_clone = Arc::clone(&routing_map);
            let entries_clone = Arc::clone(&entries);
            let max_idle = Duration::from_secs(config.max_idle_secs);

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "ManualPolicyEviction",
                move || {
                    evict_idle(&routing_map_clone, &entries_clone, max_idle, Instant::now());
                },
            ))
        } else {
            None
        };

        Self {
            routing_map,
            entries,
            assignment_mode: config.assignment_mode,
            _eviction_task: eviction_task,
        }
    }

    /// A key the map did not hold was inserted: the count and the size gauge
    /// move here, not on the selection path. The gauge moves by one rather
    /// than to the count read back: two inserts racing would publish their
    /// counts in either order, and nothing rewrites the gauge until the next
    /// change.
    fn note_new_entry(&self) {
        self.entries.fetch_add(1, Ordering::Relaxed);
        Metrics::manual_policy_cache_entry_added();
    }

    /// The entry count as kept: what the size gauge shows.
    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.entries.load(Ordering::Relaxed)
    }

    fn select_new_worker(&self, workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> usize {
        match self.assignment_mode {
            ManualAssignmentMode::Random => random_select(healthy_indices),
            ManualAssignmentMode::MinLoad | ManualAssignmentMode::Delegate => {
                min_load_select(workers, healthy_indices)
            }
            ManualAssignmentMode::MinGroup => min_group_select(workers, healthy_indices),
        }
    }

    pub(crate) fn assignment_mode(&self) -> ManualAssignmentMode {
        self.assignment_mode
    }

    /// Probe the pin for `key` without reassigning: the first of its
    /// candidates that `eligible` admits, by its position in `workers`.
    /// Refreshes `last_access` when an entry exists so probing counts as
    /// use. The key is looked up borrowed, and nothing is allocated.
    pub(crate) fn peek_pin(
        &self,
        workers: &[Arc<dyn Worker>],
        key: &str,
        eligible: impl Fn(usize) -> bool,
    ) -> PinState {
        match self.routing_map.get_mut(key) {
            Some(mut entry) => {
                entry.last_access = Instant::now();
                match find_healthy_worker(&entry.candi_worker_urls, workers, eligible) {
                    Some(idx) => PinState::Pinned(idx),
                    None => PinState::Stale,
                }
            }
            None => PinState::Vacant,
        }
    }

    /// Pin `key` to `url` as the primary candidate, keeping the previous
    /// primary as bounded failover. Used by delegated assignment and
    /// cap-respill, where the pin must follow the latest dispatch. An
    /// existing entry is updated through a borrowed lookup; only a new key
    /// is copied into the map.
    pub(crate) fn pin_front(&self, key: &str, url: &str) {
        let front = |node: &mut Node| {
            node.candi_worker_urls.retain(|u| u != url);
            node.candi_worker_urls.insert(0, url.to_string());
            node.candi_worker_urls.truncate(MAX_CANDIDATE_WORKERS);
            node.last_access = Instant::now();
        };
        if let Some(mut entry) = self.routing_map.get_mut(key) {
            front(&mut entry);
            return;
        }
        match self.routing_map.entry(RoutingId::new(key)) {
            Entry::Occupied(mut entry) => front(entry.get_mut()),
            Entry::Vacant(entry) => {
                entry.insert(Node {
                    candi_worker_urls: vec![url.to_string()],
                    last_access: Instant::now(),
                });
                self.note_new_entry();
            }
        }
    }

    /// Assign a worker for `key` via this policy's assignment mode, recording
    /// the pin exactly like keyed selection does.
    pub(crate) fn assign_pin(
        &self,
        workers: &[Arc<dyn Worker>],
        key: &str,
        healthy_indices: &[usize],
    ) -> (usize, ExecutionBranch) {
        self.select_by_routing_id(workers, key, healthy_indices)
    }

    /// Raw assignment via this policy's mode, without touching the pin map.
    pub(crate) fn assign_index(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
    ) -> usize {
        self.select_new_worker(workers, healthy_indices)
    }

    fn select_by_routing_id(
        &self,
        workers: &[Arc<dyn Worker>],
        routing_id: &str,
        healthy_indices: &[usize],
    ) -> (usize, ExecutionBranch) {
        let eligible = |idx: usize| healthy_indices.contains(&idx);
        // The hit path reads the entry through the borrowed key; only a key
        // the map does not hold is copied in, below.
        let occupied = |node: &mut Node| {
            node.last_access = Instant::now();
            if let Some(idx) = find_healthy_worker(&node.candi_worker_urls, workers, eligible) {
                (idx, ExecutionBranch::OccupiedHit)
            } else {
                let selected_idx = self.select_new_worker(workers, healthy_indices);
                node.push_bounded(workers[selected_idx].url().to_string());
                (selected_idx, ExecutionBranch::OccupiedMiss)
            }
        };
        if let Some(mut entry) = self.routing_map.get_mut(routing_id) {
            return occupied(&mut entry);
        }
        match self.routing_map.entry(RoutingId::new(routing_id)) {
            Entry::Occupied(mut entry) => occupied(entry.get_mut()),
            Entry::Vacant(entry) => {
                let selected_idx = self.select_new_worker(workers, healthy_indices);
                entry.insert(Node {
                    candi_worker_urls: vec![workers[selected_idx].url().to_string()],
                    last_access: Instant::now(),
                });
                self.note_new_entry();
                (selected_idx, ExecutionBranch::Vacant)
            }
        }
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, ExecutionBranch) {
        let healthy_indices = get_healthy_worker_indices(workers, info);
        if healthy_indices.is_empty() {
            return (None, ExecutionBranch::NoHealthyWorkers);
        }

        // Explicit conversation affinity must survive a fresh request ID on
        // each call. The rid-derived session key only fills in a missing key.
        let routing_id = info
            .routing_key
            .or_else(|| extract_routing_key_hint(info.headers))
            .or(info.rid_key);
        if let Some(routing_id) = routing_id {
            // Single is the common role; route on the bare key to skip the
            // per-request allocation. PD roles namespace so prefill and decode
            // stick independently.
            let (idx, branch) = if info.role == WorkerRole::Single {
                self.select_by_routing_id(workers, routing_id, &healthy_indices)
            } else {
                let namespaced = format!("{}{}", info.role.routing_id_prefix(), routing_id);
                self.select_by_routing_id(workers, &namespaced, &healthy_indices)
            };
            return (Some(idx), branch);
        }

        (
            Some(self.select_new_worker(workers, &healthy_indices)),
            ExecutionBranch::NoRoutingId,
        )
    }
}

impl LoadBalancingPolicy for ManualPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_manual_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "manual"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The idle eviction: every entry not used within `max_idle` of `now` leaves
/// the map, the kept count and the size gauge follow by the number removed,
/// and a pass that removed anything is logged. Returns that number.
fn evict_idle(
    routing_map: &DashMap<RoutingId, Node>,
    entries: &AtomicUsize,
    max_idle: Duration,
    now: Instant,
) -> usize {
    let mut evicted = 0;
    routing_map.retain(|_, node| {
        let kept = now.duration_since(node.last_access) < max_idle;
        if !kept {
            evicted += 1;
        }
        kept
    });
    if evicted > 0 {
        let remaining = entries
            .fetch_sub(evicted, Ordering::Relaxed)
            .saturating_sub(evicted);
        Metrics::manual_policy_cache_entries_evicted(evicted);
        info!(
            "ManualPolicy TTL eviction: evicted {} entries, remaining {} (max_idle: {}s)",
            evicted,
            remaining,
            max_idle.as_secs()
        );
    }
    evicted
}

/// The first of a pin's candidate URLs that is in `workers` and that
/// `eligible` admits, by position. A pin holds at most two URLs, each found
/// by one scan of the pool.
fn find_healthy_worker(
    urls: &[String],
    workers: &[Arc<dyn Worker>],
    eligible: impl Fn(usize) -> bool,
) -> Option<usize> {
    urls.iter()
        .filter_map(|url| find_worker_index_by_url(workers, url))
        .find(|&idx| eligible(idx))
}

fn find_worker_index_by_url(workers: &[Arc<dyn Worker>], url: &str) -> Option<usize> {
    workers.iter().position(|w| w.url() == url)
}

fn random_select(healthy_indices: &[usize]) -> usize {
    let mut rng = rand::rng();
    let random_idx = rng.random_range(0..healthy_indices.len());
    healthy_indices[random_idx]
}

fn select_min_by<K, V, F>(indices: &[K], get_value: F) -> K
where
    K: Copy,
    V: Ord,
    F: Fn(K) -> V,
{
    // One pass and no list of the ties: the n-th candidate tied for the
    // minimum replaces the pick with probability 1/n, which leaves every
    // tied candidate as likely as a draw from a list of them did.
    let mut rng = rand::rng();
    let mut pick: Option<(K, V)> = None;
    let mut ties: u32 = 0;
    for &idx in indices {
        let val = get_value(idx);
        match pick.as_ref().map(|(_, min)| val.cmp(min)) {
            None | Some(std::cmp::Ordering::Less) => {
                pick = Some((idx, val));
                ties = 1;
            }
            Some(std::cmp::Ordering::Equal) => {
                ties += 1;
                if rng.random_range(0..ties) == 0 {
                    pick = Some((idx, val));
                }
            }
            Some(std::cmp::Ordering::Greater) => {}
        }
    }
    match pick {
        Some((idx, _)) => idx,
        // Every caller returns early on an empty pool, so a pick always exists;
        // this index is the invariant's check, not a recovery path.
        None => indices[0],
    }
}

fn min_load_select(workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> usize {
    select_min_by(healthy_indices, |idx| workers[idx].load())
}

fn min_group_select(workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> usize {
    select_min_by(healthy_indices, |idx| workers[idx].routing_key_load())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::worker::WorkerStatus;

    use super::*;
    use crate::policies::test_support::{headers_with_routing_key, make_workers, with_headers};

    #[test]
    fn test_manual_role_namespaces_sticky_entries() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let headers = headers_with_routing_key("user-123");

        let prefill = SelectWorkerInfo {
            headers: Some(&headers),
            role: WorkerRole::Prefill,
            ..Default::default()
        };
        let decode = SelectWorkerInfo {
            headers: Some(&headers),
            role: WorkerRole::Decode,
            ..Default::default()
        };

        let (p1, _) = policy.select_worker_impl(&workers, &prefill);
        let (d1, _) = policy.select_worker_impl(&workers, &decode);

        // Same key under two roles -> two independent sticky entries.
        assert_eq!(policy.routing_map.len(), 2);
        for _ in 0..5 {
            assert_eq!(policy.select_worker_impl(&workers, &prefill).0, p1);
            assert_eq!(policy.select_worker_impl(&workers, &decode).0, d1);
        }
    }

    #[test]
    fn test_manual_header_key_outranks_rid_key() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let headers = headers_with_routing_key("conv");
        let turn1 = SelectWorkerInfo {
            headers: Some(&headers),
            rid_key: Some("request-1"),
            ..Default::default()
        };
        let turn2 = SelectWorkerInfo {
            headers: Some(&headers),
            rid_key: Some("request-2"),
            ..Default::default()
        };

        let (first, branch) = policy.select_worker_impl(&workers, &turn1);
        assert_eq!(branch, ExecutionBranch::Vacant);
        let (second, branch) = policy.select_worker_impl(&workers, &turn2);
        assert_eq!(second, first, "header key must outrank fresh request IDs");
        assert_eq!(branch, ExecutionBranch::OccupiedHit);

        assert_eq!(policy.routing_map.len(), 1);
        assert!(policy.routing_map.contains_key(&RoutingId::new("conv")));
    }

    #[test]
    fn test_manual_rid_key_namespaces_per_role() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let prefill = SelectWorkerInfo {
            rid_key: Some("conv"),
            role: WorkerRole::Prefill,
            ..Default::default()
        };
        let decode = SelectWorkerInfo {
            rid_key: Some("conv"),
            role: WorkerRole::Decode,
            ..Default::default()
        };

        let (p1, _) = policy.select_worker_impl(&workers, &prefill);
        let (d1, _) = policy.select_worker_impl(&workers, &decode);

        // Same rid under two roles -> two independent sticky entries.
        assert_eq!(policy.routing_map.len(), 2);
        for _ in 0..5 {
            assert_eq!(policy.select_worker_impl(&workers, &prefill).0, p1);
            assert_eq!(policy.select_worker_impl(&workers, &decode).0, d1);
        }
    }

    #[test]
    fn test_manual_consistent_routing() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let headers = headers_with_routing_key("user-123");
        let info = with_headers(&headers);

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(first_idx),
                "Same routing_id should route to same worker"
            );
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
        }
    }

    #[test]
    fn test_manual_different_routing_ids() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let mut distribution = HashMap::new();
        for i in 0..100 {
            let headers = headers_with_routing_key(&format!("user-{i}"));
            let info = with_headers(&headers);
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::Vacant);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(
            distribution.len() > 1,
            "Should distribute across multiple workers"
        );
    }

    #[test]
    fn test_manual_fallback_random() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let mut counts = HashMap::new();
        for _ in 0..100 {
            let info = SelectWorkerInfo::default();
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::NoRoutingId);
            if let Some(idx) = result {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        assert_eq!(counts.len(), 2, "Random fallback should use all workers");
    }

    #[test]
    fn test_manual_with_unhealthy_workers() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        workers[0].set_status(WorkerStatus::NotReady);

        let headers = headers_with_routing_key("test-routing-id");
        let info = with_headers(&headers);

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1), "Should only select healthy worker");
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(1), "Should only select healthy worker");
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
        }
    }

    #[test]
    fn test_manual_no_healthy_workers() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000"]);

        workers[0].set_status(WorkerStatus::NotReady);
        let headers = headers_with_routing_key("test");
        let info = with_headers(&headers);
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, ExecutionBranch::NoHealthyWorkers);
    }

    #[test]
    fn test_manual_empty_routing_id() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let mut counts = HashMap::new();
        for _ in 0..100 {
            let headers = headers_with_routing_key("");
            let info = with_headers(&headers);
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::NoRoutingId);
            if let Some(idx) = result {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        assert_eq!(
            counts.len(),
            2,
            "Empty routing_id should use random fallback"
        );
    }

    #[test]
    fn test_manual_remaps_when_worker_becomes_unhealthy() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("sticky-user");
        let info = with_headers(&headers);

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_status(WorkerStatus::NotReady);

        let (new_result, branch) = policy.select_worker_impl(&workers, &info);
        let new_idx = new_result.unwrap();
        assert_ne!(new_idx, first_idx, "Should remap to healthy worker");
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(new_idx),
                "Should consistently route to new worker"
            );
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
        }
    }

    #[test]
    fn test_manual_empty_workers() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let headers = headers_with_routing_key("test");
        let info = with_headers(&headers);
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, ExecutionBranch::NoHealthyWorkers);
    }

    #[test]
    fn test_manual_single_worker() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000"]);

        let headers = headers_with_routing_key("single-test");
        let info = with_headers(&headers);

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(0));
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(0));
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
        }
    }

    #[test]
    fn test_manual_worker_recovery() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("recovery-test");
        let info = with_headers(&headers);

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_status(WorkerStatus::NotReady);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[first_idx].set_status(WorkerStatus::Ready);

        let (after_recovery, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            after_recovery,
            Some(first_idx),
            "Should return to original worker after recovery since it's first in candidate list"
        );
        assert_eq!(branch, ExecutionBranch::OccupiedHit);
    }

    #[test]
    fn test_manual_max_candidate_workers_eviction() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let headers = headers_with_routing_key("eviction-test");
        let info = with_headers(&headers);

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_status(WorkerStatus::NotReady);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[second_idx].set_status(WorkerStatus::NotReady);

        let remaining_idx = (0..3).find(|&i| i != first_idx && i != second_idx).unwrap();
        let (third_result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            third_result,
            Some(remaining_idx),
            "Should select the only remaining healthy worker"
        );
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[first_idx].set_status(WorkerStatus::Ready);

        let (idx_after_restore, branch) = policy.select_worker_impl(&workers, &info);
        assert_ne!(
            idx_after_restore,
            Some(first_idx),
            "First worker should be evicted from candidates due to MAX_CANDIDATE_WORKERS=2"
        );
        assert_eq!(branch, ExecutionBranch::OccupiedHit);
    }

    #[test]
    fn test_manual_policy_name() {
        let policy = ManualPolicy::new();
        assert_eq!(policy.name(), "manual");
    }

    #[test]
    fn test_manual_routing_info_push_bounded() {
        let mut info = Node {
            candi_worker_urls: vec!["http://w1:8000".to_string()],
            last_access: Instant::now(),
        };

        info.push_bounded("http://w2:8000".to_string());
        assert_eq!(info.candi_worker_urls.len(), 2);
        assert_eq!(info.candi_worker_urls[0], "http://w1:8000");
        assert_eq!(info.candi_worker_urls[1], "http://w2:8000");

        info.push_bounded("http://w3:8000".to_string());
        assert_eq!(info.candi_worker_urls.len(), 2);
        assert_eq!(
            info.candi_worker_urls[0], "http://w2:8000",
            "Oldest entry should be removed"
        );
        assert_eq!(info.candi_worker_urls[1], "http://w3:8000");
    }

    #[test]
    fn test_manual_find_healthy_worker_priority() {
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let urls = vec![
            "http://w1:8000".to_string(),
            "http://w2:8000".to_string(),
            "http://w3:8000".to_string(),
        ];
        let healthy_indices = [0, 1, 2];

        let result = find_healthy_worker(&urls, &workers, |idx| healthy_indices.contains(&idx));
        assert_eq!(
            result,
            Some(0),
            "Should return first healthy worker in urls"
        );

        workers[0].set_status(WorkerStatus::NotReady);
        let healthy_indices = [1, 2];
        let result = find_healthy_worker(&urls, &workers, |idx| healthy_indices.contains(&idx));
        assert_eq!(result, Some(1), "Should skip unhealthy and return next");

        workers[1].set_status(WorkerStatus::NotReady);
        let healthy_indices = [2];
        let result = find_healthy_worker(&urls, &workers, |idx| healthy_indices.contains(&idx));
        assert_eq!(result, Some(2), "Should return last healthy worker");

        workers[2].set_status(WorkerStatus::NotReady);
        let healthy_indices: Vec<usize> = vec![];
        let result = find_healthy_worker(&urls, &workers, |idx| healthy_indices.contains(&idx));
        assert_eq!(result, None, "Should return None when no healthy workers");
    }

    #[test]
    fn test_manual_find_worker_index_by_url() {
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        assert_eq!(
            find_worker_index_by_url(&workers, "http://w1:8000"),
            Some(0)
        );
        assert_eq!(
            find_worker_index_by_url(&workers, "http://w2:8000"),
            Some(1)
        );
        assert_eq!(
            find_worker_index_by_url(&workers, "http://w3:8000"),
            None,
            "Should return None for unknown URL"
        );
    }

    #[test]
    fn test_manual_config_default() {
        let config = ManualConfig::default();
        assert_eq!(config.eviction_interval_secs, 60);
        assert_eq!(config.max_idle_secs, 4 * 3600);
    }

    /// The pin of the kept count: a selection publishes the map's size only
    /// when it inserted a key; a hit records nothing about the size.
    #[test]
    fn a_selection_publishes_the_map_size_only_for_a_new_key() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = headers_with_routing_key("conv-1");
        let info = with_headers(&headers);

        let first = PrometheusBuilder::new().build_recorder();
        let first_handle = first.handle();
        ::metrics::with_local_recorder(&first, || {
            policy.select_worker(&workers, &info);
        });
        assert!(
            first_handle
                .render()
                .contains("smg_manual_policy_cache_entries 1"),
            "the new key sets the gauge to 1"
        );

        let hits = PrometheusBuilder::new().build_recorder();
        let hits_handle = hits.handle();
        ::metrics::with_local_recorder(&hits, || {
            for _ in 0..3 {
                policy.select_worker(&workers, &info);
            }
        });
        assert!(
            !hits_handle
                .render()
                .contains("smg_manual_policy_cache_entries"),
            "a hit publishes nothing about the map's size"
        );
    }

    #[test]
    fn the_idle_eviction_takes_the_count_and_the_gauge_down_by_what_it_removed() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        ::metrics::with_local_recorder(&recorder, || {
            let policy = ManualPolicy::new();
            let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
            for key in ["conv-1", "conv-2", "conv-3"] {
                let headers = headers_with_routing_key(key);
                policy.select_worker(&workers, &with_headers(&headers));
            }
            policy.pin_front("conv-1", "http://w2:8000");
            assert_eq!(
                policy.entry_count(),
                3,
                "a pin of a held key counts nothing"
            );
            policy.pin_front("conv-4", "http://w2:8000");
            assert_eq!(policy.entry_count(), 4, "a pin of a new key counts it");
            assert!(
                handle
                    .render()
                    .contains("smg_manual_policy_cache_entries 4"),
                "every new key moved the gauge up by one"
            );

            // Two entries used an hour later: a pass at that instant with a
            // minute's bound removes the other two, not everything (a reset
            // to zero would read the same as a full pass).
            let later = Instant::now() + Duration::from_secs(3600);
            for key in ["conv-3", "conv-4"] {
                policy
                    .routing_map
                    .get_mut(&RoutingId(key.to_string()))
                    .expect("a kept key")
                    .last_access = later;
            }
            let evicted = evict_idle(
                &policy.routing_map,
                &policy.entries,
                Duration::from_secs(60),
                later,
            );
            assert_eq!(evicted, 2);
            assert_eq!(policy.entry_count(), 2);
            assert_eq!(policy.routing_map.len(), 2);
            assert!(
                handle
                    .render()
                    .contains("smg_manual_policy_cache_entries 2"),
                "the pass moved the gauge down by what it removed"
            );

            let evicted = evict_idle(&policy.routing_map, &policy.entries, Duration::ZERO, later);
            assert_eq!(evicted, 2);
            assert_eq!(policy.entry_count(), 0);
            assert_eq!(policy.routing_map.len(), 0);
            assert!(
                handle
                    .render()
                    .contains("smg_manual_policy_cache_entries 0"),
                "the last pass left the gauge at zero"
            );
        });
    }

    #[test]
    fn test_manual_with_disabled_eviction() {
        let config = ManualConfig {
            eviction_interval_secs: 0,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let policy = ManualPolicy::with_config(config);
        assert!(policy._eviction_task.is_none());
    }

    #[test]
    fn test_manual_last_access_updates() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = headers_with_routing_key("test-key");
        let info = with_headers(&headers);
        let routing_id = RoutingId::new("test-key");

        // Vacant: first access
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(branch, ExecutionBranch::Vacant);
        let first_idx = result.unwrap();
        let access_after_vacant = policy.routing_map.get(&routing_id).unwrap().last_access;
        assert!(access_after_vacant.elapsed().as_millis() < 100);

        std::thread::sleep(Duration::from_millis(10));

        // OccupiedHit: same worker still healthy
        let (_, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(branch, ExecutionBranch::OccupiedHit);
        let access_after_hit = policy.routing_map.get(&routing_id).unwrap().last_access;
        assert!(access_after_hit > access_after_vacant);

        std::thread::sleep(Duration::from_millis(10));

        // OccupiedMiss: worker becomes unhealthy
        workers[first_idx].set_status(WorkerStatus::NotReady);
        let (_, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);
        let access_after_miss = policy.routing_map.get(&routing_id).unwrap().last_access;
        assert!(access_after_miss > access_after_hit);
    }

    #[test]
    fn test_manual_ttl_eviction_logic() {
        let config = ManualConfig {
            eviction_interval_secs: 2,
            max_idle_secs: 2,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let policy = ManualPolicy::with_config(config);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("key-0");
        let info = with_headers(&headers);
        policy.select_worker_impl(&workers, &info);

        assert_eq!(policy.routing_map.len(), 1);

        std::thread::sleep(Duration::from_secs(4));

        assert_eq!(policy.routing_map.len(), 0);
    }

    #[test]
    fn test_min_group_select_distributes_evenly() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinGroup,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        for i in 0..9 {
            let routing_key = format!("key-{i}");
            let headers = headers_with_routing_key(&routing_key);
            let info = with_headers(&headers);

            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert!(result.is_some());
            assert_eq!(branch, ExecutionBranch::Vacant);

            let selected_idx = result.unwrap();
            workers[selected_idx].increment_routing_key_load(&routing_key);
        }

        let distribution: HashMap<_, usize> = policy
            .routing_map
            .iter()
            .map(|e| e.candi_worker_urls.first().unwrap().clone())
            .fold(HashMap::new(), |mut acc, url| {
                *acc.entry(url).or_default() += 1;
                acc
            });

        assert_eq!(distribution.len(), 3, "Should use all 3 workers");
        for count in distribution.values() {
            assert_eq!(*count, 3, "Each worker should have exactly 3 routing keys");
        }
    }

    #[test]
    fn test_min_group_select_prefers_worker_with_fewer_routing_keys() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinGroup,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        workers[0].increment_routing_key_load("existing-1");
        workers[0].increment_routing_key_load("existing-2");
        workers[1].increment_routing_key_load("existing-3");

        assert_eq!(workers[0].routing_key_load(), 2);
        assert_eq!(workers[1].routing_key_load(), 1);
        assert_eq!(workers[2].routing_key_load(), 0);

        let headers = headers_with_routing_key("new-key");
        let info = with_headers(&headers);
        let (result, _) = policy.select_worker_impl(&workers, &info);
        let selected_idx = result.unwrap();

        assert_eq!(selected_idx, 2, "Should select worker with 0 routing keys");
    }

    #[test]
    fn test_min_load_select_prefers_worker_with_fewer_requests() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinLoad,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        assert_eq!(workers[0].load(), 2);
        assert_eq!(workers[1].load(), 1);
        assert_eq!(workers[2].load(), 0);

        let headers = headers_with_routing_key("new-key");
        let info = with_headers(&headers);
        let (result, _) = policy.select_worker_impl(&workers, &info);
        let selected_idx = result.unwrap();

        assert_eq!(selected_idx, 2, "Should select worker with 0 load");
    }

    #[test]
    fn test_min_group_sticky_after_assignment() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinGroup,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        workers[0].increment_routing_key_load("key-0");
        workers[1].increment_routing_key_load("key-1");
        workers[1].increment_routing_key_load("key-2");

        let headers = headers_with_routing_key("new-key");
        let info = with_headers(&headers);

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);
        assert_eq!(
            first_idx, 0,
            "Should select worker 0 (has 1 routing key vs 2)"
        );

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(first_idx),
                "Same routing key should route to same worker"
            );
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
        }
    }

    #[test]
    fn test_pin_front_promotes_and_bounds_candidates() {
        let policy = ManualPolicy::new();
        policy.pin_front("k", "http://w1:8000");
        policy.pin_front("k", "http://w2:8000");
        let node = policy.routing_map.get(&RoutingId::new("k")).unwrap();
        assert_eq!(node.candi_worker_urls, ["http://w2:8000", "http://w1:8000"]);
        drop(node);

        // Re-pinning an existing candidate moves it to the front, deduped.
        policy.pin_front("k", "http://w1:8000");
        let node = policy.routing_map.get(&RoutingId::new("k")).unwrap();
        assert_eq!(node.candi_worker_urls, ["http://w1:8000", "http://w2:8000"]);
        drop(node);

        policy.pin_front("k", "http://w3:8000");
        let node = policy.routing_map.get(&RoutingId::new("k")).unwrap();
        assert_eq!(node.candi_worker_urls.len(), MAX_CANDIDATE_WORKERS);
        assert_eq!(node.candi_worker_urls[0], "http://w3:8000");
    }

    #[test]
    fn test_peek_pin_states() {
        let policy = ManualPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let healthy = [0, 1];

        assert_eq!(
            policy.peek_pin(&workers, "k", |idx| healthy.contains(&idx)),
            PinState::Vacant
        );

        policy.pin_front("k", "http://w2:8000");
        assert_eq!(
            policy.peek_pin(&workers, "k", |idx| healthy.contains(&idx)),
            PinState::Pinned(1)
        );

        workers[1].set_status(WorkerStatus::NotReady);
        assert_eq!(
            policy.peek_pin(&workers, "k", |idx| idx == 0),
            PinState::Stale
        );
    }

    #[test]
    fn test_delegate_mode_standalone_assigns_min_load() {
        // With no underlying policy to delegate to, delegate degrades to
        // min-load assignment.
        let policy = ManualPolicy::with_config(ManualConfig {
            assignment_mode: ManualAssignmentMode::Delegate,
            ..Default::default()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        workers[0].increment_load();

        let info = SelectWorkerInfo::default();
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1));
        assert_eq!(branch, ExecutionBranch::NoRoutingId);
    }

    #[test]
    fn test_random_mode_does_not_consider_load() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::Random,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        workers[0].increment_routing_key_load("key-1");
        workers[0].increment_routing_key_load("key-2");
        workers[0].increment_routing_key_load("key-3");

        let mut selected_worker_0 = false;
        for i in 0..50 {
            let headers = headers_with_routing_key(&format!("test-{i}"));
            let info = with_headers(&headers);
            let (result, _) = policy.select_worker_impl(&workers, &info);
            if result == Some(0) {
                selected_worker_0 = true;
                break;
            }
        }
        assert!(
            selected_worker_0,
            "Random mode should sometimes select worker 0 despite higher load"
        );
    }

    /// A policy dropped with its map takes its entries out of the gauge: a
    /// replacement's keys count from zero, not from the old map's worth.
    #[test]
    fn a_dropped_policy_takes_its_entries_out_of_the_gauge() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        ::metrics::with_local_recorder(&recorder, || {
            let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
            let policy = ManualPolicy::new();
            for key in ["conv-1", "conv-2", "conv-3"] {
                let headers = headers_with_routing_key(key);
                policy.select_worker(&workers, &with_headers(&headers));
            }
            assert!(
                handle
                    .render()
                    .contains("smg_manual_policy_cache_entries 3"),
                "three new keys moved the gauge to 3"
            );

            drop(policy);
            assert!(
                handle
                    .render()
                    .contains("smg_manual_policy_cache_entries 0"),
                "the dropped policy's entries left the gauge"
            );

            let replacement = ManualPolicy::new();
            let headers = headers_with_routing_key("conv-4");
            replacement.select_worker(&workers, &with_headers(&headers));
            assert!(
                handle
                    .render()
                    .contains("smg_manual_policy_cache_entries 1"),
                "a replacement's keys count from zero"
            );
        });
    }
}
