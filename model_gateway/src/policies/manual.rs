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
use smg_worker::identity::basic::Worker;
use tracing::info;

use super::{
    get_healthy_worker_indices, utils::PeriodicTask, LoadBalancingPolicy, SelectWorkerInfo,
    WorkerRole,
};
use crate::{
    config::ManualAssignmentMode, observability::metrics::Metrics,
    routers::common::header_utils::extract_routing_key_hint,
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

/// The most keys a routing map holds. A key new to a full map is not
/// recorded: its request is placed by the assignment with no pin, until the
/// idle eviction takes entries out. A map can stand past it by a key a
/// shard ([`ManualPolicy::admits_new_entry`]).
const MAX_ROUTING_KEYS: usize = 4_194_304;

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

    /// Whether the map holds [`MAX_ROUTING_KEYS`]: a read of the kept count.
    fn is_full(&self) -> bool {
        self.entries.load(Ordering::Relaxed) >= MAX_ROUTING_KEYS
    }

    /// Whether a key the map does not hold is recorded: none is while the map
    /// holds [`MAX_ROUTING_KEYS`]. The count and the size gauge move here,
    /// ahead of the insert and not on the selection path. The gauge moves by
    /// one rather than to the count read back: two inserts racing would
    /// publish their counts in either order, and nothing rewrites the gauge
    /// until the next change. Inserts of different shards that race for the
    /// last room each take it: the map can stand past the most by a key a
    /// shard.
    fn admits_new_entry(&self) -> bool {
        // A full map refuses by a read alone, and a key that enters writes
        // the count once: no writer tries twice.
        if self.is_full() {
            return false;
        }
        self.entries.fetch_add(1, Ordering::Relaxed);
        Metrics::manual_policy_cache_entry_added();
        true
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
        // A full map records nothing: the key is not copied. It is looked
        // up once more first: an entry made since the miss above takes the
        // pin.
        if self.is_full() {
            if let Some(mut entry) = self.routing_map.get_mut(key) {
                front(&mut entry);
            }
            return;
        }
        match self.routing_map.entry(RoutingId::new(key)) {
            Entry::Occupied(mut entry) => front(entry.get_mut()),
            Entry::Vacant(entry) => {
                if self.admits_new_entry() {
                    entry.insert(Node {
                        candi_worker_urls: vec![url.to_string()],
                        last_access: Instant::now(),
                    });
                }
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
        // A full map records nothing: the key is placed with no copy of it.
        // It is looked up once more first: a pin written since the miss
        // above is followed. The test under the entry's lock, below, holds
        // for a writer that found room here and none there.
        if self.is_full() {
            if let Some(mut entry) = self.routing_map.get_mut(routing_id) {
                return occupied(&mut entry);
            }
            let selected_idx = self.select_new_worker(workers, healthy_indices);
            return (selected_idx, ExecutionBranch::Vacant);
        }
        match self.routing_map.entry(RoutingId::new(routing_id)) {
            Entry::Occupied(mut entry) => occupied(entry.get_mut()),
            Entry::Vacant(entry) => {
                let selected_idx = self.select_new_worker(workers, healthy_indices);
                if self.admits_new_entry() {
                    entry.insert(Node {
                        candi_worker_urls: vec![workers[selected_idx].url().to_string()],
                        last_access: Instant::now(),
                    });
                }
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

    fn honors_routing_key(&self) -> bool {
        true
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
#[path = "../../tests/unit/policies/manual/tests.rs"]
mod tests;
