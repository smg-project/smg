//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc};

use openai_protocol::worker::WorkerLoadResponse;

use crate::{
    config::CacheIndexKind,
    worker::{HashRing, Worker},
};

mod bucket;
mod cache_aware;
mod cache_aware_length;
mod consistent_hashing;
mod dp_min_token;
mod factory;
mod least_load;
mod manual;
mod passthrough;
mod power_of_two;
mod prefix_hash;
mod random;
mod registry;
mod round_robin;
pub(crate) mod utils;

pub use bucket::BucketPolicy;
pub(crate) use cache_aware::{NoCacheStrategy, UncachedHint};
pub use cache_aware::{CacheAwarePolicy, TreeHandle, TreeKind};
pub use cache_aware_length::CacheAwareLengthPolicy;
pub use consistent_hashing::ConsistentHashingPolicy;
pub use dp_min_token::MinimumTokensPolicy;
pub use factory::PolicyFactory;
// Re-export PrefixMatchResult from kv_index for production use
pub use kv_index::PrefixMatchResult;
pub use least_load::LeastLoadPolicy;
pub use manual::{ManualConfig, ManualPolicy};
pub use passthrough::PassthroughPolicy;
pub use power_of_two::PowerOfTwoPolicy;
pub use prefix_hash::{PrefixHashConfig, PrefixHashPolicy};
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    ///
    /// # Arguments
    /// * `workers` - Available workers to select from
    /// * `info` - Additional information for routing decisions
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize>;

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, WorkerLoadResponse>) {
        // Default: no-op for policies that don't use load information
    }

    /// Whether routing consumes backend load snapshots, pushed via
    /// [`Self::update_loads`] or read from the monitor's watch feed. The
    /// `WorkerMonitor` skips backend load polling while no policy needs it.
    fn needs_backend_loads(&self) -> bool {
        false
    }

    /// Drop any cached per-worker state for a removed worker.
    ///
    /// Called when a worker leaves the registry so load-aware policies don't
    /// accumulate stale load reports under worker churn (autoscaling, rolling
    /// updates). Default is a no-op for stateless policies.
    fn remove_worker(&self, _url: &str) {
        // Default: no-op for policies that don't cache per-worker state
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait DPRankLoadPolicy: Send + Sync + Debug {
    fn select_dp_rank(&self, worker: &dyn Worker, estimated_cost: isize) -> Option<isize>;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    /// Absolute load margin for the per-request candidate gate: the selected
    /// worker spills to least-loaded when its load exceeds the healthy-fleet
    /// mean by this many requests AND by `balance_rel_threshold`. Requiring
    /// both keeps the gate quiet on steady-state variance at low means
    /// (absolute) without going blind to a deep queue at high means
    /// (relative).
    pub balance_abs_threshold: usize,
    /// Relative load margin (multiple of the healthy-fleet mean) for the
    /// per-request candidate gate; fires only together with
    /// `balance_abs_threshold`.
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
    /// Backend KV cache block size (tokens per block) for event-driven routing.
    /// Used by `compute_request_content_hashes` to chunk request tokens into blocks.
    /// Must match the backend's block size. Default: 16 (SGLang page size).
    pub block_size: usize,
    /// KV-usage **spread** (hottest minus coldest backend, 0.0–1.0) above which
    /// the pool is treated as imbalanced and cache affinity is abandoned for
    /// shortest-queue. This is the balance signal for long-context workloads
    /// where a few requests saturate one engine's KV without tripping the
    /// request-count thresholds; being backend-reported, it is invariant to the
    /// number of gateway replicas. Requires the backend to report `token_usage`
    /// (gRPC/`GetLoads`); falls back to the count spread when unavailable.
    /// `>= 1.0` disables it (default).
    pub balance_token_usage_threshold: f32,
    /// Backend KV-cache utilization **ceiling** (0.0–1.0): when the hottest
    /// engine exceeds it the pool is treated as imbalanced regardless of spread,
    /// shedding load off a critically-saturated engine. A safety valve, best set
    /// high (e.g. 0.9). Requires `token_usage`; `>= 1.0` disables it (default).
    pub overload_token_usage_threshold: f32,
    /// Anti-hotspot decay for event-driven overlap credit. Each candidate's
    /// overlap score is multiplied by `1 / (1 + overlap_decay * x)`, where `x`
    /// is the worker's waiting-prefill backlog (backend-reported waiting
    /// uncached tokens, in blocks, in excess of the minimum among candidates)
    /// divided by the request's block count. The least-backlogged candidate
    /// always keeps full credit; workers without load data are never decayed.
    /// `0.0` disables (default). Requires backend load reporting.
    pub overlap_decay: f32,
    /// Randomized selection among event-driven candidates. `0.0` (default) is
    /// exact argmax with the existing tie-breaks. Above zero, candidates are
    /// sampled by softmax over min-max-normalized scores divided by this
    /// temperature — scale-free (only relative position within the current
    /// score spread matters): small values mostly follow the best candidate,
    /// large values approach uniform.
    pub selection_temperature: f32,
    /// Index under-layer: `Tree` (default, radix prefix trees) or `Hash`
    /// (TTL'd exact-match placement map over `cache_boundaries` heads; the
    /// radix trees are neither consulted nor populated).
    pub cache_index: CacheIndexKind,
    /// Seconds a hash-index placement stays routable; should approximate
    /// serving-engine cache retention.
    pub cache_ttl_secs: u64,
    /// Ascending token positions at which serving engines retain reusable
    /// prefix state; the hash index keys request heads at these boundaries.
    pub cache_boundaries: Vec<usize>,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
            block_size: 16,
            // Both KV triggers disabled by default (>= 1.0 never trips). Set
            // balance e.g. 0.5 (spread) and/or overload e.g. 0.9 (ceiling).
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            // Both pressure knobs off by default: behavior is bit-identical
            // to pre-knob selection until explicitly tuned.
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: CacheIndexKind::Tree,
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        }
    }
}

/// Configuration for the cache_aware_length policy (long/short pool split).
///
/// Embeds [`CacheAwareConfig`] as `base` so the policy inherits all
/// cache_aware features (string tree, token tree, event-driven routing,
/// hash index, mesh sync, KV pressure) and adds 4 length-specific fields
/// for the no-cache long/short pool split.
#[derive(Debug, Clone)]
pub struct CacheAwareLengthConfig {
    /// Full cache_aware configuration: cache_threshold, balance thresholds,
    /// eviction, block_size, KV pressure knobs, overlap decay, selection
    /// temperature, cache_index, cache_ttl_secs, cache_boundaries.
    pub base: CacheAwareConfig,
    /// Divisor for char-level token estimation when `X-Prompt-Tokens` is
    /// absent: `uncached_tokens = (input_chars - matched_chars) /
    /// chars_per_token`.
    pub chars_per_token: usize,
    /// Uncached-prefill-token boundary between long and short requests.
    pub long_prefill_threshold: usize,
    /// Load ceiling for the long pool (`pool=long` workers).
    pub long_pool_max_load: usize,
    /// Load ceiling for the short pool (remaining workers).
    pub short_pool_max_load: usize,
}

impl Default for CacheAwareLengthConfig {
    fn default() -> Self {
        Self {
            base: CacheAwareConfig::default(),
            chars_per_token: 4,
            long_prefill_threshold: 100_000,
            // 0 would reject every worker as non-free; pick generous defaults so
            // the policy is usable out of the box before operators tune.
            long_pool_max_load: 4,
            short_pool_max_load: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Helper function to filter routable workers and return their indices.
///
/// [`Worker::is_available`] folds the absolute overload veto into the health +
/// circuit-breaker test it already performed, under the same guards and with
/// the same short-circuit order — the veto is one more test on a word already
/// in hand, never a second walk.
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_available())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Returns UNKNOWN_MODEL_ID for empty model_ids to ensure consistent behavior
/// across single-model and multi-model deployments.
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::worker::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// Which PD leg a selection is for. `Single` is non-PD (the default) and keeps
/// routing-key stickiness byte-identical to pre-leg behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WorkerLeg {
    #[default]
    Single,
    Prefill,
    Decode,
}

impl WorkerLeg {
    /// Prefix used to namespace sticky routing IDs per leg. `Single` is empty so
    /// non-PD entries are unchanged.
    pub fn routing_id_prefix(self) -> &'static str {
        match self {
            WorkerLeg::Single => "",
            WorkerLeg::Prefill => "prefill:",
            WorkerLeg::Decode => "decode:",
        }
    }
}

/// Information passed to policy for worker selection
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// Request text for cache-aware routing
    pub request_text: Option<&'a str>,
    /// Tokenized request for prefix-hash routing
    /// Used by PrefixHashPolicy for token-based prefix hashing
    /// The HTTP router substitutes a valid `x-smg-routing-tokens` hint here:
    /// the hint wins over body-derived tokens/text for selection.
    pub tokens: Option<&'a [u32]>,
    /// HTTP headers for header-based routing policies
    /// Policies can extract routing information from headers like:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by index
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    pub headers: Option<&'a http::HeaderMap>,
    /// Validated `x-smg-routing-key` hint (non-empty UTF-8, <= 128 bytes);
    /// consistent_hashing and prefix_hash prefer it over their other keying
    /// input.
    pub routing_key: Option<&'a str>,
    /// Session key derived from the request body's `rid` (routers with typed
    /// body access populate it); consumed by the routing-key override when
    /// its key source includes rid.
    pub rid_key: Option<&'a str>,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
    /// Which PD leg this selection is for (default `Single`); namespaces
    /// header-based sticky routing so prefill and decode stick independently.
    pub leg: WorkerLeg,
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_status(WorkerStatus::NotReady);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }

    /// Only `Ready` workers may be selected. Pending, NotReady, Failed, and
    /// Draining are all excluded. Draining specifically guards against
    /// routing new traffic to a worker that is being torn down.
    #[test]
    fn test_get_healthy_worker_indices_excludes_each_non_ready_status() {
        let cases = [
            (WorkerStatus::Pending, false),
            (WorkerStatus::Ready, true),
            (WorkerStatus::NotReady, false),
            (WorkerStatus::Failed, false),
            (WorkerStatus::Draining, false),
        ];

        for (status, expected_included) in cases {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("http://w:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("k")
                    .health_config(no_health_check())
                    .build(),
            );
            worker.set_status(status);
            let workers = vec![worker];
            let indices = get_healthy_worker_indices(&workers);
            assert_eq!(
                indices == vec![0],
                expected_included,
                "status {status:?} should be {}",
                if expected_included {
                    "included"
                } else {
                    "excluded"
                }
            );
        }
    }

    /// The overload veto rides the same eligibility filter every walking policy
    /// uses, so flagging a worker removes it everywhere at once — and clearing
    /// the flag re-admits it.
    #[test]
    fn get_healthy_worker_indices_honors_the_overload_veto() {
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://w{i}:8000"))
                        .worker_type(WorkerType::Regular)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();

        assert_eq!(get_healthy_worker_indices(&workers), vec![0, 1, 2]);

        // A busy worker is still eligible: load alone is not the veto.
        let _guard = crate::worker::WorkerLoadGuard::new(Arc::clone(&workers[1]), None);
        assert_eq!(get_healthy_worker_indices(&workers), vec![0, 1, 2]);

        workers[1].set_overloaded(true);
        assert_eq!(get_healthy_worker_indices(&workers), vec![0, 2]);

        workers[1].set_overloaded(false);
        assert_eq!(get_healthy_worker_indices(&workers), vec![0, 1, 2]);
    }

    #[test]
    fn test_select_worker_info_leg_defaults_to_single() {
        let info = SelectWorkerInfo::default();
        assert_eq!(info.leg, WorkerLeg::Single);
        assert_eq!(WorkerLeg::Single.routing_id_prefix(), "");
        assert_eq!(WorkerLeg::Prefill.routing_id_prefix(), "prefill:");
        assert_eq!(WorkerLeg::Decode.routing_id_prefix(), "decode:");
    }
}
