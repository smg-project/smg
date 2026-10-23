//! Load balancing policies for the router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc};

use crate::{
    config::CacheIndexKind,
    worker::{load_state::LoadReceiver, runtime::RoutingState, HashRing, Worker},
};

mod bucket;
mod cache_aware;
mod cache_namespace;
mod consistent_hashing;
pub mod cost;
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
pub use cache_aware::{CacheAwarePolicy, TreeHandle, TreeKind};
pub use cache_namespace::{CacheNamespace, TEXT_MARKER_LEN, TOKEN_MARKER_LEN};
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
    /// Now uses `Arc<dyn Worker>` for better performance and to avoid unnecessary cloning.
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

    /// Check if this policy needs request text for routing decisions. A
    /// policy's answer is a constant for its lifetime: the registry reads
    /// it where the policy is registered, not per request.
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Whether this policy reads the request text when the request's token
    /// ids are available as well. A policy that routes on the ids and falls
    /// back to the text only without them answers `false`, so a tokenized
    /// request's selection snapshot carries the ids alone instead of a copy
    /// of the whole rendered prompt next to them. Defaults to
    /// [`Self::needs_request_text`]; a constant for the policy's lifetime
    /// as well.
    fn needs_request_text_with_tokens(&self) -> bool {
        self.needs_request_text()
    }

    /// Hand the policy the worker monitor's load feed: the shared snapshot
    /// of every worker's latest load report, published per poll tick and
    /// per window of pushed records. A policy that scores on backend load
    /// keeps the receiver and reads the current snapshot per decision (one
    /// `Arc` clone, no copy); the default ignores it.
    fn set_load_receiver(&self, _rx: Option<LoadReceiver>) {}

    /// Whether routing reads the load feed. With load monitoring disabled,
    /// the `WorkerMonitor` polls a group only while something needs the
    /// data. A policy's answer is a constant for its lifetime.
    fn needs_backend_loads(&self) -> bool {
        false
    }

    /// Whether this policy weighs a worker's load against the cache it holds
    /// for the request and moves a request off a saturated holder itself
    /// (cache-aware's count-pressure spill). Disaggregated pair placement
    /// then leaves the absolute overload veto off a prefill pool this policy
    /// places instead of steering a returning session away from the worker
    /// holding its prefix (see `placement::select_pair`). A constant for the
    /// policy's lifetime.
    fn balances_load_against_affinity(&self) -> bool {
        false
    }

    /// Drop the state this policy keeps on a removed worker (a load-aware
    /// policy's dispatch tally), so it does not accumulate under worker
    /// churn (autoscaling, rolling updates). Default is a no-op for
    /// stateless policies.
    fn remove_worker(&self, _url: &str) {
        // Default: no-op for policies that don't keep per-worker state
    }

    /// Reconcile the state this policy booked on `worker_url` against the
    /// router's live count of requests in flight there.
    ///
    /// Every request end reaches [`Self::on_request_complete`] through the
    /// worker's load guard; this is the safety net behind it. The worker
    /// monitor calls it once per load poll with the requests the router still
    /// holds on the worker. A dispatch books once and a completion releases
    /// once, so whatever a policy holds beyond `in_flight` is a completion
    /// that never arrived (a worker that moved under another policy, a sink
    /// never installed) and is released here rather than kept for good.
    /// Default: no-op for policies that book nothing.
    fn reconcile_in_flight(&self, _worker_url: &str, _in_flight: usize) {}

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Whether a built-in policy applies the complete [`Worker::is_available`]
/// predicate before selecting. Other policies keep the caller-side filter.
pub(crate) fn policy_filters_unavailable_workers(policy: &dyn LoadBalancingPolicy) -> bool {
    let policy = policy.as_any();
    policy.is::<RandomPolicy>()
        || policy.is::<RoundRobinPolicy>()
        || policy.is::<LeastLoadPolicy>()
        || policy.is::<PowerOfTwoPolicy>()
        || policy.is::<CacheAwarePolicy>()
        || policy.is::<ManualPolicy>()
        || policy.is::<PassthroughPolicy>()
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
    /// (relative). The margin is capped at the larger of the in-flight
    /// count's natural swing at the pool's current mean (two standard
    /// deviations, never under four requests unless this value is lower) and
    /// the held prefix's price (`balance_prefix_margin_tokens`), so a pool
    /// whose per-worker in-flight count stays below this value can still
    /// spill. Outside the fleet-wide KV-pressure fallback, a request with no
    /// cache match routes among the workers under the same gate, so a burst
    /// of misses is bounded by the same margins.
    pub balance_abs_threshold: usize,
    /// Relative load margin (multiple of the healthy-fleet mean) for the
    /// per-request candidate gate; fires only together with
    /// `balance_abs_threshold`.
    pub balance_rel_threshold: f32,
    /// Price of a held prefix in the candidate gate's absolute margin, in
    /// tokens per request: a holder's margin is at least the tokens of the
    /// request's prefix it holds over this, rounded up, up to
    /// `balance_abs_threshold`. A spill gives the held prefix up and the
    /// spill target prefills it again, so a deep prefix (a long
    /// conversation's follow-up) keeps its holder through a skew that would
    /// move a shallow one. `0` (the default) leaves the margin to the swing
    /// alone. Suits prefill pools serving long multi-turn sessions (e.g.
    /// 1024, the mean prefill LeastLoad prices a queued request at); decode
    /// picks are never priced, as a decode worker prefills nothing again.
    pub balance_prefix_margin_tokens: u32,
    /// Credit a pick on a worker holding the request's prefix with the
    /// prompt it still has to prefill (its token count less the held tokens,
    /// never under the mean prefill nor over the whole prompt) instead of the
    /// whole prompt, in the expected-wait estimate later picks read: a hit on
    /// a long cached conversation is not booked as its whole context's work.
    /// Misses, spills and decode picks keep the whole prompt. Off by default.
    pub balance_uncached_credit: bool,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
    /// Backend KV cache block size (tokens per block) for event-driven routing.
    /// Used by `compute_request_content_hashes` to chunk request tokens into blocks.
    /// Must match the backend's block size. Default: 16 (one engine's page size).
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
    /// Worker selection policy run over the gathered per-worker inputs, by
    /// name from [`cost::POLICY_NAMES`]. `None` is [`cost::DEFAULT_POLICY`],
    /// the affinity-group decision this policy has always made.
    pub selection_policy: Option<String>,
    /// Lifetime of optimistic dispatch bookings (predicted prefill and prefix
    /// placement charged to the chosen worker before the engine reports it).
    /// `0` disables (default); set a little above the engine's event lag.
    pub selection_accounting_ttl_ms: u64,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            balance_prefix_margin_tokens: 0,
            balance_uncached_credit: false,
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
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
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
/// in hand, never a second walk. A selection whose caller lifted that veto
/// ([`SelectWorkerInfo::overload_lifted`]) tests every other veto instead
/// ([`RoutingState::eligible_ignoring_overload`]).
pub(crate) fn get_healthy_worker_indices(
    workers: &[Arc<dyn Worker>],
    info: &SelectWorkerInfo,
) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| {
            if info.overload_lifted {
                w.routing_state().eligible_ignoring_overload()
            } else {
                w.is_available()
            }
        })
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

/// Which PD role a selection is for. `Single` is non-PD (the default) and keeps
/// routing-key stickiness byte-identical to the behaviour before roles existed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WorkerRole {
    #[default]
    Single,
    Prefill,
    Decode,
}

impl WorkerRole {
    /// Prefix used to namespace sticky routing IDs per role. `Single` is empty so
    /// non-PD entries are unchanged.
    pub fn routing_id_prefix(self) -> &'static str {
        match self {
            WorkerRole::Single => "",
            WorkerRole::Prefill => "prefill:",
            WorkerRole::Decode => "decode:",
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
    /// The request's cache partition (cache salt / extra key / LoRA), when
    /// set. Cache-aware routing keys its affinity under it so requests the
    /// engine cannot serve from the same cache never match each other.
    pub cache_namespace: Option<CacheNamespace>,
    /// Session key derived from the request body's `rid` (routers with typed
    /// body access populate it); used when no valid routing-key header is
    /// present under the routing-key override.
    pub rid_key: Option<&'a str>,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
    /// Which PD role this selection is for (default `Single`); namespaces
    /// header-based sticky routing so prefill and decode stick independently.
    pub role: WorkerRole,
    /// The absolute overload veto does not apply to this selection: the
    /// pool's policy balances load against cache affinity itself
    /// ([`LoadBalancingPolicy::balances_load_against_affinity`]), so it
    /// takes a worker over the thresholds as it takes one under them, even
    /// when every worker is over them (health, the circuit breaker and the
    /// liveness veto still apply). Set by disaggregated pair placement for
    /// such a prefill pool (see `placement::select_pair`); every other pool
    /// keeps the veto and the least-loaded steering default.
    pub overload_lifted: bool,
}

impl SelectWorkerInfo<'_> {
    /// Whether a worker in routing `state` is a candidate for this
    /// selection: [`RoutingState::eligible`], or every veto but the overload
    /// one when the caller lifted that veto ([`Self::overload_lifted`]).
    pub(crate) fn admits(&self, state: RoutingState) -> bool {
        if self.overload_lifted {
            state.eligible_ignoring_overload()
        } else {
            state.eligible()
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{runtime::StallReason, BasicWorkerBuilder, WorkerType};

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
        let indices = get_healthy_worker_indices(&workers, &SelectWorkerInfo::default());
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_status(WorkerStatus::NotReady);
        let indices = get_healthy_worker_indices(&workers, &SelectWorkerInfo::default());
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
            let indices = get_healthy_worker_indices(&workers, &SelectWorkerInfo::default());
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

        assert_eq!(
            get_healthy_worker_indices(&workers, &SelectWorkerInfo::default()),
            vec![0, 1, 2]
        );

        // A busy worker is still eligible: load alone is not the veto.
        let _guard = crate::worker::WorkerLoadGuard::new(Arc::clone(&workers[1]), None);
        assert_eq!(
            get_healthy_worker_indices(&workers, &SelectWorkerInfo::default()),
            vec![0, 1, 2]
        );

        workers[1].set_overloaded(true);
        assert_eq!(
            get_healthy_worker_indices(&workers, &SelectWorkerInfo::default()),
            vec![0, 2]
        );

        // A caller that lifted the veto (every worker of its pool over the
        // thresholds) gets the worker back; the liveness veto still applies.
        let lifted = SelectWorkerInfo {
            overload_lifted: true,
            ..Default::default()
        };
        assert_eq!(get_healthy_worker_indices(&workers, &lifted), vec![0, 1, 2]);
        workers[1].set_stall(Some(StallReason::Wedged));
        assert_eq!(get_healthy_worker_indices(&workers, &lifted), vec![0, 2]);
        workers[1].set_stall(None);

        workers[1].set_overloaded(false);
        assert_eq!(
            get_healthy_worker_indices(&workers, &SelectWorkerInfo::default()),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn test_select_worker_info_role_defaults_to_single() {
        let info = SelectWorkerInfo::default();
        assert_eq!(info.role, WorkerRole::Single);
        assert_eq!(WorkerRole::Single.routing_id_prefix(), "");
        assert_eq!(WorkerRole::Prefill.routing_id_prefix(), "prefill:");
        assert_eq!(WorkerRole::Decode.routing_id_prefix(), "decode:");
    }

    #[test]
    fn only_policies_with_complete_availability_checks_skip_prefiltering() {
        for name in [
            "random",
            "round_robin",
            "least_load",
            "power_of_two",
            "cache_aware",
            "manual",
            "passthrough",
        ] {
            let policy = PolicyFactory::create_by_name(name).unwrap();
            assert!(
                policy_filters_unavailable_workers(policy.as_ref()),
                "{name}"
            );
        }

        for name in ["bucket", "consistent_hashing", "prefix_hash"] {
            let policy = PolicyFactory::create_by_name(name).unwrap();
            assert!(
                !policy_filters_unavailable_workers(policy.as_ref()),
                "{name}"
            );
        }
    }

    /// The allowlist above is only sound if every listed policy actually
    /// applies the complete `is_available()` predicate itself: callers skip
    /// the pre-filter for them. Two scenarios pin the two halves:
    /// the overload veto (shared with the weaker `is_healthy_and_eligible`
    /// predicate) and an open circuit breaker — the one signal that
    /// separates the full predicate from the weak one, so a policy that
    /// regresses to the weak predicate fails here.
    #[test]
    fn allowlisted_policies_reject_unavailable_workers() {
        use std::{sync::Arc, time::Duration};

        use openai_protocol::worker::WorkerStatus;

        use crate::worker::{circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder, Worker};

        fn worker(url: &str) -> Arc<dyn Worker> {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .circuit_breaker_config(CircuitBreakerConfig {
                        failure_threshold: 1,
                        success_threshold: 1,
                        timeout_duration: Duration::from_secs(600),
                        window_duration: Duration::from_secs(600),
                    })
                    .build(),
            );
            worker.set_status(WorkerStatus::Ready);
            worker
        }

        fn overloaded(url: &str) -> Arc<dyn Worker> {
            let worker = worker(url);
            worker.set_overloaded(true);
            worker
        }

        fn breaker_open(url: &str) -> Arc<dyn Worker> {
            let worker = worker(url);
            worker.record_outcome(500);
            assert!(!worker.is_available(), "breaker must be open");
            assert!(
                worker.is_healthy_and_eligible(),
                "the weak predicate must still accept it — that contrast is \
                 what this scenario tests"
            );
            worker
        }

        for name in [
            "random",
            "round_robin",
            "least_load",
            "power_of_two",
            "cache_aware",
            "manual",
            "passthrough",
        ] {
            let policy = PolicyFactory::create_by_name(name).unwrap();

            // Every worker vetoed or circuit-broken: the policy must come up
            // empty on its own.
            for unavailable in [
                [overloaded("http://a:1"), overloaded("http://b:1")],
                [breaker_open("http://a:1"), breaker_open("http://b:1")],
            ] {
                assert_eq!(
                    policy.select_worker(&unavailable, &SelectWorkerInfo::default()),
                    None,
                    "{name} selected an unavailable worker"
                );
            }

            // Mixed pools: the selection must exist and land on the
            // available worker.
            for mixed in [
                [worker("http://a:1"), overloaded("http://b:1")],
                [worker("http://a:1"), breaker_open("http://b:1")],
            ] {
                for _ in 0..16 {
                    let idx = policy
                        .select_worker(&mixed, &SelectWorkerInfo::default())
                        .unwrap_or_else(|| {
                            panic!("{name} found no worker despite an available one")
                        });
                    assert_eq!(idx, 0, "{name} selected the unavailable worker");
                }
            }
        }
    }
}
