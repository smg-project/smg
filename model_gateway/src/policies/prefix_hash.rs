//! Prefix Hash routing policy for KV cache-aware load balancing
//!
//! A lightweight alternative to the full radix tree cache_aware policy.
//! Routes requests based on a hash of their prefix tokens to maximize
//! KV cache hits across workers.
//!
//! ## Algorithm
//!
//! 1. Extract first N tokens from the request (configurable prefix length),
//!    or the equivalent span of routing text when the request is untokenized.
//!    With `cache_boundaries` configured, N is instead the deepest boundary
//!    the request reaches, so requests sharing a boundary-aligned head
//!    co-hash regardless of their total length
//! 2. Hash the prefix using xxhash for fast, stable hashing
//! 3. Use consistent hash ring to find the target worker
//! 4. If worker is overloaded (load above both the relative and absolute
//!    margins over average), retry each shallower boundary, then find least
//!    loaded
//! 5. Return least loaded worker that passes load check, or initial if all overloaded
//!
//! ## Complexity
//!
//! - Hash computation: O(prefix_length)
//! - Ring lookup: O(log n) binary search
//! - Load balance fallback: O(n) scan for least loaded
//!
//! ## Comparison with cache_aware
//!
//! | Aspect          | prefix_hash       | cache_aware (radix) |
//! |-----------------|-------------------|---------------------|
//! | Lookup          | O(log n)          | O(prefix_len)       |
//! | Memory          | O(workers × vn)   | O(total_tokens)     |
//! | Update          | O(1)              | O(prefix_len)       |
//! | Precision       | Prefix grouping   | Exact matching      |
//!
//! prefix_hash trades optimal cache utilization for predictable O(log n) performance.

use std::sync::Arc;

use tracing::debug;

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::{observability::metrics::Metrics, worker::Worker};

/// Configuration for the PrefixHash load balancing policy
#[derive(Debug, Clone)]
pub struct PrefixHashConfig {
    /// Number of prefix tokens to use for hashing, or `CHARS_PER_TOKEN` times
    /// as many characters when the request carries no tokens.
    /// Longer prefixes = more precise routing but less grouping.
    /// Shorter prefixes = more requests grouped together.
    /// Default: 256 tokens (~1 paragraph of text)
    pub prefix_token_count: usize,

    /// Relative load threshold for the overload check.
    /// A worker counts as overloaded once its load exceeds the average by
    /// this multiple as well as by `balance_abs_threshold`, at which point
    /// the request goes to the least loaded worker that is not overloaded.
    /// Default: 1.25 (125% of average load)
    pub load_factor: f64,

    /// Absolute load difference threshold for the overload check.
    /// A worker only counts as overloaded once its load exceeds the average
    /// by this many requests as well as by `load_factor`. Without it the
    /// check is purely relative, so its false-positive rate grows as each
    /// router replica observes a smaller share of a worker's true load.
    /// Default: 10 requests
    pub balance_abs_threshold: usize,

    /// Resolved copy of the shared `cache_boundaries` setting: ascending
    /// token positions at which serving engines retain reusable prefix
    /// state. When non-empty, requests hash at the deepest boundary they
    /// reach instead of at `prefix_token_count`.
    pub cache_boundaries: Vec<usize>,
}

impl Default for PrefixHashConfig {
    fn default() -> Self {
        Self {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: Vec::new(),
        }
    }
}

/// Execution branch for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    NoHealthyWorkers,
    NoRoutingKey,
    RingHit,
    LoadBalanceWalk,
    FallbackLeastLoad,
}

impl Branch {
    #[inline]
    const fn as_str(self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::NoRoutingKey => "no_routing_key",
            Self::RingHit => "ring_hit",
            Self::LoadBalanceWalk => "load_balance_walk",
            Self::FallbackLeastLoad => "fallback_least_load",
        }
    }
}

/// Characters of routing text treated as one token's worth of prefix. Four is
/// the usual rule of thumb for English text, so an untokenized request hashes
/// roughly the same span of the prompt as `prefix_token_count` would cover.
const CHARS_PER_TOKEN: usize = 4;

/// Prefix Hash load balancing policy
///
/// Routes requests based on prefix token hash for KV cache locality.
/// Uses consistent hashing with bounded load balancing.
#[derive(Debug)]
pub struct PrefixHashPolicy {
    config: PrefixHashConfig,
}

impl PrefixHashPolicy {
    /// Create a new PrefixHashPolicy with the given configuration
    pub fn new(config: PrefixHashConfig) -> Self {
        Self { config }
    }

    /// Create a new PrefixHashPolicy with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PrefixHashConfig::default())
    }

    /// Token prefix at the fixed legacy depth, as hashable bytes
    #[inline]
    fn fixed_depth_token_prefix<'a>(&self, tokens: &'a [u32]) -> &'a [u8] {
        let prefix_len = tokens.len().min(self.config.prefix_token_count);
        bytemuck::cast_slice(&tokens[..prefix_len])
    }

    /// Leading text of an untokenized request at the fixed legacy depth
    ///
    /// Only pre-tokenized requests carry token IDs; chat, completions and
    /// text-form generate requests reach the policy with routing text alone,
    /// and hashing it keeps them on a stable worker instead of leaving them
    /// unroutable.
    #[inline]
    fn fixed_depth_text_prefix<'a>(&self, text: &'a str) -> &'a [u8] {
        let budget = self
            .config
            .prefix_token_count
            .saturating_mul(CHARS_PER_TOKEN);
        let end = text
            .char_indices()
            .nth(budget)
            .map_or(text.len(), |(offset, _)| offset);

        &text.as_bytes()[..end]
    }

    /// Boundary-aligned `(level, key bytes)` candidates, deepest boundary
    /// first. Empty when no boundaries are configured or the request does
    /// not reach the smallest one (such a head-only request has nothing to
    /// group on; the caller keys it at the fixed legacy depth instead).
    fn token_boundary_prefixes<'a>(&self, tokens: &'a [u32]) -> Vec<(usize, &'a [u8])> {
        self.config
            .cache_boundaries
            .iter()
            .rev()
            .filter(|&&boundary| boundary <= tokens.len())
            .map(|&boundary| (boundary, bytemuck::cast_slice(&tokens[..boundary])))
            .collect()
    }

    /// Text analog of [`Self::token_boundary_prefixes`]: one pass over the
    /// leading `CHARS_PER_TOKEN`-scaled span collects every boundary the
    /// text reaches.
    fn text_boundary_prefixes<'a>(&self, text: &'a str) -> Vec<(usize, &'a [u8])> {
        let mut prefixes = Vec::new();
        let mut chars = text.char_indices();
        let mut consumed = 0usize;
        for &boundary in &self.config.cache_boundaries {
            let budget = boundary.saturating_mul(CHARS_PER_TOKEN);
            while consumed < budget && chars.next().is_some() {
                consumed += 1;
            }
            if consumed < budget {
                break;
            }
            let end = chars
                .clone()
                .next()
                .map_or(text.len(), |(offset, _)| offset);
            prefixes.push((boundary, &text.as_bytes()[..end]));
        }
        prefixes.reverse();
        prefixes
    }

    /// Index of the least loaded healthy worker
    fn least_loaded_healthy(workers: &[Arc<dyn Worker>]) -> Option<usize> {
        workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy_and_eligible())
            .min_by_key(|(_, w)| w.load())
            .map(|(idx, _)| idx)
    }

    /// Check if a worker's load is acceptable
    ///
    /// Overload requires clearing both the relative and the absolute margin,
    /// mirroring the imbalance test cache_aware uses.
    #[inline]
    fn load_ok(&self, worker_load: usize, total_load: usize, num_workers: usize) -> bool {
        if total_load == 0 || num_workers == 0 {
            return true;
        }

        // Average load per worker (with +1 to simulate incoming request)
        let avg_load = (total_load + 1) as f64 / num_workers as f64;
        let threshold = (avg_load * self.config.load_factor)
            .max(avg_load + self.config.balance_abs_threshold as f64);

        (worker_load as f64) <= threshold
    }

    /// Find worker using consistent hash ring with load balancing
    ///
    /// `candidates` holds `(level, key bytes)` pairs to try in order —
    /// deepest boundary first, level 0 for a fixed-depth key. The first
    /// candidate whose ring target passes the load check wins; when every
    /// target is overloaded, fall to the least loaded acceptable worker,
    /// then to the deepest target regardless.
    fn find_worker_with_load_balance(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        candidates: &[(usize, &[u8])],
    ) -> (Option<usize>, Branch, usize) {
        // Build healthy worker URL to index map
        let healthy_workers: Vec<(usize, &Arc<dyn Worker>)> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy_and_eligible())
            .collect();

        if healthy_workers.is_empty() {
            return (None, Branch::NoHealthyWorkers, 0);
        }

        // Calculate total load for load balancing
        let total_load: usize = healthy_workers.iter().map(|(_, w)| w.load()).sum();
        let num_workers = healthy_workers.len();

        // Use pre-computed ring if available
        if let Some(ref ring) = info.hash_ring {
            // Build URL to (index, worker) map for healthy workers
            let healthy_url_map: std::collections::HashMap<&str, (usize, &Arc<dyn Worker>)> =
                healthy_workers
                    .iter()
                    .map(|(idx, w)| (w.url(), (*idx, *w)))
                    .collect();

            // Deepest ring target, kept as the all-overloaded fallback.
            let mut initial: Option<(usize, usize)> = None;
            for &(level, key_bytes) in candidates {
                let prefix_hash = xxhash_rust::xxh3::xxh3_64(key_bytes);
                // Convert prefix hash to a ring key string for lookup
                let key = format!("{prefix_hash:016x}");

                // Find this level's worker from the ring
                let Some(initial_url) =
                    ring.find_healthy_url(&key, |url| healthy_url_map.contains_key(url))
                else {
                    continue;
                };
                let Some(&(idx, worker)) = healthy_url_map.get(initial_url) else {
                    continue;
                };

                // Check if this level's worker has acceptable load
                if self.load_ok(worker.load(), total_load, num_workers) {
                    return (Some(idx), Branch::RingHit, level);
                }
                if initial.is_none() {
                    initial = Some((idx, level));
                }
            }

            if let Some((initial_idx, initial_level)) = initial {
                // Every level's target overloaded, find least loaded healthy
                // worker. This is a simpler approach than walking the ring
                let least_loaded = healthy_workers
                    .iter()
                    .filter(|(_, w)| self.load_ok(w.load(), total_load, num_workers))
                    .min_by_key(|(_, w)| w.load());

                if let Some(&(idx, _)) = least_loaded {
                    return (Some(idx), Branch::LoadBalanceWalk, 0);
                }

                // All workers overloaded, use the deepest target anyway
                return (Some(initial_idx), Branch::LoadBalanceWalk, initial_level);
            }
        }

        // Fallback: no ring or ring lookup failed, use least loaded worker
        (
            Self::least_loaded_healthy(workers),
            Branch::FallbackLeastLoad,
            0,
        )
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch, usize) {
        if workers.is_empty() {
            return (None, Branch::NoHealthyWorkers, 0);
        }

        // A validated x-smg-routing-key hint overrides token/text keying.
        // Otherwise pre-tokenized requests hash their token prefix, the rest
        // hash the equivalent span of routing text.
        if let Some(key) = info.routing_key {
            return self.find_worker_with_load_balance(workers, info, &[(0, key.as_bytes())]);
        }
        match (info.tokens, info.request_text) {
            (Some(tokens), _) if !tokens.is_empty() => {
                let candidates = self.token_boundary_prefixes(tokens);
                if candidates.is_empty() {
                    return self.find_worker_with_load_balance(
                        workers,
                        info,
                        &[(0, self.fixed_depth_token_prefix(tokens))],
                    );
                }
                self.find_worker_with_load_balance(workers, info, &candidates)
            }
            (_, Some(text)) if !text.is_empty() => {
                let candidates = self.text_boundary_prefixes(text);
                if candidates.is_empty() {
                    return self.find_worker_with_load_balance(
                        workers,
                        info,
                        &[(0, self.fixed_depth_text_prefix(text))],
                    );
                }
                self.find_worker_with_load_balance(workers, info, &candidates)
            }
            // Nothing to hash: stay serviceable by falling back to load.
            _ => (Self::least_loaded_healthy(workers), Branch::NoRoutingKey, 0),
        }
    }
}

impl LoadBalancingPolicy for PrefixHashPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let (result, branch, level) = self.select_worker_impl(workers, info);
        Metrics::record_worker_prefix_hash_policy_branch(branch.as_str());
        debug!(
            branch = branch.as_str(),
            level,
            worker = result.map_or("none", |idx| workers[idx].url()),
            model_id = result.map_or("none", |idx| workers[idx].model_id()),
            "Prefix-hash selection"
        );
        result
    }

    fn name(&self) -> &'static str {
        "prefix_hash"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, HashRing, WorkerLoadGuard, WorkerType};

    fn create_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Regular)
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_prefix_hash_consistent_routing() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        // Same tokens should always route to same worker
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (first_result, _, _) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();

        // Verify consistency
        for _ in 0..10 {
            let (result, _, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(first_idx));
        }
    }

    /// The ring lookup routes on `is_healthy_and_eligible()`, so the absolute
    /// overload veto must move a key off its ring owner and back on recovery.
    #[test]
    fn overloaded_worker_is_skipped_by_the_ring_lookup() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let tokens: Vec<u32> = vec![11, 22, 33, 44, 55, 66, 77, 88];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let owner = policy.select_worker_impl(&workers, &info).0.unwrap();

        workers[owner].set_overloaded(true);
        let respilled = policy.select_worker_impl(&workers, &info).0.unwrap();
        assert_ne!(respilled, owner, "a vetoed ring owner must not be selected");

        workers[owner].set_overloaded(false);
        assert_eq!(
            policy.select_worker_impl(&workers, &info).0,
            Some(owner),
            "recovery re-admits the ring owner"
        );
    }

    #[test]
    fn test_different_prefixes_distribute() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let mut distribution = std::collections::HashMap::new();

        // Different token sequences should distribute across workers
        for i in 0..100 {
            let tokens: Vec<u32> = vec![i, i + 1, i + 2, i + 3];
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };

            let (result, _, _) = policy.select_worker_impl(&workers, &info);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(
            distribution.len() > 1,
            "Should distribute across workers, got {distribution:?}",
        );
    }

    #[test]
    fn test_shared_prefix_routes_same() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 5, // Only look at first 5 tokens
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        // Two sequences with same first 5 tokens should route to same worker
        let tokens1: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200, 300];
        let tokens2: Vec<u32> = vec![1, 2, 3, 4, 5, 999, 888, 777];

        let info1 = SelectWorkerInfo {
            tokens: Some(&tokens1),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            tokens: Some(&tokens2),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result1, _, _) = policy.select_worker_impl(&workers, &info1);
        let (result2, _, _) = policy.select_worker_impl(&workers, &info2);

        assert_eq!(result1, result2, "Same prefix should route to same worker");
    }

    #[test]
    fn test_untokenized_request_routes_consistently() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        // Chat and completions requests reach the policy with text only.
        let info = SelectWorkerInfo {
            tokens: None,
            request_text: Some("summarize the quarterly earnings report"),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (first_result, branch, _) = policy.select_worker_impl(&workers, &info);
        assert!(first_result.is_some(), "text alone must be routable");
        assert_eq!(branch, Branch::RingHit);

        for _ in 0..10 {
            let (result, _, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, first_result);
        }
    }

    #[test]
    fn test_shared_text_prefix_routes_same() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 2, // 8 characters of text
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let info1 = SelectWorkerInfo {
            request_text: Some("system: you are a helpful assistant"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            request_text: Some("system: you are a terse assistant"),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result1, _, _) = policy.select_worker_impl(&workers, &info1);
        let (result2, _, _) = policy.select_worker_impl(&workers, &info2);

        assert_eq!(
            result1, result2,
            "text past the prefix budget must not change the worker"
        );
    }

    #[test]
    fn test_text_prefix_budget_respects_char_boundaries() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 1, // 4 characters, mid-way through the emoji run
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let info = SelectWorkerInfo {
            request_text: Some("🌊🌊🌊🌊🌊🌊 tide report"),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result, branch, _) = policy.select_worker_impl(&workers, &info);
        assert!(result.is_some());
        assert_eq!(branch, Branch::RingHit);
    }

    #[test]
    fn test_tokens_take_priority_over_text() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let tokens: Vec<u32> = vec![7, 8, 9];
        let tokens_only = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let tokens_and_text = SelectWorkerInfo {
            tokens: Some(&tokens),
            request_text: Some("text that must not influence the ring key"),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result1, _, _) = policy.select_worker_impl(&workers, &tokens_only);
        let (result2, _, _) = policy.select_worker_impl(&workers, &tokens_and_text);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_routing_key_hint_overrides_token_and_text_keying() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let key_only = SelectWorkerInfo {
            routing_key: Some("session-42"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (key_only_result, branch, _) = policy.select_worker_impl(&workers, &key_only);
        assert!(key_only_result.is_some(), "a key alone must be routable");
        assert_eq!(branch, Branch::RingHit);

        // Same key with entirely different tokens and text keeps the worker.
        for tokens in [vec![1u32, 2, 3], vec![900u32, 901, 902, 903]] {
            let info = SelectWorkerInfo {
                routing_key: Some("session-42"),
                tokens: Some(&tokens),
                request_text: Some("unrelated prompt text"),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            let (result, _, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, key_only_result, "the routing key must win");
        }
    }

    #[test]
    fn test_routing_key_hint_consistent_and_distributes() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let info = SelectWorkerInfo {
            routing_key: Some("sticky-session"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (first, _, _) = policy.select_worker_impl(&workers, &info);
        for _ in 0..10 {
            let (result, _, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, first);
        }

        let mut distribution = std::collections::HashMap::new();
        for i in 0..100 {
            let key = format!("session-{i}");
            let info = SelectWorkerInfo {
                routing_key: Some(&key),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            let (result, _, _) = policy.select_worker_impl(&workers, &info);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }
        assert!(
            distribution.len() > 1,
            "distinct keys must not pile onto one worker, got {distribution:?}"
        );
    }

    #[test]
    fn test_no_routing_key_falls_back_to_load() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));
        let _busy = WorkerLoadGuard::new(workers[0].clone(), None);

        // Empty tokens, empty text
        let tokens: Vec<u32> = vec![];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            request_text: Some(""),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result, branch, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1), "a keyless request still has to be served");
        assert_eq!(branch, Branch::NoRoutingKey);

        // Neither field set
        let info_empty = SelectWorkerInfo {
            tokens: None,
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result2, branch2, _) = policy.select_worker_impl(&workers, &info_empty);
        assert_eq!(result2, Some(1));
        assert_eq!(branch2, Branch::NoRoutingKey);
    }

    #[test]
    fn test_no_healthy_workers() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000"]);
        workers[0].set_status(WorkerStatus::NotReady);

        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));
        let tokens: Vec<u32> = vec![1, 2, 3];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result, branch, _) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }

    #[test]
    fn test_load_ok_calculation() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            load_factor: 1.25,
            balance_abs_threshold: 0,
            ..Default::default()
        });

        // Total load 100, 4 workers -> avg 25.25, threshold 31.5625
        assert!(policy.load_ok(30, 100, 4));
        assert!(!policy.load_ok(35, 100, 4));

        // Edge cases
        assert!(policy.load_ok(0, 0, 4)); // No load = OK
        assert!(policy.load_ok(100, 0, 0)); // No workers = OK (shouldn't happen)
    }

    #[test]
    fn test_absolute_margin_absorbs_small_count_noise() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            load_factor: 1.25,
            balance_abs_threshold: 10,
            ..Default::default()
        });

        // Average 10.25 across 4 workers: the relative margin alone flags 13,
        // but 13 is within 10 requests of average so it stays acceptable.
        assert!(policy.load_ok(13, 40, 4));
        assert!(!policy.load_ok(21, 40, 4));
    }

    #[test]
    fn test_relative_margin_still_binds_at_high_load() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            load_factor: 1.25,
            balance_abs_threshold: 10,
            ..Default::default()
        });

        // Average 200.25: the relative margin (250.3) now exceeds the absolute
        // one (210.25), so it is the binding constraint.
        assert!(policy.load_ok(240, 800, 4));
        assert!(!policy.load_ok(260, 800, 4));
    }

    #[test]
    fn test_absolute_margin_defaults_on() {
        assert_eq!(PrefixHashConfig::default().balance_abs_threshold, 10);
    }

    #[test]
    fn test_policy_name() {
        let policy = PrefixHashPolicy::with_defaults();
        assert_eq!(policy.name(), "prefix_hash");
    }

    #[test]
    fn test_boundary_hashing_selects_deepest_applicable() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            cache_boundaries: vec![16, 64],
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        // 70 and 100 tokens sharing the first 64: both key at boundary 64,
        // so the divergent tails cannot split them.
        let tokens1: Vec<u32> = (0..70).collect();
        let tokens2: Vec<u32> = (0..64).chain(9000..9036).collect();
        let info1 = SelectWorkerInfo {
            tokens: Some(&tokens1),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            tokens: Some(&tokens2),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result1, branch1, level1) = policy.select_worker_impl(&workers, &info1);
        let (result2, _, level2) = policy.select_worker_impl(&workers, &info2);
        assert_eq!(result1, result2, "shared boundary head must co-hash");
        assert_eq!(branch1, Branch::RingHit);
        assert_eq!((level1, level2), (64, 64));

        // 63 tokens fall short of the deeper boundary and key at 16.
        let tokens3: Vec<u32> = (0..63).collect();
        let info3 = SelectWorkerInfo {
            tokens: Some(&tokens3),
            hash_ring: Some(ring),
            ..Default::default()
        };
        let (result3, _, level3) = policy.select_worker_impl(&workers, &info3);
        assert!(result3.is_some());
        assert_eq!(level3, 16);
    }

    #[test]
    fn test_below_smallest_boundary_keeps_fixed_depth_keying() {
        let boundaries = PrefixHashPolicy::new(PrefixHashConfig {
            cache_boundaries: vec![16, 64],
            ..Default::default()
        });
        let legacy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let tokens: Vec<u32> = (0..15).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result, branch, level) = boundaries.select_worker_impl(&workers, &info);
        let (legacy_result, _, _) = legacy.select_worker_impl(&workers, &info);
        assert_eq!(result, legacy_result);
        assert_eq!(branch, Branch::RingHit);
        assert_eq!(level, 0);
    }

    #[test]
    fn test_short_turn_and_long_followup_co_hash_at_shared_boundary() {
        // A short first turn and its much longer follow-up share only the
        // head; no single fixed depth can co-hash both, the boundary does.
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            cache_boundaries: vec![2048, 32768],
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let head: Vec<u32> = (0..2048).map(|i| i * 7 % 50000).collect();
        let turn1: Vec<u32> = head.iter().copied().chain(200_000..201_000).collect();
        let turn2: Vec<u32> = head.iter().copied().chain(500_000..514_952).collect();
        assert_eq!(turn1.len(), 3048);
        assert_eq!(turn2.len(), 17_000);

        let info1 = SelectWorkerInfo {
            tokens: Some(&turn1),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            tokens: Some(&turn2),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result1, branch1, level1) = policy.select_worker_impl(&workers, &info1);
        let (result2, branch2, level2) = policy.select_worker_impl(&workers, &info2);
        assert_eq!(result1, result2, "turns sharing the head must co-hash");
        assert_eq!((branch1, branch2), (Branch::RingHit, Branch::RingHit));
        assert_eq!((level1, level2), (2048, 2048));
    }

    #[test]
    fn test_overloaded_deep_target_retries_shallower_boundary() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            load_factor: 1.0,
            balance_abs_threshold: 0,
            cache_boundaries: vec![16, 32],
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        // Find a stream whose level-32 and level-16 ring targets differ.
        let mut tokens: Vec<u32> = Vec::new();
        let mut deep_idx = usize::MAX;
        let mut shallow_idx = usize::MAX;
        for seed in 0..1000u32 {
            tokens = (seed..seed + 32).collect();
            let full = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            let head = SelectWorkerInfo {
                tokens: Some(&tokens[..16]),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            deep_idx = policy.select_worker_impl(&workers, &full).0.unwrap();
            shallow_idx = policy.select_worker_impl(&workers, &head).0.unwrap();
            if deep_idx != shallow_idx {
                break;
            }
        }
        assert_ne!(deep_idx, shallow_idx, "no split stream found");

        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };

        // Overloaded level-32 target: retry lands on the level-16 target.
        let _deep_busy = [
            WorkerLoadGuard::new(workers[deep_idx].clone(), None),
            WorkerLoadGuard::new(workers[deep_idx].clone(), None),
        ];
        let (result, branch, level) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(shallow_idx));
        assert_eq!(branch, Branch::RingHit);
        assert_eq!(level, 16);

        // Both targets overloaded: the load-balance walk takes over.
        let _shallow_busy = [
            WorkerLoadGuard::new(workers[shallow_idx].clone(), None),
            WorkerLoadGuard::new(workers[shallow_idx].clone(), None),
        ];
        let (result, branch, level) = policy.select_worker_impl(&workers, &info);
        let spare = (0..3)
            .find(|i| *i != deep_idx && *i != shallow_idx)
            .unwrap();
        assert_eq!(result, Some(spare));
        assert_eq!(branch, Branch::LoadBalanceWalk);
        assert_eq!(level, 0);
    }

    #[test]
    fn test_text_boundary_co_hashing() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            cache_boundaries: vec![4], // 16 characters of text
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let info1 = SelectWorkerInfo {
            request_text: Some("shared-16-chars! then a question about tides"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            request_text: Some("shared-16-chars! and an unrelated follow-up about currents"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (result1, _, level1) = policy.select_worker_impl(&workers, &info1);
        let (result2, _, level2) = policy.select_worker_impl(&workers, &info2);
        assert_eq!(result1, result2, "shared scaled head must co-hash");
        assert_eq!((level1, level2), (4, 4));

        // Shorter than the scaled boundary: fixed-depth keying.
        let short = SelectWorkerInfo {
            request_text: Some("hi there"),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (short_result, short_branch, short_level) = policy.select_worker_impl(&workers, &short);
        assert!(short_result.is_some());
        assert_eq!(short_branch, Branch::RingHit);
        assert_eq!(short_level, 0);

        // Scaled boundary landing mid-way through a multibyte run stays on
        // a character boundary.
        let emoji = SelectWorkerInfo {
            request_text: Some("🌊🌊🌊🌊🌊🌊 tide report"),
            hash_ring: Some(ring),
            ..Default::default()
        };
        let (emoji_result, emoji_branch, emoji_level) = policy.select_worker_impl(&workers, &emoji);
        assert!(emoji_result.is_some());
        assert_eq!(emoji_branch, Branch::RingHit);
        assert_eq!(emoji_level, 4);
    }

    #[test]
    fn test_unset_boundaries_key_at_fixed_depth() {
        // Empty boundaries: the ring key is exactly the fixed-depth hash.
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 5,
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(workers.iter().map(|w| w.url())));

        let tokens: Vec<u32> = (0..40).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let (result, branch, level) = policy.select_worker_impl(&workers, &info);
        assert_eq!(branch, Branch::RingHit);
        assert_eq!(level, 0);

        let hash = xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&tokens[..5]));
        let key = format!("{hash:016x}");
        let expected_url = ring.find_healthy_url(&key, |_| true).unwrap();
        assert_eq!(workers[result.unwrap()].url(), expected_url);
    }
}
