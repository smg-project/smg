//! Prefix Hash routing policy for KV cache-aware load balancing
//!
//! A lightweight alternative to the full radix tree cache_aware policy.
//! Routes requests based on a hash of their prefix tokens to maximize
//! KV cache hits across workers.
//!
//! ## Algorithm
//!
//! 1. Extract first N tokens from the request (configurable prefix length),
//!    or the equivalent span of routing text when the request is untokenized
//! 2. Hash the prefix using xxhash for fast, stable hashing
//! 3. Use consistent hash ring to find the target worker
//! 4. If worker is overloaded (load above both the relative and absolute
//!    margins over average), find least loaded
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
}

impl Default for PrefixHashConfig {
    fn default() -> Self {
        Self {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
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

    /// Compute hash of prefix tokens using xxhash
    #[inline]
    fn compute_prefix_hash(&self, tokens: &[u32]) -> u64 {
        let prefix_len = tokens.len().min(self.config.prefix_token_count);
        let prefix = &tokens[..prefix_len];

        let bytes: &[u8] = bytemuck::cast_slice(prefix);
        xxhash_rust::xxh3::xxh3_64(bytes)
    }

    /// Compute hash of the leading text of an untokenized request
    ///
    /// Only pre-tokenized requests carry token IDs; chat, completions and
    /// text-form generate requests reach the policy with routing text alone,
    /// and hashing it keeps them on a stable worker instead of leaving them
    /// unroutable.
    #[inline]
    fn compute_text_prefix_hash(&self, text: &str) -> u64 {
        let budget = self
            .config
            .prefix_token_count
            .saturating_mul(CHARS_PER_TOKEN);
        let end = text
            .char_indices()
            .nth(budget)
            .map_or(text.len(), |(offset, _)| offset);

        xxhash_rust::xxh3::xxh3_64(&text.as_bytes()[..end])
    }

    /// Index of the least loaded healthy worker
    fn least_loaded_healthy(workers: &[Arc<dyn Worker>]) -> Option<usize> {
        workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
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
    fn find_worker_with_load_balance(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        prefix_hash: u64,
    ) -> (Option<usize>, Branch) {
        // Build healthy worker URL to index map
        let healthy_workers: Vec<(usize, &Arc<dyn Worker>)> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .collect();

        if healthy_workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Calculate total load for load balancing
        let total_load: usize = healthy_workers.iter().map(|(_, w)| w.load()).sum();
        let num_workers = healthy_workers.len();

        // Use pre-computed ring if available
        if let Some(ref ring) = info.hash_ring {
            // Convert prefix hash to a ring key string for lookup
            let key = format!("{prefix_hash:016x}");

            // Build URL to (index, worker) map for healthy workers
            let healthy_url_map: std::collections::HashMap<&str, (usize, &Arc<dyn Worker>)> =
                healthy_workers
                    .iter()
                    .map(|(idx, w)| (w.url(), (*idx, *w)))
                    .collect();

            // Find initial worker from ring
            if let Some(initial_url) =
                ring.find_healthy_url(&key, |url| healthy_url_map.contains_key(url))
            {
                if let Some(&(idx, worker)) = healthy_url_map.get(initial_url) {
                    let worker_load = worker.load();

                    // Check if initial worker has acceptable load
                    if self.load_ok(worker_load, total_load, num_workers) {
                        return (Some(idx), Branch::RingHit);
                    }

                    // Initial worker overloaded, find least loaded healthy worker
                    // This is a simpler approach than walking the ring
                    let least_loaded = healthy_workers
                        .iter()
                        .filter(|(_, w)| self.load_ok(w.load(), total_load, num_workers))
                        .min_by_key(|(_, w)| w.load());

                    if let Some(&(idx, _)) = least_loaded {
                        return (Some(idx), Branch::LoadBalanceWalk);
                    }

                    // All workers overloaded, use initial worker anyway
                    return (Some(idx), Branch::LoadBalanceWalk);
                }
            }
        }

        // Fallback: no ring or ring lookup failed, use least loaded worker
        (
            Self::least_loaded_healthy(workers),
            Branch::FallbackLeastLoad,
        )
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        if workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Pre-tokenized requests hash their token prefix, the rest hash the
        // equivalent span of routing text.
        let prefix_hash = match (info.tokens, info.request_text) {
            (Some(tokens), _) if !tokens.is_empty() => self.compute_prefix_hash(tokens),
            (_, Some(text)) if !text.is_empty() => self.compute_text_prefix_hash(text),
            // Nothing to hash: stay serviceable by falling back to load.
            _ => return (Self::least_loaded_healthy(workers), Branch::NoRoutingKey),
        };

        // Find worker using ring with load balancing
        self.find_worker_with_load_balance(workers, info, prefix_hash)
    }
}

impl LoadBalancingPolicy for PrefixHashPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_prefix_hash_policy_branch(branch.as_str());
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

        let (first_result, _) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();

        // Verify consistency
        for _ in 0..10 {
            let (result, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(first_idx));
        }
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

            let (result, _) = policy.select_worker_impl(&workers, &info);
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

        let (result1, _) = policy.select_worker_impl(&workers, &info1);
        let (result2, _) = policy.select_worker_impl(&workers, &info2);

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

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        assert!(first_result.is_some(), "text alone must be routable");
        assert_eq!(branch, Branch::RingHit);

        for _ in 0..10 {
            let (result, _) = policy.select_worker_impl(&workers, &info);
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

        let (result1, _) = policy.select_worker_impl(&workers, &info1);
        let (result2, _) = policy.select_worker_impl(&workers, &info2);

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

        let (result, branch) = policy.select_worker_impl(&workers, &info);
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

        let (result1, _) = policy.select_worker_impl(&workers, &tokens_only);
        let (result2, _) = policy.select_worker_impl(&workers, &tokens_and_text);

        assert_eq!(result1, result2);
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

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1), "a keyless request still has to be served");
        assert_eq!(branch, Branch::NoRoutingKey);

        // Neither field set
        let info_empty = SelectWorkerInfo {
            tokens: None,
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result2, branch2) = policy.select_worker_impl(&workers, &info_empty);
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

        let (result, branch) = policy.select_worker_impl(&workers, &info);
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
}
