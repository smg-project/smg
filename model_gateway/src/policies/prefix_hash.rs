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
//! ## Complexity, per request
//!
//! - Eligibility and load: one read per worker of the pool, into one list
//!   (the load sum and the least-loaded fallback read it)
//! - Hash computation: O(prefix_length) per boundary level, its ring key
//!   spelled into a stack buffer
//! - Ring lookup: O(log n) binary search, each probed node found in the list
//!   by a scan of its URLs
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

use std::{cell::Cell, sync::Arc};

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
pub(super) const CHARS_PER_TOKEN: usize = 4;

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

    /// The ring key of a prefix hash: its sixteen lowercase hex digits, the
    /// spelling the ring has always hashed, written into `buf`.
    #[expect(
        clippy::expect_used,
        reason = "sixteen hex digits are ASCII, so the buffer is always a str"
    )]
    fn ring_key(prefix_hash: u64, buf: &mut [u8; 16]) -> &str {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = DIGITS[((prefix_hash >> (60 - 4 * i)) & 0xf) as usize];
        }
        std::str::from_utf8(buf).expect("hex digits are ASCII")
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
            // The ring answers with a URL; its entry in the list is found by
            // a scan (the last eligible worker at that URL, as the map this
            // replaced answered).
            let healthy_at = |url: &str| {
                healthy_workers
                    .iter()
                    .rfind(|(_, worker)| worker.url() == url)
            };

            // Deepest ring target, kept as the all-overloaded fallback.
            let mut initial: Option<(usize, usize)> = None;
            let mut key_buf = [0u8; 16];
            for &(level, key_bytes) in candidates {
                let prefix_hash = xxhash_rust::xxh3::xxh3_64(key_bytes);
                let key = Self::ring_key(prefix_hash, &mut key_buf);

                // Find this level's worker from the ring; the probe that
                // accepts the ring's answer keeps the entry it found, one
                // scan per probed node.
                let found = Cell::new(None);
                if ring
                    .find_healthy_url(key, |url| {
                        found.set(healthy_at(url));
                        found.get().is_some()
                    })
                    .is_none()
                {
                    continue;
                }
                let Some(&(idx, worker)) = found.get() else {
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
            worker = result.map_or("none", |idx| workers[idx].redacted_url()),
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
mod tests;
