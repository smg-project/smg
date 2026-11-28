//! Factory for creating load balancing policies

use std::sync::Arc;

use super::{
    BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwarePolicy, ConsistentHashingPolicy,
    LeastLoadPolicy, LoadBalancingPolicy, ManualConfig, ManualPolicy, PassthroughPolicy,
    PowerOfTwoPolicy, PrefixHashConfig, PrefixHashPolicy, RandomPolicy, RoundRobinPolicy,
};
use crate::config::PolicyConfig;

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration
    pub fn create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::Passthrough => Arc::new(PassthroughPolicy::new()),
            PolicyConfig::PowerOfTwo { load_metric, .. } => {
                Arc::new(PowerOfTwoPolicy::with_load_metric(*load_metric))
            }
            PolicyConfig::LeastLoad {
                kv_pressure_weight,
                mean_prefill_tokens,
                default_throughput,
                ..
            } => Arc::new(LeastLoadPolicy::with_params(
                *kv_pressure_weight,
                *mean_prefill_tokens,
                *default_throughput,
            )),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                block_size,
                balance_token_usage_threshold,
                overload_token_usage_threshold,
                overlap_decay,
                selection_temperature,
                cache_index,
                cache_ttl_secs,
                cache_boundaries,
                selection_policy,
                selection_accounting_ttl_ms,
                balance_prefix_margin_tokens,
                balance_uncached_credit,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                    block_size: *block_size,
                    balance_token_usage_threshold: *balance_token_usage_threshold,
                    overload_token_usage_threshold: *overload_token_usage_threshold,
                    overlap_decay: *overlap_decay,
                    selection_temperature: *selection_temperature,
                    cache_index: *cache_index,
                    cache_ttl_secs: *cache_ttl_secs,
                    cache_boundaries: cache_boundaries.clone(),
                    selection_policy: selection_policy.clone(),
                    selection_accounting_ttl_ms: *selection_accounting_ttl_ms,
                    balance_prefix_margin_tokens: *balance_prefix_margin_tokens,
                    balance_uncached_credit: *balance_uncached_credit,
                };
                Arc::new(CacheAwarePolicy::with_config(config))
            }
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                let config = BucketConfig {
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    bucket_adjust_interval_secs: *bucket_adjust_interval_secs,
                };
                Arc::new(BucketPolicy::with_config(config))
            }
            PolicyConfig::Manual {
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode,
            } => {
                let config = ManualConfig {
                    eviction_interval_secs: *eviction_interval_secs,
                    max_idle_secs: *max_idle_secs,
                    assignment_mode: *assignment_mode,
                };
                Arc::new(ManualPolicy::with_config(config))
            }
            PolicyConfig::ConsistentHashing => Arc::new(ConsistentHashingPolicy::new()),
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
                balance_abs_threshold,
                cache_boundaries,
            } => {
                let config = PrefixHashConfig {
                    prefix_token_count: *prefix_token_count,
                    load_factor: *load_factor,
                    balance_abs_threshold: *balance_abs_threshold,
                    cache_boundaries: cache_boundaries.clone(),
                };
                Arc::new(PrefixHashPolicy::new(config))
            }
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "passthrough" => Some(Arc::new(PassthroughPolicy::new())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "least_load" | "leastload" => Some(Arc::new(LeastLoadPolicy::new())),
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "bucket" => Some(Arc::new(BucketPolicy::new())),
            "manual" => Some(Arc::new(ManualPolicy::new())),
            "consistent_hashing" | "consistenthashing" => {
                Some(Arc::new(ConsistentHashingPolicy::new()))
            }
            "prefix_hash" | "prefixhash" => Some(Arc::new(PrefixHashPolicy::with_defaults())),
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/factory/tests.rs"]
mod tests;
