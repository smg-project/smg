//! Factory for creating load balancing policies

use std::sync::Arc;

use super::{
    BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwareLengthConfig, CacheAwareLengthPolicy,
    CacheAwarePolicy, ConsistentHashingPolicy, LeastLoadPolicy, LoadBalancingPolicy, ManualConfig,
    ManualPolicy, PassthroughPolicy, PowerOfTwoPolicy, PrefixHashConfig, PrefixHashPolicy,
    RandomPolicy, RoundRobinPolicy,
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
            PolicyConfig::PowerOfTwo { .. } => {
                // TODO: Pass load_check_interval_secs to WorkerMonitor for per-policy polling intervals.
                // Currently, WorkerMonitor uses RouterConfig.load_monitor_interval_secs globally.
                Arc::new(PowerOfTwoPolicy::new())
            }
            PolicyConfig::LeastLoad {
                kv_pressure_weight,
                mean_prefill_tokens,
                default_throughput,
                max_waiting_requests,
                ..
            } => {
                // TODO: Pass load_check_interval_secs to WorkerMonitor for per-policy polling intervals.
                // Currently, WorkerMonitor uses RouterConfig.load_monitor_interval_secs globally.
                Arc::new(LeastLoadPolicy::with_params(
                    *kv_pressure_weight,
                    *mean_prefill_tokens,
                    *default_throughput,
                    *max_waiting_requests,
                ))
            }
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
                };
                Arc::new(CacheAwarePolicy::with_config(config))
            }
            PolicyConfig::CacheAwareLength {
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
                chars_per_token,
                long_prefill_threshold,
                long_pool_max_load,
                short_pool_max_load,
            } => {
                let base = CacheAwareConfig {
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
                };
                let config = CacheAwareLengthConfig {
                    base,
                    chars_per_token: *chars_per_token,
                    long_prefill_threshold: *long_prefill_threshold,
                    long_pool_max_load: *long_pool_max_load,
                    short_pool_max_load: *short_pool_max_load,
                };
                Arc::new(CacheAwareLengthPolicy::with_config(config))
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
            "cache_aware_length" | "cacheawarelength" => {
                Some(Arc::new(CacheAwareLengthPolicy::new()))
            }
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
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_from_config() {
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random);
        assert_eq!(policy.name(), "random");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        assert_eq!(policy.name(), "round_robin");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Passthrough);
        assert_eq!(policy.name(), "passthrough");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        });
        assert_eq!(policy.name(), "power_of_two");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        });
        assert_eq!(policy.name(), "cache_aware");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAwareLength {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            chars_per_token: 4,
            long_prefill_threshold: 100_000,
            long_pool_max_load: 4,
            short_pool_max_load: 32,
        });
        assert_eq!(policy.name(), "cache_aware_length");
        // Verify config values are preserved, not just the policy name.
        let cal = policy
            .as_any()
            .downcast_ref::<CacheAwareLengthPolicy>()
            .unwrap();
        let cfg = cal.config_for_test();
        assert_eq!(cfg.chars_per_token, 4);
        assert_eq!(cfg.long_prefill_threshold, 100_000);
        assert_eq!(cfg.long_pool_max_load, 4);
        assert_eq!(cfg.short_pool_max_load, 32);
        assert_eq!(cfg.base.cache_threshold, 0.5);
        assert_eq!(cfg.base.block_size, 16);

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Bucket {
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        });
        assert_eq!(policy.name(), "bucket");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: Default::default(),
        });
        assert_eq!(policy.name(), "manual");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::ConsistentHashing);
        assert_eq!(policy.name(), "consistent_hashing");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PrefixHash {
            prefix_token_count: 100,
            load_factor: 0.8,
            balance_abs_threshold: 10,
            cache_boundaries: vec![2048, 8192],
        });
        assert_eq!(policy.name(), "prefix_hash");
    }

    #[tokio::test]
    async fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert_eq!(
            PolicyFactory::create_by_name("passthrough").unwrap().name(),
            "passthrough"
        );
        assert!(PolicyFactory::create_by_name("PASSTHROUGH").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert_eq!(
            PolicyFactory::create_by_name("cache_aware_length")
                .unwrap()
                .name(),
            "cache_aware_length"
        );
        assert_eq!(
            PolicyFactory::create_by_name("CacheAwareLength")
                .unwrap()
                .name(),
            "cache_aware_length"
        );
        assert!(PolicyFactory::create_by_name("bucket").is_some());
        assert!(PolicyFactory::create_by_name("Bucket").is_some());
        assert!(PolicyFactory::create_by_name("manual").is_some());
        assert!(PolicyFactory::create_by_name("Manual").is_some());
        assert!(PolicyFactory::create_by_name("consistent_hashing").is_some());
        assert!(PolicyFactory::create_by_name("ConsistentHashing").is_some());
        assert!(PolicyFactory::create_by_name("prefix_hash").is_some());
        assert!(PolicyFactory::create_by_name("PrefixHash").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }
}
