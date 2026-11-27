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
        load_metric: Default::default(),
    });
    assert_eq!(policy.name(), "power_of_two");

    let policy =
        PolicyFactory::create_from_config(&PolicyConfig::cache_aware(0.7, 10, 1.5, 30, 1000));
    assert_eq!(policy.name(), "cache_aware");

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
