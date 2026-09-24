use llm_tokenizer::{
    cache::{cache_activity_stats, L1Cache},
    traits::{Encoder, Encoding},
    L0Cache,
};

struct Bytes;
impl Encoder for Bytes {
    fn encode(&self, input: &str, _: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(input.bytes().map(u32::from).collect()))
    }

    fn encode_batch(&self, inputs: &[&str], special: bool) -> anyhow::Result<Vec<Encoding>> {
        inputs.iter().map(|s| self.encode(s, special)).collect()
    }
}

// A dedicated test process isolates the global totals from other cache tests.
#[test]
fn activity_counts_lookups_reuse_and_capacity_evictions_across_instances() {
    let initial = cache_activity_stats();
    assert!(initial
        .iter()
        .all(|s| s.hits + s.misses + s.evictions + s.reused_bytes == 0));
    {
        let l0 = L0Cache::new(1);
        assert!(l0.get("é", false).is_none());
        l0.insert("é".into(), false, Encoding::Plain(vec![1]));
        assert!(l0.get("é", false).is_some());
        assert!(l0.get("é", true).is_none());
        l0.insert("b".into(), false, Encoding::Plain(vec![2]));
        let stats = cache_activity_stats()[0];
        assert_eq!(
            (
                stats.hits,
                stats.misses,
                stats.evictions,
                stats.reused_bytes
            ),
            (1, 2, 1, 2)
        );
        l0.clear();
        assert_eq!(l0.stats().hits, 0);
        assert_eq!(cache_activity_stats()[0].hits, 1);

        // Replacement with spare capacity, clear and drop are not capacity eviction.
        let other = L0Cache::new(10);
        other.insert("x".into(), false, Encoding::Plain(vec![3]));
        other.insert("x".into(), false, Encoding::Plain(vec![4]));
        assert!(other.get("x", false).is_some());
    }
    assert_eq!(cache_activity_stats()[0].evictions, 1);

    {
        let l1 = L1Cache::new(20);
        assert!(l1.longest_prefix_match("plain", &["|"], false).is_none());
        assert!(l1.longest_prefix_match("é|tail", &["|"], false).is_none());
        l1.insert_at_boundaries("é|tail", &Bytes, &["|"], false)
            .unwrap();
        // Multiple probes (the longer prefix is absent) still count as one hit.
        let (_, offset) = l1
            .longest_prefix_match("é|new|tail", &["|"], false)
            .unwrap();
        assert_eq!(offset, 3);
        assert!(l1.longest_prefix_match("é|tail", &["|"], true).is_none());
        // Each entry fits individually; together they exceed the budget.
        l1.insert_at_boundaries("ab|tail", &Bytes, &["|"], false)
            .unwrap();
        let stats = cache_activity_stats()[1];
        assert_eq!(
            (
                stats.hits,
                stats.misses,
                stats.evictions,
                stats.reused_bytes
            ),
            (1, 3, 1, 3)
        );
        l1.clear();
        assert_eq!(l1.stats().hits, 0);
    }
    assert_eq!(cache_activity_stats()[1].evictions, 1);

    // Disabled caches do not count lookups; an L0 hit never probes L1.
    let before_disabled = cache_activity_stats();
    let disabled = llm_tokenizer::CachedTokenizer::new(
        std::sync::Arc::new(llm_tokenizer::mock::MockTokenizer::new()),
        llm_tokenizer::CacheConfig {
            enable_l0: false,
            enable_l1: false,
            ..Default::default()
        },
    );
    disabled.encode("Hello", false).unwrap();
    for (before, after) in before_disabled.iter().zip(cache_activity_stats()) {
        assert_eq!((before.hits, before.misses), (after.hits, after.misses));
    }
    let cached = llm_tokenizer::CachedTokenizer::new(
        std::sync::Arc::new(llm_tokenizer::mock::MockTokenizer::new()),
        llm_tokenizer::CacheConfig {
            enable_l0: true,
            enable_l1: true,
            ..Default::default()
        },
    );
    cached.encode("Hello", false).unwrap();
    let before_hit = cache_activity_stats();
    cached.encode("Hello", false).unwrap();
    let after_hit = cache_activity_stats();
    assert_eq!(after_hit[0].hits, before_hit[0].hits + 1);
    assert_eq!(
        (after_hit[1].hits, after_hit[1].misses),
        (before_hit[1].hits, before_hit[1].misses)
    );

    let before = cache_activity_stats();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                let l0 = L0Cache::new(4);
                let l1 = L1Cache::new(1024);
                l0.insert("x".into(), false, Encoding::Plain(vec![1]));
                l1.insert_at_boundaries("x|tail", &Bytes, &["|"], false)
                    .unwrap();
                for _ in 0..100 {
                    assert!(l0.get("x", false).is_some());
                    assert!(l0.get("missing", false).is_none());
                    assert!(l1.longest_prefix_match("x|tail", &["|"], false).is_some());
                    assert!(l1.longest_prefix_match("missing", &["|"], false).is_none());
                }
            });
        }
    });
    for (index, (previous, current)) in before.iter().zip(cache_activity_stats()).enumerate() {
        assert_eq!(current.hits - previous.hits, 400);
        assert_eq!(current.misses - previous.misses, 400);
        assert_eq!(current.evictions, previous.evictions);
        assert_eq!(
            current.reused_bytes - previous.reused_bytes,
            400 * (index as u64 + 1)
        );
    }
}
