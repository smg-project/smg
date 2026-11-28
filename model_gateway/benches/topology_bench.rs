use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use smg::router::topology::{
    extract_conversation_prefix, PrefixCache, RendezvousRouter, TopologyRouter, WorkerNode,
};

fn bench_extract_prefix(c: &mut Criterion) {
    let json = br#"{"messages": [{"role": "user", "content": "Write a Python function"}]}"#;
    c.bench_function("extract_prefix", |b| {
        b.iter(|| {
            let _ = extract_conversation_prefix(black_box(json));
        })
    });
}

fn bench_cache_lookup(c: &mut Criterion) {
    let cache = PrefixCache::new(10000);
    cache.insert(0x1234, "worker-a".to_string());
    c.bench_function("cache_lookup_hit", |b| {
        b.iter(|| {
            let _ = cache.lookup(black_box(0x1234));
        })
    });
    c.bench_function("cache_lookup_miss", |b| {
        b.iter(|| {
            let _ = cache.lookup(black_box(0x5678));
        })
    });
}

fn bench_cache_insert(c: &mut Criterion) {
    let cache = PrefixCache::new(10000);
    c.bench_function("cache_insert", |b| {
        b.iter(|| {
            cache.insert(black_box(0x1234), black_box("worker-a".to_string()));
        })
    });
}

fn bench_hrw_route(c: &mut Criterion) {
    let workers = (0..10)
        .map(|i| WorkerNode::new(format!("worker-{}", i), format!("addr-{}", i), 0.5))
        .collect::<Vec<_>>();
    let router = RendezvousRouter::new(workers);
    c.bench_function("hrw_route_10_workers", |b| {
        b.iter(|| {
            let _ = router.route(black_box(0x1234));
        })
    });
}

fn bench_topology_route(c: &mut Criterion) {
    let workers = (0..10)
        .map(|i| WorkerNode::new(format!("worker-{}", i), format!("addr-{}", i), 0.5))
        .collect::<Vec<_>>();
    let router = TopologyRouter::new(workers, 10000);
    let json = br#"{"messages": [{"role": "user", "content": "Write a Python function"}]}"#;
    c.bench_function("topology_route_10_workers", |b| {
        b.iter(|| {
            let _ = router.route(black_box(json));
        })
    });
}

criterion_group!(
    benches,
    bench_extract_prefix,
    bench_cache_lookup,
    bench_cache_insert,
    bench_hrw_route,
    bench_topology_route
);
criterion_main!(benches);
