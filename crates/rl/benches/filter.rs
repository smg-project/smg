//! Pins the cost of the RL pre-filter at a typical rollout fleet size.

use std::{hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, Criterion};
use smg_rl::{
    table::{NoopEvictionSink, RlTable, VersionSource},
    version::Version,
    VersionPolicy,
};

fn bench_eligible(c: &mut Criterion) {
    let table = RlTable::new(Arc::new(NoopEvictionSink));
    let urls: Vec<String> = (0..8).map(|i| format!("http://engine-{i}:30000")).collect();
    for (i, url) in urls.iter().enumerate() {
        let version = if i % 2 == 0 { "42" } else { "41" };
        table.set_version(url, "m", Version::parse(version), VersionSource::Api);
    }
    let candidates: Vec<(&str, &str)> = urls.iter().map(|u| ("m", u.as_str())).collect();

    let mut group = c.benchmark_group("eligible_8_candidates");
    for (name, policy) in [
        ("any", VersionPolicy::Any),
        ("latest_only", VersionPolicy::LatestOnly),
        ("max_staleness_1", VersionPolicy::MaxStaleness(1)),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| table.eligible(black_box(candidates.iter().copied()), black_box(&policy)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_eligible);
criterion_main!(benches);
