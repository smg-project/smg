//! What one selection pays for the weighted traffic split (`--traffic-split`):
//! `PolicyRegistry::select_worker_for_model` on round robin over a fleet,
//! with the split off, on with every worker in one group (the usual case),
//! and on with two groups at 90/10, keyless and keyed.
//!
//! Run with: cargo bench --bench traffic_split
#![expect(
    clippy::unwrap_used,
    reason = "benchmark code: panicking on setup failure is expected"
)]

use std::{collections::HashMap, hint::black_box, sync::Arc};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use http::HeaderMap;
use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};
use smg::{
    config::PolicyConfig,
    policies::{
        traffic_split::{GROUP_LABEL, WEIGHT_LABEL},
        PolicyRegistry, SelectWorkerInfo,
    },
    worker::{BasicWorkerBuilder, Worker, WorkerType},
};

fn worker(index: usize, group: Option<(&str, &str)>) -> Arc<dyn Worker> {
    let mut labels = HashMap::new();
    if let Some((name, weight)) = group {
        labels.insert(GROUP_LABEL.to_string(), name.to_string());
        labels.insert(WEIGHT_LABEL.to_string(), weight.to_string());
    }
    let worker = BasicWorkerBuilder::new(format!("http://worker-{index:03}:8000"))
        .worker_type(WorkerType::Regular)
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .labels(labels)
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// `size` workers: unlabelled, all in one named group, or nine in ten in
/// `stable` at 90 and the rest in `canary` at 10.
fn fleet(size: usize, shape: &str) -> Vec<Arc<dyn Worker>> {
    (0..size)
        .map(|i| match shape {
            "unlabelled" => worker(i, None),
            "one-group" => worker(i, Some(("stable", "90"))),
            _ => worker(
                i,
                Some(if i % 10 == 9 {
                    ("canary", "10")
                } else {
                    ("stable", "90")
                }),
            ),
        })
        .collect()
}

fn bench_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("traffic_split_select");
    let keyed = {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "conversation-42".parse().unwrap());
        headers
    };
    for size in [8usize, 32, 128] {
        for (label, split, shape, headers) in [
            ("off", false, "unlabelled", None),
            ("on-one-group", true, "one-group", None),
            ("on-two-groups", true, "two-groups", None),
            ("on-two-groups-keyed", true, "two-groups", Some(&keyed)),
        ] {
            let registry = PolicyRegistry::new(PolicyConfig::RoundRobin).with_traffic_split(split);
            let policy = registry.get_default_policy();
            let workers = fleet(size, shape);
            let info = SelectWorkerInfo {
                headers,
                ..Default::default()
            };
            group.bench_function(BenchmarkId::new(label, size), |b| {
                b.iter(|| {
                    black_box(registry.select_worker_for_model(
                        &policy,
                        "model",
                        black_box(&workers),
                        &info,
                    ))
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_selection);
criterion_main!(benches);
