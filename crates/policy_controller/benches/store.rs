//! The in-memory store's health tables per call: a completed dispatch and a
//! gateway veto written for a registered worker, a veto read, and a removal
//! that clears a worker's entries across two gateways, with 64 workers
//! registered. Ids: `policy_store/health/<op>` and
//! `policy_store/remove_with_health`. The writers take the workers lock
//! before the health lock, so a write landing after the worker's removal
//! lands nowhere; this bench prices that lock on the completion path.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a store that refuses its own registration is a bug the unit tests catch"
)]

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use smg_policy_controller::{
    store::{GatewayId, HealthSink, MemoryStore, WorkerFacts, WorkersStore},
    TransportClass, WorkerId,
};

const WORKERS: u64 = 64;

fn facts(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        key: key.to_string(),
        engine: key.to_string(),
        incarnation,
        models: vec!["m".to_string()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    }
}

/// A store with `WORKERS` registered workers, and their ids in order.
fn populated() -> (MemoryStore, Vec<WorkerId>) {
    let store = MemoryStore::new();
    let ids = (0..WORKERS)
        .map(|i| store.upsert(facts(&format!("http://w{i}:8000"), 1)).id())
        .collect();
    (store, ids)
}

fn health(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store/health");
    let (store, ids) = populated();
    let sink: &dyn HealthSink = &store;
    let gateway = GatewayId(1);
    let now = Instant::now();
    let mut turn = 0usize;
    group.bench_function("completed", |b| {
        b.iter(|| {
            let id = ids[turn % ids.len()];
            turn += 1;
            sink.completed(black_box(gateway), black_box(id), TransportClass::Ok, now);
        });
    });
    group.bench_function("gateway_veto", |b| {
        b.iter(|| {
            let id = ids[turn % ids.len()];
            turn += 1;
            sink.gateway_veto(
                black_box(gateway),
                black_box(id),
                Some(now + Duration::from_secs(60)),
            );
        });
    });
    group.bench_function("vetoed", |b| {
        b.iter(|| {
            let id = ids[turn % ids.len()];
            turn += 1;
            black_box(sink.vetoed(black_box(gateway), black_box(id), now))
        });
    });
    group.finish();
}

fn remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store");
    group.bench_function("remove_with_health", |b| {
        b.iter_batched(
            || {
                let (store, ids) = populated();
                let sink: &dyn HealthSink = &store;
                let now = Instant::now();
                for id in &ids {
                    for gateway in [GatewayId(1), GatewayId(2)] {
                        sink.gateway_veto(gateway, *id, Some(now + Duration::from_secs(60)));
                        sink.completed(gateway, *id, TransportClass::Ok, now);
                    }
                }
                let id = ids[ids.len() / 2];
                let revision = WorkersStore::get(&store, id).expect("the row").revision;
                (store, id, revision)
            },
            |(store, id, revision)| {
                black_box(store.remove(black_box(id), revision).expect("removed"));
                // Returned, so criterion drops the store outside the timed section.
                store
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, health, remove);
criterion_main!(benches);
