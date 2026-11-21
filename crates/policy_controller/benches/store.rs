//! The in-memory store's health tables per call: a completed dispatch and a
//! gateway veto written for a registered worker, a veto read, and a removal
//! that clears a worker's entries across two gateways, with 64 workers
//! registered. Ids: `policy_store/health/<op>` and
//! `policy_store/remove_with_health`. The writers take the workers lock
//! before the health lock, so a write landing after the worker's removal
//! lands nowhere; this bench prices that lock on the completion path.
//!
//! `policy_store/load_answer` is one poll's answer for one of the workers
//! inside the publication window, with no build owed: the answer door as
//! every answer but a new rank's first takes it.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a store that refuses its own registration is a bug the unit tests catch"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use smg_policy_controller::{
    engine::{LoadRow, Role, WorkerId},
    store::{GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkersStore},
    tracking::TransportClass,
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
    let ids: Vec<WorkerId> = (0..WORKERS)
        .map(|i| store.upsert(facts(&format!("http://w{i}:8000"), 1)).id())
        .collect();
    store.publish_now();
    filled::filled(&store, "m", Role::Single, ids.len());
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
                store.publish_now();
                // Returned, so criterion drops the store outside the timed section.
                store
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn answer(c: &mut Criterion) {
    let (store, ids) = populated();
    let sink: &dyn LoadSink = &store;
    let row = |turn: usize| LoadRow {
        running: (turn % 12) as u32,
        ..LoadRow::default()
    };
    // The first answer is shown at once and opens the window: the timed
    // ones are written inside it and wait, so no publication is timed.
    let at = Instant::now();
    sink.report(&[(ids[0], 0, Some(row(0)))], at, LoadSource::Poll);
    let mut turn = 0usize;
    c.bench_function("policy_store/load_answer", |b| {
        b.iter(|| {
            turn += 1;
            let answer = [(ids[turn % ids.len()], 0, Some(row(turn)))];
            sink.report(black_box(&answer), at, LoadSource::Poll);
        });
    });
    // A run the filter left out wrote the first answer alone, and it was
    // shown: nothing waits.
    assert!(turn == 0 || store.publish_due().is_some());
}

criterion_group!(benches, health, remove, answer);
criterion_main!(benches);
