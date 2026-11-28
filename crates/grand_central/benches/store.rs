//! The in-memory store's health table per call: a gateway veto written for a
//! registered worker, a veto read, and a removal
//! that clears a worker's entries across two gateways, with 64 workers
//! registered. Ids: `policy_store/health/<op>` and
//! `policy_store/remove_with_health`. The writer takes the workers lock
//! before the health lock, so a write landing after the worker's removal
//! lands nowhere.

#[path = "common/filled.rs"]
mod filled;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use smg_grand_central::{
    engine::WorkerId,
    store::{
        identity, GatewayId, HealthSink, MembershipChange, MemoryStore, Position, Registrant,
        WorkerFacts, WorkerRow, WorkerStatus, WorkersStore,
    },
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

/// A store with `WORKERS` registered workers, and their ids in order. The
/// store has no holder of worker objects: it takes each row as a decided
/// change ([`MemoryStore::follow`]), stated as a registration leaves it.
fn populated() -> (MemoryStore, Vec<WorkerId>) {
    let store = MemoryStore::new();
    let now = Instant::now();
    let ids: Vec<WorkerId> = (0..WORKERS)
        .map(|i| {
            let facts = facts(&format!("http://w{i}:8000"), 1);
            let id = identity(&facts.key, facts.incarnation);
            let row = WorkerRow {
                id,
                facts,
                status: WorkerStatus::Pending,
                revision: 1,
                admitted_at: now,
                maker: Registrant::Control,
            };
            let epoch = WorkersStore::epoch(&store) + 1;
            let registered = MembershipChange::Registered { id };
            store.follow(Position::default(), epoch, registered, Some(row));
            id
        })
        .collect();
    // Each registration landed: the store's epoch counts the writes it took.
    assert_eq!(
        WorkersStore::epoch(&store),
        WORKERS,
        "a registration did not land"
    );
    store.publish_now();
    filled::filled(&store, &ids, "m");
    (store, ids)
}

fn health(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store/health");
    let (store, ids) = populated();
    let sink: &dyn HealthSink = &store;
    let gateway = GatewayId(1);
    let now = Instant::now();
    let mut turn = 0usize;
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
                    }
                }
                let id = ids[ids.len() / 2];
                let epoch = WorkersStore::epoch(&store) + 1;
                (store, id, epoch)
            },
            |(store, id, epoch)| {
                let left = MembershipChange::Removed { id: black_box(id) };
                store.follow(Position::default(), epoch, left, None);
                store.publish_now();
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
