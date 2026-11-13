//! The in-memory store's health and load tables per call, with 64 workers
//! registered: a completed dispatch and a gateway veto written for a
//! registered worker, a veto read; a rank's pushed record, a poll's report,
//! the poll arbitration and a row read; and a removal that clears a worker's
//! entries across two gateways, without and with its load rows. Ids:
//! `policy_store/health/<op>`, `policy_store/loads/<op>`,
//! `policy_store/remove_with_health` and
//! `policy_store/remove_with_health_and_loads`. The writers take the workers
//! lock before their table's lock, so a write landing after the worker's
//! removal lands nowhere; this bench prices that lock on the completion path
//! and on every load sample.
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
    store::{GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkersStore},
    LoadRow, TransportClass, WorkerId,
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

/// One rank's sample, as a stream record or a poll's row carries it.
fn sample() -> LoadRow {
    LoadRow {
        running: 3,
        queued: 1,
        kv_usage_permille: 420,
        ..LoadRow::default()
    }
}

fn loads(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store/loads");
    let now = Instant::now();
    let row = sample();
    let mut turn = 0usize;
    // A rank's record from its stream: the write of every pushed sample.
    // Every rank has its record before the first row runs, so the two reads
    // below see the same store whichever rows a filter keeps.
    let (store, ids) = populated();
    let sink: &dyn LoadSink = &store;
    for id in &ids {
        sink.record(*id, 0, row, now, LoadSource::Stream);
    }
    group.bench_function("record", |b| {
        b.iter(|| {
            let id = ids[turn % ids.len()];
            turn += 1;
            sink.record(black_box(id), 0, black_box(row), now, LoadSource::Stream);
        });
    });
    // The arbitration the driver asks for a rank at every tick, and the
    // read of its row, on a store whose ranks all have a fresh record.
    group.bench_function("poll_mode", |b| {
        b.iter(|| {
            let id = ids[turn % ids.len()];
            turn += 1;
            black_box(sink.poll_mode(black_box(id), now, Duration::from_secs(10)))
        });
    });
    group.bench_function("load", |b| {
        b.iter(|| {
            let id = ids[turn % ids.len()];
            turn += 1;
            black_box(sink.load(black_box(id), 0))
        });
    });
    // A poll's report of a worker's one rank, on a store no stream writes
    // to: the row is written, not dropped behind a fresher record.
    let (polled, ids) = populated();
    let sink: &dyn LoadSink = &polled;
    group.bench_function("report", |b| {
        b.iter(|| {
            let id = ids[turn % ids.len()];
            turn += 1;
            sink.report(black_box(id), &[(0, black_box(row))], now, LoadSource::Poll);
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
    // The same removal with the worker's load rows on file: the one
    // deletion clears them with the health entries.
    group.bench_function("remove_with_health_and_loads", |b| {
        b.iter_batched(
            || {
                let (store, ids) = populated();
                let health: &dyn HealthSink = &store;
                let loads: &dyn LoadSink = &store;
                let now = Instant::now();
                for id in &ids {
                    for gateway in [GatewayId(1), GatewayId(2)] {
                        health.gateway_veto(gateway, *id, Some(now + Duration::from_secs(60)));
                        health.completed(gateway, *id, TransportClass::Ok, now);
                    }
                    loads.record(*id, 0, sample(), now, LoadSource::Stream);
                }
                let id = ids[ids.len() / 2];
                let revision = WorkersStore::get(&store, id).expect("the row").revision;
                (store, id, revision)
            },
            |(store, id, revision)| {
                black_box(store.remove(black_box(id), revision).expect("removed"));
                store
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, health, loads, remove);
criterion_main!(benches);
