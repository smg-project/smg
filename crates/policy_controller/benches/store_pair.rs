//! The cost of a pair decision's two views built from the store, under one
//! and under eight deciding threads on one world.
//!
//! `policy_store_pair/<door>/<threads>/<n>` builds both sides of a pair
//! decision over one model with `<n>` prefill and `<n>` decode members, on
//! that many deciding threads, each on buffers of its own, and reads the
//! slowest thread's time for one pair: `owned` takes the world's pointer
//! and builds both sides from it, `guard` builds through the store's pair
//! door, which borrows.
//!
//! A bench target of its own: the store's gated target (`store_view`) is
//! built on the base branch with the pull request's bench sources, so its
//! rows call nothing the base's store lacks.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

use std::{
    hint::black_box,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, PairBuffers, WorkerFacts, WorkerStatus,
        WorkersStore, PUBLISH_WINDOW,
    },
    LoadRow, Role, SignalSource,
};

const MODEL: &str = "m";

/// Member `index`'s record: figures that differ per member, so no two rows
/// read alike.
fn record_of(index: usize) -> LoadRow {
    let x = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    LoadRow {
        running: (x >> 16) as u32 % 12,
        queued: (x >> 24) as u32 % 4,
        kv_usage_permille: ((x >> 32) % 900) as u16,
        live_rate: 1_000 + ((x >> 40) % 3_000) as u32,
        report_age_ms: 5,
        source: SignalSource::Events,
        ..LoadRow::default()
    }
}

/// A store with one model served by `n` prefill members and `n` decode
/// members, every one ready with its record on file and published; the
/// instant its last window opened.
fn pair_pool(n: usize) -> (Arc<MemoryStore>, Instant) {
    let store = Arc::new(MemoryStore::new());
    let opened = Instant::now();
    for (side, role) in [("p", Role::Prefill), ("d", Role::Decode)] {
        for index in 0..n {
            let key = format!("http://{side}{index}:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: key.clone(),
                    engine: key,
                    incarnation: 1,
                    role,
                    models: vec![MODEL.to_string()],
                    dp_size: 1,
                    max_running: Some(4),
                    ..WorkerFacts::default()
                })
                .id();
            store.publish_now();
            let revision = WorkersStore::get(&*store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            store.record(id, 0, record_of(index), opened, LoadSource::Stream);
        }
    }
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    assert_eq!(store.publish_due(), None);
    (store, opened)
}

/// `iters` pair builds through one door on buffers of this thread's own,
/// after one untimed build held to both pools' sizes; the wall time they
/// took.
fn pair_builds(store: &MemoryStore, ask: &Ask, owned: bool, n: usize, iters: u64) -> Duration {
    let mut out = PairBuffers::default();
    let one = |out: &mut PairBuffers| {
        if owned {
            store.world().build_pair(MODEL, black_box(ask), out)
        } else {
            store.build_pair(MODEL, black_box(ask), out)
        }
    };
    let (first, second) = one(&mut out).expect("both pools");
    assert_eq!(
        (first.pool_size as usize, second.pool_size as usize),
        (n, n)
    );
    let start = Instant::now();
    for _ in 0..iters {
        black_box(one(&mut out));
        black_box(out.pools());
    }
    start.elapsed()
}

/// The two doors of a pair decision under one and eight deciding threads.
/// A row's time is one pair build on one thread: the slowest thread's wall
/// time over its builds.
fn pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_pair");
    for n in [8_usize, 64] {
        let (store, opened) = pair_pool(n);
        // The decision every pair is timed for: a second after the window
        // that opened last.
        let ask = Ask {
            gateway: GatewayId(1),
            now: opened + Duration::from_secs(1),
            pinned: None,
            affinity: false,
        };
        for threads in [1_usize, 8] {
            for door in ["owned", "guard"] {
                group.bench_with_input(
                    BenchmarkId::new(format!("{door}/{threads}"), n),
                    &n,
                    |b, _| {
                        b.iter_custom(|iters| {
                            thread::scope(|s| {
                                let deciders: Vec<_> = (0..threads)
                                    .map(|_| {
                                        let (store, ask) = (&store, &ask);
                                        s.spawn(move || {
                                            pair_builds(store, ask, door == "owned", n, iters)
                                        })
                                    })
                                    .collect();
                                deciders
                                    .into_iter()
                                    .map(|h| h.join().expect("a deciding thread"))
                                    .max()
                                    .unwrap_or_default()
                            })
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, pair);
criterion_main!(benches);
