//! What the replication journal costs the store, at 8, 64 and 512 members
//! of one pool, each cell beside its twin on a store with no journal, in
//! one binary:
//!
//! - `policy_store_journal/load_write/<plain|journaled>/<n>`: one load
//!   record's write inside the publication window, which a journal is told
//!   nothing of;
//! - `policy_store_journal/status_write/<plain|journaled|told>/<n>`: one
//!   status flip of a member, published at once; `told` is the journaled
//!   write on a log with a listener of the watched kinds, as a controller
//!   with a watch open has: the write's telling is in the cell. Under the
//!   `worker-host` feature `followed` is that write with the worker
//!   objects following the store: the listener wakes their follower, which
//!   reads the store's tail on a thread of its own and waits again. The
//!   wake is in the cell. For a run pinned to two cores; no gate reads it;
//! - `policy_store_journal/member_write/<plain|journaled>/<n>`: one refresh
//!   of a member's facts, the pool derived again;
//! - `policy_store_journal/read/<events>`: one reader's copy of a batch of
//!   status events out of the log, each with its row, the longest the log's
//!   lock is held;
//! - `policy_store_journal/read/watched_at_head`: a watch's read at the
//!   log's head: nothing to copy, the lock and the look at the log alone.
//!   The log of the three read cells holds a full ring;
//! - `policy_store_journal/cut/watched/<n>`: one cut of the rows: the copy
//!   a writer of the rows waits for, and the sort after it.
//!
//! Every member is ready with a record on file, in a store that leads a
//! term as a serving controller's does. The journaled store's log
//! retains 4,096 events.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
#[cfg(feature = "worker-host")]
use smg_policy_controller::worker_host::{Members, Objects};
use smg_policy_controller::{
    engine::{LoadRow, Role, SignalSource, WorkerId},
    service::replication::Log,
    store::{
        LeasesStore, LoadSink, LoadSource, MemoryStore, Position, WorkerFacts, WorkerStatus,
        WorkersStore, PUBLISH_WINDOW,
    },
};
#[cfg(feature = "worker-host")]
use smg_worker::registry::WorkerRegistry;
#[cfg(feature = "worker-host")]
use tokio::{runtime::Builder, sync::Notify, task::JoinSet};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";
/// The log's ring.
const WATCHED: usize = 4_096;

/// Member `index`'s record as of write `turn`.
fn record_of(index: usize, turn: u64) -> LoadRow {
    let x = (index as u64 + 1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(turn.wrapping_mul(0xbf58_476d_1ce4_e5b9));
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

fn facts(index: usize, priority: i32) -> WorkerFacts {
    let key = format!("http://w{index}:8000");
    WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        max_running: Some(4),
        priority,
        ..WorkerFacts::default()
    }
}

/// One pool of `n` members, each ready with its record on file and
/// published, in a store with no journal or with a log: the store, its
/// log, its members and the instant its last window opened.
struct Pool {
    store: MemoryStore,
    log: Option<Arc<Log>>,
    ids: Vec<WorkerId>,
    opened: Instant,
}

fn pool_of(n: usize, journaled: bool) -> Pool {
    pool_on(n, journaled.then(|| Log::new(WATCHED)))
}

/// How a cell builds its pool of so many members.
type PoolOf = fn(usize) -> Pool;

/// [`pool_of`] on `log`.
fn pool_on(n: usize, log: Option<Log>) -> Pool {
    let log = log.map(Arc::new);
    let store = match &log {
        Some(log) => MemoryStore::journaled(Arc::clone(log) as _),
        None => MemoryStore::new(),
    };
    // The store leads: its writes are journaled under its term.
    store.set_term(1);
    let opened = Instant::now();
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let id = store.upsert(facts(index, 0)).id();
            store.publish_now();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            store.record(id, 0, record_of(index, 0), opened, LoadSource::Stream);
            id
        })
        .collect();
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    filled::filled(&store, MODEL, Role::Single, n);
    Pool {
        store,
        log,
        ids,
        opened,
    }
}

const KINDS: [(&str, bool); 2] = [("plain", false), ("journaled", true)];

fn load_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/load_write");
    for (kind, journaled) in KINDS {
        for n in SIZES {
            let pool = pool_of(n, journaled);
            let before = pool.log.as_ref().map(|log| log.head());
            // Inside the window: the record is folded and waits.
            let at = pool.opened + Duration::from_millis(1);
            let mut turn = 0u64;
            group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
                b.iter(|| {
                    turn += 1;
                    let index = (turn as usize).wrapping_mul(67) % n;
                    pool.store.record(
                        black_box(pool.ids[index]),
                        0,
                        record_of(index, turn),
                        at,
                        LoadSource::Stream,
                    );
                });
            });
            // The cell is held to what it claims: the log was told no
            // load write.
            assert_eq!(pool.log.as_ref().map(|log| log.head()), before);
        }
    }
    group.finish();
}

fn status_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/status_write");
    let pools: [(&str, PoolOf); 3] = [
        ("plain", |n| pool_of(n, false)),
        ("journaled", |n| pool_of(n, true)),
        // The listener does nothing: the cell holds what the log's telling
        // adds to the write, and none of a watch's work.
        ("told", |n| {
            pool_on(n, Some(Log::new(WATCHED).on_watched(|| {})))
        }),
    ];
    for (kind, build) in pools {
        for n in SIZES {
            flips(&mut group, kind, n, &build(n));
        }
    }
    #[cfg(feature = "worker-host")]
    followed(&mut group);
    group.finish();
}

/// One status flip of the pool's middle member, published at once.
fn flips(group: &mut BenchmarkGroup<'_, WallTime>, kind: &str, n: usize, pool: &Pool) {
    let id = pool.ids[n / 2];
    let mut ready = true;
    group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
        b.iter(|| {
            ready = !ready;
            let status = if ready {
                WorkerStatus::Ready
            } else {
                WorkerStatus::NotReady
            };
            let revision = WorkersStore::get(&pool.store, id)
                .expect("the row")
                .revision;
            black_box(pool.store.set_status(id, status, revision));
        });
    });
}

/// The `told` write with the worker objects following the store, as a
/// controller that hosts them has: the log's listener wakes their follower
/// on a runtime thread of its own, and each wake reads the store's tail to
/// its end. A status change builds nothing there, so the cell holds the
/// wake and what the follower's read takes of the store's lock.
#[cfg(feature = "worker-host")]
fn followed(group: &mut BenchmarkGroup<'_, WallTime>) {
    let runtime = Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .expect("a runtime");
    for n in SIZES {
        let moved = Arc::new(Notify::new());
        let told = Arc::clone(&moved);
        let log = Log::new(WATCHED).on_watched(move || told.notify_one());
        let pool = Arc::new(pool_on(n, Some(log)));
        let objects = Objects::new(Arc::new(WorkerRegistry::<Members>::new()));
        // Ended with the set, at the end of this size.
        let mut follower = JoinSet::new();
        follower.spawn_on(
            {
                let pool = Arc::clone(&pool);
                async move {
                    loop {
                        objects.follow(&pool.store);
                        moved.notified().await;
                    }
                }
            },
            runtime.handle(),
        );
        flips(group, "followed", n, &pool);
    }
}

fn member_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/member_write");
    group.sample_size(30);
    for (kind, journaled) in KINDS {
        for n in SIZES {
            let pool = pool_of(n, journaled);
            let mut turn = 0i32;
            group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
                b.iter(|| {
                    turn = turn.wrapping_add(1);
                    black_box(pool.store.upsert(facts(n / 2, turn)));
                    pool.store.publish_now();
                });
            });
        }
    }
    group.finish();
}

fn read(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/read");
    let pool = pool_of(64, true);
    let log = pool.log.as_ref().expect("the journaled pool's log");
    // The ring full and written half over again, as a controller
    // that ran a while holds it: the ring's buffer is wrapped at its
    // middle, and a reader's search of it is the search of a full ring.
    for turn in 0..WATCHED + WATCHED / 2 {
        let status = if turn % 2 == 0 {
            WorkerStatus::NotReady
        } else {
            WorkerStatus::Ready
        };
        let revision = WorkersStore::get(&pool.store, pool.ids[0])
            .expect("the row")
            .revision;
        pool.store
            .set_status(pool.ids[0], status, revision)
            .expect("a flip");
    }
    // The store took term 1 before its first write: every position stands
    // under it.
    let after = Position {
        seq: log.head().seq - 2_048,
        term: 1,
    };
    for events in [16usize, 256] {
        let mut out = Vec::with_capacity(events);
        // Checked ahead of the timed id: a filtered run skips its routine.
        log.read(after, events, &mut out).expect("retained");
        assert_eq!(out.len(), events);
        group.bench_with_input(BenchmarkId::from_parameter(events), &events, |b, _| {
            b.iter(|| {
                log.read(black_box(after), events, &mut out)
                    .expect("retained");
                black_box(out.len())
            });
        });
    }
    let head = log.head();
    let mut out = Vec::with_capacity(16);
    log.read(head, 16, &mut out).expect("retained");
    assert!(out.is_empty(), "the reader is at the head");
    group.bench_function("watched_at_head", |b| {
        b.iter(|| {
            log.read(black_box(head), 16, &mut out).expect("retained");
            black_box(out.len())
        });
    });
    group.finish();
}

fn cut(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/cut");
    for n in SIZES {
        let pool = pool_of(n, true);
        group.bench_with_input(BenchmarkId::new("watched", n), &n, |b, _| {
            b.iter(|| black_box(pool.store.checkpoint()));
        });
    }
    group.finish();
}

criterion_group!(benches, load_write, status_write, member_write, read, cut);
criterion_main!(benches);
