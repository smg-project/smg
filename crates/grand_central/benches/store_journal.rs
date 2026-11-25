//! What the replication journal costs the store, at 8, 64 and 512 members
//! of one pool, each cell beside its twin on a store with no journal, in
//! one binary:
//!
//! - `policy_store_journal/status_write/<plain|journaled|told>/<n>`: one
//!   status flip of a member, which publishes nothing; `told` is the journaled
//!   write on a log with a listener of the watched kinds, as a controller
//!   with a watch open has: the write's telling is in the cell;
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
//! Every member is ready, in a store that leads a term as a serving
//! controller's does. The journaled store's log retains 4,096 events.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{hint::black_box, sync::Arc};

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
use smg_grand_central::{
    engine::{Role, WorkerId},
    service::replication::Log,
    store::{LeasesStore, MemoryStore, Position, WorkerFacts, WorkerStatus, WorkersStore},
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";
/// The log's ring.
const WATCHED: usize = 4_096;

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

/// One pool of `n` members, each ready and published, in a store with no
/// journal or with a log: the store, its log and its members.
struct Pool {
    store: MemoryStore,
    log: Option<Arc<Log>>,
    ids: Vec<WorkerId>,
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
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let id = store.upsert(facts(index, 0)).id();
            store.publish_now();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            id
        })
        .collect();
    filled::filled(&store, MODEL, Role::Single, n);
    Pool { store, log, ids }
}

const KINDS: [(&str, bool); 2] = [("plain", false), ("journaled", true)];

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
    group.finish();
}

/// One status flip of the pool's middle member: no world is published.
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

criterion_group!(benches, status_write, member_write, read, cut);
criterion_main!(benches);
