//! What the replication journal costs the store, at 8, 64 and 512 members
//! of one pool, each cell beside its twin on a store with no journal, in
//! one binary:
//!
//! - `policy_store_journal/load_write/<plain|watched|journaled>/<n>`: one
//!   load record's write inside the publication window; `watched` is a
//!   journal of the watched kinds alone, which carries no load row;
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
//! - `policy_store_journal/view_build/<plain|journaled>/<n>`: one
//!   decision's build. A decision reads the published world and no line of
//!   the journal: the two read alike, and the count says so where the
//!   clock cannot;
//! - `policy_store_journal/read/<events>`: one reader's copy of a batch out
//!   of the log, the longest the log's lock is held;
//! - `policy_store_journal/read/watched_at_head`: a watch's read of the
//!   watched kinds at their head, beside the load rows: nothing to copy,
//!   the lock and the look at the log alone. The log of the three read
//!   cells holds a full ring of the watched kinds;
//! - `policy_store_journal/cut/<watched|every>/<n>`: one cut of the tables
//!   of the watched kinds or of every kind: the copy a writer of those
//!   tables waits for, and the sort after it;
//! - `policy_store_journal/load_write_drained/<n>`: the journaled load
//!   write while a reader on another thread drains the log, a batch a read
//!   and never resting: the most a standby's stream asks of the log's
//!   lock. For a run pinned to two cores; no gate reads it.
//!
//! Every member is ready with a record on file, in a store that leads a
//! term as a serving controller's does. The journaled store's log
//! retains 4,096 events of the watched kinds and 65,536 load rows.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{
    hint::black_box,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
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
        GatewayId, JournalKinds, LeasesStore, LoadSink, LoadSource, MemoryStore, Position,
        ViewBuffers, ViewQuery, WorkerFacts, WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
};
#[cfg(feature = "worker-host")]
use smg_worker::registry::WorkerRegistry;
#[cfg(feature = "worker-host")]
use tokio::{runtime::Builder, sync::Notify, task::JoinSet};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";
/// The watched kinds' ring.
const WATCHED: usize = 4_096;
/// The load rows' ring.
const LOADS: usize = 65_536;

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
/// published, in a store with no journal (`None`) or with a log that
/// retains so many load rows (`0`: the watched kinds alone): the store,
/// its log, its members and the instant its last window opened.
struct Pool {
    store: MemoryStore,
    log: Option<Arc<Log>>,
    ids: Vec<WorkerId>,
    opened: Instant,
}

fn pool_of(n: usize, loads: Option<usize>) -> Pool {
    pool_on(n, loads.map(|loads| Log::new(WATCHED, loads)))
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

const KINDS: [(&str, Option<usize>); 2] = [("plain", None), ("journaled", Some(LOADS))];

fn load_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/load_write");
    for (kind, loads) in [KINDS[0], ("watched", Some(0)), KINDS[1]] {
        for n in SIZES {
            let pool = pool_of(n, loads);
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
            if let Some(log) = &pool.log {
                // The cell is held to what it claims: a log that carries
                // load rows was told every write, and the other none.
                let told = log.head(JournalKinds::All).seq - log.head(JournalKinds::Watched).seq;
                match loads {
                    Some(0) => assert_eq!(told, 0),
                    _ => assert!(turn == 0 || told >= turn),
                }
            }
        }
    }
    group.finish();
}

fn status_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/status_write");
    let pools: [(&str, PoolOf); 3] = [
        ("plain", |n| pool_of(n, None)),
        ("journaled", |n| pool_of(n, Some(LOADS))),
        // The listener does nothing: the cell holds what the log's telling
        // adds to the write, and none of a watch's work.
        ("told", |n| {
            pool_on(n, Some(Log::new(WATCHED, LOADS).on_watched(|| {})))
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
        let log = Log::new(WATCHED, LOADS).on_watched(move || told.notify_one());
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
    for (kind, loads) in KINDS {
        for n in SIZES {
            let pool = pool_of(n, loads);
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

fn view_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/view_build");
    for (kind, loads) in KINDS {
        for n in SIZES {
            let pool = pool_of(n, loads);
            let ask = ViewQuery {
                gateway: GatewayId(1),
                now: pool.opened + Duration::from_secs(1),
                pinned: None,
                affinity: false,
                signal_cells: false,
            };
            let mut out = ViewBuffers::default();
            group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
                b.iter(|| {
                    let built =
                        pool.store
                            .build_view(MODEL, Role::Single, black_box(&ask), &mut out);
                    black_box(&out.candidates);
                    built.map_or(0, |built| built.pool_size)
                });
            });
        }
    }
    group.finish();
}

fn read(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/read");
    let pool = pool_of(64, Some(LOADS));
    let log = pool.log.as_ref().expect("the journaled pool's log");
    // The watched ring full and written half over again, as a controller
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
    let at = pool.opened + Duration::from_millis(1);
    for turn in 0..4_096u64 {
        let index = (turn as usize).wrapping_mul(67) % 64;
        pool.store.record(
            pool.ids[index],
            0,
            record_of(index, turn),
            at,
            LoadSource::Stream,
        );
    }
    // The store took term 1 before its first write: every position stands
    // under it.
    let after = Position {
        seq: log.head(JournalKinds::All).seq - 2_048,
        term: 1,
    };
    for events in [16usize, 256] {
        let mut out = Vec::with_capacity(events);
        // Checked ahead of the timed id: a filtered run skips its routine.
        log.read(JournalKinds::All, after, events, &mut out)
            .expect("retained");
        assert_eq!(out.len(), events);
        group.bench_with_input(BenchmarkId::from_parameter(events), &events, |b, _| {
            b.iter(|| {
                log.read(JournalKinds::All, black_box(after), events, &mut out)
                    .expect("retained");
                black_box(out.len())
            });
        });
    }
    let head = log.head(JournalKinds::Watched);
    let mut out = Vec::with_capacity(16);
    log.read(JournalKinds::Watched, head, 16, &mut out)
        .expect("retained");
    assert!(out.is_empty(), "the reader is at the head");
    group.bench_function("watched_at_head", |b| {
        b.iter(|| {
            log.read(JournalKinds::Watched, black_box(head), 16, &mut out)
                .expect("retained");
            black_box(out.len())
        });
    });
    group.finish();
}

fn cut(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/cut");
    for (kind, kinds) in [
        ("watched", JournalKinds::Watched),
        ("every", JournalKinds::All),
    ] {
        for n in SIZES {
            let pool = pool_of(n, Some(LOADS));
            group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
                b.iter(|| black_box(pool.store.checkpoint(black_box(kinds))));
            });
        }
    }
    group.finish();
}

fn load_write_drained(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/load_write_drained");
    for n in SIZES {
        let pool = pool_of(n, Some(LOADS));
        let log = Arc::clone(pool.log.as_ref().expect("the journaled pool's log"));
        let stop = Arc::new(AtomicBool::new(false));
        // The reader's start, taken before the first write: whatever the
        // run writes stands after it.
        let mut place = log.head(JournalKinds::All);
        let reader = {
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut out = Vec::with_capacity(Log::BATCH);
                let mut read = 0u64;
                loop {
                    // Asked before the read: one read follows the last
                    // write, so a run of one write is read too.
                    let asked = stop.load(Ordering::Relaxed);
                    if log
                        .read(JournalKinds::All, place, Log::BATCH, &mut out)
                        .is_err()
                    {
                        // The ring passed the reader: it starts again at
                        // the head, as a standby does from a snapshot.
                        place = log.head(JournalKinds::All);
                    } else if let Some(last) = out.last() {
                        place = last.at;
                        read += out.len() as u64;
                    } else {
                        std::hint::spin_loop();
                    }
                    if asked {
                        break read;
                    }
                }
            })
        };
        let at = pool.opened + Duration::from_millis(1);
        let mut turn = 0u64;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
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
        stop.store(true, Ordering::Relaxed);
        let read = reader.join().expect("the reader ends when asked");
        // Held to what it claims: the reader read of what the writer wrote.
        assert!(turn == 0 || read > 0, "nothing read of {turn} writes");
    }
    group.finish();
}

criterion_group!(
    benches,
    load_write,
    status_write,
    member_write,
    view_build,
    read,
    cut,
    load_write_drained
);
criterion_main!(benches);
