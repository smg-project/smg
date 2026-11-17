//! What the replication journal costs the store, at 8, 64 and 512 members
//! of one pool, each cell beside its twin on a store with no journal, in
//! one binary:
//!
//! - `policy_store_journal/load_write/<plain|watched|journaled>/<n>`: one
//!   load record's write inside the publication window; `watched` is a
//!   journal of the watched kinds alone, which carries no load row;
//! - `policy_store_journal/status_write/<plain|journaled>/<n>`: one status
//!   flip of a member, published at once;
//! - `policy_store_journal/member_write/<plain|journaled>/<n>`: one refresh
//!   of a member's facts, the pool derived again;
//! - `policy_store_journal/view_build/<plain|journaled>/<n>`: one
//!   decision's build. A decision reads the published world and no line of
//!   the journal: the two read alike, and the count says so where the
//!   clock cannot;
//! - `policy_store_journal/read/<events>`: one reader's copy of a batch out
//!   of the log, the longest the log's lock is held;
//! - `policy_store_journal/cut/<watched|every>/<n>`: one cut of the tables
//!   of the watched kinds or of every kind: the copy a writer of those
//!   tables waits for, and the sort after it.
//!
//! Every member is ready with a record on file, in a store that leads a
//! term as a serving controller's does. The journaled store's log
//! retains 4,096 events of the watched kinds and 65,536 load rows.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

use std::{
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    replication::Log,
    store::{
        Ask, GatewayId, Kinds, LeasesStore, LoadSink, LoadSource, MemoryStore, Position,
        ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
    LoadRow, Role, SignalSource, WorkerId,
};

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
    let log = loads.map(|loads| Arc::new(Log::new(WATCHED, loads)));
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
                let told = log.head(Kinds::All).seq - log.head(Kinds::Watched).seq;
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
    for (kind, loads) in KINDS {
        for n in SIZES {
            let pool = pool_of(n, loads);
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
    }
    group.finish();
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
            let ask = Ask {
                gateway: GatewayId(1),
                now: pool.opened + Duration::from_secs(1),
                pinned: None,
                affinity: false,
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
        seq: log.head(Kinds::All).seq - 2_048,
        term: 1,
    };
    for events in [16usize, 256] {
        let mut out = Vec::with_capacity(events);
        // Checked ahead of the timed id: a filtered run skips its routine.
        log.read(Kinds::All, after, events, &mut out)
            .expect("retained");
        assert_eq!(out.len(), events);
        group.bench_with_input(BenchmarkId::from_parameter(events), &events, |b, _| {
            b.iter(|| {
                log.read(Kinds::All, black_box(after), events, &mut out)
                    .expect("retained");
                black_box(out.len())
            });
        });
    }
    group.finish();
}

fn cut(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/cut");
    for (kind, kinds) in [("watched", Kinds::Watched), ("every", Kinds::All)] {
        for n in SIZES {
            let pool = pool_of(n, Some(LOADS));
            group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
                b.iter(|| black_box(pool.store.cut(black_box(kinds))));
            });
        }
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
    cut
);
criterion_main!(benches);
