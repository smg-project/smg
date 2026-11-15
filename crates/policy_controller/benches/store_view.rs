//! The cost of a decision's view built from the store, at 8, 64 and 512
//! members of one pool, and of the writes that feed it:
//!
//! - `policy_store/view_build/<n>`: one decision's build: the pool's
//!   published rows copied, the position and the report's age written;
//! - `policy_store/view_build_column/<n>`: the same under a profile that
//!   declares the affinity column;
//! - `policy_store/load_write/<n>`: one load record's write inside the
//!   publication window: on file and folded into the writer's rows;
//! - `policy_store/publish/<n>`: one window's publication of the pool: one
//!   copy, one pointer swap.
//!
//! `policy_store_cadence/*` reads the build beside a writer on another
//! core, for a run pinned to two cores (both threads spin, so each keeps
//! its core): `quiet/<n>` with the writer idle, `busy/<n>` with it writing
//! ten records per member per second through the store's own window, and
//! `after_publish/<n>`, the first build after a publication made on the
//! writer's core. Outside the `policy_store/` ids, so a filter on those
//! starts no thread. These cells run on a store of their own: the cells
//! before them move the shared store's window ahead of the host's clock,
//! which the busy writer stamps its records with. The busy cell is held to
//! what it claims: a build reads a record the writer wrote while busy.
//!
//! Every member is ready with a record on file. A timed view is written
//! into buffers on a page's start (`placed`), one placement per size for
//! the whole binary: the helper walks its earlier rejects first, and a size
//! placed a fifth time stops at its bound.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/placed.rs"]
mod placed;

use std::{
    cell::Cell,
    hint::black_box,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use placed::page_start_buffer;
use smg_policy_controller::{
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, ViewBuffers, WorkerFacts, WorkerStatus,
        WorkersStore, PUBLISH_WINDOW,
    },
    LoadRow, Role, SignalSource, WorkerId,
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";

/// Member `index`'s record as of write `turn`: figures that differ per
/// member and per write, so no two rows read alike.
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

/// One pool of `n` members in a store, and the buffers its views are timed
/// in.
struct Fixture {
    n: usize,
    store: Arc<MemoryStore>,
    ids: Vec<WorkerId>,
    /// The instant the store's last window opened; a cell that publishes
    /// moves it.
    window: Cell<Instant>,
    out: ViewBuffers,
}

/// A store with one pool of `n` members, every one registered, ready, with
/// its record on file and published: the store, its members by index and
/// the instant its last window opened.
fn pool_of(n: usize) -> (Arc<MemoryStore>, Vec<WorkerId>, Instant) {
    let store = Arc::new(MemoryStore::new());
    let opened = Instant::now();
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let key = format!("http://w{index}:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: key.clone(),
                    engine: key,
                    incarnation: 1,
                    models: vec![MODEL.to_string()],
                    dp_size: 1,
                    max_running: Some(4),
                    ..WorkerFacts::default()
                })
                .id();
            let revision = WorkersStore::get(&*store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            store.record(id, 0, record_of(index, 0), opened, LoadSource::Stream);
            id
        })
        .collect();
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    assert_eq!(store.publish_due(), None);
    (store, ids, opened)
}

/// The decision every build is timed for: a second after the window that
/// opened at `opened`.
fn ask_at(opened: Instant, affinity: bool) -> Ask {
    Ask {
        gateway: GatewayId(1),
        now: opened + Duration::from_secs(1),
        pinned: None,
        affinity,
    }
}

impl Fixture {
    fn new(n: usize) -> Self {
        let (store, ids, opened) = pool_of(n);
        Self {
            n,
            store,
            ids,
            window: Cell::new(opened),
            out: ViewBuffers {
                candidates: page_start_buffer(n),
                affinity: page_start_buffer(n),
            },
        }
    }

    fn ask(&self, affinity: bool) -> Ask {
        ask_at(self.window.get(), affinity)
    }
}

/// One build into the fixture's buffers; the rows it wrote.
fn build(store: &MemoryStore, ask: &Ask, out: &mut ViewBuffers) -> usize {
    let built = store.build_view(MODEL, Role::Single, black_box(ask), out);
    black_box(&out.candidates);
    black_box(&out.affinity);
    built.map_or(0, |built| built.pool_size as usize)
}

fn view_build(c: &mut Criterion, fixtures: &mut [Fixture]) {
    for (name, affinity) in [
        ("policy_store/view_build", false),
        ("policy_store/view_build_column", true),
    ] {
        let mut group = c.benchmark_group(name);
        for fixture in fixtures.iter_mut() {
            let n = fixture.n;
            let ask = fixture.ask(affinity);
            // Every form is held to what it must build before it is timed.
            assert_eq!(build(&fixture.store, &ask, &mut fixture.out), n);
            assert_eq!(fixture.out.affinity.len(), if affinity { n } else { 0 });
            assert!(fixture
                .out
                .candidates
                .iter()
                .all(|row| row.health.healthy && row.load.is_some()));
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| build(&fixture.store, &ask, &mut fixture.out));
            });
        }
        group.finish();
    }
}

fn load_write(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let mut group = c.benchmark_group("policy_store/load_write");
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        // Inside the window: the record is folded and waits, so the write
        // is timed without a publication.
        let at = fixture.window.get() + Duration::from_millis(1);
        let mut turn = 0u64;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                turn += 1;
                let index = (turn as usize).wrapping_mul(67) % n;
                fixture.store.record(
                    black_box(fixture.ids[index]),
                    0,
                    record_of(index, turn),
                    at,
                    LoadSource::Stream,
                );
            });
        });
        // A cell the filter left out wrote nothing, and nothing waits.
        assert!(turn == 0 || fixture.store.publish_due().is_some());
    }
    group.finish();
}

fn publish(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let mut group = c.benchmark_group("policy_store/publish");
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        let mut turn = 0u64;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || {
                    // One record folded inside the window: the pool has
                    // something to publish at the window's end.
                    turn += 1;
                    let index = (turn as usize).wrapping_mul(67) % n;
                    let opened = fixture.window.get();
                    fixture.store.record(
                        fixture.ids[index],
                        0,
                        record_of(index, turn),
                        opened + Duration::from_millis(1),
                        LoadSource::Stream,
                    );
                    let due = opened + PUBLISH_WINDOW;
                    fixture.window.set(due);
                    due
                },
                |due| fixture.store.publish(black_box(due)),
                BatchSize::PerIteration,
            );
        });
        // A cell the filter left out published nothing: the write cell's
        // records may still wait.
        assert!(turn == 0 || fixture.store.publish_due().is_none());
    }
    group.finish();
}

/// What the bench thread asks of the writer's thread.
#[derive(Default)]
struct Remote {
    /// Write at the busy fleet's rate, through the store's own window.
    busy: AtomicBool,
    /// Publications asked for, one record each, and made.
    asked: AtomicU64,
    made: AtomicU64,
    stop: AtomicBool,
}

impl Remote {
    /// One more publication, made on the writer's core before this returns.
    fn publish_one(&self) {
        let target = self.asked.fetch_add(1, Ordering::Release) + 1;
        while self.made.load(Ordering::Acquire) < target {
            std::hint::spin_loop();
        }
    }
}

/// The writer's thread: it spins on its core and writes what it is asked.
/// Busy, it writes ten records per member per second at the host's clock,
/// and the store publishes by its window. Asked for a publication, it
/// writes one record a window after the last, which publishes by itself.
fn spawn_writer(
    remote: Arc<Remote>,
    store: Arc<MemoryStore>,
    ids: Vec<WorkerId>,
    opened: Instant,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let n = ids.len();
        let period = Duration::from_secs(1) / (10 * n as u32);
        let mut turn = 0u64;
        let mut made = 0u64;
        // The asked publications run on a clock of their own, ahead of any
        // instant the host's clock reaches during the run.
        let mut window = opened + Duration::from_secs(86_400);
        let mut next = Instant::now();
        while !remote.stop.load(Ordering::Relaxed) {
            if made < remote.asked.load(Ordering::Acquire) {
                turn += 1;
                let index = (turn as usize).wrapping_mul(67) % n;
                window += PUBLISH_WINDOW;
                store.record(
                    ids[index],
                    0,
                    record_of(index, turn),
                    window,
                    LoadSource::Stream,
                );
                made += 1;
                remote.made.store(made, Ordering::Release);
            } else if remote.busy.load(Ordering::Relaxed) {
                let now = Instant::now();
                if now >= next {
                    turn += 1;
                    let index = (turn as usize).wrapping_mul(67) % n;
                    store.record(
                        ids[index],
                        0,
                        record_of(index, turn),
                        now,
                        LoadSource::Stream,
                    );
                    next = now.max(next + period);
                }
            } else {
                next = Instant::now();
                std::hint::spin_loop();
            }
        }
    })
}

/// Hold the busy cell to what it claims: with the writer busy, a build
/// reads a load that `before` does not hold, which only a publication of
/// one of the writer's records shows. The store publishes within a window
/// of a write: two seconds without one is a broken cell.
fn await_publication(
    store: &MemoryStore,
    ask: &Ask,
    out: &mut ViewBuffers,
    before: &[Option<LoadRow>],
) {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        build(store, ask, out);
        if out
            .candidates
            .iter()
            .map(|row| row.load)
            .ne(before.iter().copied())
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the busy writer's records reached no build within two seconds"
        );
        std::hint::spin_loop();
    }
}

fn cadence(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let mut group = c.benchmark_group("policy_store_cadence");
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        // A store of its own, read into the shared buffers: the cells
        // before these moved the shared store's window ahead of the host's
        // clock, and a record the busy writer stamps with that clock would
        // wait there for a window that does not end.
        let (store, ids, opened) = pool_of(n);
        let ask = ask_at(opened, false);
        let remote = Arc::new(Remote::default());
        // The thread starts with the first cell that runs: a filter that
        // selects none of them starts none.
        let writer = Cell::new(None);
        let start = || {
            if let Some(started) = writer.take() {
                writer.set(Some(started));
            } else {
                writer.set(Some(spawn_writer(
                    Arc::clone(&remote),
                    Arc::clone(&store),
                    ids.clone(),
                    opened,
                )));
            }
        };
        let out = &mut fixture.out;
        group.throughput(Throughput::Elements(n as u64));
        for state in ["quiet", "busy"] {
            let ran = Cell::new(false);
            group.bench_with_input(BenchmarkId::new(state, n), &n, |b, _| {
                start();
                ran.set(true);
                remote.busy.store(state == "busy", Ordering::Relaxed);
                b.iter(|| build(&store, &ask, out));
                remote.busy.store(false, Ordering::Relaxed);
            });
            if state == "busy" && ran.get() {
                build(&store, &ask, out);
                let before: Vec<Option<LoadRow>> =
                    out.candidates.iter().map(|row| row.load).collect();
                remote.busy.store(true, Ordering::Relaxed);
                await_publication(&store, &ask, out, &before);
                remote.busy.store(false, Ordering::Relaxed);
            }
        }
        group.bench_with_input(BenchmarkId::new("after_publish", n), &n, |b, _| {
            start();
            b.iter_batched(
                || remote.publish_one(),
                |()| build(&store, &ask, out),
                BatchSize::PerIteration,
            );
        });
        remote.stop.store(true, Ordering::Relaxed);
        if let Some(writer) = writer.take() {
            writer.join().expect("the writer's thread ends when asked");
        }
    }
    group.finish();
}

fn store_view(c: &mut Criterion) {
    let mut fixtures: Vec<Fixture> = SIZES.iter().map(|&n| Fixture::new(n)).collect();
    view_build(c, &mut fixtures);
    load_write(c, &mut fixtures);
    publish(c, &mut fixtures);
    cadence(c, &mut fixtures);
}

criterion_group!(benches, store_view);
criterion_main!(benches);
