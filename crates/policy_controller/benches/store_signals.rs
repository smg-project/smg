//! The cost of signal cells through the store, under a scorer a host
//! registers that reads two signals of every row:
//!
//! - `policy_store_signals/decide/lent/<n>`: one decision's view and the
//!   engine's answer over a publication's own rows, its block of cells lent
//!   beside them (`World::view`), at 8, 64 and 512 members of one pool;
//! - `policy_store_signals/decide/written/<n>`: the same over rows written
//!   for the decision, their cells written beside them;
//! - `policy_store_signals/write/row_alone`: one rank's load record;
//! - `policy_store_signals/write/four_values`: one rank's load record and
//!   the four named values behind it, as a source writes one report.
//!
//! The document declares four signals, so a row holds four cells and the
//! scorer reads two of them. No gate reads these rows yet.

#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, SignalSink, Taken, ViewBuffers,
        WorkerFacts, WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
    AppliedSpec, Candidate, Catalog, Constraints, Lifecycle, LoadRow, Needs, PolicySpec,
    RequestFacts, Role, Scorer, Scratch, SignalColumn, SignalSource, TieBand, WorkerId,
    API_VERSION,
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";
const ASKER: GatewayId = GatewayId(1);
/// The declared signals, in name order: a row's cells stand in it.
const NAMES: [&str; 4] = ["pending", "spare_one", "spare_two", "used"];
const UNIT: &str = "tokens";

/// Scores a row by the two values it reads: the less of both, the better.
#[derive(Debug)]
struct ReadsTwo(SignalColumn, SignalColumn);

impl Lifecycle for ReadsTwo {}

impl Scorer for ReadsTwo {
    fn name(&self) -> &'static str {
        "reads-two"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&index, score) in kept.iter().zip(out) {
            let row = &candidates[index as usize];
            let used = self.0.read(facts, row).unwrap_or(0.0);
            let pending = self.1.read(facts, row).unwrap_or(0.0);
            *score = 1.0 / (1.0 + (used + pending) as f32);
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            signal_cells: true,
            ..Needs::default()
        }
    }
}

/// The document every cell runs under: four signals, two of them read.
fn applied() -> AppliedSpec {
    let mut catalog = Catalog::builtin();
    catalog
        .scorer_with("reads-two", |build| {
            let used = build.signal_column("used")?;
            let pending = build.signal_column("pending")?;
            Ok(Box::new(ReadsTwo(used, pending)))
        })
        .expect("a free name");
    let signals: serde_json::Map<_, _> = NAMES
        .iter()
        .map(|name| ((*name).to_owned(), serde_json::json!({ "unit": UNIT })))
        .collect();
    let document: PolicySpec = serde_json::from_value(serde_json::json!({
        "apiVersion": API_VERSION,
        "generation": 2,
        "signals": signals,
        "profiles": { "work": { "score": ["reads-two"], "pick": "best" } }
    }))
    .expect("a document");
    AppliedSpec::validate_in(&document, &catalog, &[]).expect("the document validates")
}

/// Member `index`'s record: figures that differ per member.
fn record_of(index: usize) -> LoadRow {
    let x = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    LoadRow {
        running: (x >> 16) as u32 % 12,
        queued: (x >> 24) as u32 % 4,
        report_age_ms: 5,
        source: SignalSource::Events,
        ..LoadRow::default()
    }
}

/// One rank's four values behind its record, each taken.
fn state(store: &MemoryStore, id: WorkerId, index: usize, at: Instant) {
    for (nth, name) in NAMES.into_iter().enumerate() {
        let value = ((index * 7 + nth * 3) % 23) as f64;
        let taken = store.signal_value(id, 0, name, UNIT, value, at);
        assert_eq!(taken, Taken::Written, "{name}");
    }
}

/// A store with one pool of `n` members under the document, every one
/// ready, with its record and its four values on file and published: the
/// store, its members by index and the instant its last window opened.
fn pool_of(n: usize) -> (MemoryStore, Vec<WorkerId>, Instant) {
    let store = MemoryStore::new();
    store.apply(applied()).expect("a newer generation");
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let key = format!("http://w{index}:8000");
            let facts = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                models: vec![MODEL.to_string()],
                dp_size: 1,
                max_running: Some(4),
                ..WorkerFacts::default()
            };
            let id = store.upsert(facts).id();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            id
        })
        .collect();
    store.publish_now();
    let opened = Instant::now();
    for (index, &id) in ids.iter().enumerate() {
        store.record(id, 0, record_of(index), opened, LoadSource::Stream);
        state(&store, id, index, opened);
    }
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    assert_eq!(store.publish_due(), None);
    filled::filled(&store, MODEL, Role::Single, n);
    (store, ids, opened)
}

/// The whole decision over lent rows and over written ones, in turn.
fn decide_cells(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_signals/decide");
    for n in SIZES {
        let (store, _, opened) = pool_of(n);
        let ask = Ask {
            gateway: ASKER,
            now: opened + Duration::from_secs(1),
            pinned: None,
            affinity: false,
            signal_cells: true,
        };
        let mut out = ViewBuffers::default();
        let mut holds = Vec::new();
        let mut scratch = Scratch::new();
        let mut seed = 1u64;
        for (door, written) in [("lent", false), ("written", true)] {
            group.bench_with_input(BenchmarkId::new(door, n), &n, |b, _| {
                b.iter(|| {
                    seed = seed.wrapping_add(1);
                    store.with_world(|world| {
                        let view =
                            world.view(MODEL, Role::Single, black_box(&ask), written, &mut out);
                        let view = view.expect("the pool");
                        let facts = RequestFacts {
                            model: MODEL,
                            prompt_tokens: 512,
                            seed,
                            now_micros: view.built.now_micros,
                            pinned_at: view.pinned_at,
                            affinity: view.affinity,
                            lent_in_flight: view.in_flight.as_ref(),
                            signal_cells: view.signal_cells,
                            ..RequestFacts::default()
                        };
                        let cx = Constraints {
                            stalled: view.stalled,
                            held: world.held(&view, &ask, &mut holds),
                            ..Constraints::default()
                        };
                        let engine = world.spec().engine();
                        let pool = view.pool_facts();
                        black_box(engine.decide(&facts, &pool, view.candidates, &cx, &mut scratch))
                    })
                });
            });
        }
    }
    group.finish();
}

/// One report's write: the row alone, and the row with four values behind
/// it. Each write is a microsecond past the last, so the window publishes
/// as a host's loop would have it.
fn write_cells(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_signals/write");
    for (name, values) in [("row_alone", false), ("four_values", true)] {
        let (store, ids, opened) = pool_of(64);
        let mut at = opened;
        let mut index = 0usize;
        group.bench_function(name, |b| {
            b.iter(|| {
                at += Duration::from_micros(1);
                index = (index + 1) % ids.len();
                let id = ids[index];
                store.record(id, 0, record_of(index), at, LoadSource::Stream);
                if values {
                    state(&store, id, index, at);
                }
                black_box(store.publish_due())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, decide_cells, write_cells);
criterion_main!(benches);
