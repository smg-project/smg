//! The cost of a signal's write through the store:
//!
//! - `policy_store_signals/write/row_alone`: one rank's load record;
//! - `policy_store_signals/write/four_values`: one rank's load record and
//!   the four named values behind it, as a source writes one report.
//!
//! The document declares four signals, so a row holds four cells, and its
//! scorer reads two of them. No gate reads these rows yet.

#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;
#[path = "common/reads_two.rs"]
mod reads_two;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, Criterion};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec, API_VERSION},
    engine::{LoadRow, Role, SignalSource, WorkerId},
    stages::StageRegistry,
    store::{
        LoadSink, LoadSource, MemoryStore, SignalSink, SignalWriteOutcome, WorkerFacts,
        WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
};

use crate::reads_two::ReadsTwo;

const MODEL: &str = "m";
/// The declared signals, in name order: a row's cells stand in it.
const NAMES: [&str; 4] = ["pending", "spare_one", "spare_two", "used"];
const UNIT: &str = "tokens";

/// The document every cell runs under: four signals, two of them read.
fn applied() -> AppliedSpec {
    let mut catalog = StageRegistry::builtin();
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
    // A source reads the columns once a report and writes under each.
    let columns = store.signal_columns();
    for (nth, column) in columns.iter().enumerate() {
        let value = ((index * 7 + nth * 3) % 23) as f64;
        let taken = store.signal_value(id, 0, column, UNIT, value, at);
        assert_eq!(taken, SignalWriteOutcome::Written, "{}", column.name());
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

criterion_group!(benches, write_cells);
criterion_main!(benches);
