//! What a signal write comes to: the outcome it returns, and the count of
//! each outcome.

use std::{
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    document::{AppliedSpec, Needs, PolicySpec, API_VERSION},
    engine::{Candidate, RequestFacts, Role, Scorer, TieBand, WorkerId},
    load::SignalColumn,
    stages::{Lifecycle, StageRegistry},
    store::{MemoryStore, SignalSink, SignalWriteOutcome, WorkerFacts, WorkerStatus, WorkersStore},
};

const POOL: &str = "work";
const QUEUE: &str = "queue";
const REQUESTS: &str = "requests";

/// Scores a row by the value it reads of one signal.
#[derive(Debug)]
struct ByValue {
    column: SignalColumn,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Named {
    signal: String,
}

impl Lifecycle for ByValue {}

impl Scorer for ByValue {
    fn name(&self) -> &'static str {
        "by-value"
    }

    fn needs(&self) -> Needs {
        Needs {
            signal_cells: true,
            ..Needs::default()
        }
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&row, score) in kept.iter().zip(out) {
            let row = &candidates[row as usize];
            let value = self.column.read(facts, row);
            *score = value.map_or(0.0, |value| (value / 100.0) as f32);
        }
        TieBand::NORMALIZED
    }
}

/// Reads no signal.
#[derive(Debug)]
struct SeesCells;

impl Lifecycle for SeesCells {}

impl Scorer for SeesCells {
    fn name(&self) -> &'static str {
        "sees-cells"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        out[..kept.len()].fill(1.0);
        TieBand::NORMALIZED
    }
}

struct Stages {
    catalog: StageRegistry,
}

fn stages() -> Stages {
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("by-value", |build| {
            let named: Named = build.params()?;
            let column = build.signal_column(&named.signal)?;
            Ok(Box::new(ByValue { column }))
        })
        .expect("a free name");
    catalog
        .scorer_with("sees-cells", |_| Ok(Box::new(SeesCells)))
        .expect("a free name");
    Stages { catalog }
}

impl Stages {
    fn apply(&self, store: &MemoryStore, document: &Value) {
        let document: PolicySpec = serde_json::from_value(document.clone()).expect("a document");
        let digest = document.digest().expect("a digest");
        let spec = AppliedSpec::validate_digested(&document, digest, &self.catalog, &[]);
        store
            .apply(spec.expect("the document validates"))
            .expect("a newer generation");
    }
}

/// A document of `generation` that declares `signals`, whose one profile
/// scores by `score`.
fn document(generation: u64, signals: Value, score: Value) -> Value {
    json!({
        "apiVersion": API_VERSION,
        "generation": generation,
        "signals": signals,
        "profiles": { "work": { "score": [score], "pick": "best" } }
    })
}

/// The document most cases run under: one signal, read by the profile.
fn queue_document(generation: u64) -> Value {
    document(
        generation,
        json!({ QUEUE: { "unit": REQUESTS } }),
        json!({ "by-value": { "signal": QUEUE } }),
    )
}

fn facts(name: &str, role: Role) -> WorkerFacts {
    let key = format!("http://{name}:8000");
    WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role,
        models: vec![POOL.to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

/// A ready member, with its row built.
fn join(store: &MemoryStore, facts: WorkerFacts) -> WorkerId {
    let id = store.upsert(facts).id();
    let revision = WorkersStore::get(store, id).expect("the row").revision;
    store.set_status(id, WorkerStatus::Ready, revision);
    store.publish_now();
    id
}

/// A store of three ready members of one pool under the queue document.
fn fleet(stages: &Stages) -> (Arc<MemoryStore>, [WorkerId; 3]) {
    let store = Arc::new(MemoryStore::new());
    let ids = ["first", "second", "third"].map(|name| join(&store, facts(name, Role::Single)));
    stages.apply(&store, &queue_document(4));
    (store, ids)
}

fn write(store: &MemoryStore, id: WorkerId, value: f64, at: Instant) -> SignalWriteOutcome {
    store.signal_value(id, 0, &store.signal_column(QUEUE), REQUESTS, value, at)
}

/// A column of `name` under `unit` that the store under test never
/// declared: one read under another store's document.
fn foreign(stages: &Stages, name: &str, unit: &str) -> SignalColumn {
    let other = MemoryStore::new();
    let signals = json!({ name: { "unit": unit } });
    let score = json!({ "by-value": { "signal": name } });
    stages.apply(&other, &document(1, signals, score));
    other.signal_column(name)
}

#[test]
fn a_write_that_is_not_taken_says_what_became_of_it_and_is_counted() {
    let stages = stages();
    let (store, [a, ..]) = fleet(&stages);
    let at = Instant::now();
    let (queue, depth) = (
        store.signal_column(QUEUE),
        foreign(&stages, "depth", REQUESTS),
    );
    let write = |id, column, unit, value, at| store.signal_value(id, 0, column, unit, value, at);
    assert_eq!(
        write(a, &queue, REQUESTS, 5.0, at),
        SignalWriteOutcome::Written
    );
    assert_eq!(
        write(a, &depth, REQUESTS, 5.0, at),
        SignalWriteOutcome::Undeclared
    );
    assert_eq!(
        write(a, &queue, "tokens", 5.0, at),
        SignalWriteOutcome::WrongUnit
    );
    assert_eq!(
        write(a, &queue, REQUESTS, f64::NAN, at),
        SignalWriteOutcome::OutOfRange
    );
    assert_eq!(
        write(a, &queue, REQUESTS, f64::INFINITY, at),
        SignalWriteOutcome::OutOfRange
    );
    assert_eq!(
        write(WorkerId(u64::MAX), &queue, REQUESTS, 5.0, at),
        SignalWriteOutcome::NotAMember
    );
    let before = at
        .checked_sub(Duration::from_millis(1))
        .expect("a millisecond");
    assert_eq!(
        write(a, &queue, REQUESTS, 9.0, before),
        SignalWriteOutcome::Older
    );
    // One count a kind, in the outcomes' order.
    assert_eq!(
        SignalWriteOutcome::EACH.map(SignalWriteOutcome::label)[1],
        "undeclared"
    );
    assert_eq!(store.stats().signal_writes, [1, 1, 1, 2, 1, 1]);
}

/// The sink holds a value's magnitude to the bound a cost rule's own
/// parameters are held to. Past it a value times a coefficient can leave
/// the finite range, and one row's infinite cost levels the score of every
/// other row. The bound itself is taken, under either sign.
#[test]
fn a_value_past_the_magnitude_bound_is_refused_and_counted() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    let at = Instant::now();
    let past = f64::from_bits(1e15_f64.to_bits() + 1);
    assert!(past > 1e15 && past.is_finite());
    assert_eq!(write(&store, a, 1e15, at), SignalWriteOutcome::Written);
    assert_eq!(write(&store, b, -1e15, at), SignalWriteOutcome::Written);
    for value in [past, -past, f64::MAX, f64::NEG_INFINITY] {
        assert_eq!(
            write(&store, c, value, at),
            SignalWriteOutcome::OutOfRange,
            "{value}"
        );
    }
    assert_eq!(
        store.stats().signal_writes[SignalWriteOutcome::OutOfRange.place()],
        4
    );
}

#[test]
fn a_value_stated_for_another_rank_of_a_member_is_taken() {
    let stages = stages();
    let (store, [a, ..]) = fleet(&stages);
    // The worker's row stands at rank 0. A value for its rank 1 is taken
    // for that rank, as a load record would be.
    let queue = store.signal_column(QUEUE);
    let taken = store.signal_value(a, 1, &queue, REQUESTS, 7.0, Instant::now());
    assert_eq!(taken, SignalWriteOutcome::Written);
}

#[test]
fn of_two_sources_the_newer_sample_stands_whichever_arrives_last() {
    let stages = stages();
    let (store, [a, b, _]) = fleet(&stages);
    let early = Instant::now();
    let late = early + Duration::from_millis(10);
    // The newer first: the older is dropped.
    assert_eq!(write(&store, a, 5.0, late), SignalWriteOutcome::Written);
    assert_eq!(write(&store, a, 3.0, early), SignalWriteOutcome::Older);
    // The older first: the newer takes its place.
    assert_eq!(write(&store, b, 3.0, early), SignalWriteOutcome::Written);
    assert_eq!(write(&store, b, 5.0, late), SignalWriteOutcome::Written);
    // Two values of one instant: the last one written is taken too.
    assert_eq!(write(&store, a, 8.0, late), SignalWriteOutcome::Written);
}

#[test]
fn a_new_document_takes_a_changed_unit_under_its_new_unit_alone() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let a = join(&store, facts("first", Role::Single));
    let score = json!({ "by-value": { "signal": "heat" } });
    let before = json!({ "depth": { "unit": "items" }, "heat": { "unit": "celsius" } });
    stages.apply(&store, &document(4, before, score.clone()));
    let at = Instant::now();
    let write =
        |name, unit, value| store.signal_value(a, 0, &store.signal_column(name), unit, value, at);
    assert_eq!(write("depth", "items", 3.0), SignalWriteOutcome::Written);
    assert_eq!(write("heat", "celsius", 71.0), SignalWriteOutcome::Written);

    // A name before both moves the columns, and one unit changes.
    let after = json!({
        "age": { "unit": "ms" },
        "depth": { "unit": "blocks" },
        "heat": { "unit": "celsius" }
    });
    stages.apply(&store, &document(5, after, score));
    let names: Vec<_> = store
        .signal_columns()
        .iter()
        .map(|column| column.name().to_owned())
        .collect();
    assert_eq!(names, ["age", "depth", "heat"]);
    // The sink went under the new columns: a write is taken under the
    // unit the document in force declares, and under no other.
    assert_eq!(write("depth", "items", 4.0), SignalWriteOutcome::WrongUnit);
    assert_eq!(write("depth", "blocks", 4.0), SignalWriteOutcome::Written);
}

/// A source reads its columns, then writes: a document can be applied
/// between the two. A value was made under the column its source read, so
/// it is taken only where the document in force continues that column, and
/// is refused and counted where it does not.
#[test]
fn a_write_under_a_column_read_before_a_document_lands_only_where_it_is_continued() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let a = join(&store, facts("first", Role::Single));
    let score = json!({ "by-value": { "signal": "heat" } });
    let before = json!({ "depth": { "unit": "items" }, "heat": { "unit": "celsius" } });
    stages.apply(&store, &document(4, before, score.clone()));
    let at = Instant::now();
    // The source's read.
    let (depth, heat) = (store.signal_column("depth"), store.signal_column("heat"));

    // A name before both moves the columns, and one reduction changes.
    let after = json!({
        "age": { "unit": "ms" },
        "depth": { "unit": "items", "across_ranks": "max" },
        "heat": { "unit": "celsius" }
    });
    stages.apply(&store, &document(5, after, score));
    let write = |column, unit, value| store.signal_value(a, 0, column, unit, value, at);
    // Made as a sum, under a column that is now a maximum: refused.
    assert_eq!(write(&depth, "items", 3.0), SignalWriteOutcome::Undeclared);
    // Continued: taken, at the column's new place.
    assert_eq!(write(&heat, "celsius", 71.0), SignalWriteOutcome::Written);
    let counts = store.stats().signal_writes;
    assert_eq!(
        (
            counts[SignalWriteOutcome::Written.place()],
            counts[SignalWriteOutcome::Undeclared.place()]
        ),
        (1, 1)
    );
}

/// A write reads where its column is continued with no lock held, so a
/// document can be applied between that read and the write under the
/// lock. The write asks again under the generation it finds there.
#[test]
fn a_write_read_under_the_generation_before_lands_only_where_its_column_is_continued() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let a = join(&store, facts("first", Role::Single));
    let score = json!({ "by-value": { "signal": "heat" } });
    let before = json!({ "depth": { "unit": "items" }, "heat": { "unit": "celsius" } });
    stages.apply(&store, &document(4, before, score.clone()));
    let at = Instant::now();
    let (depth, heat) = (store.signal_column("depth"), store.signal_column("heat"));
    let depth_read = store.signal_read(&depth, "items");
    let heat_read = store.signal_read(&heat, "celsius");

    // A name before both moves the columns, and one reduction changes.
    let after = json!({
        "age": { "unit": "ms" },
        "depth": { "unit": "items", "across_ranks": "max" },
        "heat": { "unit": "celsius" }
    });
    stages.apply(&store, &document(5, after, score));
    let write = |column, value, read| store.signal_write(a, 0, column, value, at, read);
    // Made as a sum, under a column that is now a maximum: not taken.
    assert_eq!(
        write(&depth, 3.0, depth_read),
        SignalWriteOutcome::Undeclared
    );
    // Continued: taken, at the column's new place.
    assert_eq!(write(&heat, 71.0, heat_read), SignalWriteOutcome::Written);
}

#[test]
fn a_refused_document_leaves_the_standing_columns_in_force() {
    let stages = stages();
    let (store, [a, ..]) = fleet(&stages);
    let at = Instant::now();
    assert_eq!(write(&store, a, 7.0, at), SignalWriteOutcome::Written);
    // The standing generation's own number, under other columns.
    let document: PolicySpec = serde_json::from_value(document(
        4,
        json!({ "age": { "unit": "ms" }, QUEUE: { "unit": "tokens" } }),
        json!({ "by-value": { "signal": QUEUE } }),
    ))
    .expect("a document");
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, &stages.catalog, &[]);
    assert!(store.apply(spec.expect("the document validates")).is_err());
    // The standing generation's columns stand: a write under its unit is
    // taken, as before the refusal.
    assert_eq!(write(&store, a, 8.0, at), SignalWriteOutcome::Written);
}

#[test]
fn a_value_stated_for_a_worker_that_left_is_refused() {
    let stages = stages();
    let (store, [a, ..]) = fleet(&stages);
    let at = Instant::now();
    assert_eq!(write(&store, a, 7.0, at), SignalWriteOutcome::Written);
    // The worker leaves: a value stated for the id it held is refused.
    let row = WorkersStore::get(&*store, a).expect("the row");
    assert!(store.remove(a, row.revision).is_some());
    assert_eq!(write(&store, a, 8.0, at), SignalWriteOutcome::NotAMember);
}

#[test]
fn a_document_with_no_signal_drops_a_value_with_no_lock_taken() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let a = join(&store, facts("first", Role::Single));
    stages.apply(
        &store,
        &json!({
            "apiVersion": API_VERSION,
            "generation": 4,
            "profiles": { "work": { "score": ["sees-cells"], "pick": "best" } }
        }),
    );
    assert!(store.signal_columns().is_empty());
    // The write answers while the load lock is held: it takes none.
    let writer = Arc::clone(&store);
    let queue = foreign(&stages, QUEUE, REQUESTS);
    let (tell, told) = mpsc::channel();
    let answered = store.with_loads_held(|| {
        thread::spawn(move || {
            let taken = writer.signal_value(a, 0, &queue, REQUESTS, 5.0, Instant::now());
            tell.send(taken)
        });
        told.recv_timeout(Duration::from_secs(10))
    });
    assert_eq!(answered, Ok(SignalWriteOutcome::Undeclared));
    assert_eq!(store.stats().signal_writes, [0, 1, 0, 0, 0, 0]);
}
