//! A written value reaches a decision in a publication, beside its row.

use std::{
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    door::{Door, Request},
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, Reachability, SignalSink, Taken,
        ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore,
    },
    AppliedSpec, Candidate, Catalog, Constraints, Lifecycle, LoadRow, Needs, Phase, PolicySpec,
    RequestFacts, Role, Scorer, SignalColumn, TieBand, WireModes, WorkerId, API_VERSION,
};

const POOL: &str = "work";
const GATEWAY: GatewayId = GatewayId(37);
const OTHER: GatewayId = GatewayId(53);
const QUEUE: &str = "queue";
const REQUESTS: &str = "requests";

/// What the stages read, in the order they were asked: the signal's name,
/// the row's worker and the value.
type Read = Arc<Mutex<Vec<(String, WorkerId, Option<f64>)>>>;

/// One row's cells as a publication holds them: each cell's value and its
/// sample instant, or none.
type Held = Vec<Option<(f64, u64)>>;

/// Scores a row by the value it reads of one signal, and keeps what it
/// read.
#[derive(Debug)]
struct ByValue {
    column: SignalColumn,
    read: Read,
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
        let mut read = self.read.lock();
        for (&row, score) in kept.iter().zip(out) {
            let row = &candidates[row as usize];
            let value = self.column.read(facts, row);
            read.push((self.column.name().to_owned(), row.id, value));
            *score = value.map_or(0.0, |value| (value / 100.0) as f32);
        }
        TieBand::NORMALIZED
    }
}

/// Reads no signal, and keeps whether each decision's facts held cells.
#[derive(Debug)]
struct SeesCells(Arc<Mutex<Vec<bool>>>);

impl Lifecycle for SeesCells {}

impl Scorer for SeesCells {
    fn name(&self) -> &'static str {
        "sees-cells"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        _: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.0.lock().push(facts.signal_cells.is_some());
        out[..kept.len()].fill(1.0);
        TieBand::NORMALIZED
    }
}

struct Stages {
    catalog: Catalog,
    read: Read,
    saw_cells: Arc<Mutex<Vec<bool>>>,
}

fn stages() -> Stages {
    let read = Read::default();
    let saw_cells = Arc::<Mutex<Vec<bool>>>::default();
    let (kept, seen) = (Arc::clone(&read), Arc::clone(&saw_cells));
    let mut catalog = Catalog::builtin();
    catalog
        .scorer_with("by-value", move |build| {
            let named: Named = build.params()?;
            let column = build.signal_column(&named.signal)?;
            Ok(Box::new(ByValue {
                column,
                read: Arc::clone(&kept),
            }))
        })
        .expect("a free name");
    catalog
        .scorer_with("sees-cells", move |_| {
            Ok(Box::new(SeesCells(Arc::clone(&seen))))
        })
        .expect("a free name");
    Stages {
        catalog,
        read,
        saw_cells,
    }
}

impl Stages {
    /// What the stages read of `signal` since the last call, by worker.
    fn read_of(&self, signal: &str) -> Vec<(WorkerId, Option<f64>)> {
        let mut read: Vec<_> = self
            .read
            .lock()
            .drain(..)
            .filter(|(name, ..)| name == signal)
            .map(|(_, id, value)| (id, value))
            .collect();
        read.sort_by_key(|&(id, _)| id);
        read
    }

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

fn write(store: &MemoryStore, id: WorkerId, value: f64, at: Instant) -> Taken {
    store.signal_value(id, 0, QUEUE, REQUESTS, value, at)
}

/// Each row's cells in the block the standing publication lends for
/// `role`, by worker: a cell's value and its sample instant, or none.
fn lent(store: &MemoryStore, role: Role) -> Vec<(WorkerId, Held)> {
    store.deciding(|world| {
        let ask = Ask {
            gateway: GATEWAY,
            now: Instant::now(),
            pinned: None,
            affinity: false,
            signal_cells: true,
        };
        let mut out = ViewBuffers::default();
        let view = world
            .view(POOL, role, &ask, false, &mut out)
            .expect("the pool");
        let block = view.signal_cells.expect("the publication's block");
        assert!(block.fits(view.candidates.len()), "one cell a row a signal");
        let cells = block.cells().chunks(block.signals());
        let of = |cells: &[crate::SignalCell]| {
            let held = |cell: &crate::SignalCell| {
                (!cell.is_none()).then_some((cell.value, cell.sampled_at_micros))
            };
            cells.iter().map(held).collect()
        };
        let ids = view.candidates.iter().map(|row| row.id);
        ids.zip(cells.map(of)).collect()
    })
}

/// The one cell of each row, by worker, under a document of one signal.
fn values(store: &MemoryStore) -> Vec<(WorkerId, Option<f64>)> {
    let mut values: Vec<_> = lent(store, Role::Single)
        .into_iter()
        .map(|(id, cells)| (id, cells[0].map(|(value, _)| value)))
        .collect();
    values.sort_by_key(|&(id, _)| id);
    values
}

fn request(id: &[u8]) -> Request<'_> {
    Request {
        id,
        facts: RequestFacts {
            model: POOL,
            ..RequestFacts::default()
        },
        constraints: Constraints::default(),
    }
}

/// The worker one decision of `request` picks.
fn picked(door: &Door, gateway: GatewayId, request: &Request<'_>, now: Instant) -> WorkerId {
    let decided = door.decide(gateway, request, now).expect("a decision");
    let pick = decided.decision.pick().expect("a pick");
    pick.targets.first().expect("a target").id
}

/// `values` in the order of their workers' ids, as a read is listed.
fn by_id<const N: usize>(mut values: [(WorkerId, Option<f64>); N]) -> Vec<(WorkerId, Option<f64>)> {
    values.sort_by_key(|&(id, _)| id);
    values.to_vec()
}

/// What waits for the window's end is shown.
fn shown(store: &MemoryStore) {
    if let Some(due) = store.publish_due() {
        store.publish(due);
    }
}

#[test]
fn a_written_value_stands_at_its_rows_place_in_the_next_publication_and_a_decision_reads_it() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    let at = Instant::now();
    assert_eq!(write(&store, b, 7.0, at), Taken::Written);
    assert_eq!(
        values(&store),
        by_id([(a, None), (b, Some(7.0)), (c, None)])
    );
    let sampled: Vec<_> = lent(&store, Role::Single)
        .into_iter()
        .filter_map(|(_, cells)| cells[0])
        .collect();
    assert_eq!(sampled, [(7.0, store.micros(at))]);

    // The decision is handed the publication's own block.
    let door = Door::new(Arc::clone(&store));
    assert_eq!(picked(&door, GATEWAY, &request(b"r-1"), at), b);
    assert_eq!(
        stages.read_of(QUEUE),
        by_id([(a, None), (b, Some(7.0)), (c, None)])
    );
}

/// The three workers of a fleet where `apart` is told apart by `facts`:
/// each of them joined by `facts(name, whether it is the one apart)`.
fn fleet_with(
    stages: &Stages,
    apart: usize,
    with: impl Fn(WorkerFacts, bool) -> WorkerFacts,
) -> (Arc<MemoryStore>, [WorkerId; 3]) {
    let store = Arc::new(MemoryStore::new());
    let mut nth = 0..;
    let ids = ["first", "second", "third"].map(|name| {
        let apart = nth.next() == Some(apart);
        join(&store, with(facts(name, Role::Single), apart))
    });
    stages.apply(&store, &queue_document(4));
    for (id, value) in ids.into_iter().zip([10.0, 20.0, 30.0]) {
        assert_eq!(write(&store, id, value, Instant::now()), Taken::Written);
    }
    (store, ids)
}

/// How many of the two rows left of three stand at another place than in
/// the pool: the rows behind the one cut.
fn moved(ids: [WorkerId; 3], cut: usize) -> usize {
    ids.iter().filter(|&&id| id > ids[cut]).count()
}

#[test]
fn rows_written_for_a_gateway_read_their_own_values_at_their_new_places() {
    let stages = stages();
    let mut moves = 0;
    for local in 0..3 {
        // One worker only the first gateway reaches: another gateway's rows
        // are written without it, and the rows behind it move up.
        let (store, ids) = fleet_with(&stages, local, |facts, local| WorkerFacts {
            reachable_from: if local {
                Reachability::Gateway(GATEWAY)
            } else {
                Reachability::Any
            },
            ..facts
        });
        moves += moved(ids, local);
        let door = Door::new(Arc::clone(&store));
        let mut expected = [
            (ids[0], Some(10.0)),
            (ids[1], Some(20.0)),
            (ids[2], Some(30.0)),
        ];
        picked(&door, GATEWAY, &request(b"own"), Instant::now());
        assert_eq!(stages.read_of(QUEUE), by_id(expected), "its own gateway");
        // The row with the most is picked: 30, or 20 where 30 is cut.
        let most = if local == 2 { ids[1] } else { ids[2] };
        assert_eq!(
            picked(&door, OTHER, &request(b"other"), Instant::now()),
            most
        );
        expected[local].1 = None;
        let read = by_id(expected);
        let read: Vec<_> = read
            .into_iter()
            .filter(|&(id, _)| id != ids[local])
            .collect();
        assert_eq!(stages.read_of(QUEUE), read, "another gateway");
    }
    assert!(moves > 0, "a row moved up");
}

#[test]
fn rows_a_gateways_wires_leave_read_their_own_values_at_their_new_places() {
    let stages = stages();
    let mut moves = 0;
    for cut in 0..3 {
        // One worker the asking gateway's wire does not dial: the build's
        // rows are narrowed, and the rows behind it move up.
        let (store, ids) = fleet_with(&stages, cut, |facts, cut| WorkerFacts {
            wire_modes: if cut {
                WireModes::HTTP
            } else {
                WireModes::GRPC
            },
            ..facts
        });
        moves += moved(ids, cut);
        let door = Door::new(Arc::clone(&store));
        let mut asked = request(b"wired");
        asked.constraints.wire_modes = WireModes::GRPC;
        let most = if cut == 2 { ids[1] } else { ids[2] };
        assert_eq!(picked(&door, GATEWAY, &asked, Instant::now()), most);
        let all = by_id([
            (ids[0], Some(10.0)),
            (ids[1], Some(20.0)),
            (ids[2], Some(30.0)),
        ]);
        let read: Vec<_> = all.into_iter().filter(|&(id, _)| id != ids[cut]).collect();
        assert_eq!(stages.read_of(QUEUE), read);
    }
    assert!(moves > 0, "a row moved up");
}

#[test]
fn each_side_of_a_pair_reads_its_own_pools_values() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let prefills = ["p0", "p1"].map(|name| join(&store, facts(name, Role::Prefill)));
    let decodes = ["d0", "d1"].map(|name| join(&store, facts(name, Role::Decode)));
    stages.apply(
        &store,
        &json!({
            "apiVersion": API_VERSION,
            "generation": 4,
            "signals": { QUEUE: { "unit": REQUESTS } },
            "profiles": {
                "front": { "score": [{ "by-value": { "signal": QUEUE } }], "pick": "best" },
                "back": { "score": [{ "by-value": { "signal": QUEUE } }], "pick": "best" }
            },
            "bindings": [{
                "match": { "phase": ["prefill", "decode"] },
                "plan": "pd-pair",
                "profiles": { "prefill": "front", "decode": "back" }
            }]
        }),
    );
    // One position holds a value in each pool, and the two differ.
    let at = Instant::now();
    let stated = prefills
        .into_iter()
        .chain(decodes)
        .zip([11.0, 12.0, 61.0, 62.0]);
    for (id, value) in stated {
        assert_eq!(write(&store, id, value, at), Taken::Written);
    }
    let door = Door::new(Arc::clone(&store));
    let mut asked = request(b"pair");
    asked.facts.phase = Phase::Prefill;
    let decided = door.decide(GATEWAY, &asked, at).expect("a pair's decision");
    let pick = decided.decision.pick().expect("a pair pick");
    let target = pick.targets.first().expect("the first pair");
    let pair = target.pair.expect("the target names a decode");
    assert_eq!((target.id, pair.decode_id), (prefills[1], decodes[1]));
    let read = stages.read_of(QUEUE);
    let expected = by_id([
        (prefills[0], Some(11.0)),
        (prefills[1], Some(12.0)),
        (decodes[0], Some(61.0)),
        (decodes[1], Some(62.0)),
    ]);
    assert_eq!(read, expected);
}

#[test]
fn a_write_that_is_not_taken_says_what_became_of_it_and_is_counted() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    let at = Instant::now();
    let write = |id, name, unit, value, at| store.signal_value(id, 0, name, unit, value, at);
    assert_eq!(write(a, QUEUE, REQUESTS, 5.0, at), Taken::Written);
    assert_eq!(write(a, "depth", REQUESTS, 5.0, at), Taken::Undeclared);
    assert_eq!(write(a, QUEUE, "tokens", 5.0, at), Taken::WrongUnit);
    assert_eq!(write(a, QUEUE, REQUESTS, f64::NAN, at), Taken::NotFinite);
    assert_eq!(
        write(a, QUEUE, REQUESTS, f64::INFINITY, at),
        Taken::NotFinite
    );
    assert_eq!(
        write(WorkerId(u64::MAX), QUEUE, REQUESTS, 5.0, at),
        Taken::NotAMember
    );
    let before = at
        .checked_sub(Duration::from_millis(1))
        .expect("a millisecond");
    assert_eq!(write(a, QUEUE, REQUESTS, 9.0, before), Taken::Older);
    // One count a kind, in the outcomes' order, and the value taken stands.
    assert_eq!(Taken::EACH.map(Taken::label)[1], "undeclared");
    assert_eq!(store.counts().signal_writes, [1, 1, 1, 2, 1, 1]);
    assert_eq!(
        values(&store),
        by_id([(a, Some(5.0)), (b, None), (c, None)])
    );
}

#[test]
fn a_value_stated_for_another_rank_reaches_no_row() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    // The worker's row stands at rank 0. A value for its rank 1 is on file
    // for that rank, as a load record would be, and on no row.
    let taken = store.signal_value(a, 1, QUEUE, REQUESTS, 7.0, Instant::now());
    assert_eq!(taken, Taken::Written);
    assert_eq!(values(&store), by_id([(a, None), (b, None), (c, None)]));
}

#[test]
fn of_two_sources_the_newer_sample_stands_whichever_arrives_last() {
    let stages = stages();
    let (store, [a, b, _]) = fleet(&stages);
    let early = Instant::now();
    let late = early + Duration::from_millis(10);
    // The newer first: the older is dropped.
    assert_eq!(write(&store, a, 5.0, late), Taken::Written);
    assert_eq!(write(&store, a, 3.0, early), Taken::Older);
    // The older first: the newer takes its place.
    assert_eq!(write(&store, b, 3.0, early), Taken::Written);
    assert_eq!(write(&store, b, 5.0, late), Taken::Written);
    let cells = lent(&store, Role::Single);
    for id in [a, b] {
        let held = cells.iter().find(|(row, _)| *row == id).expect("the row");
        assert_eq!(held.1, [Some((5.0, store.micros(late)))], "{id:?}");
    }
    // Two values of one instant: the last one written stands.
    assert_eq!(write(&store, a, 8.0, late), Taken::Written);
    let cells = lent(&store, Role::Single);
    let held = cells.iter().find(|(row, _)| *row == a).expect("the row");
    assert_eq!(held.1, [Some((8.0, store.micros(late)))]);
}

#[test]
fn a_value_stands_until_its_signals_maximum_age_ends_it() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    stages.apply(
        &store,
        &document(
            5,
            json!({ QUEUE: { "unit": REQUESTS, "max_age_ms": 500 } }),
            json!({ "by-value": { "signal": QUEUE } }),
        ),
    );
    let at = Instant::now();
    assert_eq!(write(&store, a, 7.0, at), Taken::Written);
    let door = Door::new(Arc::clone(&store));
    // At exactly its maximum age the value is read.
    picked(
        &door,
        GATEWAY,
        &request(b"r-1"),
        at + Duration::from_millis(500),
    );
    assert_eq!(
        stages.read_of(QUEUE),
        by_id([(a, Some(7.0)), (b, None), (c, None)])
    );
    // Past it the cell stands and reads as no value.
    picked(
        &door,
        GATEWAY,
        &request(b"r-2"),
        at + Duration::from_millis(501),
    );
    assert_eq!(
        stages.read_of(QUEUE),
        by_id([(a, None), (b, None), (c, None)])
    );
    assert_eq!(
        values(&store),
        by_id([(a, Some(7.0)), (b, None), (c, None)])
    );
}

#[test]
fn a_value_that_rides_one_record_in_ten_is_read_between_them_under_its_own_instant() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    let door = Door::new(Arc::clone(&store));
    let start = Instant::now();
    let record = |nth: u32| {
        let at = start + Duration::from_millis(100) * nth;
        let row = LoadRow {
            running: nth,
            ..LoadRow::default()
        };
        store.record(a, 0, row, at, LoadSource::Stream);
        at
    };
    // The first record carries the value, and the nine behind it none.
    let first = record(0);
    assert_eq!(write(&store, a, 9.0, first), Taken::Written);
    let mut last = first;
    for nth in 1..10 {
        last = record(nth);
        shown(&store);
        picked(&door, GATEWAY, &request(b"between"), last);
        assert_eq!(
            stages.read_of(QUEUE),
            by_id([(a, Some(9.0)), (b, None), (c, None)]),
            "behind record {nth}"
        );
    }
    // The row is the last record's, and the value keeps its own instant.
    let cells = lent(&store, Role::Single);
    let held = cells.iter().find(|(row, _)| *row == a).expect("the row");
    assert_eq!(held.1, [Some((9.0, store.micros(first)))]);
    assert_eq!(
        LoadSink::load(&*store, a, 0).expect("a sample").row.running,
        9
    );
    assert!(last > first);
}

#[test]
fn a_new_document_keeps_a_name_it_declares_again_and_starts_a_changed_unit_empty() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let a = join(&store, facts("first", Role::Single));
    let b = join(&store, facts("second", Role::Single));
    let score = json!({ "by-value": { "signal": "heat" } });
    let before = json!({ "depth": { "unit": "items" }, "heat": { "unit": "celsius" } });
    stages.apply(&store, &document(4, before, score.clone()));
    let at = Instant::now();
    let write = |name, unit, value| store.signal_value(a, 0, name, unit, value, at);
    assert_eq!(write("depth", "items", 3.0), Taken::Written);
    assert_eq!(write("heat", "celsius", 71.0), Taken::Written);

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
    // With no new report: the kept name is read at its new column.
    let held = lent(&store, Role::Single);
    let cells = |id| &held.iter().find(|(row, _)| *row == id).expect("the row").1;
    assert_eq!(*cells(a), [None, None, Some((71.0, store.micros(at)))]);
    assert_eq!(*cells(b), [None, None, None]);
    let door = Door::new(Arc::clone(&store));
    assert_eq!(picked(&door, GATEWAY, &request(b"r-1"), at), a);
    assert_eq!(stages.read_of("heat"), by_id([(a, Some(71.0)), (b, None)]));
    // The table went under the new columns too: a write lands at them.
    assert_eq!(write("depth", "items", 4.0), Taken::WrongUnit);
    assert_eq!(write("depth", "blocks", 4.0), Taken::Written);
    let held = lent(&store, Role::Single);
    let cells = &held.iter().find(|(row, _)| *row == a).expect("the row").1;
    assert_eq!(cells[..2], [None, Some((4.0, store.micros(at)))]);
}

#[test]
fn a_new_document_that_changes_a_signals_reduction_starts_the_name_empty() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let a = join(&store, facts("first", Role::Single));
    let score = json!({ "by-value": { "signal": "heat" } });
    let before = json!({ "depth": { "unit": "items" }, "heat": { "unit": "celsius" } });
    stages.apply(&store, &document(4, before, score.clone()));
    let at = Instant::now();
    let write = |name, unit, value| store.signal_value(a, 0, name, unit, value, at);
    assert_eq!(write("depth", "items", 3.0), Taken::Written);
    assert_eq!(write("heat", "celsius", 71.0), Taken::Written);

    // The same names under the same units: one signal's ranks are made
    // one another way.
    let after = json!({
        "depth": { "unit": "items", "across_ranks": "max" },
        "heat": { "unit": "celsius" }
    });
    stages.apply(&store, &document(5, after, score));
    // The value on file was made the way that stood, so the name starts
    // empty. The other name keeps its value.
    let held = lent(&store, Role::Single);
    let cells = &held.iter().find(|(row, _)| *row == a).expect("the row").1;
    assert_eq!(**cells, [None, Some((71.0, store.micros(at)))]);
    // The next report states the name again.
    assert_eq!(write("depth", "items", 4.0), Taken::Written);
    let held = lent(&store, Role::Single);
    let cells = &held.iter().find(|(row, _)| *row == a).expect("the row").1;
    assert_eq!(cells[0], Some((4.0, store.micros(at))));
}

/// A write reads its column with no lock held, so a document can be
/// applied between that read and the write. The value was made under the
/// column the write read.
#[test]
fn a_write_read_under_the_generation_before_lands_only_where_its_column_is_continued() {
    let stages = stages();
    let store = Arc::new(MemoryStore::new());
    let a = join(&store, facts("first", Role::Single));
    let score = json!({ "by-value": { "signal": "heat" } });
    let before = json!({ "depth": { "unit": "items" }, "heat": { "unit": "celsius" } });
    stages.apply(&store, &document(4, before, score.clone()));
    let at = Instant::now();
    let depth = store.signal_read("depth", "items");
    let heat = store.signal_read("heat", "celsius");

    // A name before both moves the columns, and one reduction changes.
    let after = json!({
        "age": { "unit": "ms" },
        "depth": { "unit": "items", "across_ranks": "max" },
        "heat": { "unit": "celsius" }
    });
    stages.apply(&store, &document(5, after, score));
    let write = |names, value, read| store.signal_write(a, 0, names, value, at, read);
    // Made as a sum, under a column that is now a maximum: not taken.
    assert_eq!(write(("depth", "items"), 3.0, depth), Taken::Undeclared);
    // Continued: taken, at the column's new place.
    assert_eq!(write(("heat", "celsius"), 71.0, heat), Taken::Written);
    let held = lent(&store, Role::Single);
    let cells = &held.iter().find(|(row, _)| *row == a).expect("the row").1;
    assert_eq!(**cells, [None, None, Some((71.0, store.micros(at)))]);
}

#[test]
fn a_refused_document_moves_no_value() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    let at = Instant::now();
    assert_eq!(write(&store, a, 7.0, at), Taken::Written);
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
    assert_eq!(
        values(&store),
        by_id([(a, Some(7.0)), (b, None), (c, None)])
    );
    // A membership build reads every value from the table again: the
    // table stands under the standing generation's columns.
    let d = join(&store, facts("fourth", Role::Single));
    assert_eq!(
        values(&store),
        by_id([(a, Some(7.0)), (b, None), (c, None), (d, None)])
    );
    assert_eq!(write(&store, a, 8.0, at), Taken::Written);
}

#[test]
fn a_worker_that_leaves_and_joins_again_starts_with_no_value() {
    let stages = stages();
    let (store, [a, b, c]) = fleet(&stages);
    let at = Instant::now();
    for id in [a, b] {
        assert_eq!(write(&store, id, 7.0, at), Taken::Written);
    }
    // The first leaves and joins again as its next incarnation.
    let row = WorkersStore::get(&*store, a).expect("the row");
    assert!(store.remove(a, row.revision).is_some());
    assert_eq!(write(&store, a, 8.0, at), Taken::NotAMember);
    let again = join(
        &store,
        WorkerFacts {
            incarnation: 2,
            ..facts("first", Role::Single)
        },
    );
    assert_eq!(
        values(&store),
        by_id([(again, None), (b, Some(7.0)), (c, None)])
    );
    // The second's reports stop: its values go with its load samples.
    LoadSink::evict(&*store, b);
    assert_eq!(values(&store), by_id([(again, None), (b, None), (c, None)]));
}

#[test]
fn a_profile_that_reads_no_signal_is_handed_no_cells() {
    let stages = stages();
    let (store, _) = fleet_with(&stages, usize::MAX, |facts, _| WorkerFacts {
        wire_modes: WireModes::GRPC,
        ..facts
    });
    let door = Door::new(Arc::clone(&store));
    picked(&door, GATEWAY, &request(b"r-1"), Instant::now());
    assert_eq!(stages.read_of(QUEUE).len(), 3, "the profile that reads one");
    // The same signals under a profile whose stage resolves none: the
    // values stand in the publication, and the decision is handed none,
    // over the publication's rows and over rows written for its wires.
    stages.apply(
        &store,
        &document(
            5,
            json!({ QUEUE: { "unit": REQUESTS } }),
            json!("sees-cells"),
        ),
    );
    let held = values(&store);
    assert_eq!(held.iter().filter(|(_, value)| value.is_some()).count(), 3);
    picked(&door, GATEWAY, &request(b"r-2"), Instant::now());
    let mut wired = request(b"r-3");
    wired.constraints.wire_modes = WireModes::GRPC;
    picked(&door, GATEWAY, &wired, Instant::now());
    assert_eq!(*stages.saw_cells.lock(), [false, false]);
}

#[test]
fn a_document_with_no_signal_publishes_no_block_and_drops_a_value_with_no_lock_taken() {
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
    store.deciding(|world| {
        let ask = Ask {
            gateway: GATEWAY,
            now: Instant::now(),
            pinned: None,
            affinity: false,
            signal_cells: true,
        };
        let mut out = ViewBuffers::default();
        let lent = world.view(POOL, Role::Single, &ask, false, &mut out);
        assert!(lent.expect("the pool").signal_cells.is_none());
        let written = world.view(POOL, Role::Single, &ask, true, &mut out);
        assert!(written.expect("the pool").signal_cells.is_none());
    });
    // The write answers while the load lock is held: it takes none.
    let writer = Arc::clone(&store);
    let (tell, told) = mpsc::channel();
    let answered = store.with_loads_held(|| {
        thread::spawn(move || {
            let taken = writer.signal_value(a, 0, QUEUE, REQUESTS, 5.0, Instant::now());
            tell.send(taken)
        });
        told.recv_timeout(Duration::from_secs(10))
    });
    assert_eq!(answered, Ok(Taken::Undeclared));
    assert_eq!(store.counts().signal_writes, [0, 1, 0, 0, 0, 0]);
}

#[test]
fn a_window_of_moved_cells_allocates_no_block_past_the_second() {
    let stages = stages();
    let (store, [a, ..]) = fleet(&stages);
    let start = Instant::now();
    let mut blocks = Vec::new();
    for nth in 0..6u32 {
        let at = start + Duration::from_millis(u64::from(nth));
        assert_eq!(write(&store, a, f64::from(nth), at), Taken::Written);
        blocks.push(store.publication_allocations());
    }
    assert_eq!(
        values(&store)
            .iter()
            .filter_map(|(_, value)| *value)
            .sum::<f64>(),
        5.0
    );
    // The first publication takes a second block, and the two then trade.
    assert_eq!(blocks[1..], [blocks[1]; 5], "{blocks:?}");
    assert!(blocks[1] <= blocks[0] + 1, "{blocks:?}");
}
