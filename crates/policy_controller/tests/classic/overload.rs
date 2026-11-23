//! The classic overload rows: one worker with its data-parallel ranks, the
//! thresholds the gateway holds for every worker and the ones the worker
//! names for itself, and what the gateway's load monitor is handed, step by
//! step: a poll's answer, a poll that brought no report, a record a rank
//! pushed, the worker's loads dropped. No policy picks here. A row holds
//! whether the gateway read the worker as overloaded after each step.
//!
//! The replay hands each step to the worker crate's own load monitor, built
//! as a host of the engine over worker lists builds it ([`load_monitor`]
//! over the host's holder of worker objects), and reads the verdict off the
//! worker's object, where a decision reads it. A poll's answer goes through
//! the monitor's one poll of a group, a pushed record through the record's
//! own entry, and a worker's loads are dropped by the monitor's own loop
//! when the worker's row leaves the ready set.
//!
//! Every row replays as recorded where the worker stands as one object for
//! all its ranks, which is how the recording gateway held it. A row of
//! several ranks is replayed a second time with one object for each rank,
//! as a registration of several ranks lays them. There the monitor judges
//! each object by its own rank's figures, and [`APART`] names each row that
//! answers apart from the gateway, with the verdict on each rank's object.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::Deserialize;
use smg_policy_controller::{
    engine::WireModes,
    store::{MemoryStore, Overload, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::{load_monitor, Members, Objects},
};
use smg_worker::{
    common_proto::{EngineLoad, KvEventBatch},
    connection::engine::{AdminAnswer, EngineConnection, EngineConnector, HealthAnswer, ZmqLink},
    health::overload::OverloadThresholds,
    identity::{
        basic::Worker,
        builder::{rank_url, BasicWorkerBuilder},
    },
    load::monitor::poll_group_once_for_test,
    registry::WorkerRegistry,
    worker::{
        ConnectionMode, ProfileOptions, RuntimeType, SchedulerLoadSnapshot, WorkerGroupKey,
        WorkerLoadResponse, WorkerType,
    },
};

/// What these rows record, as a sheet's head names it: the gateway's
/// overload verdict of one worker.
pub const POLICY: &str = "overload";

const MODEL: &str = "m";

/// The worker's engine: the address of a worker of one rank, and what the
/// objects of a worker's ranks dial.
const ENGINE: &str = "grpc://w0:9000";

/// The time between two steps of a row, as the recorder sets it.
const STEP: Duration = Duration::from_secs(1);

/// The poll interval a step's poll states: shorter than a step, so a rank
/// that pushed a step ago is asked again, as the recorder asks it.
const INTERVAL: Duration = Duration::from_millis(1);

/// The monitor's own poll interval: a day, so every poll of a row is a
/// step's.
const IDLE: Duration = Duration::from_secs(86_400);

/// Lets the monitor's tasks run until each waits again.
async fn settle() {
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
}

/// A pair of thresholds as the gateway holds them: the queued requests
/// summed over a worker's ranks and the mean KV usage over them, each at or
/// above which the worker is vetoed. `None` leaves a signal off.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Thresholds {
    waiting_requests: Option<u32>,
    token_usage: Option<f64>,
}

/// One rank's figures, in a poll's answer or a pushed record: its queued
/// requests, its KV usage, and the share of that usage its running requests
/// hold where the rank reports one.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Figures {
    waiting: u32,
    usage: f64,
    #[serde(default)]
    active: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// A poll's answer: a row a rank, in rank order, `None` where the
    /// answer omits the rank.
    Poll(Vec<Option<Figures>>),
    /// A poll that brought no report.
    NoReport,
    /// A load record the rank pushed on its event stream.
    Record { rank: u16, figures: Figures },
    /// The worker leaves the ready set and its loads are dropped.
    Evict,
}

/// What a row recorded: whether the worker read as overloaded after each
/// step.
#[derive(Debug, Deserialize)]
struct Outcome {
    verdicts: Vec<bool>,
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    /// The thresholds the gateway holds for every worker.
    gateway: Thresholds,
    /// The thresholds the worker names for itself: a value it names stands
    /// over the gateway's.
    worker: Thresholds,
    /// The worker's data-parallel ranks.
    ranks: u16,
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Debug, Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// How a row's worker stands among the host's objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Laid {
    /// One object under the engine's own address, which states the width
    /// and no rank: the engine whole.
    Whole,
    /// One object for each rank, as a registration of several ranks lays
    /// them.
    Ranks,
}

/// The engine a row's objects dial: it answers a load poll with the answer
/// the step scripted, or with a failure where the step scripted none.
#[derive(Debug, Default)]
struct Engine {
    answer: Mutex<Option<WorkerLoadResponse>>,
}

fn not_scripted<T>() -> Result<T, tonic::Status> {
    Err(tonic::Status::unimplemented(
        "the replay scripts no such call",
    ))
}

#[async_trait]
impl EngineConnection for Engine {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        not_scripted()
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self.answer.lock().clone() {
            Some(answer) => Ok(answer),
            None => Err(tonic::Status::unavailable("the step scripts no report")),
        }
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        not_scripted()
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        not_scripted()
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        not_scripted()
    }

    async fn subscribe_kv_events(
        &self,
        _start_seq: u64,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        not_scripted()
    }
}

/// A connector whose every gRPC connect is answered by the one engine.
#[derive(Debug)]
struct Dials(Arc<Engine>);

#[async_trait]
impl EngineConnector for Dials {
    async fn connect_grpc(
        &self,
        _base_url: &str,
        _runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        Ok(self.0.clone())
    }

    async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        Err("the replay scripts no such connect".to_owned())
    }

    async fn dial_grpc(&self, _base_url: &str, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }
}

/// A poll's answer as an engine sends it: one entry for each rank it names.
fn answer(ranks: u16, figures: &[Option<Figures>]) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: i32::from(ranks),
        loads: figures
            .iter()
            .enumerate()
            .filter_map(|(at, figures)| {
                figures.map(|figures| SchedulerLoadSnapshot {
                    dp_rank: i32::try_from(at).expect("a rank fits the report's field"),
                    num_waiting_reqs: i32::try_from(figures.waiting)
                        .expect("a recorded count fits the report's field"),
                    token_usage: figures.usage,
                    active_token_usage: figures.active,
                    ..SchedulerLoadSnapshot::default()
                })
            })
            .collect(),
        ..WorkerLoadResponse::default()
    }
}

/// A row's own thresholds as a worker's row states them: the usage in
/// permille. `Err` where the row's usage is no whole permille, which a row
/// cannot state.
fn own(thresholds: Thresholds) -> Result<Overload, String> {
    let kv_usage_permille = match thresholds.token_usage {
        None => None,
        Some(usage) => {
            // The object reads the row's permille back as a ratio: the row
            // states the sheet's usage where that ratio is the sheet's.
            let permille = (usage * 1_000.0).round();
            let stated = (0.0..=f64::from(u16::MAX)).contains(&permille)
                && (permille / 1_000.0 - usage).abs() < f64::EPSILON;
            if !stated {
                return Err(format!("a usage of {usage} is no whole permille"));
            }
            Some(permille as u16)
        }
    };
    Ok(Overload {
        kv_usage_permille,
        waiting_requests: thresholds.waiting_requests,
    })
}

/// One worker among a host's objects, its engine and the host's load
/// monitor over them.
struct Bed {
    store: MemoryStore,
    objects: Arc<Objects<Members>>,
    engine: Arc<Engine>,
    /// The worker's objects: one, or one a rank in rank order.
    workers: Vec<Arc<dyn Worker>>,
    ranks: u16,
    laid: Laid,
}

impl Bed {
    fn new(row: &Row, laid: Laid) -> Result<Self, String> {
        let engine = Arc::new(Engine::default());
        let dials: Arc<dyn EngineConnector> = Arc::new(Dials(Arc::clone(&engine)));
        // The holder as a controller builds its own, with the gateway's
        // thresholds where a controller states its program's defaults: a
        // row's own values reach its object under any defaults, none too.
        let objects = Arc::new(
            Objects::new(Arc::new(WorkerRegistry::new()))
                .with_connector(dials)
                .with_host_local_unchecked()
                .with_status_followed()
                .with_overload_thresholds(OverloadThresholds {
                    waiting_requests: row.gateway.waiting_requests.map(|waiting| waiting as usize),
                    token_usage: row.gateway.token_usage,
                }),
        );
        let store = MemoryStore::with_publish_window(Duration::ZERO);
        let overload = own(row.worker)?;
        let facts = |key: String, dp_rank: u16| WorkerFacts {
            key,
            engine: ENGINE.to_string(),
            incarnation: 1,
            models: vec![MODEL.to_string()],
            dp_rank,
            dp_size: row.ranks,
            wire_modes: WireModes::GRPC,
            overload,
            ..WorkerFacts::default()
        };
        let keys: Vec<String> = match laid {
            Laid::Ranks if row.ranks > 1 => (0..row.ranks)
                .map(|rank| rank_url(ENGINE, usize::from(rank)))
                .collect(),
            _ => vec![ENGINE.to_string()],
        };
        let mut workers = Vec::with_capacity(keys.len());
        for (rank, key) in keys.iter().enumerate() {
            let id = store.upsert(facts(key.clone(), rank as u16)).id();
            let stored = WorkersStore::get(&store, id).ok_or("no row behind its write")?;
            store.set_status(id, WorkerStatus::Ready, stored.revision);
        }
        store.publish_now();
        objects.follow(&store);
        for key in &keys {
            let object = objects.registry().get_by_url(key);
            workers.push(object.ok_or_else(|| format!("no object under {key}"))?);
        }
        Ok(Self {
            store,
            objects,
            engine,
            workers,
            ranks: row.ranks,
            laid,
        })
    }

    /// Writes `status` on each row of the worker and has the holder read
    /// the change.
    fn status(&self, status: WorkerStatus) -> Result<(), String> {
        for worker in &self.workers {
            let row = self
                .store
                .by_key(worker.url())
                .ok_or("a worker object with no row")?;
            self.store.set_status(row.id, status, row.revision);
        }
        self.store.publish_now();
        self.objects.follow(&self.store);
        Ok(())
    }

    /// The object a rank's pushed record lands on.
    fn pusher(&self, rank: u16) -> Option<&Arc<dyn Worker>> {
        match self.laid {
            Laid::Ranks if self.ranks > 1 => self.workers.get(usize::from(rank)),
            _ => self.workers.first(),
        }
    }

    /// Whether each object of the worker reads as overloaded, in the
    /// objects' order.
    fn verdicts(&self) -> Vec<bool> {
        self.workers
            .iter()
            .map(|worker| worker.is_overloaded())
            .collect()
    }
}

/// What the host's load monitor says of the row's worker after each step:
/// the verdict on each of the worker's objects. `Err` where the row states
/// what a worker's row cannot.
fn replay(row: &Row, laid: Laid) -> Result<Vec<Vec<bool>>, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .map_err(|err| err.to_string())?;
    runtime.block_on(async {
        let bed = Bed::new(row, laid)?;
        let monitor = load_monitor(&bed.objects, IDLE);
        // The monitor's own loop hears a worker leave the ready set. A
        // group's loop polls once as it starts, ahead of the row's first
        // step: the engine scripts no answer yet, so that poll brings no
        // report to a monitor that holds none.
        monitor.start_event_loop();
        settle().await;
        let group = WorkerGroupKey {
            model_id: MODEL.to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Grpc,
        };
        let start = Instant::now();
        let mut verdicts = Vec::with_capacity(row.steps.len());
        for (at, step) in row.steps.iter().enumerate() {
            let now = start + STEP * (at as u32 + 1);
            match step {
                Step::Poll(figures) => {
                    *bed.engine.answer.lock() = Some(answer(row.ranks, figures));
                    poll_group_once_for_test(&monitor, &group, INTERVAL, now).await;
                }
                Step::NoReport => {
                    *bed.engine.answer.lock() = None;
                    poll_group_once_for_test(&monitor, &group, INTERVAL, now).await;
                }
                Step::Record { rank, figures } => {
                    let pusher = bed.pusher(*rank).ok_or("a record of no rank's object")?;
                    let record = EngineLoad {
                        waiting_requests: figures.waiting,
                        token_usage: figures.usage,
                        active_token_usage: figures.active,
                        ..EngineLoad::default()
                    };
                    monitor.apply_pushed_load(pusher, i32::from(*rank), &record, now);
                }
                Step::Evict => {
                    // The worker's rows leave the ready set, the holder
                    // moves each object's status, and the monitor's loop
                    // drops the worker's loads at that word.
                    bed.status(WorkerStatus::NotReady)?;
                    settle().await;
                    // The recorder's worker is polled again behind the
                    // drop: its rows are ready again.
                    bed.status(WorkerStatus::Ready)?;
                    settle().await;
                }
            }
            verdicts.push(bed.verdicts());
        }
        Ok(verdicts)
    })
}

/// A recorded row of several ranks that a worker laid as one object a rank
/// answers apart from the gateway: its replay under that layout must give
/// `verdicts`, the verdict on each rank's object after each step. The row
/// fails the day its rank objects all carry the recorded verdicts, so an
/// entry cannot outlive the rule, and on any third outcome.
struct Apart {
    row: &'static str,
    verdicts: &'static [&'static [bool]],
}

/// The one rule that holds the rows of [`APART`] apart. The gateway that
/// recorded the sheet held one worker for all the ranks of an engine and
/// judged it by the queue summed over its ranks and the usage averaged over
/// them. The load monitor hands each object of an engine the answer of its
/// own rank and judges the object by it.
const RANK_RULE: &str = "a rank's object is judged by its own rank's figures, not by the queue \
                         summed and the usage averaged over the engine's ranks";

/// The rows a worker of one object a rank answers apart.
const APART: &[Apart] = &[
    Apart {
        row: "the-queue-vetoes-at-its-count-and-above-summed-over-two-ranks",
        verdicts: &[
            &[false, false],
            &[false, false],
            &[true, false],
            &[false, false],
        ],
    },
    Apart {
        row: "the-usage-vetoes-at-its-mean-and-above-over-two-ranks",
        verdicts: &[
            &[false, true],
            &[false, true],
            &[true, true],
            &[true, false],
            &[false, true],
        ],
    },
    Apart {
        row: "two-ranks-that-trade-their-usage-in-one-answer-move-no-verdict",
        verdicts: &[&[true, false], &[false, true], &[true, false]],
    },
    Apart {
        row: "an-answer-that-names-no-rank-reads-as-no-report",
        verdicts: &[&[true, false], &[false, false], &[true, false]],
    },
    Apart {
        row: "a-pushed-record-moves-the-verdict-by-its-own-rank",
        verdicts: &[
            &[false, false],
            &[false, false],
            &[false, false],
            &[true, false],
            &[false, false],
        ],
    },
    Apart {
        row: "a-record-after-a-poll-is-judged-beside-the-poll-s-figures-once-every-rank-has-pushed",
        verdicts: &[
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
        ],
    },
    Apart {
        row: "a-poll-after-a-record-is-judged-by-the-poll-s-own-figures",
        verdicts: &[&[true, false], &[false, false], &[false, false]],
    },
    Apart {
        row: "a-rank-an-answer-omits-leaves-the-sum-at-once",
        verdicts: &[
            &[false, true],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
        ],
    },
    Apart {
        row: "a-rank-an-answer-omits-leaves-the-mean-at-once",
        verdicts: &[
            &[true, false],
            &[true, false],
            &[true, false],
            &[true, false],
            &[true, false],
            &[true, false],
        ],
    },
    Apart {
        row: "a-record-s-usage-is-judged-over-the-ranks-that-have-pushed",
        verdicts: &[
            &[false, false],
            &[true, false],
            &[true, false],
            &[true, false],
            &[true, false],
            &[true, false],
            &[true, true],
        ],
    },
    Apart {
        row: "an-answer-that-names-no-rank-takes-the-ranks-that-pushed-before-it-off-the-report",
        verdicts: &[
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
            &[false, false],
        ],
    },
];

/// `None` when the worker's one object carries the recorded verdict after
/// each step; the reason otherwise.
fn held_whole(row: &Row, here: &[Vec<bool>]) -> Option<String> {
    let here: Vec<&[bool]> = here.iter().map(Vec::as_slice).collect();
    let recorded: Vec<[bool; 1]> = row.outcome.verdicts.iter().map(|&v| [v]).collect();
    let recorded: Vec<&[bool]> = recorded.iter().map(<[bool; 1]>::as_slice).collect();
    (here != recorded).then(|| {
        format!(
            "recorded {:?}, replayed {here:?} on the worker's one object",
            row.outcome.verdicts
        )
    })
}

/// `None` when every rank's object carries the recorded verdict after each
/// step, or the verdicts its entry holds when [`APART`] names the row; the
/// reason otherwise. A row held apart whose rank objects all carry the
/// recorded verdicts is a defect of its entry.
fn held_ranks(row: &Row, here: &[Vec<bool>]) -> Option<String> {
    let recorded = &row.outcome.verdicts;
    let as_recorded = here.len() == recorded.len()
        && here
            .iter()
            .zip(recorded)
            .all(|(objects, verdict)| objects.iter().all(|object| object == verdict));
    match APART.iter().find(|apart| apart.row == row.name) {
        None => (!as_recorded)
            .then(|| format!("recorded {recorded:?}, replayed {here:?} on an object a rank")),
        Some(_) if as_recorded => Some(format!(
            "an object a rank replays as recorded now ({RANK_RULE}): take the row out of APART"
        )),
        Some(apart) => {
            let held: Vec<Vec<bool>> = apart.verdicts.iter().map(|step| step.to_vec()).collect();
            (here != held).then(|| {
                format!(
                    "recorded {recorded:?}, apart by its entry {:?}, replayed {here:?} on an \
                     object a rank",
                    apart.verdicts
                )
            })
        }
    }
}

/// What a row's replay leaves unheld: under the worker as one object, and
/// for a worker of several ranks under one object a rank too.
fn defects_of(row: &Row) -> Vec<String> {
    let mut defects = Vec::new();
    match replay(row, Laid::Whole) {
        Ok(here) => defects.extend(held_whole(row, &here)),
        Err(why) => defects.push(format!("refused here: {why}")),
    }
    if row.ranks > 1 {
        match replay(row, Laid::Ranks) {
            Ok(here) => defects.extend(held_ranks(row, &here)),
            Err(why) => defects.push(format!("refused here on an object a rank: {why}")),
        }
    }
    defects
}

/// The rows of a sheet replayed: how many, and what did not replay.
pub fn replay_rows(text: &str, source: &str) -> (usize, Vec<String>) {
    let sheet: Sheet = match serde_json::from_str(text) {
        Ok(sheet) => sheet,
        Err(err) => {
            return (
                0,
                vec![format!("{source} is not a sheet of {POLICY} rows: {err}")],
            )
        }
    };
    let defects = sheet
        .rows
        .iter()
        .flat_map(|row| {
            defects_of(row)
                .into_iter()
                .map(move |why| format!("{source}: {}: {why}", row.name))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// Every row held apart is an inline row, so every run of the suite checks
/// that the rule still decides it, and how: as recorded on the worker's one
/// object, and by its entry on an object a rank.
#[test]
fn every_row_held_apart_is_an_inline_row_held_to_its_entry() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    assert_eq!(sheet.rows.len(), APART.len());
    for apart in APART {
        let row = sheet
            .rows
            .iter()
            .find(|row| row.name == apart.row)
            .unwrap_or_else(|| panic!("{} is no inline row", apart.row));
        assert!(row.ranks > 1, "{}: one rank is one object", apart.row);
        let whole = replay(row, Laid::Whole).expect("the worker as one object");
        assert_eq!(held_whole(row, &whole), None, "{}", apart.row);
        let ranks = replay(row, Laid::Ranks).expect("an object a rank");
        assert_eq!(held_ranks(row, &ranks), None, "{}", apart.row);
        let held: Vec<Vec<bool>> = apart.verdicts.iter().map(|step| step.to_vec()).collect();
        assert_eq!(ranks, held, "{}", apart.row);
    }
}

/// An entry cannot outlive the rule: rank objects that all carry a row's
/// recorded verdicts are a defect of the row's entry. An answer that is
/// neither the recorded one nor the entry's is a defect as well.
#[test]
fn an_entry_fails_on_the_recorded_verdicts_and_on_a_third_answer() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    for row in &sheet.rows {
        let width = usize::from(row.ranks);
        let recorded: Vec<Vec<bool>> = row
            .outcome
            .verdicts
            .iter()
            .map(|&verdict| vec![verdict; width])
            .collect();
        let outlived = held_ranks(row, &recorded).expect("an entry that outlived the rule");
        assert!(outlived.contains("take the row out of APART"), "{outlived}");
        // The entry's own verdicts less their first step: neither answer.
        let entry = APART
            .iter()
            .find(|apart| apart.row == row.name)
            .expect("the row's entry");
        let third: Vec<Vec<bool>> = entry
            .verdicts
            .iter()
            .skip(1)
            .map(|step| step.to_vec())
            .collect();
        let neither = held_ranks(row, &third).expect("a third answer");
        assert!(neither.contains("apart by its entry"), "{neither}");
    }
}

/// A row no entry names is held to what was recorded on every rank's
/// object: a row held apart, replayed under a name no entry holds, is a
/// defect there and none on the worker's one object.
#[test]
fn a_row_no_entry_names_is_held_to_what_was_recorded_on_every_rank() {
    let mut sheet: serde_json::Value = serde_json::from_str(INLINE).expect("the inline rows");
    let row = &mut sheet["rows"][0];
    row["name"] = serde_json::Value::from("a-row-no-entry-names");
    let row: Row = serde_json::from_value(row.take()).expect("a row");
    let defects = defects_of(&row);
    assert!(
        matches!(defects.as_slice(), [one] if one.contains("on an object a rank")),
        "{defects:#?}"
    );
}

/// The sheet by the layouts that answer it: every row as recorded where the
/// worker is one object, and of its fourteen rows of several ranks eleven
/// apart where each rank is an object.
#[test]
fn the_sheet_replays_whole_and_eleven_of_its_rank_rows_stand_apart() {
    let sheet: Sheet =
        serde_json::from_str(include_str!("sheets/overload.json")).expect("the sheet");
    assert_eq!(sheet.rows.len(), 25);
    let mut several = 0;
    for row in &sheet.rows {
        let whole = replay(row, Laid::Whole).expect("the worker as one object");
        assert_eq!(held_whole(row, &whole), None, "{}", row.name);
        several += usize::from(row.ranks > 1);
    }
    assert_eq!((several, APART.len()), (14, 11));
    for apart in APART {
        let named = sheet.rows.iter().any(|row| row.name == apart.row);
        assert!(named, "{} is on no sheet", apart.row);
    }
}

/// An object the holder lays for a rank's row states what the worker
/// crate's own builder states for a rank worker: the rank's address, the
/// engine's address under it, the rank and the width. A gateway that
/// registers each rank of an engine as a worker of its own builds them so.
/// The worker as one object states the width and no rank.
#[test]
fn a_rank_object_states_what_a_rank_worker_of_the_builder_states() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    let row = sheet.rows.first().expect("an inline row");
    let width = usize::from(row.ranks);
    let bed = Bed::new(row, Laid::Ranks).expect("the rank objects");
    assert_eq!(bed.workers.len(), width);
    for (rank, object) in bed.workers.iter().enumerate() {
        let built = <BasicWorkerBuilder>::new(ENGINE)
            .dp_config(rank, width)
            .build();
        assert_eq!(object.url(), built.url());
        assert_eq!(object.base_url(), built.base_url());
        assert_eq!(object.dp_rank(), built.dp_rank());
        assert_eq!(object.dp_size(), built.dp_size());
        assert_eq!(object.is_dp_aware(), built.is_dp_aware());
    }
    let whole = Bed::new(row, Laid::Whole).expect("the one object");
    let [object] = whole.workers.as_slice() else {
        panic!("{} objects of a worker laid whole", whole.workers.len());
    };
    assert_eq!(
        (object.url(), object.dp_rank(), object.dp_size()),
        (ENGINE, None, Some(width))
    );
}

/// A row whose own usage is no whole permille is refused: a worker's row
/// states its usage in permille.
#[test]
fn a_usage_of_no_whole_permille_is_refused() {
    let both = |token_usage| Thresholds {
        waiting_requests: Some(2),
        token_usage,
    };
    assert_eq!(
        own(both(Some(0.5))).map(|own| (own.kv_usage_permille, own.waiting_requests)),
        Ok((Some(500), Some(2)))
    );
    assert_eq!(own(both(None)).map(|own| own.kv_usage_permille), Ok(None));
    assert!(own(both(Some(0.5005))).is_err());
}

/// The rows a worker of one object a rank answers apart, copied whole from
/// the sheet: every run of the suite replays them.
pub const INLINE: &str = r#"{
  "recorded_from": "436f0c8c5725b0a408b990003d76eaddc72c4be4",
  "policy": "overload",
  "rows": [
    {
      "name": "the-queue-vetoes-at-its-count-and-above-summed-over-two-ranks",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 3, "usage": 0.125}, {"waiting": 4, "usage": 0.125}]},
        {"poll": [{"waiting": 4, "usage": 0.125}, {"waiting": 4, "usage": 0.125}]},
        {"poll": [{"waiting": 9, "usage": 0.125}, {"waiting": 0, "usage": 0.125}]},
        {"poll": [{"waiting": 0, "usage": 0.125}, {"waiting": 7, "usage": 0.125}]}
      ],
      "outcome": {"verdicts": [false, true, true, false]}
    },
    {
      "name": "the-usage-vetoes-at-its-mean-and-above-over-two-ranks",
      "gateway": {"waiting_requests": 8, "token_usage": 0.5},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 0, "usage": 0.25}, {"waiting": 0, "usage": 0.5}]},
        {"poll": [{"waiting": 0, "usage": 0.25}, {"waiting": 0, "usage": 0.75}]},
        {"poll": [{"waiting": 0, "usage": 0.5}, {"waiting": 0, "usage": 0.5}]},
        {"poll": [{"waiting": 0, "usage": 1.0}, {"waiting": 0, "usage": 0.0}]},
        {"poll": [{"waiting": 0, "usage": 0.375}, {"waiting": 0, "usage": 0.5}]}
      ],
      "outcome": {"verdicts": [false, true, true, true, false]}
    },
    {
      "name": "two-ranks-that-trade-their-usage-in-one-answer-move-no-verdict",
      "gateway": {"waiting_requests": 8, "token_usage": 0.5},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 0, "usage": 0.75}, {"waiting": 0, "usage": 0.125}]},
        {"poll": [{"waiting": 0, "usage": 0.125}, {"waiting": 0, "usage": 0.75}]},
        {"poll": [{"waiting": 0, "usage": 0.75}, {"waiting": 0, "usage": 0.125}]}
      ],
      "outcome": {"verdicts": [false, false, false]}
    },
    {
      "name": "an-answer-that-names-no-rank-reads-as-no-report",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 9, "usage": 0.125}, {"waiting": 0, "usage": 0.125}]},
        {"poll": [null, null]},
        {"poll": [{"waiting": 9, "usage": 0.125}, {"waiting": 0, "usage": 0.125}]}
      ],
      "outcome": {"verdicts": [true, false, true]}
    },
    {
      "name": "a-pushed-record-moves-the-verdict-by-its-own-rank",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 1, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 1, "figures": {"waiting": 0, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 8, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 0, "usage": 0.125}}}
      ],
      "outcome": {"verdicts": [false, true, false, true, false]}
    },
    {
      "name": "a-record-after-a-poll-is-judged-beside-the-poll-s-figures-once-every-rank-has-pushed",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"record": {"rank": 0, "figures": {"waiting": 1, "usage": 0.125}}},
        {"record": {"rank": 1, "figures": {"waiting": 1, "usage": 0.125}}},
        {"poll": [{"waiting": 4, "usage": 0.125}, {"waiting": 3, "usage": 0.125}]},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 1, "figures": {"waiting": 0, "usage": 0.125}}}
      ],
      "outcome": {"verdicts": [false, false, false, true, false]}
    },
    {
      "name": "a-poll-after-a-record-is-judged-by-the-poll-s-own-figures",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"record": {"rank": 0, "figures": {"waiting": 9, "usage": 0.125}}},
        {"poll": [{"waiting": 1, "usage": 0.125}, {"waiting": 1, "usage": 0.125}]},
        {"poll": [{"waiting": 4, "usage": 0.125}, {"waiting": 4, "usage": 0.125}]}
      ],
      "outcome": {"verdicts": [true, false, true]}
    },
    {
      "name": "a-rank-an-answer-omits-leaves-the-sum-at-once",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 1, "usage": 0.125}, {"waiting": 9, "usage": 0.125}]},
        {"poll": [{"waiting": 1, "usage": 0.125}, null]},
        {"poll": [{"waiting": 1, "usage": 0.125}, null]},
        {"poll": [{"waiting": 1, "usage": 0.125}, null]},
        {"poll": [{"waiting": 1, "usage": 0.125}, null]},
        {"poll": [{"waiting": 1, "usage": 0.125}, null]}
      ],
      "outcome": {"verdicts": [true, false, false, false, false, false]}
    },
    {
      "name": "a-rank-an-answer-omits-leaves-the-mean-at-once",
      "gateway": {"waiting_requests": 8, "token_usage": 0.5},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 0, "usage": 0.75}, {"waiting": 0, "usage": 0.125}]},
        {"poll": [{"waiting": 0, "usage": 0.75}, null]},
        {"poll": [{"waiting": 0, "usage": 0.75}, null]},
        {"poll": [{"waiting": 0, "usage": 0.75}, null]},
        {"poll": [{"waiting": 0, "usage": 0.75}, null]},
        {"poll": [{"waiting": 0, "usage": 0.75}, null]}
      ],
      "outcome": {"verdicts": [false, true, true, true, true, true]}
    },
    {
      "name": "a-record-s-usage-is-judged-over-the-ranks-that-have-pushed",
      "gateway": {"waiting_requests": 8, "token_usage": 0.5},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 0, "usage": 0.125}, {"waiting": 0, "usage": 0.125}]},
        {"record": {"rank": 0, "figures": {"waiting": 0, "usage": 0.75}}},
        {"record": {"rank": 0, "figures": {"waiting": 0, "usage": 0.75}}},
        {"record": {"rank": 0, "figures": {"waiting": 0, "usage": 0.75}}},
        {"record": {"rank": 0, "figures": {"waiting": 0, "usage": 0.75}}},
        {"record": {"rank": 0, "figures": {"waiting": 0, "usage": 0.75}}},
        {"record": {"rank": 1, "figures": {"waiting": 0, "usage": 0.75}}}
      ],
      "outcome": {"verdicts": [false, true, true, true, true, true, true]}
    },
    {
      "name": "an-answer-that-names-no-rank-takes-the-ranks-that-pushed-before-it-off-the-report",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 1, "figures": {"waiting": 5, "usage": 0.125}}},
        {"poll": [null, null]},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 1, "figures": {"waiting": 5, "usage": 0.125}}}
      ],
      "outcome": {"verdicts": [false, true, false, false, false, false, false, false, true]}
    }
  ]
}
"#;
