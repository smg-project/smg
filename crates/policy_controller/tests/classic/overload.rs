//! The classic overload rows: one worker with its data-parallel ranks, the
//! thresholds the gateway holds for every worker and the ones the worker
//! names for itself, and what the gateway's load monitor is handed, step by
//! step: a poll's answer, a poll that brought no report, a record a rank
//! pushed, the worker's loads dropped. No policy picks here. A row holds
//! whether the gateway read the worker as overloaded after each step; its
//! replay writes each step through the store's load sink, as this crate's
//! driver writes it, and reads the verdict off the rows a decision is
//! handed.
//!
//! The store judges an engine by rules of its own, and [`APART`] names each
//! recorded row they decide apart from the gateway, with the verdicts here.

use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use serde::Deserialize;
use serde_json::Value;
use smg_policy_controller::{
    ingest::rows,
    spec::{BindingSpec, OverloadSpec},
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, Overload, ViewBuffers, WorkerFacts,
        WorkerStatus, WorkersStore,
    },
    view::permille,
    AppliedSpec, LoadRow, PolicySpec, Role, SignalSource, WorkerId,
};

/// What these rows record, as a sheet's head names it: the gateway's
/// overload verdict of one worker.
pub const POLICY: &str = "overload";

const MODEL: &str = "m";

/// The report period the replay tells the store, as this crate's driver
/// tells its own, and the time between two steps of a row.
const PERIOD: Duration = Duration::from_secs(10);

/// The worker's engine: the key of a worker of one rank, and what the rows
/// of a worker's ranks share.
const ENGINE: &str = "http://w0:8000";

/// A pair of thresholds as the gateway holds them: the queued requests
/// summed over a worker's ranks and the mean KV usage over them, each at or
/// above which the worker is vetoed. `None` leaves a signal off.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Thresholds {
    waiting_requests: Option<u32>,
    token_usage: Option<f32>,
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

/// A recorded row this crate decides apart from the gateway, by a rule of
/// its own that `rule` states: its replay must give `verdicts`, the verdict
/// after each step here. The row fails the day it replays as recorded, so
/// an entry cannot outlive its rule, and on any third outcome.
struct Apart {
    row: &'static str,
    verdicts: &'static [bool],
    rule: &'static str,
}

/// The rows decided apart. The store holds a usage and its threshold in
/// permille and judges an engine, at every write, over every sample on
/// file for it that lies within three report periods of the engine's
/// newest. The gateway compares ratios, judges a poll by its answer alone
/// and a pushed record by the worker's pushed report: the ranks that have
/// pushed. A rank the gateway stops reading at once is read here for
/// three periods more: a row that repeats the step that leaves a rank
/// unheard replays as recorded from its fourth repeat on.
const APART: &[Apart] = &[
    Apart {
        row: "a-usage-under-the-threshold-by-less-than-half-a-permille-is-under-it",
        verdicts: &[true, false],
        rule: "a usage within half a permille under the threshold rounds to it",
    },
    Apart {
        row: "a-rank-an-answer-omits-leaves-the-sum-at-once",
        verdicts: &[true, true, true, true, false, false],
        rule: "a rank an answer omits keeps its sample three report periods behind its engine's newest",
    },
    Apart {
        row: "a-rank-an-answer-omits-leaves-the-mean-at-once",
        verdicts: &[false, false, false, false, true, true],
        rule: "a rank an answer omits keeps its sample three report periods behind its engine's newest",
    },
    Apart {
        row: "a-record-after-a-poll-is-judged-over-the-ranks-that-have-pushed",
        verdicts: &[false, true, true, true, false, false, false],
        rule: "a record is judged beside a poll's row of another rank until that row lies three report periods behind",
    },
    Apart {
        row: "a-record-s-usage-is-judged-over-the-ranks-that-have-pushed",
        verdicts: &[false, false, false, false, true, true, true],
        rule: "a record is judged beside a poll's row of another rank until that row lies three report periods behind",
    },
    Apart {
        row: "an-answer-that-names-no-rank-takes-the-ranks-that-pushed-before-it-off-the-report",
        verdicts: &[false, true, false, true, true, false, false, false, true],
        rule: "an answer with no row changes nothing on file: a rank it names keeps its sample three report periods",
    },
];

/// The generation whose one binding holds the gateway's thresholds: both
/// signals, or the block switched off when the gateway holds neither. The
/// document's block has no spelling for one signal alone, and no recorded
/// row asks for it; the validator's words otherwise.
fn generation(gateway: Thresholds) -> Result<AppliedSpec, String> {
    let overload = match (gateway.waiting_requests, gateway.token_usage) {
        (Some(waiting), Some(usage)) => OverloadSpec {
            waiting_requests: Some(waiting),
            token_usage: Some(usage),
            ..OverloadSpec::default()
        },
        (None, None) => OverloadSpec {
            enabled: false,
            ..OverloadSpec::default()
        },
        _ => return Err("an overload block holds both signals or is switched off".to_string()),
    };
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = 1;
    let profile = document
        .profiles
        .keys()
        .next()
        .expect("the preset's profile")
        .clone();
    document.bindings = vec![BindingSpec {
        profile: Some(profile),
        overload: Some(overload),
        ..BindingSpec::default()
    }];
    AppliedSpec::validate(&document).map_err(|err| err.to_string())
}

/// One rank's figures as the row a host writes for the rank: through the
/// crate's own row builder, from the report of that rank alone, read at the
/// instant it was sampled.
fn load_row(figures: Figures, at: Instant, source: SignalSource) -> LoadRow {
    let wire = WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            num_waiting_reqs: i32::try_from(figures.waiting)
                .expect("a recorded count fits the wire's field"),
            token_usage: figures.usage,
            active_token_usage: figures.active,
            ..SchedulerLoadSnapshot::default()
        }],
        ..WorkerLoadResponse::default()
    };
    rows::load_row(&wire, Role::Single, at, at, source)
}

/// The store with the row's worker in it: a member row a rank, the ranks
/// of one engine, each with the worker's own thresholds, under a generation
/// whose one binding holds the gateway's.
struct Bed {
    store: MemoryStore,
    /// The member row of each rank, by rank.
    ids: Vec<WorkerId>,
    /// The ranks the engine's last answer left out beside a rank it named,
    /// by rank: the ranks the store marks, kept here by the same rule.
    left_out: RefCell<Vec<bool>>,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's form.
    fn new(row: &Row) -> Result<Self, String> {
        let applied = generation(row.gateway)?;
        let store = MemoryStore::new();
        let own = Overload {
            waiting_requests: row.worker.waiting_requests,
            // The one rounding a report's ratio takes.
            kv_usage_permille: row
                .worker
                .token_usage
                .map(|usage| permille(f64::from(usage))),
        };
        let ids = (0..row.ranks)
            .map(|rank| {
                let key = if row.ranks == 1 {
                    ENGINE.to_string()
                } else {
                    format!("{ENGINE}@{rank}")
                };
                let id = store
                    .upsert(WorkerFacts {
                        key,
                        engine: ENGINE.to_string(),
                        incarnation: 1,
                        models: vec![MODEL.to_string()],
                        dp_rank: rank,
                        dp_size: row.ranks,
                        overload: own,
                        ..WorkerFacts::default()
                    })
                    .id();
                let revision = WorkersStore::get(&store, id).expect("the row").revision;
                store
                    .set_status(id, WorkerStatus::Ready, revision)
                    .expect("ready");
                id
            })
            .collect();
        store.apply(applied).expect("the first generation applies");
        // The driver tells its poll interval where it is built.
        assert!(store.heard_every(PERIOD), "the first period told");
        Ok(Self {
            store,
            ids,
            left_out: RefCell::new(vec![false; usize::from(row.ranks)]),
        })
    }

    /// One step written through the load sink at `now`, each as the crate's
    /// driver writes it, and what waits shown: an engine's answer whole in
    /// one report, a poll that brought no report as an answer with no row,
    /// a record under its rank's own id, and the eviction rank by rank.
    fn step(&self, step: &Step, now: Instant) {
        let ranks = self.ids.iter().copied().zip(0u16..);
        match step {
            Step::Poll(figures) => {
                assert_eq!(figures.len(), self.ids.len(), "an entry a rank");
                // An answer that names a rank leaves out each rank it has
                // no row for. One that names none says nothing of any.
                if figures.iter().any(Option::is_some) {
                    *self.left_out.borrow_mut() = figures.iter().map(Option::is_none).collect();
                }
                let answer: Vec<_> = ranks
                    .zip(figures)
                    .map(|((id, rank), figures)| {
                        let row = figures.map(|figures| load_row(figures, now, SignalSource::Poll));
                        (id, rank, row)
                    })
                    .collect();
                self.store.report(&answer, now, LoadSource::Poll);
            }
            Step::NoReport => {
                let answer: Vec<_> = ranks.map(|(id, rank)| (id, rank, None)).collect();
                self.store.report(&answer, now, LoadSource::Poll);
            }
            Step::Record { rank, figures } => {
                let id = *self
                    .ids
                    .get(usize::from(*rank))
                    .expect("a rank of the worker");
                let row = load_row(*figures, now, SignalSource::Events);
                self.store.record(id, *rank, row, now, LoadSource::Stream);
                // A record names its rank again.
                if let Some(out) = self.left_out.borrow_mut().get_mut(usize::from(*rank)) {
                    *out = false;
                }
            }
            Step::Evict => {
                for (id, _) in ranks {
                    self.store.evict(id);
                }
                self.left_out.borrow_mut().fill(false);
            }
        }
        if let Some(due) = self.store.publish_due() {
            self.store.publish(due);
        }
    }

    /// Whether the worker reads as overloaded to a decision at `now`: the
    /// verdict of its rank rows as [`one_verdict`] takes it, each with whether the
    /// engine's last answer left the rank out.
    fn verdict(&self, now: Instant) -> Option<bool> {
        let mut out = ViewBuffers::default();
        let ask = Ask {
            gateway: GatewayId(1),
            now,
            pinned: None,
            affinity: false,
        };
        self.store
            .build_view(MODEL, Role::Single, &ask, &mut out)
            .expect("the pool");
        let left_out = self.left_out.borrow();
        one_verdict(out.candidates.iter().map(|row| {
            let rank = self.ids.iter().position(|&id| id == row.id);
            let left_out = rank.and_then(|rank| left_out.get(rank).copied());
            (left_out.unwrap_or(false), row.health.overloaded)
        }))
    }
}

/// A worker's verdict off its rank rows, each as whether the engine's last
/// answer left the rank out and whether the row carries the overload bit:
/// the bit the named ranks' rows share. A rank left out carries the bit
/// alone: that is the store's avoidance of the rank and no verdict of the
/// engine. `None` where the named ranks' rows carry two verdicts, where a
/// rank left out carries no bit, and where no rank is named.
fn one_verdict(rows: impl IntoIterator<Item = (bool, bool)>) -> Option<bool> {
    let mut named = None;
    for (left_out, bit) in rows {
        if left_out {
            if !bit {
                return None;
            }
        } else if *named.get_or_insert(bit) != bit {
            return None;
        }
    }
    named
}

#[test]
fn a_verdict_is_the_named_ranks_own_and_a_rank_left_out_carries_the_bit() {
    assert_eq!(one_verdict([(false, true), (false, true)]), Some(true));
    assert_eq!(one_verdict([(false, false), (true, true)]), Some(false));
    assert_eq!(
        one_verdict([(false, true), (false, false)]),
        None,
        "two verdicts"
    );
    assert_eq!(one_verdict([(false, false), (true, false)]), None, "no bit");
    assert_eq!(one_verdict([(true, true)]), None, "no rank named");
    assert_eq!(one_verdict([]), None, "no row");
}

/// This crate's verdict after each step of the row, or why the row has
/// none: the validator's words for a form this crate does not build, or
/// the step after which the rank rows gave no one verdict ([`one_verdict`]).
fn replay(row: &Row) -> Result<Vec<bool>, String> {
    let bed = Bed::new(row)?;
    let start = Instant::now();
    row.steps
        .iter()
        .enumerate()
        .map(|(at, step)| {
            // A step a report period: the recorder repeats the step that
            // leaves a rank unheard answer after answer, and no record is
            // newer than the poll after it.
            let now = start + PERIOD * (at as u32 + 1);
            bed.step(step, now);
            bed.verdict(now)
                .ok_or_else(|| format!("after step {at} the rank rows give no one verdict"))
        })
        .collect()
}

/// `None` when a row's verdicts here are the recorded ones, or its entry's
/// when [`APART`] names the row; the reason otherwise. A row decided apart
/// that replays as recorded is a defect of its entry, whose rule decides
/// nothing of it.
fn held(name: &str, recorded: &[bool], here: &[bool]) -> Option<String> {
    match APART.iter().find(|apart| apart.row == name) {
        None => (here != recorded).then(|| format!("recorded {recorded:?}, replayed {here:?}")),
        Some(apart) if here == recorded => Some(format!(
            "it replays as recorded now ({}): take the row out of APART",
            apart.rule
        )),
        Some(apart) => (here != apart.verdicts).then(|| {
            format!(
                "recorded {recorded:?}, apart by its entry {:?}, replayed {here:?}",
                apart.verdicts
            )
        }),
    }
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
        .filter_map(|row| {
            let why = match replay(row) {
                Ok(here) => held(&row.name, &row.outcome.verdicts, &here),
                Err(why) => Some(format!("refused here: {why}")),
            };
            why.map(|why| format!("{source}: {}: {why}", row.name))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// What separates the rows copied into this module from the sheet they were
/// copied from: a copied row the sheet does not hold, or one that is not
/// the sheet's row of its name, whole. Nothing in an overload row is drawn,
/// so nothing may differ.
pub fn copied_row_defects(inline: &str, sheet: &str, source: &str) -> Vec<String> {
    let rows = |text: &str, source: &str| {
        let rows = serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|mut sheet| sheet.get_mut("rows").map(Value::take));
        match rows {
            Some(Value::Array(rows)) => Ok(rows),
            _ => Err(format!("{source} is not a sheet of rows")),
        }
    };
    let (copied, held) = match (rows(inline, "the inline rows"), rows(sheet, source)) {
        (Ok(copied), Ok(held)) => (copied, held),
        (Err(why), _) | (_, Err(why)) => return vec![why],
    };
    copied
        .iter()
        .filter_map(|row| {
            let name = row.get("name").and_then(Value::as_str).unwrap_or("?");
            let Some(sheet_row) = held.iter().find(|held| held.get("name") == row.get("name"))
            else {
                return Some(format!("{source}: the inline row {name} is on no sheet"));
            };
            (sheet_row != row)
                .then(|| format!("{source}: the inline row {name} is not the sheet's row"))
        })
        .collect()
}

/// Every row decided apart is an inline row, so every run of the suite
/// checks that its rule still decides it, and how. An entry passes on the
/// verdicts it holds alone: the recorded ones say its rule decides nothing
/// of the row, and any other outcome is a defect.
#[test]
fn every_row_decided_apart_is_an_inline_row_held_to_its_entry() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    for apart in APART {
        let row = sheet.rows.iter().find(|row| row.name == apart.row);
        let recorded = row
            .expect("every row decided apart is an inline row")
            .outcome
            .verdicts
            .as_slice();
        assert_ne!(recorded, apart.verdicts, "{}", apart.row);
        assert_eq!(held(apart.row, recorded, apart.verdicts), None);
        let as_recorded = held(apart.row, recorded, recorded);
        assert!(
            as_recorded.is_some_and(|why| why.contains("take the row out of APART")),
            "{}",
            apart.row
        );
        let third = held(apart.row, recorded, &[]);
        assert!(
            third.is_some_and(|why| why.contains("apart by its entry")),
            "{}",
            apart.row
        );
    }
}

/// A copied row is its sheet's row whole: the copy against itself is no
/// defect, and a row the sheet lacks, another threshold and another verdict
/// are one defect each.
#[test]
fn a_copied_overload_row_is_its_sheets_row_whole() {
    assert_eq!(
        copied_row_defects(INLINE, INLINE, "sheet"),
        Vec::<String>::new()
    );
    let defect_of = |change: &dyn Fn(&mut Vec<Value>)| {
        let mut sheet: Value = serde_json::from_str(INLINE).expect("the inline sheet");
        change(
            sheet
                .get_mut("rows")
                .and_then(Value::as_array_mut)
                .expect("its rows"),
        );
        let mut defects = copied_row_defects(INLINE, &sheet.to_string(), "sheet");
        assert_eq!(defects.len(), 1, "{defects:#?}");
        defects.remove(0)
    };
    assert!(defect_of(&|rows| drop(rows.remove(0))).contains("is on no sheet"));
    assert!(defect_of(&|rows| {
        let threshold = rows
            .first_mut()
            .and_then(|row| row.pointer_mut("/gateway/waiting_requests"));
        *threshold.expect("the first row's threshold") = Value::from(9);
    })
    .contains("is not the sheet's row"));
    assert!(defect_of(&|rows| {
        let verdict = rows
            .first_mut()
            .and_then(|row| row.pointer_mut("/outcome/verdicts/0"));
        *verdict.expect("the first row's first verdict") = Value::from(true);
    })
    .contains("is not the sheet's row"));
}

/// Rows copied from the recorded sheet: each kind of step, and every row
/// this crate decides apart.
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
      "name": "two-ranks-that-trade-their-queues-in-one-answer-move-no-verdict",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 6, "usage": 0.125}, {"waiting": 1, "usage": 0.125}]},
        {"poll": [{"waiting": 1, "usage": 0.125}, {"waiting": 6, "usage": 0.125}]},
        {"poll": [{"waiting": 6, "usage": 0.125}, {"waiting": 1, "usage": 0.125}]}
      ],
      "outcome": {"verdicts": [false, false, false]}
    },
    {
      "name": "a-poll-that-brought-no-report-clears-the-veto",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 1,
      "steps": [
        {"poll": [{"waiting": 9, "usage": 0.125}]},
        "no_report",
        "no_report",
        {"poll": [{"waiting": 9, "usage": 0.125}]},
        {"poll": [{"waiting": 1, "usage": 0.125}]}
      ],
      "outcome": {"verdicts": [true, false, false, true, false]}
    },
    {
      "name": "a-worker-s-own-count-stands-over-the-gateway-s-and-the-usage-stays-the-gateway-s",
      "gateway": {"waiting_requests": 8, "token_usage": 0.5},
      "worker": {"waiting_requests": 3, "token_usage": null},
      "ranks": 1,
      "steps": [
        {"poll": [{"waiting": 2, "usage": 0.0}]},
        {"poll": [{"waiting": 3, "usage": 0.0}]},
        {"poll": [{"waiting": 0, "usage": 0.25}]},
        {"poll": [{"waiting": 0, "usage": 0.5}]}
      ],
      "outcome": {"verdicts": [false, true, false, true]}
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
      "name": "the-worker-s-loads-dropped-clears-the-veto",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 1,
      "steps": [
        {"poll": [{"waiting": 9, "usage": 0.125}]},
        "evict",
        {"poll": [{"waiting": 9, "usage": 0.125}]}
      ],
      "outcome": {"verdicts": [true, false, true]}
    },
    {
      "name": "a-usage-under-the-threshold-by-less-than-half-a-permille-is-under-it",
      "gateway": {"waiting_requests": 8, "token_usage": 0.5},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 1,
      "steps": [
        {"poll": [{"waiting": 0, "usage": 0.4996}]},
        {"poll": [{"waiting": 0, "usage": 0.4994}]}
      ],
      "outcome": {"verdicts": [false, false]}
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
      "name": "a-record-after-a-poll-is-judged-over-the-ranks-that-have-pushed",
      "gateway": {"waiting_requests": 8, "token_usage": 0.8},
      "worker": {"waiting_requests": null, "token_usage": null},
      "ranks": 2,
      "steps": [
        {"poll": [{"waiting": 4, "usage": 0.125}, {"waiting": 3, "usage": 0.125}]},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 0, "figures": {"waiting": 5, "usage": 0.125}}},
        {"record": {"rank": 1, "figures": {"waiting": 0, "usage": 0.125}}}
      ],
      "outcome": {"verdicts": [false, false, false, false, false, false, false]}
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
}"#;
