//! The classic overload rows: one worker with its data-parallel ranks, the
//! thresholds the gateway holds for every worker and the ones the worker
//! names for itself, and what the gateway's load monitor is handed, step by
//! step: a poll's answer, a poll that brought no report, a record a rank
//! pushed, the worker's loads dropped. No policy picks here. A row holds
//! whether the gateway read the worker as overloaded after each step.
//!
//! The replay hands each step to the worker crate's own load monitor, built
//! as a host of the engine over worker lists builds it (the bed of the load
//! sheets, [`Bed`]), and reads the verdict off the
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

use std::time::Instant;

use serde::Deserialize;
use smg_policy_controller::store::Overload;
use smg_worker::{
    common_proto::EngineLoad,
    health::overload::OverloadThresholds,
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    worker::{SchedulerLoadSnapshot, WorkerLoadResponse},
};

use super::load_bed::{paused, Bed, Laid, ENGINE, STEP};

/// What these rows record, as a sheet's head names it: the gateway's
/// overload verdict of one worker.
pub const POLICY: &str = "overload";

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

/// The row's worker among a host's objects, laid as `laid`: the holder
/// states the gateway's thresholds for every worker, and each of the
/// worker's rows states the worker's own. `Err` where the row states what a
/// worker's row cannot.
fn bed(row: &Row, laid: Laid) -> Result<Bed, String> {
    let gateway = OverloadThresholds {
        waiting_requests: row.gateway.waiting_requests.map(|waiting| waiting as usize),
        token_usage: row.gateway.token_usage,
    };
    Bed::new(row.ranks, laid, gateway, own(row.worker)?)
}

/// What the host's load monitor says of the row's worker after each step:
/// the verdict on each of the worker's objects. `Err` where the row states
/// what a worker's row cannot.
fn replay(row: &Row, laid: Laid) -> Result<Vec<Vec<bool>>, String> {
    paused(async {
        let bed = bed(row, laid)?;
        bed.start().await;
        let start = Instant::now();
        let mut verdicts = Vec::with_capacity(row.steps.len());
        for (at, step) in row.steps.iter().enumerate() {
            let now = start + STEP * (at as u32 + 1);
            match step {
                Step::Poll(figures) => bed.poll(Some(answer(row.ranks, figures)), now).await,
                Step::NoReport => bed.poll(None, now).await,
                Step::Record { rank, figures } => {
                    let pusher = bed.pusher(*rank).ok_or("a record of no rank's object")?;
                    let record = EngineLoad {
                        waiting_requests: figures.waiting,
                        token_usage: figures.usage,
                        active_token_usage: figures.active,
                        ..EngineLoad::default()
                    };
                    bed.monitor()
                        .apply_pushed_load(pusher, i32::from(*rank), &record, now);
                }
                Step::Evict => bed.evict().await?,
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
/// The other three answer as recorded on every rank's object.
#[test]
fn the_sheet_replays_whole_and_eleven_of_its_rank_rows_stand_apart() {
    let sheet: Sheet =
        serde_json::from_str(include_str!("sheets/overload.json")).expect("the sheet");
    assert_eq!(sheet.rows.len(), 25);
    let mut several = 0;
    let mut as_recorded = 0;
    for row in &sheet.rows {
        let whole = replay(row, Laid::Whole).expect("the worker as one object");
        assert_eq!(held_whole(row, &whole), None, "{}", row.name);
        if row.ranks > 1 {
            several += 1;
            let ranks = replay(row, Laid::Ranks).expect("an object a rank");
            assert_eq!(held_ranks(row, &ranks), None, "{}", row.name);
            as_recorded += usize::from(APART.iter().all(|apart| apart.row != row.name));
        }
    }
    assert_eq!((several, APART.len(), as_recorded), (14, 11, 3));
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
    let ranks = bed(row, Laid::Ranks).expect("the rank objects");
    assert_eq!(ranks.workers().len(), width);
    for (rank, object) in ranks.workers().iter().enumerate() {
        let built = <BasicWorkerBuilder>::new(ENGINE)
            .dp_config(rank, width)
            .build();
        assert_eq!(object.url(), built.url());
        assert_eq!(object.base_url(), built.base_url());
        assert_eq!(object.dp_rank(), built.dp_rank());
        assert_eq!(object.dp_size(), built.dp_size());
        assert_eq!(object.is_dp_aware(), built.is_dp_aware());
    }
    let whole = bed(row, Laid::Whole).expect("the one object");
    let [object] = whole.workers() else {
        panic!("{} objects of a worker laid whole", whole.workers().len());
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

/// A pushed record of a rank the worker has no object for is refused where
/// each rank is an object: the replay hands it to no other rank's object.
#[test]
fn a_record_of_a_rank_the_worker_lacks_is_refused_on_an_object_a_rank() {
    let mut sheet: serde_json::Value = serde_json::from_str(INLINE).expect("the inline rows");
    let pushes = |row: &serde_json::Value| {
        let steps = row["steps"].as_array();
        steps.is_some_and(|steps| steps.iter().any(|step| step.get("record").is_some()))
    };
    let rows = sheet["rows"].as_array_mut().expect("the rows");
    let row = rows
        .iter_mut()
        .find(|row| pushes(row))
        .expect("a row that pushes");
    let width = row["ranks"].clone();
    for step in row["steps"].as_array_mut().expect("its steps") {
        if let Some(record) = step.get_mut("record") {
            record["rank"] = width.clone();
        }
    }
    let row: Row = serde_json::from_value(row.take()).expect("a row");
    assert_eq!(
        replay(&row, Laid::Ranks),
        Err("a record of no rank's object".to_string())
    );
    assert!(replay(&row, Laid::Whole).is_ok());
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
