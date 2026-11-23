//! The classic minimum-tokens rank rows: one worker with its data-parallel
//! ranks, what the gateway's rank cache is handed step by step (a poll's
//! answer, a poll that failed, the worker's loads dropped) and the requests
//! that asked the policy for a rank. A row holds the rank each request
//! took, `null` where the gateway found none.
//!
//! What this replay proves: the worker crate's own rank choice, under the
//! load monitor a host of the engine over worker lists builds (the bed of
//! the load sheets, [`Bed`]). A poll goes through the monitor's poll of the
//! worker's group, a failed poll is the engine failing the call, the
//! worker's loads are dropped by the monitor's own loop, and a request asks
//! the monitor's rank cache for its rank, which books the request's tokens
//! on it, as the gateway's policy does. The worker stands as one object for
//! all its ranks, as the recording gateway held it, or as the object of its
//! one rank where the row says it stands for one. Every row replays as
//! recorded.
//!
//! What it does not prove: the engine's choice among rank objects. The
//! engine serves no rank option. Routing settings that ask for the
//! gateway's minimum-tokens rank choice make no document
//! (`refuse_unsupported` in `src/document/routing_settings.rs`: "under the
//! controller a rank is a candidate of the profile"), and no rank is chosen
//! behind a pick. Where each rank is a worker object the engine picks among
//! them by its profile, and these rows are not its answers.

use std::time::Instant;

use serde::Deserialize;
use smg_policy_controller::store::Overload;
use smg_worker::{
    health::overload::OverloadThresholds,
    worker::{SchedulerLoadSnapshot, WorkerLoadResponse},
};

use super::load_bed::{paused, Bed, Laid, STEP};

/// What these rows record, as a sheet's head names it: the rank the
/// gateway's minimum-tokens policy gives a request on one worker.
pub const POLICY: &str = "minimum_tokens";

/// One rank's row in a poll's answer: the tokens its cache uses, which is
/// the count the rank cache reads, its token capacity, and the context
/// tokens of the requests it holds where the rank reports them.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Figures {
    used: u32,
    capacity: u32,
    resident: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// A poll's answer: a row a rank, in rank order, `None` where the
    /// answer omits the rank.
    Poll(Vec<Option<Figures>>),
    /// A poll that failed.
    Failed,
    /// The worker leaves the ready set and its loads are dropped.
    Evict,
    /// A request of `cost` estimated tokens asks for a rank.
    Pick { cost: u32 },
}

/// What a row recorded: the rank each request took, `None` where the
/// gateway found none.
#[derive(Debug, Deserialize)]
struct Outcome {
    sequence: Vec<Option<u16>>,
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    /// The worker's data-parallel ranks.
    ranks: u16,
    /// The worker's own rank where it stands for one rank of its engine.
    pinned: Option<u16>,
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Debug, Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// A poll's answer as an engine sends it: one entry for each rank it names.
fn answer(ranks: u16, figures: &[Option<Figures>]) -> WorkerLoadResponse {
    let count = |n: u32| i32::try_from(n).expect("a recorded count fits the report's field");
    WorkerLoadResponse {
        dp_rank_count: i32::from(ranks),
        loads: figures
            .iter()
            .enumerate()
            .filter_map(|(at, figures)| {
                figures.map(|figures| SchedulerLoadSnapshot {
                    dp_rank: i32::try_from(at).expect("a rank fits the report's field"),
                    num_used_tokens: count(figures.used),
                    max_total_num_tokens: count(figures.capacity),
                    num_resident_tokens: figures.resident.map(i64::from),
                    ..SchedulerLoadSnapshot::default()
                })
            })
            .collect(),
        ..WorkerLoadResponse::default()
    }
}

/// The row's worker among a host's objects: one object for all its ranks,
/// or the object of the one rank it stands for. The rank cache reads token
/// counts alone, so the holder states no threshold and the worker none of
/// its own.
fn bed(row: &Row) -> Result<Bed, String> {
    let no_veto = OverloadThresholds {
        waiting_requests: None,
        token_usage: None,
    };
    let laid = row.pinned.map_or(Laid::Whole, Laid::Rank);
    Bed::new(row.ranks, laid, no_veto, Overload::default())
}

/// The rank the monitor's rank cache gives each request of the row, `None`
/// where it finds none.
fn replay(row: &Row) -> Result<Vec<Option<u16>>, String> {
    paused(async {
        let bed = bed(row)?;
        bed.start().await;
        let [worker] = bed.workers() else {
            return Err(format!("{} objects of one worker", bed.workers().len()));
        };
        let cache = &bed.monitor().worker_load_manager;
        let start = Instant::now();
        let mut picks = Vec::new();
        for (at, step) in row.steps.iter().enumerate() {
            let now = start + STEP * (at as u32 + 1);
            match step {
                Step::Poll(figures) => bed.poll(Some(answer(row.ranks, figures)), now).await,
                Step::Failed => bed.poll(None, now).await,
                Step::Evict => bed.evict().await?,
                Step::Pick { cost } => {
                    let cost = isize::try_from(*cost).map_err(|err| err.to_string())?;
                    // The select the gateway's policy forwards a request
                    // to: it names the rank and books the cost on it.
                    let rank = cache.select_and_increment_lowest_dp_load(worker.as_ref(), cost);
                    let rank = rank.map(u16::try_from).transpose();
                    picks.push(rank.map_err(|err| err.to_string())?);
                }
            }
        }
        Ok(picks)
    })
}

/// `None` when the ranks a row's requests take here are the recorded ones;
/// the reason otherwise.
fn held(row: &Row, here: &[Option<u16>]) -> Option<String> {
    let recorded = row.outcome.sequence.as_slice();
    (here != recorded).then(|| format!("recorded {recorded:?}, replayed {here:?}"))
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
                Ok(here) => held(row, &here),
                Err(why) => Some(format!("refused here: {why}")),
            };
            why.map(|why| format!("{source}: {}: {why}", row.name))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// The sheet whole: every row as recorded, fifteen with the worker as one
/// object for all its ranks and three with the worker as one rank's object.
#[test]
fn the_sheet_replays_whole_under_both_layouts_it_records() {
    let sheet: Sheet =
        serde_json::from_str(include_str!("sheets/minimum_tokens.json")).expect("the sheet");
    let mut pinned = 0;
    for row in &sheet.rows {
        let here = replay(row).expect("a row this crate builds");
        assert_eq!(held(row, &here), None, "{}", row.name);
        pinned += usize::from(row.pinned.is_some());
    }
    assert_eq!((sheet.rows.len(), pinned), (18, 3));
}

/// A row is held to the rank each of its requests took: a copied row whose
/// first request recorded another answer is a defect by that request.
#[test]
fn a_row_that_recorded_another_rank_is_a_defect() {
    let mut sheet: serde_json::Value = serde_json::from_str(INLINE).expect("the inline rows");
    let row = &mut sheet["rows"][0];
    let taken = row["outcome"]["sequence"][0].take();
    assert!(taken.is_u64(), "the first inline row's request took a rank");
    let row: Row = serde_json::from_value(row.take()).expect("a row");
    let here = replay(&row).expect("a row this crate builds");
    let why = held(&row, &here).expect("a request recorded with no rank");
    assert!(why.starts_with("recorded [None]"), "{why}");
}

/// A worker that stands for one rank of its engine is laid as that rank's
/// object, and one that stands for every rank states the width and no rank.
#[test]
fn a_worker_of_one_rank_is_laid_as_that_ranks_object() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    let mut seen = [false; 2];
    for row in &sheet.rows {
        let bed = bed(row).expect("a row this crate builds");
        let [worker] = bed.workers() else {
            panic!("{} objects of one worker", bed.workers().len());
        };
        let width = Some(usize::from(row.ranks));
        assert_eq!(
            (worker.dp_rank(), worker.dp_size()),
            (row.pinned.map(usize::from), width),
            "{}",
            row.name
        );
        seen[usize::from(row.pinned.is_some())] = true;
    }
    assert_eq!(seen, [true; 2], "an inline row of each layout");
}

/// A few of the sheet's rows, copied whole: every run of the suite replays
/// them.
pub const INLINE: &str = r#"{
  "recorded_from": "615542c4c69bb28fb3cb3e9e4ee29a8e2be0a7bc",
  "policy": "minimum_tokens",
  "rows": [
    {
      "name": "the-rank-with-the-fewest-used-tokens-takes-the-request",
      "ranks": 3,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 300, "capacity": 8192, "resident": 300}, {"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 10}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "a-pick-books-its-cost-on-its-rank",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 150, "capacity": 8192, "resident": 150}]},
        {"pick": {"cost": 60}},
        {"pick": {"cost": 60}},
        {"pick": {"cost": 60}},
        {"pick": {"cost": 60}}
      ],
      "outcome": {"sequence": [0, 1, 0, 1]}
    },
    {
      "name": "equal-ranks-go-to-the-lowest-rank",
      "ranks": 3,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 100, "capacity": 8192, "resident": 100}, {"used": 100, "capacity": 8192, "resident": 100}]},
        {"pick": {"cost": 10}},
        {"pick": {"cost": 10}},
        {"pick": {"cost": 10}},
        {"pick": {"cost": 10}}
      ],
      "outcome": {"sequence": [0, 1, 2, 0]}
    },
    {
      "name": "a-taken-answer-replaces-the-ranks-and-their-bookings",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 500}},
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 500}},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, 0, 1]}
    },
    {
      "name": "a-rank-an-answer-omits-is-no-rank-at-once",
      "ranks": 3,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}},
        {"poll": [null, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, 1, 1]}
    },
    {
      "name": "a-row-with-no-token-capacity-is-no-rank",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 10, "capacity": 0, "resident": 10}, {"used": 500, "capacity": 8192, "resident": 500}]},
        {"pick": {"cost": 1}},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [1, 1]}
    },
    {
      "name": "an-answer-with-no-row-drops-the-worker-s-ranks",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 1}},
        {"poll": [null, null]},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, null]}
    },
    {
      "name": "a-failed-poll-keeps-the-last-answer-and-its-bookings",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 150}},
        "failed",
        {"pick": {"cost": 1}},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, 1, 1]}
    },
    {
      "name": "an-evicted-worker-has-no-rank-until-its-next-answer",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 1}},
        "evict",
        {"pick": {"cost": 1}},
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, null, 0]}
    },
    {
      "name": "a-worker-of-one-rank-the-answer-omits-has-none",
      "ranks": 2,
      "pinned": 1,
      "steps": [
        {"poll": [{"used": 10, "capacity": 8192, "resident": 10}, null]},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [null]}
    },
    {
      "name": "the-used-tokens-decide-where-a-rank-holds-fewer-context-tokens",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 900}, {"used": 200, "capacity": 8192, "resident": 50}]},
        {"pick": {"cost": 1}},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, 0]}
    }
  ]
}
"#;
