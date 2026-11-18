//! The classic minimum-tokens rank rows: one worker with its data-parallel
//! ranks, what the gateway's rank cache is handed step by step (a poll's
//! answer, a poll that failed, the worker's loads dropped) and the requests
//! that asked the policy for a rank. A row holds the rank each request
//! took, `null` where the gateway found none. Its replay writes each step
//! through the store's load sink, as this crate's driver writes it, decides
//! each request over the rows of the worker's ranks under the resident-work
//! scorer picked by `best`, and books the request's tokens at its dispatch.
//!
//! This crate decides a rank by rules of its own, and [`APART`] names each
//! recorded row they decide apart from the gateway, with the ranks here.

use std::time::{Duration, Instant};

use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use serde::Deserialize;
use smg_policy_controller::{
    ingest::rows,
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, ViewBuffers, WorkerFacts, WorkerStatus,
        WorkersStore,
    },
    AppliedSpec, Constraints, Engine, Lease, Lifecycle, LoadRow, PolicySpec, PoolFacts,
    PowerOfTwoMetric, RequestFacts, Role, Scratch, Side, SignalSource, WorkerId,
};

/// What these rows record, as a sheet's head names it: the rank the
/// gateway's minimum-tokens policy gives a request on one worker.
pub const POLICY: &str = "minimum_tokens";

const MODEL: &str = "m";

/// The report period the replay tells the store, as this crate's driver
/// tells its own, and the time between two writes of a row's loads. A
/// request follows the step before it by [`SOON`].
const PERIOD: Duration = Duration::from_secs(10);
const SOON: Duration = Duration::from_millis(1);

/// The worker's engine: the key of a worker of one rank, and what the rows
/// of a worker's ranks share.
const ENGINE: &str = "http://w0:8000";

/// One rank's row in a poll's answer: the tokens its cache uses, its token
/// capacity, and the context tokens of the requests it holds where the rank
/// reports them.
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

/// What a request takes here in a row decided apart: one rank, or a draw
/// among the worker's ranks, which names any of them.
#[derive(Clone, Copy, Debug)]
enum Here {
    Rank(u16),
    Drawn,
}

/// A recorded row this crate decides apart from the gateway, by a rule of
/// its own that `rule` states: its replay must give `here`, request by
/// request, and its first `each_once` requests must take as many different
/// ranks. An entry that names no draw fails the day its row replays as
/// recorded, so it cannot outlive its rule; every entry fails on any other
/// outcome.
struct Apart {
    row: &'static str,
    here: &'static [Here],
    each_once: usize,
    rule: &'static str,
}

/// The rows decided apart. The gateway's rank cache reads the tokens a
/// rank's cache uses, from a row that carries token capacity, forgets a
/// worker's ranks at an answer with no entry, and takes the lowest of equal
/// ranks. The store reads the context tokens of the requests a rank holds
/// and keeps what is on file through an answer with no row. It hands a
/// decision every rank of the worker: a rank with nothing on file scores
/// beside its peers, and equal ranks are drawn. A rank an answer leaves out
/// beside a rank it names is avoided on both sides: those rows replay.
const APART: &[Apart] = &[
    Apart {
        row: "equal-ranks-go-to-the-lowest-rank",
        here: &[Here::Drawn, Here::Drawn, Here::Drawn, Here::Drawn],
        each_once: 3,
        rule: "equal ranks are the picker's draw, where the gateway takes the lowest",
    },
    Apart {
        row: "a-row-with-no-token-capacity-is-no-rank",
        here: &[Here::Rank(0), Here::Rank(0)],
        each_once: 0,
        rule: "a rank's row is read whatever token capacity it carries",
    },
    Apart {
        row: "an-answer-with-no-token-capacity-drops-the-worker-s-ranks",
        here: &[Here::Rank(0), Here::Rank(0), Here::Rank(0)],
        each_once: 0,
        rule: "a rank's row is read whatever token capacity it carries",
    },
    Apart {
        row: "an-answer-with-no-row-drops-the-worker-s-ranks",
        here: &[Here::Rank(0), Here::Rank(0)],
        each_once: 0,
        rule: "an answer with no row is the word that no report came, and nothing on file changes",
    },
    Apart {
        row: "a-worker-never-polled-has-no-rank",
        here: &[Here::Drawn, Here::Drawn],
        each_once: 2,
        rule: "a rank with nothing on file is a rank, and the requests in flight part equal ranks",
    },
    Apart {
        row: "an-evicted-worker-has-no-rank-until-its-next-answer",
        here: &[Here::Rank(0), Here::Drawn, Here::Rank(0)],
        each_once: 0,
        rule: "a rank with nothing on file is a rank, and the requests in flight part equal ranks",
    },
    Apart {
        row: "a-worker-of-one-rank-the-answer-omits-has-none",
        here: &[Here::Rank(1)],
        each_once: 0,
        rule: "a worker of one row answers with its row",
    },
    Apart {
        row: "an-answer-with-no-row-drops-the-one-rank-of-its-worker",
        here: &[Here::Rank(1), Here::Rank(1)],
        each_once: 0,
        rule: "a worker of one row answers with its row",
    },
    Apart {
        row: "the-used-tokens-decide-where-a-rank-holds-fewer-context-tokens",
        here: &[Here::Rank(1), Here::Rank(1)],
        each_once: 0,
        rule: "a rank is scored by the context tokens of its requests, not by the tokens its cache uses",
    },
    Apart {
        row: "a-rank-that-reports-no-context-tokens-is-a-rank-by-its-used-tokens",
        here: &[Here::Drawn, Here::Drawn, Here::Drawn],
        each_once: 2,
        rule: "where no rank counts its context tokens the requests in flight decide",
    },
];

/// The generation that decides a rank: the resident-work scorer over every
/// row, a request priced at no tokens of its own, picked by `best`.
fn document() -> PolicySpec {
    let mut document =
        PolicySpec::power_of_two(PowerOfTwoMetric::ResidentTokens { request_tokens: 0 })
            .expect("a preset form of this build");
    document.generation = 1;
    document
}

/// One rank's figures as the row a host writes for the rank: through the
/// crate's own row builder, from the report of that rank alone, read at the
/// instant it was sampled.
fn load_row(figures: Figures, at: Instant) -> LoadRow {
    let count = |n: u32| i32::try_from(n).expect("a recorded count fits the wire's field");
    let wire = WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            num_used_tokens: count(figures.used),
            max_total_num_tokens: count(figures.capacity),
            num_resident_tokens: figures.resident.map(i64::from),
            ..SchedulerLoadSnapshot::default()
        }],
        ..WorkerLoadResponse::default()
    };
    rows::load_row(&wire, Role::Single, at, at, SignalSource::Poll)
}

/// The store with the row's worker in it and the engine that decides over
/// its rows: a member row a rank of one engine, or the one row of a worker
/// that stands for one rank.
struct Bed {
    store: MemoryStore,
    engine: Engine,
    /// The member rows, each with its rank.
    ids: Vec<(WorkerId, u16)>,
    out: ViewBuffers,
    scratch: Scratch,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's form.
    fn new(row: &Row) -> Result<Self, String> {
        let document = document();
        let engine = document.validate().map_err(|err| err.to_string())?;
        let applied = AppliedSpec::validate(&document).map_err(|err| err.to_string())?;
        let store = MemoryStore::new();
        let ranks: Vec<u16> = match row.pinned {
            Some(rank) => vec![rank],
            None => (0..row.ranks).collect(),
        };
        let ids = ranks
            .into_iter()
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
                        ..WorkerFacts::default()
                    })
                    .id();
                let revision = WorkersStore::get(&store, id).expect("the row").revision;
                store
                    .set_status(id, WorkerStatus::Ready, revision)
                    .expect("ready");
                (id, rank)
            })
            .collect();
        store.apply(applied).expect("the first generation applies");
        // The driver tells its poll interval where it is built.
        assert!(store.heard_every(PERIOD), "the first period told");
        Ok(Self {
            store,
            engine,
            ids,
            out: ViewBuffers::default(),
            scratch: Scratch::new(),
        })
    }

    /// One step at `now`: a write of the worker's loads through the load
    /// sink, each as the crate's driver writes it, or a request, whose rank
    /// is returned. An engine's answer goes in whole in one report, a poll
    /// that failed as an answer with no row, and the eviction rank by rank.
    fn step(&mut self, step: &Step, now: Instant, seed: u64) -> Option<Option<u16>> {
        match step {
            Step::Poll(figures) => {
                let answer: Vec<_> = self
                    .ids
                    .iter()
                    .map(|&(id, rank)| {
                        let figures = figures.get(usize::from(rank)).copied().flatten();
                        (id, rank, figures.map(|figures| load_row(figures, now)))
                    })
                    .collect();
                self.store.report(&answer, now, LoadSource::Poll);
            }
            Step::Failed => {
                let answer: Vec<_> = self
                    .ids
                    .iter()
                    .map(|&(id, rank)| (id, rank, None))
                    .collect();
                self.store.report(&answer, now, LoadSource::Poll);
            }
            Step::Evict => {
                for &(id, _) in &self.ids {
                    self.store.evict(id);
                }
            }
            Step::Pick { cost } => return Some(self.pick(*cost, now, seed)),
        }
        if let Some(due) = self.store.publish_due() {
            self.store.publish(due);
        }
        None
    }

    /// One request decided over the rows of the worker's ranks and
    /// dispatched, as a host commits it: the rank it took, or `None` where
    /// the decision names no row.
    fn pick(&mut self, cost: u32, now: Instant, seed: u64) -> Option<u16> {
        let ask = Ask {
            gateway: GatewayId(1),
            now,
            pinned: None,
            affinity: false,
        };
        let built = self
            .store
            .build_view(MODEL, Role::Single, &ask, &mut self.out)?;
        let now_micros = self.store.micros(now);
        let facts = RequestFacts {
            model: MODEL,
            prompt_tokens: cost,
            seed,
            now_micros,
            ..RequestFacts::default()
        };
        let pool = PoolFacts {
            pool_size: built.pool_size,
            generation: 1,
            ..PoolFacts::default()
        };
        let decision = self.engine.decide(
            &facts,
            &pool,
            &self.out.candidates,
            &Constraints::default(),
            &mut self.scratch,
        );
        let target = *decision.pick()?.targets.first()?;
        let lease = Lease {
            issued_micros: now_micros,
            tokens: cost,
            ..Lease::default()
        };
        self.engine.on_dispatched(&lease, &target, Side::Single, 0);
        Some(target.dp_rank)
    }
}

/// The rank each request of the row takes here, or the validator's words
/// for a form this crate does not build. A write of the loads follows the
/// step before it by a report period and a request by a moment, so no
/// request is older than the answer after it.
fn replay(row: &Row) -> Result<Vec<Option<u16>>, String> {
    let mut bed = Bed::new(row)?;
    let mut now = Instant::now();
    let mut picks = Vec::new();
    for (at, step) in row.steps.iter().enumerate() {
        now += if matches!(step, Step::Pick { .. }) {
            SOON
        } else {
            PERIOD
        };
        if let Some(pick) = bed.step(step, now, at as u64) {
            picks.push(pick);
        }
    }
    Ok(picks)
}

/// `None` when a row's ranks here are the recorded ones, or what its entry
/// gives when [`APART`] names the row; the reason otherwise. A row decided
/// apart with no draw that replays as recorded is a defect of its entry,
/// whose rule decides nothing of it.
fn held(row: &Row, here: &[Option<u16>]) -> Option<String> {
    let recorded = row.outcome.sequence.as_slice();
    let Some(apart) = APART.iter().find(|apart| apart.row == row.name) else {
        return (here != recorded).then(|| format!("recorded {recorded:?}, replayed {here:?}"));
    };
    let drawn = apart.here.iter().any(|pick| matches!(pick, Here::Drawn));
    if !drawn && here == recorded {
        return Some(format!(
            "it replays as recorded now ({}): take the row out of APART",
            apart.rule
        ));
    }
    let fits = here.len() == apart.here.len()
        && here
            .iter()
            .zip(apart.here)
            .all(|(pick, want)| match (pick, want) {
                (Some(rank), Here::Rank(want)) => rank == want,
                (Some(rank), Here::Drawn) => *rank < row.ranks,
                (None, _) => false,
            });
    let mut seen = Vec::with_capacity(apart.each_once);
    let once = here.iter().take(apart.each_once).all(|pick| {
        let new = !seen.contains(pick);
        seen.push(*pick);
        new
    });
    (!(fits && once)).then(|| {
        format!(
            "recorded {recorded:?}, apart by its entry {:?} with the first {} each once, replayed {here:?}",
            apart.here, apart.each_once
        )
    })
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

/// Every row decided apart is an inline row, so every run of the suite
/// checks that its rule still decides it, and how.
#[test]
fn every_rank_row_decided_apart_is_an_inline_row() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    for apart in APART {
        assert!(
            sheet.rows.iter().any(|row| row.name == apart.row),
            "{} is decided apart and is no inline row",
            apart.row
        );
    }
}

/// Rows of the sheet, copied: every row decided apart and a few that
/// replay as recorded.
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
      "name": "an-answer-with-no-token-capacity-drops-the-worker-s-ranks",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 1}},
        {"poll": [{"used": 100, "capacity": 0, "resident": 100}, {"used": 200, "capacity": 0, "resident": 200}]},
        {"pick": {"cost": 1}},
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}]},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, null, 0]}
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
      "name": "a-worker-never-polled-has-no-rank",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"pick": {"cost": 1}},
        "failed",
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [null, null]}
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
      "name": "a-worker-of-one-rank-takes-its-own-rank-alone",
      "ranks": 2,
      "pinned": 1,
      "steps": [
        {"poll": [{"used": 10, "capacity": 8192, "resident": 10}, {"used": 500, "capacity": 8192, "resident": 500}]},
        {"pick": {"cost": 1}},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [1, 1]}
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
      "name": "an-answer-with-no-row-drops-the-one-rank-of-its-worker",
      "ranks": 2,
      "pinned": 1,
      "steps": [
        {"poll": [{"used": 10, "capacity": 8192, "resident": 10}, {"used": 500, "capacity": 8192, "resident": 500}]},
        {"pick": {"cost": 1}},
        {"poll": [null, null]},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [1, null]}
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
    },
    {
      "name": "a-rank-that-reports-no-context-tokens-is-a-rank-by-its-used-tokens",
      "ranks": 2,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": null}, {"used": 200, "capacity": 8192, "resident": null}]},
        {"pick": {"cost": 1}},
        {"pick": {"cost": 200}},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, 0, 1]}
    },
    {
      "name": "a-rank-five-answers-omit-is-no-rank-after-any-of-them",
      "ranks": 3,
      "pinned": null,
      "steps": [
        {"poll": [{"used": 100, "capacity": 8192, "resident": 100}, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}},
        {"poll": [null, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}},
        {"poll": [null, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}},
        {"poll": [null, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}},
        {"poll": [null, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}},
        {"poll": [null, {"used": 200, "capacity": 8192, "resident": 200}, {"used": 300, "capacity": 8192, "resident": 300}]},
        {"pick": {"cost": 1}}
      ],
      "outcome": {"sequence": [0, 1, 1, 1, 1, 1]}
    }
  ]
}"#;
