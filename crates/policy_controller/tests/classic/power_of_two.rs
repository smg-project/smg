//! The classic power-of-two rows: a pool of workers, the reports a router
//! has of them and one request shape, under one of the policy's four load
//! metrics.

use serde::Deserialize;
use smg_policy_controller::{
    builtins::DEFAULT_MEAN_PREFILL_TOKENS, Candidate, Constraints, Engine, Health, Lease,
    Lifecycle, LoadRow, PolicySpec, PoolFacts, PowerOfTwoMetric, RequestFacts, Scratch, Side,
    WorkerId,
};

use crate::{verdict, Outcome};

/// The classic policy these rows record, as a sheet's head names it.
pub const POLICY: &str = "power_of_two";

/// What a worker's report says: running and waiting requests, the resident
/// context tokens when the engine counts them, the token-work of its queue,
/// the tokens per second it drains (`0`: unknown) and its KV-cache usage.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Report {
    running: u32,
    queued: u32,
    resident_tokens: Option<u32>,
    waiting_tokens: u32,
    throughput: u32,
    kv_usage_permille: u16,
}

/// One worker of a row: the requests the host holds on it and its report,
/// if there is one.
#[derive(Clone, Copy, Debug, Deserialize)]
struct WorkerState {
    in_flight: u32,
    report: Option<Report>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One decision; the picked worker then holds one more request.
    Pick,
    /// A new report of every worker, sampled after the picks before it.
    Report(Vec<Option<Report>>),
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    /// The load metric in the gateway's command-line spelling.
    metric: String,
    workers: Vec<WorkerState>,
    /// `None` is a request of unknown size.
    prompt_tokens: Option<u32>,
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Debug, Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

fn metric_of(text: &str) -> PowerOfTwoMetric {
    match text {
        "expected_wait" => PowerOfTwoMetric::ExpectedWait,
        "requests" => PowerOfTwoMetric::Requests,
        "least_requests" => PowerOfTwoMetric::LeastRequests,
        other => PowerOfTwoMetric::ResidentTokens {
            request_tokens: other
                .strip_prefix("resident_tokens:")
                .and_then(|n| n.parse().ok())
                .expect("a load metric in the command-line spelling"),
        },
    }
}

/// The report as a host writes it into the view: a report with no split of
/// its KV usage carries the usage as the running requests' share too.
fn load_row(report: Report) -> LoadRow {
    LoadRow {
        running: report.running,
        queued: report.queued,
        waiting_uncached_tokens: report.waiting_tokens,
        kv_usage_permille: report.kv_usage_permille,
        active_kv_permille: report.kv_usage_permille,
        live_rate: report.throughput,
        resident_tokens: report.resident_tokens.unwrap_or(0),
        resident_counted: report.resident_tokens.is_some(),
        ..LoadRow::default()
    }
}

/// The engine of the row's form over the row's pool in its starting state,
/// with the host's side of a dispatch: the lease, the credit and the
/// in-flight count.
struct Bed {
    engine: Engine,
    pool: Vec<Candidate>,
    scratch: Scratch,
    /// The host's clock, microseconds: a report is sampled and a decision
    /// issued each at its own instant.
    clock: u64,
}

impl Bed {
    fn new(row: &Row) -> Self {
        let engine = PolicySpec::power_of_two(metric_of(&row.metric))
            .expect("the preset's form")
            .validate()
            .expect("the form validates");
        let clock = 1_000;
        let pool = row
            .workers
            .iter()
            .enumerate()
            .map(|(i, worker)| Candidate {
                position: i as u32,
                id: WorkerId(1_000 + i as u64),
                key: 1_000 + i as u32,
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                in_flight: worker.in_flight,
                load: worker.report.map(load_row),
                load_sampled_at_micros: clock,
                ..Candidate::default()
            })
            .collect();
        Self {
            engine,
            pool,
            scratch: Scratch::new(),
            clock,
        }
    }

    /// One decision over the pool as it stands.
    fn decide(&mut self, prompt_tokens: Option<u32>, seed: u64) -> usize {
        self.clock += 1_000;
        let facts = RequestFacts {
            model: "m",
            prompt_tokens: prompt_tokens.unwrap_or(0),
            seed,
            now_micros: self.clock,
            ..RequestFacts::default()
        };
        let pool_facts = PoolFacts {
            pool_size: self.pool.len() as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let decision = self.engine.decide(
            &facts,
            &pool_facts,
            &self.pool,
            &Constraints::default(),
            &mut self.scratch,
        );
        let pick = decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position as usize
    }

    /// One decision and its dispatch: the engine is told, as a host tells
    /// it, and the picked worker holds one more request.
    fn pick(&mut self, prompt_tokens: Option<u32>, seed: u64) -> usize {
        let picked = self.decide(prompt_tokens, seed);
        let candidate = &self.pool[picked];
        let target = smg_policy_controller::Target {
            position: candidate.position,
            id: candidate.id,
            dp_rank: candidate.dp_rank,
            pair: None,
        };
        let lease = Lease {
            issued_micros: self.clock,
            // A request of unknown size is booked at the host's mean.
            tokens: prompt_tokens.unwrap_or(DEFAULT_MEAN_PREFILL_TOKENS),
            ..Lease::default()
        };
        self.engine.on_dispatched(&lease, &target, Side::Single, 0);
        self.pool[picked].in_flight += 1;
        picked
    }

    fn report(&mut self, reports: &[Option<Report>]) {
        assert_eq!(reports.len(), self.pool.len(), "a report of every worker");
        self.clock += 1_000;
        for (candidate, report) in self.pool.iter_mut().zip(reports) {
            candidate.load = report.map(load_row);
            candidate.load_sampled_at_micros = self.clock;
        }
    }
}

/// This crate's outcome for the row, in the shape of the recorded one.
fn replay(row: &Row) -> Outcome {
    match &row.outcome {
        Outcome::Sequence(_) => {
            let mut bed = Bed::new(row);
            let mut picked = Vec::new();
            for (n, step) in row.steps.iter().enumerate() {
                match step {
                    Step::Pick => picked.push(bed.pick(row.prompt_tokens, n as u64 + 1)),
                    Step::Report(reports) => bed.report(reports),
                }
            }
            Outcome::Sequence(picked)
        }
        Outcome::Counts { draws, .. } => {
            // Each draw decides from the starting state: nothing is
            // dispatched, so the engine holds no credit between draws.
            let mut bed = Bed::new(row);
            let mut counts = vec![0u32; row.workers.len()];
            for seed in 1..=u64::from(*draws) {
                counts[bed.decide(row.prompt_tokens, seed)] += 1;
            }
            Outcome::Counts {
                draws: *draws,
                counts,
            }
        }
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
            verdict(&row.outcome, &replay(row)).map(|why| format!("{source}: {}: {why}", row.name))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// Rows copied from the recorded sheet, one or more of each form.
pub const INLINE: &str = r#"{
  "recorded_from": "1b141d03b5ca9097f54f0ebc4b95d72e810a62c3",
  "policy": "power_of_two",
  "rows": [
    {
      "name": "resident-tokens-each-request-priced-at-100k",
      "metric": "resident_tokens:100000",
      "workers": [
        {"in_flight": 4, "report": {"running": 4, "queued": 0, "resident_tokens": 900000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 12, "report": {"running": 12, "queued": 0, "resident_tokens": 130000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 5, "report": {"running": 5, "queued": 0, "resident_tokens": 810000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 950000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 20000,
      "steps": ["pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick"],
      "outcome": {"sequence": [3, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2]}
    },
    {
      "name": "resident-tokens-a-request-of-unknown-size-is-credited-at-the-mean-prefill",
      "metric": "resident_tokens:500",
      "workers": [
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": 3000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": 3600, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": 4500, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": null,
      "steps": ["pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick"],
      "outcome": {"sequence": [0, 1, 2, 0, 1, 2, 0, 1, 2]}
    },
    {
      "name": "resident-tokens-a-new-report-releases-the-credit",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": 2000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": 4000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1500,
      "steps": ["pick", "pick", "pick", {"report": [{"running": 1, "queued": 0, "resident_tokens": 2000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}, {"running": 1, "queued": 0, "resident_tokens": 4000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}]}, "pick", "pick", "pick"],
      "outcome": {"sequence": [0, 1, 0, 0, 1, 0]}
    },
    {
      "name": "resident-tokens-requests-in-transit-are-priced-at-the-mean-context",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 5, "report": {"running": 2, "queued": 0, "resident_tokens": 8000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 3, "report": {"running": 3, "queued": 0, "resident_tokens": 21500, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": 26500, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": ["pick", "pick", "pick", "pick", "pick", "pick"],
      "outcome": {"sequence": [1, 0, 1, 0, 2, 1]}
    },
    {
      "name": "resident-tokens-a-worker-that-does-not-count-them-scores-from-its-best-peer",
      "metric": "resident_tokens:0",
      "workers": [
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 500000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 2, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": ["pick", "pick", "pick", "pick", "pick"],
      "outcome": {"sequence": [0, 0, 0, 0, 0]}
    },
    {
      "name": "resident-tokens-nobody-counts-them-the-in-flight-count-decides",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 5, "report": {"running": 5, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": null},
        {"in_flight": 4, "report": {"running": 4, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": ["pick", "pick", "pick"],
      "outcome": {"sequence": [1, 1, 1]}
    },
    {
      "name": "resident-tokens-equal-work-is-drawn",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 6000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 6000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 6001, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": [],
      "outcome": {"counts": {"draws": 6000, "counts": [2975, 3025, 0]}}
    },
    {
      "name": "resident-tokens-an-idle-worker-that-does-not-count-them-ties-with-its-best-peer",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 6000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 0, "report": {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": [],
      "outcome": {"counts": {"draws": 6000, "counts": [2994, 3006]}}
    },
    {
      "name": "resident-tokens-a-running-worker-that-reports-zero-of-them-takes-the-request",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 6000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": 0, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": [],
      "outcome": {"counts": {"draws": 6000, "counts": [0, 6000]}}
    },
    {
      "name": "resident-tokens-an-idle-worker-with-no-report-ties-with-the-best-of-two-peers",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 6000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 9000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 0, "report": null}
      ],
      "prompt_tokens": 1000,
      "steps": [],
      "outcome": {"counts": {"draws": 6000, "counts": [3029, 0, 2971]}}
    },
    {
      "name": "resident-tokens-every-worker-counts-them-and-the-worker-idle-at-its-report-holds-three-in-transit",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 2, "report": {"running": 2, "queued": 0, "resident_tokens": 6000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 3, "report": {"running": 0, "queued": 0, "resident_tokens": 0, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": ["pick", "pick", "pick", "pick"],
      "outcome": {"sequence": [1, 0, 1, 0]}
    },
    {
      "name": "resident-tokens-nobody-counts-them-and-the-worker-idle-at-its-report-holds-thirty-over-twenty-picks",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 30, "report": {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": ["pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick", "pick"],
      "outcome": {"sequence": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    },
    {
      "name": "resident-tokens-nobody-counts-them-a-peer-the-router-holds-nothing-on-beside-the-idle-report-that-holds-five",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 5, "report": {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 0, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": [],
      "outcome": {"counts": {"draws": 6000, "counts": [0, 6000]}}
    },
    {
      "name": "resident-tokens-one-worker-counts-them-and-of-the-two-that-do-not-the-idle-report-holds-six",
      "metric": "resident_tokens:1000",
      "workers": [
        {"in_flight": 40, "report": {"running": 40, "queued": 0, "resident_tokens": 900000, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 6, "report": {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 0}}
      ],
      "prompt_tokens": 1000,
      "steps": ["pick", "pick", "pick"],
      "outcome": {"sequence": [0, 0, 0]}
    },
    {
      "name": "least-requests-the-fewest-in-flight-beats-the-shorter-queue",
      "metric": "least_requests",
      "workers": [
        {"in_flight": 4, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 100, "kv_usage_permille": 100}},
        {"in_flight": 5, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 100, "kv_usage_permille": 100}},
        {"in_flight": 1, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 50000, "throughput": 100, "kv_usage_permille": 100}},
        {"in_flight": 3, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 100, "kv_usage_permille": 100}}
      ],
      "prompt_tokens": null,
      "steps": ["pick", "pick"],
      "outcome": {"sequence": [2, 2]}
    },
    {
      "name": "least-requests-equal-counts-go-to-the-shorter-queue",
      "metric": "least_requests",
      "workers": [
        {"in_flight": 2, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 50000, "throughput": 100, "kv_usage_permille": 100}},
        {"in_flight": 2, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 100, "kv_usage_permille": 100}},
        {"in_flight": 5, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 100, "kv_usage_permille": 100}}
      ],
      "prompt_tokens": null,
      "steps": ["pick"],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "requests-the-fewer-in-flight-of-two-drawn-workers",
      "metric": "requests",
      "workers": [
        {"in_flight": 0, "report": null},
        {"in_flight": 1, "report": null},
        {"in_flight": 2, "report": null},
        {"in_flight": 3, "report": null}
      ],
      "prompt_tokens": null,
      "steps": [],
      "outcome": {"counts": {"draws": 6000, "counts": [3013, 1986, 1001, 0]}}
    },
    {
      "name": "requests-fewer-in-flight-beats-the-shorter-queue",
      "metric": "requests",
      "workers": [
        {"in_flight": 5, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 100, "kv_usage_permille": 100}},
        {"in_flight": 2, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 50000, "throughput": 100, "kv_usage_permille": 100}}
      ],
      "prompt_tokens": null,
      "steps": ["pick", "pick"],
      "outcome": {"sequence": [1, 1]}
    },
    {
      "name": "expected-wait-the-better-of-two-drawn-workers",
      "metric": "expected_wait",
      "workers": [
        {"in_flight": 0, "report": null},
        {"in_flight": 1, "report": null},
        {"in_flight": 2, "report": null},
        {"in_flight": 3, "report": null}
      ],
      "prompt_tokens": null,
      "steps": [],
      "outcome": {"counts": {"draws": 6000, "counts": [2958, 2081, 961, 0]}}
    },
    {
      "name": "expected-wait-the-shorter-queue-beats-fewer-in-flight",
      "metric": "expected_wait",
      "workers": [
        {"in_flight": 5, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 100, "kv_usage_permille": 100}},
        {"in_flight": 2, "report": {"running": 1, "queued": 0, "resident_tokens": null, "waiting_tokens": 50000, "throughput": 100, "kv_usage_permille": 100}}
      ],
      "prompt_tokens": null,
      "steps": ["pick", "pick"],
      "outcome": {"sequence": [0, 0]}
    }
  ]
}"#;
