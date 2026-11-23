//! The classic bucket rows: a pool of prefill workers the policy lists, its
//! configuration, requests of given sizes on a clock and the worker each one
//! was placed on.
//!
//! A request's size is what the classic policy read: its token count, else
//! the characters of its text, else zero. This side hands that number to the
//! decision as the prompt's tokens and books it on the worker the decision
//! named, as an issued decision's lease does. A row's clock is the host's:
//! a request at `at_ms` is decided and booked that many milliseconds after
//! the row's start, so the scorer's window and its cut run on the times the
//! recorder ran the classic policy's window and adjustment on.
//!
//! A row's workers are numbered from zero in the order they were listed; a
//! worker that joins takes the next number, and one that leaves is reported
//! to the engine as removed and leaves the view. An answer is a worker's
//! number. A row over a pool of regular workers is one nobody listed: the
//! classic policy draws for it, and this side decides it under the single
//! phase, where the scorer places nothing and the picker draws.

use serde::Deserialize;
use smg_policy_controller::{
    document::PolicySpec,
    engine::{
        Candidate, Constraints, Engine, Health, Phase, PoolFacts, RequestFacts, Scratch, WorkerId,
    },
    stages::Lifecycle,
    tracking::{Lease, Membership},
};

use crate::{Outcome, Recorded, Replayed};

/// The classic policy these rows record, as a sheet's head names it, and
/// the preset that replays it.
pub const POLICY: &str = "bucket";

/// How many times a row of steps is decided, each from a fresh engine under
/// its own seeds: a row no draw decides gives one sequence every time.
const RUNS: u32 = 32;

/// The host's clock at a row's start, in microseconds: past zero, so no
/// instant of a row reads as no clock.
const START_MICROS: u64 = 1_000_000_000;

/// The policy's configuration, in the gateway's own spelling.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Config {
    balance_abs_threshold: u32,
    balance_rel_threshold: f32,
    bucket_adjust_interval_secs: u32,
}

/// The defaults of the gateway's flags, which the preset carries.
const FLAGS: Config = Config {
    balance_abs_threshold: 64,
    balance_rel_threshold: 1.5,
    bucket_adjust_interval_secs: 5,
};

impl Config {
    /// The preset as named for a row recorded under the flags' defaults, so
    /// the preset's own numbers are held to those rows; for any other row,
    /// the preset with the row's numbers in its bucket stage.
    fn spec(self) -> PolicySpec {
        let preset = PolicySpec::preset(POLICY).expect("the preset's form");
        if self == FLAGS {
            return preset;
        }
        let mut document = serde_json::to_value(preset).expect("the preset serializes");
        document["profiles"][POLICY]["score"][0] = serde_json::json!({
            "bucket": {
                "abs_threshold": self.balance_abs_threshold,
                "rel_threshold": self.balance_rel_threshold,
                "adjust_interval_s": self.bucket_adjust_interval_secs,
            }
        });
        serde_json::from_value(document).expect("the preset with the row's numbers")
    }
}

/// One request: the row's clock at its placement and what it is sized by.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    at_ms: u64,
    #[serde(default)]
    tokens: Option<u32>,
    #[serde(default)]
    text: Option<u32>,
    /// The workers an earlier attempt of the request already tried.
    #[serde(default)]
    tried: Vec<usize>,
}

impl Request {
    /// The size the classic policy read: the token count, else the text's
    /// characters, else zero.
    fn size(&self) -> u32 {
        self.tokens.or(self.text).unwrap_or(0)
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request.
    Pick(Request),
    /// Whether each worker is ready, from here on.
    Ready(Vec<bool>),
    /// A prefill worker registers, under the next number.
    Join,
    /// The worker of this number leaves.
    Leave(usize),
}

/// Which pool a row's workers are: prefill workers the classic policy
/// listed, or regular workers nobody listed.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Pool {
    Prefill,
    Regular,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    name: String,
    config: Config,
    pool: Pool,
    /// How many workers the row starts with, numbered from zero.
    workers: usize,
    steps: Vec<Step>,
    outcome: Outcome,
}

/// A fresh engine over the row's view.
struct Bed {
    engine: Engine,
    phase: Phase,
    /// The workers that have not left, in their numbers' order, and the
    /// number of each row.
    view: Vec<Candidate>,
    numbers: Vec<usize>,
    /// Whether each worker ever numbered is ready.
    ready: Vec<bool>,
    scratch: Scratch,
    /// The seed of the next decision: each has its own.
    seed: u64,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's form. `run` separates the seeds of a row's runs.
    fn new(row: &Row, run: u32) -> Result<Self, String> {
        let engine = row
            .config
            .spec()
            .validate()
            .map_err(|err| err.to_string())?;
        let mut bed = Self {
            engine,
            phase: match row.pool {
                Pool::Prefill => Phase::Prefill,
                Pool::Regular => Phase::Single,
            },
            view: Vec::new(),
            numbers: Vec::new(),
            ready: Vec::new(),
            scratch: Scratch::new(),
            seed: u64::from(run) * 1_000_000,
        };
        for _ in 0..row.workers {
            bed.join();
        }
        Ok(bed)
    }

    /// The view's rows stand at their own positions, each as ready as its
    /// worker.
    fn settle(&mut self) {
        for (position, (row, &number)) in self.view.iter_mut().zip(&self.numbers).enumerate() {
            row.position = position as u32;
            row.health.healthy = self.ready[number];
        }
    }

    fn join(&mut self) {
        let number = self.ready.len();
        self.view.push(Candidate {
            id: WorkerId(1_000 + number as u64),
            key: 1_000 + number as u32,
            health: Health {
                healthy: true,
                ..Health::default()
            },
            ..Candidate::default()
        });
        self.numbers.push(number);
        self.ready.push(true);
        self.settle();
    }

    fn leave(&mut self, number: usize) {
        let at = self
            .numbers
            .iter()
            .position(|&listed| listed == number)
            .expect("a worker of the view");
        let gone = self.view.remove(at);
        self.numbers.remove(at);
        self.settle();
        self.engine.on_membership(&Membership::Removed(gone.id));
    }

    fn set_ready(&mut self, ready: &[bool]) {
        assert_eq!(ready.len(), self.ready.len(), "a word on every worker");
        self.ready.copy_from_slice(ready);
        self.settle();
    }

    /// One request decided and booked: the number of its first target.
    fn place(&mut self, request: &Request) -> usize {
        self.seed += 1;
        let now_micros = START_MICROS + request.at_ms * 1_000;
        let tried: Vec<WorkerId> = request
            .tried
            .iter()
            .map(|&number| WorkerId(1_000 + number as u64))
            .collect();
        let facts = RequestFacts {
            model: "m",
            phase: self.phase,
            prompt_tokens: request.size(),
            tried: &tried,
            seed: self.seed,
            now_micros,
            ..RequestFacts::default()
        };
        let pool = PoolFacts {
            pool_size: self.view.len() as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let decision = self.engine.decide(
            &facts,
            &pool,
            &self.view,
            &Constraints::default(),
            &mut self.scratch,
        );
        let lease = Lease {
            tokens: request.size(),
            issued_micros: now_micros,
            ..Lease::default()
        };
        self.engine.on_decision_issued(&lease, &decision);
        let target = *decision
            .pick()
            .expect("a pick")
            .targets
            .first()
            .expect("a target");
        self.numbers[target.position as usize]
    }
}

impl Row {
    /// The row's steps decided once from a fresh engine: the worker each
    /// request went to.
    fn run(&self, run: u32) -> Result<Vec<usize>, String> {
        let mut bed = Bed::new(self, run)?;
        let mut answers = Vec::new();
        for step in &self.steps {
            match step {
                Step::Pick(request) => answers.push(bed.place(request)),
                Step::Ready(ready) => bed.set_ready(ready),
                Step::Join => bed.join(),
                Step::Leave(number) => bed.leave(*number),
            }
        }
        Ok(answers)
    }

    /// How many workers the row ever numbers.
    fn numbered(&self) -> usize {
        self.workers
            + self
                .steps
                .iter()
                .filter(|step| matches!(step, Step::Join))
                .count()
    }

    /// How often each worker took the row's last request over `draws` runs,
    /// each from a fresh engine.
    fn counted(&self, draws: u32) -> Result<Outcome, String> {
        let mut counts = vec![0u32; self.numbered()];
        for run in 0..draws {
            if let Some(&last) = self.run(run)?.last() {
                counts[last] += 1;
            }
        }
        Ok(Outcome::Counts { draws, counts })
    }

    /// This crate's outcome for the row in the shape of the recorded one: a
    /// drawn row's counts over as many draws as were recorded, or the one
    /// sequence every run of a row of steps gives. Where a draw decides
    /// such a row, the runs differ, and the outcome is how often each
    /// worker took the row's last request, a shape no recorded sequence
    /// has.
    fn decided(&self) -> Result<Outcome, String> {
        if let Outcome::Counts { draws, .. } = self.outcome {
            return self.counted(draws);
        }
        let first = self.run(0)?;
        for run in 1..RUNS {
            if self.run(run)? != first {
                return self.counted(RUNS);
            }
        }
        Ok(Outcome::Sequence(first))
    }
}

impl Recorded for Row {
    const POLICY: &'static str = POLICY;

    fn name(&self) -> &str {
        &self.name
    }

    fn outcome(&self) -> &Outcome {
        &self.outcome
    }

    fn replay(&self) -> Replayed {
        match self.decided() {
            Ok(outcome) => Replayed::Ran(outcome),
            Err(why) => Replayed::Refused(why),
        }
    }
}

/// Rows copied from the recorded sheet, one or more of each form.
pub const INLINE: &str = r#"{
  "recorded_from": "d92647556e91a3601ef647f12ae5784df2080dc3",
  "policy": "bucket",
  "rows": [
    {
      "name": "a-size-goes-to-the-owner-of-its-range-in-an-even-split-of-four",
      "config": {"balance_abs_threshold": 1000000000, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 4,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 0}},
        {"pick": {"at_ms": 0, "tokens": 1023}},
        {"pick": {"at_ms": 0, "tokens": 1024}},
        {"pick": {"at_ms": 0, "tokens": 2047}},
        {"pick": {"at_ms": 0, "tokens": 2048}},
        {"pick": {"at_ms": 0, "tokens": 3071}},
        {"pick": {"at_ms": 0, "tokens": 3072}},
        {"pick": {"at_ms": 0, "tokens": 4096}},
        {"pick": {"at_ms": 0, "tokens": 100000}}
      ],
      "outcome": {"sequence": [0, 0, 1, 1, 2, 2, 3, 3, 3]}
    },
    {
      "name": "a-request-is-sized-by-its-tokens-then-by-its-text-then-at-zero",
      "config": {"balance_abs_threshold": 1000000000, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 4,
      "steps": [
        {"pick": {"at_ms": 0, "text": 1023}},
        {"pick": {"at_ms": 0, "text": 1024}},
        {"pick": {"at_ms": 0, "text": 3072}},
        {"pick": {"at_ms": 0, "tokens": 100, "text": 3500}},
        {"pick": {"at_ms": 0}}
      ],
      "outcome": {"sequence": [0, 1, 3, 0, 0]}
    },
    {
      "name": "an-owner-that-is-not-ready-hands-its-size-to-the-worker-below-then-above",
      "config": {"balance_abs_threshold": 1000000000, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 4,
      "steps": [
        {"ready": [true, false, true, true]},
        {"pick": {"at_ms": 0, "tokens": 1500}},
        {"ready": [false, false, true, true]},
        {"pick": {"at_ms": 0, "tokens": 1500}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"ready": [true, true, true, false]},
        {"pick": {"at_ms": 0, "tokens": 3500}},
        {"ready": [false, false, false, true]},
        {"pick": {"at_ms": 0, "tokens": 100}}
      ],
      "outcome": {"sequence": [0, 2, 2, 2, 3]}
    },
    {
      "name": "the-balance-rule-spills-a-size-to-the-least-booked-worker",
      "config": {"balance_abs_threshold": 64, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 3,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}}
      ],
      "outcome": {"sequence": [0, 1, 2, 0, 1, 2, 0]}
    },
    {
      "name": "a-difference-at-the-absolute-threshold-does-not-spill",
      "config": {"balance_abs_threshold": 64, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 2,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 64}},
        {"pick": {"at_ms": 0, "tokens": 0}},
        {"pick": {"at_ms": 0, "tokens": 1}},
        {"pick": {"at_ms": 0, "tokens": 0}}
      ],
      "outcome": {"sequence": [0, 0, 0, 1]}
    },
    {
      "name": "the-relative-part-holds-a-spill-back-under-a-ratio-of-1-5",
      "config": {"balance_abs_threshold": 64, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 2,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 2048}},
        {"pick": {"at_ms": 0, "tokens": 2048}},
        {"pick": {"at_ms": 0, "tokens": 3000}},
        {"pick": {"at_ms": 0, "tokens": 2100}},
        {"pick": {"at_ms": 0, "tokens": 2100}},
        {"pick": {"at_ms": 0, "tokens": 2100}}
      ],
      "outcome": {"sequence": [1, 0, 1, 0, 1, 0]}
    },
    {
      "name": "the-first-request-after-a-quiet-interval-is-weighed-by-the-bookings-it-ends",
      "config": {"balance_abs_threshold": 64, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 2,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 1000, "tokens": 100}},
        {"pick": {"at_ms": 2000, "tokens": 100}},
        {"pick": {"at_ms": 12000, "tokens": 100}},
        {"pick": {"at_ms": 12100, "tokens": 100}}
      ],
      "outcome": {"sequence": [0, 1, 0, 1, 0]}
    },
    {
      "name": "an-interval-s-end-cuts-the-ranges-again-by-the-sizes-seen",
      "config": {"balance_abs_threshold": 1000000000, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 3,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 100}},
        {"pick": {"at_ms": 0, "tokens": 400}},
        {"pick": {"at_ms": 0, "tokens": 400}},
        {"pick": {"at_ms": 0, "tokens": 400}},
        {"pick": {"at_ms": 0, "tokens": 400}},
        {"pick": {"at_ms": 0, "tokens": 400}},
        {"pick": {"at_ms": 0, "tokens": 1500}},
        {"pick": {"at_ms": 0, "tokens": 1500}},
        {"pick": {"at_ms": 4000, "tokens": 400}},
        {"pick": {"at_ms": 5001, "tokens": 400}},
        {"pick": {"at_ms": 5001, "tokens": 100}},
        {"pick": {"at_ms": 5001, "tokens": 401}},
        {"pick": {"at_ms": 5001, "tokens": 1500}}
      ],
      "outcome": {"sequence": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 2, 2]}
    },
    {
      "name": "fewer-sizes-than-workers-leave-a-worker-without-a-range",
      "config": {"balance_abs_threshold": 1000000000, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 3,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 200}},
        {"pick": {"at_ms": 0, "tokens": 200}},
        {"pick": {"at_ms": 0, "tokens": 200}},
        {"pick": {"at_ms": 5001, "tokens": 150}},
        {"ready": [false, true, true]},
        {"pick": {"at_ms": 5001, "tokens": 150}},
        {"ready": [true, true, false]},
        {"pick": {"at_ms": 5001, "tokens": 5000}},
        {"ready": [false, true, false]},
        {"pick": {"at_ms": 5001, "tokens": 5000}}
      ],
      "outcome": {"sequence": [0, 0, 0, 0, 2, 0, 1]}
    },
    {
      "name": "a-worker-that-leaves-hands-its-range-to-the-worker-below",
      "config": {"balance_abs_threshold": 1000000000, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "prefill",
      "workers": 3,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 1500}},
        {"leave": 1},
        {"pick": {"at_ms": 0, "tokens": 1500}},
        {"pick": {"at_ms": 0, "tokens": 2730}},
        {"pick": {"at_ms": 5001, "tokens": 1500}},
        {"pick": {"at_ms": 5001, "tokens": 1501}}
      ],
      "outcome": {"sequence": [1, 0, 2, 0, 2]}
    },
    {
      "name": "a-pool-nobody-listed-is-drawn-from",
      "config": {"balance_abs_threshold": 64, "balance_rel_threshold": 1.5, "bucket_adjust_interval_secs": 5},
      "pool": "regular",
      "workers": 4,
      "steps": [
        {"pick": {"at_ms": 0, "tokens": 1500}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [1513, 1469, 1507, 1511]}}
    }
  ]
}"#;
