//! The classic cache-aware rows in single placement: the policy's
//! configuration, the matcher the row's requests reach (the tree of text,
//! the tree of token ids, or the index of the blocks the workers report), a
//! pool of workers, what a router holds of each one and a list of steps.
//!
//! A prompt is runs of token ids, `[first, count]` each: the ids `first`,
//! `first + 1` and so on. Under the text matcher it is carried as text,
//! four characters an id. Every prompt and every shared prefix of the rows
//! is whole blocks.
//!
//! What a worker holds of a request reaches the decision as the affinity
//! column a host lends beside its rows. The matchers read apart, and the
//! row's matcher says which reading the replay hands the engine:
//!
//! - the index: every worker holds its own depth, by the blocks the row's
//!   `stored` and `removed` steps wrote for it;
//! - a tree: the classic's tree names the holders of the deepest match over
//!   the whole pool and nobody else. The replay gives those holders their
//!   depth as a predicted placement and every other worker nothing, from
//!   the row's own history: each request wrote its prompt for the worker it
//!   was placed on.
//!
//! A row under the index is handed that column by the replay's own count
//! of the blocks each worker stored.
//!
//! The profile is stated here from the row's configuration ([`Row::spec`]).
//! Every worker of a row is as young as the row, and no index holds a
//! thousand blocks, so no row reaches the warm-up slice or the diversion.

use std::time::Instant;

use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use serde::Deserialize;
use smg_policy_controller::{
    document::PolicySpec,
    engine::{
        AffinityFacts, Candidate, Constraints, Decision, Engine, HashKind, Health, LoadRow,
        PoolFacts, RequestFacts, Role, Scratch, ShedReason, SignalSource, Target, WorkerId,
    },
    stages::Lifecycle,
    tracking::{Lease, Side, TransportClass},
};

use crate::{Outcome, Recorded, Replayed};

/// The classic policy these rows record, as a sheet's head names it.
pub const POLICY: &str = "cache_aware";

const MODEL: &str = "m";

/// Characters of a text an id: a prompt carried as text spells each id as
/// four hex digits.
const CHARS_PER_ID: u32 = 4;

/// What a request's prompt is matched by.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Matcher {
    /// The tree of text: the request carries its prompt as text.
    Text,
    /// The tree of token ids.
    Tokens,
    /// The index of the blocks the workers report.
    Events,
}

/// The classic policy's configuration, each field in the spelling of the
/// gateway's configuration file.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Config {
    cache_threshold: f32,
    balance_abs_threshold: u32,
    balance_rel_threshold: f32,
    block_size: u32,
    balance_token_usage_threshold: f32,
    overload_token_usage_threshold: f32,
}

/// What the router holds of a worker besides its requests and its report.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Not ready.
    Down,
}

impl State {
    fn health(self) -> Health {
        Health {
            healthy: self != Self::Down,
            overloaded: self == Self::Over,
            ..Health::default()
        }
    }
}

/// What a worker's report says: running and waiting requests, the resident
/// context tokens when the worker counts them, the token-work of its queue,
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

/// One worker of a row: the requests the host holds on it, its state and
/// its report, if there is one.
#[derive(Clone, Copy, Debug, Deserialize)]
struct WorkerState {
    in_flight: u32,
    state: State,
    report: Option<Report>,
}

/// A prompt: runs of token ids, `[first, count]` each.
type Prompt = Vec<[u32; 2]>;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its prompt and the workers an earlier attempt of it
    /// already tried. The picked worker then holds one more request.
    Pick { prompt: Prompt, tried: Vec<usize> },
    /// The request of an earlier `Pick` ends, named by its number among the
    /// row's picks: its worker holds one fewer and the engine is told.
    Complete(usize),
    /// The requests the host holds on each worker, from here on.
    Loads(Vec<u32>),
    /// Each worker's state, from here on.
    States(Vec<State>),
    /// A new report of every worker, sampled after the picks before it.
    Reports(Vec<Option<Report>>),
    /// A worker reports that it stored the blocks of a prompt.
    Stored { worker: usize, prompt: Prompt },
    /// A worker reports that it removed the blocks of a prompt past the
    /// first `kept_blocks`.
    Removed {
        worker: usize,
        prompt: Prompt,
        kept_blocks: usize,
    },
}

#[derive(Debug, Deserialize)]
pub struct Row {
    name: String,
    matcher: Matcher,
    config: Config,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
    outcome: Outcome,
}

/// The gateway's command line defaults. The crate's preset states the same
/// numbers by stating none, so a row recorded under them asks the preset as
/// it is named.
const NAMED: Config = Config {
    cache_threshold: 0.3,
    balance_abs_threshold: 64,
    balance_rel_threshold: 1.5,
    block_size: 16,
    balance_token_usage_threshold: 1.0,
    overload_token_usage_threshold: 1.0,
};

/// Whether two numbers of a configuration are the same number as written.
fn same(one: f32, other: f32) -> bool {
    one.to_bits() == other.to_bits()
}

impl Row {
    /// The row's profile: the crate's `cache_aware` preset, with each
    /// number of the classic's configuration set on the preset's own stage
    /// where it differs from the gateway's command line default. The
    /// filters, the stages and their order, their weights and the pick are
    /// the preset's, so a change of the preset reaches every row, and a row
    /// recorded under those defaults asks the preset as it is named. A tree
    /// is a predictor of placements: a tree row names that source, and an
    /// index row names none.
    fn spec(&self) -> Result<PolicySpec, String> {
        let config = &self.config;
        let preset = PolicySpec::preset(POLICY).map_err(|err| err.to_string())?;
        let mut document = serde_json::to_value(&preset).map_err(|err| err.to_string())?;
        if config.block_size != NAMED.block_size {
            document["view"]["block_size"] = config.block_size.into();
        }
        let spill = config.balance_abs_threshold != NAMED.balance_abs_threshold
            || !same(config.balance_rel_threshold, NAMED.balance_rel_threshold);
        let gates = !same(
            config.balance_token_usage_threshold,
            NAMED.balance_token_usage_threshold,
        ) || !same(
            config.overload_token_usage_threshold,
            NAMED.overload_token_usage_threshold,
        );
        let stages = document["profiles"][POLICY]["score"]
            .as_array_mut()
            .ok_or("the preset states its score stages as a list")?;
        let (mut gated, mut pressed) = (false, false);
        for stage in stages {
            if let Some(affinity) = stage.get_mut("prefix-affinity") {
                gated = true;
                if !same(config.cache_threshold, NAMED.cache_threshold) {
                    affinity["threshold"] = config.cache_threshold.into();
                }
                if self.matcher != Matcher::Events {
                    affinity["source"] = "predictor".into();
                }
                if spill {
                    affinity["spill"] = serde_json::json!({
                        "abs": config.balance_abs_threshold,
                        "rel": config.balance_rel_threshold
                    });
                }
            }
            // A stage at its defaults is stated by its bare name.
            if stage.as_str() == Some("kv-pressure") || stage.get("kv-pressure").is_some() {
                pressed = true;
                if gates {
                    *stage = serde_json::json!({ "kv-pressure": {
                        "spread": config.balance_token_usage_threshold,
                        "ceiling": config.overload_token_usage_threshold
                    } });
                }
            }
        }
        if !(gated && pressed) {
            return Err("the preset holds no prefix-affinity stage or no kv-pressure stage".into());
        }
        serde_json::from_value(document).map_err(|err| err.to_string())
    }

    /// Token ids a block: the block size under the tree of token ids and
    /// the index, a quarter of it under the tree of text, whose blocks are
    /// characters.
    fn ids_per_block(&self) -> usize {
        let block = self.config.block_size as usize;
        match self.matcher {
            Matcher::Text => block / CHARS_PER_ID as usize,
            Matcher::Tokens | Matcher::Events => block,
        }
    }
}

/// The token ids of a prompt.
fn ids(prompt: &[[u32; 2]]) -> Vec<u32> {
    prompt
        .iter()
        .flat_map(|&[first, count]| first..first + count)
        .collect()
}

/// The report as an engine of one rank sends it.
fn wire(report: Report) -> WorkerLoadResponse {
    let count = |n: u32| i32::try_from(n).expect("a recorded count fits the wire's field");
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            num_running_reqs: count(report.running),
            num_waiting_reqs: count(report.queued),
            num_waiting_uncached_tokens: count(report.waiting_tokens),
            token_usage: f64::from(report.kv_usage_permille) / 1000.0,
            gen_throughput: f64::from(report.throughput),
            num_resident_tokens: report.resident_tokens.map(i64::from),
            ..SchedulerLoadSnapshot::default()
        }],
        ..WorkerLoadResponse::default()
    }
}

/// The report as a host writes it into the view: through the crate's own
/// row builder, as the polled report of a worker outside a pair, read at
/// the instant it was sampled.
fn load_row(report: Report) -> LoadRow {
    let now = Instant::now();
    smg_policy_controller::load::load_row(&wire(report), Role::Single, now, now, SignalSource::Poll)
}

/// The worker the row lists at `i`, as the view names it.
fn id(i: usize) -> WorkerId {
    WorkerId(1_000 + i as u64)
}

/// The engine of the row's profile over the row's pool in its starting
/// state, with the host's side of a dispatch and what each worker holds.
struct Bed {
    engine: Engine,
    pool: Vec<Candidate>,
    scratch: Scratch,
    /// The host's clock, microseconds: a report is sampled and a decision
    /// issued each at its own instant.
    clock: u64,
    matcher: Matcher,
    block_size: u32,
    ids_per_block: usize,
    /// What each worker holds, as token ids. Under a tree: the prompts of
    /// the requests placed on it. Under the index: every prefix of whole
    /// blocks it stored and did not remove.
    held: Vec<Vec<Vec<u32>>>,
    /// The lease, the target and the worker of each request so far, or
    /// nothing where the request was not placed: a row numbers its
    /// requests by every pick, as its recorder does.
    picked: Vec<Option<(Lease, Target, usize)>>,
    /// What stops the row's run, in its own words: the end of a request
    /// that no worker holds.
    parted: Option<String>,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's profile.
    fn new(row: &Row, spec: &PolicySpec) -> Result<Self, String> {
        let engine = spec.validate().map_err(|err| err.to_string())?;
        let clock = 1_000;
        let pool = row
            .workers
            .iter()
            .enumerate()
            .map(|(place, worker)| Candidate {
                position: place as u32,
                id: id(place),
                key: 1_000 + place as u32,
                health: worker.state.health(),
                in_flight: worker.in_flight,
                load: worker.report.map(load_row),
                load_sampled_at_micros: clock,
                ..Candidate::default()
            })
            .collect();
        Ok(Self {
            engine,
            pool,
            scratch: Scratch::new(),
            clock,
            matcher: row.matcher,
            block_size: row.config.block_size,
            ids_per_block: row.ids_per_block(),
            held: vec![Vec::new(); row.workers.len()],
            picked: Vec::new(),
            parted: None,
        })
    }

    /// The affinity column of a request for `prompt`: the replay's own
    /// count, under the row's matcher.
    fn column(&self, prompt: &[u32]) -> Vec<AffinityFacts> {
        let per = self.ids_per_block;
        match self.matcher {
            // Every worker by its own depth: the leading blocks of the
            // prompt it holds, each as the prefix that ends at it.
            Matcher::Events => self
                .held
                .iter()
                .map(|held| {
                    let overlap = (1..=prompt.len() / per)
                        .take_while(|&blocks| held.iter().any(|h| h[..] == prompt[..blocks * per]))
                        .count();
                    AffinityFacts {
                        overlap_blocks: overlap as u32,
                        indexed_growth: held.len() as u32,
                        indexed_blocks: held.len() as u32,
                        ..AffinityFacts::default()
                    }
                })
                .collect(),
            // The holders of the deepest match alone, by the prompts
            // placed on each worker.
            Matcher::Text | Matcher::Tokens => {
                let depths: Vec<u32> = self
                    .held
                    .iter()
                    .map(|placed| {
                        placed
                            .iter()
                            .map(|other| shared_blocks(prompt, other, per))
                            .max()
                            .unwrap_or(0)
                    })
                    .collect();
                let deepest = depths.iter().copied().max().unwrap_or(0);
                depths
                    .iter()
                    .map(|&depth| AffinityFacts {
                        predicted_blocks: if depth == deepest { depth } else { 0 },
                        ..AffinityFacts::default()
                    })
                    .collect()
            }
        }
    }

    /// One request decided and, when it is placed, dispatched as a host
    /// commits it. The answer is the worker's place in the row, or a number
    /// past the last worker for a request placed nowhere: the count of the
    /// workers for a shed as saturated, the count plus one for any other.
    fn pick(&mut self, prompt: &[[u32; 2]], tried: &[usize], seed: u64) -> usize {
        self.clock += 1_000;
        let prompt = ids(prompt);
        let column = self.column(&prompt);
        let tried: Vec<WorkerId> = tried.iter().map(|&i| id(i)).collect();
        let text = self.matcher == Matcher::Text;
        let facts = RequestFacts {
            model: MODEL,
            // A text of four characters an id is an estimate of one token
            // an id.
            prompt_tokens: prompt.len() as u32,
            hash_kind: if text {
                HashKind::Text
            } else {
                HashKind::Token
            },
            block_size: self.block_size,
            attempt: u8::from(!tried.is_empty()),
            tried: &tried,
            seed,
            now_micros: self.clock,
            affinity: &column,
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
        let pick = match &decision {
            Decision::Pick(pick) => pick,
            Decision::Shed(_) | Decision::Queue(_) => {
                self.picked.push(None);
                let saturated = matches!(&decision, Decision::Shed(shed) if shed.reason == ShedReason::Saturated);
                return self.pool.len() + usize::from(!saturated);
            }
        };
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            issued_micros: self.clock,
            // A request carried as text states no size: the load stage
            // credits its dispatch at its own mean.
            tokens: if text { 0 } else { prompt.len() as u32 },
            ..Lease::default()
        };
        self.engine.on_dispatched(&lease, &target, Side::Single, 0);
        let place = target.position as usize;
        self.pool[place].in_flight += 1;
        self.picked.push(Some((lease, target, place)));
        // A tree is written at every request, for the worker it chose.
        if self.matcher != Matcher::Events {
            self.held[place].push(prompt);
        }
        place
    }

    /// The request of pick `n` ends: its worker holds one fewer, and the
    /// engine is told. A row that ends a request no worker holds, one
    /// placed nowhere or one not picked yet, stops in those words.
    fn complete(&mut self, n: usize) {
        let Some((lease, target, place)) = self.picked.get(n).copied().flatten() else {
            self.parted.get_or_insert_with(|| {
                format!("the row ends the request of pick {n}, which no worker holds")
            });
            return;
        };
        self.pool[place].in_flight -= 1;
        let mut ended = smg_policy_controller::tracking::Outcome::reported(200, TransportClass::Ok)
            .with_served(Side::Single, target.id);
        ended.tokens_in = lease.tokens;
        self.engine.on_completed(&lease, &ended);
    }

    /// One step of the row; a request answers where it was placed.
    fn step(&mut self, step: &Step, seed: u64) -> Option<usize> {
        let per = self.ids_per_block;
        match step {
            Step::Pick { prompt, tried } => return Some(self.pick(prompt, tried, seed)),
            Step::Complete(n) => self.complete(*n),
            Step::Loads(loads) => {
                assert_eq!(loads.len(), self.pool.len(), "a load of every worker");
                for (candidate, &load) in self.pool.iter_mut().zip(loads) {
                    candidate.in_flight = load;
                }
            }
            Step::States(states) => {
                assert_eq!(states.len(), self.pool.len(), "a state of every worker");
                for (candidate, state) in self.pool.iter_mut().zip(states) {
                    candidate.health = state.health();
                }
            }
            Step::Reports(reports) => {
                assert_eq!(reports.len(), self.pool.len(), "a report of every worker");
                self.clock += 1_000;
                for (candidate, report) in self.pool.iter_mut().zip(reports) {
                    candidate.load = report.map(load_row);
                    candidate.load_sampled_at_micros = self.clock;
                }
            }
            Step::Stored { worker, prompt } => {
                let prompt = ids(prompt);
                let held = &mut self.held[*worker];
                for blocks in 1..=prompt.len() / per {
                    let prefix = &prompt[..blocks * per];
                    if !held.iter().any(|h| h[..] == *prefix) {
                        held.push(prefix.to_vec());
                    }
                }
            }
            Step::Removed {
                worker,
                prompt,
                kept_blocks,
            } => {
                let prompt = ids(prompt);
                self.held[*worker]
                    .retain(|h| h.len() <= kept_blocks * per || !prompt.starts_with(h));
            }
        }
        None
    }
}

/// The leading whole blocks two prompts share.
fn shared_blocks(a: &[u32], b: &[u32], ids_per_block: usize) -> u32 {
    let shared = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    (shared / ids_per_block) as u32
}

/// The row run once from a fresh engine: where each request was placed, or
/// the validator's words, or what stopped the run: the end of a request
/// that no worker holds. `draw` separates the seeds of a drawn row's
/// runs; a sequence row is run `0`. `spec` is the row's profile, read once
/// a row.
fn run(row: &Row, spec: &PolicySpec, draw: u64) -> Result<Vec<usize>, String> {
    let mut bed = Bed::new(row, spec)?;
    let picked = row
        .steps
        .iter()
        .enumerate()
        .filter_map(|(n, step)| bed.step(step, draw * 1_000 + n as u64 + 1))
        .collect();
    match bed.parted.take() {
        Some(why) => Err(why),
        None => Ok(picked),
    }
}

/// This crate's outcome for the row, in the shape of the recorded one, or
/// its refusal of the row's profile.
fn replay(row: &Row) -> Replayed {
    let spec = match row.spec() {
        Ok(spec) => spec,
        Err(why) => return Replayed::Refused(why),
    };
    match &row.outcome {
        Outcome::Sequence(_) => match run(row, &spec, 0) {
            Ok(picked) => Replayed::Ran(Outcome::Sequence(picked)),
            Err(why) => Replayed::Refused(why),
        },
        Outcome::Counts { draws, .. } => {
            // Each draw runs the row from a fresh engine, as the recorder
            // ran it from a fresh policy. The row's last request is the one
            // counted. A drawn row places every request: a draw that places
            // one nowhere answers as its sequence, which no count equals.
            let mut counts = vec![0u32; row.workers.len()];
            for draw in 1..=u64::from(*draws) {
                match run(row, &spec, draw) {
                    Ok(picked) => {
                        if picked.iter().any(|&place| place >= counts.len()) {
                            return Replayed::Ran(Outcome::Sequence(picked));
                        }
                        let last = *picked.last().expect("a drawn row ends with a request");
                        counts[last] += 1;
                    }
                    Err(why) => return Replayed::Refused(why),
                }
            }
            Replayed::Ran(Outcome::Counts {
                draws: *draws,
                counts,
            })
        }
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
        replay(self)
    }
}

/// Rows copied from the recorded sheet: under the tree of token ids a holder
/// kept, the count gate with its spill, a match at the threshold, the deepest
/// holder tried and a KV gate; under the index the deepest holder and equal
/// holders, the deepest holder tried, a removed block and a drawn pair of
/// holders; under the tree of text a match at the threshold.
pub const INLINE: &str = r#"{
  "recorded_from": "bfe189b332349c44cf7e3b6e1102a9b5a2699d9d",
  "policy": "cache_aware",
  "rows": [
    {
      "name": "tokens-a-holder-keeps-its-prefix-over-an-idle-worker",
      "matcher": "tokens",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null},
        {"in_flight": 2, "state": "free", "report": null}
      ],
      "steps": [
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"loads": [1, 3, 0]},
        {"pick": {"prompt": [[1000, 48], [5000, 16]], "tried": []}}
      ],
      "outcome": {"sequence": [1, 1]}
    },
    {
      "name": "tokens-a-holder-over-the-count-gate-spills-and-the-prefix-gains-a-holder",
      "matcher": "tokens",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null},
        {"in_flight": 2, "state": "free", "report": null}
      ],
      "steps": [
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"loads": [1, 9, 0]},
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"loads": [0, 2, 5]},
        {"pick": {"prompt": [[1000, 64]], "tried": []}}
      ],
      "outcome": {"sequence": [1, 2, 2, 1]}
    },
    {
      "name": "tokens-a-match-at-the-threshold-is-a-miss",
      "matcher": "tokens",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null},
        {"in_flight": 2, "state": "free", "report": null}
      ],
      "steps": [
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"loads": [0, 3, 1]},
        {"pick": {"prompt": [[1000, 32], [6000, 32]], "tried": []}}
      ],
      "outcome": {"sequence": [1, 0]}
    },
    {
      "name": "tokens-the-deepest-holder-tried-is-a-miss-beside-a-shallower-holder",
      "matcher": "tokens",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 0, "state": "free", "report": null},
        {"in_flight": 3, "state": "free", "report": null},
        {"in_flight": 3, "state": "free", "report": null}
      ],
      "steps": [
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"loads": [3, 0, 3]},
        {"pick": {"prompt": [[1000, 48], [5000, 16]], "tried": [0]}},
        {"loads": [1, 2, 0]},
        {"pick": {"prompt": [[1000, 64]], "tried": [0]}}
      ],
      "outcome": {"sequence": [0, 1, 2]}
    },
    {
      "name": "tokens-under-the-kv-spread-gate-a-hot-holder-yields-to-the-expected-wait",
      "matcher": "tokens",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 0.3, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 0, "state": "free", "report": {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 100}},
        {"in_flight": 1, "state": "free", "report": {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 100}},
        {"in_flight": 2, "state": "free", "report": {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 100}}
      ],
      "steps": [
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"reports": [{"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 900}, {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 100}, {"running": 0, "queued": 0, "resident_tokens": null, "waiting_tokens": 0, "throughput": 0, "kv_usage_permille": 100}]},
        {"pick": {"prompt": [[1000, 64]], "tried": []}}
      ],
      "outcome": {"sequence": [0, 1]}
    },
    {
      "name": "events-the-deepest-holder-keeps-the-request-and-equal-holders-go-by-load",
      "matcher": "events",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 3, "state": "free", "report": null},
        {"in_flight": 1, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null}
      ],
      "steps": [
        {"stored": {"worker": 0, "prompt": [[1000, 64]]}},
        {"stored": {"worker": 1, "prompt": [[1000, 32]]}},
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"pick": {"prompt": [[1000, 32], [6000, 32]], "tried": []}}
      ],
      "outcome": {"sequence": [0, 1]}
    },
    {
      "name": "events-the-deepest-holder-tried-a-shallower-holder-takes-the-request",
      "matcher": "events",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 1, "state": "free", "report": null},
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null}
      ],
      "steps": [
        {"stored": {"worker": 0, "prompt": [[1000, 64]]}},
        {"stored": {"worker": 1, "prompt": [[1000, 48]]}},
        {"pick": {"prompt": [[1000, 64]], "tried": [0]}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "events-a-removed-block-shortens-the-hold-and-the-last-one-ends-it",
      "matcher": "events",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 1, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null}
      ],
      "steps": [
        {"stored": {"worker": 0, "prompt": [[1000, 64]]}},
        {"stored": {"worker": 1, "prompt": [[1000, 32]]}},
        {"stored": {"worker": 2, "prompt": [[7000, 16]]}},
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"removed": {"worker": 0, "prompt": [[1000, 64]], "kept_blocks": 1}},
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"removed": {"worker": 1, "prompt": [[1000, 32]], "kept_blocks": 0}},
        {"pick": {"prompt": [[1000, 64]], "tried": []}},
        {"removed": {"worker": 0, "prompt": [[1000, 64]], "kept_blocks": 0}},
        {"pick": {"prompt": [[1000, 64]], "tried": []}}
      ],
      "outcome": {"sequence": [0, 1, 0, 2]}
    },
    {
      "name": "events-equal-holders-at-equal-loads-are-drawn",
      "matcher": "events",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null}
      ],
      "steps": [
        {"stored": {"worker": 0, "prompt": [[1000, 64]]}},
        {"stored": {"worker": 1, "prompt": [[1000, 64]]}},
        {"pick": {"prompt": [[1000, 64]], "tried": []}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [3013, 2987, 0]}}
    },
    {
      "name": "text-a-match-at-the-threshold-is-a-miss",
      "matcher": "text",
      "config": {"cache_threshold": 0.5, "balance_abs_threshold": 32, "balance_rel_threshold": 1.1, "block_size": 16, "balance_token_usage_threshold": 1.0, "overload_token_usage_threshold": 1.0},
      "workers": [
        {"in_flight": 2, "state": "free", "report": null},
        {"in_flight": 0, "state": "free", "report": null},
        {"in_flight": 2, "state": "free", "report": null}
      ],
      "steps": [
        {"pick": {"prompt": [[100, 16]], "tried": []}},
        {"loads": [0, 3, 1]},
        {"pick": {"prompt": [[100, 8], [9000, 8]], "tried": []}}
      ],
      "outcome": {"sequence": [1, 0]}
    }
  ]
}"#;

/// A row recorded under the gateway's command line defaults sets no number
/// on the preset: an index row asks it as it is named, and a tree row names
/// its source alone. The sheet holds such rows under both readings, so the
/// preset's own numbers are held to the gateway's.
#[test]
fn a_row_at_the_command_lines_defaults_asks_the_preset_as_it_is_named() {
    let sheet: serde_json::Value =
        serde_json::from_str(include_str!("sheets/cache_aware.json")).expect("the sheet parses");
    let rows: Vec<Row> = serde_json::from_value(sheet["rows"].clone()).expect("the sheet's rows");
    let preset = PolicySpec::preset(POLICY).expect("a preset of this build");
    let named = serde_json::to_value(&preset).expect("the preset is a document");
    let (mut index, mut tree) = (0, 0);
    for row in &rows {
        let config = &row.config;
        let at_defaults = same(config.cache_threshold, NAMED.cache_threshold)
            && config.balance_abs_threshold == NAMED.balance_abs_threshold
            && same(config.balance_rel_threshold, NAMED.balance_rel_threshold)
            && config.block_size == NAMED.block_size
            && same(
                config.balance_token_usage_threshold,
                NAMED.balance_token_usage_threshold,
            )
            && same(
                config.overload_token_usage_threshold,
                NAMED.overload_token_usage_threshold,
            );
        if !at_defaults {
            continue;
        }
        let spec = row.spec().expect("a row's profile");
        if row.matcher == Matcher::Events {
            assert_eq!(spec, preset, "{}", row.name);
            index += 1;
        } else {
            let mut sourced = named.clone();
            for stage in sourced["profiles"][POLICY]["score"]
                .as_array_mut()
                .expect("the preset's score list")
            {
                if let Some(affinity) = stage.get_mut("prefix-affinity") {
                    affinity["source"] = "predictor".into();
                }
            }
            let stated = serde_json::to_value(&spec).expect("a profile is a document");
            assert_eq!(stated, sourced, "{}", row.name);
            tree += 1;
        }
    }
    assert!(
        index >= 2 && tree >= 1,
        "{index} index rows and {tree} tree rows at the command line's defaults"
    );
}

/// A row that parts from what its recorder holds fails by its name, not by
/// a panic of the walk. The end of a request that no worker holds stops the
/// row in those words. A draw that places a request nowhere answers as its
/// sequence, though its last request is placed.
#[test]
fn a_request_placed_nowhere_fails_its_row() {
    let row = |steps: serde_json::Value, outcome: serde_json::Value| -> Row {
        let down = serde_json::json!({"in_flight": 0, "state": "down", "report": null});
        serde_json::from_value(serde_json::json!({
            "name": "a-request-placed-nowhere",
            "matcher": "tokens",
            "config": {
                "cache_threshold": 0.3,
                "balance_abs_threshold": 64,
                "balance_rel_threshold": 1.5,
                "block_size": 16,
                "balance_token_usage_threshold": 1.0,
                "overload_token_usage_threshold": 1.0
            },
            "workers": [down, down, down],
            "steps": steps,
            "outcome": outcome
        }))
        .expect("a row of the sheet's shape")
    };
    let pick = serde_json::json!({"pick": {"prompt": [[1000, 32]], "tried": []}});
    let refused = |row: &Row| match replay(row) {
        Replayed::Refused(why) => why,
        Replayed::Ran(outcome) => panic!("the row ran: {outcome:?}"),
    };

    // No worker is ready, so the pick places nothing, and the row ends it.
    let ended = row(
        serde_json::json!([pick, {"complete": 0}]),
        serde_json::json!({"sequence": [4]}),
    );
    assert!(refused(&ended).contains("pick 0"), "{}", refused(&ended));
    // A row that ends a request ahead of its pick stops the same way.
    let early = row(
        serde_json::json!([{"complete": 1}, pick]),
        serde_json::json!({"sequence": [4]}),
    );
    assert!(refused(&early).contains("pick 1"), "{}", refused(&early));

    // The first pick places nothing and the last, with every worker back,
    // is placed: no draw of the row is counted.
    let drawn = row(
        serde_json::json!([pick, {"states": ["free", "free", "free"]}, pick]),
        serde_json::json!({"counts": {"draws": 6, "counts": [2, 2, 2]}}),
    );
    match replay(&drawn) {
        Replayed::Ran(Outcome::Sequence(picked)) => {
            assert!(picked[0] >= 3 && picked[1] < 3, "{picked:?}");
        }
        Replayed::Ran(outcome) => panic!("the draws were counted: {outcome:?}"),
        Replayed::Refused(why) => panic!("{why}"),
    }
}
