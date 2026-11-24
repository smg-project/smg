//! The recorder of the gateway's cache-aware policy in single placement:
//! each row is the policy's configuration, the matcher its requests reach
//! (the tree of text, the tree of token ids, or the index of the blocks the
//! workers report), a pool of workers, what the router holds of each one
//! (the requests in flight, its state and its load report), a list of steps
//! and the worker the gateway's single-worker placement answered at each
//! request. The policy controller crate replays the rows through its own
//! decision (`crates/policy_controller/tests/classic_rows.rs`).
//!
//! A prompt is a list of runs of token ids, `[first, count]` each: the ids
//! `first`, `first + 1` and so on. Under the text matcher a prompt is
//! carried as text, each id spelled as four hex digits. Every prompt and
//! every shared prefix of the rows is whole blocks: sixteen token ids, or
//! sixteen characters of a text.
//!
//! A request's answer is the worker it was placed on, by its place in the
//! row's list. A request that is placed nowhere is recorded as a number past
//! the last worker: the count of the workers when the gateway sheds it as an
//! overload shed, the count plus one when it finds no worker to use.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the sheet
//! under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the commit it
//! recorded from. The sheets the policy controller crate replays are in
//! `crates/policy_controller/tests/classic/sheets`; to record this one again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib policies::classic_cache_aware_rows -- --ignored --nocapture
//! ```
//!
//! A sequence row is recorded twenty times from a fresh pool and a fresh
//! policy and must come out the same each time; a drawn row records how
//! often each worker took the row's last request over its draws, each draw
//! from a fresh pool and a fresh policy, and every request of a draw is
//! placed.
//!
//! The tree matchers write at every request: the worker a request was
//! placed on holds its prompt from then on. The index is written by the
//! row's `stored` and `removed` steps alone, as the workers' own reports
//! write it. A row under the index stores a block before its first request:
//! over an empty index the policy reads its tree of token ids.

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc, thread, time::Duration};

use kv_index::{
    compute_content_hash, request_prefix_hashes, ContentHash, SequenceHash, StoredBlock,
};
use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse, WorkerStatus};
use serde::Serialize;
use smg_worker::{identity::basic::Worker, load::state::TestFeed, test_support::ready_worker};

use super::{
    kv_index::{
        event_monitor::KvEventMonitor,
        index_backend::{KvIndex, WorkerBlocks},
    },
    normalize_model_key, PolicyRegistry,
};
use crate::{
    config::types::PolicyConfig,
    routers::common::{
        placement::{self, PlacementFailure, PlacementInputs},
        snapshot::{RoutingPool, WorkerRegistry},
    },
};

const MODEL: &str = "m";

/// What a request's prompt is matched by.
#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Matcher {
    /// The tree of text: the request carries its prompt as text.
    Text,
    /// The tree of token ids: the request carries token ids and no worker
    /// reports its blocks.
    Tokens,
    /// The index of the blocks the workers report.
    Events,
}

/// The policy's configuration, each field in the spelling of the gateway's
/// configuration file.
#[derive(Clone, Copy, Serialize)]
struct Config {
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    block_size: usize,
    balance_token_usage_threshold: f32,
    overload_token_usage_threshold: f32,
}

/// What the router holds of a worker besides its requests and its report.
#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Not ready.
    Down,
}

/// What a worker's report says: running and waiting requests, the resident
/// context tokens when the worker counts them, the token-work of its queue,
/// the tokens per second it drains (`0`: unknown) and its KV-cache usage.
#[derive(Clone, Copy, Serialize)]
struct Report {
    running: u32,
    queued: u32,
    resident_tokens: Option<u32>,
    waiting_tokens: u32,
    throughput: u32,
    kv_usage_permille: u16,
}

/// One worker of a row: the requests the router holds on it, its state and
/// its report, if the policy has one.
#[derive(Clone, Copy, Serialize)]
struct WorkerState {
    in_flight: u32,
    state: State,
    report: Option<Report>,
}

/// A prompt: runs of token ids, `[first, count]` each.
type Prompt = Vec<[u32; 2]>;

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its prompt and the workers an earlier attempt of it
    /// already tried. The picked worker then holds one more request.
    Pick { prompt: Prompt, tried: Vec<usize> },
    /// The request of an earlier `Pick` ends, named by its number among the
    /// row's picks: its worker holds one fewer and the policy is told.
    Complete(usize),
    /// The requests the router holds on each worker, from here on.
    Loads(Vec<u32>),
    /// Each worker's state, from here on.
    States(Vec<State>),
    /// A new report of every worker (`None`: no report of it), sampled
    /// after the picks before it.
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

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The answer to each `Pick` step, in order.
    Sequence(Vec<usize>),
    /// How often each worker took the row's last request over `draws` runs
    /// of the row.
    Counts { draws: u32, counts: Vec<u32> },
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    matcher: Matcher,
    config: Config,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Serialize)]
struct Sheet {
    recorded_from: String,
    policy: &'static str,
    rows: Vec<Row>,
}

fn url(i: usize) -> String {
    format!("http://w{i}:8000")
}

/// The token ids of a prompt.
fn ids(prompt: &[[u32; 2]]) -> Vec<u32> {
    prompt
        .iter()
        .flat_map(|&[first, count]| first..first + count)
        .collect()
}

/// A prompt as the text matcher's requests carry it: each id spelled as
/// four hex digits.
fn text(ids: &[u32]) -> String {
    ids.iter().map(|id| format!("{id:04x}")).collect()
}

/// The whole blocks of a prompt as a worker reports them: each block's
/// content hash, named by the hash of the prefix that ends at it.
fn blocks(ids: &[u32], block_size: usize) -> Vec<StoredBlock> {
    let contents: Vec<ContentHash> = ids
        .chunks_exact(block_size)
        .map(compute_content_hash)
        .collect();
    contents
        .iter()
        .zip(request_prefix_hashes(&contents))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect()
}

fn snapshot(report: Report) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 0,
            num_running_reqs: report.running as i32,
            num_waiting_reqs: report.queued as i32,
            num_resident_tokens: report.resident_tokens.map(i64::from),
            num_waiting_uncached_tokens: report.waiting_tokens as i32,
            gen_throughput: f64::from(report.throughput),
            token_usage: f64::from(report.kv_usage_permille) / 1000.0,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn loads_of(reports: impl Iterator<Item = Option<Report>>) -> HashMap<String, WorkerLoadResponse> {
    reports
        .enumerate()
        .filter_map(|(i, report)| Some((url(i), snapshot(report?))))
        .collect()
}

/// The policy's configuration as the gateway's own configuration file
/// spells it: the row's fields, no eviction sweep and room for every prompt
/// of a row in the trees.
fn config_of(config: &Config) -> PolicyConfig {
    let mut text = serde_json::to_value(config).expect("the row's configuration");
    let fields = text.as_object_mut().expect("an object of fields");
    fields.insert("type".into(), "cache_aware".into());
    fields.insert("eviction_interval_secs".into(), 0.into());
    fields.insert("max_tree_size".into(), 1_000_000.into());
    serde_json::from_value(text).expect("the policy's configuration")
}

/// The index of a row under the index matcher, with the blocks it holds of
/// each worker, by the index's own numbers.
struct Index {
    indexer: Arc<KvIndex>,
    held: Vec<(u32, WorkerBlocks)>,
}

/// A fresh registry of ready workers under a fresh policy, in the row's
/// starting state.
struct Bed {
    registry: WorkerRegistry,
    policies: PolicyRegistry,
    workers: Vec<Arc<dyn Worker>>,
    feed: TestFeed,
    matcher: Matcher,
    block_size: usize,
    index: Option<Index>,
    /// The worker of each request so far.
    picked: Vec<usize>,
}

impl Bed {
    fn new(matcher: Matcher, config: &Config, workers: &[WorkerState]) -> Self {
        let registry = WorkerRegistry::new();
        let pool: Vec<Arc<dyn Worker>> = (0..workers.len())
            .map(|i| {
                let worker = ready_worker(&url(i), MODEL);
                registry
                    .register(Arc::clone(&worker))
                    .expect("the worker registers");
                worker
            })
            .collect();
        let policies = PolicyRegistry::new(config_of(config));
        let feed = TestFeed::attached_to(|rx| policies.set_load_receiver(rx));
        let index = (matcher == Matcher::Events).then(|| {
            let indexer = Arc::new(KvIndex::chain());
            let held = (0..workers.len())
                .map(|i| {
                    let id = indexer.intern_worker(&url(i)).expect("an index number");
                    (id, WorkerBlocks::default())
                })
                .collect();
            let monitor = Arc::new(KvEventMonitor::new());
            monitor
                .indexers
                .insert(normalize_model_key(MODEL).to_string(), Arc::clone(&indexer));
            policies.set_kv_event_monitor(Some(monitor));
            Index { indexer, held }
        });
        let bed = Self {
            registry,
            policies,
            workers: pool,
            feed,
            matcher,
            block_size: config.block_size,
            index,
            picked: Vec::new(),
        };
        bed.loads(&workers.iter().map(|w| w.in_flight).collect::<Vec<_>>());
        bed.states(&workers.iter().map(|w| w.state).collect::<Vec<_>>());
        bed.feed
            .publish(&loads_of(workers.iter().map(|w| w.report)));
        bed
    }

    fn loads(&self, loads: &[u32]) {
        assert_eq!(loads.len(), self.workers.len(), "a load of every worker");
        for (worker, &want) in self.workers.iter().zip(loads) {
            while worker.load() < want as usize {
                worker.increment_load();
            }
            while worker.load() > want as usize {
                worker.decrement_load();
            }
        }
    }

    fn states(&self, states: &[State]) {
        assert_eq!(states.len(), self.workers.len(), "a state of every worker");
        for (worker, &state) in self.workers.iter().zip(states) {
            worker.set_status(if state == State::Down {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            });
            self.registry
                .set_worker_overloaded(worker, state == State::Over);
        }
    }

    /// One request through the gateway's single-worker placement: the
    /// worker's place, or a number past the last worker for a request that
    /// is placed nowhere. The picked worker holds one more request.
    fn pick(&mut self, prompt: &[[u32; 2]], tried: &[usize]) -> usize {
        if let Some(index) = &self.index {
            assert!(
                !index.indexer.is_empty(),
                "a request under the index matcher is decided over a block stored before it"
            );
        }
        let ids = ids(prompt);
        let text = text(&ids);
        let tried: Vec<String> = tried.iter().map(|&i| url(i)).collect();
        let inputs = match self.matcher {
            Matcher::Text => PlacementInputs {
                text: Some(&text),
                tried: &tried,
                ..Default::default()
            },
            Matcher::Tokens | Matcher::Events => PlacementInputs {
                tokens: Some(&ids),
                tried: &tried,
                ..Default::default()
            },
        };
        let picked = placement::select_single(
            &self.registry,
            &self.policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            inputs,
            placement::Stated::nothing,
        );
        let place = match picked {
            Some(worker) => {
                let place = self
                    .workers
                    .iter()
                    .position(|w| w.url() == worker.url())
                    .expect("a worker of the pool");
                worker.increment_load();
                place
            }
            None => {
                let failure = placement::single_failure(
                    &self.registry,
                    MODEL,
                    RoutingPool::HttpRegular,
                    None,
                );
                match failure {
                    PlacementFailure::AllOverloaded(_) => self.workers.len(),
                    _ => self.workers.len() + 1,
                }
            }
        };
        self.picked.push(place);
        place
    }

    /// The request of pick `n` ends: its worker holds one fewer, and the
    /// policy that placed it is told. A row ends no request that was
    /// placed nowhere.
    fn complete(&self, n: usize) {
        let worker = self
            .workers
            .get(self.picked[n])
            .expect("a row ends a request it placed");
        worker.decrement_load();
        self.policies
            .policy_for_worker(worker.as_ref())
            .on_request_complete(worker.url());
    }

    fn stored(&mut self, worker: usize, prompt: &[[u32; 2]]) {
        let blocks = blocks(&ids(prompt), self.block_size);
        let index = self.index.as_mut().expect("a row under the index matcher");
        let (id, held) = &mut index.held[worker];
        index
            .indexer
            .apply_stored(*id, &blocks, None, held)
            .expect("the blocks are stored");
    }

    fn removed(&mut self, worker: usize, prompt: &[[u32; 2]], kept_blocks: usize) {
        let hashes: Vec<SequenceHash> = blocks(&ids(prompt), self.block_size)
            .iter()
            .skip(kept_blocks)
            .map(|block| block.seq_hash)
            .collect();
        let index = self.index.as_mut().expect("a row under the index matcher");
        let (id, held) = &mut index.held[worker];
        index.indexer.apply_removed(*id, &hashes, held);
    }

    /// One step; a request answers where it was placed.
    fn step(&mut self, step: &Step) -> Option<usize> {
        match step {
            Step::Pick { prompt, tried } => return Some(self.pick(prompt, tried)),
            Step::Complete(n) => self.complete(*n),
            Step::Loads(loads) => self.loads(loads),
            Step::States(states) => self.states(states),
            Step::Reports(reports) => {
                // A report is sampled after the dispatches before it.
                thread::sleep(Duration::from_millis(2));
                self.feed.publish(&loads_of(reports.iter().copied()));
            }
            Step::Stored { worker, prompt } => self.stored(*worker, prompt),
            Step::Removed {
                worker,
                prompt,
                kept_blocks,
            } => self.removed(*worker, prompt, *kept_blocks),
        }
        None
    }

    fn run(
        matcher: Matcher,
        config: &Config,
        workers: &[WorkerState],
        steps: &[Step],
    ) -> Vec<usize> {
        let mut bed = Self::new(matcher, config, workers);
        steps.iter().filter_map(|step| bed.step(step)).collect()
    }
}

fn sequence_row(
    name: &'static str,
    matcher: Matcher,
    config: Config,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    let first = Bed::run(matcher, &config, &workers, &steps);
    for _ in 0..19 {
        assert_eq!(
            Bed::run(matcher, &config, &workers, &steps),
            first,
            "{name}: a sequence row must not depend on a draw"
        );
    }
    Row {
        name,
        matcher,
        config,
        workers,
        steps,
        outcome: Outcome::Sequence(first),
    }
}

fn drawn_row(
    name: &'static str,
    matcher: Matcher,
    config: Config,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
) -> Row {
    const DRAWS: u32 = 6_000;
    let mut counts = vec![0u32; workers.len()];
    for _ in 0..DRAWS {
        let picked = Bed::run(matcher, &config, &workers, &steps);
        assert!(
            picked.iter().all(|&place| place < workers.len()),
            "{name}: a drawn row places every request, and a draw answered {picked:?}"
        );
        counts[*picked.last().expect("a drawn row ends with a request")] += 1;
    }
    Row {
        name,
        matcher,
        config,
        workers,
        steps,
        outcome: Outcome::Counts {
            draws: DRAWS,
            counts,
        },
    }
}

/// The configuration of most rows: a match over half the prompt is a hit,
/// the count gate at a relative 1.1 and an absolute 32, blocks of sixteen,
/// both KV gates off.
const PLAIN: Config = Config {
    cache_threshold: 0.5,
    balance_abs_threshold: 32,
    balance_rel_threshold: 1.1,
    block_size: 16,
    balance_token_usage_threshold: 1.0,
    overload_token_usage_threshold: 1.0,
};

/// The gateway's command line defaults where the policy's own differ: a
/// match over three tenths of the prompt is a hit, and the count gate
/// stands at a relative 1.5 and an absolute 64.
const COMMAND_LINE: Config = Config {
    cache_threshold: 0.3,
    balance_abs_threshold: 64,
    balance_rel_threshold: 1.5,
    ..PLAIN
};

/// A count gate whose absolute margin is set under its floor of four.
const UNDER_THE_FLOOR: Config = Config {
    balance_abs_threshold: 2,
    balance_rel_threshold: 1.5,
    ..PLAIN
};

/// Free workers with no report, by their requests in flight.
fn free(in_flight: &[u32]) -> Vec<WorkerState> {
    in_flight
        .iter()
        .map(|&in_flight| WorkerState {
            in_flight,
            state: State::Free,
            report: None,
        })
        .collect()
}

/// A report of an idle queue at `kv_usage_permille` of KV-cache usage.
fn usage(kv_usage_permille: u16) -> Report {
    Report {
        running: 0,
        queued: 0,
        resident_tokens: None,
        waiting_tokens: 0,
        throughput: 0,
        kv_usage_permille,
    }
}

/// Free workers that report, by their requests in flight and their KV-cache
/// usage.
fn reporting(workers: &[(u32, u16)]) -> Vec<WorkerState> {
    workers
        .iter()
        .map(|&(in_flight, kv_usage_permille)| WorkerState {
            in_flight,
            state: State::Free,
            report: Some(usage(kv_usage_permille)),
        })
        .collect()
}

fn pick(prompt: &[[u32; 2]]) -> Step {
    Step::Pick {
        prompt: prompt.to_vec(),
        tried: vec![],
    }
}

fn retry(prompt: &[[u32; 2]], tried: &[usize]) -> Step {
    Step::Pick {
        prompt: prompt.to_vec(),
        tried: tried.to_vec(),
    }
}

fn stored(worker: usize, prompt: &[[u32; 2]]) -> Step {
    Step::Stored {
        worker,
        prompt: prompt.to_vec(),
    }
}

fn removed(worker: usize, prompt: &[[u32; 2]], kept_blocks: usize) -> Step {
    Step::Removed {
        worker,
        prompt: prompt.to_vec(),
        kept_blocks,
    }
}

/// A prompt of four blocks.
const WHOLE: [[u32; 2]; 1] = [[1000, 64]];
/// Its first block, its first two and its first three alone.
const FIRST: [[u32; 2]; 1] = [[1000, 16]];
const HALF: [[u32; 2]; 1] = [[1000, 32]];
const MOST: [[u32; 2]; 1] = [[1000, 48]];
/// Prompts of four blocks that share its first three and its first two.
const THREE: [[u32; 2]; 2] = [[1000, 48], [5000, 16]];
const TWO: [[u32; 2]; 2] = [[1000, 32], [6000, 32]];
/// Prompts that share nothing with it.
const OTHER: [[u32; 2]; 1] = [[9000, 64]];
const ANOTHER: [[u32; 2]; 1] = [[8000, 64]];
const ASIDE: [[u32; 2]; 1] = [[7000, 16]];
/// The same as text: four ids are one block of sixteen characters, and no
/// two runs start with the same character.
const TEXT_WHOLE: [[u32; 2]; 1] = [[100, 16]];
const TEXT_THREE: [[u32; 2]; 2] = [[100, 12], [9000, 4]];
const TEXT_TWO: [[u32; 2]; 2] = [[100, 8], [9000, 8]];

fn rows() -> Vec<Row> {
    use Matcher::{Events, Text, Tokens};
    use State::{Down, Free, Over};
    // The KV gates, each alone: the usage spread, then the usage ceiling.
    let spread = Config {
        balance_token_usage_threshold: 0.3,
        ..PLAIN
    };
    let ceiling = Config {
        overload_token_usage_threshold: 0.8,
        ..PLAIN
    };
    // A holder at 900 of 1,000 beside two workers at 100.
    let hot = vec![Some(usage(900)), Some(usage(100)), Some(usage(100))];
    let mut rows = vec![
        // The tree of token ids. The worker a request was placed on holds its
        // prompt: a later prompt that shares more than half of itself with it
        // goes there, over an idle worker.
        sequence_row(
            "tokens-a-holder-keeps-its-prefix-over-an-idle-worker",
            Tokens,
            PLAIN,
            free(&[2, 0, 2]),
            vec![pick(&WHOLE), Step::Loads(vec![1, 3, 0]), pick(&THREE)],
        ),
        // A holder over the count gate spills to the fewest in flight under it.
        // The spill's worker holds the prompt too, and the two holders then go
        // by their loads.
        sequence_row(
            "tokens-a-holder-over-the-count-gate-spills-and-the-prefix-gains-a-holder",
            Tokens,
            PLAIN,
            free(&[2, 0, 2]),
            vec![
                pick(&WHOLE),
                Step::Loads(vec![1, 9, 0]),
                pick(&WHOLE),
                pick(&WHOLE),
                Step::Loads(vec![0, 2, 5]),
                pick(&WHOLE),
            ],
        ),
        // Half of the prompt is not more than half: a miss.
        sequence_row(
            "tokens-a-match-at-the-threshold-is-a-miss",
            Tokens,
            PLAIN,
            free(&[2, 0, 2]),
            vec![pick(&WHOLE), Step::Loads(vec![0, 3, 1]), pick(&TWO)],
        ),
        // At the command line's defaults a match over three tenths is a
        // hit: the same half keeps its holder.
        sequence_row(
            "tokens-at-the-command-lines-defaults-a-match-of-half-is-a-hit",
            Tokens,
            COMMAND_LINE,
            free(&[2, 0, 2]),
            vec![pick(&WHOLE), Step::Loads(vec![0, 3, 1]), pick(&TWO)],
        ),
        // A miss goes to the fewest in flight, and a request that ends frees
        // its worker for the next one.
        sequence_row(
            "tokens-a-completion-frees-its-worker-for-the-next-miss",
            Tokens,
            PLAIN,
            free(&[2, 0, 3]),
            vec![
                pick(&WHOLE),
                pick(&OTHER),
                Step::Complete(0),
                pick(&ANOTHER),
            ],
        ),
        // The tree names the holders of the deepest match alone. With that
        // holder tried, the request is a miss, though another worker holds
        // three blocks of its four.
        sequence_row(
            "tokens-the-deepest-holder-tried-is-a-miss-beside-a-shallower-holder",
            Tokens,
            PLAIN,
            free(&[0, 3, 3]),
            vec![
                pick(&WHOLE),
                Step::Loads(vec![3, 0, 3]),
                retry(&THREE, &[0]),
                Step::Loads(vec![1, 2, 0]),
                retry(&WHOLE, &[0]),
            ],
        ),
        // A holder that is not ready is passed by, and holds again when it is
        // back, beside the worker that took its request meanwhile.
        sequence_row(
            "tokens-a-holder-that-is-not-ready-is-a-miss-and-holds-again-when-it-is-back",
            Tokens,
            PLAIN,
            free(&[0, 3, 3]),
            vec![
                pick(&WHOLE),
                Step::States(vec![Down, Free, Free]),
                Step::Loads(vec![0, 2, 1]),
                pick(&WHOLE),
                Step::States(vec![Free, Free, Free]),
                Step::Loads(vec![1, 0, 4]),
                pick(&WHOLE),
            ],
        ),
        sequence_row(
            "tokens-a-holder-over-its-thresholds-is-a-miss",
            Tokens,
            PLAIN,
            free(&[0, 3, 3]),
            vec![
                pick(&WHOLE),
                Step::States(vec![Over, Free, Free]),
                Step::Loads(vec![0, 2, 1]),
                pick(&WHOLE),
            ],
        ),
    ];
    // Under KV pressure affinity is dropped for the whole pool: the holder,
    // the hottest worker, yields to the lowest expected wait. With both
    // gates off the same holder keeps its prompt.
    for (name, config) in [
        (
            "tokens-under-the-kv-spread-gate-a-hot-holder-yields-to-the-expected-wait",
            spread,
        ),
        (
            "tokens-under-the-kv-ceiling-a-hot-holder-yields-to-the-expected-wait",
            ceiling,
        ),
        (
            "tokens-with-the-kv-gates-off-the-hot-holder-keeps-its-prefix",
            PLAIN,
        ),
    ] {
        rows.push(sequence_row(
            name,
            Tokens,
            config,
            reporting(&[(0, 100), (1, 100), (2, 100)]),
            vec![pick(&WHOLE), Step::Reports(hot.clone()), pick(&WHOLE)],
        ));
    }
    // Equal loads on a miss are drawn, over the tied workers alone.
    rows.push(drawn_row(
        "tokens-a-miss-over-equal-loads-is-drawn",
        Tokens,
        PLAIN,
        free(&[1, 1, 3]),
        vec![pick(&OTHER)],
    ));

    // The index of the blocks the workers report. Every worker holds its
    // own depth: the deepest holder takes the request, and equal holders go
    // by their loads.
    rows.push(sequence_row(
        "events-the-deepest-holder-keeps-the-request-and-equal-holders-go-by-load",
        Events,
        PLAIN,
        free(&[3, 1, 0]),
        vec![
            stored(0, &WHOLE),
            stored(1, &HALF),
            pick(&WHOLE),
            pick(&TWO),
        ],
    ));
    // The index counts a match at any depth: one block of four holds.
    rows.push(sequence_row(
        "events-a-one-block-overlap-keeps-its-holder",
        Events,
        PLAIN,
        free(&[2, 0, 0]),
        vec![stored(0, &FIRST), pick(&WHOLE)],
    ));
    rows.push(sequence_row(
        "events-a-holder-over-the-count-gate-spills-and-holds-again-under-it",
        Events,
        PLAIN,
        free(&[9, 1, 0]),
        vec![
            stored(0, &WHOLE),
            pick(&WHOLE),
            Step::Loads(vec![5, 1, 0]),
            pick(&WHOLE),
        ],
    ));
    rows.push(sequence_row(
        "events-a-miss-goes-to-the-fewest-in-flight",
        Events,
        PLAIN,
        free(&[2, 1, 0]),
        vec![stored(0, &WHOLE), pick(&OTHER)],
    ));
    // Where the two readings part: with the deepest holder tried or over
    // its thresholds, the index names the next holder, and it takes the
    // request over an idle worker.
    rows.push(sequence_row(
        "events-the-deepest-holder-tried-a-shallower-holder-takes-the-request",
        Events,
        PLAIN,
        free(&[1, 2, 0]),
        vec![stored(0, &WHOLE), stored(1, &MOST), retry(&WHOLE, &[0])],
    ));
    rows.push(sequence_row(
        "events-the-deepest-holder-over-its-thresholds-a-shallower-holder-takes-the-request",
        Events,
        PLAIN,
        free(&[1, 2, 0]),
        vec![
            Step::States(vec![Over, Free, Free]),
            stored(0, &WHOLE),
            stored(1, &MOST),
            pick(&WHOLE),
        ],
    ));
    // A removed block shortens what its worker holds: a worker that held
    // less then holds more and takes the request. The last removed block
    // ends a hold. A third worker's block of another prompt keeps the index
    // from standing empty.
    rows.push(sequence_row(
        "events-a-removed-block-shortens-the-hold-and-the-last-one-ends-it",
        Events,
        PLAIN,
        free(&[2, 1, 0]),
        vec![
            stored(0, &WHOLE),
            stored(1, &HALF),
            stored(2, &ASIDE),
            pick(&WHOLE),
            removed(0, &WHOLE, 1),
            pick(&WHOLE),
            removed(1, &HALF, 0),
            pick(&WHOLE),
            removed(0, &WHOLE, 0),
            pick(&WHOLE),
        ],
    ));
    // At the command line's defaults the count gate takes a factor of 1.5.
    // A holder over the mean by more than the margin keeps its prefix while
    // the factor allows it, and spills past the factor.
    for (name, in_flight) in [
        (
            "events-at-the-command-lines-defaults-a-holder-under-the-factor-keeps-its-prefix",
            [37, 19, 19],
        ),
        (
            "events-at-the-command-lines-defaults-a-holder-over-the-factor-spills",
            [39, 19, 18],
        ),
    ] {
        rows.push(sequence_row(
            name,
            Events,
            COMMAND_LINE,
            free(&in_flight),
            vec![stored(0, &WHOLE), pick(&WHOLE)],
        ));
    }
    // A margin set under the floor decides: it spills a holder the floor
    // would keep.
    rows.push(sequence_row(
        "events-a-margin-under-the-floor-spills-a-holder-the-floor-would-keep",
        Events,
        UNDER_THE_FLOOR,
        free(&[8, 3, 4]),
        vec![stored(0, &WHOLE), pick(&WHOLE)],
    ));
    for (name, config) in [
        (
            "events-under-the-kv-spread-gate-a-hot-holder-yields-to-the-expected-wait",
            spread,
        ),
        (
            "events-with-the-kv-gates-off-the-hot-holder-keeps-the-request",
            PLAIN,
        ),
    ] {
        rows.push(sequence_row(
            name,
            Events,
            config,
            reporting(&[(1, 900), (1, 100), (2, 100)]),
            vec![stored(0, &WHOLE), pick(&WHOLE)],
        ));
    }
    // Equal holders at equal loads are drawn, and an idle worker that holds
    // nothing is never picked.
    rows.push(drawn_row(
        "events-equal-holders-at-equal-loads-are-drawn",
        Events,
        PLAIN,
        free(&[2, 2, 0]),
        vec![stored(0, &WHOLE), stored(1, &WHOLE), pick(&WHOLE)],
    ));

    // The tree of text decides as the tree of token ids does, over
    // characters.
    rows.push(sequence_row(
        "text-a-holder-keeps-its-prefix-over-an-idle-worker",
        Text,
        PLAIN,
        free(&[2, 0, 2]),
        vec![
            pick(&TEXT_WHOLE),
            Step::Loads(vec![1, 3, 0]),
            pick(&TEXT_THREE),
        ],
    ));
    rows.push(sequence_row(
        "text-a-match-at-the-threshold-is-a-miss",
        Text,
        PLAIN,
        free(&[2, 0, 2]),
        vec![
            pick(&TEXT_WHOLE),
            Step::Loads(vec![0, 3, 1]),
            pick(&TEXT_TWO),
        ],
    ));
    rows.push(sequence_row(
        "text-a-holder-over-the-count-gate-spills",
        Text,
        PLAIN,
        free(&[2, 0, 2]),
        vec![
            pick(&TEXT_WHOLE),
            Step::Loads(vec![1, 9, 0]),
            pick(&TEXT_WHOLE),
        ],
    ));
    rows
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_cache_aware_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "cache_aware",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("cache_aware.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
