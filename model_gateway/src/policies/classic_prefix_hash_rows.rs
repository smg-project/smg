//! The recorder of the gateway's prefix hash policy: each row is a pool of
//! workers, what the router holds of each one (the requests in flight and
//! whether it is ready), the policy's configuration and what the gateway's
//! single-worker placement answered. The policy controller crate replays the
//! rows through its own decision
//! (`crates/policy_controller/tests/classic/prefix_hash.rs`).
//!
//! A prompt is written as runs of consecutive token ids, `[first, count]`
//! each. A request carries a prompt as tokens or as text (each token id
//! spelled as four hex digits, so the policy's four characters a token read
//! the same span), a routing key, or nothing to hash.
//!
//! Two rings put one key on different workers, so a ring row lists its
//! workers in the ring order of the request it names (`ring`): the worker
//! that request lands on with every worker ready and idle, then the one it
//! passes to when that worker is down, and so on. A ring row's loads, ready
//! lists, tried lists and answers all count in that order, so an answer of 0
//! is the key's own worker on whatever ring replays the row.
//!
//! A pairs row places pairs of prompts on an idle pool and counts the pairs
//! whose two prompts took one worker, then the pairs that took two. Prompt
//! `i` of such a row is `tokens` ids from `1_000_000 + 1_000 * i` on, and its
//! twin differs at the one place `differ_at`, where it holds
//! `2_000_000_000 + i`. Where the row's prompts are `text`, each is spelled
//! as text: prompt `i` is ids from `1_000 * i` on and its twin holds
//! `65_000 + i`. A spread row places `count` such prompts once each and
//! counts what each worker took, the largest count first: two rings give one
//! worker different shares.
//!
//! Under boundaries a prompt is keyed at the deepest one it reaches, and a
//! worker past the bound passes the key to the worker of the next boundary
//! down. Two rings agree on no pair of keys, so a level row says how its two
//! keys stand to each other: its workers are listed in the ring order of its
//! request, and `levels` lists them again in the ring order of the request's
//! head at the shallower boundary. The recorder takes the first prompt that
//! stands so, 300 ids from `3_000_000 + 1_000 * i` on, and a replay looks
//! for its own.
//!
//! Run it from the repository's root:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib policies::classic_prefix_hash_rows -- --ignored --nocapture
//! ```
//!
//! No row ties the fewest requests: between equal loads the gateway takes
//! the first worker of its pool, which is the pool's order and no rule of the
//! policy.
//!
//! A sequence row is recorded twenty times from a fresh pool and a fresh
//! policy and must come out the same each time. Nothing in a pairs row or a
//! spread row draws either: each is recorded twice and must agree.

use std::{fs, path::PathBuf, sync::Arc};

use http::HeaderMap;
use openai_protocol::worker::WorkerStatus;
use serde::Serialize;
use smg_worker::{identity::basic::Worker, test_support::ready_worker};

use super::PolicyRegistry;
use crate::{
    config::types::PolicyConfig,
    routers::common::{
        placement::{self, PlacementInputs},
        snapshot::{RoutingPool, WorkerRegistry},
    },
};

const MODEL: &str = "m";

/// The policy's configuration, in the gateway's own spelling.
#[derive(Clone, Copy, Serialize)]
struct Config {
    prefix_token_count: usize,
    load_factor: f64,
    balance_abs_threshold: usize,
    /// The boundaries a prompt is keyed at, ascending; none for a key at the
    /// one depth `prefix_token_count`.
    #[serde(skip_serializing_if = "<[usize]>::is_empty")]
    cache_boundaries: &'static [usize],
}

const DEFAULT: Config = Config {
    prefix_token_count: 256,
    load_factor: 1.25,
    balance_abs_threshold: 10,
    cache_boundaries: &[],
};

/// The default configuration under two boundaries.
const LEVELLED: Config = Config {
    cache_boundaries: &[64, 256],
    ..DEFAULT
};

impl Config {
    fn policy(self) -> PolicyConfig {
        PolicyConfig::PrefixHash {
            prefix_token_count: self.prefix_token_count,
            load_factor: self.load_factor,
            balance_abs_threshold: self.balance_abs_threshold,
            cache_boundaries: self.cache_boundaries.to_vec(),
        }
    }
}

/// A prompt: runs of consecutive token ids, `[first, count]` each.
type Runs = &'static [[u32; 2]];

fn ids(runs: Runs) -> Vec<u32> {
    runs.iter()
        .flat_map(|&[first, count]| first..first + count)
        .collect()
}

/// A prompt as text: four hex digits a token id.
fn spell(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(|&id| {
            assert!(id < 0x1_0000, "a spelled token id fits four hex digits");
            format!("{id:04x}")
        })
        .collect()
}

/// What a request carries for the policy to place it by.
#[derive(Clone, Copy, Serialize)]
struct Request {
    #[serde(skip_serializing_if = "Option::is_none")]
    key: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens: Option<Runs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<Runs>,
    /// The workers an earlier attempt of the request already tried.
    #[serde(skip_serializing_if = "<[usize]>::is_empty")]
    tried: &'static [usize],
}

const NOTHING: Request = Request {
    key: None,
    tokens: None,
    text: None,
    tried: &[],
};

const fn tokens(runs: Runs) -> Request {
    Request {
        tokens: Some(runs),
        ..NOTHING
    }
}

const fn text(runs: Runs) -> Request {
    Request {
        text: Some(runs),
        ..NOTHING
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request.
    Pick(Request),
    /// The requests the router holds on each worker, from here on.
    Loads(Vec<u32>),
    /// Whether each worker is ready, from here on.
    Ready(Vec<bool>),
}

/// The prompts of a pairs row or a spread row.
#[derive(Clone, Copy, Serialize)]
struct Prompts {
    count: u32,
    tokens: u32,
    /// A pairs row: the one place a prompt's twin differs at.
    #[serde(skip_serializing_if = "Option::is_none")]
    differ_at: Option<u32>,
    /// Whether each prompt is carried as text.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    text: bool,
}

impl Prompts {
    fn prompt(self, i: u32) -> Vec<u32> {
        let first = if self.text { 0 } else { 1_000_000 };
        (0..self.tokens)
            .map(|place| first + 1_000 * i + place)
            .collect()
    }

    fn twin(self, i: u32, differ_at: u32) -> Vec<u32> {
        let mut twin = self.prompt(i);
        twin[differ_at as usize] = if self.text { 65_000 } else { 2_000_000_000 } + i;
        twin
    }
}

#[derive(PartialEq, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The answer to each `Pick` step, in order.
    Sequence(Vec<usize>),
    /// A pairs row: over `draws` pairs, the pairs whose two prompts took one
    /// worker, then the pairs that took two. A spread row: over `draws`
    /// prompts, what each worker took, the largest count first.
    Counts { draws: u32, counts: Vec<u32> },
}

/// A boundary a level row's request reaches under its deepest one: the row's
/// workers in the ring order of the request's head at that boundary.
#[derive(Serialize)]
struct Level {
    boundary: usize,
    order: Vec<usize>,
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    config: Config,
    /// A ring row's request: the row's workers are listed in its ring order.
    #[serde(skip_serializing_if = "Option::is_none")]
    ring: Option<Request>,
    /// A level row: the shallower boundaries its request reaches.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    levels: Vec<Level>,
    /// The requests the router holds on each worker at the start.
    workers: Vec<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    steps: Vec<Step>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompts: Option<Prompts>,
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

/// A fresh registry of ready workers under a fresh policy.
struct Bed {
    registry: WorkerRegistry,
    policies: PolicyRegistry,
    workers: Vec<Arc<dyn Worker>>,
    /// Where each of the row's workers stands in the registered pool: the
    /// row's own order, or the ring order of a ring row's request.
    places: Vec<usize>,
}

impl Bed {
    fn new(config: Config, loads: &[u32], places: &[usize]) -> Self {
        assert_eq!(places.len(), loads.len(), "a place of every worker");
        let registry = WorkerRegistry::new();
        let workers: Vec<Arc<dyn Worker>> = (0..loads.len())
            .map(|i| {
                let worker = ready_worker(&url(i), MODEL);
                registry
                    .register(Arc::clone(&worker))
                    .expect("the worker registers");
                worker
            })
            .collect();
        let bed = Self {
            registry,
            policies: PolicyRegistry::new(config.policy()),
            workers,
            places: places.to_vec(),
        };
        bed.loads(loads);
        bed
    }

    fn loads(&self, loads: &[u32]) {
        assert_eq!(loads.len(), self.workers.len(), "a load of every worker");
        for (&place, &want) in self.places.iter().zip(loads) {
            let worker = &self.workers[place];
            while worker.load() < want as usize {
                worker.increment_load();
            }
            while worker.load() > want as usize {
                worker.decrement_load();
            }
        }
    }

    fn ready(&self, ready: &[bool]) {
        assert_eq!(ready.len(), self.workers.len(), "a word on every worker");
        for (&place, &ready) in self.places.iter().zip(ready) {
            self.workers[place].set_status(if ready {
                WorkerStatus::Ready
            } else {
                WorkerStatus::NotReady
            });
        }
    }

    /// One request through the gateway's single-worker placement: the place
    /// of the worker it chose among the row's workers.
    fn place(
        &self,
        key: Option<&str>,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        tried: &[usize],
    ) -> usize {
        let mut headers = HeaderMap::new();
        if let Some(key) = key {
            headers.insert("x-smg-routing-key", key.parse().expect("a header value"));
        }
        let tried: Vec<String> = tried.iter().map(|&i| url(self.places[i])).collect();
        let inputs = PlacementInputs {
            text,
            tokens,
            headers: Some(&headers),
            tried: &tried,
            ..Default::default()
        };
        let worker = placement::select_single(
            &self.registry,
            &self.policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            inputs,
            placement::Stated::nothing,
        )
        .expect("the request is placed");
        let place = self
            .workers
            .iter()
            .position(|w| w.url() == worker.url())
            .expect("a worker of the pool");
        self.places
            .iter()
            .position(|&p| p == place)
            .expect("a worker of the row")
    }

    fn pick(&self, request: Request) -> usize {
        let tokens = request.tokens.map(ids);
        let text = request.text.map(|runs| spell(&ids(runs)));
        self.place(
            request.key,
            tokens.as_deref(),
            text.as_deref(),
            request.tried,
        )
    }

    fn run(config: Config, loads: &[u32], steps: &[Step], places: &[usize]) -> Vec<usize> {
        let bed = Self::new(config, loads, places);
        let mut answers = Vec::new();
        for step in steps {
            match step {
                Step::Pick(request) => answers.push(bed.pick(*request)),
                Step::Loads(loads) => bed.loads(loads),
                Step::Ready(ready) => bed.ready(ready),
            }
        }
        answers
    }
}

/// The row's own order: worker `i` of the row is worker `i` of the pool.
fn in_order(workers: usize) -> Vec<usize> {
    (0..workers).collect()
}

/// The ring order of `request` over a pool of `workers`: the worker it lands
/// on with every worker ready and idle, then with the members found so far
/// down, until every worker is placed.
fn ring_order(config: Config, workers: usize, request: Request) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::with_capacity(workers);
    while order.len() < workers {
        let bed = Bed::new(config, &vec![0; workers], &in_order(workers));
        let ready: Vec<bool> = (0..workers).map(|place| !order.contains(&place)).collect();
        bed.ready(&ready);
        let next = bed.pick(request);
        assert!(
            !order.contains(&next),
            "the request's next member is a worker not yet placed, got {next}"
        );
        order.push(next);
    }
    order
}

/// What `record` answers, held to the same answer `times` times over.
fn recorded(name: &str, times: usize, record: impl Fn() -> Outcome) -> Outcome {
    let first = record();
    for _ in 1..times {
        assert_eq!(record(), first, "{name}: a row must not depend on a draw");
    }
    first
}

/// A row of steps. With `ring` its workers are listed in that request's ring
/// order, else in the pool's own.
fn sequence_row(
    name: &'static str,
    config: Config,
    ring: Option<Request>,
    workers: &[u32],
    steps: Vec<Step>,
) -> Row {
    let places = match ring {
        Some(request) => {
            let places = ring_order(config, workers.len(), request);
            assert_eq!(
                ring_order(config, workers.len(), request),
                places,
                "{name}: the ring's order must not depend on a draw"
            );
            places
        }
        None => in_order(workers.len()),
    };
    let outcome = recorded(name, 20, || {
        Outcome::Sequence(Bed::run(config, workers, &steps, &places))
    });
    Row {
        name,
        config,
        ring,
        levels: Vec::new(),
        workers: workers.to_vec(),
        steps,
        prompts: None,
        outcome,
    }
}

/// The idle pool a pairs row and a spread row place their prompts on.
const POOL: usize = 8;

fn prompts_row(name: &'static str, config: Config, prompts: Prompts) -> Row {
    let outcome = recorded(name, 2, || {
        let bed = Bed::new(config, &[0; POOL], &in_order(POOL));
        let place = |tokens: &[u32]| {
            if prompts.text {
                bed.place(None, None, Some(&spell(tokens)), &[])
            } else {
                bed.place(None, Some(tokens), None, &[])
            }
        };
        match prompts.differ_at {
            Some(differ_at) => {
                let together = (0..prompts.count)
                    .filter(|&i| place(&prompts.prompt(i)) == place(&prompts.twin(i, differ_at)))
                    .count();
                let together = u32::try_from(together).expect("a count of pairs");
                Outcome::Counts {
                    draws: prompts.count,
                    counts: vec![together, prompts.count - together],
                }
            }
            None => {
                let mut counts = vec![0u32; POOL];
                for i in 0..prompts.count {
                    counts[place(&prompts.prompt(i))] += 1;
                }
                counts.sort_unstable_by(|a, b| b.cmp(a));
                Outcome::Counts {
                    draws: prompts.count,
                    counts,
                }
            }
        }
    });
    Row {
        name,
        config,
        ring: None,
        levels: Vec::new(),
        workers: vec![0; POOL],
        steps: Vec::new(),
        prompts: Some(prompts),
        outcome,
    }
}

/// A head of 256 tokens with one tail, and the same head with another.
const A1: Runs = &[[100, 256], [9_000, 44]];
const A2: Runs = &[[100, 256], [9_500, 300]];
/// Another prompt, and one under 256 tokens.
const B: Runs = &[[20_000, 300]];
const SHORT: Runs = &[[700, 100]];
/// A prompt of ten tokens.
const TINY: Runs = &[[40_000, 10]];

const KEYED: Request = Request {
    key: Some("k1"),
    ..NOTHING
};

fn pick(request: Request) -> Step {
    Step::Pick(request)
}

/// A ring row under the default configuration.
fn ring(name: &'static str, request: Request, workers: &[u32], steps: Vec<Step>) -> Row {
    sequence_row(name, DEFAULT, Some(request), workers, steps)
}

/// Where a key lands, and what passes it on when its worker cannot take it.
fn key_rows() -> Vec<Row> {
    let idle = [0u32; 4];
    // Forty workers: the owner holds fifty requests, one worker far down
    // the ring holds none and every other holds three.
    let mut forty = [3u32; 40];
    forty[0] = 50;
    forty[37] = 0;
    vec![
        ring(
            "a-ten-token-prompt-keeps-its-key-s-worker-over-the-fewest-requests",
            tokens(TINY),
            &[3, 0, 1, 2],
            vec![pick(tokens(TINY)); 2],
        ),
        ring(
            "a-key-s-worker-is-found-among-forty",
            tokens(A1),
            &[0; 40],
            vec![pick(tokens(A1))],
        ),
        ring(
            "a-passed-key-takes-the-fewest-requests-among-forty",
            tokens(A1),
            &forty,
            vec![pick(tokens(A1))],
        ),
        ring(
            "a-prompt-lands-on-its-key-s-worker-every-time",
            tokens(A1),
            &idle,
            vec![pick(tokens(A1)); 3],
        ),
        ring(
            "prompts-that-share-their-first-256-tokens-land-together",
            tokens(A1),
            &idle,
            vec![pick(tokens(A1)), pick(tokens(A2))],
        ),
        ring(
            "a-prompt-under-256-tokens-lands-on-its-key-s-worker-every-time",
            tokens(SHORT),
            &idle,
            vec![pick(tokens(SHORT)); 2],
        ),
        ring(
            "a-prompt-as-text-lands-on-its-key-s-worker-every-time",
            text(B),
            &idle,
            vec![pick(text(B)); 2],
        ),
        ring(
            "texts-that-share-their-first-1024-characters-land-together",
            text(A1),
            &idle,
            vec![pick(text(A1)), pick(text(A2))],
        ),
        ring(
            "a-routing-key-leads-the-prompt",
            KEYED,
            &idle,
            vec![
                pick(Request {
                    key: Some("k1"),
                    ..tokens(A1)
                }),
                pick(Request {
                    key: Some("k1"),
                    ..tokens(B)
                }),
            ],
        ),
        ring(
            "a-key-whose-worker-is-down-passes-to-the-ring-s-next-member-and-returns",
            tokens(A1),
            &idle,
            vec![
                Step::Ready(vec![false, true, true, true]),
                pick(tokens(A1)),
                Step::Ready(vec![true; 4]),
                pick(tokens(A1)),
            ],
        ),
        ring(
            "a-retry-that-tried-the-key-s-worker-passes-to-the-ring-s-next-member",
            tokens(A1),
            &idle,
            vec![pick(Request {
                tried: &[0],
                ..tokens(A1)
            })],
        ),
    ]
}

/// The bound an owner keeps its key under, and where a passed key goes.
fn bound_rows() -> Vec<Row> {
    let one = || vec![pick(tokens(A1))];
    let wide = Config {
        load_factor: 2.0,
        balance_abs_threshold: 4,
        ..DEFAULT
    };
    vec![
        ring(
            "an-owner-past-the-relative-margin-alone-keeps-its-key",
            tokens(A1),
            &[12, 1, 2, 0],
            one(),
        ),
        ring(
            "an-owner-past-the-absolute-margin-alone-keeps-its-key",
            tokens(A1),
            &[130, 100, 110, 90],
            one(),
        ),
        ring(
            "an-owner-past-both-margins-passes-its-key-to-the-fewest-requests",
            tokens(A1),
            &[140, 100, 110, 90],
            one(),
        ),
        ring(
            "a-passed-key-takes-the-fewest-requests-and-not-the-ring-s-next-member",
            tokens(A1),
            &[30, 5, 2, 3],
            one(),
        ),
        ring(
            "a-passed-key-takes-the-ring-s-next-member-when-it-holds-the-fewest-requests",
            tokens(A1),
            &[30, 2, 5, 3],
            one(),
        ),
        ring(
            "an-owner-at-the-bound-keeps-its-key",
            tokens(A1),
            &[14, 1, 0, 0],
            one(),
        ),
        ring(
            "an-owner-one-request-past-the-bound-passes-its-key",
            tokens(A1),
            &[15, 1, 0, 2],
            one(),
        ),
        ring(
            "the-request-being-placed-counts-in-the-mean",
            tokens(A1),
            &[21, 0],
            vec![pick(tokens(A1)), Step::Loads(vec![22, 0]), pick(tokens(A1))],
        ),
        ring(
            "a-worker-that-is-down-is-outside-the-mean",
            tokens(A1),
            &[16, 40, 1, 0],
            vec![
                Step::Ready(vec![true, false, true, true]),
                pick(tokens(A1)),
                Step::Loads(vec![17, 40, 1, 0]),
                pick(tokens(A1)),
            ],
        ),
        ring(
            "a-routing-key-is-held-to-the-bound-as-a-prompt-is",
            KEYED,
            &[30, 5, 2, 3],
            vec![pick(KEYED)],
        ),
        sequence_row(
            "the-two-margins-are-the-configuration-s",
            wide,
            Some(tokens(A1)),
            &[14, 3, 4, 5],
            one(),
        ),
        sequence_row(
            "an-owner-at-1-3-times-the-mean-keeps-its-key-under-a-factor-of-1-3",
            Config {
                load_factor: 1.3,
                ..DEFAULT
            },
            Some(tokens(A1)),
            &[52, 27],
            vec![
                pick(tokens(A1)),
                Step::Loads(vec![53, 27]),
                pick(tokens(A1)),
            ],
        ),
        sequence_row(
            "a-product-that-rounds-under-the-owner-s-count-passes-its-key",
            Config {
                load_factor: 1.2,
                balance_abs_threshold: 2,
                ..DEFAULT
            },
            Some(tokens(A1)),
            &[14, 9, 11],
            one(),
        ),
        sequence_row(
            "a-request-with-nothing-to-hash-takes-the-fewest-requests",
            DEFAULT,
            None,
            &[3, 1, 2, 5],
            vec![pick(NOTHING)],
        ),
    ]
}

/// How deep the key reads, by pairs of prompts that differ at one place, and
/// how the keys spread.
fn depth_rows() -> Vec<Row> {
    let pairs = |tokens: u32, differ_at: u32| Prompts {
        count: 64,
        tokens,
        differ_at: Some(differ_at),
        text: false,
    };
    let shallow = Config {
        prefix_token_count: 64,
        ..DEFAULT
    };
    vec![
        prompts_row("the-256th-token-is-in-the-key", DEFAULT, pairs(300, 255)),
        prompts_row(
            "the-257th-token-is-outside-the-key",
            DEFAULT,
            pairs(300, 256),
        ),
        prompts_row(
            "the-last-token-of-a-prompt-under-256-tokens-is-in-the-key",
            DEFAULT,
            pairs(100, 99),
        ),
        prompts_row(
            "the-256th-token-of-a-text-is-in-the-key",
            DEFAULT,
            Prompts {
                text: true,
                ..pairs(300, 255)
            },
        ),
        prompts_row(
            "the-257th-token-of-a-text-is-outside-the-key",
            DEFAULT,
            Prompts {
                text: true,
                ..pairs(300, 256)
            },
        ),
        prompts_row(
            "at-a-depth-of-64-the-64th-token-is-in-the-key",
            shallow,
            pairs(300, 63),
        ),
        prompts_row(
            "at-a-depth-of-64-the-65th-token-is-outside-the-key",
            shallow,
            pairs(300, 64),
        ),
        prompts_row(
            "two-thousand-prompts-spread-over-eight-workers",
            DEFAULT,
            Prompts {
                count: 2_000,
                tokens: 300,
                differ_at: None,
                text: false,
            },
        ),
    ]
}

/// One run of token ids as a prompt. A recording's rows live as long as it
/// runs, so the run is leaked into them.
fn run(first: u32, count: usize) -> Runs {
    let count = u32::try_from(count).expect("a count of tokens");
    Box::leak(Box::new([[first, count]]))
}

/// How many tokens a level row's prompt holds: past the deeper boundary.
const LEVEL_TOKENS: usize = 300;

/// The first prompt of the level rows whose head at the shallower boundary
/// puts the row's workers in the order `want`, with the pool's order of the
/// row's workers. A ring spreads its keys, so a prompt among the first few
/// hundred stands so.
fn level_request(workers: usize, want: &[usize]) -> (Request, Vec<usize>) {
    let boundary = LEVELLED.cache_boundaries[0];
    (0..1_000u32)
        .find_map(|i| {
            let first = 3_000_000 + 1_000 * i;
            let request = tokens(run(first, LEVEL_TOKENS));
            let places = ring_order(LEVELLED, workers, request);
            let head = ring_order(LEVELLED, workers, tokens(run(first, boundary)));
            let order: Vec<usize> = head
                .iter()
                .map(|worker| {
                    places
                        .iter()
                        .position(|place| place == worker)
                        .expect("a worker of the pool")
                })
                .collect();
            (order == want).then_some((request, places))
        })
        .expect("a prompt whose two keys stand in the wanted order")
}

/// A row under the two boundaries: `before`, then one pick of a prompt whose
/// head at the shallower boundary stands in the order `want`.
fn level_row(name: &'static str, want: &[usize], workers: &[u32], before: Vec<Step>) -> Row {
    let (request, places) = level_request(workers.len(), want);
    let mut steps = before;
    steps.push(pick(request));
    let outcome = recorded(name, 20, || {
        Outcome::Sequence(Bed::run(LEVELLED, workers, &steps, &places))
    });
    Row {
        name,
        config: LEVELLED,
        ring: Some(request),
        levels: vec![Level {
            boundary: LEVELLED.cache_boundaries[0],
            order: want.to_vec(),
        }],
        workers: workers.to_vec(),
        steps,
        prompts: None,
        outcome,
    }
}

/// Under boundaries: the deepest one a prompt reaches is tried first, then
/// the next one down, and the fewest requests take a key that no boundary's
/// worker keeps.
fn level_rows() -> Vec<Row> {
    // The head's worker is the third of the prompt's own ring order: neither
    // the prompt's own worker nor that ring's next member.
    let apart = [2, 0, 1, 3];
    vec![
        level_row(
            "under-boundaries-a-worker-under-the-bound-keeps-the-deepest-boundary-s-key",
            &apart,
            &[12, 0, 5, 1],
            Vec::new(),
        ),
        level_row(
            "a-worker-past-the-bound-passes-the-key-to-the-next-boundary-s-worker",
            &apart,
            &[30, 0, 5, 1],
            Vec::new(),
        ),
        level_row(
            "workers-past-the-bound-at-every-boundary-pass-the-key-to-the-fewest-requests",
            &apart,
            &[30, 3, 28, 0],
            Vec::new(),
        ),
        level_row(
            "one-worker-at-both-boundaries-passes-the-key-to-the-fewest-requests",
            &[0, 2, 1, 3],
            &[30, 5, 0, 1],
            Vec::new(),
        ),
        level_row(
            "a-down-worker-at-the-next-boundary-hands-the-key-to-that-ring-s-next-member",
            &[2, 3, 0, 1],
            &[30, 0, 5, 1],
            vec![Step::Ready(vec![true, true, false, true])],
        ),
        level_row(
            "a-down-worker-at-the-deepest-boundary-hands-the-key-to-that-ring-s-next-member",
            &apart,
            &[0, 5, 0, 1],
            vec![Step::Ready(vec![false, true, true, true])],
        ),
    ]
}

/// How deep the key reads under boundaries, by pairs of prompts that differ
/// at one place.
fn level_depth_rows() -> Vec<Row> {
    let pairs = |tokens: u32, differ_at: u32| Prompts {
        count: 64,
        tokens,
        differ_at: Some(differ_at),
        text: false,
    };
    let other_depth = Config {
        prefix_token_count: 128,
        ..LEVELLED
    };
    vec![
        prompts_row(
            "under-boundaries-the-256th-token-of-a-prompt-that-reaches-256-is-in-the-key",
            LEVELLED,
            pairs(300, 255),
        ),
        prompts_row(
            "under-boundaries-the-257th-token-of-a-prompt-that-reaches-256-is-outside-the-key",
            LEVELLED,
            pairs(300, 256),
        ),
        prompts_row(
            "under-boundaries-the-64th-token-of-a-prompt-that-reaches-64-alone-is-in-the-key",
            LEVELLED,
            pairs(100, 63),
        ),
        prompts_row(
            "under-boundaries-the-65th-token-of-a-prompt-that-reaches-64-alone-is-outside-the-key",
            LEVELLED,
            pairs(100, 64),
        ),
        prompts_row(
            "under-boundaries-a-prompt-that-reaches-none-is-keyed-at-the-one-depth",
            LEVELLED,
            pairs(48, 47),
        ),
        prompts_row(
            "a-prompt-that-reaches-a-boundary-is-not-keyed-at-the-one-depth",
            other_depth,
            pairs(300, 200),
        ),
        prompts_row(
            "under-boundaries-the-256th-token-of-a-text-that-reaches-256-is-in-the-key",
            LEVELLED,
            Prompts {
                text: true,
                ..pairs(300, 255)
            },
        ),
        prompts_row(
            "under-boundaries-the-257th-token-of-a-text-that-reaches-256-is-outside-the-key",
            LEVELLED,
            Prompts {
                text: true,
                ..pairs(300, 256)
            },
        ),
    ]
}

fn rows() -> Vec<Row> {
    let mut rows = key_rows();
    rows.extend(bound_rows());
    rows.extend(depth_rows());
    rows.extend(level_rows());
    rows.extend(level_depth_rows());
    rows
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_prefix_hash_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    let sheet = Sheet {
        recorded_from: commit,
        policy: "prefix_hash",
        rows: rows(),
    };
    fs::create_dir_all(&out).expect("the output directory");
    let path = out.join("prefix_hash.json");
    let text = serde_json::to_string_pretty(&sheet).expect("the sheet serializes");
    fs::write(&path, text + "\n").expect("the sheet is written");
    println!("{} rows -> {}", sheet.rows.len(), path.display());
}
