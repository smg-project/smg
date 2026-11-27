//! A toy fleet simulation that checks each policy does what it claims, before any engine is
//! involved: it beats a load-blind baseline when prefixes repeat (cache awareness) and it does not
//! collapse onto one worker under a hot prefix (load awareness). Only orderings are asserted; the
//! numbers are printed so a `--nocapture` run reports the relative TTFTs.
//!
//! The model: `N` workers, each with a FIFO block cache of `cache_blocks` and a prefill queue that
//! drains `drain_tokens_per_tick` per tick. A request's TTFT, in ticks, is the queue ahead of it
//! plus its own uncached tokens, over the drain rate. Dispatching inserts the prompt's blocks and
//! queues its uncached tokens. One tick is one simulated second for the policies' time parameters.
#![expect(
    clippy::print_stderr,
    reason = "the simulation reports its numbers under --nocapture"
)]

use std::collections::{HashSet, VecDeque};

use rand::{rngs::StdRng, RngExt, SeedableRng};

use super::{build, CandidateInputs, Pick, RequestInputs, WorkerSelectionPolicy};

const BLOCK: usize = 16;

struct SimWorker {
    url: String,
    cache: VecDeque<u64>,
    cached: HashSet<u64>,
    cache_blocks: usize,
    /// Prefill tokens queued, oldest first; each entry is one request.
    queue: VecDeque<f64>,
}

impl SimWorker {
    fn backlog(&self) -> f64 {
        self.queue.iter().sum()
    }

    fn overlap(&self, prompt: &[u64]) -> usize {
        prompt
            .iter()
            .take_while(|hash| self.cached.contains(hash))
            .count()
    }

    fn insert(&mut self, prompt: &[u64]) {
        for &hash in prompt {
            if self.cached.insert(hash) {
                self.cache.push_back(hash);
                if self.cache.len() > self.cache_blocks {
                    if let Some(evicted) = self.cache.pop_front() {
                        self.cached.remove(&evicted);
                    }
                }
            }
        }
    }

    /// Drain the queue by `tokens`; returns how many requests completed.
    fn drain(&mut self, mut tokens: f64) -> usize {
        let mut completed = 0;
        while tokens > 0.0 {
            let Some(front) = self.queue.front_mut() else {
                break;
            };
            if *front <= tokens {
                tokens -= *front;
                self.queue.pop_front();
                completed += 1;
            } else {
                *front -= tokens;
                tokens = 0.0;
            }
        }
        completed
    }
}

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    workers: usize,
    cache_blocks: usize,
    drain_tokens_per_tick: f64,
    sessions: usize,
    prefix_blocks: usize,
    suffix_blocks: usize,
    arrivals_per_tick: usize,
    /// Share of requests that go to session 0.
    hot_share: f64,
    requests: usize,
}

enum Chooser<'a> {
    Random,
    Sticky,
    Policy(&'a WorkerSelectionPolicy),
}

struct Outcome {
    mean_ttft: f64,
    p99_ttft: f64,
}

fn chain_hashes(
    session: u64,
    prefix_blocks: usize,
    suffix_seed: u64,
    suffix_blocks: usize,
) -> Vec<u64> {
    (0..prefix_blocks as u64)
        .map(|i| (session << 32) | i)
        .chain((0..suffix_blocks as u64).map(|i| (1 << 63) | (suffix_seed << 16) | i))
        .collect()
}

fn run(scenario: Scenario, chooser: Chooser<'_>, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut workers: Vec<SimWorker> = (0..scenario.workers)
        .map(|i| SimWorker {
            url: format!("http://w{i:03}:8000"),
            cache: VecDeque::new(),
            cached: HashSet::new(),
            cache_blocks: scenario.cache_blocks,
            queue: VecDeque::new(),
        })
        .collect();
    let mut ttfts: Vec<f64> = Vec::with_capacity(scenario.requests);
    let warmup = scenario.requests / 5;
    let mut issued = 0usize;
    let mut suffix_seed = 0u64;
    while issued < scenario.requests {
        for worker in &mut workers {
            let completed = worker.drain(scenario.drain_tokens_per_tick);
            if let Chooser::Policy(policy) = &chooser {
                for _ in 0..completed {
                    policy.on_request_complete(&worker.url);
                }
            }
        }
        for _ in 0..scenario.arrivals_per_tick {
            if issued == scenario.requests {
                break;
            }
            let session = if rng.random::<f64>() < scenario.hot_share {
                0
            } else {
                rng.random_range(0..scenario.sessions as u64)
            };
            suffix_seed += 1;
            let prompt = chain_hashes(
                session,
                scenario.prefix_blocks,
                suffix_seed,
                scenario.suffix_blocks,
            );
            let prompt_tokens = prompt.len() * BLOCK;
            let overlaps: Vec<usize> = workers.iter().map(|w| w.overlap(&prompt)).collect();
            let backlogs: Vec<f64> = workers.iter().map(SimWorker::backlog).collect();
            let inflight: Vec<usize> = workers.iter().map(|w| w.queue.len()).collect();
            let avg_load = inflight.iter().sum::<usize>() as f64 / workers.len() as f64;

            let lowest_backlog = |rows: &[usize]| {
                rows.iter()
                    .copied()
                    .min_by(|&a, &b| {
                        backlogs[a]
                            .total_cmp(&backlogs[b])
                            .then_with(|| workers[a].url.cmp(&workers[b].url))
                    })
                    .expect("non-empty")
            };
            let all: Vec<usize> = (0..workers.len()).collect();
            let chosen = match &chooser {
                Chooser::Random => rng.random_range(0..workers.len()),
                Chooser::Sticky => {
                    let best = overlaps.iter().copied().max().unwrap_or(0);
                    let tied: Vec<usize> = all
                        .iter()
                        .copied()
                        .filter(|&i| overlaps[i] == best)
                        .collect();
                    lowest_backlog(&tied)
                }
                Chooser::Policy(policy) => {
                    let request = RequestInputs {
                        prompt_tokens,
                        block_size: BLOCK,
                        request_blocks: prompt.len().max(1),
                        avg_load,
                        prefix_hashes: Some(&prompt),
                    };
                    let inputs: Vec<CandidateInputs<'_>> = workers
                        .iter()
                        .enumerate()
                        .map(|(i, w)| CandidateInputs {
                            idx: i,
                            url: &w.url,
                            device_blocks: overlaps[i] as f64,
                            effective_score: overlaps[i] as f64,
                        })
                        .collect();
                    let chosen = match policy.select(&request, &inputs) {
                        Pick::None => lowest_backlog(&all),
                        Pick::Final(row) => inputs[row].idx,
                        Pick::Group(rows) => {
                            let group: Vec<usize> = rows.iter().map(|&r| inputs[r].idx).collect();
                            lowest_backlog(&group)
                        }
                    };
                    policy.on_dispatch(&request, &inputs[chosen]);
                    chosen
                }
            };
            let uncached = (prompt.len() - overlaps[chosen]) as f64 * BLOCK as f64;
            let ttft = (backlogs[chosen] + uncached) / scenario.drain_tokens_per_tick;
            if issued >= warmup {
                ttfts.push(ttft);
            }
            workers[chosen].queue.push_back(uncached);
            workers[chosen].insert(&prompt);
            issued += 1;
        }
    }
    ttfts.sort_by(f64::total_cmp);
    let mean_ttft = ttfts.iter().sum::<f64>() / ttfts.len() as f64;
    let p99_ttft = ttfts[(ttfts.len() * 99 / 100).min(ttfts.len() - 1)];
    Outcome {
        mean_ttft,
        p99_ttft,
    }
}

/// Policies under test: the product's default.
fn policies() -> Vec<(&'static str, WorkerSelectionPolicy)> {
    vec![(
        "cache-aware-default",
        build("cache-aware-default", 0.0).unwrap(),
    )]
}

const REPEAT_HEAVY: Scenario = Scenario {
    name: "repeat_heavy",
    workers: 8,
    cache_blocks: 1024,
    drain_tokens_per_tick: 512.0,
    sessions: 48,
    prefix_blocks: 128,
    suffix_blocks: 8,
    arrivals_per_tick: 4,
    hot_share: 0.0,
    requests: 1500,
};

/// Three quarters of eight arrivals per tick share one prefix: a worker that keeps all of them
/// drains 512 tokens per tick against 768 queued, so staying sticky means an unbounded queue.
const HOT_PREFIX: Scenario = Scenario {
    name: "hot_prefix",
    sessions: 24,
    arrivals_per_tick: 8,
    hot_share: 0.75,
    requests: 2_400,
    ..REPEAT_HEAVY
};

#[test]
fn every_policy_beats_random_when_prefixes_repeat() {
    let random = run(REPEAT_HEAVY, Chooser::Random, 1);
    let sticky = run(REPEAT_HEAVY, Chooser::Sticky, 1);
    eprintln!(
        "{}: random mean {:.2} p99 {:.2}; sticky mean {:.2} p99 {:.2}",
        REPEAT_HEAVY.name, random.mean_ttft, random.p99_ttft, sticky.mean_ttft, sticky.p99_ttft
    );
    let outcomes: Vec<(&str, Outcome)> = policies()
        .iter()
        .map(|(name, policy)| (*name, run(REPEAT_HEAVY, Chooser::Policy(policy), 1)))
        .collect();
    for (name, outcome) in &outcomes {
        eprintln!(
            "{}: {name} mean {:.2} p99 {:.2}",
            REPEAT_HEAVY.name, outcome.mean_ttft, outcome.p99_ttft
        );
    }
    for (name, outcome) in &outcomes {
        assert!(
            outcome.mean_ttft < 0.5 * random.mean_ttft,
            "{name}: mean TTFT {:.2} is not under half of random's {:.2}",
            outcome.mean_ttft,
            random.mean_ttft
        );
    }
}

#[test]
fn every_policy_spreads_a_hot_prefix() {
    let random = run(HOT_PREFIX, Chooser::Random, 2);
    let sticky = run(HOT_PREFIX, Chooser::Sticky, 2);
    eprintln!(
        "{}: random mean {:.2} p99 {:.2}; sticky mean {:.2} p99 {:.2}",
        HOT_PREFIX.name, random.mean_ttft, random.p99_ttft, sticky.mean_ttft, sticky.p99_ttft
    );
    for (name, policy) in policies() {
        let outcome = run(HOT_PREFIX, Chooser::Policy(&policy), 2);
        eprintln!(
            "{}: {name} mean {:.2} p99 {:.2}",
            HOT_PREFIX.name, outcome.mean_ttft, outcome.p99_ttft
        );
        // The cache-aware default is the sticky baseline by construction: its
        // load relief is the host's spill gate and expected-wait selector,
        // which this simulation does not model.
        if name != "cache-aware-default" {
            assert!(
                outcome.p99_ttft < sticky.p99_ttft,
                "{name}: p99 TTFT {:.2} does not beat sticky's {:.2}",
                outcome.p99_ttft,
                sticky.p99_ttft
            );
        }
        assert!(
            outcome.mean_ttft < random.mean_ttft,
            "{name}: mean TTFT {:.2} does not beat random's {:.2}",
            outcome.mean_ttft,
            random.mean_ttft
        );
    }
}
