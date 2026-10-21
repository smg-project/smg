//! What a servicer can tell about its engine's load from the requests it
//! forwards: queued token-work, generation throughput and the prefix-cache
//! hit rate, per data-parallel rank. The gateway's expected-wait score reads
//! them from `GetLoads`; an engine whose stats do not carry them (vLLM
//! reports running, waiting and KV usage) would otherwise be scored on
//! defaults and routed unlike its peers.
//!
//! The servicer sees every request it submits and the rank it goes to, the
//! engine's first output for it (with the prompt and cached token counts) and
//! every token streamed, so, for each rank:
//!
//! - queued token-work is the uncached prompt tokens of the requests the
//!   rank has not started: of the submitted requests without a first
//!   output, the youngest `num_waiting_reqs` (the engine admits FCFS, so the
//!   older ones are the ones in prefill), each prompt discounted by the
//!   rank's recent hit rate since its own cached count is not known until it
//!   starts;
//! - generation throughput is the tokens the rank streamed over the last
//!   [`THROUGHPUT_WINDOW`];
//! - the hit rate is cached over prompt tokens across the rank's last
//!   [`HIT_RATE_SAMPLES`] first outputs.
//!
//! A rank keeps its own figures: a report that put one rank's figures on
//! another, or the whole servicer's on its first rank, would read a serving
//! rank as idle or misses as hits. These are the field semantics the SGLang
//! servicer reports from its scheduler and the mock engine from its queue.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

/// How far back streamed tokens count toward the throughput.
pub(crate) const THROUGHPUT_WINDOW: Duration = Duration::from_secs(2);
/// How many first outputs the hit rate averages over.
pub(crate) const HIT_RATE_SAMPLES: usize = 64;

/// The three fields, as `GetLoads` reports them for one rank.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LoadEstimate {
    /// Uncached prompt tokens of the requests the rank has not started.
    pub queued_token_work: i32,
    /// Tokens the rank streamed per second over the window.
    pub gen_throughput: f64,
    /// Cached over prompt tokens of the rank's recent first outputs, in
    /// `[0, 1]`.
    pub cache_hit_rate: f64,
}

struct Pending {
    request_id: String,
    prompt_tokens: u32,
    /// The rank the request went to.
    rank: i32,
}

/// One rank's recent first outputs and streamed tokens.
#[derive(Default)]
struct Rank {
    /// Recent first outputs: (prompt tokens, cached tokens).
    hits: VecDeque<(u64, u64)>,
    prompt_sum: u64,
    cached_sum: u64,
    /// Tokens streamed, by time, within the window.
    generated: VecDeque<(Instant, u64)>,
    generated_sum: u64,
}

impl Rank {
    fn trim(&mut self, now: Instant) {
        while let Some(&(at, tokens)) = self.generated.front() {
            if now.duration_since(at) <= THROUGHPUT_WINDOW {
                break;
            }
            self.generated.pop_front();
            self.generated_sum -= tokens;
        }
    }

    fn hit_rate(&self) -> f64 {
        if self.prompt_sum == 0 {
            0.0
        } else {
            (self.cached_sum as f64 / self.prompt_sum as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Default)]
struct Inner {
    /// Submitted requests without a first output yet, oldest first, each
    /// under the rank it went to.
    pending: VecDeque<Pending>,
    /// The ranks that have started a request or streamed a token.
    ranks: BTreeMap<i32, Rank>,
}

/// Per-servicer load bookkeeping, kept per rank; every method is cheap and
/// lock-scoped.
#[derive(Default)]
pub(crate) struct LoadTracker {
    inner: Mutex<Inner>,
}

impl LoadTracker {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A request with `prompt_tokens` was handed to `rank`.
    pub(crate) fn submitted(&self, rank: i32, request_id: &str, prompt_tokens: u32) {
        self.lock().pending.push_back(Pending {
            request_id: request_id.to_string(),
            prompt_tokens,
            rank,
        });
    }

    /// `rank`'s first output for a request: it has started, and its prompt
    /// had `cached_tokens` of `prompt_tokens` in that rank's prefix cache.
    pub(crate) fn first_output(
        &self,
        rank: i32,
        request_id: &str,
        prompt_tokens: u32,
        cached_tokens: u32,
    ) {
        let mut inner = self.lock();
        drop_pending(&mut inner, request_id);
        if prompt_tokens == 0 {
            return;
        }
        let sample = (
            u64::from(prompt_tokens),
            u64::from(cached_tokens.min(prompt_tokens)),
        );
        let rank = inner.ranks.entry(rank).or_default();
        rank.hits.push_back(sample);
        rank.prompt_sum += sample.0;
        rank.cached_sum += sample.1;
        if rank.hits.len() > HIT_RATE_SAMPLES {
            if let Some((prompt, cached)) = rank.hits.pop_front() {
                rank.prompt_sum -= prompt;
                rank.cached_sum -= cached;
            }
        }
    }

    /// `rank` streamed `tokens` to the client just now.
    pub(crate) fn generated(&self, rank: i32, tokens: u32) {
        self.generated_at(rank, tokens, Instant::now());
    }

    pub(crate) fn generated_at(&self, rank: i32, tokens: u32, at: Instant) {
        if tokens == 0 {
            return;
        }
        let mut inner = self.lock();
        let rank = inner.ranks.entry(rank).or_default();
        rank.generated.push_back((at, u64::from(tokens)));
        rank.generated_sum += u64::from(tokens);
        rank.trim(at);
    }

    /// A request ended (or was aborted) without the engine starting it.
    pub(crate) fn finished(&self, request_id: &str) {
        drop_pending(&mut self.lock(), request_id);
    }

    /// `rank`'s estimate now, given the engine's own count of that rank's
    /// waiting requests.
    pub(crate) fn estimate(&self, rank: i32, num_waiting_reqs: i32) -> LoadEstimate {
        self.estimate_at(rank, num_waiting_reqs, Instant::now())
    }

    pub(crate) fn estimate_at(
        &self,
        rank: i32,
        num_waiting_reqs: i32,
        now: Instant,
    ) -> LoadEstimate {
        let mut inner = self.lock();
        let (hit_rate, gen_throughput) = match inner.ranks.get_mut(&rank) {
            Some(figures) => {
                figures.trim(now);
                (
                    figures.hit_rate(),
                    figures.generated_sum as f64 / THROUGHPUT_WINDOW.as_secs_f64(),
                )
            }
            None => (0.0, 0.0),
        };
        let waiting = usize::try_from(num_waiting_reqs).unwrap_or(0);
        let queued: f64 = inner
            .pending
            .iter()
            .rev()
            .filter(|pending| pending.rank == rank)
            .take(waiting)
            .map(|pending| f64::from(pending.prompt_tokens) * (1.0 - hit_rate))
            .sum();
        LoadEstimate {
            queued_token_work: queued.round().clamp(0.0, f64::from(i32::MAX)) as i32,
            gen_throughput,
            cache_hit_rate: hit_rate,
        }
    }

    /// Submitted requests the engine has not started.
    #[cfg(test)]
    pub(crate) fn pending(&self) -> usize {
        self.lock().pending.len()
    }
}

fn drop_pending(inner: &mut Inner, request_id: &str) {
    if let Some(index) = inner
        .pending
        .iter()
        .position(|pending| pending.request_id == request_id)
    {
        inner.pending.remove(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_token_work_is_the_youngest_waiting_prompts_discounted_by_the_hit_rate() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        assert_eq!(tracker.estimate_at(0, 5, now), LoadEstimate::default());

        // Three in flight, none started: the engine says two are waiting, so
        // the oldest is in prefill and the two youngest are queued work.
        tracker.submitted(0, "a", 1_000);
        tracker.submitted(0, "b", 300);
        tracker.submitted(0, "c", 200);
        assert_eq!(tracker.estimate_at(0, 2, now).queued_token_work, 500);
        assert_eq!(tracker.estimate_at(0, 0, now).queued_token_work, 0);
        assert_eq!(
            tracker.estimate_at(0, 10, now).queued_token_work,
            1_500,
            "never more than we hold"
        );

        // The first output of `a`: half its prompt was cached. Queued prompts
        // are discounted by that rate.
        tracker.first_output(0, "a", 1_000, 500);
        assert_eq!(tracker.pending(), 2);
        let estimate = tracker.estimate_at(0, 2, now);
        assert_eq!(estimate.cache_hit_rate, 0.5);
        assert_eq!(estimate.queued_token_work, 250);

        // A request that ends before starting leaves the queue.
        tracker.finished("b");
        assert_eq!(tracker.estimate_at(0, 2, now).queued_token_work, 100);
        tracker.first_output(0, "c", 200, 200);
        assert_eq!(tracker.pending(), 0);
        assert_eq!(tracker.estimate_at(0, 3, now).queued_token_work, 0);
        // 700 cached of 1_200 prompt tokens.
        assert!((tracker.estimate_at(0, 0, now).cache_hit_rate - 700.0 / 1_200.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_counts_tokens_inside_the_window_only() {
        let tracker = LoadTracker::default();
        let start = Instant::now();
        tracker.generated_at(0, 1_000, start);
        tracker.generated_at(0, 3_000, start + Duration::from_secs(1));
        let per_second = |estimate: LoadEstimate| estimate.gen_throughput;
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(1))),
            2_000.0
        );
        // Two seconds after the first sample it falls out of the window.
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_millis(2_500))),
            1_500.0
        );
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(4))),
            0.0
        );
        tracker.generated_at(0, 0, start + Duration::from_secs(4));
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(4))),
            0.0
        );
    }

    #[test]
    fn the_hit_rate_averages_recent_first_outputs_and_ignores_empty_prompts() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        tracker.submitted(0, "x", 0);
        tracker.first_output(0, "x", 0, 0);
        assert_eq!(tracker.estimate_at(0, 0, now).cache_hit_rate, 0.0);
        for index in 0..HIT_RATE_SAMPLES {
            tracker.first_output(0, &format!("old-{index}"), 100, 0);
        }
        assert_eq!(tracker.estimate_at(0, 0, now).cache_hit_rate, 0.0);
        for index in 0..HIT_RATE_SAMPLES {
            tracker.first_output(0, &format!("new-{index}"), 100, 100);
        }
        assert_eq!(
            tracker.estimate_at(0, 0, now).cache_hit_rate,
            1.0,
            "the old samples aged out"
        );
        // Cached can never exceed the prompt.
        tracker.first_output(0, "odd", 10, 50);
        assert!(tracker.estimate_at(0, 0, now).cache_hit_rate <= 1.0);
    }

    /// Two ranks serve: each rank's hit rate, throughput and queued work are
    /// its own, and a rank that served nothing reads as idle, not as its
    /// peer.
    #[test]
    fn each_rank_keeps_its_own_hit_rate_throughput_and_queue() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        tracker.submitted(0, "a", 1_000);
        tracker.submitted(1, "b", 400);
        tracker.submitted(1, "c", 600);
        tracker.first_output(0, "a", 1_000, 0);
        tracker.first_output(1, "b", 400, 400);
        tracker.generated_at(0, 100, now);
        tracker.generated_at(1, 900, now);

        let rank0 = tracker.estimate_at(0, 1, now);
        let rank1 = tracker.estimate_at(1, 1, now);
        assert_eq!((rank0.cache_hit_rate, rank1.cache_hit_rate), (0.0, 1.0));
        assert_eq!((rank0.gen_throughput, rank1.gen_throughput), (50.0, 450.0));
        // Rank 0 has nothing waiting; rank 1's `c` waits behind `b`, with
        // rank 1's own hit rate discounting it to nothing.
        assert_eq!((rank0.queued_token_work, rank1.queued_token_work), (0, 0));
        tracker.first_output(1, "b", 400, 0);
        assert_eq!(tracker.estimate_at(1, 1, now).queued_token_work, 300);
        assert_eq!(tracker.estimate_at(2, 1, now), LoadEstimate::default());
    }
}
