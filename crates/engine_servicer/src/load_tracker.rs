//! What a servicer can tell about its engine's load from the requests it
//! forwards: queued token-work, generation throughput and the prefix-cache
//! hit rate. The gateway's expected-wait score reads them from `GetLoads`;
//! an engine whose stats do not carry them (vLLM reports running, waiting
//! and KV usage) would otherwise be scored on defaults and routed unlike its
//! peers.
//!
//! The servicer sees every request it submits, the engine's first output for
//! it (with the prompt and cached token counts) and every token streamed, so:
//!
//! - queued token-work is the uncached prompt tokens of the requests the
//!   engine has not started: of the submitted requests without a first
//!   output, the youngest `num_waiting_reqs` (the engine admits FCFS, so the
//!   older ones are the ones in prefill), each prompt discounted by the
//!   recent hit rate since its own cached count is not known until it starts;
//! - generation throughput is the tokens streamed over the last
//!   [`THROUGHPUT_WINDOW`];
//! - the hit rate is cached over prompt tokens across the last
//!   [`HIT_RATE_SAMPLES`] first outputs.
//!
//! These are the field semantics the SGLang servicer reports from its
//! scheduler and the mock engine from its queue.

use std::{
    collections::VecDeque,
    sync::{Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

/// How far back streamed tokens count toward the throughput.
pub(crate) const THROUGHPUT_WINDOW: Duration = Duration::from_secs(2);
/// How many first outputs the hit rate averages over.
pub(crate) const HIT_RATE_SAMPLES: usize = 64;

/// The three fields, as `GetLoads` reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LoadEstimate {
    /// Uncached prompt tokens of the requests the engine has not started.
    pub queued_token_work: i32,
    /// Tokens streamed per second over the window.
    pub gen_throughput: f64,
    /// Cached over prompt tokens of recent first outputs, in `[0, 1]`.
    pub cache_hit_rate: f64,
}

struct Pending {
    request_id: String,
    prompt_tokens: u32,
}

#[derive(Default)]
struct Inner {
    /// Submitted requests without a first output yet, oldest first.
    pending: VecDeque<Pending>,
    /// Recent first outputs: (prompt tokens, cached tokens).
    hits: VecDeque<(u64, u64)>,
    prompt_sum: u64,
    cached_sum: u64,
    /// Tokens streamed, by time, within the window.
    generated: VecDeque<(Instant, u64)>,
    generated_sum: u64,
}

impl Inner {
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

/// Per-servicer load bookkeeping; every method is cheap and lock-scoped.
#[derive(Default)]
pub(crate) struct LoadTracker {
    inner: Mutex<Inner>,
}

impl LoadTracker {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A request with `prompt_tokens` was handed to the engine.
    pub(crate) fn submitted(&self, request_id: &str, prompt_tokens: u32) {
        self.lock().pending.push_back(Pending {
            request_id: request_id.to_string(),
            prompt_tokens,
        });
    }

    /// The engine's first output for a request: it has started, and its
    /// prompt had `cached_tokens` of `prompt_tokens` in the prefix cache.
    pub(crate) fn first_output(&self, request_id: &str, prompt_tokens: u32, cached_tokens: u32) {
        let mut inner = self.lock();
        if let Some(index) = inner
            .pending
            .iter()
            .position(|pending| pending.request_id == request_id)
        {
            inner.pending.remove(index);
        }
        if prompt_tokens == 0 {
            return;
        }
        let sample = (
            u64::from(prompt_tokens),
            u64::from(cached_tokens.min(prompt_tokens)),
        );
        inner.hits.push_back(sample);
        inner.prompt_sum += sample.0;
        inner.cached_sum += sample.1;
        if inner.hits.len() > HIT_RATE_SAMPLES {
            if let Some((prompt, cached)) = inner.hits.pop_front() {
                inner.prompt_sum -= prompt;
                inner.cached_sum -= cached;
            }
        }
    }

    /// `tokens` were streamed to the client just now.
    pub(crate) fn generated(&self, tokens: u32) {
        self.generated_at(tokens, Instant::now());
    }

    pub(crate) fn generated_at(&self, tokens: u32, at: Instant) {
        if tokens == 0 {
            return;
        }
        let mut inner = self.lock();
        inner.generated.push_back((at, u64::from(tokens)));
        inner.generated_sum += u64::from(tokens);
        inner.trim(at);
    }

    /// A request ended (or was aborted) without the engine starting it.
    pub(crate) fn finished(&self, request_id: &str) {
        let mut inner = self.lock();
        if let Some(index) = inner
            .pending
            .iter()
            .position(|pending| pending.request_id == request_id)
        {
            inner.pending.remove(index);
        }
    }

    /// The estimate now, given the engine's own count of waiting requests.
    pub(crate) fn estimate(&self, num_waiting_reqs: i32) -> LoadEstimate {
        self.estimate_at(num_waiting_reqs, Instant::now())
    }

    pub(crate) fn estimate_at(&self, num_waiting_reqs: i32, now: Instant) -> LoadEstimate {
        let mut inner = self.lock();
        inner.trim(now);
        let hit_rate = inner.hit_rate();
        let waiting = usize::try_from(num_waiting_reqs)
            .unwrap_or(0)
            .min(inner.pending.len());
        let queued: f64 = inner
            .pending
            .iter()
            .rev()
            .take(waiting)
            .map(|pending| f64::from(pending.prompt_tokens) * (1.0 - hit_rate))
            .sum();
        LoadEstimate {
            queued_token_work: queued.round().clamp(0.0, f64::from(i32::MAX)) as i32,
            gen_throughput: inner.generated_sum as f64 / THROUGHPUT_WINDOW.as_secs_f64(),
            cache_hit_rate: hit_rate,
        }
    }

    /// Submitted requests the engine has not started.
    #[cfg(test)]
    pub(crate) fn pending(&self) -> usize {
        self.lock().pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_token_work_is_the_youngest_waiting_prompts_discounted_by_the_hit_rate() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        assert_eq!(tracker.estimate_at(5, now), LoadEstimate::default());

        // Three in flight, none started: the engine says two are waiting, so
        // the oldest is in prefill and the two youngest are queued work.
        tracker.submitted("a", 1_000);
        tracker.submitted("b", 300);
        tracker.submitted("c", 200);
        assert_eq!(tracker.estimate_at(2, now).queued_token_work, 500);
        assert_eq!(tracker.estimate_at(0, now).queued_token_work, 0);
        assert_eq!(
            tracker.estimate_at(10, now).queued_token_work,
            1_500,
            "never more than we hold"
        );

        // The first output of `a`: half its prompt was cached. Queued prompts
        // are discounted by that rate.
        tracker.first_output("a", 1_000, 500);
        assert_eq!(tracker.pending(), 2);
        let estimate = tracker.estimate_at(2, now);
        assert_eq!(estimate.cache_hit_rate, 0.5);
        assert_eq!(estimate.queued_token_work, 250);

        // A request that ends before starting leaves the queue.
        tracker.finished("b");
        assert_eq!(tracker.estimate_at(2, now).queued_token_work, 100);
        tracker.first_output("c", 200, 200);
        assert_eq!(tracker.pending(), 0);
        assert_eq!(tracker.estimate_at(3, now).queued_token_work, 0);
        // 700 cached of 1_200 prompt tokens.
        assert!((tracker.estimate_at(0, now).cache_hit_rate - 700.0 / 1_200.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_counts_tokens_inside_the_window_only() {
        let tracker = LoadTracker::default();
        let start = Instant::now();
        tracker.generated_at(1_000, start);
        tracker.generated_at(3_000, start + Duration::from_secs(1));
        let per_second = |estimate: LoadEstimate| estimate.gen_throughput;
        assert_eq!(
            per_second(tracker.estimate_at(0, start + Duration::from_secs(1))),
            2_000.0
        );
        // Two seconds after the first sample it falls out of the window.
        assert_eq!(
            per_second(tracker.estimate_at(0, start + Duration::from_millis(2_500))),
            1_500.0
        );
        assert_eq!(
            per_second(tracker.estimate_at(0, start + Duration::from_secs(4))),
            0.0
        );
        tracker.generated_at(0, start + Duration::from_secs(4));
        assert_eq!(
            per_second(tracker.estimate_at(0, start + Duration::from_secs(4))),
            0.0
        );
    }

    #[test]
    fn the_hit_rate_averages_recent_first_outputs_and_ignores_empty_prompts() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        tracker.submitted("x", 0);
        tracker.first_output("x", 0, 0);
        assert_eq!(tracker.estimate_at(0, now).cache_hit_rate, 0.0);
        for index in 0..HIT_RATE_SAMPLES {
            tracker.first_output(&format!("old-{index}"), 100, 0);
        }
        assert_eq!(tracker.estimate_at(0, now).cache_hit_rate, 0.0);
        for index in 0..HIT_RATE_SAMPLES {
            tracker.first_output(&format!("new-{index}"), 100, 100);
        }
        assert_eq!(
            tracker.estimate_at(0, now).cache_hit_rate,
            1.0,
            "the old samples aged out"
        );
        // Cached can never exceed the prompt.
        tracker.first_output("odd", 10, 50);
        assert!(tracker.estimate_at(0, now).cache_hit_rate <= 1.0);
    }
}
