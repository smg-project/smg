//! Optimistic self-accounting: what the router has already decided but the engines have not yet
//! reported.
//!
//! Between a dispatch and the engine's first KV event (or load report) the index and the load
//! snapshot are stale by exactly that request. Under a burst of sibling requests (best-of-N, agent
//! fan-out, replayed sessions) that window is enough to herd them all onto one worker. This keeps
//! two short-lived views, in the spirit of a predict-on-route side index and an active-sequence
//! booking:
//!
//! - **booked prefill**: the predicted uncached tokens of each dispatched request, charged to the
//!   chosen worker until it completes or the booking expires;
//! - **predicted placement**: the prefix of each dispatched request and the worker that will hold
//!   its blocks. A prefix is keyed by its chain hash at block depths `1, 2, 4, 8, …` and at its full
//!   length, so a later request sharing `d` leading blocks finds the placement at the deepest power
//!   of two not above `d` with `O(log d)` lookups and no per-block index.
//!
//! Both expire after `ttl`, which should be a little longer than the engine's event lag. Output
//! blocks can be credited as generation crosses block boundaries through `on_output_blocks`.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

#[derive(Debug, Clone, Copy)]
struct Booking {
    tokens: u64,
    expires: Instant,
}

#[derive(Debug, Clone)]
struct Placement {
    url: Arc<str>,
    expires: Instant,
}

#[derive(Debug, Default)]
struct State {
    booked: HashMap<Arc<str>, VecDeque<Booking>>,
    output_blocks: HashMap<Arc<str>, f64>,
    /// Prefix chain hash → workers predicted to hold that prefix, newest last.
    predicted: HashMap<u64, Vec<Placement>>,
    /// Insertion order of prediction keys, for bounded eviction.
    predicted_order: VecDeque<(u64, Instant)>,
}

/// Bound on remembered prediction keys; the oldest go first once reached.
const MAX_PREDICTED_KEYS: usize = 1 << 16;
/// Workers remembered per prefix key (a prefix spilled to several workers).
const MAX_PLACEMENTS_PER_KEY: usize = 4;

#[derive(Debug)]
pub struct OptimisticAccounting {
    ttl: Duration,
    state: Mutex<State>,
}

/// Block positions (zero-based) at which a prefix of `blocks` blocks is keyed: block counts
/// `1, 2, 4, …` below `blocks`, then `blocks` itself.
fn key_positions(blocks: usize) -> impl Iterator<Item = usize> {
    let mut count = 1usize;
    std::iter::from_fn(move || {
        if count < blocks {
            let position = count - 1;
            count *= 2;
            Some(position)
        } else if count == usize::MAX {
            None
        } else {
            count = usize::MAX;
            blocks.checked_sub(1)
        }
    })
}

impl OptimisticAccounting {
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            state: Mutex::new(State::default()),
        }
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    fn intern(state: &State, url: &str) -> Arc<str> {
        state
            .booked
            .get_key_value(url)
            .map(|(key, _)| Arc::clone(key))
            .unwrap_or_else(|| Arc::from(url))
    }

    /// Book a dispatch: `uncached_tokens` of prefill on `url`, and the prompt's prefix
    /// (`prefix_hashes`, chain hashes by block) predicted to become resident there.
    pub fn record_dispatch(&self, url: &str, uncached_tokens: u64, prefix_hashes: &[u64]) {
        let now = Instant::now();
        let expires = now + self.ttl;
        let mut state = self.state.lock();
        let key = Self::intern(&state, url);
        let queue = state.booked.entry(Arc::clone(&key)).or_default();
        while queue.front().is_some_and(|b| b.expires <= now) {
            queue.pop_front();
        }
        queue.push_back(Booking {
            tokens: uncached_tokens,
            expires,
        });
        if prefix_hashes.is_empty() {
            return;
        }
        let State {
            predicted,
            predicted_order,
            ..
        } = &mut *state;
        for position in key_positions(prefix_hashes.len()) {
            let hash = prefix_hashes[position];
            let placements = predicted.entry(hash).or_default();
            if placements.is_empty() {
                predicted_order.push_back((hash, expires));
            }
            placements.retain(|p| p.expires > now);
            match placements.iter_mut().find(|p| p.url == key) {
                Some(existing) => existing.expires = expires,
                None => {
                    if placements.len() == MAX_PLACEMENTS_PER_KEY {
                        placements.remove(0);
                    }
                    placements.push(Placement {
                        url: Arc::clone(&key),
                        expires,
                    });
                }
            }
        }
        // Beyond the key bound the oldest keys go, live or not, so the walk
        // shortens the queue; below it, an expired head whose placements a
        // later dispatch re-armed goes back to the tail under its newest
        // expiry, which is not yet due, so it is not seen again this walk.
        while let Some(&(hash, expiry)) = predicted_order.front() {
            let over_bound = predicted_order.len() > MAX_PREDICTED_KEYS;
            if !over_bound && expiry > now {
                break;
            }
            predicted_order.pop_front();
            if over_bound {
                predicted.remove(&hash);
                continue;
            }
            let Some(placements) = predicted.get_mut(&hash) else {
                continue;
            };
            placements.retain(|p| p.expires > now);
            if placements.is_empty() {
                predicted.remove(&hash);
            } else {
                let latest = placements.iter().map(|p| p.expires).max().unwrap_or(now);
                predicted_order.push_back((hash, latest));
            }
        }
    }

    /// A request on `url` finished (or produced its first token): release its oldest booking.
    pub fn release(&self, url: &str) {
        let mut state = self.state.lock();
        if let Some(queue) = state.booked.get_mut(url) {
            queue.pop_front();
            if queue.is_empty() {
                state.booked.remove(url);
            }
        }
    }

    /// Trim the live bookings on `url` to the router's in-flight count there,
    /// oldest first, and return how many were released. A dispatch books once
    /// and a completion releases once, so bookings beyond the live count are
    /// completions that never arrived. Expired bookings are dropped on the way
    /// and not counted: they were already out of every sum.
    pub fn reconcile(&self, url: &str, in_flight: usize) -> usize {
        let now = Instant::now();
        let mut state = self.state.lock();
        let Some(queue) = state.booked.get_mut(url) else {
            return 0;
        };
        queue.retain(|booking| booking.expires > now);
        let excess = queue.len().saturating_sub(in_flight);
        queue.drain(..excess);
        if queue.is_empty() {
            state.booked.remove(url);
        }
        excess
    }

    /// Prefill tokens booked on `url` that have not expired or been released.
    pub fn pending_prefill_tokens(&self, url: &str) -> u64 {
        let now = Instant::now();
        let state = self.state.lock();
        state.booked.get(url).map_or(0, |queue| {
            queue
                .iter()
                .filter(|b| b.expires > now)
                .map(|b| b.tokens)
                .sum()
        })
    }

    /// Workers predicted to hold a prefix of the prompt described by `prefix_hashes`, each with the
    /// number of leading blocks predicted resident there (the deepest keyed depth that matched).
    pub fn predicted_overlaps(&self, prefix_hashes: &[u64]) -> Vec<(Arc<str>, f64)> {
        if prefix_hashes.is_empty() {
            return Vec::new();
        }
        let now = Instant::now();
        let positions: Vec<usize> = key_positions(prefix_hashes.len()).collect();
        let state = self.state.lock();
        let mut found: Vec<(Arc<str>, f64)> = Vec::new();
        for &position in positions.iter().rev() {
            let Some(placements) = state.predicted.get(&prefix_hashes[position]) else {
                continue;
            };
            for placement in placements.iter().filter(|p| p.expires > now) {
                if !found.iter().any(|(url, _)| *url == placement.url) {
                    found.push((Arc::clone(&placement.url), (position + 1) as f64));
                }
            }
        }
        found
    }

    /// Credit `blocks` of generated output on `url` (decode-side growth the engine will report
    /// later); `release_output` forgets it when the request ends.
    pub fn on_output_blocks(&self, url: &str, blocks: f64) {
        let mut state = self.state.lock();
        let key = Self::intern(&state, url);
        *state.output_blocks.entry(key).or_default() += blocks;
    }

    pub fn release_output(&self, url: &str, blocks: f64) {
        let mut state = self.state.lock();
        if let Some(current) = state.output_blocks.get_mut(url) {
            *current = (*current - blocks).max(0.0);
            if *current == 0.0 {
                state.output_blocks.remove(url);
            }
        }
    }

    pub fn output_blocks(&self, url: &str) -> f64 {
        self.state
            .lock()
            .output_blocks
            .get(url)
            .copied()
            .unwrap_or(0.0)
    }

    /// Drop every booking and prediction for a worker that left the fleet.
    pub fn forget_worker(&self, url: &str) {
        let mut state = self.state.lock();
        state.booked.remove(url);
        state.output_blocks.remove(url);
        for placements in state.predicted.values_mut() {
            placements.retain(|p| &*p.url != url);
        }
        state
            .predicted
            .retain(|_, placements| !placements.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    #[test]
    fn key_positions_are_powers_of_two_then_full_length() {
        assert_eq!(key_positions(0).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(key_positions(1).collect::<Vec<_>>(), vec![0]);
        assert_eq!(key_positions(2).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(key_positions(5).collect::<Vec<_>>(), vec![0, 1, 3, 4]);
        assert_eq!(key_positions(8).collect::<Vec<_>>(), vec![0, 1, 3, 7]);
        assert_eq!(key_positions(9).collect::<Vec<_>>(), vec![0, 1, 3, 7, 8]);
    }

    #[test]
    fn bookings_sum_until_released_or_expired() {
        let acc = OptimisticAccounting::new(Duration::from_millis(50));
        acc.record_dispatch("w1", 100, &[]);
        acc.record_dispatch("w1", 200, &[]);
        assert_eq!(acc.pending_prefill_tokens("w1"), 300);
        acc.release("w1");
        assert_eq!(acc.pending_prefill_tokens("w1"), 200);
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(acc.pending_prefill_tokens("w1"), 0);
    }

    #[test]
    fn reconcile_releases_bookings_beyond_the_live_count_oldest_first() {
        let acc = OptimisticAccounting::new(Duration::from_secs(5));
        acc.record_dispatch("w1", 100, &[]);
        acc.record_dispatch("w1", 200, &[]);
        acc.record_dispatch("w1", 400, &[]);
        assert_eq!(acc.reconcile("w1", 3), 0, "nothing beyond the live count");
        assert_eq!(acc.reconcile("w1", 1), 2, "two completions never arrived");
        assert_eq!(
            acc.pending_prefill_tokens("w1"),
            400,
            "the newest booking is the one still in flight"
        );
        assert_eq!(acc.reconcile("w1", 0), 1);
        assert_eq!(acc.pending_prefill_tokens("w1"), 0);
        assert_eq!(
            acc.reconcile("w1", 0),
            0,
            "a worker with no bookings releases nothing"
        );
    }

    #[test]
    fn predicted_overlap_matches_shared_prefixes_at_keyed_depths() {
        let acc = OptimisticAccounting::new(Duration::from_secs(5));
        let prefix: Vec<u64> = (1..=10).collect();
        acc.record_dispatch("w1", 0, &prefix);

        // A longer sibling sharing all ten blocks matches the full-length key.
        let longer: Vec<u64> = (1..=12).collect();
        let found = acc.predicted_overlaps(&longer);
        assert_eq!(found.len(), 1);
        assert_eq!(&*found[0].0, "w1");
        assert_eq!(
            found[0].1, 8.0,
            "deepest keyed depth below 12 shared with 10 is 8"
        );

        // The identical prompt gets the full ten blocks.
        let found = acc.predicted_overlaps(&prefix);
        assert_eq!(found[0].1, 10.0);

        // A prompt sharing only three blocks matches the depth-2 key.
        let short: Vec<u64> = vec![1, 2, 3, 99, 98];
        let found = acc.predicted_overlaps(&short);
        assert_eq!(found[0].1, 2.0);

        // Nothing shared, nothing predicted.
        assert!(acc.predicted_overlaps(&[70, 80, 90]).is_empty());

        // Another worker gets the same prefix: both are reported.
        acc.record_dispatch("w2", 0, &prefix);
        let mut urls: Vec<String> = acc
            .predicted_overlaps(&prefix)
            .into_iter()
            .map(|(url, _)| url.to_string())
            .collect();
        urls.sort();
        assert_eq!(urls, vec!["w1", "w2"]);

        acc.forget_worker("w1");
        let found = acc.predicted_overlaps(&prefix);
        assert_eq!(found.len(), 1);
        assert_eq!(&*found[0].0, "w2");
    }

    /// More live keys than the bound under one long TTL: the eviction walk
    /// must drop the oldest and return, not cycle a live head through the
    /// queue for ever under the lock.
    #[test]
    fn more_live_keys_than_the_bound_evict_the_oldest_and_return() {
        let acc = Arc::new(OptimisticAccounting::new(Duration::from_secs(3600)));
        let extra = 100u64;
        let total = MAX_PREDICTED_KEYS as u64 + extra;
        let (done_tx, done_rx) = mpsc::channel();
        let worker = Arc::clone(&acc);
        std::thread::spawn(move || {
            for key in 1..=total {
                worker.record_dispatch("w1", 0, &[key]);
            }
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(Duration::from_secs(60))
            .expect("record_dispatch returns with more live keys than the bound");
        let state = acc.state.lock();
        assert!(state.predicted_order.len() <= MAX_PREDICTED_KEYS);
        assert_eq!(state.predicted.len(), state.predicted_order.len());
        assert!(
            state.predicted.contains_key(&total),
            "the newest keys are the ones kept"
        );
        assert!(
            !state.predicted.contains_key(&1),
            "the oldest keys are the ones evicted"
        );
    }

    #[test]
    fn predictions_expire() {
        let acc = OptimisticAccounting::new(Duration::from_millis(30));
        acc.record_dispatch("w1", 0, &[1, 2, 3, 4]);
        assert_eq!(acc.predicted_overlaps(&[1, 2, 3, 4]).len(), 1);
        std::thread::sleep(Duration::from_millis(40));
        assert!(acc.predicted_overlaps(&[1, 2, 3, 4]).is_empty());
    }

    #[test]
    fn output_blocks_accumulate_and_release() {
        let acc = OptimisticAccounting::new(Duration::from_secs(1));
        acc.on_output_blocks("w1", 2.0);
        acc.on_output_blocks("w1", 3.0);
        assert_eq!(acc.output_blocks("w1"), 5.0);
        acc.release_output("w1", 4.0);
        assert_eq!(acc.output_blocks("w1"), 1.0);
        acc.release_output("w1", 4.0);
        assert_eq!(acc.output_blocks("w1"), 0.0);
    }
}
