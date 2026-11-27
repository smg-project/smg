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
//! Both expire after `ttl`, which should be a little longer than the engine's event lag.

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

    /// Drop every booking and prediction for a worker that left the fleet.
    pub fn forget_worker(&self, url: &str) {
        let mut state = self.state.lock();
        state.booked.remove(url);
        for placements in state.predicted.values_mut() {
            placements.retain(|p| &*p.url != url);
        }
        state
            .predicted
            .retain(|_, placements| !placements.is_empty());
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/policies/cost/accounting/tests.rs"]
mod tests;
