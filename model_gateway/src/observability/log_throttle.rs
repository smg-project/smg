//! A per-callsite cap on log lines written for a sustained condition.
//!
//! A refusal the gateway answers thousands of times per second (every worker
//! of a model down while a client retries at once) is one condition, not
//! thousands of events: the counters carry the volume, and the log needs a
//! few lines per second to show that the condition is still on, not a line
//! per request. [`LogThrottle`] admits at most `limit` lines per `window`
//! and counts the rest; the count rides on the next admitted line as its
//! `suppressed` figure, so a reader sees how much the window hid.
//! [`KeyedLogThrottle`] keeps one such allowance per key (a model id), so one
//! subject's storm never silences another's first line.

use std::{
    collections::VecDeque,
    sync::{Mutex, PoisonError},
    time::{Duration, Instant},
};

/// At most `limit` admissions per `window`, the rest counted.
///
/// Windows are fixed-length and anchored on the first call at or after the
/// previous window's end, so a burst that starts mid-window still gets the
/// full allowance. One short critical section per call; built `const`, so a
/// callsite holds its throttle in a `static`.
pub struct LogThrottle {
    limit: u32,
    window: Duration,
    state: Mutex<State>,
}

struct State {
    /// Start of the current window; `None` until the first call.
    window_start: Option<Instant>,
    /// Lines admitted in the current window.
    admitted: u32,
    /// Calls suppressed since the last admitted line, across windows.
    suppressed: u64,
}

impl State {
    const NEW: State = State {
        window_start: None,
        admitted: 0,
        suppressed: 0,
    };

    /// One call at `now` under `limit` per `window`: admitted with the count
    /// suppressed since the previous admitted call, or counted.
    fn admit(&mut self, limit: u32, window: Duration, now: Instant) -> Option<Admitted> {
        let window_over = self
            .window_start
            .is_none_or(|start| now.saturating_duration_since(start) >= window);
        if window_over {
            self.window_start = Some(now);
            self.admitted = 0;
        }
        if self.admitted < limit {
            self.admitted += 1;
            let suppressed = std::mem::take(&mut self.suppressed);
            Some(Admitted { suppressed })
        } else {
            self.suppressed += 1;
            None
        }
    }
}

/// An admitted line, with the number of calls the throttle suppressed since
/// the previous admitted one: the figure the line reports so the suppressed
/// volume is not lost. Zero while the callsite stays under its limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admitted {
    pub suppressed: u64,
}

impl LogThrottle {
    /// A throttle admitting `limit` lines per `window`.
    pub const fn new(limit: u32, window: Duration) -> Self {
        Self {
            limit,
            window,
            state: Mutex::new(State::NEW),
        }
    }

    /// Admit a line now, or count it as suppressed.
    pub fn admit(&self) -> Option<Admitted> {
        self.admit_at(Instant::now())
    }

    /// [`Self::admit`] at an explicit instant.
    pub fn admit_at(&self, now: Instant) -> Option<Admitted> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .admit(self.limit, self.window, now)
    }
}

/// A [`LogThrottle`] per key, for a line about one of many subjects (a
/// model id): one subject's storm never consumes another's allowance, and a
/// line's `suppressed` count is its own subject's. The key set is bounded:
/// past `max_keys` the key seen least recently is evicted with its state, so
/// a stream of unique names cannot grow it; a key that returns after an
/// eviction opens a fresh window. Built `const`, so a callsite holds its
/// throttle in a `static`; the lookup is a scan of at most `max_keys`
/// entries under one short critical section.
pub struct KeyedLogThrottle {
    limit: u32,
    window: Duration,
    max_keys: usize,
    /// The keys in order of last use, the most recent at the back.
    state: Mutex<VecDeque<(String, State)>>,
}

impl KeyedLogThrottle {
    /// A throttle admitting `limit` lines per `window` for each of at most
    /// `max_keys` keys.
    pub const fn new(limit: u32, window: Duration, max_keys: usize) -> Self {
        Self {
            limit,
            window,
            max_keys,
            state: Mutex::new(VecDeque::new()),
        }
    }

    /// Admit a line about `key` now, or count it as suppressed for `key`.
    pub fn admit(&self, key: &str) -> Option<Admitted> {
        self.admit_at(key, Instant::now())
    }

    /// [`Self::admit`] at an explicit instant.
    pub fn admit_at(&self, key: &str, now: Instant) -> Option<Admitted> {
        let mut keys = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let mut entry = match keys.iter().position(|(known, _)| known == key) {
            Some(index) => keys.remove(index),
            None => {
                if keys.len() >= self.max_keys.max(1) {
                    keys.pop_front();
                }
                None
            }
        }
        .unwrap_or_else(|| (key.to_owned(), State::NEW));
        let admitted = entry.1.admit(self.limit, self.window, now);
        // The key just used goes to the back: the front is the eviction end.
        keys.push_back(entry);
        admitted
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/log_throttle/tests.rs"]
mod tests;
