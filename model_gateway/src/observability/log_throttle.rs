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
mod tests {
    use super::*;

    const WINDOW: Duration = Duration::from_secs(1);

    /// A second of refusals writes the configured lines and no more; the
    /// first line of the next window carries the count of what was hidden,
    /// once.
    #[test]
    fn a_burst_in_one_window_emits_the_limit_and_the_next_line_carries_the_suppressed_count() {
        let throttle = LogThrottle::new(5, WINDOW);
        let t0 = Instant::now();

        let admitted = (0..1000u64)
            .filter(|i| {
                throttle
                    .admit_at(t0 + Duration::from_micros(i * 500))
                    .is_some()
            })
            .count();
        assert_eq!(admitted, 5, "the window admits exactly its limit");

        let first_of_next = throttle
            .admit_at(t0 + WINDOW)
            .expect("the next window admits again");
        assert_eq!(
            first_of_next.suppressed, 995,
            "the summary names every line the burst hid"
        );
        let second_of_next = throttle
            .admit_at(t0 + WINDOW + Duration::from_millis(1))
            .expect("still inside the new window's allowance");
        assert_eq!(second_of_next.suppressed, 0, "the summary is reported once");
    }

    /// Under the limit nothing is held back and no summary appears.
    #[test]
    fn lines_under_the_limit_are_all_admitted_with_nothing_suppressed() {
        let throttle = LogThrottle::new(5, WINDOW);
        let t0 = Instant::now();
        for window in 0..3u32 {
            for line in 0..3u32 {
                let at = t0 + WINDOW * window + Duration::from_millis(u64::from(line) * 100);
                assert_eq!(throttle.admit_at(at), Some(Admitted { suppressed: 0 }));
            }
        }
    }

    /// A window that fills, then silence: the next call, however much later,
    /// opens a fresh window with the whole allowance and reports the backlog.
    #[test]
    fn a_quiet_gap_starts_a_fresh_window_with_the_full_allowance() {
        let throttle = LogThrottle::new(2, WINDOW);
        let t0 = Instant::now();
        assert!(throttle.admit_at(t0).is_some());
        assert!(throttle.admit_at(t0).is_some());
        assert!(throttle.admit_at(t0).is_none());
        assert!(throttle.admit_at(t0 + Duration::from_millis(999)).is_none());

        let later = t0 + Duration::from_secs(10);
        assert_eq!(throttle.admit_at(later), Some(Admitted { suppressed: 2 }));
        assert_eq!(throttle.admit_at(later), Some(Admitted { suppressed: 0 }));
        assert!(throttle.admit_at(later).is_none());
    }

    /// A clock that reads before the window's start is treated as inside it.
    #[test]
    fn an_earlier_clock_reading_stays_in_the_current_window() {
        let throttle = LogThrottle::new(1, WINDOW);
        let t0 = Instant::now() + Duration::from_secs(5);
        assert!(throttle.admit_at(t0).is_some());
        assert!(throttle.admit_at(t0 - Duration::from_secs(1)).is_none());
    }

    /// Two models refused in the same second each get the whole allowance,
    /// and each one's summary counts only its own hidden lines.
    #[test]
    fn each_key_has_its_own_allowance_and_its_own_suppressed_count() {
        let throttle = KeyedLogThrottle::new(2, WINDOW, 64);
        let t0 = Instant::now();
        for _ in 0..10 {
            throttle.admit_at("a", t0);
        }
        assert_eq!(
            throttle.admit_at("b", t0),
            Some(Admitted { suppressed: 0 }),
            "a's storm leaves b's first line untouched"
        );
        assert!(throttle.admit_at("b", t0).is_some());
        assert!(throttle.admit_at("b", t0).is_none());

        let next = t0 + WINDOW;
        assert_eq!(
            throttle.admit_at("a", next),
            Some(Admitted { suppressed: 8 })
        );
        assert_eq!(
            throttle.admit_at("b", next),
            Some(Admitted { suppressed: 1 })
        );
    }

    /// Past the key bound the key seen least recently goes, and comes back
    /// with a fresh window when it returns.
    #[test]
    fn the_key_set_is_bounded_and_evicts_the_least_recently_seen_key() {
        let throttle = KeyedLogThrottle::new(1, WINDOW, 2);
        let t0 = Instant::now();
        assert!(throttle.admit_at("a", t0).is_some());
        assert!(throttle.admit_at("b", t0).is_some());
        assert!(throttle.admit_at("a", t0).is_none(), "a's window is full");
        // a was used after b, so c evicts b.
        assert!(throttle.admit_at("c", t0).is_some());
        assert!(
            throttle.admit_at("b", t0).is_some(),
            "b returns as a new key with a fresh allowance"
        );
        assert!(throttle.admit_at("c", t0).is_none(), "c's window is full");
        assert!(
            throttle.admit_at("a", t0 + WINDOW).is_some(),
            "a was evicted by b's return and starts afresh"
        );
        assert_eq!(
            throttle.admit_at("a", t0 + WINDOW),
            None,
            "a fresh key still has one line per window"
        );
    }
}
