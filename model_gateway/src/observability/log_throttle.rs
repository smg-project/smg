//! A per-callsite cap on log lines written for a sustained condition.
//!
//! A refusal the gateway answers thousands of times per second (every worker
//! of a model down while a client retries at once) is one condition, not
//! thousands of events: the counters carry the volume, and the log needs a
//! few lines per second to show that the condition is still on, not a line
//! per request. [`LogThrottle`] admits at most `limit` lines per `window`
//! and counts the rest; the count rides on the next admitted line as its
//! `suppressed` figure, so a reader sees how much the window hid.

use std::{
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
            state: Mutex::new(State {
                window_start: None,
                admitted: 0,
                suppressed: 0,
            }),
        }
    }

    /// Admit a line now, or count it as suppressed.
    pub fn admit(&self) -> Option<Admitted> {
        self.admit_at(Instant::now())
    }

    /// [`Self::admit`] at an explicit instant.
    pub fn admit_at(&self, now: Instant) -> Option<Admitted> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let window_over = state
            .window_start
            .is_none_or(|start| now.saturating_duration_since(start) >= self.window);
        if window_over {
            state.window_start = Some(now);
            state.admitted = 0;
        }
        if state.admitted < self.limit {
            state.admitted += 1;
            let suppressed = std::mem::take(&mut state.suppressed);
            Some(Admitted { suppressed })
        } else {
            state.suppressed += 1;
            None
        }
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
}
