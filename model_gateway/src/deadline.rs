//! Deadlines from configured seconds.
//!
//! A flag gives a number of seconds as any `u64`, and "never" is often
//! written as the largest one. The clock cannot hold the sum of such a
//! number and now, and a plain `+` panics on it, in the task that made the
//! sum: a request, a streamed answer, the shutdown. The sums here saturate
//! instead.

use std::time::Duration;

use tokio::time::Instant;

/// Where a wait the clock cannot hold ends: thirty years on, as
/// `tokio::time::sleep` ends a wait past the clock's range.
const FAR_FUTURE: Duration = Duration::from_secs(86_400 * 365 * 30);

/// `from + wait`, saturated to the far future when the sum does not fit an
/// instant. A wait the clock holds ends where the plain sum ends.
pub(crate) fn deadline_after(from: Instant, wait: Duration) -> Instant {
    from.checked_add(wait).unwrap_or_else(|| from + FAR_FUTURE)
}

/// `now + wait`, saturated as [`deadline_after`] saturates.
pub(crate) fn deadline_from_now(wait: Duration) -> Instant {
    deadline_after(Instant::now(), wait)
}

#[cfg(test)]
mod tests {
    use super::*;

    const YEAR: u64 = 86_400 * 365;

    /// A wait the clock holds ends where the plain sum ends, one past the
    /// far future too.
    #[tokio::test(start_paused = true)]
    async fn a_wait_the_clock_holds_ends_where_the_plain_sum_ends() {
        let now = Instant::now();
        for secs in [0, 1, 3600, 10 * YEAR, 40 * YEAR] {
            let wait = Duration::from_secs(secs);
            assert_eq!(deadline_after(now, wait), now + wait, "{secs}");
            assert_eq!(deadline_from_now(wait), now + wait, "{secs}");
        }
    }

    /// A wait the clock cannot hold ends thirty years on: the largest
    /// number, an operator's "never", and 2^63, the first that a signed
    /// count of seconds cannot hold.
    #[tokio::test(start_paused = true)]
    async fn a_wait_past_the_clocks_range_ends_at_the_far_future() {
        let now = Instant::now();
        let far_future = now + Duration::from_secs(30 * YEAR);
        for secs in [u64::MAX, 1 << 63] {
            let wait = Duration::from_secs(secs);
            assert_eq!(deadline_after(now, wait), far_future, "{secs}");
            assert_eq!(deadline_from_now(wait), far_future, "{secs}");
        }
    }

    /// The far future starts where the clock's range ends and no sooner:
    /// the longest wait the clock holds from now ends at the plain sum, and
    /// a second more ends thirty years on.
    #[tokio::test(start_paused = true)]
    async fn the_far_future_starts_where_the_clocks_range_ends() {
        let now = Instant::now();
        // The longest whole number of seconds the clock holds from now,
        // found by halving.
        let (mut held, mut past) = (0, u64::MAX);
        while past - held > 1 {
            let middle = held + (past - held) / 2;
            if now.checked_add(Duration::from_secs(middle)).is_some() {
                held = middle;
            } else {
                past = middle;
            }
        }
        let (held, past) = (Duration::from_secs(held), Duration::from_secs(past));
        assert_eq!(deadline_after(now, held), now + held);
        assert_eq!(
            deadline_after(now, past),
            now + Duration::from_secs(30 * YEAR)
        );
    }
}
