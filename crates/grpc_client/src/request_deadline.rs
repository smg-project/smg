//! The deadline of one request to a worker: the gateway's `--request-timeout-secs`,
//! counted from the request's dispatch, over the call that opens the stream
//! and over every message on it. The gateway arms it on the stream it got
//! back ([`crate::AbortOnDropStream::with_deadline`]); when it passes, the
//! stream is dropped the way a client that left drops it, which sends the
//! abort to the worker, and the reader meets one status that names the
//! deadline ([`RequestDeadline::status`]), told from a worker's own
//! `DEADLINE_EXCEEDED` by [`is_request_timeout`]. Nothing of it goes on the
//! wire: the worker learns of the deadline only through the abort.

use std::time::Duration;

use tokio::time::Instant;
use tonic::{metadata::MetadataValue, Code, Status};

/// The metadata key the deadline's status carries, by which the gateway tells
/// its own cut from a status the worker answered; its value is the timeout
/// in whole seconds.
const REQUEST_TIMEOUT_METADATA_KEY: &str = "smg-request-timeout";

/// The deadline of one request to a worker: `timeout` after its dispatch.
#[derive(Clone, Copy, Debug)]
pub struct RequestDeadline {
    at: Instant,
    timeout: Duration,
}

impl RequestDeadline {
    /// The deadline `timeout` from now; one the clock cannot add never passes.
    #[must_use]
    pub fn after(timeout: Duration) -> Self {
        let now = Instant::now();
        let at = now.checked_add(timeout).unwrap_or_else(far_future);
        Self { at, timeout }
    }

    /// The instant the deadline passes.
    #[must_use]
    pub fn at(self) -> Instant {
        self.at
    }

    /// The timeout the deadline was made from.
    #[must_use]
    pub fn timeout(self) -> Duration {
        self.timeout
    }

    /// The status a request ends with when its deadline passes: a
    /// `DEADLINE_EXCEEDED` of the gateway's own, marked as such (see
    /// [`is_request_timeout`]).
    #[must_use]
    pub fn status(self) -> Status {
        let mut status = Status::deadline_exceeded(format!(
            "request timed out after {:?}: the worker did not finish it within the request timeout",
            self.timeout
        ));
        if let Ok(value) = MetadataValue::try_from(self.timeout.as_secs().to_string()) {
            status
                .metadata_mut()
                .insert(REQUEST_TIMEOUT_METADATA_KEY, value);
        }
        status
    }
}

/// Whether `status` is the gateway's own request timeout (see
/// [`RequestDeadline::status`]) and not a worker's `DEADLINE_EXCEEDED`.
#[must_use]
pub fn is_request_timeout(status: &Status) -> bool {
    status.code() == Code::DeadlineExceeded
        && status.metadata().contains_key(REQUEST_TIMEOUT_METADATA_KEY)
}

/// An instant the clock cannot reach: a deadline that never passes.
fn far_future() -> Instant {
    // tokio's timers reach about two years ahead; a longer sleep is clamped
    // there, which is as never as this needs.
    Instant::now() + Duration::from_secs(365 * 24 * 60 * 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_deadlines_status_is_told_from_a_workers_deadline_exceeded() {
        let deadline = RequestDeadline::after(Duration::from_secs(15));
        let status = deadline.status();
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert!(status.message().starts_with("request timed out after 15s"));
        assert!(is_request_timeout(&status));
        assert_eq!(deadline.timeout(), Duration::from_secs(15));
        assert!(!is_request_timeout(&Status::deadline_exceeded(
            "engine scheduling deadline exceeded"
        )));
        assert!(!is_request_timeout(&Status::cancelled("Timeout expired")));
    }

    #[tokio::test]
    async fn a_deadline_the_clock_cannot_add_never_passes() {
        let deadline = RequestDeadline::after(Duration::MAX);
        assert!(deadline.at() > Instant::now() + Duration::from_secs(60 * 60));
    }
}
