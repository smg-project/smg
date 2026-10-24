//! The bounds of one request to a worker, counted from its dispatch: the
//! gateway's `--request-timeout-secs` (the total: over the call that opens the
//! stream and over every message on it) and, when set, the two silence bounds,
//! `--first-token-timeout-secs` (the dispatch to the stream's first message,
//! the call that opens it included) and `--inter-token-timeout-secs` (between
//! one message and the next). The gateway arms them on the stream it got back
//! ([`crate::AbortOnDropStream::with_deadline`]); when the earliest passes, the
//! stream is dropped the way a client that left drops it, which sends the
//! abort to the worker, and the reader meets one status that names the bound
//! ([`RequestDeadline::status`]), told from a worker's own `DEADLINE_EXCEEDED`
//! by [`is_request_timeout`] and by kind through [`request_timeout_kind`].
//! Nothing of it goes on the wire: the worker learns of the bound only through
//! the abort.

use std::time::Duration;

use tokio::time::Instant;
use tonic::{metadata::MetadataValue, Code, Status};

/// The metadata key the bound's status carries, by which the gateway tells
/// its own cut from a status the worker answered; its value is the bound in
/// whole seconds.
const REQUEST_TIMEOUT_METADATA_KEY: &str = "smg-request-timeout";
/// The metadata key naming which bound passed ([`TimeoutKind::label`]).
const REQUEST_TIMEOUT_KIND_METADATA_KEY: &str = "smg-request-timeout-kind";

/// Which of a request's bounds passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeoutKind {
    /// The total, `--request-timeout-secs`: dispatch to last message.
    Total,
    /// `--first-token-timeout-secs`: dispatch to the stream's first message.
    FirstToken,
    /// `--inter-token-timeout-secs`: one message to the next.
    InterToken,
}

impl TimeoutKind {
    /// The kind's name, as the status metadata and the logs carry it.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Total => "total",
            Self::FirstToken => "first_token",
            Self::InterToken => "inter_token",
        }
    }

    fn parse(label: &str) -> Option<Self> {
        match label {
            "total" => Some(Self::Total),
            "first_token" => Some(Self::FirstToken),
            "inter_token" => Some(Self::InterToken),
            _ => None,
        }
    }
}

/// The bounds configured for a request to a worker, each `None` when off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestBounds {
    /// Dispatch to last message (`--request-timeout-secs`).
    pub total: Option<Duration>,
    /// Dispatch to the first message (`--first-token-timeout-secs`).
    pub first_token: Option<Duration>,
    /// One message to the next (`--inter-token-timeout-secs`).
    pub inter_token: Option<Duration>,
}

impl RequestBounds {
    /// No bound at all.
    pub const NONE: Self = Self {
        total: None,
        first_token: None,
        inter_token: None,
    };

    /// The total bound alone.
    #[must_use]
    pub fn total(timeout: Duration) -> Self {
        Self {
            total: Some(timeout),
            ..Self::NONE
        }
    }

    /// Whether no bound is set.
    #[must_use]
    pub fn is_none(&self) -> bool {
        *self == Self::NONE
    }
}

/// The bounds of one request to a worker, counted from its dispatch.
#[derive(Clone, Copy, Debug)]
pub struct RequestDeadline {
    dispatched: Instant,
    bounds: RequestBounds,
}

impl RequestDeadline {
    /// The total bound of `timeout` from now and nothing else.
    #[must_use]
    pub fn after(timeout: Duration) -> Self {
        Self {
            dispatched: Instant::now(),
            bounds: RequestBounds::total(timeout),
        }
    }

    /// `bounds` counted from now, the dispatch; `None` when none is set, so
    /// an unbounded request arms no timer.
    #[must_use]
    pub fn from_dispatch(bounds: RequestBounds) -> Option<Self> {
        (!bounds.is_none()).then(|| Self {
            dispatched: Instant::now(),
            bounds,
        })
    }

    /// The bounds this deadline was made from.
    #[must_use]
    pub fn bounds(self) -> RequestBounds {
        self.bounds
    }

    /// The instant the total bound passes (as never as the clock allows when
    /// no total is set; see [`Self::dispatch_at`] and [`Self::next_at`] for
    /// the instants that matter).
    #[must_use]
    pub fn at(self) -> Instant {
        self.total_at().unwrap_or_else(far_future)
    }

    fn total_at(self) -> Option<Instant> {
        self.bounds.total.map(|total| self.plus(total))
    }

    fn plus(self, bound: Duration) -> Instant {
        self.dispatched
            .checked_add(bound)
            .unwrap_or_else(far_future)
    }

    /// The bound a call that opens the stream is under, with the instant it
    /// passes: the earlier of the total and the first-token bound (the call
    /// is answered with the first message on servicers that send their
    /// headers with it). `None` when neither is set.
    #[must_use]
    pub fn dispatch_bound(self) -> Option<(Instant, TimeoutKind)> {
        self.earliest(
            self.bounds
                .first_token
                .map(|b| (self.plus(b), TimeoutKind::FirstToken)),
        )
    }

    /// The instant the call that opens the stream must have been answered by
    /// (as never as the clock allows when nothing bounds it).
    #[must_use]
    pub fn dispatch_at(self) -> Instant {
        self.dispatch_bound().map_or_else(far_future, |(at, _)| at)
    }

    /// The bound the stream's next message is under, given the instant of the
    /// last one (`None` before the first): before the first message, the
    /// earlier of the total and the first-token bound; after it, the earlier
    /// of the total and `last + inter_token`. `None` when nothing bounds this
    /// phase.
    #[must_use]
    pub fn next_at(self, last_item: Option<Instant>) -> Option<(Instant, TimeoutKind)> {
        match last_item {
            None => self.dispatch_bound(),
            Some(last) => self.earliest(self.bounds.inter_token.map(|b| {
                (
                    last.checked_add(b).unwrap_or_else(far_future),
                    TimeoutKind::InterToken,
                )
            })),
        }
    }

    /// The earlier of the total bound and `other`.
    fn earliest(self, other: Option<(Instant, TimeoutKind)>) -> Option<(Instant, TimeoutKind)> {
        let total = self.total_at().map(|at| (at, TimeoutKind::Total));
        match (total, other) {
            (Some(total), Some(other)) => Some(if other.0 < total.0 { other } else { total }),
            (total, other) => total.or(other),
        }
    }

    /// The bound of `kind` this deadline was made with (zero when it was not
    /// set, which no status of that kind is made for).
    #[must_use]
    pub fn timeout(self, kind: TimeoutKind) -> Duration {
        match kind {
            TimeoutKind::Total => self.bounds.total,
            TimeoutKind::FirstToken => self.bounds.first_token,
            TimeoutKind::InterToken => self.bounds.inter_token,
        }
        .unwrap_or_default()
    }

    /// The status a request ends with when its bound of `kind` passes: a
    /// `DEADLINE_EXCEEDED` of the gateway's own, marked as such and with the
    /// kind (see [`is_request_timeout`], [`request_timeout_kind`]).
    #[must_use]
    pub fn status(self, kind: TimeoutKind) -> Status {
        let timeout = self.timeout(kind);
        let message = match kind {
            TimeoutKind::Total => format!(
                "request timed out after {timeout:?}: the worker did not finish it within the request timeout"
            ),
            TimeoutKind::FirstToken => format!(
                "no first token within {timeout:?}: the worker did not answer within the first-token timeout"
            ),
            TimeoutKind::InterToken => format!(
                "no token for {timeout:?}: the worker went silent past the inter-token timeout"
            ),
        };
        let mut status = Status::deadline_exceeded(message);
        if let Ok(value) = MetadataValue::try_from(timeout.as_secs().to_string()) {
            status
                .metadata_mut()
                .insert(REQUEST_TIMEOUT_METADATA_KEY, value);
        }
        status.metadata_mut().insert(
            REQUEST_TIMEOUT_KIND_METADATA_KEY,
            MetadataValue::from_static(kind.label()),
        );
        status
    }
}

/// Whether `status` is one of the gateway's own request bounds (see
/// [`RequestDeadline::status`]) and not a worker's `DEADLINE_EXCEEDED`.
#[must_use]
pub fn is_request_timeout(status: &Status) -> bool {
    status.code() == Code::DeadlineExceeded
        && status.metadata().contains_key(REQUEST_TIMEOUT_METADATA_KEY)
}

/// Which of the gateway's own bounds `status` names; `None` for any other
/// status, a worker's `DEADLINE_EXCEEDED` included.
#[must_use]
pub fn request_timeout_kind(status: &Status) -> Option<TimeoutKind> {
    if !is_request_timeout(status) {
        return None;
    }
    status
        .metadata()
        .get(REQUEST_TIMEOUT_KIND_METADATA_KEY)
        .and_then(|value| value.to_str().ok())
        .and_then(TimeoutKind::parse)
        // A status marked without a kind is the total's (an older gateway's).
        .or(Some(TimeoutKind::Total))
}

/// An instant the clock cannot reach: a bound that never passes.
pub(crate) fn far_future() -> Instant {
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
        let status = deadline.status(TimeoutKind::Total);
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert!(status.message().starts_with("request timed out after 15s"));
        assert!(is_request_timeout(&status));
        assert_eq!(request_timeout_kind(&status), Some(TimeoutKind::Total));
        assert_eq!(
            deadline.timeout(TimeoutKind::Total),
            Duration::from_secs(15)
        );
        assert!(!is_request_timeout(&Status::deadline_exceeded(
            "engine scheduling deadline exceeded"
        )));
        assert_eq!(
            request_timeout_kind(&Status::deadline_exceeded(
                "engine scheduling deadline exceeded"
            )),
            None
        );
        assert!(!is_request_timeout(&Status::cancelled("Timeout expired")));
    }

    #[test]
    fn each_silence_bound_has_its_own_status_and_kind() {
        let bounds = RequestBounds {
            total: Some(Duration::from_secs(60)),
            first_token: Some(Duration::from_secs(5)),
            inter_token: Some(Duration::from_secs(2)),
        };
        let deadline = RequestDeadline::from_dispatch(bounds).expect("bounded");
        let first = deadline.status(TimeoutKind::FirstToken);
        assert!(
            first.message().starts_with("no first token within 5s"),
            "{first:?}"
        );
        assert_eq!(request_timeout_kind(&first), Some(TimeoutKind::FirstToken));
        let inter = deadline.status(TimeoutKind::InterToken);
        assert!(inter.message().starts_with("no token for 2s"), "{inter:?}");
        assert_eq!(request_timeout_kind(&inter), Some(TimeoutKind::InterToken));
        assert!(is_request_timeout(&first) && is_request_timeout(&inter));
    }

    #[test]
    fn the_next_bound_is_the_earliest_applicable_one() {
        let bounds = RequestBounds {
            total: Some(Duration::from_secs(60)),
            first_token: Some(Duration::from_secs(5)),
            inter_token: Some(Duration::from_secs(2)),
        };
        let deadline = RequestDeadline::from_dispatch(bounds).expect("bounded");
        let dispatched = deadline.dispatched;
        assert_eq!(
            deadline.dispatch_bound(),
            Some((dispatched + Duration::from_secs(5), TimeoutKind::FirstToken))
        );
        assert_eq!(deadline.next_at(None), deadline.dispatch_bound());
        let last = dispatched + Duration::from_secs(10);
        assert_eq!(
            deadline.next_at(Some(last)),
            Some((last + Duration::from_secs(2), TimeoutKind::InterToken))
        );
        // Past the total's reach, the total is the earlier one.
        let late = dispatched + Duration::from_secs(59);
        assert_eq!(
            deadline.next_at(Some(late)),
            Some((dispatched + Duration::from_secs(60), TimeoutKind::Total))
        );

        // The total alone bounds every phase; a first-token bound alone
        // bounds nothing after the first message.
        let total_only = RequestDeadline::after(Duration::from_secs(60));
        assert_eq!(
            total_only.next_at(Some(last)).map(|(_, kind)| kind),
            Some(TimeoutKind::Total)
        );
        let first_only = RequestDeadline::from_dispatch(RequestBounds {
            first_token: Some(Duration::from_secs(5)),
            ..RequestBounds::NONE
        })
        .expect("bounded");
        assert_eq!(
            first_only.next_at(None).map(|(_, kind)| kind),
            Some(TimeoutKind::FirstToken)
        );
        assert_eq!(first_only.next_at(Some(last)), None);
        assert!(RequestDeadline::from_dispatch(RequestBounds::NONE).is_none());
    }

    #[tokio::test]
    async fn a_deadline_the_clock_cannot_add_never_passes() {
        let deadline = RequestDeadline::after(Duration::MAX);
        assert!(deadline.at() > Instant::now() + Duration::from_secs(60 * 60));
        assert!(deadline.dispatch_at() > Instant::now() + Duration::from_secs(60 * 60));
    }
}
