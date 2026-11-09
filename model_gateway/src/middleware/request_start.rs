//! The request's entry stamp: the one clock read the edge takes when a
//! request comes in, which the layers inside share instead of reading the
//! clock again. The metrics layer, the outermost layer with a clock, takes
//! it and runs the request inside its scope; the access log counts its
//! latency from it and the in-flight guard ages the request from it. Each
//! keeps a read of its own for a request that reaches it outside the scope
//! (a bare stack in a test). The stamps inside the admission layers (the
//! acceptance at tenant resolution, the request-body deadline) stay reads of
//! their own: a wait for a permit sits between the edge and them.
//!
//! The stamp is a task-local of the request ([`RequestStarted::sync_scope`]
//! around the inner service's `call`, [`RequestStarted::scope`] around its
//! future), not an entry on the request's extensions: an entry is a box per
//! request and, at the edge's entry count, the map's growth too (the eighth
//! entry rehashes the table into one twice the size), for one `Instant` the
//! layers inside read once.
//!
//! The stamp is on the runtime's clock ([`clock_now`]): the wall clock in
//! production, the paused clock of a test that advances time itself, which
//! is also the clock the acceptance stamp is read on.

use std::{future::Future, time::Instant};

use tokio::task::futures::TaskLocalFuture;

tokio::task_local! {
    /// The entry stamp of the request whose call or future runs.
    static STARTED: RequestStarted;
}

/// When the request came in at the edge.
#[derive(Clone, Copy, Debug)]
pub struct RequestStarted(Instant);

impl RequestStarted {
    /// A stamp taken now, on the runtime's clock.
    pub fn now() -> Self {
        Self(clock_now())
    }

    /// A stamp taken at `at`.
    pub fn at(at: Instant) -> Self {
        Self(at)
    }

    /// The instant the request came in.
    pub fn instant(self) -> Instant {
        self.0
    }

    /// Runs `f` as the request's: inside it, [`RequestStarted::current`]
    /// reads this stamp. The edge calls the layers inside from its own
    /// `call` under this, so they read the stamp in their `call`s.
    pub fn sync_scope<R>(self, f: impl FnOnce() -> R) -> R {
        STARTED.sync_scope(self, f)
    }

    /// Runs `future` as the request's: at every poll inside it,
    /// [`RequestStarted::current`] reads this stamp.
    pub fn scope<F: Future>(self, future: F) -> TaskLocalFuture<RequestStarted, F> {
        STARTED.scope(self, future)
    }

    /// The entry stamp of the request whose call or future runs, when the
    /// edge put one on it.
    pub fn current() -> Option<Instant> {
        STARTED.try_with(|started| started.0).ok()
    }
}

/// The runtime's clock: the wall clock in production, the paused clock of a
/// test that advances time itself.
pub fn clock_now() -> Instant {
    tokio::time::Instant::now().into_std()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn outside_a_request_s_scope_there_is_no_stamp() {
        assert!(RequestStarted::current().is_none());
    }

    #[tokio::test]
    async fn inside_the_scope_the_stamp_is_the_instant_it_was_taken_at() {
        let at = Instant::now() - Duration::from_secs(3);
        let read = RequestStarted::at(at)
            .scope(async { RequestStarted::current() })
            .await;
        assert_eq!(read, Some(at));
        assert!(
            RequestStarted::current().is_none(),
            "the scope ends with the request's future"
        );
    }

    #[test]
    fn inside_the_sync_scope_the_stamp_is_read_from_a_call() {
        let at = Instant::now() - Duration::from_secs(2);
        let read = RequestStarted::at(at).sync_scope(RequestStarted::current);
        assert_eq!(read, Some(at));
        assert!(
            RequestStarted::current().is_none(),
            "the scope ends with the call"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_stamp_is_on_the_runtime_s_clock() {
        let before = RequestStarted::now();
        tokio::time::advance(Duration::from_secs(5)).await;
        let after = RequestStarted::now();
        assert_eq!(
            after.instant().duration_since(before.instant()),
            Duration::from_secs(5)
        );
    }
}
