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
//!
//! The scope holds a second value beside it: how long the read of the
//! request's body waited for the client (`RequestStarted::upload_waited`,
//! written by a reader of the body where it ends a read that had to wait:
//! the request-body deadline layer, or one of the two readers outside it
//! under the same measure, the WASM request hook ahead of the layer and the
//! multipart route, which stands under no body deadline). The dispatch wait
//! counted without the upload is the dispatch wait less this. A body that
//! came whole leaves it zero: its reader reads no clock and writes nothing.

use std::{
    cell::Cell,
    future::Future,
    time::{Duration, Instant},
};

use tokio::task::futures::TaskLocalFuture;

tokio::task_local! {
    /// The stamp of the request whose call or future runs: in a cell, since
    /// the body's upload wait is written to it inside the scope.
    static STARTED: Cell<RequestStarted>;
}

/// When the request came in at the edge, and how long the read of its body
/// has waited for the client.
#[derive(Clone, Copy, Debug)]
pub struct RequestStarted {
    at: Instant,
    /// In nanoseconds, not a `Duration`: the stamp is swapped in and out at
    /// every poll of the request's future, and this keeps it eight bytes
    /// smaller.
    upload_wait_nanos: u64,
}

impl RequestStarted {
    /// A stamp taken now, on the runtime's clock.
    pub fn now() -> Self {
        Self::at(clock_now())
    }

    /// A stamp taken at `at`.
    pub fn at(at: Instant) -> Self {
        Self {
            at,
            upload_wait_nanos: 0,
        }
    }

    /// The instant the request came in.
    pub fn instant(self) -> Instant {
        self.at
    }

    /// Runs `f` as the request's: inside it, [`RequestStarted::current`]
    /// reads this stamp. The edge calls the layers inside from its own
    /// `call` under this, so they read the stamp in their `call`s.
    pub fn sync_scope<R>(self, f: impl FnOnce() -> R) -> R {
        STARTED.sync_scope(Cell::new(self), f)
    }

    /// Runs `future` as the request's: at every poll inside it,
    /// [`RequestStarted::current`] reads this stamp.
    pub fn scope<F: Future>(self, future: F) -> TaskLocalFuture<Cell<RequestStarted>, F> {
        STARTED.scope(Cell::new(self), future)
    }

    /// The entry stamp of the request whose call or future runs, when the
    /// edge put one on it.
    pub fn current() -> Option<Instant> {
        STARTED.try_with(|started| started.get().at).ok()
    }

    /// The whole stamp of the request whose call or future runs, the upload
    /// wait of its body with it: what a task of its own runs under again
    /// ([`RequestStarted::scope`]) to count as the request's future does.
    pub(crate) fn carried() -> Option<Self> {
        STARTED.try_with(Cell::get).ok()
    }

    /// Adds `waited` to the upload wait of the request whose call or future
    /// runs: what a reader of its body says where it ends a read that had to
    /// wait for the client. A call outside a scope does nothing: there is no
    /// stamp to write to.
    pub(crate) fn upload_waited(waited: Duration) {
        let nanos = u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX);
        STARTED
            .try_with(|started| {
                let stamp = started.get();
                started.set(Self {
                    upload_wait_nanos: stamp.upload_wait_nanos.saturating_add(nanos),
                    ..stamp
                });
            })
            .unwrap_or_default();
    }

    /// `wait`, a time counted from this stamp's entry, less the upload wait
    /// of the body: `wait` itself where the body never waited.
    pub(crate) fn less_upload_wait(self, wait: Duration) -> Duration {
        match self.upload_wait_nanos {
            0 => wait,
            nanos => wait.saturating_sub(Duration::from_nanos(nanos)),
        }
    }

    /// This stamp with `waited` as its upload wait: what a test hands a
    /// pipeline it runs bare, where no reader of a body runs.
    #[cfg(test)]
    pub(crate) fn with_upload_wait(self, waited: Duration) -> Self {
        Self {
            upload_wait_nanos: u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX),
            ..self
        }
    }
}

/// The runtime's clock: the wall clock in production, the paused clock of a
/// test that advances time itself.
pub fn clock_now() -> Instant {
    tokio::time::Instant::now().into_std()
}

#[cfg(test)]
mod tests {
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

    /// A wait counted from the entry, as the dispatch wait is.
    const WAIT: Duration = Duration::from_secs(9);

    #[test]
    fn outside_a_scope_an_upload_wait_is_written_nowhere() {
        RequestStarted::upload_waited(Duration::from_secs(1));
        assert!(RequestStarted::carried().is_none());
    }

    #[tokio::test]
    async fn an_upload_wait_is_taken_off_a_wait_counted_from_the_entry() {
        let entered = RequestStarted::now();
        let (before, first, second) = entered
            .scope(async {
                let before = RequestStarted::carried();
                RequestStarted::upload_waited(Duration::from_secs(3));
                let first = RequestStarted::carried();
                RequestStarted::upload_waited(Duration::from_secs(4));
                (before, first, RequestStarted::carried())
            })
            .await;
        let before = before.expect("the stamp inside the scope");
        assert_eq!(
            before.less_upload_wait(WAIT),
            WAIT,
            "a body that never waited takes nothing off"
        );
        let first = first.expect("the stamp inside the scope");
        assert_eq!(first.instant(), entered.instant(), "the entry stamp stays");
        assert_eq!(first.less_upload_wait(WAIT), Duration::from_secs(6));
        let second = second.expect("the stamp inside the scope");
        assert_eq!(
            second.less_upload_wait(WAIT),
            Duration::from_secs(2),
            "a second read's wait adds to the first"
        );
        assert_eq!(
            second.less_upload_wait(Duration::from_secs(5)),
            Duration::ZERO,
            "a wait shorter than the upload wait is zero"
        );
    }

    #[tokio::test]
    async fn a_task_of_its_own_carries_the_upload_wait_and_a_new_stamp_holds_none() {
        let entered = RequestStarted::now();
        let carried = entered
            .scope(async {
                RequestStarted::upload_waited(Duration::from_secs(2));
                RequestStarted::carried()
            })
            .await
            .expect("the stamp inside the scope");
        // A task of its own, run under the carried stamp, counts as the
        // request's future does.
        let again = carried
            .scope(async { RequestStarted::carried() })
            .await
            .expect("the carried stamp");
        assert_eq!(again.instant(), entered.instant());
        assert_eq!(again.less_upload_wait(WAIT), Duration::from_secs(7));
        // A stamp taken anew (a later turn's) holds no upload wait.
        assert_eq!(RequestStarted::now().less_upload_wait(WAIT), WAIT);
    }
}
