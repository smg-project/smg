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
//! the request-body deadline layer, or a reader ahead of it under the same
//! measure). The dispatch wait counted without the upload is the dispatch
//! wait less this. A body that came whole leaves it zero: its reader reads
//! no clock and writes nothing.

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
#[path = "../../tests/unit/middleware/request_start/tests.rs"]
mod tests;
