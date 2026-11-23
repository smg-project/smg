//! What a load monitor's polls did, as counts its host reads
//! ([`WorkerMonitor::poll_totals`]).
//!
//! The monitor writes series of its own through the process recorder. A
//! host that renders a recorder of its own cannot read those, so the
//! monitor keeps these counts beside them and the host reads them where it
//! renders. Every count moves in a group's tick alone: no request, no
//! decision and no pushed record adds to one.
//!
//! [`WorkerMonitor::poll_totals`]: super::monitor::WorkerMonitor::poll_totals

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::port::PollMode;
use crate::metrics::writers::Metrics;

/// Why one fetch of an engine's load brought no report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FetchFailure {
    /// The connect, the request or the read of the answer failed on the
    /// way: no answer of the engine's.
    Transport,
    /// The engine, or the host's load source in its place, answered with an
    /// error or with a body that is no load report.
    Answer,
    /// The poll's deadline, or the request's own, ended the fetch.
    Deadline,
    /// The engine's metrics state no load gauge.
    NoLoad,
    /// The engine has no load read for the worker: the call is not
    /// implemented, or no native load route is there and the worker has no
    /// gauges to read instead (its runtime has no gauge table, or it is one
    /// rank of a data-parallel engine).
    Unsupported,
    /// The host had no client for the worker, or no word of its load source
    /// in time: the host's failure, no engine's.
    Client,
}

/// Fetches that brought no report, by reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FailureTotals {
    /// The connect, the request or the read of the answer failed on the
    /// way.
    pub transport: u64,
    /// The engine, or the host's load source in its place, answered with an
    /// error or with a body that is no load report.
    pub answer: u64,
    /// The poll's deadline, or the request's own, ended the fetch.
    pub deadline: u64,
    /// The engine's metrics stated no load gauge.
    pub no_load: u64,
    /// The engine has no load read for the worker: the call is not
    /// implemented, or no native load route is there and the worker has no
    /// gauges to read instead (its runtime has no gauge table, or it is one
    /// rank of a data-parallel engine).
    pub unsupported: u64,
    /// The host had no client for the worker, or no word of its load source
    /// in time.
    pub client: u64,
}

/// What a load monitor's polls did since it was built, as one read answers
/// it ([`WorkerMonitor::poll_totals`]). The numbers stand through a stop
/// and a start of the monitor: they are the monitor's, not one run's.
///
/// A worker on a connection is one that is asked over gRPC or ZMQ. A
/// worker over HTTP has no stream and no arbitration: its read is counted
/// by its outcome.
///
/// [`WorkerMonitor::poll_totals`]: super::monitor::WorkerMonitor::poll_totals
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PollTotals {
    /// Ticks at which a worker on a connection was polled with no pushed
    /// record on file ([`PollMode::Poll`]).
    pub polls: u64,
    /// Ticks at which a worker on a connection was polled because a
    /// pushing rank's records went quiet ([`PollMode::Fallback`]).
    pub fallbacks: u64,
    /// Ticks at which a worker on a connection was left to its stream
    /// ([`PollMode::Suppressed`]): no call was made.
    pub suppressed: u64,
    /// Fetches over a connection that brought no report.
    pub poll_failures: FailureTotals,
    /// Reads over HTTP that brought a report.
    pub scrapes: u64,
    /// Reads over HTTP that brought no report.
    pub scrape_failures: FailureTotals,
    /// Rank rows of polled answers the monitor published.
    pub rows: u64,
    /// Named values the report's bounds took out of polled answers.
    pub values_dropped: u64,
}

/// Failed fetches, one count a [`FetchFailure`].
#[derive(Debug, Default)]
struct Failures([AtomicU64; 6]);

impl Failures {
    fn count(&self, reason: FetchFailure) {
        let at = match reason {
            FetchFailure::Transport => 0,
            FetchFailure::Answer => 1,
            FetchFailure::Deadline => 2,
            FetchFailure::NoLoad => 3,
            FetchFailure::Unsupported => 4,
            FetchFailure::Client => 5,
        };
        self.0[at].fetch_add(1, Relaxed);
    }

    fn totals(&self) -> FailureTotals {
        let [transport, answer, deadline, no_load, unsupported, client] =
            [0, 1, 2, 3, 4, 5].map(|at: usize| self.0[at].load(Relaxed));
        FailureTotals {
            transport,
            answer,
            deadline,
            no_load,
            unsupported,
            client,
        }
    }
}

/// The counts behind [`PollTotals`]: one a monitor, added to by its ticks.
#[derive(Debug, Default)]
pub(crate) struct PollCounts {
    polls: AtomicU64,
    fallbacks: AtomicU64,
    suppressed: AtomicU64,
    poll_failures: Failures,
    scrapes: AtomicU64,
    scrape_failures: Failures,
    rows: AtomicU64,
    values_dropped: AtomicU64,
}

impl PollCounts {
    /// One worker's poll decision at a tick, in one step: the monitor's
    /// series, and the count where the worker is on a connection.
    pub(crate) fn decided(&self, url: &str, mode: PollMode, on_connection: bool) {
        Metrics::record_engine_load_poll(url, mode.as_str());
        if on_connection {
            let count = match mode {
                PollMode::Poll => &self.polls,
                PollMode::Fallback => &self.fallbacks,
                PollMode::Suppressed => &self.suppressed,
            };
            count.fetch_add(1, Relaxed);
        }
    }

    /// One fetch of an engine's load ended: over HTTP a report is counted
    /// and a failure by its reason, over a connection a failure alone.
    pub(crate) fn fetched<T>(&self, over_http: bool, outcome: &Result<T, FetchFailure>) {
        match (over_http, outcome) {
            (true, Ok(_)) => {
                self.scrapes.fetch_add(1, Relaxed);
            }
            (true, Err(reason)) => self.scrape_failures.count(*reason),
            (false, Ok(_)) => {}
            (false, Err(reason)) => self.poll_failures.count(*reason),
        }
    }

    /// The rank rows of polled answers a tick hands the publish.
    pub(crate) fn published(&self, rows: usize) {
        self.rows.fetch_add(rows as u64, Relaxed);
    }

    /// The named values the bounds took out of one polled answer, in one
    /// step: the monitor's series and the count.
    pub(crate) fn dropped(&self, url: &str, values: usize) {
        Metrics::record_engine_load_values_dropped(url, values);
        self.values_dropped.fetch_add(values as u64, Relaxed);
    }

    /// The counts as they stand: each number is read by itself, so a read
    /// beside a tick may hold one count of the tick and not the next.
    pub(crate) fn totals(&self) -> PollTotals {
        PollTotals {
            polls: self.polls.load(Relaxed),
            fallbacks: self.fallbacks.load(Relaxed),
            suppressed: self.suppressed.load(Relaxed),
            poll_failures: self.poll_failures.totals(),
            scrapes: self.scrapes.load(Relaxed),
            scrape_failures: self.scrape_failures.totals(),
            rows: self.rows.load(Relaxed),
            values_dropped: self.values_dropped.load(Relaxed),
        }
    }
}
