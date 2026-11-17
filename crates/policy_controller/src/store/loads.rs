//! The load sink: one row per rank from any of the three sources, with the
//! poll arbitration per worker (design 5.1, the loads row).

use std::time::{Duration, Instant};

use crate::{
    store::GatewayId,
    view::{LoadRow, SignalSource, WorkerId},
};

/// Where a load row came from, as the ingress knows it. The view's
/// [`SignalSource`] is its narrow form, without the relay's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadSource {
    /// The records riding the worker's event stream.
    Stream,
    /// The `GetLoads` poll a servicer falls back to when its stream stops.
    Poll,
    /// The scrape of an engine without a servicer.
    Scrape,
    /// A gateway relaying a host-local worker's records.
    Relay(GatewayId),
}

impl From<LoadSource> for SignalSource {
    fn from(source: LoadSource) -> Self {
        match source {
            LoadSource::Stream => Self::Events,
            LoadSource::Poll => Self::Poll,
            LoadSource::Scrape => Self::Scrape,
            LoadSource::Relay(_) => Self::Relay,
        }
    }
}

/// The poll decision for one worker at one tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PollMode {
    /// No pushed record on file: the poll is the only source.
    Poll,
    /// A pushing rank's newest record is older than the interval: the poll
    /// stands in for it.
    Fallback,
    /// Every pushing rank is fresher than the interval: no call.
    Suppressed,
}

/// One rank's row as the store holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadSample {
    pub row: LoadRow,
    /// When the store received it, on the host's clock: what the poll
    /// decision reads.
    pub received_at: Instant,
    /// When the engine's state behind it was sampled: the receipt less the
    /// row's age. A load scorer releases the dispatches it made at or
    /// before it.
    pub sampled_at: Instant,
    pub source: LoadSource,
}

/// Where load rows are written.
///
/// A record replaces its rank's row: the stream is in order per rank and a
/// relay forwards it in order, so the newest record is the one on file; a
/// record applied twice changes nothing. A poll's report fills the ranks that
/// pushed nothing since the poll began and leaves the others to the stream,
/// whose row is the newer; a poll never counts as the stream working, so the
/// worker stays polled until its records resume.
///
/// Every write that changes what is on file for an engine judges the
/// engine's overload verdict once, when the write is whole: a record, a
/// report, the word that no report came, an eviction. Once the caller
/// told how often it hears a rank ([`LoadSink::heard_every`]), a judgment
/// counts a rank's sample only while its engine still says it.
///
/// A worker's rows live exactly as long as its membership row: the write
/// that removes or replaces the worker deletes them, and a record or a
/// report for a worker that is not a member is dropped, so a read that lands
/// after the removal re-creates nothing.
pub trait LoadSink: Send + Sync {
    /// One rank's record from the stream, a relay or the scrape: the
    /// demultiplexer's per-rank delivery of one engine subscription.
    fn record(
        &self,
        id: WorkerId,
        rank: u16,
        row: LoadRow,
        received_at: Instant,
        source: LoadSource,
    );

    /// One engine's answer to the poll asked at `polled_at`, whole: for
    /// each rank the poll was for, under the rank's own id, the row the
    /// answer brought, or `None` where it brought none. The rows are
    /// written and the engine is judged once, so two ranks that trade
    /// their load in one answer are never read as one new rank beside one
    /// old. A rank the answer omits keeps what is on file. An answer with
    /// no row for any rank of an engine is the word that no report came:
    /// nothing on file changes, and the engine's overload verdict reads
    /// clear until its next record or report. The word stands over no
    /// engine a rank of which pushed since the poll began, whether the
    /// answer names that rank or not. Whether a stream still serves a
    /// rank the poll was not for is the caller's to know, by the
    /// freshness it chose the poll's ranks with: it says the word only
    /// for a poll that left out no such rank.
    fn report(
        &self,
        answer: &[(WorkerId, u16, Option<LoadRow>)],
        polled_at: Instant,
        source: LoadSource,
    );

    /// The poll decision for the worker at `now` under the poll `interval`.
    fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> PollMode;

    /// The caller hears every ready rank once a `period`: by its stream's
    /// records, or by the poll it falls back to. From this call on a
    /// sample is a fact while its engine still says it: a judgment counts
    /// the row of a data-parallel engine's rank only where its receipt
    /// lies no further behind the engine's newest than three periods,
    /// which keep a rank through two lost reports. Both instants are the
    /// sink's own receipts (a record's arrival, a poll's start), so no
    /// clock crosses, and no timer runs: the age is read at the write
    /// that judges, and an engine none of whose ranks is heard keeps its
    /// verdict until the word that no report came. Every engine is judged
    /// again at the call. Until a period is told no sample ends by its
    /// age. A period is told once: a call with another period changes
    /// nothing and answers `false`.
    fn heard_every(&self, period: Duration) -> bool;

    /// The row of one rank, when one is on file.
    fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample>;

    /// How many ranks have a row on file.
    fn ranks(&self, id: WorkerId) -> usize;

    /// The worker left, or its reports stopped: nothing on file for it.
    fn evict(&self, id: WorkerId);

    /// When the writes this sink took and has not yet shown its readers are
    /// due to be shown; `None` when nothing waits. A sink that shows each
    /// write at once never names an instant. The crate's load driver keeps
    /// the instant; a host that writes from a task of its own reads it
    /// after a write and calls [`LoadSink::publish`] at it. A host that
    /// does neither shows a waiting write with its next one, and the row's
    /// age says so meanwhile.
    fn publish_due(&self) -> Option<Instant> {
        None
    }

    /// Show the readers what waits, when `now` is at or past the instant
    /// [`LoadSink::publish_due`] names; a call before it shows nothing.
    fn publish(&self, now: Instant) {
        let _ = now;
    }
}
