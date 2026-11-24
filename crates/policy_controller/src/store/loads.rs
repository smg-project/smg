//! The load sink: one row per rank from any of the three sources, with the
//! poll arbitration per worker (design 5.1, the loads row).

use std::time::{Duration, Instant};

use crate::{
    engine::view::{LoadRow, SignalSource, WorkerId},
    store::GatewayId,
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
    /// A pushed row's receipt or a poll's start, on the host's clock.
    pub received_at: Instant,
    /// When the state behind the report was sampled, on the host's clock.
    /// A load scorer releases the dispatches it made at or before it.
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
/// engine's overload verdict when the write is whole: a record, a
/// report, the word that no report came, an eviction. Once the caller
/// told how often it hears a rank ([`LoadSink::heard_every`]), a judgment
/// counts a rank's sample only while its engine still says it.
///
/// A rank its engine stops naming is avoided at once. Where an answer
/// brings a row for one rank of an engine and none for another, the rank it
/// left out carries the overload bit on its rows while a rank the answer
/// named has a record on file, whether or not a threshold stands. The mark
/// ends when an answer or a record names the rank again. It goes with the
/// member's rows, and with the build that follows the member's
/// registration under another rank number. A rank alone is never avoided:
/// with no named rank on file it reads as its engine does.
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
    /// written whole before the engine is judged, so two ranks that trade
    /// their load in one answer are never read as one new rank beside one
    /// old. A rank the answer omits keeps what is on file, and beside a
    /// rank the answer names it is avoided from this write on. An answer
    /// with no row for any rank of an engine is the word that no report
    /// came: nothing on file changes, no rank is marked or named again,
    /// and the engine's overload verdict reads
    /// clear until its next record or report. The word stands over no
    /// engine a rank of which pushed since the poll began, whether the
    /// answer names that rank or not. Whether a stream still serves a
    /// rank the poll was not for is the caller's to know, by the
    /// freshness it chose the poll's ranks with: it says the word only
    /// for a poll that left out no such rank. The crate's store pays the
    /// build a rank the answer names still waits for before it writes the
    /// answer, so an answer is read over the rows of every rank it names.
    /// A row the answer brings for a rank that pushed since the poll began
    /// is passed over, and names its engine all the same: a rank left out
    /// beside it is avoided.
    ///
    /// The door answers the positions in `answer` of the rows it passed
    /// over, in the answer's order: the rows a caller that keeps a file of
    /// its own must not lay over it, since the record that stands is the
    /// newer. A position that brought no row is never on the list, nor is
    /// a row for an id that is no member, which is dropped. The list says
    /// what the door passed over at this write, and it is empty, with no
    /// allocation, where it passed over nothing.
    fn report(
        &self,
        answer: &[(WorkerId, u16, Option<LoadRow>)],
        polled_at: Instant,
        source: LoadSource,
    ) -> Vec<usize>;

    /// A poll answer with a known sample instant, on the host's clock.
    /// The rows carry the age at receipt. The sample instant may lie
    /// before or during the poll, whose start still arbitrates with
    /// pushed records. A sink that only keeps the rows can use `report`.
    fn report_sampled(
        &self,
        answer: &[(WorkerId, u16, Option<LoadRow>)],
        polled_at: Instant,
        sampled_at: Instant,
        source: LoadSource,
    ) -> Vec<usize> {
        let _ = sampled_at;
        self.report(answer, polled_at, source)
    }

    /// The poll decision for the worker at `now` under the poll `interval`.
    fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> PollMode;

    /// The caller hears every ready rank once a `period`: by its stream's
    /// records, or by the poll it falls back to. From this call on a
    /// sample is a fact while its engine still says it: a judgment counts
    /// the row of a data-parallel engine's rank only where its receipt
    /// lies no further behind the engine's newest than three periods,
    /// which keep a rank through one lost report at the least. Both
    /// instants are the sink's own receipts (a record's arrival, a poll's
    /// start), so no clock crosses, and no timer runs: the age is read at
    /// the write that judges, and an engine none of whose ranks is heard
    /// keeps its verdict until the word that no report came. Every engine
    /// is judged again at the call. Until a period is told no sample ends
    /// by its age. A period is told once: a call with another period
    /// changes nothing and answers `false`.
    fn heard_every(&self, period: Duration) -> bool;

    /// The row of one rank, when one is on file.
    fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample>;

    /// How many ranks have a row on file.
    fn ranks(&self, id: WorkerId) -> usize;

    /// The worker left, or its reports stopped: nothing on file for it.
    fn evict(&self, id: WorkerId);

    /// When the writes this sink took and has not yet shown its readers are
    /// due to be shown; `None` when nothing waits. A sink that shows each
    /// write at once never names an instant. The controller's serve keeps
    /// the instant; a host that writes from a task of its own reads it
    /// after a write and calls [`LoadSink::publish`] at it. A host that
    /// does neither shows a waiting write with its next one, and the row's
    /// age says so meanwhile.
    ///
    /// The crate's store names the instant for one thing more: the build
    /// its membership writes owe. The store pays that build ahead of the
    /// instant for one load write alone, a poll's answer for a rank that
    /// has no row yet, and no load write is shown before it, so a host with
    /// no loop asks for it ([`crate::store::MemoryStore::publish_now`]).
    fn publish_due(&self) -> Option<Instant> {
        None
    }

    /// Show the readers what waits, when `now` is at or past the instant
    /// [`LoadSink::publish_due`] names; a call before it shows nothing.
    fn publish(&self, now: Instant) {
        let _ = now;
    }

    /// Hand the sink the host's wake. The sink calls it, on the writing
    /// thread, where a write that is no load call leaves an instant for
    /// [`LoadSink::publish_due`] to name and none stood: in the crate's
    /// store, the build a membership write owes. The host's loop reads the
    /// instant again at each call. The last wake handed stands. A sink
    /// that never names an instant drops it.
    fn wake_with(&self, wake: Wake) {
        drop(wake);
    }
}

/// The host's wake: what a sink calls to have its host read
/// [`LoadSink::publish_due`] again. It is called under the writer's locks:
/// it sets a flag or a notification and returns.
pub type Wake = Box<dyn Fn() + Send + Sync>;
