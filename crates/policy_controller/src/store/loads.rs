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

    /// A poll's report of the worker, one row per rank, asked at
    /// `polled_at`.
    fn report(&self, id: WorkerId, rows: &[(u16, LoadRow)], polled_at: Instant, source: LoadSource);

    /// The poll decision for the worker at `now` under the poll `interval`.
    fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> PollMode;

    /// The row of one rank, when one is on file.
    fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample>;

    /// How many ranks have a row on file.
    fn ranks(&self, id: WorkerId) -> usize;

    /// The worker left, or its reports stopped: nothing on file for it.
    fn evict(&self, id: WorkerId);
}
