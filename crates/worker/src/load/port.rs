//! The port between the load collector and the sink of what its facts mean
//! for routing: the calls of a sink and what they answer. The collector (the
//! monitor of [`monitor`](super::monitor)) keeps each worker's report, which
//! is monitoring: the gauges, liveness and the published snapshot read it.
//! The sink keeps the receipts of the pushed records, the poll decision they
//! give, the overload verdict, and what routing reads beside the snapshot.
//!
//! The monitor's own sink is built by
//! [`WorkerMonitor::new`](super::monitor::WorkerMonitor::new) over the
//! registry and the routing side it is handed. A host states the port over
//! its own store and hands it to
//! [`WorkerMonitor::with_sink`](super::monitor::WorkerMonitor::with_sink).
//!
//! The port is not the stream's hand to the monitor: a worker's event stream
//! gives the monitor each pushed record through `PushedLoadSink`
//! (`kv_events::subscription`), and the monitor says what the record means
//! to a [`LoadSink`].

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{identity::basic::Worker, worker::WorkerLoadResponse};

/// Where a worker's load comes from at a tick. An engine that reports its
/// load on its KV-event stream is not asked for it again: in gRPC mode the
/// `GetLoads` poll is the fallback, for servicers that predate the pushed
/// record, for a stream that is down and for heartbeats that stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PollMode {
    /// No pushed record on file: the poll is the only source.
    Poll,
    /// Pushed records on file, but some pushing rank's newest is older than
    /// the tick interval: the poll takes over until the records resume.
    Fallback,
    /// Every pushing rank delivered a record within the tick interval: the
    /// stream is the source, no RPC.
    Suppressed,
}

impl PollMode {
    /// The `mode` label of `smg_engine_load_polls_total`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Poll => "poll",
            Self::Fallback => "fallback",
            Self::Suppressed => "skipped_fresh_push",
        }
    }
}

/// One polled member of an engine, as a poll's answer reaches the sink: the
/// engine's one worker, or one rank worker of a data-parallel engine.
#[derive(Clone, Copy)]
pub struct PolledMember<'a> {
    /// The polled worker.
    pub worker: &'a Arc<dyn Worker>,
    /// The member's answer: the engine's, scoped to the member's own rank
    /// where the member is one rank of it. An answer with no row says the
    /// engine named none for the member; `None` says the fetch failed.
    pub answer: Option<&'a WorkerLoadResponse>,
    /// The member's report on file with the collector, when its stream
    /// pushed one: the rows a sink may send off the file.
    pub on_file: Option<&'a WorkerLoadResponse>,
}

/// What a sink answers to one engine's poll, by rank number. A rank is one
/// member's: the one worker's of an engine, or the rank worker's own.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PolledRanks {
    /// The ranks whose stream is the newer: a record of theirs arrived since
    /// the poll was asked. The collector keeps their rows on file and lays
    /// them over the poll's answer.
    pub newer: Vec<i32>,
    /// The ranks that leave the file. The collector drops their rows, so
    /// the next pushed record of another rank does not publish them again.
    pub leaving: Vec<i32>,
}

/// One polled member's report at the end of a group's tick.
#[derive(Clone, Copy)]
pub struct PolledReport<'a> {
    /// The polled worker the report is of.
    pub worker: &'a Arc<dyn Worker>,
    /// The report as the collector publishes it: the poll's answer with the
    /// rows of the ranks whose stream is the newer laid over it.
    pub report: &'a WorkerLoadResponse,
    /// Whether the answer itself, before those rows, carried token capacity
    /// on rows that are ranks: what a cache of rank loads asks before it
    /// takes the report's rows as the worker's ranks. It does not read the
    /// answer's rank count: an answer short of its own count reads true.
    pub carries_rank_capacity: bool,
}

/// What a group's tick leaves, once every polled engine has answered and the
/// collector has laid the stream's rows over each answer.
#[derive(Clone, Copy)]
pub struct GroupTick<'a> {
    /// Every `Ready` worker of the group the sink says yes of
    /// ([`LoadSink::polls`]), polled or served by its stream.
    pub workers: &'a [Arc<dyn Worker>],
    /// The polled members that answered with rows.
    pub reports: &'a [PolledReport<'a>],
}

/// Where the load collector's facts go.
///
/// The lock: the collector calls [`Self::record`], [`Self::report`] and
/// [`Self::report_empty`] while it holds the lock of its own reports, and
/// lays what the sink answers under the same hold, so a record that lands
/// between a sink's answer and the collector's lay is not undone. It calls
/// [`Self::forget`] there too, so a record lands ahead of a report's
/// removal or behind the sink's word of it. A sink called there neither
/// calls the collector nor waits. The other calls are made with no lock of
/// the collector's held.
///
/// A poll's unit is one engine: its polled members cross together, so a
/// sink that judges an engine reads the answer whole.
///
/// Seven calls have no default: each is a fact a sink takes or a question it
/// answers. Three have one, where doing nothing is a true answer: a sink
/// whose readers always read loads says yes at [`Self::needs_loads`], one
/// whose host asks every worker for its load says yes at [`Self::polls`],
/// and a sink that keeps no rank cache and trims nothing has no work at
/// [`Self::tick`].
pub trait LoadSink: Send + Sync {
    /// A record pushed on the worker's stream for `rank`, received at
    /// `received_at`. `report` is the worker's report after the record, in
    /// the poll's shape with its ranks sorted, and `sampled_at` is when the
    /// engine's state behind the record was sampled. Hot: once a record.
    /// The report is borrowed, so a sink that keeps none of it allocates
    /// nothing.
    fn record(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        received_at: Instant,
        sampled_at: Instant,
    );

    /// The poll decision for the worker at `now` under the tick's
    /// `interval`.
    fn poll_mode(&self, worker: &Arc<dyn Worker>, now: Instant, interval: Duration) -> PollMode;

    /// One engine's answer to the poll asked at `polled_at`, whole: its
    /// polled members, at least one of which brought a row. Answers the
    /// ranks whose stream is the newer and the ranks that leave the file.
    fn report(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> PolledRanks;

    /// One engine's answer with no row for any of its polled members, a
    /// member whose own fetch failed among them. Answers the ranks that
    /// leave the file.
    fn report_empty(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> Vec<i32>;

    /// The fetch of one engine failed: no answer for any polled member of
    /// the poll asked at `polled_at`.
    fn fetch_failed(&self, members: &[Arc<dyn Worker>], polled_at: Instant);

    /// The worker's loads are forgotten: it left the registry, was
    /// replaced, or its status left `Ready`. It is also said for a worker
    /// whose pushed report a record of the registry's worker at the URL
    /// replaced. It is not said at the eviction of a worker whose URL holds
    /// another worker's report: what the sink holds at the URL is that
    /// worker's since its first record. A worker that never pushed is then
    /// not said at all: a sink that keys its rows by the worker and not by
    /// the URL lets them go at the first record of the URL's next worker.
    fn forget(&self, worker: &Arc<dyn Worker>);

    /// The collector reset itself: it stopped its groups and dropped every
    /// load it published. It does so on three paths: at its start, after
    /// it lost registry events, and when its host stops it. What the sink
    /// derived from loads and cannot tie to a standing worker by itself is
    /// void: its verdicts and its rows. Just before this call the collector
    /// says [`Self::forget`] for every worker whose pushed report it held
    /// and that is no longer the registry's `Ready` worker at its URL. So a
    /// sink may keep what it knows of a stream, as the collector keeps the
    /// report of a pusher that stays. A polled worker that left while
    /// events were lost is never said: in a sink that keeps rows of its
    /// own, nothing of a polled worker outlives this call. A sink whose
    /// rows end with the write that removes their worker keeps them: no row
    /// of a worker that left stands there.
    ///
    /// A dropped collector says nothing: its drop stops its tasks and calls
    /// no sink. The host that drops a collector owns the sink too: it voids
    /// its store when it chooses, or it starts the next collector over the
    /// same sink, whose start says the reset.
    fn reset(&self);

    /// Whether a reader wants worker loads, asked by a collector that is
    /// told to fetch them on demand alone. A sink whose readers always read
    /// them keeps this answer.
    fn needs_loads(&self) -> bool {
        true
    }

    /// Whether the collector asks `worker` for its load. A worker the sink
    /// says no of is no member of a tick: it is not polled, no load stream
    /// is wanted for it, and [`Self::tick`] does not name it. A sink whose
    /// host can dial every worker keeps this answer.
    ///
    /// The collector says nothing where a worker's answer turns from yes to
    /// no: the worker keeps its last published report, and the sink what it
    /// judged of it, until [`Self::forget`] or [`Self::reset`]. A record
    /// pushed for a worker the sink says no of still reaches
    /// [`Self::record`]: the push asks nothing here.
    fn polls(&self, _worker: &dyn Worker) -> bool {
        true
    }

    /// The end of a group's tick. A sink that keeps rows of its own took
    /// them at [`Self::report`] and needs nothing here.
    fn tick(&self, _tick: &GroupTick<'_>) {}
}
