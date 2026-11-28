//! The records-only stream of `SubscribeKvEvents` with `loads_only`: the
//! servicer's load record for every data-parallel rank, at the record's
//! cadence, and nothing else.
//!
//! A gateway that routes by load and keeps no KV index asks for this instead
//! of the event stream. It is served from the servicer's own `GetLoads`
//! bookkeeping ([`RankLoads`] through [`LoadFromState`]), so it needs no
//! KV-event relay: an engine whose events are off streams its load all the
//! same. Every batch is `load_only`, numbered 0 (there is no sequence to
//! repeat and the subscriber admits nothing), stamped with its rank, and
//! carries the record with its telemetry. A rank's record goes out when it
//! changed ([`RankClock::due`], checked every `load_tick`) or when its
//! heartbeat is due (`heartbeat_interval`, `heartbeat_backoff` after two
//! unchanged heartbeats), each rank on its own clock, so a quiet rank keeps
//! reporting while another is busy: the gateway prices each rank worker by
//! its own rank's current figures. The stream ends when the servicer state is
//! gone.

use std::{collections::VecDeque, sync::Arc, time::Duration};

use futures::stream;
use smg_grpc_client::common_proto as common;
use tokio::time::Instant;
use tonic::Status;

use crate::{
    kv_events::{
        unix_seconds, LoadFromState, LoadSource, RankClock, RankLoads, DEFAULT_HEARTBEAT_BACKOFF,
        DEFAULT_HEARTBEAT_INTERVAL, DEFAULT_LOAD_TICK,
    },
    BoxStream,
};

/// How often a rank's record is checked and how long a silence may last
/// before its heartbeat: the relay's figures.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cadence {
    pub(crate) load_tick: Duration,
    pub(crate) heartbeat_interval: Duration,
    pub(crate) heartbeat_backoff: Duration,
}

impl Default for Cadence {
    fn default() -> Self {
        Self {
            load_tick: DEFAULT_LOAD_TICK,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
        }
    }
}

/// Where the stream reads: the record of each rank and how many ranks the
/// servicer serves now (0 once the servicer is gone, which ends the stream).
pub(crate) trait RankedLoads: Send + Sync {
    fn rank_count(&self) -> usize;
    fn rank_record(&self, rank: i32) -> Option<common::EngineLoad>;
}

impl<S: RankLoads> RankedLoads for LoadFromState<S> {
    fn rank_count(&self) -> usize {
        self.0.upgrade().map_or(0, |state| state.rank_count())
    }

    fn rank_record(&self, rank: i32) -> Option<common::EngineLoad> {
        LoadSource::load(self, Some(rank))
    }
}

struct LoadRecordStream {
    source: Arc<dyn RankedLoads>,
    cadence: Cadence,
    ranks: Vec<RankClock>,
    /// The batches a tick found due, sent one per poll.
    pending: VecDeque<common::KvEventBatch>,
    next_tick: Instant,
    /// Records sent on this stream (`EngineLoad.sample`).
    sample: u64,
}

impl LoadRecordStream {
    fn new(source: Arc<dyn RankedLoads>, cadence: Cadence) -> Self {
        Self {
            source,
            cadence,
            ranks: Vec::new(),
            pending: VecDeque::new(),
            // The first tick is now: every rank's record goes out at once.
            next_tick: Instant::now(),
            sample: 0,
        }
    }

    /// Queue the record of every rank whose record moved or whose heartbeat
    /// is due at `now`; `false` once the servicer is gone.
    fn collect_due(&mut self, now: Instant) -> bool {
        let count = self.source.rank_count();
        if count == 0 {
            return false;
        }
        self.ranks.truncate(count);
        self.ranks.resize_with(count, RankClock::new);
        for (index, clock) in self.ranks.iter_mut().enumerate() {
            let rank = i32::try_from(index).unwrap_or(i32::MAX);
            let Some(mut record) = self.source.rank_record(rank) else {
                continue;
            };
            if !clock.due(
                &record,
                now,
                self.cadence.heartbeat_interval,
                self.cadence.heartbeat_backoff,
            ) {
                continue;
            }
            self.sample += 1;
            record.sample = self.sample;
            record.load_only = true;
            clock.sent(&record, now);
            self.pending.push_back(common::KvEventBatch {
                sequence_number: 0,
                timestamp: unix_seconds(),
                events: Vec::new(),
                dp_rank: Some(rank),
                snapshot: None,
                load: Some(record),
            });
        }
        true
    }

    async fn next(&mut self) -> Option<common::KvEventBatch> {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                return Some(batch);
            }
            tokio::time::sleep_until(self.next_tick).await;
            let now = Instant::now();
            // The next tick a whole period from this one; a late tick (a
            // slow consumer) does not burst to catch up.
            self.next_tick = now + self.cadence.load_tick;
            if !self.collect_due(now) {
                return None;
            }
        }
    }
}

/// The records-only stream over `source` at `cadence`.
pub(crate) fn load_records(
    source: Arc<dyn RankedLoads>,
    cadence: Cadence,
) -> BoxStream<common::KvEventBatch> {
    let records = LoadRecordStream::new(source, cadence);
    Box::pin(stream::unfold(records, |mut records| async move {
        records
            .next()
            .await
            .map(|batch| (Ok::<_, Status>(batch), records))
    }))
}

/// The records-only stream of a servicer state: its ranks, each rank's
/// `GetLoads` figures with the engine link's liveness on the record.
pub(crate) fn load_records_of<S: RankLoads + 'static>(
    state: &Arc<S>,
) -> BoxStream<common::KvEventBatch> {
    load_records(
        Arc::new(LoadFromState(Arc::downgrade(state))),
        Cadence::default(),
    )
}

#[cfg(test)]
#[path = "../tests/unit/load_stream/tests.rs"]
mod tests;
