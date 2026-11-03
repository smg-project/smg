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
//! changed ([`load_changed`], checked every `load_tick`) or when its
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
        load_changed, unix_seconds, LoadFromState, RankLoads, DEFAULT_HEARTBEAT_BACKOFF,
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
        crate::kv_events::LoadSource::load(self, Some(rank))
    }
}

/// One rank's clock: what it last sent and when.
struct RankClock {
    last_record: Option<common::EngineLoad>,
    last_sent_at: Option<Instant>,
    idle_heartbeats: u32,
}

impl RankClock {
    const fn new() -> Self {
        Self {
            last_record: None,
            last_sent_at: None,
            idle_heartbeats: 0,
        }
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
            let changed = clock
                .last_record
                .as_ref()
                .is_none_or(|last| load_changed(last, &record));
            let due_after = if clock.idle_heartbeats >= 2 {
                self.cadence.heartbeat_backoff
            } else {
                self.cadence.heartbeat_interval
            };
            let silent_for = clock
                .last_sent_at
                .map_or(Duration::MAX, |at| now.saturating_duration_since(at));
            if !changed && silent_for < due_after {
                continue;
            }
            self.sample += 1;
            record.sample = self.sample;
            record.load_only = true;
            clock.idle_heartbeats = if changed {
                0
            } else {
                clock.idle_heartbeats.saturating_add(1)
            };
            clock.last_sent_at = Some(now);
            clock.last_record = Some(record.clone());
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
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex, PoisonError,
    };

    use futures::StreamExt;

    use super::*;

    /// A source of `ranks` ranks whose records are set by the test.
    struct Ranks {
        count: AtomicUsize,
        records: Mutex<Vec<Option<common::EngineLoad>>>,
    }

    fn record(running: u32, waiting: u32) -> common::EngineLoad {
        common::EngineLoad {
            running_requests: running,
            waiting_requests: waiting,
            token_usage: 0.25,
            gen_throughput: 1_000.0,
            max_running_requests: 64,
            cache_hit_rate: Some(0.5),
            ..Default::default()
        }
    }

    impl Ranks {
        fn new(records: Vec<Option<common::EngineLoad>>) -> Arc<Self> {
            Arc::new(Self {
                count: AtomicUsize::new(records.len()),
                records: Mutex::new(records),
            })
        }

        fn set(&self, rank: usize, record: Option<common::EngineLoad>) {
            self.records.lock().unwrap_or_else(PoisonError::into_inner)[rank] = record;
        }
    }

    impl RankedLoads for Ranks {
        fn rank_count(&self) -> usize {
            self.count.load(Ordering::Relaxed)
        }

        fn rank_record(&self, rank: i32) -> Option<common::EngineLoad> {
            let records = self.records.lock().unwrap_or_else(PoisonError::into_inner);
            records.get(usize::try_from(rank).ok()?).cloned().flatten()
        }
    }

    fn cadence() -> Cadence {
        Cadence {
            load_tick: Duration::from_millis(10),
            heartbeat_interval: Duration::from_millis(100),
            heartbeat_backoff: Duration::from_millis(300),
        }
    }

    async fn read(stream: &mut BoxStream<common::KvEventBatch>) -> common::KvEventBatch {
        tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("a batch in time")
            .expect("the stream is open")
            .expect("a batch")
    }

    /// Reads until `within` passes: what the stream sent in that window.
    async fn read_for(
        stream: &mut BoxStream<common::KvEventBatch>,
        within: Duration,
    ) -> Vec<common::KvEventBatch> {
        let mut batches = Vec::new();
        let deadline = tokio::time::sleep(within);
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                item = stream.next() => batches.push(item.expect("open").expect("a batch")),
                () = &mut deadline => return batches,
            }
        }
    }

    fn rank_of(batch: &common::KvEventBatch) -> i32 {
        batch.dp_rank.expect("a rank")
    }

    #[tokio::test(start_paused = true)]
    async fn every_rank_reports_at_once_then_on_change_and_heartbeat() {
        let ranks = Ranks::new(vec![
            Some(record(1, 0)),
            Some(record(2, 0)),
            Some(record(3, 0)),
        ]);
        let mut stream = load_records(Arc::clone(&ranks) as Arc<dyn RankedLoads>, cadence());

        // The first tick: one batch per rank, numbered 0, load_only, with the
        // telemetry, samples counted per stream.
        let mut first = Vec::new();
        for _ in 0..3 {
            first.push(read(&mut stream).await);
        }
        let mut seen: Vec<i32> = first.iter().map(rank_of).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2]);
        for (index, batch) in first.iter().enumerate() {
            assert_eq!(batch.sequence_number, 0);
            assert!(batch.events.is_empty());
            let load = batch.load.as_ref().expect("a record");
            assert!(load.load_only);
            assert_eq!(load.sample, index as u64 + 1);
            assert_eq!(
                load.cache_hit_rate,
                Some(0.5),
                "telemetry rides every record"
            );
            assert_eq!(
                load.running_requests,
                u32::try_from(rank_of(batch)).expect("rank") + 1,
                "each rank's own figures"
            );
        }

        // One rank moves: only its record goes out before the heartbeat
        // (time is paused: the windows below are exact).
        ranks.set(1, Some(record(2, 5)));
        let change = read(&mut stream).await;
        assert_eq!(rank_of(&change), 1);
        assert_eq!(change.load.as_ref().expect("a record").waiting_requests, 5);
        let quiet = read_for(&mut stream, Duration::from_millis(40)).await;
        assert!(quiet.is_empty(), "nothing else changed: {quiet:?}");

        // Nothing changes: every rank heartbeats at the interval on its own
        // clock (ranks 0 and 2 at 100 and 200 ms after their first record,
        // rank 1 at 110 and 210 ms, after its change), then, after two
        // unchanged heartbeats, at the backoff (300 ms later).
        let beats = read_for(&mut stream, Duration::from_millis(210)).await;
        let mut per_rank = [0_usize; 3];
        for beat in &beats {
            per_rank[usize::try_from(rank_of(beat)).expect("rank")] += 1;
        }
        assert_eq!(per_rank, [2, 2, 2], "two heartbeats per rank: {beats:?}");
        let backed_off = read_for(&mut stream, Duration::from_millis(220)).await;
        assert!(
            backed_off.is_empty(),
            "after two unchanged heartbeats the next waits for the backoff: {backed_off:?}"
        );
        let third = read_for(&mut stream, Duration::from_millis(60)).await;
        assert_eq!(
            third.len(),
            3,
            "the backoff heartbeat of every rank: {third:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_rank_without_a_record_is_skipped_and_the_count_may_grow() {
        let ranks = Ranks::new(vec![Some(record(1, 0)), None]);
        let mut stream = load_records(Arc::clone(&ranks) as Arc<dyn RankedLoads>, cadence());
        let first = read(&mut stream).await;
        assert_eq!(rank_of(&first), 0);
        let quiet = read_for(&mut stream, Duration::from_millis(40)).await;
        assert!(quiet.is_empty(), "rank 1 has nothing to report: {quiet:?}");

        // The engine of rank 1 comes up: its record joins at the next tick.
        ranks.set(1, Some(record(0, 0)));
        let joined = read(&mut stream).await;
        assert_eq!(rank_of(&joined), 1);
        assert_eq!(
            joined.load.as_ref().expect("a record").sample,
            2,
            "samples count the records actually sent"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_stream_ends_with_the_servicer() {
        let ranks = Ranks::new(vec![Some(record(1, 0))]);
        let mut stream = load_records(Arc::clone(&ranks) as Arc<dyn RankedLoads>, cadence());
        read(&mut stream).await;
        ranks.count.store(0, Ordering::Relaxed);
        let end = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("the stream ends in time");
        assert!(end.is_none(), "the stream ends once the state is gone");
    }

    #[test]
    fn the_default_cadence_is_the_relays() {
        let cadence = Cadence::default();
        assert_eq!(cadence.load_tick, DEFAULT_LOAD_TICK);
        assert_eq!(cadence.heartbeat_interval, DEFAULT_HEARTBEAT_INTERVAL);
        assert_eq!(cadence.heartbeat_backoff, DEFAULT_HEARTBEAT_BACKOFF);
    }
}
