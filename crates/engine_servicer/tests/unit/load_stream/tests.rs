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
