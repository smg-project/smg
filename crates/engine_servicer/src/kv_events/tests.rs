use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use smg_grpc_client::common_proto::{kv_cache_event, KvCacheLocality, KvCacheTier};
use tokio::time::timeout;
use zeromq::{PubSocket, RouterSocket, SocketEvent};

use super::{golden, *};
use crate::kv_wire::WireBatch;

/// A state the relay reads loads from: one rank's fixed record and a
/// fresh engine link (nothing reported yet).
struct FixedLoads {
    rank: i32,
    record: common::EngineLoad,
    engine: EngineLink,
}

impl RankLoads for FixedLoads {
    fn rank_load(&self, dp_rank: Option<i32>) -> Option<common::EngineLoad> {
        (dp_rank.is_none() || dp_rank == Some(self.rank)).then(|| self.record.clone())
    }

    fn engine_link(&self) -> &EngineLink {
        &self.engine
    }
}

#[test]
fn load_from_state_adds_the_links_liveness_and_ends_with_the_state() {
    // The state's record carries stale figures in the three liveness
    // fields; the link has reported nothing, so the source must replace
    // them with its `None`s.
    let state = Arc::new(FixedLoads {
        rank: 1,
        record: common::EngineLoad {
            running_requests: 3,
            engine_silence_ms: Some(7),
            engine_reports_steps: Some(true),
            prefill_pending_tokens: Some(11),
            ..common::EngineLoad::default()
        },
        engine: EngineLink::default(),
    });
    assert_eq!(state.engine.output_silence_ms(), None, "a fresh link");
    let source = LoadFromState(Arc::downgrade(&state));
    let record = source.load(Some(1)).expect("the state's record");
    assert_eq!(record.running_requests, 3);
    assert_eq!(
        record.engine_silence_ms, None,
        "the link's silence, not the stale figure"
    );
    assert_eq!(
        record.engine_reports_steps, None,
        "the link's step reporting"
    );
    assert_eq!(
        record.prefill_pending_tokens, None,
        "the link's prefill backlog"
    );
    assert!(
        source.load(Some(0)).is_none(),
        "a rank the state has no record for"
    );
    drop(state);
    assert!(
        source.load(None).is_none(),
        "nothing after the state is gone"
    );
}

fn decode(hex: &str) -> WireBatch {
    rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&golden::bytes(hex))
        .expect("golden batch decodes")
        .0
}

fn convert_batch(
    batch: WireBatch,
    sequence_number: u64,
    event_id: &mut u64,
) -> common::KvEventBatch {
    Normalizer::new().normalize_batch(batch, sequence_number, event_id)
}

fn stored(event: &common::KvCacheEvent) -> &common::KvBlocksStored {
    match &event.data {
        Some(kv_cache_event::Data::Stored(stored)) => stored,
        other => panic!("expected a stored event, got {other:?}"),
    }
}

fn block(block_hash: i64, token_ids: Vec<u32>, lora_id: Option<i64>) -> common::KvBlock {
    common::KvBlock {
        block_hash,
        block_size: i32::try_from(token_ids.len()).unwrap(),
        token_ids,
        lora_id,
        cache_level: None,
        ..Default::default()
    }
}

#[test]
fn the_relay_starts_at_boot_unless_told_to_wait() {
    assert!(starts_at_boot(None));
    assert!(starts_at_boot(Some("")));
    assert!(starts_at_boot(Some("eager")));
    assert!(!starts_at_boot(Some("lazy")));
    assert!(!starts_at_boot(Some(" Lazy ")));
}

fn relay_dialling(endpoint: &str) -> Arc<KvEventRelay> {
    KvEventRelay::new(RelayConfig {
        sources: vec![RankSource {
            rank: 0,
            endpoint: endpoint.to_string(),
            replay_endpoint: None,
        }],
        topic: "kv".to_string(),
        history_batches: 100,
        history_bytes: 64 << 20,
        replay_timeout: Duration::from_secs(2),
        load_tick: DEFAULT_LOAD_TICK,
        heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
    })
}

/// The relay dials at the servicer's start, minutes before the engine
/// binds its publisher: a publisher bound seconds later is joined within
/// a second of the bind (the relay's own knock every second), not at the
/// ZMQ crate's own back-off (50 ms doubling to 30 s), which would put the
/// join after a planned warm-up.
#[tokio::test]
async fn a_publisher_bound_after_the_relays_start_is_joined_within_a_second() {
    let endpoint = format!(
        "tcp://127.0.0.1:{}",
        portpicker::pick_unused_port().expect("a free publisher port")
    );
    let relay = relay_dialling(&endpoint);
    assert!(relay.subscribed(), "nothing to wait for before the start");
    relay.start();
    assert!(!relay.subscribed(), "dialling a port nobody has bound");
    // The engine's start: by now the crate's own back-off would be
    // sleeping for seconds between dials.
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(!relay.subscribed(), "still nobody bound");
    let mut publisher = PubSocket::new();
    publisher.bind(&endpoint).await.expect("publisher binds");
    let bound = Instant::now();
    while !relay.subscribed() {
        assert!(
            bound.elapsed() < Duration::from_millis(1_500),
            "the relay took {:?} to join a publisher bound after its start",
            bound.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // The subscription is live: the publisher's first batch is relayed.
    let batch1 = golden::bytes(golden::BATCH1);
    for _ in 0..250 {
        publisher
            .send(golden::frame(b"kv", 0, &batch1))
            .await
            .expect("publish");
        if relay.counts().relayed >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(relay.counts().relayed, 1, "the first batch after the join");
}

/// A relay whose dial fails for good (an endpoint no transport serves)
/// has nothing a warm-up should wait for; its subscriptions say why.
#[tokio::test]
async fn a_relay_that_gave_up_has_nothing_to_wait_for() {
    let relay = relay_dialling("bogus://nowhere");
    relay.start();
    for _ in 0..250 {
        if relay.subscribed() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(relay.subscribed(), "the task gave up and said so");
    let refused = relay
        .subscribe(common::SubscribeKvEventsRequest {
            start_sequence_number: 0,
            load_records_before_first_batch: false,
        })
        .err()
        .expect("a down relay refuses");
    assert_eq!(refused.code(), tonic::Code::Internal);
    assert!(refused.message().contains("is down"), "{refused}");
}

#[test]
fn endpoint_for_rank_mirrors_the_python_helper() {
    assert_eq!(endpoint_for_rank("tcp://*:5557", 0), "tcp://127.0.0.1:5557");
    assert_eq!(
        endpoint_for_rank("tcp://0.0.0.0:5557", 0),
        "tcp://127.0.0.1:5557"
    );
    assert_eq!(endpoint_for_rank("tcp://*:5557", 2), "tcp://127.0.0.1:5559");
    assert_eq!(
        endpoint_for_rank("tcp://10.0.0.1:5557", 1),
        "tcp://10.0.0.1:5558"
    );
    assert_eq!(endpoint_for_rank("tcp://host:port", 1), "tcp://host:port");
    assert_eq!(endpoint_for_rank("ipc:///tmp/kv", 1), "ipc:///tmp/kv");
}

/// The sources of a data-parallel publisher are laid out as the engines
/// lay their ranks out (and as the Python relay derives them): rank `r`
/// on the base port plus `r`, its replay socket likewise.
#[test]
fn rank_sources_lay_the_ranks_out_like_the_python_relay() {
    assert_eq!(
        rank_sources("tcp://*:5557", Some("tcp://*:5600"), 2),
        vec![
            RankSource {
                rank: 0,
                endpoint: "tcp://127.0.0.1:5557".to_string(),
                replay_endpoint: Some("tcp://127.0.0.1:5600".to_string()),
            },
            RankSource {
                rank: 1,
                endpoint: "tcp://127.0.0.1:5558".to_string(),
                replay_endpoint: Some("tcp://127.0.0.1:5601".to_string()),
            },
        ]
    );
    // No replay socket; a width below one is one rank.
    let lone = rank_sources("tcp://*:6000", Some(""), 0);
    assert_eq!(lone.len(), 1);
    assert_eq!(
        (
            lone[0].rank,
            lone[0].endpoint.as_str(),
            lone[0].replay_endpoint.as_deref()
        ),
        (0, "tcp://127.0.0.1:6000", None)
    );
    let config = RelayConfig::for_publisher("tcp://0.0.0.0:5557", None, "kv", 3);
    assert_eq!(
        config
            .sources
            .iter()
            .map(|source| source.endpoint.as_str())
            .collect::<Vec<_>>(),
        [
            "tcp://127.0.0.1:5557",
            "tcp://127.0.0.1:5558",
            "tcp://127.0.0.1:5559"
        ]
    );
    assert_eq!(config.label(), "tcp://127.0.0.1:5557");
}

/// The golden batches convert to exactly what the Python relay produced
/// for them: sha256 hashes reduced to their low 64 bits, an unaligned
/// store skipped but its event id consumed, `lora_id` and the parent
/// carried, `dp_rank` set only when the publisher set it.
#[test]
fn golden_batches_convert_like_the_python_relay() {
    let mut event_id = 0;
    let batch = convert_batch(decode(golden::BATCH1), 9, &mut event_id);
    assert_eq!(event_id, 5);
    assert_eq!(batch.sequence_number, 9);
    assert!((batch.timestamp - 1_700_000_000.5).abs() < f64::EPSILON);
    assert_eq!(batch.dp_rank, None);
    assert_eq!(
        batch
            .events
            .iter()
            .map(|event| event.event_id)
            .collect::<Vec<_>>(),
        vec![1, 2, 4, 5]
    );
    let first = stored(&batch.events[0]);
    assert_eq!(first.parent_block_hash, Some(7));
    assert_eq!(
        first.blocks,
        vec![
            block(i64::MIN, vec![1, 2, 3, 4], None),
            block(-2, vec![5, 6, 7, 8], None),
        ]
    );
    let second = stored(&batch.events[1]);
    assert_eq!(second.parent_block_hash, None);
    assert_eq!(second.blocks, vec![block(0x1234, vec![9, 10], Some(3))]);
    assert_eq!(
        batch.events[2].data,
        Some(kv_cache_event::Data::Removed(common::KvBlocksRemoved {
            block_hashes: vec![0x1234, -1],
            cache_level: None,
            tier: Some(KvCacheTier::Device as i32),
            medium: Some("GPU".to_string()),
            locality: Some(KvCacheLocality::Local as i32),
            ..Default::default()
        }))
    );
    assert_eq!(
        batch.events[3].data,
        Some(kv_cache_event::Data::Cleared(
            common::KvCacheCleared::default()
        ))
    );

    let batch = convert_batch(decode(golden::BATCH2), 10, &mut event_id);
    assert_eq!(event_id, 6);
    assert_eq!(batch.sequence_number, 10);
    assert_eq!(batch.dp_rank, Some(1));
    assert_eq!(batch.events[0].event_id, 6);
    let only = stored(&batch.events[0]);
    assert_eq!(only.parent_block_hash, Some(41));
    assert_eq!(only.blocks, vec![block(42, vec![100, 101], None)]);
}

/// The batch array may omit the trailing rank and may grow new fields;
/// an event of a type this relay does not convert is skipped on its own
/// (consuming its event id), as the Python relay skips unknown types.
#[test]
fn batch_layout_tolerates_an_omitted_rank_and_trailing_fields() {
    let short = rmp_serde::to_vec(&(1.5f64, Vec::<u8>::new())).unwrap();
    let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&short)
        .unwrap()
        .0;
    assert!(batch.events.is_empty());
    assert_eq!(batch.dp_rank, None);

    let long = rmp_serde::to_vec(&(1.5f64, Vec::<u8>::new(), 2i32, "future")).unwrap();
    let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&long)
        .unwrap()
        .0;
    assert_eq!(batch.dp_rank, Some(2));

    let unknown = rmp_serde::to_vec(&serde_json::json!([
        1.5,
        [{"type": "Mystery", "x": 1}, {"type": "AllBlocksCleared"}]
    ]))
    .unwrap();
    let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&unknown)
        .unwrap()
        .0;
    let mut event_id = 4;
    let converted = convert_batch(batch, 7, &mut event_id);
    assert_eq!(event_id, 6, "the skipped event still consumed an id");
    assert_eq!(converted.events.len(), 1);
    assert_eq!(converted.events[0].event_id, 6);
    assert!(matches!(
        converted.events[0].data,
        Some(kv_cache_event::Data::Cleared(_))
    ));
}

#[test]
fn frames_are_split_like_the_python_relay() {
    let payload = golden::bytes(golden::BATCH2);
    let message = golden::frame(b"kv", 5, &payload);
    let (sequence, body) = split_frames(&message).expect("three frames");
    assert_eq!(sequence, 5);
    assert_eq!(body, payload.as_slice());

    let mut short = ZmqMessage::from(b"kv".to_vec());
    short.push_back(5u64.to_be_bytes().to_vec().into());
    assert!(split_frames(&short).is_none());

    // A shorter sequence frame zero-extends, as `int.from_bytes` does.
    let mut narrow = ZmqMessage::from(b"kv".to_vec());
    narrow.push_back(vec![1, 2].into());
    narrow.push_back(payload.into());
    assert_eq!(
        split_frames(&narrow).map(|(sequence, _)| sequence),
        Some(0x0102)
    );
}

struct Lab {
    publisher: PubSocket,
    router: Option<RouterSocket>,
    relay: Arc<KvEventRelay>,
}

/// A local publisher (and replay ROUTER when asked) with a started relay.
async fn start_lab(history_batches: usize, with_replay: bool) -> Lab {
    let mut publisher = PubSocket::new();
    let endpoint = publisher
        .bind("tcp://127.0.0.1:0")
        .await
        .expect("publisher binds")
        .to_string();
    let (router, replay_endpoint) = if with_replay {
        let mut router = RouterSocket::new();
        let endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("router binds")
            .to_string();
        (Some(router), Some(endpoint))
    } else {
        (None, None)
    };
    let relay = KvEventRelay::new(RelayConfig {
        sources: vec![RankSource {
            rank: 0,
            endpoint,
            replay_endpoint,
        }],
        topic: "kv".to_string(),
        history_batches,
        history_bytes: 64 << 20,
        replay_timeout: Duration::from_secs(2),
        load_tick: DEFAULT_LOAD_TICK,
        heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
    });
    relay.start();
    let mut lab = Lab {
        publisher,
        router,
        relay,
    };
    if with_replay {
        // The relay asks the replay socket for the publisher's start
        // before anything else; this publisher has nothing yet.
        lab.answer_replay(0, &[]).await;
    }
    lab
}

/// A load source the tests steer: the record the relay attaches, and how
/// often it was asked.
struct StubLoads {
    record: Mutex<common::EngineLoad>,
    asked: AtomicU64,
}

impl StubLoads {
    fn new(running: u32, waiting: u32) -> Arc<Self> {
        Arc::new(Self {
            record: Mutex::new(common::EngineLoad {
                running_requests: running,
                waiting_requests: waiting,
                waiting_uncached_tokens: Some(4_096),
                token_usage: 0.25,
                gen_throughput: 1_200.0,
                max_running_requests: 64,
                cache_hit_rate: Some(0.5),
                num_used_tokens: Some(1_024),
                max_total_num_tokens: Some(4_096),
                ..Default::default()
            }),
            asked: AtomicU64::new(0),
        })
    }

    fn set(&self, running: u32, waiting: u32) {
        let mut record = self.record.lock().unwrap_or_else(PoisonError::into_inner);
        record.running_requests = running;
        record.waiting_requests = waiting;
    }
}

impl LoadSource for StubLoads {
    fn load(&self, _dp_rank: Option<i32>) -> Option<common::EngineLoad> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        Some(
            self.record
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone(),
        )
    }
}

/// A relay with a load source and short record intervals (`tick`,
/// heartbeat, backoff), primed with sequence 0.
async fn start_lab_with_loads(
    loads: Arc<StubLoads>,
    tick: Duration,
    heartbeat: Duration,
    backoff: Duration,
) -> Lab {
    start_lab_with_loads_at(loads, tick, heartbeat, backoff, true).await
}

/// [`start_lab_with_loads`], with or without the publisher's first batch.
async fn start_lab_with_loads_at(
    loads: Arc<StubLoads>,
    tick: Duration,
    heartbeat: Duration,
    backoff: Duration,
    prime: bool,
) -> Lab {
    let mut publisher = PubSocket::new();
    let endpoint = publisher
        .bind("tcp://127.0.0.1:0")
        .await
        .expect("publisher binds")
        .to_string();
    let relay = KvEventRelay::new(RelayConfig {
        sources: vec![RankSource {
            rank: 0,
            endpoint,
            replay_endpoint: None,
        }],
        topic: "kv".to_string(),
        history_batches: 100,
        history_bytes: 64 << 20,
        replay_timeout: Duration::from_secs(2),
        load_tick: tick,
        heartbeat_interval: heartbeat,
        heartbeat_backoff: backoff,
    });
    relay.set_load_source(loads);
    relay.start();
    let mut lab = Lab {
        publisher,
        router: None,
        relay,
    };
    if prime {
        lab.prime().await;
    }
    lab
}

fn record_of(batch: &common::KvEventBatch) -> &common::EngineLoad {
    batch.load.as_ref().expect("a load record on the batch")
}

/// Every batch a subscriber receives, whether from the history, live or
/// a snapshot chunk, carries the servicer's load record, numbered per
/// stream; a relay without a source sends none.
#[tokio::test]
async fn every_batch_a_subscriber_receives_carries_the_load_record() {
    let loads = StubLoads::new(3, 1);
    let mut lab = start_lab_with_loads(
        loads,
        Duration::from_secs(10),
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await;
    let batch2 = golden::bytes(golden::BATCH2);
    lab.publish(1, &batch2).await;
    lab.wait_relayed(2).await;
    // History first (0 and 1), then live (2).
    let mut stream = lab.subscribe(0).expect("history");
    let first = read(&mut stream).await;
    assert_eq!(first.sequence_number, 0);
    let record = record_of(&first);
    assert_eq!(
        (
            record.running_requests,
            record.waiting_requests,
            record.waiting_uncached_tokens,
            record.sample,
            record.load_only
        ),
        (3, 1, Some(4_096), 1, false)
    );
    // The first record carries the telemetry; the next event batch's
    // record is the core only.
    assert_eq!(record.cache_hit_rate, Some(0.5));
    assert_eq!(record.max_total_num_tokens, Some(4_096));
    let second = read(&mut stream).await;
    assert_eq!(record_of(&second).sample, 2);
    assert_eq!(record_of(&second).cache_hit_rate, None);
    assert_eq!(record_of(&second).max_total_num_tokens, None);
    lab.publish(2, &batch2).await;
    let live = read(&mut stream).await;
    assert_eq!((live.sequence_number, record_of(&live).sample), (2, 3));
    // A snapshot chunk (the window of this second lab has no start).
    let mut rolled = start_lab_with_loads(
        StubLoads::new(1, 0),
        Duration::from_secs(10),
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await;
    rolled.publish(1, &batch2).await;
    rolled.wait_relayed(2).await;
    let mut whole = rolled.subscribe(0).expect("history");
    assert_eq!(record_of(&read(&mut whole).await).running_requests, 1);
    // No source: no record.
    let mut plain = start_lab(10, false).await;
    plain.prime().await;
    let mut bare = plain.subscribe(0).expect("history");
    assert!(read(&mut bare).await.load.is_none());
}

/// While the publisher is quiet the stream still speaks: a `load_only`
/// batch (no events, the last sequence repeated) when the record moved,
/// a heartbeat after the interval, and after two unchanged heartbeats
/// only every backoff interval.
#[tokio::test]
async fn a_quiet_publishers_stream_sends_load_only_batches_and_heartbeats() {
    let loads = StubLoads::new(2, 0);
    let mut lab = start_lab_with_loads(
        Arc::clone(&loads),
        Duration::from_millis(20),
        Duration::from_millis(200),
        Duration::from_millis(600),
    )
    .await;
    let mut stream = lab.subscribe(0).expect("history");
    let first = read(&mut stream).await;
    assert_eq!(first.sequence_number, 0);
    // A change with no KV event: a load-only batch within a tick or two.
    let started = Instant::now();
    loads.set(5, 2);
    let change = read(&mut stream).await;
    let record = record_of(&change);
    assert!(load_only_marker(&change), "{change:?}");
    assert_eq!((change.sequence_number, change.events.len()), (0, 0));
    assert_eq!((record.running_requests, record.waiting_requests), (5, 2));
    // A heartbeat carries the telemetry.
    assert_eq!(record.cache_hit_rate, Some(0.5));
    assert!(
        started.elapsed() < Duration::from_millis(150),
        "the change took {:?}",
        started.elapsed()
    );
    // Nothing changes: heartbeats at the interval, then at the backoff.
    let mut gaps = Vec::new();
    let mut last = Instant::now();
    for _ in 0..4 {
        let beat = read(&mut stream).await;
        assert!(load_only_marker(&beat));
        assert_eq!(record_of(&beat).running_requests, 5);
        gaps.push(last.elapsed());
        last = Instant::now();
    }
    assert!(
        gaps[0] >= Duration::from_millis(150) && gaps[0] < Duration::from_millis(500),
        "first heartbeat after {:?}",
        gaps[0]
    );
    assert!(
        gaps[1] < Duration::from_millis(500),
        "second heartbeat after {:?}",
        gaps[1]
    );
    assert!(
        gaps[2] >= Duration::from_millis(500) && gaps[3] >= Duration::from_millis(500),
        "backed off: {gaps:?}"
    );
    // A real batch resets the backoff and carries the record itself.
    lab.publish(1, &golden::bytes(golden::BATCH2)).await;
    let live = read(&mut stream).await;
    assert_eq!(live.sequence_number, 1);
    assert!(!load_only_marker(&live));
    let beat = read(&mut stream).await;
    assert!(load_only_marker(&beat));
    assert_eq!(beat.sequence_number, 1, "repeats the last sequence sent");
}

fn load_only_marker(batch: &common::KvEventBatch) -> bool {
    batch.load.as_ref().is_some_and(|load| load.load_only)
}

/// A frozen engine's record changes in nothing but its silence; that goes
/// out once a second, so a subscriber's wedge threshold is met within a
/// second of the engine crossing it instead of at the heartbeat backoff.
#[test]
fn a_growing_engine_silence_changes_the_record_once_a_second() {
    let mut last = common::EngineLoad::default();
    let mut current = last.clone();
    current.engine_silence_ms = Some(900);
    assert!(load_changed(&last, &current), "a silence appearing");
    last.engine_silence_ms = Some(100);
    assert!(!load_changed(&last, &current), "within the same second");
    current.engine_silence_ms = Some(1_000);
    assert!(load_changed(&last, &current), "a whole second crossed");
    current.engine_silence_ms = None;
    assert!(load_changed(&last, &current), "the silence ending");
}

/// A relay whose replay socket is not bound yet: the relay keeps asking
/// for the publisher's start until it is.
async fn start_lab_without_replay_socket_yet() -> (Lab, String) {
    let mut publisher = PubSocket::new();
    let endpoint = publisher
        .bind("tcp://127.0.0.1:0")
        .await
        .expect("publisher binds")
        .to_string();
    let replay_endpoint = format!(
        "tcp://127.0.0.1:{}",
        portpicker::pick_unused_port().expect("a free replay port")
    );
    let relay = KvEventRelay::new(RelayConfig {
        sources: vec![RankSource {
            rank: 0,
            endpoint,
            replay_endpoint: Some(replay_endpoint.clone()),
        }],
        topic: "kv".to_string(),
        history_batches: 100,
        history_bytes: 64 << 20,
        replay_timeout: Duration::from_secs(2),
        load_tick: DEFAULT_LOAD_TICK,
        heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
    });
    relay.start();
    (
        Lab {
            publisher,
            router: None,
            relay,
        },
        replay_endpoint,
    )
}

impl Lab {
    async fn publish(&mut self, sequence: u64, payload: &[u8]) {
        self.publisher
            .send(golden::frame(b"kv", sequence, payload))
            .await
            .expect("publish");
    }

    /// The subscription reaches the publisher a moment after the
    /// connect; publish sequence 0 until the relay has it.
    async fn prime(&mut self) {
        let batch1 = golden::bytes(golden::BATCH1);
        for _ in 0..250 {
            self.publish(0, &batch1).await;
            if self.relay.counts().relayed >= 1 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the relay never saw sequence 0");
    }

    async fn wait_relayed(&self, count: u64) {
        for _ in 0..250 {
            if self.relay.counts().relayed >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!(
            "the relay relayed {} batches, not {count}",
            self.relay.counts().relayed
        );
    }

    fn subscribe(&self, cursor: u64) -> Result<BoxStream<common::KvEventBatch>, Status> {
        self.relay.subscribe(common::SubscribeKvEventsRequest {
            start_sequence_number: cursor,
            load_records_before_first_batch: false,
        })
    }

    /// [`Self::subscribe`] as a gateway that knows `load_only` does: it
    /// asks for the records before the publisher's first batch.
    fn subscribe_from_the_start(
        &self,
        cursor: u64,
    ) -> Result<BoxStream<common::KvEventBatch>, Status> {
        self.relay.subscribe(common::SubscribeKvEventsRequest {
            start_sequence_number: cursor,
            load_records_before_first_batch: true,
        })
    }

    /// Answer the next replay request as vLLM's ROUTER does, with
    /// `replies` and the end marker; the request must ask from `start`.
    async fn answer_replay(&mut self, start: u64, replies: &[(u64, Vec<u8>)]) {
        let router = self.router.as_mut().expect("a replay socket");
        let request = timeout(Duration::from_secs(5), router.recv())
            .await
            .expect("a replay request in time")
            .expect("request");
        let frames = replay_request_frames(&request, start);
        self.send_replay(&frames, replies).await;
    }

    /// Publish `sequence` (the subscription reaches the publisher a
    /// moment after the connect) until the relay asks the replay socket,
    /// which it must do from `start`; the request's frames.
    async fn publish_until_replay_requested(
        &mut self,
        sequence: u64,
        payload: &[u8],
        start: u64,
    ) -> Vec<Vec<u8>> {
        let router = self.router.as_mut().expect("a replay socket");
        for _ in 0..250 {
            self.publisher
                .send(golden::frame(b"kv", sequence, payload))
                .await
                .expect("publish");
            if let Ok(request) = timeout(Duration::from_millis(20), router.recv()).await {
                return replay_request_frames(&request.expect("request"), start);
            }
        }
        panic!("the relay never asked the replay socket");
    }

    /// Reply to the request `frames` with `replies` and the end marker.
    async fn send_replay(&mut self, frames: &[Vec<u8>], replies: &[(u64, Vec<u8>)]) {
        let router = self.router.as_mut().expect("a replay socket");
        for (sequence, payload) in replies {
            let mut reply = ZmqMessage::from(frames[0].clone());
            reply.push_back(Vec::new().into());
            reply.push_back(b"kv".to_vec().into());
            reply.push_back(sequence.to_be_bytes().to_vec().into());
            reply.push_back(payload.clone().into());
            router.send(reply).await.expect("reply");
        }
        let mut end = ZmqMessage::from(frames[0].clone());
        end.push_back(Vec::new().into());
        end.push_back(Vec::new().into());
        end.push_back(END_SEQUENCE.to_vec().into());
        end.push_back(Vec::new().into());
        router.send(end).await.expect("end marker");
    }
}

/// A replay request's frames (`[identity, empty, start]`), checked to
/// ask from `start`.
fn replay_request_frames(request: &ZmqMessage, start: u64) -> Vec<Vec<u8>> {
    let frames: Vec<Vec<u8>> = request.iter().map(|frame| frame.to_vec()).collect();
    assert_eq!(frames.len(), 3, "[identity, empty, start]");
    assert_eq!(frames[1], b"");
    assert_eq!(frames[2], start.to_be_bytes());
    frames
}

fn refused(result: Result<BoxStream<common::KvEventBatch>, Status>) -> Status {
    match result {
        Err(status) => status,
        Ok(_) => panic!("the subscription was accepted"),
    }
}

async fn read(stream: &mut BoxStream<common::KvEventBatch>) -> common::KvEventBatch {
    timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("a batch in time")
        .expect("stream open")
        .expect("a batch")
}

async fn read_error(stream: &mut BoxStream<common::KvEventBatch>) -> Status {
    timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("an item in time")
        .expect("stream open")
        .expect_err("an error")
}

async fn read_end(stream: &mut BoxStream<common::KvEventBatch>) {
    assert!(timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("the end in time")
        .is_none());
}

#[tokio::test]
async fn history_serves_a_cursor_inside_the_window_then_live() {
    let mut lab = start_lab(100, false).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    for sequence in 1..=5 {
        lab.publish(sequence, &batch2).await;
    }
    lab.wait_relayed(6).await;

    let mut stream = lab.subscribe(2).expect("inside the window");
    for expected in 3..=5 {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }
    lab.publish(6, &batch2).await;
    assert_eq!(read(&mut stream).await.sequence_number, 6);

    // The newest sequence as the cursor: nothing owed, live from here.
    let mut caught_up = lab.subscribe(6).expect("at the newest");
    lab.publish(7, &batch2).await;
    assert_eq!(read(&mut caught_up).await.sequence_number, 7);
    assert_eq!(read(&mut stream).await.sequence_number, 7);
    assert_eq!(lab.relay.counts().served_from_history, 2);
}

#[tokio::test]
async fn cursors_outside_the_window_are_out_of_range() {
    let fresh = start_lab(10, false).await;
    let status = refused(fresh.subscribe(1));
    assert_eq!(status.code(), tonic::Code::OutOfRange);
    assert!(status.message().contains("holds no history yet"));

    let mut lab = start_lab(3, false).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    for sequence in 1..=9 {
        lab.publish(sequence, &batch2).await;
    }
    lab.wait_relayed(10).await;
    // The window is 7..=9: cursor 6 wants 7, served; cursor 5 wants 6, gone.
    let mut stream = lab.subscribe(6).expect("the oldest batch is wanted");
    for expected in 7..=9 {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }
    let behind = refused(lab.subscribe(5));
    assert_eq!(behind.code(), tonic::Code::OutOfRange);
    assert!(behind.message().contains("from sequence 7"));
    let ahead = refused(lab.subscribe(20));
    assert_eq!(ahead.code(), tonic::Code::OutOfRange);
    assert!(ahead.message().contains("publisher restarted"));
    assert_eq!(lab.relay.counts().out_of_range, 2);
}

#[tokio::test]
async fn a_subscriber_without_a_cursor_gets_the_whole_history_or_a_snapshot() {
    let mut lab = start_lab(100, false).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    lab.publish(1, &batch2).await;
    lab.publish(2, &batch2).await;
    lab.wait_relayed(3).await;
    // Everything since the publisher's first batch is here: hand it over.
    let mut stream = lab.subscribe(0).expect("live");
    for expected in 0..=2 {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }
    lab.publish(3, &batch2).await;
    assert_eq!(read(&mut stream).await.sequence_number, 3);

    // A relay that joined after the publisher's first batches has an
    // incomplete window: what it knows of the state, as a snapshot cut
    // at its newest sequence, then live.
    let mut late = start_lab(100, false).await;
    for _ in 0..250 {
        late.publish(5, &batch2).await;
        if late.relay.counts().relayed >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(late.relay.counts().relayed, 1, "the relay saw sequence 5");
    let mut stream = late.subscribe(0).expect("a snapshot");
    let chunk = read(&mut stream).await;
    assert_eq!(chunk.sequence_number, 5, "cut at the newest sequence");
    assert_eq!(
        chunk.snapshot,
        Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: 1,
            unknown_before: 5,
        }),
        "the five sequences before the relay joined are unknown"
    );
    assert_eq!(
        chunk.dp_rank,
        Some(0),
        "the socket's rank, not the rank the payload names"
    );
    assert!(is_cleared(&chunk.events[0]));
    assert_eq!(stored_hashes(&chunk), vec![(Some(41), vec![42])]);
    late.publish(6, &batch2).await;
    assert_eq!(
        read(&mut stream).await.sequence_number,
        6,
        "live from the next sequence"
    );
    assert_eq!(late.relay.counts().served_snapshots, 1);
}

#[tokio::test]
async fn a_publisher_gap_is_filled_from_the_engines_replay() {
    let mut lab = start_lab(100, true).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    lab.publish(1, &batch2).await;
    lab.wait_relayed(2).await;
    let mut stream = lab.subscribe(1).expect("caught up");

    // 2 and 3 never reach the SUB; 4 reveals the gap.
    lab.publish(4, &batch2).await;
    lab.answer_replay(
        2,
        &[
            (2, batch2.clone()),
            (3, batch2.clone()),
            (4, batch2.clone()),
        ],
    )
    .await;
    for expected in 2..=4 {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }
    lab.publish(5, &batch2).await;
    assert_eq!(read(&mut stream).await.sequence_number, 5);
    let counts = lab.relay.counts();
    assert_eq!(
        (
            counts.publisher_gaps,
            counts.gap_batches_recovered,
            counts.gap_batches_lost
        ),
        (1, 3, 0)
    );
    assert_eq!(
        counts.relayed, 6,
        "the replay's copy of 4 was relayed, the live one skipped"
    );
    // The window is complete: a cursor inside it is served across the
    // recovered stretch.
    let mut resumed = lab.subscribe(1).expect("served");
    for expected in 2..=5 {
        assert_eq!(read(&mut resumed).await.sequence_number, expected);
    }
}

#[tokio::test]
async fn a_gap_the_engine_cannot_fill_leaves_a_hole_not_a_dead_end() {
    let mut lab = start_lab(100, false).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    lab.publish(1, &batch2).await;
    lab.wait_relayed(2).await;
    let mut stream = lab.subscribe(1).expect("caught up");
    lab.publish(4, &batch2).await;
    assert_eq!(
        read(&mut stream).await.sequence_number,
        4,
        "the live stream jumps"
    );
    let counts = lab.relay.counts();
    assert_eq!(
        (
            counts.publisher_gaps,
            counts.gap_batches_recovered,
            counts.gap_batches_lost
        ),
        (1, 0, 2)
    );
    // A resume from inside the hole or before it skips it, as the gateway
    // then settles the gap itself instead of looping on OUT_OF_RANGE.
    let mut from_before = lab.subscribe(1).expect("served");
    assert_eq!(read(&mut from_before).await.sequence_number, 4);
    let mut from_inside = lab.subscribe(2).expect("served");
    assert_eq!(read(&mut from_inside).await.sequence_number, 4);
    // The window has a hole, so it is not the publisher's whole state: a
    // subscriber without a cursor gets the snapshot (both copies of block
    // 42, from sequences 1 and 4) cut at 4, then live.
    let mut no_cursor = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut no_cursor).await;
    assert_eq!(chunk.sequence_number, 4);
    assert_eq!(chunk.snapshot.as_ref().map(|chunk| chunk.blocks), Some(2));
    assert_eq!(
        stored_hashes(&chunk),
        vec![(Some(41), vec![42]), (Some(41), vec![42])]
    );
    lab.publish(5, &batch2).await;
    assert_eq!(read(&mut no_cursor).await.sequence_number, 5);
}

#[tokio::test]
async fn a_partial_replay_recovers_what_it_can_and_loses_the_rest() {
    let mut lab = start_lab(100, true).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    let mut stream = lab.subscribe(0).expect("live");
    assert_eq!(read(&mut stream).await.sequence_number, 0);
    lab.publish(5, &batch2).await;
    // The engine's buffer starts at 3: 1 and 2 are gone for good.
    lab.answer_replay(1, &[(3, batch2.clone()), (4, batch2.clone())])
        .await;
    for expected in [3, 4, 5] {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }
    let counts = lab.relay.counts();
    assert_eq!(
        (counts.gap_batches_recovered, counts.gap_batches_lost),
        (2, 2)
    );
}

#[tokio::test]
async fn undecodable_payloads_relay_as_empty_batches() {
    let mut lab = start_lab(100, false).await;
    lab.prime().await;
    let mut stream = lab.subscribe(0).expect("live");
    assert_eq!(read(&mut stream).await.events.len(), 4);
    lab.publish(1, b"not msgpack").await;
    let empty = read(&mut stream).await;
    assert_eq!((empty.sequence_number, empty.events.len()), (1, 0));
    let mut short = ZmqMessage::from(b"kv".to_vec());
    short.push_back(2u64.to_be_bytes().to_vec().into());
    lab.publisher.send(short).await.expect("publish");
    lab.publish(2, &golden::bytes(golden::BATCH2)).await;
    assert_eq!(
        read(&mut stream).await.sequence_number,
        2,
        "a short frame is nothing"
    );
    assert_eq!(lab.relay.counts().undecodable_batches, 1);
}

#[tokio::test]
async fn a_sequence_regression_ends_live_streams_and_clears_the_history() {
    let mut lab = start_lab(100, false).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    for sequence in 1..=3 {
        lab.publish(sequence, &batch2).await;
    }
    lab.wait_relayed(4).await;
    let mut stream = lab.subscribe(0).expect("whole history");
    for expected in 0..=3 {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }

    // The publisher restarts and counts from 0 again.
    lab.publish(0, &batch2).await;
    let status = read_error(&mut stream).await;
    assert_eq!(status.code(), tonic::Code::DataLoss);
    assert!(status.message().contains("restarted"));
    read_end(&mut stream).await;
    assert_eq!(lab.relay.counts().publisher_restarts, 1);

    // The old incarnation's cursor is refused; the new one's complete
    // history is handed to a fresh subscriber.
    let stale = refused(lab.subscribe(3));
    assert_eq!(stale.code(), tonic::Code::OutOfRange);
    let mut fresh = lab
        .subscribe(0)
        .expect("the new incarnation from its start");
    assert_eq!(read(&mut fresh).await.sequence_number, 0);
    lab.publish(1, &batch2).await;
    assert_eq!(read(&mut fresh).await.sequence_number, 1);
}

fn cleared_payload() -> Vec<u8> {
    let batch = serde_json::json!([1700000002.0, [{"type": "AllBlocksCleared"}], 0]);
    rmp_serde::to_vec_named(&batch).expect("encodes")
}

#[test]
fn restart_rules_read_the_same_on_every_wire() {
    let mut record = RankRecord {
        history: History::new(10, usize::MAX),
        cursor: Some(500),
        started_at: Some(0),
        unknown_before: 0,
        cleared: false,
    };
    let restart = |reason| Admission::Restart { reason, last: 500 };
    assert_eq!(record.admit(501, false), Admission::Accept);
    assert_eq!(
        record.admit(501, true),
        Admission::Accept,
        "a flush continues the sequence"
    );
    assert_eq!(
        record.admit(503, false),
        Admission::Gap { from: 501, to: 502 }
    );
    assert_eq!(record.admit(500, false), Admission::Duplicate);
    assert_eq!(
        record.admit(500, true),
        restart(RestartReason::StartupClear)
    );
    assert_eq!(
        record.admit(499, false),
        restart(RestartReason::SequenceRegression)
    );
    assert_eq!(
        record.admit(0, false),
        restart(RestartReason::CounterRestarted)
    );
    assert_eq!(
        record.admit(1, true),
        restart(RestartReason::CounterRestarted)
    );
    // Under a cursor of 1, a repeated 1 is a duplicate unless it clears.
    record.cursor = Some(1);
    assert_eq!(record.admit(1, false), Admission::Duplicate);
    assert_eq!(
        record.admit(1, true),
        Admission::Restart {
            reason: RestartReason::StartupClear,
            last: 1
        }
    );
    assert_eq!(
        record.admit(0, false),
        Admission::Restart {
            reason: RestartReason::SequenceRegression,
            last: 1
        }
    );
    record.cursor = None;
    assert_eq!(
        record.admit(7, true),
        Admission::Accept,
        "no cursor yet: anything goes"
    );
}

/// SGLang's first batch after a start carries `AllBlocksCleared`; under a
/// sequence the relay already passed it is a restart, not a duplicate.
#[tokio::test]
async fn the_engines_startup_clear_under_a_passed_cursor_is_a_restart() {
    let mut lab = start_lab(100, false).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    for sequence in 1..=3 {
        lab.publish(sequence, &batch2).await;
    }
    lab.wait_relayed(4).await;
    let mut stream = lab.subscribe(3).expect("caught up");
    // A repeated sequence without a clear is a duplicate.
    lab.publish(3, &batch2).await;
    lab.publish(4, &batch2).await;
    assert_eq!(read(&mut stream).await.sequence_number, 4);

    // The engine comes back and its startup clear lands on a sequence
    // the relay already passed.
    lab.publish(2, &cleared_payload()).await;
    let status = read_error(&mut stream).await;
    assert_eq!(status.code(), tonic::Code::DataLoss);
    assert_eq!(lab.relay.counts().publisher_restarts, 1);
    // The new incarnation began at 2 with the clear: nothing is live, and
    // a fresh subscriber is told so by a snapshot of one clear, cut at 2.
    let mut fresh = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!((chunk.sequence_number, chunk.events.len()), (2, 1));
    assert!(is_cleared(&chunk.events[0]));
    lab.publish(3, &batch2).await;
    assert_eq!(read(&mut fresh).await.sequence_number, 3);
}

/// A counter back at 0 or 1 after a cursor above them is a restart even
/// when nothing else says so (the mock engine's restart-publisher hook,
/// a vLLM process restart: no clear on that wire).
#[tokio::test]
async fn a_counter_back_at_its_start_is_a_restart() {
    let mut lab = start_lab(100, false).await;
    lab.prime().await;
    let batch2 = golden::bytes(golden::BATCH2);
    for sequence in 1..=5 {
        lab.publish(sequence, &batch2).await;
    }
    lab.wait_relayed(6).await;
    let mut stream = lab.subscribe(5).expect("caught up");
    lab.publish(1, &batch2).await;
    let status = read_error(&mut stream).await;
    assert_eq!(status.code(), tonic::Code::DataLoss);
    read_end(&mut stream).await;
    // The new incarnation started at 1, not 0: its window is not complete
    // from the publisher's first batch, so a fresh subscriber gets the
    // new incarnation's state as a snapshot (the restart emptied the old
    // one), then live.
    let mut fresh = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!(chunk.sequence_number, 1);
    assert_eq!(stored_hashes(&chunk), vec![(Some(41), vec![42])]);
    lab.publish(2, &batch2).await;
    assert_eq!(read(&mut fresh).await.sequence_number, 2);
    let mut resumed = lab.subscribe(1).expect("inside the new window");
    assert_eq!(read(&mut resumed).await.sequence_number, 2);
}

/// A publisher already counting when the relay's subscription reaches
/// it: the relay asks the engine's replay for everything from 0 before
/// relaying what it saw, and the window is the publisher's whole life.
#[tokio::test]
async fn a_publisher_already_counting_when_the_relay_joins_is_replayed_from_its_start() {
    let mut lab = start_lab(100, true).await;
    let batch2 = golden::bytes(golden::BATCH2);
    // Sequences 0..=2 went out before the subscription landed; 3 is the
    // first the relay sees, and it must ask for 0 before relaying it.
    let request = lab.publish_until_replay_requested(3, &batch2, 0).await;
    lab.send_replay(
        &request,
        &[
            (0, batch2.clone()),
            (1, batch2.clone()),
            (2, batch2.clone()),
            (3, batch2.clone()),
        ],
    )
    .await;
    lab.wait_relayed(4).await;
    let counts = lab.relay.counts();
    assert_eq!(
        (
            counts.gap_batches_recovered,
            counts.gap_batches_lost,
            counts.unknown_before_start,
            counts.publisher_gaps,
        ),
        (4, 0, 0, 0),
        "{counts:?}"
    );
    let mut stream = lab.subscribe(0).expect("the whole history");
    for expected in 0..=3 {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }
    lab.publish(4, &batch2).await;
    assert_eq!(read(&mut stream).await.sequence_number, 4);
    let counts = lab.relay.counts();
    assert_eq!(
        (counts.served_from_history, counts.served_snapshots),
        (1, 0)
    );
}

/// The replay that answers a late join may itself start past 0 (the
/// engine's buffer rolled): what it gives is relayed, the sequences
/// before it are holes and counted as unknown, a subscriber from zero
/// gets a snapshot that says so, and cursors inside the holes are served
/// past them.
#[tokio::test]
async fn a_replay_that_starts_past_zero_leaves_the_earlier_sequences_unknown() {
    let mut lab = start_lab(100, true).await;
    let batch2 = golden::bytes(golden::BATCH2);
    let request = lab.publish_until_replay_requested(7, &batch2, 0).await;
    lab.send_replay(
        &request,
        &[
            (5, batch2.clone()),
            (6, batch2.clone()),
            (7, batch2.clone()),
        ],
    )
    .await;
    lab.wait_relayed(3).await;
    let counts = lab.relay.counts();
    assert_eq!(
        (
            counts.gap_batches_recovered,
            counts.gap_batches_lost,
            counts.unknown_before_start,
        ),
        (3, 5, 5),
        "{counts:?}"
    );
    let mut fresh = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!(chunk.sequence_number, 7);
    assert_eq!(
        chunk.snapshot,
        Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: 3,
            unknown_before: 5,
        })
    );
    // A cursor inside the unknown stretch resumes at the first batch
    // after it; the gateway settles the jump as a gap of its own.
    let mut resumed = lab.subscribe(2).expect("served past the holes");
    for expected in [5, 6, 7] {
        assert_eq!(read(&mut resumed).await.sequence_number, expected);
    }
    lab.publish(8, &batch2).await;
    assert_eq!(read(&mut fresh).await.sequence_number, 8);
    assert_eq!(read(&mut resumed).await.sequence_number, 8);
}

/// Without a replay socket a late join cannot be filled: the sequences
/// before the first seen are holes, counted as unknown, and the snapshot
/// a fresh subscriber gets carries the count instead of passing as whole.
#[tokio::test]
async fn a_relay_without_a_replay_socket_marks_the_batches_before_its_start_unknown() {
    let mut late = start_lab(100, false).await;
    let batch2 = golden::bytes(golden::BATCH2);
    for _ in 0..250 {
        late.publish(5, &batch2).await;
        if late.relay.counts().relayed >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let counts = late.relay.counts();
    assert_eq!(
        (
            counts.relayed,
            counts.gap_batches_lost,
            counts.unknown_before_start,
            counts.publisher_gaps,
        ),
        (1, 5, 5, 0),
        "{counts:?}"
    );
    let mut fresh = late.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!((chunk.sequence_number, chunk.dp_rank), (5, Some(0)));
    assert_eq!(
        chunk.snapshot.as_ref().map(|chunk| chunk.unknown_before),
        Some(5)
    );
    assert_eq!(stored_hashes(&chunk), vec![(Some(41), vec![42])]);
    // Cursors in the unknown stretch are served from the first batch.
    let mut resumed = late.subscribe(3).expect("served past the holes");
    assert_eq!(read(&mut resumed).await.sequence_number, 5);
    late.publish(6, &batch2).await;
    assert_eq!(read(&mut resumed).await.sequence_number, 6);
    assert_eq!(read(&mut fresh).await.sequence_number, 6);
    assert_eq!(late.relay.counts().out_of_range, 0);
}

/// The batches before a late join matter until the engine clears its
/// cache: nothing from before a clear is live, so a snapshot taken after
/// one is whole and says so, while the lifetime counter keeps the late
/// start on record.
#[tokio::test]
async fn a_clear_after_a_late_join_ends_the_unknown_stretch() {
    let mut late = start_lab(100, false).await;
    let batch2 = golden::bytes(golden::BATCH2);
    for _ in 0..250 {
        late.publish(5, &batch2).await;
        if late.relay.counts().relayed >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(late.relay.counts().unknown_before_start, 5);
    // The engine resets its cache: BATCH1 ends with its clear.
    late.publish(6, &golden::bytes(golden::BATCH1)).await;
    late.wait_relayed(2).await;
    let mut fresh = late.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!((chunk.sequence_number, chunk.dp_rank), (6, Some(0)));
    assert_eq!(
        chunk.snapshot,
        Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: 0,
            unknown_before: 0,
        }),
        "nothing from before the join is live any more"
    );
    assert_eq!(chunk.events.len(), 1);
    assert!(is_cleared(&chunk.events[0]));
    assert_eq!(late.relay.counts().unknown_before_start, 5);
    late.publish(7, &batch2).await;
    assert_eq!(read(&mut fresh).await.sequence_number, 7);
}

/// The same when the clear comes back from the engine's replay: a replay
/// that starts past 0 leaves the earlier sequences unknown, but a clear
/// among the batches it gave makes them irrelevant to a snapshot.
#[tokio::test]
async fn a_clear_among_the_replayed_batches_ends_the_unknown_stretch_too() {
    let mut lab = start_lab(100, true).await;
    let batch2 = golden::bytes(golden::BATCH2);
    let request = lab.publish_until_replay_requested(7, &batch2, 0).await;
    lab.send_replay(
        &request,
        &[
            (5, batch2.clone()),
            (6, golden::bytes(golden::BATCH1)),
            (7, batch2.clone()),
        ],
    )
    .await;
    lab.wait_relayed(3).await;
    let counts = lab.relay.counts();
    assert_eq!(
        (counts.gap_batches_lost, counts.unknown_before_start),
        (5, 5),
        "{counts:?}"
    );
    let mut fresh = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!(chunk.sequence_number, 7);
    assert_eq!(
        chunk.snapshot,
        Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: 1,
            unknown_before: 0,
        }),
        "sequence 6 cleared the cache: only 7's block is live and nothing before the \
         record matters"
    );
    assert_eq!(stored_hashes(&chunk), vec![(Some(41), vec![42])]);
}

/// A publisher that counts from 1 (the mock engine) or whose first batch
/// is its startup clear (SGLang) has nothing before it to ask for.
#[test]
fn a_first_batch_at_the_counters_start_or_with_a_clear_is_not_a_late_join() {
    assert!(!joined_late(0, false));
    assert!(!joined_late(1, false));
    assert!(joined_late(2, false));
    assert!(!joined_late(2, true));
    assert!(joined_late(500, false));
}

/// A publisher that published at registration and nothing since: the
/// relay never gets a live batch to notice it by, so it takes the
/// replay's buffer at start, before any live batch, and a subscriber
/// from zero gets those blocks.
#[tokio::test]
async fn the_relay_takes_the_publishers_replay_at_start_before_any_live_batch() {
    let mut lab = start_lab_without_replay_socket_yet().await;
    let (lab, replay_endpoint) = (&mut lab.0, lab.1);
    let batch2 = golden::bytes(golden::BATCH2);
    // The engine comes up a moment after the servicer: the first attempt
    // found no replay socket, the retry finds it.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut router = RouterSocket::new();
    router.bind(&replay_endpoint).await.expect("replay binds");
    lab.router = Some(router);
    // The mock counts from 1: sequences 1..=3 went out at registration.
    lab.answer_replay(
        0,
        &[
            (1, batch2.clone()),
            (2, batch2.clone()),
            (3, batch2.clone()),
        ],
    )
    .await;
    lab.wait_relayed(3).await;
    let counts = lab.relay.counts();
    assert_eq!(
        (
            counts.primed_batches,
            counts.unknown_before_start,
            counts.gap_batches_lost,
            counts.publisher_gaps,
        ),
        (3, 0, 0, 0),
        "{counts:?}"
    );
    // Nothing live ever came; the state is served from zero as a snapshot
    // cut at 3 (a window from 1 is not complete from 0).
    let mut fresh = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!(chunk.sequence_number, 3);
    assert_eq!(chunk.snapshot.as_ref().map(|chunk| chunk.blocks), Some(3));
    assert_eq!(
        chunk.snapshot.as_ref().map(|chunk| chunk.unknown_before),
        Some(0)
    );
    // Live continues from the publisher's next sequence.
    lab.publish(4, &batch2).await;
    assert_eq!(read(&mut fresh).await.sequence_number, 4);
    let mut resumed = lab.subscribe(2).expect("inside the window");
    for expected in [3, 4] {
        assert_eq!(read(&mut resumed).await.sequence_number, expected);
    }
}

/// The start replay answered empty, then the publisher spoke (the mock's
/// registration batches) without the subscription catching them, and
/// never again: the first subscriber from zero finds the relay empty,
/// which makes it ask the replay once more, and receives the batches.
#[tokio::test]
async fn a_first_subscriber_makes_an_empty_relay_ask_the_replay_again() {
    let mut lab = start_lab(100, true).await;
    let batch2 = golden::bytes(golden::BATCH2);
    let mut stream = lab.subscribe(0).expect("live, nothing held");
    lab.answer_replay(
        0,
        &[
            (1, batch2.clone()),
            (2, batch2.clone()),
            (3, batch2.clone()),
        ],
    )
    .await;
    for expected in [1, 2, 3] {
        assert_eq!(read(&mut stream).await.sequence_number, expected);
    }
    let counts = lab.relay.counts();
    assert_eq!(
        (
            counts.primed_batches,
            counts.unknown_before_start,
            counts.relayed
        ),
        (3, 0, 3),
        "{counts:?}"
    );
    lab.publish(4, &batch2).await;
    assert_eq!(read(&mut stream).await.sequence_number, 4);
    // The next subscriber from zero gets the state the relay now holds
    // (a snapshot: the window starts at 1), and asks nothing more.
    let mut next = lab.subscribe(0).expect("a snapshot");
    assert_eq!(read(&mut next).await.sequence_number, 4);
    assert_eq!(lab.relay.counts().served_snapshots, 1);
}

/// A replay whose buffer rolled before the relay started marks what it
/// no longer holds as unknown, as the late-join path does.
#[tokio::test]
async fn a_start_replay_that_begins_past_the_publishers_start_marks_the_rest_unknown() {
    let mut lab = start_lab_without_replay_socket_yet().await;
    let (lab, replay_endpoint) = (&mut lab.0, lab.1);
    let batch2 = golden::bytes(golden::BATCH2);
    let mut router = RouterSocket::new();
    router.bind(&replay_endpoint).await.expect("replay binds");
    lab.router = Some(router);
    lab.answer_replay(0, &[(6, batch2.clone()), (7, batch2.clone())])
        .await;
    lab.wait_relayed(2).await;
    let counts = lab.relay.counts();
    assert_eq!(
        (
            counts.primed_batches,
            counts.unknown_before_start,
            counts.gap_batches_lost,
        ),
        (2, 6, 6),
        "{counts:?}"
    );
    let mut fresh = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut fresh).await;
    assert_eq!(chunk.sequence_number, 7);
    assert_eq!(
        chunk.snapshot.as_ref().map(|chunk| chunk.unknown_before),
        Some(6)
    );
}

#[tokio::test]
async fn dropping_the_relay_closes_the_publisher_subscription() {
    let mut lab = start_lab(10, false).await;
    let mut monitor = lab.publisher.monitor();
    lab.prime().await;
    let mut stream = lab.subscribe(0).expect("live");
    assert_eq!(read(&mut stream).await.sequence_number, 0);
    drop(lab.relay);
    read_end(&mut stream).await;
    let disconnected = timeout(Duration::from_secs(5), async {
        while let Some(event) = monitor.next().await {
            if matches!(event, SocketEvent::Disconnected(_)) {
                return true;
            }
        }
        false
    })
    .await
    .expect("the publisher notices in time");
    assert!(disconnected);
}
/// One publisher per rank, as a data-parallel engine runs them, each on
/// its own port, with a started relay over all of them.
struct RankLab {
    publishers: Vec<PubSocket>,
    relay: Arc<KvEventRelay>,
}

async fn start_rank_lab(ranks: i32, history_batches: usize) -> RankLab {
    let mut publishers = Vec::new();
    let mut sources = Vec::new();
    for rank in 0..ranks {
        let mut publisher = PubSocket::new();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string();
        publishers.push(publisher);
        sources.push(RankSource {
            rank,
            endpoint,
            replay_endpoint: None,
        });
    }
    let relay = KvEventRelay::new(RelayConfig {
        sources,
        topic: "kv".to_string(),
        history_batches,
        history_bytes: 64 << 20,
        replay_timeout: Duration::from_secs(2),
        load_tick: DEFAULT_LOAD_TICK,
        heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
    });
    relay.start();
    RankLab { publishers, relay }
}

impl RankLab {
    async fn publish(&mut self, rank: usize, sequence: u64, payload: &[u8]) {
        self.publishers[rank]
            .send(golden::frame(b"kv", sequence, payload))
            .await
            .expect("publish");
    }

    /// Each subscription reaches its publisher a moment after the
    /// connect: publish sequence 0 (a store of block 1) on every rank
    /// until the relay has them all.
    async fn prime(&mut self) {
        let payload = store_payload(0.0, &[1], None);
        let ranks = self.publishers.len();
        for _ in 0..250 {
            for rank in 0..ranks {
                self.publish(rank, 0, &payload).await;
            }
            if self.relay.counts().relayed >= ranks as u64 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("the relay never saw every rank's sequence 0");
    }

    async fn wait_relayed(&self, count: u64) {
        for _ in 0..250 {
            if self.relay.counts().relayed >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!(
            "the relay relayed {} batches, not {count}",
            self.relay.counts().relayed
        );
    }

    fn subscribe(&self, cursor: u64) -> Result<BoxStream<common::KvEventBatch>, Status> {
        self.relay.subscribe(common::SubscribeKvEventsRequest {
            start_sequence_number: cursor,
            load_records_before_first_batch: false,
        })
    }
}

/// `(rank, sequence)` of the next `count` batches of `stream`.
async fn tagged(
    stream: &mut BoxStream<common::KvEventBatch>,
    count: usize,
) -> Vec<(Option<i32>, u64)> {
    let mut seen = Vec::with_capacity(count);
    for _ in 0..count {
        let batch = read(stream).await;
        seen.push((batch.dp_rank, batch.sequence_number));
    }
    seen
}

/// A worker with two attention-DP ranks publishes two streams, each under
/// its own sequence counter: the relay subscribes to both endpoints,
/// keeps each rank's own numbers, stamps every batch with its socket's
/// rank (the payloads here all name rank 0) and records both ranks'
/// blocks; a subscriber from zero gets both windows, rank by rank, then
/// the live stream of each.
#[tokio::test]
async fn every_rank_of_a_data_parallel_publisher_is_relayed_under_its_own_sequence() {
    let mut lab = start_rank_lab(2, 100).await;
    lab.prime().await;
    lab.publish(0, 1, &store_payload(1.0, &[2], Some(1))).await;
    lab.publish(0, 2, &store_payload(2.0, &[3], Some(2))).await;
    lab.publish(1, 1, &store_payload(1.0, &[12], Some(1))).await;
    lab.wait_relayed(5).await;
    assert_eq!(lab.relay.live_ranks(), 2, "both ranks' blocks are recorded");

    let mut stream = lab.subscribe(0).expect("both windows");
    assert_eq!(
        tagged(&mut stream, 5).await,
        vec![
            (Some(0), 0),
            (Some(0), 1),
            (Some(0), 2),
            (Some(1), 0),
            (Some(1), 1)
        ]
    );
    lab.publish(1, 2, &store_payload(2.0, &[13], Some(12)))
        .await;
    lab.publish(0, 3, &store_payload(3.0, &[4], Some(3))).await;
    let mut live = tagged(&mut stream, 2).await;
    live.sort_unstable();
    assert_eq!(live, vec![(Some(0), 3), (Some(1), 2)]);
    assert_eq!(lab.relay.counts().served_from_history, 1);
}

/// The request carries one cursor and a relay over several ranks cannot
/// tell what the gateway holds of the others (nothing below it may be
/// resent on a fresh stream): the cursor is refused, and the
/// resubscription from zero the gateway answers with gets every window.
#[tokio::test]
async fn a_cursor_on_a_relay_over_several_ranks_is_refused_for_a_resubscription_from_zero() {
    let mut lab = start_rank_lab(2, 100).await;
    lab.prime().await;
    lab.publish(0, 1, &store_payload(1.0, &[2], Some(1))).await;
    lab.publish(0, 2, &store_payload(2.0, &[3], Some(2))).await;
    lab.publish(1, 1, &store_payload(1.0, &[12], Some(1))).await;
    lab.wait_relayed(5).await;

    let status = refused(lab.subscribe(1));
    assert_eq!(status.code(), tonic::Code::OutOfRange);
    assert!(
        status.message().contains("follows 2 ranks")
            && status.message().contains("resubscribe from zero"),
        "{}",
        status.message()
    );
    assert_eq!(lab.relay.counts().out_of_range, 1);
    let mut stream = lab.subscribe(0).expect("every window");
    assert_eq!(
        tagged(&mut stream, 5).await,
        vec![
            (Some(0), 0),
            (Some(0), 1),
            (Some(0), 2),
            (Some(1), 0),
            (Some(1), 1)
        ]
    );
}

/// Once a window rolled, a subscriber from zero gets the state snapshot:
/// one chunk per rank here, each stamped at its own rank's cursor, so
/// each rank's live stream continues right after its chunk.
#[tokio::test]
async fn a_snapshot_stamps_each_rank_at_its_own_cursor() {
    let mut lab = start_rank_lab(2, 2).await;
    lab.prime().await;
    for seq in 1..=4 {
        lab.publish(0, seq, &store_payload(seq as f64, &[seq as i64 * 10], None))
            .await;
    }
    lab.publish(1, 1, &store_payload(1.0, &[100], None)).await;
    lab.wait_relayed(7).await;

    let mut stream = lab.subscribe(0).expect("a snapshot");
    let first = read(&mut stream).await;
    let second = read(&mut stream).await;
    assert_eq!((first.dp_rank, first.sequence_number), (Some(0), 4));
    assert_eq!(
        first
            .snapshot
            .as_ref()
            .map(|chunk| (chunk.index, chunk.count)),
        Some((0, 2))
    );
    assert!(is_cleared(&first.events[0]));
    assert_eq!(
        stored_hashes(&first)
            .into_iter()
            .flat_map(|(_, hashes)| hashes)
            .collect::<Vec<_>>(),
        vec![1, 10, 20, 30, 40],
        "rank 0's live set, the rolled window included"
    );
    assert_eq!((second.dp_rank, second.sequence_number), (Some(1), 1));
    assert_eq!(
        second
            .snapshot
            .as_ref()
            .map(|chunk| (chunk.index, chunk.count)),
        Some((1, 2))
    );
    assert_eq!(
        stored_hashes(&second),
        vec![(None, vec![1]), (None, vec![100])]
    );

    lab.publish(0, 5, &store_payload(5.0, &[50], None)).await;
    lab.publish(1, 2, &store_payload(2.0, &[101], None)).await;
    let mut live = tagged(&mut stream, 2).await;
    live.sort_unstable();
    assert_eq!(live, vec![(Some(0), 5), (Some(1), 2)]);
    assert_eq!(lab.relay.counts().served_snapshots, 1);
}

/// One rank's publisher starting over ends live streams as any restart
/// does, but clears that rank's window and blocks only: a fresh
/// subscriber gets the other rank's window as it was and the restarted
/// rank's new incarnation from its start.
#[tokio::test]
async fn a_restart_on_one_rank_clears_that_ranks_window_only() {
    let mut lab = start_rank_lab(2, 100).await;
    lab.prime().await;
    lab.publish(0, 1, &store_payload(1.0, &[2], Some(1))).await;
    lab.publish(1, 1, &store_payload(1.0, &[12], Some(1))).await;
    lab.publish(1, 2, &store_payload(2.0, &[13], Some(12)))
        .await;
    lab.wait_relayed(5).await;
    let mut stream = lab.subscribe(0).expect("both windows");
    assert_eq!(tagged(&mut stream, 5).await.len(), 5);

    lab.publish(1, 0, &store_payload(9.0, &[99], None)).await;
    let status = read_error(&mut stream).await;
    assert_eq!(status.code(), tonic::Code::DataLoss);
    assert!(status.message().contains("restarted"));
    lab.wait_relayed(6).await;
    assert_eq!(lab.relay.counts().publisher_restarts, 1);

    let mut fresh = lab.subscribe(0).expect("both windows again");
    assert_eq!(
        tagged(&mut fresh, 3).await,
        vec![(Some(0), 0), (Some(0), 1), (Some(1), 0)]
    );
    assert_eq!(lab.relay.live_ranks(), 2);
}

/// A publisher batch on rank 0 with one `BlockStored`: `hashes` chained
/// from `parent`, two tokens per block, stamped `ts`.
fn store_payload(ts: f64, hashes: &[i64], parent: Option<i64>) -> Vec<u8> {
    let tokens: Vec<u32> = (0..hashes.len() as u32 * 2).collect();
    let batch = serde_json::json!([ts, [{
        "type": "BlockStored",
        "block_hashes": hashes,
        "parent_block_hash": parent,
        "token_ids": tokens,
        "block_size": 2,
        "lora_id": null,
        "medium": "GPU",
    }], 0]);
    rmp_serde::to_vec_named(&batch).expect("encodes")
}

fn remove_payload(hashes: &[i64]) -> Vec<u8> {
    let batch = serde_json::json!([1700000003.0, [{
        "type": "BlockRemoved",
        "block_hashes": hashes,
        "medium": "GPU",
    }], 0]);
    rmp_serde::to_vec_named(&batch).expect("encodes")
}

fn is_cleared(event: &common::KvCacheEvent) -> bool {
    matches!(event.data, Some(kv_cache_event::Data::Cleared(_)))
}

/// `(parent, hashes)` of every stored event of a batch, in order.
fn stored_hashes(batch: &common::KvEventBatch) -> Vec<(Option<i64>, Vec<i64>)> {
    batch
        .events
        .iter()
        .filter_map(|event| match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => Some((
                stored.parent_block_hash,
                stored.blocks.iter().map(|block| block.block_hash).collect(),
            )),
            _ => None,
        })
        .collect()
}

/// Once the window has rolled, a subscriber without a cursor gets the
/// live set as a snapshot cut at the newest sequence, then live events
/// from the next one; a batch published between the subscription and
/// the first poll follows the snapshot, once.
#[tokio::test]
async fn after_the_window_rolled_a_subscriber_without_a_cursor_gets_a_snapshot_then_live() {
    let mut lab = start_lab(3, false).await;
    lab.prime().await; // sequence 0: BATCH1, whose clear leaves nothing live
    lab.publish(1, &store_payload(1.0, &[10, 11], None)).await;
    lab.publish(2, &store_payload(2.0, &[12], Some(11))).await;
    lab.publish(3, &store_payload(3.0, &[20], None)).await;
    lab.publish(4, &remove_payload(&[20])).await;
    lab.publish(5, &store_payload(5.0, &[21, 22], None)).await;
    lab.wait_relayed(6).await;
    let mut stream = lab.subscribe(0).expect("a snapshot");
    // Published before the first poll: must follow the snapshot.
    lab.publish(6, &store_payload(6.0, &[30], Some(22))).await;
    lab.wait_relayed(7).await;
    let chunk = read(&mut stream).await;
    assert_eq!(chunk.sequence_number, 5, "stamped at the cut");
    assert_eq!(
        chunk.snapshot,
        Some(common::KvSnapshotChunk {
            index: 0,
            count: 1,
            blocks: 5,
            unknown_before: 0,
        })
    );
    assert_eq!(chunk.dp_rank, Some(0));
    assert!(is_cleared(&chunk.events[0]));
    assert_eq!(
        stored_hashes(&chunk),
        vec![(None, vec![10, 11, 12]), (None, vec![21, 22])],
        "the live set as the engine stored it, chains merged"
    );
    let live = read(&mut stream).await;
    assert_eq!((live.sequence_number, live.snapshot), (6, None));
    assert_eq!(stored_hashes(&live), vec![(Some(22), vec![30])]);
    lab.publish(7, &store_payload(7.0, &[31], Some(30))).await;
    assert_eq!(read(&mut stream).await.sequence_number, 7);
    let counts = lab.relay.counts();
    assert_eq!(
        (counts.served_snapshots, counts.served_from_history),
        (1, 0)
    );
}

/// A cursor below the window is still refused, and the resubscription
/// from zero the gateway answers with gets the snapshot.
#[tokio::test]
async fn a_stale_cursor_below_the_window_is_refused_and_zero_gets_the_snapshot() {
    let mut lab = start_lab(2, false).await;
    lab.prime().await;
    for seq in 1..=4 {
        lab.publish(seq, &store_payload(seq as f64, &[seq as i64 * 10], None))
            .await;
    }
    lab.wait_relayed(5).await;
    // The window is 3..=4; cursor 1 wants 2, gone.
    let status = refused(lab.subscribe(1));
    assert_eq!(status.code(), tonic::Code::OutOfRange);
    assert!(
        status
            .message()
            .contains("resubscribe from zero for a state snapshot"),
        "{}",
        status.message()
    );
    let mut stream = lab.subscribe(0).expect("a snapshot");
    let chunk = read(&mut stream).await;
    assert_eq!(chunk.sequence_number, 4);
    assert_eq!(chunk.snapshot.as_ref().map(|chunk| chunk.blocks), Some(4));
    assert_eq!(
        stored_hashes(&chunk),
        vec![
            (None, vec![10]),
            (None, vec![20]),
            (None, vec![30]),
            (None, vec![40])
        ]
    );
    lab.publish(5, &store_payload(5.0, &[50], None)).await;
    assert_eq!(read(&mut stream).await.sequence_number, 5);
    let counts = lab.relay.counts();
    assert_eq!((counts.out_of_range, counts.served_snapshots), (1, 1));
}

/// A relayed batch of `blocks` chained device blocks of 16 tokens on rank 0.
fn synthetic_batch(seq: u64, blocks: i64) -> common::KvEventBatch {
    let first = seq as i64 * blocks + 1;
    common::KvEventBatch {
        sequence_number: seq,
        timestamp: 1.0,
        events: vec![common::KvCacheEvent {
            event_id: seq,
            data: Some(kv_cache_event::Data::Stored(common::KvBlocksStored {
                blocks: (first..first + blocks)
                    .map(|hash| common::KvBlock {
                        block_hash: hash,
                        token_ids: (0..16).map(|i| hash as u32 ^ i).collect(),
                        block_size: 16,
                        ..Default::default()
                    })
                    .collect(),
                parent_block_hash: None,
                tier: Some(KvCacheTier::Device as i32),
                medium: Some("GPU".to_string()),
                ..Default::default()
            })),
        }],
        dp_rank: Some(0),
        snapshot: None,
        load: None,
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("after the epoch")
        .as_secs_f64()
}

/// Publish `sequences` one at a time, each stamped with its publish time
/// and read back on `live` before the next: the relay's latency per
/// batch in milliseconds.
async fn publish_timed(
    lab: &mut Lab,
    sequences: std::ops::Range<u64>,
    live: &mut BoxStream<common::KvEventBatch>,
) -> Vec<f64> {
    let mut latencies = Vec::with_capacity((sequences.end - sequences.start) as usize);
    for seq in sequences {
        lab.publish(seq, &store_payload(unix_now(), &[-(seq as i64)], None))
            .await;
        let batch = read(live).await;
        assert_eq!(batch.sequence_number, seq);
        latencies.push((unix_now() - batch.timestamp) * 1e3);
    }
    latencies
}

/// A large worker's pool (676k blocks) behind a rolled window: the
/// snapshot is cut atomically at the relay's cursor while live batches
/// stream through to another subscriber, which sees no stall beyond the
/// one pass under the lock; the snapshot subscriber, not read until the
/// cut is long past, gets every chunk and then the live stream from the
/// cut, nothing twice and nothing missing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::print_stderr,
    reason = "the measured pause is this test's report; read it with --nocapture"
)]
async fn a_676k_block_snapshot_is_cut_atomically_and_does_not_stall_the_live_stream() {
    const BATCHES: u64 = 10_564;
    const PER_BATCH: i64 = 64;
    let mut lab = start_lab(50, false).await;
    lab.relay
        .preload((0..BATCHES).map(|seq| synthetic_batch(seq, PER_BATCH)));
    let preloaded = BATCHES * PER_BATCH as u64;
    let last = BATCHES - 1;
    let mut live = lab.subscribe(last).expect("at the newest sequence");
    // The SUB connects asynchronously: publish the next sequence until it lands.
    let next = last + 1;
    for _ in 0..250 {
        lab.publish(next, &store_payload(unix_now(), &[-1], None))
            .await;
        if lab.relay.counts().relayed > BATCHES {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(read(&mut live).await.sequence_number, next);
    let control = publish_timed(&mut lab, next + 1..next + 201, &mut live).await;

    // The snapshot is taken while live batches keep flowing to `live`.
    let relay = Arc::clone(&lab.relay);
    let taking = tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        let stream = relay.subscribe(common::SubscribeKvEventsRequest {
            start_sequence_number: 0,
            load_records_before_first_batch: false,
        });
        (stream, started.elapsed())
    });
    let during = publish_timed(&mut lab, next + 201..next + 1_201, &mut live).await;
    let (snapshot, subscribe_took) = taking.await.expect("the subscribe call ran");
    let mut snapshot = snapshot.expect("a snapshot");
    let max = |values: &[f64]| values.iter().copied().fold(0.0f64, f64::max);
    let median = |values: &[f64]| {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    eprintln!(
        "676k snapshot: subscribe call (lock held for the collection) {subscribe_took:?}; \
         live latency ms: control median {:.3} max {:.3}, during median {:.3} max {:.3}",
        median(&control),
        max(&control),
        median(&during),
        max(&during)
    );
    assert!(
        subscribe_took < Duration::from_secs(2),
        "{subscribe_took:?}"
    );
    assert!(
        max(&during) < 2_000.0,
        "a live batch waited {:.1} ms on the snapshot",
        max(&during)
    );

    // Not read until now: the chunks come whole, then live from the cut.
    let mut chunks = Vec::new();
    let first_live = loop {
        let batch = read(&mut snapshot).await;
        if batch.snapshot.is_none() {
            break batch;
        }
        chunks.push(batch);
    };
    let through = chunks.last().expect("chunks").sequence_number;
    assert!(
        (next..next + 1_201).contains(&through),
        "the cut fell inside the timed publishes: {through}"
    );
    let blocks_at_cut = preloaded + (through - last);
    let count = u32::try_from(chunks.len()).unwrap();
    assert_eq!(
        count as usize,
        (blocks_at_cut as usize).div_ceil(crate::kv_state::CHUNK_BLOCKS)
    );
    for (index, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.sequence_number,
            through + 1 - u64::from(count) + index as u64,
            "stamps are consecutive up to the cut"
        );
        let marker = chunk.snapshot.as_ref().unwrap();
        assert_eq!((marker.index, marker.count), (index as u32, count));
        // Every live batch up to the cut stored one block: the state and
        // the cut agree, so the collection and the cursor were one guard.
        assert_eq!(marker.blocks, blocks_at_cut);
    }
    assert!(is_cleared(&chunks[0].events[0]));
    let emitted: usize = chunks
        .iter()
        .map(|chunk| {
            stored_hashes(chunk)
                .iter()
                .map(|(_, hashes)| hashes.len())
                .sum::<usize>()
        })
        .sum();
    assert_eq!(emitted as u64, blocks_at_cut);
    assert_eq!(
        first_live.sequence_number,
        through + 1,
        "live continues right after the cut"
    );
    let mut expected = through + 2;
    while expected < next + 1_201 {
        assert_eq!(read(&mut snapshot).await.sequence_number, expected);
        expected += 1;
    }
}

/// The normalizer's counters ride on the relay: readable after every
/// relayed batch and logged with the relay's own counts, so a live run
/// shows what was forwarded, dropped by reason and, with the engine-hash
/// check on, verified.
#[tokio::test]
async fn the_relays_wire_counts_follow_the_normalizer() {
    let mut lab = start_lab(8, false).await;
    assert_eq!(lab.relay.wire_counts(), WireCounts::default());
    lab.prime().await;
    lab.publish(1, &golden::bytes(golden::BATCH2)).await;
    lab.wait_relayed(2).await;
    let wire = lab.relay.wire_counts();
    assert!(
        wire.forwarded_stored + wire.forwarded_removed + wire.forwarded_cleared > 0,
        "{wire:?}"
    );
    assert_eq!(
        wire.hash_checked, 0,
        "the check is off unless the environment asks"
    );
    assert_eq!(wire.window_only_stores, 0);
}

/// Before the stream has sent a batch there is no sequence a heartbeat
/// could repeat, so a quiet publisher at the start gets none for a
/// subscriber that did not ask for records before the first batch: a
/// gateway that predates the load field would take the publisher's
/// first batch for a duplicate of a heartbeat numbered 0. The first real
/// batch opens the heartbeats.
#[tokio::test]
async fn no_heartbeat_goes_out_before_the_streams_first_batch() {
    let loads = StubLoads::new(2, 0);
    let mut lab = start_lab_with_loads_at(
        Arc::clone(&loads),
        Duration::from_millis(20),
        Duration::from_millis(60),
        Duration::from_millis(100),
        false,
    )
    .await;
    let mut stream = lab.subscribe(0).expect("live from the start");
    loads.set(5, 1);
    assert!(
        timeout(Duration::from_millis(300), stream.next())
            .await
            .is_err(),
        "no batch may precede the publisher's first"
    );
    lab.prime().await;
    let first = read(&mut stream).await;
    assert_eq!(first.sequence_number, 0);
    assert!(!load_only_marker(&first));
    let beat = read(&mut stream).await;
    assert!(load_only_marker(&beat));
    assert_eq!(beat.sequence_number, 0);
}

/// A subscriber that asks for records before the publisher's first batch
/// (every gateway that knows `load_only`) hears from the relay from the
/// subscription on: a `load_only` batch numbered 0 within a tick, one
/// for each change of the record after it (a frozen engine's growing
/// silence is such a change) and the heartbeats between; the publisher's
/// first batch is then relayed as itself, not skipped as a repeat of
/// their number, and the heartbeats repeat its sequence from there.
#[tokio::test]
async fn a_subscriber_that_asks_gets_records_before_the_publishers_first_batch() {
    let loads = StubLoads::new(0, 0);
    let mut lab = start_lab_with_loads_at(
        Arc::clone(&loads),
        Duration::from_millis(20),
        Duration::from_millis(60),
        Duration::from_millis(100),
        false,
    )
    .await;
    let mut stream = lab
        .subscribe_from_the_start(0)
        .expect("live from the start");
    let first = timeout(Duration::from_millis(300), stream.next())
        .await
        .expect("a record before the publisher's first batch")
        .expect("an item")
        .expect("a batch");
    assert!(load_only_marker(&first), "{first:?}");
    assert_eq!((first.sequence_number, first.events.len()), (0, 0));
    assert_eq!(record_of(&first).running_requests, 0);
    // The record moves: a batch within a tick or two, numbered 0 still
    // (heartbeats of the unchanged record may come first).
    loads.set(1, 0);
    let started = Instant::now();
    loop {
        let batch = read(&mut stream).await;
        assert!(load_only_marker(&batch), "{batch:?}");
        assert_eq!(batch.sequence_number, 0);
        if record_of(&batch).running_requests == 1 {
            break;
        }
    }
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "the change took {:?}",
        started.elapsed()
    );
    // The publisher's first batch is relayed as itself, and the
    // heartbeats repeat its sequence.
    lab.prime().await;
    let live = loop {
        let batch = read(&mut stream).await;
        if !load_only_marker(&batch) {
            break batch;
        }
    };
    assert_eq!(live.sequence_number, 0);
    assert!(!live.events.is_empty(), "{live:?}");
    let beat = read(&mut stream).await;
    assert!(load_only_marker(&beat));
    assert_eq!(beat.sequence_number, 0);
}
