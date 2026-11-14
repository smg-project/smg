//! The driver against the memory store and a probe in memory, under paused
//! time: the arbitration end to end, the per-rank writes, the failures, the
//! scrape and its interval, the rows it never dials, the engine that does
//! not answer and the worker that leaves with a call out.

use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use openai_protocol::worker::{RuntimeType, SchedulerLoadSnapshot, WorkerLoadResponse};
use parking_lot::Mutex;
use tokio::{
    sync::{watch, Notify},
    task::JoinSet,
};

use super::{
    now, peer, Failures, LoadDriver, LoadDriverConfig, LoadProbe, ReadError, Schedule, Target,
};
use crate::{
    store::{
        GatewayId, LoadSink, LoadSource, MemoryStore, Reachability, Signals, WorkerFacts,
        WorkerStatus, WorkersStore,
    },
    view::{LoadRow, Role, SignalSource, WireModes, WorkerId},
};

/// A metrics body of the first gauge family: three running, five waiting,
/// three quarters of the KV cache in use.
const METRICS_TEXT: &str = "vllm:num_requests_running{model_name=\"m\"} 3\n\
                           vllm:num_requests_waiting{model_name=\"m\"} 5\n\
                           vllm:kv_cache_usage_perc{model_name=\"m\"} 0.75\n";

/// One rank's figures the fake engine answers with.
#[derive(Clone, Copy)]
struct Figures {
    running: i32,
    waiting: i32,
    usage: f64,
}

#[derive(Default)]
struct FakeProbe {
    /// The endpoints polled, in order.
    polls: Mutex<Vec<String>>,
    /// The URLs scraped, in order.
    scrapes: Mutex<Vec<String>>,
    /// The ranks an endpoint answers with, by rank index.
    answers: Mutex<HashMap<String, Vec<Figures>>>,
    /// Endpoints whose poll fails on the transport.
    failing: Mutex<HashSet<String>>,
    /// The metrics text a URL serves.
    texts: Mutex<HashMap<String, String>>,
    forgotten: Mutex<Vec<String>>,
    /// The endpoints a connection is kept for, as the probe keeps them: a
    /// poll keeps one from the moment its dial completes, `forget` drops it.
    connected: Mutex<HashSet<String>>,
    /// Endpoints whose poll or scrape does not answer until the test releases
    /// it: an engine that takes its time to be dialled, or never answers.
    held: Mutex<HashSet<String>>,
    release: Notify,
}

impl FakeProbe {
    /// How many connections the probe keeps.
    fn connections(&self) -> usize {
        self.connected.lock().len()
    }

    fn answers(&self, endpoint: &str, ranks: &[Figures]) {
        self.answers
            .lock()
            .insert(endpoint.to_owned(), ranks.to_vec());
    }

    fn polls(&self) -> Vec<String> {
        self.polls.lock().clone()
    }

    /// How many polls went to `endpoint`.
    fn polls_of(&self, endpoint: &str) -> usize {
        self.polls
            .lock()
            .iter()
            .filter(|polled| *polled == endpoint)
            .count()
    }
}

impl LoadProbe for FakeProbe {
    async fn poll(
        &self,
        target: &Target,
        _deadline: Duration,
    ) -> Result<WorkerLoadResponse, ReadError> {
        self.polls.lock().push(target.endpoint.clone());
        if self.held.lock().contains(&target.endpoint) {
            self.release.notified().await;
        }
        if self.failing.lock().contains(&target.endpoint) {
            return Err(ReadError::Transport("connection refused".to_owned()));
        }
        self.connected.lock().insert(target.endpoint.clone());
        let ranks = self
            .answers
            .lock()
            .get(&target.endpoint)
            .cloned()
            .unwrap_or_default();
        let loads = ranks
            .iter()
            .enumerate()
            .map(|(rank, figures)| SchedulerLoadSnapshot {
                dp_rank: i32::try_from(rank).unwrap_or(i32::MAX),
                num_running_reqs: figures.running,
                num_waiting_reqs: figures.waiting,
                token_usage: figures.usage,
                max_total_num_tokens: 8_192,
                num_used_tokens: 1_024,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        Ok(WorkerLoadResponse {
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
            ..Default::default()
        })
    }

    async fn scrape(&self, url: &str, _timeout: Duration) -> Result<String, ReadError> {
        self.scrapes.lock().push(url.to_owned());
        if self.held.lock().contains(url) {
            self.release.notified().await;
        }
        self.texts
            .lock()
            .get(url)
            .cloned()
            .ok_or_else(|| ReadError::Transport("connection refused".to_owned()))
    }

    fn forget(&self, endpoint: &str) {
        self.forgotten.lock().push(endpoint.to_owned());
        self.connected.lock().remove(endpoint);
    }
}

/// A ready row: one rank of `engine` at `key`.
fn ready(
    store: &MemoryStore,
    key: &str,
    engine: &str,
    rank: u16,
    size: u16,
    wires: WireModes,
    scrape_ms: u32,
) -> WorkerId {
    let id = store
        .upsert(WorkerFacts {
            key: key.to_owned(),
            engine: engine.to_owned(),
            incarnation: 1,
            role: Role::Single,
            wire_modes: wires,
            engine_kind: RuntimeType::Vllm.as_str().to_owned(),
            models: vec!["m".to_owned()],
            dp_rank: rank,
            dp_size: size,
            signals: Signals {
                events: false,
                loads: false,
                scrape_interval_ms: scrape_ms,
            },
            ..WorkerFacts::default()
        })
        .id();
    let revision = store.get(id).expect("the row").revision;
    store.set_status(id, WorkerStatus::Ready, revision);
    id
}

fn driver(
    store: &Arc<MemoryStore>,
    probe: &Arc<FakeProbe>,
    poll_interval: Duration,
) -> LoadDriver<MemoryStore, FakeProbe> {
    LoadDriver::new(
        Arc::clone(store),
        Arc::clone(probe),
        LoadDriverConfig {
            poll_interval,
            tick: Duration::from_millis(100),
            ..LoadDriverConfig::default()
        },
    )
}

fn row(store: &MemoryStore, id: WorkerId, rank: u16) -> Option<(LoadRow, LoadSource)> {
    store
        .load(id, rank)
        .map(|sample| (sample.row, sample.source))
}

const TEN: Duration = Duration::from_secs(10);

/// One tick, then every call it started lands: the fake engines answer at
/// once, so the test reads the rows the tick asked for.
async fn tick(driver: &LoadDriver<MemoryStore, FakeProbe>, schedule: &mut Schedule) {
    driver.tick(schedule);
    while let Some(joined) = schedule.calls.join_next().await {
        driver.land(&mut schedule.due, joined);
    }
}

#[tokio::test(start_paused = true)]
async fn the_first_tick_polls_every_ready_engine_once_and_writes_a_row_per_rank() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let single = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    // A data-parallel engine's rows carry the registry's rank keys; the poll
    // dials the engine, never a rank key.
    let rank0 = ready(
        &store,
        "grpc://b:1@0",
        "grpc://b:1",
        0,
        2,
        WireModes::GRPC,
        0,
    );
    let rank1 = ready(
        &store,
        "grpc://b:1@1",
        "grpc://b:1",
        1,
        2,
        WireModes::GRPC,
        0,
    );
    let busy = Figures {
        running: 4,
        waiting: 2,
        usage: 0.5,
    };
    let idle = Figures {
        running: 0,
        waiting: 0,
        usage: 0.1,
    };
    probe.answers("grpc://a:1", &[busy]);
    probe.answers("grpc://b:1", &[idle, busy]);
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert_eq!(
        probe.polls(),
        vec!["grpc://a:1", "grpc://b:1"],
        "one call per engine, at the engine's address"
    );
    let (one, source) = row(&store, single, 0).expect("the single engine's row");
    assert_eq!(
        (one.running, one.queued, one.kv_usage_permille),
        (4, 2, 500)
    );
    assert_eq!((one.source, source), (SignalSource::Poll, LoadSource::Poll));
    let (zero, _) = row(&store, rank0, 0).expect("rank 0's row");
    assert_eq!((zero.running, zero.kv_usage_permille), (0, 100));
    let (one, _) = row(&store, rank1, 1).expect("rank 1's row");
    assert_eq!((one.running, one.kv_usage_permille), (4, 500));
    assert!(
        row(&store, rank0, 1).is_none(),
        "a rank writes its own row alone"
    );
    let stats = driver.stats();
    assert_eq!(stats.polls.load(Ordering::Relaxed), 2);
    assert_eq!(stats.rows.load(Ordering::Relaxed), 3);
}

#[tokio::test(start_paused = true)]
async fn a_rank_served_by_its_stream_is_not_polled_and_a_quiet_one_falls_back() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    probe.answers(
        "grpc://a:1",
        &[Figures {
            running: 9,
            waiting: 0,
            usage: 0.9,
        }],
    );
    let streamed = LoadRow {
        running: 1,
        source: SignalSource::Events,
        ..LoadRow::default()
    };
    store.record(id, 0, streamed, now(), LoadSource::Stream);
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;
    assert!(probe.polls().is_empty(), "a fresh record serves the rank");
    assert_eq!(row(&store, id, 0).map(|(r, _)| r.running), Some(1));

    tokio::time::advance(TEN + Duration::from_secs(1)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(
        probe.polls(),
        vec!["grpc://a:1"],
        "a quiet record falls back to the poll"
    );
    let (polled, source) = row(&store, id, 0).expect("the poll's row");
    assert_eq!((polled.running, source), (9, LoadSource::Poll));
    let stats = driver.stats();
    assert_eq!(stats.suppressed.load(Ordering::Relaxed), 1);
    assert_eq!(stats.fallbacks.load(Ordering::Relaxed), 1);
}

#[tokio::test(start_paused = true)]
async fn a_stream_that_stops_is_polled_within_one_tick_of_the_freshness_bound() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    probe.answers(
        "grpc://a:1",
        &[Figures {
            running: 3,
            waiting: 0,
            usage: 0.3,
        }],
    );
    let record = LoadRow {
        running: 1,
        ..LoadRow::default()
    };
    store.record(id, 0, record, now(), LoadSource::Stream);
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    // t = 0: the record is fresh, the engine is skipped, not postponed.
    tick(&driver, &mut schedule).await;
    assert!(probe.polls().is_empty());
    // t = 1: the stream's last record.
    tokio::time::advance(Duration::from_secs(1)).await;
    store.record(id, 0, record, now(), LoadSource::Stream);
    tick(&driver, &mut schedule).await;
    // t = 10: nine seconds old, still within the bound.
    tokio::time::advance(Duration::from_secs(9)).await;
    tick(&driver, &mut schedule).await;
    assert!(
        probe.polls().is_empty(),
        "a record nine seconds old still serves its rank"
    );
    // t = 11: the bound passed one second ago; the fallback goes now, not at
    // the next interval.
    tokio::time::advance(Duration::from_secs(1)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(probe.polls(), vec!["grpc://a:1"]);
    let stats = driver.stats();
    assert_eq!(
        stats.suppressed.load(Ordering::Relaxed),
        2,
        "three ticks asked and skipped, counted once per interval (t = 0 and t = 10)"
    );
    assert_eq!(stats.fallbacks.load(Ordering::Relaxed), 1);
    // The call moved the due instant: the next tick asks nothing.
    tokio::time::advance(Duration::from_secs(1)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(probe.polls().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_dp_engine_polls_once_for_its_quiet_ranks_and_leaves_the_fresh_rank_alone() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let rank0 = ready(
        &store,
        "grpc://b:1@0",
        "grpc://b:1",
        0,
        2,
        WireModes::GRPC,
        0,
    );
    let rank1 = ready(
        &store,
        "grpc://b:1@1",
        "grpc://b:1",
        1,
        2,
        WireModes::GRPC,
        0,
    );
    let figures = Figures {
        running: 7,
        waiting: 1,
        usage: 0.3,
    };
    probe.answers("grpc://b:1", &[figures, figures]);
    store.record(
        rank0,
        0,
        LoadRow {
            running: 2,
            ..LoadRow::default()
        },
        now(),
        LoadSource::Stream,
    );
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert_eq!(probe.polls(), vec!["grpc://b:1"]);
    assert_eq!(
        row(&store, rank0, 0),
        Some((
            LoadRow {
                running: 2,
                ..LoadRow::default()
            },
            LoadSource::Stream
        )),
        "the fresh rank keeps its stream's row"
    );
    let (one, source) = row(&store, rank1, 1).expect("the quiet rank's row");
    assert_eq!((one.running, source), (7, LoadSource::Poll));
}

#[tokio::test(start_paused = true)]
async fn a_rank_ready_alone_takes_its_own_figures_from_the_engines_answer() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    // Rank 0 is registered and pending; rank 1 alone is ready.
    let pending = store
        .upsert(WorkerFacts {
            key: "grpc://b:1@0".to_owned(),
            engine: "grpc://b:1".to_owned(),
            incarnation: 1,
            wire_modes: WireModes::GRPC,
            dp_rank: 0,
            dp_size: 2,
            ..WorkerFacts::default()
        })
        .id();
    let rank1 = ready(
        &store,
        "grpc://b:1@1",
        "grpc://b:1",
        1,
        2,
        WireModes::GRPC,
        0,
    );
    probe.answers(
        "grpc://b:1",
        &[
            Figures {
                running: 9,
                waiting: 9,
                usage: 0.9,
            },
            Figures {
                running: 2,
                waiting: 1,
                usage: 0.2,
            },
        ],
    );
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert_eq!(probe.polls(), vec!["grpc://b:1"]);
    let (one, _) = row(&store, rank1, 1).expect("rank 1's row");
    assert_eq!(
        (one.running, one.queued, one.kv_usage_permille),
        (2, 1, 200),
        "rank 1 takes its own figures, never the engine's sums"
    );
    assert!(
        row(&store, pending, 0).is_none(),
        "a pending rank is not written"
    );
}

#[tokio::test(start_paused = true)]
async fn a_failed_poll_leaves_the_row_on_file_and_counts() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    probe.answers(
        "grpc://a:1",
        &[Figures {
            running: 5,
            waiting: 0,
            usage: 0.5,
        }],
    );
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();
    tick(&driver, &mut schedule).await;
    assert_eq!(row(&store, id, 0).map(|(r, _)| r.running), Some(5));

    probe.failing.lock().insert("grpc://a:1".to_owned());
    tokio::time::advance(TEN + Duration::from_secs(1)).await;
    tick(&driver, &mut schedule).await;

    assert_eq!(probe.polls().len(), 2);
    assert_eq!(
        row(&store, id, 0).map(|(r, _)| r.running),
        Some(5),
        "the old row stands"
    );
    let stats = driver.stats();
    assert_eq!(stats.poll_failures.transport.load(Ordering::Relaxed), 1);
    assert_eq!(stats.rows.load(Ordering::Relaxed), 1);
}

#[tokio::test(start_paused = true)]
async fn rows_the_controller_never_dials_are_not_polled() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    // A host-local wire, relayed by its gateway.
    let relayed = store
        .upsert(WorkerFacts {
            key: "zmq://local".to_owned(),
            engine: "zmq://local".to_owned(),
            incarnation: 1,
            wire_modes: WireModes::ZMQ.union(WireModes::GRPC),
            dp_size: 1,
            reachable_from: Reachability::Gateway(GatewayId(7)),
            ..WorkerFacts::default()
        })
        .id();
    let revision = store.get(relayed).expect("the row").revision;
    store.set_status(relayed, WorkerStatus::Ready, revision);
    // An HTTP engine with no scrape interval: nothing to ask it.
    ready(&store, "http://h:1", "http://h:1", 0, 1, WireModes::HTTP, 0);
    // A row not yet ready.
    store.upsert(WorkerFacts {
        key: "grpc://pending:1".to_owned(),
        engine: "grpc://pending:1".to_owned(),
        incarnation: 1,
        wire_modes: WireModes::GRPC,
        dp_size: 1,
        ..WorkerFacts::default()
    });
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert!(probe.polls().is_empty());
    assert!(probe.scrapes.lock().is_empty());
    assert_eq!(driver.stats().rows.load(Ordering::Relaxed), 0);
}

#[tokio::test(start_paused = true)]
async fn the_poll_waits_the_interval_between_calls() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;
    tokio::time::advance(Duration::from_secs(5)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(probe.polls().len(), 1, "half an interval later: no call");
    tokio::time::advance(Duration::from_secs(5)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(probe.polls().len(), 2, "the interval passed: one call");
}

#[tokio::test(start_paused = true)]
async fn an_engine_without_a_servicer_is_scraped_at_its_own_interval() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(
        &store,
        "http://h:1",
        "http://h:1",
        0,
        1,
        WireModes::HTTP,
        500,
    );
    probe
        .texts
        .lock()
        .insert("http://h:1".to_owned(), METRICS_TEXT.to_owned());
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert_eq!(*probe.scrapes.lock(), vec!["http://h:1"]);
    assert!(probe.polls().is_empty(), "an HTTP engine has no GetLoads");
    let (scraped, source) = row(&store, id, 0).expect("the scrape's row");
    assert_eq!(
        (scraped.running, scraped.queued, scraped.kv_usage_permille),
        (3, 5, 750)
    );
    assert_eq!(
        (scraped.source, source),
        (SignalSource::Scrape, LoadSource::Scrape)
    );

    tokio::time::advance(Duration::from_millis(400)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(
        probe.scrapes.lock().len(),
        1,
        "before the interval: no call"
    );
    tokio::time::advance(Duration::from_millis(200)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(probe.scrapes.lock().len(), 2);
    assert_eq!(driver.stats().scrapes.load(Ordering::Relaxed), 2);
}

#[tokio::test(start_paused = true)]
async fn a_scrape_without_the_load_gauge_writes_no_row() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(
        &store,
        "http://h:1",
        "http://h:1",
        0,
        1,
        WireModes::HTTP,
        500,
    );
    probe.texts.lock().insert(
        "http://h:1".to_owned(),
        "vllm:num_requests_running{model_name=\"m\"} 3\n".to_owned(),
    );
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert!(row(&store, id, 0).is_none());
    let stats = driver.stats();
    assert_eq!(stats.scrape_failures.no_load.load(Ordering::Relaxed), 1);
    assert_eq!(stats.scrapes.load(Ordering::Relaxed), 0);
}

/// A KV-usage gauge that is not a number is no reading: the scrape writes no
/// row, where a row at zero usage would make an engine of unknown pressure
/// look idle, and it counts as a body without its load.
#[tokio::test(start_paused = true)]
async fn a_scrape_whose_kv_usage_is_not_a_number_writes_no_row() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(
        &store,
        "http://h:1",
        "http://h:1",
        0,
        1,
        WireModes::HTTP,
        500,
    );
    probe.texts.lock().insert(
        "http://h:1".to_owned(),
        "vllm:num_requests_running{model_name=\"m\"} 3\n\
         vllm:kv_cache_usage_perc{model_name=\"m\"} NaN\n"
            .to_owned(),
    );
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert!(row(&store, id, 0).is_none(), "no row at zero usage");
    let stats = driver.stats();
    assert_eq!(stats.scrape_failures.no_load.load(Ordering::Relaxed), 1);
    assert_eq!(stats.scrapes.load(Ordering::Relaxed), 0);
    assert_eq!(stats.rows.load(Ordering::Relaxed), 0);
}

#[tokio::test(start_paused = true)]
async fn a_two_rank_engine_without_a_servicer_has_no_row_after_a_scrape_tick_and_a_one_rank_engine_has_its_row(
) {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let rank0 = ready(
        &store,
        "http://h:1@0",
        "http://h:1",
        0,
        2,
        WireModes::HTTP,
        500,
    );
    let rank1 = ready(
        &store,
        "http://h:1@1",
        "http://h:1",
        1,
        2,
        WireModes::HTTP,
        500,
    );
    let single = ready(
        &store,
        "http://s:1",
        "http://s:1",
        0,
        1,
        WireModes::HTTP,
        500,
    );
    for url in ["http://h:1", "http://s:1"] {
        probe
            .texts
            .lock()
            .insert(url.to_owned(), METRICS_TEXT.to_owned());
    }
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    // The gauges are engine-wide: the two-rank engine is not asked and
    // neither rank reads a figure that is not its own.
    assert_eq!(*probe.scrapes.lock(), vec!["http://s:1"]);
    for id in [rank0, rank1] {
        assert_eq!(store.ranks(id), 0, "no aggregate on a rank");
    }
    let (sample, source) = row(&store, single, 0).expect("the one-rank engine's row");
    assert_eq!(
        (sample.running, sample.queued, sample.kv_usage_permille),
        (3, 5, 750)
    );
    assert_eq!(source, LoadSource::Scrape);
    assert_eq!(driver.stats().rows.load(Ordering::Relaxed), 1);

    // A rank's row is its own record, once one arrives: the data-parallel
    // rank reads its per-rank source alone, and the one-rank engine, served
    // by its own records now, is not scraped while they are fresh.
    let own = LoadRow {
        running: 1,
        source: SignalSource::Events,
        ..LoadRow::default()
    };
    store.record(rank0, 0, own, now(), LoadSource::Stream);
    store.record(single, 0, own, now(), LoadSource::Stream);
    tokio::time::advance(Duration::from_millis(500)).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(
        *probe.scrapes.lock(),
        vec!["http://s:1"],
        "its own records serve the one-rank engine: no scrape"
    );
    assert_eq!(row(&store, rank0, 0), Some((own, LoadSource::Stream)));
    assert_eq!(store.ranks(rank1), 0);
    assert_eq!(row(&store, single, 0), Some((own, LoadSource::Stream)));
    assert_eq!(
        driver.stats().scrapes.load(Ordering::Relaxed),
        1,
        "the one scrape that produced a row"
    );

    // The records stop: past the freshness bound the engine is scraped
    // again, at the first tick.
    tokio::time::advance(TEN).await;
    tick(&driver, &mut schedule).await;
    assert_eq!(
        *probe.scrapes.lock(),
        vec!["http://s:1", "http://s:1"],
        "its records went quiet: scraped again"
    );
}

/// A record that arrives while the scrape is out is the rank's row: the
/// scrape's sample is dropped, and it is not counted as a scrape that
/// produced a row.
#[tokio::test(start_paused = true)]
async fn a_scrape_that_lands_after_the_ranks_own_record_writes_no_row_and_is_not_counted() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(
        &store,
        "http://h:1",
        "http://h:1",
        0,
        1,
        WireModes::HTTP,
        500,
    );
    probe
        .texts
        .lock()
        .insert("http://h:1".to_owned(), METRICS_TEXT.to_owned());
    probe.held.lock().insert("http://h:1".to_owned());
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    driver.tick(&mut schedule);
    tokio::task::yield_now().await;
    assert_eq!(probe.scrapes.lock().len(), 1, "the scrape is out");
    let own = LoadRow {
        running: 1,
        source: SignalSource::Events,
        ..LoadRow::default()
    };
    store.record(id, 0, own, now(), LoadSource::Stream);
    probe.release.notify_one();
    while let Some(joined) = schedule.calls.join_next().await {
        driver.land(&mut schedule.due, joined);
    }

    assert_eq!(row(&store, id, 0), Some((own, LoadSource::Stream)));
    let stats = driver.stats();
    assert_eq!(
        stats.scrapes.load(Ordering::Relaxed),
        0,
        "no row from the scrape: not counted"
    );
    assert_eq!(stats.rows.load(Ordering::Relaxed), 0);
}

#[tokio::test(start_paused = true)]
async fn an_engine_with_a_servicer_is_polled_not_scraped() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    ready(
        &store,
        "grpc://a:1",
        "grpc://a:1",
        0,
        1,
        WireModes::GRPC.union(WireModes::HTTP),
        500,
    );
    probe.answers(
        "grpc://a:1",
        &[Figures {
            running: 4,
            waiting: 0,
            usage: 0.4,
        }],
    );
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    tick(&driver, &mut schedule).await;

    assert_eq!(probe.polls(), vec!["grpc://a:1"]);
    assert!(
        probe.scrapes.lock().is_empty(),
        "the scrape stands in for a servicer alone"
    );
}

#[tokio::test(start_paused = true)]
async fn an_engine_that_left_is_forgotten_by_the_probe() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();
    tick(&driver, &mut schedule).await;
    assert!(probe.forgotten.lock().is_empty());

    let revision = store.get(id).expect("the row").revision;
    store.remove(id, revision);
    tokio::time::advance(Duration::from_secs(1)).await;
    tick(&driver, &mut schedule).await;

    assert_eq!(*probe.forgotten.lock(), vec!["grpc://a:1"]);
    assert!(schedule.due.is_empty());
}

#[tokio::test(start_paused = true)]
async fn run_ticks_until_told_to_stop() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    let driver = Arc::new(driver(&store, &probe, TEN));
    let (stop, stopped) = watch::channel(false);
    let mut tasks = JoinSet::new();
    let running = Arc::clone(&driver);
    tasks.spawn(async move { running.run(stopped).await });

    tokio::time::sleep(Duration::from_millis(350)).await;
    assert_eq!(
        probe.polls().len(),
        1,
        "the first tick polled, the next ones waited"
    );
    stop.send_replace(true);
    tasks
        .join_next()
        .await
        .expect("the run task")
        .expect("it ended without a panic");
    assert!(tasks.is_empty());
}

#[tokio::test(start_paused = true)]
async fn with_one_silent_engine_the_others_advance_every_period_and_it_holds_exactly_one_call() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let silent = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    let answering = [
        ready(&store, "grpc://b:1", "grpc://b:1", 0, 1, WireModes::GRPC, 0),
        ready(&store, "grpc://c:1", "grpc://c:1", 0, 1, WireModes::GRPC, 0),
    ];
    let figures = Figures {
        running: 2,
        waiting: 0,
        usage: 0.2,
    };
    for endpoint in ["grpc://a:1", "grpc://b:1", "grpc://c:1"] {
        probe.answers(endpoint, &[figures]);
    }
    // The first engine never answers, and its deadline is far past the test.
    probe.held.lock().insert("grpc://a:1".to_owned());
    let period = Duration::from_secs(1);
    let driver = Arc::new(LoadDriver::new(
        Arc::clone(&store),
        Arc::clone(&probe),
        LoadDriverConfig {
            poll_interval: period,
            poll_deadline: Duration::from_secs(60),
            tick: Duration::from_millis(100),
            ..LoadDriverConfig::default()
        },
    ));
    let (stop, stopped) = watch::channel(false);
    let mut tasks = JoinSet::new();
    let running = Arc::clone(&driver);
    tasks.spawn(async move { running.run(stopped).await });

    // Read between the ticks: every period the two engines that answer have
    // a newer row, and the silent one has its one call out and no row.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let mut newest = [None; 2];
    for round in 1..=5 {
        for (seen, id) in newest.iter_mut().zip(answering) {
            let sample = store
                .load(id, 0)
                .expect("the row of an engine that answers");
            assert!(
                *seen < Some(sample.received_at),
                "round {round}: the row advanced within the period"
            );
            *seen = Some(sample.received_at);
        }
        assert_eq!(store.ranks(silent), 0);
        assert_eq!(
            probe.polls_of("grpc://a:1"),
            1,
            "round {round}: one call out, none queued behind it"
        );
        tokio::time::sleep(period).await;
    }
    assert_eq!(probe.polls_of("grpc://b:1"), 6, "one poll per period");

    // The stop does not wait for the call that is still out.
    stop.send_replace(true);
    tasks
        .join_next()
        .await
        .expect("the run task")
        .expect("it ended without a panic");
}

#[tokio::test(start_paused = true)]
async fn an_engine_whose_call_passed_its_deadline_is_asked_again() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    probe.held.lock().insert("grpc://a:1".to_owned());
    // A poll due every second under the default three-second deadline.
    let driver = driver(&store, &probe, Duration::from_secs(1));
    let mut schedule = Schedule::default();

    driver.tick(&mut schedule);
    tokio::task::yield_now().await;
    for second in 1..3 {
        tokio::time::advance(Duration::from_secs(1)).await;
        driver.tick(&mut schedule);
        tokio::task::yield_now().await;
        assert_eq!(
            probe.polls_of("grpc://a:1"),
            1,
            "t = {second} s: due, and its call still out"
        );
    }
    // The mark is the call's deadline: a call that never came back (an
    // aborted task, a probe that ignored its deadline) frees its engine then.
    tokio::time::advance(Duration::from_secs(1)).await;
    driver.tick(&mut schedule);
    tokio::task::yield_now().await;
    assert_eq!(probe.polls_of("grpc://a:1"), 2);
}

#[tokio::test(start_paused = true)]
async fn a_suppressed_poll_counts_once_per_interval() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();
    let record = LoadRow {
        running: 1,
        ..LoadRow::default()
    };

    // Thirty seconds of a stream that never stops: a record and a tick every
    // second, three poll intervals.
    for _ in 0..30 {
        store.record(id, 0, record, now(), LoadSource::Stream);
        tick(&driver, &mut schedule).await;
        tokio::time::advance(Duration::from_secs(1)).await;
    }

    assert!(probe.polls().is_empty());
    let stats = driver.stats();
    assert_eq!(
        stats.suppressed.load(Ordering::Relaxed),
        3,
        "thirty ticks asked and skipped, counted on the scale of a poll: once an interval"
    );
    assert_eq!(stats.polls.load(Ordering::Relaxed), 0);
}

#[tokio::test(start_paused = true)]
async fn a_poll_that_lands_after_its_worker_left_writes_nothing() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    probe.answers(
        "grpc://a:1",
        &[Figures {
            running: 4,
            waiting: 1,
            usage: 0.4,
        }],
    );
    probe.held.lock().insert("grpc://a:1".to_owned());
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    driver.tick(&mut schedule);
    tokio::task::yield_now().await;
    assert_eq!(probe.polls_of("grpc://a:1"), 1, "the call is out");

    // The worker leaves with its call out; the engine answers afterwards.
    let revision = store.get(id).expect("the row").revision;
    store.remove(id, revision);
    probe.release.notify_one();
    while let Some(joined) = schedule.calls.join_next().await {
        driver.land(&mut schedule.due, joined);
    }

    assert_eq!(store.ranks(id), 0, "a late answer brings no row back");
    assert!(row(&store, id, 0).is_none());
}

#[tokio::test(start_paused = true)]
async fn a_call_that_lands_after_its_engine_left_leaves_no_connection_behind() {
    let store = Arc::new(MemoryStore::new());
    let probe = Arc::new(FakeProbe::default());
    let id = ready(&store, "grpc://a:1", "grpc://a:1", 0, 1, WireModes::GRPC, 0);
    probe.answers(
        "grpc://a:1",
        &[Figures {
            running: 4,
            waiting: 1,
            usage: 0.4,
        }],
    );
    probe.held.lock().insert("grpc://a:1".to_owned());
    let driver = driver(&store, &probe, TEN);
    let mut schedule = Schedule::default();

    // The engine's first call is out and its dial has not completed.
    driver.tick(&mut schedule);
    tokio::task::yield_now().await;
    assert_eq!(probe.polls_of("grpc://a:1"), 1, "the call is out");

    // The engine leaves. The next tick drops it and has the probe forget
    // it, which finds nothing to drop yet.
    let revision = store.get(id).expect("the row").revision;
    store.remove(id, revision);
    driver.tick(&mut schedule);
    assert_eq!(*probe.forgotten.lock(), vec!["grpc://a:1"]);
    assert_eq!(probe.connections(), 0);

    // The dial completes and the engine answers: no later tick names the
    // engine, so the landing is the last chance to drop what the call kept.
    probe.release.notify_one();
    while let Some(joined) = schedule.calls.join_next().await {
        driver.land(&mut schedule.due, joined);
    }
    assert_eq!(
        probe.connections(),
        0,
        "the connection a late call made is dropped"
    );
    assert_eq!(*probe.forgotten.lock(), vec!["grpc://a:1", "grpc://a:1"]);
    assert_eq!(store.ranks(id), 0);
}

#[test]
fn a_failure_counts_under_its_own_label() {
    let errors = [
        ReadError::Transport(String::new()),
        ReadError::Answer(String::new()),
        ReadError::Deadline(Duration::ZERO),
        ReadError::NoLoad(String::new()),
        ReadError::Unsupported(String::new()),
    ];
    assert_eq!(
        errors.len(),
        Failures::LABELS.len(),
        "one label per variant"
    );
    for error in &errors {
        let failures = Failures::default();
        failures.count(error);
        let counted: Vec<&str> = failures
            .each()
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(label, _)| label)
            .collect();
        assert_eq!(counted, vec![error.label()], "{error}");
    }
}

#[test]
fn a_log_line_names_an_engine_by_host_and_port_never_its_url() {
    for (engine, named) in [
        ("grpc://host:50051", "host:50051"),
        ("grpc://user:secret@host:50051", "host:50051"),
        ("https://user:p%40ss@host:8443/v1?token=t#frag", "host:8443"),
        ("grpc://us@er:se@cret@host:1", "host:1"),
        ("http://[::1]:8000/metrics", "[::1]:8000"),
        ("host:9000", "host:9000"),
    ] {
        assert_eq!(peer(engine), named, "{engine}");
    }
}

#[test]
fn an_address_whose_host_cannot_be_told_from_its_userinfo_is_named_by_a_marker() {
    // A raw `/`, `?` or `#` inside the userinfo ends the authority early, and
    // the text before it is part of the userinfo. A query that carries an
    // `@` of its own reads the same, so it is named the same.
    for engine in [
        "grpc://user:pa/ss@host:1",
        "grpc://user:pa?ss@host:1",
        "grpc://user:pa#ss@host:1",
        "http://host:1?next=a@b",
    ] {
        assert_eq!(peer(engine), "<unparsable address>", "{engine}");
    }
}
