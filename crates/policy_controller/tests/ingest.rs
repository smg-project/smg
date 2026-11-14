//! The load path against engines in-process: the mock worker's gRPC engine
//! answers the `GetLoads` poll through the controller's own probe, a small
//! HTTP server answers the scrape, and the store's rows and the admin
//! listener's counters are read back. The Numbers cell at the end drives
//! the driver against a fleet of mock engines and prints its cost.

#[path = "common/net.rs"]
mod net;

use std::{
    io::Write,
    net::SocketAddr,
    sync::{atomic::Ordering, Arc},
    time::{Duration, Instant},
};

use axum::{routing::get, Router};
use net::{http_get, loopback, vm_rss_kib};
use openai_protocol::worker::RuntimeType;
use smg_policy_controller::{
    controller::{Config, Controller, Listener, Scope},
    ingest::{
        loads::{LoadDriver, LoadDriverConfig, LoadProbe, ReadError, Target},
        probe::EngineProbe,
    },
    store::{LoadSink, LoadSource, MemoryStore, Signals, WorkerFacts, WorkerStatus, WorkersStore},
    view::{Role, SignalSource, WireModes, WorkerId},
};
use tokio::{
    net::TcpListener,
    sync::{oneshot, watch},
    task::JoinSet,
};

/// A metrics body of the first gauge family: three requests running, five
/// waiting, three quarters of the KV cache in use.
const METRICS_TEXT: &str = "# HELP vllm:num_requests_running Requests running.\n\
                            vllm:num_requests_running{model_name=\"m\"} 3\n\
                            vllm:num_requests_waiting{model_name=\"m\"} 5\n\
                            vllm:kv_cache_usage_perc{model_name=\"m\"} 0.75\n";

/// A mock gRPC engine of the vLLM wire with `ranks` data-parallel ranks,
/// served until the returned sender drops. Returns its address.
async fn mock_engine(ranks: u16) -> Result<(SocketAddr, oneshot::Sender<()>), std::io::Error> {
    let listener = TcpListener::bind(loopback()).await?;
    let addr = listener.local_addr()?;
    let cfg = Arc::new(mock_worker::config::Config {
        grpc_base_port: addr.port(),
        grpc_count: 1,
        grpc_dp_size: ranks,
        grpc_wire: mock_worker::config::SchedulerWire::Vllm,
        loads_like: mock_worker::engine::LoadsLike::Vllm,
        realistic: true,
        ..mock_worker::config::Config::default()
    });
    let (stop, stopped) = oneshot::channel();
    let mut served = JoinSet::new();
    served.spawn(mock_worker::grpc::serve_with_listener_until(
        cfg,
        listener,
        async move {
            let _ = stopped.await;
        },
    ));
    served.detach_all();
    Ok((addr, stop))
}

/// An HTTP server answering `/metrics` with `body`, served until the
/// returned sender drops. Returns its base URL.
async fn metrics_server(
    body: &'static str,
) -> Result<(String, oneshot::Sender<()>), std::io::Error> {
    let listener = TcpListener::bind(loopback()).await?;
    let addr = listener.local_addr()?;
    let app = Router::new().route("/metrics", get(move || async move { body }));
    let (stop, stopped) = oneshot::channel::<()>();
    let mut served = JoinSet::new();
    served.spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = stopped.await;
            })
            .await;
    });
    served.detach_all();
    Ok((format!("http://{addr}"), stop))
}

/// A ready row of `engine` at `key`, rank `rank` of `size`.
fn ready(
    store: &MemoryStore,
    key: &str,
    engine: &str,
    rank: u16,
    size: u16,
    wires: WireModes,
    scrape_ms: u32,
) -> Option<WorkerId> {
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
    let revision = store.get(id)?.revision;
    store.set_status(id, WorkerStatus::Ready, revision);
    Some(id)
}

/// Waits until `check` passes or `within` runs out.
async fn eventually(within: Duration, mut check: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if check() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    check()
}

#[tokio::test]
async fn the_probe_polls_a_mock_engine_and_every_rank_gets_its_row() {
    let (addr, _engine) = mock_engine(2).await.expect("a mock engine");
    let store = Arc::new(MemoryStore::new());
    let endpoint = format!("http://{addr}");
    let rank0 = ready(
        &store,
        &format!("{endpoint}@0"),
        &endpoint,
        0,
        2,
        WireModes::GRPC,
        0,
    )
    .expect("rank 0");
    let rank1 = ready(
        &store,
        &format!("{endpoint}@1"),
        &endpoint,
        1,
        2,
        WireModes::GRPC,
        0,
    )
    .expect("rank 1");
    let probe = Arc::new(EngineProbe::new().expect("an HTTP client"));
    let driver = Arc::new(LoadDriver::new(
        Arc::clone(&store),
        Arc::clone(&probe),
        LoadDriverConfig {
            poll_interval: Duration::from_secs(1),
            tick: Duration::from_millis(100),
            ..LoadDriverConfig::default()
        },
    ));
    let (stop, stopped) = watch::channel(false);
    let mut tasks = JoinSet::new();
    let running = Arc::clone(&driver);
    tasks.spawn(async move { running.run(stopped).await });

    let both = eventually(Duration::from_secs(5), || {
        store.load(rank0, 0).is_some() && store.load(rank1, 1).is_some()
    })
    .await;
    assert!(both, "both ranks' rows arrive from one poll");
    let sample = store.load(rank0, 0).expect("rank 0's row");
    assert_eq!(sample.source, LoadSource::Poll);
    assert_eq!(sample.row.source, SignalSource::Poll);
    assert!(
        sample.row.max_running > 0,
        "the engine reports its admission window"
    );
    assert!(store.load(rank0, 1).is_none(), "a rank's row is its own");
    assert_eq!(
        probe.connections(),
        1,
        "one connection per engine, kept across polls"
    );

    let twice = eventually(Duration::from_secs(5), || {
        driver.stats().polls.load(Ordering::Relaxed) >= 2
    })
    .await;
    assert!(twice, "the poll repeats at its interval");
    stop.send_replace(true);
    tasks
        .join_next()
        .await
        .expect("the run task")
        .expect("no panic");
}

#[tokio::test]
async fn an_engine_that_goes_away_fails_on_the_transport_and_its_connection_is_dropped() {
    let (addr, engine) = mock_engine(1).await.expect("a mock engine");
    let probe = EngineProbe::new().expect("an HTTP client");
    let target = Target {
        endpoint: format!("http://{addr}"),
        kind: RuntimeType::Vllm,
    };
    let answer = probe
        .poll(&target, Duration::from_secs(3))
        .await
        .expect("the engine answers");
    assert!(!answer.loads.is_empty());
    assert_eq!(
        probe.connections(),
        1,
        "the connection is kept for the next poll"
    );

    drop(engine);
    let deadline = Instant::now() + Duration::from_secs(10);
    let error = loop {
        match probe.poll(&target, Duration::from_secs(3)).await {
            Err(error) => break error,
            Ok(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Ok(_) => panic!("the engine kept answering after it stopped"),
        }
    };
    assert!(matches!(error, ReadError::Transport(_)), "{error}");
    assert_eq!(
        probe.connections(),
        0,
        "a failed connection is dropped for a redial"
    );

    let listener = TcpListener::bind(loopback()).await.expect("a port");
    let closed = listener.local_addr().expect("its address");
    drop(listener);
    let error = probe
        .poll(
            &Target {
                endpoint: format!("http://{closed}"),
                kind: RuntimeType::Vllm,
            },
            Duration::from_secs(3),
        )
        .await
        .expect_err("nothing listens");
    assert!(matches!(error, ReadError::Transport(_)), "{error}");
    // The poll's failure line prints this text: it holds no URL.
    assert!(!error.to_string().contains("://"), "{error}");

    let mlx = Target {
        endpoint: format!("http://{addr}"),
        kind: RuntimeType::Mlx,
    };
    let error = probe
        .poll(&mlx, Duration::from_secs(3))
        .await
        .expect_err("no GetLoads on that family");
    assert!(matches!(error, ReadError::Unsupported(_)), "{error}");
}

#[tokio::test]
async fn an_engine_re_registered_under_another_kind_at_one_address_is_asked_through_its_own_family()
{
    // The mock engine speaks the vLLM wire. Its first registration names
    // another family: the engine refuses that family's call, an answer,
    // which keeps the connection.
    let (addr, _engine) = mock_engine(1).await.expect("a mock engine");
    let probe = EngineProbe::new().expect("an HTTP client");
    let endpoint = format!("http://{addr}");
    let wrong = Target {
        endpoint: endpoint.clone(),
        kind: RuntimeType::Sglang,
    };
    let error = probe
        .poll(&wrong, Duration::from_secs(3))
        .await
        .expect_err("the engine refuses another family's call");
    assert!(matches!(error, ReadError::Answer(_)), "{error}");
    assert_eq!(probe.connections(), 1, "a refusal keeps the connection");

    // The registration is corrected at the same address: the poll goes
    // through a client of the engine's own family, not the one kept.
    let right = Target {
        endpoint,
        kind: RuntimeType::Vllm,
    };
    let answer = probe
        .poll(&right, Duration::from_secs(3))
        .await
        .expect("the engine's own family answers");
    assert!(!answer.loads.is_empty());
    assert_eq!(
        probe.connections(),
        1,
        "one client per address: the other family's was replaced"
    );

    // And back: the client kept is the last family asked, never both.
    let error = probe
        .poll(&wrong, Duration::from_secs(3))
        .await
        .expect_err("refused again");
    assert!(matches!(error, ReadError::Answer(_)), "{error}");
    assert_eq!(probe.connections(), 1);
}

#[tokio::test]
async fn the_scrape_reads_the_metrics_endpoint_and_a_closed_port_fails() {
    let (url, _server) = metrics_server(METRICS_TEXT)
        .await
        .expect("a metrics server");
    let probe = EngineProbe::new().expect("an HTTP client");
    let text = probe
        .scrape(&url, Duration::from_secs(3))
        .await
        .expect("the metrics text");
    assert_eq!(text, METRICS_TEXT);

    let listener = TcpListener::bind(loopback()).await.expect("a port");
    let closed = format!("http://{}", listener.local_addr().expect("its address"));
    drop(listener);
    let error = probe
        .scrape(&closed, Duration::from_secs(3))
        .await
        .expect_err("nothing listens");
    assert!(matches!(error, ReadError::Transport(_)), "{error}");
    // The scrape's failure line prints this text: the cause, and no URL
    // (the HTTP client's own text ends with the request's).
    let line = error.to_string();
    assert!(!line.contains("://"), "{line}");
    assert!(line.contains("Connect"), "{line}");
}

#[tokio::test]
async fn the_controller_runs_the_driver_and_renders_its_counters() {
    let (addr, _engine) = mock_engine(1).await.expect("a mock engine");
    let (url, _server) = metrics_server(METRICS_TEXT)
        .await
        .expect("a metrics server");
    let controller = Controller::bind(Config {
        controller_id: "c-1".to_owned(),
        listeners: vec![Listener {
            name: "all".to_owned(),
            addr: loopback(),
            scopes: Scope::ALL.to_vec(),
            tls: None,
        }],
        admin: loopback(),
        shutdown_settle: Duration::ZERO,
        shutdown_grace: Duration::from_secs(5),
        // Long enough that the counter below reads the first poll alone,
        // however slow the host; the first tick polls at once.
        load_poll_interval: Duration::from_secs(10),
    })
    .await
    .expect("the controller binds");
    let admin = controller.addresses().admin;
    let store = controller.store();
    let endpoint = format!("http://{addr}");
    let polled =
        ready(&store, &endpoint, &endpoint, 0, 1, WireModes::GRPC, 0).expect("the polled row");
    let scraped = ready(&store, &url, &url, 0, 1, WireModes::HTTP, 200).expect("the scraped row");
    let (stop, stopped) = oneshot::channel::<()>();
    let mut serving = JoinSet::new();
    serving.spawn(async move {
        controller
            .serve(async move {
                let _ = stopped.await;
            })
            .await
    });

    let both = eventually(Duration::from_secs(5), || {
        store.load(polled, 0).is_some() && store.load(scraped, 0).is_some()
    })
    .await;
    assert!(both, "the driver polls and scrapes from the serve");
    let scrape = store.load(scraped, 0).expect("the scrape's row");
    assert_eq!(scrape.source, LoadSource::Scrape);
    assert_eq!(
        (
            scrape.row.running,
            scrape.row.queued,
            scrape.row.kv_usage_permille
        ),
        (3, 5, 750)
    );

    let (status, body) = http_get(admin, "/metrics").await.expect("the metrics page");
    assert_eq!(status, 200);
    assert!(
        body.contains("smg_policy_controller_load_polls_total{mode=\"poll\"} 1"),
        "{body}"
    );
    assert!(
        body.contains("smg_policy_controller_load_scrapes_total "),
        "{body}"
    );
    assert!(
        body.contains("smg_policy_controller_load_scrape_failures_total{reason=\"no_load\"} 0"),
        "{body}"
    );
    assert!(
        body.contains("smg_policy_controller_load_rows_total "),
        "{body}"
    );

    stop.send(()).expect("the serve is listening for the stop");
    serving
        .join_next()
        .await
        .expect("the serve task")
        .expect("no panic")
        .expect("the serve ends clean");
}

/// CPU seconds of this process (user + system), from `/proc/self/stat`.
fn cpu_seconds() -> f64 {
    let ticks = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            let after = stat.rsplit_once(')')?.1;
            let fields: Vec<&str> = after.split_whitespace().collect();
            let utime: f64 = fields.get(11)?.parse().ok()?;
            let stime: f64 = fields.get(12)?.parse().ok()?;
            Some(utime + stime)
        })
        .unwrap_or(0.0);
    ticks / 100.0
}

/// The Numbers cell: the driver alone against `SMG_LOAD_BED_ENGINES` mock
/// engines (8 by default) at a 1 s poll interval for 30 s, its CPU and RSS
/// printed. Run by name with `--ignored --nocapture`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement, not a check: prints the driver's cost against a fleet of mock engines"]
async fn cpu_and_rss_of_the_driver_against_a_fleet_of_mock_engines() {
    let engines: usize = std::env::var("SMG_LOAD_BED_ENGINES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let seconds: u64 = std::env::var("SMG_LOAD_BED_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let store = Arc::new(MemoryStore::new());
    let mut stops = Vec::with_capacity(engines);
    for _ in 0..engines {
        let (addr, stop) = mock_engine(1).await.expect("a mock engine");
        let endpoint = format!("http://{addr}");
        ready(&store, &endpoint, &endpoint, 0, 1, WireModes::GRPC, 0).expect("the row");
        stops.push(stop);
    }
    let probe = Arc::new(EngineProbe::new().expect("an HTTP client"));
    let driver = Arc::new(LoadDriver::new(
        Arc::clone(&store),
        probe,
        LoadDriverConfig {
            poll_interval: Duration::from_secs(1),
            ..LoadDriverConfig::default()
        },
    ));
    let idle_rss = vm_rss_kib();
    let cpu_before = cpu_seconds();
    let started = Instant::now();
    let (stop, stopped) = watch::channel(false);
    let mut tasks = JoinSet::new();
    let running = Arc::clone(&driver);
    tasks.spawn(async move { running.run(stopped).await });
    tokio::time::sleep(Duration::from_secs(seconds)).await;
    stop.send_replace(true);
    tasks
        .join_next()
        .await
        .expect("the run task")
        .expect("no panic");
    let elapsed = started.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu_before;
    let stats = driver.stats();
    let polls = stats.polls.load(Ordering::Relaxed) + stats.fallbacks.load(Ordering::Relaxed);
    let mut out = std::io::stdout();
    writeln!(out, "BED_ENGINES {engines}").expect("stdout");
    writeln!(out, "BED_SECONDS {elapsed:.1}").expect("stdout");
    writeln!(out, "BED_POLLS {polls}").expect("stdout");
    writeln!(
        out,
        "BED_POLL_FAILURES {}",
        stats
            .poll_failures
            .each()
            .iter()
            .map(|(_, n)| n)
            .sum::<u64>()
    )
    .expect("stdout");
    writeln!(out, "BED_ROWS {}", stats.rows.load(Ordering::Relaxed)).expect("stdout");
    writeln!(out, "BED_CPU_SECONDS {cpu:.3}").expect("stdout");
    writeln!(
        out,
        "BED_CPU_PERCENT_OF_ONE_CORE {:.2}",
        cpu / elapsed * 100.0
    )
    .expect("stdout");
    writeln!(
        out,
        "BED_CPU_MICROS_PER_POLL {:.1}",
        if polls == 0 {
            0.0
        } else {
            cpu * 1e6 / polls as f64
        }
    )
    .expect("stdout");
    writeln!(out, "BED_RSS_IDLE_KIB {idle_rss}").expect("stdout");
    writeln!(out, "BED_RSS_END_KIB {}", vm_rss_kib()).expect("stdout");
    drop(stops);
}
