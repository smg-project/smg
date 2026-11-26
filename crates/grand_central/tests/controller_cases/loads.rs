//! The load monitor under a serve: its counts on the admin page, the overload
//! thresholds a start states, and a document that switches the default off.

use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

use axum::{routing::get, Router};
use openai_protocol::worker::RuntimeType;
use smg_grand_central::{
    engine::{Role, WireModes, WorkerId},
    service::controller::{Config, Controller},
    store::{MemoryStore, Overload, WorkerFacts},
    worker_host::{Members, Objects},
};
use smg_worker::health::overload::OverloadThresholds;
use tokio::{net::TcpListener, sync::oneshot, task::JoinSet, time::timeout};

use crate::{
    config,
    engines::polled_engine,
    net::{http_get, loopback},
    ready,
};

/// A metrics body of the first gauge family: three requests running, five
/// waiting, three quarters of the KV cache in use.
const METRICS_TEXT: &str = "# HELP vllm:num_requests_running Requests running.\n\
                            vllm:num_requests_running{model_name=\"m\"} 3\n\
                            vllm:num_requests_waiting{model_name=\"m\"} 5\n\
                            vllm:kv_cache_usage_perc{model_name=\"m\"} 0.75\n";

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

/// A ready worker at `key` on `wires`, of the first gauge family's engine
/// kind, with the queue bound `waiting` where the row states one.
fn ready_row(
    holder: &Objects<Members>,
    store: &MemoryStore,
    key: &str,
    wires: WireModes,
    waiting: Option<u32>,
) -> Option<WorkerId> {
    let stated = WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation: 1,
        role: Role::Single,
        wire_modes: wires,
        engine_kind: RuntimeType::Vllm.as_str().to_owned(),
        models: vec!["m".to_owned()],
        dp_size: 1,
        overload: Overload {
            waiting_requests: waiting,
            ..Overload::default()
        },
        ..WorkerFacts::default()
    };
    ready(holder, store, stated)
}

/// The value of the controller's series `name` on a metrics `page`.
fn series(page: &str, name: &str) -> Option<u64> {
    let line = format!("smg_grand_central_{name} ");
    let value = page.lines().find_map(|read| read.strip_prefix(&line))?;
    value.trim().parse().ok()
}

/// The serve runs the load monitor while the replica leads, and the admin
/// page's load series read the monitor's counts. A worker on a connection
/// that serves no stream of its load records is polled, and a worker with
/// a gauge page alone is read over HTTP. The monitor publishes a report of
/// each. The second's states its page's gauges, over the queue bound its
/// row states: one overload verdict moves, once.
#[tokio::test]
async fn the_serve_runs_the_load_monitor_and_renders_its_counts() {
    const PATIENCE: Duration = Duration::from_secs(30);

    let (addr, _engine) = polled_engine(1).await.expect("a mock engine");
    let (read, _server) = metrics_server(METRICS_TEXT)
        .await
        .expect("a metrics server");
    let polled = format!("http://{addr}");
    let controller = Controller::bind(Config {
        load_poll_interval: Duration::from_secs(1),
        ..config()
    })
    .await
    .expect("the controller binds");
    let admin = controller.addresses().admin;
    let (store, loads) = (controller.store(), controller.loads());
    let holder = controller.workers();
    ready_row(&holder, &store, &polled, WireModes::GRPC, None).expect("the polled row");
    ready_row(&holder, &store, &read, WireModes::HTTP, Some(4)).expect("the read row");
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        // Both reports are published, and the page has counted them.
        let started = Instant::now();
        let page = loop {
            let (status, page) = http_get(admin, "/metrics").await.expect("the metrics page");
            assert_eq!(status, 200);
            let reported = {
                let reports = loads.borrow();
                reports.report(&polled).is_some() && reports.report(&read).is_some()
            };
            let counted = [
                ("load_polls_total{mode=\"poll\"}", 1),
                ("load_scrapes_total", 1),
                ("load_rows_total", 2),
                ("overload_flips_total", 1),
            ]
            .iter()
            .all(|&(name, least)| series(&page, name).is_some_and(|count| count >= least));
            if reported && counted {
                break page;
            }
            assert!(
                started.elapsed() < PATIENCE,
                "the load monitor polls both workers from the serve: {page}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        };
        assert_eq!(
            series(&page, "load_scrape_failures_total{reason=\"no_load\"}"),
            Some(0),
            "{page}"
        );
        // The read worker stays over its bound: its verdict moved once,
        // however many reports followed.
        assert_eq!(series(&page, "overload_flips_total"), Some(1), "{page}");
        let gauges = loads.borrow().report(&read).and_then(|(report, _)| {
            let rank = report.loads.first()?;
            Some((rank.num_running_reqs, rank.num_waiting_reqs))
        });
        assert_eq!(gauges, Some((3, 5)), "the report states the page's gauges");
        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

/// A metrics body of the first gauge family that reads over both of a
/// start's default thresholds: nine requests waiting, nine tenths of the KV
/// cache in use.
const BUSY_METRICS_TEXT: &str = "vllm:num_requests_running{model_name=\"m\"} 3\n\
                                 vllm:num_requests_waiting{model_name=\"m\"} 9\n\
                                 vllm:kv_cache_usage_perc{model_name=\"m\"} 0.9\n";

/// A start states the two overload thresholds that a worker with none of
/// its own is judged by. One worker's page reads 5 queued requests and a
/// usage of 0.75, and its row states no threshold. A controller started
/// with a count of 4 vetoes it, and so does one started with a usage of
/// 0.7: each beside the other signal's default, and each alone. Under the
/// thresholds a start states by default, 8 queued requests and a usage of
/// 0.8, no verdict moves. A signal a start states none for has no default:
/// under a start that states neither, no verdict moves for a worker with no
/// threshold of its own whose page reads 9 queued requests and a usage of
/// 0.9.
#[tokio::test]
async fn a_start_states_the_overload_thresholds_a_worker_with_none_is_judged_by() {
    const PATIENCE: Duration = Duration::from_secs(30);

    let (calm, _calm) = metrics_server(METRICS_TEXT)
        .await
        .expect("a metrics server");
    let (busy, _busy) = metrics_server(BUSY_METRICS_TEXT)
        .await
        .expect("a metrics server");
    let stating = |waiting_requests, token_usage| OverloadThresholds {
        waiting_requests,
        token_usage,
    };
    let stock = Config::DEFAULT_WORKER_OVERLOAD;
    let starts = [
        (&calm, (5, 0.75), stating(Some(4), stock.token_usage), true),
        (
            &calm,
            (5, 0.75),
            stating(stock.waiting_requests, Some(0.7)),
            true,
        ),
        (&calm, (5, 0.75), stock, false),
        (&calm, (5, 0.75), stating(Some(4), None), true),
        (&calm, (5, 0.75), stating(None, Some(0.7)), true),
        (&busy, (9, 0.9), stating(None, None), false),
    ];
    for (worker, (queued, used), worker_overload, vetoed) in starts {
        let controller = Controller::bind(Config {
            load_poll_interval: Duration::from_secs(1),
            worker_overload,
            ..config()
        })
        .await
        .expect("the controller binds");
        let admin = controller.addresses().admin;
        let (store, loads) = (controller.store(), controller.loads());
        let holder = controller.workers();
        ready_row(&holder, &store, worker, WireModes::HTTP, None).expect("the read row");
        let (stop, stopped) = oneshot::channel::<()>();
        let serve = controller.serve(async {
            let _ = stopped.await;
        });
        let probe = async {
            let started = Instant::now();
            let page = loop {
                let read = timeout(PATIENCE, http_get(admin, "/metrics"));
                let (status, page) = read
                    .await
                    .expect("the page answers inside the patience")
                    .expect("the metrics page");
                assert_eq!(status, 200);
                let reported = loads.borrow().report(worker).is_some();
                // Two reads of the worker's page are counted: the first
                // one's report is judged by then.
                let read_twice =
                    series(&page, "load_scrapes_total").is_some_and(|count| count >= 2);
                // A verdict that stands has moved an odd number of times:
                // once, and twice more where a read of the worker's page
                // failed between two good ones. The failed read lifts the
                // verdict and the next good one sets it again.
                let stands =
                    series(&page, "overload_flips_total").is_some_and(|flips| flips % 2 == 1);
                if reported && read_twice && (stands || !vetoed) {
                    break page;
                }
                assert!(
                    started.elapsed() < PATIENCE,
                    "{worker_overload:?} vetoes a worker at {queued} and {used}: {page}"
                );
                tokio::time::sleep(Duration::from_millis(25)).await;
            };
            let stated = loads
                .borrow()
                .report(worker)
                .map(|(report, _)| (report.total_waiting_reqs(), report.effective_kv_pressure()));
            assert!(
                stated.is_some_and(
                    |(waiting, usage)| waiting == queued && (usage - used).abs() < 1e-6
                ),
                "the report states the page's gauges: {stated:?}"
            );
            if !vetoed {
                // No failed read moves a verdict that never stood.
                assert_eq!(
                    series(&page, "overload_flips_total"),
                    Some(0),
                    "{worker_overload:?} at {queued} and {used}: {page}"
                );
            }
            stop.send(()).expect("the serve is listening for the stop");
        };
        let (served, ()) = tokio::join!(serve, probe);
        served.expect("a clean stop");
    }
}

/// A standalone controller reads the policy spec's overload switch at each
/// document that takes force. One worker's page reads 5 queued requests,
/// and its row states no threshold. A controller started with a count of 4
/// vetoes it under a spec file whose binding leaves the switch on. The file
/// is written again with the switch off while the controller runs: the
/// verdict leaves when the file's watch applies the document, and no later
/// read of the worker's page sets it again. Written again with the switch
/// on, the verdict returns at the next report.
#[tokio::test]
async fn a_document_that_switches_the_default_off_lifts_a_served_controllers_verdict() {
    use smg_grand_central::document::API_VERSION;

    const PATIENCE: Duration = Duration::from_secs(30);

    /// The admin page's two counts once `holds` says so of them: the reads
    /// of the worker's page, and the overload verdicts that moved.
    async fn counts(admin: SocketAddr, what: &str, holds: impl Fn(u64, u64) -> bool) -> (u64, u64) {
        let started = Instant::now();
        loop {
            let read = timeout(PATIENCE, http_get(admin, "/metrics"));
            let (status, page) = read
                .await
                .expect("the page answers inside the patience")
                .expect("the metrics page");
            assert_eq!(status, 200);
            let scrapes = series(&page, "load_scrapes_total").unwrap_or(0);
            let flips = series(&page, "overload_flips_total").unwrap_or(0);
            if holds(scrapes, flips) {
                return (scrapes, flips);
            }
            assert!(started.elapsed() < PATIENCE, "{what}: {page}");
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    let (worker, _worker) = metrics_server(METRICS_TEXT)
        .await
        .expect("a metrics server");
    // The controller's source is a file: each document enters through its
    // door, as an operator's does.
    let switched = |enabled: bool| {
        format!(
            "apiVersion: {API_VERSION}\nprofiles:\n  p:\n    pick: random\nbindings:\n  - profile: p\n    overload:\n      enabled: {enabled}\n"
        )
    };
    let file = tempfile::NamedTempFile::new().expect("a policy file");
    std::fs::write(file.path(), switched(true)).expect("the document is written");
    let mut controller = Controller::bind(Config {
        load_poll_interval: Duration::from_secs(1),
        worker_overload: OverloadThresholds {
            waiting_requests: Some(4),
            token_usage: None,
        },
        ..config()
    })
    .await
    .expect("the controller binds");
    controller
        .spec_file(file.path())
        .await
        .expect("the document validates");
    let admin = controller.addresses().admin;
    let (store, holder) = (controller.store(), controller.workers());
    ready_row(&holder, &store, &worker, WireModes::HTTP, None).expect("the read row");
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        // A verdict that stands has moved an odd number of times, and one
        // that is off an even number: a failed read of the worker's page
        // lifts a verdict and the next good one sets it again.
        let stands = |_: u64, flips: u64| flips % 2 == 1;
        let (_, stood) = counts(admin, "the start's count vetoes the worker", stands).await;
        std::fs::write(file.path(), switched(false)).expect("the file is written again");
        let off = |_: u64, flips: u64| flips.is_multiple_of(2);
        let (read, lifted) = counts(admin, "the switch off lifts the verdict", off).await;
        assert!(lifted > stood, "the verdict moved: {stood} then {lifted}");
        // Two more reads of the worker's page: the object states no
        // threshold, so no report sets a verdict.
        let twice = |scrapes: u64, _: u64| scrapes >= read + 2;
        let (_, still) = counts(admin, "two more reads of the worker's page", twice).await;
        assert_eq!(still, lifted, "no read of the page set a verdict");
        std::fs::write(file.path(), switched(true)).expect("the file is written again");
        let (_, back) = counts(admin, "the switch on brings the verdict back", stands).await;
        assert!(back > still, "the verdict moved: {still} then {back}");
        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}
