//! The counts a host reads of a monitor's polls
//! ([`WorkerMonitor::poll_totals`]): each moves where a tick says its fact,
//! and nowhere else.

use std::collections::VecDeque;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::{
    common_proto::KvEventBatch,
    connection::engine::{AdminAnswer, HealthAnswer},
    load::counts::{FailureTotals, PollTotals},
    worker::ProfileOptions,
};

/// What a scripted load read does at one call: it answers a report, it
/// refuses with a status of the engine's, or it never returns.
type Step = Option<Result<WorkerLoadResponse, tonic::Code>>;

/// An engine on a connection whose load read follows a script, one step a
/// call. Past the script it never returns. It answers no other call.
#[derive(Debug)]
struct Scripted(Mutex<VecDeque<Step>>);

fn unscripted<T>() -> Result<T, tonic::Status> {
    Err(tonic::Status::unimplemented(
        "the case scripts no such call",
    ))
}

#[async_trait::async_trait]
impl EngineConnection for Scripted {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_answer(&self) -> Result<HealthAnswer, tonic::Status> {
        unscripted()
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        let step = self.0.lock().pop_front().flatten();
        match step {
            Some(answer) => answer.map_err(|code| tonic::Status::new(code, "scripted")),
            None => std::future::pending().await,
        }
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        unscripted()
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        unscripted()
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        unscripted()
    }

    async fn model_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        unscripted()
    }

    async fn server_labels(&self) -> Result<HashMap<String, String>, tonic::Status> {
        unscripted()
    }

    async fn subscribe_kv_events(
        &self,
        _start_seq: u64,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        unscripted()
    }

    async fn subscribe_load_records(
        &self,
    ) -> Result<tonic::Streaming<KvEventBatch>, tonic::Status> {
        unscripted()
    }
}

/// A host's load source that never answers.
#[derive(Debug)]
struct NeverLoads;

#[async_trait::async_trait]
impl LoadSource for NeverLoads {
    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        std::future::pending().await
    }
}

/// A host's connector that hands the one connection it was made with, and
/// keeps a load source that never answers for an HTTP worker.
#[derive(Debug)]
struct Hands(Arc<Scripted>);

#[async_trait::async_trait]
impl EngineConnector for Hands {
    async fn connect_grpc(
        &self,
        _base_url: &str,
        _runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        Ok(Arc::clone(&self.0) as Arc<dyn EngineConnection>)
    }

    async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        Err("the case scripts no ZMQ link".to_string())
    }

    async fn dial_grpc(&self, _base_url: &str, _timeout: Duration) -> Result<(), String> {
        Ok(())
    }

    async fn http_load_source(
        &self,
        _base_url: &str,
        _runtime: RuntimeType,
    ) -> Option<Arc<dyn LoadSource>> {
        Some(Arc::new(NeverLoads))
    }
}

/// A connector over a load read that follows `script`.
fn scripted(script: impl IntoIterator<Item = Step>) -> Arc<dyn EngineConnector> {
    Arc::new(Hands(Arc::new(Scripted(Mutex::new(
        script.into_iter().collect(),
    )))))
}

/// A worker of model `a` at `url` on `mode`, of the runtime `runtime`,
/// built with `connector`.
fn worker_on(
    url: &str,
    mode: ConnectionMode,
    runtime: RuntimeType,
    connector: Arc<dyn EngineConnector>,
) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(mode)
            .runtime_type(runtime)
            .model(ModelCard::new("a"))
            .engine_connector(connector)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

/// A worker on a connection at `url`.
fn connected(url: &str, connector: Arc<dyn EngineConnector>) -> Arc<dyn Worker> {
    worker_on(url, ConnectionMode::Grpc, RuntimeType::Vllm, connector)
}

/// A loopback page: `/metrics` answers `status` with `body`, and every
/// other route is absent.
async fn spawn_page(status: StatusCode, body: &'static str) -> String {
    let app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || async move { (status, body) }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the page");
    let addr = listener.local_addr().expect("the page's address");
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve the page");
    });
    format!("http://{addr}")
}

/// A loopback address nothing listens at: a connect there is refused.
async fn closed_address() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    format!("http://{}", listener.local_addr().expect("an address"))
}

/// A worker over HTTP at `url` of a runtime with no gauge table, whose host
/// keeps no load source.
fn no_table_worker(url: &str) -> Arc<dyn Worker> {
    worker_on(
        url,
        ConnectionMode::Http,
        RuntimeType::Unspecified,
        Arc::new(SourceConnector::default()),
    )
}

/// What a cutting engine does behind the headers and half the body of a
/// native answer: it sends no more and keeps the connection, or closes it.
#[derive(Clone, Copy)]
enum Cut {
    Stalls,
    Closes,
}

/// One request's head off `socket`, or nothing where the peer is gone.
async fn request_head(socket: &mut tokio::net::TcpStream) -> Option<String> {
    let mut head = Vec::new();
    let mut chunk = [0u8; 1024];
    while !head.windows(4).any(|end| end == b"\r\n\r\n") {
        match socket.read(&mut chunk).await {
            Ok(0) | Err(_) => return None,
            Ok(read) => head.extend_from_slice(&chunk[..read]),
        }
    }
    Some(String::from_utf8_lossy(&head).into_owned())
}

/// A loopback engine that speaks HTTP by hand. Its gateway route is absent.
/// Its native route answers the two ranks' body whole once, and from then
/// on sends an answer's headers and half its body, and cuts it as `cut`
/// says. The second value is told once a half is sent.
async fn spawn_cutting(cut: Cut) -> (String, Arc<Notify>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the cutting engine");
    let addr = listener.local_addr().expect("its address");
    let asked = Arc::new(AtomicUsize::new(0));
    let sent_half = Arc::new(Notify::new());
    let told = Arc::clone(&sent_half);
    let whole = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{NATIVE_DP_BODY}",
        NATIVE_DP_BODY.len()
    );
    let serve = move |mut socket: tokio::net::TcpStream| {
        let (asked, whole, told) = (Arc::clone(&asked), whole.clone(), Arc::clone(&told));
        async move {
            while let Some(head) = request_head(&mut socket).await {
                let answer = if !head.starts_with("GET /v1/loads") {
                    "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n"
                } else if asked.fetch_add(1, Ordering::SeqCst) == 0 {
                    whole.as_str()
                } else {
                    let half = &whole[..whole.len() - NATIVE_DP_BODY.len() / 2];
                    socket.write_all(half.as_bytes()).await.expect("half");
                    told.notify_one();
                    match cut {
                        Cut::Stalls => std::future::pending().await,
                        Cut::Closes => return,
                    }
                };
                socket.write_all(answer.as_bytes()).await.expect("answer");
            }
        }
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            tokio::spawn(serve(socket));
        }
    });
    (format!("http://{addr}"), sent_half)
}

/// Two fetches of the two ranks at `url` over one memo: the first finds the
/// native route, the second reads it again.
async fn fetch_twice(counts: &PollCounts, url: &str) {
    let memo = DashMap::new();
    for _ in 0..2 {
        WorkerMonitor::fetch_engine_loads(two_ranks(url), ConnectionMode::Http, &memo, counts)
            .await;
    }
}

/// One fetch of the engine that `worker` is, on its own wire.
async fn fetch(counts: &PollCounts, worker: &Arc<dyn Worker>) {
    fetch_engine(counts, vec![Arc::clone(worker)]).await;
}

/// One fetch of the engine whose polled workers are `members`, on their
/// wire.
async fn fetch_engine(counts: &PollCounts, members: Vec<Arc<dyn Worker>>) {
    let memo = DashMap::new();
    let mode = *members[0].connection_mode();
    WorkerMonitor::fetch_engine_loads(members, mode, &memo, counts).await;
}

/// The two rank workers of one engine over HTTP at `url`.
fn two_ranks(url: &str) -> Vec<Arc<dyn Worker>> {
    vec![gauge_dp_worker(url, 0, 2), gauge_dp_worker(url, 1, 2)]
}

/// A fetch over a connection that brings no report is counted by its
/// reason, and one that brings a report counts nothing: the engine that
/// does not implement the call, the status of a connection, any other
/// status, the call the deadline ends, the connect that fails and the
/// connect the deadline ends. The host's own failure is no member to ask
/// and a worker with no client. No read over HTTP is counted, and the one
/// answer's nameless value is the one value dropped.
#[tokio::test(start_paused = true)]
async fn a_fetch_over_a_connection_counts_its_failure_by_its_reason() {
    let counts = PollCounts::default();
    let engine = connected(
        "grpc://engine-a.invalid:50051",
        scripted([
            Some(Ok(host_answer(1))),
            Some(Err(tonic::Code::Unimplemented)),
            Some(Err(tonic::Code::Unavailable)),
            Some(Err(tonic::Code::InvalidArgument)),
        ]),
    );
    for _ in 0..5 {
        fetch(&counts, &engine).await;
    }
    let refused = SourceConnector::keeping(None) as Arc<dyn EngineConnector>;
    fetch(
        &counts,
        &connected("grpc://engine-b.invalid:50051", refused),
    )
    .await;
    fetch(
        &counts,
        &silent_worker("grpc://engine-c.invalid:50051", "a"),
    )
    .await;
    assert_eq!(
        counts.totals(),
        PollTotals {
            poll_failures: FailureTotals {
                transport: 2,
                answer: 1,
                deadline: 2,
                unsupported: 1,
                ..FailureTotals::default()
            },
            values_dropped: 1,
            ..PollTotals::default()
        }
    );

    let no_member = WorkerMonitor::fetch_backend_load(&[]).await;
    assert_eq!(no_member.err(), Some(FetchFailure::Client));
    let over_http = gauge_worker("http://engine-d.invalid:8000");
    let no_client = WorkerMonitor::fetch_backend_load(&[over_http]).await;
    assert_eq!(no_client.err(), Some(FetchFailure::Client));
}

/// A read over HTTP that brings a report counts one, off the native route,
/// off the gauges and off a host's source alike, and the rank workers of
/// one engine count the one read they share. One that brings none is
/// counted by its reason: a status that is no success, a page of either
/// family with no load gauge, an address nothing answers at, a runtime with
/// no gauge table and no native route, a rank worker whose engine has no
/// native route, two ranks of such an engine once, and a host's source that
/// fails. No fetch over a connection is counted.
#[tokio::test]
async fn a_read_over_http_counts_a_report_or_its_failure_by_its_reason() {
    let counts = PollCounts::default();
    let native = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
    fetch(&counts, &gauge_worker(&native.url)).await;
    let gauges = spawn_engine(StatusCode::NOT_FOUND, "").await;
    fetch(&counts, &gauge_worker(&gauges.url)).await;
    let hosted = SourceConnector::keeping(Some(host_answer(1)));
    fetch(
        &counts,
        &hosted_worker("http://engine-a.invalid:8000", &hosted, None),
    )
    .await;
    let ranked = spawn_engine(StatusCode::OK, NATIVE_DP_BODY).await;
    fetch_engine(&counts, two_ranks(&ranked.url)).await;
    assert_eq!(
        counts.totals(),
        PollTotals {
            scrapes: 4,
            values_dropped: 1,
            ..PollTotals::default()
        },
        "four reports, and the one nameless value of the source's answer"
    );

    let refusing = spawn_page(StatusCode::INTERNAL_SERVER_ERROR, "").await;
    fetch(&counts, &gauge_worker(&refusing)).await;
    let bare = spawn_page(StatusCode::OK, "up 1\n").await;
    fetch(&counts, &gauge_worker(&bare)).await;
    fetch(&counts, &gauge_worker(&closed_address().await)).await;
    // The second family at the page with no gauge: its host keeps no source.
    let second = worker_on(
        &bare,
        ConnectionMode::Http,
        RuntimeType::Sglang,
        Arc::new(SourceConnector::default()),
    );
    fetch(&counts, &second).await;
    // No native route and no gauges to read: no load read, and the two
    // ranks of one engine count the one read they share.
    fetch(&counts, &no_table_worker(&gauges.url)).await;
    fetch(&counts, &gauge_dp_worker(&gauges.url, 1, 2)).await;
    fetch_engine(&counts, two_ranks(&gauges.url)).await;
    let unread = SourceConnector::keeping(None);
    fetch(
        &counts,
        &hosted_worker("http://engine-b.invalid:8000", &unread, None),
    )
    .await;
    assert_eq!(
        counts.totals().scrape_failures,
        FailureTotals {
            transport: 1,
            answer: 2,
            no_load: 2,
            unsupported: 3,
            ..FailureTotals::default()
        }
    );
    assert_eq!(counts.totals().poll_failures, FailureTotals::default());
}

/// A worker with no gauges to read fails as its native read did, and has
/// no load read only where no native route is there: a rank worker and a
/// worker of a runtime with no gauge table, at an address nothing answers
/// at and at an engine whose native route refuses. The rank workers of one
/// engine count the one read they share, once. A native route that
/// answered, and then answers a body that is no load report, is the
/// engine's answer, and one that then closes on half a body is the
/// transport's. A body that is no load report on a route that never
/// answered is the engine's answer once: the route is absent from then on.
#[tokio::test]
async fn a_native_read_that_fails_counts_its_reason_once_an_engine() {
    let counts = PollCounts::default();
    let closed = closed_address().await;
    fetch(&counts, &gauge_dp_worker(&closed, 1, 2)).await;
    fetch(&counts, &no_table_worker(&closed)).await;
    let ranks = two_ranks(&closed);
    let answers =
        WorkerMonitor::fetch_engine_loads(ranks, ConnectionMode::Http, &DashMap::new(), &counts)
            .await;
    assert!(
        matches!(answers.as_slice(), [(_, None), (_, None)]),
        "each rank's report is missing"
    );
    let unreached = FailureTotals {
        transport: 3,
        ..FailureTotals::default()
    };
    assert_eq!(counts.totals().scrape_failures, unreached);

    let refusing = spawn_engine(StatusCode::INTERNAL_SERVER_ERROR, "").await;
    fetch(&counts, &gauge_dp_worker(&refusing.url, 1, 2)).await;
    fetch(&counts, &no_table_worker(&refusing.url)).await;
    fetch_engine(&counts, two_ranks(&refusing.url)).await;
    let refused = FailureTotals {
        answer: 3,
        ..unreached
    };
    assert_eq!(counts.totals().scrape_failures, refused);

    let turning = spawn_backend(
        &[(StatusCode::NOT_FOUND, "")],
        &[
            (StatusCode::OK, NATIVE_DP_BODY),
            (StatusCode::OK, UNREADABLE_NATIVE_BODY),
        ],
    )
    .await;
    fetch_twice(&counts, &turning.url).await;
    assert_eq!(
        counts.totals(),
        PollTotals {
            scrapes: 1,
            scrape_failures: FailureTotals {
                answer: 4,
                ..unreached
            },
            ..PollTotals::default()
        },
        "the route answered once: its unread body is one bad answer"
    );

    fetch_twice(&counts, &spawn_cutting(Cut::Closes).await.0).await;
    let strange = spawn_engine(StatusCode::OK, UNREADABLE_NATIVE_BODY).await;
    fetch_twice(&counts, &strange.url).await;
    assert_eq!(
        counts.totals(),
        PollTotals {
            scrapes: 2,
            scrape_failures: FailureTotals {
                transport: 4,
                answer: 5,
                unsupported: 1,
                ..FailureTotals::default()
            },
            ..PollTotals::default()
        },
        "a cut body, then a strange one and the route it leaves absent"
    );
}

/// A native body that stalls behind its headers is a deadline: the
/// request's timeout ends the body's read. The engine answers whole once,
/// then sends half a body and no more. The clock stops once the half is
/// sent and is moved past the timeout.
#[tokio::test]
async fn a_native_body_that_stalls_behind_its_headers_counts_a_deadline() {
    let counts = PollCounts::default();
    let (stalling, sent_half) = spawn_cutting(Cut::Stalls).await;
    let timeout = async {
        sent_half.notified().await;
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        tokio::time::pause();
        tokio::time::advance(REQUEST_TIMEOUT + Duration::from_secs(1)).await;
    };
    tokio::join!(fetch_twice(&counts, &stalling), timeout);
    assert_eq!(
        counts.totals(),
        PollTotals {
            scrapes: 1,
            scrape_failures: FailureTotals {
                deadline: 1,
                ..FailureTotals::default()
            },
            ..PollTotals::default()
        }
    );
}

/// A read over HTTP the request's timeout ends is a deadline, the gauges'
/// read and the native read of a rank worker alike, as a host's source
/// that never answers is under the poll's own. A connector that is late to
/// say whether it keeps a source is the host's failure.
#[tokio::test(start_paused = true)]
async fn a_read_over_http_that_a_timeout_ends_counts_a_deadline() {
    let counts = PollCounts::default();
    // A listener that takes no connection: the request is sent and nothing
    // answers it.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let mute = format!("http://{}", listener.local_addr().expect("an address"));
    fetch(&counts, &gauge_worker(&mute)).await;
    fetch(&counts, &gauge_dp_worker(&mute, 1, 2)).await;
    let never = worker_on(
        "http://engine-a.invalid:8000",
        ConnectionMode::Http,
        RuntimeType::Vllm,
        scripted([]),
    );
    fetch(&counts, &never).await;
    let late = SourceConnector::keeping(Some(host_answer(1)));
    late.silent_asks.store(1, Ordering::Relaxed);
    fetch(
        &counts,
        &hosted_worker("http://engine-b.invalid:8000", &late, None),
    )
    .await;
    assert_eq!(
        counts.totals(),
        PollTotals {
            scrape_failures: FailureTotals {
                deadline: 3,
                client: 1,
                ..FailureTotals::default()
            },
            ..PollTotals::default()
        }
    );
    drop(listener);
}

/// A group's tick counts each worker's poll decision where the group is on
/// a connection: polled with no record on file, polled behind records that
/// went quiet, and left to its stream. It counts the rank rows it hands
/// the publish and the named values the bounds took out of the answers.
/// The tick of a group over HTTP counts no decision: its read is counted
/// by its outcome.
#[tokio::test]
async fn a_tick_counts_the_decisions_of_a_group_on_a_connection_and_its_rows() {
    let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
    let answering = || scripted([Some(Ok(host_answer(1)))]);
    let polled = connected("grpc://engine-a.invalid:50051", answering());
    let quiet = connected("grpc://engine-b.invalid:50051", answering());
    let pushing = connected("grpc://engine-c.invalid:50051", scripted([]));
    let page = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
    let over_http = gauge_worker(&page.url);
    for worker in [&polled, &quiet, &pushing, &over_http] {
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(worker)).unwrap();
    }
    let key = |connection_mode| WorkerGroupKey {
        model_id: "a".to_string(),
        worker_type: WorkerType::Regular,
        connection_mode,
    };
    let interval = Duration::from_secs(10);
    let tick = Instant::now();
    let long_ago = tick
        .checked_sub(3 * interval)
        .expect("the clock is older than three intervals");
    let record = PushedRecord::requests(1, 0);
    monitor.apply_pushed_load(&quiet, 0, &record, long_ago);
    monitor.apply_pushed_load(&pushing, 0, &record, tick);
    assert_eq!(monitor.poll_totals(), PollTotals::default(), "no tick yet");

    poll_group_once(&monitor, &key(ConnectionMode::Grpc), interval, tick).await;
    let on_connection = PollTotals {
        polls: 1,
        fallbacks: 1,
        suppressed: 1,
        rows: 2,
        values_dropped: 2,
        ..PollTotals::default()
    };
    assert_eq!(monitor.poll_totals(), on_connection);

    poll_group_once(&monitor, &key(ConnectionMode::Http), interval, tick).await;
    assert_eq!(
        monitor.poll_totals(),
        PollTotals {
            scrapes: 1,
            rows: 3,
            ..on_connection
        }
    );
}
