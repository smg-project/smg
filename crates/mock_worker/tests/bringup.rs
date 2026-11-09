//! The process when a worker cannot come up. A ZMQ rank's frontend may bind
//! its handshake endpoint late (a gateway binds a start-up worker's at that
//! worker's first health probe): the rank dials in rounds and takes it. A
//! worker that stops on its own leaves the others serving; the last one's
//! stop ends the process with a failure status, so a launcher that waits for
//! the fleet to answer reads the failure at once, with the log that says why,
//! instead of waiting on a process that serves nothing.
#![cfg(unix)]

use std::{
    io::Read,
    net::{SocketAddr, TcpListener, TcpStream},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex, PoisonError},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use engine_zmq_client::{
    connect_handshake,
    mock_engine::MOCK_DEADLINE,
    protocol::vllm::{request::EngineCoreRequest, sampling::EngineCoreSamplingParams},
    EngineCoreClient,
};
use futures::StreamExt;

mod common;
use common::free_port;

const BIN: &str = env!("CARGO_BIN_EXE_mock-worker");

/// The binary under test: this build's, or the one `MOCK_WORKER_BIN` names
/// (to run the suite against another build for a before/after).
fn binary() -> String {
    std::env::var("MOCK_WORKER_BIN").unwrap_or_else(|_| BIN.to_string())
}

/// A port held by this test, so a worker told to bind it cannot.
fn held_port() -> (TcpListener, String) {
    let held = TcpListener::bind("127.0.0.1:0").expect("hold a port");
    let port = held.local_addr().expect("listener address").port();
    (held, port.to_string())
}

/// A mock worker process started with `args`, its log read as it comes (so a
/// chatty process never blocks on a full pipe while the test waits) into a
/// buffer the test can read while the process runs.
struct Process {
    child: Child,
    log: Arc<Mutex<String>>,
    reader: Option<JoinHandle<()>>,
}

/// A test that panics while the child runs leaves no mock worker behind on
/// the host (std's `Child` does not kill its process when dropped).
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Process {
    fn start(args: &[&str]) -> Self {
        let mut child = Command::new(binary())
            .args(["--host", "127.0.0.1"])
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn the mock worker");
        let mut out = child.stdout.take().expect("piped stdout");
        let log = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&log);
        let reader = thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match out.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => sink
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push_str(&String::from_utf8_lossy(&buf[..n])),
                }
            }
        });
        Self {
            child,
            log,
            reader: Some(reader),
        }
    }

    /// The log so far.
    fn log(&self) -> String {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Wait up to `limit` for `needle` to appear in the log.
    fn wait_for_log(&self, needle: &str, limit: Duration) {
        let deadline = Instant::now() + limit;
        while !self.log().contains(needle) {
            assert!(
                Instant::now() < deadline,
                "no {needle:?} in the log within {limit:?}; log:\n{}",
                self.log()
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    /// SIGTERM, as a container runtime sends it.
    fn terminate(&self) {
        let status = Command::new("kill")
            .arg("-TERM")
            .arg(self.child.id().to_string())
            .status()
            .expect("run kill");
        assert!(status.success(), "kill -TERM failed: {status}");
    }

    /// Wait up to `limit` for the exit; a process still alive by then is
    /// killed and the test fails. Returns the exit status and the log.
    fn exit(mut self, limit: Duration) -> (ExitStatus, String) {
        let deadline = Instant::now() + limit;
        loop {
            if let Some(status) = self.child.try_wait().expect("try_wait") {
                self.join_reader();
                return (status, self.log());
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                self.join_reader();
                panic!(
                    "the mock worker did not exit within {limit:?}; log:\n{}",
                    self.log()
                );
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// The whole log is in the buffer once the reader has seen the pipe close.
    fn join_reader(&mut self) {
        if let Some(reader) = self.reader.take() {
            reader.join().expect("stdout reader");
        }
    }
}

fn wait_until_listening(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(20);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    while TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_err() {
        assert!(
            Instant::now() < deadline,
            "the mock worker did not listen on port {port} within 20 s"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn the_last_workers_stop_ends_the_process_with_a_failure_status() {
    // The one worker cannot bind its port (held here); the admin API can
    // bind its own. The process ends all the same, with status 1, and the
    // worker's own error line comes before the line that says the process
    // stops: a launcher that quotes a log's last error line quotes the cause.
    let (held, held_port) = held_port();
    let admin_port = free_port().to_string();
    let mock = Process::start(&[
        "--grpc-base-port",
        &held_port,
        "--grpc-count",
        "1",
        "--admin-port",
        &admin_port,
    ]);
    let (status, log) = mock.exit(Duration::from_secs(20));
    drop(held);
    assert_eq!(status.code(), Some(1), "exit code; log:\n{log}");
    let cause = log
        .find("grpc worker bind")
        .expect("the bind failure is in the log");
    let stop = log
        .find("the last mock worker stopped; nothing left to serve, exiting with a failure status")
        .expect("the stop is in the log");
    assert!(cause < stop, "the cause comes before the stop; log:\n{log}");
    assert!(!log.contains("still serving"), "log:\n{log}");
}

#[test]
fn a_worker_that_stops_leaves_the_others_serving() {
    // One worker cannot bind, one can: the process stays up on the one that
    // serves, says so, and still stops cleanly on a signal.
    let (held, held_port) = held_port();
    let http_port = free_port();
    let mock = Process::start(&[
        "--grpc-base-port",
        &held_port,
        "--grpc-count",
        "1",
        "--http-base-port",
        &http_port.to_string(),
        "--http-count",
        "1",
    ]);
    wait_until_listening(http_port);
    // The stopped worker's line first: a signal that won the supervisor's
    // select against the join would take the drain path without it.
    mock.wait_for_log(
        "a mock worker stopped (likely a port bind failure); 1 still serving",
        Duration::from_secs(10),
    );
    mock.terminate();
    let (status, log) = mock.exit(Duration::from_secs(10));
    drop(held);
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(
        log.contains("a mock worker stopped (likely a port bind failure); 1 still serving"),
        "log:\n{log}"
    );
    assert!(log.contains("drained; exiting"), "log:\n{log}");
}

/// Longer than the mock engine module's one read or wait.
fn past_the_mock_deadline() -> Duration {
    MOCK_DEADLINE + Duration::from_secs(2)
}

/// A rank's three ipc endpoints (handshake, input, output) in a directory of
/// their own, which lives as long as the returned handle.
fn ipc_endpoints() -> (tempfile::TempDir, String, String, String) {
    let dir = tempfile::tempdir().expect("a directory for the ipc endpoints");
    let endpoint = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (
        endpoint("hs.sock"),
        endpoint("in.sock"),
        endpoint("out.sock"),
    );
    (dir, handshake, input, output)
}

/// A request of four prompt tokens for four output tokens.
fn a_request(id: &str) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: id.to_string(),
        prompt_token_ids: Some(vec![1, 2, 3, 4]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens: 4,
            ..Default::default()
        }),
        ..Default::default()
    }
}

// The two tests below each wait out one mock deadline (30 s), the behaviour
// under test; they run side by side, so the file's wall time is one deadline.

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn a_zmq_rank_waits_for_a_frontend_that_binds_after_its_first_round() {
    // The frontend binds its handshake endpoint after the rank's first wait
    // for it has ended (the rank's own line says so; no fixed margin, which a
    // slow start could eat): the rank registers, answers, and still stops
    // cleanly on a signal.
    let (_dir, handshake, input, output) = ipc_endpoints();
    let mock = Process::start(&["--zmq-handshake", &handshake, "--zmq-count", "1"]);
    mock.wait_for_log(
        "again (round 2 of 3)",
        MOCK_DEADLINE + Duration::from_secs(20),
    );
    let transport = connect_handshake(&handshake, 1, &input, &output, Duration::from_secs(20))
        .await
        .expect("the rank registers with a frontend that bound after its first round");
    let client = EngineCoreClient::new(transport);
    let mut answer = client
        .submit(a_request("after-the-late-bind"))
        .await
        .expect("submit a request to the rank");
    answer
        .next()
        .await
        .expect("a batch of the request")
        .expect("ok");
    mock.terminate();
    let (status, log) = mock.exit(Duration::from_secs(10));
    drop(answer);
    drop(client);
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(log.contains("again (round 2 of 3)"), "log:\n{log}");
    assert!(log.contains("zmq mock engine 0 connected"), "log:\n{log}");
    assert!(!log.contains("handshake failed"), "log:\n{log}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn a_connected_zmq_rank_waits_past_the_read_bound_for_its_requests() {
    // The frontend is there for the rank's first round. Then nothing arrives
    // for longer than the mock's read bound: the rank is still there for the
    // next request, as an engine is, and still stops cleanly on a signal.
    let (_dir, handshake, input, output) = ipc_endpoints();
    let mock = Process::start(&["--zmq-handshake", &handshake, "--zmq-count", "1"]);
    let transport = connect_handshake(&handshake, 1, &input, &output, Duration::from_secs(20))
        .await
        .expect("the rank registers in its first round");
    let client = EngineCoreClient::new(transport);
    tokio::time::sleep(past_the_mock_deadline()).await;
    let mut answer = client
        .submit(a_request("after-the-idle"))
        .await
        .expect("submit a request after the idle");
    answer
        .next()
        .await
        .expect("a batch of the request after the idle")
        .expect("ok");
    mock.terminate();
    let (status, log) = mock.exit(Duration::from_secs(10));
    drop(answer);
    drop(client);
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(log.contains("zmq mock engine 0 connected"), "log:\n{log}");
    assert!(!log.contains("again (round"), "log:\n{log}");
    assert!(!log.contains("input closed"), "log:\n{log}");
}
