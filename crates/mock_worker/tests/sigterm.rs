//! The process under a termination signal: SIGTERM (what a container runtime
//! sends when a pod is deleted) or SIGINT stops the mock worker cleanly. No
//! new connection is accepted, the requests in flight finish up to the drain
//! deadline, the exit code is 0, and the log says what was drained.
#![cfg(unix)]

use std::{
    io::Read,
    net::{SocketAddr, TcpListener, TcpStream},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Mutex, MutexGuard, PoisonError},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use engine_zmq_client::{
    connect_handshake,
    protocol::vllm::{
        output::EngineCoreFinishReason, request::EngineCoreRequest,
        sampling::EngineCoreSamplingParams,
    },
    EngineCoreClient,
};
use futures::StreamExt;
use serde_json::json;
use smg_grpc_client::{tokenspeed_proto as ts, TokenSpeedSchedulerClient};

mod common;

const BIN: &str = env!("CARGO_BIN_EXE_mock-worker");

/// The binary under test: this build's, or the one `MOCK_WORKER_BIN` names
/// (to run the suite against another build for a before/after).
fn binary() -> String {
    std::env::var("MOCK_WORKER_BIN").unwrap_or_else(|_| BIN.to_string())
}

/// A slow stream: one token per decode pass of this many milliseconds.
const DECODE_MS: &str = "20";

/// The tests run one at a time: each hands free ports to its own process,
/// and two of them picking at once could pick the same port.
static SERIAL: Mutex<()> = Mutex::new(());

/// Two distinct ports nothing listens on right now.
fn free_ports() -> (u16, u16) {
    let first = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
    let second = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
    let port = |listener: &TcpListener| listener.local_addr().expect("listener address").port();
    (port(&first), port(&second))
}

fn addr(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

/// A mock worker process with one gRPC and one HTTP worker, reachable.
struct Mock {
    /// Held from the start to the exit: see [`SERIAL`].
    _serial: MutexGuard<'static, ()>,
    child: Child,
    stdout: JoinHandle<String>,
    grpc_port: u16,
    http_port: u16,
}

impl Mock {
    fn start(extra: &[&str]) -> Self {
        let serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let (grpc_port, http_port) = free_ports();
        let mut child = Command::new(binary())
            .args(["--grpc-base-port", &grpc_port.to_string()])
            .args(["--grpc-count", "1"])
            .args(["--http-base-port", &http_port.to_string()])
            .args(["--http-count", "1"])
            .args(extra)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn the mock worker");
        // The log is read as it comes, so a chatty process never blocks on a
        // full pipe while the test waits for its exit.
        let mut out = child.stdout.take().expect("piped stdout");
        let stdout = thread::spawn(move || {
            let mut text = String::new();
            let _ = out.read_to_string(&mut text);
            text
        });
        let mock = Self {
            _serial: serial,
            child,
            stdout,
            grpc_port,
            http_port,
        };
        for port in [grpc_port, http_port] {
            let deadline = Instant::now() + Duration::from_secs(20);
            while TcpStream::connect_timeout(&addr(port), Duration::from_millis(200)).is_err() {
                assert!(
                    Instant::now() < deadline,
                    "the mock worker did not listen on port {port} within 20 s"
                );
                thread::sleep(Duration::from_millis(20));
            }
        }
        mock
    }

    /// Send `signal` (`TERM`, `INT`) to the process; returns when it was sent.
    fn signal(&self, signal: &str) -> Instant {
        let status = Command::new("kill")
            .arg(format!("-{signal}"))
            .arg(self.child.id().to_string())
            .status()
            .expect("run kill");
        assert!(status.success(), "kill -{signal} failed: {status}");
        Instant::now()
    }

    /// Wait up to `limit` for the exit; a process still alive by then is
    /// killed and the test fails. Returns the exit status, when it happened
    /// and the process's log.
    fn exit(mut self, limit: Duration) -> (ExitStatus, Instant, String) {
        let deadline = Instant::now() + limit;
        loop {
            if let Some(status) = self.child.try_wait().expect("try_wait") {
                let exited = Instant::now();
                let log = self.stdout.join().expect("stdout reader");
                return (status, exited, log);
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                let log = self.stdout.join().expect("stdout reader");
                panic!("the mock worker did not exit within {limit:?}; log:\n{log}");
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

/// The ports stop accepting connections soon after the signal.
fn assert_refuses_connections(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if TcpStream::connect_timeout(&addr(port), Duration::from_millis(200)).is_err() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "port {port} still accepted connections 3 s after the signal"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

/// A streamed gRPC request for `max_new_tokens` tokens.
fn generate_request(request_id: &str, max_new_tokens: u32) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: request_id.to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: vec![1, 2, 3, 4],
            original_text: "hello".to_string(),
        }),
        sampling_params: Some(ts::SamplingParams {
            max_new_tokens: Some(max_new_tokens),
            ..Default::default()
        }),
        stream: true,
        ..Default::default()
    }
}

/// What a gRPC stream delivered: chunks seen, whether `Complete` arrived.
struct Delivered {
    chunks: usize,
    complete: bool,
    error: Option<tonic::Status>,
}

/// Read the rest of a generate stream.
async fn drain_generate(
    stream: &mut smg_grpc_client::tokenspeed_scheduler::AbortOnDropStream,
    mut delivered: Delivered,
) -> Delivered {
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(frame) => match frame.response {
                Some(ts::generate_response::Response::Chunk(_)) => delivered.chunks += 1,
                Some(ts::generate_response::Response::Complete(_)) => delivered.complete = true,
                _ => {}
            },
            Err(status) => {
                delivered.error = Some(status);
                break;
            }
        }
    }
    delivered
}

#[test]
fn sigterm_with_nothing_in_flight_exits_zero_at_once() {
    let mock = Mock::start(&[]);
    let sent = mock.signal("TERM");
    let (status, exited, log) = mock.exit(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(
        exited - sent < Duration::from_secs(2),
        "an idle process exits at once, not at the deadline: {:?}",
        exited - sent
    );
    assert!(
        log.contains("SIGTERM: draining 0 in-flight request(s)"),
        "the drain is logged; log:\n{log}"
    );
    assert!(log.contains("drained; exiting"), "log:\n{log}");
}

#[test]
fn sigint_is_handled_the_same_way() {
    let mock = Mock::start(&[]);
    let sent = mock.signal("INT");
    let (status, exited, log) = mock.exit(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(
        exited - sent < Duration::from_secs(2),
        "{:?}",
        exited - sent
    );
    assert!(
        log.contains("SIGINT: draining 0 in-flight request(s)"),
        "log:\n{log}"
    );
}

#[tokio::test]
async fn sigterm_lets_the_streams_in_flight_finish_then_exits_zero() {
    // ~40 tokens at 20 ms each: streams of about a second, well inside the
    // default 5 s drain.
    let mock = Mock::start(&["--engine", "realistic", "--decode-base-ms", DECODE_MS]);

    // One gRPC and one HTTP stream in flight, each past its first frame.
    let client =
        TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{}", mock.grpc_port))
            .await
            .expect("connect to the gRPC worker");
    let mut grpc = client
        .generate(generate_request("in-flight", 40))
        .await
        .expect("generate");
    let first = grpc.next().await.expect("a first frame").expect("ok");
    assert!(matches!(
        first.response,
        Some(ts::generate_response::Response::Chunk(_))
    ));
    let http = common::http_client()
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            mock.http_port
        ))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 40,
        }))
        .send()
        .await
        .expect("post the chat completion");
    assert_eq!(http.status(), 200);
    let mut sse = http.bytes_stream();
    let mut body =
        String::from_utf8_lossy(&sse.next().await.expect("a first SSE chunk").expect("ok"))
            .into_owned();

    let sent = mock.signal("TERM");
    // No new connection from the signal on, on either worker.
    assert_refuses_connections(mock.grpc_port);
    assert_refuses_connections(mock.http_port);

    // The streams already running are served to their end.
    let delivered = drain_generate(
        &mut grpc,
        Delivered {
            chunks: 1,
            complete: false,
            error: None,
        },
    )
    .await;
    grpc.mark_completed();
    assert!(
        delivered.error.is_none(),
        "the gRPC stream was cut: {:?}",
        delivered.error
    );
    assert!(delivered.complete, "the gRPC stream ended with Complete");
    assert_eq!(
        delivered.chunks, 40,
        "every token of the gRPC stream arrived"
    );
    while let Some(chunk) = sse.next().await {
        body.push_str(&String::from_utf8_lossy(&chunk.expect("an SSE chunk")));
    }
    assert!(
        body.contains("\"finish_reason\":\"stop\"") && body.contains("[DONE]"),
        "the HTTP stream ended with its finish frame and [DONE]:\n{body}"
    );

    let (status, exited, log) = mock.exit(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(
        exited - sent < Duration::from_secs(4),
        "the process exits when its streams are done, before the 5 s deadline: {:?}",
        exited - sent
    );
    assert!(
        log.contains("SIGTERM: draining 2 in-flight request(s)"),
        "both streams were counted; log:\n{log}"
    );
    assert!(log.contains("drained; exiting"), "log:\n{log}");
}

#[tokio::test]
async fn the_drain_deadline_bounds_the_exit() {
    // A stream of several seconds against a half-second drain: the process
    // exits at the deadline, cleanly, and says what it left behind.
    let mock = Mock::start(&[
        "--engine",
        "realistic",
        "--decode-base-ms",
        DECODE_MS,
        "--drain-timeout-secs",
        "0.5",
    ]);
    let client =
        TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{}", mock.grpc_port))
            .await
            .expect("connect to the gRPC worker");
    let mut grpc = client
        .generate(generate_request("long", 400))
        .await
        .expect("generate");
    grpc.next().await.expect("a first frame").expect("ok");

    let sent = mock.signal("TERM");
    let (status, exited, log) = mock.exit(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    let took = exited - sent;
    assert!(
        took >= Duration::from_millis(500) && took < Duration::from_secs(4),
        "the exit comes at the deadline: {took:?}"
    );
    assert!(
        log.contains("drain deadline of 0.5 s reached with 1 request(s) still in flight"),
        "log:\n{log}"
    );

    // The stream that outlived the deadline ended with the process, short of
    // its tokens.
    let delivered = drain_generate(
        &mut grpc,
        Delivered {
            chunks: 1,
            complete: false,
            error: None,
        },
    )
    .await;
    assert!(
        !delivered.complete,
        "a stream cut by the exit has no Complete"
    );
    assert!(delivered.chunks < 400);
}

#[tokio::test]
async fn sigterm_ends_a_kv_event_subscription_held_open() {
    // The gateway's cache-aware policy keeps a SubscribeKvEvents stream open
    // on every worker; a stop that waited for it would never drain.
    let mock = Mock::start(&["--engine", "realistic"]);
    let client =
        TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{}", mock.grpc_port))
            .await
            .expect("connect to the gRPC worker");
    let mut events = client
        .subscribe_kv_events(0)
        .await
        .expect("subscribe to KV events");
    // The subscription is live: it carries a load heartbeat before any event.
    let first = tokio::time::timeout(Duration::from_secs(5), events.next())
        .await
        .expect("a first batch within 5 s");
    assert!(first.is_some(), "a subscription that ends by itself");

    let sent = mock.signal("TERM");
    let ended = tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(batch) = events.next().await {
            if batch.is_err() {
                break;
            }
        }
    })
    .await;
    assert!(ended.is_ok(), "the subscription ends at the stop");
    let (status, exited, log) = mock.exit(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    eprintln!(
        "kv subscription held open: exit {:?} after SIGTERM",
        exited - sent
    );
    assert!(
        exited - sent < Duration::from_secs(2),
        "the held subscription does not run the drain to its deadline: {:?}",
        exited - sent
    );
    assert!(log.contains("drained; exiting"), "log:\n{log}");
}

#[tokio::test]
async fn a_second_signal_during_the_drain_exits_at_once() {
    // A long stream against the default 5 s drain; a second SIGTERM a moment
    // later ends the wait.
    let mock = Mock::start(&["--engine", "realistic", "--decode-base-ms", DECODE_MS]);
    let client =
        TokenSpeedSchedulerClient::connect(&format!("http://127.0.0.1:{}", mock.grpc_port))
            .await
            .expect("connect to the gRPC worker");
    let mut grpc = client
        .generate(generate_request("long", 400))
        .await
        .expect("generate");
    grpc.next().await.expect("a first frame").expect("ok");

    let first = mock.signal("TERM");
    thread::sleep(Duration::from_millis(300));
    mock.signal("TERM");
    let (status, exited, log) = mock.exit(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(
        exited - first < Duration::from_secs(2),
        "the second signal cuts the drain short: {:?}",
        exited - first
    );
    assert!(
        log.contains("SIGTERM during the drain: exiting now with 1 request(s) in flight"),
        "log:\n{log}"
    );
}

/// A request to a ZMQ rank for `max_tokens` tokens.
fn zmq_request(request_id: &str, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: request_id.to_string(),
        prompt_token_ids: Some(vec![1, 2, 3, 4]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn sigterm_puts_the_last_batch_of_a_drained_zmq_request_on_the_socket() {
    // Two requests in flight on a ZMQ rank: one the frontend gives up on
    // during the drain (an Abort; the forwarder it cancels is no fault to
    // log), one read to its end. A rank's output writer is what puts a batch
    // on the socket, and the process exits the moment its last rank returns,
    // so the terminal batch of the drained request has to be on the wire by
    // then.
    //
    // The rank's engine runs one request at a time, and the request to drain
    // is queued behind the one to abandon: the rank counts it from its
    // arrival, but it cannot produce a token, let alone finish, until the
    // abandoned request is gone, which is after the signal. So the count at
    // the signal is two however late the signal lands. (A request that has
    // produced output proves nothing about that count: it may have finished
    // by the time the signal arrives.)
    let dir = tempfile::tempdir().expect("a directory for the ipc endpoints");
    let endpoint = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (
        endpoint("hs.sock"),
        endpoint("in.sock"),
        endpoint("out.sock"),
    );
    let mock = Mock::start(&[
        "--engine",
        "realistic",
        "--decode-base-ms",
        DECODE_MS,
        "--max-running",
        "1",
        "--zmq-handshake",
        &handshake,
        "--zmq-count",
        "1",
    ]);
    // The rank dials the frontend: binding the frontend's sockets here lets
    // it register.
    let transport = connect_handshake(&handshake, 1, &input, &output, Duration::from_secs(20))
        .await
        .expect("the ZMQ rank registers with the frontend");
    let client = EngineCoreClient::new(transport);

    let mut abandoned = client
        .submit(zmq_request("abandoned", 400))
        .await
        .expect("submit the request to abandon");
    let mut drained = client
        .submit(zmq_request("drained", 40))
        .await
        .expect("submit the request to drain");
    // Both are registered at the rank once its load, which rides on every
    // output batch, reports the abandoned request running and the drained
    // one waiting behind it.
    tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            abandoned
                .next()
                .await
                .expect("a batch of the running request")
                .expect("ok");
            let load = client
                .engine_load(0)
                .expect("the rank's load rides on its batches");
            if load.num_running == 1 && load.num_waiting == 1 {
                break;
            }
        }
    })
    .await
    .expect("the rank reports one request running and one waiting within 8 s");

    let sent = mock.signal("TERM");
    // The stop is in effect once a port refuses connections: the drain's
    // count is logged before anything else of this test reaches the rank.
    assert_refuses_connections(mock.grpc_port);
    // The frontend gives the long request up during the drain: its Abort
    // reaches the rank, whose forwarder for it is cancelled, and the queued
    // request takes its place in the engine.
    drop(abandoned);

    // The queued request runs to its end, terminal batch included.
    let mut tokens = 0;
    let finish = tokio::time::timeout(Duration::from_secs(8), async {
        while let Some(item) = drained.next().await {
            let output = item.expect("an output of the drained request");
            tokens += output.new_token_ids.len();
            if output.finished() {
                return output.finish_reason;
            }
        }
        None
    })
    .await
    .expect("the drained request ends within 8 s");
    assert_eq!(
        finish,
        Some(EngineCoreFinishReason::Length),
        "the terminal batch of the drained request arrived"
    );
    assert_eq!(tokens, 40, "every token of the drained request arrived");

    let (status, exited, log) = mock.exit(Duration::from_secs(10));
    assert_eq!(status.code(), Some(0), "exit code; log:\n{log}");
    assert!(
        exited - sent < Duration::from_secs(4),
        "the rank returns with its last batch sent, before the deadline: {:?}",
        exited - sent
    );
    assert!(
        log.contains("SIGTERM: draining 2 in-flight request(s)"),
        "both requests were counted; log:\n{log}"
    );
    assert!(log.contains("drained; exiting"), "log:\n{log}");
    assert!(
        !log.contains("forwarder panicked"),
        "a forwarder cancelled by an Abort is not a panic; log:\n{log}"
    );
    assert!(
        !log.contains("ERROR"),
        "nothing is an error here; log:\n{log}"
    );
}
