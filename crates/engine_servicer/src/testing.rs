//! Helpers the servicer tests share: the configuration they start from and
//! the deadline every wait on the servicer runs under.

use std::{
    future::Future,
    io::{self, Read, Write},
    net::{SocketAddr, TcpStream},
    panic::Location,
    path::Path,
    sync::{Arc, Mutex, OnceLock, PoisonError},
    time::Duration,
};

use engine_zmq_client::mock_engine::MOCK_DEADLINE;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;

use crate::{kv_events::KvEventRelay, ServicerConfig};

/// The servicer configuration every servicer test starts from: an ephemeral
/// port, the engine's data-plane sockets under the test's own directory,
/// one engine, no tokenizer directory, a ten-second silence bound on the
/// start and no ceiling.
pub(crate) fn servicer_config(dir: &Path, handshake: &str) -> ServicerConfig {
    ServicerConfig {
        engine_startup_timeout: Duration::from_secs(10),
        engine_startup_ceiling: None,
        ..ServicerConfig::new(
            "127.0.0.1:0",
            format!("ipc://{}", dir.join("engine").display()),
            handshake,
        )
    }
}

/// A wait on the servicer (a response stream's next message, a watch's next
/// event) under the mock engine's deadline: a servicer that stalls fails the
/// test at the waiting line after thirty seconds instead of hanging the gate.
pub(crate) trait Bounded: Future + Sized {
    #[track_caller]
    fn bounded(self) -> impl Future<Output = Self::Output> {
        let caller = Location::caller();
        async move {
            match timeout(MOCK_DEADLINE, self).await {
                Ok(output) => output,
                Err(_) => panic!("{caller}: nothing from the servicer within {MOCK_DEADLINE:?}"),
            }
        }
    }
}

impl<F: Future> Bounded for F {}

/// `count` publishers bound on consecutive ports from the returned base, as
/// a data-parallel engine lays its ranks' publishers out (rank `r` on the
/// base port plus `r`).
pub(crate) async fn consecutive_publishers(count: u16) -> (u16, Vec<zeromq::PubSocket>) {
    use zeromq::prelude::*;

    for _ in 0..20 {
        let Some(base) = portpicker::pick_unused_port() else {
            continue;
        };
        let Some(last) = base.checked_add(count.saturating_sub(1)) else {
            continue;
        };
        if (base..=last).any(|port| !portpicker::is_free_tcp(port)) {
            continue;
        }
        let mut publishers = Vec::with_capacity(usize::from(count));
        for port in base..=last {
            let mut publisher = zeromq::PubSocket::new();
            if publisher
                .bind(&format!("tcp://127.0.0.1:{port}"))
                .await
                .is_err()
            {
                break;
            }
            publishers.push(publisher);
        }
        if publishers.len() == usize::from(count) {
            return (base, publishers);
        }
    }
    panic!("no run of {count} consecutive free ports");
}

/// One HTTP/1.1 GET against a listener (a servicer's metrics endpoint):
/// the status code and the body. A dependency-free client, so the servicer
/// crate's tests need no HTTP client of their own.
pub(crate) fn http_get(address: SocketAddr, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(address).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("read timeout");
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .expect("request written");
    let mut raw = String::new();
    stream.read_to_string(&mut raw).expect("response read");
    let (head, body) = raw.split_once("\r\n\r\n").expect("a header block");
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .expect("a status code");
    (status, body.to_string())
}

/// A Prometheus exposition reduced to what a golden comparison pins: every
/// `# HELP`, `# TYPE` and sample line but the histogram buckets (pinned by
/// [`assert_bucket_edges`]), sorted (the exporter renders families in hash
/// order), with the values that depend on the clock (`_sum` of a latency
/// histogram, the start time, the scrape duration) replaced by `<v>`; the
/// counters, the gauges and the `_count`s stay literal.
pub(crate) fn golden_exposition(body: &str) -> Vec<String> {
    let timed = |name: &str| {
        name.ends_with("_seconds_sum")
            || name == "smg_servicer_start_time_seconds"
            || name == "smg_servicer_metrics_scrape_seconds"
    };
    let mut lines: Vec<String> = body
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.contains("_bucket{"))
        .map(|line| {
            if line.starts_with('#') {
                return line.to_string();
            }
            let (series, _value) = line.rsplit_once(' ').expect("a sample has a value");
            let name = series.split('{').next().unwrap_or(series);
            if timed(name) {
                format!("{series} <v>")
            } else {
                line.to_string()
            }
        })
        .collect();
    lines.sort();
    lines.dedup();
    lines
}

/// The exposition `body` reduced by [`golden_exposition`] equals `expected`
/// (the golden text, one line each, in any order); the whole actual text is
/// in the failure so a reviewed change can be pasted.
pub(crate) fn assert_golden(body: &str, expected: &str) {
    let actual = golden_exposition(body).join("\n");
    let mut expected: Vec<&str> = expected
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(
        actual,
        expected.join("\n"),
        "the exposition differs; actual:\n{actual}\n"
    );
}

/// Every histogram family of the exposition carries its table's bucket edges
/// (rank 0's series): the engine's own tables for the latencies, the 1-2-5
/// series up to `max_model_len` for the token counts.
pub(crate) fn assert_bucket_edges(body: &str, max_model_len: u32) {
    use crate::metrics::{
        names, one_two_five_buckets, INTER_TOKEN_BUCKETS, PARAMS_N_BUCKETS,
        REQUEST_LATENCY_BUCKETS, TTFT_BUCKETS,
    };

    let tokens = one_two_five_buckets(max_model_len);
    let families: [(&str, &[f64]); 8] = [
        (names::TIME_TO_FIRST_TOKEN_SECONDS, TTFT_BUCKETS),
        (names::E2E_REQUEST_LATENCY_SECONDS, REQUEST_LATENCY_BUCKETS),
        (
            names::REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS,
            INTER_TOKEN_BUCKETS,
        ),
        (names::INTER_TOKEN_LATENCY_SECONDS, INTER_TOKEN_BUCKETS),
        (names::REQUEST_PROMPT_TOKENS, &tokens),
        (names::REQUEST_GENERATION_TOKENS, &tokens),
        (names::REQUEST_PARAMS_MAX_TOKENS, &tokens),
        (names::REQUEST_PARAMS_N, PARAMS_N_BUCKETS),
    ];
    for (name, edges) in families {
        let prefix = format!("{name}_bucket{{");
        let mut seen: Vec<f64> = body
            .lines()
            .filter(|line| line.starts_with(&prefix) && line.contains("rank=\"0\""))
            .filter_map(|line| {
                let (_, rest) = line.split_once("le=\"")?;
                let (le, _) = rest.split_once('"')?;
                le.parse::<f64>().ok()
            })
            .collect();
        seen.retain(|le| le.is_finite());
        seen.sort_by(f64::total_cmp);
        assert_eq!(seen, edges, "the bucket edges of {name}");
    }
}

/// The warnings and errors this test process logs, for the tests that assert
/// a log line. The process-wide subscriber is installed on the first call (a
/// global subscriber is set once per process; the servicers install none of
/// their own, that is the launcher's `init_tracing`), every event from then
/// on lands in one buffer, and a test reads the lines that name its request.
pub(crate) fn captured_logs() -> CapturedLogs {
    static LOGS: OnceLock<CapturedLogs> = OnceLock::new();
    LOGS.get_or_init(|| {
        let logs = CapturedLogs(Arc::new(Mutex::new(Vec::new())));
        let writer = logs.clone();
        let _ = tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::new("warn"))
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .try_init();
        logs
    })
    .clone()
}

#[derive(Clone)]
pub(crate) struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

impl CapturedLogs {
    /// The captured lines that contain `needle`, oldest first.
    pub(crate) fn lines_containing(&self, needle: &str) -> Vec<String> {
        let bytes = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        String::from_utf8_lossy(&bytes)
            .lines()
            .filter(|line| line.contains(needle))
            .map(str::to_string)
            .collect()
    }
}

impl Write for CapturedLogs {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The port a socket bound to `tcp://127.0.0.1:0` was given.
pub(crate) fn bound_port(endpoint: &str) -> u16 {
    endpoint
        .rsplit(':')
        .next()
        .and_then(|port| port.parse().ok())
        .expect("a bound tcp endpoint ends with its port")
}

/// A publisher bound on a free local port for a KV-events relay test, with
/// the `tcp://*:<port>` endpoint a model info names for it.
pub(crate) async fn bound_publisher() -> (zeromq::PubSocket, String) {
    use zeromq::prelude::*;

    let mut publisher = zeromq::PubSocket::new();
    let port = bound_port(
        &publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string(),
    );
    (publisher, format!("tcp://*:{port}"))
}

/// The first two batches of a publisher the relay subscribed to at boot,
/// before any gateway: publishes sequence 0 until the relay, with no
/// subscriber of its own yet, has taken it (the SUB connect is asynchronous;
/// repeats are duplicates), then sequence 1, and reads two relayed.
pub(crate) async fn relay_takes_the_first_two_batches(
    publisher: &mut zeromq::PubSocket,
    relay: &KvEventRelay,
) {
    use zeromq::prelude::*;

    use crate::kv_events::golden;

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
    assert_eq!(
        relay.counts().relayed,
        1,
        "the relay took sequence 0 before any gateway subscribed"
    );
    publisher
        .send(golden::frame(b"kv", 1, &golden::bytes(golden::BATCH2)))
        .await
        .expect("publish");
    for _ in 0..250 {
        if relay.counts().relayed >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(relay.counts().relayed, 2);
}

/// A publisher and a replay socket bound on free local ports for a KV-events
/// relay test, with the two `tcp://*:<port>` endpoints a model info names
/// for them.
pub(crate) async fn publisher_and_replay_socket(
) -> (zeromq::PubSocket, zeromq::RouterSocket, String, String) {
    use zeromq::prelude::*;

    let (publisher, pub_endpoint) = bound_publisher().await;
    let mut router = zeromq::RouterSocket::new();
    let replay_port = bound_port(
        &router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("replay socket binds")
            .to_string(),
    );
    (
        publisher,
        router,
        pub_endpoint,
        format!("tcp://*:{replay_port}"),
    )
}

/// The replay round of a publisher that was already counting when the
/// servicer started. Sequences 0..=2 went out before the subscription landed:
/// publishes sequence 3 until the relay asks the replay socket, which it must
/// do from 0; answers 0..=3 (the first golden batch, then the second three
/// times) and the end marker; then reads the relay's counts: whichever asked
/// first, the start replay or the late join, all four came from the replay
/// socket and nothing is unknown.
pub(crate) async fn replayed_from_the_publishers_start(
    publisher: &mut zeromq::PubSocket,
    router: &mut zeromq::RouterSocket,
    relay: &KvEventRelay,
) {
    use zeromq::{prelude::*, ZmqMessage};

    use crate::kv_events::golden;

    let batch1 = golden::bytes(golden::BATCH1);
    let batch2 = golden::bytes(golden::BATCH2);
    let mut request = None;
    for _ in 0..250 {
        publisher
            .send(golden::frame(b"kv", 3, &batch2))
            .await
            .expect("publish");
        if let Ok(message) = timeout(Duration::from_millis(20), router.recv()).await {
            request = Some(message.expect("a replay request"));
            break;
        }
    }
    let request = request.expect("the relay asked the replay socket");
    let frames: Vec<Vec<u8>> = request.iter().map(|frame| frame.to_vec()).collect();
    assert_eq!(frames.len(), 3, "[identity, empty, start]");
    assert_eq!(
        frames[2],
        0u64.to_be_bytes(),
        "asked from the publisher's start"
    );
    for (sequence, payload) in [(0u64, &batch1), (1, &batch2), (2, &batch2), (3, &batch2)] {
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
    end.push_back([0xff; 8].to_vec().into());
    end.push_back(Vec::new().into());
    router.send(end).await.expect("end marker");
    for _ in 0..250 {
        if relay.counts().relayed >= 4 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let counts = relay.counts();
    assert_eq!(
        (
            counts.relayed,
            counts.gap_batches_recovered + counts.primed_batches,
            counts.unknown_before_start
        ),
        (4, 4, 0),
        "{counts:?}"
    );
}
