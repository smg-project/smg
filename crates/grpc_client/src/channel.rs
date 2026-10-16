//! Shared `tonic::Channel` builder for SMG gRPC clients.
//!
//! Each engine client (sglang, vllm, trtllm, mlx) connects to its backend
//! the same way: accept either an `http(s)://` or a `grpc(s)://` endpoint,
//! convert the gRPC schemes to tonic-compatible HTTP(S) ones, and build a
//! `Channel` with the same keep-alive / window-size profile. This module
//! centralises that pipeline so adding a new engine — or tuning the
//! transport profile — touches one file instead of four.

use std::time::Duration;

use tonic::transport::{Channel, Endpoint};

/// Convert a `grpc://` or `grpcs://` endpoint to a tonic-compatible
/// `http://` or `https://` URI. Other schemes (or schemeless inputs) are
/// returned unchanged so callers can mix `http(s)://` and `grpc(s)://`
/// freely.
pub fn normalize_grpc_endpoint(endpoint: &str) -> String {
    match endpoint.split_once("://") {
        Some(("grpc", rest)) => format!("http://{rest}"),
        Some(("grpcs", rest)) => format!("https://{rest}"),
        _ => endpoint.to_string(),
    }
}

/// Default ceiling on a single TCP/TLS connect attempt.
///
/// tonic applies no connect timeout of its own, so without this a dial to a
/// black-holing peer (SYN dropped rather than refused — a pod IP whose
/// container is not listening yet) sits in `SYN_SENT` until the kernel gives
/// up: `net.ipv4.tcp_syn_retries` defaults to 6, i.e. ~127s. Bounding the
/// attempt here keeps those sockets from accumulating faster than they drain.
/// Matches the upstream HTTP client's connect timeout in `AppContext`.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// HTTP/2 keepalive: a ping every 30 seconds, answered within 10. The
/// interval is bounded from below by the engines' gRPC servers, not by how
/// fast a dead peer should be noticed: a grpc-core server (vLLM's and
/// SGLang's native gRPC servers) answers a ping that arrives within its
/// `min_ping_interval_without_data` (five minutes by default) of the previous
/// one with a strike, and the second strike is a GOAWAY "too_many_pings"
/// that fails every in-flight stream with `UNAVAILABLE`. A ping every second
/// did exactly that whenever a prefill batch left the connection without
/// data for two seconds. A peer that drops its connection still fails its
/// streams at once; one that falls silent is noticed at the keepalive
/// timeout or at the next poll's deadline, whichever comes first, and the
/// gateway's liveness veto turns the failure into exclusion from routing.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);
const TCP_KEEPALIVE: Duration = Duration::from_secs(60);

/// HTTP/2 receive window per stream (one in-flight RPC).
///
/// These windows only govern what the worker sends back; requests flow under
/// the worker's own settings. The stream window caps how much of a response h2
/// buffers while SMG is not reading it, e.g. while a slow SSE client keeps the
/// bounded body channel full. Beyond that, flow control pauses the worker.
///
/// h2 charges every buffered DATA frame under 256 bytes against a single
/// budget per *connection*, and a streamed token chunk is a few dozen bytes.
/// Once the budget is gone the client sends
/// `GOAWAY ENHANCE_YOUR_CALM too_many_data_frames`, failing every RPC on the
/// connection. A 16 MiB stream window let one unread high-rate stream (an
/// `n=128` request behind a slow client) do that in about 90 seconds.
const STREAM_WINDOW: u32 = 256 * 1024;

/// HTTP/2 receive window per connection.
///
/// Every stream is already capped at [`STREAM_WINDOW`], so this does not bound
/// memory in practice. It sizes h2's small-DATA-frame budget, which is half of
/// this window.
const CONNECTION_WINDOW: u32 = 512 * 1024 * 1024;

// A stream window packed with 1-byte DATA frames costs `STREAM_WINDOW * 255`
// budget units. Require room for four of them, so a few stalled streams
// cannot tear down the connection they share.
const _: () = assert!((STREAM_WINDOW as usize) * 255 * 4 <= (CONNECTION_WINDOW as usize) / 2);

/// Connect a `tonic::Channel` to the given endpoint with the SMG-standard
/// keep-alive and HTTP/2 window profile applied, using
/// [`DEFAULT_CONNECT_TIMEOUT`].
///
/// The endpoint may use any of `http://`, `https://`, `grpc://`, or
/// `grpcs://` — gRPC schemes are normalised to their HTTP(S) equivalents
/// before tonic parses them.
pub async fn connect_channel(
    endpoint: &str,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    connect_channel_with_timeout(endpoint, DEFAULT_CONNECT_TIMEOUT).await
}

/// Same as [`connect_channel`], but with an explicit connect timeout.
///
/// Callers that already bound the dial with their own deadline (health and
/// reachability probes) should pass that deadline through so tonic reaps the
/// socket itself, rather than relying on the outer future being dropped.
pub async fn connect_channel_with_timeout(
    endpoint: &str,
    connect_timeout: Duration,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    let channel = configured_endpoint(endpoint, connect_timeout)?
        .connect()
        .await?;
    Ok(channel)
}

fn configured_endpoint(
    endpoint: &str,
    connect_timeout: Duration,
) -> Result<Endpoint, tonic::transport::Error> {
    let http_endpoint = normalize_grpc_endpoint(endpoint);
    Ok(Endpoint::from_shared(http_endpoint)?
        .connect_timeout(connect_timeout)
        .http2_keep_alive_interval(KEEPALIVE_INTERVAL)
        .keep_alive_timeout(KEEPALIVE_TIMEOUT)
        .keep_alive_while_idle(true)
        .tcp_keepalive(Some(TCP_KEEPALIVE))
        .tcp_nodelay(true)
        .http2_adaptive_window(false)
        .initial_stream_window_size(Some(STREAM_WINDOW))
        .initial_connection_window_size(Some(CONNECTION_WINDOW)))
}

/// Whether `status` is what tonic makes of a failure in the HTTP/2 layer
/// under a call, rather than an answer the worker gave: the connection went
/// away (a GOAWAY of either side, the one this client issues once too many
/// of its locally-reset streams received late frames included), the peer
/// reset the stream below gRPC, a protocol fault. tonic picks such a
/// status's code from the frame's reason (`ENHANCE_YOUR_CALM` reads as
/// `RESOURCE_EXHAUSTED`, the code of a worker's own capacity pushback;
/// `PROTOCOL_ERROR` as `INTERNAL`, the code of its own failure on a
/// request), so the code cannot tell the two apart: the `h2::Error` in the
/// status's source chain can. A status the worker answered with (its
/// trailers) carries no such source.
pub fn is_transport_failure(status: &tonic::Status) -> bool {
    let mut source = std::error::Error::source(status);
    while let Some(error) = source {
        if error.is::<h2::Error>() {
            return true;
        }
        source = error.source();
    }
    false
}

/// For the tests that need a dial to fail.
#[cfg(test)]
pub(crate) mod testing {
    use std::net::{Ipv4Addr, SocketAddr};

    use tokio::net::TcpSocket;

    /// A loopback endpoint whose dial is refused at once on any host: the
    /// port is bound, so nothing else can take it while this value lives, and
    /// not listening, so the host's own stack answers the connect with a
    /// reset. No name is resolved and nothing leaves the host. (A made-up
    /// scheme such as `invalid://endpoint` is no such endpoint: it is dialled
    /// as plain HTTP/2 at port 80 of the host `endpoint`, and the connect
    /// succeeds wherever that name resolves to a listener.)
    pub(crate) struct RefusedEndpoint {
        _reservation: TcpSocket,
        pub(crate) url: String,
    }

    pub(crate) fn refused_endpoint() -> RefusedEndpoint {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let port = socket.local_addr().expect("the bound port").port();
        RefusedEndpoint {
            _reservation: socket,
            url: format!("grpc://127.0.0.1:{port}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::{pending, Pending},
        io,
        task::{Context, Poll},
        time::{Duration, Instant},
    };

    use hyper_util::rt::TokioIo;
    use tokio::io::DuplexStream;
    use tonic::{codegen::http::Uri, transport::Endpoint};
    use tower::Service;

    use super::{
        configured_endpoint, is_transport_failure, normalize_grpc_endpoint,
        DEFAULT_CONNECT_TIMEOUT, STREAM_WINDOW,
    };

    #[derive(Clone, Copy)]
    struct PendingConnector;

    impl Service<Uri> for PendingConnector {
        type Response = TokioIo<DuplexStream>;
        type Error = io::Error;
        type Future = Pending<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _request: Uri) -> Self::Future {
            pending()
        }
    }

    #[test]
    fn default_connect_timeout_is_below_the_kernel_syn_ceiling() {
        // The point of the default is to beat the ~127s kernel SYN timeout
        // (tcp_syn_retries=6); if it ever grows past that it stops doing its job.
        assert!(DEFAULT_CONNECT_TIMEOUT < Duration::from_secs(127));
    }

    /// A connector that never completes must be bounded by the caller's
    /// `connect_timeout`, preventing the kernel SYN retry ceiling described by
    /// [`DEFAULT_CONNECT_TIMEOUT`] from governing a black-holed dial.
    ///
    /// The upper bound has to sit *below* [`DEFAULT_CONNECT_TIMEOUT`], otherwise
    /// an implementation that silently ignored the argument and fell back to the
    /// default would still pass. `PROBE_TIMEOUT` is generous enough to absorb
    /// scheduler variance on a loaded CI box.
    #[tokio::test]
    async fn connect_timeout_bounds_a_pending_connector() {
        const PROBE_TIMEOUT: Duration = Duration::from_millis(300);
        const LOWER_BOUND: Duration = Duration::from_millis(200);
        const UPPER_BOUND: Duration = Duration::from_secs(2);

        // Guards the discriminating power of the assertion below: if the
        // default ever drops to within the bound, this test silently stops
        // distinguishing "argument honored" from "default used".
        assert!(
            UPPER_BOUND < DEFAULT_CONNECT_TIMEOUT,
            "UPPER_BOUND ({UPPER_BOUND:?}) must stay below DEFAULT_CONNECT_TIMEOUT \
             ({DEFAULT_CONNECT_TIMEOUT:?}) for this test to prove the argument is used"
        );

        let start = Instant::now();
        let result = configured_endpoint("grpc://unused.invalid", PROBE_TIMEOUT)
            .expect("build test endpoint")
            .connect_with_connector(PendingConnector)
            .await;
        let elapsed = start.elapsed();

        assert!(result.is_err(), "pending connector should time out");
        assert!(
            elapsed >= LOWER_BOUND,
            "dial failed in {elapsed:?}, before the {PROBE_TIMEOUT:?} timeout; \
             the test did not exercise timeout handling"
        );
        assert!(
            elapsed < UPPER_BOUND,
            "dial took {elapsed:?}, over the {UPPER_BOUND:?} bound; \
             the {PROBE_TIMEOUT:?} connect_timeout argument was not applied"
        );
    }

    #[test]
    fn normalize_grpc_to_http() {
        assert_eq!(
            normalize_grpc_endpoint("grpc://worker:8080"),
            "http://worker:8080"
        );
    }

    #[test]
    fn normalize_grpcs_to_https() {
        assert_eq!(
            normalize_grpc_endpoint("grpcs://worker:8443"),
            "https://worker:8443"
        );
    }

    #[test]
    fn normalize_passes_http_through() {
        assert_eq!(
            normalize_grpc_endpoint("http://worker:8080"),
            "http://worker:8080"
        );
    }

    #[test]
    fn normalize_passes_https_through() {
        assert_eq!(
            normalize_grpc_endpoint("https://worker:8443"),
            "https://worker:8443"
        );
    }

    #[test]
    fn normalize_passes_unknown_scheme_through() {
        // Tonic will reject this, but normalize is not a validator —
        // it only rewrites gRPC schemes.
        assert_eq!(
            normalize_grpc_endpoint("tcp://worker:9000"),
            "tcp://worker:9000"
        );
    }

    #[test]
    fn normalize_passes_schemeless_through() {
        assert_eq!(normalize_grpc_endpoint("worker:8080"), "worker:8080");
    }

    #[test]
    fn normalize_handles_path_after_authority() {
        assert_eq!(
            normalize_grpc_endpoint("grpc://worker:8080/some/path"),
            "http://worker:8080/some/path"
        );
    }

    #[test]
    fn normalize_is_case_sensitive_on_scheme() {
        // Schemes are conventionally lowercase; tonic itself is case
        // sensitive on the URI, so we don't rewrite uppercased gRPC.
        assert_eq!(
            normalize_grpc_endpoint("GRPC://worker:8080"),
            "GRPC://worker:8080"
        );
    }

    // --- HTTP/2 window on the wire ------------------------------------------
    //
    // tonic applies `http2_adaptive_window` after the window setters and hyper
    // resets both windows to 65535 when it is enabled, so what reaches the
    // peer can only be checked on the wire. h2 also sizes its budget for
    // sub-256-byte DATA frames from that window (`max(window / 2, 25_600)`),
    // which is what turns a backlog of streamed token frames into
    // `GOAWAY ENHANCE_YOUR_CALM too_many_data_frames`.

    /// Read `SETTINGS_INITIAL_WINDOW_SIZE` from the client's first frame.
    /// `None` means the client omitted it, which h2 does at the 65535 default.
    async fn advertised_initial_stream_window<F>(build: F) -> Option<u32>
    where
        F: FnOnce(String) -> Endpoint,
    {
        use tokio::io::AsyncReadExt as _;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let endpoint = build(format!("http://{addr}"));

        let read_settings = async {
            let (mut sock, _) = listener.accept().await.expect("accept");
            let mut preface = [0u8; 24];
            sock.read_exact(&mut preface).await.expect("read preface");
            assert_eq!(&preface[..16], b"PRI * HTTP/2.0\r\n");
            let mut header = [0u8; 9];
            sock.read_exact(&mut header)
                .await
                .expect("read frame header");
            assert_eq!(header[3], 0x4, "first frame must be SETTINGS");
            let len = u32::from_be_bytes([0, header[0], header[1], header[2]]) as usize;
            let mut payload = vec![0u8; len];
            sock.read_exact(&mut payload).await.expect("read SETTINGS");
            payload
        };
        // `connect` resolves once the preface is flushed, so join rather
        // than race the two.
        let (payload, _connected) = tokio::join!(read_settings, endpoint.connect());

        let (entries, _) = payload.as_chunks::<6>();
        entries
            .iter()
            .find(|e| u16::from_be_bytes([e[0], e[1]]) == 0x4)
            .map(|e| u32::from_be_bytes([e[2], e[3], e[4], e[5]]))
    }

    #[tokio::test]
    async fn configured_endpoint_advertises_its_stream_window() {
        let advertised = advertised_initial_stream_window(|uri| {
            configured_endpoint(&uri, DEFAULT_CONNECT_TIMEOUT).expect("endpoint")
        })
        .await;
        assert_eq!(advertised, Some(STREAM_WINDOW));
    }

    #[derive(Debug)]
    enum BlastOutcome {
        ClientGoAway(h2::Reason),
        StillOpen,
    }

    /// Have an in-process h2 server push `frames` one-byte DATA frames onto a
    /// stream whose body the client never polls, and report whether the client
    /// tore the connection down.
    async fn blast_unread_data_frames<F>(build: F, frames: usize) -> BlastOutcome
    where
        F: FnOnce(String) -> Endpoint,
    {
        use tonic::codegen::{http, Bytes};

        // Keep the HTTP/2 exchange in memory: closing TCP with unread DATA
        // can surface as a connection reset on macOS instead of the GOAWAY.
        // Buffer the entire blast (including frame headers) so the peer can
        // finish writing before reading the client's GOAWAY.
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        let endpoint = build("http://unused.invalid".to_owned());

        #[expect(clippy::disallowed_methods, reason = "test-only h2 peer, joined below")]
        let server = tokio::spawn(async move {
            let mut conn = h2::server::handshake(server_io)
                .await
                .expect("h2 handshake");
            let (_request, mut respond) = conn
                .accept()
                .await
                .expect("client opened a stream")
                .expect("valid request");
            let response = http::Response::builder()
                .status(200)
                .header("content-type", "application/grpc")
                .body(())
                .expect("response");
            let mut send = respond
                .send_response(response, false)
                .expect("send headers");
            send.reserve_capacity(frames);
            for sent in 0..frames {
                // Every frame must be queued; a short blast would let
                // `StillOpen` below pass vacuously.
                send.send_data(Bytes::from_static(b"x"), false)
                    .unwrap_or_else(|err| panic!("send_data failed after {sent} frames: {err}"));
            }
            match tokio::time::timeout(Duration::from_secs(5), conn.accept()).await {
                Ok(Some(Err(err))) if err.is_go_away() && err.is_remote() => {
                    BlastOutcome::ClientGoAway(err.reason().expect("go_away reason"))
                }
                Ok(Some(Err(err))) => panic!("unexpected h2 error: {err}"),
                Ok(Some(Ok(_))) => panic!("client opened a second stream"),
                Ok(None) => panic!("client closed mid-blast"),
                Err(_elapsed) => BlastOutcome::StillOpen,
            }
        });

        let mut client_io = Some(client_io);
        let mut channel = endpoint
            .connect_with_connector(tower::service_fn(move |_uri: Uri| {
                let io = client_io.take().map(TokioIo::new).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotConnected, "test connection already used")
                });
                std::future::ready(io)
            }))
            .await
            .expect("connect");
        std::future::poll_fn(|cx| channel.poll_ready(cx))
            .await
            .expect("channel ready");
        let request = http::Request::post("/smg.test.Blast/Stream")
            .header("content-type", "application/grpc")
            .header("te", "trailers")
            .body(tonic::body::Body::empty())
            .expect("request");
        let response = channel.call(request).await.expect("response headers");
        let _unread_body = response.into_body();

        server.await.expect("server task")
    }

    /// With adaptive windows the budget is 32_767, so ~130 unread token
    /// frames make the client GOAWAY and kill every stream on the connection.
    #[tokio::test]
    async fn adaptive_window_goaways_on_unread_token_frames() {
        let outcome = blast_unread_data_frames(
            |uri| {
                Endpoint::from_shared(uri)
                    .expect("uri")
                    .http2_adaptive_window(true)
                    .initial_stream_window_size(Some(STREAM_WINDOW))
                    .initial_connection_window_size(Some(2 * STREAM_WINDOW))
            },
            2_000,
        )
        .await;
        assert!(
            matches!(
                outcome,
                BlastOutcome::ClientGoAway(h2::Reason::ENHANCE_YOUR_CALM)
            ),
            "expected GOAWAY ENHANCE_YOUR_CALM, got {outcome:?}"
        );
    }

    /// The previous profile (16 MiB stream / 32 MiB connection window) let a
    /// single unread stream buffer enough token frames to exhaust the 16 MiB
    /// budget (~66k one-byte frames), and the client GOAWAYed every stream on
    /// the connection.
    #[tokio::test]
    async fn oversized_stream_window_goaways_on_unread_token_frames() {
        const MIB: u32 = 1024 * 1024;
        let outcome = blast_unread_data_frames(
            |uri| {
                Endpoint::from_shared(uri)
                    .expect("uri")
                    .http2_adaptive_window(false)
                    .initial_stream_window_size(Some(16 * MIB))
                    .initial_connection_window_size(Some(32 * MIB))
            },
            70_000,
        )
        .await;
        assert!(
            matches!(
                outcome,
                BlastOutcome::ClientGoAway(h2::Reason::ENHANCE_YOUR_CALM)
            ),
            "expected GOAWAY ENHANCE_YOUR_CALM, got {outcome:?}"
        );
    }

    /// Flow control stops the peer once an unread stream holds `STREAM_WINDOW`
    /// bytes, and a full window of 1-byte frames still fits in the budget. The
    /// connection therefore survives no matter how much the peer tries to send.
    #[tokio::test]
    async fn configured_endpoint_survives_unread_token_frames() {
        let outcome = blast_unread_data_frames(
            |uri| configured_endpoint(&uri, DEFAULT_CONNECT_TIMEOUT).expect("endpoint"),
            STREAM_WINDOW as usize + 1_024,
        )
        .await;
        assert!(
            matches!(outcome, BlastOutcome::StillOpen),
            "expected the connection to survive, got {outcome:?}"
        );
    }

    // --- transport faults under a call ---------------------------------------

    /// tonic's status for an HTTP/2 fault carries the h2 error as its source,
    /// whatever code it picked from the frame's reason; a status a worker
    /// answers with, or one made of some other error, carries none.
    #[test]
    fn a_status_made_of_an_http2_fault_is_a_transport_failure() {
        for (reason, code) in [
            (
                h2::Reason::ENHANCE_YOUR_CALM,
                tonic::Code::ResourceExhausted,
            ),
            (h2::Reason::PROTOCOL_ERROR, tonic::Code::Internal),
            (h2::Reason::REFUSED_STREAM, tonic::Code::Unavailable),
            (h2::Reason::CANCEL, tonic::Code::Cancelled),
        ] {
            let status = tonic::Status::from_error(Box::new(h2::Error::from(reason)));
            assert_eq!(status.code(), code, "tonic's own code for {reason:?}");
            assert!(is_transport_failure(&status), "{reason:?}");
        }
        assert!(!is_transport_failure(&tonic::Status::resource_exhausted(
            "busy"
        )));
        assert!(!is_transport_failure(&tonic::Status::internal("boom")));
        let other = tonic::Status::from_error(Box::new(io::Error::other("not http/2")));
        assert_eq!(other.code(), tonic::Code::Unknown);
        assert!(!is_transport_failure(&other));
    }

    // --- late frames on forgotten streams ------------------------------------
    //
    // A stream the client reset (a dropped response body) is remembered for
    // one second, fifty at a time; a frame the peer sends on it after that is
    // a frame on an unknown stream, which h2 answers with a library reset
    // that counts toward the connection's limit of 1024, and past the limit
    // it tears the connection down: GOAWAY ENHANCE_YOUR_CALM
    // "too_many_internal_resets", every call in flight failed. The status
    // tonic makes of that failure is a transport fault.

    /// Serve `io` at the frame level: answer every request's HEADERS with
    /// response headers at once and one DATA frame `late` afterwards, whatever
    /// RST_STREAM the client sent in between; report the client's GOAWAY as
    /// its (error code, debug data).
    async fn serve_late_frames(io: DuplexStream, late: Duration) -> (u32, Vec<u8>) {
        use std::sync::Arc;

        use tokio::{
            io::{AsyncReadExt as _, AsyncWriteExt as _, WriteHalf},
            sync::Mutex,
        };

        async fn write_frame(
            writer: &Mutex<WriteHalf<DuplexStream>>,
            kind: u8,
            flags: u8,
            stream_id: u32,
            payload: &[u8],
        ) {
            let mut frame = Vec::with_capacity(9 + payload.len());
            frame.extend_from_slice(&(payload.len() as u32).to_be_bytes()[1..]);
            frame.push(kind);
            frame.push(flags);
            frame.extend_from_slice(&stream_id.to_be_bytes());
            frame.extend_from_slice(payload);
            let mut writer = writer.lock().await;
            // The client closes its side after its GOAWAY: late writes may fail.
            let _ = writer.write_all(&frame).await;
            let _ = writer.flush().await;
        }

        let (mut reader, writer) = tokio::io::split(io);
        let writer = Arc::new(Mutex::new(writer));
        let mut preface = [0u8; 24];
        reader
            .read_exact(&mut preface)
            .await
            .expect("client preface");
        assert_eq!(&preface[..16], b"PRI * HTTP/2.0\r\n");
        write_frame(&writer, 0x4, 0, 0, &[]).await; // the server's SETTINGS
        loop {
            let mut header = [0u8; 9];
            reader
                .read_exact(&mut header)
                .await
                .expect("a frame header (the client left without a GOAWAY)");
            let len = u32::from_be_bytes([0, header[0], header[1], header[2]]) as usize;
            let (kind, flags) = (header[3], header[4]);
            let stream_id =
                u32::from_be_bytes([header[5], header[6], header[7], header[8]]) & 0x7fff_ffff;
            let mut payload = vec![0u8; len];
            reader
                .read_exact(&mut payload)
                .await
                .expect("a frame payload");
            match kind {
                // SETTINGS and PING are acknowledged; everything else the
                // client sends, RST_STREAM above all, is ignored.
                0x4 if flags & 0x1 == 0 => write_frame(&writer, 0x4, 0x1, 0, &[]).await,
                0x6 if flags & 0x1 == 0 => write_frame(&writer, 0x6, 0x1, 0, &payload).await,
                0x1 => {
                    // `:status: 200` (static table index 8), END_HEADERS.
                    write_frame(&writer, 0x1, 0x4, stream_id, &[0x88]).await;
                    let writer = Arc::clone(&writer);
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "test-only h2 peer; the task ends with its one frame"
                    )]
                    tokio::spawn(async move {
                        tokio::time::sleep(late).await;
                        write_frame(&writer, 0x0, 0, stream_id, b"x").await;
                    });
                }
                0x7 => {
                    let code = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                    return (code, payload[8..].to_vec());
                }
                _ => {}
            }
        }
    }

    /// The limit under the production profile, reached by late frames alone:
    /// the connection goes away with "too_many_internal_resets", and the
    /// status a call in flight fails with is a transport fault (tonic reads
    /// the reason as RESOURCE_EXHAUSTED, which answered as a 429 would tell
    /// the client of a rate limit it never hit).
    #[tokio::test]
    async fn late_frames_on_forgotten_streams_end_the_connection_with_a_transport_failure() {
        use std::pin::Pin;

        use tonic::codegen::{http, Body as _};

        // Past h2's limit of 1024 local resets per connection.
        const ABANDONED: usize = 1_100;
        // Past the one second h2 remembers a reset stream for.
        const LATE: Duration = Duration::from_millis(1_500);

        let (client_io, server_io) = tokio::io::duplex(256 * 1024);
        #[expect(clippy::disallowed_methods, reason = "test-only h2 peer, joined below")]
        let server = tokio::spawn(serve_late_frames(server_io, LATE));

        let mut client_io = Some(client_io);
        let mut channel = configured_endpoint("http://unused.invalid", DEFAULT_CONNECT_TIMEOUT)
            .expect("endpoint")
            .connect_with_connector(tower::service_fn(move |_uri: Uri| {
                let io = client_io.take().map(TokioIo::new).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotConnected, "test connection already used")
                });
                std::future::ready(io)
            }))
            .await
            .expect("connect");
        let request = || {
            http::Request::post("/smg.test.Late/Stream")
                .header("content-type", "application/grpc")
                .header("te", "trailers")
                .body(tonic::body::Body::empty())
                .expect("request")
        };
        // One call stays in flight, to read the connection's end from.
        std::future::poll_fn(|cx| channel.poll_ready(cx))
            .await
            .expect("channel ready");
        let mut probe = channel
            .call(request())
            .await
            .expect("probe headers")
            .into_body();
        for _ in 0..ABANDONED {
            std::future::poll_fn(|cx| channel.poll_ready(cx))
                .await
                .expect("channel ready");
            let response = channel.call(request()).await.expect("response headers");
            // The body goes with it: RST_STREAM CANCEL.
            drop(response);
        }

        let (code, debug) = tokio::time::timeout(Duration::from_secs(20), server)
            .await
            .expect("the client went away within the deadline")
            .expect("server task");
        assert_eq!(h2::Reason::from(code), h2::Reason::ENHANCE_YOUR_CALM);
        assert_eq!(debug, b"too_many_internal_resets");

        // The probe's own late frame arrives before the connection's end.
        let failure = loop {
            match std::future::poll_fn(|cx| Pin::new(&mut probe).poll_frame(cx)).await {
                Some(Ok(_frame)) => continue,
                Some(Err(failure)) => break failure,
                None => panic!("the probe ended cleanly, not with the connection"),
            }
        };
        assert!(is_transport_failure(&failure), "{failure:?}");
        assert_eq!(
            failure.code(),
            tonic::Code::ResourceExhausted,
            "{failure:?}"
        );
    }
}
