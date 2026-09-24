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
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(10))
        .keep_alive_while_idle(true)
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .tcp_nodelay(true)
        .http2_adaptive_window(false)
        // 16MB stream window, 32MB connection window — sized for the
        // typical inference response (multi-MB tokenized payloads +
        // streaming chunks) without head-of-line blocking.
        .initial_stream_window_size(Some(16 * 1024 * 1024))
        .initial_connection_window_size(Some(32 * 1024 * 1024)))
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

    use super::{configured_endpoint, normalize_grpc_endpoint, DEFAULT_CONNECT_TIMEOUT};

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

    const STREAM_WINDOW: u32 = 16 * 1024 * 1024;

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

    #[tokio::test]
    async fn configured_endpoint_survives_unread_token_frames() {
        let outcome = blast_unread_data_frames(
            |uri| configured_endpoint(&uri, DEFAULT_CONNECT_TIMEOUT).expect("endpoint"),
            2_000,
        )
        .await;
        assert!(
            matches!(outcome, BlastOutcome::StillOpen),
            "expected the connection to survive, got {outcome:?}"
        );
    }
}
