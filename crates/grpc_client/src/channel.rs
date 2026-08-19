//! Shared `tonic::Channel` builder for SMG gRPC clients.
//!
//! Each engine client (sglang, vllm, trtllm, mlx) connects to its backend
//! the same way: accept either an `http(s)://` or a `grpc(s)://` endpoint,
//! convert the gRPC schemes to tonic-compatible HTTP(S) ones, and build a
//! `Channel` with the same keep-alive / window-size profile. This module
//! centralises that pipeline so adding a new engine — or tuning the
//! transport profile — touches one file instead of four.

use std::time::Duration;

use tonic::transport::Channel;

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
    let http_endpoint = normalize_grpc_endpoint(endpoint);
    let channel = Channel::from_shared(http_endpoint)?
        .connect_timeout(connect_timeout)
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(10))
        .keep_alive_while_idle(true)
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .tcp_nodelay(true)
        .http2_adaptive_window(true)
        // 16MB stream window, 32MB connection window — sized for the
        // typical inference response (multi-MB tokenized payloads +
        // streaming chunks) without head-of-line blocking.
        .initial_stream_window_size(Some(16 * 1024 * 1024))
        .initial_connection_window_size(Some(32 * 1024 * 1024))
        .connect()
        .await?;
    Ok(channel)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{connect_channel_with_timeout, normalize_grpc_endpoint, DEFAULT_CONNECT_TIMEOUT};

    #[test]
    fn default_connect_timeout_is_below_the_kernel_syn_ceiling() {
        // The point of the default is to beat the ~127s kernel SYN timeout
        // (tcp_syn_retries=6); if it ever grows past that it stops doing its job.
        assert!(DEFAULT_CONNECT_TIMEOUT < Duration::from_secs(127));
    }

    /// A dial to a black-holing peer must be bounded by the caller's
    /// `connect_timeout` rather than by the kernel's SYN retry ceiling.
    /// `192.0.2.1` is TEST-NET-1 (RFC 5737) and is not routable, so the connect
    /// either fails fast with an unreachable error or is cut off by our
    /// timeout — never hangs for the ~127s a retrying SYN would otherwise take.
    ///
    /// The upper bound has to sit *below* [`DEFAULT_CONNECT_TIMEOUT`], otherwise
    /// an implementation that silently ignored the argument and fell back to the
    /// default would still pass. `PROBE_TIMEOUT` is generous enough to absorb
    /// scheduler variance on a loaded CI box.
    #[tokio::test]
    async fn connect_timeout_bounds_a_blackholed_dial() {
        const PROBE_TIMEOUT: Duration = Duration::from_millis(300);
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
        let result = connect_channel_with_timeout("grpc://192.0.2.1:1", PROBE_TIMEOUT).await;
        let elapsed = start.elapsed();

        assert!(result.is_err(), "unroutable dial should not succeed");
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
}
