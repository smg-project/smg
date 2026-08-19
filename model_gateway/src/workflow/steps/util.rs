//! Shared URL normalization and network probe utilities for worker steps.

use std::time::Duration;

use reqwest::Client;
use smg_grpc_client::connect_channel_with_timeout;

use crate::routers::grpc::client::GrpcClient;

fn strip_scheme<'a>(url: &'a str, scheme: &str) -> Option<&'a str> {
    url.get(..scheme.len())
        .filter(|prefix| prefix.eq_ignore_ascii_case(scheme))
        .map(|_| &url[scheme.len()..])
}

fn url_scheme(url: &str) -> Option<String> {
    url.split_once("://")
        .map(|(scheme, _)| scheme.to_ascii_lowercase())
}

/// Strip protocol prefix (http://, https://, grpc://, grpcs://) from URL.
pub(crate) fn strip_protocol(url: &str) -> String {
    for scheme in ["http://", "https://", "grpc://", "grpcs://"] {
        if let Some(rest) = strip_scheme(url, scheme) {
            return rest.to_string();
        }
    }
    url.to_string()
}

/// Ensure URL has an HTTP(S) scheme — handles bare `host:port` and gRPC inputs.
pub(crate) fn http_base_url(url: &str) -> String {
    if strip_scheme(url, "http://").is_some() || strip_scheme(url, "https://").is_some() {
        url.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", strip_protocol(url).trim_end_matches('/'))
    }
}

/// Ensure URL has a gRPC scheme — handles bare `host:port` and HTTP(S) inputs.
pub(crate) fn grpc_base_url(url: &str) -> String {
    if strip_scheme(url, "grpc://").is_some() || strip_scheme(url, "grpcs://").is_some() {
        url.trim_end_matches('/').to_string()
    } else {
        format!("grpc://{}", strip_protocol(url).trim_end_matches('/'))
    }
}

fn http_health_url(url: &str) -> Result<String, String> {
    match url_scheme(url).as_deref() {
        Some("http") | Some("https") => Ok(format!("{}/health", url.trim_end_matches('/'))),
        Some("grpc") | Some("grpcs") => Err(format!(
            "HTTP health check does not accept gRPC URL scheme: {url}"
        )),
        Some(scheme) => Err(format!(
            "HTTP health check does not accept URL scheme '{scheme}': {url}"
        )),
        None => Ok(format!("http://{}/health", url.trim_end_matches('/'))),
    }
}

fn grpc_reachable_url(url: &str) -> Result<String, String> {
    match url_scheme(url).as_deref() {
        Some("grpc") | Some("grpcs") => Ok(url.trim_end_matches('/').to_string()),
        Some("http") | Some("https") => Err(format!(
            "gRPC health check does not accept HTTP URL scheme: {url}"
        )),
        Some(scheme) => Err(format!(
            "gRPC health check does not accept URL scheme '{scheme}': {url}"
        )),
        None => Ok(format!("grpc://{}", url.trim_end_matches('/'))),
    }
}

/// Try HTTP health check (2xx response required).
pub(crate) async fn try_http_reachable(
    url: &str,
    timeout_secs: u64,
    client: &Client,
) -> Result<(), String> {
    let health_url = http_health_url(url)?;

    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {e}"))?;

    Ok(())
}

/// Perform a single gRPC health check with a specific runtime type.
///
/// Also used by `DetectBackendStep` for runtime identification.
pub(crate) async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {e}"))?;

    Ok(())
}

/// Dial the endpoint once to decide whether it is worth probing per-runtime.
///
/// All five runtime clients dial the same authority, so an endpoint that
/// cannot be reached at the transport level fails five identical connects.
/// One dial answers that question, which matters during a mass worker
/// registration: an unready pod that black-holes SYN would otherwise hold
/// five in-flight sockets per attempt instead of one.
async fn grpc_transport_reachable(grpc_url: &str, timeout_secs: u64) -> Result<(), String> {
    let timeout = Duration::from_secs(timeout_secs);
    let connect_future = connect_channel_with_timeout(grpc_url, timeout);
    tokio::time::timeout(timeout, connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;
    Ok(())
}

/// Check if gRPC is reachable by trying all known runtime types in parallel.
///
/// We don't care which runtime it is here — that's `DetectBackendStep`'s job.
/// We just need to know: does this endpoint speak gRPC at all?
///
/// The per-runtime fan-out is gated behind a single transport probe so an
/// unreachable endpoint costs one connect rather than five.
pub(crate) async fn try_grpc_reachable(url: &str, timeout_secs: u64) -> Result<(), String> {
    let grpc_url = grpc_reachable_url(url)?;

    grpc_transport_reachable(&grpc_url, timeout_secs).await?;

    let (sglang, vllm, trtllm, mlx, tokenspeed) = tokio::join!(
        do_grpc_health_check(&grpc_url, timeout_secs, "sglang"),
        do_grpc_health_check(&grpc_url, timeout_secs, "vllm"),
        do_grpc_health_check(&grpc_url, timeout_secs, "trtllm"),
        do_grpc_health_check(&grpc_url, timeout_secs, "mlx"),
        do_grpc_health_check(&grpc_url, timeout_secs, "tokenspeed"),
    );

    match (sglang, vllm, trtllm, mlx, tokenspeed) {
        (Ok(()), _, _, _, _)
        | (_, Ok(()), _, _, _)
        | (_, _, Ok(()), _, _)
        | (_, _, _, Ok(()), _)
        | (_, _, _, _, Ok(())) => Ok(()),
        (Err(e1), Err(e2), Err(e3), Err(e4), Err(e5)) => Err(format!(
            "gRPC not reachable (tried sglang, vllm, trtllm, mlx, tokenspeed): sglang={e1}, vllm={e2}, trtllm={e3}, mlx={e4}, tokenspeed={e5}",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_health_url_accepts_http_https_and_bare_urls() {
        assert_eq!(
            http_health_url("http://localhost:30000").unwrap(),
            "http://localhost:30000/health"
        );
        assert_eq!(
            http_health_url("https://example.com/").unwrap(),
            "https://example.com/health"
        );
        assert_eq!(
            http_health_url("localhost:30000").unwrap(),
            "http://localhost:30000/health"
        );
    }

    #[test]
    fn http_health_url_rejects_grpc_schemes() {
        assert!(http_health_url("grpc://localhost:30001").is_err());
        assert!(http_health_url("grpcs://localhost:30001").is_err());
    }

    #[test]
    fn grpc_reachable_url_accepts_grpc_grpcs_and_bare_urls() {
        assert_eq!(
            grpc_reachable_url("grpc://localhost:30001").unwrap(),
            "grpc://localhost:30001"
        );
        assert_eq!(
            grpc_reachable_url("grpcs://localhost:30001/").unwrap(),
            "grpcs://localhost:30001"
        );
        assert_eq!(
            grpc_reachable_url("localhost:30001").unwrap(),
            "grpc://localhost:30001"
        );
    }

    #[test]
    fn grpc_reachable_url_rejects_http_schemes() {
        assert!(grpc_reachable_url("http://localhost:30000").is_err());
        assert!(grpc_reachable_url("https://localhost:30000").is_err());
    }

    /// An endpoint that cannot be reached at the transport level must
    /// short-circuit before the per-runtime fan-out, so one unreachable worker
    /// costs one dial rather than five.
    ///
    /// The two failure paths are distinguishable by their error text, and only
    /// the aggregate can be constructed *after* all five `do_grpc_health_check`
    /// calls have run. So asserting the transport error — and the absence of any
    /// runtime name — is what proves no per-runtime probe was started.
    /// `192.0.2.1` is TEST-NET-1 (RFC 5737) and is not routable.
    #[tokio::test]
    async fn unreachable_transport_short_circuits_the_runtime_fanout() {
        let err = try_grpc_reachable("grpc://192.0.2.1:1", 1)
            .await
            .expect_err("unroutable endpoint should not be reachable");

        assert!(
            err.starts_with("gRPC connection"),
            "expected the transport-gate error, got: {err}"
        );
        assert!(
            !err.contains("gRPC not reachable"),
            "fan-out aggregate was produced, so the gate did not short-circuit: {err}"
        );
        for runtime in ["sglang", "vllm", "trtllm", "mlx", "tokenspeed"] {
            assert!(
                !err.contains(runtime),
                "error names {runtime}, so a per-runtime probe ran: {err}"
            );
        }
    }
}
