//! Shared URL normalization and network probe utilities for worker steps.

use std::{future::Future, time::Duration};

use futures::stream::{FuturesUnordered, StreamExt};
use reqwest::Client;
use smg_grpc_client::connect_channel_with_timeout;

use crate::routers::grpc::client::{GrpcClient, HealthCheckResponse};

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

/// One gRPC health RPC to the engine at `grpc_url`, spoken to as
/// `runtime_type`: its answer, healthy or not, or why there was none (the
/// endpoint unreachable, the RPC failed or timed out, another runtime's
/// service there). An answer identifies the runtime whatever it says about
/// the engine; `DetectBackendStep` reads it so.
pub(crate) async fn grpc_health_answer(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<HealthCheckResponse, String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {e}"))
}

/// Perform a single gRPC health check with a specific runtime type: the
/// engine answered on that runtime's service and reports itself healthy.
///
/// The answer counts, not only the RPC: an engine that reports itself not
/// healthy (a servicer that bound its port while its engine still loads the
/// model answers so, as does one that is draining) is not reachable yet, as
/// an HTTP engine whose health route answers an error status is not. The
/// startup detection keeps polling until the engine can serve, instead of
/// admitting a worker whose every request would fail until then.
pub(crate) async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let response = grpc_health_answer(grpc_url, timeout_secs, runtime_type).await?;
    if !response.healthy {
        return Err(not_healthy_error(&response.message));
    }

    Ok(())
}

/// The error for a health answer that says the engine is not healthy, with
/// the engine's own reason when it gives one.
fn not_healthy_error(message: &str) -> String {
    let message = message.trim();
    if message.is_empty() {
        "gRPC health check reports the engine not healthy".to_string()
    } else {
        format!("gRPC health check reports the engine not healthy: {message}")
    }
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

    // tonic's connect timeout is applied to the connector. Keep an outer
    // deadline as a safety net for the remaining Channel::connect setup.
    tokio::time::timeout(timeout, connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;
    Ok(())
}

const GRPC_RUNTIME_TYPES: [&str; 5] = ["sglang", "vllm", "trtllm", "mlx", "tokenspeed"];

async fn first_success_or_all_errors<F>(
    mut checks: FuturesUnordered<F>,
    runtimes: &[&str],
) -> Result<(), String>
where
    F: Future<Output = (usize, Result<(), String>)>,
{
    let mut errors = vec![None; runtimes.len()];

    while let Some((index, result)) = checks.next().await {
        match result {
            Ok(()) => return Ok(()),
            Err(error) => {
                if let Some(slot) = errors.get_mut(index) {
                    *slot = Some(error);
                }
            }
        }
    }

    let details = runtimes
        .iter()
        .enumerate()
        .map(|(index, runtime)| match errors[index].as_deref() {
            Some(error) => format!("{runtime}={error}"),
            None => format!("{runtime}=health check did not complete"),
        })
        .collect::<Vec<_>>()
        .join(", ");

    Err(format!(
        "gRPC not reachable (tried {}): {details}",
        runtimes.join(", ")
    ))
}

/// Check if gRPC is reachable by trying all known runtime types in parallel.
///
/// We don't care which runtime it is here — that's `DetectBackendStep`'s job.
/// We just need to know: does this endpoint speak gRPC at all?
///
/// The per-runtime fan-out is gated behind a single transport probe so an
/// unreachable endpoint costs one connect rather than five.
/// The remaining runtime probes are cancelled after the first success.
/// `timeout_secs` bounds the gate and fan-out together, not each phase.
pub(crate) async fn try_grpc_reachable(url: &str, timeout_secs: u64) -> Result<(), String> {
    let grpc_url = grpc_reachable_url(url)?;
    let timeout = Duration::from_secs(timeout_secs);

    let reachability = Box::pin(async {
        grpc_transport_reachable(&grpc_url, timeout_secs).await?;

        let checks = FuturesUnordered::new();
        for (index, runtime) in GRPC_RUNTIME_TYPES.iter().copied().enumerate() {
            let grpc_url = &grpc_url;
            checks.push(async move {
                (
                    index,
                    do_grpc_health_check(grpc_url, timeout_secs, runtime).await,
                )
            });
        }

        first_success_or_all_errors(checks, &GRPC_RUNTIME_TYPES).await
    });

    tokio::time::timeout(timeout, reachability)
        .await
        .map_err(|_| "gRPC reachability timeout".to_string())?
}

/// A mock gRPC engine on an ephemeral loopback port for the probes' tests,
/// answering its health check as its overrides say (healthy until told
/// otherwise): its `grpc://` URL and those overrides.
#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "test helper: the mock engine's serve task runs for the test process's lifetime"
)]
pub(crate) async fn mock_grpc_engine() -> (
    String,
    std::sync::Arc<mock_worker::overrides::WorkerOverrides>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind the mock engine");
    let port = listener
        .local_addr()
        .expect("the mock engine's address")
        .port();
    let config = std::sync::Arc::new(mock_worker::config::Config {
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        model_id: "probe-model".to_string(),
        tokenizer_path: "probe-model".to_string(),
        ..mock_worker::config::Config::default()
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(config, listener));
    (
        format!("grpc://127.0.0.1:{port}"),
        mock_worker::overrides::for_port(port),
    )
}

#[cfg(test)]
mod tests {
    use std::{
        future::pending,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    use super::*;

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

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

    #[tokio::test]
    async fn runtime_fanout_returns_on_first_success_and_cancels_stalled_checks() {
        let stalled_started = Arc::new(AtomicBool::new(false));
        let stalled_cancelled = Arc::new(AtomicBool::new(false));
        let checks = FuturesUnordered::new();

        for (index, succeeds) in [false, true].into_iter().enumerate() {
            let stalled_started = Arc::clone(&stalled_started);
            let stalled_cancelled = Arc::clone(&stalled_cancelled);
            checks.push(async move {
                let result = if succeeds {
                    tokio::task::yield_now().await;
                    Ok(())
                } else {
                    stalled_started.store(true, Ordering::SeqCst);
                    let _drop_flag = DropFlag(stalled_cancelled);
                    pending::<Result<(), String>>().await
                };
                (index, result)
            });
        }

        let result = tokio::time::timeout(
            Duration::from_secs(1),
            first_success_or_all_errors(checks, &["stalled", "healthy"]),
        )
        .await;

        assert!(matches!(result, Ok(Ok(()))));
        assert!(stalled_started.load(Ordering::SeqCst));
        assert!(stalled_cancelled.load(Ordering::SeqCst));
    }

    /// An endpoint that cannot be reached at the transport level must
    /// short-circuit before the per-runtime fan-out, so one unreachable worker
    /// costs one dial rather than five.
    ///
    /// TCP port 0 cannot be owned by a listener: binding it asks the OS for an
    /// ephemeral nonzero port. This makes the local transport failure
    /// deterministic without a bind-then-drop port-reuse race.
    /// The fan-out aggregate can only be constructed after all runtime probes
    /// run, so the transport error proves the gate returned first.
    #[tokio::test]
    async fn unreachable_transport_short_circuits_the_runtime_fanout() {
        let err = try_grpc_reachable("grpc://127.0.0.1:0", 1)
            .await
            .expect_err("closed local endpoint should not be reachable");

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

    /// An engine that answers its health check but reports itself not healthy
    /// (its model still loading, say) is not reachable yet: the probe fails
    /// with the engine's reason, for the runtime it speaks and in the
    /// fan-out, and passes once the engine reports healthy.
    #[tokio::test]
    async fn an_engine_that_reports_not_healthy_is_not_reachable_yet() {
        let (url, engine) = mock_grpc_engine().await;
        engine.set_serving(false);

        let answer = grpc_health_answer(&url, 5, "tokenspeed")
            .await
            .expect("the runtime answers its health RPC whether or not the engine is healthy");
        assert!(!answer.healthy);
        assert_eq!(answer.message, "not serving");
        grpc_health_answer(&url, 5, "vllm")
            .await
            .expect_err("another runtime's service does not answer there");
        let err = do_grpc_health_check(&url, 5, "tokenspeed")
            .await
            .expect_err("an engine reporting not healthy must not pass the probe");
        assert_eq!(
            err,
            "gRPC health check reports the engine not healthy: not serving"
        );
        let err = try_grpc_reachable(&url, 5)
            .await
            .expect_err("the fan-out must not count a not-healthy answer as reachable");
        assert!(
            err.contains(
                "tokenspeed=gRPC health check reports the engine not healthy: not serving"
            ),
            "the fan-out error does not carry the engine's reason: {err}"
        );

        engine.set_serving(true);
        do_grpc_health_check(&url, 5, "tokenspeed")
            .await
            .expect("a healthy engine passes the probe");
        try_grpc_reachable(&url, 5)
            .await
            .expect("a healthy engine is reachable");
    }

    #[test]
    fn the_not_healthy_error_carries_the_engines_reason_when_it_gives_one() {
        assert_eq!(
            not_healthy_error("Engine is starting"),
            "gRPC health check reports the engine not healthy: Engine is starting"
        );
        assert_eq!(
            not_healthy_error("  "),
            "gRPC health check reports the engine not healthy"
        );
    }
}
