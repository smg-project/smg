//! How a registration reaches a worker before it knows what the worker is:
//! the addresses its probes ask, the HTTP probe and the gRPC reach.

use std::{future::Future, time::Duration};

use futures::stream::{FuturesUnordered, StreamExt};
use reqwest::Client;

use super::runtimes::{listed, GRPC_RUNTIMES};
use crate::{
    connection::engine::{EngineConnector, HealthAnswer},
    identity::{
        basic::RuntimeType,
        endpoint::{strip_protocol, strip_scheme, Endpoint},
    },
};

/// The scheme `url` starts with, lowercased: a word of scheme characters
/// (RFC 3986's, a digit or a sign first too, and the empty word) before a
/// `://`, as every scheme the address type knows is.
/// An address that starts with no such word has no scheme, as the address
/// type reads it. A `://` further in belongs to its path or its query, and
/// what stands before it may hold a userinfo, which a refusal would name.
fn url_scheme(url: &str) -> Option<String> {
    let (word, _) = url.split_once("://")?;
    word.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        .then(|| word.to_ascii_lowercase())
}

/// Ensure URL has an HTTP(S) scheme — handles bare `host:port` and gRPC inputs.
pub fn http_base_url(url: &str) -> String {
    if strip_scheme(url, "http://").is_some() || strip_scheme(url, "https://").is_some() {
        url.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", strip_protocol(url).trim_end_matches('/'))
    }
}

/// The URL of the worker's HTTP health route. Keep the worker's HTTP or
/// HTTPS scheme and use HTTP for an address without a scheme. Refuse any
/// other scheme and mask credentials in the error.
pub fn http_health_url(url: &str) -> Result<String, String> {
    match url_scheme(url).as_deref() {
        Some("http") | Some("https") => Ok(format!("{}/health", url.trim_end_matches('/'))),
        // The refusals name the address masked: on the detection step's
        // auto-detect arm (a URL whose scheme names no mode) a scheme neither
        // probe accepts ends in the step's failure message, which a log line
        // carries.
        Some("grpc") | Some("grpcs") => Err(format!(
            "HTTP health check does not accept gRPC URL scheme: {}",
            Endpoint::redacted_str(url)
        )),
        Some(scheme) => Err(format!(
            "HTTP health check does not accept URL scheme '{scheme}': {}",
            Endpoint::redacted_str(url)
        )),
        None => Ok(format!("http://{}/health", url.trim_end_matches('/'))),
    }
}

/// The address the gRPC reach dials for `url`: `url` itself under a gRPC
/// scheme, or under `grpc://` for an address with none. A scheme the reach
/// does not take is refused, the address masked.
pub fn grpc_reachable_url(url: &str) -> Result<String, String> {
    match url_scheme(url).as_deref() {
        Some("grpc") | Some("grpcs") => Ok(url.trim_end_matches('/').to_string()),
        Some("http") | Some("https") => Err(format!(
            "gRPC health check does not accept HTTP URL scheme: {}",
            Endpoint::redacted_str(url)
        )),
        Some(scheme) => Err(format!(
            "gRPC health check does not accept URL scheme '{scheme}': {}",
            Endpoint::redacted_str(url)
        )),
        None => Ok(format!("grpc://{}", url.trim_end_matches('/'))),
    }
}

/// Try HTTP health check (2xx response required).
pub async fn try_http_reachable(
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
        .map_err(|e| format!("Health check failed: {}", e.without_url()))?;

    Ok(())
}

/// One gRPC health RPC to the engine at `grpc_url`, spoken to as `runtime`
/// through `connector`: its answer, healthy or not, or why there was none
/// (the endpoint unreachable, the RPC failed or timed out, another runtime's
/// service there). An answer identifies the runtime whatever it says about
/// the engine; the runtime detection reads it so.
pub async fn grpc_health_answer(
    connector: &dyn EngineConnector,
    grpc_url: &str,
    timeout_secs: u64,
    runtime: RuntimeType,
) -> Result<HealthAnswer, String> {
    let connect_future = connector.connect_grpc(grpc_url, runtime);
    let connection = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;

    let health_future = connection.health_answer();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {e}"))
}

/// Perform a single gRPC health check with a specific runtime: the
/// engine answered on that runtime's service and reports itself healthy.
///
/// The answer counts, not only the RPC: an engine that reports itself not
/// healthy (a servicer that bound its port while its engine still loads the
/// model answers so, as does one that is draining) is not reachable yet, as
/// an HTTP engine whose health route answers an error status is not. The
/// startup detection keeps polling until the engine can serve, instead of
/// admitting a worker whose every request would fail until then.
async fn do_grpc_health_check(
    connector: &dyn EngineConnector,
    grpc_url: &str,
    timeout_secs: u64,
    runtime: RuntimeType,
) -> Result<(), String> {
    let response = grpc_health_answer(connector, grpc_url, timeout_secs, runtime).await?;
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
/// Every runtime's client dials the same authority, so an endpoint that
/// cannot be reached at the transport level fails one identical connect a
/// runtime. One dial answers that question, which matters during a mass
/// worker registration: an unready worker that black-holes SYN would
/// otherwise hold one in-flight socket a runtime per attempt instead of one.
async fn grpc_transport_reachable(
    connector: &dyn EngineConnector,
    grpc_url: &str,
    timeout_secs: u64,
) -> Result<(), String> {
    let timeout = Duration::from_secs(timeout_secs);
    let connect_future = connector.dial_grpc(grpc_url, timeout);

    // The dial bounds its own connect with the timeout. Keep an outer
    // deadline as a safety net for the rest of the channel's setup.
    tokio::time::timeout(timeout, connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;
    Ok(())
}

async fn first_success_or_all_errors<F>(
    mut checks: FuturesUnordered<F>,
    runtimes: &[RuntimeType],
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
        listed(runtimes)
    ))
}

/// Check if gRPC is reachable by trying all known runtimes in parallel,
/// through `connector`.
///
/// Which runtime answers is the runtime detection's question. This one is:
/// does this endpoint speak gRPC at all?
///
/// The per-runtime fan-out is gated behind a single transport probe so an
/// unreachable endpoint costs one connect rather than one a runtime.
/// The remaining runtime probes are cancelled after the first success.
/// `timeout_secs` bounds the gate and fan-out together, not each phase.
pub async fn try_grpc_reachable(
    connector: &dyn EngineConnector,
    url: &str,
    timeout_secs: u64,
) -> Result<(), String> {
    let grpc_url = grpc_reachable_url(url)?;
    let timeout = Duration::from_secs(timeout_secs);

    let reachability = Box::pin(async {
        grpc_transport_reachable(connector, &grpc_url, timeout_secs).await?;

        let checks = FuturesUnordered::new();
        for (index, runtime) in GRPC_RUNTIMES.iter().copied().enumerate() {
            let grpc_url = &grpc_url;
            checks.push(async move {
                (
                    index,
                    do_grpc_health_check(connector, grpc_url, timeout_secs, runtime).await,
                )
            });
        }

        first_success_or_all_errors(checks, &GRPC_RUNTIMES).await
    });

    tokio::time::timeout(timeout, reachability)
        .await
        .map_err(|_| "gRPC reachability timeout".to_string())?
}

#[cfg(test)]
mod tests;
