//! WebRTC signaling handlers for `/v1/realtime/calls`.
//!
//! SMG acts as a WebRTC relay: it terminates the client's peer connection,
//! establishes its own peer connection to upstream, and bridges data-channel
//! messages plus audio RTP packets between the two.

use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::Arc,
};

use axum::{
    body::Bytes,
    http::{header::CONTENT_TYPE, request::Parts, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::{debug, error, info, warn};

use super::{
    webrtc_bridge::{BridgeSetupError, WebRtcBridge},
    RealtimeLabels, RealtimeRegistry,
};
use crate::{
    error,
    header_utils::extract_auth_header,
    metrics::{metrics_labels, Metrics},
    worker::ExternalWorker,
};

/// The WebRTC bind address when none is configured: the unspecified address
/// of the serving host's family, so a gateway serving on an IPv6 address
/// gathers IPv6 candidates. A `::` socket is dual-stack on Linux and still
/// answers IPv4 peers through their mapped addresses.
pub fn default_bind_addr(serving_host: &str) -> IpAddr {
    let literal = serving_host
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');
    if literal.parse::<Ipv6Addr>().is_ok() {
        IpAddr::V6(Ipv6Addr::UNSPECIFIED)
    } else {
        IpAddr::V4(Ipv4Addr::UNSPECIFIED)
    }
}

/// The STUN server address a socket bound to `bind_addr` can talk to: the
/// first answer of its family; for a `::` socket (dual-stack) an IPv4 answer
/// through its mapped address when there is no IPv6 one.
fn pick_stun_address(
    addrs: impl IntoIterator<Item = SocketAddr>,
    bind_addr: IpAddr,
) -> Option<SocketAddr> {
    let addrs: Vec<SocketAddr> = addrs.into_iter().collect();
    match bind_addr {
        IpAddr::V4(_) => addrs.iter().find(|a| a.is_ipv4()).copied(),
        IpAddr::V6(bind) => addrs.iter().find(|a| a.is_ipv6()).copied().or_else(|| {
            if !bind.is_unspecified() {
                return None;
            }
            addrs.iter().find_map(|a| match a.ip() {
                IpAddr::V4(v4) => Some(SocketAddr::new(IpAddr::V6(v4.to_ipv6_mapped()), a.port())),
                IpAddr::V6(_) => None,
            })
        }),
    }
}

/// Resolve a STUN server hostname to a `SocketAddr` of the bind address's
/// family (see [`pick_stun_address`]).
/// Times out after 3 seconds to avoid blocking bridge setup on slow DNS.
/// Returns `None` if disabled ("none"), resolution fails or no answer fits
/// the family.
async fn resolve_stun_server(server: Option<&str>, bind_addr: IpAddr) -> Option<SocketAddr> {
    use std::time::Duration;

    let host = server?;
    if host.eq_ignore_ascii_case("none") {
        return None;
    }
    match tokio::time::timeout(Duration::from_secs(3), tokio::net::lookup_host(host)).await {
        Ok(Ok(addrs)) => {
            let picked = pick_stun_address(addrs, bind_addr);
            if picked.is_none() {
                tracing::warn!(
                    stun_server = host,
                    %bind_addr,
                    "STUN server has no address of the bind address's family"
                );
            }
            picked
        }
        Ok(Err(e)) => {
            tracing::warn!(stun_server = host, error = %e, "Failed to resolve STUN server");
            None
        }
        Err(_) => {
            tracing::warn!(stun_server = host, "STUN server DNS resolution timed out");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

/// Pre-parsed WebRTC signaling request.
///
/// Created by [`parse_webrtc_request`] so the router can extract the model
/// for worker selection before handing off to the bridge setup.
pub struct WebRtcParsedRequest {
    pub model: String,
    pub sdp: String,
    pub session_config: Option<serde_json::Value>,
}

/// Parse a WebRTC signaling request body.
///
/// Supports two content types:
/// - `multipart/form-data`: Contains `sdp` (SDP offer) and `session` (JSON
///   session config) fields. Model is extracted from `session.model`.
/// - `application/sdp`: Raw SDP offer body. Model comes from `query_model`.
pub async fn parse_webrtc_request(
    parts: &Parts,
    body: &Bytes,
    query_model: &str,
) -> Result<WebRtcParsedRequest, Response> {
    let content_type = parts
        .headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type
        .split(';')
        .next()
        .is_some_and(|mt| mt.trim().eq_ignore_ascii_case("multipart/form-data"))
    {
        parse_multipart(content_type, body).await
    } else if content_type
        .split(';')
        .next()
        .is_some_and(|mt| mt.trim().eq_ignore_ascii_case("application/sdp"))
    {
        parse_sdp(body, query_model)
    } else {
        error!(
            content_type,
            "Unsupported Content-Type for /v1/realtime/calls"
        );
        Err(error::bad_request(
            "invalid_content_type",
            "Expected Content-Type: multipart/form-data or application/sdp",
        ))
    }
}

/// Parse a `multipart/form-data` body into SDP + session config.
async fn parse_multipart(
    content_type: &str,
    body: &Bytes,
) -> Result<WebRtcParsedRequest, Response> {
    let boundary = match multer::parse_boundary(content_type) {
        Ok(b) => b,
        Err(e) => {
            error!(error = %e, "Failed to parse multipart boundary");
            return Err(error::bad_request(
                "invalid_multipart",
                "Missing or invalid multipart boundary",
            ));
        }
    };

    let body_clone = body.clone();
    let mut multipart = multer::Multipart::new(
        futures::stream::once(async move { Ok::<_, std::io::Error>(body_clone) }),
        boundary,
    );

    let mut sdp_offer: Option<Vec<u8>> = None;
    let mut session_json: Option<serde_json::Value> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => {
                return Err(error::bad_request(
                    "malformed_multipart",
                    format!("Failed to read multipart field: {e}"),
                ));
            }
        };
        match field.name() {
            Some("sdp") => match field.bytes().await {
                Ok(b) => sdp_offer = Some(b.to_vec()),
                Err(e) => {
                    return Err(error::bad_request(
                        "unreadable_sdp",
                        format!("Failed to read 'sdp' field: {e}"),
                    ));
                }
            },
            Some("session") => match field.text().await {
                Ok(text) => match serde_json::from_str(&text) {
                    Ok(parsed) => session_json = Some(parsed),
                    Err(e) => {
                        return Err(error::bad_request(
                            "invalid_session_json",
                            format!("Invalid JSON in 'session' field: {e}"),
                        ));
                    }
                },
                Err(e) => {
                    return Err(error::bad_request(
                        "unreadable_session",
                        format!("Failed to read 'session' field: {e}"),
                    ));
                }
            },
            _ => {}
        }
    }

    let Some(sdp_bytes) = sdp_offer else {
        return Err(error::bad_request(
            "missing_sdp",
            "multipart 'sdp' field is required",
        ));
    };

    let sdp = validate_sdp(&sdp_bytes)?;

    // Trim the model in place so the normalized value is used for both
    // local worker selection and the upstream request body.
    if let Some(m) = session_json
        .as_mut()
        .and_then(|s| s.get_mut("model"))
        .and_then(|v| v.as_str().map(|s| s.trim().to_string()))
    {
        if let Some(model_val) = session_json.as_mut().and_then(|s| s.get_mut("model")) {
            *model_val = serde_json::Value::String(m);
        }
    }

    let model = session_json
        .as_ref()
        .and_then(|s| s.get("model"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if model.is_empty() {
        return Err(error::bad_request(
            "missing_model",
            "session.model is required",
        ));
    }

    Ok(WebRtcParsedRequest {
        model,
        sdp,
        session_config: session_json,
    })
}

/// Parse an `application/sdp` body.
fn parse_sdp(body: &Bytes, query_model: &str) -> Result<WebRtcParsedRequest, Response> {
    let query_model = query_model.trim();
    if query_model.is_empty() {
        return Err(error::bad_request(
            "missing_model",
            "query parameter 'model' is required for application/sdp requests",
        ));
    }

    let sdp = validate_sdp(body)?;

    Ok(WebRtcParsedRequest {
        model: query_model.to_string(),
        sdp,
        session_config: None,
    })
}

/// Validate and decode raw bytes as a valid SDP offer.
fn validate_sdp(bytes: &[u8]) -> Result<String, Response> {
    let sdp = std::str::from_utf8(bytes)
        .map_err(|_| error::bad_request("invalid_sdp", "SDP is not valid UTF-8"))?;
    if !sdp.starts_with("v=0") {
        return Err(error::bad_request(
            "invalid_sdp",
            "SDP offer must start with 'v=0'",
        ));
    }
    // Full parse to reject malformed offers before any upstream work.
    str0m::change::SdpOffer::from_sdp_string(sdp)
        .map_err(|e| error::bad_request("invalid_sdp", format!("Malformed SDP offer: {e}")))?;
    Ok(sdp.to_string())
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Handle a pre-parsed realtime WebRTC signaling request.
///
/// The caller (router) is responsible for:
/// 1. Parsing the request via [`parse_webrtc_request`]
/// 2. Selecting a worker using the parsed model
/// 3. Extracting the auth header
#[expect(
    clippy::too_many_arguments,
    reason = "bridge setup requires all params"
)]
pub async fn handle_realtime_webrtc(
    labels: RealtimeLabels,
    headers: HeaderMap,
    parsed: WebRtcParsedRequest,
    worker: Result<Arc<dyn ExternalWorker>, Response>,
    auth_header: Option<HeaderValue>,
    bind_addr: IpAddr,
    stun_server: Option<String>,
    realtime_registry: Arc<RealtimeRegistry>,
) -> Response {
    let worker = match worker {
        Ok(w) => w,
        Err(response) => {
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_WEBRTC,
                &parsed.model,
                metrics_labels::ENDPOINT_REALTIME,
                metrics_labels::ERROR_NO_WORKERS,
            );
            return response;
        }
    };

    // Resolve auth: user-provided or fall back to worker's API key
    let auth_str = match resolve_auth(labels, &parsed.model, auth_header, worker.api_key()) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let label = if parsed.session_config.is_some() {
        "multipart"
    } else {
        "direct SDP"
    };

    setup_and_spawn_bridge(
        &headers,
        &parsed.sdp,
        &parsed.model,
        parsed.session_config,
        &auth_str,
        worker,
        bind_addr,
        stun_server,
        realtime_registry,
        label,
    )
    .await
}

/// Resolve the effective auth string from user header or worker API key.
fn resolve_auth(
    labels: RealtimeLabels,
    model: &str,
    auth_header: Option<HeaderValue>,
    worker_api_key: Option<&String>,
) -> Result<String, Response> {
    let effective_auth = auth_header.or_else(|| extract_auth_header(None, worker_api_key));
    match effective_auth {
        Some(v) => match v.to_str() {
            Ok(s) => Ok(s.to_string()),
            Err(_) => {
                Metrics::record_router_error(
                    labels.router,
                    labels.backend,
                    metrics_labels::CONNECTION_WEBRTC,
                    model,
                    metrics_labels::ENDPOINT_REALTIME,
                    metrics_labels::ERROR_VALIDATION,
                );
                Err((
                    StatusCode::BAD_REQUEST,
                    "Authorization header contains invalid UTF-8 characters",
                )
                    .into_response())
            }
        },
        None => {
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_WEBRTC,
                model,
                metrics_labels::ENDPOINT_REALTIME,
                metrics_labels::ERROR_VALIDATION,
            );
            Err(StatusCode::UNAUTHORIZED.into_response())
        }
    }
}

// ---------------------------------------------------------------------------
// Shared bridge setup
// ---------------------------------------------------------------------------

/// Bridge creation → spawn relay task → return SDP answer.
///
/// Shared by both multipart and direct SDP paths. Worker selection is done
/// by the caller (router layer).
#[expect(
    clippy::too_many_arguments,
    reason = "bridge setup requires all params"
)]
async fn setup_and_spawn_bridge(
    headers: &HeaderMap,
    sdp_str: &str,
    model: &str,
    session_config: Option<serde_json::Value>,
    auth_str: &str,
    worker: Arc<dyn ExternalWorker>,
    bind_addr: IpAddr,
    configured_stun_server: Option<String>,
    realtime_registry: Arc<RealtimeRegistry>,
    label: &str,
) -> Response {
    // Create the load guard now but move it into the spawned bridge task
    // so the worker's load count remains elevated for the bridge lifetime,
    // not just until the SDP answer is returned.
    let load_guard = worker.hold_load(Some(headers));

    let query = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("model", model)
        .finish();
    let upstream_url = format!(
        "{}/v1/realtime/calls?{query}",
        worker.url().trim_end_matches('/')
    );

    let call_id = uuid::Uuid::now_v7().to_string();
    let stun_server = resolve_stun_server(configured_stun_server.as_deref(), bind_addr).await;

    info!(
        call_id,
        model,
        upstream_url,
        ?stun_server,
        "Creating WebRTC bridge ({label})"
    );

    let (mut bridge, client_sdp_answer) = match WebRtcBridge::setup(
        sdp_str,
        &upstream_url,
        auth_str,
        session_config,
        call_id.clone(),
        worker.http_client(),
        bind_addr,
        stun_server,
    )
    .await
    {
        Ok(result) => result,
        Err(e) => {
            return match e {
                BridgeSetupError::UpstreamHttp {
                    status,
                    body,
                    content_type,
                } => {
                    worker.record_outcome(status.as_u16());
                    warn!(call_id, model, %status, "Upstream rejected WebRTC bridge setup ({label})");
                    let mut builder = Response::builder().status(status);
                    if let Some(ct) = content_type {
                        builder = builder.header("Content-Type", ct);
                    }
                    builder
                        .body(axum::body::Body::from(body))
                        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
                }
                BridgeSetupError::Other(ref err) => {
                    error!(call_id, model, error = %err, "Failed to create WebRTC bridge ({label})");
                    StatusCode::BAD_GATEWAY.into_response()
                }
            };
        }
    };

    // -- Register call and spawn bridge task --------------------------------
    let entry = realtime_registry.register_call(
        call_id.clone(),
        model.to_string(),
        worker.url().to_string(),
    );
    // Use the registry's cancel token so hangup cancellation reaches the bridge.
    bridge.set_cancel_token(entry.cancel_token);

    let bridge_registry = Arc::clone(&realtime_registry);
    let bridge_call_id = call_id.clone();
    #[expect(
        clippy::disallowed_methods,
        reason = "bridge task self-terminates on disconnect/cancel"
    )]
    tokio::spawn(async move {
        let _guard = load_guard; // keep worker load elevated until bridge ends
        let success = Box::pin(bridge.run(bridge_registry.clone())).await;
        worker.record_outcome(if success { 200 } else { 502 });
        bridge_registry.remove_call(&bridge_call_id);
        debug!(
            call_id = bridge_call_id,
            success, "WebRTC bridge task completed"
        );
    });

    debug!(call_id, model, "WebRTC bridge started ({label})");

    // -- Return SMG-generated SDP answer ------------------------------------
    #[expect(
        clippy::expect_used,
        reason = "infallible: static header names and valid body"
    )]
    Response::builder()
        .status(StatusCode::CREATED)
        .header("Content-Type", "application/sdp")
        .body(axum::body::Body::from(client_sdp_answer))
        .expect("static response builder")
}

#[cfg(test)]
mod tests {
    use axum::body::Bytes;

    use super::{parse_sdp, validate_sdp};

    #[test]
    fn validate_sdp_rejects_non_v0() {
        assert!(validate_sdp(b"not-an-sdp-offer").is_err());
    }

    #[test]
    fn validate_sdp_rejects_empty() {
        assert!(validate_sdp(b"").is_err());
    }

    #[test]
    fn parse_sdp_requires_model() {
        // Empty/whitespace query model is rejected before SDP validation.
        let body = Bytes::from_static(b"v=0\r\n");
        assert!(parse_sdp(&body, "   ").is_err());
    }
}

#[cfg(test)]
mod family_tests {
    use super::*;

    fn v4(last: u8) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(198, 51, 100, last)), 3478)
    }

    fn v6(last: u16) -> SocketAddr {
        SocketAddr::new(
            IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, last)),
            3478,
        )
    }

    #[test]
    fn an_ipv4_bind_takes_the_first_ipv4_answer() {
        assert_eq!(
            pick_stun_address([v6(1), v4(1), v4(2)], IpAddr::V4(Ipv4Addr::UNSPECIFIED)),
            Some(v4(1))
        );
        assert_eq!(
            pick_stun_address([v6(1)], IpAddr::V4(Ipv4Addr::UNSPECIFIED)),
            None
        );
    }

    #[test]
    fn an_ipv6_bind_takes_the_first_ipv6_answer() {
        let bind = IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 7));
        assert_eq!(pick_stun_address([v4(1), v6(1), v6(2)], bind), Some(v6(1)));
        assert_eq!(
            pick_stun_address([v4(1)], bind),
            None,
            "a socket bound to one IPv6 address cannot reach an IPv4 server"
        );
    }

    #[test]
    fn a_dual_stack_bind_falls_back_to_a_mapped_ipv4_answer() {
        let bind = IpAddr::V6(Ipv6Addr::UNSPECIFIED);
        assert_eq!(pick_stun_address([v4(1), v6(1)], bind), Some(v6(1)));
        assert_eq!(
            pick_stun_address([v4(1)], bind),
            Some(SocketAddr::new(
                IpAddr::V6(Ipv4Addr::new(198, 51, 100, 1).to_ipv6_mapped()),
                3478
            ))
        );
    }

    #[test]
    fn the_default_bind_follows_the_serving_hosts_family() {
        for host in ["0.0.0.0", "127.0.0.1", "10.0.0.7", "localhost", ""] {
            assert_eq!(
                default_bind_addr(host),
                IpAddr::V4(Ipv4Addr::UNSPECIFIED),
                "{host}"
            );
        }
        for host in ["::", "[::]", "::1", "[fd00::7]", " :: "] {
            assert_eq!(
                default_bind_addr(host),
                IpAddr::V6(Ipv6Addr::UNSPECIFIED),
                "{host}"
            );
        }
    }
}
