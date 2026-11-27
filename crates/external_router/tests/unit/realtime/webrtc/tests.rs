use std::{
    net::{IpAddr, Ipv4Addr},
    sync::Arc,
};

use axum::{body::Bytes, http::HeaderMap};

use super::{parse_sdp, setup_and_spawn_bridge, validate_sdp, RealtimeRegistry};
use crate::test_support::{StubWorker, CREDENTIALED_WORKER_URL, MASKED_WORKER_URL};

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

/// The bridge set-up line names the worker masked: a worker registered
/// with credentials at a port nobody listens on cannot take the upstream
/// offer, and neither the set-up line nor the failure line carries the
/// credentials. The model rides as its own field, so the query the
/// upstream URL carries it in is not missed when the address is cut at
/// its path.
#[tracing_test::traced_test]
#[tokio::test]
async fn the_bridge_set_up_line_names_the_worker_masked() {
    let response = setup_and_spawn_bridge(
        &HeaderMap::new(),
        "v=0\r\n",
        "m",
        None,
        "Bearer k",
        StubWorker::at(CREDENTIALED_WORKER_URL),
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        None,
        Arc::new(RealtimeRegistry::new()),
        "direct SDP",
    )
    .await;

    assert!(!logs_contain("secret"));
    assert!(!logs_contain("user:"));
    assert!(response.status().is_server_error(), "{}", response.status());
    assert!(logs_contain("Creating WebRTC bridge"));
    assert!(logs_contain(&format!(
        "{MASKED_WORKER_URL}/v1/realtime/calls"
    )));
}
