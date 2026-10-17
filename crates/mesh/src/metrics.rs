//! Mesh cluster metrics for Prometheus
//!
//! Implements all metrics required by issue #10839:
//! - Convergence latency
//! - Traffic metrics (batches, bytes)
//! - Snapshot metrics
//! - Peer health metrics
//! - State integrity metrics
//! - Rate-limit/LB drift metrics

use std::time::Duration;

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};

/// Initialize mesh metrics descriptions
pub fn init_mesh_metrics() {
    // Peer health metrics
    describe_gauge!(
        "router_mesh_peer_connections",
        "Open sync streams accepted from each peer"
    );
    describe_counter!(
        "router_mesh_peer_reconnects_total",
        "Total number of peer reconnections"
    );
    describe_counter!("router_mesh_peer_ack_total", "Total number of ACK messages");
    describe_counter!(
        "router_mesh_peer_nack_total",
        "Total number of NACK messages"
    );

    // State integrity metrics (drift gauges currently retained as scaffolding;
    // recorder helpers below are `#[expect(dead_code)]`).
    describe_gauge!(
        "router_mesh_store_cardinality",
        "Number of entries in each store"
    );
    describe_gauge!(
        "router_mesh_store_hash",
        "Hash of store state for integrity checking"
    );

    // Sync round profiling
    describe_histogram!(
        "router_mesh_sync_round_duration_seconds",
        "Duration of a mesh sync round"
    );

    // Rate-limit and LB drift gauges
    describe_gauge!(
        "router_rl_drift_ratio",
        "Rate-limit drift ratio (actual vs expected)"
    );
    describe_gauge!(
        "router_lb_drift_ratio",
        "Load balance drift ratio (actual vs expected)"
    );
}

/// A sync stream accepted from `peer`, counted in
/// `router_mesh_peer_connections` for as long as this value lives. The gauge
/// is the number of such streams open now, so two that overlap (the dialer
/// redialed before the handler of its old stream reached the idle timeout)
/// add up, and the older one's close leaves the live one counted. The count
/// goes with the drop, so a stream ends counted however its handler ends.
#[must_use = "the stream is counted for as long as this value lives"]
pub struct PeerConnection {
    peer: String,
}

/// Count a sync stream accepted from `peer` as open, until the returned value
/// is dropped.
pub fn peer_connection_opened(peer: &str) -> PeerConnection {
    gauge!("router_mesh_peer_connections",
        "peer" => peer.to_string()
    )
    .increment(1.0);
    PeerConnection {
        peer: peer.to_string(),
    }
}

impl Drop for PeerConnection {
    fn drop(&mut self) {
        gauge!("router_mesh_peer_connections",
            "peer" => self.peer.clone()
        )
        .decrement(1.0);
    }
}

/// Record peer reconnection
pub fn record_peer_reconnect(peer: &str) {
    counter!("router_mesh_peer_reconnects_total",
        "peer" => peer.to_string()
    )
    .increment(1);
}

/// Record ACK
pub fn record_ack(peer: &str, success: bool) {
    let status = if success { "success" } else { "failure" };
    counter!("router_mesh_peer_ack_total",
        "peer" => peer.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
}

/// Record NACK
pub fn record_nack(peer: &str) {
    counter!("router_mesh_peer_nack_total",
        "peer" => peer.to_string()
    )
    .increment(1);
}

#[expect(dead_code)]
/// Update store cardinality
pub fn update_store_cardinality(store: &str, count: usize) {
    gauge!("router_mesh_store_cardinality",
        "store" => store.to_string()
    )
    .set(count as f64);
}

#[expect(dead_code)]
/// Update store hash (for integrity checking)
pub fn update_store_hash(store: &str, hash: u64) {
    gauge!("router_mesh_store_hash",
        "store" => store.to_string()
    )
    .set(hash as f64);
}

#[expect(dead_code)]
/// Update rate-limit drift ratio
pub fn update_rl_drift_ratio(key: &str, ratio: f64) {
    gauge!("router_rl_drift_ratio",
        "key" => key.to_string()
    )
    .set(ratio);
}

#[expect(dead_code)]
/// Update load balance drift ratio
pub fn update_lb_drift_ratio(model: &str, ratio: f64) {
    gauge!("router_lb_drift_ratio",
        "model" => model.to_string()
    )
    .set(ratio);
}

/// Record a mesh sync round's duration
pub fn record_sync_round_duration(peer: &str, duration: Duration) {
    histogram!("router_mesh_sync_round_duration_seconds",
        "peer" => peer.to_string()
    )
    .record(duration.as_secs_f64());
}
