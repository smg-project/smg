//! Prometheus metrics for the RL control plane. Names carry the `smg_rl_`
//! prefix; histograms end in `_duration_seconds` so the gateway's bucket
//! matcher picks them up.

use std::time::Duration;

use metrics::{counter, describe_counter, describe_histogram, histogram};

const CONTROL_CALLS: &str = "smg_rl_control_calls_total";
const CONTROL_CALL_DURATION: &str = "smg_rl_control_call_duration_seconds";
const FANOUT: &str = "smg_rl_fanout_total";
const FANOUT_DURATION: &str = "smg_rl_fanout_duration_seconds";

/// Known engine control operations. Anything else is reported as `other`
/// so a caller cannot mint unbounded label values by inventing paths.
const KNOWN_OPS: &[&str] = &[
    "pause_generation",
    "continue_generation",
    "update_weights_from_disk",
    "update_weights_from_tensor",
    "update_weights_from_distributed",
    "init_weights_update_group",
    "destroy_weights_update_group",
    "update_weight_version",
    "flush_cache",
    "abort_request",
    "release_memory_occupation",
    "resume_memory_occupation",
    "pause",
    "resume",
    "sleep",
    "wake_up",
    "collective_rpc",
    "server_info",
    "get_server_info",
    "health",
];

/// Register HELP/TYPE text. Called once from the gateway's `init_metrics()`.
pub fn init_rl_metrics() {
    describe_counter!(
        CONTROL_CALLS,
        "RL control-plane calls proxied to engines, by operation and result"
    );
    describe_histogram!(
        CONTROL_CALL_DURATION,
        "Latency of one proxied RL control call, by operation"
    );
    describe_counter!(FANOUT, "RL fan-out requests, by result");
    describe_histogram!(FANOUT_DURATION, "Wall time of one RL fan-out request");
}

/// Bounded metric label for an engine path: the whole path when it is a
/// single known operation, else `other`.
pub fn op_label(path: &str) -> &'static str {
    KNOWN_OPS
        .iter()
        .copied()
        .find(|op| *op == path)
        .unwrap_or("other")
}

pub fn record_control_call(op: &'static str, result: &'static str, elapsed: Duration) {
    counter!(CONTROL_CALLS, "op" => op, "result" => result).increment(1);
    histogram!(CONTROL_CALL_DURATION, "op" => op).record(elapsed.as_secs_f64());
}

pub fn record_fanout(result: &'static str, elapsed: Duration) {
    counter!(FANOUT, "result" => result).increment(1);
    histogram!(FANOUT_DURATION).record(elapsed.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_label_is_bounded() {
        assert_eq!(op_label("pause_generation"), "pause_generation");
        assert_eq!(
            op_label("update_weights_from_disk"),
            "update_weights_from_disk"
        );
        assert_eq!(op_label("inference/v1/generate"), "other");
        assert_eq!(op_label("v1/chat/completions/render"), "other");
        assert_eq!(op_label("../etc"), "other");
        assert_eq!(op_label("sleep"), "sleep");
        assert_eq!(op_label("server_info"), "server_info");
        assert_eq!(op_label(""), "other");
    }
}
