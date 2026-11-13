//! The routing flags compiled into a policy specification at start: one
//! deprecation line per flag the command line carried (a flag given at its
//! default included) and one `smg_config_deprecated_flags` series each on
//! `/metrics`; a start without such flags logs no line and serves no series.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::net::SocketAddr;

use crate::gateway_launch::{base_args, free_ports, request, serial, Gateway, IP};

/// The log and the metrics scrape of a gateway started with `flags` beside
/// the common ones, once it serves.
fn start(flags: &[&str], log_name: &str) -> (String, String) {
    let _serial = serial();
    let [port, metrics_port] = free_ports();
    let mut args = base_args(port, metrics_port);
    args.extend(flags.iter().map(|flag| (*flag).to_string()));
    let dir = tempfile::tempdir().expect("a temp dir");
    let gateway = Gateway::start(
        &args,
        SocketAddr::from((IP, port)),
        &dir.path().join(log_name),
    );
    let (status, body) = request(
        SocketAddr::from((IP, metrics_port)),
        "GET",
        "/metrics",
        None,
        None,
    )
    .expect("the metrics listener answers");
    assert_eq!(status, 200, "{body}");
    (gateway.log(), body)
}

#[test]
fn a_routing_flag_the_command_line_carries_logs_its_line_and_serves_its_series() {
    let (log, metrics) = start(
        &[
            "--policy",
            "least_load",
            "--least-load-kv-pressure-weight",
            "0.3",
            "--worker-overload-waiting-requests",
            "8",
        ],
        "flags.log",
    );
    for line in [
        "smg: --policy is deprecated and is removed in ",
        "; it compiled to bindings[0].profile and profiles.least_load = least_load; write it \
         in --policy-spec",
        "smg: --least-load-kv-pressure-weight is deprecated and is removed in ",
        "; it compiled to profiles.least_load.score.least-load.kv_pressure_weight = 0.3; \
         write it in --policy-spec",
        "smg: --worker-overload-waiting-requests is deprecated and is removed in ",
        "; it compiled to bindings[0].overload.waiting_requests = 8; write it in --policy-spec",
    ] {
        assert!(log.contains(line), "missing {line:?} in\n{log}");
    }
    for series in [
        "smg_config_deprecated_flags{flag=\"policy\"} 1",
        "smg_config_deprecated_flags{flag=\"least_load_kv_pressure_weight\"} 1",
        "smg_config_deprecated_flags{flag=\"worker_overload_waiting_requests\"} 1",
    ] {
        assert!(metrics.contains(series), "missing {series} in\n{metrics}");
    }
    assert!(
        metrics.contains("# HELP smg_config_deprecated_flags "),
        "{metrics}"
    );
}

#[test]
fn a_start_without_routing_flags_logs_no_line_and_serves_no_series() {
    let (log, metrics) = start(&[], "no-flags.log");
    assert!(!log.contains("is deprecated and is removed in"), "{log}");
    assert!(
        !metrics.contains("smg_config_deprecated_flags{"),
        "{metrics}"
    );
}
