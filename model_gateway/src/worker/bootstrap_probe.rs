//! A bounded TCP probe of the rendezvous port a disaggregated leg advertises.
//!
//! A leg registered with a `bootstrap_port` hands that port to every partner
//! it is paired with: the partner's engine connects there to rendezvous
//! before the KV handoff. The gateway itself never connects to it on the
//! request path, so a port nothing listens on used to be found only by the
//! engines, one request at a time, after their own bootstrap deadline. This
//! probe connects once, after the leg has been registered, off the
//! registration path: the leg serves either way, and a port that does not
//! accept the connection is named in one WARN and counted.

use std::time::Duration;

use tokio::net::TcpStream;
use tracing::{debug, warn};

use super::{worker::WorkerTypeExt, Worker};
use crate::observability::metrics::Metrics;

/// How long the probe waits for the connection: long enough for a far host,
/// short enough that a black-holed port is reported promptly.
pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// The `phase` label a failed probe counts under in
/// `smg_pd_rendezvous_failures_total`.
pub(crate) const PROBE_PHASE: &str = "probe";

/// Connect to `host:port` once, within `timeout`. `Ok` when the port accepted
/// the connection; otherwise why it did not, in words for a log line.
pub(crate) async fn probe_bootstrap_port(
    host: &str,
    port: u16,
    timeout: Duration,
) -> Result<(), String> {
    let address = format!("{host}:{port}");
    match tokio::time::timeout(timeout, TcpStream::connect(address.as_str())).await {
        Ok(Ok(_connected)) => Ok(()),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_elapsed) => Err(format!("no answer within {timeout:?}")),
    }
}

/// Probe the bootstrap port `worker` was registered with, off the
/// registration path. Nothing happens for a worker without one, or for a
/// transport that carries no host.
pub(crate) fn spawn_bootstrap_port_probe(worker: &dyn Worker) {
    let Some(port) = worker.bootstrap_port() else {
        return;
    };
    let host = worker.bootstrap_host();
    if host.is_empty() {
        return;
    }
    let host = host.to_string();
    let url = worker.url().to_string();
    let leg = worker.worker_type().as_metric_label();
    #[expect(
        clippy::disallowed_methods,
        reason = "an advisory probe that ends within its own timeout; nothing waits on it at shutdown"
    )]
    tokio::spawn(async move {
        report_bootstrap_port_probe(&url, leg, &host, port).await;
    });
}

/// Run the probe and report its verdict: a port that did not accept the
/// connection is one WARN naming the worker, the port and the error, and one
/// count under `smg_pd_rendezvous_failures_total{leg, phase="probe"}`; a port
/// that did is a DEBUG line. Returns whether the port accepted.
pub(crate) async fn report_bootstrap_port_probe(
    url: &str,
    leg: &'static str,
    host: &str,
    port: u16,
) -> bool {
    match probe_bootstrap_port(host, port, PROBE_TIMEOUT).await {
        Ok(()) => {
            debug!(
                worker = url,
                worker_type = leg,
                bootstrap_host = host,
                bootstrap_port = port,
                "Bootstrap port accepted a connection at registration"
            );
            true
        }
        Err(error) => {
            warn!(
                worker = url,
                worker_type = leg,
                bootstrap_host = host,
                bootstrap_port = port,
                error = %error,
                "Bootstrap port did not accept a connection at registration: the leg is \
                 registered, but its disaggregated partners cannot rendezvous there"
            );
            Metrics::record_pd_rendezvous_failure(leg, PROBE_PHASE);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use metrics_exporter_prometheus::PrometheusBuilder;

    use super::*;
    use crate::observability::metrics::metrics_labels;

    /// A loopback port nothing listens on.
    fn closed_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("address").port();
        drop(listener);
        port
    }

    #[tokio::test]
    async fn a_bootstrap_port_probe_passes_a_listening_port() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("address").port();
        assert_eq!(
            probe_bootstrap_port("127.0.0.1", port, PROBE_TIMEOUT).await,
            Ok(())
        );
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let accepted = {
            let _guard = metrics::set_default_local_recorder(&recorder);
            report_bootstrap_port_probe(
                "http://127.0.0.1:8000",
                metrics_labels::WORKER_PREFILL,
                "127.0.0.1",
                port,
            )
            .await
        };
        assert!(accepted);
        assert!(
            !handle.render().contains("smg_pd_rendezvous_failures_total"),
            "{}",
            handle.render()
        );
    }

    #[tokio::test]
    async fn a_bootstrap_port_probe_fails_a_closed_port_and_counts_it() {
        let port = closed_port();
        let error = probe_bootstrap_port("127.0.0.1", port, PROBE_TIMEOUT)
            .await
            .expect_err("nothing listens there");
        assert!(!error.is_empty());

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let accepted = {
            let _guard = metrics::set_default_local_recorder(&recorder);
            report_bootstrap_port_probe(
                "http://127.0.0.1:8000",
                metrics_labels::WORKER_PREFILL,
                "127.0.0.1",
                port,
            )
            .await
        };
        assert!(!accepted);
        let rendered = handle.render();
        let line = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
            .unwrap_or_else(|| panic!("the failed probe is counted:\n{rendered}"));
        assert!(
            line.contains(r#"leg="prefill""#)
                && line.contains(r#"phase="probe""#)
                && line.ends_with(" 1"),
            "{line}"
        );
    }
}
