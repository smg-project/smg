//! A bounded TCP probe of the rendezvous port a disaggregated leg advertises.
//!
//! A leg registered with a `bootstrap_port` hands that port to every partner
//! it is paired with: the partner's engine connects there to rendezvous
//! before the KV handoff. A leg registered without one (the CLI's
//! `--prefill <url>` with no port, a `POST /workers` without the field, a
//! discovered pod without the bootstrap annotation) is handed the default
//! port on the request path instead, so that is the port probed for it. The
//! gateway itself never connects to the port on the request path, so a port
//! nothing listens on used to be found only by the engines, one request at a
//! time, after their own bootstrap deadline. This probe connects once, after
//! the leg has been registered, off the registration path: the leg serves
//! either way, and a port that does not accept the connection is named in one
//! WARN and counted.

use std::time::Duration;

use openai_protocol::worker::{RuntimeType, WorkerType};
use tokio::net::TcpStream;
use tracing::{debug, warn};

use super::{worker::WorkerTypeExt, Worker, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR};
use crate::observability::metrics::Metrics;

/// How long the probe waits for the connection: long enough for a far host,
/// short enough that a black-holed port is reported promptly.
pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// The `phase` label a failed probe counts under in
/// `smg_pd_rendezvous_failures_total`.
pub(crate) const PROBE_PHASE: &str = "probe";

/// Where the port a probe connects to came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BootstrapPortSource {
    /// The port the leg was registered with: the worker API's
    /// `bootstrap_port`, the CLI's `--prefill <url> <port>`, or the
    /// discovery annotation.
    Registered,
    /// The leg was registered without a port, so the request path hands its
    /// partners [`DEFAULT_BOOTSTRAP_PORT`].
    Default,
}

impl BootstrapPortSource {
    /// The value of the `bootstrap_port_source` log field.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Registered => "registered",
            Self::Default => "default",
        }
    }
}

/// The bootstrap port the gateway hands a partner of `worker`, and where it
/// comes from; `None` when no partner is handed one.
///
/// A registered port is the one, for any worker. Without one, the request
/// path applies [`DEFAULT_BOOTSTRAP_PORT`] to the legs whose partners
/// rendezvous on a bootstrap server: an encode leg, and a prefill leg of every
/// runtime whose rendezvous travels in the request (the SGLang-style
/// bootstrap metadata and the KV bootstrap room), or of vLLM on the Mooncake
/// connector, whose decode-leg `remote_bootstrap_addr` is minted with it. A
/// vLLM prefill leg on another connector (NIXL, MoRI-IO, none) carries its own
/// side channel and is never handed a bootstrap port, and neither is a decode
/// or a regular worker.
pub(crate) fn effective_bootstrap_port(worker: &dyn Worker) -> Option<(u16, BootstrapPortSource)> {
    if let Some(port) = worker.bootstrap_port() {
        return Some((port, BootstrapPortSource::Registered));
    }
    let spec = &worker.metadata().spec;
    let default_applies = match worker.worker_type() {
        WorkerType::Encode => true,
        WorkerType::Prefill => match spec.runtime_type {
            RuntimeType::Vllm => spec.kv_connector.as_deref() == Some(MOONCAKE_CONNECTOR),
            // Listed, as the PD protocol table lists them, so that a runtime
            // added later decides its probe here instead of inheriting one.
            RuntimeType::Sglang
            | RuntimeType::TokenSpeed
            | RuntimeType::Trtllm
            | RuntimeType::Mlx
            | RuntimeType::Generic
            | RuntimeType::External
            | RuntimeType::Unspecified => true,
        },
        WorkerType::Regular | WorkerType::Decode => false,
    };
    default_applies.then_some((DEFAULT_BOOTSTRAP_PORT, BootstrapPortSource::Default))
}

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

/// Probe the bootstrap port the partners of `worker` are handed (see
/// [`effective_bootstrap_port`]), off the registration path. Nothing happens
/// for a worker no partner is handed one for, or for a transport that
/// carries no host.
pub(crate) fn spawn_bootstrap_port_probe(worker: &dyn Worker) {
    let Some((port, source)) = effective_bootstrap_port(worker) else {
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
        report_bootstrap_port_probe(&url, leg, &host, port, source).await;
    });
}

/// Run the probe and report its verdict: a port that did not accept the
/// connection is one WARN naming the worker, the port, where the port came
/// from (`bootstrap_port_source`: `registered`, or `default` for a leg
/// registered without one) and the error, and one count under
/// `smg_pd_rendezvous_failures_total{leg, phase="probe"}`; a port that did is
/// a DEBUG line. Returns whether the port accepted.
pub(crate) async fn report_bootstrap_port_probe(
    url: &str,
    leg: &'static str,
    host: &str,
    port: u16,
    source: BootstrapPortSource,
) -> bool {
    match probe_bootstrap_port(host, port, PROBE_TIMEOUT).await {
        Ok(()) => {
            debug!(
                worker = url,
                worker_type = leg,
                bootstrap_host = host,
                bootstrap_port = port,
                bootstrap_port_source = source.label(),
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
                bootstrap_port_source = source.label(),
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
    use crate::{
        observability::metrics::metrics_labels,
        routers::common::kv_transfer::{kv_connector_mode, KvConnectorMode},
        worker::BasicWorkerBuilder,
    };

    /// A loopback port nothing listens on.
    fn closed_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("address").port();
        drop(listener);
        port
    }

    /// A worker of one type and runtime, with the connector and the
    /// bootstrap port it was registered with.
    fn leg(
        worker_type: WorkerType,
        runtime: RuntimeType,
        kv_connector: Option<&str>,
        bootstrap_port: Option<u16>,
    ) -> impl Worker {
        let mut builder = BasicWorkerBuilder::new("http://prefill-1:8000")
            .worker_type(worker_type)
            .runtime_type(runtime)
            .bootstrap_port(bootstrap_port);
        if let Some(connector) = kv_connector {
            builder = builder.kv_connector(connector);
        }
        builder.build()
    }

    /// The port a leg was registered with is the one probed, whatever the
    /// leg: the operator said so.
    #[test]
    fn a_registered_bootstrap_port_is_the_one_probed_whatever_the_leg() {
        for (worker_type, runtime, connector) in [
            (WorkerType::Prefill, RuntimeType::Sglang, None),
            (
                WorkerType::Prefill,
                RuntimeType::Vllm,
                Some("NixlConnector"),
            ),
            (WorkerType::Encode, RuntimeType::Sglang, None),
            (WorkerType::Decode, RuntimeType::TokenSpeed, None),
            (WorkerType::Regular, RuntimeType::Vllm, None),
        ] {
            let worker = leg(worker_type, runtime, connector, Some(9090));
            assert_eq!(
                effective_bootstrap_port(&worker),
                Some((9090, BootstrapPortSource::Registered)),
                "{worker_type} {runtime} {connector:?}"
            );
        }
    }

    /// A leg registered without a port whose partners are handed the default
    /// on the request path is probed on that default: a prefill leg of a
    /// runtime whose rendezvous travels in the request, a vLLM prefill leg on
    /// the Mooncake connector, an encode leg.
    #[test]
    fn a_leg_without_a_bootstrap_port_is_probed_on_the_default_its_partners_are_handed() {
        for (worker_type, runtime, connector) in [
            (WorkerType::Prefill, RuntimeType::Sglang, None),
            (WorkerType::Prefill, RuntimeType::TokenSpeed, None),
            (WorkerType::Prefill, RuntimeType::Unspecified, None),
            (WorkerType::Prefill, RuntimeType::Generic, None),
            (
                WorkerType::Prefill,
                RuntimeType::Vllm,
                Some(MOONCAKE_CONNECTOR),
            ),
            (WorkerType::Encode, RuntimeType::Sglang, None),
            (WorkerType::Encode, RuntimeType::Vllm, None),
        ] {
            let worker = leg(worker_type, runtime, connector, None);
            assert_eq!(
                effective_bootstrap_port(&worker),
                Some((DEFAULT_BOOTSTRAP_PORT, BootstrapPortSource::Default)),
                "{worker_type} {runtime} {connector:?}"
            );
        }
    }

    /// A leg no partner is handed a bootstrap port for is not probed: a
    /// decode or regular worker, and a vLLM prefill leg whose connector
    /// carries its own side channel.
    #[test]
    fn a_leg_no_partner_is_handed_a_bootstrap_port_for_is_not_probed() {
        for (worker_type, runtime, connector) in [
            (WorkerType::Decode, RuntimeType::Sglang, None),
            (
                WorkerType::Decode,
                RuntimeType::Vllm,
                Some(MOONCAKE_CONNECTOR),
            ),
            (WorkerType::Regular, RuntimeType::Sglang, None),
            (WorkerType::Prefill, RuntimeType::Vllm, None),
            (
                WorkerType::Prefill,
                RuntimeType::Vllm,
                Some("NixlConnector"),
            ),
            (
                WorkerType::Prefill,
                RuntimeType::Vllm,
                Some("MoRIIOConnector"),
            ),
            (
                WorkerType::Prefill,
                RuntimeType::Vllm,
                Some("LMCacheConnector"),
            ),
        ] {
            let worker = leg(worker_type, runtime, connector, None);
            assert_eq!(
                effective_bootstrap_port(&worker),
                None,
                "{worker_type} {runtime} {connector:?}"
            );
        }
    }

    /// For a vLLM prefill leg the probe's decision is the request path's: the
    /// default is probed exactly when the connector mode minted for the leg
    /// hands the decode leg a bootstrap address on that port.
    #[test]
    fn the_default_port_decision_follows_the_connector_mode_the_request_path_mints() {
        for connector in [
            None,
            Some(MOONCAKE_CONNECTOR),
            Some("NixlConnector"),
            Some("MoRIIOConnector"),
            Some("LMCacheConnector"),
        ] {
            let worker = leg(WorkerType::Prefill, RuntimeType::Vllm, connector, None);
            let minted = matches!(
                kv_connector_mode(connector, worker.bootstrap_host(), None, None),
                KvConnectorMode::Mooncake { port, .. } if port == u32::from(DEFAULT_BOOTSTRAP_PORT)
            );
            assert_eq!(
                effective_bootstrap_port(&worker).is_some(),
                minted,
                "{connector:?}"
            );
        }
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
                BootstrapPortSource::Registered,
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

    /// A discovered pod's URL carries an IPv6 literal in brackets, and that
    /// is how the bootstrap host is kept; the probe reaches it as such.
    #[tokio::test]
    async fn a_bootstrap_port_probe_reaches_an_ipv6_host_as_a_worker_url_spells_it() {
        // No IPv6 loopback on this host: nothing to probe.
        let Ok(listener) = TcpListener::bind("[::1]:0") else {
            return;
        };
        let port = listener.local_addr().expect("address").port();
        assert_eq!(
            probe_bootstrap_port("[::1]", port, PROBE_TIMEOUT).await,
            Ok(())
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
                BootstrapPortSource::Default,
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
