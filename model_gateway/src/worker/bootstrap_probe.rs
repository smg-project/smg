//! A bounded TCP probe of the rendezvous port a disaggregated worker advertises.
//!
//! A worker registered with a `bootstrap_port` hands that port to every partner
//! it is paired with: the partner's engine connects there to rendezvous
//! before the KV handoff. A worker registered without one (the CLI's
//! `--prefill <url>` with no port, a `POST /workers` without the field, a
//! discovered pod without the bootstrap annotation) is handed the default
//! port on the request path instead, so that is the port probed for it. The
//! gateway itself never connects to the port on the request path, so a port
//! nothing listens on used to be found only by the engines, one request at a
//! time, after their own bootstrap deadline. This probe connects once, after
//! the worker has been registered, off the registration path: the worker serves
//! either way, and a port that does not accept the connection is named in one
//! WARN and counted.

use std::time::Duration;

use openai_protocol::worker::{RuntimeType, WorkerType};
use tokio::net::TcpStream;
use tracing::{debug, info, warn};

use super::{
    basic::WorkerTypeExt, pd_pairing::KvTransferProfile, Worker, DEFAULT_BOOTSTRAP_PORT,
    MOONCAKE_CONNECTOR,
};
use crate::{observability::metrics::Metrics, worker::endpoint::Endpoint};

/// How long the probe waits for the connection: long enough for a far host,
/// short enough that a black-holed port is reported promptly.
pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// The `phase` label a failed probe counts under in
/// `smg_pd_rendezvous_failures_total`.
pub(crate) const PROBE_PHASE: &str = "probe";

/// The `phase` label the same probe counts under when the request path runs
/// it: a decode side's engine failed the request without a reason, and the
/// port that side had to reach, its prefill partner's, did not accept a
/// connection (see `routers::grpc::utils::pd_bootstrap`).
pub(crate) const REQUEST_PHASE: &str = "request";

/// A bootstrap port that did not accept the probe's connection: the port,
/// where it came from, and why, for the line and the message that name it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeadBootstrapPort {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) source: BootstrapPortSource,
    pub(crate) error: String,
}

/// Where the port a probe connects to came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BootstrapPortSource {
    /// The port the worker was registered with: the worker API's
    /// `bootstrap_port`, the CLI's `--prefill <url> <port>`, or the
    /// discovery annotation.
    Registered,
    /// The worker was registered without a port, so the request path hands its
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

/// A prefill worker whose connector writes to a shared store: its partners read
/// its blocks back from the store and are handed no port, whatever port the
/// worker was registered with.
fn is_store_prefill_worker(worker: &dyn Worker) -> bool {
    matches!(worker.worker_type(), WorkerType::Prefill)
        && worker.pd_pairing().profile() == Some(KvTransferProfile::Store)
}

/// The bootstrap port the gateway hands a partner of `worker`, and where it
/// comes from; `None` when no partner is handed one.
///
/// A registered port is the one, for any worker but a prefill worker on a KV
/// store (see [`is_store_prefill_worker`]). Without one, the request path
/// applies [`DEFAULT_BOOTSTRAP_PORT`] to the workers whose partners rendezvous
/// on a bootstrap server: an encode worker, and a prefill worker of every runtime
/// whose rendezvous travels in the request (the SGLang-style bootstrap
/// metadata and the KV bootstrap room), or of vLLM on the Mooncake connector,
/// whose decode-side `remote_bootstrap_addr` is minted with it. A vLLM prefill
/// worker on another connector (NIXL, MoRI-IO, a store, none) carries its own
/// side channel or meets its partners in the store and is never handed a
/// bootstrap port, and neither is a decode or a regular worker.
pub(crate) fn effective_bootstrap_port(worker: &dyn Worker) -> Option<(u16, BootstrapPortSource)> {
    if is_store_prefill_worker(worker) {
        return None;
    }
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

/// The address the probes check for `worker`: the host the side was
/// registered under and the port of [`effective_bootstrap_port`] (the one it
/// was registered with, or the default its partners are handed), with where
/// the port came from. `None` for a worker no partner is handed a port for,
/// or whose transport carries no host.
fn partners_bootstrap_address(worker: &dyn Worker) -> Option<(String, u16, BootstrapPortSource)> {
    let (port, source) = effective_bootstrap_port(worker)?;
    let host = worker.bootstrap_host();
    (!host.is_empty()).then(|| (host.to_string(), port, source))
}

/// Whether the port of [`effective_bootstrap_port`] is one the request path
/// hands the decode partner of `worker`. A vLLM prefill side's is on the
/// Mooncake connector alone, whose decode-side `remote_bootstrap_addr` is
/// minted with it: a port registered on a side whose connector carries its own
/// side channel (NIXL, MoRI-IO) or meets its partners in a store plays no
/// part in the handoff, so it is the registration probe's concern alone and
/// never turns a request's failure into a rendezvous failure. Every other
/// side's port is handed whenever one is.
fn handed_on_the_request_path(worker: &dyn Worker) -> bool {
    let spec = &worker.metadata().spec;
    !(matches!(worker.worker_type(), WorkerType::Prefill)
        && matches!(spec.runtime_type, RuntimeType::Vllm)
        && spec.kv_connector.as_deref() != Some(MOONCAKE_CONNECTOR))
}

/// Probe, now, the bootstrap port the partners of `worker` are handed: the
/// registration probe's connection and bound, run by the request path once a
/// side has failed, on a port the request path really hands the decode
/// partner. `None` for a worker that is not probed (see
/// [`partners_bootstrap_address`]) or whose port is not handed on the request
/// path (see [`handed_on_the_request_path`]); otherwise whether the port
/// accepted the connection, with the port and the error when it did not.
pub(crate) async fn probe_partners_bootstrap_port(
    worker: &dyn Worker,
) -> Option<Result<(), DeadBootstrapPort>> {
    if !handed_on_the_request_path(worker) {
        return None;
    }
    let (host, port, source) = partners_bootstrap_address(worker)?;
    let probed = probe_bootstrap_port(&host, port, PROBE_TIMEOUT).await;
    Some(probed.map_err(|error| DeadBootstrapPort {
        host,
        port,
        source,
        error,
    }))
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
/// carries no host; a port registered on a prefill worker whose partners meet
/// it in a KV store is said to be unused instead of probed.
pub(crate) fn spawn_bootstrap_port_probe(worker: &dyn Worker) {
    if is_store_prefill_worker(worker) {
        if let Some(port) = worker.bootstrap_port() {
            info!(
                worker = %worker.redacted_url(),
                bootstrap_port = port,
                kv_connector = worker.metadata().spec.kv_connector.as_deref().unwrap_or(""),
                "Bootstrap port registered on a prefill worker whose KV connector writes to a \
                 shared store is not used: its decode partners read the blocks back from \
                 the store, nothing rendezvous on the port"
            );
        }
        return;
    }
    let Some((host, port, source)) = partners_bootstrap_address(worker) else {
        return;
    };
    let url = worker.url().to_string();
    let worker_type = worker.worker_type().as_metric_label();
    #[expect(
        clippy::disallowed_methods,
        reason = "an advisory probe that ends within its own timeout; nothing waits on it at shutdown"
    )]
    tokio::spawn(async move {
        report_bootstrap_port_probe(&url, worker_type, &host, port, source).await;
    });
}

/// Run the probe and report its verdict: a port that did not accept the
/// connection is one WARN naming the worker, the port, where the port came
/// from (`bootstrap_port_source`: `registered`, or `default` for a worker
/// registered without one) and the error, and one count under
/// `smg_pd_rendezvous_failures_total{leg, phase="probe"}`; a port that did is
/// a DEBUG line. Returns whether the port accepted.
pub(crate) async fn report_bootstrap_port_probe(
    url: &str,
    worker_type: &'static str,
    host: &str,
    port: u16,
    source: BootstrapPortSource,
) -> bool {
    match probe_bootstrap_port(host, port, PROBE_TIMEOUT).await {
        Ok(()) => {
            debug!(
                worker = %Endpoint::redacted_str(url),
                worker_type = worker_type,
                bootstrap_host = host,
                bootstrap_port = port,
                bootstrap_port_source = source.label(),
                "Bootstrap port accepted a connection at registration"
            );
            true
        }
        Err(error) => {
            warn!(
                worker = %Endpoint::redacted_str(url),
                worker_type = worker_type,
                bootstrap_host = host,
                bootstrap_port = port,
                bootstrap_port_source = source.label(),
                error = %error,
                "Bootstrap port did not accept a connection at registration: the worker is \
                 registered, but its disaggregated partners cannot rendezvous there"
            );
            Metrics::record_pd_rendezvous_failure(worker_type, PROBE_PHASE);
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
        test_support::RefusedAddress,
        worker::BasicWorkerBuilder,
    };

    /// A worker of one type and runtime, with the connector and the
    /// bootstrap port it was registered with.
    fn worker_of(
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

    /// The port a worker was registered with is the one probed, whatever the
    /// worker: the operator said so.
    #[test]
    fn a_registered_bootstrap_port_is_the_one_probed_whatever_the_worker() {
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
            let worker = worker_of(worker_type, runtime, connector, Some(9090));
            assert_eq!(
                effective_bootstrap_port(&worker),
                Some((9090, BootstrapPortSource::Registered)),
                "{worker_type} {runtime} {connector:?}"
            );
        }
    }

    /// A worker registered without a port whose partners are handed the default
    /// on the request path is probed on that default: a prefill worker of a
    /// runtime whose rendezvous travels in the request, a vLLM prefill worker on
    /// the Mooncake connector, an encode worker.
    #[test]
    fn a_worker_without_a_bootstrap_port_is_probed_on_the_default_its_partners_are_handed() {
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
            let worker = worker_of(worker_type, runtime, connector, None);
            assert_eq!(
                effective_bootstrap_port(&worker),
                Some((DEFAULT_BOOTSTRAP_PORT, BootstrapPortSource::Default)),
                "{worker_type} {runtime} {connector:?}"
            );
        }
    }

    /// A worker no partner is handed a bootstrap port for is not probed: a
    /// decode or regular worker, and a vLLM prefill worker whose connector
    /// carries its own side channel.
    #[test]
    fn a_worker_no_partner_is_handed_a_bootstrap_port_for_is_not_probed() {
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
            let worker = worker_of(worker_type, runtime, connector, None);
            assert_eq!(
                effective_bootstrap_port(&worker),
                None,
                "{worker_type} {runtime} {connector:?}"
            );
        }
    }

    /// A prefill worker whose connector writes to a shared store hands no
    /// partner a port, whatever port it was registered with: its decode
    /// partners read the blocks back from the store. The rendezvous variant
    /// of the same family keeps its probe.
    #[test]
    fn a_store_prefill_worker_is_not_probed_whatever_port_it_was_registered_with() {
        for connector in [
            "MooncakeStoreConnector",
            "SharedStorageConnector",
            "FlexKVConnectorV1",
        ] {
            for port in [Some(8998), Some(9090), None] {
                let worker = worker_of(
                    WorkerType::Prefill,
                    RuntimeType::Vllm,
                    Some(connector),
                    port,
                );
                assert_eq!(
                    effective_bootstrap_port(&worker),
                    None,
                    "{connector} {port:?}"
                );
            }
        }
        let rendezvous = worker_of(
            WorkerType::Prefill,
            RuntimeType::Vllm,
            Some(MOONCAKE_CONNECTOR),
            Some(8998),
        );
        assert_eq!(
            effective_bootstrap_port(&rendezvous),
            Some((8998, BootstrapPortSource::Registered))
        );
    }

    /// For a vLLM prefill worker the probe's decision is the request path's: the
    /// default is probed exactly when the connector mode minted for the worker
    /// hands the decode worker a bootstrap address on that port.
    #[test]
    fn the_default_port_decision_follows_the_connector_mode_the_request_path_mints() {
        for connector in [
            None,
            Some(MOONCAKE_CONNECTOR),
            Some("NixlConnector"),
            Some("MoRIIOConnector"),
            Some("LMCacheConnector"),
        ] {
            let worker = worker_of(WorkerType::Prefill, RuntimeType::Vllm, connector, None);
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

    /// A prefill side on the loopback host, registered with `bootstrap_port`.
    fn loopback_prefill(bootstrap_port: u16) -> impl Worker {
        BasicWorkerBuilder::new("grpc://127.0.0.1:30000")
            .worker_type(WorkerType::Prefill)
            .runtime_type(RuntimeType::Sglang)
            .bootstrap_port(Some(bootstrap_port))
            .build()
    }

    /// A loopback prefill side of one runtime and connector, registered with a
    /// bootstrap port.
    fn loopback_leg(runtime: RuntimeType, kv_connector: Option<&str>, port: u16) -> impl Worker {
        let mut builder = BasicWorkerBuilder::new("grpc://127.0.0.1:30000")
            .worker_type(WorkerType::Prefill)
            .runtime_type(runtime)
            .bootstrap_port(Some(port));
        if let Some(connector) = kv_connector {
            builder = builder.kv_connector(connector);
        }
        builder.build()
    }

    /// A port the request path does not hand the decode partner is not
    /// probed for a request, whatever port the side was registered with: a
    /// vLLM prefill side on a connector with its own side channel, or on none.
    /// The registration probe keeps the operator's port for it. The Mooncake
    /// side's port, and a runtime's whose rendezvous travels in the request,
    /// are probed.
    #[tokio::test]
    async fn a_port_the_request_path_does_not_hand_is_not_probed_for_a_request() {
        let refused = RefusedAddress::reserve();
        let port = refused.port();
        for connector in [None, Some("NixlConnector"), Some("MoRIIOConnector")] {
            let worker = loopback_leg(RuntimeType::Vllm, connector, port);
            assert_eq!(
                effective_bootstrap_port(&worker),
                Some((port, BootstrapPortSource::Registered)),
                "{connector:?}"
            );
            assert_eq!(
                probe_partners_bootstrap_port(&worker).await,
                None,
                "{connector:?}"
            );
        }
        for (runtime, connector) in [
            (RuntimeType::Vllm, Some(MOONCAKE_CONNECTOR)),
            (RuntimeType::Sglang, None),
            (RuntimeType::TokenSpeed, None),
        ] {
            let dead = probe_partners_bootstrap_port(&loopback_leg(runtime, connector, port))
                .await
                .expect("the port is handed on the request path")
                .expect_err("nothing listens there");
            assert_eq!(
                (dead.port, dead.source),
                (port, BootstrapPortSource::Registered),
                "{runtime} {connector:?}"
            );
        }
    }

    /// The request path's probe of a side's port: a port nothing listens on
    /// is the dead port, with its address and the error; a listening one
    /// accepted; a side no partner is handed a port for is not probed.
    #[tokio::test]
    async fn the_partners_port_probe_names_a_dead_port_and_passes_a_live_one() {
        let refused = RefusedAddress::reserve();
        let port = refused.port();
        let dead = probe_partners_bootstrap_port(&loopback_prefill(port))
            .await
            .expect("a prefill side is probed")
            .expect_err("nothing listens there");
        assert_eq!(
            (dead.host.as_str(), dead.port, dead.source),
            ("127.0.0.1", port, BootstrapPortSource::Registered)
        );
        assert!(!dead.error.is_empty());

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("address").port();
        assert_eq!(
            probe_partners_bootstrap_port(&loopback_prefill(port)).await,
            Some(Ok(()))
        );

        let decode = BasicWorkerBuilder::new("grpc://127.0.0.1:30001")
            .worker_type(WorkerType::Decode)
            .build();
        assert_eq!(probe_partners_bootstrap_port(&decode).await, None);
    }

    #[tokio::test]
    async fn a_bootstrap_port_probe_fails_a_closed_port_and_counts_it() {
        let refused = RefusedAddress::reserve();
        let port = refused.port();
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
