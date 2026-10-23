//! What a disaggregated side's failure says about the rendezvous.
//!
//! The sides of a disaggregated request meet on the prefill's bootstrap port;
//! a side that cannot (the port is closed, the partner never arrived, the
//! handoff broke) is retired by its engine with a message that names the
//! phase it died in. Those failures count under
//! `smg_pd_rendezvous_failures_total{leg, phase}`, next to the registration
//! probe's, so a dead rendezvous port shows as a rate at the gateway and not
//! only as late client errors. An engine whose wire carries no reason
//! retires the side with a bare report; for a decode side, the gateway then
//! asks the question itself, with the registration probe, of the port that
//! side had to reach (see [`dead_rendezvous_of`]).

use std::{fmt, sync::Arc};

use super::stream_failure::{is_the_engines_own_report, UpstreamClass};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    worker::{
        bootstrap_probe::{probe_partners_bootstrap_port, DeadBootstrapPort, REQUEST_PHASE},
        Worker,
    },
};

/// Words an engine puts on a side that never met its partner: the bootstrap
/// (its server, its room, the wait in the bootstrapping state) and the
/// handshake the decode side runs against it.
const BOOTSTRAP_WORDS: &[&str] = &["bootstrap", "handshake"];

/// Words an engine puts on a side whose KV handoff failed after the sides met.
const TRANSFER_WORDS: &[&str] = &["transfer failed", "kv transfer", "kvcache from prefill"];

/// The phase of the rendezvous a side's failure names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RendezvousPhase {
    /// The sides never met: a bootstrap or handshake that was refused or
    /// timed out.
    Bootstrap,
    /// The sides met and the KV handoff between them failed.
    Transfer,
}

impl RendezvousPhase {
    /// The `phase` label value.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Transfer => "transfer",
        }
    }

    /// The phase `status` names, when it is the worker's failure (a server
    /// error or a timeout: a request's own fault is no rendezvous failure,
    /// whatever it mentions) and its text names one. The earliest mention
    /// decides: an engine's verdict leads its message, and the cause it
    /// appends may belong to the other phase.
    pub(crate) fn of(status: &tonic::Status) -> Option<Self> {
        if !UpstreamClass::of(status).is_worker_failure() {
            return None;
        }
        let message = status.message().to_ascii_lowercase();
        let earliest = |words: &[&str]| words.iter().filter_map(|word| message.find(word)).min();
        match (earliest(BOOTSTRAP_WORDS), earliest(TRANSFER_WORDS)) {
            (None, None) => None,
            (Some(_), None) => Some(Self::Bootstrap),
            (None, Some(_)) => Some(Self::Transfer),
            (Some(bootstrap), Some(transfer)) => Some(if bootstrap <= transfer {
                Self::Bootstrap
            } else {
                Self::Transfer
            }),
        }
    }
}

/// Count `status`, the failure of the `side` of a disaggregated request, under
/// the rendezvous family when it names a phase of it. Any other
/// failure of the side is left to its own accounting.
pub(crate) fn account_pd_side_failure(side: &'static str, status: &tonic::Status) {
    if let Some(phase) = RendezvousPhase::of(status) {
        Metrics::record_pd_rendezvous_failure(side, phase.label());
    }
}

/// A dead rendezvous port behind a decode side's failure: the prefill side's
/// bootstrap port, as the request path found it (see [`dead_rendezvous_of`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeadRendezvous(pub(crate) DeadBootstrapPort);

impl fmt::Display for DeadRendezvous {
    /// The words a client reads after the engine's report. The port's
    /// address is for the gateway's own line, not for the client.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the prefill side's bootstrap port did not accept a connection ({}), so the sides \
             could not rendezvous",
            self.0.error
        )
    }
}

/// The rendezvous failure behind a decode side its engine failed without a
/// reason. The engine's own report (see [`is_the_engines_own_report`]) names
/// no phase: its wire carries none. When it ends the decode side of a
/// disaggregated request, the port that side's engine had to reach, the
/// prefill side's bootstrap port, is probed as it was at registration (the
/// same port decision, connection and bound). A port that does not accept
/// the connection is the failure the engine could not name: counted under
/// the rendezvous family's `request` phase on the prefill side, whose port it
/// is, and handed back for the line and the message. `None` for a status
/// that carries the engine's words (classed by [`RendezvousPhase`] instead),
/// a prefill side that hands its partners no port, or a port that accepted.
pub(crate) async fn dead_rendezvous_of(
    prefill: Option<&Arc<dyn Worker>>,
    status: &tonic::Status,
) -> Option<DeadRendezvous> {
    if !is_the_engines_own_report(status) {
        return None;
    }
    match probe_partners_bootstrap_port(prefill?.as_ref()).await? {
        Ok(()) => None,
        Err(dead) => {
            Metrics::record_pd_rendezvous_failure(metrics_labels::WORKER_PREFILL, REQUEST_PHASE);
            Some(DeadRendezvous(dead))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::worker::{RuntimeType, WorkerType};

    use super::*;
    use crate::{routers::grpc::proto_wrapper::ENGINE_ERROR_MESSAGE, worker::BasicWorkerBuilder};

    /// A loopback port nothing listens on.
    fn closed_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("address").port();
        drop(listener);
        port
    }

    /// A prefill side on the loopback host whose partners are handed `port`.
    fn prefill_on(port: u16) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:30000")
                .worker_type(WorkerType::Prefill)
                .runtime_type(RuntimeType::Sglang)
                .bootstrap_port(Some(port))
                .build(),
        )
    }

    /// The rendered rendezvous series.
    fn rendezvous_series(rendered: &str) -> Vec<&str> {
        rendered
            .lines()
            .filter(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
            .collect()
    }

    /// The engine's own report behind a prefill side whose bootstrap port
    /// nothing listens on is the dead rendezvous: counted under the request
    /// phase on the prefill side, worded for the client without the address.
    #[tokio::test]
    async fn the_engines_own_report_behind_a_dead_bootstrap_port_is_a_rendezvous_failure() {
        let port = closed_port();
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let dead = {
            let _guard = metrics::set_default_local_recorder(&recorder);
            dead_rendezvous_of(
                Some(&prefill_on(port)),
                &tonic::Status::internal(ENGINE_ERROR_MESSAGE),
            )
            .await
        }
        .expect("the dead port is the rendezvous failure");
        assert_eq!((dead.0.host.as_str(), dead.0.port), ("127.0.0.1", port));
        let words = dead.to_string();
        assert!(
            words.starts_with("the prefill side's bootstrap port did not accept a connection (")
                && words.ends_with("), so the sides could not rendezvous"),
            "{words}"
        );
        assert!(
            !words.contains("127.0.0.1"),
            "the address stays out of the client's words: {words}"
        );
        let rendered = handle.render();
        let counted = rendezvous_series(&rendered);
        assert_eq!(counted.len(), 1, "{rendered}");
        assert!(
            counted[0].contains(r#"leg="prefill""#)
                && counted[0].contains(r#"phase="request""#)
                && counted[0].ends_with(" 1"),
            "{rendered}"
        );
    }

    /// Nothing is probed or counted without both a report that carries no
    /// reason and a dead port: a status with the engine's words (classed by
    /// phase above), a port that accepts, no prefill side, or a prefill side
    /// no partner is handed a port for.
    #[tokio::test]
    async fn nothing_is_probed_without_a_reasonless_report_and_a_dead_port() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let live = listener.local_addr().expect("address").port();
        let side_channel: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:30000")
                .worker_type(WorkerType::Prefill)
                .runtime_type(RuntimeType::Vllm)
                .kv_connector("NixlConnector")
                .build(),
        );
        let report = tonic::Status::internal(ENGINE_ERROR_MESSAGE);
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let verdicts = {
            let _guard = metrics::set_default_local_recorder(&recorder);
            [
                dead_rendezvous_of(Some(&prefill_on(live)), &report).await,
                dead_rendezvous_of(
                    Some(&prefill_on(closed_port())),
                    &tonic::Status::internal(
                        "Decode handshake failed for request rank=0 req.rid='r2' \
                         req.bootstrap_room=42",
                    ),
                )
                .await,
                dead_rendezvous_of(
                    Some(&prefill_on(closed_port())),
                    &tonic::Status::unavailable("connection reset"),
                )
                .await,
                dead_rendezvous_of(None, &report).await,
                dead_rendezvous_of(Some(&side_channel), &report).await,
            ]
        };
        assert!(verdicts.iter().all(Option::is_none), "{verdicts:?}");
        assert!(
            rendezvous_series(&handle.render()).is_empty(),
            "{}",
            handle.render()
        );
    }

    /// The texts a disaggregated engine's scheduler and transfer layer write
    /// for a side they retire, as their sources spell them, and the phase
    /// each names.
    #[test]
    fn an_engine_reported_rendezvous_failure_is_classed_by_phase() {
        let cases = [
            // The prefill waited for the decode's KV indices past its deadline.
            (
                "Prefill bootstrap failed for request rank=0 req.rid='r1' req.bootstrap_room=41 \
                 with exception KVTransferError(bootstrap_room=41): Request 41 timed out after \
                 120.0s in KVPoll.Bootstrapping",
                RendezvousPhase::Bootstrap,
            ),
            // The decode never reached the prefill's bootstrap server.
            (
                "Decode handshake failed for request rank=0 req.rid='r2' req.bootstrap_room=42 \
                 with exception KVTransferError(bootstrap_room=42): Failed to get kvcache from \
                 prefill instance, it might be dead",
                RendezvousPhase::Bootstrap,
            ),
            (
                "Some requests timed out when bootstrapping, which means prefill instances fail \
                 to receive the KV indices from the decode instance of this request.",
                RendezvousPhase::Bootstrap,
            ),
            // The sides met; the handoff itself failed.
            (
                "Decode transfer failed for request rank=0 req.rid='r3' req.bootstrap_room=43 \
                 with exception KVTransferError(bootstrap_room=43): KV transfer failed",
                RendezvousPhase::Transfer,
            ),
            (
                "Prefill transfer failed for request rank=0 req.rid='r4' req.bootstrap_room=44",
                RendezvousPhase::Transfer,
            ),
            (
                "Failed to get kvcache from prefill instance, it might be dead",
                RendezvousPhase::Transfer,
            ),
        ];
        for (message, phase) in cases {
            assert_eq!(
                RendezvousPhase::of(&tonic::Status::internal(message)),
                Some(phase),
                "{message}"
            );
            assert_eq!(
                RendezvousPhase::of(&tonic::Status::unavailable(message)),
                Some(phase),
                "{message}"
            );
        }
    }

    #[test]
    fn a_failure_that_does_not_name_the_rendezvous_is_not_classed() {
        for status in [
            tonic::Status::internal("engine finished the request with an error (see engine logs)"),
            tonic::Status::unavailable("connection reset"),
            tonic::Status::deadline_exceeded("decode step took too long"),
            tonic::Status::internal("h2 protocol error: error reading a body from connection"),
        ] {
            assert_eq!(RendezvousPhase::of(&status), None, "{status:?}");
        }
    }

    /// A request's own fault, and capacity pushback, are not rendezvous
    /// failures whatever they mention.
    #[test]
    fn a_request_fault_is_not_a_rendezvous_failure_whatever_it_mentions() {
        for status in [
            tonic::Status::invalid_argument(
                "bootstrap_room must be set for a disaggregated request",
            ),
            tonic::Status::resource_exhausted("bootstrap queue full"),
        ] {
            assert_eq!(RendezvousPhase::of(&status), None, "{status:?}");
        }
    }

    #[test]
    fn a_side_failure_that_names_the_rendezvous_is_counted_under_the_rendezvous_family() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            account_pd_side_failure(
                metrics_labels::WORKER_PREFILL,
                &tonic::Status::internal(
                    "Prefill bootstrap failed for request rank=0 req.rid='r1' req.bootstrap_room=41",
                ),
            );
            account_pd_side_failure(
                metrics_labels::WORKER_DECODE,
                &tonic::Status::internal(
                    "Decode transfer failed for request rank=0 req.rid='r3' req.bootstrap_room=43",
                ),
            );
            account_pd_side_failure(
                metrics_labels::WORKER_DECODE,
                &tonic::Status::unavailable("connection reset"),
            );
        });
        let rendered = handle.render();
        let counted: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
            .collect();
        assert_eq!(counted.len(), 2, "{rendered}");
        assert!(
            counted.iter().any(|line| line.contains(r#"leg="prefill""#)
                && line.contains(r#"phase="bootstrap""#)
                && line.ends_with(" 1")),
            "{rendered}"
        );
        assert!(
            counted.iter().any(|line| line.contains(r#"leg="decode""#)
                && line.contains(r#"phase="transfer""#)
                && line.ends_with(" 1")),
            "{rendered}"
        );
    }
}
