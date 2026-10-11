//! What a disaggregated leg's failure says about the rendezvous.
//!
//! The legs of a disaggregated request meet on the prefill's bootstrap port;
//! a leg that cannot (the port is closed, the partner never arrived, the
//! handoff broke) is retired by its engine with a message that names the
//! phase it died in. Those failures count under
//! `smg_pd_bootstrap_failures_total{leg, phase}`, next to the registration
//! probe's, so a dead rendezvous port shows as a rate at the gateway and not
//! only as late client errors.

use super::stream_failure::UpstreamClass;
use crate::observability::metrics::Metrics;

/// Words an engine puts on a leg that never met its partner: the bootstrap
/// (its server, its room, the wait in the bootstrapping state) and the
/// handshake the decode side runs against it.
const BOOTSTRAP_WORDS: &[&str] = &["bootstrap", "handshake"];

/// Words an engine puts on a leg whose KV handoff failed after the legs met.
const TRANSFER_WORDS: &[&str] = &["transfer failed", "kv transfer", "kvcache from prefill"];

/// The phase of the rendezvous a leg's failure names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RendezvousPhase {
    /// The legs never met: a bootstrap or handshake that was refused or
    /// timed out.
    Bootstrap,
    /// The legs met and the KV handoff between them failed.
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

/// Count `status`, the failure of the `leg` of a disaggregated request, under
/// the bootstrap-failure family when it names the rendezvous. Any other
/// failure of the leg is left to its own accounting.
pub(crate) fn account_pd_leg_failure(leg: &'static str, status: &tonic::Status) {
    if let Some(phase) = RendezvousPhase::of(status) {
        Metrics::record_pd_bootstrap_failure_for(leg, phase.label());
    }
}

#[cfg(test)]
mod tests {
    use metrics_exporter_prometheus::PrometheusBuilder;

    use super::*;
    use crate::observability::metrics::metrics_labels;

    /// The texts a disaggregated engine's scheduler and transfer layer write
    /// for a leg they retire, as their sources spell them, and the phase
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
            // The legs met; the handoff itself failed.
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
    fn a_leg_failure_that_names_the_rendezvous_is_counted_under_the_bootstrap_family() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            account_pd_leg_failure(
                metrics_labels::WORKER_PREFILL,
                &tonic::Status::internal(
                    "Prefill bootstrap failed for request rank=0 req.rid='r1' req.bootstrap_room=41",
                ),
            );
            account_pd_leg_failure(
                metrics_labels::WORKER_DECODE,
                &tonic::Status::internal(
                    "Decode transfer failed for request rank=0 req.rid='r3' req.bootstrap_room=43",
                ),
            );
            account_pd_leg_failure(
                metrics_labels::WORKER_DECODE,
                &tonic::Status::unavailable("connection reset"),
            );
        });
        let rendered = handle.render();
        let counted: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with("smg_pd_bootstrap_failures_total{"))
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
