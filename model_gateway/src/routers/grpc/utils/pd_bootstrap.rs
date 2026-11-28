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

use smg_worker::{
    health::bootstrap_probe::{probe_partners_bootstrap_port, DeadBootstrapPort, REQUEST_PHASE},
    identity::basic::Worker,
    metrics::writers::Metrics,
};

use super::{stream_failure::UpstreamClass, tonic_ext::is_the_engines_own_report};
use crate::observability::metrics::metrics_labels;

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
#[path = "../../../../tests/unit/routers/grpc/utils/pd_bootstrap/tests.rs"]
mod tests;
