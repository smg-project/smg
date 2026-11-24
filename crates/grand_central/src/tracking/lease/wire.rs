//! The lifecycle's words as the protocol spells them. Every writer of a
//! lifecycle frame says them through these: the plugin host's view and a
//! gateway's client.

use super::{LeaseId, Outcome, ReleaseReason, Side, TransportClass};
use crate::service::proto;

impl From<Side> for proto::Side {
    fn from(side: Side) -> Self {
        match side {
            Side::Single => Self::Single,
            Side::Prefill => Self::Prefill,
            Side::Decode => Self::Decode,
            Side::Encode => Self::Encode,
        }
    }
}

impl From<TransportClass> for proto::completed::Class {
    fn from(class: TransportClass) -> Self {
        match class {
            TransportClass::Ok => Self::Ok,
            TransportClass::ClientError => Self::ClientError,
            TransportClass::ServerError => Self::ServerError,
            TransportClass::ConnectFailed => Self::ConnectFailed,
            TransportClass::Timeout => Self::Timeout,
            TransportClass::Cancelled => Self::Cancelled,
        }
    }
}

impl From<ReleaseReason> for proto::released::Reason {
    fn from(reason: ReleaseReason) -> Self {
        match reason {
            ReleaseReason::Abandoned => Self::Abandoned,
            ReleaseReason::LateDecision => Self::LateDecision,
            ReleaseReason::ClientGone => Self::ClientGone,
            ReleaseReason::CommitFailed => Self::CommitFailed,
            ReleaseReason::Expired => Self::Expired,
            ReleaseReason::WorkerRemoved => Self::WorkerRemoved,
        }
    }
}

/// The completion of `lease` as its frame carries it.
pub(crate) fn completed(lease: LeaseId, outcome: &Outcome) -> proto::Completed {
    proto::Completed {
        lease_id: lease.0,
        side: proto::Side::from(outcome.side).into(),
        served: outcome.served.0,
        status: u32::from(outcome.status),
        class: proto::completed::Class::from(outcome.class).into(),
        ttft_micros: outcome.ttft_micros,
        tokens_in: outcome.tokens_in,
        tokens_out: outcome.tokens_out,
        duration_micros: outcome.duration_micros,
    }
}
