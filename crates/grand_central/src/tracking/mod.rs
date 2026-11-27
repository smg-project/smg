//! Request tracking: one placed request from its decision to its end, the
//! requests in flight on each worker and the dispatches since a worker's last
//! report.

pub(crate) mod decision_guard;
pub(crate) mod dispatch_credit;
pub(crate) mod in_flight;
pub(crate) mod lease;
#[cfg(feature = "in-process")]
pub(crate) mod placement;
#[cfg(feature = "in-process")]
pub(crate) mod placement_side;
pub(crate) mod prefix_predictor;
pub(crate) mod rank_credit;

pub use decision_guard::DecisionGuard;
pub use dispatch_credit::{CreditedDispatches, DispatchCredit, DispatchCreditPass};
pub use in_flight::HostCount;
pub use lease::{
    Dispatch, EventKind, EventSet, Lease, LeaseId, LeaseKind, Membership, Outcome, ReleaseReason,
    Said, Side, Signal, TransportClass,
};
#[cfg(feature = "in-process")]
pub use placement::Encoders;
#[cfg(feature = "in-process")]
pub use placement::Placement;
#[cfg(feature = "in-process")]
pub use placement_side::{PlacementSide, SideEnd};
pub use prefix_predictor::{DeepestPrefix, PrefixPredictor};
pub use rank_credit::DispatchCounts;
