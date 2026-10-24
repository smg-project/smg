//! Per-tenant LLM token/request rate limiting.
//!
//! Config schema, policy compilation, the reserve/settle engine, and an
//! in-memory backend. The gRPC router's typed routes reserve once per
//! logical request in the ingress stage (`RateLimitReserveStage`) and settle
//! through the handle it returns; the gRPC Responses iterations reserve
//! nothing today, and the HTTP routers are not wired to the manager.

mod backend;
mod bucket;
mod config;
mod local_backend;
mod manager;
mod policy;
mod rejection;
mod reservation;
#[cfg(test)]
pub(crate) mod test_support;

pub use backend::{RateLimitBackend, ReserveOutcome, ReserveRequest, UsageSettlement};
pub use config::{
    ModelMatcherSpec, ModelRuleSpec, RateLimitConfigError, RateLimitYaml, TenantPolicySpec,
};
pub use manager::{RateLimitManager, Reservation};
pub use policy::{CompiledPolicySet, ScopeLimits};
pub use rejection::{rejection_response, RATE_LIMIT_ERROR_CODE};
pub use reservation::{ReservationAttachment, SharedReservationHandle};
