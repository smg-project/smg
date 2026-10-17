//! Per-tenant LLM token/request rate limiting.
//!
//! Config schema, policy compilation, the reserve/settle engine, and an
//! in-memory backend. Nothing in the request path calls `reserve()` yet
//! — that's wired in per protocol in a later phase.

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
