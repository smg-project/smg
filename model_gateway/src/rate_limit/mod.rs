//! Per-tenant LLM token/request rate limiting.
//!
//! Config schema, policy compilation, the reserve/settle engine, and an
//! in-memory backend. The gRPC router's typed routes reserve once per
//! logical request in the ingress stage (`RateLimitReserveStage`), and each
//! iteration of a gRPC Responses request reserves on its own; the regular
//! and the disaggregated HTTP routers' typed routes and the external
//! routers' gateway side reserve once per logical request before dispatch
//! (`routers/http/tenant_rate_limit.rs`).

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
