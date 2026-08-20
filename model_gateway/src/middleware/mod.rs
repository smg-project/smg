//! HTTP middleware stack for the gateway server.
//!
//! Each submodule owns a single concern; this `mod.rs` re-exports the
//! types and functions that `server.rs` and other call sites already
//! reference, so this split is invisible to downstream callers.

pub mod auth;
pub mod concurrency;
pub mod logging;
pub mod metrics;
pub mod request_id;
pub mod scheduler;
pub mod storage_context;
pub mod tenant_resolution;
pub mod token_bucket;
pub mod wasm;

pub use auth::{auth_middleware, deny_all_middleware, AuthConfig};
pub use concurrency::{
    concurrency_limit_middleware, ConcurrencyLimiter, QueueProcessor, QueuedRequest, TokenGuardBody,
};
pub use logging::{
    create_logging_layer, ProbeResponse, RequestLogger, RequestSpan, ResponseLogger,
    StreamFailureLogger,
};
pub use metrics::{HttpMetricsLayer, HttpMetricsMiddleware};
pub use request_id::{RequestId, RequestIdLayer, RequestIdMiddleware};
pub use storage_context::storage_context_middleware;
pub use tenant_resolution::{
    ordinary_tenant_resolution_middleware, route_request_meta_middleware, TenantResolutionState,
};
pub use token_bucket::TokenBucket;
pub use wasm::wasm_middleware;

pub use crate::tenant::{
    resolve_admin_target_tenant_id, resolve_admin_target_tenant_key, DataPlaneCaller,
    RouteRequestMeta, TenantIdentity, TenantKey, TenantResolutionError,
};

/// `Retry-After` seconds on overload sheds: sheds must signal back-off; 408
/// reads as a client transport error and gets instant proxy retries.
pub(crate) const SHED_RETRY_AFTER_SECS: u32 = 2;

/// Backward-compatible alias for the older tenant metadata name used in a few
/// router-path tests and plumbing call sites.
pub type TenantRequestMeta = RouteRequestMeta;
