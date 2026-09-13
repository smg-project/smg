//! Common pipeline stages shared across all endpoints and model types.
//!
//! The pipeline is two-phase: ingress stages run on [`RequestContext`] (which
//! owns the parsed request) through request building; the dispatch phase runs
//! on [`DispatchContext`], which has no request field.

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::grpc::{
    context::{BuildOutput, DispatchContext, RequestContext},
    spec::ResponseSpec,
};

/// Ingress-phase stage: full access to the parsed request.
#[async_trait]
pub trait PipelineStage: Send + Sync {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response>;

    /// Stage name for logging
    fn name(&self) -> &'static str;
}

/// Final ingress stage: the last reader of the parsed request. Produces
/// the retained execution plan, the response spec, and the per-attempt stamp.
#[async_trait]
pub trait BuildStage: Send + Sync {
    async fn build(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response>;

    fn name(&self) -> &'static str;

    /// Stable descriptor of the stage plus its mode-bearing args, compared
    /// against golden literals in the pipeline parity test.
    #[cfg(test)]
    fn signature(&self) -> String;
}

/// Dispatch-phase stage: consumes one attempt's execution result. The spec is
/// its only request-derived input.
///
/// Returns:
/// - `Ok(Some(response))` — early response (streaming SSE), return as-is
/// - `Ok(None)` — final response stored in `ctx.response`
/// - `Err(response)` — error
#[async_trait]
pub trait ProcessStage: Send + Sync {
    async fn process(
        &self,
        ctx: &mut DispatchContext,
        spec: ResponseSpec,
    ) -> Result<Option<Response>, Response>;

    fn name(&self) -> &'static str;
}

mod client_acquisition;
mod dispatch_metadata;
pub(crate) mod encode;
pub(crate) mod helpers;
pub(crate) mod pd_protocol;
mod rate_limit;
mod request_execution;
mod worker_selection;

// Export stage implementations
pub(crate) use client_acquisition::acquire_clients;
pub(crate) use dispatch_metadata::prepare_dispatch_metadata;
pub(crate) use encode::EncodeStage;
pub(crate) use rate_limit::{RateLimitCell, RateLimitOutcome, RateLimitReserveStage};
pub(crate) use request_execution::execute_plan;
pub(crate) use worker_selection::{WorkerSelectionMode, WorkerSelectionStage};
