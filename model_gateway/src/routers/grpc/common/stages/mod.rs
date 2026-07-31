//! Common pipeline stages shared across all endpoints and model types
//!
//! These stages are endpoint-agnostic and model-agnostic:
//! - Worker selection
//! - Client acquisition
//! - Dispatch metadata generation
//! - Request execution

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::grpc::context::RequestContext;

/// Trait for pipeline stages that process requests
#[async_trait]
pub trait PipelineStage: Send + Sync {
    /// Execute this stage, mutating the context
    ///
    /// Returns:
    /// - `Ok(None)` - Continue to next stage
    /// - `Ok(Some(response))` - Pipeline complete, return this response (e.g., streaming)
    /// - `Err(response)` - Error occurred, return this error response
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response>;

    /// Stage name for logging
    fn name(&self) -> &'static str;

    /// Stable descriptor of the stage plus its mode-bearing args, compared
    /// against golden literals in the pipeline parity test. Stages whose
    /// construction args vary by mode override this to include them; the default
    /// emits the short type name (last `::` segment).
    #[cfg(test)]
    fn signature(&self) -> String {
        std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or_default()
            .to_string()
    }
}

mod client_acquisition;
mod dispatch_metadata;
pub(crate) mod encode;
pub(crate) mod helpers;
mod rate_limit;
mod request_execution;
mod worker_selection;

// Export stage implementations
pub(crate) use client_acquisition::ClientAcquisitionStage;
pub(crate) use dispatch_metadata::DispatchMetadataStage;
pub(crate) use encode::EncodeStage;
pub(crate) use rate_limit::{RateLimitCell, RateLimitOutcome, RateLimitReserveStage};
pub(crate) use request_execution::RequestExecutionStage;
pub(crate) use worker_selection::{WorkerSelectionMode, WorkerSelectionStage};
