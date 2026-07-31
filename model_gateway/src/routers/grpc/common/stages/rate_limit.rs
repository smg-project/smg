//! Tenant rate-limit reserve stage: admit or reject a logical request once,
//! before dispatch, reusing the exact input-token count preparation just
//! produced.
//!
//! `RetryExecutor` reruns the whole pipeline (this stage included) fresh on
//! every retry attempt, so [`RateLimitCell`] is threaded in from the router
//! and shared across attempts of one logical request: the first attempt to
//! reach this stage reserves (or is denied) and caches the outcome; every
//! later attempt sees the cached outcome and skips straight through.

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use parking_lot::Mutex;
use tracing::error;

use super::PipelineStage;
use crate::{
    rate_limit::{
        rejection_response, RateLimitManager, Reservation, ReserveRequest, SharedReservationHandle,
    },
    routers::{error, grpc::context::RequestContext},
};

/// Outcome of the first reserve attempt for one logical request, cached so
/// later retry attempts don't reserve again.
#[derive(Clone)]
pub(crate) enum RateLimitOutcome {
    Admitted(Arc<SharedReservationHandle>),
    Denied,
}

/// Shared across every retry attempt of one logical request. `parking_lot::Mutex`
/// (not `tokio::sync::Mutex`) is fine: `peek`/`set` never hold the lock
/// across an `.await`, and `RetryExecutor` runs attempts strictly
/// sequentially, so there's no real contention -- this is just correct
/// check-then-act bookkeeping, not a concurrency primitive.
pub(crate) struct RateLimitCell(Mutex<Option<RateLimitOutcome>>);

impl RateLimitCell {
    pub(crate) fn new() -> Self {
        Self(Mutex::new(None))
    }

    pub(crate) fn peek(&self) -> Option<RateLimitOutcome> {
        self.0.lock().clone()
    }

    fn set(&self, outcome: RateLimitOutcome) {
        *self.0.lock() = Some(outcome);
    }
}

/// Reserve tenant rate-limit budget once per logical request. No-ops when
/// the feature is disabled (`manager: None`) or the calling endpoint hasn't
/// opted in (`ctx.input.rate_limit_cell: None` -- Responses/embeddings/classify
/// today).
pub(crate) struct RateLimitReserveStage {
    manager: Option<Arc<RateLimitManager>>,
}

impl RateLimitReserveStage {
    pub fn new(manager: Option<Arc<RateLimitManager>>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl PipelineStage for RateLimitReserveStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let Some(manager) = &self.manager else {
            return Ok(None);
        };
        let Some(cell) = ctx.input.rate_limit_cell.as_ref() else {
            return Ok(None);
        };

        match cell.peek() {
            Some(RateLimitOutcome::Admitted(_)) => Ok(None),
            Some(RateLimitOutcome::Denied) => {
                // Defensive: `should_retry` in router.rs stops the retry loop
                // as soon as a Denied outcome is cached, so a second attempt
                // should never actually reach this stage.
                Err(rejection_response(0))
            }
            None => {
                let Some(tenant_meta) = ctx.input.tenant_request_meta.as_ref() else {
                    // No tenant identity resolved (shouldn't happen once
                    // tenant_resolution middleware runs, but fail open rather
                    // than block a request over missing rate-limit context).
                    return Ok(None);
                };
                let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
                    error!(
                        function = "RateLimitReserveStage::execute",
                        "Preparation stage not completed"
                    );
                    error::internal_error(
                        "preparation_stage_not_completed",
                        "Preparation stage not completed",
                    )
                })?;

                let request = ReserveRequest {
                    request_charge_id: tenant_meta.request_charge_id,
                    tenant_key: tenant_meta.tenant_key.clone(),
                    model_id: Some(ctx.input.model_id.clone()),
                    estimated_input_tokens: prep.token_ids().len() as u32,
                };

                match manager.reserve(request).await {
                    Reservation::Admitted(handle) => {
                        cell.set(RateLimitOutcome::Admitted(handle));
                        Ok(None)
                    }
                    Reservation::Denied { retry_after_secs } => {
                        cell.set(RateLimitOutcome::Denied);
                        Err(rejection_response(retry_after_secs))
                    }
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "RateLimitReserve"
    }
}
