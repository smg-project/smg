//! Tenant rate-limit reserve stage: admit or reject a logical request once,
//! at ingress, reusing the exact input-token count preparation just produced.
//!
//! Ingress runs once per logical request (retries re-dispatch the retained
//! plan without re-running this stage), so the reservation is naturally
//! single-shot. [`RateLimitCell`] (`routers/common/rate_limit_cell.rs`, the
//! reservation every router takes) is threaded in from the router (one per
//! typed request) or from the pipeline's Responses entry points (one per
//! iteration of a Responses request: every chat turn of a tool loop is a
//! generation of its own) and holds the outcome for the paths that resolve
//! it later: the settle of a buffered answer, the streaming handoff, and the
//! close of a reservation a failed answer never settled.
//!
//! A charge id names one generation to the backend: `reserve` is idempotent
//! per id (a retry of a generation reuses its reservation, and a backend may
//! keep the id after its resolution), so the iterations of a Responses
//! request, generations of their own, each reserve under a fresh id
//! ([`iteration_charge`]).

use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::warn;
use uuid::Uuid;

use super::PipelineStage;
pub(crate) use crate::routers::common::rate_limit_cell::{RateLimitCell, RateLimitOutcome};
use crate::{
    middleware::TenantRequestMeta,
    rate_limit::{rejection_response, RateLimitManager},
    routers::grpc::context::{stage_output, RequestContext},
};

/// The tenant meta one iteration of a Responses request reserves under: the
/// request's tenant with a charge id of its own. A charge id names one
/// generation to the backend (`reserve` is idempotent per id, so a retry of a
/// generation reuses its reservation, and a backend may keep the id after its
/// resolution); the iterations of a tool loop are generations of their own,
/// so each reserves under a fresh id, whatever the backend remembers of the
/// ones before.
pub(crate) fn iteration_charge(meta: &TenantRequestMeta) -> TenantRequestMeta {
    let mut iteration = meta.clone();
    iteration.request_charge_id = Uuid::now_v7();
    iteration
}

/// Reserve tenant rate-limit budget once per logical request. No-ops when
/// the feature is disabled (`manager: None`) or the caller set no cell
/// (`ctx.input.rate_limit_cell: None`; every typed route and every
/// Responses iteration sets one, so that is a unit test's context).
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
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let Some(manager) = &self.manager else {
            return Ok(());
        };
        let Some(cell) = ctx.input.rate_limit_cell.as_ref() else {
            return Ok(());
        };

        match cell.peek() {
            Some(RateLimitOutcome::Admitted(_)) => Ok(()),
            // Defensive: ingress runs once per logical request, so a cached
            // Denied outcome should never be seen here again.
            Some(RateLimitOutcome::Denied) => Err(rejection_response(0)),
            None => {
                let Some(tenant_meta) = ctx.input.tenant_request_meta.as_ref() else {
                    // No tenant identity resolved (shouldn't happen once
                    // tenant_resolution middleware runs, but fail open rather
                    // than block a request over missing rate-limit context).
                    warn!(
                        function = "RateLimitReserveStage::execute",
                        "tenant_request_meta missing; skipping rate-limit reservation"
                    );
                    return Ok(());
                };
                let prep = stage_output(
                    ctx.state.preparation.as_ref(),
                    "RateLimitReserveStage::execute",
                )?;

                cell.reserve(
                    manager,
                    tenant_meta,
                    &ctx.input.model_id,
                    prep.total_input_token_count() as u32,
                )
                .await
            }
        }
    }

    fn name(&self) -> &'static str {
        "RateLimitReserve"
    }
}

#[cfg(test)]
mod iteration_charge_tests {
    use super::*;
    use crate::tenant::TenantKey;

    /// Two iterations of one request reserve as the request's tenant under
    /// charge ids of their own, distinct from each other and from the
    /// request's.
    #[test]
    fn each_iteration_reserves_under_its_own_charge_id() {
        let request = TenantRequestMeta::new(TenantKey::new("auth:tenant"));
        let first = iteration_charge(&request);
        let second = iteration_charge(&request);

        assert_eq!(first.tenant_key, request.tenant_key);
        assert_eq!(second.tenant_key, request.tenant_key);
        assert_ne!(first.request_charge_id, request.request_charge_id);
        assert_ne!(second.request_charge_id, request.request_charge_id);
        assert_ne!(first.request_charge_id, second.request_charge_id);
    }
}
