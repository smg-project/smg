//! Tenant rate-limit reserve stage: admit or reject a logical request once,
//! before dispatch, reusing the exact input-token count preparation just
//! produced.
//!
//! `RetryExecutor` reruns the whole pipeline (this stage included) fresh on
//! every retry attempt, so [`RateLimitCell`] is threaded in from the router
//! and shared across attempts of one logical request: the first attempt to
//! reach this stage reserves (or is denied) and caches the outcome; every
//! later attempt sees the cached outcome and skips straight through.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use async_trait::async_trait;
use axum::response::Response;
use parking_lot::Mutex;
use tracing::{error, warn};

use super::PipelineStage;
use crate::{
    rate_limit::{
        rejection_response, RateLimitManager, Reservation, ReservationAttachment, ReserveRequest,
        SharedReservationHandle,
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
///
/// `Drop` is the safety net for a reservation that never gets the chance to
/// resolve through any of the normal paths -- e.g. the priority scheduler
/// preempting the route future (dropping it) before a response of any kind
/// (streaming or not) is ever produced. Non-streaming success (settled
/// inline) and the router's post-retry-loop close-if-unsettled both resolve
/// the handle *before* this cell would otherwise drop, so `Drop`'s own
/// `abandon_if_open` is a safe no-op there (CAS-guarded, first resolution
/// wins). Streaming success is the one case that needs an explicit opt-out
/// (`handed_off`): resolution there is intentionally deferred to whenever
/// the response body's `ReservationAttachment` drops, which happens *after*
/// this cell would otherwise drop (`route_*_impl` returns the initial SSE
/// response long before the stream finishes) -- without the flag, `Drop`
/// would race that deferred resolution and always win, permanently
/// abandoning the reservation before real usage is ever settled.
pub(crate) struct RateLimitCell {
    outcome: Mutex<Option<RateLimitOutcome>>,
    handed_off: AtomicBool,
}

impl RateLimitCell {
    pub(crate) fn new() -> Self {
        Self {
            outcome: Mutex::new(None),
            handed_off: AtomicBool::new(false),
        }
    }

    pub(crate) fn peek(&self) -> Option<RateLimitOutcome> {
        self.outcome.lock().clone()
    }

    fn set(&self, outcome: RateLimitOutcome) {
        *self.outcome.lock() = Some(outcome);
    }

    /// For a streaming response about to attach a `ReservationAttachment`:
    /// extract the admitted handle (if any), and mark this cell as having
    /// handed off resolution responsibility so `Drop` doesn't race the
    /// attachment's own, correctly-timed resolution. No-op (returns `None`)
    /// if nothing was ever reserved or the request was denied.
    pub(crate) fn take_for_streaming_handoff(&self) -> Option<Arc<SharedReservationHandle>> {
        match self.peek() {
            Some(RateLimitOutcome::Admitted(handle)) => {
                self.handed_off.store(true, Ordering::Release);
                Some(handle)
            }
            _ => None,
        }
    }
}

impl Drop for RateLimitCell {
    fn drop(&mut self) {
        if self.handed_off.load(Ordering::Acquire) {
            return;
        }
        if let Some(RateLimitOutcome::Admitted(handle)) = self.outcome.get_mut().take() {
            // `SharedReservationHandle::abandon_if_open` is private to the
            // rate_limit module; going through the public `ReservationAttachment`
            // RAII wrapper (construct then immediately drop) triggers the
            // same CAS-guarded abandon without needing to widen that
            // visibility.
            drop(ReservationAttachment::new(handle));
        }
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
                    warn!(
                        function = "RateLimitReserveStage::execute",
                        "tenant_request_meta missing; skipping rate-limit reservation"
                    );
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
                    estimated_input_tokens: prep.total_input_token_count() as u32,
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

#[cfg(test)]
mod cell_drop_tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use async_trait::async_trait;
    use uuid::Uuid;

    use super::*;
    use crate::rate_limit::{RateLimitBackend, ReserveOutcome, UsageSettlement};

    #[derive(Default)]
    struct CountingBackend {
        abandon_calls: AtomicUsize,
    }

    #[async_trait]
    impl RateLimitBackend for CountingBackend {
        async fn reserve(&self, _request: ReserveRequest) -> ReserveOutcome {
            ReserveOutcome::Admitted
        }
        async fn settle_success(&self, _request_charge_id: Uuid, _usage: UsageSettlement) {}
        async fn close_reserved_only(&self, _request_charge_id: Uuid) {}
        async fn abandon(&self, _request_charge_id: Uuid) {
            self.abandon_calls.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    fn admitted_cell(backend: &Arc<CountingBackend>) -> RateLimitCell {
        let handle = Arc::new(SharedReservationHandle::new(
            backend.clone() as Arc<dyn RateLimitBackend>,
            Uuid::now_v7(),
        ));
        let cell = RateLimitCell::new();
        cell.set(RateLimitOutcome::Admitted(handle));
        cell
    }

    /// Simulates preemption: the cell is dropped while still holding an
    /// open, un-handed-off reservation (no streaming response body ever
    /// took it over) -- `Drop` must abandon it rather than leak it.
    #[tokio::test]
    async fn drop_without_handoff_abandons_the_reservation() {
        let backend = Arc::new(CountingBackend::default());
        let cell = admitted_cell(&backend);

        drop(cell);
        tokio::task::yield_now().await; // let abandon_if_open's spawned task run

        assert_eq!(backend.abandon_calls.load(AtomicOrdering::SeqCst), 1);
    }

    /// Simulates a streaming response taking ownership via
    /// `take_for_streaming_handoff`: the cell's own `Drop` must not also
    /// abandon, or it would win the CAS race against the real, later
    /// settle and permanently abandon a reservation that should have been
    /// trued up with real usage instead.
    #[tokio::test]
    async fn drop_after_streaming_handoff_does_not_abandon() {
        let backend = Arc::new(CountingBackend::default());
        let cell = admitted_cell(&backend);

        let handed_off = cell.take_for_streaming_handoff();
        assert!(handed_off.is_some());

        drop(cell);
        tokio::task::yield_now().await;

        assert_eq!(
            backend.abandon_calls.load(AtomicOrdering::SeqCst),
            0,
            "a cell that handed off must not also abandon on drop"
        );

        // The handed-off handle is still independently resolvable (e.g. via
        // the streaming path's own later settle, or -- as production code
        // does -- by wrapping it in a `ReservationAttachment`, whose own
        // Drop is the actual cleanup trigger; a bare handle alone does not
        // self-abandon).
        let handle = handed_off.expect("handed off");
        drop(ReservationAttachment::new(handle));
        tokio::task::yield_now().await;
        assert_eq!(backend.abandon_calls.load(AtomicOrdering::SeqCst), 1);
    }

    /// A denied reservation never reaches `Admitted`, so `Drop` has nothing
    /// to abandon.
    #[tokio::test]
    async fn drop_after_denial_does_not_abandon() {
        let backend = Arc::new(CountingBackend::default());
        let cell = RateLimitCell::new();
        cell.set(RateLimitOutcome::Denied);

        drop(cell);
        tokio::task::yield_now().await;

        assert_eq!(backend.abandon_calls.load(AtomicOrdering::SeqCst), 0);
    }
}
