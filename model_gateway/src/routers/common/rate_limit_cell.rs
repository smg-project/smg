//! One logical request's tenant rate-limit reservation, shared by every
//! router that reserves: [`RateLimitCell::reserve`] takes it once, before
//! any dispatch, and the cell holds the outcome for the paths that resolve
//! it later -- the settle of a buffered answer, the streaming handoff, and
//! the close of a reservation a failed answer never settled.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use axum::response::Response;
use parking_lot::Mutex;

use crate::{
    middleware::TenantRequestMeta,
    rate_limit::{
        rejection_response, RateLimitManager, Reservation, ReservationAttachment, ReserveRequest,
        SharedReservationHandle, UsageSettlement,
    },
};

/// Outcome of the first reserve attempt for one logical request, cached so
/// later retry attempts don't reserve again.
#[derive(Clone)]
pub(crate) enum RateLimitOutcome {
    Admitted(Arc<SharedReservationHandle>),
    Denied,
}

/// One cell per logical request, holding the reservation outcome for every
/// later resolver (settle, streaming handoff, close-if-unsettled).
/// `parking_lot::Mutex` (not `tokio::sync::Mutex`) is fine: `peek`/`set`
/// never hold the lock across an `.await` -- this is just correct
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
/// this cell would otherwise drop (the router returns the initial streaming
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

    /// Place an outcome from outside [`Self::reserve`], for tests only.
    #[cfg(test)]
    pub(crate) fn set_for_test(&self, outcome: RateLimitOutcome) {
        self.set(outcome);
    }

    fn set(&self, outcome: RateLimitOutcome) {
        *self.outcome.lock() = Some(outcome);
    }

    /// Reserve `estimated_input_tokens` of `tenant_meta`'s budget for
    /// `model_id`, under the request's charge id, and keep the outcome for
    /// this cell's resolvers. A denial is answered with the tenant's 429
    /// ([`rejection_response`]: `Retry-After` when the wait is finite).
    /// Called once per logical request, before any dispatch: a second call
    /// would build a second handle over the same charge id.
    pub(crate) async fn reserve(
        &self,
        manager: &RateLimitManager,
        tenant_meta: &TenantRequestMeta,
        model_id: &str,
        estimated_input_tokens: u32,
    ) -> Result<(), Response> {
        let request = ReserveRequest {
            request_charge_id: tenant_meta.request_charge_id,
            tenant_key: tenant_meta.tenant_key.clone(),
            model_id: Some(model_id),
            estimated_input_tokens,
        };
        match manager.reserve(request).await {
            Reservation::Admitted(handle) => {
                self.set(RateLimitOutcome::Admitted(handle));
                Ok(())
            }
            Reservation::Denied { retry_after_secs } => {
                self.set(RateLimitOutcome::Denied);
                Err(rejection_response(retry_after_secs))
            }
        }
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

    /// [`Self::take_for_streaming_handoff`] on the request's cell, if it has
    /// one (only a unit test's context has none). A handle comes back only
    /// when tenant rate limiting admitted the request: the gRPC streaming
    /// processor settles it with real usage on success, and the safety net
    /// it hands the response body (`ReservationAttachment`'s `Drop`)
    /// abandons it on an early disconnect or error.
    pub(crate) fn stream_reservation(cell: Option<&Self>) -> Option<Arc<SharedReservationHandle>> {
        cell.and_then(Self::take_for_streaming_handoff)
    }

    /// Settle a buffered success's reservation with the engine's counts, if
    /// this logical request reserved one. No-op if nothing was reserved
    /// (feature disabled) or the request was never admitted (preparation
    /// failed before the reserve stage ran). Idempotent: `settle_success` is
    /// a no-op once the handle is resolved.
    pub(crate) async fn settle(&self, usage: UsageSettlement) {
        if let Some(RateLimitOutcome::Admitted(handle)) = self.peek() {
            handle.settle_success(usage).await;
        }
    }

    /// Close a reservation that a failed answer never got the chance to
    /// settle (a preparation failure repeated across every attempt, retries
    /// exhausted, a non-retryable dispatch failure, a stream that broke):
    /// the reserved debit is kept. No-op if nothing was reserved. Safe to
    /// call alongside a success path's own settle or streaming attachment:
    /// `close_reserved_only` is CAS-guarded, the first resolution of a
    /// handle wins.
    pub(crate) async fn close_unsettled(&self) {
        if let Some(RateLimitOutcome::Admitted(handle)) = self.peek() {
            handle.close_reserved_only().await;
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

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/rate_limit_cell/tests.rs"]
mod tests;
