//! Abort-safe reservation lifecycle: [`SharedReservationHandle`] is held
//! by whoever owns a reservation, and [`ReservationAttachment`] rides in a
//! streaming response's body (see `routers::common::stream::GuardedStream`) so an
//! aborted/disconnected stream still releases it.

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use uuid::Uuid;

use super::{backend::UsageSettlement, manager::Backend};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReservationState {
    Open = 0,
    Settled = 1,
    Closed = 2,
    Abandoned = 3,
}

/// Shared handle to an in-flight reservation. Only the first of
/// [`Self::settle_success`], [`Self::close_reserved_only`], or
/// `Self::abandon_if_open` takes effect; later calls are no-ops, which
/// makes `Drop`-triggered `abandon_if_open` safe even after an explicit
/// settle/close.
pub struct SharedReservationHandle {
    state: AtomicU8,
    backend: Backend,
    request_charge_id: Uuid,
}

impl SharedReservationHandle {
    pub(crate) fn new(backend: impl Into<Backend>, request_charge_id: Uuid) -> Self {
        Self {
            state: AtomicU8::new(ReservationState::Open as u8),
            backend: backend.into(),
            request_charge_id,
        }
    }

    fn try_transition(&self, target: ReservationState) -> bool {
        self.state
            .compare_exchange(
                ReservationState::Open as u8,
                target as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Settle from final, authoritative usage.
    pub async fn settle_success(&self, usage: UsageSettlement) {
        if self.try_transition(ReservationState::Settled) {
            self.backend
                .settle_success(self.request_charge_id, usage)
                .await;
        }
    }

    /// Close without settlement (success-with-no-usage, or terminal
    /// failure after admission) — keeps the reserved-input debit.
    pub async fn close_reserved_only(&self) {
        if self.try_transition(ReservationState::Closed) {
            self.backend
                .close_reserved_only(self.request_charge_id)
                .await;
        }
    }

    /// Abandon on drop/cancel/disconnect before an explicit close — keeps
    /// the debit rather than refunding it (already-admitted budget
    /// shouldn't be silently returned).
    fn abandon_if_open(self: &Arc<Self>) {
        if self.try_transition(ReservationState::Abandoned) {
            let backend = self.backend.clone();
            let request_charge_id = self.request_charge_id;
            #[expect(
                clippy::disallowed_methods,
                reason = "best-effort abandon cleanup; must not block the caller's Drop"
            )]
            tokio::spawn(async move {
                backend.abandon(request_charge_id).await;
            });
        }
    }
}

/// RAII guard: held by a streaming response's body (`routers::common::stream::GuardedStream`)
/// so an aborted/disconnected stream still abandons an open reservation
/// instead of leaking it as permanently reserved.
pub struct ReservationAttachment {
    handle: Arc<SharedReservationHandle>,
}

impl ReservationAttachment {
    pub fn new(handle: Arc<SharedReservationHandle>) -> Self {
        Self { handle }
    }
}

impl Drop for ReservationAttachment {
    fn drop(&mut self) {
        self.handle.abandon_if_open();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/rate_limit/reservation/tests.rs"]
mod tests;
