//! Abort-safe reservation lifecycle: [`SharedReservationHandle`] is held
//! by whoever owns a reservation, and [`ReservationAttachment`] rides in a
//! streaming response's body (see `worker::GuardedStream`) so an
//! aborted/disconnected stream still releases it.

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use uuid::Uuid;

use super::backend::{RateLimitBackend, UsageSettlement};

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
    backend: Arc<dyn RateLimitBackend>,
    request_charge_id: Uuid,
}

impl SharedReservationHandle {
    pub(crate) fn new(backend: Arc<dyn RateLimitBackend>, request_charge_id: Uuid) -> Self {
        Self {
            state: AtomicU8::new(ReservationState::Open as u8),
            backend,
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
            let backend = Arc::clone(&self.backend);
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

/// RAII guard: held by a streaming response's body (`worker::GuardedStream`)
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
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use async_trait::async_trait;

    use super::*;
    use crate::rate_limit::backend::{ReserveOutcome, ReserveRequest};

    #[derive(Default)]
    struct CountingBackend {
        settle_calls: AtomicUsize,
        close_calls: AtomicUsize,
        abandon_calls: AtomicUsize,
    }

    #[async_trait]
    impl RateLimitBackend for CountingBackend {
        async fn reserve(&self, _request: ReserveRequest) -> ReserveOutcome {
            ReserveOutcome::Admitted
        }

        async fn settle_success(&self, _request_charge_id: Uuid, _usage: UsageSettlement) {
            self.settle_calls.fetch_add(1, AtomicOrdering::SeqCst);
        }

        async fn close_reserved_only(&self, _request_charge_id: Uuid) {
            self.close_calls.fetch_add(1, AtomicOrdering::SeqCst);
        }

        async fn abandon(&self, _request_charge_id: Uuid) {
            self.abandon_calls.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    #[tokio::test]
    async fn settle_then_drop_does_not_also_abandon() {
        let backend = Arc::new(CountingBackend::default());
        let handle = Arc::new(SharedReservationHandle::new(
            backend.clone() as Arc<dyn RateLimitBackend>,
            Uuid::now_v7(),
        ));
        handle.settle_success(UsageSettlement::default()).await;
        assert_eq!(backend.settle_calls.load(AtomicOrdering::SeqCst), 1);

        let attachment = ReservationAttachment::new(handle);
        drop(attachment);
        // Give the spawned abandon task (if any) a chance to run.
        tokio::task::yield_now().await;

        assert_eq!(backend.settle_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(backend.close_calls.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(backend.abandon_calls.load(AtomicOrdering::SeqCst), 0);
    }

    #[tokio::test]
    async fn drop_without_explicit_settle_abandons_exactly_once() {
        let backend = Arc::new(CountingBackend::default());
        let handle = Arc::new(SharedReservationHandle::new(
            backend.clone() as Arc<dyn RateLimitBackend>,
            Uuid::now_v7(),
        ));
        let attachment = ReservationAttachment::new(handle);
        drop(attachment);
        tokio::task::yield_now().await;

        assert_eq!(backend.abandon_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(backend.settle_calls.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(backend.close_calls.load(AtomicOrdering::SeqCst), 0);
    }

    #[tokio::test]
    async fn double_close_is_a_no_op() {
        let backend = Arc::new(CountingBackend::default());
        let handle = Arc::new(SharedReservationHandle::new(
            backend.clone() as Arc<dyn RateLimitBackend>,
            Uuid::now_v7(),
        ));
        handle.close_reserved_only().await;
        handle.close_reserved_only().await;
        handle.settle_success(UsageSettlement::default()).await;

        assert_eq!(backend.close_calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(backend.settle_calls.load(AtomicOrdering::SeqCst), 0);
    }
}
