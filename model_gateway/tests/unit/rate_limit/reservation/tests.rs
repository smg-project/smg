use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use async_trait::async_trait;

use super::*;
use crate::rate_limit::backend::{RateLimitBackend, ReserveOutcome, ReserveRequest};

#[derive(Default)]
struct CountingBackend {
    settle_calls: AtomicUsize,
    close_calls: AtomicUsize,
    abandon_calls: AtomicUsize,
}

#[async_trait]
impl RateLimitBackend for CountingBackend {
    async fn reserve(&self, _request: ReserveRequest<'_>) -> ReserveOutcome {
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
