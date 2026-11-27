use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use async_trait::async_trait;
use uuid::Uuid;

use super::*;
use crate::{
    config::types::RouterConfig,
    rate_limit::{
        test_support::RecordingBackend, RateLimitBackend, ReserveOutcome, UsageSettlement,
    },
    tenant::TenantKey,
};

#[derive(Default)]
struct CountingBackend {
    abandon_calls: AtomicUsize,
}

#[async_trait]
impl RateLimitBackend for CountingBackend {
    async fn reserve(&self, _request: ReserveRequest<'_>) -> ReserveOutcome {
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

/// The cell of an admitted reservation, on a backend that records what
/// the cell does with it.
fn recorded_cell() -> (RateLimitCell, Arc<RecordingBackend>) {
    let (handle, backend) = RecordingBackend::reservation();
    let cell = RateLimitCell::new();
    cell.set(RateLimitOutcome::Admitted(handle));
    (cell, backend)
}

/// `settle` resolves an admitted reservation with the counts it is
/// given, once: a second settle and the cell's own `Drop` change
/// nothing. An empty cell (feature off) settles nothing.
#[tokio::test]
async fn settle_resolves_an_admitted_reservation_once() {
    let (cell, backend) = recorded_cell();
    let usage = UsageSettlement {
        actual_input_tokens: 7,
        completion_tokens: 3,
        cached_input_tokens: 0,
        reasoning_tokens: 0,
    };

    cell.settle(usage).await;
    cell.settle(usage).await;
    drop(cell);
    tokio::task::yield_now().await;

    assert_eq!(backend.settlements(), vec![usage]);
    assert_eq!(backend.closes(), 0);

    RateLimitCell::new().settle(usage).await;
}

/// `close_unsettled` closes an admitted reservation with its debit kept,
/// once, and leaves nothing for `Drop` to abandon; a cell that was never
/// admitted has nothing to close.
#[tokio::test]
async fn close_unsettled_closes_an_admitted_reservation_once() {
    let (cell, backend) = recorded_cell();

    cell.close_unsettled().await;
    cell.close_unsettled().await;
    cell.settle(UsageSettlement::default()).await;
    drop(cell);
    tokio::task::yield_now().await;

    assert_eq!(backend.closes(), 1);
    assert!(backend.settlements().is_empty());

    let denied = RateLimitCell::new();
    denied.set(RateLimitOutcome::Denied);
    denied.close_unsettled().await;
    RateLimitCell::new().close_unsettled().await;
}

/// A manager over the in-memory backend: one tenant budget of
/// `tokens_per_minute` under a request budget nothing here spends.
fn local_manager(tokens_per_minute: u32) -> Arc<RateLimitManager> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rate_limit.yaml");
    std::fs::write(
        &path,
        format!(
            "default_policy:\n  tokens_per_minute: {tokens_per_minute}\n  \
                 requests_per_minute: 1000\n"
        ),
    )
    .unwrap();
    let config = RouterConfig::builder()
        .tenant_rate_limit_enabled(true)
        .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
        .build_unchecked();
    RateLimitManager::from_config(&config).unwrap().unwrap()
}

/// `reserve` keeps an admission for the cell's resolvers, and answers a
/// denial with the tenant's 429 and keeps it as the cell's outcome.
#[tokio::test]
async fn reserve_keeps_the_outcome_and_answers_a_denial_with_429() {
    let manager = local_manager(5);
    let tenant = TenantRequestMeta::new(TenantKey::new("auth:tenant"));

    let admitted = RateLimitCell::new();
    assert!(admitted
        .reserve(&manager, &tenant, "model", 4)
        .await
        .is_ok());
    assert!(matches!(
        admitted.peek(),
        Some(RateLimitOutcome::Admitted(_))
    ));

    let denied = RateLimitCell::new();
    let other = TenantRequestMeta::new(TenantKey::new("auth:tenant"));
    let response = denied
        .reserve(&manager, &other, "model", 4)
        .await
        .expect_err("1 of 5 tokens left cannot admit 4");
    assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);
    assert!(matches!(denied.peek(), Some(RateLimitOutcome::Denied)));
}
