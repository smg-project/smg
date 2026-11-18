use std::{
    future,
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::oneshot;

use super::*;
use crate::{
    policies::prefill_cost::reservation::{PrefillCandidate, Reservation, ReservationLedger},
    routers::{
        common::sse::sse_channel,
        grpc::regular::streaming::tests::eof::{chunk_of_ids, complete, gated_stream},
    },
    worker::BasicWorkerBuilder,
};

fn reservation() -> (Arc<dyn Worker>, Arc<ReservationLedger>, Reservation) {
    let worker: Arc<dyn Worker> =
        Arc::new(BasicWorkerBuilder::new("grpc://cost-worker:30000").build());
    let ledger = Arc::new(ReservationLedger::default());
    let (_, _, lease) = ledger
        .select_and_reserve(&[PrefillCandidate {
            idx: 0,
            worker: worker.url(),
            uncached_tokens: 100,
            active_request_tokens: 1024,
            backend_work_tokens: 0,
            sampled_at: None,
            tokens_per_second: 2000.0,
            kv_penalty_seconds: 0.0,
        }])
        .unwrap();
    (worker, ledger, lease)
}

#[tokio::test]
async fn cost_reservation_survives_empty_frames_replay_and_rewrapping() {
    let (worker, ledger, lease) = reservation();
    let (gate_tx, gate_rx) = oneshot::channel();
    let (engine, server) = gated_stream(
        vec![chunk_of_ids(0, Vec::new()), chunk_of_ids(0, vec![7])],
        gate_rx,
        vec![complete(0, "stop")],
        "0",
    )
    .await;
    let mut stream = dispatch_single_booked(
        async { Ok(engine) },
        Some(&worker),
        100,
        true,
        Instant::now(),
        Some(lease),
    )
    .await
    .unwrap()
    .defer_abort_until_first_item();
    assert_eq!(ledger.pending(worker.url()), 100);
    let _ = stream
        .first_response_within(Instant::now() + Duration::from_secs(1))
        .await;
    assert_eq!(
        ledger.pending(worker.url()),
        100,
        "an empty control chunk is not a token"
    );
    assert!(
        matches!(stream.next().await, Some(Ok(_))),
        "replayed empty frame"
    );
    assert_eq!(ledger.pending(worker.url()), 100);
    assert!(
        matches!(stream.next().await, Some(Ok(_))),
        "first output token"
    );
    assert_eq!(ledger.pending(worker.url()), 0);
    assert_eq!(
        ledger.active(worker.url()),
        1,
        "decode still owns the attempt"
    );
    gate_tx.send(()).unwrap();
    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert!(stream.next().await.is_none());
    assert_eq!(ledger.active(worker.url()), 0);
    server.abort();
}

#[tokio::test]
async fn cost_reservation_is_released_when_a_parked_client_disconnects() {
    let (worker, ledger, lease) = reservation();
    let (_gate_tx, gate_rx) = oneshot::channel();
    let (engine, server) = gated_stream(Vec::new(), gate_rx, vec![complete(0, "stop")], "0").await;
    let stream = dispatch_single_booked(
        async { Ok(engine) },
        Some(&worker),
        100,
        true,
        Instant::now(),
        Some(lease),
    )
    .await
    .unwrap();
    let (sender, receiver) = sse_channel();
    let mut stream = stream.until_client_gone(&sender, "cost-cancel");
    drop(receiver);
    assert!(tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .is_none());
    assert_eq!(ledger.pending(worker.url()), 0);
    assert_eq!(ledger.active(worker.url()), 0);
    server.abort();
}

#[tokio::test]
async fn cost_reservation_ends_prefill_on_a_non_streaming_terminal_without_tokens() {
    let (worker, ledger, lease) = reservation();
    let (_gate_tx, gate_rx) = oneshot::channel();
    let (engine, server) = gated_stream(vec![complete(0, "stop")], gate_rx, Vec::new(), "0").await;
    let mut stream = dispatch_single_booked(
        async { Ok(engine) },
        Some(&worker),
        100,
        false,
        Instant::now(),
        Some(lease),
    )
    .await
    .unwrap();
    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert_eq!(ledger.pending(worker.url()), 0);
    assert_eq!(ledger.active(worker.url()), 1);
    drop(stream);
    assert_eq!(ledger.active(worker.url()), 0);
    server.abort();
}

#[tokio::test]
async fn cost_reservation_is_released_on_upstream_status_without_dropping_the_stream() {
    let (worker, ledger, lease) = reservation();
    let (gate_tx, gate_rx) = oneshot::channel();
    let (engine, server) = gated_stream(Vec::new(), gate_rx, Vec::new(), "13").await;
    let mut stream = dispatch_single_booked(
        async { Ok(engine) },
        Some(&worker),
        100,
        true,
        Instant::now(),
        Some(lease),
    )
    .await
    .unwrap();
    gate_tx.send(()).unwrap();
    assert!(matches!(stream.next().await, Some(Err(_))));
    assert_eq!(ledger.active(worker.url()), 0);
    server.abort();
}

#[tokio::test]
async fn cost_reservation_is_released_if_the_rpc_fails_before_headers() {
    let (worker, ledger, lease) = reservation();
    let result = dispatch_single_booked(
        async { Err(tonic::Status::unavailable("local failure")) },
        Some(&worker),
        100,
        true,
        Instant::now(),
        Some(lease),
    )
    .await;
    assert!(result.is_err());
    assert_eq!(ledger.active(worker.url()), 0);
}

#[tokio::test]
async fn cost_reservation_is_released_if_the_rpc_future_is_cancelled() {
    let (worker, ledger, lease) = reservation();
    let mut call = Box::pin(dispatch_single_booked(
        future::pending::<StreamResult>(),
        Some(&worker),
        100,
        true,
        Instant::now(),
        Some(lease),
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(10), call.as_mut())
            .await
            .is_err()
    );
    assert_eq!(ledger.pending(worker.url()), 100);
    drop(call);
    assert_eq!(ledger.active(worker.url()), 0);
}

#[tokio::test]
async fn cost_reservation_keeps_a_batch_sibling_pending_until_it_starts_or_is_cancelled() {
    let (worker, ledger, lease) = reservation();
    let mut leases = lease.split(2);
    let (_gate_tx, gate_rx) = oneshot::channel();
    let (engine, server) = gated_stream(
        vec![chunk_of_ids(0, vec![7])],
        gate_rx,
        vec![complete(0, "stop")],
        "0",
    )
    .await;
    let mut stream = dispatch_single_booked(
        async { Ok(engine) },
        Some(&worker),
        100,
        true,
        Instant::now(),
        leases.pop(),
    )
    .await
    .unwrap();
    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert_eq!(
        ledger.pending(worker.url()),
        100,
        "other sibling has not started"
    );
    drop(leases);
    assert_eq!(ledger.pending(worker.url()), 0);
    assert_eq!(ledger.active(worker.url()), 1);
    drop(stream);
    assert_eq!(ledger.active(worker.url()), 0);
    server.abort();
}
