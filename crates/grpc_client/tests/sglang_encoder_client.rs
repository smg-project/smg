//! Actual tonic/protobuf transport; no SGLang model computation is performed.

#[path = "common/sglang_encoder.rs"]
mod encoder_peer;

use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use encoder_peer::{register, Peer};
use smg_grpc_client::{
    is_request_timeout, sglang_encoder_proto as ep, RequestDeadline, SglangEncoderClient,
    TraceInjector,
};
use tonic::Code;

#[tokio::test]
async fn encode_and_scheduler_registration_use_the_wire_contract() {
    let peer = Peer::new().await.unwrap();
    let request = ep::EncodeRequest {
        req_id: "transport-final".into(),
        mm_items: vec!["https://example.test/image.png".into()],
        num_parts: 1,
        ..Default::default()
    };
    let (response, ()) = futures::try_join!(
        peer.client.encode(
            request.clone(),
            RequestDeadline::after(Duration::from_secs(2))
        ),
        register(&peer, &request.req_id),
    )
    .unwrap();
    assert_eq!(response, ep::EncodeResponse::default());
    assert_eq!(*peer.state.requests.lock().unwrap(), [request]);
    assert!(peer.state.timeouts.lock().unwrap()[0].is_empty());
}

#[tokio::test]
async fn encode_uses_its_owner_deadline_and_marks_local_expiration() {
    let peer = Peer::new().await.unwrap();
    let request = ep::EncodeRequest {
        req_id: "unregistered".into(),
        ..Default::default()
    };
    let error = peer
        .client
        .encode(request.clone(), RequestDeadline::after(Duration::ZERO))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::DeadlineExceeded);
    assert!(is_request_timeout(&error));
    assert!(peer.state.requests.lock().unwrap().is_empty());
    let error = peer
        .client
        .encode(request, RequestDeadline::after(Duration::from_millis(30)))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::DeadlineExceeded);
    assert!(is_request_timeout(&error));
}

#[tokio::test]
async fn a_rounded_transport_timeout_cannot_beat_the_owner_deadline() {
    let peer = Peer::new().await.unwrap();
    tokio::time::pause();
    // Keep the runtime awake so real loopback I/O completes without advancing
    // its paused clock to a timer automatically. Both sets abort on failure.
    let mut awake = tokio::task::JoinSet::new();
    awake.spawn(async {
        loop {
            tokio::task::yield_now().await;
        }
    });
    // grpc-timeout has eight digits: this duration loses half a second when
    // encoded in seconds. Virtual time makes that real tonic race bounded.
    let deadline = RequestDeadline::after(Duration::from_millis(100_000_500));
    let client = peer.client.clone();
    let mut calls = tokio::task::JoinSet::new();
    calls.spawn(async move { client.encode(ep::EncodeRequest::default(), deadline).await });
    let entered_by = std::time::Instant::now() + Duration::from_secs(2);
    while peer.state.requests.lock().unwrap().is_empty() {
        assert!(
            std::time::Instant::now() < entered_by,
            "encoder peer did not receive request"
        );
        tokio::task::yield_now().await;
    }
    tokio::time::advance(Duration::from_secs(100_000) + Duration::from_millis(20)).await;
    for _ in 0..128 {
        tokio::task::yield_now().await;
    }
    assert!(tokio::time::Instant::now() < deadline.at());
    let premature = calls.try_join_next();
    assert!(
        premature.is_none(),
        "Encode ended before its owner deadline: {premature:?}; grpc-timeout={:?}",
        peer.state.timeouts.lock().unwrap()
    );
    tokio::time::advance(Duration::from_millis(481)).await;
    let error = calls.join_next().await.unwrap().unwrap().unwrap_err();
    assert_eq!(error.code(), Code::DeadlineExceeded);
    assert!(is_request_timeout(&error));
}

#[tokio::test]
async fn remote_errors_are_not_relabelled_as_owner_timeouts() {
    let peer = Peer::new().await.unwrap();
    for (mode, code, message) in [
        (1, Code::Internal, "encode control"),
        (2, Code::Cancelled, "Timeout expired"),
        (3, Code::DeadlineExceeded, "remote encoder deadline"),
    ] {
        peer.state.fail.store(mode, Ordering::SeqCst);
        let error = peer
            .client
            .encode(
                ep::EncodeRequest::default(),
                RequestDeadline::after(Duration::from_secs(2)),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code(), code);
        assert_eq!(error.message(), message);
        assert!(!is_request_timeout(&error));
    }
}

struct EncoderTrace;

impl TraceInjector for EncoderTrace {
    fn inject(
        &self,
        metadata: &mut tonic::metadata::MetadataMap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        metadata.insert(
            "traceparent",
            "00-12345678901234567890123456789012-1234567890123456-01".parse()?,
        );
        Ok(())
    }
}

#[tokio::test]
async fn encode_propagates_trace_context_and_accepts_a_large_owner_deadline() {
    let peer = Peer::new().await.unwrap();
    let client =
        SglangEncoderClient::connect_with_trace_injector(&peer.url, Arc::new(EncoderTrace))
            .await
            .unwrap();
    let request = ep::EncodeRequest {
        req_id: "trace".into(),
        ..Default::default()
    };
    futures::try_join!(
        client.encode(request, RequestDeadline::after(Duration::MAX)),
        register(&peer, "trace"),
    )
    .unwrap();
    assert_eq!(
        peer.state.traceparents.lock().unwrap().as_slice(),
        ["00-12345678901234567890123456789012-1234567890123456-01"]
    );
}
