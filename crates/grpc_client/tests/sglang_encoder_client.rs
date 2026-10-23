//! Actual tonic/protobuf transport; no SGLang model computation is performed.

#[path = "common/sglang_encoder.rs"]
mod encoder_peer;

use std::{sync::atomic::Ordering, time::Duration};

use encoder_peer::{register, Peer};
use smg_grpc_client::sglang_encoder_proto as ep;
use tonic::Code;

#[tokio::test]
async fn encode_and_scheduler_registration_use_the_wire_contract() {
    let peer = Peer::new().await;
    let request = ep::EncodeRequest {
        req_id: "transport-final".into(),
        mm_items: vec!["https://example.test/image.png".into()],
        num_parts: 1,
        ..Default::default()
    };
    let (response, ()) = futures::try_join!(
        peer.client.encode(request.clone(), Duration::from_secs(2)),
        register(&peer, &request.req_id),
    )
    .unwrap();
    assert_eq!(response, ep::EncodeResponse::default());
    assert_eq!(*peer.state.requests.lock().unwrap(), [request]);
    assert!(!peer.state.timeouts.lock().unwrap()[0].is_empty());
    assert!(peer.url.starts_with("grpc://127.0.0.1:"));
}

#[tokio::test]
async fn encode_uses_its_owner_deadline_and_rejects_zero_timeout() {
    let peer = Peer::new().await;
    let request = ep::EncodeRequest {
        req_id: "unregistered".into(),
        ..Default::default()
    };
    let error = peer
        .client
        .encode(request.clone(), Duration::ZERO)
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(peer.state.requests.lock().unwrap().is_empty());
    let error = peer
        .client
        .encode(request, Duration::from_millis(30))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::DeadlineExceeded);
    assert_eq!(error.message(), "SGLang encode deadline elapsed");
}

#[tokio::test]
async fn remote_errors_are_not_relabelled_as_owner_timeouts() {
    let peer = Peer::new().await;
    for (mode, code, message) in [
        (1, Code::Internal, "encode control"),
        (2, Code::Cancelled, "Timeout expired"),
    ] {
        peer.state.fail.store(mode, Ordering::SeqCst);
        let error = peer
            .client
            .encode(ep::EncodeRequest::default(), Duration::from_secs(2))
            .await
            .unwrap_err();
        assert_eq!(error.code(), code);
        assert_eq!(error.message(), message);
    }
}
