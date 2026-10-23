use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use axum::{
    body::{to_bytes, Body},
    http::StatusCode,
    response::Response,
};

use super::{
    encoder_peer::{register, Peer},
    *,
};

fn request(peer: &Peer, id: &str) -> sglang_proto::GenerateRequest {
    let images = super::media::originals();
    sglang_proto::GenerateRequest {
        request_id: id.into(),
        mm_inputs: Some(sglang_proto::MultimodalInputs {
            image_urls: images.clone(),
            ..Default::default()
        }),
        encode_handoff: Some(sglang_proto::EncodeHandoff {
            request_id: id.into(),
            encoder_urls: vec![peer.url.clone()],
            num_items_assigned: vec![images.len() as u32],
        }),
        ..Default::default()
    }
}

struct Dropped(Arc<AtomicUsize>);
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn epd_sglang_real_encode_rpc_and_original_pd_response() {
    let peer = Peer::new().await;
    let peer_ref = &peer;
    for id in ["attempt-1", "attempt-2"] {
        let req = request(&peer, id);
        dispatch(req, Duration::from_secs(2), |prefill| async move {
            assert_eq!(prefill.request_id, id);
            assert_eq!(prefill.encode_handoff.as_ref().unwrap().request_id, id);
            register(peer_ref, id).await.unwrap();
            Ok::<_, Response>(())
        })
        .await
        .unwrap();
    }
    let requests = peer.state.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].mm_items, super::media::originals());
    assert_eq!(requests[1].req_id, "attempt-2");
    assert_eq!((requests[0].num_parts, requests[0].part_idx), (1, 0));
    let error = dispatch(
        request(&peer, "pd-error"),
        Duration::from_secs(2),
        |_| async {
            Err::<(), _>(
                Response::builder()
                    .status(StatusCode::IM_A_TEAPOT)
                    .header("x-original-pd", "retained")
                    .body(Body::from("exact-pd-body"))
                    .unwrap(),
            )
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error.status(), StatusCode::IM_A_TEAPOT);
    assert_eq!(error.headers()["x-original-pd"], "retained");
    assert_eq!(
        to_bytes(error.into_body(), 1024).await.unwrap(),
        "exact-pd-body"
    );
}

#[tokio::test]
async fn epd_sglang_error_deadline_and_owner_drop_retire_local_pd_future() {
    let peer = Peer::new().await;
    for (fail, expected) in [
        (1, StatusCode::BAD_GATEWAY),
        (2, StatusCode::BAD_GATEWAY),
        (0, StatusCode::GATEWAY_TIMEOUT),
    ] {
        peer.state.fail.store(fail, Ordering::SeqCst);
        let dropped = Arc::new(AtomicUsize::new(0));
        let guard = Dropped(dropped.clone());
        let result = dispatch(
            request(&peer, &format!("error-{fail}")),
            Duration::from_millis(80),
            |_| async move {
                let _guard = guard;
                std::future::pending::<Result<(), Response>>().await
            },
        )
        .await
        .unwrap_err();
        assert_eq!(result.status(), expected);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
    let dropped = Arc::new(AtomicUsize::new(0));
    let guard = Dropped(dropped.clone());
    let count = peer.state.requests.lock().unwrap().len();
    let req = request(&peer, "client-drop");
    let owner = tokio::spawn(dispatch(req, Duration::from_secs(2), |_| async move {
        let _guard = guard;
        std::future::pending::<Result<(), Response>>().await
    }));
    tokio::time::timeout(Duration::from_secs(1), async {
        while peer.state.requests.lock().unwrap().len() == count {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    owner.abort();
    assert!(owner.await.unwrap_err().is_cancelled());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}
