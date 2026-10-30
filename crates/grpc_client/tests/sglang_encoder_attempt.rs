//! Real loopback tonic/protobuf; encoding and the prefill scheduler are stand-ins.

#[path = "common/sglang_encoder.rs"]
mod encoder_peer;

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use encoder_peer::{register, Peer};
use smg_grpc_client::{encode_with_prefill, sglang_proto as sg};
use tonic::Status;

fn prefill(id: &str, urls: Vec<String>, counts: Vec<u32>) -> sg::GenerateRequest {
    sg::GenerateRequest {
        request_id: id.into(),
        mm_inputs: Some(sg::MultimodalInputs {
            image_urls: vec!["a".into()],
            ..Default::default()
        }),
        encode_handoff: Some(sg::EncodeHandoff {
            request_id: id.into(),
            encoder_urls: urls,
            num_items_assigned: counts,
        }),
        ..Default::default()
    }
}
#[tokio::test]
async fn encoder_waits_for_registration_and_concurrent_attempts_keep_final_ids() {
    let first = Peer::new().await.unwrap();
    let second = Peer::new().await.unwrap();
    for id in ["attempt-one", "attempt-two"] {
        let envelope = prefill(
            id,
            vec![
                second.url.clone(),
                "grpc://unused:1".into(),
                first.url.clone(),
            ],
            vec![2, 0, 1],
        );
        let mut envelope = envelope;
        envelope.mm_inputs.as_mut().unwrap().image_urls = vec!["a".into(), "b".into(), "c".into()];
        let (first_ref, second_ref) = (&first, &second);
        let (_, forwarded) = encode_with_prefill(
            vec![second.client.clone(), first.client.clone()],
            envelope.clone(),
            Duration::from_secs(2),
            |actual| async move {
                assert_eq!(actual, envelope);
                let first_done = first_ref.state.finished.notified();
                let second_done = second_ref.state.finished.notified();
                register(first_ref, id).await?;
                register(second_ref, id).await?;
                futures::join!(first_done, second_done);
                Ok(actual)
            },
        )
        .await
        .unwrap();
        assert_eq!(forwarded.request_id, id);
    }
    let seen = second.state.requests.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].mm_items, ["a", "b"]);
    assert_eq!((seen[0].part_idx, seen[0].num_parts), (0, 2));
    assert!(seen[0].embedding_port.is_empty());
    assert_ne!(seen[0].req_id, seen[1].req_id);
}

#[tokio::test]
async fn stale_ids_counts_endpoint_order_and_non_images_reject_before_dispatch() {
    let peer = Peer::new().await.unwrap();
    let calls = AtomicUsize::new(0);
    for mutation in 0..9 {
        let mut generate = prefill("final", vec![peer.url.clone()], vec![1]);
        let encoders = vec![peer.client.clone()];
        match mutation {
            0 => generate.request_id = "new-attempt".into(),
            1 => generate.encode_handoff.as_mut().unwrap().request_id = "old-attempt".into(),
            2 => generate.encode_handoff.as_mut().unwrap().num_items_assigned[0] = 2,
            3 => {
                generate.encode_handoff.as_mut().unwrap().encoder_urls[0] =
                    "grpc://foreign:1".into();
            }
            4 => generate
                .mm_inputs
                .as_mut()
                .unwrap()
                .video_urls
                .push("video".into()),
            5 => generate
                .mm_inputs
                .as_mut()
                .unwrap()
                .image_data
                .push(vec![1]),
            6 => generate
                .mm_inputs
                .as_mut()
                .unwrap()
                .video_data
                .push(vec![1]),
            7 => generate
                .mm_inputs
                .as_mut()
                .unwrap()
                .audio_data
                .push(vec![1]),
            _ => {
                generate
                    .mm_inputs
                    .as_mut()
                    .unwrap()
                    .model_specific_tensors
                    .insert("image_grid_thw".into(), sg::TensorData::default());
            }
        }
        let result = encode_with_prefill(encoders, generate, Duration::from_secs(1), |_| async {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .await;
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(peer.state.requests.lock().unwrap().is_empty());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(peer.state.requests.lock().unwrap().is_empty());
}

struct Dropped(Arc<AtomicBool>);
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn deadline_drops_local_prefill_future_without_detaching_it() {
    let peer = Peer::new().await.unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Dropped(dropped.clone());
    let future = async move {
        let _guard = guard;
        std::future::pending::<Result<(), Status>>().await
    };
    let result = encode_with_prefill(
        vec![peer.client.clone()],
        prefill("bound", vec![peer.url.clone()], vec![1]),
        Duration::from_millis(30),
        |_| future,
    )
    .await;
    assert!(dropped.load(Ordering::SeqCst));
    let error = result.unwrap_err();
    assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
    assert_eq!(error.message(), "SGLang EPD attempt deadline elapsed");
}

#[tokio::test]
async fn encode_error_is_propagated_and_drops_local_prefill_future() {
    let peer = Peer::new().await.unwrap();
    for code in [tonic::Code::Internal, tonic::Code::Cancelled] {
        peer.state.fail.store(
            if code == tonic::Code::Internal { 1 } else { 2 },
            Ordering::SeqCst,
        );
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = Dropped(dropped.clone());
        let future = async move {
            let _guard = guard;
            std::future::pending::<Result<(), Status>>().await
        };
        let result = encode_with_prefill(
            vec![peer.client.clone()],
            prefill("failed", vec![peer.url.clone()], vec![1]),
            Duration::from_secs(2),
            |_| future,
        )
        .await;
        let error = result.unwrap_err();
        assert_eq!(error.code(), code);
        if code == tonic::Code::Cancelled {
            assert_eq!(error.message(), "Timeout expired");
        }
        assert!(dropped.load(Ordering::SeqCst));
    }
}
