//! Synthetic native vLLM gRPC frames, including the actual abort RPC boundary.

use std::{convert::Infallible, fs, path::PathBuf, sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Response, StatusCode},
    routing::post,
    Router,
};
use base64::Engine as _;
use bytes::{BufMut, Bytes};
use futures::{stream, Stream};
use http_body::Frame;
use http_body_util::StreamBody;
use serde_json::Value;
use smg_grpc_client::vllm_engine::VllmEngineClient;
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::timeout,
};
use tonic::codec::Codec;

use super::*;
use crate::{
    diagnostic_capture::{CaptureLimits, CapturePool, SharedCapture},
    routers::grpc::proto_wrapper::{PrefillCachedTokens, ProtoGenerateResponse, ProtoStream},
    worker::{BasicWorkerBuilder, Worker},
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

struct Endpoint {
    client: VllmEngineClient,
    aborts: mpsc::UnboundedReceiver<Vec<String>>,
    task: JoinHandle<()>,
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A real HTTP/2 endpoint records the native client's abort requests; no generation RPC is sent.
#[expect(
    clippy::disallowed_methods,
    reason = "bounded test server; Endpoint aborts its task on drop"
)]
async fn endpoint() -> TestResult<Endpoint> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    let (tx, aborts) = mpsc::unbounded_channel();
    let app = Router::new()
        .route("/vllm.grpc.engine.VllmEngine/Abort", post(abort))
        .with_state(tx);
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve synthetic abort endpoint");
    });
    let client = VllmEngineClient::connect(&url).await?;
    Ok(Endpoint {
        client,
        aborts,
        task,
    })
}

async fn abort(
    State(tx): State<mpsc::UnboundedSender<Vec<String>>>,
    body: Bytes,
) -> Result<Response<Body>, StatusCode> {
    let bytes = body.get(5..).ok_or(StatusCode::BAD_REQUEST)?;
    let request = vllm::AbortRequest::decode(bytes).map_err(|_| StatusCode::BAD_REQUEST)?;
    tx.send(request.request_ids)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let frames: Vec<Result<_, Infallible>> =
        vec![Ok(data(&vllm::AbortResponse {})), Ok(trailers("0"))];
    Response::builder()
        .header("content-type", "application/grpc")
        .body(Body::new(StreamBody::new(stream::iter(frames))))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Encode controlled test messages as length-prefixed native gRPC frames.
fn data(message: &impl Message) -> Frame<Bytes> {
    let encoded = message.encode_to_vec();
    let mut bytes = Vec::with_capacity(encoded.len() + 5);
    bytes.put_u8(0);
    bytes.put_u32(encoded.len().try_into().expect("small test frame"));
    bytes.extend_from_slice(&encoded);
    Frame::data(Bytes::from(bytes))
}

fn trailers(code: &'static str) -> Frame<Bytes> {
    let mut headers = HeaderMap::new();
    headers.insert("grpc-status", HeaderValue::from_static(code));
    if code != "0" {
        headers.insert(
            "grpc-message",
            HeaderValue::from_static("transport%20failed"),
        );
        headers.insert("grpc-status-details-bin", HeaderValue::from_static("AP8="));
    }
    Frame::trailers(headers)
}

/// The production native stream/codec, with controlled frames and the test endpoint's client.
fn native(
    endpoint: &Endpoint,
    body: impl Stream<Item = Result<Frame<Bytes>, tonic::Status>> + Send + 'static,
) -> VllmStream {
    let mut codec =
        tonic_prost::ProstCodec::<vllm::GenerateResponse, vllm::GenerateResponse>::default();
    let decoded = tonic::Streaming::new_response(
        codec.decoder(),
        StreamBody::new(body),
        StatusCode::OK,
        None,
        None,
    );
    VllmStream::new(
        decoded,
        "captured-native".to_owned(),
        endpoint.client.clone(),
    )
    .with_release_drain_bound(Duration::from_millis(100))
}

fn scripted(
    endpoint: &Endpoint,
    messages: &[vllm::GenerateResponse],
    code: &'static str,
) -> VllmStream {
    let frames: Vec<Result<_, tonic::Status>> = messages
        .iter()
        .map(|m| Ok(data(m)))
        .chain(std::iter::once(Ok(trailers(code))))
        .collect();
    native(endpoint, stream::iter(frames))
}

async fn capture(
    record_bytes: usize,
    file_bytes: u64,
) -> TestResult<(tempfile::TempDir, SharedCapture, CapturePool)> {
    let root = tempfile::tempdir()?;
    let pool = CapturePool::new(
        root.path().to_owned(),
        CaptureLimits {
            max_active: 1,
            max_attempts: 2,
            max_retained_bytes: file_bytes * 2,
            file_bytes,
            queue_capacity: 64,
            record_bytes,
        },
    )?;
    let owner = SharedCapture::new(pool.start().await?);
    Ok((root, owner, pool))
}

/// These records come only from this compact encoder, not arbitrary captured RawValue payloads.
async fn records(owner: SharedCapture) -> TestResult<Vec<Value>> {
    let directory = owner.finish().await?;
    let lines = fs::read_to_string(directory.join("records.jsonl"))?;
    let records = lines
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    let mut payloads = Vec::new();
    for (sequence, record) in records.into_iter().enumerate() {
        assert_eq!(record["sequence"], sequence);
        let payload = record["payload"].clone();
        assert_eq!(payload["schema"], "smg.vllm_grpc_ingress.v1");
        assert_eq!(payload["source"], 7);
        assert_eq!(payload["ordinal"], sequence);
        payloads.push(payload);
    }
    Ok(payloads)
}

fn chunk() -> vllm::GenerateResponse {
    vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk {
                token_ids: vec![12, 34, 56],
                cached_tokens: 9,
                index: 3,
                output_logprobs: Some(vllm::OutputLogProbs {
                    token_logprobs: vec![-0.0, f32::INFINITY, f32::from_bits(0x7fc01234)],
                    token_ids: vec![12, 34, 56],
                    top_logprobs: Vec::new(),
                }),
                ..Default::default()
            },
        )),
    }
}

fn decoded(payload: &Value) -> TestResult<vllm::GenerateResponse> {
    let bytes = STANDARD.decode(
        payload["protobuf_base64"]
            .as_str()
            .ok_or("missing protobuf")?,
    )?;
    Ok(vllm::GenerateResponse::decode(bytes.as_slice())?)
}

#[tokio::test]
async fn native_messages_precede_error_mapping_restamping_and_ahead_replay() -> TestResult {
    let endpoint = endpoint().await?;
    let (_root, owner, _pool) = capture(16384, 131072).await?;
    let first = chunk();
    let failed = vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Complete(
            vllm::GenerateComplete {
                finish_reason: "error".to_owned(),
                output_ids: vec![12, 34, 56],
                index: 3,
                cached_tokens: 9,
                kv_transfer_params_json: Some("{\"native\":true}".to_owned()),
                ..Default::default()
            },
        )),
    };
    let mut stream = ProtoStream::capture_vllm(
        scripted(&endpoint, &[first.clone(), failed.clone()], "0"),
        owner.observer(),
        7,
    );
    assert!(stream.first_item_ahead().await.is_some());
    let worker: Arc<dyn Worker> =
        Arc::new(BasicWorkerBuilder::new("grpc://captured-worker:30000").build());
    let mut stream = stream
        .tracked(worker, 0, false)
        .decode_side(PrefillCachedTokens {
            by_index: [(3, 27)].into(),
        });
    assert_eq!(
        stream.next().await.ok_or("missing chunk")??.cached_tokens(),
        27
    );
    let error = match stream.next().await {
        Some(Err(error)) => error,
        _ => return Err("expected mapped engine failure".into()),
    };
    assert_eq!(error.code(), tonic::Code::Internal);
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
    stream.mark_completed();
    drop(stream);
    let records = records(owner).await?;
    assert_eq!(
        records
            .iter()
            .map(|r| r["kind"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["open", "message", "message", "eof", "closed"]
    );
    // Includes non-finite logprob bit patterns, which a lossy JSON conversion could drop.
    assert_eq!(decoded(&records[1])?.encode_to_vec(), first.encode_to_vec());
    assert_eq!(
        decoded(&records[2])?.encode_to_vec(),
        failed.encode_to_vec()
    );
    assert_eq!(records[4]["items"], 2);
    assert_eq!(records[4]["eof"], true);
    assert_eq!(records[4]["completion_marked"], true);
    Ok(())
}

#[tokio::test]
async fn native_status_details_and_eof_are_retained_without_replacing_the_error() -> TestResult {
    let endpoint = endpoint().await?;
    let (_root, owner, _pool) = capture(16384, 131072).await?;
    let mut plain = ProtoStream::Vllm(scripted(&endpoint, &[], "14"));
    let mut observed =
        ProtoStream::capture_vllm(scripted(&endpoint, &[], "14"), owner.observer(), 7);
    let error = match observed.next().await {
        Some(Err(error)) => error,
        _ => return Err("expected native status".into()),
    };
    let ordinary = match plain.next().await {
        Some(Err(error)) => error,
        _ => return Err("expected ordinary status".into()),
    };
    assert_eq!(
        (error.code(), error.message(), error.details()),
        (ordinary.code(), ordinary.message(), ordinary.details())
    );
    assert_eq!(error.details(), &[0, 255]);
    assert!(observed.next().await.is_none());
    assert!(observed.next().await.is_none());
    plain.mark_completed();
    observed.mark_completed();
    drop((plain, observed));
    let records = records(owner).await?;
    assert_eq!(records.len(), 4);
    assert_eq!(records[1]["kind"], "status");
    assert_eq!(records[1]["code"], 14);
    assert_eq!(records[1]["message"], error.message());
    assert_eq!(
        STANDARD.decode(
            records[1]["details_base64"]
                .as_str()
                .ok_or("missing details")?
        )?,
        error.details()
    );
    assert_eq!(records[2]["kind"], "eof");
    assert_eq!(records[3]["items"], 1);
    Ok(())
}

#[tokio::test]
async fn cancelled_poll_does_not_invent_a_message_or_eof() -> TestResult {
    let endpoint = endpoint().await?;
    let (_root, owner, _pool) = capture(16384, 131072).await?;
    let (tx, rx) = oneshot::channel();
    let body = stream::once(async move {
        rx.await.expect("release gated frame");
        Ok(data(&chunk()))
    })
    .chain(stream::once(async { Ok(trailers("0")) }));
    let mut observed = ProtoStream::capture_vllm(native(&endpoint, body), owner.observer(), 7);
    assert!(timeout(Duration::from_millis(20), observed.next())
        .await
        .is_err());
    tx.send(()).map_err(|_| "gated stream dropped")?;
    assert!(matches!(observed.next().await, Some(Ok(_))));
    assert!(observed.next().await.is_none());
    observed.mark_completed();
    drop(observed);
    let records = records(owner).await?;
    assert_eq!(
        records
            .iter()
            .map(|r| r["kind"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["open", "message", "eof", "closed"]
    );
    Ok(())
}

#[tokio::test]
async fn capture_size_and_writer_failures_do_not_change_native_output() -> TestResult {
    let endpoint = endpoint().await?;
    for (record_bytes, file_bytes) in [(256, 8192), (16384, 512)] {
        let (_root, owner, _pool) = capture(record_bytes, file_bytes).await?;
        let directory: PathBuf = owner.directory().to_owned();
        let large = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Chunk(
                vllm::GenerateStreamChunk {
                    token_ids: vec![1234; 2048],
                    ..Default::default()
                },
            )),
        };
        let mut observed = ProtoStream::capture_vllm(
            scripted(&endpoint, &[large.clone()], "0"),
            owner.observer(),
            7,
        );
        let received = match observed.next().await {
            Some(Ok(ProtoGenerateResponse::Vllm(message))) => message,
            _ => return Err("capture changed native output".into()),
        };
        assert_eq!(received.encode_to_vec(), large.encode_to_vec());
        assert!(observed.next().await.is_none());
        observed.mark_completed();
        drop(observed);
        assert!(owner.finish().await.is_err());
        assert!(!directory.join("complete.json").exists());
    }
    Ok(())
}

#[tokio::test]
async fn drop_preserves_immediate_deferred_and_suppressed_native_abort() -> TestResult {
    let mut endpoint = endpoint().await?;
    for deferred in [false, true] {
        let (_root, owner, _pool) = capture(16384, 131072).await?;
        let (tx, rx) = oneshot::channel();
        let body = stream::once(async move {
            let _ = rx.await; // An immediate abort may release the body before the gate fires.
            Ok(data(&chunk()))
        })
        .chain(stream::once(async { Ok(trailers("0")) }));
        let observed = ProtoStream::capture_vllm(native(&endpoint, body), owner.observer(), 7);
        let observed = if deferred {
            observed.defer_abort_until_first_item()
        } else {
            observed
        };
        drop(observed);
        if deferred {
            assert!(timeout(Duration::from_millis(100), endpoint.aborts.recv())
                .await
                .is_err());
            tx.send(())
                .map_err(|_| "deferred stream did not retain its body")?;
        }
        assert_eq!(
            timeout(Duration::from_secs(5), endpoint.aborts.recv())
                .await?
                .ok_or("abort endpoint closed")?,
            ["captured-native"]
        );
        // Immediate abort may still be draining its body; closing the gate releases it.
        let records = records(owner).await?;
        assert_eq!(records.len(), 2);
        assert_eq!(records[1]["kind"], "closed");
        assert_eq!(records[1]["items"], 0);
        assert_eq!(records[1]["eof"], false);
        assert_eq!(records[1]["completion_marked"], false);
    }
    let (_root, owner, _pool) = capture(16384, 131072).await?;
    let mut observed =
        ProtoStream::capture_vllm(scripted(&endpoint, &[chunk()], "0"), owner.observer(), 7)
            .defer_abort_until_first_item();
    observed.mark_completed();
    drop(observed);
    assert!(timeout(Duration::from_millis(100), endpoint.aborts.recv())
        .await
        .is_err());
    let records = records(owner).await?;
    assert_eq!(records[1]["items"], 0);
    assert_eq!(records[1]["completion_marked"], true);
    Ok(())
}

#[tokio::test]
async fn abandoned_capture_owner_does_not_stop_the_transport() -> TestResult {
    let endpoint = endpoint().await?;
    let (root, owner, pool) = capture(16384, 131072).await?;
    let directory = owner.directory().to_owned();
    let mut observed =
        ProtoStream::capture_vllm(scripted(&endpoint, &[chunk()], "0"), owner.observer(), 7);
    drop(owner);
    assert!(matches!(observed.next().await, Some(Ok(_))));
    assert!(observed.next().await.is_none());
    observed.mark_completed();
    drop(observed);
    timeout(Duration::from_secs(5), async {
        loop {
            match pool.start().await {
                Ok(session) => return session.finish().await.map(|_| ()),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                Err(error) => return Err(error),
            }
        }
    })
    .await??;
    assert!(!directory.join("complete.json").exists());
    drop(root);
    Ok(())
}
