//! The Rust TensorRT-LLM servicer against a mock shim on the msgpack wire:
//! the mock decodes the `TrtllmGenerateReqInput` frames the servicer sends
//! and answers with the slim records, as the shim beside the engine does.

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use bytes::Bytes;
use engine_zmq_client::{
    codec::{decode_msgpack, encode_msgpack},
    mock_engine::{
        connect_to_frontend, default_ready_response, MockEngineInput, MockEngineOutput,
        MOCK_DEADLINE,
    },
    protocol::{
        handshake::EngineCoreReadyResponse,
        sglang::output::{BatchTokenIDSlimOutput, MatchedStop},
        trtllm::request::{
            TrtllmGenerateReqInput, TrtllmRequestType, GENERATE_REQ_TAG, GENERATE_WIRE_FIELDS,
            SAMPLING_PARAMS_TAG, SAMPLING_WIRE_FIELDS,
        },
    },
    EngineId,
};
use smg_grpc_client::{
    common_proto as common,
    trtllm_proto::{self as trt, trtllm_service_client::TrtllmServiceClient},
};
use tonic::{transport::Channel, Code};

use super::*;
use crate::{
    kv_events,
    testing::{servicer_config, Bounded},
};

fn model_info() -> TrtllmModelInfo {
    TrtllmModelInfo {
        model_path: "org/model".to_string(),
        served_model_name: "served".to_string(),
        max_input_len: 4096,
        max_seq_len: 8192,
        vocab_size: 1024,
        eos_token_ids: vec![999],
        backend: "pytorch".to_string(),
        trtllm_version: "1.3.0".to_string(),
        tensor_parallel_size: 2,
        pipeline_parallel_size: 1,
        context_parallel_size: 1,
        data_parallel_size: 1,
        guided_decoding_backend: "xgrammar".to_string(),
        structural_tag_vocabulary: "older".to_string(),
        max_running_requests: 64,
        ..Default::default()
    }
}

fn config(dir: &std::path::Path, handshake: &str, model: TrtllmModelInfo) -> TrtllmServicerConfig {
    TrtllmServicerConfig {
        servicer: servicer_config(dir, handshake),
        model,
    }
}

/// The handshake endpoint of one test: an IPC socket under its own
/// directory, so two tests running in parallel never share one.
fn handshake_address(dir: &std::path::Path) -> String {
    format!("ipc://{}", dir.join("handshake").display())
}

/// The layouts the shim declares at its handshake: the field order of the
/// two structs it decodes (what the capability read compares against).
fn declared_layouts() -> BTreeMap<String, Vec<String>> {
    let names = |fields: &[&str]| fields.iter().map(|f| f.to_string()).collect::<Vec<_>>();
    BTreeMap::from([
        (GENERATE_REQ_TAG.to_string(), names(&GENERATE_WIRE_FIELDS)),
        (
            SAMPLING_PARAMS_TAG.to_string(),
            names(&SAMPLING_WIRE_FIELDS),
        ),
    ])
}

fn ready_response() -> EngineCoreReadyResponse {
    EngineCoreReadyResponse {
        positional_layouts: declared_layouts(),
        ..default_ready_response()
    }
}

async fn wait_until(mut condition: impl FnMut() -> bool) {
    for _ in 0..400 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("condition not met within 10s");
}

/// A channel to the servicer whose every request fails after [`MOCK_DEADLINE`]
/// instead of waiting on a servicer that never answers.
async fn grpc_channel(address: impl std::fmt::Display) -> Channel {
    Channel::from_shared(format!("http://{address}"))
        .expect("grpc address")
        .timeout(MOCK_DEADLINE)
        .connect()
        .await
        .expect("grpc client")
}

/// A bound servicer, a handshaken mock shim, and a gRPC client.
struct Harness {
    server: TrtllmServicerServer,
    engine_in: MockEngineInput,
    engine_out: MockEngineOutput,
    client: TrtllmServiceClient<Channel>,
    _dir: tempfile::TempDir,
}

async fn harness_with(model: TrtllmModelInfo, ready: EngineCoreReadyResponse) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = TrtllmServicerServer::start(config(dir.path(), &handshake, model))
        .expect("servicer starts");
    let engine = connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready)
        .await
        .expect("mock shim handshake");
    wait_until(|| server.engine_ready()).await;
    let client = TrtllmServiceClient::new(grpc_channel(server.address()).await);
    let (engine_in, engine_out) = engine.split();
    Harness {
        server,
        engine_in,
        engine_out,
        client,
        _dir: dir,
    }
}

async fn harness(model: TrtllmModelInfo) -> Harness {
    harness_with(model, ready_response()).await
}

/// One engine step for `rid`: `tokens` generated, finished with `finish`
/// (`None` = still running) and the matched stop when there is one.
fn batch(
    rid: &str,
    tokens: Vec<u32>,
    completion_tokens: u32,
    finish: Option<&str>,
    matched: Option<MatchedStop>,
) -> BatchTokenIDSlimOutput {
    BatchTokenIDSlimOutput {
        rids: vec![rid.to_string()],
        output_ids: vec![tokens],
        finished_reasons: vec![finish.unwrap_or_default().to_string()],
        finished_messages: vec![None],
        finished_matched: vec![matched],
        prompt_tokens: vec![3],
        completion_tokens: vec![completion_tokens],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![Vec::new()],
        output_token_logprobs_idx: vec![Vec::new()],
        engine_index: 0,
        num_running: 1,
        num_waiting: 0,
        kv_used_tokens: 10,
        kv_total_tokens: 100,
        finished_status: vec![None],
        ..Default::default()
    }
}

async fn send(engine_out: &mut MockEngineOutput, batch: &BatchTokenIDSlimOutput) {
    engine_out
        .send_frames(vec![Bytes::from(encode_msgpack(batch).unwrap())])
        .await
        .unwrap();
}

/// The next ADD the shim receives: the request and the frames behind it.
async fn recv_add(engine: &mut MockEngineInput) -> (TrtllmGenerateReqInput, Vec<Bytes>) {
    let mut frames = engine.recv_frames().await.unwrap();
    assert_eq!(
        TrtllmRequestType::from_frame(frames[0].as_ref()),
        Some(TrtllmRequestType::Add)
    );
    let request = decode_msgpack(frames[1].as_ref()).unwrap();
    (request, frames.split_off(2))
}

async fn recv_abort(engine: &mut MockEngineInput) -> Vec<String> {
    let frames = engine.recv_frames().await.unwrap();
    assert_eq!(
        TrtllmRequestType::from_frame(frames[0].as_ref()),
        Some(TrtllmRequestType::Abort)
    );
    decode_msgpack(frames[1].as_ref()).unwrap()
}

fn generate_request(id: &str, streaming: bool) -> trt::GenerateRequest {
    trt::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(trt::TokenizedInput {
            original_text: String::new(),
            input_token_ids: vec![1, 2, 3],
            query_token_ids: vec![],
        }),
        sampling_config: Some(trt::SamplingConfig {
            beam_width: 1,
            num_return_sequences: 1,
            seed: Some(7),
            ..Default::default()
        }),
        output_config: Some(trt::OutputConfig {
            exclude_input_from_output: true,
            ..Default::default()
        }),
        max_tokens: 8,
        streaming,
        ..Default::default()
    }
}

fn chunk(response: trt::GenerateResponse) -> trt::GenerateStreamChunk {
    match response.response {
        Some(trt::generate_response::Response::Chunk(chunk)) => chunk,
        other => panic!("expected a chunk, got {other:?}"),
    }
}

fn complete(response: trt::GenerateResponse) -> trt::GenerateComplete {
    match response.response {
        Some(trt::generate_response::Response::Complete(complete)) => complete,
        other => panic!("expected a complete, got {other:?}"),
    }
}

/// Nothing reaches a mock shim that was only handshaken.
async fn assert_engine_idle(engine_in: &mut MockEngineInput) {
    assert!(
        tokio::time::timeout(Duration::from_millis(200), engine_in.recv())
            .await
            .is_err(),
        "nothing should have reached the engine"
    );
}

/// A streaming request: every engine step is a chunk, the finish is the
/// `Complete` with the cumulative ids and the matched stop, under the
/// request's id; the engine is asked to stream whatever the client asked,
/// with the Python servicer's safety defaults filled in; a drain refuses new
/// requests and leaves the one in flight.
#[tokio::test]
async fn streaming_generate_maps_steps_to_chunks_and_a_complete() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r1", true))
        .await
        .expect("generate")
        .into_inner();

    assert_eq!(h.server.active_requests(), 1);
    h.server.set_serving(false);
    let status = h
        .client
        .generate(generate_request("late", true))
        .await
        .expect_err("draining");
    assert_eq!(status.code(), Code::Unavailable);
    assert_eq!(h.server.active_requests(), 1);
    let (request, aux) = recv_add(&mut h.engine_in).await;
    assert_eq!(request.rid, "r1");
    assert_eq!(request.input_ids, vec![1, 2, 3]);
    assert_eq!(request.sampling.max_tokens, 8);
    assert_eq!(request.sampling.seed, Some(7));
    assert_eq!(request.sampling.temperature, Some(1.0));
    assert_eq!(request.sampling.top_p, Some(1.0));
    assert!(request.streaming);
    assert_eq!(request.image_frames, 0);
    assert!(aux.is_empty());

    send(&mut h.engine_out, &batch("r1", vec![10], 1, None, None)).await;
    send(
        &mut h.engine_out,
        &batch(
            "r1",
            vec![11],
            2,
            Some("stop"),
            Some(MatchedStop::TokenId(11)),
        ),
    )
    .await;

    let first = stream.message().bounded().await.unwrap().unwrap();
    assert_eq!(first.request_id, "r1");
    let first = chunk(first);
    assert_eq!(first.token_ids, vec![10]);
    assert_eq!(first.sequence_index, 0);
    assert_eq!(first.prompt_tokens, 3);
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).token_ids,
        vec![11]
    );
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_token_ids, vec![10, 11]);
    assert_eq!(done.prompt_tokens, 3);
    assert_eq!(done.completion_tokens, 2);
    assert_eq!(done.sequence_index, 0);
    assert_eq!(
        done.matched_stop,
        Some(trt::generate_complete::MatchedStop::MatchedTokenId(11))
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(h.server.active_requests(), 0);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Non-streaming delivers exactly the terminal `Complete`; the engine is
/// still asked to stream (one engine path).
#[tokio::test]
async fn non_streaming_generate_delivers_only_the_complete() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r2", false))
        .await
        .expect("generate")
        .into_inner();
    let (request, _) = recv_add(&mut h.engine_in).await;
    assert!(
        request.streaming,
        "the engine streams for both kinds of client"
    );
    send(&mut h.engine_out, &batch("r2", vec![10], 1, None, None)).await;
    send(
        &mut h.engine_out,
        &batch("r2", vec![11, 12], 3, Some("length"), None),
    )
    .await;
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.output_token_ids, vec![10, 11, 12]);
    assert!(done.matched_stop.is_none());
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Images ride behind the payload, one frame each, counted by the array.
#[tokio::test]
async fn images_ride_as_frames_behind_the_request() {
    let mut h = harness(model_info()).await;
    let mut req = generate_request("img", true);
    req.multimodal_input = Some(trt::MultimodalInput {
        image_data: vec![b"first".to_vec(), b"second".to_vec()],
    });
    let _stream = h.client.generate(req).await.expect("generate").into_inner();
    let (request, aux) = recv_add(&mut h.engine_in).await;
    assert_eq!(request.image_frames, 2);
    assert_eq!(
        aux,
        vec![Bytes::from_static(b"first"), Bytes::from_static(b"second")]
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `Abort` ends the stream as the Python servicer ends it (a terminal
/// `abort` completion with the output so far), sends ABORT on the wire, and
/// reports an unknown id as not found.
#[tokio::test]
async fn abort_rpc_cancels_an_in_flight_stream() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r4", true))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("r4", vec![10], 1, None, None)).await;
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).token_ids,
        vec![10]
    );

    let response = h
        .client
        .abort(trt::AbortRequest {
            request_id: "r4".to_string(),
        })
        .await
        .expect("abort")
        .into_inner();
    assert!(response.success);
    assert_eq!(response.message, "Request r4 aborted");
    let aborted = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(aborted.finish_reason, "abort");
    assert_eq!(aborted.output_token_ids, vec![10]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r4".to_string()]);
    let unknown = h
        .client
        .abort(trt::AbortRequest {
            request_id: "never".to_string(),
        })
        .await
        .expect("abort of an unknown id")
        .into_inner();
    assert!(!unknown.success);
    assert_eq!(unknown.message, "Request never not found");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[tokio::test]
async fn duplicate_in_flight_request_ids_are_refused() {
    let mut h = harness(model_info()).await;
    let _stream = h
        .client
        .generate(generate_request("dup", true))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let status = h
        .client
        .generate(generate_request("dup", true))
        .await
        .expect_err("duplicate id");
    assert_eq!(status.code(), Code::AlreadyExists);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `num_return_sequences > 1` fans out to one engine request per sample under
/// the parent's id, each with its own derived seed; every response carries
/// the parent id and its `sequence_index`.
#[tokio::test]
async fn choices_fan_out_under_the_parent_id() {
    let mut h = harness(model_info()).await;
    let mut req = generate_request("p", true);
    req.sampling_config.as_mut().unwrap().num_return_sequences = 2;
    let mut stream = h.client.generate(req).await.expect("generate").into_inner();
    let (first, _) = recv_add(&mut h.engine_in).await;
    let (second, _) = recv_add(&mut h.engine_in).await;
    let mut rids = vec![first.rid.clone(), second.rid.clone()];
    rids.sort();
    assert_eq!(rids, vec!["p-0".to_string(), "p-1".to_string()]);
    let mut seeds = vec![first.sampling.seed, second.sampling.seed];
    seeds.sort();
    assert_eq!(seeds, vec![Some(7), Some(8)]);

    send(
        &mut h.engine_out,
        &batch("p-1", vec![21], 1, Some("stop"), None),
    )
    .await;
    let response = stream.message().bounded().await.unwrap().unwrap();
    assert_eq!(response.request_id, "p");
    let first_out = chunk(response);
    assert_eq!(first_out.sequence_index, 1);
    assert_eq!(first_out.token_ids, vec![21]);
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.sequence_index, 1);
    send(
        &mut h.engine_out,
        &batch("p-0", vec![20], 1, Some("stop"), None),
    )
    .await;
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).sequence_index,
        0
    );
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.sequence_index, 0);
    assert_eq!(done.output_token_ids, vec![20]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `HealthCheck.status` is the Python servicer's `OK` while serving, and the
/// shared message otherwise.
#[tokio::test]
async fn health_reports_ok_while_serving_and_the_drain_after() {
    let mut h = harness(model_info()).await;
    let status = h
        .client
        .health_check(trt::HealthCheckRequest {})
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(status, "OK");
    h.server.set_serving(false);
    let status = h
        .client
        .health_check(trt::HealthCheckRequest {})
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(status, "Draining");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The info RPCs report the launcher's facts, and the handshake's figures
/// where the launcher reported none.
#[tokio::test]
async fn info_rpcs_report_the_launcher_facts_and_the_handshake() {
    let mut h = harness(model_info()).await;
    let model = h
        .client
        .get_model_info(trt::GetModelInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(model.model_id, "served");
    assert_eq!(model.max_input_len, 4096);
    assert_eq!(model.max_seq_len, 8192);
    assert_eq!(model.vocab_size, 1024);
    let info = h
        .client
        .get_server_info(trt::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.version, "1.3.0");
    assert_eq!(info.backend, "pytorch");
    assert_eq!(info.tensor_parallel_size, 2);
    assert_eq!(info.pipeline_parallel_size, 1);
    assert_eq!(info.context_parallel_size, 1);
    assert_eq!(info.world_size, 2);
    // The vocabulary of structural-tag nodes the launcher read off the
    // engine's xgrammar rides along for the router's label.
    assert_eq!(info.structural_tag_vocabulary, "older");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    // A launcher that reported no bounds and no sizes: the handshake's; one
    // that said nothing of the structural-tag vocabulary advertises nothing.
    let bare = TrtllmModelInfo {
        max_input_len: 0,
        max_seq_len: 0,
        served_model_name: String::new(),
        tensor_parallel_size: 0,
        pipeline_parallel_size: 0,
        context_parallel_size: 0,
        trtllm_version: String::new(),
        backend: String::new(),
        structural_tag_vocabulary: String::new(),
        ..model_info()
    };
    let ready = EngineCoreReadyResponse {
        max_model_len: 2048,
        tensor_parallel_size: 4,
        pipeline_parallel_size: 2,
        world_size: 8,
        vllm_version: "trtllm-1.4.0".to_string(),
        ..ready_response()
    };
    let mut h = harness_with(bare, ready).await;
    let model = h
        .client
        .get_model_info(trt::GetModelInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(model.model_id, "org/model");
    assert_eq!(model.max_input_len, 2048);
    assert_eq!(model.max_seq_len, 2048);
    let info = h
        .client
        .get_server_info(trt::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.version, "1.4.0");
    assert_eq!(info.backend, "tensorrt-llm");
    assert_eq!(info.tensor_parallel_size, 4);
    assert_eq!(info.pipeline_parallel_size, 2);
    assert_eq!(info.world_size, 8);
    assert_eq!(info.structural_tag_vocabulary, "");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// What neither servicer serves says why: `Embed`, the KV-event relay
/// without a publisher, the tokenizer bundle without a directory.
#[tokio::test]
async fn the_unimplemented_rpcs_say_why() {
    let mut h = harness(model_info()).await;
    let embed = h
        .client
        .embed(trt::EmbedRequest::default())
        .await
        .expect_err("no embed");
    assert_eq!(embed.code(), Code::Unimplemented);
    assert!(embed.message().contains("Embed"), "{embed}");
    let events = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .expect_err("no publisher");
    assert_eq!(events.code(), Code::Unimplemented);
    assert_eq!(events.message(), kv_events::TRTLLM_DISABLED_MESSAGE);
    let bundle = h
        .client
        .get_tokenizer(common::GetTokenizerRequest::default())
        .await
        .expect_err("no tokenizer dir");
    assert_eq!(bundle.code(), Code::FailedPrecondition);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// An `abort` from the engine side (the shim refused the request while it
/// built the engine's parameters) is the caller's error with the status its
/// HTTP code means.
#[tokio::test]
async fn an_engine_abort_is_the_callers_error() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("bad", true))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let refused = BatchTokenIDSlimOutput {
        finished_messages: vec![Some(
            "embedding_bias length (3) must match vocab_size".into(),
        )],
        finished_status: vec![Some(400)],
        ..batch("bad", vec![], 0, Some("abort"), None)
    };
    send(&mut h.engine_out, &refused).await;
    let error = stream.message().bounded().await.unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("embedding_bias"), "{error}");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// What the wire does not carry is refused before the engine sees anything:
/// beam search, and a constraint the engine's backend does not enforce (the
/// launcher said `none`).
#[tokio::test]
async fn beam_search_and_an_unenforceable_constraint_are_refused_before_the_engine() {
    let mut h = harness(model_info()).await;
    let mut req = generate_request("beam", true);
    req.sampling_config.as_mut().unwrap().beam_width = 4;
    let status = h.client.generate(req).await.expect_err("beam search");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("beam search"), "{status}");
    assert_engine_idle(&mut h.engine_in).await;
    assert_eq!(h.server.active_requests(), 0);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    let mut h = harness(TrtllmModelInfo {
        guided_decoding_backend: "none".to_string(),
        ..model_info()
    })
    .await;
    let mut req = generate_request("guided", true);
    req.guided_decoding = Some(trt::GuidedDecodingParams {
        guide_type: trt::guided_decoding_params::GuideType::JsonSchema as i32,
        guide: "{}".to_string(),
    });
    let status = h
        .client
        .generate(req)
        .await
        .expect_err("no grammar backend");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The capability read at the handshake: a shim whose struct layout moved
/// under this servicer fails the link by name, before any request.
#[tokio::test]
async fn a_shim_whose_layout_moved_fails_the_link_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = TrtllmServicerServer::start(config(dir.path(), &handshake, model_info()))
        .expect("servicer starts");
    let mut layouts = declared_layouts();
    layouts.get_mut(SAMPLING_PARAMS_TAG).unwrap().swap(1, 2);
    let ready = EngineCoreReadyResponse {
        positional_layouts: layouts,
        ..default_ready_response()
    };
    let engine = connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready).await;
    drop(engine);
    let started = Instant::now();
    loop {
        if let Some(error) = server.last_error().unwrap() {
            assert!(
                error.contains("`TrtllmSamplingParams` field 1 is `temperature` on this side"),
                "{error}"
            );
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the link should have failed"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(!server.engine_ready());
    server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[test]
fn structured_outputs_support_follows_the_launcher_facts() {
    use engine_zmq_adapter::{StructuredOutputsBackendConfig, StructuredOutputsSupport};
    let support = |backend: &str| {
        structured_outputs_support(&TrtllmModelInfo {
            guided_decoding_backend: backend.to_string(),
            ..model_info()
        })
    };
    assert_eq!(support("none"), StructuredOutputsSupport::DISABLED);
    assert_eq!(
        support("xgrammar"),
        StructuredOutputsBackendConfig::Pinned(StructuredOutputBackend::Xgrammar).into()
    );
    assert_eq!(
        support("llguidance"),
        StructuredOutputsBackendConfig::Pinned(StructuredOutputBackend::Guidance).into()
    );
    assert_eq!(support(""), StructuredOutputsBackendConfig::Other.into());
    let _ = Arc::new(());
}
