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
use futures::StreamExt;
use smg_grpc_client::{
    common_proto as common,
    trtllm_proto::{self as trt, trtllm_service_client::TrtllmServiceClient},
};
use tonic::{transport::Channel, Code};

use super::*;
use crate::{
    kv_events,
    testing::{handshake_address, servicer_config, Bounded},
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
        rids: vec![rid.into()],
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
        image_data: vec![Bytes::from_static(b"first"), Bytes::from_static(b"second")],
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

/// A fan-out's open choices end on an `Abort` RPC with their own indexes: the
/// terminal the servicer builds for the second choice carries index 1, as the
/// engine's own completes do.
#[tokio::test]
async fn an_abort_ends_each_choice_of_a_fan_out_with_its_own_index() {
    let mut h = harness(model_info()).await;
    let mut req = generate_request("r7", true);
    req.sampling_config.as_mut().unwrap().num_return_sequences = 2;
    let mut stream = h.client.generate(req).await.expect("generate").into_inner();
    recv_add(&mut h.engine_in).await;
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("r7-1", vec![21], 1, None, None)).await;
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).token_ids,
        vec![21]
    );

    let response = h
        .client
        .abort(trt::AbortRequest {
            request_id: "r7".to_string(),
        })
        .await
        .expect("abort")
        .into_inner();
    assert!(response.success);
    let mut terminals = [
        complete(stream.message().bounded().await.unwrap().unwrap()),
        complete(stream.message().bounded().await.unwrap().unwrap()),
    ];
    terminals.sort_by_key(|done| done.sequence_index);
    let seen: Vec<_> = terminals
        .iter()
        .map(|done| {
            (
                done.sequence_index,
                done.finish_reason.clone(),
                done.output_token_ids.clone(),
            )
        })
        .collect();
    assert_eq!(
        seen,
        vec![
            (0, "abort".to_string(), vec![]),
            (1, "abort".to_string(), vec![21])
        ]
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    let mut aborted = recv_abort(&mut h.engine_in).await;
    while aborted.len() < 2 {
        aborted.extend(recv_abort(&mut h.engine_in).await);
    }
    aborted.sort();
    assert_eq!(aborted, vec!["r7-0".to_string(), "r7-1".to_string()]);
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

/// A shutdown with a stream still open is the engine side going away under
/// the request, not an abort the caller asked for: the stream ends with
/// `UNAVAILABLE` (the status the Router counts and logs as the worker's
/// failure), never with a `Complete` whose finish reason is `abort`.
#[tokio::test]
async fn a_shutdown_ends_an_open_stream_with_unavailable_not_an_abort_completion() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r8", true))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("r8", vec![10], 1, None, None)).await;
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).token_ids,
        vec![10]
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
    let status = stream
        .message()
        .bounded()
        .await
        .expect_err("a shutdown ends the stream with a status, not a completion");
    assert_eq!(status.code(), Code::Unavailable);
    assert!(
        status.message().contains("shutting down"),
        "{}",
        status.message()
    );
    assert!(!h.server.running());
}

/// A request the engine finished on the tick before the shutdown keeps its
/// completion: the finish tick's chunk went out and its `Complete` was
/// parked for the next poll when `stop()` landed, so the stream ends with
/// that `Complete` and no status (nothing was cut), never with `UNAVAILABLE`.
#[tokio::test]
async fn a_shutdown_keeps_the_completion_of_a_choice_the_engine_had_finished() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r11", true))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(
        &mut h.engine_out,
        &batch("r11", vec![10], 1, Some("length"), None),
    )
    .await;
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).token_ids,
        vec![10]
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
    let done = complete(
        stream
            .message()
            .bounded()
            .await
            .expect("the completion the engine had already produced, not a status")
            .expect("a completion, not the end of the stream"),
    );
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.output_token_ids, vec![10]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert!(!h.server.running());
}

/// The engine's own `Complete` went out and the shutdown lands before the
/// poll that would have ended the stream (the gap h2 backpressure opens):
/// the choice has ended and is not open, so the stream ends cleanly, never
/// with `UNAVAILABLE` after a completion. Driven poll by poll with no
/// transport in between, so nothing races; without the choice dropping its
/// engine stream as the `Complete` goes out, this ends with a status.
#[tokio::test]
async fn a_shutdown_after_the_engines_completion_ends_the_stream_cleanly() {
    let mut h = harness(model_info()).await;
    let mut stream = generate::generate(&h.server.state, generate_request("r12", true), None)
        .await
        .expect("generate");
    recv_add(&mut h.engine_in).await;
    send(
        &mut h.engine_out,
        &batch("r12", vec![10], 1, Some("length"), None),
    )
    .await;
    assert_eq!(
        chunk(stream.next().bounded().await.unwrap().unwrap()).token_ids,
        vec![10]
    );
    let done = complete(stream.next().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.output_token_ids, vec![10]);
    // The shutdown lands between the completion and the poll that ends the
    // stream: `cancel_all` is what `stop()` fires first.
    h.server
        .state
        .shared
        .registry
        .cancel_all()
        .expect("cancel_all");
    assert!(
        stream.next().bounded().await.is_none(),
        "a clean end after the completion, not a status"
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// String stops ride the request to the engine, which keeps its tokenizer
/// and reports the matched string back.
#[tokio::test]
async fn string_stops_reach_the_engine_and_its_match_comes_back() {
    let mut h = harness(model_info()).await;
    let mut request = generate_request("r3", false);
    request.stop = vec!["###".to_string()];
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let (request, _) = recv_add(&mut h.engine_in).await;
    assert_eq!(request.sampling.stop, vec!["###".to_string()]);
    send(
        &mut h.engine_out,
        &batch(
            "r3",
            vec![10, 11],
            2,
            Some("stop"),
            Some(MatchedStop::Text("###".to_string())),
        ),
    )
    .await;
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(
        done.matched_stop,
        Some(trt::generate_complete::MatchedStop::MatchedStopStr(
            "###".to_string()
        ))
    );
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
    // The refusal says what is served and what is not: the load records
    // (served since the records-only stream) and the KV events, with the
    // reason, never that the RPC is unimplemented.
    assert!(
        events.message().contains("loads_only")
            && events.message().contains("no publisher endpoint")
            && !events.message().contains("not implemented"),
        "{events}"
    );
    let bundle = h
        .client
        .get_tokenizer(common::GetTokenizerRequest::default())
        .await
        .expect_err("no tokenizer dir");
    assert_eq!(bundle.code(), Code::FailedPrecondition);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The record of a connected rank before any batch has carried a load tail
/// is zero-filled (an idle scheduler, as on the other wires), what the
/// records-only stream reads for it from its first tick; a rank past the
/// connected ones has none.
#[tokio::test]
async fn a_connected_ranks_record_is_zero_filled_before_its_first_load_tail() {
    let h = harness(model_info()).await;
    let source = kv_events::LoadFromState(Arc::downgrade(&h.server.state));
    let record = kv_events::LoadSource::load(&source, Some(0)).expect("rank 0's record");
    assert_eq!(
        (
            record.running_requests,
            record.waiting_requests,
            record.token_usage
        ),
        (0, 0, 0.0)
    );
    assert_eq!(
        kv_events::LoadSource::load(&source, None).map(|record| record.running_requests),
        Some(0)
    );
    assert!(kv_events::LoadSource::load(&source, Some(1)).is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Two connected shims, one of which has reported a load tail: the record
/// of the other stays zero-filled (an idle scheduler, not a gone one), never
/// the sibling's figures; a rank past the connected ones has none. `GetLoads`
/// answers the same figures for every connected rank, the named rank alone
/// when the request names one, and the record pushed for a rank is that
/// rank's poll entry.
#[tokio::test]
async fn an_idle_shims_record_stays_zero_filled_once_a_sibling_reports() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let mut model = model_info();
    model.data_parallel_size = 2;
    let mut config = config(dir.path(), &handshake, model);
    config.servicer.engine_count = 2;
    let server = TrtllmServicerServer::start(config).expect("servicer starts");
    // A KV capacity on the handshake, so the token figures below are real
    // counts (the shim's batch reports 10 of 100 KV tokens), not zeros.
    let ready = |rank: u32| EngineCoreReadyResponse {
        data_parallel_rank: rank,
        kv_cache_size_tokens: Some(100),
        ..ready_response()
    };
    let (first, second) = tokio::join!(
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready(0)),
        connect_to_frontend(&handshake, EngineId::from_engine_index(1), ready(1)),
    );
    let (mut rank0_in, mut rank0_out) = first.expect("shim 0 handshake").split();
    let (mut rank1_in, mut rank1_out) = second.expect("shim 1 handshake").split();
    wait_until(|| server.engine_ready()).await;
    let mut client = TrtllmServiceClient::new(grpc_channel(server.address()).await);

    // The servicer places the request on either shim; that shim answers with
    // a load tail (one running request) and the other stays silent.
    let mut stream = client
        .generate(generate_request("r0", true))
        .await
        .expect("generate")
        .into_inner();
    let served: u32 = tokio::select! {
        _ = recv_add(&mut rank0_in) => 0,
        _ = recv_add(&mut rank1_in) => 1,
    };
    let out = if served == 0 {
        &mut rank0_out
    } else {
        &mut rank1_out
    };
    send(
        out,
        &BatchTokenIDSlimOutput {
            engine_index: served,
            ..batch("r0", vec![10], 1, None, None)
        },
    )
    .await;
    stream.message().bounded().await.unwrap().unwrap();
    let source = kv_events::LoadFromState(Arc::downgrade(&server.state));
    let served_rank = i32::try_from(served).unwrap();
    wait_until(|| {
        kv_events::LoadSource::load(&source, Some(served_rank))
            .is_some_and(|record| record.running_requests == 1)
    })
    .await;
    let idle = kv_events::LoadSource::load(&source, Some(1 - served_rank))
        .expect("the idle shim's record");
    assert_eq!(
        (
            idle.running_requests,
            idle.waiting_requests,
            idle.token_usage
        ),
        (0, 0, 0.0),
        "{idle:?}"
    );
    assert!(kv_events::LoadSource::load(&source, Some(2)).is_none());

    let loads = client
        .get_loads(smg_grpc_client::trtllm_proto::GetLoadsRequest::default())
        .await
        .expect("get_loads")
        .into_inner();
    assert_eq!(loads.dp_rank_count, 2, "{loads:?}");
    assert_eq!(
        loads
            .loads
            .iter()
            .map(|load| (load.dp_rank, load.num_running_reqs, load.num_total_reqs))
            .collect::<Vec<_>>(),
        vec![
            (0, i32::from(served == 0), i32::from(served == 0)),
            (1, i32::from(served == 1), i32::from(served == 1))
        ]
    );
    assert!(loads
        .loads
        .iter()
        .all(|load| load.prefill_throughput.is_none()));
    // The served rank's entry carries the shim's real counts over the
    // handshake's capacity; the idle rank reads the capacity and no use.
    assert_eq!(
        loads
            .loads
            .iter()
            .map(|load| (
                load.dp_rank,
                load.num_used_tokens,
                load.max_total_num_tokens
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, if served == 0 { 10 } else { 0 }, 100),
            (1, if served == 1 { 10 } else { 0 }, 100)
        ]
    );
    // The record the relay pushes for a rank is the poll's entry for it: one
    // field set on both wires, the token counts and the KV capacity included,
    // so a gateway's DP-rank token cache reads the same from a push as from
    // a poll.
    for load in &loads.loads {
        let pushed = kv_events::LoadSource::load(&source, Some(load.dp_rank))
            .expect("the rank's pushed record");
        assert_eq!(
            (pushed.num_used_tokens, pushed.max_total_num_tokens),
            (Some(load.num_used_tokens), Some(load.max_total_num_tokens)),
            "{pushed:?}"
        );
        let mut from_push = openai_protocol::worker::SchedulerLoadSnapshot::from(&pushed);
        from_push.dp_rank = load.dp_rank;
        assert_eq!(
            format!(
                "{:?}",
                openai_protocol::worker::SchedulerLoadSnapshot::from(*load)
            ),
            format!("{from_push:?}")
        );
    }
    let named = client
        .get_loads(smg_grpc_client::trtllm_proto::GetLoadsRequest {
            dp_rank: Some(served_rank),
            include: vec![],
        })
        .await
        .expect("get_loads of one rank")
        .into_inner();
    assert_eq!(
        (
            named.dp_rank_count,
            named.loads.len(),
            named.loads[0].dp_rank
        ),
        (1, 1, served_rank)
    );

    drop(stream);
    server.stop(Duration::from_secs(5)).expect("clean stop");
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

/// After the shim's first output for a request (one of its three prompt
/// tokens cached) and four streamed tokens, with a second request the shim
/// holds waiting, `GetLoads` and the pushed record both carry the rank's hit
/// rate, a throughput and the waiting request's queued token-work, the
/// servicer's own estimate as on every wire; the two readers agree.
#[tokio::test]
async fn the_loads_and_the_record_carry_the_servicers_estimate_of_queued_work_rate_and_hit_rate() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r1", true))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let mut started = batch("r1", vec![10, 11, 12, 13], 4, None, None);
    started.cached_tokens = vec![1];
    started.num_waiting = 1;
    send(&mut h.engine_out, &started).await;
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).token_ids,
        vec![10, 11, 12, 13]
    );
    let second = h
        .client
        .generate(generate_request("r2", true))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;

    let loads = h
        .client
        .get_loads(smg_grpc_client::trtllm_proto::GetLoadsRequest::default())
        .await
        .expect("get_loads")
        .into_inner();
    assert_eq!(loads.loads.len(), 1, "one entry per rank: {loads:?}");
    let rank0 = &loads.loads[0];
    assert_eq!((rank0.num_running_reqs, rank0.num_waiting_reqs), (1, 1));
    assert!(
        (rank0.cache_hit_rate - 1.0 / 3.0).abs() < 1e-9,
        "the rank cached one token in three: {rank0:?}"
    );
    assert!(
        rank0.gen_throughput > 0.0,
        "the rank streamed the tokens: {rank0:?}"
    );
    assert_eq!(
        rank0.num_waiting_uncached_tokens, 2,
        "the waiting request's three prompt tokens at the rank's hit rate: {rank0:?}"
    );
    assert!((rank0.utilization - rank0.token_usage).abs() < f64::EPSILON);

    let source = kv_events::LoadFromState(Arc::downgrade(&h.server.state));
    let record = kv_events::LoadSource::load(&source, Some(0)).expect("rank 0's record");
    assert_eq!((record.running_requests, record.waiting_requests), (1, 1));
    assert_eq!(
        record.waiting_uncached_tokens,
        Some(2),
        "the waiting request's three prompt tokens at the rank's hit rate: {record:?}"
    );
    assert_eq!(
        record.cache_hit_rate,
        Some(rank0.cache_hit_rate),
        "the record and GetLoads read the one estimate: {record:?}"
    );
    assert!(record.gen_throughput > 0.0, "{record:?}");

    drop(second);
    drop(stream);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The production path on this wire, which carries no engine timing: two
/// requests that met no queue on the shim, each answered after its prefill,
/// calibrate the rank's input capacity on the servicer's clock, and
/// `GetLoads` and the pushed record carry it with the figures the load tail
/// does not (queued work, hit rate, generation rate). A request handed over
/// behind one still in flight samples too: its wait is the other's prefill,
/// which the union counts once.
#[tokio::test]
async fn prefill_capacity_is_measured_on_the_servicers_clock_and_published() {
    let mut h = harness(model_info()).await;
    let loads = &h.server.state.shared.loads;
    // Two idle arrivals, each prefilled alone: 3 prompt tokens, the second
    // with one of them cached, each answered 20 ms or more after the submit.
    for (id, cached) in [("c1", 0), ("c2", 1)] {
        let mut stream = h
            .client
            .generate(generate_request(id, true))
            .await
            .unwrap()
            .into_inner();
        recv_add(&mut h.engine_in).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        let mut done = batch(id, vec![10], 1, Some("length"), None);
        done.cached_tokens = vec![cached];
        done.num_running = 0;
        send(&mut h.engine_out, &done).await;
        while stream.message().bounded().await.unwrap().is_some() {}
    }
    assert_eq!((loads.prefill_samples(0), loads.pending()), (2, 0));
    let polled = h
        .client
        .get_loads(trt::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    let rank0 = &polled.loads[0];
    let rate = rank0
        .prefill_throughput
        .expect("two samples calibrate the rank");
    // 5 computed tokens over two prefills of 20 ms or more each.
    assert!(rate > 0.0 && rate <= 125.0, "{rate} tokens per second");
    assert!((rank0.cache_hit_rate - 1.0 / 6.0).abs() < 1e-9);
    assert_eq!(rank0.num_waiting_uncached_tokens, 0);
    let source = kv_events::LoadFromState(Arc::downgrade(&h.server.state));
    let record = kv_events::LoadSource::load(&source, Some(0)).expect("rank 0's record");
    assert_eq!(record.prefill_throughput, Some(rate));
    assert_eq!(record.waiting_uncached_tokens, Some(0));
    assert_eq!(record.cache_hit_rate, Some(rank0.cache_hit_rate));

    // `q1` and `q2` are handed over together: `q2` counts as queued work
    // until it starts, and both sample, `q2` from a submit that waited
    // behind `q1`.
    let mut first = h
        .client
        .generate(generate_request("q1", true))
        .await
        .unwrap()
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let mut second = h
        .client
        .generate(generate_request("q2", true))
        .await
        .unwrap()
        .into_inner();
    recv_add(&mut h.engine_in).await;
    assert_eq!(loads.pending(), 2);
    // A sample must clear the estimator's 1 ms rule: the batches below
    // follow their submits by more than that.
    tokio::time::sleep(Duration::from_millis(5)).await;
    let mut step = batch("q1", vec![10], 1, None, None);
    step.num_waiting = 1;
    send(&mut h.engine_out, &step).await;
    chunk(first.message().bounded().await.unwrap().unwrap());
    assert_eq!(loads.prefill_samples(0), 3, "q1 sampled");
    let busy = h
        .client
        .get_loads(trt::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    // q2's 3 prompt tokens, discounted by the rank's 1-in-9 hit rate.
    assert_eq!(busy.loads[0].num_waiting_uncached_tokens, 3);
    let mut step = batch("q2", vec![11], 1, None, None);
    step.num_running = 2;
    send(&mut h.engine_out, &step).await;
    chunk(second.message().bounded().await.unwrap().unwrap());
    assert_eq!(loads.prefill_samples(0), 4, "q2 sampled behind q1");
    // Every request finishes on the shim's word (a stream dropped mid-flight
    // would send the shim an abort frame).
    for (id, stream) in [("q1", &mut first), ("q2", &mut second)] {
        send(
            &mut h.engine_out,
            &batch(id, Vec::new(), 1, Some("length"), None),
        )
        .await;
        while stream.message().bounded().await.unwrap().is_some() {}
    }
    assert_eq!(loads.pending(), 0);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A burst of 256 requests handed over together and answered together after
/// one prefill batch reads the capacity of the whole batch: every one of
/// them is a sample, merged into one busy period as they overlap, so 256 x 3
/// computed tokens over the batch time (50 ms or more: at most 15,360 tokens
/// per second), never below the run's own tokens over its wall time. A rule
/// that sampled the burst's first arrival alone read one 256th; a ring of 64
/// per-request samples read the newest 64 over the seconds of all.
#[tokio::test]
async fn a_burst_of_overlapping_prefills_reads_the_capacity_of_the_whole_batch() {
    let mut h = harness(model_info()).await;
    // A second channel: one carries at most 200 open streams.
    let mut second = TrtllmServiceClient::new(grpc_channel(h.server.address()).await);
    let loads = &h.server.state.shared.loads;
    let batch_time = Duration::from_millis(50);
    let started = Instant::now();
    let ids: Vec<String> = (0..256).map(|index| format!("b{index}")).collect();
    let mut streams = Vec::new();
    for (index, id) in ids.iter().enumerate() {
        let client = if index % 2 == 0 {
            &mut h.client
        } else {
            &mut second
        };
        streams.push(
            client
                .generate(generate_request(id, true))
                .await
                .unwrap()
                .into_inner(),
        );
        recv_add(&mut h.engine_in).await;
    }
    assert_eq!(loads.pending(), 256);
    tokio::time::sleep(batch_time).await;
    // The batch ends: every request's first (and last) output together.
    for id in &ids {
        let mut done = batch(id, vec![10], 1, Some("length"), None);
        done.num_running = 256;
        send(&mut h.engine_out, &done).await;
    }
    for stream in &mut streams {
        while stream.message().bounded().await.unwrap().is_some() {}
    }
    assert_eq!(loads.pending(), 0);
    let wall = started.elapsed().as_secs_f64();
    let samples = loads.prefill_samples(0);
    let polled = h
        .client
        .get_loads(trt::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    let rate = polled.loads[0]
        .prefill_throughput
        .expect("the burst calibrates the rank");
    assert_eq!(
        samples, 256,
        "every request of the burst is a sample ({rate} tokens per second)"
    );
    // 256 x 3 computed tokens: over a batch of 50 ms or more at most, over
    // the run's whole wall time at least (the union never exceeds it).
    let floor = 768.0 / wall;
    assert!(
        rate <= 15_360.0 && rate >= floor,
        "{rate} tokens per second from {samples} samples, expected within [{floor}, 15360]"
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The record the relay attaches to a batch for a rank is the `GetLoads`
/// entry of that rank, read alone (here before any load tail: the
/// zero-filled entry with the capacity figures and the servicer's own
/// estimate); `None` names rank 0; a rank past the connected ones has no
/// record.
#[tokio::test]
async fn the_relays_rank_record_is_the_get_loads_entry_of_the_rank() {
    use crate::kv_events::RankLoads;

    let mut h = harness(model_info()).await;
    let idle = h
        .client
        .get_loads(trt::GetLoadsRequest {
            dp_rank: Some(0),
            include: Vec::new(),
        })
        .await
        .unwrap()
        .into_inner();
    let load = &idle.loads[0];
    let mut expected = common::EngineLoad::from(load);
    expected.waiting_uncached_tokens =
        Some(u32::try_from(load.num_waiting_uncached_tokens).unwrap_or(0));
    let state = &h.server.state;
    assert_eq!(state.rank_load(Some(0)), Some(expected));
    assert_eq!(state.rank_load(None), state.rank_load(Some(0)));
    assert!(state.rank_load(Some(1)).is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}
