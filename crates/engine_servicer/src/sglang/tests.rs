//! The Rust SGLang servicer against a mock scheduler on the msgpack wire: the
//! mock decodes the `TokenizedGenerateReqInput` frames the servicer sends and
//! answers with `BatchTokenIDSlimOutput` batches, as the SMG plugin inside a
//! headless scheduler does.

use std::time::Duration;

use bytes::Bytes;
use engine_zmq_client::{
    codec::{decode_msgpack, encode_msgpack, OpaqueValue},
    mock_engine::{
        connect_to_frontend, default_ready_response, MockEngineInput, MockEngineOutput,
        MOCK_DEADLINE,
    },
    protocol::sglang::{
        output::{BatchEmbeddingSlimOutput, BatchTokenIDSlimOutput, ControlReplySlim, MatchedStop},
        request::{SglangRequestType, TokenizedEmbeddingReqInput, TokenizedGenerateReqInput},
    },
    EngineId,
};
use prost_types::value::Kind;
use smg_grpc_client::{
    common_proto as common,
    sglang_proto::{self as sg, sglang_scheduler_client::SglangSchedulerClient},
};
use tonic::{transport::Channel, Code};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

use super::*;
use crate::{testing::Bounded, ServicerError};

fn model_info() -> SglangModelInfo {
    SglangModelInfo {
        model_path: "org/model".to_string(),
        served_model_name: "served".to_string(),
        tokenizer_path: "org/model".to_string(),
        is_generation: true,
        model_type: "qwen3".to_string(),
        architectures: vec!["Qwen3ForCausalLM".to_string()],
        max_context_length: 4096,
        vocab_size: 1024,
        eos_token_ids: vec![999],
        pad_token_id: 0,
        bos_token_id: 1,
        default_sampling_params_json: "{\"temperature\":0.7}".to_string(),
        server_args_json: r#"{"model_path":"org/model","tp_size":2,"dp_size":1,"context_length":4096,"pairing_protocol":"p1"}"#.to_string(),
        scheduler_info_json: r#"{"is_generation":true}"#.to_string(),
        sglang_version: "0.5.20".to_string(),
        max_running_requests: 64,
        data_parallel_size: 1,
        ..Default::default()
    }
}

fn config(dir: &std::path::Path, handshake: &str, model: SglangModelInfo) -> SglangServicerConfig {
    SglangServicerConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ipc_base_url: format!("ipc://{}", dir.join("engine").display()),
        handshake_address: handshake.to_string(),
        engine_count: 1,
        tokenizer_dir: None,
        model,
        engine_startup_timeout: Duration::from_secs(10),
    }
}

/// The handshake endpoint of one test: an IPC socket under its own
/// directory. A probed TCP port is not reserved, so two tests running in
/// parallel could pick the same one and a mock engine would handshake with
/// the other test's servicer and wait forever for its INIT.
fn handshake_address(dir: &std::path::Path) -> String {
    format!("ipc://{}", dir.join("handshake").display())
}

/// The port a socket bound to `tcp://127.0.0.1:0` was given.
fn bound_port(endpoint: &str) -> u16 {
    endpoint
        .rsplit(':')
        .next()
        .and_then(|port| port.parse().ok())
        .expect("a bound tcp endpoint ends with its port")
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

/// A bound servicer, a handshaken mock scheduler, and a gRPC client.
struct Harness {
    server: SglangServicerServer,
    engine_in: MockEngineInput,
    engine_out: MockEngineOutput,
    client: SglangSchedulerClient<Channel>,
    _dir: tempfile::TempDir,
}

async fn harness(model: SglangModelInfo) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = SglangServicerServer::start(config(dir.path(), &handshake, model))
        .expect("servicer starts");
    let engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock scheduler handshake");
    wait_until(|| server.engine_ready()).await;
    let client = SglangSchedulerClient::new(grpc_channel(server.address()).await);
    let (engine_in, engine_out) = engine.split();
    Harness {
        server,
        engine_in,
        engine_out,
        client,
        _dir: dir,
    }
}

/// One scheduler step for `rid`: `tokens` generated, finished with `finish`
/// (`None` = still running), carrying a load snapshot when given.
fn batch(
    rid: &str,
    tokens: Vec<u32>,
    completion_tokens: u32,
    finish: Option<&str>,
    load: Option<(u64, u64, u64, u64)>,
) -> BatchTokenIDSlimOutput {
    let (num_running, num_waiting, kv_used_tokens, kv_total_tokens) = load.unwrap_or_default();
    BatchTokenIDSlimOutput {
        rids: vec![rid.to_string()],
        output_ids: vec![tokens],
        finished_reasons: vec![finish.unwrap_or_default().to_string()],
        finished_messages: vec![None],
        finished_matched: vec![None],
        prompt_tokens: vec![3],
        completion_tokens: vec![completion_tokens],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![Vec::new()],
        output_token_logprobs_idx: vec![Vec::new()],
        engine_index: 0,
        num_running,
        num_waiting,
        kv_used_tokens,
        kv_total_tokens,
        ..Default::default()
    }
}

async fn send(engine_out: &mut MockEngineOutput, batch: &BatchTokenIDSlimOutput) {
    engine_out
        .send_frames(vec![Bytes::from(encode_msgpack(batch).unwrap())])
        .await
        .unwrap();
}

async fn recv_add(engine: &mut MockEngineInput) -> TokenizedGenerateReqInput {
    let frames = engine.recv_frames().await.unwrap();
    assert_eq!(
        SglangRequestType::from_frame(frames[0].as_ref()),
        Some(SglangRequestType::Add)
    );
    decode_msgpack(frames[1].as_ref()).unwrap()
}

async fn recv_abort(engine: &mut MockEngineInput) -> Vec<String> {
    let frames = engine.recv_frames().await.unwrap();
    assert_eq!(
        SglangRequestType::from_frame(frames[0].as_ref()),
        Some(SglangRequestType::Abort)
    );
    decode_msgpack(frames[1].as_ref()).unwrap()
}

fn generate_request(id: &str, stream: bool, stops: Vec<String>) -> sg::GenerateRequest {
    sg::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(sg::TokenizedInput {
            input_ids: vec![1, 2, 3],
            original_text: String::new(),
        }),
        sampling_params: Some(sg::SamplingParams {
            max_new_tokens: Some(8),
            stop: stops,
            ..Default::default()
        }),
        logprob_start_len: -1,
        stream,
        ..Default::default()
    }
}

fn chunk(response: sg::GenerateResponse) -> sg::GenerateStreamChunk {
    match response.response {
        Some(sg::generate_response::Response::Chunk(chunk)) => chunk,
        other => panic!("expected a chunk, got {other:?}"),
    }
}

fn complete(response: sg::GenerateResponse) -> sg::GenerateComplete {
    match response.response {
        Some(sg::generate_response::Response::Complete(complete)) => complete,
        other => panic!("expected a complete, got {other:?}"),
    }
}

#[test]
fn config_is_validated_before_binding() {
    let dir = tempfile::tempdir().unwrap();
    let bad_bind = SglangServicerConfig {
        bind_address: "nope".to_string(),
        ..config(dir.path(), "tcp://127.0.0.1:1", model_info())
    };
    assert!(matches!(
        SglangServicerServer::start(bad_bind),
        Err(ServicerError::InvalidConfig(_))
    ));
    let no_engines = SglangServicerConfig {
        engine_count: 0,
        ..config(dir.path(), "tcp://127.0.0.1:1", model_info())
    };
    assert!(matches!(
        SglangServicerServer::start(no_engines),
        Err(ServicerError::InvalidConfig(_))
    ));
    let no_model = SglangServicerConfig {
        model: SglangModelInfo::default(),
        ..config(dir.path(), "tcp://127.0.0.1:1", model_info())
    };
    assert!(matches!(
        SglangServicerServer::start(no_model),
        Err(ServicerError::InvalidConfig(_))
    ));
    // Launcher JSON the Router would read labels from must be an object.
    let bad_args = SglangServicerConfig {
        model: SglangModelInfo {
            server_args_json: "NaN".to_string(),
            ..model_info()
        },
        ..config(dir.path(), "tcp://127.0.0.1:1", model_info())
    };
    assert!(matches!(
        SglangServicerServer::start(bad_args),
        Err(ServicerError::InvalidConfig(message)) if message.contains("server_args_json")
    ));
}

/// Health is NOT_SERVING until the scheduler handshakes, SERVING after, and
/// NOT_SERVING again once the lifecycle owner drains.
#[tokio::test]
async fn health_follows_the_engine_link_and_the_drain_flag() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = SglangServicerServer::start(config(dir.path(), &handshake, model_info()))
        .expect("servicer starts");
    let channel = Channel::from_shared(format!("http://{}", server.address()))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut health = HealthClient::new(channel);
    let status = |response: tonic_health::pb::HealthCheckResponse| response.status;
    let check = HealthCheckRequest {
        service: SERVICE_NAME.to_string(),
    };
    assert_eq!(
        status(health.check(check.clone()).await.unwrap().into_inner()),
        ServingStatus::NotServing as i32
    );
    let mut client = SglangSchedulerClient::connect(format!("http://{}", server.address()))
        .await
        .unwrap();
    let before = client
        .health_check(sg::HealthCheckRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(!before.healthy);
    assert_eq!(before.message, "Engine is starting");

    let _engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock scheduler handshake");
    wait_until(|| server.engine_ready()).await;
    assert_eq!(
        status(health.check(check.clone()).await.unwrap().into_inner()),
        ServingStatus::Serving as i32
    );
    server.set_serving(false);
    assert_eq!(
        status(health.check(check).await.unwrap().into_inner()),
        ServingStatus::NotServing as i32
    );
    let draining = client
        .health_check(sg::HealthCheckRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(draining.message, "Draining");
    server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Streaming: one chunk per scheduler step, then the scheduler's own
/// `Complete` with the accumulated ids, all under the request's id.
#[tokio::test]
async fn streaming_generate_maps_steps_to_chunks_and_a_complete() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r1", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();

    let request = recv_add(&mut h.engine_in).await;
    assert_eq!(request.rid, "r1");
    assert_eq!(request.input_ids, vec![1, 2, 3]);
    assert_eq!(request.sampling_params.max_new_tokens, Some(8));
    assert_eq!(request.sampling_params.n, 1);
    assert!(request.stream);

    send(&mut h.engine_out, &batch("r1", vec![10], 1, None, None)).await;
    send(
        &mut h.engine_out,
        &batch("r1", vec![11], 2, Some("length"), None),
    )
    .await;

    let first = stream.message().bounded().await.unwrap().unwrap();
    assert_eq!(first.request_id, "r1");
    let first = chunk(first);
    assert_eq!(first.token_ids, vec![10]);
    assert_eq!(first.prompt_tokens, 3);
    assert_eq!(
        chunk(stream.message().bounded().await.unwrap().unwrap()).token_ids,
        vec![11]
    );
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.output_ids, vec![10, 11]);
    assert_eq!(done.prompt_tokens, 3);
    assert_eq!(done.completion_tokens, 2);
    assert_eq!(done.index, 0);
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A score-only scheduler result must survive the wire and proto conversion,
/// even though no sampled-token logprobs exist to carry it.
#[tokio::test]
async fn scoring_preserves_selected_logprobs_without_generated_tokens() {
    let mut h = harness(model_info()).await;
    let mut request = generate_request("score", false, Vec::new());
    request.sampling_params.as_mut().unwrap().max_new_tokens = Some(0);
    request.return_logprob = true;
    request.token_ids_logprob = vec![42, 5];
    let mut stream = h.client.generate(request).await.unwrap().into_inner();
    let sent = recv_add(&mut h.engine_in).await;
    assert_eq!(sent.token_ids_logprob, Some(vec![42, 5]));
    assert_eq!(sent.sampling_params.max_new_tokens, Some(0));

    // A scheduler scoring result has no sampled/output tokens. The SMG
    // plugin appends the selected candidate columns at slots 24 and 25.
    let encoded = encode_msgpack(&batch("score", vec![], 0, Some("length"), None)).unwrap();
    let OpaqueValue::Array(mut fields) = decode_msgpack(&encoded).unwrap() else {
        panic!("expected a positional batch");
    };
    fields.truncate(24);
    fields.push(OpaqueValue::Array(vec![OpaqueValue::Array(vec![
        OpaqueValue::Array(vec![OpaqueValue::F64(-0.25), OpaqueValue::F64(-12.0)]),
    ])]));
    fields.push(OpaqueValue::Array(vec![OpaqueValue::Array(vec![
        OpaqueValue::Array(vec![OpaqueValue::from(42u32), OpaqueValue::from(5u32)]),
    ])]));
    h.engine_out
        .send_frames(vec![Bytes::from(
            encode_msgpack(&OpaqueValue::Array(fields)).unwrap(),
        )])
        .await
        .unwrap();

    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert!(done.output_ids.is_empty());
    assert_eq!((done.prompt_tokens, done.completion_tokens), (3, 0));
    let logprobs = done.output_logprobs.expect("prefill-only selected scores");
    assert!(logprobs.token_ids.is_empty() && logprobs.top_logprobs.is_empty());
    assert_eq!(logprobs.token_ids_logprobs.len(), 1);
    assert_eq!(logprobs.token_ids_logprobs[0].token_ids, vec![42, 5]);
    assert_eq!(logprobs.token_ids_logprobs[0].values, vec![-0.25, -12.0]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Selected rows follow token deltas in chunks and accumulate in the terminal
/// response, including when the terminal response is parked behind a chunk.
#[tokio::test]
async fn selected_logprobs_stream_deltas_and_complete_accumulates() {
    let mut h = harness(model_info()).await;
    let request = sg::GenerateRequest {
        return_logprob: true,
        token_ids_logprob: vec![42, 5],
        ..generate_request("scores", true, Vec::new())
    };
    let mut stream = h.client.generate(request).await.unwrap().into_inner();
    recv_add(&mut h.engine_in).await;

    for (token, values, count, finish) in [
        (10, vec![-0.25, -12.0], 1, None),
        (11, vec![-2.0, -0.5], 2, Some("length")),
    ] {
        send(
            &mut h.engine_out,
            &BatchTokenIDSlimOutput {
                output_token_ids_logprobs_val: vec![vec![values.clone()]],
                output_token_ids_logprobs_idx: vec![vec![vec![42, 5]]],
                ..batch("scores", vec![token], count, finish, None)
            },
        )
        .await;
        let item = chunk(stream.message().bounded().await.unwrap().unwrap());
        assert_eq!(item.token_ids, vec![token]);
        let scores = item.output_logprobs.unwrap().token_ids_logprobs;
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].token_ids, vec![42, 5]);
        assert_eq!(
            scores[0].values,
            values
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>()
        );
    }

    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![10, 11]);
    let scores = done.output_logprobs.unwrap().token_ids_logprobs;
    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0].values, vec![-0.25, -12.0]);
    assert_eq!(scores[1].values, vec![-2.0, -0.5]);
    assert!(scores.iter().all(|row| row.token_ids == vec![42, 5]));
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A malformed terminal batch must fail its request, not disappear in the
/// connector's tolerance of isolated undecodable batches.
#[tokio::test]
async fn malformed_selected_logprobs_fail_without_waiting_for_another_batch() {
    let mut h = harness(model_info()).await;
    for (rid, ids) in [("row", vec![vec![vec![42]]]), ("column", vec![])] {
        let request = sg::GenerateRequest {
            return_logprob: true,
            token_ids_logprob: vec![42, 5],
            ..generate_request(rid, false, Vec::new())
        };
        let mut stream = h.client.generate(request).await.unwrap().into_inner();
        recv_add(&mut h.engine_in).await;
        send(
            &mut h.engine_out,
            &BatchTokenIDSlimOutput {
                output_token_ids_logprobs_val: vec![vec![vec![-0.25, -12.0]]],
                output_token_ids_logprobs_idx: ids,
                ..batch(rid, vec![], 0, Some("length"), None)
            },
        )
        .await;
        let error = stream.message().bounded().await.unwrap_err();
        assert_eq!(error.code(), Code::Internal);
        assert!(error.message().contains("selected-token logprobs"));
    }
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Non-streaming delivers exactly the terminal `Complete`.
#[tokio::test]
async fn non_streaming_generate_delivers_only_the_complete() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r2", false, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("r2", vec![10], 1, None, None)).await;
    send(
        &mut h.engine_out,
        &batch("r2", vec![11, 12], 3, Some("stop"), None),
    )
    .await;
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![10, 11, 12]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// String stops ride the request to the scheduler, which keeps its tokenizer
/// and reports the matched string back.
#[tokio::test]
async fn string_stops_reach_the_scheduler_and_its_match_comes_back() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r3", false, vec!["###".to_string()]))
        .await
        .expect("generate")
        .into_inner();
    let request = recv_add(&mut h.engine_in).await;
    assert_eq!(
        request.sampling_params.stop.as_deref(),
        Some(&["###".to_string()][..])
    );
    let mut done = batch("r3", vec![10, 11], 2, Some("stop"), None);
    done.finished_matched = vec![Some(MatchedStop::Text("###".to_string()))];
    send(&mut h.engine_out, &done).await;
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(
        done.matched_stop,
        Some(sg::generate_complete::MatchedStop::MatchedStopStr(
            "###".to_string()
        ))
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `Abort` ends the stream with an `abort` Complete and tells the scheduler;
/// an unknown id is a no-op reported as not found.
#[tokio::test]
async fn abort_rpc_ends_the_stream_and_reaches_the_scheduler() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r4", true, Vec::new()))
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
        .abort(sg::AbortRequest {
            request_id: "r4".to_string(),
            reason: "client left".to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(response.success);
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "abort");
    assert_eq!(done.output_ids, vec![10]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r4".to_string()]);

    let unknown = h
        .client
        .abort(sg::AbortRequest {
            request_id: "nobody".to_string(),
            reason: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!unknown.success, "an unknown id is reported as not found");
    assert!(unknown.message.contains("not found"));
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[tokio::test]
async fn duplicate_request_ids_are_refused() {
    let mut h = harness(model_info()).await;
    let _first = h
        .client
        .generate(generate_request("dup", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let second = h
        .client
        .generate(generate_request("dup", true, Vec::new()))
        .await;
    assert_eq!(second.err().map(|s| s.code()), Some(Code::AlreadyExists));
    let no_ids = h
        .client
        .generate(sg::GenerateRequest {
            request_id: "empty".to_string(),
            ..Default::default()
        })
        .await;
    assert_eq!(no_ids.err().map(|s| s.code()), Some(Code::InvalidArgument));
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `n = 2`: two single-sample scheduler requests under the parent id, demuxed
/// back into two indexed `Complete`s on the one gRPC stream.
#[tokio::test]
async fn n2_fans_out_under_the_parent_id() {
    let mut h = harness(model_info()).await;
    let mut req = generate_request("r5", false, Vec::new());
    if let Some(params) = req.sampling_params.as_mut() {
        params.n = 2;
    }
    let mut stream = h.client.generate(req).await.expect("generate").into_inner();
    let mut rids = vec![
        recv_add(&mut h.engine_in).await.rid,
        recv_add(&mut h.engine_in).await.rid,
    ];
    rids.sort();
    assert_eq!(rids, vec!["r5-0".to_string(), "r5-1".to_string()]);
    let both = BatchTokenIDSlimOutput {
        rids: vec!["r5-0".into(), "r5-1".into()],
        output_ids: vec![vec![10], vec![11]],
        finished_reasons: vec!["stop".into(), "stop".into()],
        finished_messages: vec![None, None],
        finished_matched: vec![None, None],
        prompt_tokens: vec![3, 3],
        completion_tokens: vec![1, 1],
        cached_tokens: vec![0, 0],
        output_token_logprobs_val: vec![vec![], vec![]],
        output_token_logprobs_idx: vec![vec![], vec![]],
        ..Default::default()
    };
    send(&mut h.engine_out, &both).await;
    let mut completes = [
        complete(stream.message().bounded().await.unwrap().unwrap()),
        complete(stream.message().bounded().await.unwrap().unwrap()),
    ];
    completes.sort_by_key(|done| done.index);
    assert_eq!(
        (completes[0].index, &completes[0].output_ids),
        (0, &vec![10])
    );
    assert_eq!(
        (completes[1].index, &completes[1].output_ids),
        (1, &vec![11])
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Sampled-token logprobs and ranked candidates pass through per chunk and
/// cumulatively on the Complete.
#[tokio::test]
async fn logprobs_and_ranked_candidates_pass_through() {
    let mut h = harness(model_info()).await;
    let mut req = generate_request("r6", true, Vec::new());
    req.return_logprob = true;
    req.top_logprobs_num = 2;
    let mut stream = h.client.generate(req).await.expect("generate").into_inner();
    let request = recv_add(&mut h.engine_in).await;
    assert!(request.return_logprob);
    assert_eq!(request.top_logprobs_num, 2);
    let mut step = batch("r6", vec![10], 1, Some("length"), None);
    step.output_token_logprobs_val = vec![vec![-0.5]];
    step.output_token_logprobs_idx = vec![vec![10]];
    step.output_top_logprobs_val = vec![vec![vec![-0.5, -1.5]]];
    step.output_top_logprobs_idx = vec![vec![vec![10, 12]]];
    send(&mut h.engine_out, &step).await;
    let first = chunk(stream.message().bounded().await.unwrap().unwrap());
    let logprobs = first.output_logprobs.expect("chunk logprobs");
    assert_eq!(logprobs.token_logprobs, vec![-0.5]);
    assert_eq!(logprobs.top_logprobs[0].token_ids, vec![10, 12]);
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    let logprobs = done.output_logprobs.expect("complete logprobs");
    assert_eq!(logprobs.token_ids, vec![10]);
    assert_eq!(logprobs.top_logprobs.len(), 1);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A request the scheduler refuses (its 400) is the caller's error on the
/// stream, as the Python servicer's `context.abort(INVALID_ARGUMENT)`.
#[tokio::test]
async fn a_scheduler_refusal_is_the_callers_error() {
    let mut h = harness(model_info()).await;
    let mut stream = h
        .client
        .generate(generate_request("r7", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let mut refusal = batch("r7", vec![], 0, Some("abort"), None);
    refusal.finished_messages = vec![Some("n=2 is not served on this wire".to_string())];
    refusal.finished_status = vec![Some(400)];
    send(&mut h.engine_out, &refusal).await;
    let error = stream.message().bounded().await.expect_err("a status");
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("n=2 is not served"), "{error}");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Model and server info come from the launcher's facts plus the handshake's
/// capacity figures; loads are zero-filled until a batch carries a snapshot.
#[tokio::test]
async fn info_rpcs_report_launcher_facts_and_handshake_figures() {
    let mut h = harness(model_info()).await;
    let ready = default_ready_response();
    let info = h
        .client
        .get_model_info(sg::GetModelInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.model_path, "org/model");
    assert_eq!(info.served_model_name, "served");
    assert_eq!(info.max_context_length, 4096);
    assert_eq!(info.max_req_input_len, 4096);
    assert_eq!(info.eos_token_ids, vec![999]);
    assert!(info.is_generation);
    assert_eq!(info.default_sampling_params_json, "{\"temperature\":0.7}");

    let server = h
        .client
        .get_server_info(sg::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    let args = server.server_args.expect("server_args");
    assert_eq!(
        args.fields["tp_size"].kind,
        Some(Kind::NumberValue(2.0)),
        "the Router reads its labels off server_args"
    );
    assert_eq!(
        args.fields["pairing_protocol"].kind,
        Some(Kind::StringValue("p1".to_string()))
    );
    let scheduler_info = server.scheduler_info.expect("scheduler_info");
    assert_eq!(
        scheduler_info.fields["page_size"].kind,
        Some(Kind::NumberValue(f64::from(
            i32::try_from(ready.block_size).unwrap()
        )))
    );
    assert_eq!(
        scheduler_info.fields["max_running_requests"].kind,
        Some(Kind::NumberValue(64.0))
    );
    assert_eq!(server.sglang_version, "0.5.20");
    assert_eq!(server.server_type, "grpc");
    let expected_capacity = ready
        .kv_cache_size_tokens
        .map(|tokens| i32::try_from(tokens).unwrap())
        .unwrap_or(0);
    assert_eq!(server.max_total_num_tokens, expected_capacity);
    assert_eq!(server.active_requests, 0);

    let idle = h
        .client
        .get_loads(sg::GetLoadsRequest {
            dp_rank: None,
            include: vec!["all".to_string()],
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(idle.dp_rank_count, 1);
    assert_eq!(idle.loads[0].num_running_reqs, 0);
    assert_eq!(idle.loads[0].max_running_requests, 64);
    assert_eq!(idle.version, "0.5.20");

    // A batch with a load tail updates the snapshot the next GetLoads reports.
    let mut stream = h
        .client
        .generate(generate_request("r8", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(
        &mut h.engine_out,
        &batch("r8", vec![10], 1, None, Some((1, 2, 50, 1000))),
    )
    .await;
    chunk(stream.message().bounded().await.unwrap().unwrap());
    let busy = h
        .client
        .get_loads(sg::GetLoadsRequest {
            dp_rank: Some(0),
            include: Vec::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(busy.loads.len(), 1);
    assert_eq!(busy.loads[0].num_running_reqs, 1);
    assert_eq!(busy.loads[0].num_waiting_reqs, 2);
    assert!((busy.loads[0].token_usage - 0.05).abs() < 1e-9);
    assert_eq!(busy.aggregate.unwrap().total_reqs, 3);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A control call as the mock scheduler sees it: `[CONTROL, (call_id,
/// method, args)]`.
async fn recv_control(engine: &mut MockEngineInput) -> (i64, String, Vec<OpaqueValue>) {
    let frames = engine.recv_frames().await.unwrap();
    assert_eq!(
        SglangRequestType::from_frame(frames[0].as_ref()),
        Some(SglangRequestType::Control)
    );
    decode_msgpack(frames[1].as_ref()).unwrap()
}

async fn reply_control(
    engine_out: &mut MockEngineOutput,
    call_id: i64,
    success: bool,
    message: Option<&str>,
) {
    let reply = ControlReplySlim {
        call_id,
        success,
        message: message.map(str::to_string),
        engine_index: 0,
    };
    engine_out
        .send_frames(vec![Bytes::from(encode_msgpack(&reply).unwrap())])
        .await
        .unwrap();
}

/// `FlushCache`, `StartProfile` and `StopProfile` are the scheduler's own
/// control requests, carried by the plugin's control call and answered under
/// the call id; the answer is the scheduler's `success`/`message`.
#[tokio::test]
async fn flush_cache_and_profiling_reach_the_scheduler() {
    let mut h = harness(model_info()).await;
    let (engine_in, engine_out) = (&mut h.engine_in, &mut h.engine_out);
    let (flushed, ()) = tokio::join!(
        h.client
            .flush_cache(common::FlushCacheRequest { timeout_s: 0.0 }),
        async {
            let (call_id, method, args) = recv_control(engine_in).await;
            assert_eq!(method, "flush_cache");
            assert_eq!(args, vec![OpaqueValue::F64(0.0)]);
            reply_control(engine_out, call_id, true, None).await;
        }
    );
    let flushed = flushed.unwrap().into_inner();
    assert!(flushed.success);
    assert_eq!(flushed.message, "Cache flushed successfully");

    let (started, ()) = tokio::join!(
        h.client.start_profile(common::StartProfileRequest {
            output_dir: Some("/tmp/traces".to_string()),
            num_steps: Some(3),
            activities: vec!["CPU".to_string()],
            profile_by_stage: false,
            ..Default::default()
        }),
        async {
            let (call_id, method, args) = recv_control(engine_in).await;
            assert_eq!(method, "start_profile");
            let options = args[0].as_map().unwrap();
            let get = |key: &str| {
                options
                    .iter()
                    .find(|(k, _)| k.as_str() == Some(key))
                    .map(|(_, v)| v.clone())
            };
            assert_eq!(get("output_dir"), Some(OpaqueValue::from("/tmp/traces")));
            assert_eq!(get("num_steps"), Some(OpaqueValue::from(3i64)));
            assert_eq!(get("profile_by_stage"), Some(OpaqueValue::Boolean(false)));
            assert!(get("profile_id").is_some_and(|id| id.as_str().is_some_and(|s| !s.is_empty())));
            // Unset options are left to the scheduler side's defaults.
            assert_eq!(get("with_stack"), None);
            reply_control(engine_out, call_id, false, Some("profiler already running")).await;
        }
    );
    let started = started.unwrap().into_inner();
    assert!(!started.success);
    assert_eq!(started.message, "profiler already running");

    let (stopped, ()) = tokio::join!(
        h.client.stop_profile(common::StopProfileRequest {}),
        async {
            let (call_id, method, args) = recv_control(engine_in).await;
            assert_eq!(method, "stop_profile");
            assert!(args.is_empty());
            reply_control(engine_out, call_id, true, None).await;
        }
    );
    let stopped = stopped.unwrap().into_inner();
    assert!(stopped.success);
    assert_eq!(stopped.message, "Stop profiling succeeded");
    // A negative timeout is refused before anything reaches the scheduler.
    let bad = h
        .client
        .flush_cache(common::FlushCacheRequest { timeout_s: -1.0 })
        .await;
    assert_eq!(bad.err().map(|s| s.code()), Some(Code::InvalidArgument));
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

fn embed_request(id: &str) -> sg::EmbedRequest {
    sg::EmbedRequest {
        request_id: id.to_string(),
        tokenized: Some(sg::TokenizedInput {
            input_ids: vec![1, 2, 3],
            original_text: "hi".to_string(),
        }),
        ..Default::default()
    }
}

/// `Embed` rides the `ADD` frame as the scheduler's embedding request and is
/// answered by its one finished output; a scheduler abort is the caller's
/// error, and a generation model refuses up front.
#[tokio::test]
async fn embed_returns_the_schedulers_vector() {
    let mut h = harness(SglangModelInfo {
        is_generation: false,
        ..model_info()
    })
    .await;
    let (engine_in, engine_out) = (&mut h.engine_in, &mut h.engine_out);
    let (embedded, ()) = tokio::join!(h.client.embed(embed_request("e1")), async {
        let frames = engine_in.recv_frames().await.unwrap();
        assert_eq!(
            SglangRequestType::from_frame(frames[0].as_ref()),
            Some(SglangRequestType::Add)
        );
        let request: TokenizedEmbeddingReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
        assert_eq!(request.rid, "e1");
        assert_eq!(request.input_ids, vec![1, 2, 3]);
        assert_eq!(request.input_text.as_deref(), Some("hi"));
        assert_eq!(request.sampling_params.max_new_tokens, Some(0));
        let batch = BatchEmbeddingSlimOutput {
            rids: vec!["e1".to_string()],
            embeddings: vec![vec![0.25, -0.5]],
            prompt_tokens: vec![3],
            cached_tokens: vec![0],
            finished_reasons: vec!["stop".to_string()],
            finished_messages: vec![None],
            finished_status: vec![None],
            ..Default::default()
        };
        engine_out
            .send_frames(vec![Bytes::from(encode_msgpack(&batch).unwrap())])
            .await
            .unwrap();
    });
    let embedded = embedded.unwrap().into_inner();
    assert_eq!(embedded.embedding, vec![0.25, -0.5]);
    assert_eq!(embedded.embedding_dim, 2);
    assert_eq!(embedded.prompt_tokens, 3);

    let (refused, ()) = tokio::join!(h.client.embed(embed_request("e2")), async {
        let _ = engine_in.recv_frames().await.unwrap();
        let batch = BatchEmbeddingSlimOutput {
            rids: vec!["e2".to_string()],
            embeddings: vec![Vec::new()],
            prompt_tokens: vec![0],
            cached_tokens: vec![0],
            finished_reasons: vec!["abort".to_string()],
            finished_messages: vec![Some("input too long".to_string())],
            finished_status: vec![Some(400)],
            ..Default::default()
        };
        engine_out
            .send_frames(vec![Bytes::from(encode_msgpack(&batch).unwrap())])
            .await
            .unwrap();
    });
    let refused = refused.err().unwrap();
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(refused.message().contains("input too long"));
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    let mut generation = harness(model_info()).await;
    let refused = generation.client.embed(embed_request("e3")).await;
    assert_eq!(refused.err().map(|s| s.code()), Some(Code::InvalidArgument));
    generation
        .server
        .stop(Duration::from_secs(5))
        .expect("clean stop");
}

/// Prompt logprobs and the reasoning-token count ride the appended slim
/// columns: the prompt logprobs go out once, on the first token-bearing
/// chunk, and again on the `Complete`; the reasoning count is on every
/// response.
#[tokio::test]
async fn prompt_logprobs_and_reasoning_tokens_pass_through() {
    let mut h = harness(model_info()).await;
    let request = sg::GenerateRequest {
        return_logprob: true,
        logprob_start_len: 0,
        top_logprobs_num: 1,
        require_reasoning: true,
        ..generate_request("p1", true, Vec::new())
    };
    let mut stream = h.client.generate(request).await.unwrap().into_inner();
    let sent = recv_add(&mut h.engine_in).await;
    assert!(sent.return_logprob && sent.logprob_start_len == 0 && sent.require_reasoning);
    let first = BatchTokenIDSlimOutput {
        reasoning_tokens: vec![1],
        input_token_logprobs_val: vec![vec![None, Some(-0.7), Some(-1.1)]],
        input_token_logprobs_idx: vec![vec![1, 2, 3]],
        input_top_logprobs_val: vec![vec![vec![], vec![-0.7], vec![-1.1]]],
        input_top_logprobs_idx: vec![vec![vec![], vec![2], vec![3]]],
        ..batch("p1", vec![10], 1, None, None)
    };
    send(&mut h.engine_out, &first).await;
    let chunk1 = chunk(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(chunk1.reasoning_tokens, 1);
    let input = chunk1
        .input_logprobs
        .expect("prompt logprobs on the first chunk");
    assert_eq!(input.token_ids, vec![1, 2, 3]);
    assert_eq!(input.token_logprobs[0].value, None);
    assert_eq!(input.token_logprobs[1].value, Some(-0.7));
    assert_eq!(input.top_logprobs.len(), 3);
    assert_eq!(input.top_logprobs[1].token_ids, vec![2]);
    let last = BatchTokenIDSlimOutput {
        reasoning_tokens: vec![2],
        ..batch("p1", vec![11], 2, Some("stop"), None)
    };
    send(&mut h.engine_out, &last).await;
    let chunk2 = chunk(stream.message().bounded().await.unwrap().unwrap());
    assert!(
        chunk2.input_logprobs.is_none(),
        "prompt logprobs go out once"
    );
    assert_eq!(chunk2.reasoning_tokens, 2);
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.reasoning_tokens, 2);
    assert_eq!(
        done.input_logprobs.map(|input| input.token_ids),
        Some(vec![1, 2, 3])
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// With a publisher configured, `SubscribeKvEvents` relays it: the call
/// resolves before any event, batches arrive under the publisher's sequence
/// numbers, and stopping the servicer closes the subscription.
#[tokio::test]
async fn subscribe_kv_events_relays_a_configured_publisher() {
    use zeromq::{prelude::*, PubSocket, SocketEvent};

    use crate::kv_events::golden;

    let mut publisher = PubSocket::new();
    let mut monitor = publisher.monitor();
    let port = bound_port(
        &publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string(),
    );
    let mut model = model_info();
    // A bind wildcard, as SGLang's config spells it; the relay resolves it.
    model.kv_events_endpoint = format!("tcp://*:{port}");
    model.kv_events_topic = "kv".to_string();
    let mut h = harness(model).await;
    let mut stream = tokio::time::timeout(
        Duration::from_secs(5),
        h.client
            .subscribe_kv_events(common::SubscribeKvEventsRequest::default()),
    )
    .await
    .expect("the call resolves before any event is published")
    .expect("subscribe")
    .into_inner();

    // The subscription reaches the publisher a moment after the connect;
    // probe with sequence 0 until a batch comes through.
    let batch1 = golden::bytes(golden::BATCH1);
    let mut first = None;
    for _ in 0..200 {
        publisher
            .send(golden::frame(b"kv", 0, &batch1))
            .await
            .expect("publish");
        if let Ok(item) = tokio::time::timeout(Duration::from_millis(50), stream.message()).await {
            first = Some(item.expect("stream open").expect("a batch"));
            break;
        }
    }
    let first = first.expect("the subscription went live");
    assert_eq!(first.sequence_number, 0);
    assert_eq!(first.events.len(), 4);

    // The relay keeps its publisher subscription for the servicer's
    // lifetime (its history outlives any one stream); stopping the servicer
    // closes it.
    drop(stream);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
    let disconnected = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(event) = monitor.next().bounded().await {
            if matches!(event, SocketEvent::Disconnected(_)) {
                return true;
            }
        }
        false
    })
    .await
    .expect("the publisher notices the stopped servicer in time");
    assert!(disconnected);
}

/// The relay follows the publisher from the servicer's start, before any
/// gateway subscribes: batches published with nobody listening are in its
/// history, and the first subscription gets them as the engine's whole state.
#[tokio::test]
async fn the_relay_subscribes_to_the_publisher_at_boot_before_any_gateway() {
    use zeromq::{prelude::*, PubSocket};

    use crate::kv_events::golden;

    let mut publisher = PubSocket::new();
    let port = bound_port(
        &publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string(),
    );
    let mut model = model_info();
    model.kv_events_endpoint = format!("tcp://*:{port}");
    model.kv_events_topic = "kv".to_string();
    let mut h = harness(model).await;
    let relay = h
        .server
        .state
        .kv_relay
        .clone()
        .expect("a relay for the publisher");
    // The SUB connect is asynchronous: publish sequence 0 until the relay,
    // with no subscriber of its own yet, has taken it (repeats are duplicates).
    let batch1 = golden::bytes(golden::BATCH1);
    for _ in 0..250 {
        publisher
            .send(golden::frame(b"kv", 0, &batch1))
            .await
            .expect("publish");
        if relay.counts().relayed >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        relay.counts().relayed,
        1,
        "the relay took sequence 0 before any gateway subscribed"
    );
    publisher
        .send(golden::frame(b"kv", 1, &golden::bytes(golden::BATCH2)))
        .await
        .expect("publish");
    for _ in 0..250 {
        if relay.counts().relayed >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(relay.counts().relayed, 2);

    // The first gateway gets both from the history: the engine's whole state.
    let mut stream = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .expect("subscribe")
        .into_inner();
    for expected in [0, 1] {
        let batch = tokio::time::timeout(Duration::from_secs(5), stream.message())
            .await
            .expect("a batch in time")
            .expect("stream open")
            .expect("a batch");
        assert_eq!(batch.sequence_number, expected);
    }
    assert_eq!(relay.counts().served_from_history, 1);
    drop(stream);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The relay asks the engine's replay for the batches it missed before its
/// subscription joined: a publisher already at sequence 3 when the servicer
/// starts, whose replay covers 0..=3, leaves the window whole from the
/// publisher's first batch, and the first gateway gets all four from it.
#[tokio::test]
async fn a_publisher_already_counting_when_the_servicer_starts_is_replayed_from_its_start() {
    use zeromq::{prelude::*, PubSocket, RouterSocket, ZmqMessage};

    use crate::kv_events::golden;

    let mut publisher = PubSocket::new();
    let pub_port = bound_port(
        &publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string(),
    );
    let mut router = RouterSocket::new();
    let replay_port = bound_port(
        &router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("replay socket binds")
            .to_string(),
    );
    let mut model = model_info();
    model.kv_events_endpoint = format!("tcp://*:{pub_port}");
    model.kv_events_replay_endpoint = format!("tcp://*:{replay_port}");
    model.kv_events_topic = "kv".to_string();
    let mut h = harness(model).await;
    let relay = h
        .server
        .state
        .kv_relay
        .clone()
        .expect("a relay for the publisher");
    // Sequences 0..=2 went out before the subscription landed: publish 3
    // until the relay asks the replay socket, which it must do from 0.
    let batch1 = golden::bytes(golden::BATCH1);
    let batch2 = golden::bytes(golden::BATCH2);
    let mut request = None;
    for _ in 0..250 {
        publisher
            .send(golden::frame(b"kv", 3, &batch2))
            .await
            .expect("publish");
        if let Ok(message) = tokio::time::timeout(Duration::from_millis(20), router.recv()).await {
            request = Some(message.expect("a replay request"));
            break;
        }
    }
    let request = request.expect("the relay asked the replay socket");
    let frames: Vec<Vec<u8>> = request.iter().map(|frame| frame.to_vec()).collect();
    assert_eq!(frames.len(), 3, "[identity, empty, start]");
    assert_eq!(
        frames[2],
        0u64.to_be_bytes(),
        "asked from the publisher's start"
    );
    for (sequence, payload) in [(0u64, &batch1), (1, &batch2), (2, &batch2), (3, &batch2)] {
        let mut reply = ZmqMessage::from(frames[0].clone());
        reply.push_back(Vec::new().into());
        reply.push_back(b"kv".to_vec().into());
        reply.push_back(sequence.to_be_bytes().to_vec().into());
        reply.push_back(payload.clone().into());
        router.send(reply).await.expect("reply");
    }
    let mut end = ZmqMessage::from(frames[0].clone());
    end.push_back(Vec::new().into());
    end.push_back(Vec::new().into());
    end.push_back([0xff; 8].to_vec().into());
    end.push_back(Vec::new().into());
    router.send(end).await.expect("end marker");
    for _ in 0..250 {
        if relay.counts().relayed >= 4 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Whichever asked first, the start replay or the late join: all four came
    // from the replay socket and nothing is unknown.
    let counts = relay.counts();
    assert_eq!(
        (
            counts.relayed,
            counts.gap_batches_recovered + counts.primed_batches,
            counts.unknown_before_start
        ),
        (4, 4, 0),
        "{counts:?}"
    );

    // The window is the publisher's whole life: the first gateway gets it.
    let mut stream = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .expect("subscribe")
        .into_inner();
    for expected in 0..=3 {
        let batch = tokio::time::timeout(Duration::from_secs(5), stream.message())
            .await
            .expect("a batch in time")
            .expect("stream open")
            .expect("a batch");
        assert_eq!(batch.sequence_number, expected);
    }
    assert_eq!(relay.counts().served_from_history, 1);
    drop(stream);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// What neither servicer serves is reported, not emulated.
#[tokio::test]
async fn unserved_rpcs_report_the_gap() {
    let mut h = harness(model_info()).await;
    let lora = h
        .client
        .load_lo_ra_adapter(sg::LoadLoRaAdapterRequest::default())
        .await;
    assert_eq!(lora.err().map(|s| s.code()), Some(Code::Unimplemented));
    let kv = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await;
    assert_eq!(kv.err().map(|s| s.code()), Some(Code::Unimplemented));
    let tokenizer = h
        .client
        .get_tokenizer(common::GetTokenizerRequest::default())
        .await;
    assert_eq!(
        tokenizer.err().map(|s| s.code()),
        Some(Code::FailedPrecondition),
        "no tokenizer directory was configured"
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A scheduler that never dials in: the server stays up (health NOT_SERVING),
/// the error is reported, and the listener port is released on stop.
#[tokio::test]
async fn startup_timeout_keeps_the_server_up_and_reports_the_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = config(dir.path(), &handshake_address(dir.path()), model_info());
    config.engine_startup_timeout = Duration::from_millis(300);
    let server = SglangServicerServer::start(config).expect("servicer starts");
    wait_until(|| matches!(server.last_error(), Ok(Some(_)))).await;
    assert!(!server.engine_ready());
    assert!(server.running());
    let mut client = SglangSchedulerClient::connect(format!("http://{}", server.address()))
        .await
        .unwrap();
    let health = client
        .health_check(sg::HealthCheckRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(!health.healthy);
    assert_eq!(health.message, "Engine connection failed");
    let generate = client
        .generate(generate_request("late", true, Vec::new()))
        .await;
    assert_eq!(generate.err().map(|s| s.code()), Some(Code::Unavailable));
    server.stop(Duration::from_secs(5)).expect("clean stop");
}
