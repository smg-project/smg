//! The Rust TokenSpeed servicer against a mock scheduler on the msgpack wire:
//! the mock decodes the `TokenizedGenerateReqInput` frames the servicer sends
//! and answers with `BatchTokenIDOutSlim` batches, as the headless scheduler
//! does.

use std::{
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
        tokenspeed::{
            output::BatchTokenIDOutSlim,
            request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
        },
    },
    EngineId,
};
use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
use prost_types::value::Kind;
use smg_grpc_client::{
    common_proto as common,
    tokenspeed_proto::{self as ts, token_speed_scheduler_client::TokenSpeedSchedulerClient},
};
use tonic::{transport::Channel, Code};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

use super::*;
use crate::{kv_events, testing::Bounded, ServicerError};

fn model_info() -> TokenSpeedModelInfo {
    TokenSpeedModelInfo {
        model_path: "org/model".to_string(),
        served_model_name: "served".to_string(),
        tokenizer_path: "org/model".to_string(),
        model_type: "qwen3".to_string(),
        architectures: vec!["Qwen3ForCausalLM".to_string()],
        max_context_length: 4096,
        vocab_size: 1024,
        eos_token_ids: vec![999],
        pad_token_id: 0,
        bos_token_id: 1,
        default_sampling_params_json: "{\"temperature\":0.7}".to_string(),
        server_args_json:
            r#"{"model":"org/model","max_num_seqs":64,"dp_size":2,"pairing_protocol":"p1"}"#
                .to_string(),
        scheduler_info_json: r#"{"shm_namespace_id":"boot:1"}"#.to_string(),
        tokenspeed_version: "0.1.0.post20261003".to_string(),
        max_running_requests: 64,
        data_parallel_size: 1,
        ..Default::default()
    }
}

fn config(
    dir: &std::path::Path,
    handshake: &str,
    model: TokenSpeedModelInfo,
) -> TokenSpeedServicerConfig {
    TokenSpeedServicerConfig {
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

/// The handshake endpoint is free again: ZMQ unlinks the ipc socket file once
/// the bound socket is dropped, so a fresh listener can take the path.
async fn assert_handshake_released(handshake: &str) {
    use std::os::unix::net::UnixListener;
    let path = handshake.trim_start_matches("ipc://");
    let deadline = Instant::now() + Duration::from_secs(5);
    while std::fs::metadata(path).is_ok() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    UnixListener::bind(path).expect("handshake endpoint released");
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
    server: TokenSpeedServicerServer,
    engine_in: MockEngineInput,
    engine_out: MockEngineOutput,
    client: TokenSpeedSchedulerClient<Channel>,
    _dir: tempfile::TempDir,
}

async fn harness(model: TokenSpeedModelInfo, tokenizer: Option<Arc<dyn Tokenizer>>) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let config = config(dir.path(), &handshake, model);
    let server = match tokenizer {
        Some(tokenizer) => TokenSpeedServicerServer::start_with_tokenizer(config, tokenizer),
        None => TokenSpeedServicerServer::start(config),
    }
    .expect("servicer starts");
    let engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        EngineCoreReadyResponse {
            cache_trace_epochs: vec!["capture-epoch".to_string()],
            ..default_ready_response()
        },
    )
    .await
    .expect("mock scheduler handshake");
    wait_until(|| server.engine_ready()).await;
    let client = TokenSpeedSchedulerClient::new(grpc_channel(server.address()).await);
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
) -> BatchTokenIDOutSlim {
    let (num_running, num_waiting, kv_active_pages, kv_total_pages) = load.unwrap_or_default();
    BatchTokenIDOutSlim {
        rids: vec![rid.to_string()],
        output_ids: vec![tokens],
        finished_reasons: vec![finish.unwrap_or_default().to_string()],
        prompt_tokens: vec![3],
        completion_tokens: vec![completion_tokens],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![Vec::new()],
        output_token_logprobs_idx: vec![Vec::new()],
        engine_index: 0,
        num_running,
        num_waiting,
        kv_active_pages,
        kv_total_pages,
    }
}

async fn send(engine_out: &mut MockEngineOutput, batch: &BatchTokenIDOutSlim) {
    engine_out
        .send_frames(vec![Bytes::from(encode_msgpack(batch).unwrap())])
        .await
        .unwrap();
}

async fn recv_add(engine: &mut MockEngineInput) -> TokenizedGenerateReqInput {
    let frames = engine.recv_frames().await.unwrap();
    assert_eq!(
        TokenSpeedRequestType::from_frame(frames[0].as_ref()),
        Some(TokenSpeedRequestType::Add)
    );
    decode_msgpack(frames[1].as_ref()).unwrap()
}

async fn recv_abort(engine: &mut MockEngineInput) -> Vec<String> {
    let frames = engine.recv_frames().await.unwrap();
    assert_eq!(
        TokenSpeedRequestType::from_frame(frames[0].as_ref()),
        Some(TokenSpeedRequestType::Abort)
    );
    decode_msgpack(frames[1].as_ref()).unwrap()
}

fn generate_request(id: &str, stream: bool, stops: Vec<String>) -> ts::GenerateRequest {
    ts::GenerateRequest {
        request_id: id.to_string(),
        tokenized: Some(ts::TokenizedInput {
            input_ids: vec![1, 2, 3],
            original_text: String::new(),
        }),
        sampling_params: Some(ts::SamplingParams {
            max_new_tokens: Some(8),
            stop: stops,
            ..Default::default()
        }),
        stream,
        ..Default::default()
    }
}

fn chunk_tokens(response: ts::GenerateResponse) -> Vec<u32> {
    match response.response {
        Some(ts::generate_response::Response::Chunk(chunk)) => chunk.token_ids,
        other => panic!("expected a chunk, got {other:?}"),
    }
}

fn complete(response: ts::GenerateResponse) -> ts::GenerateComplete {
    match response.response {
        Some(ts::generate_response::Response::Complete(complete)) => complete,
        other => panic!("expected a complete, got {other:?}"),
    }
}

#[test]
fn config_is_validated_before_binding() {
    let dir = tempfile::tempdir().unwrap();
    let good = config(dir.path(), "tcp://127.0.0.1:1", model_info());
    for (name, bad) in [
        (
            "bind",
            TokenSpeedServicerConfig {
                bind_address: "nope".into(),
                ..good.clone()
            },
        ),
        (
            "ipc",
            TokenSpeedServicerConfig {
                ipc_base_url: "tcp://x".into(),
                ..good.clone()
            },
        ),
        (
            "handshake",
            TokenSpeedServicerConfig {
                handshake_address: "udp://x".into(),
                ..good.clone()
            },
        ),
        (
            "engines",
            TokenSpeedServicerConfig {
                engine_count: 0,
                ..good.clone()
            },
        ),
        (
            "model",
            TokenSpeedServicerConfig {
                model: TokenSpeedModelInfo::default(),
                ..good.clone()
            },
        ),
        (
            "startup timeout",
            TokenSpeedServicerConfig {
                engine_startup_timeout: Duration::ZERO,
                ..good.clone()
            },
        ),
    ] {
        assert!(
            matches!(
                TokenSpeedServicerServer::start(bad),
                Err(ServicerError::InvalidConfig(_))
            ),
            "{name}"
        );
    }
}

#[tokio::test]
async fn health_gates_on_the_engine_link_and_the_drain_flag() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = TokenSpeedServicerServer::start(config(dir.path(), &handshake, model_info()))
        .expect("servicer starts");
    let address = format!("http://{}", server.address());
    let mut client = TokenSpeedSchedulerClient::connect(address.clone())
        .await
        .unwrap();
    let mut health = HealthClient::new(
        Channel::from_shared(address)
            .unwrap()
            .connect()
            .await
            .unwrap(),
    );

    let before = client
        .health_check(ts::HealthCheckRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert!(!before.healthy);
    assert_eq!(before.message, "Engine is starting");
    let status = health
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(status, ServingStatus::NotServing as i32);

    let _engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock scheduler handshake");
    wait_until(|| server.engine_ready()).await;
    let after = client
        .health_check(ts::HealthCheckRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert!(after.healthy);
    for service in ["", SERVICE_NAME] {
        let status = health
            .check(HealthCheckRequest {
                service: service.to_string(),
            })
            .await
            .unwrap()
            .into_inner()
            .status;
        assert_eq!(status, ServingStatus::Serving as i32, "{service:?}");
    }

    server.set_serving(false);
    let draining = client
        .health_check(ts::HealthCheckRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert!(!draining.healthy);
    assert_eq!(draining.message, "Draining");
    server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Streaming: one chunk per scheduler step, then the scheduler's own
/// `Complete` with the accumulated ids, all under the request's id.
#[tokio::test]
async fn streaming_generate_maps_steps_to_chunks_and_a_complete() {
    let mut h = harness(model_info(), None).await;
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
    assert!(request.sampling_params.is_normalized);
    assert!(request.stream);

    send(&mut h.engine_out, &batch("r1", vec![10], 1, None, None)).await;
    send(
        &mut h.engine_out,
        &batch("r1", vec![11], 2, Some("length"), None),
    )
    .await;

    let first = stream.message().bounded().await.unwrap().unwrap();
    assert_eq!(first.request_id, "r1");
    assert_eq!(chunk_tokens(first), vec![10]);
    let second = stream.message().bounded().await.unwrap().unwrap();
    assert_eq!(chunk_tokens(second), vec![11]);
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.output_ids, vec![10, 11]);
    assert_eq!(done.prompt_tokens, 3);
    assert_eq!(done.completion_tokens, 2);
    assert_eq!(done.index, 0);
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Non-streaming delivers exactly the terminal `Complete`.
#[tokio::test]
async fn non_streaming_generate_delivers_only_the_complete() {
    let mut h = harness(model_info(), None).await;
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

/// A multi-token stop string never reaches the scheduler; the servicer
/// matches it on the decoded output, ends the choice with
/// `matched_stop_str`, and aborts the scheduler-side request.
#[tokio::test]
async fn string_stops_are_matched_by_the_servicer() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let mut h = harness(model_info(), Some(tokenizer)).await;
    // "Hello world" is two mock tokens (1, 2), so it stays a string stop.
    let mut stream = h
        .client
        .generate(generate_request(
            "r3",
            true,
            vec!["Hello world".to_string()],
        ))
        .await
        .expect("generate")
        .into_inner();

    let request = recv_add(&mut h.engine_in).await;
    assert!(request.sampling_params.stop_token_ids.is_none());
    assert!(request.sampling_params.stop_strs.is_empty());
    send(&mut h.engine_out, &batch("r3", vec![1], 1, None, None)).await;
    send(&mut h.engine_out, &batch("r3", vec![2], 2, None, None)).await;

    assert_eq!(
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
        vec![1]
    );
    assert_eq!(
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
        vec![2]
    );
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![1, 2]);
    assert_eq!(
        done.matched_stop,
        Some(ts::generate_complete::MatchedStop::MatchedStopStr(
            "Hello world".to_string()
        ))
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r3".to_string()]);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A stop string that is one token (`world` is mock token 2) also reaches
/// the scheduler as a stop id, so it finishes on the tick the string
/// matches; the choice ends with the scheduler's own `Complete`.
#[tokio::test]
async fn a_single_token_stop_reaches_the_scheduler_as_a_stop_id() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let mut h = harness(model_info(), Some(tokenizer)).await;
    let mut stream = h
        .client
        .generate(generate_request("r3b", false, vec!["world".to_string()]))
        .await
        .expect("generate")
        .into_inner();
    let request = recv_add(&mut h.engine_in).await;
    assert_eq!(request.sampling_params.stop_token_ids, Some(vec![2]));
    send(&mut h.engine_out, &batch("r3b", vec![7], 1, None, None)).await;
    send(
        &mut h.engine_out,
        &batch("r3b", vec![2], 2, Some("stop"), None),
    )
    .await;
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![7, 2]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[tokio::test]
async fn string_stops_without_a_tokenizer_are_refused() {
    let mut h = harness(model_info(), None).await;
    let status = h
        .client
        .generate(generate_request(
            "r3c",
            true,
            vec!["Hello world".to_string()],
        ))
        .await
        .expect_err("string stops need a tokenizer");
    assert_eq!(status.code(), Code::FailedPrecondition);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[tokio::test]
async fn abort_rpc_cancels_an_in_flight_stream() {
    let mut h = harness(model_info(), None).await;
    let mut stream = h
        .client
        .generate(generate_request("r4", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("r4", vec![10], 1, None, None)).await;
    assert_eq!(
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
        vec![10]
    );

    let response = h
        .client
        .abort(ts::AbortRequest {
            request_id: "r4".to_string(),
            reason: "test".to_string(),
        })
        .await
        .expect("abort")
        .into_inner();
    assert!(response.success);
    // Ends as on the Python servicer: a terminal `abort` Complete with the
    // output so far, then the stream closes; the scheduler side is aborted.
    let aborted = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(aborted.finish_reason, "abort");
    assert_eq!(aborted.output_ids, vec![10]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r4".to_string()]);
    // An unknown id is a no-op, not an error (idempotent cleanup).
    h.client
        .abort(ts::AbortRequest {
            request_id: "never".to_string(),
            reason: String::new(),
        })
        .await
        .expect("abort of an unknown id");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[tokio::test]
async fn duplicate_in_flight_request_ids_are_refused() {
    let mut h = harness(model_info(), None).await;
    let _stream = h
        .client
        .generate(generate_request("dup", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let status = h
        .client
        .generate(generate_request("dup", true, Vec::new()))
        .await
        .expect_err("duplicate id");
    assert_eq!(status.code(), Code::AlreadyExists);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `n > 1` fans out to one scheduler request per choice under the parent's
/// id; every response is stamped with the parent id and its choice index.
#[tokio::test]
async fn choices_fan_out_under_the_parent_id() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("r6", true, Vec::new());
    request.sampling_params.as_mut().unwrap().n = 2;
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let first = recv_add(&mut h.engine_in).await;
    let second = recv_add(&mut h.engine_in).await;
    assert_eq!(first.rid, "r6-0");
    assert_eq!(second.rid, "r6-1");
    assert_eq!(first.sampling_params.n, 1);

    send(
        &mut h.engine_out,
        &batch("r6-1", vec![21], 1, Some("stop"), None),
    )
    .await;
    send(
        &mut h.engine_out,
        &batch("r6-0", vec![20], 1, Some("stop"), None),
    )
    .await;
    let mut seen = Vec::new();
    for _ in 0..4 {
        let response = stream.message().bounded().await.unwrap().unwrap();
        assert_eq!(response.request_id, "r6");
        match response.response.unwrap() {
            ts::generate_response::Response::Chunk(chunk) => {
                seen.push((chunk.index, chunk.token_ids, false));
            }
            ts::generate_response::Response::Complete(done) => {
                seen.push((done.index, done.output_ids, true));
            }
        }
    }
    seen.sort();
    assert_eq!(
        seen,
        vec![
            (0, vec![20], false),
            (0, vec![20], true),
            (1, vec![21], false),
            (1, vec![21], true),
        ]
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Sampled-token logprobs ride through; what the wire cannot carry (ranked
/// candidates, prompt logprobs) is refused rather than silently truncated.
#[tokio::test]
async fn logprobs_pass_through_and_the_unsupported_kinds_are_refused() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("lp", true, Vec::new());
    request.return_logprob = true;
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let wire = recv_add(&mut h.engine_in).await;
    assert!(wire.return_logprob);
    let mut step = batch("lp", vec![10], 1, Some("length"), None);
    step.output_token_logprobs_val = vec![vec![-0.5]];
    step.output_token_logprobs_idx = vec![vec![10]];
    send(&mut h.engine_out, &step).await;
    let chunk = match stream
        .message()
        .bounded()
        .await
        .unwrap()
        .unwrap()
        .response
        .unwrap()
    {
        ts::generate_response::Response::Chunk(chunk) => chunk,
        other @ ts::generate_response::Response::Complete(_) => {
            panic!("expected a chunk, got {other:?}")
        }
    };
    let logprobs = chunk.output_logprobs.expect("chunk logprobs");
    assert_eq!(logprobs.token_ids, vec![10]);
    assert!((logprobs.token_logprobs[0] + 0.5).abs() < 1e-6);
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.output_logprobs.unwrap().token_ids, vec![10]);

    let mut request = generate_request("lp2", true, Vec::new());
    request.top_logprobs_num = 2;
    let status = h.client.generate(request).await.expect_err("top logprobs");
    assert_eq!(status.code(), Code::InvalidArgument);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

#[tokio::test]
async fn requests_without_token_ids_are_refused() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("nt", true, Vec::new());
    request.tokenized = None;
    let status = h.client.generate(request).await.expect_err("no ids");
    assert_eq!(status.code(), Code::InvalidArgument);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The info RPCs report the launcher's facts, with the handshake's capacity
/// figures under `scheduler_info`, and the loads the scheduler piggybacks on
/// its output (zero-filled per rank before any step).
#[tokio::test]
async fn info_rpcs_report_the_launcher_facts_and_the_handshake() {
    let mut h = harness(model_info(), None).await;
    let model = h
        .client
        .get_model_info(ts::GetModelInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(model.model_path, "org/model");
    assert_eq!(model.served_model_name, "served");
    assert_eq!(model.max_context_length, 4096);
    assert_eq!(model.max_req_input_len, 4096);
    assert_eq!(model.vocab_size, 1024);
    assert_eq!(model.eos_token_ids, vec![999]);
    assert_eq!(model.architectures, vec!["Qwen3ForCausalLM".to_string()]);
    assert_eq!(model.default_sampling_params_json, "{\"temperature\":0.7}");

    let info = h
        .client
        .get_server_info(ts::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    let server_args = info.server_args.expect("server_args");
    assert_eq!(
        server_args.fields["dp_size"].kind,
        Some(Kind::NumberValue(2.0))
    );
    assert_eq!(
        server_args.fields["pairing_protocol"].kind,
        Some(Kind::StringValue("p1".to_string()))
    );
    let scheduler_info = info.scheduler_info.expect("scheduler_info");
    assert_eq!(
        scheduler_info.fields["cache_trace_epochs"].kind,
        Some(Kind::ListValue(prost_types::ListValue {
            values: vec![prost_types::Value {
                kind: Some(Kind::StringValue("capture-epoch".to_string()))
            }]
        }))
    );
    assert_eq!(
        scheduler_info.fields["shm_namespace_id"].kind,
        Some(Kind::StringValue("boot:1".to_string()))
    );
    assert_eq!(
        scheduler_info.fields["page_size"].kind,
        Some(Kind::NumberValue(16.0))
    );
    assert_eq!(info.tokenspeed_version, "0.1.0.post20261003");
    assert_eq!(info.active_requests, 0);
    assert!(info.start_time.is_some());

    let idle = h
        .client
        .get_loads(ts::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(idle.dp_rank_count, 1);
    assert_eq!(idle.loads[0].max_running_requests, 64);
    assert_eq!(idle.loads[0].num_running_reqs, 0);
    assert_eq!(idle.version, "tokenspeed");

    // A step carrying a load snapshot updates the rank's report while the
    // request is in flight.
    let mut stream = h
        .client
        .generate(generate_request("ld", false, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(
        &mut h.engine_out,
        &batch("ld", vec![10], 1, None, Some((2, 1, 10, 100))),
    )
    .await;
    wait_until(|| {
        h.server
            .state
            .engine
            .client
            .get()
            .is_some_and(|client| !client.get_loads().loads.is_empty())
    })
    .await;
    let loads = h
        .client
        .get_loads(ts::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(loads.loads.len(), 1);
    let load = &loads.loads[0];
    assert_eq!((load.num_running_reqs, load.num_waiting_reqs), (2, 1));
    assert_eq!(load.num_total_reqs, 3);
    assert!((load.token_usage - 0.1).abs() < 1e-9);
    let aggregate = loads.aggregate.expect("aggregate");
    assert_eq!(aggregate.total_reqs, 3);

    // Once the rank's last request finished, its queue counts read idle (the
    // scheduler samples its snapshot before the finish is committed, so the
    // client, which routed every request, zeroes them); KV usage stays as
    // reported.
    send(
        &mut h.engine_out,
        &batch("ld", vec![11], 2, Some("length"), Some((2, 1, 10, 100))),
    )
    .await;
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![10, 11]);
    let idle = h
        .client
        .get_loads(ts::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        (
            idle.loads[0].num_running_reqs,
            idle.loads[0].num_waiting_reqs
        ),
        (0, 0)
    );
    assert!((idle.loads[0].token_usage - 0.1).abs() < 1e-9);
    assert_eq!(idle.loads[0].active_token_usage, Some(0.1));
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The msgpack wire has no control messages, so the scheduler's control RPCs
/// say so instead of pretending; a disabled KV publisher reads as the Python
/// servicer's UNIMPLEMENTED.
#[tokio::test]
async fn control_rpcs_report_what_the_wire_cannot_carry() {
    let mut h = harness(model_info(), None).await;
    let flush = h
        .client
        .flush_cache(common::FlushCacheRequest::default())
        .await
        .expect_err("no control wire");
    assert_eq!(flush.code(), Code::Unimplemented);
    let profile = h
        .client
        .start_profile(common::StartProfileRequest::default())
        .await
        .expect_err("no control wire");
    assert_eq!(profile.code(), Code::Unimplemented);
    let events = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .expect_err("no publisher");
    assert_eq!(events.code(), Code::Unimplemented);
    assert_eq!(events.message(), kv_events::TOKENSPEED_DISABLED_MESSAGE);
    let bundle = h
        .client
        .get_tokenizer(common::GetTokenizerRequest::default())
        .await
        .expect_err("no tokenizer dir");
    assert_eq!(bundle.code(), Code::FailedPrecondition);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
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
    let mut h = harness(model, None).await;
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
    let mut h = harness(model, None).await;
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

/// A scheduler that never dials in fails the link at the configured bound (not
/// a fixed one: a cold kernel cache makes a real start exceed ten minutes), the
/// server stays up to report it, and `stop` releases the handshake endpoint.
#[tokio::test]
async fn a_scheduler_that_never_dials_in_fails_the_link_at_the_startup_bound() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = TokenSpeedServicerServer::start(TokenSpeedServicerConfig {
        engine_startup_timeout: Duration::from_millis(300),
        ..config(dir.path(), &handshake, model_info())
    })
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let error = loop {
        if let Some(error) = server.last_error().unwrap() {
            break error;
        }
        assert!(Instant::now() < deadline, "no link failure reported");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(error.contains("timed out"), "{error}");
    assert!(!server.engine_ready());
    assert!(
        server.running(),
        "the server keeps answering after the link failed"
    );

    server.stop(Duration::from_secs(5)).unwrap();
    assert_handshake_released(&handshake).await;
}
