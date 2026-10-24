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
        connect_to_frontend, default_ready_response, hello, MockEngineInput, MockEngineOutput,
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
use futures::StreamExt;
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
use crate::{
    kv_events,
    testing::{assert_bucket_edges, assert_golden, http_get, servicer_config, Bounded},
    ServicerError,
};

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
            r#"{"model":"org/model","max_num_seqs":64,"dp_size":2,"pairing_protocol":"p1","engine_reasoning_parser":"kimi_k25","constrained_decoding_start":"first_token"}"#
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
        servicer: servicer_config(dir, handshake),
        model,
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

/// Nothing reaches a mock scheduler that was only handshaken.
async fn assert_scheduler_idle(engine_in: &mut MockEngineInput) {
    assert!(
        tokio::time::timeout(Duration::from_millis(200), engine_in.recv())
            .await
            .is_err(),
        "nothing should have reached the scheduler"
    );
}

/// Between HELLO and READY a real scheduler loads its model, and nothing
/// crosses the wire meanwhile. The lifecycle owner's reports that the
/// scheduler process is alive keep the link waiting well past the silence
/// bound, and the start then completes.
#[tokio::test]
async fn a_slow_scheduler_stays_linked_while_the_launcher_reports_it_alive() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let silence = Duration::from_millis(300);
    let mut config = config(dir.path(), &handshake, model_info());
    config.servicer.engine_startup_timeout = silence;
    let server = TokenSpeedServicerServer::start(config).unwrap();
    // The scheduler dials in at once, then "loads" for four silences.
    let loading = hello(&handshake, EngineId::from_engine_index(0))
        .bounded()
        .await
        .expect("HELLO answered with INIT");
    let load_time = Instant::now() + 4 * silence;
    while Instant::now() < load_time {
        server.note_engine_alive();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        server.last_error().unwrap(),
        None,
        "the link gave up on a living scheduler"
    );
    assert!(!server.engine_ready());

    let engine = loading
        .ready(default_ready_response())
        .bounded()
        .await
        .expect("READY and registration accepted");
    wait_until(|| server.engine_ready()).await;
    let (mut engine_in, _engine_out) = engine.split();
    assert_scheduler_idle(&mut engine_in).await;
    server.stop(Duration::from_secs(5)).unwrap();
}

/// A scheduler that dials in and then falls silent, with nobody reporting it
/// alive, fails the link at the silence bound as before.
#[tokio::test]
async fn a_scheduler_that_falls_silent_after_hello_fails_the_link_at_the_bound() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let silence = Duration::from_millis(300);
    let mut config = config(dir.path(), &handshake, model_info());
    config.servicer.engine_startup_timeout = silence;
    let server = TokenSpeedServicerServer::start(config).unwrap();
    let _loading = hello(&handshake, EngineId::from_engine_index(0))
        .bounded()
        .await
        .expect("HELLO answered with INIT");
    let since_hello = Instant::now();
    let deadline = since_hello + Duration::from_secs(10);
    let error = loop {
        if let Some(error) = server.last_error().unwrap() {
            break error;
        }
        assert!(Instant::now() < deadline, "no link failure reported");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(since_hello.elapsed() >= silence);
    assert!(
        error.contains("timed out while waiting for READY"),
        "{error}"
    );
    assert!(error.contains("without a sign of life"), "{error}");
    assert!(
        error.starts_with("TokenSpeed scheduler connection failed: "),
        "{error}"
    );
    assert!(!server.engine_ready());
    server.stop(Duration::from_secs(5)).unwrap();
}

/// The ceiling bounds a start however alive the scheduler is reported.
#[tokio::test]
async fn the_startup_ceiling_bounds_a_start_the_launcher_keeps_reporting_alive() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let silence = Duration::from_millis(300);
    let ceiling = Duration::from_millis(900);
    let mut config = config(dir.path(), &handshake, model_info());
    config.servicer.engine_startup_timeout = silence;
    config.servicer.engine_startup_ceiling = Some(ceiling);
    let server = TokenSpeedServicerServer::start(config).unwrap();
    let started = Instant::now();
    let _loading = hello(&handshake, EngineId::from_engine_index(0))
        .bounded()
        .await
        .expect("HELLO answered with INIT");
    let deadline = started + Duration::from_secs(10);
    let error = loop {
        server.note_engine_alive();
        if let Some(error) = server.last_error().unwrap() {
            break error;
        }
        assert!(Instant::now() < deadline, "no link failure reported");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(started.elapsed() >= ceiling);
    assert!(error.contains("reached its ceiling"), "{error}");
    assert!(!server.engine_ready());
    server.stop(Duration::from_secs(5)).unwrap();
}

#[test]
fn config_is_validated_before_binding() {
    let dir = tempfile::tempdir().unwrap();
    let good = config(dir.path(), "tcp://127.0.0.1:1", model_info());
    let with = |edit: fn(&mut ServicerConfig)| {
        let mut config = good.clone();
        edit(&mut config.servicer);
        config
    };
    for (name, bad) in [
        (
            "bind",
            with(|servicer| servicer.bind_address = "nope".into()),
        ),
        (
            "ipc",
            with(|servicer| servicer.ipc_base_url = "tcp://x".into()),
        ),
        (
            "handshake",
            with(|servicer| servicer.handshake_address = "udp://x".into()),
        ),
        ("engines", with(|servicer| servicer.engine_count = 0)),
        (
            "timeout",
            with(|servicer| servicer.engine_startup_timeout = Duration::ZERO),
        ),
        (
            "ceiling",
            with(|servicer| {
                servicer.engine_startup_ceiling = Some(Duration::ZERO);
            }),
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
            with(|servicer| servicer.engine_startup_timeout = Duration::ZERO),
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

    // Readiness held alone (the lifecycle owner's planned warm-up): health
    // answers NOT_SERVING with its own message while admission stays open to
    // the warm-up's own requests only, then the release restores SERVING and
    // admits every request; the drain below closes admission.
    server.set_ready(false);
    let held = client
        .health_check(ts::HealthCheckRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert!(!held.healthy);
    assert_eq!(held.message, "Warming up");
    let refused = server
        .state
        .shared
        .registry
        .register("held-probe")
        .err()
        .expect("a request that is not the warm-up's is refused while readiness is held");
    assert_eq!(refused.code(), Code::Unavailable);
    assert_eq!(refused.message(), "servicer is warming up");
    let (admitted, _cancel) = server
        .state
        .shared
        .registry
        .register("smg-warmup-1")
        .expect("the warm-up's own request is admitted while readiness is held");
    drop(admitted);
    server.set_ready(true);
    let (admitted, _cancel) = server
        .state
        .shared
        .registry
        .register("held-probe")
        .expect("the release admits every request");
    drop(admitted);
    assert!(
        client
            .health_check(ts::HealthCheckRequest::default())
            .await
            .unwrap()
            .into_inner()
            .healthy
    );

    server.set_serving(false);
    let draining = client
        .health_check(ts::HealthCheckRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert!(!draining.healthy);
    assert_eq!(draining.message, "Draining");
    assert!(
        server.state.shared.registry.register("late").is_err(),
        "the drain closes admission"
    );
    server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Readiness held for the lifecycle owner's planned warm-up: a request that
/// is not the warm-up's own is refused as not serving before anything of it
/// reaches the engine (the engine's next admission is the warm-up's), the
/// warm-up's own request is served, and the release admits the other again.
#[tokio::test]
async fn a_request_that_is_not_the_warm_ups_is_refused_while_readiness_is_held() {
    let mut h = harness(model_info(), None).await;
    h.server.set_ready(false);
    let status = h
        .client
        .generate(generate_request("chatcmpl-1", true, Vec::new()))
        .await
        .expect_err("readiness held: not the warm-up's request");
    assert_eq!(status.code(), Code::Unavailable);
    assert_eq!(status.message(), "servicer is warming up");
    assert_eq!(h.server.active_requests(), 0);

    let mut own = h
        .client
        .generate(generate_request("smg-warmup-1", true, Vec::new()))
        .await
        .expect("the warm-up's own request is admitted")
        .into_inner();
    let request = recv_add(&mut h.engine_in).await;
    assert_eq!(
        request.rid, "smg-warmup-1",
        "the engine's first admission is the warm-up's"
    );
    send(
        &mut h.engine_out,
        &batch("smg-warmup-1", vec![10], 1, Some("length"), None),
    )
    .await;
    while own.message().bounded().await.unwrap().is_some() {}

    h.server.set_ready(true);
    let mut late = h
        .client
        .generate(generate_request("chatcmpl-1", true, Vec::new()))
        .await
        .expect("the release admits every request")
        .into_inner();
    let request = recv_add(&mut h.engine_in).await;
    assert_eq!(request.rid, "chatcmpl-1");
    send(
        &mut h.engine_out,
        &batch("chatcmpl-1", vec![10], 1, Some("length"), None),
    )
    .await;
    while late.message().bounded().await.unwrap().is_some() {}
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
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

    assert_eq!(h.server.active_requests(), 1);
    h.server.set_serving(false);
    let status = h
        .client
        .generate(generate_request("late", true, Vec::new()))
        .await
        .expect_err("draining");
    assert_eq!(status.code(), Code::Unavailable);
    assert_eq!(h.server.active_requests(), 1);
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
    assert_eq!(h.server.active_requests(), 0);
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

/// `Abort` ends the stream with an `abort` Complete and reaches the
/// scheduler. The caller's own abort stays a completion; only a shutdown
/// under an open request is a status
/// (`a_shutdown_ends_an_open_stream_with_unavailable_not_an_abort_completion`).
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

/// The launcher's constrained-decoding policy (the engine's declaration that
/// it takes a grammar on every turn) reaches `GetServerInfo` as a field of its
/// own, for the Router's `constrained_decoding` label.
#[tokio::test]
async fn server_info_reports_the_engines_constrained_decoding_policy() {
    let mut h = harness(
        TokenSpeedModelInfo {
            constrained_decoding: "always".to_string(),
            ..model_info()
        },
        None,
    )
    .await;
    let info = h
        .client
        .get_server_info(ts::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.constrained_decoding, "always");
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
    // A launcher that reports no constrained-decoding policy leaves the field
    // at its default; the Router then sets no `constrained_decoding` label.
    assert_eq!(info.constrained_decoding, "");
    let server_args = info.server_args.expect("server_args");
    assert_eq!(
        server_args.fields["dp_size"].kind,
        Some(Kind::NumberValue(2.0))
    );
    assert_eq!(
        server_args.fields["pairing_protocol"].kind,
        Some(Kind::StringValue("p1".to_string()))
    );
    // The launcher's grammar-start facts (the engine's reasoning parser and
    // where its grammar starts on a thinking prompt) reach the Router's label
    // source as the launcher wrote them.
    assert_eq!(
        server_args.fields["engine_reasoning_parser"].kind,
        Some(Kind::StringValue("kimi_k25".to_string()))
    );
    assert_eq!(
        server_args.fields["constrained_decoding_start"].kind,
        Some(Kind::StringValue("first_token".to_string()))
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
            .shared
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
        .shared
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
        .shared
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
    let mut config = config(dir.path(), &handshake, model_info());
    config.servicer.engine_startup_timeout = Duration::from_millis(300);
    let server = TokenSpeedServicerServer::start(config).unwrap();
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

/// A shutdown with a stream still open is the engine side going away under
/// the request, not an abort the caller asked for: the stream ends with
/// `UNAVAILABLE` (the status the Router counts and logs as the worker's
/// failure), never with a `Complete` whose finish reason is `abort`.
#[tokio::test]
async fn a_shutdown_ends_an_open_stream_with_unavailable_not_an_abort_completion() {
    let mut h = harness(model_info(), None).await;
    let mut stream = h
        .client
        .generate(generate_request("r8", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("r8", vec![10], 1, None, None)).await;
    assert_eq!(
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
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
    let mut h = harness(model_info(), None).await;
    let mut stream = h
        .client
        .generate(generate_request("r11", true, Vec::new()))
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
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
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
    assert_eq!(done.output_ids, vec![10]);
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
    let mut h = harness(model_info(), None).await;
    let mut stream = generate::generate(&h.server.state, generate_request("r12", true, Vec::new()))
        .await
        .expect("generate");
    recv_add(&mut h.engine_in).await;
    send(
        &mut h.engine_out,
        &batch("r12", vec![10], 1, Some("length"), None),
    )
    .await;
    assert_eq!(
        chunk_tokens(stream.next().bounded().await.unwrap().unwrap()),
        vec![10]
    );
    let done = complete(stream.next().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.output_ids, vec![10]);
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

/// A worker with two attention-DP ranks publishes on `port` and `port + 1`,
/// each under its own sequence counter. The launcher's width
/// (`data_parallel_size`) tells the relay how many publishers to follow: a
/// gateway's stream carries both ranks' batches, each under its rank's own
/// numbers and stamped with its socket's rank, which is what the gateway's
/// per-rank admission cursors expect.
#[tokio::test]
async fn every_rank_of_a_data_parallel_publisher_reaches_the_gateway() {
    use zeromq::prelude::*;

    use crate::{kv_events::golden, testing::consecutive_publishers};

    let (base, mut publishers) = consecutive_publishers(2).await;
    let mut model = model_info();
    model.kv_events_endpoint = format!("tcp://*:{base}");
    model.kv_events_topic = "kv".to_string();
    model.data_parallel_size = 2;
    let mut h = harness(model, None).await;
    let relay = h
        .server
        .state
        .shared
        .kv_relay
        .clone()
        .expect("a relay for the publishers");
    // Each SUB connect is asynchronous: publish sequence 0 on both ranks
    // until the relay has taken both (repeats are duplicates).
    let batch2 = golden::bytes(golden::BATCH2);
    for _ in 0..250 {
        for publisher in &mut publishers {
            publisher
                .send(golden::frame(b"kv", 0, &batch2))
                .await
                .expect("publish");
        }
        if relay.counts().relayed >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        relay.counts().relayed,
        2,
        "both ranks' sequence 0 reached the relay"
    );

    let mut stream = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .expect("subscribe")
        .into_inner();
    let mut seen = Vec::new();
    for _ in 0..2 {
        let batch = tokio::time::timeout(Duration::from_secs(5), stream.message())
            .await
            .expect("a batch in time")
            .expect("stream open")
            .expect("a batch");
        seen.push((batch.dp_rank, batch.sequence_number));
    }
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![(Some(0), 0), (Some(1), 0)],
        "each rank under its own sequence, stamped with its socket's rank"
    );
    // Each rank continues its own count.
    publishers[1]
        .send(golden::frame(b"kv", 1, &batch2))
        .await
        .expect("publish");
    let batch = tokio::time::timeout(Duration::from_secs(5), stream.message())
        .await
        .expect("a batch in time")
        .expect("stream open")
        .expect("a batch");
    assert_eq!((batch.dp_rank, batch.sequence_number), (Some(1), 1));
    drop(stream);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// What the scheduler enforces of a structured request, as the launcher's
/// `server_args` say: the grammar backend it resolved; nothing is judged
/// when the launcher could not tell.
#[test]
fn structured_outputs_support_follows_the_launcher_args() {
    use engine_zmq_adapter::{
        StructuredOutputBackend,
        StructuredOutputsBackendConfig::{Other, Pinned},
        StructuredOutputsSupport,
    };
    let support = |server_args_json: &str| {
        structured_outputs_support(&TokenSpeedModelInfo {
            server_args_json: server_args_json.to_string(),
            ..model_info()
        })
    };
    assert_eq!(
        support(r#"{"grammar_backend":"xgrammar"}"#),
        Pinned(StructuredOutputBackend::Xgrammar).into()
    );
    assert_eq!(
        support(r#"{"grammar_backend":"none"}"#),
        StructuredOutputsSupport::DISABLED
    );
    // A launcher that wrote no backend could not tell (a checkpoint default
    // not resolved yet, or an older launcher): nothing is judged here.
    assert_eq!(support(r#"{"model":"org/model"}"#), Other.into());
    assert_eq!(support(""), Other.into());
    assert_eq!(support(r#"{"grammar_backend":"other"}"#), Other.into());
}

/// A constraint the scheduler's grammar backend cannot enforce is refused
/// before the scheduler, as the caller's error, on this wire as on the
/// others: every kind under `--grammar-backend none` (the scheduler would
/// admit the request and abort it, which this wire reports as a bare
/// `abort`, an empty answer), and a schema with a feature xgrammar drops;
/// what the backend compiles reaches the scheduler with its constraint, an
/// unconstrained request is served as before, and nothing is judged when
/// the launcher wrote no backend.
#[tokio::test]
async fn a_constraint_the_grammar_backend_cannot_enforce_is_refused_before_the_scheduler() {
    use ts::sampling_params::Constraint;

    let with_backend = |server_args_json: &str| TokenSpeedModelInfo {
        server_args_json: server_args_json.to_string(),
        ..model_info()
    };
    let constrained = |id: &str, constraint: Constraint| {
        let mut request = generate_request(id, false, Vec::new());
        request.sampling_params.as_mut().unwrap().constraint = Some(constraint);
        request
    };
    const SCHEMA: &str = r#"{"type":"object","properties":{"city":{"type":"string"}}}"#;
    const MULTIPLE_OF: &str = r#"{"type":"integer","multipleOf":3}"#;
    let schema = || Constraint::JsonSchema(SCHEMA.to_string());
    let regex = || Constraint::Regex("[0-9]+".to_string());

    // No backend: every kind is refused up front; nothing reaches the
    // scheduler but the unconstrained request.
    {
        let mut h = harness(with_backend(r#"{"grammar_backend":"none"}"#), None).await;
        for (id, constraint) in [
            ("n1", schema()),
            ("n2", regex()),
            ("n3", Constraint::EbnfGrammar("root ::= \"a\"".to_string())),
            (
                "n4",
                Constraint::StructuralTag(r#"{"format":{}}"#.to_string()),
            ),
        ] {
            let status = h
                .client
                .generate(constrained(id, constraint))
                .await
                .expect_err("refused");
            assert_eq!(status.code(), Code::InvalidArgument, "{id}");
            assert!(
                status
                    .message()
                    .contains("grammar backend disabled (`none`)"),
                "{id}: {}",
                status.message()
            );
        }
        assert_scheduler_idle(&mut h.engine_in).await;
        let _stream = h
            .client
            .generate(generate_request("n5", false, Vec::new()))
            .await
            .expect("generate")
            .into_inner();
        assert_eq!(recv_add(&mut h.engine_in).await.rid, "n5");
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }

    // xgrammar: a schema and a regex reach the scheduler with their
    // constraint; a schema with a feature xgrammar drops is refused.
    {
        let mut h = harness(with_backend(r#"{"grammar_backend":"xgrammar"}"#), None).await;
        let _schema_stream = h
            .client
            .generate(constrained("x1", schema()))
            .await
            .expect("generate")
            .into_inner();
        let sent = recv_add(&mut h.engine_in).await;
        assert_eq!(sent.sampling_params.json_schema.as_deref(), Some(SCHEMA));
        let _regex_stream = h
            .client
            .generate(constrained("x2", regex()))
            .await
            .expect("generate")
            .into_inner();
        let sent = recv_add(&mut h.engine_in).await;
        assert_eq!(sent.sampling_params.regex.as_deref(), Some("[0-9]+"));
        let status = h
            .client
            .generate(constrained(
                "x3",
                Constraint::JsonSchema(MULTIPLE_OF.to_string()),
            ))
            .await
            .expect_err("refused");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(
            status.message().contains("`multipleOf` at its root")
                && status.message().contains("not supported by xgrammar"),
            "{}",
            status.message()
        );
        assert_scheduler_idle(&mut h.engine_in).await;
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }

    // No backend written: nothing is judged ahead of the engine.
    {
        let mut h = harness(with_backend(r#"{"model":"org/model"}"#), None).await;
        let _stream = h
            .client
            .generate(constrained(
                "u1",
                Constraint::JsonSchema(MULTIPLE_OF.to_string()),
            ))
            .await
            .expect("generate")
            .into_inner();
        let sent = recv_add(&mut h.engine_in).await;
        assert_eq!(
            sent.sampling_params.json_schema.as_deref(),
            Some(MULTIPLE_OF)
        );
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }
}

/// A prompt that fills the scheduler's context window, or an empty one, is
/// refused before the scheduler sees it, with both lengths named; the
/// refusal releases the registration, so the same id then serves a healthy
/// request.
#[tokio::test]
async fn a_prompt_the_window_cannot_hold_is_refused_before_the_scheduler() {
    let mut h = harness(model_info(), None).await;
    for (input_ids, max_new_tokens) in [
        (vec![7; 1_049_692], Some(16_384)),
        (vec![7; 1_048_576], None),
        (Vec::new(), Some(8)),
    ] {
        let mut request = generate_request("window", true, Vec::new());
        request
            .tokenized
            .as_mut()
            .unwrap()
            .input_ids
            .clone_from(&input_ids);
        request.sampling_params.as_mut().unwrap().max_new_tokens = max_new_tokens;
        let status = h
            .client
            .generate(request)
            .await
            .expect_err("the prompt leaves no room for output");
        assert_eq!(status.code(), Code::InvalidArgument);
        if input_ids.is_empty() {
            assert!(status.message().contains("empty"), "{}", status.message());
        } else {
            assert!(
                status.message().contains(&input_ids.len().to_string()),
                "{}",
                status.message()
            );
            assert!(status.message().contains("1048576"), "{}", status.message());
        }
    }
    let mut request = generate_request("window", true, Vec::new());
    request.tokenized = None;
    let status = h
        .client
        .generate(request)
        .await
        .expect_err("no tokenized input");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_scheduler_idle(&mut h.engine_in).await;
    assert!(h.server.engine_ready());
    // Reusing the refused id proves the registration was released, and a
    // healthy request still reaches the scheduler.
    let _stream = h
        .client
        .generate(generate_request("window", true, Vec::new()))
        .await
        .expect("healthy request after refusal")
        .into_inner();
    let accepted = recv_add(&mut h.engine_in).await;
    assert_eq!(accepted.rid, "window");
    assert_eq!(accepted.input_ids, vec![1, 2, 3]);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The configured engine-silence bound and prefill allowance reach the
/// scheduler's link: with a 300 ms window and one token per second allowed
/// for the prompt pending prefill (three tokens: three seconds), a scheduler
/// that reads a request and never answers is given up on at the window plus
/// the allowance, well under the crate's five-minute default, and the stream
/// ends with the engine's death.
#[tokio::test]
async fn the_configured_silence_bound_and_prefill_allowance_reach_the_scheduler_link() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let window = Duration::from_millis(300);
    let allowance = Duration::from_secs(3);
    let mut config = config(dir.path(), &handshake, model_info());
    config.servicer.engine_silence_timeout = window;
    config.servicer.engine_prefill_tokens_per_sec = 1;
    let server = TokenSpeedServicerServer::start(config).unwrap();
    let engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock scheduler handshake");
    wait_until(|| server.engine_ready()).await;
    let mut client = TokenSpeedSchedulerClient::new(grpc_channel(server.address()).await);
    let (mut engine_in, _engine_out) = engine.split();
    let started = Instant::now();
    let mut stream = client
        .generate(generate_request("r-silent", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    assert_eq!(recv_add(&mut engine_in).await.input_ids, vec![1, 2, 3]);
    let status = stream
        .message()
        .bounded()
        .await
        .expect_err("a scheduler silent past the window and the allowance is dead");
    assert!(
        status
            .message()
            .contains("engine core reported fatal failure"),
        "{}",
        status.message()
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed >= window + allowance && elapsed < Duration::from_secs(15),
        "given up on at the window plus the prompt's allowance, not before or much after: {elapsed:?}"
    );
    server.stop(Duration::from_secs(5)).unwrap();
}

// ---------------------------------------------------------------------------
// Metrics: the `/metrics` listener and the families the request path records.
// ---------------------------------------------------------------------------

/// A bound servicer with its metrics listener on an ephemeral port, a
/// handshaken mock scheduler and a gRPC client.
async fn metrics_harness(
    model: TokenSpeedModelInfo,
    tokenizer: Option<Arc<dyn Tokenizer>>,
) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let mut config = config(dir.path(), &handshake, model);
    config.servicer.metrics_bind_address = Some("127.0.0.1:0".to_string());
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

/// One scrape of the servicer's metrics listener.
fn scrape(server: &TokenSpeedServicerServer) -> String {
    let address = server
        .metrics_address()
        .expect("a metrics listener")
        .parse()
        .expect("a socket address");
    let (status, body) = http_get(address, "/metrics");
    assert_eq!(status, 200);
    body
}

/// The value of the sample of `name` whose labels contain `labels`.
fn sample(body: &str, name: &str, labels: &str) -> f64 {
    let prefix = format!("{name}{{");
    body.lines()
        .find(|line| {
            line.starts_with(&prefix) && labels.split(',').all(|label| line.contains(label))
        })
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("no sample {name}{{{labels}}} in:\n{body}"))
}

/// The exposition after the scripted sequence of
/// `the_scrape_pins_the_families_after_a_scripted_sequence`: every family,
/// type, label set and value, the histogram buckets aside (their edges are
/// pinned apart) and the clock-dependent values replaced by `<v>`.
const GOLDEN_SCRAPE: &str = r#"
# HELP smg_servicer_e2e_request_latency_seconds Seconds from a request's arrival at the servicer to the end of its stream.
# HELP smg_servicer_generation_tokens_total Generated tokens streamed to callers.
# HELP smg_servicer_inter_token_latency_seconds Seconds between consecutive output chunks of a choice, per token when a chunk carries several (servicer clock; sampled above a pending-observation bound).
# HELP smg_servicer_metrics_dropped_total Observations the pressure sampler dropped, by family.
# HELP smg_servicer_metrics_scrape_seconds How long the last scrape took to render.
# HELP smg_servicer_metrics_scrapes_total Scrapes answered, by HTTP status code.
# HELP smg_servicer_prompt_tokens_cached_total Prompt tokens the engine served from its prefix cache.
# HELP smg_servicer_prompt_tokens_total Prompt tokens as the engine reported them at each request's first output.
# HELP smg_servicer_request_errors_total Requests that failed with a gRPC status, by code.
# HELP smg_servicer_request_generation_tokens Generated tokens per request.
# HELP smg_servicer_request_params_max_tokens The max_tokens parameter of requests that set it.
# HELP smg_servicer_request_params_n The n parameter of requests.
# HELP smg_servicer_request_prompt_tokens Prompt tokens per request.
# HELP smg_servicer_request_time_per_output_token_seconds Mean seconds per generated token after the first, per request (servicer clock).
# HELP smg_servicer_requests_active Admitted requests still preprocessing, streaming, or unwinding an abort.
# HELP smg_servicer_requests_refused_total Requests refused before they reached the engine, by reason.
# HELP smg_servicer_requests_total Requests whose stream ended, by how they ended; a fan-out (n > 1) is one request with the reason of its last choice to end, an abort winning.
# HELP smg_servicer_start_time_seconds When the servicer started, in seconds since the Unix epoch.
# HELP smg_servicer_state The servicer's lifecycle state, one-hot: starting, serving, warming_up (readiness held with admission open), draining (admission closed) or engine_down.
# HELP smg_servicer_string_stops_total Requests this servicer ended on a stop string it matched itself.
# HELP smg_servicer_time_to_first_token_seconds Seconds from a request's arrival at the servicer to its first output.
# TYPE smg_servicer_e2e_request_latency_seconds histogram
# TYPE smg_servicer_generation_tokens_total counter
# TYPE smg_servicer_inter_token_latency_seconds histogram
# TYPE smg_servicer_metrics_dropped_total counter
# TYPE smg_servicer_metrics_scrape_seconds gauge
# TYPE smg_servicer_metrics_scrapes_total counter
# TYPE smg_servicer_prompt_tokens_cached_total counter
# TYPE smg_servicer_prompt_tokens_total counter
# TYPE smg_servicer_request_errors_total counter
# TYPE smg_servicer_request_generation_tokens histogram
# TYPE smg_servicer_request_params_max_tokens histogram
# TYPE smg_servicer_request_params_n histogram
# TYPE smg_servicer_request_prompt_tokens histogram
# TYPE smg_servicer_request_time_per_output_token_seconds histogram
# TYPE smg_servicer_requests_active gauge
# TYPE smg_servicer_requests_refused_total counter
# TYPE smg_servicer_requests_total counter
# TYPE smg_servicer_start_time_seconds gauge
# TYPE smg_servicer_state gauge
# TYPE smg_servicer_string_stops_total counter
# TYPE smg_servicer_time_to_first_token_seconds histogram
smg_servicer_e2e_request_latency_seconds_count{model="served",wire="tokenspeed",rank="0"} 5
smg_servicer_e2e_request_latency_seconds_sum{model="served",wire="tokenspeed",rank="0"} <v>
smg_servicer_generation_tokens_total{model="served",wire="tokenspeed",rank="0"} 8
smg_servicer_inter_token_latency_seconds_count{model="served",wire="tokenspeed",rank="0"} 2
smg_servicer_inter_token_latency_seconds_sum{model="served",wire="tokenspeed",rank="0"} <v>
smg_servicer_metrics_dropped_total{model="served",wire="tokenspeed",family="inter_token_latency"} 0
smg_servicer_metrics_scrape_seconds{model="served",wire="tokenspeed"} <v>
smg_servicer_metrics_scrapes_total{model="served",wire="tokenspeed",code="200"} 0
smg_servicer_metrics_scrapes_total{model="served",wire="tokenspeed",code="500"} 0
smg_servicer_prompt_tokens_cached_total{model="served",wire="tokenspeed",rank="0"} 1
smg_servicer_prompt_tokens_total{model="served",wire="tokenspeed",rank="0"} 15
smg_servicer_request_errors_total{model="served",wire="tokenspeed",code="unavailable"} 1
smg_servicer_request_generation_tokens_count{model="served",wire="tokenspeed",rank="0"} 5
smg_servicer_request_generation_tokens_sum{model="served",wire="tokenspeed",rank="0"} 8
smg_servicer_request_params_max_tokens_count{model="served",wire="tokenspeed",rank="0"} 5
smg_servicer_request_params_max_tokens_sum{model="served",wire="tokenspeed",rank="0"} 40
smg_servicer_request_params_n_count{model="served",wire="tokenspeed",rank="0"} 5
smg_servicer_request_params_n_sum{model="served",wire="tokenspeed",rank="0"} 5
smg_servicer_request_prompt_tokens_count{model="served",wire="tokenspeed",rank="0"} 5
smg_servicer_request_prompt_tokens_sum{model="served",wire="tokenspeed",rank="0"} 15
smg_servicer_request_time_per_output_token_seconds_count{model="served",wire="tokenspeed",rank="0"} 2
smg_servicer_request_time_per_output_token_seconds_sum{model="served",wire="tokenspeed",rank="0"} <v>
smg_servicer_requests_active{model="served",wire="tokenspeed"} 0
smg_servicer_requests_refused_total{model="served",wire="tokenspeed",reason="draining"} 0
smg_servicer_requests_refused_total{model="served",wire="tokenspeed",reason="duplicate_id"} 1
smg_servicer_requests_refused_total{model="served",wire="tokenspeed",reason="invalid_argument"} 0
smg_servicer_requests_refused_total{model="served",wire="tokenspeed",reason="media"} 0
smg_servicer_requests_total{model="served",wire="tokenspeed",rank="0",finish_reason="abort"} 1
smg_servicer_requests_total{model="served",wire="tokenspeed",rank="0",finish_reason="cut"} 1
smg_servicer_requests_total{model="served",wire="tokenspeed",rank="0",finish_reason="error"} 0
smg_servicer_requests_total{model="served",wire="tokenspeed",rank="0",finish_reason="length"} 1
smg_servicer_requests_total{model="served",wire="tokenspeed",rank="0",finish_reason="stop"} 2
smg_servicer_start_time_seconds{model="served",wire="tokenspeed"} <v>
smg_servicer_state{model="served",wire="tokenspeed",state="draining"} 1
smg_servicer_state{model="served",wire="tokenspeed",state="engine_down"} 0
smg_servicer_state{model="served",wire="tokenspeed",state="serving"} 0
smg_servicer_state{model="served",wire="tokenspeed",state="starting"} 0
smg_servicer_state{model="served",wire="tokenspeed",state="warming_up"} 0
smg_servicer_string_stops_total{model="served",wire="tokenspeed",rank="0"} 0
smg_servicer_time_to_first_token_seconds_count{model="served",wire="tokenspeed",rank="0"} 5
smg_servicer_time_to_first_token_seconds_sum{model="served",wire="tokenspeed",rank="0"} <v>
"#;

/// One streamed request (three steps, the scheduler's stop), one
/// non-streamed (the budget), one the caller aborts, one duplicate id
/// refused, one a shutdown cuts: the scrape's families, types, label sets
/// and values are exactly these.
#[tokio::test]
async fn the_scrape_pins_the_families_after_a_scripted_sequence() {
    let mut h = metrics_harness(model_info(), None).await;
    // Streamed: three single-token steps, the scheduler's stop on the last;
    // the first step reports one cached prompt token.
    let mut s1 = h
        .client
        .generate(generate_request("s1", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let mut first = batch("s1", vec![10], 1, None, None);
    first.cached_tokens = vec![1];
    send(&mut h.engine_out, &first).await;
    send(&mut h.engine_out, &batch("s1", vec![11], 2, None, None)).await;
    send(
        &mut h.engine_out,
        &batch("s1", vec![12], 3, Some("stop"), None),
    )
    .await;
    for _ in 0..3 {
        chunk_tokens(s1.message().bounded().await.unwrap().unwrap());
    }
    assert_eq!(
        complete(s1.message().bounded().await.unwrap().unwrap()).finish_reason,
        "stop"
    );
    assert!(s1.message().bounded().await.unwrap().is_none());
    // Non-streamed: one two-token step that ends on the budget.
    let mut n1 = h
        .client
        .generate(generate_request("n1", false, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(
        &mut h.engine_out,
        &batch("n1", vec![20, 21], 2, Some("length"), None),
    )
    .await;
    assert_eq!(
        complete(n1.message().bounded().await.unwrap().unwrap()).finish_reason,
        "length"
    );
    assert!(n1.message().bounded().await.unwrap().is_none());
    // The caller's abort after the first step.
    let mut a1 = h
        .client
        .generate(generate_request("a1", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("a1", vec![30], 1, None, None)).await;
    chunk_tokens(a1.message().bounded().await.unwrap().unwrap());
    h.client
        .abort(ts::AbortRequest {
            request_id: "a1".to_string(),
            reason: "client left".to_string(),
        })
        .await
        .expect("abort");
    assert_eq!(
        complete(a1.message().bounded().await.unwrap().unwrap()).finish_reason,
        "abort"
    );
    assert!(a1.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["a1".to_string()]);
    // A duplicate id refused while the first is in flight; the first then
    // ends on the scheduler's stop.
    let mut d1 = h
        .client
        .generate(generate_request("d1", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let duplicate = h
        .client
        .generate(generate_request("d1", true, Vec::new()))
        .await
        .expect_err("a live id is refused");
    assert_eq!(duplicate.code(), Code::AlreadyExists);
    send(
        &mut h.engine_out,
        &batch("d1", vec![40], 1, Some("stop"), None),
    )
    .await;
    chunk_tokens(d1.message().bounded().await.unwrap().unwrap());
    complete(d1.message().bounded().await.unwrap().unwrap());
    assert!(d1.message().bounded().await.unwrap().is_none());
    // A shutdown cuts a request still open: the drain announced, every
    // registered cancellation fired, the stream ends with the status.
    let mut c1 = h
        .client
        .generate(generate_request("c1", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("c1", vec![50], 1, None, None)).await;
    chunk_tokens(c1.message().bounded().await.unwrap().unwrap());
    h.server.set_serving(false);
    h.server.state.shared.registry.cancel_all().unwrap();
    let cut = c1.message().bounded().await.expect_err("a cut is a status");
    assert_eq!(cut.code(), Code::Unavailable);
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["c1".to_string()]);
    wait_until(|| h.server.active_requests() == 0).await;

    let body = scrape(&h.server);
    assert_bucket_edges(&body, 4096);
    assert_golden(&body, GOLDEN_SCRAPE);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A stop string this servicer matches is its own stop: counted as one,
/// and the request ends as `stop`.
#[tokio::test]
async fn a_string_stop_counts_as_the_servicers_own_stop() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let mut h = metrics_harness(model_info(), Some(tokenizer)).await;
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
    recv_add(&mut h.engine_in).await;
    send(&mut h.engine_out, &batch("r3", vec![1], 1, None, None)).await;
    send(&mut h.engine_out, &batch("r3", vec![2], 2, None, None)).await;
    chunk_tokens(stream.message().bounded().await.unwrap().unwrap());
    chunk_tokens(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(
        complete(stream.message().bounded().await.unwrap().unwrap()).finish_reason,
        "stop"
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r3".to_string()]);
    wait_until(|| h.server.active_requests() == 0).await;
    let body = scrape(&h.server);
    assert_eq!(
        sample(&body, "smg_servicer_string_stops_total", "rank=\"0\""),
        1.0
    );
    assert_eq!(
        sample(
            &body,
            "smg_servicer_requests_total",
            "finish_reason=\"stop\""
        ),
        1.0
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The listener is bound before the scheduler connects, so a scrape during
/// its start answers `starting`; once it is up, `serving`.
#[tokio::test]
async fn the_listener_binds_before_the_engine_connects_and_answers_starting() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let mut config = config(dir.path(), &handshake, model_info());
    config.servicer.metrics_bind_address = Some("127.0.0.1:0".to_string());
    let server = TokenSpeedServicerServer::start(config).expect("servicer starts");
    let body = scrape(&server);
    assert_eq!(
        sample(&body, "smg_servicer_state", "state=\"starting\""),
        1.0
    );
    let engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock scheduler handshake");
    wait_until(|| server.engine_ready()).await;
    let body = scrape(&server);
    assert_eq!(
        sample(&body, "smg_servicer_state", "state=\"serving\""),
        1.0
    );
    drop(engine);
    server.stop(Duration::from_secs(5)).expect("clean stop");
}
