//! The Rust SGLang servicer against a mock scheduler on the msgpack wire: the
//! mock decodes the `TokenizedGenerateReqInput` frames the servicer sends and
//! answers with `BatchTokenIDSlimOutput` batches, as the SMG plugin inside a
//! headless scheduler does.

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use engine_zmq_client::{
    codec::{decode_msgpack, encode_msgpack, OpaqueValue},
    mock_engine::{
        connect_to_frontend, default_ready_response, hello, MockEngineInput, MockEngineOutput,
        MOCK_DEADLINE,
    },
    protocol::{
        handshake::EngineCoreReadyResponse,
        sglang::{
            output::{
                BatchEmbeddingSlimOutput, BatchTokenIDSlimOutput, ControlReplySlim, MatchedStop,
            },
            request::{
                RequestMeta, SglangRequestType, TokenizedEmbeddingReqInput,
                TokenizedGenerateReqInput,
            },
        },
    },
    EngineId,
};
use futures::StreamExt;
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
use crate::{
    testing::{assert_bucket_edges, assert_golden, http_get, servicer_config, Bounded},
    EncodeDispatchErrorSink, EncodeDispatchOutcome, EncodeDispatchRequest, EncodeFuture,
    EncodeRelease, EncoderDispatchErrorReq, ServicerError, SglangEncodeDispatcher,
};

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
        server_args_json: r#"{"model_path":"org/model","tp_size":2,"dp_size":1,"context_length":4096,"pairing_protocol":"p1","engine_reasoning_parser":"qwen3","constrained_decoding_start":"after_reasoning"}"#.to_string(),
        scheduler_info_json: r#"{"is_generation":true}"#.to_string(),
        sglang_version: "0.5.20".to_string(),
        max_running_requests: 64,
        data_parallel_size: 1,
        ..Default::default()
    }
}

fn config(dir: &std::path::Path, handshake: &str, model: SglangModelInfo) -> SglangServicerConfig {
    SglangServicerConfig {
        servicer: servicer_config(dir, handshake),
        model,
        encode_dispatcher: None,
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

/// A bound servicer, a handshaken mock scheduler (rank 0; the further ranks
/// of a data-parallel group in `more_ranks`, from index 1), and a gRPC
/// client.
struct Harness {
    server: SglangServicerServer,
    engine_in: MockEngineInput,
    engine_out: MockEngineOutput,
    more_ranks: Vec<(MockEngineInput, MockEngineOutput)>,
    client: SglangSchedulerClient<Channel>,
    _dir: tempfile::TempDir,
}

async fn harness(model: SglangModelInfo) -> Harness {
    harness_with(model, |_| {}).await
}

/// [`harness`] with the servicer's config adjusted before it starts (a
/// metrics listener, an encode dispatcher, the rank count): as many mock
/// schedulers handshake as the config expects, in index order.
async fn harness_with(
    model: SglangModelInfo,
    configure: impl FnOnce(&mut SglangServicerConfig),
) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let mut config = config(dir.path(), &handshake, model);
    configure(&mut config);
    let ranks = u32::try_from(config.servicer.engine_count).expect("a small rank count");
    let server = SglangServicerServer::start(config).expect("servicer starts");
    let mut engines = Vec::new();
    for index in 0..ranks {
        engines.push(
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(index),
                default_ready_response(),
            )
            .await
            .expect("mock scheduler handshake"),
        );
    }
    wait_until(|| server.engine_ready()).await;
    let client = SglangSchedulerClient::new(grpc_channel(server.address()).await);
    let mut ranks = engines.into_iter().map(|engine| engine.split());
    let (engine_in, engine_out) = ranks.next().expect("rank 0");
    Harness {
        server,
        engine_in,
        engine_out,
        more_ranks: ranks.collect(),
        client,
        _dir: dir,
    }
}

/// A bound servicer over two data-parallel scheduler ranks (two handshaken
/// mock schedulers, each dialing with its rank as its engine index, as the
/// plugin does) and a gRPC client: the harness holds rank 0, the pair
/// returned with it is rank 1.
async fn two_rank_harness(model: SglangModelInfo) -> (Harness, MockEngineInput, MockEngineOutput) {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let mut config = config(dir.path(), &handshake, model);
    config.servicer.engine_count = 2;
    let server = SglangServicerServer::start(config).expect("servicer starts");
    let rank = |index: u32| {
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(index),
            EngineCoreReadyResponse {
                data_parallel_size: 2,
                data_parallel_rank: index,
                ..default_ready_response()
            },
        )
    };
    let (rank0, rank1) = tokio::join!(rank(0), rank(1));
    let rank0 = rank0.expect("rank 0 handshake");
    let rank1 = rank1.expect("rank 1 handshake");
    wait_until(|| server.engine_ready()).await;
    let client = SglangSchedulerClient::new(grpc_channel(server.address()).await);
    let (engine_in, engine_out) = rank0.split();
    let (rank1_in, rank1_out) = rank1.split();
    (
        Harness {
            server,
            engine_in,
            engine_out,
            client,
            more_ranks: Vec::new(),
            _dir: dir,
        },
        rank1_in,
        rank1_out,
    )
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
    let server = SglangServicerServer::start(config).unwrap();
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
    let server = SglangServicerServer::start(config).unwrap();
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
    let server = SglangServicerServer::start(config).unwrap();
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
    let with = |edit: fn(&mut ServicerConfig)| {
        let mut config = config(dir.path(), "tcp://127.0.0.1:1", model_info());
        edit(&mut config.servicer);
        config
    };
    let bad_bind = with(|servicer| servicer.bind_address = "nope".to_string());
    assert!(matches!(
        SglangServicerServer::start(bad_bind),
        Err(ServicerError::InvalidConfig(_))
    ));
    let no_engines = with(|servicer| servicer.engine_count = 0);
    assert!(matches!(
        SglangServicerServer::start(no_engines),
        Err(ServicerError::InvalidConfig(_))
    ));
    let zero_timeout = with(|servicer| servicer.engine_startup_timeout = Duration::ZERO);
    assert!(matches!(
        SglangServicerServer::start(zero_timeout),
        Err(ServicerError::InvalidConfig(_))
    ));
    let zero_ceiling = with(|servicer| {
        servicer.engine_startup_ceiling = Some(Duration::ZERO);
    });
    assert!(matches!(
        SglangServicerServer::start(zero_ceiling),
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
    // Readiness held alone (the lifecycle owner's planned warm-up): health
    // answers NOT_SERVING with its own message while admission stays open to
    // the warm-up's own requests only, then the release restores SERVING and
    // admits every request; the drain below closes admission.
    server.set_ready(false);
    let held = client
        .health_check(sg::HealthCheckRequest {})
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
            .health_check(sg::HealthCheckRequest {})
            .await
            .unwrap()
            .into_inner()
            .healthy
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
    let mut h = harness(model_info()).await;
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
    let mut h = harness(model_info()).await;
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
    assert_eq!(h.server.active_requests(), 0);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A prefill/decode request's rendezvous reaches the scheduler in its own
/// slots, as the Python servicer forwards it (an empty host and a zero port
/// unset, the room kept as given, 0 included); a plain request writes none;
/// the response path is the same either way.
#[tokio::test]
async fn a_disaggregated_request_carries_its_bootstrap_to_the_scheduler() {
    let mut h = harness(model_info()).await;
    let mut request = generate_request("pd1", false, Vec::new());
    request.disaggregated_params = Some(sg::DisaggregatedParams {
        bootstrap_host: "10.0.0.7".to_string(),
        bootstrap_port: 8998,
        bootstrap_room: 41,
    });
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let add = recv_add(&mut h.engine_in).await;
    assert_eq!(add.rid, "pd1");
    assert_eq!(add.bootstrap_host.as_deref(), Some("10.0.0.7"));
    assert_eq!(add.bootstrap_port, Some(8998));
    assert_eq!(add.bootstrap_room, Some(41));
    // A prefill side answers its one token and finishes on length once the
    // KV is handed over; the servicer relays it as any completion.
    send(
        &mut h.engine_out,
        &batch("pd1", vec![10], 1, Some("length"), None),
    )
    .await;
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.output_ids, vec![10]);
    assert!(stream.message().bounded().await.unwrap().is_none());

    // The fake-transfer probe: a host, no port, room 0 (kept).
    let mut probe = generate_request("probe", false, Vec::new());
    probe.disaggregated_params = Some(sg::DisaggregatedParams {
        bootstrap_host: "2.2.2.2".to_string(),
        bootstrap_port: 0,
        bootstrap_room: 0,
    });
    let mut stream = h
        .client
        .generate(probe)
        .await
        .expect("generate")
        .into_inner();
    let add = recv_add(&mut h.engine_in).await;
    assert_eq!(add.bootstrap_host.as_deref(), Some("2.2.2.2"));
    assert_eq!((add.bootstrap_port, add.bootstrap_room), (None, Some(0)));
    send(
        &mut h.engine_out,
        &batch("probe", vec![11], 1, Some("length"), None),
    )
    .await;
    assert_eq!(
        complete(stream.message().bounded().await.unwrap().unwrap()).finish_reason,
        "length"
    );
    assert!(stream.message().bounded().await.unwrap().is_none());

    let mut stream = h
        .client
        .generate(generate_request("plain", false, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    let add = recv_add(&mut h.engine_in).await;
    assert_eq!(
        (add.bootstrap_host, add.bootstrap_port, add.bootstrap_room),
        (None, None, None)
    );
    send(
        &mut h.engine_out,
        &batch("plain", vec![12], 1, Some("stop"), None),
    )
    .await;
    assert_eq!(
        complete(stream.message().bounded().await.unwrap().unwrap()).finish_reason,
        "stop"
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    wait_until(|| h.server.active_requests() == 0).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A request the Router pins to a data-parallel rank reaches that rank's
/// scheduler, with the rank in the request's own `routed_dp_rank` slot, the
/// way the engine's data-parallel controller dispatches a pinned request for
/// its frontend; a pin past the connected ranks is the caller's error, in
/// the frontend's words, and reaches no scheduler.
#[tokio::test]
async fn a_pinned_request_reaches_its_data_parallel_rank() {
    let mut model = model_info();
    model.data_parallel_size = 2;
    let (mut h, mut rank1_in, mut rank1_out) = two_rank_harness(model).await;

    let mut request = generate_request("dp1", false, Vec::new());
    request.data_parallel_rank = Some(1);
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let add = recv_add(&mut rank1_in).await;
    assert_eq!(add.rid, "dp1");
    assert_eq!(add.routed_dp_rank, Some(1));
    rank1_out
        .send_frames(vec![Bytes::from(
            encode_msgpack(&BatchTokenIDSlimOutput {
                engine_index: 1,
                ..batch("dp1", vec![10], 1, Some("stop"), None)
            })
            .unwrap(),
        )])
        .await
        .unwrap();
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![10]);
    assert!(stream.message().bounded().await.unwrap().is_none());

    let mut request = generate_request("dp0", false, Vec::new());
    request.data_parallel_rank = Some(0);
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let add = recv_add(&mut h.engine_in).await;
    assert_eq!((add.rid.as_str(), add.routed_dp_rank), ("dp0", Some(0)));
    send(
        &mut h.engine_out,
        &batch("dp0", vec![11], 1, Some("stop"), None),
    )
    .await;
    assert_eq!(
        complete(stream.message().bounded().await.unwrap().unwrap()).finish_reason,
        "stop"
    );
    assert!(stream.message().bounded().await.unwrap().is_none());

    let mut request = generate_request("dp2", false, Vec::new());
    request.data_parallel_rank = Some(2);
    let status = h
        .client
        .generate(request)
        .await
        .expect_err("a pin past the connected ranks is refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(status.message(), "routed_dp_rank=2 out of range [0, 2)");
    assert_scheduler_idle(&mut h.engine_in).await;
    assert_scheduler_idle(&mut rank1_in).await;
    wait_until(|| h.server.active_requests() == 0).await;
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
/// an unknown id is a no-op reported as not found. The caller's own abort
/// stays a completion; only a shutdown under an open request is a status
/// (`a_shutdown_ends_an_open_stream_with_unavailable_not_an_abort_completion`).
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

/// A model that takes multimodal input is advertised as the launcher reports
/// it (`GetModelInfo.supports_vision`, SGLang's own verdict for the launch),
/// and `GetServerInfo.server_args` stays the launcher's: no verdict of the
/// servicer's own there (the Python servicer writes none either), so the
/// gateway's media pipeline preprocesses for the worker and the wire carries
/// the tensors.
#[tokio::test]
async fn a_multimodal_model_is_advertised_with_the_launchers_vision_verdict() {
    let mut vision = model_info();
    vision.model_type = "qwen3_vl".to_string();
    vision.architectures = vec!["Qwen3VLForConditionalGeneration".to_string()];
    vision.supports_vision = true;
    let mut h = harness(vision).await;
    let info = h
        .client
        .get_model_info(sg::GetModelInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(info.supports_vision, "the launcher's verdict, as reported");
    let server = h
        .client
        .get_server_info(sg::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    let args = server.server_args.expect("server_args");
    assert!(
        !args.fields.contains_key("supports_vision"),
        "no verdict of the servicer's own in server_args"
    );
    assert_eq!(
        args.fields["tp_size"].kind,
        Some(Kind::NumberValue(2.0)),
        "the launcher's server_args stay as written"
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    // A text model reads as the launcher reported it too.
    let mut h = harness(model_info()).await;
    let info = h
        .client
        .get_model_info(sg::GetModelInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(!info.supports_vision);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The request's preprocessed media reaches the scheduler as its own
/// processor output: one image item with the pixel tensor as its feature,
/// the model-specific tensors beside it, the placeholder ranges as inclusive
/// offsets and the image token id; every choice of an `n > 1` fan-out
/// carries it (shared bytes).
#[tokio::test]
async fn multimodal_inputs_reach_the_scheduler_as_its_processor_output() {
    use engine_zmq_client::protocol::sglang::multimodal::SglangWireModality;
    let mut vision = model_info();
    vision.supports_vision = true;
    let mut h = harness(vision).await;
    let pixels: Vec<u8> = (0..8)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();
    let grid: Vec<u8> = [1i64, 2, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut request = generate_request("mm", true, Vec::new());
    request.tokenized = Some(sg::TokenizedInput {
        original_text: String::new(),
        input_ids: vec![1, 7, 7, 7, 2, 3, 4, 5, 6],
    });
    request.sampling_params.as_mut().unwrap().n = 2;
    request.mm_inputs = Some(sg::MultimodalInputs {
        modalities: vec!["image".to_string()],
        pixel_values: Some(sg::TensorData {
            data: pixels.clone().into(),
            shape: vec![2, 4],
            dtype: "float32".to_string(),
        }),
        model_specific_tensors: [(
            "image_grid_thw".to_string(),
            sg::TensorData {
                data: grid.clone().into(),
                shape: vec![1, 3],
                dtype: "int64".to_string(),
            },
        )]
        .into_iter()
        .collect(),
        im_token_id: Some(7),
        mm_placeholders: vec![sg::PlaceholderRange {
            offset: 1,
            length: 3,
        }],
        ..Default::default()
    });
    let _stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let mut rids = Vec::new();
    for _ in 0..2 {
        let engine_request = recv_add(&mut h.engine_in).await;
        let media = engine_request
            .mm_inputs
            .as_ref()
            .expect("the processor output on every choice");
        assert_eq!(media.im_token_id, Some(7));
        assert_eq!(media.mm_items.len(), 1);
        let item = &media.mm_items[0];
        assert_eq!(item.modality, SglangWireModality::Image);
        assert_eq!(item.offsets, Some(vec![(1, 3)]));
        let feature = item.feature.as_ref().expect("the pixel tensor");
        assert_eq!(
            (feature.dtype.as_str(), feature.shape.as_slice()),
            ("float32", &[2, 4][..])
        );
        assert_eq!(feature.data.as_ref(), pixels.as_slice());
        let grid_tensor = &item.model_specific_data["image_grid_thw"];
        assert_eq!(
            (grid_tensor.dtype.as_str(), grid_tensor.shape.as_slice()),
            ("int64", &[1, 3][..])
        );
        assert_eq!(grid_tensor.data.as_ref(), grid.as_slice());
        assert_eq!(engine_request.input_ids, vec![1, 7, 7, 7, 2, 3, 4, 5, 6]);
        rids.push(engine_request.rid);
    }
    rids.sort();
    assert_eq!(rids, vec!["mm-0".to_string(), "mm-1".to_string()]);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

const ENCODERS: [&str; 2] = ["grpc://10.0.0.8:30001", "grpc://10.0.0.9:30001"];
const UNEXPANDED_PROMPT: &str = "<|vision_start|><|image_pad|><|vision_end|>What is this?";
const IMAGE_REFERENCE: &str = "data:image/png;base64,iVBORw0KGgo=";

/// How a scripted dispatcher answers every call.
enum DispatchAnswer {
    /// Two encoders registered, the one image assigned to the first: the
    /// engine's frontend with encoders behind it.
    Encoders,
    /// No encoder registered: the frontend leaves the request alone.
    NoEncoder,
    /// The dispatch itself fails before anything went out.
    Failure(&'static str),
}

/// The lifecycle owner's dispatcher, scripted: records every call and the
/// sink it was handed, counts its releases.
struct ScriptedDispatcher {
    answer: DispatchAnswer,
    calls: Mutex<Vec<(String, Vec<String>)>>,
    sinks: Mutex<Vec<EncodeDispatchErrorSink>>,
    released: Arc<AtomicUsize>,
}

impl ScriptedDispatcher {
    fn new(answer: DispatchAnswer) -> Arc<Self> {
        Arc::new(Self {
            answer,
            calls: Mutex::default(),
            sinks: Mutex::default(),
            released: Arc::default(),
        })
    }

    fn calls(&self) -> Vec<(String, Vec<String>)> {
        self.calls.lock().unwrap().clone()
    }

    fn released(&self) -> usize {
        self.released.load(Ordering::SeqCst)
    }

    /// The sink handed over with the call for `rid`.
    fn sink_for(&self, rid: &str) -> EncodeDispatchErrorSink {
        let calls = self.calls.lock().unwrap();
        let index = calls
            .iter()
            .position(|(called, _)| called == rid)
            .expect("a dispatched rid");
        Arc::clone(&self.sinks.lock().unwrap()[index])
    }

    /// The servicer's config with this dispatcher and `ranks` scheduler ranks.
    fn configure(self: &Arc<Self>, ranks: usize) -> impl FnOnce(&mut SglangServicerConfig) {
        let dispatcher: Arc<dyn SglangEncodeDispatcher> = Arc::<Self>::clone(self);
        move |config| {
            config.encode_dispatcher = Some(dispatcher);
            config.servicer.engine_count = ranks;
        }
    }
}

impl SglangEncodeDispatcher for ScriptedDispatcher {
    fn dispatch(
        &self,
        request: EncodeDispatchRequest,
    ) -> EncodeFuture<Result<EncodeDispatchOutcome, String>> {
        self.calls
            .lock()
            .unwrap()
            .push((request.rid, request.image_urls));
        self.sinks.lock().unwrap().push(request.on_error);
        let released = Arc::clone(&self.released);
        // Built only where the dispatch answers: the guard runs on drop, and a
        // failed dispatch hands out no release.
        let sent = move || {
            EncodeRelease::new(
                Box::new(move || {
                    Box::pin(async move {
                        released.fetch_add(1, Ordering::SeqCst);
                    })
                }),
                tokio::runtime::Handle::current(),
            )
        };
        let answer = match &self.answer {
            DispatchAnswer::Failure(message) => Err((*message).to_string()),
            DispatchAnswer::Encoders => Ok(EncodeDispatchOutcome {
                need_wait_for_mm_inputs: true,
                num_items_assigned: Some(BTreeMap::from([(1, vec![1, 0])])),
                encoder_urls: Some(ENCODERS.iter().map(ToString::to_string).collect()),
                sent: Some(sent()),
            }),
            DispatchAnswer::NoEncoder => Ok(EncodeDispatchOutcome {
                need_wait_for_mm_inputs: false,
                num_items_assigned: None,
                encoder_urls: None,
                sent: Some(sent()),
            }),
        };
        Box::pin(std::future::ready(answer))
    }
}

/// A vision model served by `ranks` scheduler ranks.
fn vision_model(ranks: i32) -> SglangModelInfo {
    let mut model = model_info();
    model.supports_vision = true;
    model.data_parallel_size = ranks;
    model
}

/// A request naming one image by reference beside its unexpanded prompt,
/// `n` choices: what reaches a language-side worker.
fn media_reference_request(id: &str, n: u32) -> sg::GenerateRequest {
    let mut request = generate_request(id, false, Vec::new());
    request.tokenized = Some(sg::TokenizedInput {
        original_text: UNEXPANDED_PROMPT.to_string(),
        input_ids: vec![1, 7, 7, 7, 2],
    });
    request.sampling_params.as_mut().unwrap().n = n;
    request.mm_inputs = Some(sg::MultimodalInputs {
        modalities: vec!["image".to_string()],
        image_urls: vec![IMAGE_REFERENCE.to_string()],
        ..Default::default()
    });
    request
}

/// On a language-side worker a request's image references go to the
/// encoders before the request goes to its scheduler rank, one dispatch per
/// choice under the choice's own id; the scheduler's request carries the
/// unexpanded prompt and the three slots the dispatch set, and no processor
/// output. A failure the dispatcher reports once the request is out reaches
/// the rank that holds the request, after it, as the scheduler's own notice;
/// the other rank sees nothing.
#[tokio::test]
async fn a_language_side_request_is_dispatched_to_the_encoders_before_its_rank() {
    let dispatcher = ScriptedDispatcher::new(DispatchAnswer::Encoders);
    let mut h = harness_with(vision_model(2), dispatcher.configure(2)).await;
    let mut stream = h
        .client
        .generate(media_reference_request("epd", 2))
        .await
        .expect("generate")
        .into_inner();
    let (mut rank1_in, mut rank1_out) = h.more_ranks.pop().expect("the second rank");
    // The two choices spread over the two idle ranks (the connector reserves
    // its choice): one scheduler request on each.
    let mut rid_by_rank = BTreeMap::new();
    for (rank, engine) in [(0u32, &mut h.engine_in), (1, &mut rank1_in)] {
        let add = recv_add(engine).await;
        assert_eq!(add.need_wait_for_mm_inputs, Some(true), "{}", add.rid);
        assert_eq!(
            add.num_items_assigned,
            Some(BTreeMap::from([(1, vec![1, 0])]))
        );
        assert_eq!(
            add.encoder_urls,
            Some(ENCODERS.iter().map(ToString::to_string).collect())
        );
        assert_eq!(add.input_text.as_deref(), Some(UNEXPANDED_PROMPT));
        assert_eq!(add.input_ids, vec![1, 7, 7, 7, 2]);
        assert!(
            add.mm_inputs.is_none(),
            "no processor output on the language side"
        );
        rid_by_rank.insert(rank, add.rid);
    }
    let mut rids: Vec<String> = rid_by_rank.values().cloned().collect();
    rids.sort();
    assert_eq!(rids, ["epd-0", "epd-1"]);
    let calls = dispatcher.calls();
    assert_eq!(calls.len(), 2, "{calls:?}");
    for (rid, urls) in &calls {
        assert!(rids.contains(rid), "{rid}");
        assert_eq!(urls, &[IMAGE_REFERENCE.to_string()]);
    }
    // Released once each request was on the wire.
    wait_until(|| dispatcher.released() == 2).await;

    // The encoder behind rank 1's request fails: the notice follows the
    // request on rank 1 (the bytes the wire test pins); rank 0 sees nothing.
    let failing = rid_by_rank[&1].clone();
    let error = EncoderDispatchErrorReq {
        rid: failing.clone(),
        error_msg: format!("Encode failed for request {failing}: boom"),
        error_code: 502,
    };
    dispatcher.sink_for(&failing)(error.clone());
    let frames = rank1_in.recv_frames().await.unwrap();
    assert_eq!(
        SglangRequestType::from_frame(frames[0].as_ref()),
        Some(SglangRequestType::Add)
    );
    let notice: EncoderDispatchErrorReq = decode_msgpack(frames[1].as_ref()).unwrap();
    assert_eq!(notice, error);
    assert_scheduler_idle(&mut h.engine_in).await;

    // Both ranks finish their request; the caller gets its two completions.
    for (rank, engine_out) in [(0u32, &mut h.engine_out), (1, &mut rank1_out)] {
        send(
            engine_out,
            &batch(&rid_by_rank[&rank], vec![10], 1, Some("stop"), None),
        )
        .await;
    }
    for _ in 0..2 {
        let done = complete(stream.message().bounded().await.unwrap().unwrap());
        assert_eq!(done.finish_reason, "stop");
    }
    assert!(stream.message().bounded().await.unwrap().is_none());
    wait_until(|| h.server.active_requests() == 0).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Media references a worker cannot have encoded are refused before the
/// scheduler, by name: on a worker with no encode dispatch (not a
/// language-side worker), on a language-side worker whose dispatch finds no
/// encoder registered (the dispatcher is released even so), and when the
/// dispatch itself fails; a failure reported for a request that never went
/// out reaches no rank. The refused id is free again: a text request under
/// it reaches the scheduler.
#[tokio::test]
async fn media_references_a_worker_cannot_encode_are_refused_before_the_scheduler() {
    let mut h = harness(vision_model(1)).await;
    let status = h
        .client
        .generate(media_reference_request("refs", 1))
        .await
        .expect_err("no encode dispatch on this worker");
    assert_eq!(
        status.code(),
        Code::FailedPrecondition,
        "{}",
        status.message()
    );
    assert!(
        status.message().contains("runs no encode dispatch"),
        "{}",
        status.message()
    );
    assert_scheduler_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    for (answer, code, needle) in [
        (
            DispatchAnswer::NoEncoder,
            Code::FailedPrecondition,
            "no encoder is registered with this language-only worker",
        ),
        (
            DispatchAnswer::Failure("the encoder list is empty"),
            Code::Internal,
            "encode dispatch failed: the encoder list is empty",
        ),
    ] {
        let releases = usize::from(matches!(answer, DispatchAnswer::NoEncoder));
        let dispatcher = ScriptedDispatcher::new(answer);
        let mut h = harness_with(vision_model(1), dispatcher.configure(1)).await;
        let status = h
            .client
            .generate(media_reference_request("refs", 1))
            .await
            .expect_err("refused before the scheduler");
        assert_eq!(status.code(), code, "{}", status.message());
        assert!(status.message().contains(needle), "{}", status.message());
        assert_eq!(
            dispatcher.calls(),
            vec![("refs".to_string(), vec![IMAGE_REFERENCE.to_string()])]
        );
        assert_eq!(dispatcher.released(), releases);
        // A failure reported now is for a request no rank holds: dropped.
        dispatcher.sink_for("refs")(EncoderDispatchErrorReq {
            rid: "refs".to_string(),
            error_msg: "late".to_string(),
            error_code: 500,
        });
        assert_scheduler_idle(&mut h.engine_in).await;
        let _stream = h
            .client
            .generate(generate_request("refs", true, Vec::new()))
            .await
            .expect("the id is free again")
            .into_inner();
        assert_eq!(recv_add(&mut h.engine_in).await.rid, "refs");
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }
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
    // The launcher's grammar-start facts (the engine's reasoning parser and
    // where its grammar starts on a thinking prompt) reach the Router's label
    // source as the launcher wrote them.
    assert_eq!(
        args.fields["engine_reasoning_parser"].kind,
        Some(Kind::StringValue("qwen3".to_string()))
    );
    assert_eq!(
        args.fields["constrained_decoding_start"].kind,
        Some(Kind::StringValue("after_reasoning".to_string()))
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

/// Hidden states ride the appended slim column as the rows each tick adds:
/// each chunk carries them flattened (the prefill block on the first, a decode
/// step's rows on each later one); the `Complete` one entry per position a
/// forward ran on (the prompt positions the prefill computed, then each
/// generated token fed back; the last sampled token has none), the final
/// layer, the positions absolute (a prefix the radix cache served has no
/// rows: the anchor comes from the scheduler's counters); a request that did
/// not ask carries none.
#[tokio::test]
async fn hidden_states_pass_through_as_tick_rows_and_positions() {
    let mut model = model_info();
    model.server_args_json = r#"{"return_hidden_states_mode":"full"}"#.to_string();
    let mut h = harness(model).await;
    let request = sg::GenerateRequest {
        return_hidden_states: true,
        ..generate_request("h1", true, Vec::new())
    };
    let mut stream = h.client.generate(request).await.unwrap().into_inner();
    let sent = recv_add(&mut h.engine_in).await;
    assert!(sent.return_hidden_states);
    // The prefill block: one row per prompt position (three), hidden size 2.
    // The first token is sampled from the last of them, so the prefill's tick
    // carries no row of its own.
    let mut rows = vec![vec![0.5, -0.5], vec![0.25, 0.75], vec![1.0, 0.0]];
    let first = BatchTokenIDSlimOutput {
        output_hidden_states: vec![Some(rows.clone())],
        ..batch("h1", vec![10], 1, None, None)
    };
    send(&mut h.engine_out, &first).await;
    let chunk1 = chunk(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(chunk1.hidden_states, vec![0.5, -0.5, 0.25, 0.75, 1.0, 0.0]);
    // The decode step that samples the second token fed the first one back:
    // its row, the only one this tick adds, and the request's last
    // (prompt + generated - 1 = 4).
    rows.push(vec![3.0, 4.0]);
    let last = BatchTokenIDSlimOutput {
        output_hidden_states: vec![Some(vec![vec![3.0, 4.0]])],
        ..batch("h1", vec![11], 2, Some("stop"), None)
    };
    send(&mut h.engine_out, &last).await;
    let chunk2 = chunk(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(chunk2.hidden_states, vec![3.0, 4.0], "the tick's row only");
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![10, 11]);
    assert_eq!(done.all_hidden_states.len(), rows.len());
    for (position, (entry, row)) in done.all_hidden_states.iter().zip(&rows).enumerate() {
        assert_eq!(entry.values, *row);
        assert_eq!(entry.layer, -1, "the final hidden layer");
        assert_eq!(entry.position, i32::try_from(position).unwrap());
    }
    assert!(stream.message().bounded().await.unwrap().is_none());

    // A prefix the radix cache served has no rows: the prefill block holds the
    // computed positions only, and the positions stay absolute, anchored from
    // the scheduler's counters (a prompt of five, three of them cached, two
    // generated: rows at 3, 4 and 5).
    let request = sg::GenerateRequest {
        return_hidden_states: true,
        ..generate_request("h2", true, Vec::new())
    };
    let mut stream = h.client.generate(request).await.unwrap().into_inner();
    recv_add(&mut h.engine_in).await;
    let first = BatchTokenIDSlimOutput {
        prompt_tokens: vec![5],
        cached_tokens: vec![3],
        output_hidden_states: vec![Some(vec![vec![5.0, 5.5], vec![6.0, 6.5]])],
        ..batch("h2", vec![10], 1, None, None)
    };
    send(&mut h.engine_out, &first).await;
    let chunk1 = chunk(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(chunk1.hidden_states, vec![5.0, 5.5, 6.0, 6.5]);
    let last = BatchTokenIDSlimOutput {
        prompt_tokens: vec![5],
        cached_tokens: vec![3],
        output_hidden_states: vec![Some(vec![vec![7.0, 7.5]])],
        ..batch("h2", vec![11], 2, Some("stop"), None)
    };
    send(&mut h.engine_out, &last).await;
    let chunk2 = chunk(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(chunk2.hidden_states, vec![7.0, 7.5]);
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(
        done.all_hidden_states
            .iter()
            .map(|entry| entry.position)
            .collect::<Vec<_>>(),
        vec![3, 4, 5]
    );
    assert_eq!(done.all_hidden_states[2].values, vec![7.0, 7.5]);
    assert!(stream.message().bounded().await.unwrap().is_none());

    // Without the flag nothing is asked and nothing rides the responses.
    let mut stream = h
        .client
        .generate(generate_request("h2", true, Vec::new()))
        .await
        .unwrap()
        .into_inner();
    assert!(!recv_add(&mut h.engine_in).await.return_hidden_states);
    send(
        &mut h.engine_out,
        &batch("h2", vec![10], 1, Some("stop"), None),
    )
    .await;
    let chunk1 = chunk(stream.message().bounded().await.unwrap().unwrap());
    assert!(chunk1.hidden_states.is_empty());
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert!(done.all_hidden_states.is_empty());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `return_hidden_states` is kept to the scheduler's launch-time cap ahead of
/// the scheduler, with the engine frontend's own words: refused under
/// `--return-hidden-states-mode last` or none, forwarded under `full` (the
/// earlier `--enable-return-hidden-states` included) and when the launcher's
/// server args name no cap at all.
#[tokio::test]
async fn hidden_states_are_kept_to_the_schedulers_launch_time_cap() {
    for (args, words) in [
        (
            r#"{"return_hidden_states_mode":"last","enable_return_hidden_states":false}"#,
            "exceeds the server maximum `last`",
        ),
        (
            // The engine's order: `last` wins over the legacy flag it sets itself.
            r#"{"return_hidden_states_mode":"last","enable_return_hidden_states":true}"#,
            "exceeds the server maximum `last`",
        ),
        (
            r#"{"return_hidden_states_mode":null,"enable_return_hidden_states":false}"#,
            "not configured to return hidden states",
        ),
    ] {
        let mut model = model_info();
        model.server_args_json = args.to_string();
        let mut h = harness(model).await;
        let request = sg::GenerateRequest {
            return_hidden_states: true,
            ..generate_request("c1", false, Vec::new())
        };
        let refused = h.client.generate(request).await.expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert!(
            refused.message().contains(words)
                && refused
                    .message()
                    .contains("`--return-hidden-states-mode full`"),
            "{}",
            refused.message()
        );
        assert_scheduler_idle(&mut h.engine_in).await;
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }
    for args in [
        r#"{"return_hidden_states_mode":"full","enable_return_hidden_states":false}"#,
        r#"{"return_hidden_states_mode":null,"enable_return_hidden_states":true}"#,
        r#"{"tp_size":1}"#,
    ] {
        let mut model = model_info();
        model.server_args_json = args.to_string();
        let mut h = harness(model).await;
        let request = sg::GenerateRequest {
            return_hidden_states: true,
            ..generate_request("c2", false, Vec::new())
        };
        let mut stream = h.client.generate(request).await.unwrap().into_inner();
        assert!(
            recv_add(&mut h.engine_in).await.return_hidden_states,
            "{args}"
        );
        send(
            &mut h.engine_out,
            &batch("c2", vec![10], 1, Some("stop"), None),
        )
        .await;
        let done = complete(stream.message().bounded().await.unwrap().unwrap());
        assert_eq!(done.output_ids, vec![10]);
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }
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
    let mut h = harness(model).await;
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
    config.servicer.engine_startup_timeout = Duration::from_millis(300);
    let server = SglangServicerServer::start(config).expect("servicer starts");
    wait_until(|| matches!(server.last_error(), Ok(Some(_)))).await;
    let error = server.last_error().unwrap().unwrap();
    assert!(
        error.starts_with("SGLang scheduler connection failed: "),
        "{error}"
    );
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

/// The Router's W3C trace context (gRPC metadata) rides to the scheduler as
/// the request's metadata frame beside the positional array, which stays the
/// scheduler's own struct; a call without the context sends no such frame.
#[tokio::test]
async fn trace_context_metadata_reaches_the_scheduler_beside_the_request() {
    const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let mut h = harness(model_info()).await;
    let mut request = tonic::Request::new(generate_request("tr1", false, Vec::new()));
    request
        .metadata_mut()
        .insert("traceparent", TRACEPARENT.parse().unwrap());
    request
        .metadata_mut()
        .insert("tracestate", "vendor=a".parse().unwrap());
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let frames = h.engine_in.recv_frames().await.unwrap();
    assert_eq!(
        SglangRequestType::from_frame(frames[0].as_ref()),
        Some(SglangRequestType::Add)
    );
    let sent: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
    assert_eq!(sent.rid, "tr1");
    assert_eq!(frames.len(), 3, "the metadata frame follows the request");
    let meta: RequestMeta = decode_msgpack(frames[2].as_ref()).unwrap();
    assert_eq!(
        meta.trace_headers,
        Some(BTreeMap::from([
            ("traceparent".to_string(), TRACEPARENT.to_string()),
            ("tracestate".to_string(), "vendor=a".to_string()),
        ]))
    );
    send(
        &mut h.engine_out,
        &batch("tr1", vec![7], 1, Some("length"), None),
    )
    .await;
    assert_eq!(
        complete(stream.message().bounded().await.unwrap().unwrap()).finish_reason,
        "length"
    );
    assert!(stream.message().bounded().await.unwrap().is_none());

    let mut stream = h
        .client
        .generate(generate_request("tr2", false, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    let frames = h.engine_in.recv_frames().await.unwrap();
    assert_eq!(frames.len(), 2, "no metadata frame without a context");
    send(
        &mut h.engine_out,
        &batch("tr2", vec![7], 1, Some("length"), None),
    )
    .await;
    complete(stream.message().bounded().await.unwrap().unwrap());
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
        .generate(generate_request("r8", true, Vec::new()))
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
    let mut h = harness(model_info()).await;
    let mut stream = generate::generate(
        &h.server.state,
        generate_request("r12", true, Vec::new()),
        None,
    )
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
    let mut h = harness(model).await;
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

/// A scheduler release whose sampling struct moved a field inside the prefix
/// this servicer fills: the link fails at the handshake with the struct and
/// the field named, before any request is sent to be refused one by one.
#[tokio::test]
async fn a_scheduler_whose_request_layout_moved_fails_the_link_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = SglangServicerServer::start(config(dir.path(), &handshake, model_info()))
        .expect("servicer starts");
    let mut moved = default_ready_response();
    moved.positional_layouts.insert(
        "SamplingParams".to_string(),
        ["max_new_tokens", "stop", "watermark", "stop_token_ids"]
            .map(str::to_string)
            .to_vec(),
    );
    // The mock scheduler registers; the servicer refuses the link right after
    // and drops the transport under the mock, whose connect may then fail.
    let scheduler = tokio::time::timeout(
        MOCK_DEADLINE,
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), moved),
    );
    let failed = wait_until(|| matches!(server.last_error(), Ok(Some(_))));
    let (_scheduler, ()) = tokio::join!(scheduler, failed);
    let error = server.last_error().unwrap().unwrap();
    assert!(
        error.contains(
            "wire layout mismatch: engine EngineId(0000): `SamplingParams` field 2 is \
             `stop_token_ids` on this side and `watermark` on the engine's"
        ),
        "{error}"
    );
    assert!(!server.engine_ready());
    let mut client = SglangSchedulerClient::new(grpc_channel(server.address()).await);
    let health = client
        .health_check(sg::HealthCheckRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(!health.healthy);
    assert_eq!(health.message, "Engine connection failed");
    server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// What the scheduler enforces of a structured request, as the launcher's
/// `server_args` say: the grammar backend it resolved, the kinds that
/// backend compiles, and no backend at all without a tokenizer.
#[test]
fn structured_outputs_support_follows_the_launcher_args() {
    use engine_zmq_adapter::{
        ConstraintKinds, StructuredOutputBackend,
        StructuredOutputsBackendConfig::{Other, Pinned},
        StructuredOutputsSupport,
    };
    let support = |server_args_json: &str| {
        structured_outputs_support(&SglangModelInfo {
            server_args_json: server_args_json.to_string(),
            ..model_info()
        })
    };
    assert_eq!(
        support(r#"{"grammar_backend":"xgrammar"}"#),
        Pinned(StructuredOutputBackend::Xgrammar).into()
    );
    assert_eq!(
        support(r#"{"grammar_backend":"llguidance"}"#),
        Pinned(StructuredOutputBackend::Guidance).into()
    );
    assert_eq!(
        support(r#"{"grammar_backend":"outlines"}"#),
        StructuredOutputsSupport::new(
            Pinned(StructuredOutputBackend::Outlines),
            ConstraintKinds {
                json_schema: true,
                regex: true,
                ebnf: false,
                structural_tag: false,
            },
        )
    );
    assert_eq!(
        support(r#"{"grammar_backend":"none"}"#),
        StructuredOutputsSupport::DISABLED
    );
    // Without its tokenizer the scheduler builds no grammar backend, whatever
    // the flag says.
    assert_eq!(
        support(r#"{"grammar_backend":"xgrammar","skip_tokenizer_init":true}"#),
        StructuredOutputsSupport::DISABLED
    );
    // A plugin backend of the scheduler's is not judged here.
    assert_eq!(support(r#"{"grammar_backend":"my_backend"}"#), Other.into());
    // A launcher that wrote no backend: the engine's default.
    assert_eq!(
        support(r#"{"model_path":"org/model"}"#),
        StructuredOutputsSupport::default()
    );
    assert_eq!(support(""), StructuredOutputsSupport::default());
}

/// A constraint the scheduler's grammar backend cannot enforce is refused
/// before the scheduler, as the caller's error, on this wire as on the
/// others: every kind under `--grammar-backend none` (the scheduler would
/// admit the request and abort it), a kind outside the backend's set, and a
/// schema with a feature xgrammar drops; what the backend compiles reaches
/// the scheduler with its constraint, and an unconstrained request is served
/// as before.
#[tokio::test]
async fn a_constraint_the_grammar_backend_cannot_enforce_is_refused_before_the_scheduler() {
    use sg::sampling_params::Constraint;

    let with_backend = |server_args_json: &str| SglangModelInfo {
        server_args_json: server_args_json.to_string(),
        ..model_info()
    };
    let constrained = |id: &str, constraint: Constraint| {
        let mut request = generate_request(id, false, Vec::new());
        request.sampling_params.as_mut().unwrap().constraint = Some(constraint);
        request
    };
    const SCHEMA: &str = r#"{"type":"object","properties":{"city":{"type":"string"}}}"#;
    let schema = || Constraint::JsonSchema(SCHEMA.to_string());
    let regex = || Constraint::Regex("[0-9]+".to_string());
    let ebnf = || Constraint::EbnfGrammar("root ::= \"a\"".to_string());

    // No backend: every kind is refused up front; nothing reaches the
    // scheduler but the unconstrained request.
    {
        let mut h = harness(with_backend(r#"{"grammar_backend":"none"}"#)).await;
        for (id, constraint) in [
            ("n1", schema()),
            ("n2", regex()),
            ("n3", ebnf()),
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
        let mut h = harness(with_backend(r#"{"grammar_backend":"xgrammar"}"#)).await;
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
                Constraint::JsonSchema(r#"{"type":"integer","multipleOf":3}"#.to_string()),
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

    // outlines compiles JSON schemas and regular expressions only.
    {
        let mut h = harness(with_backend(r#"{"grammar_backend":"outlines"}"#)).await;
        let status = h
            .client
            .generate(constrained("o1", ebnf()))
            .await
            .expect_err("refused");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(
            status.message().contains(
                "`outlines`, the structured-output backend this engine runs, compiles JSON \
                 schemas and regular expressions only"
            ),
            "{}",
            status.message()
        );
        assert_scheduler_idle(&mut h.engine_in).await;
        let _regex_stream = h
            .client
            .generate(constrained("o2", regex()))
            .await
            .expect("generate")
            .into_inner();
        assert_eq!(recv_add(&mut h.engine_in).await.rid, "o2");
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }
}

/// A prompt that fills the scheduler's context window, or an empty one, is
/// refused before the scheduler sees it, with both lengths named; the
/// refusal releases the registration, so the same id then serves a healthy
/// request.
#[tokio::test]
async fn a_prompt_the_window_cannot_hold_is_refused_before_the_scheduler() {
    let mut h = harness(model_info()).await;
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

/// A scheduler started with `--allow-auto-truncate` cuts a long prompt down
/// to its own bound instead of aborting the request: its servicer leaves the
/// upper bound to it, so the window-filling prompt reaches the scheduler
/// whole, while an empty prompt is still refused before it.
#[tokio::test]
async fn a_scheduler_started_to_truncate_keeps_the_upper_bound_to_itself() {
    let mut model = model_info();
    let args = model.server_args_json.replace(
        "\"tp_size\":2",
        "\"tp_size\":2,\"allow_auto_truncate\":true",
    );
    assert_ne!(
        args, model.server_args_json,
        "the flag was added to the launcher's arguments"
    );
    model.server_args_json = args;
    let mut h = harness(model).await;
    let mut request = generate_request("long", true, Vec::new());
    request.tokenized.as_mut().unwrap().input_ids = vec![7; 1_049_692];
    let _stream = h
        .client
        .generate(request)
        .await
        .expect("left to the scheduler's own bound")
        .into_inner();
    let accepted = recv_add(&mut h.engine_in).await;
    assert_eq!(accepted.rid, "long");
    assert_eq!(accepted.input_ids.len(), 1_049_692);
    let mut empty = generate_request("empty", true, Vec::new());
    empty.tokenized.as_mut().unwrap().input_ids.clear();
    let status = h
        .client
        .generate(empty)
        .await
        .expect_err("an empty prompt is refused whatever the scheduler truncates");
    assert_eq!(status.code(), Code::InvalidArgument);
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
    let server = SglangServicerServer::start(config).unwrap();
    let engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock scheduler handshake");
    wait_until(|| server.engine_ready()).await;
    let mut client = SglangSchedulerClient::new(grpc_channel(server.address()).await);
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
async fn metrics_harness(model: SglangModelInfo) -> Harness {
    harness_with(model, |config| {
        config.servicer.metrics_bind_address = Some("127.0.0.1:0".to_string());
    })
    .await
}

/// One scrape of the servicer's metrics listener.
fn scrape(server: &SglangServicerServer) -> String {
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
/// pinned apart) and the clock-dependent values replaced by `<v>`. This
/// wire has no string-stop family: the scheduler matches stop strings.
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
# HELP smg_servicer_state The servicer's lifecycle state, one-hot: starting, serving, warming_up (readiness held, admitting the warm-up's own requests only), draining (admission closed) or engine_down.
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
# TYPE smg_servicer_time_to_first_token_seconds histogram
smg_servicer_e2e_request_latency_seconds_count{model="served",wire="sglang",rank="0"} 5
smg_servicer_e2e_request_latency_seconds_sum{model="served",wire="sglang",rank="0"} <v>
smg_servicer_generation_tokens_total{model="served",wire="sglang",rank="0"} 8
smg_servicer_inter_token_latency_seconds_count{model="served",wire="sglang",rank="0"} 2
smg_servicer_inter_token_latency_seconds_sum{model="served",wire="sglang",rank="0"} <v>
smg_servicer_metrics_dropped_total{model="served",wire="sglang",family="inter_token_latency"} 0
smg_servicer_metrics_scrape_seconds{model="served",wire="sglang"} <v>
smg_servicer_metrics_scrapes_total{model="served",wire="sglang",code="200"} 0
smg_servicer_metrics_scrapes_total{model="served",wire="sglang",code="500"} 0
smg_servicer_prompt_tokens_cached_total{model="served",wire="sglang",rank="0"} 1
smg_servicer_prompt_tokens_total{model="served",wire="sglang",rank="0"} 15
smg_servicer_request_errors_total{model="served",wire="sglang",code="unavailable"} 1
smg_servicer_request_generation_tokens_count{model="served",wire="sglang",rank="0"} 5
smg_servicer_request_generation_tokens_sum{model="served",wire="sglang",rank="0"} 8
smg_servicer_request_params_max_tokens_count{model="served",wire="sglang",rank="0"} 5
smg_servicer_request_params_max_tokens_sum{model="served",wire="sglang",rank="0"} 40
smg_servicer_request_params_n_count{model="served",wire="sglang",rank="0"} 5
smg_servicer_request_params_n_sum{model="served",wire="sglang",rank="0"} 5
smg_servicer_request_prompt_tokens_count{model="served",wire="sglang",rank="0"} 5
smg_servicer_request_prompt_tokens_sum{model="served",wire="sglang",rank="0"} 15
smg_servicer_request_time_per_output_token_seconds_count{model="served",wire="sglang",rank="0"} 2
smg_servicer_request_time_per_output_token_seconds_sum{model="served",wire="sglang",rank="0"} <v>
smg_servicer_requests_active{model="served",wire="sglang"} 0
smg_servicer_requests_refused_total{model="served",wire="sglang",reason="draining"} 0
smg_servicer_requests_refused_total{model="served",wire="sglang",reason="duplicate_id"} 1
smg_servicer_requests_refused_total{model="served",wire="sglang",reason="invalid_argument"} 0
smg_servicer_requests_refused_total{model="served",wire="sglang",reason="media"} 0
smg_servicer_requests_total{model="served",wire="sglang",rank="0",finish_reason="abort"} 1
smg_servicer_requests_total{model="served",wire="sglang",rank="0",finish_reason="cut"} 1
smg_servicer_requests_total{model="served",wire="sglang",rank="0",finish_reason="error"} 0
smg_servicer_requests_total{model="served",wire="sglang",rank="0",finish_reason="length"} 1
smg_servicer_requests_total{model="served",wire="sglang",rank="0",finish_reason="stop"} 2
smg_servicer_start_time_seconds{model="served",wire="sglang"} <v>
smg_servicer_state{model="served",wire="sglang",state="draining"} 1
smg_servicer_state{model="served",wire="sglang",state="engine_down"} 0
smg_servicer_state{model="served",wire="sglang",state="serving"} 0
smg_servicer_state{model="served",wire="sglang",state="starting"} 0
smg_servicer_state{model="served",wire="sglang",state="warming_up"} 0
smg_servicer_time_to_first_token_seconds_count{model="served",wire="sglang",rank="0"} 5
smg_servicer_time_to_first_token_seconds_sum{model="served",wire="sglang",rank="0"} <v>
"#;

/// One streamed request (three chunks, the scheduler's stop), one
/// non-streamed (the budget), one the caller aborts, one duplicate id
/// refused, one a shutdown cuts: the scrape's families, types, label sets
/// and values are exactly these.
#[tokio::test]
async fn the_scrape_pins_the_families_after_a_scripted_sequence() {
    let mut h = metrics_harness(model_info()).await;
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
        chunk(s1.message().bounded().await.unwrap().unwrap());
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
    chunk(a1.message().bounded().await.unwrap().unwrap());
    h.client
        .abort(sg::AbortRequest {
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
    chunk(d1.message().bounded().await.unwrap().unwrap());
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
    chunk(c1.message().bounded().await.unwrap().unwrap());
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

/// The in-flight gauge reads the registry at scrape: one while a stream is
/// open, zero once the caller dropped it (which counts as the caller's
/// abort).
#[tokio::test]
async fn the_active_gauge_falls_when_the_stream_drops() {
    let mut h = metrics_harness(model_info()).await;
    let stream = h
        .client
        .generate(generate_request("g1", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let body = scrape(&h.server);
    assert_eq!(
        sample(&body, "smg_servicer_requests_active", "wire=\"sglang\""),
        1.0
    );
    drop(stream);
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["g1".to_string()]);
    wait_until(|| h.server.active_requests() == 0).await;
    let body = scrape(&h.server);
    assert_eq!(
        sample(&body, "smg_servicer_requests_active", "wire=\"sglang\""),
        0.0
    );
    assert_eq!(
        sample(
            &body,
            "smg_servicer_requests_total",
            "finish_reason=\"abort\""
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
    let server = SglangServicerServer::start(config).expect("servicer starts");
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
