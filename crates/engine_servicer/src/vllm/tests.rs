use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Cursor,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use engine_zmq_client::{
    codec::{decode_msgpack, tensor::WireTensor, OpaqueValue},
    mock_engine::{
        connect_to_frontend, default_ready_response, hello, EngineInbound, MockEngineInput,
        MockEngineOutput, MOCK_DEADLINE,
    },
    protocol::vllm::{
        multimodal::MmKwargValue,
        output::{
            EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
            SpecDecodeMetrics, StopReason,
        },
        pooling::{PoolingOutput, PoolingParams},
        request::{EngineCoreRequest, MmFeaturesPayload},
        stats::SchedulerStats,
        structured_outputs::{StructuredOutputBackend, StructuredOutputConstraint},
    },
    EngineId,
};
use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
use smg_grpc_client::{
    common_proto as common,
    tokenizer_bundle::{validate_bundle_sha256, with_extracted_bundle, StreamBundle},
    vllm_proto as vllm,
    vllm_proto::vllm_engine_client::VllmEngineClient,
};
use tonic::{transport::Channel, Code};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};
use zip::{CompressionMethod, ZipArchive};

use super::*;
use crate::{kv_events, testing::Bounded, tokenizer_bundle, ServicerError};

fn model_info() -> VllmModelInfo {
    VllmModelInfo {
        model_path: "org/model".to_string(),
        served_model_name: "served".to_string(),
        tokenizer_path: "org/model".to_string(),
        is_generation: true,
        max_context_length: 4096,
        vocab_size: 1024,
        model_type: "qwen3".to_string(),
        architectures: vec!["Qwen3ForCausalLM".to_string()],
        pad_token_id: 0,
        bos_token_id: 1,
        default_sampling_params_json: "{\"temperature\":0.7}".to_string(),
        data_parallel_size: 1,
        ..Default::default()
    }
}

fn config(dir: &std::path::Path, handshake: &str, model: VllmModelInfo) -> VllmServicerConfig {
    VllmServicerConfig {
        bind_address: "127.0.0.1:0".to_string(),
        ipc_base_url: format!("ipc://{}", dir.join("engine").display()),
        handshake_address: handshake.to_string(),
        engine_count: 1,
        tokenizer_dir: None,
        model,
        media_processor: None,
        engine_startup_timeout: Duration::from_secs(10),
        engine_startup_ceiling: None,
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
    while fs::metadata(path).is_ok() && Instant::now() < deadline {
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

/// A bound servicer, a handshaken mock engine, and a gRPC client.
struct Harness {
    server: VllmServicerServer,
    engine_in: MockEngineInput,
    engine_out: MockEngineOutput,
    client: VllmEngineClient<Channel>,
    _dir: tempfile::TempDir,
}

async fn harness(model: VllmModelInfo, tokenizer: Option<Arc<dyn Tokenizer>>) -> Harness {
    harness_with(model, tokenizer, None).await
}

async fn harness_with(
    model: VllmModelInfo,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    media_processor: Option<Arc<dyn MediaProcessor>>,
) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let mut config = config(dir.path(), &handshake, model);
    config.media_processor = media_processor;
    let server = match tokenizer {
        Some(tokenizer) => VllmServicerServer::start_with_tokenizer(config, tokenizer),
        None => VllmServicerServer::start(config),
    }
    .expect("servicer starts");
    let engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock engine handshake");
    wait_until(|| server.engine_ready()).await;
    let client = VllmEngineClient::new(grpc_channel(server.address()).await);
    let (engine_in, engine_out) = engine.split();
    Harness {
        server,
        engine_in,
        engine_out,
        client,
        _dir: dir,
    }
}

fn batch(
    request_id: &str,
    tokens: Vec<u32>,
    finish: Option<EngineCoreFinishReason>,
    stats: Option<SchedulerStats>,
) -> EngineCoreOutputs {
    EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids: tokens,
            finish_reason: finish,
            ..Default::default()
        }],
        finished_requests: finish.map(|_| BTreeSet::from([request_id.to_string()])),
        scheduler_stats: stats.map(Box::new),
        ..Default::default()
    })
}

fn generate_request(id: &str, stream: bool, stop: Vec<String>) -> vllm::GenerateRequest {
    vllm::GenerateRequest {
        request_id: id.to_string(),
        input: Some(vllm::generate_request::Input::Tokenized(
            vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            },
        )),
        sampling_params: Some(vllm::SamplingParams {
            max_tokens: Some(8),
            stop,
            ..Default::default()
        }),
        stream,
        ..Default::default()
    }
}

async fn recv_add(engine: &mut MockEngineInput) -> Box<EngineCoreRequest> {
    match engine.recv().await.expect("inbound") {
        EngineInbound::Add(request) => request,
        other => panic!("expected Add, got {other:?}"),
    }
}

async fn recv_abort(engine: &mut MockEngineInput) -> Vec<String> {
    match engine.recv().await.expect("inbound") {
        EngineInbound::Abort(ids) => ids,
        other => panic!("expected Abort, got {other:?}"),
    }
}

/// Answer the engine's one-time `get_supported_tasks` utility call, which
/// the first pooling request of a connection makes before its submit.
async fn answer_supported_tasks(
    engine_in: &mut MockEngineInput,
    engine_out: &mut MockEngineOutput,
    tasks: &[&str],
) {
    let call = match engine_in.recv().await.expect("inbound") {
        EngineInbound::Utility(call) => call,
        other => panic!("expected the supported-tasks call, got {other:?}"),
    };
    assert_eq!(call.method, "get_supported_tasks");
    let tasks = tasks.iter().map(|task| OpaqueValue::from(*task)).collect();
    engine_out
        .send_utility_reply(0, call.call_id, Ok(OpaqueValue::Array(tasks)))
        .await
        .expect("reply");
}

/// Nothing reaches the engine within a grace period.
async fn assert_engine_idle(engine_in: &mut MockEngineInput) {
    assert!(
        tokio::time::timeout(Duration::from_millis(200), engine_in.recv())
            .await
            .is_err(),
        "nothing should have reached the engine"
    );
}

fn chunk_tokens(response: vllm::GenerateResponse) -> Vec<u32> {
    match response.response {
        Some(vllm::generate_response::Response::Chunk(chunk)) => chunk.token_ids,
        other => panic!("expected chunk, got {other:?}"),
    }
}

fn complete(response: vllm::GenerateResponse) -> vllm::GenerateComplete {
    match response.response {
        Some(vllm::generate_response::Response::Complete(complete)) => complete,
        other => panic!("expected complete, got {other:?}"),
    }
}

#[test]
fn start_rejects_malformed_config() {
    let dir = tempfile::tempdir().unwrap();
    let good = config(dir.path(), "tcp://127.0.0.1:29999", model_info());
    let bad_ipc = VllmServicerConfig {
        ipc_base_url: "/tmp/not-ipc".to_string(),
        ..good.clone()
    };
    let bad_handshake = VllmServicerConfig {
        handshake_address: "udp://127.0.0.1:1".to_string(),
        ..good.clone()
    };
    let no_engines = VllmServicerConfig {
        engine_count: 0,
        ..good.clone()
    };
    let zero_timeout = VllmServicerConfig {
        engine_startup_timeout: Duration::ZERO,
        ..good.clone()
    };
    let zero_ceiling = VllmServicerConfig {
        engine_startup_ceiling: Some(Duration::ZERO),
        ..good.clone()
    };
    let no_model = VllmServicerConfig {
        model: VllmModelInfo::default(),
        ..good
    };
    for config in [
        bad_ipc,
        bad_handshake,
        no_engines,
        zero_timeout,
        zero_ceiling,
        no_model,
    ] {
        assert!(matches!(
            VllmServicerServer::start(config),
            Err(ServicerError::InvalidConfig(_))
        ));
    }
}

/// The gRPC listener is up before the engine is; health says so until the
/// handshake lands, and flips back to NOT_SERVING on drain.
#[tokio::test]
async fn health_gates_on_the_engine_link_and_the_drain_flag() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = VllmServicerServer::start(config(dir.path(), &handshake, model_info()))
        .expect("servicer starts");
    let address = format!("http://{}", server.address());
    let mut client = VllmEngineClient::connect(address.clone()).await.unwrap();
    let mut health = HealthClient::new(
        Channel::from_shared(address)
            .unwrap()
            .connect()
            .await
            .unwrap(),
    );

    let before = client
        .health_check(vllm::HealthCheckRequest::default())
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
    assert!(!server.engine_ready());

    let _engine = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("mock engine handshake");
    wait_until(|| server.engine_ready()).await;

    let after = client
        .health_check(vllm::HealthCheckRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert!(after.healthy);
    assert_eq!(after.message, "Health");
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
    let unknown = health
        .check(HealthCheckRequest {
            service: "other.Service".to_string(),
        })
        .await
        .unwrap_err();
    assert_eq!(unknown.code(), Code::NotFound);

    server.set_serving(false);
    let draining = client
        .health_check(vllm::HealthCheckRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert!(!draining.healthy);
    assert_eq!(draining.message, "Draining");

    server.stop(Duration::from_secs(5)).expect("clean stop");
    assert!(!server.running());
}

/// Streaming: every engine tick is a delta chunk, the finish tick's delta
/// goes out before the cumulative `Complete`, and the piggybacked scheduler
/// stats feed `GetLoads` while the rank has work (the connector clears a
/// rank's load once its last request finishes).
#[tokio::test]
async fn streams_chunks_then_a_cumulative_complete() {
    let mut h = harness(model_info(), None).await;
    let mut stream = h
        .client
        .generate(generate_request("r1", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();

    let request = recv_add(&mut h.engine_in).await;
    assert_eq!(request.request_id, "r1");
    assert_eq!(request.prompt_token_ids, Some(vec![1, 2, 3]));
    assert_eq!(request.sampling_params.as_ref().unwrap().max_tokens, 8);
    let stats = SchedulerStats {
        num_running_reqs: 1,
        num_waiting_reqs: 2,
        kv_cache_usage: 0.25,
        ..Default::default()
    };
    h.engine_out
        .send_outputs(&batch("r1", vec![10], None, Some(stats)))
        .await
        .unwrap();
    assert_eq!(
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
        vec![10]
    );

    let loads = h
        .client
        .get_loads(vllm::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(loads.dp_rank_count, 1);
    assert_eq!(loads.loads.len(), 1);
    assert_eq!(loads.loads[0].num_running_reqs, 1);
    assert_eq!(loads.loads[0].num_waiting_reqs, 2);
    assert_eq!(loads.loads[0].num_total_reqs, 3);
    assert!((loads.loads[0].token_usage - 0.25).abs() < 1e-9);
    assert_eq!(
        loads.loads[0].max_running_requests,
        i32::try_from(default_ready_response().max_num_seqs).unwrap()
    );

    h.engine_out
        .send_outputs(&batch(
            "r1",
            vec![11],
            Some(EngineCoreFinishReason::Length),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
        vec![11]
    );
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![10, 11]);
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.completion_tokens, 2);
    assert!(stream.message().bounded().await.unwrap().is_none());
}

/// A non-streaming request gets the terminal `Complete` only, as from the
/// Python servicer's `FINAL_ONLY` output kind.
#[tokio::test]
async fn non_streaming_yields_only_the_complete() {
    let mut h = harness(model_info(), None).await;
    let mut stream = h
        .client
        .generate(generate_request("r2", false, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    h.engine_out
        .send_outputs(&batch("r2", vec![10], None, None))
        .await
        .unwrap();
    h.engine_out
        .send_outputs(&batch(
            "r2",
            vec![11],
            Some(EngineCoreFinishReason::Stop),
            None,
        ))
        .await
        .unwrap();
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![10, 11]);
    assert_eq!(done.finish_reason, "stop");
    assert!(stream.message().bounded().await.unwrap().is_none());
}

/// EngineCore cannot match string stops: the servicer strips them from the
/// engine request, backstops EOS from the tokenizer, matches the string on
/// the decoded output, ends the choice with `matched_stop_str`, and aborts
/// the engine-side request.
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
    let sampling = request.sampling_params.as_ref().unwrap();
    // The tokenizer's EOS (999) backstops a config without ids: primary id
    // on `_eos_token_id`, and in the stop set.
    assert_eq!(sampling.eos_token_id, Some(999));
    assert!(sampling.all_stop_token_ids.contains(&999));
    h.engine_out
        .send_outputs(&batch("r3", vec![1], None, None))
        .await
        .unwrap();
    h.engine_out
        .send_outputs(&batch("r3", vec![2], None, None))
        .await
        .unwrap();

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
        Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
            "Hello world".to_string()
        ))
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    // The engine is still generating from its point of view: it gets the
    // abort for the choice the servicer ended.
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r3".to_string()]);
}

/// A stop string that is one token (`world` is mock token 2) also reaches
/// the engine as a stop id, so the engine finishes on the tick the string
/// matches. The choice must end with the engine's own `Complete`, carrying
/// every generated id, not an empty frontend-synthesized one; and the
/// non-streaming shape must deliver exactly that one message.
#[tokio::test]
async fn an_engine_finish_on_the_matching_tick_keeps_the_engine_complete() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let mut h = harness(model_info(), Some(tokenizer)).await;
    for (id, streaming) in [("r3c", true), ("r3d", false)] {
        let mut stream = h
            .client
            .generate(generate_request(id, streaming, vec!["world".to_string()]))
            .await
            .expect("generate")
            .into_inner();
        let request = recv_add(&mut h.engine_in).await;
        // Forwarded to the engine as a stop id (not just a string).
        let sampling = request.sampling_params.as_ref().unwrap();
        assert!(
            sampling.stop_token_ids.contains(&2),
            "{:?}",
            sampling.stop_token_ids
        );
        h.engine_out
            .send_outputs(&batch(id, vec![1], None, None))
            .await
            .unwrap();
        // The engine stops on token 2 itself and reports the stop id.
        let mut finish = batch(id, vec![2], Some(EngineCoreFinishReason::Stop), None);
        if let EngineCoreOutputs::RequestBatch(batch) = &mut finish {
            batch.outputs[0].stop_reason = Some(StopReason::TokenId(2));
        }
        h.engine_out.send_outputs(&finish).await.unwrap();

        if streaming {
            assert_eq!(
                chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
                vec![1]
            );
            assert_eq!(
                chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
                vec![2]
            );
        }
        let done = complete(stream.message().bounded().await.unwrap().unwrap());
        assert_eq!(done.output_ids, vec![1, 2], "streaming={streaming}");
        assert_eq!(done.finish_reason, "stop");
        assert_eq!(done.completion_tokens, 2);
        assert_eq!(
            done.matched_stop,
            Some(vllm::generate_complete::MatchedStop::MatchedTokenId(2))
        );
        assert!(stream.message().bounded().await.unwrap().is_none());
    }
}

/// Without a tokenizer there is nothing to match string stops with, so such
/// a request is refused up front rather than silently generating past them.
#[tokio::test]
async fn string_stops_without_a_tokenizer_are_refused() {
    let mut h = harness(model_info(), None).await;
    let status = h
        .client
        .generate(generate_request(
            "r3b",
            true,
            vec!["Hello world".to_string()],
        ))
        .await
        .expect_err("no tokenizer");
    assert_eq!(status.code(), Code::FailedPrecondition);
}

/// `Abort` ends the response stream and reaches the engine.
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
    h.engine_out
        .send_outputs(&batch("r4", vec![10], None, None))
        .await
        .unwrap();
    assert_eq!(
        chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
        vec![10]
    );

    h.client
        .abort(vllm::AbortRequest {
            request_ids: vec!["r4".to_string()],
        })
        .await
        .expect("abort");
    // Ends as on the Python servicer: a terminal `abort` Complete with the
    // output so far, then the stream closes; the engine side is aborted.
    let aborted = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(aborted.finish_reason, "abort");
    assert_eq!(aborted.output_ids, vec![10]);
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r4".to_string()]);
    // An unknown id is a no-op, not an error (idempotent cleanup).
    h.client
        .abort(vllm::AbortRequest {
            request_ids: vec!["r4".to_string(), "never".to_string()],
        })
        .await
        .expect("abort of unknown ids");
}

/// PD disaggregation rides through as on the Python servicer: the request's
/// connector params reach the engine's sampling params, and the finished
/// output's params come back on the `Complete` (JSON plus the legacy mirror).
#[tokio::test]
async fn kv_transfer_params_pass_through_both_ways() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("pd1", false, Vec::new());
    request.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let engine_request = recv_add(&mut h.engine_in).await;
    let extra = engine_request
        .sampling_params
        .as_ref()
        .and_then(|sp| sp.extra_args.as_ref())
        .expect("extra_args carry the connector params");
    assert_eq!(
        extra.get("kv_transfer_params"),
        Some(&serde_json::json!({"do_remote_decode": true}))
    );

    let mut outputs = batch("pd1", vec![7], Some(EngineCoreFinishReason::Length), None);
    if let EngineCoreOutputs::RequestBatch(batch) = &mut outputs {
        batch.outputs[0].kv_transfer_params = Some(serde_json::json!({
            "do_remote_prefill": true,
            "remote_block_ids": [1, 2],
            "remote_engine_id": "eng-a",
            "remote_host": "10.0.0.1",
            "remote_port": 5600,
        }));
    }
    h.engine_out.send_outputs(&outputs).await.unwrap();
    let finished = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(finished.finish_reason, "length");
    let returned: serde_json::Value = serde_json::from_str(
        finished
            .kv_transfer_params_json
            .as_deref()
            .expect("json params"),
    )
    .unwrap();
    assert_eq!(returned["remote_engine_id"], "eng-a");
    assert_eq!(returned["remote_block_ids"], serde_json::json!([1, 2]));
    let legacy = finished.kv_transfer_params.expect("legacy mirror");
    assert_eq!(legacy.remote_host, "10.0.0.1");
    assert_eq!(legacy.remote_port, 5600);
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The Router's W3C trace context (gRPC metadata) reaches the engine request
/// as `trace_headers`, the field vLLM's own frontend fills for its tracer; a
/// call without the context sets none.
#[tokio::test]
async fn trace_context_metadata_reaches_the_engine_request() {
    const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let mut h = harness(model_info(), None).await;
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
    let engine_request = recv_add(&mut h.engine_in).await;
    assert_eq!(
        engine_request.trace_headers,
        Some(BTreeMap::from([
            ("traceparent".to_string(), TRACEPARENT.to_string()),
            ("tracestate".to_string(), "vendor=a".to_string()),
        ]))
    );
    h.engine_out
        .send_outputs(&batch(
            "tr1",
            vec![7],
            Some(EngineCoreFinishReason::Length),
            None,
        ))
        .await
        .unwrap();
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
    assert_eq!(recv_add(&mut h.engine_in).await.trace_headers, None);
    h.engine_out
        .send_outputs(&batch(
            "tr2",
            vec![7],
            Some(EngineCoreFinishReason::Length),
            None,
        ))
        .await
        .unwrap();
    assert!(stream.message().bounded().await.unwrap().is_some());
    assert!(stream.message().bounded().await.unwrap().is_none());
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A PD decode leg refused before admission (here: string stops with no
/// tokenizer) still tells the engine, so the connector releases the blocks
/// the prefill side pinned: vLLM's immediately aborted one-token request.
#[tokio::test]
async fn a_refused_decode_leg_notifies_the_engine() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("pd2", false, vec!["STOP".to_string()]);
    request.kv_transfer_params_json =
        Some(r#"{"do_remote_prefill":true,"remote_block_ids":[3]}"#.to_string());
    let status = h.client.generate(request).await.expect_err("refused");
    assert_eq!(status.code(), Code::FailedPrecondition);
    let notice = recv_add(&mut h.engine_in).await;
    assert_eq!(notice.request_id, "pd2");
    assert!(notice.abort_immediately);
    assert_eq!(notice.prompt_token_ids.as_deref(), Some(&[0][..]));
    let params = notice.sampling_params.expect("sampling params");
    assert_eq!(params.max_tokens, 1);
    assert_eq!(
        params.extra_args.expect("extra_args")["kv_transfer_params"]["remote_block_ids"],
        serde_json::json!([3])
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A decode leg refused as a duplicate id sends no rejection notice: the id
/// names a request that is still live, and the notice would reuse it.
#[tokio::test]
async fn a_duplicate_decode_leg_sends_no_rejection_notice() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("pd3", true, Vec::new());
    request.kv_transfer_params_json =
        Some(r#"{"do_remote_prefill":true,"remote_block_ids":[1]}"#.to_string());
    let stream = h
        .client
        .generate(request.clone())
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let status = h.client.generate(request).await.expect_err("duplicate id");
    assert_eq!(status.code(), Code::AlreadyExists);
    assert_engine_idle(&mut h.engine_in).await;
    drop(stream);
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["pd3".to_string()]);
}

/// Router-preprocessed media serves every choice of an `n > 1` request: the
/// `/dev/shm` payload is read (and unlinked) once, and each engine request
/// carries the per-item features.
#[tokio::test]
#[cfg(target_os = "linux")]
async fn shm_media_serves_every_choice_of_a_fan_out() {
    let mut h = harness(model_info(), None).await;
    let name = format!(
        "smg-servicer-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    );
    let path = std::path::Path::new("/dev/shm").join(&name);
    let payload: Vec<u8> = (0..8).flat_map(|v| (v as f32).to_le_bytes()).collect();
    fs::write(&path, &payload).expect("write shm file");
    let mut request = generate_request("mm3", true, Vec::new());
    request.input = Some(vllm::generate_request::Input::Tokenized(
        vllm::TokenizedInput {
            original_text: String::new(),
            input_ids: vec![0; 9],
        },
    ));
    request.sampling_params.as_mut().unwrap().n = 2;
    request.mm_inputs = Some(vllm::MultimodalInputs {
        pixel_values: Some(vllm::TensorData {
            shape: vec![2, 4],
            dtype: "float32".to_string(),
            payload: Some(vllm::tensor_data::Payload::Shm(common::ShmHandle {
                name: name.clone(),
                offset: 0,
                nbytes: payload.len() as u64,
                owner_id: "smg:test".to_string(),
            })),
        }),
        mm_placeholders: vec![
            vllm::PlaceholderRange {
                offset: 1,
                length: 3,
            },
            vllm::PlaceholderRange {
                offset: 6,
                length: 3,
            },
        ],
        mm_hashes: vec!["h0".to_string(), "h1".to_string()],
        batched_keys: vec!["pixel_values".to_string()],
        modality: common::Modality::Image as i32,
        ..Default::default()
    });
    let _stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let mut ids = Vec::new();
    for _ in 0..2 {
        let engine_request = recv_add(&mut h.engine_in).await;
        let features = engine_request
            .mm_features
            .as_ref()
            .and_then(MmFeaturesPayload::typed)
            .expect("features on every choice");
        assert_eq!(features.len(), 2);
        assert_eq!(features[1].mm_position.offset, 6);
        ids.push(engine_request.request_id.clone());
    }
    ids.sort();
    assert_eq!(ids, vec!["mm3-0".to_string(), "mm3-1".to_string()]);
    assert!(!path.exists(), "the shm payload is read and unlinked once");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Without a media processor, references are refused up front in the Python
/// servicer's words (the adapter would otherwise run the request text-only);
/// Router-preprocessed batches, including extra modality batches, go
/// through to the engine.
#[tokio::test]
async fn media_refs_are_refused_and_extra_batches_translated() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("mm1", false, Vec::new());
    request.media_refs = Some(media_refs(&["https://example.com/x.png"]));
    let status = h.client.generate(request).await.expect_err("refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("--mm-processor"),
        "{}",
        status.message()
    );
    assert_engine_idle(&mut h.engine_in).await;

    // A tensor-less identity batch (a grid-less PD decode leg) rides the
    // cache salt; an empty extra batch is simply nothing to attach.
    let mut request = generate_request("mm2", false, Vec::new());
    request.mm_inputs = Some(vllm::MultimodalInputs {
        mm_hashes: vec!["h1".to_string(), "h2".to_string()],
        ..Default::default()
    });
    request.extra_mm_inputs = vec![vllm::MultimodalInputs::default()];
    let _stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let engine_request = recv_add(&mut h.engine_in).await;
    assert!(engine_request.mm_features.is_none());
    assert_eq!(engine_request.cache_salt.as_deref(), Some("mm:h1,h2"));
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// vLLM checks string stops only once the output exceeds `min_tokens`, and
/// the text up to that point never matches: the early "Hello world" is
/// ignored, the later one ends the choice.
#[tokio::test]
async fn string_stops_wait_for_min_tokens() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let mut h = harness(model_info(), Some(tokenizer)).await;
    let mut request = generate_request("mt1", true, vec!["Hello world".to_string()]);
    request.sampling_params.as_mut().unwrap().min_tokens = 3;
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    for token in [1, 2, 1, 2] {
        h.engine_out
            .send_outputs(&batch("mt1", vec![token], None, None))
            .await
            .unwrap();
    }
    for expected in [1, 2, 1, 2] {
        assert_eq!(
            chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
            vec![expected]
        );
    }
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![1, 2, 1, 2]);
    assert_eq!(
        done.matched_stop,
        Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
            "Hello world".to_string()
        ))
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["mt1".to_string()]);
}

/// A stop may end just past the `min_tokens` window though it began inside
/// it, as vLLM searches back by the stop's length: the "world world" matched
/// at token 2 is dropped, the one ending at token 3 counts.
#[tokio::test]
async fn string_stops_may_span_the_min_tokens_boundary() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let mut h = harness(model_info(), Some(tokenizer)).await;
    let mut request = generate_request("mt2", true, vec!["world world".to_string()]);
    request.sampling_params.as_mut().unwrap().min_tokens = 2;
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    for _ in 0..3 {
        h.engine_out
            .send_outputs(&batch("mt2", vec![2], None, None))
            .await
            .unwrap();
    }
    for _ in 0..3 {
        assert_eq!(
            chunk_tokens(stream.message().bounded().await.unwrap().unwrap()),
            vec![2]
        );
    }
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![2, 2, 2]);
    assert_eq!(
        done.matched_stop,
        Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
            "world world".to_string()
        ))
    );
    assert!(stream.message().bounded().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["mt2".to_string()]);
}

/// Per-request speculative-decoding counters ride the final output onto the
/// `Complete`, summed as the Python servicer sums them.
#[tokio::test]
async fn spec_decode_counts_reach_the_complete() {
    let mut h = harness(model_info(), None).await;
    let mut stream = h
        .client
        .generate(generate_request("sd1", false, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let mut outputs = batch(
        "sd1",
        vec![5, 6],
        Some(EngineCoreFinishReason::Length),
        None,
    );
    if let EngineCoreOutputs::RequestBatch(batch) = &mut outputs {
        batch.outputs[0].spec_decode_metrics = Some(SpecDecodeMetrics {
            num_spec_tokens: 3,
            histogram: vec![1, 2, 0, 1],
            num_draft_tokens: 9,
            ..Default::default()
        });
    }
    h.engine_out.send_outputs(&outputs).await.unwrap();
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![5, 6]);
    assert_eq!(done.spec_accepted_tokens, 5);
    assert_eq!(done.spec_draft_tokens, 9);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The grammar backend on every structured request is the engine's
/// configured one (`auto` settles on xgrammar as vLLM's frontend does), not
/// a per-constraint guess: the engine pins one backend at first use.
#[tokio::test]
async fn structured_output_backend_follows_the_launcher_config() {
    use vllm::sampling_params::Constraint;

    for (configured, expected) in [
        ("", StructuredOutputBackend::Xgrammar),
        ("guidance", StructuredOutputBackend::Guidance),
    ] {
        let mut model = model_info();
        model.structured_outputs_backend = configured.to_string();
        let mut h = harness(model, None).await;
        let mut request = generate_request("so1", false, Vec::new());
        request.sampling_params.as_mut().unwrap().constraint = Some(Constraint::JsonObject(true));
        let _stream = h
            .client
            .generate(request)
            .await
            .expect("generate")
            .into_inner();
        let engine_request = recv_add(&mut h.engine_in).await;
        let backend = engine_request
            .sampling_params
            .as_ref()
            .and_then(|sp| sp.structured_outputs.as_ref())
            .map(|so| so.backend)
            .expect("structured outputs");
        assert_eq!(backend, expected, "configured backend {configured:?}");
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }

    // `auto` resolves per constraint as vLLM's frontend does: a Lark grammar
    // goes to xgrammar unchanged (vLLM 0.30 parses Lark there), a schema with
    // a feature xgrammar lacks goes to guidance, and a choice is lowered to
    // the grammar xgrammar compiles. The streams stay open so no abort
    // interleaves with the engine's inbound requests.
    let mut h = harness(model_info(), None).await;
    let lark = "start: \"yes\" | \"no\"";
    let mut streams = Vec::new();
    for (id, constraint, expected_backend, expected_constraint) in [
        (
            "so2",
            Constraint::Grammar(lark.to_string()),
            StructuredOutputBackend::Xgrammar,
            StructuredOutputConstraint::Grammar(lark.to_string()),
        ),
        (
            "so3",
            Constraint::JsonSchema(r#"{"type":"integer","multipleOf":3}"#.to_string()),
            StructuredOutputBackend::Guidance,
            StructuredOutputConstraint::Json(
                serde_json::json!({"type": "integer", "multipleOf": 3}),
            ),
        ),
        (
            "so4",
            Constraint::Choice(vllm::ChoiceConstraint {
                choices: vec!["a".to_string(), "b".to_string()],
            }),
            StructuredOutputBackend::Xgrammar,
            StructuredOutputConstraint::Grammar("root ::= \"a\" | \"b\"".to_string()),
        ),
    ] {
        let mut request = generate_request(id, true, Vec::new());
        request.sampling_params.as_mut().unwrap().constraint = Some(constraint);
        streams.push(
            h.client
                .generate(request)
                .await
                .expect("generate")
                .into_inner(),
        );
        let engine_request = recv_add(&mut h.engine_in).await;
        let structured = engine_request
            .sampling_params
            .and_then(|sp| sp.structured_outputs)
            .expect("structured outputs");
        assert_eq!(structured.backend, expected_backend, "{id}");
        assert_eq!(structured.constraint, expected_constraint, "{id}");
    }
    drop(streams);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The request id is the Router's cancellation handle, so a live duplicate
/// is refused; dropping the first stream aborts its engine request.
#[tokio::test]
async fn duplicate_in_flight_request_ids_are_refused() {
    let mut h = harness(model_info(), None).await;
    let stream = h
        .client
        .generate(generate_request("r5", true, Vec::new()))
        .await
        .expect("generate")
        .into_inner();
    recv_add(&mut h.engine_in).await;
    let status = h
        .client
        .generate(generate_request("r5", true, Vec::new()))
        .await
        .expect_err("duplicate id");
    assert_eq!(status.code(), Code::AlreadyExists);
    drop(stream);
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["r5".to_string()]);
}

/// Metadata comes from the launcher's config and the handshake, in the
/// shape the Router's discovery step reads off the Python servicer.
#[tokio::test]
async fn info_rpcs_report_config_and_handshake_facts() {
    let mut model = model_info();
    model.kv_connector = "NixlConnector".to_string();
    model.kv_role = "kv_producer".to_string();
    model.kv_engine_id = "eng-a".to_string();
    model.kv_cache_dtype = "auto".to_string();
    model.model_dtype = "torch.bfloat16".to_string();
    model.shm_namespace_id = "boot:42".to_string();
    let mut h = harness(model, None).await;
    let info = h
        .client
        .get_model_info(vllm::GetModelInfoRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.model_path, "org/model");
    assert_eq!(info.served_model_name, "served");
    assert!(info.is_generation);
    assert_eq!(info.max_context_length, 4096);
    assert_eq!(info.max_req_input_len, 4096);
    assert_eq!(info.vocab_size, 1024);
    assert_eq!(info.model_type, "qwen3");
    assert_eq!(info.architectures, vec!["Qwen3ForCausalLM".to_string()]);
    assert_eq!(info.default_sampling_params_json, "{\"temperature\":0.7}");

    let server = h
        .client
        .get_server_info(vllm::GetServerInfoRequest::default())
        .await
        .unwrap()
        .into_inner();
    let ready = default_ready_response();
    assert_eq!(server.server_type, SERVER_TYPE);
    assert_eq!(server.data_parallel_size, 1);
    assert_eq!(server.block_size, i32::try_from(ready.block_size).unwrap());
    // The launcher's dtype label wins (the spelling PD pairing compares).
    assert_eq!(server.model_dtype, "torch.bfloat16");
    assert_eq!(server.active_requests, 0);
    // PD identity and pairing facts, off the launcher's config.
    assert_eq!(server.kv_connector, "NixlConnector");
    assert_eq!(server.kv_role, "kv_producer");
    assert_eq!(server.kv_engine_id, "eng-a");
    assert_eq!(server.kv_cache_dtype, "auto");
    assert_eq!(server.shm_namespace_id, "boot:42");
    // The running window the Router's PD admission gate bounds dispatch by:
    // the handshake's, for a launcher that reported none.
    assert_eq!(
        server.max_num_seqs,
        i32::try_from(ready.max_num_seqs).unwrap()
    );

    // Before any output batch, loads are zero-filled per rank (the Router
    // reads an empty list as no report), stamped with the engine's version.
    let loads = h
        .client
        .get_loads(vllm::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(loads.dp_rank_count, 1);
    assert_eq!(loads.loads.len(), 1);
    assert_eq!(loads.loads[0].num_running_reqs, 0);
    assert_eq!(
        loads.loads[0].max_running_requests,
        i32::try_from(ready.max_num_seqs).unwrap()
    );
    assert_eq!(loads.version, ready.vllm_version);

    // With no KV-event publisher configured, the relay is the one RPC that
    // still answers UNIMPLEMENTED.
    let status = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(status.code(), Code::Unimplemented);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The launcher's `--max-num-seqs` is the window `GetServerInfo` advertises
/// and `GetLoads` reports, over the handshake's figure.
#[tokio::test]
async fn the_launchers_running_window_wins_over_the_handshakes() {
    let mut model = model_info();
    model.max_num_seqs = 64;
    let mut h = harness(model, None).await;
    let server = h
        .client
        .get_server_info(vllm::GetServerInfoRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(server.max_num_seqs, 64);
    let loads = h
        .client
        .get_loads(vllm::GetLoadsRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(loads.loads[0].max_running_requests, 64);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The relay follows the publisher from the servicer's start, before any
/// gateway subscribes: batches published with nobody listening are in its
/// history, and the first subscription gets them as the engine's whole state.
#[tokio::test]
async fn the_relay_subscribes_to_the_publisher_at_boot_before_any_gateway() {
    use kv_events::golden;
    use zeromq::{prelude::*, PubSocket};

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

/// Every relayed batch carries the servicer's load record: the figures
/// `GetLoads` answers with, so the gateway reads the queue, running set, KV
/// usage and window at every scheduler step.
#[tokio::test]
async fn relayed_batches_carry_the_servicers_load_record() {
    use zeromq::{prelude::*, PubSocket};

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
    let mut stream = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .expect("subscribe")
        .into_inner();
    let batch1 = kv_events::golden::bytes(kv_events::golden::BATCH1);
    let mut first = None;
    for _ in 0..200 {
        publisher
            .send(kv_events::golden::frame(b"kv", 0, &batch1))
            .await
            .expect("publish");
        if let Ok(item) = tokio::time::timeout(Duration::from_millis(50), stream.message()).await {
            first = Some(item.expect("stream open").expect("a batch"));
            break;
        }
    }
    let first = first.expect("the subscription went live");
    let record = first.load.expect("the batch carries the load record");
    let loads = h
        .client
        .get_loads(vllm::GetLoadsRequest::default())
        .await
        .expect("loads")
        .into_inner();
    let rank0 = &loads.loads[0];
    assert_eq!(
        (
            record.running_requests,
            record.waiting_requests,
            record.max_running_requests
        ),
        (
            u32::try_from(rank0.num_running_reqs).unwrap(),
            u32::try_from(rank0.num_waiting_reqs).unwrap(),
            u32::try_from(rank0.max_running_requests).unwrap()
        )
    );
    assert_eq!(
        record.waiting_uncached_tokens,
        Some(u32::try_from(rank0.num_waiting_uncached_tokens).unwrap()),
        "the vLLM servicer estimates the queued token-work"
    );
    assert!((record.token_usage - rank0.token_usage).abs() < f64::EPSILON);
    assert!(record.sample >= 1 && !record.load_only);
    drop(stream);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The relay asks the engine's replay for the batches it missed before its
/// subscription joined: a publisher already at sequence 3 when the servicer
/// starts, whose replay covers 0..=3, leaves the window whole from the
/// publisher's first batch, and the first gateway gets all four from it.
#[tokio::test]
async fn a_publisher_already_counting_when_the_servicer_starts_is_replayed_from_its_start() {
    use kv_events::golden;
    use zeromq::{prelude::*, PubSocket, RouterSocket, ZmqMessage};

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

/// Representative tokenizer files plus ones the Python builder excludes.
/// `tokenizer.json` is incompressible and larger than a chunk, so the bundle
/// streams as several.
fn write_tokenizer_dir(dir: &std::path::Path) -> Vec<(&'static str, Vec<u8>)> {
    let mut seed = 0x9E37_79B9_u32;
    let noise: Vec<u8> = (0..2 * tokenizer_bundle::CHUNK_SIZE)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        })
        .collect();
    let files = vec![
        ("tokenizer.json", noise),
        (
            "tokenizer_config.json",
            br#"{"model_max_length": 4096}"#.to_vec(),
        ),
        (
            "special_tokens_map.json",
            br#"{"eos_token": "<|im_end|>"}"#.to_vec(),
        ),
        ("chat_template.jinja", b"{{ messages }}".to_vec()),
        ("model.safetensors", vec![0; 64]),
        ("README.md", b"excluded".to_vec()),
    ];
    for (name, content) in &files {
        fs::write(dir.join(name), content).unwrap();
    }
    fs::create_dir(dir.join("original")).unwrap();
    fs::write(dir.join("original/tokenizer.model"), b"nested, excluded").unwrap();
    files
}

/// `GetTokenizer` streams the Python builder's selection as a Deflate zip in
/// `CHUNK_SIZE` frames with the fingerprint on the last one, and the Router's
/// own bundle loader (sha256 check, zip validation, extraction) accepts it.
#[tokio::test]
async fn get_tokenizer_streams_a_bundle_the_router_loader_accepts() {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer_dir = dir.path().join("tokenizer");
    fs::create_dir(&tokenizer_dir).unwrap();
    let files = write_tokenizer_dir(&tokenizer_dir);
    let mut config = config(dir.path(), &handshake_address(dir.path()), model_info());
    config.tokenizer_dir = Some(tokenizer_dir.to_string_lossy().into_owned());
    // The bundle comes off the configured directory; no engine is needed.
    let server = VllmServicerServer::start(config).expect("servicer starts");
    let mut client = VllmEngineClient::connect(format!("http://{}", server.address()))
        .await
        .unwrap();

    let mut stream = client
        .get_tokenizer(common::GetTokenizerRequest::default())
        .await
        .expect("get_tokenizer")
        .into_inner();
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.message().bounded().await.unwrap() {
        chunks.push(chunk);
    }
    let (last, full) = chunks.split_last().expect("at least one chunk");
    assert!(!full.is_empty(), "the bundle should span several chunks");
    for chunk in full {
        assert_eq!(chunk.data.len(), tokenizer_bundle::CHUNK_SIZE);
        assert!(chunk.sha256.is_empty());
    }
    assert!(!last.data.is_empty() && last.data.len() <= tokenizer_bundle::CHUNK_SIZE);
    assert_eq!(last.sha256.len(), 64);

    // Reassembled as the Router does, then its own validation and extraction.
    let bundle = StreamBundle {
        sha256: last.sha256.clone(),
        compressed_data: chunks
            .iter()
            .flat_map(|chunk| chunk.data.iter().copied())
            .collect(),
    };
    validate_bundle_sha256(&bundle).expect("fingerprint matches");
    let mut archive = ZipArchive::new(Cursor::new(bundle.compressed_data.as_slice())).unwrap();
    let names: Vec<String> = (0..archive.len())
        .map(|index| {
            let entry = archive.by_index(index).unwrap();
            assert!(entry.is_file());
            assert_eq!(entry.compression(), CompressionMethod::Deflated);
            entry.name().to_string()
        })
        .collect();
    assert_eq!(
        names,
        [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ]
    );
    let extracted = with_extracted_bundle(&bundle, |extracted| {
        Ok(files
            .iter()
            .map(|(name, _)| fs::read(extracted.join(name)).ok())
            .collect::<Vec<_>>())
    })
    .expect("extracts");
    for ((name, content), got) in files.iter().zip(extracted) {
        let expected = names.iter().any(|n| n == name).then(|| content.clone());
        assert_eq!(got, expected, "{name}");
    }
    server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Without a tokenizer directory the RPC is refused as on the Python
/// servicer; a configured directory with nothing to bundle is its
/// `FileNotFoundError`, an internal error.
#[tokio::test]
async fn get_tokenizer_without_a_tokenizer_dir_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let server = VllmServicerServer::start(config(
        dir.path(),
        &handshake_address(dir.path()),
        model_info(),
    ))
    .expect("servicer starts");
    let mut client = VllmEngineClient::connect(format!("http://{}", server.address()))
        .await
        .unwrap();
    let status = client
        .get_tokenizer(common::GetTokenizerRequest::default())
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(status.code(), Code::FailedPrecondition);
    assert_eq!(
        status.message(),
        "Tokenizer path is not configured on this server."
    );
    server.stop(Duration::from_secs(5)).expect("clean stop");

    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing");
    let mut config = config(dir.path(), &handshake_address(dir.path()), model_info());
    config.tokenizer_dir = Some(missing.to_string_lossy().into_owned());
    let server = VllmServicerServer::start(config).expect("servicer starts");
    let mut client = VllmEngineClient::connect(format!("http://{}", server.address()))
        .await
        .unwrap();
    let status = client
        .get_tokenizer(common::GetTokenizerRequest::default())
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(status.code(), Code::Internal);
    assert_eq!(
        status.message(),
        format!("No tokenizer files found in {}", missing.display())
    );
    server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `SubscribeKvEvents` without a publisher is UNIMPLEMENTED with the Python
/// servicer's message. With one configured, the call resolves before any
/// event (headers go out eagerly, as the Python relay's initial metadata),
/// batches arrive under the publisher's sequence numbers, and stopping the
/// servicer closes the subscription on the publisher's side.
#[tokio::test]
async fn subscribe_kv_events_relays_a_publisher_or_is_unimplemented() {
    use zeromq::{prelude::*, PubSocket, SocketEvent};

    let mut h = harness(model_info(), None).await;
    let status = h
        .client
        .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(status.code(), Code::Unimplemented);
    assert_eq!(status.message(), kv_events::VLLM_DISABLED_MESSAGE);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

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
    // A bind wildcard, as vLLM's config spells it; the relay resolves it.
    model.kv_events_endpoint = format!("tcp://*:{port}");
    model.kv_events_topic = "kv".to_string();
    let mut h = harness(model, None).await;
    let mut stream = tokio::time::timeout(
        Duration::from_secs(5),
        h.client
            .subscribe_kv_events(common::SubscribeKvEventsRequest {
                start_sequence_number: 0,
            }),
    )
    .await
    .expect("the call resolves before any event is published")
    .expect("subscribe")
    .into_inner();

    // The subscription reaches the publisher a moment after the connect;
    // probe with sequence 0 until a batch comes through.
    let batch1 = kv_events::golden::bytes(kv_events::golden::BATCH1);
    let mut first = None;
    for _ in 0..200 {
        publisher
            .send(kv_events::golden::frame(b"kv", 0, &batch1))
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
    publisher
        .send(kv_events::golden::frame(
            b"kv",
            1,
            &kv_events::golden::bytes(kv_events::golden::BATCH2),
        ))
        .await
        .expect("publish");
    let second = loop {
        let batch = tokio::time::timeout(Duration::from_secs(5), stream.message())
            .await
            .expect("a batch in time")
            .expect("stream open")
            .expect("a batch");
        if batch.sequence_number != 0 {
            break batch;
        }
    };
    assert_eq!(second.sequence_number, 1);
    assert_eq!(second.dp_rank, Some(1));
    let stored = match &second.events[0].data {
        Some(common::kv_cache_event::Data::Stored(stored)) => stored,
        other => panic!("expected a stored event, got {other:?}"),
    };
    assert_eq!(stored.parent_block_hash, Some(41));
    assert_eq!(stored.blocks[0].block_hash, 42);
    assert_eq!(stored.blocks[0].token_ids, vec![100, 101]);

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

/// `FlushCache`, under the Python servicer's `admin.flush_cache` contract.
mod flush_cache {
    use std::time::Instant;

    use engine_zmq_client::{codec::OpaqueValue, protocol::vllm::request::UtilityCall};

    use super::*;

    const RESET_CALL: &str = "reset_prefix_cache";

    async fn recv_utility(engine: &mut MockEngineInput) -> UtilityCall {
        match engine.recv().await.expect("inbound") {
            EngineInbound::Utility(call) => call,
            other => panic!("expected Utility, got {other:?}"),
        }
    }

    fn request(timeout_s: f32) -> common::FlushCacheRequest {
        common::FlushCacheRequest { timeout_s }
    }

    /// Answer every reset the servicer issues with `answer` until it stops
    /// asking; returns how many it issued.
    async fn answer_resets(
        engine_in: &mut MockEngineInput,
        engine_out: &mut MockEngineOutput,
        answer: bool,
    ) -> usize {
        let mut answered = 0;
        while let Ok(Ok(EngineInbound::Utility(call))) =
            tokio::time::timeout(Duration::from_secs(1), engine_in.recv()).await
        {
            engine_out
                .send_utility_reply(0, call.call_id, Ok(OpaqueValue::from(answer)))
                .await
                .expect("reply");
            answered += 1;
        }
        answered
    }

    /// The RPC is vLLM's `reset_prefix_cache(False, False)` as a utility call;
    /// `true` from the engine is the Python servicer's success response.
    #[tokio::test]
    async fn resets_the_prefix_cache_through_a_utility_call() {
        let mut h = harness(model_info(), None).await;
        let (response, call) = tokio::join!(h.client.flush_cache(request(0.0)), async {
            let call = recv_utility(&mut h.engine_in).await;
            h.engine_out
                .send_utility_reply(0, call.call_id, Ok(OpaqueValue::from(true)))
                .await
                .unwrap();
            call
        });
        assert_eq!(call.method, RESET_CALL);
        assert_eq!(
            call.args,
            vec![OpaqueValue::from(false), OpaqueValue::from(false)]
        );
        assert_eq!(call.client_index, 0);
        let response = response.expect("flush").into_inner();
        assert!(response.success);
        assert_eq!(
            response.message,
            "Local KV prefix cache flushed successfully"
        );
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }

    /// `timeout_s == 0` is one attempt: an engine still holding KV blocks
    /// answers `false`, which is the refusal response (OK status), not a retry.
    #[tokio::test]
    async fn an_immediate_attempt_reports_a_refusal() {
        let mut h = harness(model_info(), None).await;
        let (response, ()) = tokio::join!(h.client.flush_cache(request(0.0)), async {
            let call = recv_utility(&mut h.engine_in).await;
            h.engine_out
                .send_utility_reply(0, call.call_id, Ok(OpaqueValue::from(false)))
                .await
                .unwrap();
        });
        let response = response.expect("a refusal is not an error").into_inner();
        assert!(!response.success);
        assert_eq!(
            response.message,
            "KV prefix cache reset refused; requests may be in flight"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(300), h.engine_in.recv())
                .await
                .is_err(),
            "no retry after an immediate attempt"
        );
    }

    /// A positive `timeout_s` retries every 100 ms while the engine refuses,
    /// then fails as DEADLINE_EXCEEDED; an engine that never answers runs out
    /// the same budget.
    #[tokio::test]
    async fn retries_until_the_deadline_then_times_out() {
        let mut h = harness(model_info(), None).await;
        let started = Instant::now();
        let (status, attempts) = tokio::join!(
            h.client.flush_cache(request(0.45)),
            answer_resets(&mut h.engine_in, &mut h.engine_out, false)
        );
        let status = status.expect_err("deadline");
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert_eq!(
            status.message(),
            "Flush cache timed out; the engine may still complete an issued reset"
        );
        assert!(attempts >= 2, "{attempts}");
        assert!(started.elapsed() >= Duration::from_millis(450));

        let started = Instant::now();
        let status = h
            .client
            .flush_cache(request(0.2))
            .await
            .expect_err("deadline");
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert!(started.elapsed() >= Duration::from_millis(200));
        // The unanswered call did reach the engine.
        assert_eq!(recv_utility(&mut h.engine_in).await.method, RESET_CALL);
    }

    /// An engine-side failure is INTERNAL carrying the engine's message, as
    /// the Python servicer reports an exception from the reset.
    #[tokio::test]
    async fn an_engine_failure_is_internal() {
        let mut h = harness(model_info(), None).await;
        let failure = "Call to reset_prefix_cache method failed: boom";
        let (status, ()) = tokio::join!(h.client.flush_cache(request(0.0)), async {
            let call = recv_utility(&mut h.engine_in).await;
            h.engine_out
                .send_utility_reply(0, call.call_id, Err(failure.to_string()))
                .await
                .unwrap();
        });
        let status = status.expect_err("engine failure");
        assert_eq!(status.code(), Code::Internal);
        assert!(
            status.message().starts_with("Flush cache failed: "),
            "{status:?}"
        );
        assert!(status.message().ends_with(failure), "{status:?}");
    }

    /// A negative or non-finite `timeout_s` is refused before the engine is
    /// asked.
    #[tokio::test]
    async fn rejects_an_invalid_timeout() {
        let mut h = harness(model_info(), None).await;
        for timeout_s in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let status = h
                .client
                .flush_cache(request(timeout_s))
                .await
                .expect_err("invalid timeout_s");
            assert_eq!(status.code(), Code::InvalidArgument, "{timeout_s}");
            assert_eq!(
                status.message(),
                "timeout_s must be finite and non-negative"
            );
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(300), h.engine_in.recv())
                .await
                .is_err()
        );
    }
}

// ---- Embed ----

fn embed_request(id: &str, input_ids: Vec<u32>) -> vllm::EmbedRequest {
    vllm::EmbedRequest {
        request_id: id.to_string(),
        tokenized: Some(vllm::TokenizedInput {
            original_text: String::new(),
            input_ids,
        }),
    }
}

fn pooling_model_info() -> VllmModelInfo {
    VllmModelInfo {
        is_generation: false,
        model_type: "bert".to_string(),
        architectures: vec!["BertModel".to_string()],
        ..model_info()
    }
}

/// One finished pooling output for `request_id`, encoded by the Rust side.
fn pooled_batch(request_id: &str, values: Vec<f32>) -> EngineCoreOutputs {
    EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        outputs: vec![EngineCoreOutput {
            request_id: request_id.to_string(),
            pooling_output: Some(PoolingOutput::new(
                WireTensor::from_f32(vec![values.len()], values).unwrap(),
            )),
            finish_reason: Some(EngineCoreFinishReason::Stop),
            ..Default::default()
        }],
        finished_requests: Some(BTreeSet::from([request_id.to_string()])),
        ..Default::default()
    })
}

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit"))
        .collect()
}

/// vLLM's own `MsgpackEncoder` bytes (vLLM 0.30.1rc1) for the finished
/// pooling output of request `emb-1`: `tensor([0.25, -1.5, 3.0])` inline as a
/// raw-view ext, finish reason STOP, as the engine answers an embedding
/// request.
const VLLM_EMBED_OUTPUT_INLINE: &str = "980091dc0012a5656d622d3190c0c093a7666c6f617433329103c70c030000803e0000c0bf0000404000c0c0c0c0c0c0c000c0c0c0c0c0cb415e4c4313f62b42c091a5656d622d31c0c0";
/// The same for `emb-2` with a 96-float tensor, which exceeds vLLM's 256-byte
/// inline threshold and rides as aux frame 1 (the primary frame carries its
/// index).
const VLLM_EMBED_OUTPUT_AUX_FRAME0: &str = "980091dc0012a5656d622d3290c0c093a7666c6f6174333291600100c0c0c0c0c0c0c000c0c0c0c0c0cb415e4c4313f6c03dc091a5656d622d32c0c0";

/// `Embed` submits a pooling request (no sampling params, the verified
/// `PoolingParams(task="embed")`) and answers from the engine's single
/// finished output, decoded from vLLM's own wire bytes: an inline tensor and
/// an aux-frame one.
#[tokio::test]
async fn embed_answers_from_the_engines_pooling_output() {
    let mut h = harness(pooling_model_info(), None).await;
    let (response, request) = tokio::join!(
        h.client
            .embed(embed_request("emb-1", vec![101, 7592, 2088, 102])),
        async {
            answer_supported_tasks(&mut h.engine_in, &mut h.engine_out, &["embed"]).await;
            let request = recv_add(&mut h.engine_in).await;
            h.engine_out
                .send_frames(vec![Bytes::from(unhex(VLLM_EMBED_OUTPUT_INLINE))])
                .await
                .unwrap();
            request
        }
    );
    assert_eq!(request.request_id, "emb-1");
    assert_eq!(request.prompt_token_ids, Some(vec![101, 7592, 2088, 102]));
    assert!(request.sampling_params.is_none());
    assert_eq!(request.pooling_params, Some(PoolingParams::embed()));
    let response = response.expect("embed").into_inner();
    assert_eq!(response.embedding, vec![0.25, -1.5, 3.0]);
    assert_eq!(response.prompt_tokens, 4);
    assert_eq!(response.embedding_dim, 3);

    let values: Vec<f32> = (0..96).map(|i| i as f32 / 8.0).collect();
    let aux = Bytes::from(
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let (response, ()) = tokio::join!(
        h.client.embed(embed_request("emb-2", vec![101, 102])),
        async {
            recv_add(&mut h.engine_in).await;
            h.engine_out
                .send_frames(vec![Bytes::from(unhex(VLLM_EMBED_OUTPUT_AUX_FRAME0)), aux])
                .await
                .unwrap();
        }
    );
    let response = response.expect("embed").into_inner();
    assert_eq!(response.embedding, values);
    assert_eq!(response.embedding_dim, 96);
    assert_eq!(response.prompt_tokens, 2);
    // The registry entry is released with the answer.
    let info = h
        .client
        .get_server_info(vllm::GetServerInfoRequest::default())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.active_requests, 0);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Pooling needs the prompt's ids: a request without tokenized input (the
/// proto's only input form), with an empty one, or without an id is refused
/// before the engine sees it, as the Python servicer's `ValueError` maps to
/// INVALID_ARGUMENT.
#[tokio::test]
async fn embed_without_tokenized_input_is_invalid() {
    let mut h = harness(pooling_model_info(), None).await;
    for request in [
        vllm::EmbedRequest {
            request_id: "emb-3".to_string(),
            tokenized: None,
        },
        embed_request("emb-4", Vec::new()),
        embed_request("", vec![1, 2]),
    ] {
        let status = h.client.embed(request).await.expect_err("refused");
        assert_eq!(status.code(), Code::InvalidArgument, "{}", status.message());
    }
    // None of them reached the engine: the next Add is the first valid one.
    let (response, request) =
        tokio::join!(h.client.embed(embed_request("emb-5", vec![1])), async {
            answer_supported_tasks(&mut h.engine_in, &mut h.engine_out, &["embed"]).await;
            let request = recv_add(&mut h.engine_in).await;
            h.engine_out
                .send_outputs(&pooled_batch("emb-5", vec![1.0, 2.0]))
                .await
                .unwrap();
            request
        });
    assert_eq!(request.request_id, "emb-5");
    assert_eq!(
        response.expect("embed").into_inner().embedding,
        vec![1.0, 2.0]
    );
}

/// `Abort` ends a waiting `Embed` as it ends a generate stream: the RPC
/// answers ABORTED and the engine-side request is aborted.
#[tokio::test]
async fn abort_rpc_cancels_an_in_flight_embed() {
    let mut h = harness(pooling_model_info(), None).await;
    let mut aborter = h.client.clone();
    let (response, aborted) = tokio::join!(
        h.client.embed(embed_request("emb-6", vec![1, 2, 3])),
        async {
            answer_supported_tasks(&mut h.engine_in, &mut h.engine_out, &["embed"]).await;
            recv_add(&mut h.engine_in).await;
            aborter
                .abort(vllm::AbortRequest {
                    request_ids: vec!["emb-6".to_string()],
                })
                .await
                .expect("abort");
            recv_abort(&mut h.engine_in).await
        }
    );
    assert_eq!(response.expect_err("aborted").code(), Code::Aborted);
    assert_eq!(aborted, vec!["emb-6".to_string()]);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A generation runner serves no pooling task and a pooling runner no
/// generation task: both refusals are vLLM's frontend's, before the engine
/// (whose own check would take it down) sees anything. A pooling runner
/// without the `embed` task is refused with the engine's own task list,
/// fetched once per connection.
#[tokio::test]
async fn embed_and_generate_are_refused_on_the_wrong_runner() {
    let mut h = harness(model_info(), None).await;
    let status = h
        .client
        .embed(embed_request("emb-7", vec![1]))
        .await
        .expect_err("refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(status.message(), "This model does not support pooling");
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    let mut h = harness(pooling_model_info(), None).await;
    let status = h
        .client
        .generate(generate_request("g1", false, Vec::new()))
        .await
        .expect_err("refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(status.message(), "This model does not support generation");
    let (status, ()) = tokio::join!(
        h.client.embed(embed_request("emb-8", vec![1])),
        answer_supported_tasks(&mut h.engine_in, &mut h.engine_out, &["classify", "render"]),
    );
    let status = status.expect_err("refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(
        status.message(),
        "Unsupported task: 'embed' Supported tasks: [\"classify\"]"
    );
    // Cached: the second refusal makes no engine call.
    let status = h
        .client
        .embed(embed_request("emb-9", vec![1]))
        .await
        .expect_err("refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The launcher's pooler config reaches the engine's pooling params the way
/// vLLM's frontend merges it: an activation turned off and a Matryoshka
/// dimension ride every `Embed`.
#[tokio::test]
async fn embed_params_carry_the_pooler_config() {
    let mut model = pooling_model_info();
    model.pooler_use_activation = Some(false);
    model.pooler_dimensions = Some(3);
    let mut h = harness(model, None).await;
    let (response, request) =
        tokio::join!(h.client.embed(embed_request("emb-10", vec![1, 2])), async {
            answer_supported_tasks(&mut h.engine_in, &mut h.engine_out, &["embed"]).await;
            let request = recv_add(&mut h.engine_in).await;
            h.engine_out
                .send_outputs(&pooled_batch("emb-10", vec![1.0, 2.0, 3.0]))
                .await
                .unwrap();
            request
        });
    let params = request.pooling_params.expect("pooling params");
    assert_eq!(params.use_activation, Some(false));
    assert_eq!(params.dimensions, Some(3));
    assert_eq!(
        response.expect("embed").into_inner().embedding,
        vec![1.0, 2.0, 3.0]
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

// ---------------------------------------------------------------------------
// Worker-side media processing (`media_refs`)
// ---------------------------------------------------------------------------

fn media_refs(urls: &[&str]) -> vllm::MediaRefs {
    vllm::MediaRefs {
        items: urls
            .iter()
            .map(|url| vllm::MediaRef {
                modality: common::Modality::Image as i32,
                url: url.to_string(),
            })
            .collect(),
    }
}

/// `mm_features` the way vLLM's encoder writes one image item whose pixel
/// tensor went to aux frame 1.
fn encoded_features() -> Bytes {
    let features = serde_json::json!([{
        "data": {"pixel_values": {
            "data": ["bfloat16", [4], 1],
            "field": ["batched", {"keep_on_cpu": false}],
        }},
        "modality": "image",
        "identifier": "h0",
        "mm_position": {"offset": 1, "length": 3, "is_embed": null},
        "mm_hash": "h0",
    }]);
    Bytes::from(rmp_serde::to_vec_named(&features).unwrap())
}

fn processed_media() -> ProcessedMedia {
    ProcessedMedia {
        prompt_token_ids: vec![7, 8, 9, 10],
        features: MediaFeatures::Encoded {
            mm_features: Some(encoded_features()),
            aux_frames: vec![Bytes::from_static(&[1, 2, 3, 4, 5, 6, 7, 8])],
            cache_salt: Some("salt".to_string()),
        },
        media_identity: None,
    }
}

/// A scripted processor: one outcome for every request, the requests it was
/// handed, an optional delay, and a switchable probe.
struct MockMediaProcessor {
    outcome: Mutex<Result<ProcessedMedia, MediaError>>,
    requests: Mutex<Vec<MediaRequest>>,
    /// When each request's processing began.
    started: Mutex<Vec<Instant>>,
    delay: Duration,
    probe_ok: AtomicBool,
    max_inflight: usize,
}

impl MockMediaProcessor {
    fn answering(outcome: Result<ProcessedMedia, MediaError>) -> Arc<Self> {
        Self::slow(outcome, Duration::ZERO, 4)
    }

    fn slow(
        outcome: Result<ProcessedMedia, MediaError>,
        delay: Duration,
        max_inflight: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            outcome: Mutex::new(outcome),
            requests: Mutex::new(Vec::new()),
            started: Mutex::new(Vec::new()),
            delay,
            probe_ok: AtomicBool::new(true),
            max_inflight,
        })
    }

    fn requests(&self) -> Vec<MediaRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn started(&self) -> Vec<Instant> {
        self.started.lock().unwrap().clone()
    }
}

impl MediaProcessor for MockMediaProcessor {
    fn name(&self) -> &str {
        "mock"
    }

    fn schemes(&self) -> String {
        "http,https,data".to_string()
    }

    fn source(&self) -> &str {
        "flag"
    }

    fn max_inflight(&self) -> usize {
        self.max_inflight
    }

    fn probe(&self) -> BoxFuture<bool> {
        let ok = self.probe_ok.load(Ordering::Acquire);
        Box::pin(async move { ok })
    }

    fn process(&self, request: MediaRequest) -> BoxFuture<Result<ProcessedMedia, MediaError>> {
        self.requests.lock().unwrap().push(request);
        self.started.lock().unwrap().push(Instant::now());
        let outcome = self.outcome.lock().unwrap().clone();
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            outcome
        })
    }
}

/// A processor that answers in the Router's batch shape (smg's own pipeline)
/// has its batches translated for the engine as a Router request's are: one
/// typed feature per item, the pixels in the dtype the pipeline wrote (raw
/// `uint8` here), and the expanded prompt.
#[tokio::test]
async fn media_processed_as_router_batches_is_translated_for_the_engine() {
    let batch = vllm::MultimodalInputs {
        pixel_values: Some(vllm::TensorData {
            shape: vec![1, 4],
            dtype: "uint8".to_string(),
            payload: Some(vllm::tensor_data::Payload::Inline(vec![9, 8, 7, 6])),
        }),
        mm_placeholders: vec![vllm::PlaceholderRange {
            offset: 1,
            length: 2,
        }],
        mm_hashes: vec!["h0".to_string()],
        batched_keys: vec!["pixel_values".to_string()],
        modality: common::Modality::Image as i32,
        ..Default::default()
    };
    let processor = MockMediaProcessor::answering(Ok(ProcessedMedia {
        prompt_token_ids: vec![7, 8, 8, 10],
        features: MediaFeatures::Batches(vec![batch]),
        media_identity: None,
    }));
    let mut h = harness_with(model_info(), None, Some(processor)).await;
    let mut request = media_request("mr22");
    request.stream = false;
    let _stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let engine_request = recv_add(&mut h.engine_in).await;
    assert_eq!(engine_request.prompt_token_ids, Some(vec![7, 8, 8, 10]));
    let Some(MmFeaturesPayload::Typed(features)) = engine_request.mm_features else {
        panic!(
            "expected typed features, got {:?}",
            engine_request.mm_features
        );
    };
    assert_eq!(features.len(), 1);
    assert_eq!(features[0].mm_hash.as_deref(), Some("h0"));
    assert_eq!(features[0].mm_position.offset, 1);
    assert_eq!(features[0].mm_position.length, 2);
    let item = features[0].data.as_ref().expect("item kwargs");
    let tensor = match item["pixel_values"].data.as_ref().expect("pixel data") {
        MmKwargValue::Tensor(tensor) => tensor,
        other => panic!("expected a tensor, got {other:?}"),
    };
    assert_eq!(tensor.dtype, "uint8");
    assert_eq!(tensor.shape, vec![4]);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A request with references is handed to the processor as the Router sent
/// it; what comes back reaches the engine as is: the expanded prompt, the
/// encoder's `mm_features` bytes in the payload, its aux frames after it, and
/// the cache salt.
#[tokio::test]
async fn media_refs_are_processed_worker_side_and_relayed_as_encoded() {
    let processor = MockMediaProcessor::answering(Ok(processed_media()));
    let mut h = harness_with(model_info(), None, Some(processor.clone())).await;
    let mut request = generate_request("mr1", true, Vec::new());
    request.input = Some(vllm::generate_request::Input::Tokenized(
        vllm::TokenizedInput {
            original_text: "describe".to_string(),
            input_ids: vec![1, 2, 3],
        },
    ));
    request.media_refs = Some(media_refs(&["https://example.com/x.png"]));
    let _stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();

    let frames = h.engine_in.recv_frames().await.expect("add frames");
    assert_eq!(frames.len(), 3, "type, payload, one aux frame");
    let engine_request: EngineCoreRequest = decode_msgpack(&frames[1]).expect("add payload");
    assert_eq!(engine_request.request_id, "mr1");
    assert_eq!(engine_request.prompt_token_ids, Some(vec![7, 8, 9, 10]));
    assert_eq!(engine_request.cache_salt.as_deref(), Some("salt"));
    assert!(engine_request.mm_features.is_some());
    let primary = encoded_features();
    assert!(
        frames[1]
            .windows(primary.len())
            .any(|window| window == primary),
        "the encoder's bytes are relayed verbatim"
    );
    assert_eq!(frames[2], Bytes::from_static(&[1, 2, 3, 4, 5, 6, 7, 8]));

    let seen = processor.requests();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].request_id, "mr1");
    assert_eq!(seen[0].prompt_token_ids, vec![1, 2, 3]);
    assert_eq!(seen[0].prompt_text.as_deref(), Some("describe"));
    assert_eq!(
        seen[0].items,
        vec![MediaRefItem {
            modality: "image".to_string(),
            url: "https://example.com/x.png".to_string(),
        }]
    );
    assert!(!seen[0].want_identity);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// The Python servicer's own refusals, before the processor runs: references
/// next to preprocessed batches or without tokenized input, an unsupported
/// modality, an empty reference.
#[tokio::test]
async fn malformed_media_refs_are_refused_before_processing() {
    let processor = MockMediaProcessor::answering(Ok(processed_media()));
    let mut h = harness_with(model_info(), None, Some(processor.clone())).await;
    let mut twice = generate_request("mr2", false, Vec::new());
    twice.media_refs = Some(media_refs(&["https://example.com/x.png"]));
    twice.mm_inputs = Some(vllm::MultimodalInputs {
        mm_hashes: vec!["h1".to_string()],
        ..Default::default()
    });
    let mut text = generate_request("mr3", false, Vec::new());
    text.input = Some(vllm::generate_request::Input::Text("hi".to_string()));
    text.media_refs = Some(media_refs(&["https://example.com/x.png"]));
    let mut audio = generate_request("mr4", false, Vec::new());
    audio.media_refs = Some(vllm::MediaRefs {
        items: vec![vllm::MediaRef {
            modality: common::Modality::Audio as i32,
            url: "https://example.com/x.wav".to_string(),
        }],
    });
    let mut empty = generate_request("mr5", false, Vec::new());
    empty.media_refs = Some(media_refs(&[""]));
    for (request, needle) in [
        (twice, "cannot be combined"),
        (text, "requires tokenized input"),
        (audio, "unsupported modality"),
        (empty, "empty url"),
    ] {
        let status = h.client.generate(request).await.expect_err("refused");
        assert_eq!(status.code(), Code::InvalidArgument, "{}", status.message());
        assert!(status.message().contains(needle), "{}", status.message());
    }
    assert!(processor.requests().is_empty());
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A processor's verdict becomes the Python servicer's status: a caller
/// error INVALID_ARGUMENT, a retryable one UNAVAILABLE (the Router re-selects
/// a worker), a failure INTERNAL; a refused PD decode leg still notifies the
/// engine.
#[tokio::test]
async fn media_processor_errors_map_to_statuses() {
    for (outcome, code) in [
        (
            MediaError::Invalid("media_refs[0]: unsupported image".to_string()),
            Code::InvalidArgument,
        ),
        (
            MediaError::Unavailable("sidecar_timeout: no result".to_string()),
            Code::Unavailable,
        ),
        (MediaError::Internal("boom".to_string()), Code::Internal),
    ] {
        let processor = MockMediaProcessor::answering(Err(outcome.clone()));
        let mut h = harness_with(model_info(), None, Some(processor)).await;
        let mut request = generate_request("mr6", false, Vec::new());
        request.media_refs = Some(media_refs(&["https://example.com/x.png"]));
        request.kv_transfer_params_json =
            Some(r#"{"do_remote_prefill":true,"remote_block_ids":[3]}"#.to_string());
        let status = h.client.generate(request).await.expect_err("refused");
        assert_eq!(status.code(), code, "{outcome:?}");
        let notice = recv_add(&mut h.engine_in).await;
        assert_eq!(notice.request_id, "mr6");
        assert!(notice.abort_immediately);
        h.server.stop(Duration::from_secs(5)).expect("clean stop");
    }
}

/// The in-flight cap as the Python servicer applies it: `max_inflight`
/// requests process at once, as many again wait, the next is shed with a
/// retryable refusal.
#[tokio::test]
async fn media_processing_is_capped_and_sheds_beyond_the_cap() {
    let processor = MockMediaProcessor::slow(Ok(processed_media()), Duration::from_millis(1500), 1);
    let mut h = harness_with(model_info(), None, Some(processor.clone())).await;
    let request = |id: &str| {
        let mut request = generate_request(id, true, Vec::new());
        request.media_refs = Some(media_refs(&["https://example.com/x.png"]));
        request
    };
    let mut first_client = h.client.clone();
    let mut second_client = h.client.clone();
    // Staggered: one in flight, one waiting, and the third is shed at once.
    let (first, second, third) = tokio::join!(
        first_client.generate(request("mr7")),
        async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            second_client.generate(request("mr8")).await
        },
        async {
            tokio::time::sleep(Duration::from_millis(600)).await;
            h.client.generate(request("mr9")).await
        },
    );
    let status = third.expect_err("shed");
    assert_eq!(status.code(), Code::Unavailable);
    assert!(
        status.message().contains("saturated"),
        "{}",
        status.message()
    );
    let _first = first.expect("first admitted");
    let _second = second.expect("second admitted");
    let mut ids: Vec<String> = Vec::new();
    for _ in 0..2 {
        ids.push(recv_add(&mut h.engine_in).await.request_id);
    }
    ids.sort();
    assert_eq!(ids, vec!["mr7".to_string(), "mr8".to_string()]);
    assert_eq!(processor.requests().len(), 2);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A PD prefill leg asks the processor for the media identity and returns
/// it on every `Complete`, so the decode leg is served without pixels.
#[tokio::test]
async fn a_pd_prefill_leg_returns_the_media_identity() {
    let identity = vllm::MediaIdentity {
        prompt_token_ids: vec![7, 8, 9, 10],
        mm_inputs: Some(vllm::MultimodalInputs {
            mm_hashes: vec!["h0".to_string()],
            modality: common::Modality::Image as i32,
            ..Default::default()
        }),
        extra_mm_inputs: Vec::new(),
    };
    let mut processed = processed_media();
    processed.media_identity = Some(identity.clone());
    let processor = MockMediaProcessor::answering(Ok(processed));
    let mut h = harness_with(model_info(), None, Some(processor.clone())).await;
    let mut request = generate_request("mr10", true, Vec::new());
    request.media_refs = Some(media_refs(&["https://example.com/x.png"]));
    request.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
    let mut stream = h
        .client
        .generate(request)
        .await
        .expect("generate")
        .into_inner();
    let engine_request = recv_add(&mut h.engine_in).await;
    assert_eq!(engine_request.prompt_token_ids, Some(vec![7, 8, 9, 10]));
    assert!(processor.requests()[0].want_identity);
    h.engine_out
        .send_outputs(&batch(
            "mr10",
            vec![5],
            Some(EngineCoreFinishReason::Stop),
            None,
        ))
        .await
        .unwrap();
    let _chunk = stream.message().bounded().await.unwrap().unwrap();
    let done = complete(stream.message().bounded().await.unwrap().unwrap());
    assert_eq!(done.media_identity, Some(identity));
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Device-side pixel normalization is a fact of the engine's config,
/// advertised whether or not this worker processes media itself: the Router
/// preprocessing for it needs it to send raw pixels.
#[tokio::test]
async fn server_info_advertises_device_side_normalization() {
    let mut model = model_info();
    model.mm_device_do_normalize = true;
    let mut h = harness_with(model, None, None).await;
    let info = h
        .client
        .get_server_info(vllm::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(info.mm_device_do_normalize);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// `GetServerInfo` advertises the processor only while it answers its probe
/// and the engine takes multimodal input, as the Python servicer does.
#[tokio::test]
async fn server_info_advertises_a_serving_media_processor() {
    let processor = MockMediaProcessor::answering(Ok(processed_media()));
    let mut vision = model_info();
    vision.supports_vision = true;
    let mut h = harness_with(vision.clone(), None, Some(processor.clone())).await;
    let info = h
        .client
        .get_server_info(vllm::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.mm_processor, "mock");
    assert_eq!(info.mm_media_ref_schemes, "http,https,data");
    assert_eq!(info.mm_processor_source, "flag");
    processor.probe_ok.store(false, Ordering::Release);
    let info = h
        .client
        .get_server_info(vllm::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.mm_processor, "");
    assert_eq!(info.mm_media_ref_schemes, "");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    // A text-only engine never advertises one.
    let processor = MockMediaProcessor::answering(Ok(processed_media()));
    let mut h = harness_with(model_info(), None, Some(processor)).await;
    let info = h
        .client
        .get_server_info(vllm::GetServerInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.mm_processor, "");
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

fn media_request(id: &str) -> vllm::GenerateRequest {
    let mut request = generate_request(id, true, Vec::new());
    request.media_refs = Some(media_refs(&["https://example.com/x.png"]));
    request
}

/// A caller that gives up while queued behind the cap leaves no phantom
/// waiter behind: the next request is admitted, not shed.
#[tokio::test]
async fn a_cancelled_wait_frees_its_place_in_the_queue() {
    let processor = MockMediaProcessor::slow(Ok(processed_media()), Duration::from_millis(1500), 1);
    let mut h = harness_with(model_info(), None, Some(processor.clone())).await;
    let mut first_client = h.client.clone();
    let mut second_client = h.client.clone();
    let (first, (), third) = tokio::join!(
        first_client.generate(media_request("mr14")),
        async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            // Queued behind the one slot, then abandoned mid-wait.
            let _ = tokio::time::timeout(
                Duration::from_millis(100),
                second_client.generate(media_request("mr15")),
            )
            .await;
        },
        async {
            tokio::time::sleep(Duration::from_millis(700)).await;
            h.client.generate(media_request("mr16")).await
        },
    );
    let _first = first.expect("first admitted");
    let _third = third.expect("admitted once the cancelled waiter left the queue");
    let mut ids = Vec::new();
    for _ in 0..2 {
        ids.push(recv_add(&mut h.engine_in).await.request_id);
    }
    ids.sort();
    assert_eq!(ids, vec!["mr14".to_string(), "mr16".to_string()]);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A caller that gives up mid-processing does not free the slot early: the
/// processor finishes first, so the cap bounds the work actually in flight.
#[tokio::test]
async fn an_abandoned_request_keeps_its_slot_until_the_processor_finishes() {
    let processor = MockMediaProcessor::slow(Ok(processed_media()), Duration::from_millis(1200), 1);
    let mut h = harness_with(model_info(), None, Some(processor.clone())).await;
    let mut first_client = h.client.clone();
    let ((), second) = tokio::join!(
        async {
            // Abandoned at 200 ms; the processor runs on until 1200 ms.
            let _ = tokio::time::timeout(
                Duration::from_millis(200),
                first_client.generate(media_request("mr17")),
            )
            .await;
        },
        async {
            tokio::time::sleep(Duration::from_millis(400)).await;
            h.client.generate(media_request("mr18")).await
        },
    );
    let _second = second.expect("admitted");
    let started = processor.started();
    assert_eq!(started.len(), 2);
    assert!(
        started[1].duration_since(started[0]) >= Duration::from_millis(1100),
        "the second request waited for the abandoned slot: {:?}",
        started[1].duration_since(started[0])
    );
    // Only the request whose caller stayed reached the engine.
    assert_eq!(recv_add(&mut h.engine_in).await.request_id, "mr18");
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A PD decode leg abandoned while its media is still being fetched sends
/// the rejection notice all the same, as the Python servicer shields it
/// through cancellation.
#[tokio::test]
async fn a_cancelled_decode_leg_still_notifies_the_engine() {
    let processor = MockMediaProcessor::slow(Ok(processed_media()), Duration::from_millis(1500), 4);
    let mut h = harness_with(model_info(), None, Some(processor)).await;
    let mut request = media_request("mr19");
    request.kv_transfer_params_json =
        Some(r#"{"do_remote_prefill":true,"remote_block_ids":[3]}"#.to_string());
    let _ = tokio::time::timeout(Duration::from_millis(200), h.client.generate(request)).await;
    let notice = recv_add(&mut h.engine_in).await;
    assert_eq!(notice.request_id, "mr19");
    assert!(notice.abort_immediately);
    assert_eq!(
        notice
            .sampling_params
            .expect("sampling params")
            .extra_args
            .expect("extra_args")["kv_transfer_params"]["remote_block_ids"],
        serde_json::json!([3])
    );
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// vLLM refuses `n > 1` under greedy sampling on the request as a whole;
/// the fan-out hands each choice `n = 1`, so the check runs before it, and
/// nothing reaches the engine.
#[tokio::test]
async fn greedy_sampling_with_several_choices_is_refused_before_the_fan_out() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("g2", false, Vec::new());
    if let Some(params) = request.sampling_params.as_mut() {
        params.n = 2;
        params.temperature = Some(0.0);
    }
    let status = h.client.generate(request).await.expect_err("refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(
        status.message(),
        "n must be 1 when using greedy sampling, got 2."
    );
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// A PD decode leg refused past its media stage (here: by the engine wire's
/// sampling validation) still owes the notice: the engine never held it.
#[tokio::test]
async fn a_decode_leg_refused_after_its_media_still_notifies_the_engine() {
    let processor = MockMediaProcessor::answering(Ok(processed_media()));
    let mut h = harness_with(model_info(), None, Some(processor)).await;
    let mut request = media_request("mr20");
    request.kv_transfer_params_json =
        Some(r#"{"do_remote_prefill":true,"remote_block_ids":[3]}"#.to_string());
    if let Some(params) = request.sampling_params.as_mut() {
        params.top_p = 2.0;
    }
    let status = h.client.generate(request).await.expect_err("refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    let notice = recv_add(&mut h.engine_in).await;
    assert_eq!(notice.request_id, "mr20");
    assert!(notice.abort_immediately);
    // The notice's own stream is dropped once the add is out (the engine
    // finishes that request by itself), which sends its abort; nothing else
    // follows: one notice, no retry under the refused id.
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["mr20".to_string()]);
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// Once admitted, a decode leg whose caller leaves is aborted, not refused:
/// the engine holds it, and a notice under its id would reuse it.
#[tokio::test]
async fn a_caller_leaving_an_admitted_decode_leg_sends_no_notice() {
    let processor = MockMediaProcessor::answering(Ok(processed_media()));
    let mut h = harness_with(model_info(), None, Some(processor)).await;
    let mut request = media_request("mr21");
    request.kv_transfer_params_json =
        Some(r#"{"do_remote_prefill":true,"remote_block_ids":[3]}"#.to_string());
    let stream = h
        .client
        .generate(request)
        .await
        .expect("admitted")
        .into_inner();
    assert_eq!(recv_add(&mut h.engine_in).await.request_id, "mr21");
    drop(stream);
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["mr21".to_string()]);
    assert_engine_idle(&mut h.engine_in).await;
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}

/// An engine that never dials in fails the link at the configured bound, the
/// server stays up to report it, and `stop` releases the handshake endpoint.
#[tokio::test]
async fn an_engine_that_never_dials_in_fails_the_link_at_the_startup_bound() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let server = VllmServicerServer::start(VllmServicerConfig {
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

/// Between HELLO and READY a real engine loads its model, and nothing crosses
/// the wire meanwhile. The lifecycle owner's reports that the engine process
/// is alive keep the link waiting well past the silence bound, and the start
/// then completes.
#[tokio::test]
async fn a_slow_engine_stays_linked_while_the_launcher_reports_it_alive() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let silence = Duration::from_millis(300);
    let server = VllmServicerServer::start(VllmServicerConfig {
        engine_startup_timeout: silence,
        ..config(dir.path(), &handshake, model_info())
    })
    .unwrap();
    // The engine dials in at once, then "loads" for four silences.
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
        "the link gave up on a living engine"
    );
    assert!(!server.engine_ready());

    let engine = loading
        .ready(default_ready_response())
        .bounded()
        .await
        .expect("READY and registration accepted");
    wait_until(|| server.engine_ready()).await;
    let (mut engine_in, _engine_out) = engine.split();
    assert_engine_idle(&mut engine_in).await;
    server.stop(Duration::from_secs(5)).unwrap();
}

/// An engine that dials in and then falls silent, with nobody reporting it
/// alive, fails the link at the silence bound as before.
#[tokio::test]
async fn an_engine_that_falls_silent_after_hello_fails_the_link_at_the_bound() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let silence = Duration::from_millis(300);
    let server = VllmServicerServer::start(VllmServicerConfig {
        engine_startup_timeout: silence,
        ..config(dir.path(), &handshake, model_info())
    })
    .unwrap();
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

/// The ceiling bounds a start however alive the engine is reported.
#[tokio::test]
async fn the_startup_ceiling_bounds_a_start_the_launcher_keeps_reporting_alive() {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address(dir.path());
    let silence = Duration::from_millis(300);
    let ceiling = Duration::from_millis(900);
    let server = VllmServicerServer::start(VllmServicerConfig {
        engine_startup_timeout: silence,
        engine_startup_ceiling: Some(ceiling),
        ..config(dir.path(), &handshake, model_info())
    })
    .unwrap();
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

/// Many streams on one connection keep their own chunk sequences: every
/// engine step carries one token for each of them, and each stream sees
/// exactly its tokens, in order, then its own `Complete` with the full output
/// and finish reason.
#[tokio::test]
async fn concurrent_streams_keep_their_own_chunk_sequences() {
    const STREAMS: usize = 48;
    const STEPS: u32 = 6;
    fn token(stream: usize, step: u32) -> u32 {
        1000 + u32::try_from(stream).unwrap() * 16 + step
    }
    let mut h = harness(model_info(), None).await;
    let mut streams = Vec::with_capacity(STREAMS);
    for i in 0..STREAMS {
        let mut request = generate_request(&format!("c{i}"), true, Vec::new());
        request.sampling_params.as_mut().unwrap().max_tokens = Some(STEPS);
        streams.push(
            h.client
                .generate(request)
                .await
                .expect("generate")
                .into_inner(),
        );
    }
    let mut ids = BTreeSet::new();
    for _ in 0..STREAMS {
        ids.insert(recv_add(&mut h.engine_in).await.request_id);
    }
    assert_eq!(ids.len(), STREAMS);
    for step in 0..STEPS {
        let last = step + 1 == STEPS;
        let outputs = (0..STREAMS)
            .map(|i| EngineCoreOutput {
                request_id: format!("c{i}"),
                new_token_ids: vec![token(i, step)],
                finish_reason: last.then_some(EngineCoreFinishReason::Length),
                ..Default::default()
            })
            .collect();
        h.engine_out
            .send_outputs(&EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
                engine_index: 0,
                outputs,
                finished_requests: last.then(|| ids.clone()),
                ..Default::default()
            }))
            .await
            .unwrap();
    }
    for (i, stream) in streams.iter_mut().enumerate() {
        let expected: Vec<u32> = (0..STEPS).map(|step| token(i, step)).collect();
        let mut got = Vec::new();
        loop {
            let message = stream
                .message()
                .bounded()
                .await
                .unwrap()
                .expect("a stream ends with its Complete");
            match message.response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    got.extend(chunk.token_ids);
                }
                Some(vllm::generate_response::Response::Complete(done)) => {
                    assert_eq!(done.output_ids, expected, "stream {i}");
                    assert_eq!(done.finish_reason, "length", "stream {i}");
                    assert_eq!(done.completion_tokens, STEPS, "stream {i}");
                    break;
                }
                other => panic!("stream {i}: unexpected response {other:?}"),
            }
        }
        assert_eq!(
            got, expected,
            "stream {i}: the chunks are its tokens, in order"
        );
        assert!(stream.message().bounded().await.unwrap().is_none());
    }
}
