use std::{collections::BTreeSet, sync::Arc, time::Duration};

use engine_zmq_client::{
    mock_engine::{
        connect_to_frontend, default_ready_response, EngineInbound, MockEngineInput,
        MockEngineOutput,
    },
    protocol::vllm::{
        output::{
            EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
            StopReason,
        },
        request::EngineCoreRequest,
        stats::SchedulerStats,
    },
    EngineId,
};
use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
use portpicker::pick_unused_port;
use smg_grpc_client::{
    common_proto as common, vllm_proto as vllm, vllm_proto::vllm_engine_client::VllmEngineClient,
};
use tonic::{transport::Channel, Code};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

use super::*;
use crate::ServicerError;

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
    }
}

fn handshake_address() -> String {
    format!(
        "tcp://127.0.0.1:{}",
        pick_unused_port().expect("a free handshake port")
    )
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

/// A bound servicer, a handshaken mock engine, and a gRPC client.
struct Harness {
    server: VllmServicerServer,
    engine_in: MockEngineInput,
    engine_out: MockEngineOutput,
    client: VllmEngineClient<Channel>,
    _dir: tempfile::TempDir,
}

async fn harness(model: VllmModelInfo, tokenizer: Option<Arc<dyn Tokenizer>>) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let handshake = handshake_address();
    let config = config(dir.path(), &handshake, model);
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
    let client = VllmEngineClient::connect(format!("http://{}", server.address()))
        .await
        .expect("grpc client");
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
        handshake_address: "ipc:///tmp/hs".to_string(),
        ..good.clone()
    };
    let no_engines = VllmServicerConfig {
        engine_count: 0,
        ..good.clone()
    };
    let no_model = VllmServicerConfig {
        model: VllmModelInfo::default(),
        ..good
    };
    for config in [bad_ipc, bad_handshake, no_engines, no_model] {
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
    let handshake = handshake_address();
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
        chunk_tokens(stream.message().await.unwrap().unwrap()),
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
        chunk_tokens(stream.message().await.unwrap().unwrap()),
        vec![11]
    );
    let done = complete(stream.message().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![10, 11]);
    assert_eq!(done.finish_reason, "length");
    assert_eq!(done.completion_tokens, 2);
    assert!(stream.message().await.unwrap().is_none());
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
    let done = complete(stream.message().await.unwrap().unwrap());
    assert_eq!(done.output_ids, vec![10, 11]);
    assert_eq!(done.finish_reason, "stop");
    assert!(stream.message().await.unwrap().is_none());
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
        chunk_tokens(stream.message().await.unwrap().unwrap()),
        vec![1]
    );
    assert_eq!(
        chunk_tokens(stream.message().await.unwrap().unwrap()),
        vec![2]
    );
    let done = complete(stream.message().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![1, 2]);
    assert_eq!(
        done.matched_stop,
        Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
            "Hello world".to_string()
        ))
    );
    assert!(stream.message().await.unwrap().is_none());
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
                chunk_tokens(stream.message().await.unwrap().unwrap()),
                vec![1]
            );
            assert_eq!(
                chunk_tokens(stream.message().await.unwrap().unwrap()),
                vec![2]
            );
        }
        let done = complete(stream.message().await.unwrap().unwrap());
        assert_eq!(done.output_ids, vec![1, 2], "streaming={streaming}");
        assert_eq!(done.finish_reason, "stop");
        assert_eq!(done.completion_tokens, 2);
        assert_eq!(
            done.matched_stop,
            Some(vllm::generate_complete::MatchedStop::MatchedTokenId(2))
        );
        assert!(stream.message().await.unwrap().is_none());
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
        chunk_tokens(stream.message().await.unwrap().unwrap()),
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
    let aborted = complete(stream.message().await.unwrap().unwrap());
    assert_eq!(aborted.finish_reason, "abort");
    assert_eq!(aborted.output_ids, vec![10]);
    assert!(stream.message().await.unwrap().is_none());
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
    let finished = complete(stream.message().await.unwrap().unwrap());
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
    assert!(stream.message().await.unwrap().is_none());
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

/// What the adapter would silently drop is refused up front.
#[tokio::test]
async fn media_refs_and_extra_batches_are_refused_not_dropped() {
    let mut h = harness(model_info(), None).await;
    let mut request = generate_request("mm1", false, Vec::new());
    request.media_refs = Some(vllm::MediaRefs {
        items: vec![vllm::MediaRef {
            modality: common::Modality::Image as i32,
            url: "https://example.com/x.png".to_string(),
        }],
    });
    let status = h.client.generate(request).await.expect_err("refused");
    assert_eq!(status.code(), Code::Unimplemented);
    let mut request = generate_request("mm2", false, Vec::new());
    request.extra_mm_inputs = vec![vllm::MultimodalInputs::default()];
    let status = h.client.generate(request).await.expect_err("refused");
    assert_eq!(status.code(), Code::Unimplemented);
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

    for status in [
        h.client
            .embed(vllm::EmbedRequest::default())
            .await
            .map(|_| ())
            .unwrap_err(),
        h.client
            .flush_cache(common::FlushCacheRequest::default())
            .await
            .map(|_| ())
            .unwrap_err(),
        h.client
            .get_tokenizer(common::GetTokenizerRequest::default())
            .await
            .map(|_| ())
            .unwrap_err(),
        h.client
            .subscribe_kv_events(common::SubscribeKvEventsRequest::default())
            .await
            .map(|_| ())
            .unwrap_err(),
    ] {
        assert_eq!(status.code(), Code::Unimplemented);
    }
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
}
