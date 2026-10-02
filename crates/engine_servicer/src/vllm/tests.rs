use std::{collections::BTreeSet, fs, io::Cursor, sync::Arc, time::Duration};

use bytes::Bytes;
use engine_zmq_client::{
    codec::tensor::WireTensor,
    mock_engine::{
        connect_to_frontend, default_ready_response, EngineInbound, MockEngineInput,
        MockEngineOutput,
    },
    protocol::vllm::{
        output::{
            EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
            SpecDecodeMetrics, StopReason,
        },
        pooling::{PoolingOutput, PoolingParams},
        request::EngineCoreRequest,
        stats::SchedulerStats,
        structured_outputs::StructuredOutputBackend,
    },
    EngineId,
};
use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
use portpicker::pick_unused_port;
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

/// Worker-side media processing is refused up front (the adapter would
/// otherwise run the request text-only); Router-preprocessed batches,
/// including extra modality batches, go through to the engine.
#[tokio::test]
async fn media_refs_are_refused_and_extra_batches_translated() {
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
            chunk_tokens(stream.message().await.unwrap().unwrap()),
            vec![expected]
        );
    }
    let done = complete(stream.message().await.unwrap().unwrap());
    assert_eq!(done.finish_reason, "stop");
    assert_eq!(done.output_ids, vec![1, 2, 1, 2]);
    assert_eq!(
        done.matched_stop,
        Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
            "Hello world".to_string()
        ))
    );
    assert!(stream.message().await.unwrap().is_none());
    assert_eq!(recv_abort(&mut h.engine_in).await, vec!["mt1".to_string()]);
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
    let done = complete(stream.message().await.unwrap().unwrap());
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
    for (configured, expected) in [
        ("", StructuredOutputBackend::Xgrammar),
        ("guidance", StructuredOutputBackend::Guidance),
    ] {
        let mut model = model_info();
        model.structured_outputs_backend = configured.to_string();
        let mut h = harness(model, None).await;
        let mut request = generate_request("so1", false, Vec::new());
        request.sampling_params.as_mut().unwrap().constraint =
            Some(vllm::sampling_params::Constraint::JsonObject(true));
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
    let mut config = config(dir.path(), &handshake_address(), model_info());
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
    while let Some(chunk) = stream.message().await.unwrap() {
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
    let server = VllmServicerServer::start(config(dir.path(), &handshake_address(), model_info()))
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
    let mut config = config(dir.path(), &handshake_address(), model_info());
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
/// batches arrive under the publisher's sequence numbers, and dropping the
/// stream closes the subscription on the publisher's side.
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
    assert_eq!(status.message(), kv_events::DISABLED_MESSAGE);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");

    let port = pick_unused_port().expect("a free publisher port");
    let mut publisher = PubSocket::new();
    let mut monitor = publisher.monitor();
    publisher
        .bind(&format!("tcp://127.0.0.1:{port}"))
        .await
        .expect("publisher binds");
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

    drop(stream);
    let disconnected = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(event) = monitor.next().await {
            if matches!(event, SocketEvent::Disconnected(_)) {
                return true;
            }
        }
        false
    })
    .await
    .expect("the publisher notices the dropped stream in time");
    assert!(disconnected);
    h.server.stop(Duration::from_secs(5)).expect("clean stop");
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
