//! How a request's string stops are resolved per wire: forwarded as stop token ids, kept for the
//! router's decoder, or cleared; and the prefill-decode metadata injection, which leaves a request
//! untouched when its wire is not the one its selection names.

use std::sync::Arc;

use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
use smg_grpc_client::{sglang_proto, tokenspeed_proto, vllm_proto};

use super::{resolve_string_stops, ProtoGenerateRequest};

fn mock_tokenizer() -> Arc<dyn Tokenizer> {
    // MockTokenizer vocab: "." => 6, "Hello" => 1, "world" => 2. `encode`
    // splits on whitespace, so "." => [6] (single) and "Hello world" =>
    // [1, 2] (multi); unknown words encode to [].
    Arc::new(MockTokenizer::new())
}

fn sglang_request(stop: Vec<&str>, stop_token_ids: Vec<u32>) -> ProtoGenerateRequest {
    ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
        sampling_params: Some(sglang_proto::SamplingParams {
            stop: stop.into_iter().map(str::to_string).collect(),
            stop_token_ids,
            ..Default::default()
        }),
        ..Default::default()
    }))
}

fn vllm_request(stop: Vec<&str>, stop_token_ids: Vec<u32>) -> ProtoGenerateRequest {
    ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
        sampling_params: Some(vllm_proto::SamplingParams {
            stop: stop.into_iter().map(str::to_string).collect(),
            stop_token_ids,
            ..Default::default()
        }),
        ..Default::default()
    }))
}

fn tokenspeed_request(stop: Vec<&str>, stop_token_ids: Vec<u32>) -> ProtoGenerateRequest {
    ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
        sampling_params: Some(tokenspeed_proto::SamplingParams {
            stop: stop.into_iter().map(str::to_string).collect(),
            stop_token_ids,
            ..Default::default()
        }),
        ..Default::default()
    }))
}

fn tokenspeed_params(req: &ProtoGenerateRequest) -> &tokenspeed_proto::SamplingParams {
    match req {
        ProtoGenerateRequest::TokenSpeed(r) => r.sampling_params.as_ref().unwrap(),
        _ => panic!("expected TokenSpeed request"),
    }
}

fn sglang_params(req: &ProtoGenerateRequest) -> &sglang_proto::SamplingParams {
    match req {
        ProtoGenerateRequest::Sglang(r) => r.sampling_params.as_ref().unwrap(),
        _ => panic!("expected SGLang request"),
    }
}

fn vllm_params(req: &ProtoGenerateRequest) -> &vllm_proto::SamplingParams {
    match req {
        ProtoGenerateRequest::Vllm(r) => r.sampling_params.as_ref().unwrap(),
        _ => panic!("expected vLLM request"),
    }
}

#[test]
fn sglang_single_token_becomes_stop_token_id() {
    let mut req = sglang_request(vec!["."], vec![]);
    resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

    let params = sglang_params(&req);
    assert!(params.stop.is_empty(), "string stop should be cleared");
    assert_eq!(params.stop_token_ids, vec![6]);
}

#[test]
fn sglang_multi_token_relies_on_router_decoder() {
    // "Hello world" => [1, 2]: can't be a flat stop_token_id, so it must not
    // be forwarded (would over-eagerly stop on any subtoken).
    let mut req = sglang_request(vec!["Hello world"], vec![]);
    resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

    let params = sglang_params(&req);
    assert!(params.stop.is_empty());
    assert!(params.stop_token_ids.is_empty());
}

#[test]
fn sglang_mixed_only_single_token_forwarded_and_dedups() {
    let mut req = sglang_request(vec![".", "Hello world"], vec![6, 42]);
    resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

    let params = sglang_params(&req);
    assert!(params.stop.is_empty());
    assert_eq!(
        params.stop_token_ids,
        vec![6, 42],
        "existing ids kept, no dup"
    );
}

#[test]
fn sglang_without_tokenizer_still_clears_strings() {
    let mut req = sglang_request(vec!["."], vec![]);
    resolve_string_stops(&mut req, None, false);

    let params = sglang_params(&req);
    assert!(
        params.stop.is_empty(),
        "strings dropped so worker won't 400"
    );
    assert!(params.stop_token_ids.is_empty());
}

#[test]
fn vllm_resolved_only_over_zmq() {
    // gRPC vLLM keeps its strings (the servicer detokenizes engine-side).
    let mut grpc = vllm_request(vec!["."], vec![]);
    resolve_string_stops(&mut grpc, Some(&mock_tokenizer()), false);
    let params = vllm_params(&grpc);
    assert_eq!(
        params.stop,
        vec![".".to_string()],
        "gRPC vLLM stop preserved"
    );
    assert!(params.stop_token_ids.is_empty());

    // ZMQ vLLM (EngineCore sees token ids only) resolves like SGLang.
    // EOS injection is the ZMQ client's own step, not stop resolution's
    // (see zmq_client::fold_tokenizer_eos_backstop tests).
    let mut zmq = vllm_request(vec!["."], vec![]);
    resolve_string_stops(&mut zmq, Some(&mock_tokenizer()), true);
    let params = vllm_params(&zmq);
    assert!(params.stop.is_empty(), "ZMQ vLLM stop cleared");
    assert_eq!(params.stop_token_ids, vec![6]);
}

#[test]
fn noop_when_no_string_stops() {
    let mut req = sglang_request(vec![], vec![7]);
    resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

    let params = sglang_params(&req);
    assert_eq!(params.stop_token_ids, vec![7], "unrelated ids untouched");
}

#[test]
fn tokenspeed_resolved_only_over_zmq() {
    // A gRPC TokenSpeed request is never produced, but guard the gate: the
    // strings must survive when is_zmq is false.
    let mut grpc = tokenspeed_request(vec!["."], vec![]);
    resolve_string_stops(&mut grpc, Some(&mock_tokenizer()), false);
    let params = tokenspeed_params(&grpc);
    assert_eq!(params.stop, vec![".".to_string()], "non-zmq stop preserved");
    assert!(params.stop_token_ids.is_empty());

    // Over ZMQ the token-only wire drops raw strings, so a single-token stop
    // must ride as a stop_token_ids entry instead.
    let mut zmq = tokenspeed_request(vec!["."], vec![]);
    resolve_string_stops(&mut zmq, Some(&mock_tokenizer()), true);
    let params = tokenspeed_params(&zmq);
    assert!(params.stop.is_empty(), "ZMQ TokenSpeed stop cleared");
    assert_eq!(params.stop_token_ids, vec![6]);
}

#[test]
fn tokenspeed_zmq_does_not_fold_eos() {
    // Unlike vLLM EngineCore, the TokenSpeed scheduler stops at EOS itself,
    // so resolution must not append the tokenizer's EOS ids (999).
    let mut req = tokenspeed_request(vec!["."], vec![]);
    resolve_string_stops(&mut req, Some(&mock_tokenizer()), true);
    assert_eq!(
        tokenspeed_params(&req).stop_token_ids,
        vec![6],
        "only the single-token stop, no EOS fold"
    );
}

#[test]
fn resolution_returns_router_obligations() {
    // Strings stripped for the engine come back as the router's trim duty.
    let mut req = sglang_request(vec![".", "Hello world"], vec![]);
    let obligations = resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);
    assert_eq!(
        obligations,
        vec![".".to_string(), "Hello world".to_string()]
    );

    // gRPC vLLM matches stops server-side: nothing left for the router.
    let mut req = vllm_request(vec!["."], vec![]);
    assert!(resolve_string_stops(&mut req, Some(&mock_tokenizer()), false).is_empty());
}

#[test]
fn pd_bootstrap_injection_skips_non_sglang_requests() {
    use super::{Worker, WorkerSelection};
    use crate::worker::{BasicWorkerBuilder, RuntimeType, WorkerType};

    // An SGLang-runtime worker selection paired with a non-SGLang proto
    // (e.g. a misreporting backend) must skip injection, not panic.
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill:30000")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: worker.clone(),
        decode: worker,
        runtime_type: RuntimeType::Sglang,
    };

    let mut req = vllm_request(vec!["."], vec![7]);
    let before = match &req {
        ProtoGenerateRequest::Vllm(inner) => (**inner).clone(),
        _ => panic!("vllm_request builds a Vllm variant"),
    };
    super::maybe_inject_pd_metadata(&mut req, &selection);
    match &req {
        ProtoGenerateRequest::Vllm(inner) => {
            assert_eq!(**inner, before, "request must be untouched");
        }
        _ => panic!("variant must be unchanged"),
    }
}

#[test]
fn tokenspeed_zmq_multi_token_relies_on_router_decoder() {
    // "Hello world" => [1, 2]: not a flat stop id, so it must not forward.
    let mut req = tokenspeed_request(vec!["Hello world"], vec![42]);
    resolve_string_stops(&mut req, Some(&mock_tokenizer()), true);
    let params = tokenspeed_params(&req);
    assert!(params.stop.is_empty());
    assert_eq!(params.stop_token_ids, vec![42], "existing ids kept");
}
