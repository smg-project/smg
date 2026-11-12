//! A retry re-applies the sampling defaults from the request's baseline, not from the previous
//! worker.

use std::sync::Arc;

use openai_protocol::chat::ChatCompletionRequest;
use smg_grpc_client::sglang_proto;

use super::*;
use crate::{
    routers::grpc::{context::ExecutionPlanKind, proto_wrapper::ProtoRequest},
    worker::{BasicWorkerBuilder, RuntimeType, Worker, WorkerType},
};

fn worker_with_defaults(url: &str, defaults_json: Option<&str>) -> WorkerSelection {
    let mut builder = BasicWorkerBuilder::new(url).worker_type(WorkerType::Regular);
    if let Some(json) = defaults_json {
        builder = builder.label(DEFAULT_SAMPLING_PARAMS_LABEL, json);
    }
    WorkerSelection::Single {
        worker: Arc::new(builder.build()),
    }
}

fn sglang_request() -> ProtoGenerateRequest {
    ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
        request_id: "sampling-restamp".to_string(),
        sampling_params: Some(sglang_proto::SamplingParams {
            temperature: 1.0,
            top_k: 3,
            ..Default::default()
        }),
        ..Default::default()
    }))
}

fn sglang_params(plan: &ExecutionPlan) -> &sglang_proto::SamplingParams {
    match plan {
        ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::Sglang(req))) => {
            req.sampling_params.as_ref().unwrap()
        }
        _ => panic!("expected single SGLang plan"),
    }
}

/// Attempt N's effective sampling params must be exactly (request values)
/// + (worker N's defaults). A retry worker that omits a field the first
/// worker defaulted must fall back to the build-time baseline, never to
/// the first worker's value.
#[test]
fn retry_reapplies_defaults_from_the_baseline_not_the_previous_worker() {
    let mask = SamplingDefaultsMask::from_request_type(&RequestType::Chat(Arc::new(
        ChatCompletionRequest::default(),
    )))
    .expect("chat requests always carry a mask");

    let worker_a = worker_with_defaults(
        "grpc://worker-a:30000",
        Some(r#"{"temperature":0.5,"top_k":7}"#),
    );
    let mut request = sglang_request();
    let baseline = apply_sampling_defaults(&mut request, Some(mask), Some(&worker_a));
    let stamp = AttemptStamp {
        id: IdStamp::Exact,
        sampling_mask: Some(mask),
        sampling_baseline: baseline,
        inject_pd_metadata: false,
        tool_constraint: None,
    };
    let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);
    {
        let params = sglang_params(&plan);
        assert_eq!(params.temperature, 0.5, "worker A's default applied");
        assert_eq!(params.top_k, 7);
    }

    // Worker B omits temperature: it must return to the baseline (1.0),
    // not keep worker A's 0.5.
    let worker_b = worker_with_defaults("grpc://worker-b:30000", Some(r#"{"top_k":9}"#));
    restamp_plan_for_attempt(&mut plan, &stamp, &worker_b, &ToolParserFactory::new()).unwrap();
    {
        let params = sglang_params(&plan);
        assert_eq!(params.temperature, 1.0, "baseline restored, not worker A's");
        assert_eq!(params.top_k, 9, "worker B's default applied");
    }

    // Worker C has no defaults label at all: everything returns to the
    // baseline.
    let worker_c = worker_with_defaults("grpc://worker-c:30000", None);
    restamp_plan_for_attempt(&mut plan, &stamp, &worker_c, &ToolParserFactory::new()).unwrap();
    {
        let params = sglang_params(&plan);
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_k, 3);
    }
}

/// The mask of a chat request that set no sampling field: every default applies.
fn chat_mask() -> SamplingDefaultsMask {
    SamplingDefaultsMask::from_request_type(&RequestType::Chat(Arc::new(
        ChatCompletionRequest::default(),
    )))
    .expect("chat requests always carry a mask")
}

/// A disaggregated pair whose sides carry the given default-sampling labels
/// (`None` leaves a side unlabelled).
fn labelled_pair(prefill_json: Option<&str>, decode_json: Option<&str>) -> WorkerSelection {
    fn side(url: &str, worker_type: WorkerType, json: Option<&str>) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url).worker_type(worker_type);
        if let Some(json) = json {
            builder = builder.label(DEFAULT_SAMPLING_PARAMS_LABEL, json);
        }
        Arc::new(builder.build())
    }
    WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: side("grpc://prefill:30000", WorkerType::Prefill, prefill_json),
        decode: side("grpc://decode:30000", WorkerType::Decode, decode_json),
        runtime_type: RuntimeType::Sglang,
    }
}

fn sglang_params_of(request: &ProtoGenerateRequest) -> &sglang_proto::SamplingParams {
    match request {
        ProtoGenerateRequest::Sglang(req) => req.sampling_params.as_ref().unwrap(),
        _ => panic!("expected an SGLang request"),
    }
}

/// A disaggregated request samples under the decode worker's defaults: that
/// side produces the visible tokens, so its label is the one read.
#[test]
fn a_disaggregated_request_takes_the_decode_workers_defaults() {
    let pair = labelled_pair(
        Some(r#"{"temperature":0.2,"top_k":5}"#),
        Some(r#"{"temperature":0.8,"top_k":11}"#),
    );
    let mut request = sglang_request();
    apply_sampling_defaults(&mut request, Some(chat_mask()), Some(&pair));
    let params = sglang_params_of(&request);
    assert_eq!(params.temperature, 0.8, "the decode worker's default");
    assert_eq!(params.top_k, 11, "the decode worker's default");
}

/// The prefill worker's label is not read: a pair whose decode side carries
/// none leaves the request as it was built.
#[test]
fn a_prefill_workers_label_alone_defaults_nothing() {
    let pair = labelled_pair(Some(r#"{"temperature":0.2,"top_k":5}"#), None);
    let mut request = sglang_request();
    apply_sampling_defaults(&mut request, Some(chat_mask()), Some(&pair));
    let params = sglang_params_of(&request);
    assert_eq!(params.temperature, 1.0, "the request's own value");
    assert_eq!(params.top_k, 3, "the request's own value");
}
