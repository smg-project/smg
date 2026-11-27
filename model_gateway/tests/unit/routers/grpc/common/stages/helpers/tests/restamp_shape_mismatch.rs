//! A restamp over a plan of the other shape, or a sampling baseline from another backend, is an
//! error, not a silent stamp.

use std::sync::Arc;

use openai_protocol::chat::ChatCompletionRequest;
use smg_grpc_client::sglang_proto;

use super::*;
use crate::routers::{
    error::extract_error_code_from_response,
    grpc::{context::ExecutionPlanKind, proto_wrapper::ProtoRequest},
};

fn single_plan() -> ExecutionPlan {
    ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::Sglang(
        Box::default(),
    )))
}

fn batch_plan() -> ExecutionPlan {
    ExecutionPlan::Batch {
        kind: ExecutionPlanKind::Single,
        shared_request_id: "cmpl_shared".to_string(),
        requests: vec![ProtoGenerateRequest::Sglang(Box::default())],
    }
}

/// A stamp/plan shape mismatch is a build-stage wiring bug; a warn-and-
/// continue would re-dispatch the previous attempt's engine ids.
#[test]
fn id_stamp_shape_mismatch_is_an_error() {
    let batch_stamp = IdStamp::Batch(BatchIdStamp {
        stable_shared: None,
        prefix: "cmpl_",
        unique_subs: false,
    });
    let response = batch_stamp
        .restamp(&mut single_plan())
        .expect_err("batch stamp on a single plan must fail");
    assert_eq!(
        extract_error_code_from_response(&response),
        "id_stamp_plan_mismatch"
    );

    let minted_stamp = IdStamp::Minted { prefix: "cmpl_" };
    let response = minted_stamp
        .restamp(&mut batch_plan())
        .expect_err("single stamp on a batch plan must fail");
    assert_eq!(
        extract_error_code_from_response(&response),
        "id_stamp_plan_mismatch"
    );
}

/// A baseline captured from one backend flavor cannot restore another.
#[test]
fn sampling_baseline_backend_mismatch_is_an_error() {
    let baseline = SamplingBaseline::Vllm(vllm_proto::SamplingParams::default());
    let mut request = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
        sampling_params: Some(sglang_proto::SamplingParams::default()),
        ..Default::default()
    }));
    let mask = SamplingDefaultsMask::from_request_type(&RequestType::Chat(Arc::new(
        ChatCompletionRequest::default(),
    )))
    .expect("chat requests always carry a mask");
    let response = baseline
        .restore_masked(&mut request, mask)
        .expect_err("cross-backend baseline must fail");
    assert_eq!(
        extract_error_code_from_response(&response),
        "sampling_baseline_plan_mismatch"
    );
}
