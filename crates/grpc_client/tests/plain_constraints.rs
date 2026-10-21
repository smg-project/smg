//! A plain generate request names at most one structured constraint (a JSON
//! schema, a regex or an EBNF grammar). Every engine client refuses two with the
//! same words, so the gateway's answer does not depend on the worker's engine.

use openai_protocol::generate::GenerateRequest;
use smg_grpc_client::{
    sglang_scheduler::SglangSchedulerClient, tokenspeed_scheduler::TokenSpeedSchedulerClient,
    trtllm_service::TrtllmServiceClient, vllm_engine::VllmEngineClient,
};

const REFUSAL: &str = "Multiple structured constraints are not allowed";

fn plain(sampling_params: serde_json::Value) -> serde_json::Result<GenerateRequest> {
    serde_json::from_value(serde_json::json!({
        "text": "hi",
        "sampling_params": sampling_params
    }))
}

#[test]
fn two_structured_constraints_on_a_plain_request_are_refused_alike_by_every_engine_client() {
    let two = plain(serde_json::json!({"json_schema": "{\"type\":\"object\"}", "regex": "[a-z]+"}))
        .unwrap();
    let refusals = [
        (
            "vllm_engine",
            VllmEngineClient::build_plain_generate_request("r".into(), &two, None, vec![3]).err(),
        ),
        (
            "sglang_scheduler",
            SglangSchedulerClient::build_plain_generate_request_parts(
                "r".into(),
                &two,
                None,
                vec![3],
            )
            .err(),
        ),
        (
            "tokenspeed_scheduler",
            TokenSpeedSchedulerClient::build_plain_generate_request(
                "r".into(),
                &two,
                None,
                vec![3],
            )
            .err(),
        ),
        (
            "trtllm_service",
            TrtllmServiceClient::build_plain_generate_request_parts(
                "r".into(),
                &two,
                None,
                vec![3],
            )
            .err(),
        ),
    ];
    for (client, refusal) in refusals {
        assert_eq!(refusal.as_deref(), Some(REFUSAL), "{client}");
    }
}

#[test]
fn one_structured_constraint_on_a_plain_request_is_accepted_by_every_engine_client() {
    let one = plain(serde_json::json!({"regex": "[a-z]+"})).unwrap();
    assert!(
        VllmEngineClient::build_plain_generate_request("r".into(), &one, None, vec![3]).is_ok()
    );
    assert!(SglangSchedulerClient::build_plain_generate_request_parts(
        "r".into(),
        &one,
        None,
        vec![3]
    )
    .is_ok());
    assert!(TokenSpeedSchedulerClient::build_plain_generate_request(
        "r".into(),
        &one,
        None,
        vec![3]
    )
    .is_ok());
    assert!(TrtllmServiceClient::build_plain_generate_request_parts(
        "r".into(),
        &one,
        None,
        vec![3]
    )
    .is_ok());
}
