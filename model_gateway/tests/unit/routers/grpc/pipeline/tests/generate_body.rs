//! The generate endpoint's answer body, `GenerateBody`: one object for a single prompt with one
//! response, a list otherwise.

use openai_protocol::generate::{GenerateFinishReason, GenerateFinishType, GenerateMetaInfo};

use super::*;

fn response(weight_version: &str) -> GenerateResponse {
    GenerateResponse {
        text: "hi".to_string(),
        output_ids: vec![1, 2, 3],
        meta_info: GenerateMetaInfo {
            id: "req-1".to_string(),
            finish_reason: GenerateFinishReason::Stop {
                finish_type: GenerateFinishType::Stop,
            },
            prompt_tokens: 3,
            weight_version: weight_version.to_string(),
            input_token_logprobs: None,
            output_token_logprobs: None,
            completion_tokens: 3,
            cached_tokens: 0,
            reasoning_tokens: None,
            e2e_latency: 0.1,
            matched_stop: None,
        },
    }
}

fn to_value(body: GenerateBody) -> serde_json::Value {
    serde_json::to_value(body).expect("GenerateBody always serializes")
}

#[test]
fn single_prompt_with_one_response_is_an_object() {
    let value = to_value(generate_response_body(true, 1, vec![response("v1")]));
    assert!(value.is_object(), "expected an object, got {value}");
    assert_eq!(value["meta_info"]["weight_version"], "v1");
}

#[test]
fn n_greater_than_one_stays_a_list_even_for_a_single_prompt() {
    let value = to_value(generate_response_body(
        true,
        2,
        vec![response("v1"), response("v1")],
    ));
    assert!(value.is_array(), "expected a list, got {value}");
    assert_eq!(value.as_array().map(Vec::len), Some(2));
}

#[test]
fn a_batch_request_stays_a_list() {
    // `single_prompt = false` here stands in for a batch `input_ids`
    // request. That shape is unreachable over gRPC today --
    // `GeneratePreparationStage::resolve_generate_input` rejects
    // `InputIds::Batch` with 400 before any response exists -- but the
    // helper still needs to answer it correctly if that ever changes.
    let value = to_value(generate_response_body(
        false,
        1,
        vec![response("v1"), response("v1")],
    ));
    assert!(value.is_array(), "expected a list, got {value}");
}

#[test]
fn single_prompt_with_more_than_one_response_stays_a_list() {
    // Shouldn't happen in practice (a single prompt dispatches once),
    // but the helper must not silently drop a response if it did.
    let value = to_value(generate_response_body(
        true,
        1,
        vec![response("v1"), response("v2")],
    ));
    assert!(value.is_array(), "expected a list, got {value}");
    assert_eq!(value.as_array().map(Vec::len), Some(2));
}
