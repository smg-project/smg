use openai_protocol::{
    chat::ChatCompletionMessage,
    common::{StreamOptions, StringOrArray},
    responses::ReasoningEffort,
};

use super::*;
use crate::routers::grpc::common::responses::utils::{
    assert_responses_api_usage, namespace_test_request, responses_api_test_usage,
};

#[test]
fn chat_to_responses_serializes_responses_api_usage() {
    let chat_response = ChatCompletionResponse::builder("chatcmpl_test", "test-model")
        .choices(vec![ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("done".to_string()),
                refusal: None,
                annotations: Vec::new(),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        }])
        .usage(responses_api_test_usage())
        .build();

    let response = chat_to_responses(
        &chat_response,
        &ResponsesRequest::default(),
        None,
        Some("resp_test".to_string()),
    )
    .expect("chat response should convert");
    let wire = serde_json::to_value(response).expect("response should serialize");
    assert_responses_api_usage(&wire);
}

/// The public API returns these fields on every Response object (the
/// schema marks `error`, `instructions`, `top_p` and both usage detail
/// blocks required); a plain chat completion without reasoning or cache
/// hits must still produce all of them.
#[test]
fn chat_to_responses_carries_the_always_present_fields() {
    let chat_response = ChatCompletionResponse::builder("chatcmpl_plain", "test-model")
        .choices(vec![ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("done".to_string()),
                refusal: None,
                annotations: Vec::new(),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        }])
        .usage(Usage::from_counts(5, 3))
        .build();

    let response = chat_to_responses(
        &chat_response,
        &ResponsesRequest::default(),
        None,
        Some("resp_plain".to_string()),
    )
    .expect("chat response should convert");
    let wire = serde_json::to_value(response).expect("response should serialize");
    let body = wire.as_object().expect("object");

    for key in [
        "error",
        "instructions",
        "incomplete_details",
        "previous_response_id",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "max_tool_calls",
    ] {
        assert_eq!(
            body.get(key),
            Some(&serde_json::Value::Null),
            "{key} must be present and null"
        );
    }
    assert_eq!(body["top_p"], serde_json::json!(1.0));
    assert_eq!(body["truncation"], "disabled");
    assert_eq!(body["service_tier"], "default");
    assert_eq!(body["top_logprobs"], 0);
    assert_eq!(body["background"], false);
    assert_eq!(body["frequency_penalty"], serde_json::json!(0.0));
    assert_eq!(body["presence_penalty"], serde_json::json!(0.0));
    assert_eq!(
        body["reasoning"],
        serde_json::json!({"effort": null, "summary": null})
    );
    assert_eq!(
        body["text"],
        serde_json::json!({"format": {"type": "text"}, "verbosity": "medium"})
    );
    assert!(
        body["completed_at"].is_i64(),
        "completed responses are stamped"
    );
    assert_eq!(
        body["usage"]["input_tokens_details"],
        serde_json::json!({"cached_tokens": 0})
    );
    assert_eq!(
        body["usage"]["output_tokens_details"],
        serde_json::json!({"reasoning_tokens": 0})
    );
}

/// The assistant's reasoning is the turn's first output item, ahead of
/// its message, and carries the opaque encrypted copy only when the
/// request asked for it.
#[test]
fn a_reasoning_turn_leads_the_output_and_carries_its_encrypted_copy_only_when_asked() {
    let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
        "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
        "choices":[{"index":0,"message":{"role":"assistant","content":"done",
            "reasoning_content":"because"},"finish_reason":"stop"}]
    }))
    .unwrap();

    let plain = serde_json::to_value(
        chat_to_responses(&chat, &ResponsesRequest::default(), None, None).unwrap(),
    )
    .unwrap();
    assert_eq!(plain["output"][0]["type"], "reasoning");
    assert!(
        plain["output"][0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("rs_")),
        "{plain}"
    );
    assert_eq!(plain["output"][0]["content"][0]["text"], "because");
    assert!(
        plain["output"][0]["encrypted_content"].is_null(),
        "the encrypted copy comes only when asked for"
    );
    assert_eq!(plain["output"][1]["type"], "message");
    assert_eq!(plain["output"][1]["content"][0]["text"], "done");

    let asking = ResponsesRequest {
        include: Some(vec![IncludeField::ReasoningEncryptedContent]),
        ..Default::default()
    };
    let encrypted = serde_json::to_value(
        chat_to_responses(&chat, &asking, asking.tools.as_deref(), None).unwrap(),
    )
    .unwrap();
    assert_eq!(encrypted["output"][0]["content"][0]["text"], "because");
    assert!(
        encrypted["output"][0]["encrypted_content"].is_string(),
        "the request asked for the encrypted copy"
    );
}

/// A Responses request for `model`, from its input items.
fn items_request(model: &str, input: serde_json::Value) -> ResponsesRequest {
    serde_json::from_value(serde_json::json!({"model": model, "input": input}))
        .expect("a responses request")
}

/// The conversion's refusal of `input` for a model under the DeepSeek
/// contract.
fn refused_under_deepseek(input: serde_json::Value) -> String {
    responses_to_chat(&items_request("deepseek-flash", input))
        .expect_err("a tool result without its call is a malformed history")
        .to_string()
}

/// The call ids of the assistant turn at `index` of a converted history.
fn issued_call_ids(chat: &ChatCompletionRequest, index: usize) -> Vec<&str> {
    match &chat.messages[index] {
        ChatMessage::Assistant {
            tool_calls: Some(calls),
            ..
        } => calls.iter().map(|call| call.id.as_str()).collect(),
        other => panic!("expected an assistant turn with calls, got {other:?}"),
    }
}

/// The call id the tool message at `index` of a converted history answers.
fn answered_call_id(chat: &ChatCompletionRequest, index: usize) -> &str {
    match &chat.messages[index] {
        ChatMessage::Tool { tool_call_id, .. } => tool_call_id,
        other => panic!("expected a tool message, got {other:?}"),
    }
}

/// A tool result under a call id the previous turn did not issue is a
/// malformed history under the DeepSeek contract (as under OpenAI's): the
/// conversion refuses it in the public API's words, naming the id; the
/// right id converts, and a model under no such contract converts either
/// way.
#[test]
fn a_tool_result_under_an_unknown_call_id_does_not_convert_under_the_deepseek_contract() {
    let with_call = |call_id: &str| {
        serde_json::json!([
            {"type": "message", "role": "user", "content": "Add 17 and 25."},
            {"type": "function_call", "call_id": "call_1", "name": "add_integers",
             "arguments": "{\"a\":17,\"b\":25}"},
            {"type": "function_call_output", "call_id": call_id, "output": "42"}
        ])
    };
    assert_eq!(
        refused_under_deepseek(with_call("call_doesnotexist")),
        "No tool call found for function call output with call_id 'call_doesnotexist'"
    );
    assert!(responses_to_chat(&items_request("deepseek-flash", with_call("call_1"))).is_ok());
    assert!(responses_to_chat(&items_request(
        "local-model",
        with_call("call_doesnotexist")
    ))
    .is_ok());
}

/// A result with no call before it answers nothing: refused in the same
/// words after the user's message and as the only item. The stored turns
/// a request continues arrive with their calls, and the loader drops a
/// stored result whose call did not survive the reload, so such a result
/// is the client's, not a reload that lost the call; a model under no
/// such contract converts it either way.
#[test]
fn an_orphan_tool_result_does_not_convert_under_the_deepseek_contract() {
    let after_user = serde_json::json!([
        {"type": "message", "role": "user", "content": "Add 17 and 25."},
        {"type": "function_call_output", "call_id": "call_a", "output": "42"}
    ]);
    let alone = serde_json::json!([
        {"type": "function_call_output", "call_id": "call_a", "output": "42"}
    ]);
    for input in [after_user, alone] {
        assert_eq!(
            refused_under_deepseek(input.clone()),
            "No tool call found for function call output with call_id 'call_a'"
        );
        assert!(responses_to_chat(&items_request("local-model", input)).is_ok());
    }
}

/// A result ahead of the call it names, or ahead of a later call and its
/// result, has no call before it either.
#[test]
fn a_tool_result_ahead_of_its_call_does_not_convert_under_the_deepseek_contract() {
    let before_its_call = serde_json::json!([
        {"type": "function_call_output", "call_id": "call_a", "output": "42"},
        {"type": "function_call", "call_id": "call_a", "name": "add_integers",
         "arguments": "{\"a\":17,\"b\":25}"}
    ]);
    let before_a_later_call = serde_json::json!([
        {"type": "function_call_output", "call_id": "call_a", "output": "42"},
        {"type": "function_call", "call_id": "call_m", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_m", "output": "ok"}
    ]);
    for input in [before_its_call, before_a_later_call] {
        assert_eq!(
            refused_under_deepseek(input),
            "No tool call found for function call output with call_id 'call_a'"
        );
    }
}

/// The call and the result that answers it convert to the assistant's
/// call and the tool message under its id.
#[test]
fn a_matching_call_and_result_convert_under_the_deepseek_contract() {
    let chat = responses_to_chat(&items_request(
        "deepseek-flash",
        serde_json::json!([
            {"type": "message", "role": "user", "content": "Add 17 and 25."},
            {"type": "function_call", "call_id": "call_a", "name": "add_integers",
             "arguments": "{\"a\":17,\"b\":25}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "42"}
        ]),
    ))
    .expect("a paired result converts");
    assert_eq!(chat.messages.len(), 3);
    assert_eq!(issued_call_ids(&chat, 1), ["call_a"]);
    assert_eq!(answered_call_id(&chat, 2), "call_a");
}

/// Two calls in one turn and a result for one of them: the result answers
/// a call the turn issued, so it converts, both calls in the one assistant
/// turn (the call still without a result is the model's to answer, not
/// this rule's subject); a result for neither call is refused.
#[test]
fn a_result_for_one_of_two_calls_converts_under_the_deepseek_contract() {
    let two_calls = |call_id: &str| {
        serde_json::json!([
            {"type": "message", "role": "user", "content": "Add 17 and 25, then 3 and 4."},
            {"type": "function_call", "call_id": "call_a", "name": "add_integers",
             "arguments": "{\"a\":17,\"b\":25}"},
            {"type": "function_call", "call_id": "call_b", "name": "add_integers",
             "arguments": "{\"a\":3,\"b\":4}"},
            {"type": "function_call_output", "call_id": call_id, "output": "7"}
        ])
    };
    let chat = responses_to_chat(&items_request("deepseek-flash", two_calls("call_b")))
        .expect("a result for one of the turn's calls converts");
    assert_eq!(chat.messages.len(), 3);
    assert_eq!(issued_call_ids(&chat, 1), ["call_a", "call_b"]);
    assert_eq!(answered_call_id(&chat, 2), "call_b");
    assert_eq!(
        refused_under_deepseek(two_calls("call_c")),
        "No tool call found for function call output with call_id 'call_c'"
    );
}

#[test]
fn test_text_input_conversion() {
    let req = ResponsesRequest {
        input: ResponseInput::Text("Hello, world!".to_string()),
        instructions: Some("You are a helpful assistant.".to_string()),
        model: "gpt-4".to_string(),
        temperature: Some(0.7),
        ..Default::default()
    };

    let chat_req = responses_to_chat(&req).unwrap();
    assert_eq!(chat_req.messages.len(), 2); // system + user
    assert_eq!(chat_req.model, "gpt-4");
    assert_eq!(chat_req.temperature, Some(0.7));
}

/// The stop sequences a Responses request sets are the chat request's, in
/// the string form and in the list form: the engine's stop list and the
/// gateway's own trim of a matched sequence both read them there.
#[test]
fn a_responses_request_s_stop_sequences_reach_the_chat_request_in_both_forms() {
    let as_string = ResponsesRequest {
        input: ResponseInput::Text("Count from 1 to 10.".to_string()),
        model: "local-model".to_string(),
        stop: Some(StringOrArray::String(",".to_string())),
        ..Default::default()
    };
    let chat = responses_to_chat(&as_string).unwrap();
    assert_eq!(
        chat.stop,
        Some(StringOrArray::String(",".to_string())),
        "a stop string is the chat request's stop"
    );

    let as_list = ResponsesRequest {
        stop: Some(StringOrArray::Array(vec![
            ",".to_string(),
            "10".to_string(),
        ])),
        ..as_string
    };
    let chat = responses_to_chat(&as_list).unwrap();
    assert_eq!(
        chat.stop,
        Some(StringOrArray::Array(vec![
            ",".to_string(),
            "10".to_string()
        ])),
        "a stop list is the chat request's stop"
    );
}

#[test]
fn test_reasoning_effort_flows_through() {
    use openai_protocol::responses::ResponseReasoningParam;

    let req = ResponsesRequest {
        input: ResponseInput::Text("hi".to_string()),
        reasoning: Some(ResponseReasoningParam {
            effort: Some(ReasoningEffort::High),
            summary: None,
            mode: None,
            context: None,
            generate_summary: None,
        }),
        ..Default::default()
    };

    let chat_req = responses_to_chat(&req).unwrap();
    assert_eq!(chat_req.reasoning_effort.as_deref(), Some("high"));
}

/// The outer OpenAI tiers reach the Chat pipeline verbatim, where `none`
/// already means thinking off.
#[test]
fn test_reasoning_effort_outer_tiers_flow_through() {
    use openai_protocol::responses::ResponseReasoningParam;

    for effort in [
        ReasoningEffort::None,
        ReasoningEffort::Xhigh,
        ReasoningEffort::Max,
    ] {
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            reasoning: Some(ResponseReasoningParam {
                effort: Some(effort),
                summary: None,
                mode: None,
                context: None,
                generate_summary: None,
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.reasoning_effort.as_deref(), Some(effort.as_str()));
    }
}

#[test]
fn test_reasoning_effort_absent_when_reasoning_none() {
    let req = ResponsesRequest {
        input: ResponseInput::Text("hi".to_string()),
        ..Default::default()
    };

    let chat_req = responses_to_chat(&req).unwrap();
    assert_eq!(chat_req.reasoning_effort, None);
}

/// The five sampling knobs a Responses request sets are the chat
/// request's. Three of them are plain values with serde defaults on the
/// Responses request (`top_k` -1, `min_p` 0, `repetition_penalty` 1) and
/// cross only when set: an unset knob stays `None`, so the model's own
/// defaults still fill it.
#[test]
fn a_responses_request_s_sampling_knobs_reach_the_chat_request_only_when_set() {
    let unset = ResponsesRequest {
        input: ResponseInput::Text("What is 2+2?".to_string()),
        model: "local-model".to_string(),
        ..Default::default()
    };
    let chat = responses_to_chat(&unset).unwrap();
    assert_eq!(chat.frequency_penalty, None, "no frequency_penalty was set");
    assert_eq!(chat.presence_penalty, None, "no presence_penalty was set");
    assert_eq!(
        chat.top_k, None,
        "the serde default -1 is not a top_k the client set"
    );
    assert_eq!(
        chat.min_p, None,
        "the serde default 0 is not a min_p the client set"
    );
    assert_eq!(
        chat.repetition_penalty, None,
        "the serde default 1 is not a repetition_penalty the client set"
    );

    let set = ResponsesRequest {
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.3),
        top_k: 40,
        min_p: 0.05,
        repetition_penalty: 1.1,
        ..unset
    };
    let chat = responses_to_chat(&set).unwrap();
    assert_eq!(
        chat.frequency_penalty,
        Some(0.5),
        "frequency_penalty crosses as set"
    );
    assert_eq!(
        chat.presence_penalty,
        Some(0.3),
        "presence_penalty crosses as set"
    );
    assert_eq!(chat.top_k, Some(40), "top_k crosses as set");
    assert_eq!(chat.min_p, Some(0.05), "min_p crosses as set");
    assert_eq!(
        chat.repetition_penalty,
        Some(1.1),
        "repetition_penalty crosses as set"
    );
}

#[test]
fn test_items_input_conversion() {
    let req = ResponsesRequest {
        input: ResponseInput::Items(vec![
            ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: vec![ResponseContentPart::InputText {
                    text: "Hello!".to_string(),
                }],
                status: None,
                phase: None,
            },
            ResponseInputOutputItem::Message {
                id: "msg_2".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: "Hi there!".to_string(),
                    annotations: vec![],
                    logprobs: None,
                }],
                status: None,
                phase: None,
            },
        ]),
        ..Default::default()
    };

    let chat_req = responses_to_chat(&req).unwrap();
    assert_eq!(chat_req.messages.len(), 2); // user + assistant
}

#[test]
fn test_function_call_history_uses_call_id_for_chat_tool_messages() {
    let req = ResponsesRequest {
        input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionToolCall {
            id: Some("fc_item_id".to_string()),
            call_id: "call_tool_id".to_string(),
            caller: None,
            name: "lookup".to_string(),
            namespace: Some("weather".to_string()),
            arguments: "{\"q\":\"rust\"}".to_string(),
            output: Some("done".to_string()),
            status: Some("completed".to_string()),
        }]),
        ..Default::default()
    };

    let chat_req = responses_to_chat(&req).unwrap();
    assert_eq!(chat_req.messages.len(), 2);

    match &chat_req.messages[0] {
        ChatMessage::Assistant {
            tool_calls: Some(tool_calls),
            ..
        } => {
            assert_eq!(tool_calls[0].id, "call_tool_id");
            assert_eq!(tool_calls[0].function.name, "weather.lookup");
        }
        other => panic!("expected assistant tool call, got {other:?}"),
    }

    match &chat_req.messages[1] {
        ChatMessage::Tool { tool_call_id, .. } => {
            assert_eq!(tool_call_id, "call_tool_id");
        }
        other => panic!("expected tool message, got {other:?}"),
    }
}

/// A hosted tool call of a stored turn replays as the call and its result;
/// the tools listing stored next to it says nothing to the model.
#[test]
fn stored_hosted_tool_calls_replay_as_a_call_and_its_result() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "input": [
            {"role": "user", "content": "search it"},
            {"type": "mcp_list_tools", "id": "mcpl_1", "server_label": "wiki", "tools": []},
            {"type": "mcp_call", "id": "mcp_1", "name": "search", "server_label": "wiki",
             "arguments": "{\"q\":\"rust\"}", "output": "found", "status": "completed"},
            {"type": "message", "id": "msg_a", "role": "assistant",
             "content": [{"type": "output_text", "text": "Found it.", "annotations": []}]}
        ]
    }))
    .unwrap();

    let chat_req = responses_to_chat(&req).unwrap();

    assert_eq!(chat_req.messages.len(), 4, "{:?}", chat_req.messages);
    match &chat_req.messages[1] {
        ChatMessage::Assistant {
            tool_calls: Some(tool_calls),
            ..
        } => {
            assert_eq!(tool_calls[0].id, "mcp_1");
            assert_eq!(tool_calls[0].function.name, "search");
            assert_eq!(
                tool_calls[0].function.arguments.as_deref(),
                Some("{\"q\":\"rust\"}")
            );
        }
        other => panic!("expected the assistant's call, got {other:?}"),
    }
    match &chat_req.messages[2] {
        ChatMessage::Tool {
            tool_call_id,
            content,
        } => {
            assert_eq!(tool_call_id, "mcp_1");
            assert!(matches!(content, MessageContent::Text(text) if text == "found"));
        }
        other => panic!("expected the call's result, got {other:?}"),
    }
}

/// A hosted tool call that failed stores an empty output and the failure
/// under `error`: it replays as the call and a result that says the call
/// failed, not as a successful empty result.
#[test]
fn a_failed_stored_hosted_tool_call_replays_its_error_as_the_result() {
    for (error, expected) in [
        (
            serde_json::json!({"type": "mcp_tool_execution_error", "content": "wiki is down"}),
            "The tool call failed: wiki is down",
        ),
        (
            serde_json::json!({"type": "http_error", "code": 503, "message": "unavailable"}),
            "The tool call failed (HTTP 503): unavailable",
        ),
        (
            serde_json::json!("legacy text"),
            "The tool call failed: legacy text",
        ),
    ] {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "input": [
                {"role": "user", "content": "search it"},
                {"type": "mcp_call", "id": "mcp_1", "name": "search", "server_label": "wiki",
                 "arguments": "{}", "output": "", "status": "failed", "error": error}
            ]
        }))
        .unwrap();
        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 3, "{:?}", chat_req.messages);
        match &chat_req.messages[2] {
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "mcp_1");
                assert!(
                    matches!(content, MessageContent::Text(text) if text == expected),
                    "{content:?}"
                );
            }
            other => panic!("expected the failure as the result, got {other:?}"),
        }
    }
    // A failed call with no error text still says it failed.
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "input": [
            {"role": "user", "content": "search it"},
            {"type": "mcp_call", "id": "mcp_1", "name": "search", "server_label": "wiki",
             "arguments": "{}", "status": "failed"}
        ]
    }))
    .unwrap();
    let chat_req = responses_to_chat(&req).unwrap();
    assert!(
        matches!(&chat_req.messages[2], ChatMessage::Tool { content: MessageContent::Text(text), .. } if text == "The tool call failed."),
        "{:?}",
        chat_req.messages
    );
}

/// A Responses request's `json_schema` text format keeps its description in
/// the Chat request the gRPC path builds from it.
#[test]
fn a_json_schema_text_format_keeps_its_description() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "input": "Name a city.",
        "text": {"format": {
            "type": "json_schema", "name": "city",
            "description": "A city and its country.",
            "schema": {"type": "object"}, "strict": true
        }}
    }))
    .unwrap();
    let chat_req = responses_to_chat(&req).unwrap();
    let Some(ResponseFormat::JsonSchema { json_schema }) = &chat_req.response_format else {
        panic!(
            "expected a json_schema format, got {:?}",
            chat_req.response_format
        );
    };
    assert_eq!(
        json_schema.description.as_deref(),
        Some("A city and its country.")
    );
    assert_eq!(json_schema.name, "city");
}

#[test]
fn test_empty_input_error() {
    let req = ResponsesRequest {
        input: ResponseInput::Text(String::new()),
        ..Default::default()
    };

    // Empty text should still create a user message, so this should succeed
    let result = responses_to_chat(&req);
    assert!(result.is_ok());
}

#[test]
fn test_stream_options_include_obfuscation_roundtrip() {
    // Regression: ensure caller-provided stream_options (e.g. `include_obfuscation`)
    // are preserved through the Responses → Chat conversion when streaming.
    let req = ResponsesRequest {
        input: ResponseInput::Text("hi".to_string()),
        stream: Some(true),
        stream_options: Some(StreamOptions {
            include_usage: None,
            include_obfuscation: Some(false),
            ..StreamOptions::default()
        }),
        ..Default::default()
    };

    let chat_req = responses_to_chat(&req).unwrap();
    assert!(chat_req.stream);
    let opts = chat_req
        .stream_options
        .expect("stream_options populated when streaming");
    // Caller-provided value is preserved verbatim.
    assert_eq!(opts.include_obfuscation, Some(false));
    // include_usage defaults to true when absent so downstream consumers
    // still emit the usage block at end-of-stream.
    assert_eq!(opts.include_usage, Some(true));
}

#[test]
fn test_stream_options_caller_include_usage_preserved() {
    // Caller-set `include_usage` must not be clobbered by the conversion layer.
    let req = ResponsesRequest {
        input: ResponseInput::Text("hi".to_string()),
        stream: Some(true),
        stream_options: Some(StreamOptions {
            include_usage: Some(false),
            include_obfuscation: Some(true),
            ..StreamOptions::default()
        }),
        ..Default::default()
    };

    let opts = responses_to_chat(&req).unwrap().stream_options.unwrap();
    assert_eq!(opts.include_usage, Some(false));
    assert_eq!(opts.include_obfuscation, Some(true));
}

#[test]
fn test_stream_options_unknown_fields_survive_grpc_chat_builder() {
    // Regression: the gRPC path rebuilds a ChatCompletionRequest field by
    // field, so engine-specific streaming options carried in the catch-all
    // map must survive the conversion and reach the SSE assembler, which is
    // what renders usage chunks in gRPC mode.
    let mut extra = serde_json::Map::new();
    extra.insert(
        "step_usage_chunks".to_string(),
        serde_json::Value::String("all".to_string()),
    );

    let req = ResponsesRequest {
        input: ResponseInput::Text("hi".to_string()),
        stream: Some(true),
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
            other: extra,
            ..StreamOptions::default()
        }),
        ..Default::default()
    };

    let opts = responses_to_chat(&req).unwrap().stream_options.unwrap();
    assert_eq!(
        opts.other.get("step_usage_chunks").and_then(|v| v.as_str()),
        Some("all")
    );
}

#[test]
fn test_stream_options_non_streaming_dropped() {
    // stream=false must produce None stream_options even if caller set it.
    let req = ResponsesRequest {
        input: ResponseInput::Text("hi".to_string()),
        stream: Some(false),
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
            include_obfuscation: Some(false),
            ..StreamOptions::default()
        }),
        ..Default::default()
    };

    let chat_req = responses_to_chat(&req).unwrap();
    assert!(!chat_req.stream);
    assert!(chat_req.stream_options.is_none());
}

#[test]
fn test_image_generation_call_input_rejected() {
    // Regression: `image_generation_call` items are server-produced
    // output (populated via the shared MCP transformer) and must not
    // be round-tripped back into the chat conversion as input.
    // The regular gRPC path — used by non-Harmony text LLMs that only do
    // function calling — rejects this variant with the same contract as
    // sibling hosted-tool items (Computer/Shell/Custom/ApplyPatch).
    let req = ResponsesRequest {
        input: ResponseInput::Items(vec![ResponseInputOutputItem::ImageGenerationCall {
            id: Some("ig_test".to_string()),
            action: None,
            background: None,
            error: None,
            output_format: None,
            quality: None,
            result: Some("base64data".to_string()),
            revised_prompt: Some("a cat".to_string()),
            size: None,
            status: None,
            svg_usage: None,
        }]),
        ..Default::default()
    };

    let result = responses_to_chat(&req);
    assert_eq!(result.unwrap_err(), ConversionError::UnsupportedItem);
}

/// A message item's `input_file`, and an `input_image` that names only a
/// `file_id`, have no form in the chat pipeline: the conversion refuses
/// the request under the chat route's `unsupported_content_part`, naming
/// the part, instead of answering as if the part had not been sent. An
/// `input_image` with a URL converts to an `image_url` part as before,
/// and the refusal of an item kind keeps `convert_request_failed`.
#[test]
fn a_part_the_chat_pipeline_has_no_form_for_is_refused_not_dropped() {
    let message = |parts: serde_json::Value| serde_json::json!([{"type": "message", "role": "user", "content": parts}]);
    let file = responses_to_chat(&items_request(
        "local-model",
        message(serde_json::json!([
            {"type": "input_text", "text": "Summarize the attached document."},
            {"type": "input_file", "file_data": "JVBERi0xLjQK", "filename": "a.pdf"}
        ])),
    ))
    .expect_err("a file part has no chat form");
    assert_eq!(file.code(), "unsupported_content_part");
    assert!(file.to_string().contains("\"input_file\""), "{file}");

    let by_id = responses_to_chat(&items_request(
        "local-model",
        message(serde_json::json!([
            {"type": "input_image", "file_id": "file_123", "detail": "auto"}
        ])),
    ))
    .expect_err("an image named by a file_id alone has no chat form");
    assert_eq!(by_id.code(), "unsupported_content_part");
    assert!(by_id.to_string().contains("\"input_image\""), "{by_id}");

    let simple = responses_to_chat(&items_request(
        "local-model",
        serde_json::json!([{"role": "user", "content": [
            {"type": "input_file", "file_id": "file_123"}
        ]}]),
    ))
    .expect_err("a simple input message is read the same way");
    assert_eq!(simple.code(), "unsupported_content_part");

    let chat = responses_to_chat(&items_request(
        "local-model",
        message(serde_json::json!([
            {"type": "input_text", "text": "What is in the picture?"},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
        ])),
    ))
    .expect("an image with a URL converts");
    match &chat.messages[0] {
        ChatMessage::User {
            content: MessageContent::Parts(parts),
            ..
        } => {
            assert!(
                matches!(&parts[0], ContentPart::Text { text } if text == "What is in the picture?")
            );
            assert!(matches!(
                &parts[1],
                ContentPart::ImageUrl { image_url } if *image_url.url == "data:image/png;base64,iVBORw0KGgo="
            ));
        }
        other => panic!("the user turn carries parts: {other:?}"),
    }
    assert_eq!(
        ConversionError::UnsupportedItem.code(),
        "convert_request_failed"
    );
    assert_eq!(ConversionError::NoMessages.code(), "convert_request_failed");
    // A function output with a part the text-only chat path cannot carry
    // is a content-part refusal too (it used to answer convert_request_failed).
    assert_eq!(
        ConversionError::UnsupportedToolOutput("image".to_string()).code(),
        "unsupported_content_part"
    );
}

/// A `developer` item folds into the leading system message as text, so a
/// media part in it has nowhere to go: the conversion refuses the request
/// under `unsupported_content_part`, naming the part, instead of folding
/// the text and dropping the image (the request used to answer 200 as if
/// the part had not been sent). A text-only developer item folds as
/// before, in the message form and in the simple-input form.
#[test]
fn a_media_part_in_a_developer_item_is_refused_not_folded_away() {
    let developer = |parts: serde_json::Value| serde_json::json!([{"type": "message", "role": "developer", "content": parts}]);
    let image = responses_to_chat(&items_request(
        "local-model",
        developer(serde_json::json!([
            {"type": "input_text", "text": "Describe what you see."},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
        ])),
    ))
    .expect_err("a developer item's image has no place in the fold");
    assert_eq!(image.code(), "unsupported_content_part");
    assert!(image.to_string().contains("\"input_image\""), "{image}");
    assert!(image.to_string().contains("developer message"), "{image}");

    let simple = responses_to_chat(&items_request(
        "local-model",
        serde_json::json!([{"role": "developer", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
        ]}]),
    ))
    .expect_err("a simple developer message is read the same way");
    assert_eq!(simple.code(), "unsupported_content_part");

    let chat = responses_to_chat(&items_request(
        "local-model",
        serde_json::json!([
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Answer in French."}]},
            {"role": "developer", "content": "Keep it short."},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
        ]),
    ))
    .expect("text-only developer items fold");
    assert_eq!(chat.messages.len(), 2, "{:?}", chat.messages);
    match &chat.messages[0] {
        ChatMessage::System {
            content: MessageContent::Text(text),
            ..
        } => assert_eq!(
            text,
            "Developer instructions:\nAnswer in French.\n\nDeveloper instructions:\nKeep it short."
        ),
        other => panic!("the developer turns fold into the leading system message: {other:?}"),
    }
}

/// A function output with a part that has no text (an image, a file, a
/// refusal) is refused under the chat route's `unsupported_content_part`,
/// and the refusal names the part, as the refusal of a message part does;
/// it used to answer one fixed message whatever the part was.
#[test]
fn a_function_output_part_without_text_is_refused_by_name() {
    let input = serde_json::json!([
        {"type": "message", "role": "user", "content": "What is in the picture?"},
        {"type": "function_call", "call_id": "call_1", "name": "take_picture", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
        ]}
    ]);
    let refused = responses_to_chat(&items_request("local-model", input))
        .expect_err("an image part in a function output has no text form");
    assert_eq!(refused.code(), "unsupported_content_part");
    assert_eq!(
        refused.to_string(),
        "Unsupported content part type \"input_image\" in a function output for text-only backends: only text parts are carried as a tool result"
    );
}

#[test]
fn namespace_chat_response_roundtrips_identity() {
    let request: ResponsesRequest = namespace_test_request();
    let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
        "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
        "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
            {"id":"call_weather","type":"function","function":{"name":"weather.lookup","arguments":"{}"}}
        ]},"finish_reason":"tool_calls"}]
    })).unwrap();
    let response = chat_to_responses(&chat, &request, request.tools.as_deref(), None).unwrap();
    let wire = serde_json::to_value(&response.output[0]).unwrap();
    assert_eq!(wire["name"], "lookup");
    assert_eq!(wire["namespace"], "weather");
    let mut replay = request;
    replay.input = ResponseInput::Items(vec![serde_json::from_value(wire).unwrap()]);
    let converted = responses_to_chat(&replay).unwrap();
    let wire = serde_json::to_value(converted).unwrap();
    assert_eq!(
        wire["messages"][0]["tool_calls"][0]["function"]["name"],
        "weather.lookup"
    );
    assert_eq!(wire["tools"][0]["function"]["name"], "weather.lookup");
}
#[test]
fn generated_tool_calls_complete_the_response_without_executing_tools() {
    let request = ResponsesRequest::default();
    let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
        "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
        "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
            {"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}},
            {"id":"call_time","type":"function","function":{"name":"time","arguments":"{}"}}
        ]},"finish_reason":"tool_calls"}]
    }))
    .unwrap();
    let wire = serde_json::to_value(
        chat_to_responses(&chat, &request, request.tools.as_deref(), None).unwrap(),
    )
    .unwrap();
    assert_eq!(wire["status"], "completed");
    assert_eq!(wire["output"].as_array().unwrap().len(), 2);
    for (item, call_id) in wire["output"]
        .as_array()
        .unwrap()
        .iter()
        .zip(["call_weather", "call_time"])
    {
        assert_eq!(item["status"], "completed");
        assert_eq!(item["call_id"], call_id);
        assert_eq!(item["arguments"], "{}");
    }
}

/// A call the parser did not read whole, delivered under `tool_calls` with arguments that
/// are no JSON value (a tagged syntax that refused bytes inside the call writes no close for
/// it): the item is `incomplete`, and so is the response, with no stated reason, as a call
/// cut by an abort leaves it.
#[test]
fn a_generated_call_the_parser_did_not_read_whole_leaves_the_response_incomplete() {
    let request = ResponsesRequest::default();
    let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
        "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
        "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
            {"id":"call_add","type":"function","function":{"name":"add_integers","arguments":""}}
        ]},"finish_reason":"tool_calls"}]
    }))
    .unwrap();
    let wire = serde_json::to_value(
        chat_to_responses(&chat, &request, request.tools.as_deref(), None).unwrap(),
    )
    .unwrap();
    assert_eq!(wire["status"], "incomplete", "{wire}");
    assert_eq!(
        wire["incomplete_details"],
        serde_json::Value::Null,
        "{wire}"
    );
    assert_eq!(wire["output"][0]["type"], "function_call", "{wire}");
    assert_eq!(wire["output"][0]["status"], "incomplete", "{wire}");
    assert_eq!(wire["output"][0]["arguments"], "", "{wire}");
}

#[test]
fn failed_generation_does_not_complete_partial_messages_or_tool_calls() {
    for finish_reason in ["failed", "error"] {
        let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","content":"Partial reply","tool_calls":[
                {"id":"call_weather","type":"function","function":{
                    "name":"weather","arguments":"{\"city\":"
                }}
            ]},"finish_reason":finish_reason}]
        }))
        .unwrap();
        let response = chat_to_responses(&chat, &ResponsesRequest::default(), None, None).unwrap();
        let wire = serde_json::to_value(response).unwrap();
        assert_eq!(wire["status"], "failed", "{finish_reason}");
        assert_eq!(wire["output"].as_array().unwrap().len(), 2);
        assert_eq!(wire["output"][0]["type"], "message");
        assert_eq!(
            wire["output"][0]["status"], "in_progress",
            "{finish_reason}"
        );
        assert_eq!(wire["output"][0]["content"][0]["text"], "Partial reply");
        let item = &wire["output"][1];
        assert_eq!(item["status"], "in_progress", "{finish_reason}");
        assert_eq!(item["call_id"], "call_weather");
        assert_eq!(item["arguments"], "{\"city\":");
    }
}

#[test]
fn length_truncated_tool_call_preserves_earlier_completed_calls() {
    let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
        "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
        "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
            {"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}},
            {"id":"call_time","type":"function","function":{"name":"time","arguments":"{\"tz\":"}}
        ]},"finish_reason":"length"}]
    }))
    .unwrap();
    let response = chat_to_responses(&chat, &ResponsesRequest::default(), None, None).unwrap();
    let wire = serde_json::to_value(response).unwrap();
    assert_eq!(wire["status"], "incomplete");
    assert_eq!(wire["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(wire["output"].as_array().unwrap().len(), 2);
    assert_eq!(wire["output"][0]["status"], "completed");
    assert_eq!(wire["output"][0]["arguments"], "{}");
    assert_eq!(wire["output"][1]["status"], "incomplete");
    assert_eq!(wire["output"][1]["call_id"], "call_time");
    assert_eq!(wire["output"][1]["arguments"], "{\"tz\":");
}

#[test]
fn length_finish_preserves_a_fully_parsed_tool_call() {
    let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
        "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
        "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
            {"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}}
        ]},"finish_reason":"length"}]
    }))
    .unwrap();
    let wire = serde_json::to_value(
        chat_to_responses(&chat, &ResponsesRequest::default(), None, None).unwrap(),
    )
    .unwrap();
    assert_eq!(wire["status"], "incomplete");
    assert_eq!(wire["output"][0]["status"], "completed");
    assert_eq!(wire["output"][0]["arguments"], "{}");
}

#[test]
fn a_cut_call_under_a_finish_other_than_length_leaves_the_response_incomplete() {
    for finish_reason in [
        serde_json::json!("abort"),
        serde_json::json!("stop"),
        serde_json::json!(null),
    ] {
        let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
                {"id":"call_time","type":"function","function":{"name":"time","arguments":"{\"tz\":"}}
            ]},"finish_reason":finish_reason}]
        }))
        .unwrap();
        let wire = serde_json::to_value(
            chat_to_responses(&chat, &ResponsesRequest::default(), None, None).unwrap(),
        )
        .unwrap();
        assert_eq!(wire["status"], "incomplete", "{finish_reason}");
        assert!(wire["incomplete_details"].is_null(), "{finish_reason}");
        assert_eq!(wire["output"][0]["status"], "incomplete", "{finish_reason}");
        assert_eq!(
            wire["output"][0]["arguments"], "{\"tz\":",
            "{finish_reason}"
        );
    }
}

#[test]
fn a_cut_custom_tool_call_leaves_the_response_incomplete_too() {
    let request: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model", "input": "hi",
        "tools": [{"type": "custom", "name": "emit_command"}]
    }))
    .unwrap();
    for finish_reason in [
        serde_json::json!("abort"),
        serde_json::json!("stop"),
        serde_json::json!(null),
    ] {
        let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
                {"id":"call_c","type":"function","function":{"name":"emit_command","arguments":"{\"input\": \"pw"}}
            ]},"finish_reason":finish_reason}]
        }))
        .unwrap();
        let wire = serde_json::to_value(
            chat_to_responses(&chat, &request, request.tools.as_deref(), None).unwrap(),
        )
        .unwrap();
        assert_eq!(wire["status"], "incomplete", "{finish_reason}");
        assert!(wire["incomplete_details"].is_null(), "{finish_reason}");
        assert_eq!(
            wire["output"][0]["type"], "custom_tool_call",
            "{finish_reason}"
        );
        assert_eq!(wire["output"][0]["call_id"], "call_c", "{finish_reason}");
        let id = wire["output"][0]["id"].as_str().unwrap_or_default();
        assert!(id.starts_with("ctc_"), "{finish_reason}: {wire}");
    }
}

/// The MCP tool loop's conversion: the request's function tools and the
/// MCP tools are one list (nothing is wrapped twice), `tool_choice: none`
/// withholds the request's own tools and keeps the MCP tools offered, and a
/// request without either still carries an (empty) list, as the loop has
/// always sent one; the plain conversion carries none.
#[test]
fn the_tool_loop_offers_the_mcp_tools_in_one_list_with_the_request_s_function_tools() {
    use openai_protocol::responses::{ResponsesToolChoice, ToolChoiceOptions};
    let mcp: Tool = serde_json::from_value(serde_json::json!({
        "type": "function",
        "function": {"name": "mcp_echo", "parameters": {"type": "object", "properties": {}}}
    }))
    .unwrap();
    let names = |chat: &ChatCompletionRequest| {
        chat.tools.as_deref().map(|tools| {
            tools
                .iter()
                .map(|tool| tool.function.name.clone())
                .collect::<Vec<_>>()
        })
    };

    let request = namespace_test_request();
    let chat = responses_to_chat_with_mcp_tools(&request, std::slice::from_ref(&mcp)).unwrap();
    assert_eq!(
        names(&chat).unwrap(),
        ["weather.lookup", "travel.lookup", "mcp_echo"]
    );

    let mut forbidden = namespace_test_request();
    forbidden.tool_choice = Some(ResponsesToolChoice::Options(ToolChoiceOptions::None));
    let chat = responses_to_chat_with_mcp_tools(&forbidden, std::slice::from_ref(&mcp)).unwrap();
    assert_eq!(names(&chat).unwrap(), ["mcp_echo"]);

    let bare: ResponsesRequest =
        serde_json::from_value(serde_json::json!({"model": "test-model", "input": "hi"})).unwrap();
    assert_eq!(
        names(&responses_to_chat_with_mcp_tools(&bare, &[]).unwrap()),
        Some(Vec::new())
    );
    assert_eq!(names(&responses_to_chat(&bare).unwrap()), None);
}

/// The message content of an input item's parts: no part is an empty
/// string, one text part is that text, one part of another kind stays a
/// one-part list, and several parts stay a list in their order.
#[test]
fn input_parts_become_text_or_an_ordered_list() {
    let text = |text: &str| ResponseContentPart::InputText {
        text: text.to_string(),
    };
    let image = ResponseContentPart::InputImage {
        image_url: Some("https://example.com/a.png".to_string()),
        file_id: None,
        detail: None,
    };
    assert_eq!(
        response_parts_to_message_content(&[]).unwrap(),
        MessageContent::Text(String::new())
    );
    assert_eq!(
        response_parts_to_message_content(&[text("one")]).unwrap(),
        MessageContent::Text("one".to_string())
    );
    let lone_image = response_parts_to_message_content(std::slice::from_ref(&image)).unwrap();
    assert!(
        matches!(&lone_image, MessageContent::Parts(parts) if parts.len() == 1 && matches!(parts[0], ContentPart::ImageUrl { .. })),
        "a lone part of another kind stays a one-part list, got {lone_image:?}"
    );
    let several = response_parts_to_message_content(&[text("first"), image, text("last")]).unwrap();
    let MessageContent::Parts(parts) = several else {
        panic!("several parts stay a list, got {several:?}");
    };
    assert_eq!(parts.len(), 3);
    assert!(matches!(&parts[0], ContentPart::Text { text } if text == "first"));
    assert!(matches!(&parts[1], ContentPart::ImageUrl { .. }));
    assert!(matches!(&parts[2], ContentPart::Text { text } if text == "last"));
}

#[test]
fn responses_to_chat_converts_plaintext_agent_message_and_rejects_encrypted() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "input": [{
            "type": "agent_message",
            "id": "am_1",
            "author": "/root",
            "recipient": "/root/worker",
            "content": [{"type": "input_text", "text": "Reply OK"}]
        }]
    }))
    .unwrap();

    let chat = responses_to_chat(&req).expect("plaintext agent_message converts");
    assert_eq!(chat.messages.len(), 1);
    match &chat.messages[0] {
        ChatMessage::User {
            content: MessageContent::Text(text),
            ..
        } => {
            assert!(text.contains("Sender: /root"), "{text}");
            assert!(text.contains("Recipient: /root/worker"), "{text}");
            assert!(text.ends_with("Reply OK"), "{text}");
        }
        other => panic!("expected user text message, got {other:?}"),
    }

    let encrypted: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "input": [{
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
            "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}]
        }]
    }))
    .unwrap();
    let err = responses_to_chat(&encrypted).expect_err("encrypted agent_message must fail");
    assert_eq!(err.code(), "unsupported_content_part");
}
