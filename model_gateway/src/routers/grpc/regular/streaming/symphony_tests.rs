//! Symphony reading the chat stream: the assignment on the response spec switches the drivers
//! from the tool and reasoning parsers to one engine per choice, with the same chunks on the wire.

use llm_tokenizer::StopSequenceConfig;
use openai_protocol::{
    chat::ChatChoice,
    common::{ChatLogProbs, StringOrArray},
    profile::ProviderProfile,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto::GenerateComplete, vllm_engine::proto};
use symphony::{
    formats::{plain, qwen3},
    CallSyntax, Format,
};

use super::{
    eof_tests::{
        chat_spec, chat_text, chunk, collect_events, complete, dispatch, processor,
        scripted_stream, CharacterTokenizer,
    },
    *,
};
use crate::routers::grpc::{
    proto_wrapper::ProtoGenerateComplete,
    regular::{processor::ResponseProcessor, symphony::Assignment},
};

/// The spec with Symphony assigned `format`, replaying `prompt`.
fn assigned(mut spec: ChatResponseSpec, format: Format, prompt: &str) -> ChatResponseSpec {
    spec.symphony = Some(Assignment {
        format,
        prompt: prompt.to_string(),
    });
    spec
}

/// A Qwen3 output: a thought, content, and one call of the declared `lookup` tool.
const QWEN3_OUTPUT: &str = "<think>plan</think>Hello <tool_call>\n\
                            {\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>";

async fn stream_events(
    responses: Vec<proto::GenerateResponse>,
    with_tools: bool,
    spec: ChatResponseSpec,
) -> (Result<(), utils::StreamFailure>, Vec<Value>) {
    let (stream, server) = scripted_stream(responses, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor(with_tools)
        .process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    (result, events)
}

/// The tool-call deltas of choice 0, in order.
fn tool_call_deltas(events: &[Value]) -> Vec<Value> {
    events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .cloned()
        .collect()
}

#[tokio::test]
async fn symphony_streams_reasoning_content_and_a_tool_call_from_a_qwen3_output() {
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    // Cut inside the thought and inside a marker, so the chunking crosses both.
    let responses = vec![
        chunk(0, &QWEN3_OUTPUT[..9]),
        chunk(0, &QWEN3_OUTPUT[9..32]),
        chunk(0, &QWEN3_OUTPUT[32..]),
        complete(0, "stop"),
    ];
    let (result, events) = stream_events(responses, true, spec).await;
    assert!(result.is_ok(), "{result:?}");

    assert_eq!(chat_text(&events, 0, "reasoning_content"), "plan");
    let content = chat_text(&events, 0, "content");
    assert_eq!(content.trim(), "Hello", "{content:?}");

    let calls = tool_call_deltas(&events);
    assert!(!calls.is_empty(), "{events:#?}");
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "lookup");
    assert!(calls[0]["id"].is_string(), "{:?}", calls[0]);
    let arguments: String = calls
        .iter()
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    let arguments: Value = serde_json::from_str(&arguments).expect("the arguments as JSON");
    assert_eq!(arguments, json!({"q": 1}));

    let finish = events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk");
    assert_eq!(finish, "tool_calls");

    // The opening chunk carries the role; no later delta repeats it.
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(events[1..]
        .iter()
        .all(|event| event["choices"][0]["delta"]["role"].is_null()));
}

/// [`stream_events`] with tools and a stop sequence the decoder matches.
async fn stream_events_stopping(
    responses: Vec<proto::GenerateResponse>,
    spec: ChatResponseSpec,
    stop: &str,
) -> (Result<(), utils::StreamFailure>, Vec<Value>) {
    let (stream, server) = scripted_stream(responses, "0").await;
    let (tx, rx) = sse_channel();
    let stop = Some(StringOrArray::String(stop.to_string()));
    let result = processor(true)
        .process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (stop, None, false, false, false),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    (result, events)
}

#[tokio::test]
async fn a_stop_the_decoder_matches_without_a_complete_frame_still_ends_with_tool_calls() {
    // The decoder matches `DONE` after the call and no Complete frame arrives: the output ends as
    // an abort at the transport's end, and the `Finish` event's count of calls still decides the
    // finish reason, as it does when a Complete frame ends the output.
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let output = format!("{QWEN3_OUTPUT} DONE");
    let responses = vec![chunk(0, &output[..20]), chunk(0, &output[20..])];
    let (result, events) = stream_events_stopping(responses, spec, "DONE").await;
    assert!(result.is_ok(), "{result:?}");
    let calls = tool_call_deltas(&events);
    assert!(!calls.is_empty(), "{events:#?}");
    assert_eq!(calls[0]["function"]["name"], "lookup");
    let finish = events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk");
    assert_eq!(finish, "tool_calls");

    // Without a stop or a Complete frame, the transport's end gives no finish chunk, as on the
    // older path; the events still carry the whole call.
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let responses = vec![chunk(0, QWEN3_OUTPUT)];
    let (result, events) = stream_events(responses, true, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert!(!tool_call_deltas(&events).is_empty(), "{events:#?}");
    assert!(events
        .iter()
        .all(|event| event["choices"][0]["finish_reason"].is_null()));
}

#[tokio::test]
async fn the_prompts_open_thought_is_replayed_so_the_output_starts_inside_it() {
    let spec = assigned(
        chat_spec(false),
        qwen3(CallSyntax::Json),
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
    let responses = vec![chunk(0, "plan</think>Hi"), complete(0, "stop")];
    let (result, events) = stream_events(responses, false, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "plan");
    assert_eq!(chat_text(&events, 0, "content").trim(), "Hi");
    let finish = events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk");
    assert_eq!(finish, "stop");
}

#[tokio::test]
async fn without_an_assignment_the_older_parsers_read_the_stream() {
    let responses = || vec![chunk(0, "<think>plan</think>Hi"), complete(0, "stop")];
    // Assigned the plain table, Symphony reads the thought as content.
    let spec = assigned(chat_spec(false), plain(), "");
    let (result, events) = stream_events(responses(), false, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "content"), "<think>plan</think>Hi");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "");
    // With no assignment, the configured reasoning parser takes it, as before.
    let (result, events) = stream_events(responses(), false, chat_spec(false)).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "plan");
    assert_eq!(chat_text(&events, 0, "content"), "Hi");
}

/// A whole `output` read by the non-streaming driver with the Qwen3 table, one logprob per
/// character token: the choice and its reasoning token count.
async fn fold(output: &str) -> (ChatChoice, u32) {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let output_ids: Vec<u32> = output.chars().map(u32::from).collect();
    let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids: output_ids.clone(),
        finish_reason: "stop".to_string(),
        output_logprobs: Some(smg_grpc_client::tokenspeed_proto::OutputLogProbs {
            token_logprobs: vec![-0.5; output_ids.len()],
            token_ids: output_ids,
            top_logprobs: Vec::new(),
        }),
        ..Default::default()
    });
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), false);
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    processor
        .process_single_choice(
            "test-request",
            &complete,
            0,
            &spec,
            "Qwen/Qwen3-8B",
            &tokenizer,
            &mut decoder,
            0,
            false,
            false,
            None,
            None,
        )
        .await
        .expect("a choice")
}

/// The tokens whose logprobs the choice kept, joined.
fn kept_logprob_tokens(choice: &ChatChoice) -> String {
    let Some(ChatLogProbs::Detailed {
        content: Some(items),
        ..
    }) = &choice.logprobs
    else {
        panic!("detailed logprobs");
    };
    items.iter().map(|item| item.token.as_str()).collect()
}

#[tokio::test]
async fn symphony_folds_a_whole_output_into_the_message() {
    // One logprob per output token: the choice keeps the content's alone, as the older path does.
    let (choice, reasoning_tokens) = fold(QWEN3_OUTPUT).await;
    assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
    assert_eq!(
        choice.message.content.as_deref().map(str::trim),
        Some("Hello")
    );
    let calls = choice.message.tool_calls.as_ref().expect("tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "lookup");
    let arguments: Value =
        serde_json::from_str(calls[0].function.arguments.as_deref().unwrap_or(""))
            .expect("the arguments as JSON");
    assert_eq!(arguments, json!({"q": 1}));
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    // One token per character: the thought's four.
    assert_eq!(reasoning_tokens, 4);
    assert_eq!(
        kept_logprob_tokens(&choice),
        "Hello ",
        "the content's tokens alone keep their logprobs"
    );
}

#[tokio::test]
async fn a_message_without_content_keeps_no_logprobs() {
    // The template's newlines between the thought and the call are content to the engine, but
    // whitespace alone is no message content: no token keeps its logprob, as on the older path.
    let (choice, _) = fold(concat!(
        "<think>plan</think>\n\n<tool_call>\n",
        "{\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>"
    ))
    .await;
    assert_eq!(choice.message.content, None);
    assert_eq!(choice.message.tool_calls.as_ref().map(Vec::len), Some(1));
    assert_eq!(kept_logprob_tokens(&choice), "");
}

#[tokio::test]
async fn malformed_call_text_is_content_and_keeps_its_logprobs() {
    // A call region that never forms a call comes back malformed and reaches the client as
    // content, so its tokens keep their logprobs with the rest of the content's.
    let (choice, _) = fold("<think>plan</think>Hello <tool_call>\nnot a call\n</tool_call>").await;
    let content = choice.message.content.clone().expect("content");
    assert!(content.contains("not a call"), "{content}");
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(kept_logprob_tokens(&choice), content);
}

/// The `lookup` tool the outputs call.
fn lookup_tools() -> Vec<Tool> {
    serde_json::from_value(json!([{
        "type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object", "properties": {}}
        }
    }]))
    .expect("tools")
}

/// A Messages spec with Symphony assigned `format`, replaying `prompt`, with the `lookup` tool.
fn assigned_message(format: Format, prompt: &str, thinking: bool) -> MessagesResponseSpec {
    MessagesResponseSpec {
        provider: ProviderProfile::Generic,
        continues_final_assistant: false,
        thinking: thinking.then_some(messages::ThinkingConfig::Enabled {
            budget_tokens: 1024,
            display: None,
        }),
        reasoning_in_prefill: None,
        tool_choice: None,
        has_tools: true,
        history_tool_calls_count: 0,
        chat_tools: lookup_tools(),
        stop_sequences: None,
        symphony: Some(Assignment {
            format,
            prompt: prompt.to_string(),
        }),
    }
}

async fn message_events(
    responses: Vec<proto::GenerateResponse>,
    spec: MessagesResponseSpec,
    stop: Option<&str>,
) -> (Result<(), utils::StreamFailure>, Vec<Value>) {
    let (stream, server) = scripted_stream(responses, "0").await;
    let (tx, rx) = sse_channel();
    let stop = stop.map(|stop| StringOrArray::String(stop.to_string()));
    let result = processor(true)
        .process_messages_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (stop, None, false, false, false),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    (result, events)
}

/// The content blocks of a streamed message as `(type, text or name, input JSON)`, rebuilt from
/// the deltas: blocks must open and close one at a time, with indices in order.
fn message_blocks(events: &[Value]) -> Vec<(String, String, Value)> {
    let mut blocks: Vec<(String, String, String)> = Vec::new();
    let mut open: Option<u64> = None;
    for event in events {
        match event["type"].as_str() {
            Some("content_block_start") => {
                assert_eq!(open, None, "a block starts while another is open: {event}");
                assert_eq!(
                    event["index"].as_u64(),
                    Some(blocks.len() as u64),
                    "{event}"
                );
                open = Some(blocks.len() as u64);
                let block = &event["content_block"];
                let kind = block["type"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                blocks.push((kind, name, String::new()));
            }
            Some("content_block_delta") => {
                assert_eq!(
                    open,
                    event["index"].as_u64(),
                    "a delta for a closed block: {event}"
                );
                let delta = &event["delta"];
                let last = blocks.last_mut().expect("an open block");
                for key in ["thinking", "text", "partial_json"] {
                    if let Some(part) = delta[key].as_str() {
                        last.2.push_str(part);
                    }
                }
            }
            Some("content_block_stop") => {
                assert_eq!(open.take(), event["index"].as_u64(), "{event}");
            }
            _ => {}
        }
    }
    assert_eq!(open, None, "a block is left open");
    blocks
        .into_iter()
        .map(|(kind, name, body)| {
            let value = if kind == "tool_use" {
                serde_json::from_str(&body).expect("the input as JSON")
            } else {
                Value::String(body)
            };
            (kind, name, value)
        })
        .collect()
}

fn message_stop_reason(events: &[Value]) -> Value {
    events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .expect("a message_delta")["delta"]["stop_reason"]
        .clone()
}

#[tokio::test]
async fn symphony_streams_a_message_as_thinking_text_and_tool_use_blocks() {
    let spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", true);
    let responses = vec![
        chunk(0, &QWEN3_OUTPUT[..9]),
        chunk(0, &QWEN3_OUTPUT[9..32]),
        chunk(0, &QWEN3_OUTPUT[32..]),
        complete(0, "stop"),
    ];
    let (result, events) = message_events(responses, spec, None).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(events[0]["type"], "message_start");
    let blocks = message_blocks(&events);
    assert_eq!(blocks.len(), 3, "{blocks:?}");
    assert_eq!(blocks[0].0, "thinking");
    assert_eq!(blocks[0].2, "plan");
    assert_eq!(blocks[1].0, "text");
    assert_eq!(blocks[1].2.as_str().map(str::trim), Some("Hello"));
    assert_eq!(blocks[2].0, "tool_use");
    assert_eq!(blocks[2].1, "lookup");
    assert_eq!(blocks[2].2, json!({"q": 1}));
    assert_eq!(message_stop_reason(&events), "tool_use");
    assert_eq!(events.last().expect("events")["type"], "message_stop");
    // The table's framing never reaches the client.
    let text: String = events
        .iter()
        .filter_map(|event| event["delta"]["text"].as_str())
        .collect();
    assert!(!text.contains("<tool_call>"), "{text:?}");
}

#[tokio::test]
async fn symphony_message_stop_reasons_follow_the_finish_and_the_matched_stop() {
    // The token limit.
    let spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    let (result, events) =
        message_events(vec![chunk(0, "Hello"), complete(0, "length")], spec, None).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(message_blocks(&events)[0].2, "Hello");
    assert_eq!(message_stop_reason(&events), "max_tokens");
    // The turn's end.
    let spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    let (_, events) =
        message_events(vec![chunk(0, "Hello"), complete(0, "stop")], spec, None).await;
    assert_eq!(message_stop_reason(&events), "end_turn");
    // A stop sequence the decoder matched.
    let spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    let (_, events) = message_events(
        vec![chunk(0, "Hello END more"), complete(0, "stop")],
        spec,
        Some("END"),
    )
    .await;
    assert_eq!(message_blocks(&events)[0].2, "Hello ");
    assert_eq!(message_stop_reason(&events), "stop_sequence");
    let delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .expect("a message_delta");
    assert_eq!(delta["delta"]["stop_sequence"], "END");
}
