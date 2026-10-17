//! Symphony reading the chat stream: the assignment on the response spec switches the drivers
//! from the tool and reasoning parsers to one engine per choice, with the same chunks on the wire.

use std::sync::Arc;

use llm_tokenizer::{
    encoders::deepseek_v4::{encode_messages, EncodeParams, ThinkingMode},
    registry::TokenizerRegistry,
    StopSequenceConfig,
};
use openai_protocol::{
    chat::{ChatChoice, ChatCompletionRequest},
    common::{ChatLogProbs, StringOrArray},
    smg::ParseReport,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto::GenerateComplete, vllm_engine::proto};
use symphony::{
    formats::{deepseek_v4, glm, plain, qwen3},
    CallSyntax, Format,
};

use super::{
    eof_tests::{
        chat_spec, chat_text, chunk, collect_events, complete, dispatch, processor,
        scripted_stream, CharacterTokenizer,
    },
    *,
};
use crate::{
    config::ParserChoice,
    routers::grpc::{
        proto_wrapper::ProtoGenerateComplete,
        regular::{
            processor::ResponseProcessor,
            symphony::{assign, telemetry::Report, template::Templates, Asks, Assignment},
        },
        utils::ModelNames,
    },
};

/// The spec with Symphony assigned `format`, replaying `prompt`.
fn assigned(mut spec: ChatResponseSpec, format: Format, prompt: &str) -> ChatResponseSpec {
    spec.symphony = Some(Assignment {
        format: Arc::new(format),
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

/// A chat request with the conversation of a retry and no tools: the user asked for
/// `list_skills`, the assistant called it, the tool was unavailable, the user asks again.
fn retry_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "MiniMax-M3", "stream": true, "separate_reasoning": true,
        "messages": [
            {"role": "user", "content": "Call the `list_skills` tool once to see what we can do, \
                                         and reply with nothing else."},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "list_skills", "arguments": "{}"}
            }]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": "`list_skills` is unavailable for now; call it again later."},
            {"role": "user", "content": "The cooldown is over. Try again."}
        ]
    }))
    .expect("chat request")
}

/// The prompt the model's public chat template renders for the retry conversation
/// ([`retry_request`], thinking adaptive, no `tools`): the earlier call in the history is
/// written as an invoke with no parameters, and the generation prompt ends at the turn
/// opener, so the replay starts the output in the content with the thought the model's to
/// open.
const RETRY_PROMPT: &str = concat!(
    "]~!b[]~b]system\n",
    "Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers of AI towards AGI.\n",
    "\n",
    "<thinking_instructions>\n",
    "You have a thinking capability that allows you to reason step by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. When thinking is adaptive, decide on your own whether to think for the current turn.\n",
    "Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n",
    "</thinking_instructions>[e~[\n",
    "]~b]developer\n",
    "You are a helpful assistant.[e~[\n",
    "]~b]user\n",
    "Call the `list_skills` tool once to see what we can do, and reply with nothing else.[e~[\n",
    "]~b]ai\n",
    "</mm:think>]<]minimax[>[<tool_call>\n",
    "]<]minimax[>[<invoke name=\"list_skills\">]<]minimax[>[</invoke>\n",
    "]<]minimax[>[</tool_call>[e~[\n",
    "]~b]tool\n",
    "<response>Tool `list_skills` unavailable for now. Please call it later.</response>[e~[\n",
    "]~b]user\n",
    "<system-reminder>The cooldown is over, try again.</system-reminder>[e~[\n",
    "]~b]ai\n"
);

#[tokio::test]
async fn a_call_the_conversation_established_is_a_call_without_a_tool_inventory() {
    // The request declares no tools and the model's provider reads calls without them, so the
    // older path runs its tool parser; the assignment keeps the table's calls the same way, and
    // the block the model writes for the retry comes back as a call, not as content.
    let request = retry_request();
    let assignment = assign(
        ParserChoice::Symphony,
        &ModelNames::default(),
        Asks::Chat(&request),
        None,
        RETRY_PROMPT,
        &Templates::new(Arc::new(TokenizerRegistry::new())),
    )
    .expect("no template named")
    .expect("the registry's table");
    assert_eq!(
        assignment.prompt, "]~b]ai\n",
        "the replay starts at the turn opener"
    );
    let mut spec = ChatResponseSpec::from(&request);
    assert!(
        spec.tools.as_deref().is_some_and(<[Tool]>::is_empty),
        "scanned for calls, with no inventory"
    );
    spec.symphony = Some(assignment);
    let output = "<mm:think>The user asks for the tool again.</mm:think>\
                  <tool_call>\n<invoke name=\"list_skills\"></invoke>\n</tool_call>";
    let responses = vec![
        chunk(0, &output[..30]),
        chunk(0, &output[30..]),
        complete(0, "stop"),
    ];
    let (result, events) = stream_events(responses, false, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        chat_text(&events, 0, "reasoning_content"),
        "The user asks for the tool again."
    );
    assert_eq!(
        chat_text(&events, 0, "content").trim(),
        "",
        "the block is no content"
    );
    let calls = tool_call_deltas(&events);
    assert!(!calls.is_empty(), "{events:#?}");
    assert_eq!(calls[0]["function"]["name"], "list_skills");
    let arguments: String = calls
        .iter()
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    let arguments: Value = serde_json::from_str(&arguments).expect("the arguments as JSON");
    assert_eq!(arguments, json!({}));
    let finish = events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk");
    assert_eq!(finish, "tool_calls");
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
    fold_with(output, qwen3(CallSyntax::Json), "<|im_start|>assistant\n").await
}

/// [`fold`] over `format` after `prompt`.
async fn fold_with(output: &str, format: Format, prompt: &str) -> (ChatChoice, u32) {
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
    let spec = assigned(chat_spec(true), format, prompt);
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
        .map(|(choice, reasoning_tokens, _)| (choice, reasoning_tokens))
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
async fn a_refused_block_released_as_the_answer_keeps_its_tokens_logprobs() {
    // The whole answer after the thought is a call written as Python inside the call tag: the
    // reader releases the refused bytes as the content, and their tokens keep their logprobs,
    // placed where the output wrote them, not where the released event stands.
    let output = "plan</think><tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    let (choice, reasoning_tokens) = fold_with(output, glm(), "<|assistant|><think>").await;
    let released = "get_stock_info(company=\"Apple\")</tool_call>";
    assert_eq!(choice.message.content.as_deref(), Some(released));
    assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    assert_eq!(reasoning_tokens, 4);
    assert_eq!(kept_logprob_tokens(&choice), released);
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
        parse_report: Report::Off,
        parse_silent: false,
        parse_strict: false,
        first_tool_call_only: false,
        thinking_requested: thinking,
        thinking_text_returned: true,
        reasoning_in_prefill: false,
        tool_choice: None,
        has_tools: true,
        history_tool_calls_count: 0,
        chat_tools: lookup_tools().into(),
        stop_sequences: None,
        symphony: Some(Assignment {
            format: Arc::new(format),
            prompt: prompt.to_string(),
        }),
    }
}

pub(super) async fn message_events(
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
pub(super) fn message_blocks(events: &[Value]) -> Vec<(String, String, Value)> {
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

/// The `tool_use` ids a streamed message opened, in order.
fn tool_use_ids(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .filter(|event| {
            event["type"] == "content_block_start" && event["content_block"]["type"] == "tool_use"
        })
        .map(|event| {
            event["content_block"]["id"]
                .as_str()
                .expect("a tool_use id")
                .to_string()
        })
        .collect()
}

#[tokio::test]
async fn two_symphony_messages_never_share_a_tool_use_id() {
    // A client keeps its tool_use blocks and tool_results by id across a conversation: a second
    // turn's call must not come back under the first turn's id, and the id has the shape the
    // older parsers mint (`toolu_` and 24 hex characters).
    let mut ids = Vec::new();
    for _ in 0..2 {
        let spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
        let responses = vec![chunk(0, QWEN3_OUTPUT), complete(0, "stop")];
        let (result, events) = message_events(responses, spec, None).await;
        assert!(result.is_ok(), "{result:?}");
        let turn = tool_use_ids(&events);
        assert_eq!(turn.len(), 1, "{turn:?}");
        ids.extend(turn);
    }
    assert_ne!(ids[0], ids[1], "{ids:?}");
    for id in &ids {
        let hex = id.strip_prefix("toolu_").unwrap_or_else(|| panic!("{id}"));
        assert!(
            hex.len() == 24 && hex.bytes().all(|b| b.is_ascii_hexdigit()),
            "{id}"
        );
    }
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

/// The Messages body a coding agent's client sends, as recorded: its environment notes as a
/// system-role message after the user turn, rendered in place. The family's template writes the
/// turn opener after a user message only, so the prompt opens no turn for the model, which starts
/// inside its thought: the thought and its close are a thinking block (none when the thought is
/// empty), never a text block, on the Messages API and on chat completions alike.
#[tokio::test]
async fn a_conversation_that_ends_with_a_system_turn_keeps_the_thought_out_of_the_text() {
    let messages = json!([
        {
            "role": "system",
            "content": "You are a coding assistant working in a Python repository. Use the tools \
                        to inspect and edit files. Be brief."
        },
        {
            "role": "user",
            "content": "The test suite (python3 -m pytest -q) has a failing test in \
                        tests/test_versions.py. Find the bug in the source code under src/ and fix \
                        it so the whole suite passes. Do not modify anything under tests/."
        },
        {
            "role": "system",
            "content": "# Environment\nYou have been invoked in the following environment:\n - \
                        Primary working directory: /work/repo\n - Is a git repository: true\n - \
                        Platform: linux\n\nToday's date is 2026-10-16."
        },
    ]);
    let prompt = encode_messages(
        messages.as_array().expect("messages"),
        ThinkingMode::Thinking,
        &EncodeParams::default(),
    )
    .expect("the rendered prompt");
    // The template opened no turn: no assistant opener, no `<think>`.
    assert!(!prompt.contains("<｜Assistant｜>"), "{prompt:?}");
    assert!(!prompt.contains("<think>"), "{prompt:?}");
    let call = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n\n</｜DSML｜invoke>\n\
                </｜DSML｜tool_calls>";
    // The thought, its close, the template's spacing and the call, in an engine's pieces.
    let spec = assigned_message(deepseek_v4(), &prompt, true);
    let responses = vec![
        chunk(0, "The date is in the notes."),
        chunk(0, "</think>"),
        chunk(0, "\n\n"),
        chunk(0, call),
        complete(0, "stop"),
    ];
    let (result, events) = message_events(responses, spec, None).await;
    assert!(result.is_ok(), "{result:?}");
    let blocks = message_blocks(&events);
    assert_eq!(blocks.len(), 2, "{blocks:?}");
    assert_eq!(blocks[0].0, "thinking");
    assert_eq!(blocks[0].2, "The date is in the notes.");
    assert_eq!(blocks[1].0, "tool_use");
    assert_eq!(blocks[1].1, "lookup");
    assert_eq!(blocks[1].2, json!({}));
    assert_eq!(message_stop_reason(&events), "tool_use");
    // An empty thought, its close alone before the call: no thinking block and no text block.
    let spec = assigned_message(deepseek_v4(), &prompt, true);
    let responses = vec![
        chunk(0, "</think>\n\n"),
        chunk(0, call),
        complete(0, "stop"),
    ];
    let (result, events) = message_events(responses, spec, None).await;
    assert!(result.is_ok(), "{result:?}");
    let blocks = message_blocks(&events);
    assert_eq!(blocks.len(), 1, "{blocks:?}");
    assert_eq!(blocks[0].0, "tool_use");
    let text: String = events
        .iter()
        .filter_map(|event| event["delta"]["text"].as_str())
        .collect();
    assert_eq!(text, "");
    // Chat completions on the same prompt: the thought is `reasoning_content`, the content empty.
    let spec = assigned(chat_spec(true), deepseek_v4(), &prompt);
    let responses = vec![
        chunk(0, "The date is in the notes.</think>\n\n"),
        chunk(0, call),
        complete(0, "stop"),
    ];
    let (result, events) = stream_events(responses, true, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        chat_text(&events, 0, "reasoning_content"),
        "The date is in the notes."
    );
    assert_eq!(chat_text(&events, 0, "content").trim(), "");
    assert!(!tool_call_deltas(&events).is_empty(), "{events:#?}");
}

/// A `tool_use` block never carries `{}` for arguments the model wrote as JSON that no input
/// value holds: the message is refused with the call named, where the stream carries the text as
/// written for the client's own reader.
#[tokio::test]
async fn a_message_whose_tool_use_input_cannot_carry_the_arguments_is_refused() {
    let output = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"days\": 9.99e999999}}\n\
                  </tool_call>";
    let spec = || assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    // The whole message: refused, the call and the reader's reason in the error.
    let mut frame = complete(0, "stop");
    if let Some(proto::generate_response::Response::Complete(complete)) = &mut frame.response {
        complete.output_ids = output.chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
    }
    let (stream, server) = scripted_stream(vec![frame], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, true);
    let result = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    )
    .process_non_streaming_messages_response(
        context::ExecutionResult::Single { stream },
        spec(),
        dispatch(),
        tokenizer,
        &mut decoder,
    )
    .await;
    server.abort();
    let response = match result {
        Ok((message, _)) => panic!("a message with {:?}", message.content),
        Err(response) => response,
    };
    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    let body = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        axum::body::to_bytes(response.into_body(), 65536),
    )
    .await
    .expect("the body in time")
    .expect("the body");
    let body = String::from_utf8(body.to_vec()).expect("text");
    assert!(
        body.contains("symphony_tool_use_input_refused")
            && body.contains("call 0 (lookup)")
            && body.contains("number out of range"),
        "{body}"
    );
    // The stream: the fragments as the model wrote them, and the stop reason the calls give.
    let (result, events) =
        message_events(vec![chunk(0, output), complete(0, "stop")], spec(), None).await;
    assert!(result.is_ok(), "{result:?}");
    let partial: String = events
        .iter()
        .filter_map(|event| event["delta"]["partial_json"].as_str())
        .collect();
    assert_eq!(partial, "{\"days\": 9.99e999999}");
    assert_eq!(message_stop_reason(&events), "tool_use");
}

/// The scrape of a recorder of this test's own after `read` ran on this thread (the test
/// runtime is single-threaded, so the local recorder sees every record of the future).
async fn scraped<F: std::future::Future>(read: F) -> (F::Output, String) {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let guard = metrics::set_default_local_recorder(&recorder);
    let output = read.await;
    drop(guard);
    (output, handle.render())
}

/// The line of `scrape` for the series `name` with every label in `labels`.
fn series_line<'a>(scrape: &'a str, name: &str, labels: &[&str]) -> Option<&'a str> {
    scrape
        .lines()
        .find(|line| line.starts_with(name) && labels.iter().all(|label| line.contains(label)))
}

#[tracing_test::traced_test]
#[tokio::test]
async fn a_malformed_region_in_a_whole_output_is_counted_and_written_once() {
    // A call region that never forms a call: the content keeps its bytes, the counters say
    // which table gave up and why, and one line names the request, the choice and the model.
    let ((choice, _), scrape) = scraped(fold("Hello <tool_call>\nnot a call\n</tool_call>")).await;
    let content = choice.message.content.expect("content");
    assert!(content.contains("not a call"), "{content}");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(
        series_line(
            &scrape,
            "smg_symphony_malformed_total",
            &["table=\"qwen3\""]
        )
        .is_some(),
        "{scrape}"
    );
    assert!(
        series_line(
            &scrape,
            "smg_symphony_malformed_bytes_count",
            &["table=\"qwen3\""]
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "{scrape}"
    );
    assert!(logs_contain(
        "Symphony did not read the whole output as content, reasoning and calls"
    ));
    assert!(logs_contain("request_id=\"test-request\""));
    assert!(logs_contain("model=\"Qwen/Qwen3-8B\""));
    assert!(logs_contain("table=\"qwen3\""));
    assert!(logs_contain("choice=0"));
}

/// A call whose arguments are not JSON (an expression where a number was due): the client gets
/// the call with the arguments exactly as the model wrote them, nothing of the block as content,
/// and the output is counted as malformed under `invalid_arguments`, the line naming the byte
/// where the arguments stopped being JSON (`at`) and a run of no bytes.
#[tracing_test::traced_test]
#[tokio::test]
async fn arguments_that_are_not_json_reach_the_client_whole_and_are_counted() {
    let output =
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 2.0 * Math.PI}}\n</tool_call>";
    let ((choice, _), scrape) = scraped(fold(output)).await;
    let calls = choice.message.tool_calls.as_ref().expect("tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "lookup");
    assert_eq!(
        calls[0].function.arguments.as_deref(),
        Some("{\"q\": 2.0 * Math.PI}"),
        "the arguments as the model wrote them"
    );
    assert!(
        choice
            .message
            .content
            .as_deref()
            .is_none_or(|content| content.trim().is_empty()),
        "{:?}",
        choice.message.content
    );
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let malformed = series_line(
        &scrape,
        "smg_symphony_malformed_total",
        &["table=\"qwen3\"", "reason=\"invalid_arguments\""],
    )
    .expect("the malformed counter");
    assert!(malformed.ends_with(" 1"), "{malformed}");
    assert!(logs_contain("reason=\"invalid_arguments\""));
    // The break stands at the `*`: the output's bytes up to it, and the run has no bytes.
    let at = output.find('*').expect("the expression");
    assert!(logs_contain(&format!("at={at} bytes=0")));

    // The same in a stream cut inside the expression: the argument deltas, joined, are the
    // arguments as written, and no content delta carries any of them.
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let ((result, events), scrape) = scraped(stream_events(
        vec![
            chunk(
                0,
                "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 2.0 * Ma",
            ),
            chunk(0, "th.PI}}\n</tool_call>"),
            complete(0, "stop"),
        ],
        true,
        spec,
    ))
    .await;
    result.expect("the stream ends well");
    let arguments: String = tool_call_deltas(&events)
        .iter()
        .filter_map(|delta| delta["function"]["arguments"].as_str())
        .collect();
    assert_eq!(arguments, "{\"q\": 2.0 * Math.PI}");
    assert_eq!(chat_text(&events, 0, "content").trim(), "");
    let finish = events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk");
    assert_eq!(finish, "tool_calls");
    let malformed = series_line(
        &scrape,
        "smg_symphony_malformed_total",
        &["table=\"qwen3\"", "reason=\"invalid_arguments\""],
    )
    .expect("the malformed counter");
    assert!(malformed.ends_with(" 1"), "{malformed}");
}

#[tracing_test::traced_test]
#[tokio::test]
async fn a_clean_whole_output_is_counted_and_writes_nothing() {
    let ((choice, _), scrape) = scraped(fold("Hello there")).await;
    assert_eq!(choice.message.content.as_deref(), Some("Hello there"));
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"clean\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(!scrape.contains("smg_symphony_malformed"), "{scrape}");
    assert!(!logs_contain("Symphony did not read the whole output"));
}

#[tracing_test::traced_test]
#[tokio::test]
async fn a_malformed_region_in_a_stream_is_counted_and_written_once() {
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let ((result, events), scrape) = scraped(stream_events(
        vec![
            chunk(0, "Hi <tool_call>\nnot a call\n"),
            chunk(0, "</tool_call>"),
            complete(0, "stop"),
        ],
        true,
        spec,
    ))
    .await;
    result.expect("the stream ends well");
    assert!(chat_text(&events, 0, "content").contains("not a call"));
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(logs_contain(
        "Symphony did not read the whole output as content, reasoning and calls"
    ));
    assert!(logs_contain("table=\"qwen3\""));
    assert!(logs_contain("choice=0"));
}

/// An output of whitespace alone: the table holds it as the lead of the content and drops it
/// when nothing follows; the reader delivers it as one content delta before the finish chunk,
/// so the stream carries the answer the engine's own server would stream. The output is
/// recorded as `empty`, with the engine's finish and the bytes released, once.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_stream_of_whitespace_alone_carries_it_as_one_delta_and_is_recorded_empty() {
    let spec = assigned(
        chat_spec(false),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let mut responses: Vec<proto::GenerateResponse> = std::iter::repeat_n("\n", 24)
        .map(|piece| chunk(0, piece))
        .collect();
    responses.push(complete(0, "length"));
    let ((result, events), scrape) = scraped(stream_events(responses, false, spec)).await;
    result.expect("the stream ends well");
    // The opening chunk, one content delta of the whole answer, the finish chunk: the opening
    // chunk's empty content is the public API's shape, not a delta.
    let chunks: Vec<&Value> = events
        .iter()
        .filter(|event| event["choices"][0].is_object())
        .collect();
    assert_eq!(chunks.len(), 3, "{events:#?}");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(
        chunks[1]["choices"][0]["delta"]["content"],
        "\n".repeat(24),
        "{events:#?}"
    );
    assert!(chunks[1]["choices"][0]["finish_reason"].is_null());
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "length");
    assert_eq!(chat_text(&events, 0, "content"), "\n".repeat(24));
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"empty\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(logs_contain(
        "Symphony read an output with no text, reasoning or call: whitespace and framing alone, or nothing"
    ));
    assert!(logs_contain("finish=\"length\""));
    assert!(logs_contain("whitespace_bytes=24"));
    assert!(logs_contain("released_bytes=24"));
}

/// A choice whose first engine output is its `Complete`: the opening chunk, the finish chunk,
/// and the output recorded as `empty` with the finish the engine gave.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_stream_of_no_tokens_is_recorded_empty_too() {
    let spec = assigned(
        chat_spec(false),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let ((result, events), scrape) =
        scraped(stream_events(vec![complete(0, "stop")], false, spec)).await;
    result.expect("the stream ends well");
    let chunks: Vec<&Value> = events
        .iter()
        .filter(|event| event["choices"][0].is_object())
        .collect();
    assert_eq!(chunks.len(), 2, "{events:#?}");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[1]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chat_text(&events, 0, "content"), "");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"empty\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(logs_contain("finish=\"stop\""));
    assert!(logs_contain("bytes=0"));
    assert!(logs_contain("released_bytes=0"));
}

/// An output whose only answer is a call written as Python inside the block's markers, after
/// a thought: the table refuses the block, so the stream would end with the reasoning deltas and
/// a finish chunk alone. The reader releases the refused bytes as one content delta before the
/// finish chunk, which stays last; no call delta is made of them; the output is counted as
/// malformed under the reason, and its line says how many bytes went out.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_stream_whose_only_answer_is_a_python_call_in_the_tag_carries_it_as_content_before_the_finish(
) {
    let spec = assigned(chat_spec(true), glm(), "<|assistant|><think>");
    let output = "plan</think><tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    let released = "get_stock_info(company=\"Apple\")</tool_call>";
    let ((result, events), scrape) = scraped(stream_events(
        vec![
            chunk(0, &output[..12]),
            chunk(0, &output[12..40]),
            chunk(0, &output[40..]),
            complete(0, "stop"),
        ],
        true,
        spec,
    ))
    .await;
    result.expect("the stream ends well");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "plan");
    assert_eq!(chat_text(&events, 0, "content"), released);
    assert!(tool_call_deltas(&events).is_empty(), "{events:#?}");
    let chunks: Vec<&Value> = events
        .iter()
        .filter(|event| event["choices"][0].is_object())
        .collect();
    let content_at = chunks
        .iter()
        .position(|chunk| chunk["choices"][0]["delta"]["content"].is_string())
        .expect("a content delta");
    let finish_at = chunks
        .iter()
        .position(|chunk| chunk["choices"][0]["finish_reason"].is_string())
        .expect("a finish chunk");
    assert!(content_at < finish_at, "{events:#?}");
    assert_eq!(finish_at, chunks.len() - 1, "{events:#?}");
    assert_eq!(chunks[finish_at]["choices"][0]["finish_reason"], "stop");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"glm\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let reason = series_line(
        &scrape,
        "smg_symphony_malformed_total",
        &["table=\"glm\"", "reason=\"text_where_the_name_should_be\""],
    )
    .expect("the reason counter");
    assert!(reason.ends_with(" 1"), "{reason}");
    assert!(logs_contain(
        "Symphony did not read the whole output as content, reasoning and calls"
    ));
    assert!(logs_contain(&format!("released_bytes={}", released.len())));
}

/// [`fold`] under a report mode: the choice and the whole answer's `smg` object as the
/// non-streaming processor builds it, from the spec's report mode.
async fn fold_reported(output: &str, report: Report, with_symphony: bool) -> Option<ParseReport> {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let output_ids: Vec<u32> = output.chars().map(u32::from).collect();
    let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids,
        finish_reason: "stop".to_string(),
        ..Default::default()
    });
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), false);
    let mut spec = chat_spec(true);
    if with_symphony {
        spec = assigned(spec, qwen3(CallSyntax::Json), "<|im_start|>assistant\n");
    }
    spec.parse_report = report;
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    let (_, _, report) = processor
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
        .expect("a choice");
    report
}

/// [`fold`] under strictness: the choice as the non-streaming processor builds it, with the
/// report mode `report` (strictness raises `Off` to `OnFailure` in preparation; here the spec is
/// set as preparation would leave it).
async fn fold_strict(output: &str, report: Report) -> (ChatChoice, Option<ParseReport>) {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let output_ids: Vec<u32> = output.chars().map(u32::from).collect();
    let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids,
        finish_reason: "stop".to_string(),
        ..Default::default()
    });
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), false);
    let mut spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    spec.parse_report = report.at_least_on_failure();
    spec.parse_strict = true;
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    let (choice, _, report) = processor
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
        .expect("a choice");
    (choice, report)
}

#[tokio::test]
async fn under_strictness_a_folded_call_finishes_as_a_parse_failure_with_its_report() {
    let (choice, report) =
        fold_strict("Hello <tool_call>\nnot a call\n</tool_call>", Report::Off).await;
    assert_eq!(choice.finish_reason.as_deref(), Some("parse_failure"));
    // The bytes are still delivered.
    assert!(choice
        .message
        .content
        .as_deref()
        .is_some_and(|c| c.contains("not a call")));
    let report = report.expect("a report, whatever the mode asked");
    assert_eq!(report.outcome.as_deref(), Some("malformed"));
    assert_eq!(report.malformed.len(), 1);
}

#[tokio::test]
async fn under_strictness_a_clean_answer_keeps_its_finish_reason() {
    let (choice, report) = fold_strict("Hello there", Report::Off).await;
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    assert_eq!(report, None);
    let (choice, _) = fold_strict(
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {}}\n</tool_call>",
        Report::Off,
    )
    .await;
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
}

/// [`fold_strict`] with the engine's finish reason chosen.
async fn fold_strict_finishing(output: &str, finish: &str, first_call_only: bool) -> ChatChoice {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let output_ids: Vec<u32> = output.chars().map(u32::from).collect();
    let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids,
        finish_reason: finish.to_string(),
        ..Default::default()
    });
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), false);
    let mut spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    spec.parse_report = Report::OnFailure;
    spec.parse_strict = true;
    spec.first_tool_call_only = first_call_only;
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    let (choice, _, _) = processor
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
        .expect("a choice");
    choice
}

#[tokio::test]
async fn under_strictness_a_truncated_call_stays_a_truncation() {
    // The token limit cut the output inside the call region: the table gives the region up as
    // unterminated, and the finish stays `length`, which a client's truncation guard reads.
    let choice = fold_strict_finishing(
        "<tool_call>\n{\"name\": \"lookup\", \"argu",
        "length",
        false,
    )
    .await;
    assert_eq!(choice.finish_reason.as_deref(), Some("length"));
}

#[tokio::test]
async fn under_strictness_a_suppressed_call_is_not_a_parse_failure() {
    // Two calls under `parallel_tool_calls: false`: the table read both, the reader dropped the
    // second as asked; the answer is a call and finishes as one.
    let choice = fold_strict_finishing(
        concat!(
            "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {}}\n</tool_call>\n",
            "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {}}\n</tool_call>"
        ),
        "stop",
        true,
    )
    .await;
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(choice.message.tool_calls.as_ref().map(Vec::len), Some(1));
}

#[tokio::test]
async fn under_strictness_a_streamed_folded_call_finishes_as_a_parse_failure() {
    let mut spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    spec.parse_report = Report::OnFailure;
    spec.parse_strict = true;
    let (result, events) = stream_events(
        vec![
            chunk(0, "Hi <tool_call>\nnot a call\n</tool_call>"),
            complete(0, "stop"),
        ],
        true,
        spec,
    )
    .await;
    result.expect("the stream ends well");
    let finish = events
        .iter()
        .find(|e| e["choices"][0]["finish_reason"].is_string())
        .expect("a finish chunk");
    assert_eq!(finish["choices"][0]["finish_reason"], "parse_failure");
    assert_eq!(finish["smg"]["parse"]["outcome"], "malformed");
}

#[tokio::test]
async fn under_strictness_a_streamed_message_stops_with_a_parse_failure() {
    let mut spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    spec.parse_report = Report::OnFailure;
    spec.parse_strict = true;
    let (result, events) = message_events(
        vec![
            chunk(0, "Hi <tool_call>\nnot a call\n</tool_call>"),
            complete(0, "stop"),
        ],
        spec,
        None,
    )
    .await;
    result.expect("the stream ends well");
    assert_eq!(message_stop_reason(&events), "parse_failure");
    let delta = events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .expect("a message_delta");
    assert_eq!(delta["delta"]["smg"]["parse"]["outcome"], "malformed");
}

#[tokio::test]
async fn a_whole_answer_reports_its_malformed_runs_when_asked() {
    let report = fold_reported(
        "Hello <tool_call>\nnot a call\n</tool_call>",
        Report::Always,
        true,
    )
    .await
    .expect("a report");
    assert_eq!(report.parser, "symphony");
    assert_eq!(report.table.as_deref(), Some("qwen3"));
    assert_eq!(report.outcome.as_deref(), Some("malformed"));
    assert_eq!(report.malformed.len(), 1);
    let run = &report.malformed[0];
    assert_eq!(run.choice, 0);
    assert_eq!(run.at, "Hello <tool_call>".len() as u64);
    assert_eq!(run.excerpt, "\nnot a call\n");
    assert!(report.dropped.is_empty());
    // `detail` adds the wrapping the table dropped.
    let detailed = fold_reported(
        "Hello <tool_call>\nnot a call\n</tool_call>",
        Report::Detail,
        true,
    )
    .await
    .expect("a report");
    assert_eq!(detailed.malformed.len(), 1);
    assert!(
        detailed
            .dropped
            .iter()
            .any(|run| run.reason == "wrapper" && run.excerpt == "<tool_call>"),
        "{detailed:?}"
    );
}

#[tokio::test]
async fn a_clean_answer_reports_only_when_asked_and_a_legacy_one_names_its_parser() {
    assert_eq!(fold_reported("Hello there", Report::Off, true).await, None);
    assert_eq!(
        fold_reported("Hello there", Report::OnFailure, true).await,
        None
    );
    let clean = fold_reported("Hello there", Report::Always, true)
        .await
        .expect("a report");
    assert_eq!(clean.outcome.as_deref(), Some("clean"));
    assert!(clean.malformed.is_empty());
    let legacy = fold_reported("Hello there", Report::Always, false)
        .await
        .expect("a report");
    assert_eq!(legacy.parser, "legacy");
    assert_eq!(legacy.table, None);
    assert_eq!(fold_reported("Hello there", Report::Off, false).await, None);
}

#[tokio::test]
async fn a_streamed_choice_carries_the_report_on_its_finish_chunk() {
    let mut spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    spec.parse_report = Report::Always;
    let (result, events) = stream_events(
        vec![
            chunk(0, "Hi <tool_call>\nnot a call\n"),
            chunk(0, "</tool_call>"),
            complete(0, "stop"),
        ],
        true,
        spec,
    )
    .await;
    result.expect("the stream ends well");
    let with_smg: Vec<&Value> = events.iter().filter(|e| e.get("smg").is_some()).collect();
    assert_eq!(with_smg.len(), 1, "{events:?}");
    let chunk = with_smg[0];
    assert_eq!(chunk["choices"][0]["finish_reason"], "stop");
    let parse = &chunk["smg"]["parse"];
    assert_eq!(parse["parser"], "symphony");
    assert_eq!(parse["table"], "qwen3");
    assert_eq!(parse["outcome"], "malformed");
    // The run starts after "Hi " and the opening marker the table dropped: offsets count every
    // byte of the output, dropped wrapping included.
    assert_eq!(parse["malformed"][0]["at"], "Hi <tool_call>".len());
    assert_eq!(parse["malformed"][0]["choice"], 0);
    assert!(parse.get("dropped").is_none());
}

#[tokio::test]
async fn a_choice_a_table_was_assigned_but_never_read_reports_the_table_not_the_legacy_parsers() {
    // The Complete frame comes with no chunk before it (an empty output): no reader was made,
    // and the report still names the table that was assigned, clean.
    let mut spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    spec.parse_report = Report::Always;
    let (result, events) = stream_events(vec![complete(0, "stop")], true, spec).await;
    result.expect("the stream ends well");
    let with_smg: Vec<&Value> = events.iter().filter(|e| e.get("smg").is_some()).collect();
    assert_eq!(with_smg.len(), 1, "{events:?}");
    let parse = &with_smg[0]["smg"]["parse"];
    assert_eq!(parse["parser"], "symphony");
    assert_eq!(parse["table"], "qwen3");
    assert_eq!(parse["outcome"], "clean");
    // Without an assignment the legacy parsers read the stream, and the report says so.
    let mut legacy = chat_spec(true);
    legacy.parse_report = Report::Always;
    let (result, events) =
        stream_events(vec![chunk(0, "Hi"), complete(0, "stop")], true, legacy).await;
    result.expect("the stream ends well");
    let with_smg: Vec<&Value> = events.iter().filter(|e| e.get("smg").is_some()).collect();
    assert_eq!(with_smg.len(), 1, "{events:?}");
    assert_eq!(with_smg[0]["smg"]["parse"]["parser"], "legacy");
}

#[tokio::test]
async fn a_stream_without_a_report_mode_carries_no_smg_member() {
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let (result, events) = stream_events(
        vec![
            chunk(0, "Hi <tool_call>\nnot a call\n</tool_call>"),
            complete(0, "stop"),
        ],
        true,
        spec,
    )
    .await;
    result.expect("the stream ends well");
    assert!(events.iter().all(|e| e.get("smg").is_none()), "{events:?}");
}

#[tokio::test]
async fn a_streamed_message_carries_the_report_on_its_message_delta() {
    let mut spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    spec.parse_report = Report::Always;
    let (result, events) = message_events(
        vec![
            chunk(0, "Hi <tool_call>\nnot a call\n</tool_call>"),
            complete(0, "stop"),
        ],
        spec,
        None,
    )
    .await;
    result.expect("the stream ends well");
    let delta = events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .expect("a message_delta");
    let parse = &delta["delta"]["smg"]["parse"];
    assert_eq!(parse["parser"], "symphony");
    assert_eq!(parse["outcome"], "malformed");
    assert_eq!(parse["malformed"][0]["excerpt"], "\nnot a call\n");
}
