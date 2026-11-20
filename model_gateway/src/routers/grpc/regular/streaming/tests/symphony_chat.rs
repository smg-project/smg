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
    messages::CreateMessageRequest,
    profile::ProviderProfile,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto::GenerateComplete, vllm_engine::proto};
use symphony::{
    formats::{
        deepseek_v4, deepseek_v4_1, gemma4, glm, glm4, kimi_k3, llama4, plain, qwen3, Table,
    },
    grammar::GrammarStart,
    CallSyntax, Format,
};

use super::{
    super::*,
    eof::{
        chat_spec, chat_text, chunk, chunk_with_logprobs, collect_events, complete, dispatch,
        logprob_carriers, named_processor, processor, scripted_stream, CharacterTokenizer,
    },
};
use crate::{
    config::ParserChoice,
    routers::grpc::{
        proto_wrapper::ProtoGenerateComplete,
        regular::{
            processor::ResponseProcessor,
            symphony::{
                assign, assign_for, telemetry,
                template::Templates,
                testing::{tools, Refusing},
                Asks, Assignment, Forbids, Reading,
            },
        },
        utils::ModelNames,
    },
};

/// The spec with Symphony assigned `format`, replaying `prompt`.
pub(super) fn assigned(
    mut spec: ChatResponseSpec,
    format: Format,
    prompt: &str,
) -> ChatResponseSpec {
    spec.symphony = Some(Assignment {
        format: Arc::new(format),
        prompt: prompt.to_string(),
        forbids_calls: Forbids::Nothing,
        past_the_end: false,
        forcing: None,
        opener: None,
        forced_ids: None,
        json_answer: false,
        span: Span::none(),
        marked: Arc::default(),
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
    stream_events_through(
        Arc::new(CharacterTokenizer::default()),
        responses,
        with_tools,
        spec,
    )
    .await
}

/// [`stream_events`] decoding through `tokenizer`.
async fn stream_events_through(
    tokenizer: Arc<dyn Tokenizer>,
    responses: Vec<proto::GenerateResponse>,
    with_tools: bool,
    spec: ChatResponseSpec,
) -> (Result<(), utils::StreamFailure>, Vec<Value>) {
    stream_events_by(processor(with_tools), tokenizer, responses, spec).await
}

/// The events of `responses` streamed by `processor` under `spec`, decoding through `tokenizer`.
async fn stream_events_by(
    processor: StreamingProcessor,
    tokenizer: Arc<dyn Tokenizer>,
    responses: Vec<proto::GenerateResponse>,
    spec: ChatResponseSpec,
) -> (Result<(), utils::StreamFailure>, Vec<Value>) {
    let (stream, server) = scripted_stream(responses, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor
        .process_streaming_chunks(
            stream,
            dispatch(),
            tokenizer,
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

#[tokio::test]
async fn symphony_streams_every_chunks_logprobs_once() {
    // A chunk of the thought's opener alone makes no delta: its logprobs go out on a chunk with
    // an empty delta, as the engine's own server sends such a step. Otherwise a chunk's
    // logprobs ride on its first delta, a call's start included.
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    let responses = vec![
        chunk_with_logprobs(0, &QWEN3_OUTPUT[..7]),
        chunk_with_logprobs(0, &QWEN3_OUTPUT[7..25]),
        chunk_with_logprobs(0, &QWEN3_OUTPUT[25..]),
        complete(0, "stop"),
    ];
    let (result, events) = stream_events(responses, true, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "plan");
    assert_eq!(
        logprob_carriers(&events),
        [
            ("empty", QWEN3_OUTPUT[..7].to_string()),
            ("reasoning", QWEN3_OUTPUT[7..25].to_string()),
            ("tool_calls", QWEN3_OUTPUT[25..].to_string()),
        ],
        "{events:?}"
    );
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
    fold_finished(output, format, prompt, "stop").await
}

/// [`fold_with`] with the engine's finish reason chosen.
async fn fold_finished(
    output: &str,
    format: Format,
    prompt: &str,
    finish_reason: &str,
) -> (ChatChoice, u32) {
    fold_through(
        Arc::new(CharacterTokenizer::default()),
        output,
        format,
        prompt,
        finish_reason,
    )
    .await
    .expect("a choice")
}

/// [`fold_finished`] decoding through `tokenizer`, with the unary path's own verdict.
async fn fold_through(
    tokenizer: Arc<dyn Tokenizer>,
    output: &str,
    format: Format,
    prompt: &str,
    finish_reason: &str,
) -> Result<(ChatChoice, u32), String> {
    let output_ids: Vec<u32> = output.chars().map(u32::from).collect();
    let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids: output_ids.clone(),
        finish_reason: finish_reason.to_string(),
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
    // One logprob per output token: the choice keeps every one of them, as the older path does.
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
        QWEN3_OUTPUT,
        "every token keeps its logprob: the thought's, the markers' and the call's too"
    );
}

/// A DSML call the token limit cut, the shape of a tool request whose `max_tokens` is below the
/// call's length: the declared `lookup` tool, its first parameter whole, its second cut inside a
/// `string="true"` value or inside a `string="false"` object.
const CUT_CALL_PREFIX: &str = concat!(
    "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n",
    "<｜DSML｜ parameter name=\"q\" string=\"true\">rain</｜DSML｜ parameter>\n"
);

/// The two cut tails and the arguments a client reads for each: the prefix as the model wrote it.
const CUT_CALL_TAILS: [(&str, &str); 2] = [
    (
        "<｜DSML｜ parameter name=\"city\" string=\"true\">Par",
        r#"{"q": "rain", "city": "Par"#,
    ),
    (
        "<｜DSML｜ parameter name=\"options\" string=\"false\">{\"days\": 3, \"uni",
        r#"{"q": "rain", "options": {"days": 3, "uni"#,
    ),
];

/// A streamed call the token limit cut: the stream finishes `length` with the call's name and
/// `arguments`, its bytes as the model wrote them, and nothing of the call as content; the
/// arguments as streamed, for a check of their shape.
async fn streamed_cut_call_keeps_its_bytes(
    responses: Vec<proto::GenerateResponse>,
    spec: ChatResponseSpec,
    output: &str,
    arguments: &str,
) -> String {
    let (result, events) = stream_events(responses, true, spec).await;
    assert!(result.is_ok(), "{output:?}: {result:?}");
    let calls = tool_call_deltas(&events);
    assert_eq!(
        calls[0]["function"]["name"], "lookup",
        "{output:?}: {events:#?}"
    );
    let streamed: String = calls
        .iter()
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    assert_eq!(streamed, arguments, "{output:?}: {events:#?}");
    assert_eq!(
        chat_text(&events, 0, "content"),
        "",
        "nothing of the call is content: {output:?}: {events:#?}"
    );
    let finish = events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk");
    assert_eq!(finish, "length", "{output:?}");
    streamed
}

/// A whole call the token limit cut, folded over `format` after `prompt`: the choice finishes
/// `length` with the call's name and `arguments`, its bytes as the model wrote them, and nothing
/// of the call as content.
async fn whole_cut_call_keeps_its_bytes(
    output: &str,
    arguments: &str,
    format: Format,
    prompt: &str,
) {
    let (choice, _) = fold_finished(output, format, prompt, "length").await;
    let calls = choice.message.tool_calls.as_ref().expect("the call");
    assert_eq!(calls.len(), 1, "{output:?}");
    assert_eq!(calls[0].function.name, "lookup");
    assert_eq!(
        calls[0].function.arguments.as_deref(),
        Some(arguments),
        "{output:?}: {choice:?}"
    );
    assert_eq!(
        choice.message.content, None,
        "nothing of the call is content: {output:?}: {choice:?}"
    );
    assert_eq!(
        choice.finish_reason.as_deref(),
        Some("length"),
        "{output:?}"
    );
}

#[tokio::test]
async fn a_streamed_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length() {
    for (tail, arguments) in CUT_CALL_TAILS {
        let output = format!("{CUT_CALL_PREFIX}{tail}");
        // Chunked at the cut member's tag and inside its value, then the engine's `length`.
        let at = output.find(tail).expect("the tail");
        let inside = at + tail.len() - 2;
        let spec = assigned(chat_spec(true), deepseek_v4_1(), "<think>");
        let responses = vec![
            chunk(0, &output[..at]),
            chunk(0, &output[at..inside]),
            chunk(0, &output[inside..]),
            complete(0, "length"),
        ];
        streamed_cut_call_keeps_its_bytes(responses, spec, &output, arguments).await;
    }
}

#[tokio::test]
async fn a_whole_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length() {
    for (tail, arguments) in CUT_CALL_TAILS {
        let output = format!("{CUT_CALL_PREFIX}{tail}");
        whole_cut_call_keeps_its_bytes(&output, arguments, deepseek_v4_1(), "<think>").await;
    }
}

/// A DSML block the token limit cut before its invoke named a call, after a refusal inside it (a
/// parameter tag where the name's quote should have come, and the text after it): no call, so the
/// reader releases the output's own slice from the invoke's opener as the content at the end, the
/// held newline the cut found included, where a `Malformed` event with bytes at the cut ended that
/// release and left the client the newline alone. The `calls` marker before it is the table's
/// framing, dropped as always.
#[tokio::test]
async fn a_block_the_token_limit_cut_before_a_call_was_named_is_released_whole_as_content() {
    let invoke = "<｜DSML｜ invoke name=\"f<｜DSML｜ parameter name=\"s\" string=\"true\">x\n";
    let output = format!("</think>\n\n<｜DSML｜ calls>\n{invoke}");
    let (choice, _) = fold_finished(&output, deepseek_v4_1(), "<think>", "length").await;
    assert!(choice.message.tool_calls.is_none(), "{choice:?}");
    assert_eq!(
        choice.message.content.as_deref(),
        Some(invoke),
        "the slice from the invoke's opener, whole: {choice:?}"
    );
    assert_eq!(
        choice.finish_reason.as_deref(),
        Some("length"),
        "{choice:?}"
    );
}

#[tokio::test]
async fn a_refused_block_released_as_the_answer_keeps_every_tokens_logprobs() {
    // The whole answer after the thought is a call written as Python inside the call tag: the
    // reader releases the block, from its opener, as the content; the logprobs are the engine's
    // whole list, the thought's tokens and the block's alike. Two closed blocks with the
    // template's newline between them the same.
    let output = "plan</think><tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    let (choice, reasoning_tokens) = fold_with(output, glm(), "<|assistant|><think>").await;
    let released = "<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    assert_eq!(choice.message.content.as_deref(), Some(released));
    assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    assert_eq!(reasoning_tokens, 4);
    assert_eq!(kept_logprob_tokens(&choice), output);
    let output = "plan</think><tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
    let (choice, _) = fold_with(output, glm(), "<|assistant|><think>").await;
    let released = "<tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
    assert_eq!(choice.message.content.as_deref(), Some(released));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(kept_logprob_tokens(&choice), output);
    // The answer's text, then the block: the whole list all the same.
    let output = "plan</think>Let me look:<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    let (choice, _) = fold_with(output, glm(), "<|assistant|><think>").await;
    let content = "Let me look:<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    assert_eq!(choice.message.content.as_deref(), Some(content));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    assert_eq!(kept_logprob_tokens(&choice), output);
}

#[tokio::test]
async fn a_message_without_content_keeps_every_tokens_logprobs() {
    // The template's newlines between the thought and the call are content to the engine, but
    // whitespace alone is no message content; the logprobs are the engine's whole list all the
    // same, as on the older path: the thought's, the newlines' and the call's tokens.
    let output = concat!(
        "<think>plan</think>\n\n<tool_call>\n",
        "{\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>"
    );
    let (choice, _) = fold(output).await;
    assert_eq!(choice.message.content, None);
    assert_eq!(choice.message.tool_calls.as_ref().map(Vec::len), Some(1));
    assert_eq!(kept_logprob_tokens(&choice), output);
}

#[tokio::test]
async fn malformed_call_text_is_content_and_keeps_its_logprobs() {
    // A call region that never forms a call comes back malformed and reaches the client as
    // content; the logprobs are the engine's whole list, its tokens among them.
    let output = "<think>plan</think>Hello <tool_call>\nnot a call\n</tool_call>";
    let (choice, _) = fold(output).await;
    let content = choice.message.content.clone().expect("content");
    assert!(content.contains("not a call"), "{content}");
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(kept_logprob_tokens(&choice), output);
}

/// A Messages spec with Symphony assigned `format`, replaying `prompt`, with the `lookup` tool.
pub(super) fn assigned_message(
    format: Format,
    prompt: &str,
    thinking: bool,
) -> MessagesResponseSpec {
    MessagesResponseSpec {
        prompt_tokens: 0,
        max_tokens: 4096,
        context_window: None,
        first_tool_call_only: false,
        thinking_requested: thinking,
        thinking_text_returned: true,
        reasoning_in_prefill: false,
        tool_choice: None,
        has_tools: true,
        history_tool_calls_count: 0,
        chat_tools: tools().into(),
        stop_sequences: None,
        symphony: Some(Assignment {
            format: Arc::new(format),
            prompt: prompt.to_string(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: Span::none(),
            marked: Arc::default(),
        }),
        parser_names: None,
    }
}

pub(super) async fn message_events(
    responses: Vec<proto::GenerateResponse>,
    spec: MessagesResponseSpec,
    stop: Option<&str>,
) -> (Result<(), utils::StreamFailure>, Vec<Value>) {
    message_events_through(
        Arc::new(CharacterTokenizer::default()),
        responses,
        spec,
        stop,
    )
    .await
}

/// [`message_events`] decoding through `tokenizer`.
async fn message_events_through(
    tokenizer: Arc<dyn Tokenizer>,
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
            tokenizer,
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
    // The token limit: the five tokens are the whole budget.
    let mut spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    spec.max_tokens = 5;
    let (result, events) =
        message_events(vec![chunk(0, "Hello"), complete(0, "length")], spec, None).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(message_blocks(&events)[0].2, "Hello");
    assert_eq!(message_stop_reason(&events), "max_tokens");
    // The same cut short of the budget: the window was filled.
    let spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    let (_, events) =
        message_events(vec![chunk(0, "Hello"), complete(0, "length")], spec, None).await;
    assert_eq!(
        message_stop_reason(&events),
        "model_context_window_exceeded"
    );
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
/// system-role message after the user turn, rendered in place. The family's reference opens the
/// assistant turn after a system message that ends the conversation, so the prompt ends with the
/// turn opener and the open thought, and the model starts inside its thought: the thought and its
/// close are a thinking block (none when the thought is empty), never a text block, on the
/// Messages API and on chat completions alike.
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
    // The reference opens the turn after the trailing system message: the system text bare, then
    // the opener and the open thought end the prompt.
    assert!(
        prompt.ends_with("Today's date is 2026-10-16.<｜Assistant｜><think>"),
        "{prompt:?}"
    );
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
/// written for the client's own reader. The refusal is counted once under the table by its
/// reason; the output's own count stays the read's verdict, clean.
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
    let (result, scrape) = scraped(
        ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        )
        .process_non_streaming_messages_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            spec(),
            dispatch(),
            tokenizer,
            &mut decoder,
        ),
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
    // Counted once under the table by its reason; the output's own count is the read's: clean.
    let refused = series_line(
        &scrape,
        "smg_symphony_refused_total",
        &["table=\"qwen3\"", "reason=\"tool_use_input\""],
    )
    .expect("the refused count");
    assert!(refused.ends_with(" 1"), "{refused}");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"clean\""],
    )
    .expect("the clean count");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert_eq!(
        scrape.matches("smg_symphony_outputs_total{").count(),
        1,
        "{scrape}"
    );
    // The stream: the fragments as the model wrote them, and the stop reason the calls give;
    // nothing refused, nothing counted as refused.
    let ((result, events), scrape) = scraped(message_events(
        vec![chunk(0, output), complete(0, "stop")],
        spec(),
        None,
    ))
    .await;
    assert!(result.is_ok(), "{result:?}");
    let partial: String = events
        .iter()
        .filter_map(|event| event["delta"]["partial_json"].as_str())
        .collect();
    assert_eq!(partial, "{\"days\": 9.99e999999}");
    assert_eq!(message_stop_reason(&events), "tool_use");
    assert!(!scrape.contains("smg_symphony_refused_total"), "{scrape}");
}

/// A call whose arguments are a whole value of another kind than an object (a double-encoded
/// string here) reaches the client as a `tool_use` block whose `input` is `{}`, as on the older
/// path, through the real reader: nothing refused, nothing counted as refused, the output's own
/// count clean. (The unit test beside `object_inputs` pins the rule; this one pins its wiring in
/// `read_messages_with_symphony`.)
#[tokio::test]
async fn a_message_whose_tool_use_input_is_a_string_carries_an_empty_object() {
    let output =
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\": \"\\\"Paris\\\"\"}\n</tool_call>";
    let spec = || assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    let mut frame = complete(0, "stop");
    if let Some(proto::generate_response::Response::Complete(complete)) = &mut frame.response {
        complete.output_ids = output.chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
    }
    let (stream, server) = scripted_stream(vec![frame], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, true);
    let (result, scrape) = scraped(
        ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        )
        .process_non_streaming_messages_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            spec(),
            dispatch(),
            tokenizer,
            &mut decoder,
        ),
    )
    .await;
    server.abort();
    let (message, _) = match result {
        Ok(message) => message,
        Err(response) => panic!("the message was refused with {}", response.status()),
    };
    let blocks = serde_json::to_value(&message.content).expect("the blocks");
    assert_eq!(blocks.as_array().map(Vec::len), Some(1), "{blocks}");
    assert_eq!(blocks[0]["type"], "tool_use", "{blocks}");
    assert_eq!(blocks[0]["name"], "lookup", "{blocks}");
    assert_eq!(
        blocks[0]["input"],
        json!({}),
        "a string input leaves as an empty object: {blocks}"
    );
    assert!(!scrape.contains("smg_symphony_refused_total"), "{scrape}");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"clean\""],
    )
    .expect("the clean count");
    assert!(outputs.ends_with(" 1"), "{outputs}");
}

/// The scrape of a recorder of this test's own after `read` ran on this thread (the test
/// runtime is single-threaded, so the local recorder sees every record of the future), the
/// Symphony series it read being this test's own too (`telemetry::own_series`).
pub(super) async fn scraped<F: Future>(read: F) -> (F::Output, String) {
    let _own = telemetry::own_series();
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let guard = metrics::set_default_local_recorder(&recorder);
    let output = read.await;
    drop(guard);
    (output, handle.render())
}

/// The line of `scrape` for the series `name` with every label in `labels`.
pub(super) fn series_line<'a>(scrape: &'a str, name: &str, labels: &[&str]) -> Option<&'a str> {
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

/// The same output on the Messages API, whole and streamed: the `tool_use` block's `input` is the
/// value the accepted prefix's complete members make on both paths (the stream's deltas end at
/// the break and a client's partial JSON reader closes them; the whole message closes the prefix
/// the same way), nothing of the tail is text, the stop reason is `tool_use`, and the output is
/// counted as malformed once, as on the chat path, whose arguments string begins with the prefix.
#[tokio::test]
async fn a_messages_call_whose_arguments_are_not_json_carries_the_prefixes_value_whole_and_streamed(
) {
    let output =
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 2.0 * Math.PI}}\n</tool_call>";
    let spec = || assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    // The whole message.
    let mut frame = complete(0, "stop");
    if let Some(proto::generate_response::Response::Complete(complete)) = &mut frame.response {
        complete.output_ids = output.chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
    }
    let (stream, server) = scripted_stream(vec![frame], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, true);
    let (result, scrape) = scraped(
        ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        )
        .process_non_streaming_messages_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            spec(),
            dispatch(),
            tokenizer,
            &mut decoder,
        ),
    )
    .await;
    server.abort();
    let (message, _) = match result {
        Ok(message) => message,
        Err(response) => panic!("the message was refused with {}", response.status()),
    };
    let blocks = serde_json::to_value(&message.content).expect("the blocks");
    assert_eq!(blocks.as_array().map(Vec::len), Some(1), "{blocks}");
    assert_eq!(blocks[0]["type"], "tool_use", "{blocks}");
    assert_eq!(blocks[0]["name"], "lookup", "{blocks}");
    assert_eq!(
        blocks[0]["input"],
        json!({"q": 2.0}),
        "the prefix's complete members, nothing of the tail: {blocks}"
    );
    assert_eq!(
        serde_json::to_value(message.stop_reason).expect("the stop reason"),
        "tool_use"
    );
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

    // The stream, cut inside the expression: the block's deltas are the prefix a strict reader
    // accepts and nothing after the break, no text delta carries any of the tail, and a client's
    // partial JSON reader closes the prefix into the whole message's input.
    let ((result, events), scrape) = scraped(message_events(
        vec![
            chunk(
                0,
                "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": 2.0 * Ma",
            ),
            chunk(0, "th.PI}}\n</tool_call>"),
            complete(0, "stop"),
        ],
        spec(),
        None,
    ))
    .await;
    assert!(result.is_ok(), "{result:?}");
    let partial: String = events
        .iter()
        .filter_map(|event| event["delta"]["partial_json"].as_str())
        .collect();
    assert_eq!(partial, "{\"q\": 2.0 ");
    assert!(
        events
            .iter()
            .all(|event| event["delta"]["type"] != "text_delta"),
        "nothing of the tail is text: {events:?}"
    );
    let closed_by_a_client: Value =
        serde_json::from_str(&format!("{}}}", partial.trim_end())).expect("closed");
    assert_eq!(closed_by_a_client, blocks[0]["input"]);
    assert_eq!(message_stop_reason(&events), "tool_use");
    let malformed = series_line(
        &scrape,
        "smg_symphony_malformed_total",
        &["table=\"qwen3\"", "reason=\"invalid_arguments\""],
    )
    .expect("the malformed counter");
    assert!(malformed.ends_with(" 1"), "{malformed}");

    // The chat path carries the arguments as the model wrote them, a string that begins with the
    // prefix the Messages paths close.
    let (choice, _) = fold(output).await;
    let arguments = choice
        .message
        .tool_calls
        .as_ref()
        .and_then(|calls| calls.first())
        .and_then(|call| call.function.arguments.clone())
        .expect("the chat arguments");
    assert!(
        arguments.starts_with(&partial),
        "{arguments:?} vs {partial:?}"
    );
}

#[tracing_test::traced_test]
#[tokio::test]
async fn scraped_takes_its_lines_from_series_of_its_own() {
    // Forty malformed outputs read under one scrape drain the table's bucket; the output read
    // under the next scrape still writes its line, and counts none of the flow's suppressed
    // lines: the series are the scrape's own. The flow's written count is read after it, so the
    // pin does not lean on the bucket's burst size.
    let flow = 40;
    let written = |lines: &[&str]| {
        lines
            .iter()
            .filter(|line| line.contains("did not read the whole output"))
            .count()
    };
    scraped(async {
        for _ in 0..flow {
            fold("Hello <tool_call>\nnot a call\n</tool_call>").await;
        }
    })
    .await;
    let after_the_flow = std::cell::Cell::new(0);
    logs_assert(|lines: &[&str]| {
        let count = written(lines);
        after_the_flow.set(count);
        if count < flow {
            Ok(())
        } else {
            Err(format!(
                "no line of the flow was suppressed: {count} written"
            ))
        }
    });
    scraped(fold("Hello <tool_call>\nnot a call\n</tool_call>")).await;
    logs_assert(|lines: &[&str]| {
        let count = written(lines);
        let last = lines
            .iter()
            .rev()
            .find(|line| line.contains("did not read the whole output"));
        match last {
            Some(line)
                if count == after_the_flow.get() + 1 && line.contains("suppressed_lines=0") =>
            {
                Ok(())
            }
            Some(line) => Err(format!(
                "the line after the flow is missing or counts the flow's suppressed lines \
                 ({count} written after {}): {line}",
                after_the_flow.get()
            )),
            None => Err("no line at all".to_string()),
        }
    });
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

/// A stream the request ran past the model's own end (`ignore_eos`): the thought's close the
/// unconstrained tail wrote reaches the client as content like any marker out of place, the
/// output is counted clean, no malformed run is counted, and the line such a marker writes in
/// any other output stays unwritten; one line says so at debug.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_marker_in_a_stream_run_past_the_models_end_is_content_and_counts_as_nothing() {
    let mut spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    spec.symphony.as_mut().expect("assigned").past_the_end = true;
    let ((result, events), scrape) = scraped(stream_events(
        vec![
            chunk(0, "OK"),
            chunk(0, "OK</think> OK"),
            complete(0, "length"),
        ],
        true,
        spec,
    ))
    .await;
    result.expect("the stream ends well");
    assert_eq!(chat_text(&events, 0, "content"), "OKOK</think> OK");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"clean\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(
        series_line(
            &scrape,
            "smg_symphony_outputs_total",
            &["table=\"qwen3\"", "outcome=\"malformed\""]
        )
        .is_none_or(|line| line.ends_with(" 0")),
        "{scrape}"
    );
    assert!(!scrape.contains("smg_symphony_malformed"), "{scrape}");
    assert!(!logs_contain("Symphony did not read the whole output"));
    assert!(logs_contain(
        "Symphony kept markers of the table as text in an output the request ran past the model's own end"
    ));
    assert!(logs_contain("markers=1"));
    assert!(logs_contain("finish=\"length\""));
}

/// A streamed call to a function the request did not declare: the deltas carry the name as the
/// model wrote it and the arguments whole, the wire's finish is `tool_calls`, and the output is counted
/// once under `unknown_tool`, the call under its reason, with one line naming the function and
/// the table. Nothing on the wire says so.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_streamed_call_to_an_undeclared_function_goes_out_as_written_and_is_counted() {
    let output = QWEN3_OUTPUT.replace("\"lookup\"", "\"lookup_v2\"");
    let spec = assigned(
        chat_spec(true),
        qwen3(CallSyntax::Json),
        "<|im_start|>assistant\n",
    );
    // Cut inside the thought and inside the call's opening marker, as the declared call is.
    let responses = vec![
        chunk(0, &output[..9]),
        chunk(0, &output[9..32]),
        chunk(0, &output[32..]),
        complete(0, "stop"),
    ];
    let ((result, events), scrape) = scraped(stream_events(responses, true, spec)).await;
    assert!(result.is_ok(), "{result:?}");
    let calls = tool_call_deltas(&events);
    assert_eq!(calls[0]["function"]["name"], "lookup_v2", "{calls:#?}");
    assert!(
        calls[1..]
            .iter()
            .all(|call| call["function"]["name"].is_null()),
        "{calls:#?}"
    );
    let arguments: String = calls
        .iter()
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).expect("the arguments as JSON"),
        json!({"q": 1})
    );
    let finish = events
        .iter()
        .filter_map(|event| event["choices"][0]["finish_reason"].as_str())
        .next_back()
        .expect("a finish");
    assert_eq!(finish, "tool_calls");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"unknown_tool\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    let runs = series_line(
        &scrape,
        "smg_symphony_malformed_total",
        &["table=\"qwen3\"", "reason=\"unknown_tool\""],
    )
    .expect("the reason counter");
    assert!(runs.ends_with(" 1"), "{runs}");
    assert!(logs_contain(
        "Symphony read a call to a function the request did not declare; the call went out as the model named it"
    ));
    assert!(logs_contain("table=\"qwen3\""));
    assert!(logs_contain("tool=\"lookup_v2\""));
    assert!(logs_contain("undeclared_calls=1"));
    // The line carries the engine's finish; `tool_calls` is the wire's refinement of it.
    assert!(logs_contain("finish=\"stop\""));
}

/// The same call in a whole response: the message keeps it as the model named it, and the output
/// is counted the same way.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_whole_output_calling_an_undeclared_function_keeps_the_call_and_is_counted() {
    let output = QWEN3_OUTPUT.replace("\"lookup\"", "\"lookup_v2\"");
    let ((choice, _), scrape) = scraped(fold(&output)).await;
    assert_eq!(
        choice.message.content.as_deref().map(str::trim),
        Some("Hello")
    );
    let calls = choice.message.tool_calls.as_ref().expect("tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "lookup_v2");
    let arguments: Value =
        serde_json::from_str(calls[0].function.arguments.as_deref().unwrap_or(""))
            .expect("the arguments as JSON");
    assert_eq!(arguments, json!({"q": 1}));
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"unknown_tool\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(logs_contain("tool=\"lookup_v2\""));
    assert!(logs_contain("tool_calls=1"));
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
/// a finish chunk alone. The reader releases the block, from its opener, as one content delta
/// before the finish chunk, which stays last; no call delta is made of it; the output is counted
/// as malformed under the reason, and its line says how many bytes went out.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_stream_whose_only_answer_is_a_python_call_in_the_tag_carries_it_as_content_before_the_finish(
) {
    let spec = assigned(chat_spec(true), glm(), "<|assistant|><think>");
    let output = "plan</think><tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    let released = "<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
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

/// Two closed blocks after the thought, with the template's newline between them, cut by the
/// stream at the newline and inside the second opener: the one content delta before the finish
/// chunk is the output's own slice from the first opener to the last refused byte, the newline
/// and the second opener in their places; the finish chunk stays last, and the line counts the
/// whole slice.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_stream_of_two_refused_blocks_carries_the_bytes_between_them_before_the_finish() {
    let spec = assigned(chat_spec(true), glm(), "<|assistant|><think>");
    let output = "plan</think><tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
    let released = "<tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
    let ((result, events), scrape) = scraped(stream_events(
        vec![
            chunk(0, &output[..12]),
            chunk(0, &output[12..41]),
            chunk(0, &output[41..42]),
            chunk(0, &output[42..47]),
            chunk(0, &output[47..]),
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
    let content_at: Vec<usize> = chunks
        .iter()
        .enumerate()
        .filter(|(_, chunk)| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        })
        .map(|(at, _)| at)
        .collect();
    let finish_at = chunks
        .iter()
        .position(|chunk| chunk["choices"][0]["finish_reason"].is_string())
        .expect("a finish chunk");
    assert_eq!(content_at, vec![finish_at - 1], "{events:#?}");
    assert_eq!(finish_at, chunks.len() - 1, "{events:#?}");
    assert_eq!(chunks[finish_at]["choices"][0]["finish_reason"], "stop");
    let outputs = series_line(
        &scrape,
        "smg_symphony_outputs_total",
        &["table=\"glm\"", "outcome=\"malformed\""],
    )
    .expect("the outputs counter");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(logs_contain(&format!("released_bytes={}", released.len())));
}

/// The answer's text, then a call written as Python inside the block's markers (the recorded
/// shape): the text streams as it comes, and the block, the tail of the answer, goes out from its
/// opener as one content delta right before the finish chunk, which stays last; no call delta is
/// made of it. The output counts as malformed under its reason, and its line says how many bytes
/// went out.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_stream_whose_text_ends_in_a_python_call_in_the_tag_carries_the_block_as_content_before_the_finish(
) {
    let spec = assigned(chat_spec(true), glm(), "<|assistant|><think>");
    let text = "I'll place both orders for you.";
    let released = "<tool_call>whole_foods_order(location=\"downtown\")</tool_call>";
    let ((result, events), scrape) = scraped(stream_events(
        vec![
            chunk(0, "plan</think>"),
            chunk(0, "I'll place both "),
            chunk(0, "orders for you."),
            chunk(0, "<tool_call>"),
            chunk(0, "whole_foods_order(location="),
            chunk(0, "\"downtown\")</tool_call>"),
            complete(0, "stop"),
        ],
        true,
        spec,
    ))
    .await;
    result.expect("the stream ends well");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "plan");
    assert_eq!(
        chat_text(&events, 0, "content"),
        format!("{text}{released}")
    );
    assert!(tool_call_deltas(&events).is_empty(), "{events:#?}");
    let chunks: Vec<&Value> = events
        .iter()
        .filter(|event| event["choices"][0].is_object())
        .collect();
    let contents: Vec<(usize, &str)> = chunks
        .iter()
        .enumerate()
        .filter_map(|(at, chunk)| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .filter(|text| !text.is_empty())
                .map(|text| (at, text))
        })
        .collect();
    let finish_at = chunks
        .iter()
        .position(|chunk| chunk["choices"][0]["finish_reason"].is_string())
        .expect("a finish chunk");
    // The text's deltas as they came, then the block as one delta right before the finish chunk.
    let (last_at, last) = contents.last().copied().expect("a content delta");
    assert_eq!((last_at, last), (finish_at - 1, released), "{events:#?}");
    let streamed: String = contents[..contents.len() - 1]
        .iter()
        .map(|(_, text)| *text)
        .collect();
    assert_eq!(streamed, text, "{events:#?}");
    assert!(contents.len() >= 2, "{events:#?}");
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
    assert!(logs_contain(&format!("released_bytes={}", released.len())));
}

/// The scrape of an output whose decoder failed: counted `failed` for the table, once, and under
/// no other outcome.
fn counted_failed_once(scrape: &str) {
    let outputs = series_line(
        scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"failed\""],
    )
    .expect("the failed count");
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert_eq!(
        scrape.matches("smg_symphony_outputs_total{").count(),
        1,
        "{scrape}"
    );
}

/// The frames of an output whose decoder's end comes at the engine's `Complete`, and of one
/// whose end comes with the transport's, without a `Complete`.
fn ends_of(output: &str) -> [Vec<proto::GenerateResponse>; 2] {
    [
        vec![chunk(0, output), complete(0, "stop")],
        vec![chunk(0, output)],
    ]
}

#[tokio::test]
async fn a_decoders_end_that_fails_counts_the_output_as_failed_on_the_chat_paths() {
    let format = || qwen3(CallSyntax::Json);
    let prompt = "<|im_start|>assistant\n";
    // The whole response: the request's error names the decoder's end.
    let (result, scrape) = scraped(fold_through(
        Arc::new(Refusing::the_end()),
        "Hello",
        format(),
        prompt,
        "stop",
    ))
    .await;
    let error = result.expect_err("the decoder's end");
    assert!(
        error.contains("Failed to finish token decoding") && error.contains("refused"),
        "{error}"
    );
    counted_failed_once(&scrape);
    // The stream: the decoder's end at the engine's Complete, and at the transport's end.
    for frames in ends_of("Hello") {
        let ((result, _), scrape) = scraped(stream_events_through(
            Arc::new(Refusing::the_end()),
            frames,
            true,
            assigned(chat_spec(true), format(), prompt),
        ))
        .await;
        let error = result.expect_err("the decoder's end");
        assert!(format!("{error:?}").contains("refused"), "{error:?}");
        counted_failed_once(&scrape);
    }
}

#[tokio::test]
async fn a_decoders_end_that_fails_counts_the_output_as_failed_on_the_messages_paths() {
    let spec = || assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
    // The whole message: a 500 under the read failure's code.
    let mut frame = complete(0, "stop");
    if let Some(proto::generate_response::Response::Complete(complete)) = &mut frame.response {
        complete.output_ids = "Hello".chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
    }
    let (stream, server) = scripted_stream(vec![frame], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(Refusing::the_end());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, true);
    let (result, scrape) = scraped(
        ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::disabled(),
        )
        .process_non_streaming_messages_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            spec(),
            dispatch(),
            tokenizer,
            &mut decoder,
        ),
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
        body.contains("symphony_read_failed") && body.contains("Failed to finish token decoding"),
        "{body}"
    );
    counted_failed_once(&scrape);
    // The stream: the decoder's end at the engine's Complete, and at the transport's end.
    for frames in ends_of("Hello") {
        let ((result, _), scrape) = scraped(message_events_through(
            Arc::new(Refusing::the_end()),
            frames,
            spec(),
            None,
        ))
        .await;
        let error = result.expect_err("the decoder's end");
        assert!(format!("{error:?}").contains("refused"), "{error:?}");
        counted_failed_once(&scrape);
    }
}

/// The generation prompt's tail of a table whose calls are a bare Python list.
const PYTHONIC_PROMPT: &str = "<|header_start|>assistant<|header_end|>\n\n";

/// A pythonic call the token limit cut, the shape of a tool request whose `max_tokens` is below
/// the call's length: the declared `lookup` tool, cut inside its first quoted value and inside a
/// second one; the arguments a client reads for each are the prefix as the model wrote it, the
/// cut member's key and its open string included.
const PYTHONIC_CUT_CALLS: [(&str, &str); 2] = [
    ("[lookup(city=\"Par", r#"{"city": "Par"#),
    (
        "[lookup(city=\"Paris\", days=\"3",
        r#"{"city": "Paris", "days": "3"#,
    ),
];

#[tokio::test]
async fn a_streamed_pythonic_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length(
) {
    for (output, arguments) in PYTHONIC_CUT_CALLS {
        // Chunked before the cut member's key and inside its value, then the engine's `length`.
        let at = output.rfind(", ").map_or(8, |at| at + 2);
        let spec = assigned(chat_spec(true), llama4(), PYTHONIC_PROMPT);
        let responses = vec![
            chunk(0, &output[..at]),
            chunk(0, &output[at..output.len() - 2]),
            chunk(0, &output[output.len() - 2..]),
            complete(0, "length"),
        ];
        streamed_cut_call_keeps_its_bytes(responses, spec, output, arguments).await;
    }
}

#[tokio::test]
async fn a_whole_pythonic_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length()
{
    for (output, arguments) in PYTHONIC_CUT_CALLS {
        whole_cut_call_keeps_its_bytes(output, arguments, llama4(), PYTHONIC_PROMPT).await;
    }
}

/// The generation prompt's tail of the Gemma 4 table: the model's turn opened.
const GEMMA_PROMPT: &str = "<|turn>model\n";

/// A Gemma 4 call the token limit cut, the shape of a tool request whose `max_tokens` is below
/// the call's length: the declared `lookup` tool, cut inside a bare member after a string member
/// and cut inside its first string; the arguments a client reads for each are the prefix as the
/// model wrote it, the cut member's key and bytes included and no closing brace.
const GEMMA_CUT_CALLS: [(&str, &str); 2] = [
    (
        "<|tool_call>call:lookup{artist:<|\"|>Taylor<|\"|>,n:1",
        r#"{"artist": "Taylor", "n": 1"#,
    ),
    (
        "<|tool_call>call:lookup{artist:<|\"|>Tay",
        r#"{"artist": "Tay"#,
    ),
];

#[tokio::test]
async fn a_streamed_gemma_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length()
{
    for (output, arguments) in GEMMA_CUT_CALLS {
        // Chunked before the cut member and inside it, then the engine's `length`.
        let spec = assigned(chat_spec(true), gemma4(), GEMMA_PROMPT);
        let responses = vec![
            chunk(0, &output[..output.len() - 4]),
            chunk(0, &output[output.len() - 4..output.len() - 1]),
            chunk(0, &output[output.len() - 1..]),
            complete(0, "length"),
        ];
        let streamed = streamed_cut_call_keeps_its_bytes(responses, spec, output, arguments).await;
        assert!(
            serde_json::from_str::<Value>(&streamed).is_err(),
            "the arguments stay cut: {streamed:?}"
        );
    }
}

#[tokio::test]
async fn a_whole_gemma_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length() {
    for (output, arguments) in GEMMA_CUT_CALLS {
        whole_cut_call_keeps_its_bytes(output, arguments, gemma4(), GEMMA_PROMPT).await;
    }
}

/// The generation prompt's tail of the GLM 4.5 and 4.6 table: the turn opened.
const KEYED_PROMPT: &str = "<|assistant|>";

/// A keyed call the token limit cut, the shape of a tool request whose `max_tokens` is below the
/// call's length: the declared `lookup` tool, its first value whole, then cut inside an object
/// value and inside an integer, both held for their close; the arguments a client reads for each
/// are the prefix as the model wrote it, the cut member's key and bytes included.
const KEYED_CUT_CALLS: [(&str, &str); 2] = [
    (
        "\n<think></think>\n<tool_call>lookup\n<arg_key>q</arg_key>\n<arg_value>rain</arg_value>\n\
         <arg_key>options</arg_key>\n<arg_value>{\"days\": 3, \"uni",
        r#"{"q": "rain", "options": {"days": 3, "uni"#,
    ),
    (
        "\n<think></think>\n<tool_call>lookup\n<arg_key>q</arg_key>\n<arg_value>rain</arg_value>\n\
         <arg_key>n</arg_key>\n<arg_value>12",
        r#"{"q": "rain", "n": 12"#,
    ),
];

#[tokio::test]
async fn a_streamed_keyed_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length()
{
    for (output, arguments) in KEYED_CUT_CALLS {
        // Chunked before the cut member's key tag and inside its value, then the engine's `length`.
        let at = output.rfind("<arg_key>").expect("the cut member's tag");
        let spec = assigned(chat_spec(true), glm4(), KEYED_PROMPT);
        let responses = vec![
            chunk(0, &output[..at]),
            chunk(0, &output[at..output.len() - 2]),
            chunk(0, &output[output.len() - 2..]),
            complete(0, "length"),
        ];
        streamed_cut_call_keeps_its_bytes(responses, spec, output, arguments).await;
    }
}

#[tokio::test]
async fn a_whole_keyed_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length() {
    for (output, arguments) in KEYED_CUT_CALLS {
        whole_cut_call_keeps_its_bytes(output, arguments, glm4(), KEYED_PROMPT).await;
    }
}

/// The Kimi K3 generation prompt's tail with thinking on: the assistant's message opened and its
/// thought.
const KIMI_K3_THINKING_PROMPT: &str =
    "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";

/// A Kimi K3 reply written straight into the prefilled thought and closed by the answer's closer
/// and the end of the turn, never by the thought's closer (seen on about 1 % of very long turns),
/// one marker a token as the engine streams them.
const KIMI_K3_SKIPPED_THOUGHT: [&str; 8] = [
    "The answer",
    " is 42.",
    "<|close|>",
    "response",
    "<|sep|>",
    "<|close|>",
    "message",
    "<|sep|>",
];

/// Every text a client reads from `events`: a chat delta's content and reasoning, a Messages
/// delta's text and thinking.
fn client_text(events: &[Value]) -> String {
    events
        .iter()
        .flat_map(|event| {
            let delta = if event["choices"].is_array() {
                &event["choices"][0]["delta"]
            } else {
                &event["delta"]
            };
            ["content", "reasoning_content", "text", "thinking"]
                .into_iter()
                .filter_map(move |field| delta[field].as_str())
        })
        .collect()
}

#[tokio::test]
async fn a_kimi_k3_thought_closed_by_the_answers_closer_reaches_the_client_once_without_the_marker()
{
    let output = KIMI_K3_SKIPPED_THOUGHT.concat();
    let frames = || -> Vec<proto::GenerateResponse> {
        let mut frames: Vec<_> = KIMI_K3_SKIPPED_THOUGHT
            .iter()
            .map(|token| chunk(0, token))
            .collect();
        frames.push(complete(0, "stop"));
        frames
    };

    // The chat stream: the reply once, as the thought it was streamed as; no content, and the
    // closer never in the text.
    let spec = assigned(chat_spec(false), kimi_k3(), KIMI_K3_THINKING_PROMPT);
    let (result, events) = stream_events(frames(), false, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        chat_text(&events, 0, "reasoning_content"),
        "The answer is 42.",
        "{events:#?}"
    );
    assert_eq!(chat_text(&events, 0, "content"), "", "{events:#?}");
    assert!(
        !client_text(&events).contains("<|close|>"),
        "a marker reached the client: {events:#?}"
    );

    // The whole chat message: the same reading.
    let (choice, _) = fold_with(&output, kimi_k3(), KIMI_K3_THINKING_PROMPT).await;
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("The answer is 42."),
        "{choice:?}"
    );
    assert_eq!(choice.message.content, None, "{choice:?}");
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"), "{choice:?}");

    // The Messages stream: one thinking block with the reply, no text block, no marker.
    let spec = || assigned_message(kimi_k3(), KIMI_K3_THINKING_PROMPT, true);
    let (result, events) = message_events(frames(), spec(), None).await;
    assert!(result.is_ok(), "{result:?}");
    let blocks = message_blocks(&events);
    assert_eq!(blocks.len(), 1, "{blocks:?}");
    assert_eq!(blocks[0].0, "thinking", "{blocks:?}");
    assert_eq!(blocks[0].2, "The answer is 42.", "{blocks:?}");
    assert_eq!(message_stop_reason(&events), "end_turn");
    assert!(
        !client_text(&events).contains("<|close|>"),
        "a marker reached the client: {events:#?}"
    );

    // The whole Messages answer: the thinking block and nothing else.
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
        context::ExecutionResult::Single {
            stream,
            dispatch_start: Instant::now(),
        },
        spec(),
        dispatch(),
        tokenizer,
        &mut decoder,
    )
    .await;
    server.abort();
    let (message, _) = match result {
        Ok(message) => message,
        Err(response) => panic!("the message was refused with {}", response.status()),
    };
    let blocks = serde_json::to_value(&message.content).expect("the blocks");
    assert_eq!(blocks.as_array().map(Vec::len), Some(1), "{blocks}");
    assert_eq!(blocks[0]["type"], "thinking", "{blocks}");
    assert_eq!(blocks[0]["thinking"], "The answer is 42.", "{blocks}");
    assert_eq!(
        serde_json::to_value(message.stop_reason).expect("the stop reason"),
        "end_turn"
    );
}

/// A chat request to the Qwen3 model that offers `lookup` under `tool_choice` and separates the
/// reasoning.
fn forced_request(tool_choice: &Value) -> ChatCompletionRequest {
    forced_request_keeping(tool_choice, true)
}

/// [`forced_request`] with the reasoning separated, or kept in the content.
fn forced_request_keeping(tool_choice: &Value, separate_reasoning: bool) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "Qwen/Qwen3-8B", "messages": [], "stream": true,
        "separate_reasoning": separate_reasoning, "tool_choice": tool_choice,
        "tools": [{"type": "function", "function": {"name": "lookup",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    }))
    .expect("a chat request")
}

/// The spec of [`forced_request`] as the gateway's JSON schema makes `table` read the response
/// after `prompt`, for workers that do not say where they start a grammar.
fn forced_spec(table: Table, tool_choice: &Value, prompt: &str) -> ChatResponseSpec {
    spec_under_the_schema(
        table,
        &forced_request(tool_choice),
        prompt,
        GrammarStart::Unknown,
    )
}

/// The spec of `request` as the gateway's JSON schema makes `table` read the response after
/// `prompt`, for a model whose workers start a grammar at `start` together. The schema is the
/// older registry's over the tools the request's choice calls, as request building passes it.
fn spec_under_the_schema(
    table: Table,
    request: &ChatCompletionRequest,
    prompt: &str,
    start: GrammarStart,
) -> ChatResponseSpec {
    let declared = request.tools.as_deref().unwrap_or_default();
    let choice = request.tool_choice.as_ref().expect("a choice");
    let callable = choice
        .narrow_tools(declared)
        .unwrap_or_else(|| declared.to_vec());
    let schema = ToolParserFactory::new()
        .registry()
        .generate_tool_constraint(None, &callable, choice, false)
        .expect("a constraint")
        .expect("the older registry's schema")
        .to_tuple();
    assert_eq!(schema.0, "json_schema", "{} reads", table.name());
    let mut spec = ChatResponseSpec::from(request);
    spec.symphony = assign_for(
        Reading::Table(table.shared(), false),
        Asks::Chat(request),
        Some(&schema),
        prompt,
        ProviderProfile::for_model(&request.model),
        || start,
    );
    assert!(spec.symphony.is_some(), "{} reads", table.name());
    spec
}

/// `output` read whole under `spec` and ended with the engine's `finish_reason`: the choice.
async fn forced_whole(spec: &ChatResponseSpec, output: &str, finish_reason: &str) -> ChatChoice {
    whole_choice(spec, output, finish_reason, None).await
}

/// [`forced_whole`]; with `reasoning`, a spec without a table is read as the older arm reads it
/// behind the reasoning parser of that name, the `json` parser named for the calls.
async fn whole_choice(
    spec: &ChatResponseSpec,
    output: &str,
    finish_reason: &str,
    reasoning: Option<&str>,
) -> ChatChoice {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let whole = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids: output.chars().map(u32::from).collect(),
        finish_reason: finish_reason.to_string(),
        ..Default::default()
    });
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), false);
    let (choice, _) = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    )
    .process_single_choice(
        "test-request",
        &whole,
        0,
        spec,
        "Qwen/Qwen3-8B",
        &tokenizer,
        &mut decoder,
        0,
        reasoning.is_some(),
        reasoning.is_some(),
        reasoning,
        reasoning.map(|_| "json"),
    )
    .await
    .expect("a choice");
    choice
}

/// What `output` says under `spec`, streamed one character at a time and collected. A spec with
/// no table is read by the older arm, behind the reasoning parser `older` names.
async fn said_both_ways(
    spec: &ChatResponseSpec,
    output: &str,
    older: Option<&str>,
) -> (Said, Said) {
    let mut frames: Vec<_> = output.chars().map(|ch| chunk(0, &ch.to_string())).collect();
    frames.push(complete(0, "stop"));
    let (result, events) = stream_events_by(
        named_processor("json", older.unwrap_or("qwen3")),
        Arc::new(CharacterTokenizer::default()),
        frames,
        spec.clone(),
    )
    .await;
    assert!(result.is_ok(), "{output}: {result:?}");
    let whole = whole_choice(spec, output, "stop", older).await;
    (stream_said(&events), whole_said(&whole))
}

/// What a reader said of choice 0: the calls by name with their arguments as written, the
/// reasoning, the content and the finish reason.
type Said = (Vec<(String, String)>, String, String, String);

/// What a stream said of choice 0.
fn stream_said(events: &[Value]) -> Said {
    (
        streamed_calls(events),
        chat_text(events, 0, "reasoning_content"),
        chat_text(events, 0, "content"),
        streamed_finish(events).to_string(),
    )
}

/// What a collected choice said.
fn whole_said(choice: &ChatChoice) -> Said {
    let calls = choice
        .message
        .tool_calls
        .iter()
        .flatten()
        .map(|call| {
            (
                call.function.name.clone(),
                call.function.arguments.clone().unwrap_or_default(),
            )
        })
        .collect();
    (
        calls,
        choice.message.reasoning_content.clone().unwrap_or_default(),
        choice.message.content.clone().unwrap_or_default(),
        choice.finish_reason.clone().unwrap_or_default(),
    )
}

/// What a stream said of choice 0's calls: each call's name and its arguments as the deltas
/// spell them, by the call's index.
fn streamed_calls(events: &[Value]) -> Vec<(String, String)> {
    let mut calls: Vec<(String, String)> = Vec::new();
    for delta in tool_call_deltas(events) {
        let index = delta["index"].as_u64().expect("a call's index") as usize;
        if calls.len() <= index {
            calls.resize(index + 1, (String::new(), String::new()));
        }
        let function = &delta["function"];
        calls[index]
            .0
            .push_str(function["name"].as_str().unwrap_or_default());
        calls[index]
            .1
            .push_str(function["arguments"].as_str().unwrap_or_default());
    }
    calls
}

/// The finish reason of choice 0's stream.
fn streamed_finish(events: &[Value]) -> &str {
    events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk")
}

/// A call the gateway's JSON schema forces, on a prompt that opened the thought, streamed one
/// character at a time and then collected: the thought is reasoning, and the JSON after its
/// close is the call, the arguments alone for the function the request names and the list for a
/// required call. The model's table reads both with its forced call; no legacy parser reads.
#[tokio::test]
async fn symphony_reads_the_thought_and_then_the_call_a_json_schema_forces() {
    let schema = ("json_schema".to_string(), "{}".to_string());
    for (tool_choice, output) in [
        (
            json!("required"),
            r#"plan</think>[{"name":"lookup","parameters":{"q":"Paris"}}]"#,
        ),
        (
            json!({"type": "function", "function": {"name": "lookup"}}),
            r#"plan</think>{"q":"Paris"}"#,
        ),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen3-8B", "messages": [], "stream": true,
            "separate_reasoning": true, "tool_choice": tool_choice,
            "tools": [{"type": "function", "function": {"name": "lookup",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        }))
        .expect("a chat request");
        let spec = || {
            let mut spec = ChatResponseSpec::from(&request);
            spec.symphony = assign(
                ParserChoice::Symphony,
                &ModelNames::default(),
                Asks::Chat(&request),
                Some(&schema),
                "<|im_start|>assistant\n<think>\n",
                &Templates::new(Arc::new(TokenizerRegistry::new())),
            )
            .expect("no template named");
            assert_eq!(
                spec.symphony.as_ref().map(Assignment::table),
                Some("qwen3"),
                "the model's table reads the forced call"
            );
            spec
        };
        let mut frames: Vec<_> = output.chars().map(|ch| chunk(0, &ch.to_string())).collect();
        frames.push(complete(0, "stop"));
        let (result, events) = stream_events(frames, true, spec()).await;
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(chat_text(&events, 0, "reasoning_content"), "plan");
        assert_eq!(chat_text(&events, 0, "content"), "");
        let calls = tool_call_deltas(&events);
        assert!(
            calls
                .iter()
                .any(|call| call["function"]["name"] == "lookup"),
            "{events:#?}"
        );
        let arguments: String = calls
            .iter()
            .filter_map(|call| call["function"]["arguments"].as_str())
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&arguments).expect("the arguments as JSON"),
            json!({"q": "Paris"})
        );
        assert!(
            events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == "tool_calls"),
            "{events:#?}"
        );

        // The same output collected: the message holds the thought and the one call.
        let choice = forced_whole(&spec(), output, "stop").await;
        assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
        assert_eq!(choice.message.content.as_deref().unwrap_or_default(), "");
        let calls = choice.message.tool_calls.as_ref().expect("tool calls");
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(
            serde_json::from_str::<Value>(
                calls[0].function.arguments.as_deref().expect("arguments")
            )
            .expect("the arguments as JSON"),
            json!({"q": "Paris"})
        );
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }
}

/// A forced call the token limit cut keeps the bytes that arrived, and its stream ends with the
/// engine's `length`, never `tool_calls`. What a required call's list still held at the cut is
/// no content: behind a whole call the comma after an item and the next item cut before its
/// arguments began, behind an item's whole arguments the rest of its object (the whitespace an
/// indented list writes before the closing brace, a name written after the parameters). A list
/// cut inside its first item made no call: the reader releases the output's own bytes from the
/// bracket, as it releases any call block the table refused in an answer that made no call. A
/// required call's item without parameters is a call with an empty object. The stream and the
/// collected message agree on every call byte for byte.
#[tokio::test]
async fn a_forced_call_cut_short_or_without_parameters_reads_the_same_streamed_and_whole() {
    let named = json!({"type": "function", "function": {"name": "lookup"}});
    let required = json!("required");
    for (tool_choice, output, engine_finish, arguments, content, finish) in [
        // The limit cut the arguments of the function the request names.
        (
            &named,
            r#"plan</think>{"q":"Par"#,
            "length",
            vec![r#"{"q":"Par"#],
            "",
            "length",
        ),
        // The limit cut a required call's list inside its second item.
        (
            &required,
            r#"plan</think>[{"name":"lookup","parameters":{"q":"a"}},{"name":"lookup","parameters":{"q":"Par"#,
            "length",
            vec![r#"{"q":"a"}"#, r#"{"q":"Par"#],
            "",
            "length",
        ),
        // The limit cut a required call's list between two items: the comma is no content.
        (
            &required,
            r#"plan</think>[{"name":"lookup","parameters":{"q":"a"}},"#,
            "length",
            vec![r#"{"q":"a"}"#],
            "",
            "length",
        ),
        // The limit cut the list's second item before its arguments began.
        (
            &required,
            r#"plan</think>[{"name":"lookup","parameters":{"q":"a"}}, {"name":"loo"#,
            "length",
            vec![r#"{"q":"a"}"#],
            "",
            "length",
        ),
        // The limit cut an indented list behind an item's whole arguments, before the item's
        // closing brace: the newline and the indent are no content, streamed or whole.
        (
            &required,
            "plan</think>[\n  {\"name\":\"lookup\",\"parameters\":{\"q\":\"a\"}\n  ",
            "length",
            vec![r#"{"q":"a"}"#],
            "",
            "length",
        ),
        // The same cut behind a name written after the parameters.
        (
            &required,
            r#"plan</think>[{"parameters":{"q":"a"},"name":"lookup""#,
            "length",
            vec![r#"{"q":"a"}"#],
            "",
            "length",
        ),
        // The limit cut the list's first item before its arguments began: no call was made,
        // so the reader releases the output's own bytes from the bracket.
        (
            &required,
            r#"plan</think>[{"name":"loo"#,
            "length",
            vec![],
            r#"[{"name":"loo"#,
            "length",
        ),
        // A required call's item without parameters.
        (
            &required,
            r#"plan</think>[{"name":"lookup"}]"#,
            "stop",
            vec!["{}"],
            "",
            "tool_calls",
        ),
    ] {
        let spec = forced_spec(
            Table::Qwen3,
            tool_choice,
            "<|im_start|>assistant\n<think>\n",
        );
        let expected: Vec<(String, String)> = arguments
            .iter()
            .map(|arguments| ("lookup".to_string(), (*arguments).to_string()))
            .collect();
        let mut frames: Vec<_> = output.chars().map(|ch| chunk(0, &ch.to_string())).collect();
        frames.push(complete(0, engine_finish));
        let (result, events) = stream_events(frames, true, spec.clone()).await;
        assert!(result.is_ok(), "{output}: {result:?}");
        assert_eq!(streamed_calls(&events), expected, "{output}: {events:#?}");
        assert_eq!(
            chat_text(&events, 0, "reasoning_content"),
            "plan",
            "{output}"
        );
        assert_eq!(chat_text(&events, 0, "content"), content, "{output}");
        assert_eq!(streamed_finish(&events), finish, "{output}");

        let choice = forced_whole(&spec, output, engine_finish).await;
        let whole: Vec<(String, String)> = choice
            .message
            .tool_calls
            .iter()
            .flatten()
            .map(|call| {
                (
                    call.function.name.clone(),
                    call.function.arguments.clone().unwrap_or_default(),
                )
            })
            .collect();
        assert_eq!(whole, expected, "{output}: {choice:?}");
        assert_eq!(
            choice.message.reasoning_content.as_deref(),
            Some("plan"),
            "{output}"
        );
        assert_eq!(
            choice.message.content.as_deref().unwrap_or_default(),
            content,
            "{output}"
        );
        assert_eq!(choice.finish_reason.as_deref(), Some(finish), "{output}");
    }
}

/// The stream of a call the JSON schema forces reads through a table as through the older arm:
/// the same calls by name and by JSON value, no content and the same finish reason, at every
/// split of the output into two frames, for the function the request names and for a required
/// call's list. A list of two calls arrives whole through a table however the frames fall. The
/// older arm reads the end of one call per frame, so it stands as the reference only where the
/// first frame holds the whole first call; elsewhere it loses the second call, which the table
/// keeps. Five tables stand for the registry here: `plain`, which reads no thought, and four
/// that read one (`qwen3`, `inkling`, `kimi_k3`, `deepseek_v4`). The reader's own tests read
/// every table at every cut against the older arm's whole answer
/// (`every_table_reads_the_json_a_schema_forces_as_the_older_arm_does`).
#[tokio::test]
async fn a_forced_calls_stream_reads_through_a_table_as_through_the_older_arm_at_every_split() {
    /// A stream's calls by name and JSON value, its content and its finish reason.
    fn said(events: &[Value]) -> (Vec<(String, Value)>, String, String) {
        let calls = streamed_calls(events)
            .into_iter()
            .map(|(name, arguments)| {
                let arguments = serde_json::from_str(&arguments).expect("the arguments as JSON");
                (name, arguments)
            })
            .collect();
        (
            calls,
            chat_text(events, 0, "content"),
            streamed_finish(events).to_string(),
        )
    }
    let named = json!({"type": "function", "function": {"name": "lookup"}});
    let required = json!("required");
    let two_calls =
        r#"[{"name":"lookup","parameters":{"q":1}},{"name":"lookup","parameters":{"q":"b"}}]"#;
    // Where the list's first call ends: the older arm reads both calls from this split on.
    let first_call = two_calls.find("},{").expect("two calls") + 1;
    for (tool_choice, output, older_from, calls) in [
        (
            &required,
            r#"[{"name":"lookup","parameters":{"q":1}}]"#,
            1,
            vec![json!({"q": 1})],
        ),
        (
            &required,
            two_calls,
            first_call,
            vec![json!({"q": 1}), json!({"q": "b"})],
        ),
        (
            &named,
            r#"{"q":"P","n":[1,{"d":null}]}"#,
            1,
            vec![json!({"q": "P", "n": [1, {"d": null}]})],
        ),
    ] {
        let expected = (
            calls
                .into_iter()
                .map(|arguments| ("lookup".to_string(), arguments))
                .collect::<Vec<_>>(),
            String::new(),
            "tool_calls".to_string(),
        );
        for split in output.char_indices().map(|(at, _)| at).skip(1) {
            let frames = || {
                vec![
                    chunk(0, &output[..split]),
                    chunk(0, &output[split..]),
                    complete(0, "stop"),
                ]
            };
            if split >= older_from {
                // The older arm: the same request with no table assigned.
                let older = ChatResponseSpec::from(&forced_request(tool_choice));
                let (result, events) = stream_events(frames(), true, older).await;
                assert!(result.is_ok(), "split {split}: {result:?}");
                assert_eq!(
                    said(&events),
                    expected,
                    "the older arm, split {split}: {events:#?}"
                );
            }
            for table in [
                Table::Plain,
                Table::Qwen3,
                Table::Inkling,
                Table::KimiK3,
                Table::DeepSeekV4,
            ] {
                let spec = forced_spec(table, tool_choice, "");
                let (result, events) = stream_events(frames(), true, spec).await;
                let case = format!("{} split {split}", table.name());
                assert!(result.is_ok(), "{case}: {result:?}");
                assert_eq!(said(&events), expected, "{case}: {events:#?}");
            }
        }
    }
}

/// A Messages request whose `tool_choice` names a tool takes the same JSON schema, and the
/// model's table reads its response the same way: the thought is a thinking block and the JSON
/// after its close is the tool's input, with the stop reason of a tool use.
#[tokio::test]
async fn a_message_reads_the_thought_and_then_the_tool_a_json_schema_forces() {
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "Qwen/Qwen3-8B", "max_tokens": 64, "messages": [],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tool_choice": {"type": "tool", "name": "lookup"},
        "tools": [{"name": "lookup", "input_schema": {
            "type": "object", "properties": {"q": {"type": "string"}}}}]
    }))
    .expect("a Messages request");
    let schema = ("json_schema".to_string(), "{}".to_string());
    let mut spec = MessagesResponseSpec::from(&request);
    spec.symphony = assign_for(
        Reading::Table(Table::Qwen3.shared(), false),
        Asks::for_messages(&request, false),
        Some(&schema),
        "<|im_start|>assistant\n<think>\n",
        ProviderProfile::for_model(&request.model),
        || GrammarStart::Unknown,
    );
    assert_eq!(
        spec.symphony
            .as_ref()
            .and_then(|assignment| assignment.forcing.as_deref()),
        Some("lookup"),
        "the table reads the forced tool"
    );
    let output = r#"plan</think>{"q":"Paris"}"#;
    let mut frames: Vec<_> = output.chars().map(|ch| chunk(0, &ch.to_string())).collect();
    frames.push(complete(0, "stop"));
    let (result, events) = message_events(frames, spec, None).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        message_blocks(&events),
        vec![
            ("thinking".to_string(), String::new(), json!("plan")),
            (
                "tool_use".to_string(),
                "lookup".to_string(),
                json!({"q": "Paris"})
            ),
        ]
    );
    assert_eq!(message_stop_reason(&events), "tool_use");
}

/// A message whose `tool_choice` asks for any tool reads the list the JSON schema forces, and
/// what that list still held when the token limit cut it (the comma after a tool, the next
/// tool cut before its input began, a tool's name written after its input) opens no text block
/// beside the tool's: the message is the thought and the tool, and its stop reason the limit's.
#[tokio::test]
async fn a_message_whose_required_tools_the_token_limit_cut_has_no_text_block() {
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "Qwen/Qwen3-8B", "max_tokens": 64, "messages": [],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tool_choice": {"type": "any"},
        "tools": [{"name": "lookup", "input_schema": {
            "type": "object", "properties": {"q": {"type": "string"}}}}]
    }))
    .expect("a Messages request");
    let schema = ("json_schema".to_string(), "{}".to_string());
    for output in [
        r#"plan</think>[{"name":"lookup","parameters":{"q":"Paris"}},"#,
        r#"plan</think>[{"name":"lookup","parameters":{"q":"Paris"}}, {"name":"loo"#,
        r#"plan</think>[{"parameters":{"q":"Paris"},"name":"lookup""#,
    ] {
        let mut spec = MessagesResponseSpec::from(&request);
        spec.symphony = assign_for(
            Reading::Table(Table::Qwen3.shared(), false),
            Asks::for_messages(&request, false),
            Some(&schema),
            "<|im_start|>assistant\n<think>\n",
            ProviderProfile::for_model(&request.model),
            || GrammarStart::Unknown,
        );
        assert_eq!(
            spec.symphony.as_ref().map(Assignment::table),
            Some("qwen3"),
            "the model's table reads the required tools"
        );
        let mut frames: Vec<_> = output.chars().map(|ch| chunk(0, &ch.to_string())).collect();
        frames.push(complete(0, "length"));
        let (result, events) = message_events(frames, spec, None).await;
        assert!(result.is_ok(), "{output}: {result:?}");
        assert_eq!(
            message_blocks(&events),
            vec![
                ("thinking".to_string(), String::new(), json!("plan")),
                (
                    "tool_use".to_string(),
                    "lookup".to_string(),
                    json!({"q": "Paris"})
                ),
            ],
            "{output}"
        );
        // Which of the limit's two reasons it is reads the budget against the window: the
        // truncation cases' subject.
        let stop = message_stop_reason(&events);
        assert!(
            stop == "max_tokens" || stop == "model_context_window_exceeded",
            "{output}: {stop}"
        );
    }
}

/// Two calls a required choice forces arrive when one frame ends the first call and holds the
/// second: the frame shape at which the older arm's stream delivered the first call alone. The
/// reader is the one the gateway's own assignment gives the request, so the case is red where
/// the assignment leaves the response to the older arm.
#[tokio::test]
async fn a_frame_that_ends_one_forced_call_and_holds_the_next_delivers_both() {
    let request = forced_request(&json!("required"));
    let schema = ("json_schema".to_string(), "{}".to_string());
    let mut spec = ChatResponseSpec::from(&request);
    spec.symphony = assign_for(
        Reading::Table(Table::Qwen3.shared(), false),
        Asks::Chat(&request),
        Some(&schema),
        "",
        ProviderProfile::for_model(&request.model),
        || GrammarStart::Unknown,
    );
    let frames = vec![
        chunk(0, r#"[{"name":"lookup","parameters":{"q":"a"#),
        chunk(0, r#""}},{"name":"lookup","parameters":{"q":"b"}}]"#),
        complete(0, "stop"),
    ];
    let (result, events) = stream_events(frames, true, spec).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(
        streamed_calls(&events),
        [
            ("lookup".to_string(), r#"{"q":"a"}"#.to_string()),
            ("lookup".to_string(), r#"{"q":"b"}"#.to_string()),
        ],
        "{events:#?}"
    );
    assert_eq!(chat_text(&events, 0, "content"), "");
    assert_eq!(streamed_finish(&events), "tool_calls");
}

/// One reading: the calls by their arguments as written (each a call of `lookup`), the reasoning,
/// the content and the finish reason.
fn said(calls: &[&str], reasoning: &str, content: &str, finish: &str) -> Said {
    (
        calls
            .iter()
            .map(|arguments| ("lookup".to_string(), (*arguments).to_string()))
            .collect(),
        reasoning.to_string(),
        content.to_string(),
        finish.to_string(),
    )
}

/// A forced call under the gateway's JSON schema after a prompt that opened the thought, cell by
/// cell: where the model's workers start a grammar (at the first token, once the thought is
/// over, without saying, at both places), the reasoning separated or kept in the content, the
/// JSON alone or after the thought and its close, the function named or a call required, each
/// streamed one character at a time and collected. The reader starts where the grammar starts:
///
/// - at the first token the schema holds the output before the model writes anything, so the
///   reader starts in the call and the JSON alone is the call;
/// - once the thought is over the thought comes first, so the prompt's replay stands: the
///   thought is reasoning, or content where the request keeps it there, and the JSON after its
///   close is the call;
/// - workers that do not say, or that start at both places, leave the shape to the output, and
///   its first bytes say which: the JSON's opener and then a byte the JSON writes after it make
///   the output the call from the opener, and any other output is read where the prompt's
///   replay leaves the reader.
///
/// An output the workers' start cannot produce reads as that start reads it: the thought first
/// where the reader starts in the call is the call's bytes or, for a list, content, and the JSON
/// alone where the replay stands is the thought's text. Workers that do not say produce either
/// shape, and both read as the call with the thought where the request asks for it. The older
/// arm reads the same whatever the workers say, here behind the `qwen3` reasoning parser; its
/// stream and its collected answer disagree where the thought is kept in the content and
/// written first.
#[tokio::test]
async fn a_forced_calls_reader_starts_where_the_workers_start_a_grammar() {
    const OPENED: &str = "<|im_start|>assistant\n<think>\n";
    let named = json!({"type": "function", "function": {"name": "lookup"}});
    let required = json!("required");
    let arguments = r#"{"q":"Paris"}"#;
    let list = r#"[{"name":"lookup","parameters":{"q":"Paris"}}]"#;
    let call = said(&[arguments], "", "", "tool_calls");
    for (tool_choice, json_alone) in [(&named, arguments), (&required, list)] {
        let by_name = tool_choice.is_object();
        let thought_first = format!("plan</think>{json_alone}");
        let thought_first = thought_first.as_str();
        for separated in [true, false] {
            let request = forced_request_keeping(tool_choice, separated);
            for output in [json_alone, thought_first] {
                let alone = output == json_alone;
                let case = format!("separated {separated}, {output}");
                // The older arm: no table assigned, the open thought recorded on the spec.
                let mut older = ChatResponseSpec::from(&request);
                older.reasoning_in_prefill = true;
                let expected = match (separated, alone) {
                    (true, true) => {
                        let thought = said(&[], json_alone, "", "stop");
                        (thought.clone(), thought)
                    }
                    (true, false) => {
                        let read = said(&[arguments], "plan", "", "tool_calls");
                        (read.clone(), read)
                    }
                    (false, true) => (call.clone(), call.clone()),
                    // Its stream reads the thought into a named call's arguments and as content
                    // ahead of a required call; its collected answer is content and no call.
                    (false, false) => (
                        if by_name {
                            said(&[thought_first], "", "", "tool_calls")
                        } else {
                            said(&[arguments], "", "plan</think>", "tool_calls")
                        },
                        said(&[], "", thought_first, "stop"),
                    ),
                };
                assert_eq!(
                    said_both_ways(&older, output, Some("qwen3")).await,
                    expected,
                    "the older arm, {case}"
                );
                for start in [
                    GrammarStart::FirstToken,
                    GrammarStart::AfterReasoning,
                    GrammarStart::Unknown,
                    GrammarStart::Either,
                ] {
                    // In the call, after the prompt's replay, or where the output's first bytes
                    // say: in the call for the JSON alone, after the replay for the thought.
                    let in_the_call = match start {
                        GrammarStart::FirstToken => true,
                        GrammarStart::AfterReasoning => false,
                        GrammarStart::Unknown | GrammarStart::Either => alone,
                    };
                    let expected = match (in_the_call, alone) {
                        (true, true) => call.clone(),
                        (true, false) if by_name => said(&[thought_first], "", "", "tool_calls"),
                        (true, false) => said(&[], "", thought_first, "stop"),
                        (false, false) if separated => said(&[arguments], "plan", "", "tool_calls"),
                        (false, false) => said(&[arguments], "", "plan", "tool_calls"),
                        (false, true) if separated => said(&[], json_alone, "", "stop"),
                        (false, true) => said(&[], "", json_alone, "stop"),
                    };
                    let table = spec_under_the_schema(Table::Qwen3, &request, OPENED, start);
                    assert_eq!(
                        said_both_ways(&table, output, None).await,
                        (expected.clone(), expected),
                        "{start:?}, {case}"
                    );
                }
            }
        }
    }
}

/// The one place a thinking table of the registry has no tag for a forced call: the prompt's
/// thought open, on workers that start a grammar at the first token. Where no tool-call parser
/// is named the call takes the JSON schema there and the output is the JSON alone: the table
/// reads the call, streamed and collected, the reasoning separated or kept in the content. The
/// older arm, behind the reasoning parser that answers to the table's name, reads a separated
/// request's stream as the thought and no call, and its collected answer as the call.
#[tokio::test]
async fn a_forced_call_the_schema_holds_from_the_first_token_is_the_call_on_a_thinking_table() {
    const THINKING: &str = "<|message_model|><|content_thinking|>";
    let named = json!({"type": "function", "function": {"name": "lookup"}});
    let output = r#"{"q":"Paris"}"#;
    let call = said(&[output], "", "", "tool_calls");
    for separated in [true, false] {
        let request = forced_request_keeping(&named, separated);
        let table =
            spec_under_the_schema(Table::Inkling, &request, THINKING, GrammarStart::FirstToken);
        assert_eq!(
            said_both_ways(&table, output, None).await,
            (call.clone(), call.clone()),
            "separated {separated}"
        );
        let mut older = ChatResponseSpec::from(&request);
        older.reasoning_in_prefill = true;
        let streamed = if separated {
            said(&[], output, "", "stop")
        } else {
            call.clone()
        };
        assert_eq!(
            said_both_ways(&older, output, Some("inkling")).await,
            (streamed, call.clone()),
            "the older arm, separated {separated}"
        );
    }
}

/// The output's first bytes decide on a table whose unopened turn starts inside the thought
/// too: after a prompt that opened the thought and after one with no turn opener, the JSON alone
/// is the call, with nothing of the prompt replayed, and the thought first is read where the
/// replay leaves the reader, its text in the content as the request keeps it there. The workers
/// do not say where they start a grammar.
#[tokio::test]
async fn a_forced_calls_first_bytes_decide_on_a_table_with_an_unopened_turn() {
    let named = json!({"type": "function", "function": {"name": "lookup"}});
    let request = forced_request_keeping(&named, false);
    let output = r#"{"q":"Paris"}"#;
    let call = said(&[output], "", "", "tool_calls");
    let thought_first = format!("plan</think>{output}");
    let thought_then_call = said(&[output], "", "plan", "tool_calls");
    for prompt in ["<｜User｜>hi<｜Assistant｜><think>", "<｜User｜>hi"] {
        let spec =
            spec_under_the_schema(Table::DeepSeekV4, &request, prompt, GrammarStart::Unknown);
        assert_eq!(
            said_both_ways(&spec, output, None).await,
            (call.clone(), call.clone()),
            "{prompt}"
        );
        assert_eq!(
            said_both_ways(&spec, &thought_first, None).await,
            (thought_then_call.clone(), thought_then_call.clone()),
            "{prompt}, the thought first"
        );
    }
}

/// A thought that begins with the forced JSON's own opener, on workers that do not say or that
/// start at both places: the byte after the opener is none the JSON writes there, so the output
/// is read where the prompt's replay leaves the reader: the thought, as reasoning or kept in the
/// content, and then the call, streamed one character at a time and collected. A table whose own
/// markers begin with the bracket closes an empty thought ahead of a required call's list with
/// that byte, and the list is the call there too.
#[tokio::test]
async fn a_thought_that_begins_with_the_forced_jsons_opener_is_the_thought_and_then_the_call() {
    const OPENED: &str = "<|im_start|>assistant\n<think>\n";
    let named = json!({"type": "function", "function": {"name": "lookup"}});
    let required = json!("required");
    let arguments = r#"{"q":"Paris"}"#;
    let list = r#"[{"name":"lookup","parameters":{"q":"Paris"}}]"#;
    for (tool_choice, json_alone, thought) in [
        (&named, arguments, "{ plan"),
        (&named, arguments, "{thinking} plan"),
        (&required, list, "[ plan"),
        (&required, list, "[analysis] plan"),
    ] {
        let output = format!("{thought}</think>{json_alone}");
        for separated in [true, false] {
            let request = forced_request_keeping(tool_choice, separated);
            let expected = if separated {
                said(&[arguments], thought, "", "tool_calls")
            } else {
                said(&[arguments], "", thought, "tool_calls")
            };
            for start in [GrammarStart::Unknown, GrammarStart::Either] {
                let table = spec_under_the_schema(Table::Qwen3, &request, OPENED, start);
                assert_eq!(
                    said_both_ways(&table, &output, None).await,
                    (expected.clone(), expected.clone()),
                    "{start:?}, separated {separated}, {output}"
                );
            }
        }
    }
    // The close of an empty thought begins with the list's own opener.
    let closed_at_once = format!("[/THINK]{list}");
    let call = said(&[arguments], "", "", "tool_calls");
    for separated in [true, false] {
        let request = forced_request_keeping(&required, separated);
        for start in [GrammarStart::Unknown, GrammarStart::Either] {
            let table = spec_under_the_schema(Table::MistralV13, &request, "[/INST][THINK]", start);
            assert_eq!(
                said_both_ways(&table, &closed_at_once, None).await,
                (call.clone(), call.clone()),
                "{start:?}, separated {separated}, {closed_at_once}"
            );
        }
    }
}

/// Whitespace is the forced JSON's until a byte says otherwise, on workers that do not say or
/// that start at both places: a run of it that fills the output's 64 deciding bytes, ahead of
/// the opener or behind it, starts the reader in the call, so the JSON behind it is the call,
/// streamed one character at a time and collected. One byte short of them the next byte still
/// decides, and a thought there is the thought and then the call. A thought behind a run that
/// fills them is the stated limit: it reads as workers that hold the output to the schema from
/// the first token read it, and so does the run alone, to the output's end.
#[tokio::test]
async fn whitespace_that_fills_a_forced_calls_deciding_bytes_is_the_calls() {
    const OPENED: &str = "<|im_start|>assistant\n<think>\n";
    const DECIDING_BYTES: usize = 64;
    let named = json!({"type": "function", "function": {"name": "lookup"}});
    let required = json!("required");
    let arguments = r#"{"q":"Paris"}"#;
    let list = r#"[{"name":"lookup","parameters":{"q":"Paris"}}]"#;
    let filled = " ".repeat(DECIDING_BYTES);
    let short = &filled[1..];
    // Behind the opener, itself the first of the deciding bytes.
    let (opened_filled, opened_short) = (&filled[1..], &filled[2..]);
    for (tool_choice, json_alone) in [(&named, arguments), (&required, list)] {
        let (opener, rest) = json_alone.split_at(1);
        let thought_first = format!("plan</think>{json_alone}");
        for separated in [true, false] {
            let request = forced_request_keeping(tool_choice, separated);
            let known = |start| spec_under_the_schema(Table::Qwen3, &request, OPENED, start);
            let (in_the_call, after_the_replay) = (
                known(GrammarStart::FirstToken),
                known(GrammarStart::AfterReasoning),
            );
            for (output, reads_as, the_call) in [
                (format!("{filled}{json_alone}"), &in_the_call, true),
                (format!("{short}{json_alone}"), &in_the_call, true),
                (format!("{opener}{opened_filled}{rest}"), &in_the_call, true),
                (format!("{short}{thought_first}"), &after_the_replay, true),
                (
                    format!("{opener}{opened_short}{thought_first}"),
                    &after_the_replay,
                    true,
                ),
                // The stated limit.
                (format!("{filled}{thought_first}"), &in_the_call, false),
                (
                    format!("{opener}{opened_filled}{thought_first}"),
                    &in_the_call,
                    false,
                ),
            ] {
                let expected = said_both_ways(reads_as, &output, None).await;
                let whole_call = |said: &Said| {
                    matches!(said.0.as_slice(), [(name, written)] if name == "lookup"
                        && serde_json::from_str::<Value>(written).ok()
                            == Some(json!({"q": "Paris"})))
                };
                assert_eq!(
                    (whole_call(&expected.0), whole_call(&expected.1)),
                    (the_call, the_call),
                    "{output:?}: {expected:?}"
                );
                for start in [GrammarStart::Unknown, GrammarStart::Either] {
                    let table = spec_under_the_schema(Table::Qwen3, &request, OPENED, start);
                    assert_eq!(
                        said_both_ways(&table, &output, None).await,
                        expected,
                        "{start:?}, separated {separated}, {output:?}"
                    );
                }
            }
            // The run alone, to the output's end: a call cut short, as on first-token workers.
            let run = format!("{opener}{}", " ".repeat(4 * DECIDING_BYTES));
            let table =
                spec_under_the_schema(Table::Qwen3, &request, OPENED, GrammarStart::Unknown);
            assert_eq!(
                said_both_ways(&table, &run, None).await,
                said_both_ways(&in_the_call, &run, None).await,
                "separated {separated}, the run alone"
            );
        }
    }
}

/// A named function whose arguments the schema does not hold to an object, on workers that do
/// not say or that start at both places, with the reasoning kept in the content and the prompt's
/// thought open: the JSON alone is the call whatever byte it opens with, as the older arm reads
/// it, streamed one character at a time and collected. No opener is named for it: read after
/// the prompt's replay, a string, a number or a list alone would be content and no call.
#[tokio::test]
async fn a_named_call_not_held_to_an_object_is_the_call_where_the_thought_is_kept() {
    const OPENED: &str = "<|im_assistant|>assistant<|im_middle|><think>";
    for (parameters, arguments) in [
        (json!({"type": "string"}), json!("Paris")),
        (json!({"type": "integer"}), json!(7)),
        (
            json!({"type": "array", "items": {"type": "string"}}),
            json!(["Paris"]),
        ),
        (json!({}), json!(null)),
        (json!({}), json!({"city": "Paris"})),
    ] {
        let written = arguments.to_string();
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen3-8B", "messages": [], "stream": true,
            "separate_reasoning": false,
            "tool_choice": {"type": "function", "function": {"name": "look up"}},
            "tools": [{"type": "function",
                "function": {"name": "look up", "parameters": parameters}}]
        }))
        .expect("a chat request");
        let call: Said = (
            vec![("look up".to_string(), written.clone())],
            String::new(),
            String::new(),
            "tool_calls".to_string(),
        );
        let mut older = ChatResponseSpec::from(&request);
        older.reasoning_in_prefill = true;
        assert_eq!(
            said_both_ways(&older, &written, Some("kimi_k25")).await,
            (call.clone(), call.clone()),
            "the older arm, {written}"
        );
        for start in [GrammarStart::Unknown, GrammarStart::Either] {
            let spec = spec_under_the_schema(Table::KimiK2, &request, OPENED, start);
            assert_eq!(
                said_both_ways(&spec, &written, None).await,
                (call.clone(), call.clone()),
                "{start:?}, {written}"
            );
        }
    }
}

/// The rows of workers that do not say, and of a pool of both kinds, for the tools a table's
/// syntax writes and for the ones it cannot write, where a forced call takes the gateway's JSON
/// schema on that table: a tool with no name; on `kimi_k2` a name with a space, a newline, or a
/// quote and a bracket; on `muse_spark` a name or a required key with a quote and a bracket, and
/// an empty required key. After the prompt's tail, the JSON alone is the call, and the thought
/// first is the thought, as reasoning or kept in the content, and then the call: the function
/// named and a call required, streamed one character at a time and collected.
#[tokio::test]
async fn a_forced_call_on_workers_that_do_not_say_reads_the_call_for_every_tool_shape() {
    // A table, the prompt's tail, and the thought as the model writes it up to the JSON.
    const QWEN: (&str, &str, &str) = ("qwen3", "<|im_start|>assistant\n<think>\n", "plan</think>");
    const KIMI: (&str, &str, &str) = (
        "kimi_k2",
        "<|im_assistant|>assistant<|im_middle|><think>",
        "plan</think>",
    );
    const MUSE: (&str, &str, &str) = (
        "muse_spark",
        "<|start|>assistant",
        " to=self<|message|>plan<|eom|>",
    );
    let city = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let keyed = |key: &str| json!({"type": "object", "properties": {key: {"type": "string"}}, "required": [key]});
    let paris = json!({"city": "Paris"});
    // The tool's name, its parameters, and the arguments the model writes.
    let shapes = [
        (QWEN, "lookup", city.clone(), paris.clone()),
        (QWEN, "", city.clone(), paris.clone()),
        (KIMI, "lookup", city.clone(), paris.clone()),
        (KIMI, "look up", city.clone(), paris.clone()),
        (KIMI, "look\nup", city.clone(), paris.clone()),
        (KIMI, "look\"up<", city.clone(), paris.clone()),
        (MUSE, "lookup", city.clone(), paris.clone()),
        (MUSE, "look\"up<", city.clone(), paris.clone()),
        (
            MUSE,
            "lookup",
            keyed("ci\"ty<"),
            json!({"ci\"ty<": "Paris"}),
        ),
        (MUSE, "lookup", keyed(""), json!({"": "Paris"})),
    ];
    for ((table, prompt, thought), name, parameters, arguments) in &shapes {
        let table = Table::ALL
            .into_iter()
            .find(|listed| listed.name() == *table)
            .expect("a table of the registry");
        let written = arguments.to_string();
        // One call of the tool with the arguments as written, after `reasoning` and `content`.
        let call = |reasoning: &str, content: &str| -> Said {
            (
                vec![((*name).to_string(), written.clone())],
                reasoning.to_string(),
                content.to_string(),
                "tool_calls".to_string(),
            )
        };
        for by_name in [true, false] {
            let (tool_choice, json_alone) = if by_name {
                (
                    json!({"type": "function", "function": {"name": name}}),
                    written.clone(),
                )
            } else {
                (
                    json!("required"),
                    json!([{"name": name, "parameters": arguments}]).to_string(),
                )
            };
            let thought_first = format!("{thought}{json_alone}");
            for separated in [true, false] {
                let request: ChatCompletionRequest = serde_json::from_value(json!({
                    "model": "Qwen/Qwen3-8B", "messages": [], "stream": true,
                    "separate_reasoning": separated, "tool_choice": tool_choice,
                    "tools": [{"type": "function",
                        "function": {"name": name, "parameters": parameters}}]
                }))
                .expect("a chat request");
                let after_the_thought = if separated {
                    call("plan", "")
                } else {
                    call("", "plan")
                };
                for start in [GrammarStart::Unknown, GrammarStart::Either] {
                    let case = format!(
                        "{}, {name:?}, named {by_name}, separated {separated}, {start:?}",
                        table.name()
                    );
                    let spec = spec_under_the_schema(table, &request, prompt, start);
                    assert_eq!(
                        said_both_ways(&spec, &json_alone, None).await,
                        (call("", ""), call("", "")),
                        "{case}: the JSON alone"
                    );
                    assert_eq!(
                        said_both_ways(&spec, &thought_first, None).await,
                        (after_the_thought.clone(), after_the_thought.clone()),
                        "{case}: the thought first"
                    );
                }
            }
        }
    }
}
