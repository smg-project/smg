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
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto::GenerateComplete, vllm_engine::proto};
use symphony::{
    formats::{deepseek_v4, deepseek_v4_1, gemma4, glm, glm4, llama4, plain, qwen3},
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
            symphony::{assign, template::Templates, testing::Refusing, Asks, Assignment},
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
        #[cfg(feature = "diagnostic-capture")]
        recipe: None,
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
    let (stream, server) = scripted_stream(responses, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor(with_tools)
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
        let (result, events) = stream_events(responses, true, spec).await;
        assert!(result.is_ok(), "{tail:?}: {result:?}");
        let calls = tool_call_deltas(&events);
        assert_eq!(
            calls[0]["function"]["name"], "lookup",
            "{tail:?}: {events:#?}"
        );
        let streamed: String = calls
            .iter()
            .filter_map(|call| call["function"]["arguments"].as_str())
            .collect();
        assert_eq!(streamed, arguments, "{tail:?}: {events:#?}");
        assert_eq!(
            chat_text(&events, 0, "content"),
            "",
            "nothing of the call is content: {tail:?}: {events:#?}"
        );
        let finish = events
            .iter()
            .find_map(|event| event["choices"][0]["finish_reason"].as_str())
            .expect("a finish chunk");
        assert_eq!(finish, "length", "{tail:?}");
    }
}

#[tokio::test]
async fn a_whole_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length() {
    for (tail, arguments) in CUT_CALL_TAILS {
        let output = format!("{CUT_CALL_PREFIX}{tail}");
        let (choice, _) = fold_finished(&output, deepseek_v4_1(), "<think>", "length").await;
        let calls = choice.message.tool_calls.as_ref().expect("the call");
        assert_eq!(calls.len(), 1, "{tail:?}");
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(
            calls[0].function.arguments.as_deref(),
            Some(arguments),
            "{tail:?}: {choice:?}"
        );
        assert_eq!(
            choice.message.content, None,
            "nothing of the call is content: {tail:?}: {choice:?}"
        );
        assert_eq!(choice.finish_reason.as_deref(), Some("length"), "{tail:?}");
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
async fn a_refused_block_released_as_the_answer_keeps_its_tokens_logprobs() {
    // The whole answer after the thought is a call written as Python inside the call tag: the
    // reader releases the block, from its opener, as the content, and its tokens keep their
    // logprobs, placed where the output wrote them, not where the released event stands. Two
    // closed blocks with the template's newline between them the same, over the one slice.
    let output = "plan</think><tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    let (choice, reasoning_tokens) = fold_with(output, glm(), "<|assistant|><think>").await;
    let released = "<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    assert_eq!(choice.message.content.as_deref(), Some(released));
    assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    assert_eq!(reasoning_tokens, 4);
    assert_eq!(kept_logprob_tokens(&choice), released);
    let output = "plan</think><tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
    let (choice, _) = fold_with(output, glm(), "<|assistant|><think>").await;
    let released = "<tool_call>a(x=1)</tool_call>\n<tool_call>b(y=2)</tool_call>";
    assert_eq!(choice.message.content.as_deref(), Some(released));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(kept_logprob_tokens(&choice), released);
    // The answer's text, then the block: the text's tokens and the block's keep their logprobs,
    // the block's where the output wrote them.
    let output = "plan</think>Let me look:<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    let (choice, _) = fold_with(output, glm(), "<|assistant|><think>").await;
    let content = "Let me look:<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
    assert_eq!(choice.message.content.as_deref(), Some(content));
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    assert_eq!(kept_logprob_tokens(&choice), content);
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
pub(super) fn assigned_message(
    format: Format,
    prompt: &str,
    thinking: bool,
) -> MessagesResponseSpec {
    MessagesResponseSpec {
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
            #[cfg(feature = "diagnostic-capture")]
            recipe: None,
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
            context::ExecutionResult::Single { stream },
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

/// The scrape of a recorder of this test's own after `read` ran on this thread (the test
/// runtime is single-threaded, so the local recorder sees every record of the future).
pub(super) async fn scraped<F: std::future::Future>(read: F) -> (F::Output, String) {
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
            context::ExecutionResult::Single { stream },
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
            context::ExecutionResult::Single { stream },
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
    }
}

#[tokio::test]
async fn a_whole_pythonic_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length()
{
    for (output, arguments) in PYTHONIC_CUT_CALLS {
        let (choice, _) = fold_finished(output, llama4(), PYTHONIC_PROMPT, "length").await;
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
        assert!(
            serde_json::from_str::<Value>(&streamed).is_err(),
            "the arguments stay cut: {streamed:?}"
        );
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
    }
}

#[tokio::test]
async fn a_whole_gemma_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length() {
    for (output, arguments) in GEMMA_CUT_CALLS {
        let (choice, _) = fold_finished(output, gemma4(), GEMMA_PROMPT, "length").await;
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
    }
}

#[tokio::test]
async fn a_whole_keyed_call_the_token_limit_cut_keeps_its_bytes_in_the_call_and_finishes_length() {
    for (output, arguments) in KEYED_CUT_CALLS {
        let (choice, _) = fold_finished(output, glm4(), KEYED_PROMPT, "length").await;
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
}
