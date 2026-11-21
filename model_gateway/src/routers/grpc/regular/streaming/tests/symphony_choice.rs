//! A narrowed tool choice returns only the calls it leaves callable. Under a named function, an
//! allowed list or `required`, a call to any other function is not delivered, streamed or whole,
//! on the chat and Messages paths; the calls beside it go out as before, and `auto` is unchanged.

use std::sync::Arc;

use llm_tokenizer::StopSequenceConfig;
use openai_protocol::{
    chat::ChatCompletionRequest, messages::CreateMessageRequest, profile::ProviderProfile,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto::GenerateComplete, vllm_engine::proto};
use symphony::{
    formats::{deepseek_v4_1, glm, Table},
    grammar::GrammarStart,
    Format,
};

use super::{
    super::*,
    eof::{
        chat_text, chunk, collect_events, complete, dispatch, processor, scripted_stream,
        CharacterTokenizer,
    },
    symphony_chat::{message_blocks, message_events, scraped, series_line},
};
use crate::routers::grpc::{
    proto_wrapper::ProtoGenerateComplete,
    regular::{
        processor::ResponseProcessor,
        symphony::{assign_for, Asks, Reading},
    },
};

/// A call as a test writes it: the function, its one argument's name and the argument's text.
type Call = (&'static str, &'static str, &'static str);

const SAVE_NOTE: Call = ("save_note", "text", "t");
const DELETE_FILE: Call = ("delete_file", "path", "/");
const LOOKUP: Call = ("lookup", "text", "q");

/// One table under test: the table, the prompt its reader replays, and how it writes an output
/// of whole calls.
struct Family {
    format: fn() -> Arc<Format>,
    prompt: &'static str,
    write: fn(&[Call]) -> String,
}

const FAMILIES: [Family; 3] = [
    Family {
        format: glm_table,
        prompt: "<|assistant|><think>",
        write: keyed_calls,
    },
    Family {
        format: deepseek_table,
        prompt: "",
        write: attributed_calls,
    },
    Family {
        format: tagged_table,
        prompt: "",
        write: tagged_calls,
    },
];

fn glm_table() -> Arc<Format> {
    Arc::new(glm())
}

fn deepseek_table() -> Arc<Format> {
    Arc::new(deepseek_v4_1())
}

fn tagged_table() -> Arc<Format> {
    Table::Qwen3Tagged.shared()
}

/// The calls as GLM writes them, after the thought's close.
fn keyed_calls(calls: &[Call]) -> String {
    let mut output = String::from("</think>");
    for (name, key, value) in calls {
        output.push_str(&format!(
            "<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>"
        ));
    }
    output
}

/// The calls as DeepSeek V4.1 writes them: one block of invokes.
fn attributed_calls(calls: &[Call]) -> String {
    let mut output = String::from("\n\n<｜DSML｜ calls>\n");
    for (name, key, value) in calls {
        output.push_str(&format!(
            "<｜DSML｜ invoke name=\"{name}\">\n<｜DSML｜ parameter name=\"{key}\" \
             string=\"true\">{value}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
        ));
    }
    output.push_str("</｜DSML｜ calls>");
    output
}

/// The calls as the tagged Qwen table writes them, a line break between two.
fn tagged_calls(calls: &[Call]) -> String {
    calls
        .iter()
        .map(|(name, key, value)| {
            format!(
                "<tool_call>\n<function={name}>\n<parameter={key}>\n{value}\n</parameter>\n\
                 </function>\n</tool_call>"
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The parameters every tool of these requests takes.
fn parameters() -> Value {
    json!({"type": "object", "properties": {
        "text": {"type": "string"}, "path": {"type": "string"}, "meta": {"type": "object"}}})
}

/// The spec of a chat request that declares `tools` under `choice` (no choice at all for
/// `None`), read by `family`'s table.
fn chat_spec_of(
    family: &Family,
    tools: &[&str],
    choice: Option<Value>,
    parallel: bool,
) -> ChatResponseSpec {
    let tools: Value = tools
        .iter()
        .map(|name| json!({"type": "function", "function": {"name": name, "parameters": parameters()}}))
        .collect();
    let mut request = json!({
        "model": "Qwen/Qwen3-8B", "messages": [], "stream": true, "separate_reasoning": true,
        "parallel_tool_calls": parallel, "tools": tools,
    });
    if let Some(choice) = choice {
        request["tool_choice"] = choice;
    }
    let request: ChatCompletionRequest = serde_json::from_value(request).expect("a chat request");
    let mut spec = ChatResponseSpec::from(&request);
    spec.symphony = assign_for(
        Reading::Table((family.format)(), false),
        Asks::Chat(&request),
        None,
        family.prompt,
        ProviderProfile::for_model(&request.model),
        || GrammarStart::Unknown,
    );
    assert!(spec.symphony.is_some(), "a table reads");
    spec
}

/// A choice that names `function`.
fn named(function: &str) -> Value {
    json!({"type": "function", "function": {"name": function}})
}

/// A choice that allows `functions`, in `mode`.
fn allowed(mode: &str, functions: &[&str]) -> Value {
    let tools: Value = functions
        .iter()
        .map(|name| json!({"type": "function", "name": name}))
        .collect();
    json!({"type": "allowed_tools", "mode": mode, "tools": tools})
}

/// What a reader said of choice 0: the calls in the order of their indexes, each the function
/// and its arguments as written; the content; the finish reason.
struct Said {
    calls: Vec<(String, String)>,
    content: String,
    finish: String,
}

/// `output` one character a frame, then the engine's stop.
fn by_character(output: &str) -> Vec<proto::GenerateResponse> {
    let mut frames: Vec<_> = output.chars().map(|ch| chunk(0, &ch.to_string())).collect();
    frames.push(complete(0, "stop"));
    frames
}

/// What the stream of `frames` said under `spec`.
async fn streamed(spec: &ChatResponseSpec, frames: Vec<proto::GenerateResponse>) -> Said {
    let (stream, server) = scripted_stream(frames, "0").await;
    let (tx, rx) = sse_channel();
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let result = processor(true)
        .process_streaming_chunks(
            stream,
            dispatch(),
            tokenizer,
            (None, None, false, false, false),
            spec.clone(),
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    let mut calls: Vec<(String, String)> = Vec::new();
    let deltas = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten();
    for delta in deltas {
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
    let finish = events
        .iter()
        .find_map(|event| event["choices"][0]["finish_reason"].as_str())
        .expect("a finish chunk");
    Said {
        calls,
        content: chat_text(&events, 0, "content"),
        finish: finish.to_string(),
    }
}

/// What `output` said under `spec`, read whole.
async fn whole(spec: &ChatResponseSpec, output: &str) -> Said {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let engine = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids: output.chars().map(u32::from).collect(),
        finish_reason: "stop".to_string(),
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
        &engine,
        0,
        spec,
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
    Said {
        calls: choice
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
            .collect(),
        content: choice.message.content.clone().unwrap_or_default(),
        finish: choice.finish_reason.clone().unwrap_or_default(),
    }
}

/// What `output` says under `spec` each way these tests read it, with the way's name: whole, and
/// streamed one character a frame.
async fn both_ways(spec: &ChatResponseSpec, output: &str) -> Vec<(String, Said)> {
    vec![
        ("whole".to_string(), whole(spec, output).await),
        (
            "a character a frame".to_string(),
            streamed(spec, by_character(output)).await,
        ),
    ]
}

/// [`both_ways`], and streamed in two frames at every cut of `output`.
async fn every_way(spec: &ChatResponseSpec, output: &str) -> Vec<(String, Said)> {
    let mut said = both_ways(spec, output).await;
    for (at, _) in output.char_indices().skip(1) {
        let frames = vec![
            chunk(0, &output[..at]),
            chunk(0, &output[at..]),
            complete(0, "stop"),
        ];
        said.push((format!("cut at byte {at}"), streamed(spec, frames).await));
    }
    said
}

/// `said` holds `calls` and nothing else: each with its arguments, no text beside them, and the
/// finish reason `finish`.
fn assert_said(said: &Said, calls: &[Call], finish: &str, at: &str) {
    let got: Vec<(String, Value)> = said
        .calls
        .iter()
        .map(|(name, arguments)| {
            let arguments = serde_json::from_str(arguments)
                .unwrap_or_else(|e| panic!("{at}: the arguments {arguments:?} of {name}: {e}"));
            (name.clone(), arguments)
        })
        .collect();
    let want: Vec<(String, Value)> = calls
        .iter()
        .map(|(name, key, value)| {
            let arguments = [((*key).to_string(), Value::from(*value))];
            (
                (*name).to_string(),
                Value::Object(arguments.into_iter().collect()),
            )
        })
        .collect();
    assert_eq!(got, want, "{at}");
    assert_eq!(said.finish, finish, "{at}");
    assert_eq!(said.content.trim(), "", "{at}");
}

/// The name of `family`'s table.
fn table_of(family: &Family) -> String {
    (family.format)().name().to_string()
}

#[tokio::test]
async fn a_named_choice_returns_the_named_call_alone() {
    for family in &FAMILIES {
        let spec = chat_spec_of(
            family,
            &["save_note", "delete_file"],
            Some(named("save_note")),
            true,
        );
        let output = (family.write)(&[SAVE_NOTE, DELETE_FILE]);
        for (way, said) in every_way(&spec, &output).await {
            let at = format!("{} {way}", table_of(family));
            assert_said(&said, &[SAVE_NOTE], "tool_calls", &at);
        }
    }
}

/// The call the choice leaves out comes first: the named call is the first one returned.
#[tokio::test]
async fn a_call_the_choice_leaves_out_takes_no_place_among_the_calls_returned() {
    for family in &FAMILIES {
        let spec = chat_spec_of(
            family,
            &["save_note", "delete_file"],
            Some(named("save_note")),
            true,
        );
        let output = (family.write)(&[DELETE_FILE, SAVE_NOTE]);
        for (way, said) in every_way(&spec, &output).await {
            let at = format!("{} {way}", table_of(family));
            assert_said(&said, &[SAVE_NOTE], "tool_calls", &at);
        }
    }
}

#[tokio::test]
async fn an_allowed_list_returns_its_calls_alone() {
    for mode in ["auto", "required"] {
        for family in &FAMILIES {
            let spec = chat_spec_of(
                family,
                &["save_note", "delete_file", "lookup"],
                Some(allowed(mode, &["save_note", "lookup"])),
                true,
            );
            let output = (family.write)(&[SAVE_NOTE, DELETE_FILE, LOOKUP]);
            for (way, said) in both_ways(&spec, &output).await {
                let at = format!("{} {mode} {way}", table_of(family));
                assert_said(&said, &[SAVE_NOTE, LOOKUP], "tool_calls", &at);
            }
        }
    }
}

/// An allowed list that names no function the request declares leaves none callable: no call is
/// returned, and the output ends as the engine ended it.
#[tokio::test]
async fn an_allowed_list_of_no_declared_tool_returns_no_call() {
    for family in &FAMILIES {
        let spec = chat_spec_of(family, &["delete_file"], Some(allowed("auto", &[])), true);
        let output = (family.write)(&[DELETE_FILE]);
        for (way, said) in both_ways(&spec, &output).await {
            let at = format!("{} {way}", table_of(family));
            assert_said(&said, &[], "stop", &at);
        }
    }
}

/// Under `required` every declared function is callable, and no other: a call to a function the
/// request did not declare is not returned, and an output of that call alone ends as the engine
/// ended it, with no call.
#[tokio::test]
async fn a_required_choice_returns_no_call_to_a_function_the_request_did_not_declare() {
    for family in &FAMILIES {
        let spec = chat_spec_of(family, &["save_note"], Some(json!("required")), true);
        let output = (family.write)(&[SAVE_NOTE, DELETE_FILE]);
        for (way, said) in both_ways(&spec, &output).await {
            let at = format!("{} beside a declared call, {way}", table_of(family));
            assert_said(&said, &[SAVE_NOTE], "tool_calls", &at);
        }
        let output = (family.write)(&[DELETE_FILE]);
        for (way, said) in both_ways(&spec, &output).await {
            let at = format!("{} alone, {way}", table_of(family));
            assert_said(&said, &[], "stop", &at);
        }
    }
}

/// With parallel calls off, the one call returned is the first the choice allows.
#[tokio::test]
async fn one_call_only_keeps_the_first_call_the_choice_allows() {
    for family in &FAMILIES {
        let spec = chat_spec_of(
            family,
            &["save_note", "delete_file", "lookup"],
            Some(allowed("auto", &["save_note", "lookup"])),
            false,
        );
        let output = (family.write)(&[DELETE_FILE, SAVE_NOTE, LOOKUP]);
        for (way, said) in both_ways(&spec, &output).await {
            let at = format!("{} {way}", table_of(family));
            assert_said(&said, &[SAVE_NOTE], "tool_calls", &at);
        }
    }
}

/// `auto`, spelled or left out, returns every call the model made, one to a function the request
/// did not declare included.
#[tokio::test]
async fn auto_returns_every_call() {
    for choice in [None, Some(json!("auto"))] {
        for family in &FAMILIES {
            let spec = chat_spec_of(family, &["save_note"], choice.clone(), true);
            let output = (family.write)(&[SAVE_NOTE, DELETE_FILE]);
            for (way, said) in both_ways(&spec, &output).await {
                let at = format!("{} {choice:?} {way}", table_of(family));
                assert_said(&said, &[SAVE_NOTE, DELETE_FILE], "tool_calls", &at);
            }
        }
    }
}

/// A call the choice leaves out is counted where the calls a request forbade are counted, once,
/// whole and streamed.
#[tokio::test]
async fn a_call_the_choice_leaves_out_is_counted_with_the_forbidden_calls() {
    for family in &FAMILIES {
        let spec = chat_spec_of(
            family,
            &["save_note", "delete_file"],
            Some(named("save_note")),
            true,
        );
        let table = spec.symphony.as_ref().expect("a table reads").table();
        let label = format!("table=\"{table}\"");
        let output = (family.write)(&[SAVE_NOTE, DELETE_FILE]);
        let (_, scrape) = scraped(whole(&spec, &output)).await;
        let counted = series_line(&scrape, "smg_symphony_calls_forbidden_total", &[&label])
            .unwrap_or_else(|| panic!("{table} whole: no forbidden call counted"));
        assert!(counted.ends_with(" 1"), "{table} whole: {counted}");
        let (_, scrape) = scraped(streamed(&spec, by_character(&output))).await;
        let counted = series_line(&scrape, "smg_symphony_calls_forbidden_total", &[&label])
            .unwrap_or_else(|| panic!("{table} streamed: no forbidden call counted"));
        assert!(counted.ends_with(" 1"), "{table} streamed: {counted}");
    }
}

/// The spec of a Messages request that declares `tools` under `choice`, read by `family`'s
/// table.
fn message_spec_of(family: &Family, tools: &[&str], choice: Value) -> MessagesResponseSpec {
    let tools: Value = tools
        .iter()
        .map(|name| json!({"name": name, "input_schema": parameters()}))
        .collect();
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "Qwen/Qwen3-8B", "max_tokens": 64, "messages": [],
        "tool_choice": choice, "tools": tools,
    }))
    .expect("a Messages request");
    let mut spec = MessagesResponseSpec::from(&request);
    spec.symphony = assign_for(
        Reading::Table((family.format)(), false),
        Asks::for_messages(&request, false),
        None,
        family.prompt,
        ProviderProfile::for_model(&request.model),
        || GrammarStart::Unknown,
    );
    assert!(spec.symphony.is_some(), "a table reads");
    spec
}

/// A content block as a test reads it: its type, its text or its tool's name, and the tool's
/// input.
type Block = (String, String, Value);

/// The content blocks of a streamed message of `output` under `spec`, every one but a thought,
/// and the message's stop reason.
async fn message_said(spec: MessagesResponseSpec, output: &str) -> (Vec<Block>, String) {
    let (result, events) = message_events(by_character(output), spec, None).await;
    assert!(result.is_ok(), "{result:?}");
    let blocks = message_blocks(&events)
        .into_iter()
        .filter(|(kind, _, _)| kind != "thinking")
        .collect();
    let stop = events
        .iter()
        .find_map(|event| event["delta"]["stop_reason"].as_str())
        .expect("a stop reason");
    (blocks, stop.to_string())
}

/// The content blocks of the whole message of `output` under `spec`, every one but a thought.
async fn whole_message(spec: MessagesResponseSpec, output: &str) -> Vec<Block> {
    let mut frame = complete(0, "stop");
    if let Some(proto::generate_response::Response::Complete(done)) = &mut frame.response {
        done.output_ids = output.chars().map(u32::from).collect();
        done.completion_tokens = done.output_ids.len() as u32;
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
        spec,
        dispatch(),
        tokenizer,
        &mut decoder,
    )
    .await;
    server.abort();
    match result {
        Ok((message, _)) => serde_json::to_value(&message.content)
            .expect("the content blocks")
            .as_array()
            .expect("a list of blocks")
            .iter()
            .filter(|block| block["type"] != "thinking")
            .map(|block| {
                let named = block["name"].as_str().or_else(|| block["text"].as_str());
                (
                    block["type"].as_str().unwrap_or_default().to_string(),
                    named.unwrap_or_default().to_string(),
                    block["input"].clone(),
                )
            })
            .collect(),
        Err(response) => panic!("a message, not the status {}", response.status()),
    }
}

#[tokio::test]
async fn a_message_returns_the_tool_its_choice_names_alone() {
    for family in &FAMILIES {
        let spec = || {
            message_spec_of(
                family,
                &["save_note", "delete_file"],
                json!({"type": "tool", "name": "save_note"}),
            )
        };
        let output = (family.write)(&[SAVE_NOTE, DELETE_FILE]);
        let (blocks, stop) = message_said(spec(), &output).await;
        let table = table_of(family);
        let want = [(
            "tool_use".to_string(),
            "save_note".to_string(),
            json!({"text": "t"}),
        )];
        assert_eq!(blocks, want, "{table}");
        assert_eq!(stop, "tool_use", "{table}");
        assert_eq!(whole_message(spec(), &output).await, want, "{table}");
    }
}

/// Under `any` a tool the request did not declare is not returned: the message of that tool
/// alone uses no tool and ends its turn.
#[tokio::test]
async fn a_message_under_any_returns_no_tool_the_request_did_not_declare() {
    for family in &FAMILIES {
        let spec = || message_spec_of(family, &["save_note"], json!({"type": "any"}));
        let output = (family.write)(&[DELETE_FILE]);
        let (blocks, stop) = message_said(spec(), &output).await;
        let table = table_of(family);
        assert_eq!(blocks, [], "{table}");
        assert_eq!(stop, "end_turn", "{table}");
        assert_eq!(whole_message(spec(), &output).await, [], "{table}");
    }
}

/// A call to `delete_file` as the table of JSON objects reads it, whose object the output never
/// closes: a member stands behind its arguments.
const OPEN_CALL: &str =
    "<tool_call>\n{\"name\": \"delete_file\", \"arguments\": {\"path\": \"/\"}, \"note\": \"n\"";

/// A whole call to `lookup` as that table reads it.
const WHOLE_CALL: &str =
    "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"text\": \"q\"}}\n</tool_call>";

/// Under `choice`, and with one call only unless `parallel`, `output` of the table of JSON
/// objects is said as `calls` and no text beside them, whole and at every cut.
async fn assert_objects_said(
    choice: Option<Value>,
    parallel: bool,
    output: &str,
    calls: &[Call],
    finish: &str,
) {
    let family = Family {
        format: || Table::Qwen3.shared(),
        prompt: "",
        write: tagged_calls,
    };
    let spec = chat_spec_of(&family, &["lookup", "delete_file"], choice, parallel);
    for (way, said) in every_way(&spec, output).await {
        assert_said(&said, calls, finish, &format!("{way} of {output:?}"));
    }
}

/// A call the choice leaves out whose object the output never closes: the bytes behind its
/// arguments go with the call, so the client is sent no call and no text, and a call the choice
/// allows after it is returned alone.
#[tokio::test]
async fn a_call_the_choice_leaves_out_that_never_closes_returns_no_text() {
    let then = format!("{OPEN_CALL}\n</tool_call>\n{WHOLE_CALL}");
    for choice in [named("lookup"), allowed("auto", &["lookup"])] {
        assert_objects_said(Some(choice.clone()), true, OPEN_CALL, &[], "stop").await;
        assert_objects_said(Some(choice), true, &then, &[LOOKUP], "tool_calls").await;
    }
}

/// Under `none` a call whose object the output never closes returns no call and no text.
#[tokio::test]
async fn a_call_under_none_that_never_closes_returns_no_text() {
    let then = format!("{OPEN_CALL}\n</tool_call>\n{WHOLE_CALL}");
    assert_objects_said(Some(json!("none")), true, OPEN_CALL, &[], "stop").await;
    assert_objects_said(Some(json!("none")), true, &then, &[], "stop").await;
}

/// Where the request takes one call, a later call whose object the output never closes returns
/// no text beside the first call.
#[tokio::test]
async fn a_later_call_that_never_closes_returns_no_text() {
    let after = format!("{WHOLE_CALL}\n{OPEN_CALL}");
    let closed = format!("{after}\n</tool_call>");
    assert_objects_said(None, false, &after, &[LOOKUP], "tool_calls").await;
    assert_objects_said(None, false, &closed, &[LOOKUP], "tool_calls").await;
}
