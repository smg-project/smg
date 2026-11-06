//! The exact SSE bytes of Messages streams, pinned per scenario: text, text
//! under Symphony (the answer over several chunks: the first delta opens the
//! block through the content-block machine, the following ones go straight
//! from the borrowed text to the frame), thinking then text, a tool call and a
//! length stop, driven over scripted
//! engine frames and compared byte for byte with
//! `fixtures/messages_<scenario>.sse`. The fixtures hold the bytes the
//! per-event copies produced, so any change to the wire format of a Messages
//! event, intended or not, fails here. Set `SMG_RECORD_SSE_FIXTURES=1` to
//! rewrite them from the current code.

use smg_grpc_client::vllm_engine::proto;
use symphony::{formats::qwen3, CallSyntax};

use super::{
    super::*,
    eof::{
        chat_spec, chunk, complete, complete_with_prompt, processor, scripted_stream,
        CharacterTokenizer,
    },
    sse_bytes::assert_recorded_bytes,
};
use crate::routers::grpc::regular::symphony::Assignment;

struct MessagesScenario {
    name: &'static str,
    with_tools: bool,
    thinking: bool,
    /// Symphony's Qwen3 table reads the message (the content-block machine writes the events)
    /// instead of the reasoning and tool parsers.
    symphony: bool,
    /// The served model; a Kimi name gives the tool call its deterministic id.
    model: &'static str,
    frames: Vec<proto::GenerateResponse>,
    /// A substring the scenario's bytes must carry, so the scenario keeps
    /// exercising the path it was written for.
    must_contain: &'static str,
}

fn scenarios() -> Vec<MessagesScenario> {
    vec![
        MessagesScenario {
            name: "messages_text",
            with_tools: false,
            thinking: false,
            symphony: false,
            model: "fixture-model",
            frames: vec![
                chunk(0, "Hello"),
                chunk(0, ", world"),
                complete_with_prompt(0, "stop", 10),
            ],
            must_contain: r#""text":", world""#,
        },
        MessagesScenario {
            name: "messages_symphony_text",
            with_tools: false,
            thinking: false,
            symphony: true,
            model: "fixture-model",
            frames: vec![
                chunk(0, "Hello"),
                chunk(0, ", wor"),
                chunk(0, "ld"),
                chunk(0, " "),
                chunk(0, "caf\u{e9}"),
                chunk(0, " \"again\""),
                complete_with_prompt(0, "stop", 10),
            ],
            must_contain: r#""text":"ld""#,
        },
        MessagesScenario {
            name: "messages_thinking_then_text",
            with_tools: false,
            thinking: true,
            symphony: false,
            model: "fixture-model",
            frames: vec![
                chunk(0, "<think>because"),
                chunk(0, " of this</think>the answer"),
                complete(0, "stop"),
            ],
            must_contain: r#""thinking":"because""#,
        },
        MessagesScenario {
            name: "messages_tool_use",
            with_tools: true,
            thinking: false,
            symphony: false,
            model: "kimi-fixture",
            frames: vec![
                chunk(0, r#"{"name": "lookup", "#),
                chunk(0, r#""parameters": {"q": "x"}}"#),
                complete(0, "stop"),
            ],
            must_contain: r#""type":"tool_use""#,
        },
        MessagesScenario {
            name: "messages_length",
            with_tools: false,
            thinking: false,
            symphony: false,
            model: "fixture-model",
            frames: vec![chunk(0, "a"), complete(0, "length")],
            must_contain: r#""stop_reason":"max_tokens""#,
        },
    ]
}

/// The SSE bytes of one scenario.
async fn stream_bytes(scenario: &MessagesScenario) -> Vec<u8> {
    let (stream, server) = scripted_stream(scenario.frames.clone(), "0").await;
    let dispatch = context::DispatchMetadata {
        request_id: "msg-bytes".to_string(),
        model: scenario.model.to_string(),
        created: 1,
        weight_version: None,
    };
    let spec = MessagesResponseSpec {
        prompt_tokens: 0,
        thinking_requested: scenario.thinking,
        thinking_text_returned: true,
        reasoning_in_prefill: false,
        tool_choice: None,
        has_tools: scenario.with_tools,
        history_tool_calls_count: 0,
        chat_tools: if scenario.with_tools {
            chat_spec(true).tools.unwrap()
        } else {
            Vec::new().into()
        },
        stop_sequences: None,
        symphony: scenario.symphony.then(|| Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: "<|im_start|>assistant\n".to_string(),
            forbids_calls: false,
            past_the_end: false,
        }),
        first_tool_call_only: false,
        parser_names: None,
    };
    let (tx, mut rx) = sse_channel();
    let result = processor(scenario.with_tools)
        .process_messages_streaming_chunks(
            stream,
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, true),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let mut bytes = Vec::new();
    while let Some(frame) = rx.recv().await {
        bytes.extend_from_slice(&frame.expect("successful SSE write"));
    }
    server.abort();
    assert!(result.is_ok(), "{}: {result:?}", scenario.name);
    bytes
}

#[tokio::test]
async fn messages_stream_bytes_match_the_recorded_fixtures() {
    for scenario in scenarios() {
        let actual = stream_bytes(&scenario).await;
        assert_recorded_bytes(scenario.name, scenario.must_contain, &actual);
    }
}
