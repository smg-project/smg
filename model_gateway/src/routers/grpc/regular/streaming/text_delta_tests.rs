//! The text delta a chat stream writes without the serializer carries the
//! bytes the serializer would have written: the same chunk, pinned against
//! [`ChatChunkFrame::write_chunk`] over a [`ChatStreamChoiceRef`] for every
//! field, index, usage shape and the texts a model can produce.

use std::borrow::Cow;

use openai_protocol::{
    chat::{ChatMessageDeltaRef, ChatStreamChoiceRef},
    common::ChatLogProbsContent,
};
use symphony::Text;

use super::*;

/// Texts with every escape the JSON serializer knows, the bytes it leaves
/// alone (`/`, DEL, the non-ASCII planes) and the shapes a token produces.
fn texts() -> Vec<String> {
    let mut texts = vec![
        String::new(),
        " ".to_string(),
        "Hello".to_string(),
        " world,".to_string(),
        "say \"hi\"".to_string(),
        "back\\slash".to_string(),
        "tab\there".to_string(),
        "line\nbreak\r\n".to_string(),
        "form\u{0C}feed back\u{08}space".to_string(),
        "a/b".to_string(),
        "del\u{7F}char".to_string(),
        "caf\u{E9} na\u{EF}ve".to_string(),
        "\u{4E2D}\u{6587}".to_string(),
        "\u{1F600}\u{1F3FD}".to_string(),
        "e\u{301}".to_string(),
        "\u{2028}\u{2029}".to_string(),
        "{\"json\": [1, 2]}".to_string(),
        "x".repeat(5000),
        "\"".repeat(64),
    ];
    // Every C0 control on its own and between letters.
    for code in 0u8..0x20 {
        let c = char::from(code);
        texts.push(c.to_string());
        texts.push(format!("a{c}b"));
    }
    texts
}

fn frame() -> ChatChunkFrame {
    ChatChunkFrame::new(
        "chatcmpl-test",
        "fixture-model",
        1_700_000_000,
        Some("fp_1"),
    )
}

fn usage() -> Usage {
    Usage::from_counts(3, 2).with_cached_tokens(1)
}

fn serialized(
    frame: &ChatChunkFrame,
    index: u32,
    field: TextDeltaField,
    text: &str,
    usage: Option<&Usage>,
    emit_usage_null: bool,
) -> Vec<u8> {
    let (content, reasoning_content) = match field {
        TextDeltaField::Content => (Some(Cow::Borrowed(text)), None),
        TextDeltaField::Reasoning => (None, Some(Cow::Borrowed(text))),
    };
    let choice = ChatStreamChoiceRef {
        index,
        delta: ChatMessageDeltaRef {
            role: None,
            content,
            tool_calls: None,
            reasoning_content,
        },
        logprobs: None,
        finish_reason: None,
        matched_stop: None,
    };
    let mut buffer = BytesMut::new();
    frame.write_chunk(&mut buffer, &choice, usage, emit_usage_null);
    buffer.to_vec()
}

fn written(
    frame: &ChatChunkFrame,
    index: u32,
    field: TextDeltaField,
    text: &str,
    usage: Option<&Usage>,
    emit_usage_null: bool,
) -> Vec<u8> {
    let mut buffer = BytesMut::new();
    frame.write_text_delta(&mut buffer, index, field, text, usage, emit_usage_null);
    buffer.to_vec()
}

#[test]
fn text_delta_bytes_equal_the_serialized_choice() {
    let frame = frame();
    let usage = usage();
    let usages: [(Option<&Usage>, bool); 3] = [(None, false), (None, true), (Some(&usage), false)];
    for text in texts() {
        for field in [TextDeltaField::Content, TextDeltaField::Reasoning] {
            for index in [0u32, 1, 7, 10, 4_294_967_295] {
                for (usage, emit_usage_null) in usages {
                    let expected = serialized(&frame, index, field, &text, usage, emit_usage_null);
                    let actual = written(&frame, index, field, &text, usage, emit_usage_null);
                    assert_eq!(
                        String::from_utf8_lossy(&actual),
                        String::from_utf8_lossy(&expected),
                        "field {field:?}, index {index}, usage {:?}/{emit_usage_null}, text {text:?}",
                        usage.is_some()
                    );
                }
            }
        }
    }
}

#[test]
fn a_text_delta_is_appended_after_what_the_buffer_holds() {
    let frame = frame();
    let mut buffer = BytesMut::from(&b"data: earlier\n\n"[..]);
    frame.write_text_delta(&mut buffer, 2, TextDeltaField::Content, "x", None, true);
    let mut expected = b"data: earlier\n\n".to_vec();
    expected.extend(serialized(
        &frame,
        2,
        TextDeltaField::Content,
        "x",
        None,
        true,
    ));
    assert_eq!(buffer.to_vec(), expected);
}

#[test]
fn json_string_contents_escape_as_the_serializer_does() {
    for text in texts() {
        let mut buffer = BytesMut::new();
        push_json_string_contents(&mut buffer, &text);
        let quoted = serde_json::to_string(&text).expect("a string serializes");
        assert_eq!(
            String::from_utf8_lossy(&buffer),
            quoted[1..quoted.len() - 1],
            "text {text:?}"
        );
    }
}

#[test]
fn decimal_digits_are_the_number_s() {
    for value in [0u32, 1, 9, 10, 99, 100, 12_345, 4_294_967_295] {
        let mut buffer = BytesMut::new();
        push_decimal(&mut buffer, value);
        assert_eq!(String::from_utf8_lossy(&buffer), value.to_string());
    }
}

/// The logprobs a chunk carries: a detailed entry for one token.
fn logprobs() -> ChatLogProbs {
    ChatLogProbs::Detailed {
        content: Some(vec![ChatLogProbsContent {
            token: "why".to_string(),
            logprob: -0.25,
            bytes: None,
            top_logprobs: Vec::new(),
        }]),
        refusal: None,
    }
}

/// The serializer's bytes for one text delta of choice `index`, with or
/// without logprobs: what `push_chunk` wrote before the direct form existed.
fn serialized_delta(
    frame: &ChatChunkFrame,
    index: u32,
    field: TextDeltaField,
    text: &str,
    logprobs: Option<ChatLogProbs>,
) -> Vec<u8> {
    let (content, reasoning_content) = match field {
        TextDeltaField::Content => (Some(Cow::Borrowed(text)), None),
        TextDeltaField::Reasoning => (None, Some(Cow::Borrowed(text))),
    };
    let choice = ChatStreamChoiceRef {
        index,
        delta: ChatMessageDeltaRef {
            role: None,
            content,
            tool_calls: None,
            reasoning_content,
        },
        logprobs,
        finish_reason: None,
        matched_stop: None,
    };
    let mut buffer = BytesMut::new();
    frame.write_chunk(&mut buffer, &choice, None, false);
    buffer.to_vec()
}

/// With logprobs on, the first text delta of a step carries them (through the
/// serializer), whichever kind it is, and the deltas after it, content and
/// reasoning alike, are written in the direct form: byte for byte what the
/// serializer wrote for every one of them.
#[tokio::test]
async fn push_events_puts_the_logprobs_on_the_first_text_delta_and_writes_the_serializer_s_bytes() {
    let frame = frame();
    let (tx, _rx) = sse_channel();
    // `Events` merges consecutive text of one kind and drops empty text, so
    // the kinds alternate here: three deltas, the first of them reasoning.
    let mut events = Events::new();
    events.push_content(Text::new("", 0));
    events.push_reasoning(Text::new("why", 1));
    events.push_content(Text::new("Hello \"there\"\n", 2));
    events.push_reasoning(Text::new("", 0));
    events.push_reasoning(Text::new("because", 1));
    let mut buffer = BytesMut::new();
    StreamingProcessor::push_events(
        &tx,
        &mut buffer,
        &frame,
        3,
        &mut events,
        Some(logprobs()),
        None,
        false,
    )
    .await
    .expect("the events are written");

    let mut expected = serialized_delta(
        &frame,
        3,
        TextDeltaField::Reasoning,
        "why",
        Some(logprobs()),
    );
    expected.extend(serialized_delta(
        &frame,
        3,
        TextDeltaField::Content,
        "Hello \"there\"\n",
        None,
    ));
    expected.extend(serialized_delta(
        &frame,
        3,
        TextDeltaField::Reasoning,
        "because",
        None,
    ));
    assert_eq!(
        String::from_utf8_lossy(&buffer),
        String::from_utf8_lossy(&expected)
    );
    assert!(
        String::from_utf8_lossy(&buffer)
            .matches("\"logprobs\":{")
            .count()
            == 1,
        "exactly the first text delta carries the logprobs"
    );
}

/// Without logprobs every text delta takes the direct form, and the empty
/// texts write nothing.
#[tokio::test]
async fn push_events_without_logprobs_writes_every_text_delta_in_the_direct_form() {
    let frame = frame();
    let (tx, _rx) = sse_channel();
    let mut events = Events::new();
    events.push_content(Text::new("a", 1));
    events.push_content(Text::new("", 0));
    events.push_reasoning(Text::new("b", 1));
    let mut buffer = BytesMut::new();
    StreamingProcessor::push_events(&tx, &mut buffer, &frame, 0, &mut events, None, None, true)
        .await
        .expect("the events are written");
    let mut expected = written(&frame, 0, TextDeltaField::Content, "a", None, true);
    expected.extend(written(
        &frame,
        0,
        TextDeltaField::Reasoning,
        "b",
        None,
        true,
    ));
    assert_eq!(buffer.to_vec(), expected);
}
