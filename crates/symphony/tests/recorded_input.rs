//! Diagnostic input ownership and replay through the real parser, without gateway claims.

#![expect(
    clippy::expect_used,
    reason = "test helpers report invalid fixtures and parser failures"
)]

use std::{
    io::{self, Write},
    sync::Arc,
};

use serde_json::json;
use symphony::{
    formats, CallIds, CallSyntax, Declared, Engine, EngineFinish, Event, Events, Input, Parser,
    RecordedInput, TokenSpan,
};

fn round_trip(input: &Input<'_>) -> RecordedInput {
    let recorded = RecordedInput::from(input);
    let mut json = Vec::new();
    serde_json::to_writer(&mut json, input).expect("serialize borrowed input");
    assert_eq!(
        json,
        serde_json::to_vec(&recorded).expect("serialize owned input")
    );
    let restored: RecordedInput = serde_json::from_slice(&json).expect("deserialize input");
    assert_eq!(restored, recorded);
    assert_eq!(RecordedInput::from(&restored.as_input()), recorded);
    restored
}

#[test]
fn snapshots_own_the_exact_prompt_after_its_buffers_are_dropped() {
    let recorded = {
        let text = String::from("<think>réfléchir\n");
        let tokens = vec![7, 8];
        round_trip(&Input::Prompt {
            token_ids: &tokens,
            text: &text,
        })
    };
    assert_eq!(
        serde_json::to_value(recorded).expect("json"),
        json!({
            "kind": "prompt", "token_ids": [7, 8], "text": "<think>réfléchir\n"
        })
    );
    let empty = round_trip(&Input::Prompt {
        token_ids: &[],
        text: "<think>",
    });
    assert!(matches!(empty, RecordedInput::Prompt { token_ids, .. } if token_ids.is_empty()));
}

#[test]
fn held_tokens_and_continued_unicode_bytes_keep_their_alignment() {
    let first = Input::Delta {
        token_ids: &[4, 5],
        text: "é",
        spans: &[
            TokenSpan {
                token_id: 4,
                start: 0,
                end: 2,
                continued: false,
            },
            TokenSpan {
                token_id: 5,
                start: 2,
                end: 2,
                continued: false,
            },
        ],
    };
    round_trip(&first);
    let tail = round_trip(&Input::Delta {
        token_ids: &[],
        text: "<",
        spans: &[TokenSpan {
            token_id: 5,
            start: 0,
            end: 1,
            continued: true,
        }],
    });
    assert_eq!(
        serde_json::to_value(tail).expect("json"),
        json!({
            "kind": "delta", "token_ids": [], "text": "<",
            "spans": [{"token_id": 5, "start": 0, "end": 1, "continued": true}]
        })
    );
}

#[test]
fn malformed_alignment_is_evidence_and_is_not_repaired_by_recording() {
    round_trip(&Input::Delta {
        token_ids: &[99],
        text: "é",
        spans: &[TokenSpan {
            token_id: 7,
            start: 1,
            end: 9,
            continued: true,
        }],
    });
    round_trip(&Input::Delta {
        token_ids: &[],
        text: "",
        spans: &[],
    });
}

#[test]
fn all_finish_reasons_survive_including_unknown_engine_text() {
    for (finish, expected) in [
        (EngineFinish::Stop, json!("stop")),
        (EngineFinish::Length, json!("length")),
        (EngineFinish::Abort, json!("abort")),
        (
            EngineFinish::Other("worker/停止\0".into()),
            json!({"other": "worker/停止\0"}),
        ),
    ] {
        let recorded = round_trip(&Input::End { finish });
        assert_eq!(
            serde_json::to_value(recorded).expect("finish json"),
            json!({"kind": "end", "finish": expected}),
        );
    }
}

#[test]
fn missing_alignment_and_unknown_fields_do_not_silently_disappear() {
    for record in [
        json!({"kind": "delta", "token_ids": [1], "text": "x"}),
        json!({"kind": "prompt", "token_ids": [], "text": "", "extra": true}),
        json!({"kind": "delta", "token_ids": [1], "text": "x", "spans": [
            {"token_id": 1, "start": 0, "end": 1, "continued": false, "extra": true}
        ]}),
    ] {
        assert!(serde_json::from_value::<RecordedInput>(record).is_err());
    }
}

fn engine() -> Engine {
    let tools = serde_json::from_value(json!([{
        "type": "function", "function": {
            "name": "echo", "parameters": {"type": "object", "properties": {
                "text": {"type": "string"}
            }}
        }
    }]))
    .expect("declared tools");
    Engine::with_call_ids(
        Arc::new(formats::qwen3(CallSyntax::Json)),
        Declared::shared(tools),
        CallIds::from_nonce("0123456789abcdef00000000").expect("fixed replay nonce"),
    )
}

fn feed(engine: &mut Engine, input: Input<'_>) -> Vec<Event<'static>> {
    let mut events = Events::new();
    engine.feed(input, &mut events).expect("feed input");
    events.into_owned().drain()
}

#[test]
fn serialized_inputs_reproduce_each_consumed_event_batch() {
    let mut original = engine();
    let mut records = Vec::new();
    let mut expected = Vec::new();
    let prompt = Input::Prompt {
        token_ids: &[],
        text: "<|im_start|>assistant\n<think>",
    };
    records.push(round_trip(&prompt));
    expected.push(feed(&mut original, prompt));
    for (index, text) in [
        "réflé",
        "chir</think><tool_",
        "call>{\"name\":\"echo\",\"arguments\":{\"text\":\"é",
        "\"}}</tool_call>",
    ]
    .into_iter()
    .enumerate()
    {
        let tokens = [index as u32 + 100];
        let spans = [TokenSpan {
            token_id: tokens[0],
            start: 0,
            end: text.len(),
            continued: false,
        }];
        let input = Input::Delta {
            token_ids: &tokens,
            text,
            spans: &spans,
        };
        records.push(round_trip(&input));
        expected.push(feed(&mut original, input));
    }
    let end = Input::End {
        finish: EngineFinish::Stop,
    };
    records.push(round_trip(&end));
    expected.push(feed(&mut original, end));
    let mut replay = engine();
    let actual: Vec<_> = records
        .iter()
        .map(|record| feed(&mut replay, record.as_input()))
        .collect();
    assert_eq!(actual, expected);
    assert!(actual
        .iter()
        .flatten()
        .any(|event| matches!(event, Event::Reasoning(_))));
    assert!(actual
        .iter()
        .flatten()
        .any(|event| matches!(event, Event::ToolCallStart { .. })));
    assert!(matches!(
        actual.last().and_then(|batch| batch.last()),
        Some(Event::Finish { .. })
    ));
}

/// Exercise partial writes and a storage error without an intermediate whole-record buffer.
struct CappedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for CappedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let n = bytes.len().min(3).min(self.limit - self.bytes.len());
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "capture budget exhausted",
            ));
        }
        self.bytes.extend_from_slice(&bytes[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn borrowed_input_survives_short_writes_and_unicode_escaping() {
    let input = Input::Delta {
        token_ids: &[7],
        text: "é\n\0\"",
        spans: &[TokenSpan {
            token_id: 7,
            start: 0,
            end: 5,
            continued: false,
        }],
    };
    let mut writer = CappedWriter {
        bytes: Vec::new(),
        limit: 512,
    };
    serde_json::to_writer(&mut writer, &input).expect("write input in short pieces");
    let restored: RecordedInput = serde_json::from_slice(&writer.bytes).expect("decode input");
    assert_eq!(restored, RecordedInput::from(&input));
}

#[test]
fn bounded_writers_refuse_large_borrowed_fields_without_a_complete_record() {
    let text = "é\n".repeat(128 * 1024);
    let tokens = vec![u32::MAX; 16 * 1024];
    let spans = vec![
        TokenSpan {
            token_id: 1,
            start: 0,
            end: 0,
            continued: true
        };
        8 * 1024
    ];
    for input in [
        Input::Prompt {
            token_ids: &[],
            text: &text,
        },
        Input::Prompt {
            token_ids: &tokens,
            text: "",
        },
        Input::Delta {
            token_ids: &[],
            text: "",
            spans: &spans,
        },
        Input::End {
            finish: EngineFinish::Other("stop\0".repeat(128 * 1024)),
        },
    ] {
        let mut writer = CappedWriter {
            bytes: Vec::new(),
            limit: 64,
        };
        let error =
            serde_json::to_writer(&mut writer, &input).expect_err("bounded write must fail");
        assert_eq!(error.io_error_kind(), Some(io::ErrorKind::StorageFull));
        assert_eq!(writer.bytes.len(), writer.limit);
        assert!(serde_json::from_slice::<RecordedInput>(&writer.bytes).is_err());
    }
}
