//! Synthetic Reader boundary tests; these do not qualify a tokenizer/model deployment.

use std::{
    fs,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use llm_tokenizer::{stop::StopSequenceDecoder, StopSequenceConfig};
use serde_json::{json, Value};
use symphony::{adapt::chat, formats, CallSyntax, Format, RecordedInput};

use super::{
    super::{Delta, Reader},
    *,
};
use crate::{
    diagnostic_capture::{
        CaptureLimits, CapturePool, CaptureReadLimits, CaptureSnapshot, SharedCapture,
    },
    routers::grpc::regular::symphony::testing::{tools, who, Refusing, PROMPT},
};

async fn capture(
    record_bytes: usize,
    file_bytes: u64,
) -> (tempfile::TempDir, SharedCapture, CapturePool) {
    let root = tempfile::tempdir().unwrap();
    let pool = CapturePool::new(
        root.path().to_owned(),
        CaptureLimits {
            max_active: 1,
            max_attempts: 1,
            max_retained_bytes: file_bytes,
            file_bytes,
            record_bytes,
            queue_capacity: 256,
        },
    )
    .unwrap();
    let owner = SharedCapture::new(pool.start().await.unwrap());
    (root, owner, pool)
}

async fn records(owner: SharedCapture) -> Vec<Value> {
    let directory = owner.finish().await.unwrap();
    let snapshot = CaptureSnapshot::read(
        &directory,
        CaptureReadLimits {
            file_bytes: 1_000_000,
            record_bytes: 100_000,
            records: 256,
        },
    )
    .unwrap();
    let records: Vec<Value> = snapshot
        .payloads()
        .map(|payload| serde_json::from_str(payload).unwrap())
        .collect();
    for (ordinal, record) in records.iter().enumerate() {
        assert_eq!(record["schema"], "smg.symphony_reader.v2");
        assert_eq!(record["source"], 7);
        assert_eq!(record["choice"], 0);
        assert_eq!(record["ordinal"], ordinal);
    }
    records
}

fn assignment(format: Format, prompt: &str) -> Assignment {
    Assignment {
        format: Arc::new(format),
        prompt: prompt.to_owned(),
        recipe: None,
    }
}

fn text_assignment() -> Assignment {
    assignment(
        Format::new("text").state("content", symphony::Emits::Content),
        "",
    )
}

fn nonce() -> CallIds {
    CallIds::from_nonce("ffffffffffffffffffffffff").unwrap()
}

fn observed(assignment: &Assignment, owner: &SharedCapture, out: &mut Events<'_>) -> Reader {
    Reader::observed(
        assignment,
        tools().into(),
        nonce(),
        out,
        owner.observer(),
        7,
        0,
    )
    .unwrap()
}

#[tokio::test]
async fn reader_capture_binds_nonce_prompt_tools_and_actual_policy_order() {
    for (format, prompt, output) in [
        (
            formats::json(),
            "",
            "{\"name\":\"lookup\",\"arguments\":{}};{\"name\":\"lookup\",\"arguments\":{}}",
        ),
        (
            formats::qwen3(CallSyntax::Json),
            PROMPT,
            "plan</think><tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>",
        ),
    ] {
        let (_root, owner, _pool) = capture(100_000, 1_000_000).await;
        let assignment = assignment(format, prompt);
        let mut events = Events::new();
        let mut reader = observed(&assignment, &owner, &mut events).first_call_only(true);
        let mut delta = Delta::default();
        reader
            .tail(output.to_owned(), &mut delta, &mut events)
            .unwrap();
        reader.end(EngineFinish::Stop, who(), &mut events).unwrap();
        let choice = chat::message(0, events.as_slice());
        assert_eq!(choice.message.tool_calls.unwrap()[0].id, nonce().id(0));
        reader.observe_chat_batch(0, &events);
        let expected = serde_json::to_value(events.as_slice()).unwrap();
        drop(reader);
        let records = records(owner).await;
        assert_eq!(records[0]["nonce"], nonce().nonce());
        assert_eq!(records[0]["tools"], serde_json::to_value(tools()).unwrap());
        assert_eq!(records[0]["first_call_only"], false);
        assert_eq!(
            records[1]["input"],
            json!({"kind":"prompt", "token_ids":[], "text":prompt})
        );
        assert_eq!(records[2]["operation"], "construct");
        assert_eq!(records[3]["kind"], "policy");
        assert_eq!(records[3]["first_call_only"], true);
        let batch = records.iter().find(|r| r["kind"] == "chat_batch").unwrap();
        assert_eq!(batch["events"], expected);
        assert_eq!(records.last().unwrap()["finished"], true);
        for record in records.iter().filter(|r| r["kind"] == "input") {
            let _: RecordedInput = serde_json::from_value(record["input"].clone()).unwrap();
        }
    }
}

#[tokio::test]
async fn reader_capture_reconstructs_bound_selector_before_prompt() {
    use crate::{
        config::ParserChoice,
        routers::grpc::regular::symphony::{assign, testing, Asks},
    };

    let request = testing::request(json!({
        "model": "skt/A.X-3.1-Light", "tools": tools()
    }));
    let assigned = assign(
        ParserChoice::Symphony,
        &testing::names(None, None),
        Asks::Chat(&request),
        None,
        "prefix<|im_start|><|assistant|>",
        &testing::templates(),
    )
    .unwrap()
    .unwrap();
    drop(request);
    let output = "<tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>";
    let (_root, owner, _pool) = capture(100_000, 1_000_000).await;
    let mut events = Events::new();
    let mut reader = observed(&assigned, &owner, &mut events).first_call_only(true);
    let mut delta = Delta::default();
    reader
        .tail(output.to_owned(), &mut delta, &mut events)
        .unwrap();
    reader.end(EngineFinish::Stop, who(), &mut events).unwrap();
    reader.observe_chat_batch(0, &events);
    let expected = serde_json::to_value(events.as_slice()).unwrap();
    let message = serde_json::to_value(chat::message(0, events.as_slice())).unwrap();
    assert_eq!(
        message["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
    drop(reader);
    let records = records(owner).await;
    assert_eq!(records[0]["kind"], "open");
    assert_eq!(records[0]["recipe"]["selector_model"], "skt/A.X-3.1-Light");
    assert_eq!(records[1]["kind"], "input");
    assert_eq!(records[1]["input"]["kind"], "prompt");
    let recipe: FormatRecipe =
        serde_json::from_value(records[0]["recipe"].clone()).unwrap();
    let replay_assignment = Assignment {
        format: recipe.resolve().unwrap(),
        prompt: records[1]["input"]["text"].as_str().unwrap().to_owned(),
        recipe: Some(recipe),
    };
    let retained_tools: Vec<Tool> = serde_json::from_value(records[0]["tools"].clone()).unwrap();
    let retained_ids = CallIds::from_nonce(records[0]["nonce"].as_str().unwrap()).unwrap();
    let mut replay_events = Events::new();
    let mut replay = Reader::with_call_ids(
        &replay_assignment,
        retained_tools.into(),
        retained_ids,
        &mut replay_events,
    )
    .unwrap()
    .first_call_only(true);
    let mut replay_delta = Delta::default();
    replay
        .tail(output.to_owned(), &mut replay_delta, &mut replay_events)
        .unwrap();
    replay
        .end(EngineFinish::Stop, who(), &mut replay_events)
        .unwrap();
    assert_eq!(
        serde_json::to_value(replay_events.as_slice()).unwrap(),
        expected
    );
    assert_eq!(
        serde_json::to_value(chat::message(0, replay_events.as_slice())).unwrap(),
        message
    );
}

#[tokio::test]
async fn reader_capture_keeps_coalesced_batches_and_end_releases() {
    for (assignment, pieces, expected_text) in [
        (text_assignment(), vec!["hello", " world"], "hello world"),
        (
            assignment(formats::qwen3(CallSyntax::Json), ""),
            vec![" \n\t"],
            " \n\t",
        ),
        (text_assignment(), vec![], ""),
    ] {
        let (_root, owner, _pool) = capture(100_000, 1_000_000).await;
        let mut events = Events::new();
        let mut reader = observed(&assignment, &owner, &mut events);
        // Keep every delta alive while the one complete adapter batch borrows them.
        let mut deltas: Vec<_> = pieces.iter().map(|_| Delta::default()).collect();
        for (piece, delta) in pieces.iter().zip(&mut deltas) {
            reader
                .tail((*piece).to_owned(), delta, &mut events)
                .unwrap();
        }
        reader.end(EngineFinish::Stop, who(), &mut events).unwrap();
        let content: String = events
            .as_slice()
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(content, expected_text);
        if pieces.len() == 2 {
            assert_eq!(
                events
                    .as_slice()
                    .iter()
                    .filter(|e| matches!(e, Event::Content(_)))
                    .count(),
                1
            );
        }
        let expected = serde_json::to_value(events.as_slice()).unwrap();
        reader.observe_chat_batch(0, &events);
        drop(reader);
        let records = records(owner).await;
        let batches: Vec<_> = records
            .iter()
            .filter(|r| r["kind"] == "chat_batch")
            .collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0]["events"], expected);
        assert_eq!(records.last().unwrap()["batches"], 1);
    }
}

#[tokio::test]
async fn reader_capture_distinguishes_tokens_tail_stop_and_exact_inputs() {
    for stop in [false, true] {
        let (_root, owner, _pool) = capture(100_000, 1_000_000).await;
        let mut events = Events::new();
        let mut reader = observed(&text_assignment(), &owner, &mut events);
        let config = StopSequenceConfig::default().with_stop_sequence("STOP");
        let mut decoder = StopSequenceDecoder::new(Arc::new(Refusing::default()), config, false);
        let text = if stop { "héSTOPignored" } else { "héST" };
        let tokens: Vec<_> = text.chars().map(u32::from).collect();
        let mut delta = Delta::default();
        assert_eq!(
            reader
                .read(&mut decoder, &tokens, &mut delta, &mut events)
                .unwrap(),
            stop
        );
        // End the event buffer's borrow before inspecting the decoded input itself.
        let mut events: Events<'_> = events.into_owned();
        let expected_input = RecordedInput::from(&delta.input());
        assert_eq!(delta.text, "hé");
        let mut tail = Delta::default();
        if !stop {
            let flushed = decoder.finish().unwrap();
            let llm_tokenizer::stop::SequenceDecoderOutput::Text(text) = flushed else {
                panic!("expected held tail")
            };
            assert_eq!(text, "ST");
            reader.tail(text, &mut tail, &mut events).unwrap();
        }
        reader.end(EngineFinish::Stop, who(), &mut events).unwrap();
        reader.observe_chat_batch(0, &events);
        drop(reader);
        let records = records(owner).await;
        let read = records
            .iter()
            .find(|r| r["kind"] == "operation" && r["operation"] == "read")
            .unwrap();
        assert_eq!(read["token_ids"], json!(tokens));
        let input = records
            .iter()
            .filter(|r| r["kind"] == "input")
            .nth(1)
            .unwrap();
        assert_eq!(
            serde_json::from_value::<RecordedInput>(input["input"].clone()).unwrap(),
            expected_input
        );
        let returned = records
            .iter()
            .find(|r| r["kind"] == "returned" && r["operation"] == "read")
            .unwrap();
        assert_eq!(returned["stopped"], stop);
        if !stop {
            let input = records
                .iter()
                .filter(|r| r["kind"] == "input")
                .nth(2)
                .unwrap();
            assert_eq!(input["input"]["token_ids"], json!([]));
            assert_eq!(input["input"]["spans"][0]["continued"], true);
            assert_eq!(input["input"]["text"], "ST");
        }
    }
}

#[tokio::test]
async fn reader_capture_retains_decoder_and_feed_errors_without_replacing_them() {
    for decoder_error in [false, true] {
        let (_root, owner, _pool) = capture(100_000, 1_000_000).await;
        let mut results = Vec::new();
        for enabled in [false, true] {
            let mut events = Events::new();
            let mut reader = if enabled {
                observed(&text_assignment(), &owner, &mut events)
            } else {
                Reader::with_call_ids(&text_assignment(), tools().into(), nonce(), &mut events)
                    .unwrap()
            };
            let mut delta = Delta::default();
            let error = if decoder_error {
                let mut decoder = StopSequenceDecoder::new(
                    Arc::new(Refusing::default()),
                    StopSequenceConfig::default(),
                    false,
                );
                reader
                    .read(
                        &mut decoder,
                        &[u32::from('H'), Refusing::REFUSED],
                        &mut delta,
                        &mut events,
                    )
                    .unwrap_err()
            } else {
                reader.end(EngineFinish::Stop, who(), &mut events).unwrap();
                reader
                    .tail("after end".to_owned(), &mut delta, &mut events)
                    .unwrap_err()
            };
            results.push((
                error.to_string(),
                serde_json::to_value(events.as_slice()).unwrap(),
            ));
        }
        assert_eq!(results[0], results[1]);
        let records = records(owner).await;
        let failed = records.iter().position(|r| r["kind"] == "failed").unwrap();
        assert_eq!(records[failed]["message"], results[0].0);
        if decoder_error {
            assert_eq!(
                records.iter().filter(|r| r["kind"] == "input").count(),
                1,
                "no decoded input was fed"
            );
            assert_eq!(records.last().unwrap()["finished"], false);
        } else {
            assert_eq!(records[failed - 1]["kind"], "input");
            assert_eq!(records[failed - 1]["input"]["text"], "after end");
        }
        assert_eq!(records.last().unwrap()["failed"], true);
    }
}

#[tokio::test]
async fn reader_capture_retains_unwinding_prefix_without_inventing_end() {
    let (_root, owner, _pool) = capture(100_000, 1_000_000).await;
    let observer = owner.observer();
    let result = catch_unwind(AssertUnwindSafe(move || {
        let mut events = Events::new();
        let mut reader = Reader::observed(
            &text_assignment(),
            tools().into(),
            nonce(),
            &mut events,
            observer,
            7,
            0,
        )
        .unwrap();
        reader.trap = Some("boom");
        let mut decoder = StopSequenceDecoder::new(
            Arc::new(Refusing::default()),
            StopSequenceConfig::default(),
            false,
        );
        let mut delta = Delta::default();
        reader
            .read(
                &mut decoder,
                &"boom".chars().map(u32::from).collect::<Vec<_>>(),
                &mut delta,
                &mut events,
            )
            .unwrap();
    }));
    assert!(result.is_err());
    let records = records(owner).await;
    let closed = records.last().unwrap();
    assert_eq!(closed["unwinding"], true);
    assert_eq!(closed["pending"], "read");
    assert_eq!(closed["finished"], false);
    assert_eq!(closed["inputs"], 1, "only Prompt reached the engine");
    assert!(!records.iter().any(|r| r["input"]["kind"] == "end"));
}

#[tokio::test]
async fn reader_capture_rejects_wrong_choice_and_abandoned_owner() {
    for abandoned in [false, true] {
        let (_root, owner, _pool) = capture(100_000, 1_000_000).await;
        let directory = owner.directory().to_owned();
        let mut events = Events::new();
        let mut reader = observed(&text_assignment(), &owner, &mut events);
        let mut owner = Some(owner);
        if abandoned {
            drop(owner.take());
        }
        let mut delta = Delta::default();
        reader
            .tail("still served".to_owned(), &mut delta, &mut events)
            .unwrap();
        reader.end(EngineFinish::Stop, who(), &mut events).unwrap();
        reader.observe_chat_batch(if abandoned { 0 } else { 1 }, &events);
        assert_eq!(
            chat::message(0, events.as_slice())
                .message
                .content
                .as_deref(),
            Some("still served")
        );
        drop(reader);
        if let Some(owner) = owner {
            assert!(owner.finish().await.is_err());
            let prefix: Vec<Value> = fs::read_to_string(directory.join("records.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let rejected = &prefix.last().unwrap()["payload"];
            assert_eq!(rejected["kind"], "rejected");
            assert_eq!(
                rejected["reason"],
                "chat batch choice differs from reader choice"
            );
            assert_eq!(rejected["attempted_choice"], 1);
            assert_eq!(rejected["choice"], 0);
        }
        assert!(!fs::exists(directory.join("complete.json")).unwrap());
    }
}
