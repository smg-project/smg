//! Actual streaming Chat adapter consumption, with synthetic Reader output.

use super::*;
use crate::diagnostic_capture::{
    CaptureLimits, CaptureObserver, CapturePool, CaptureReadLimits, CaptureSnapshot, SharedCapture,
};

const PROMPT: &str = "<|im_start|>assistant\n<think>\n";

fn tools() -> Vec<Tool> {
    serde_json::from_value(serde_json::json!([{
        "type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object", "properties": {}}
        }
    }]))
    .expect("valid declared tool fixture")
}

fn who() -> Who<'static> {
    Who {
        request_id: "test-request",
        model: "test-model",
        choice: 0,
    }
}

async fn render(
    observer: Option<CaptureObserver>,
    first_call_only: bool,
    disconnected: bool,
) -> (Vec<u8>, Option<String>, Vec<Value>) {
    let assignment = Assignment {
        format: Arc::new(::symphony::formats::qwen3(::symphony::CallSyntax::Json)),
        prompt: PROMPT.to_owned(),
        recipe: None,
    };
    let ids = ::symphony::CallIds::from_nonce("0123456789abcdef00000000").expect("recorded nonce");
    let mut events = Events::new();
    let mut reader = match observer {
        Some(observer) => Reader::observed(
            &assignment,
            tools().into(),
            ids,
            &mut events,
            observer,
            7,
            0,
        ),
        None => Reader::with_call_ids(&assignment, tools().into(), ids, &mut events),
    }
    .expect("construct reader")
    .first_call_only(first_call_only);
    let (tx, rx) = sse_channel();
    let mut rx = if disconnected {
        drop(rx);
        None
    } else {
        Some(rx)
    };
    let frame = ChatChunkFrame::new("test-id", "test-model", 1, None);
    let mut buffer = BytesMut::new();
    let mut batches = Vec::new();
    let mut error = None;
    let mut pieces = vec![
        "plan</think>Hello".to_owned(),
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":1}}</tool_call><tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":2}}</tool_call>".to_owned(),
    ];
    if disconnected {
        pieces[0].push_str(&"x".repeat(CHUNK_FLUSH_BYTES * 2));
    }
    let mut deltas: Vec<_> = pieces.iter().map(|_| Delta::default()).collect();
    for (piece, delta) in pieces.into_iter().zip(&mut deltas) {
        reader
            .tail(piece, delta, &mut events)
            .expect("read synthetic output");
        batches.push(serde_json::to_value(events.as_slice()).expect("serialize adapter batch"));
        if let Err(why) = StreamingProcessor::push_events(
            &tx,
            &mut buffer,
            &frame,
            0,
            &mut events,
            Some(&mut reader),
            None,
            None,
            false,
        )
        .await
        {
            error = Some(why);
            break;
        }
    }
    if error.is_none() {
        reader
            .end(::symphony::EngineFinish::Stop, who(), &mut events)
            .expect("finish reader");
        batches
            .push(serde_json::to_value(events.as_slice()).expect("serialize final adapter batch"));
        StreamingProcessor::push_events(
            &tx,
            &mut buffer,
            &frame,
            0,
            &mut events,
            Some(&mut reader),
            None,
            None,
            false,
        )
        .await
        .expect("adapt final batch");
        StreamingProcessor::flush_chunks(&tx, &mut buffer)
            .await
            .expect("flush synthetic frames");
    }
    drop(tx);
    let mut wire = Vec::new();
    if let Some(rx) = &mut rx {
        while let Some(frame) = rx.recv().await {
            wire.extend_from_slice(&frame.expect("successful synthetic frame"));
        }
    }
    (wire, error, batches)
}

async fn owner(
    record_bytes: usize,
    file_bytes: u64,
) -> (tempfile::TempDir, CapturePool, SharedCapture) {
    let root = tempfile::tempdir().expect("capture root");
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
    .expect("bounded capture pool");
    let owner = SharedCapture::new(pool.start().await.expect("admit capture session"));
    (root, pool, owner)
}

#[tokio::test]
async fn reader_capture_chat_adapter_preserves_wire_on_capture_failure() {
    for first_call_only in [false, true] {
        let original = render(None, first_call_only, false).await;
        assert!(!original.0.is_empty());
        assert!(original.1.is_none());
        for (record_bytes, file_bytes, complete) in [
            (100_000, 1_000_000, true),
            (64, 1_000_000, false),
            (100_000, 600, false),
            (512, 1_000_000, false),
            (100_000, 1600, false),
        ] {
            let (_root, _pool, owner) = owner(record_bytes, file_bytes).await;
            let directory = owner.directory().to_owned();
            let observed = render(Some(owner.observer()), first_call_only, false).await;
            assert_eq!(
                observed, original,
                "capture limits {record_bytes}/{file_bytes}"
            );
            let receipt = owner.finish().await;
            assert_eq!(receipt.is_ok(), complete);
            assert_eq!(directory.join("complete.json").exists(), complete);
            if record_bytes == 512 || file_bytes == 1600 {
                // These budgets accept Open/Prompt before a later record or file write fails.
                // Check the retained prefix as well as parity, so early admission failure
                // cannot accidentally satisfy this partial-capture regression.
                let prefix: Vec<Value> = std::fs::read_to_string(directory.join("records.jsonl"))
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str::<Value>(line).unwrap()["payload"].clone())
                    .collect();
                assert_eq!(prefix[0]["kind"], "open");
                assert!(prefix
                    .iter()
                    .any(|record| record["input"]["kind"] == "prompt"));
            }
            if complete {
                let snapshot = CaptureSnapshot::read(
                    &directory,
                    CaptureReadLimits {
                        file_bytes: 1_000_000,
                        record_bytes: 100_000,
                        records: 256,
                    },
                )
                .unwrap();
                let batches: Vec<_> = snapshot
                    .payloads()
                    .map(|p| serde_json::from_str::<Value>(p).unwrap())
                    .filter(|r| r["kind"] == "chat_batch")
                    .map(|r| r["events"].clone())
                    .collect();
                assert_eq!(batches, original.2, "all actual adapter input boundaries");
            }
        }
    }
}

#[tokio::test]
async fn reader_capture_chat_batch_does_not_claim_delivery_after_disconnect() {
    let original = render(None, false, true).await;
    assert_eq!(original.1.as_deref(), Some("Failed to send chunk"));
    assert!(original.0.is_empty());
    let (_root, _pool, owner) = owner(100_000, 1_000_000).await;
    let observed = render(Some(owner.observer()), false, true).await;
    assert_eq!(observed, original);
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
        .map(|p| serde_json::from_str(p).unwrap())
        .collect();
    assert_eq!(
        records.iter().filter(|r| r["kind"] == "chat_batch").count(),
        1
    );
    assert_eq!(records.last().unwrap()["finished"], false);
    // Storage completion preserves an observed prefix; request/send outcomes remain the
    // future coordinator's job and cannot be inferred from this adapter entrance record.
}
