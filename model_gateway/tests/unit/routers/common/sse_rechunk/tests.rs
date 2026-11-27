use std::{
    io,
    sync::{Arc, Mutex},
};

use futures::stream;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::*;
use crate::routers::grpc::wire::Wire;

/// The one policy in the tree: MiniMax's.
fn policy() -> RechunkPolicy {
    Wire::Minimax
        .rechunk_policy()
        .expect("MiniMax's streams are re-sliced")
}

fn bytes(data: impl AsRef<[u8]>) -> Bytes {
    Bytes::copy_from_slice(data.as_ref())
}

fn events(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(bytes)
        .split("\n\n")
        .filter(|f| !f.is_empty())
        .filter_map(|f| f.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

fn content_event(text: &str) -> String {
    format!(
        "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
        serde_json::to_string(text).unwrap()
    )
}

fn run(frames: &[&str]) -> Vec<u8> {
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    for frame in frames {
        all.extend_from_slice(r.feed(bytes(frame)).as_ref());
    }
    all.extend_from_slice(r.finish().as_ref());
    all
}

fn content_sizes(evs: &[Value]) -> Vec<usize> {
    evs.iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .map(|s| s.chars().count())
        .collect()
}

#[test]
fn tiny_deltas_merge() {
    let frames: Vec<String> = (0..100).map(|_| content_event("ab")).collect();
    let refs: Vec<&str> = frames.iter().map(String::as_str).collect();
    let sizes = content_sizes(&events(&run(&refs)));
    assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 200);
}

#[test]
fn large_delta_splits() {
    let big = "x".repeat(1000);
    let sizes = content_sizes(&events(&run(&[&content_event(&big)])));
    assert!(sizes.iter().all(|&s| s <= 200), "sizes: {sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 1000);
}

#[test]
fn tail_slices_never_fall_into_the_tiny_band() {
    let sizes = content_sizes(&events(&run(&[&content_event(&"x".repeat(161))])));
    assert_eq!(sizes, vec![156, 5]);
}

#[test]
fn multibyte_payload_is_sliced_by_chars() {
    let all = run(&[&content_event(&"中".repeat(500))]);
    let evs = events(&all);
    for e in &evs {
        let s = e["choices"][0]["delta"]["content"].as_str().unwrap();
        assert!(s.chars().count() <= 200, "chars: {}", s.chars().count());
    }
    assert!(evs
        .iter()
        .any(|e| { e["choices"][0]["delta"]["content"].as_str().unwrap().len() > 200 }));
    assert_eq!(content_sizes(&evs).iter().sum::<usize>(), 500);
}

#[test]
fn escaped_text_is_rebuilt_exactly() {
    let text = "line one\n\t\"quoted\" \\ 中文 \u{1F600}";
    let evs = events(&run(&[&content_event(text)]));
    let joined: String = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(joined, text);
}

#[test]
fn frames_split_across_chunks_are_reassembled() {
    let frame = content_event(&"y".repeat(100));
    let frame = frame.as_bytes();
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    for k in [7usize, 40, 90] {
        all.extend_from_slice(r.feed(bytes(&frame[..k])).as_ref());
        assert!(all.is_empty(), "nothing before the frame is complete");
        all.extend_from_slice(r.feed(bytes(&frame[k..])).as_ref());
        all.extend_from_slice(r.finish().as_ref());
        let sizes = content_sizes(&events(&all));
        assert_eq!(sizes, vec![100], "split at {k}");
        r = SseRechunker::new(policy());
        all.clear();
    }
}

#[test]
fn crlf_frames_are_recognised() {
    let frame =
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ab\"}}]}\r\n\r\n";
    let frames = [frame; 50];
    let sizes = content_sizes(&events(&run(&frames)));
    assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 100);
}

#[test]
fn bare_cr_frames_are_recognised() {
    let frame =
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ab\"}}]}\r\r";
    let frames = [frame; 50];
    let sizes = content_sizes(&events(&run(&frames)));
    assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 100);
}

#[test]
fn a_delimiter_split_across_feeds_is_found() {
    let frame = content_event(&"z".repeat(100));
    let frame = frame.as_bytes();
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    all.extend_from_slice(r.feed(bytes(&frame[..frame.len() - 1])).as_ref());
    assert!(all.is_empty());
    all.extend_from_slice(r.feed(bytes(&frame[frame.len() - 1..])).as_ref());
    assert_eq!(content_sizes(&events(&all)), vec![100]);
}

#[test]
fn oversized_unterminated_frames_switch_to_passthrough() {
    let mut r = SseRechunker::new(policy());
    assert!(r.feed(bytes(content_event("ab"))).is_empty());
    let fragment = vec![b'q'; 64 * 1024];
    let mut forwarded = Vec::new();
    for _ in 0..17 {
        forwarded.extend_from_slice(r.feed(bytes(&fragment)).as_ref());
    }
    // The pending payload came out first, then every raw byte.
    let text = String::from_utf8_lossy(&forwarded);
    assert!(text.starts_with("data: "), "pending payload flushed first");
    assert!(text.ends_with(&"q".repeat(64 * 1024)));
    assert_eq!(text.matches('q').count(), 17 * 64 * 1024);
    assert_eq!(r.feed(bytes(b"more")).as_ref(), b"more");
}

#[test]
fn passthrough_returns_the_chunk_it_was_given() {
    let mut r = SseRechunker::new(policy());
    r.feed(bytes(b"data: [DONE]\n\n"));
    let chunk = Bytes::from_static(b"trailing bytes");
    let returned = r.feed(chunk.clone());
    assert_eq!(returned.as_ptr(), chunk.as_ptr(), "no copy in passthrough");
}

#[test]
fn choice_level_metadata_is_forwarded_once() {
    let with_logprobs = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"logprobs\":{\"content\":[{\"token\":\"Hi\",\"logprob\":-0.1}]},\"finish_reason\":null}]}\n\n";
    let evs = events(&run(&[
        &content_event("ab"),
        with_logprobs,
        &content_event("cd"),
    ]));
    let logprob_events: Vec<&Value> = evs
        .iter()
        .filter(|e| e["choices"][0]["logprobs"].is_object())
        .collect();
    assert_eq!(logprob_events.len(), 1);
    assert_eq!(
        logprob_events[0]["choices"][0]["delta"]["content"],
        Value::from("Hi")
    );
    // The logprobs stay attached to the content they describe.
    let texts: Vec<&str> = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(texts, vec!["ab", "Hi", "cd"]);
    // A null logprobs field, the common shape when none were requested, still merges.
    let null_logprobs = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ab\"},\"logprobs\":null}]}\n\n";
    let frames = [null_logprobs; 50];
    let sizes = content_sizes(&events(&run(&frames)));
    assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
}

/// A stream under `include_usage` writes `usage: null` on every chunk
/// before the usage-only one: the slices synthesized from such chunks
/// carry the placeholder after their choices, an opening keeps it, and
/// no slice invents one for a stream without it.
#[test]
fn the_usage_placeholder_rides_every_synthesized_slice() {
    let text = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"abcdefghij\"},\"finish_reason\":null}],\"usage\":null}\n\n";
    let last = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"end\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n";
    let mut frames = vec![text; 30];
    frames.push(last);
    let bytes = run(&frames);
    let evs = events(&bytes);
    assert!(evs.len() > 2, "{evs:?}");
    assert!(
        evs.iter().all(|e| e.get("usage") == Some(&Value::Null)),
        "{evs:?}"
    );
    assert_eq!(
        evs.last().unwrap()["choices"][0]["finish_reason"],
        Value::from("stop")
    );
    let text_bytes = std::str::from_utf8(&bytes).unwrap();
    assert!(
        text_bytes.contains("}],\"usage\":null}\n\n"),
        "{text_bytes}"
    );
    assert!(
        !text_bytes.contains("\"usage\":null,\"choices\""),
        "{text_bytes}"
    );

    let plain = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"abcdefghij\"},\"finish_reason\":null}]}\n\n";
    let evs = events(&run(&[plain; 30]));
    assert!(evs.len() > 1, "{evs:?}");
    assert!(evs.iter().all(|e| e.get("usage").is_none()), "{evs:?}");

    // A closing event with a usage object: the text buffered before it
    // keeps its placeholder, the closing event keeps its object.
    let closing = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"end\"},\"finish_reason\":\"stop\"}],\"usage\":{\"total_tokens\":5}}\n\n";
    let mut frames = vec![text; 30];
    frames.push(closing);
    let evs = events(&run(&frames));
    let (last, earlier) = evs.split_last().expect("events");
    assert!(earlier.len() > 1, "{evs:?}");
    assert!(
        earlier.iter().all(|e| e.get("usage") == Some(&Value::Null)),
        "{evs:?}"
    );
    assert_eq!(last["usage"]["total_tokens"], Value::from(5), "{last}");
    assert_eq!(last["choices"][0]["finish_reason"], Value::from("stop"));
}

#[test]
fn choice_level_usage_is_forwarded() {
    let with_usage = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"usage\":{\"total_tokens\":3}}]}\n\n";
    let evs = events(&run(&[with_usage]));
    assert!(evs
        .iter()
        .any(|e| e["choices"][0]["usage"]["total_tokens"] == 3));
}

#[test]
fn finish_and_usage_flush_in_order() {
    let fin = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"total_tokens\":5}}\n\n";
    let evs = events(&run(&[&content_event("hello"), fin]));
    assert_eq!(
        evs.last().unwrap()["choices"][0]["finish_reason"],
        Value::from("stop")
    );
    let content_pos = evs
        .iter()
        .position(|e| e["choices"][0]["delta"]["content"].is_string())
        .unwrap();
    assert!(content_pos < evs.len() - 1);
}

#[test]
fn content_and_finish_in_one_event_keep_their_order() {
    let last = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"end\"},\"finish_reason\":\"stop\"}]}\n\n";
    let evs = events(&run(&[&content_event("hello "), last]));
    let text: String = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "hello end");
    let fin = evs.last().unwrap();
    assert_eq!(fin["choices"][0]["finish_reason"], Value::from("stop"));
    assert!(fin["choices"][0]["delta"]["content"].is_null());
}

#[test]
fn a_closing_event_does_not_split_its_own_payload() {
    // 79 pending chars plus a final "!" with finish_reason must leave as
    // one 80-char event, not 79 + 1.
    let last = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n";
    let sizes = content_sizes(&events(&run(&[&content_event(&"y".repeat(79)), last])));
    assert_eq!(sizes, vec![80]);
}

#[test]
fn residual_tool_call_fields_are_forwarded_in_order() {
    let frag = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ab\",\"parsed\":true}}]}}]}\n\n";
    let evs = events(&run(&[frag]));
    let args: String = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str())
        .collect();
    assert_eq!(args, "ab");
    assert!(evs.iter().any(|e| {
        e["choices"][0]["delta"]["tool_calls"][0]["function"]["parsed"] == Value::Bool(true)
    }));
}

#[test]
fn role_and_content_in_one_event_put_the_role_first() {
    let first = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"}}]}\n\n";
    let evs = events(&run(&[first, &content_event(" there")]));
    assert_eq!(
        evs[0]["choices"][0]["delta"]["role"],
        Value::from("assistant")
    );
    assert!(evs[0]["choices"][0]["delta"]["content"].is_null());
    let text: String = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "Hi there");
}

#[test]
fn prompt_text_goes_out_once_with_its_events_own_keys() {
    // The first event carries the rendered prompt the request asked
    // back, a role, text and a usage object: it leaves as an opening
    // event, its text under the envelope, and a closing event, and the
    // prompt rides the opening alone.
    let first = "data: {\"id\":\"x\",\"prompt_text\":\"user: hi\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi there\"}}],\"usage\":{\"total_tokens\":5}}\n\n";
    let evs = events(&run(&[first]));
    assert_eq!(evs.len(), 3, "{evs:?}");
    assert_eq!(evs[0]["prompt_text"], Value::from("user: hi"));
    assert!(
        evs[1..].iter().all(|e| e.get("prompt_text").is_none()),
        "{evs:?}"
    );

    // A content event that brings the prompt behind the role, with no
    // finish and no usage, would leave as synthesized text alone: an
    // event of its own carries the key once, and the text is whole.
    let role =
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n";
    let content = "data: {\"id\":\"x\",\"prompt_text\":\"user: hi\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n";
    let evs = events(&run(&[role, content]));
    let prompts: Vec<&Value> = evs.iter().filter_map(|e| e.get("prompt_text")).collect();
    assert_eq!(prompts, [&Value::from("user: hi")], "{evs:?}");
    // The key's event leaves ahead of the event's text.
    let key_at = evs.iter().position(|e| e.get("prompt_text").is_some());
    let text_at = evs.iter().position(|e| {
        e["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|text| !text.is_empty())
    });
    assert!(key_at < text_at, "{evs:?}");
    let text: String = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "hello");

    // A content event that brings the prompt behind the role and closes
    // the choice goes out after its own text: the closing event carries
    // the key once, behind the event's text, and the text is whole.
    let closing = "data: {\"id\":\"x\",\"prompt_text\":\"user: hi\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n";
    let evs = events(&run(&[role, closing]));
    assert_eq!(evs.len(), 3, "{evs:?}");
    assert!(
        evs[..2].iter().all(|e| e.get("prompt_text").is_none()),
        "{evs:?}"
    );
    assert_eq!(
        evs[1]["choices"][0]["delta"]["content"],
        Value::from("hello")
    );
    assert_eq!(evs[2]["prompt_text"], Value::from("user: hi"));
    assert_eq!(evs[2]["choices"][0]["finish_reason"], Value::from("stop"));
    assert!(
        evs[2]["choices"][0]["delta"]["content"].is_null(),
        "{evs:?}"
    );
}

#[test]
fn repeated_roles_do_not_break_merging() {
    let frame = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ab\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n";
    let frames = [frame; 50];
    let evs = events(&run(&frames));
    let roles = evs
        .iter()
        .filter(|e| e["choices"][0]["delta"]["role"].is_string())
        .count();
    assert_eq!(roles, 1, "only the first role is forwarded");
    let sizes = content_sizes(&evs);
    assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 100);
}

#[test]
fn tool_call_identity_precedes_arguments() {
    let start = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}}]}}]}\n\n";
    let args = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"k\\\":1}\"}}]}}]}\n\n";
    let evs = events(&run(&[start, args]));
    let name_pos = evs
        .iter()
        .position(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["name"].is_string())
        .unwrap();
    let args_pos = evs
        .iter()
        .position(|e| {
            e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        })
        .unwrap();
    assert!(name_pos < args_pos);
}

#[test]
fn identity_and_arguments_in_one_event_stay_ordered() {
    let start = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n";
    let rest = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n";
    let evs = events(&run(&[start, rest]));
    assert_eq!(
        evs[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
        Value::from("c1")
    );
    assert!(evs[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].is_null());
    let args: String = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str())
        .collect();
    assert_eq!(args, "{\"a\":1}");
}

#[test]
fn omitted_tool_index_is_not_invented() {
    let start = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}}]}\n\n";
    let frag = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n";
    let evs = events(&run(&[start, frag]));
    let args_event = evs
        .iter()
        .find(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].is_string())
        .unwrap();
    assert!(args_event["choices"][0]["delta"]["tool_calls"][0]["index"].is_null());
}

#[test]
fn null_valued_fields_are_not_structural() {
    let frag = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":null,\"refusal\":null,\"tool_calls\":[{\"index\":0,\"function\":{\"name\":null,\"arguments\":\"ab\"}}]}}]}\n\n";
    let frames = [frag; 50];
    let evs = events(&run(&frames));
    let sizes: Vec<usize> = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str())
        .map(str::len)
        .collect();
    assert!(sizes.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 100);
    assert_eq!(
        evs.len(),
        sizes.len(),
        "no event was forwarded structurally"
    );
}

#[test]
fn reasoning_is_flushed_before_content_starts() {
    let reasoning = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n";
    let evs = events(&run(&[reasoning, &content_event(&"c".repeat(100))]));
    let think_pos = evs
        .iter()
        .position(|e| e["choices"][0]["delta"]["reasoning_content"].is_string())
        .unwrap();
    let content_pos = evs
        .iter()
        .position(|e| e["choices"][0]["delta"]["content"].is_string())
        .unwrap();
    assert!(think_pos < content_pos);
}

#[test]
fn keepalive_comments_do_not_flush() {
    let all = run(&[&content_event("ab"), ": ping\n\n", &content_event("cd")]);
    let text = String::from_utf8_lossy(&all);
    assert!(text.contains(": ping\n\n"));
    let sizes = content_sizes(&events(&all));
    assert_eq!(
        sizes,
        vec![4],
        "the two fragments were merged across the comment"
    );
}

#[test]
fn multi_choice_streams_pass_through_verbatim() {
    let two = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}},{\"index\":1,\"delta\":{\"content\":\"b\"}}]}\n\n";
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    all.extend_from_slice(r.feed(bytes(two)).as_ref());
    all.extend_from_slice(r.feed(bytes(two)).as_ref());
    all.extend_from_slice(r.finish().as_ref());
    assert_eq!(all, [two.as_bytes(), two.as_bytes()].concat());
}

#[test]
fn legacy_completion_chunks_pass_through() {
    let text = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"text\":\"hi\",\"finish_reason\":null}]}\n\n";
    let all = run(&[text, text]);
    assert_eq!(all, [text.as_bytes(), text.as_bytes()].concat());
}

#[test]
fn empty_choices_events_pass_through() {
    let opener = "data: {\"id\":\"x\",\"choices\":[],\"prompt_filter_results\":[1]}\n\n";
    let all = run(&[opener, &content_event(&"z".repeat(100))]);
    let evs = events(&all);
    assert_eq!(evs[0]["prompt_filter_results"], serde_json::json!([1]));
    assert!(evs[1]["prompt_filter_results"].is_null());
}

#[test]
fn envelope_follows_the_latest_chunk() {
    let first = "data: {\"id\":\"x\",\"created\":1,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n";
    let second = "data: {\"id\":\"x\",\"created\":2,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}\n\n";
    let evs = events(&run(&[first, second]));
    assert_eq!(evs.last().unwrap()["created"], Value::from(2));
}

#[test]
fn error_frames_do_not_become_the_envelope() {
    let error = "data: {\"error\":{\"message\":\"boom\"}}\n\n";
    let evs = events(&run(&[error, &content_event("hello")]));
    assert_eq!(evs[0]["error"]["message"], Value::from("boom"));
    assert!(evs[1]["error"].is_null());
}

#[test]
fn unknown_delta_keys_are_forwarded_in_order() {
    let audio =
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"audio\":{\"id\":\"a\"}}}]}\n\n";
    let evs = events(&run(&[&content_event("ab"), audio, &content_event("cd")]));
    let audio_pos = evs
        .iter()
        .position(|e| e["choices"][0]["delta"]["audio"].is_object())
        .unwrap();
    let texts: Vec<&str> = evs
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(texts, vec!["ab", "cd"]);
    assert_eq!(audio_pos, 1);
}

#[test]
fn flush_pending_emits_what_is_buffered() {
    let mut r = SseRechunker::new(policy());
    assert!(r.feed(bytes(content_event("hello"))).is_empty());
    assert!(r.has_pending());
    let evs = events(&r.flush_pending());
    assert_eq!(
        evs[0]["choices"][0]["delta"]["content"],
        Value::from("hello")
    );
    assert!(!r.has_pending());
}

#[test]
fn flush_pending_holds_a_remainder_shorter_than_the_tiny_band() {
    let mut r = SseRechunker::new(policy());
    assert!(r.feed(bytes(content_event("hi"))).is_empty());
    assert!(!r.has_pending(), "two characters are not worth an event");
    assert!(r.flush_pending().is_empty());
    assert!(r.feed(bytes(content_event(" there"))).is_empty());
    assert!(r.has_pending());
    let evs = events(&r.flush_pending());
    assert_eq!(
        evs[0]["choices"][0]["delta"]["content"],
        Value::from("hi there")
    );
    assert!(!r.has_pending());
    // The end of the stream still releases a tiny remainder.
    assert!(r.feed(bytes(content_event("ok"))).is_empty());
    let evs = events(&r.finish());
    assert_eq!(evs[0]["choices"][0]["delta"]["content"], Value::from("ok"));
}

/// The frames of a forced function call: the call's identity, then one
/// `arguments` delta per decoded token.
fn tool_identity_event() -> String {
    "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"functions.write_file:0\",\"type\":\"function\",\"function\":{\"name\":\"write_file\"}}]},\"logprobs\":null,\"finish_reason\":null}]}\n\n".to_string()
}

fn arguments_event(text: &str) -> String {
    format!(
        "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{}}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n",
        serde_json::to_string(text).unwrap()
    )
}

/// An upstream that repeats null identity fields on every piece.
fn arguments_event_with_nulls(text: &str) -> String {
    format!(
        "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":null,\"content\":null,\"tool_calls\":[{{\"id\":null,\"index\":0,\"type\":\"function\",\"function\":{{\"name\":null,\"arguments\":{}}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n",
        serde_json::to_string(text).unwrap()
    )
}

/// An upstream that sends the identity together with the first piece.
fn identity_with_arguments_event(text: &str) -> String {
    format!(
        "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{{\"name\":\"write_file\",\"arguments\":{}}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n",
        serde_json::to_string(text).unwrap()
    )
}

const FINISH_EVENT: &str = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"tool_calls\"}]}\n\n";

fn argument_pieces(evs: &[Value]) -> Vec<String> {
    evs.iter()
        .filter_map(|e| {
            e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// The tokens of `{"path":"a.txt","content":"<hex>"}` as a byte-level
/// tokenizer cuts them: the JSON punctuation is one to three characters.
fn forced_call_tokens() -> Vec<String> {
    let mut tokens: Vec<String> = [
        "{\"", "path", "\":\"", "a", ".txt", "\",\"", "content", "\":\"",
    ]
    .iter()
    .map(|t| (*t).to_string())
    .collect();
    tokens.extend((0..40).map(|i| format!("{i:04x}")));
    tokens.push("\"}".to_string());
    tokens
}

/// Streams `frames` with the idle timer firing after every one of them,
/// as it does when the engine pauses between tokens.
fn run_with_pauses(frames: &[String]) -> Vec<u8> {
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    for frame in frames {
        all.extend_from_slice(r.feed(bytes(frame)).as_ref());
        all.extend_from_slice(r.flush_pending().as_ref());
    }
    all.extend_from_slice(r.feed(bytes(FINISH_EVENT)).as_ref());
    all.extend_from_slice(r.finish().as_ref());
    all
}

fn assert_no_tiny_pieces_before_the_tail(all: &[u8], tokens: &[String]) {
    let pieces = argument_pieces(&events(all));
    let sizes: Vec<usize> = pieces.iter().map(|p| p.chars().count()).collect();
    let (tail, body) = sizes.split_last().expect("some argument piece");
    assert!(body.iter().all(|&s| s >= 5), "sizes: {sizes:?}");
    assert!(*tail >= 1);
    assert_eq!(pieces.concat(), tokens.concat());
}

#[test]
fn an_engine_pausing_between_tokens_does_not_make_tiny_argument_events() {
    let tokens = forced_call_tokens();
    let mut frames = vec![tool_identity_event()];
    frames.extend(tokens.iter().map(|t| arguments_event(t)));
    assert_no_tiny_pieces_before_the_tail(&run_with_pauses(&frames), &tokens);
}

#[test]
fn null_identity_fields_on_every_piece_still_merge_across_pauses() {
    let tokens = forced_call_tokens();
    let mut frames = vec![tool_identity_event()];
    frames.extend(tokens.iter().map(|t| arguments_event_with_nulls(t)));
    assert_no_tiny_pieces_before_the_tail(&run_with_pauses(&frames), &tokens);
}

#[test]
fn identity_sent_with_the_first_piece_still_merges_across_pauses() {
    let tokens = forced_call_tokens();
    let mut frames = vec![identity_with_arguments_event(&tokens[0])];
    frames.extend(tokens[1..].iter().map(|t| arguments_event(t)));
    let all = run_with_pauses(&frames);
    let evs = events(&all);
    assert_eq!(
        evs[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        Value::from("write_file")
    );
    assert_no_tiny_pieces_before_the_tail(&all, &tokens);
}

/// The identity of tool call `index`: the engine opening one of
/// several parallel calls.
fn tool_identity_event_for(index: u64) -> String {
    format!(
        "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{index},\"id\":\"call_{index}\",\"type\":\"function\",\"function\":{{\"name\":\"write_file\"}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n"
    )
}

fn arguments_event_for(index: u64, text: &str) -> String {
    format!(
        "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{index},\"function\":{{\"arguments\":{}}}}}]}},\"logprobs\":null,\"finish_reason\":null}}]}}\n\n",
        serde_json::to_string(text).unwrap()
    )
}

/// The argument pieces of tool call `index`, in order.
fn argument_pieces_of(evs: &[Value], index: u64) -> Vec<String> {
    evs.iter()
        .filter_map(|e| {
            let call = &e["choices"][0]["delta"]["tool_calls"][0];
            if call["index"].as_u64() != Some(index) {
                return None;
            }
            call["function"]["arguments"].as_str().map(str::to_owned)
        })
        .collect()
}

/// Where the identity of tool call `index` is written.
fn identity_position(evs: &[Value], index: u64) -> usize {
    evs.iter()
        .position(|e| {
            let call = &e["choices"][0]["delta"]["tool_calls"][0];
            call["index"].as_u64() == Some(index) && call["function"]["name"].is_string()
        })
        .expect("the call's identity")
}

/// A parallel tool-call stream: a call's value text in engine-sized
/// deltas, a pause, its closer on a delta of its own, then the next call
/// opens. The closer must ride with text of its own call, not leave
/// alone when the next call's identity flushes the buffers.
#[test]
fn a_closer_arriving_alone_rides_with_its_call_when_the_next_call_opens() {
    let call_0 = format!("{{\"path\":\"{}\"}}", "v".repeat(231));
    let (text, closer) = call_0.split_at(240);
    assert_eq!(closer, "\"}");
    let call_1 = "{\"path\":\"b.txt\"}";
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    all.extend_from_slice(r.feed(bytes(tool_identity_event_for(0))).as_ref());
    for piece in [&text[..80], &text[80..160], &text[160..]] {
        all.extend_from_slice(r.feed(bytes(arguments_event_for(0, piece))).as_ref());
    }
    // The engine pauses before the closer: the idle timer fires.
    all.extend_from_slice(r.flush_pending().as_ref());
    all.extend_from_slice(r.feed(bytes(arguments_event_for(0, closer))).as_ref());
    all.extend_from_slice(r.feed(bytes(tool_identity_event_for(1))).as_ref());
    all.extend_from_slice(r.feed(bytes(arguments_event_for(1, call_1))).as_ref());
    all.extend_from_slice(r.feed(bytes(FINISH_EVENT)).as_ref());
    all.extend_from_slice(r.finish().as_ref());
    let evs = events(&all);
    let pieces = argument_pieces_of(&evs, 0);
    let sizes: Vec<usize> = pieces.iter().map(|p| p.chars().count()).collect();
    assert!(
        sizes.iter().all(|&s| s >= policy().min_tail_chars),
        "call 0 pieces: {sizes:?}"
    );
    assert_eq!(pieces.concat(), call_0);
    assert_eq!(argument_pieces_of(&evs, 1).concat(), call_1);
    // Call 0 is complete before call 1 opens.
    let last_of_0 = evs
        .iter()
        .rposition(|e| {
            let call = &e["choices"][0]["delta"]["tool_calls"][0];
            call["index"].as_u64() == Some(0) && call["function"]["arguments"].is_string()
        })
        .unwrap();
    assert!(last_of_0 < identity_position(&evs, 1));
}

/// An idle flush of a call's arguments keeps a tail for the closer, and
/// the tail alone is not worth an idle flush; the end of the stream
/// releases it with whatever followed.
#[test]
fn an_idle_flush_keeps_a_tail_of_a_calls_arguments_for_its_closer() {
    let mut r = SseRechunker::new(policy());
    assert!(!r.feed(bytes(tool_identity_event())).is_empty());
    let text = format!("{{\"path\":\"{}", "p".repeat(59));
    assert!(r.feed(bytes(arguments_event(&text))).is_empty());
    assert!(r.has_pending());
    let evs = events(&r.flush_pending());
    assert_eq!(argument_pieces(&evs), vec![text[..63].to_owned()]);
    assert!(!r.has_pending(), "the held tail waits for the closer");
    assert!(r.flush_pending().is_empty());
    assert!(r.feed(bytes(arguments_event("\"}"))).is_empty());
    assert!(!r.has_pending());
    let mut all = r.feed(bytes(FINISH_EVENT)).to_vec();
    all.extend_from_slice(r.finish().as_ref());
    let tail = format!("{}\"}}", &text[63..]);
    assert_eq!(argument_pieces(&events(&all)), vec![tail]);
}

/// Content is not held back: a reader is waiting for it. Its slices are
/// what they were before tool-call arguments kept a tail, the same shape
/// included.
#[test]
fn content_before_a_tool_call_is_not_held_back() {
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    for _ in 0..3 {
        all.extend_from_slice(r.feed(bytes(content_event(&"c".repeat(80)))).as_ref());
    }
    all.extend_from_slice(r.flush_pending().as_ref());
    all.extend_from_slice(r.feed(bytes(content_event("ok"))).as_ref());
    all.extend_from_slice(r.feed(bytes(tool_identity_event())).as_ref());
    all.extend_from_slice(
        r.feed(bytes(arguments_event("{\"path\":\"a.txt\"}")))
            .as_ref(),
    );
    all.extend_from_slice(r.feed(bytes(FINISH_EVENT)).as_ref());
    all.extend_from_slice(r.finish().as_ref());
    let evs = events(&all);
    assert_eq!(content_sizes(&evs), vec![80, 80, 80, 2]);
    assert_eq!(argument_pieces(&evs), vec!["{\"path\":\"a.txt\"}"]);
}

/// A call whose whole arguments are tiny still goes out, before the
/// next call opens: only a delta that is tiny in total is a tiny event.
#[test]
fn a_call_whose_whole_arguments_are_tiny_still_precedes_the_next_call() {
    let frames = [
        tool_identity_event_for(0),
        arguments_event_for(0, "{}"),
        tool_identity_event_for(1),
        arguments_event_for(1, "{\"k\":1}"),
        FINISH_EVENT.to_string(),
    ];
    let refs: Vec<&str> = frames.iter().map(String::as_str).collect();
    let evs = events(&run(&refs));
    assert_eq!(argument_pieces_of(&evs, 0), vec!["{}"]);
    let piece_0 = evs
        .iter()
        .position(|e| e["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "{}")
        .unwrap();
    assert!(piece_0 < identity_position(&evs, 1));
    assert_eq!(argument_pieces_of(&evs, 1), vec!["{\"k\":1}"]);
}

#[test]
fn done_passthrough_after_flush() {
    let mut r = SseRechunker::new(policy());
    let mut all = Vec::new();
    all.extend_from_slice(r.feed(bytes(content_event("hi"))).as_ref());
    all.extend_from_slice(r.feed(bytes(b"data: [DONE]\n\n")).as_ref());
    let text = String::from_utf8_lossy(&all).to_string();
    assert!(text.find("\"hi\"").unwrap() < text.find("[DONE]").unwrap());
    assert_eq!(r.feed(bytes(b"trailing")).as_ref(), b"trailing");
}

fn chunk(text: &str) -> Bytes {
    Bytes::from(content_event(text))
}

fn contents(frames: &[Bytes]) -> Vec<String> {
    let mut all = Vec::new();
    for frame in frames {
        all.extend_from_slice(frame);
    }
    events(&all)
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .map(str::to_string)
        .collect()
}

#[tokio::test(start_paused = true)]
async fn stream_adapter_merges_tiny_deltas_and_flushes_the_tail() {
    let upstream = stream::iter((0..100).map(|_| Ok::<_, io::Error>(chunk("ab"))));
    let frames: Vec<Bytes> = rechunk_stream(policy(), upstream, NoopObserver)
        .map(|frame| frame.unwrap())
        .collect()
        .await;
    let sizes: Vec<usize> = contents(&frames)
        .iter()
        .map(|c| c.chars().count())
        .collect();
    assert!(
        sizes.iter().all(|&s| s >= policy().min_tail_chars),
        "sizes: {sizes:?}"
    );
    assert_eq!(sizes.iter().sum::<usize>(), 200);
}

#[tokio::test(start_paused = true)]
async fn stream_adapter_flushes_pending_payload_when_upstream_idles() {
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
    let mut out = Box::pin(rechunk_stream(
        policy(),
        ReceiverStream::new(rx),
        NoopObserver,
    ));
    for _ in 0..3 {
        tx.send(Ok(chunk("ab"))).await.unwrap();
    }
    // Below the emit threshold: only the idle timer releases it.
    let frame = out.next().await.unwrap().unwrap();
    assert_eq!(contents(&[frame]), vec!["ababab"]);
    drop(tx);
    assert!(out.next().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn stream_adapter_holds_a_tiny_remainder_across_an_upstream_pause() {
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
    let mut out = Box::pin(rechunk_stream(
        policy(),
        ReceiverStream::new(rx),
        NoopObserver,
    ));
    tx.send(Ok(chunk("ab"))).await.unwrap();
    // Quiet for far longer than the idle flush: nothing worth an event yet.
    assert!(tokio::time::timeout(policy().idle_flush * 4, out.next())
        .await
        .is_err());
    tx.send(Ok(chunk("cde"))).await.unwrap();
    let frame = out.next().await.unwrap().unwrap();
    assert_eq!(contents(&[frame]), vec!["abcde"]);
    // The end of the stream releases what the idle flush held.
    tx.send(Ok(chunk("f"))).await.unwrap();
    drop(tx);
    let frame = out.next().await.unwrap().unwrap();
    assert_eq!(contents(&[frame]), vec!["f"]);
    assert!(out.next().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn stream_adapter_measures_the_idle_flush_from_the_last_chunk() {
    let window = policy().idle_flush;
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
    let mut out = Box::pin(rechunk_stream(
        policy(),
        ReceiverStream::new(rx),
        NoopObserver,
    ));
    // Under the emit threshold and inside the window: held.
    tx.send(Ok(chunk("abcde"))).await.unwrap();
    assert!(tokio::time::timeout(window / 5, out.next()).await.is_err());
    // A second delta late in the window starts a new one: nothing leaves
    // until a whole window has passed since that delta, not since the
    // stream began.
    tokio::time::sleep(window * 3 / 5).await;
    tx.send(Ok(chunk("fghij"))).await.unwrap();
    assert!(
        tokio::time::timeout(window * 4 / 5, out.next())
            .await
            .is_err(),
        "held text left before a window had passed since the last delta"
    );
    // And it leaves once that window has passed: bounded from above too,
    // so a window that grew would be noticed.
    let frame = tokio::time::timeout(window / 2, out.next())
        .await
        .expect("the held text leaves once a window has passed since the last delta")
        .unwrap()
        .unwrap();
    assert_eq!(contents(&[frame]), vec!["abcdefghij"]);
    drop(tx);
    assert!(out.next().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn stream_adapter_delivers_the_tail_before_an_upstream_error() {
    let upstream = stream::iter(vec![Ok(chunk("ab")), Err(io::Error::other("boom"))]);
    let mut out = Box::pin(rechunk_stream(policy(), upstream, NoopObserver));
    let tail = out.next().await.unwrap().unwrap();
    assert_eq!(contents(&[tail]), vec!["ab"]);
    assert!(out.next().await.unwrap().is_err());
    assert!(out.next().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn stream_adapter_ends_cleanly_when_the_veto_lands_on_the_tail() {
    // The upstream errors with text still held; the observer vetoes the
    // tail's slice, so the policy's error frame is the last thing the
    // client sees and the upstream's error is not yielded after it.
    let upstream = stream::iter(vec![Ok(chunk("ab")), Err(io::Error::other("boom"))]);
    let mut out = Box::pin(rechunk_stream(
        policy(),
        upstream,
        Watcher::vetoing_after(0),
    ));
    let frame = out.next().await.unwrap().unwrap();
    assert_eq!(frame.as_ref(), POLICY_ERROR_FRAME.as_bytes());
    assert!(out.next().await.is_none());
}

/// What an observer was shown, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Seen {
    Open,
    Slice(DeltaField, String),
    End,
}

/// Records what it is shown, and lets `allow` slices out before it ends
/// the stream on the next one.
#[derive(Clone)]
struct Watcher {
    seen: Arc<Mutex<Vec<Seen>>>,
    allow: usize,
}

impl Watcher {
    fn recording() -> Self {
        Self::vetoing_after(usize::MAX)
    }

    fn vetoing_after(allow: usize) -> Self {
        Self {
            seen: Arc::default(),
            allow,
        }
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl DeltaObserver for Watcher {
    fn open(&mut self) {
        self.seen.lock().unwrap().push(Seen::Open);
    }

    fn on_slice(&mut self, slice: DeltaSlice<'_>) -> Verdict {
        self.seen
            .lock()
            .unwrap()
            .push(Seen::Slice(slice.field, slice.text.to_owned()));
        if self.allow == 0 {
            return Verdict::Terminate(policy_stop());
        }
        self.allow -= 1;
        Verdict::Continue
    }

    fn end(&mut self) {
        self.seen.lock().unwrap().push(Seen::End);
    }
}

fn policy_stop() -> StreamStop {
    StreamStop {
        message: "the response was ended by a policy".to_owned(),
        error_type: "policy_violation".to_owned(),
    }
}

const POLICY_ERROR_FRAME: &str = "data: {\"error\":{\"message\":\"the response was ended by a policy\",\"type\":\"policy_violation\"}}\n\n";

/// The slices of the emitted events, in order: what the client sees.
fn emitted_slices(all: &[u8]) -> Vec<Seen> {
    events(all)
        .iter()
        .filter_map(|event| {
            let delta = &event["choices"][0]["delta"];
            for field in PAYLOAD_FIELDS {
                if let Some(text) = delta[field].as_str() {
                    return Some(Seen::Slice(DeltaField::Text(field), text.to_owned()));
                }
            }
            let call = &delta["tool_calls"][0];
            call["function"]["arguments"]
                .as_str()
                .filter(|text| !text.is_empty())
                .map(|text| {
                    Seen::Slice(
                        DeltaField::ToolArguments(call["index"].as_u64()),
                        text.to_owned(),
                    )
                })
        })
        .collect()
}

/// A stream with every kind of slice: a role, reasoning in tokens, one
/// large content delta, a tool call's identity and its arguments in
/// pieces, then the finish.
fn mixed_stream() -> Vec<String> {
    let mut frames = vec![
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"
            .to_string(),
    ];
    frames.extend((0..60).map(|_| {
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"th\"}}]}\n\n"
            .to_string()
    }));
    frames.push(content_event(&"c".repeat(400)));
    frames.push(tool_identity_event());
    frames.extend((0..30).map(|i| arguments_event(&format!("{{\"k{i}\":1}}"))));
    frames.push(FINISH_EVENT.to_string());
    frames
}

fn run_watched(frames: &[String], watcher: Watcher) -> Vec<u8> {
    let mut r = SseRechunker::with_observer(policy(), watcher);
    let mut all = Vec::new();
    for frame in frames {
        all.extend_from_slice(r.feed(bytes(frame)).as_ref());
    }
    all.extend_from_slice(r.finish().as_ref());
    all
}

#[test]
fn the_observer_sees_every_slice_in_the_order_the_client_does() {
    let watcher = Watcher::recording();
    let all = run_watched(&mixed_stream(), watcher.clone());
    let mut expected = vec![Seen::Open];
    expected.extend(emitted_slices(&all));
    expected.push(Seen::End);
    let seen = watcher.seen();
    assert_eq!(seen, expected);
    // Every kind of slice was in it, more than one of each, and the text
    // of a field came through whole.
    let of = |field: DeltaField| -> Vec<&str> {
        seen.iter()
            .filter_map(|s| match s {
                Seen::Slice(f, text) if *f == field => Some(text.as_str()),
                _ => None,
            })
            .collect()
    };
    let reasoning = of(DeltaField::Text("reasoning_content"));
    let content = of(DeltaField::Text("content"));
    let arguments = of(DeltaField::ToolArguments(Some(0)));
    assert!(reasoning.len() > 1 && content.len() > 1 && arguments.len() > 1);
    assert_eq!(reasoning.concat(), "th".repeat(60));
    assert_eq!(content.concat(), "c".repeat(400));
    let pieces: String = (0..30).map(|i| format!("{{\"k{i}\":1}}")).collect();
    assert_eq!(arguments.concat(), pieces);
}

#[test]
fn an_observer_that_never_objects_leaves_the_bytes_as_they_are() {
    let frames = mixed_stream();
    let refs: Vec<&str> = frames.iter().map(String::as_str).collect();
    assert_eq!(run_watched(&frames, Watcher::recording()), run(&refs));
}

#[test]
fn a_vetoing_observer_ends_the_stream_with_the_error_event_and_nothing_after_it() {
    let watcher = Watcher::vetoing_after(2);
    let mut r = SseRechunker::with_observer(policy(), watcher.clone());
    // 500 characters: three slices are ready at once, the third is vetoed.
    let all = r.feed(bytes(content_event(&"a".repeat(500))));
    let text = String::from_utf8(all.to_vec()).unwrap();
    let evs = events(&all);
    assert_eq!(content_sizes(&evs), vec![160, 160]);
    assert_eq!(evs.len(), 3, "{text}");
    assert_eq!(
        evs[2],
        serde_json::json!({"error": {
            "message": "the response was ended by a policy",
            "type": "policy_violation",
        }})
    );
    assert!(text.ends_with(POLICY_ERROR_FRAME), "{text}");
    assert_eq!(r.stopped(), Some(&policy_stop()));
    // The vetoed slice and the remainder never leave, and neither does
    // anything the upstream sends afterwards: no finish, no [DONE].
    assert!(!r.has_pending());
    assert!(r.flush_pending().is_empty());
    assert!(r.feed(bytes(content_event("more"))).is_empty());
    assert!(r.feed(bytes(FINISH_EVENT)).is_empty());
    assert!(r.feed(bytes(b"data: [DONE]\n\n")).is_empty());
    assert!(r.finish().is_empty());
    // The observer saw the slice it vetoed and nothing after it; the
    // stream was its to end, so there is no end call.
    let seen = watcher.seen();
    assert_eq!(seen.len(), 4, "{seen:?}");
    assert_eq!(seen[0], Seen::Open);
    assert!(matches!(&seen[3], Seen::Slice(DeltaField::Text("content"), t) if t.len() == 160));
}

#[test]
fn a_veto_on_the_tail_replaces_it_with_the_error_event() {
    let watcher = Watcher::vetoing_after(0);
    let mut r = SseRechunker::with_observer(policy(), watcher.clone());
    // Held back: too short for a slice until the stream ends.
    assert!(r.feed(bytes(content_event("hi"))).is_empty());
    assert_eq!(r.finish().as_ref(), POLICY_ERROR_FRAME.as_bytes());
    assert_eq!(
        watcher.seen(),
        vec![
            Seen::Open,
            Seen::Slice(DeltaField::Text("content"), "hi".to_owned())
        ]
    );
}

#[test]
fn a_veto_on_arguments_ends_the_stream_too() {
    let mut r = SseRechunker::with_observer(policy(), Watcher::vetoing_after(0));
    let mut all = Vec::new();
    all.extend_from_slice(r.feed(bytes(tool_identity_event())).as_ref());
    all.extend_from_slice(r.feed(bytes(arguments_event(&"x".repeat(100)))).as_ref());
    let evs = events(&all);
    assert_eq!(evs.len(), 2, "{}", String::from_utf8_lossy(&all));
    assert_eq!(
        evs[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        Value::from("write_file")
    );
    assert_eq!(evs[1]["error"]["type"], Value::from("policy_violation"));
    assert!(r.feed(bytes(FINISH_EVENT)).is_empty());
}

#[tokio::test(start_paused = true)]
async fn stream_adapter_ends_on_the_observers_stop_and_lets_the_upstream_go() {
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
    let mut out = Box::pin(rechunk_stream(
        policy(),
        ReceiverStream::new(rx),
        Watcher::vetoing_after(0),
    ));
    tx.send(Ok(chunk(&"a".repeat(100)))).await.unwrap();
    let frame = out.next().await.unwrap().unwrap();
    assert_eq!(frame.as_ref(), POLICY_ERROR_FRAME.as_bytes());
    // The stream ends while the upstream is still sending, and ending it
    // lets the upstream go: the sender finds its reader gone, and what it
    // sends next has nowhere to go.
    assert!(out.next().await.is_none());
    assert!(tx.is_closed());
    assert!(tx.send(Ok(chunk("more"))).await.is_err());
}
