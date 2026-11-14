//! What the conversation item records allocate, pinned at today's shape: the
//! object the Conversations API answers for a stored item moves the item's
//! content instead of copying it part by part, and the record a turn's item is
//! stored under moves the item instead of copying it once more, so the next
//! change that brings a content-sized copy back fails a test rather than
//! moving a bench inside its noise. A separate executable, so it can declare
//! the counting allocator the pins need.
#![expect(
    clippy::print_stderr,
    reason = "the counts a pin reads are printed for the Numbers table"
)]

mod common;

use std::{sync::Arc, time::Instant};

use chrono::Utc;
use common::counting_alloc::{bytes, calls, Counting};
use openai_protocol::responses::ResponsesRequest;
use serde_json::{json, Value};
use smg_data_connector::{
    ConversationItem, ConversationItemId, ConversationItemStorage, ConversationStorage,
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    NewConversation, ResponseStorage,
};
use smg_external_router::persistence_utils::{item_to_json, persist_conversation_items};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls and bytes this thread made
/// during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize, usize) {
    let (calls_before, bytes_before) = (calls(), bytes());
    let out = work();
    (out, calls() - calls_before, bytes() - bytes_before)
}

/// A stored assistant message of `parts` text parts, each a short sentence.
fn message(parts: usize) -> ConversationItem {
    let part = json!({
        "type": "output_text",
        "text": "a dozen words of answer text that a model wrote as one part ",
        "annotations": []
    });
    ConversationItem {
        id: ConversationItemId::from("msg_0123456789abcdef0123456789abcdef"),
        response_id: Some("resp_0123456789abcdef".to_string()),
        item_type: "message".to_string(),
        role: Some("assistant".to_string()),
        content: Value::Array(vec![part; parts]),
        status: Some("completed".to_string()),
        created_at: Utc::now(),
        tenant: None,
    }
}

/// A stored tool call whose output is `output_len` bytes long (a listed type:
/// the object carries the listed fields of its content).
fn tool_call(output_len: usize) -> ConversationItem {
    ConversationItem {
        id: ConversationItemId::from("mcp_0123456789abcdef0123456789abcdef"),
        response_id: Some("resp_0123456789abcdef".to_string()),
        item_type: "mcp_call".to_string(),
        role: None,
        content: json!({
            "name": "search",
            "arguments": "{\"query\":\"q\"}",
            "output": "x".repeat(output_len),
            "server_label": "tools"
        }),
        status: Some("completed".to_string()),
        created_at: Utc::now(),
        tenant: None,
    }
}

/// An item stored whole (the shape a replayed structural item is stored in)
/// whose result is `len` bytes long: the object carries its fields hoisted.
fn whole_item(len: usize) -> ConversationItem {
    ConversationItem {
        id: ConversationItemId::from("ig_0123456789abcdef0123456789abcdef"),
        response_id: Some("resp_0123456789abcdef".to_string()),
        item_type: "image_generation_call".to_string(),
        role: None,
        content: json!({
            "id": "ig_0123456789abcdef0123456789abcdef",
            "type": "image_generation_call",
            "status": "completed",
            "revised_prompt": "a cat",
            "result": "x".repeat(len)
        }),
        status: Some("completed".to_string()),
        created_at: Utc::now(),
        tenant: None,
    }
}

/// The object of a message allocates the same whether its content is one part
/// or sixty-four: the parts move into the object, nothing of them is copied.
/// (A copy of the content grew with the parts: a map, two keys and two strings
/// per part.)
#[test]
fn a_message_object_allocates_the_same_whatever_its_content_length() {
    let one = message(1);
    let many = message(64);
    let (short, calls_one, bytes_one) = allocations_during(|| item_to_json(one));
    let (long, calls_many, bytes_many) = allocations_during(|| item_to_json(many));
    assert_eq!(short["content"].as_array().map(Vec::len), Some(1));
    assert_eq!(long["content"].as_array().map(Vec::len), Some(64));
    eprintln!(
        "allocation calls for a message object: one part {calls_one} ({bytes_one} bytes), sixty-four parts {calls_many} ({bytes_many} bytes)"
    );
    assert_eq!(
        calls_many, calls_one,
        "allocation calls for a sixty-four-part message object against a one-part one"
    );
}

/// The object of a tool call whose output is 64 KiB long allocates under half
/// that: the listed fields move out of the content, the output is not copied.
#[test]
fn a_tool_call_object_does_not_copy_its_output() {
    let len = 64 * 1024;
    let call = tool_call(len);
    let (object, allocation_calls, allocated) = allocations_during(|| item_to_json(call));
    assert_eq!(object["output"].as_str().map(str::len), Some(len));
    eprintln!(
        "allocation calls for a tool call object with a 64 KiB output: {allocation_calls} ({allocated} bytes)"
    );
    assert!(
        allocated < len / 2,
        "bytes allocated for a tool call object with a 64 KiB output: {allocated}"
    );
}

/// The object of an item stored whole, with a 64 KiB result, allocates under
/// half that: its fields move up into the object, the result is not copied.
#[test]
fn a_whole_item_object_does_not_copy_its_fields() {
    let len = 64 * 1024;
    let item = whole_item(len);
    let (object, allocation_calls, allocated) = allocations_during(|| item_to_json(item));
    assert_eq!(object["result"].as_str().map(str::len), Some(len));
    assert_eq!(object["revised_prompt"], json!("a cat"));
    eprintln!(
        "allocation calls for a whole item object with a 64 KiB result: {allocation_calls} ({allocated} bytes)"
    );
    assert!(
        allocated < len / 2,
        "bytes allocated for a whole item object with a 64 KiB result: {allocated}"
    );
}

/// The time a page of a thousand sixty-four-part message objects takes, per
/// object, printed for the Numbers table (not pinned: a clock, not a count).
#[test]
fn a_page_of_message_objects_timed() {
    let page: Vec<ConversationItem> = (0..1000).map(|_| message(64)).collect();
    let started = Instant::now();
    let objects: Vec<Value> = page.into_iter().map(item_to_json).collect();
    let elapsed = started.elapsed();
    assert_eq!(objects.len(), 1000);
    eprintln!(
        "a thousand sixty-four-part message objects: {:.1} us per object",
        elapsed.as_secs_f64() * 1e6 / 1000.0
    );
}

/// A response of eight messages whose texts are 64 KiB each.
fn response(id: &str) -> Value {
    let output: Vec<Value> = (0..8)
        .map(|i| {
            json!({
                "type": "message",
                "id": format!("msg_{i}"),
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "x".repeat(64 * 1024), "annotations": []}]
            })
        })
        .collect();
    json!({
        "id": id,
        "object": "response",
        "model": "m",
        "status": "completed",
        "output": output
    })
}

/// The three memory stores a persist writes.
type Stores = (
    Arc<dyn ConversationStorage>,
    Arc<dyn ConversationItemStorage>,
    Arc<dyn ResponseStorage>,
);

fn stores() -> Stores {
    (
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
        Arc::new(MemoryResponseStorage::new()),
    )
}

/// Linking a turn's items into a conversation stores each item's content once
/// (the memory store's own copy of the record it keeps) and not again on the
/// way: the bytes a persist with a conversation allocates beyond the same
/// persist without one stay under one and a half times the items' content.
/// (The record built from a borrowed item copied the content once more: two
/// times the content and above.)
#[test]
fn linking_a_turns_items_stores_each_once() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");
    let content_bytes = 8 * 64 * 1024;

    let (conversations, items, responses) = stores();
    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi"})).expect("a request");
    let alone = response("resp_alone");
    let (stored, _, bytes_alone) = allocations_during(|| {
        runtime.block_on(persist_conversation_items(
            conversations,
            items,
            responses,
            &alone,
            &request,
            None,
            None,
        ))
    });
    stored.expect("the response is stored");

    let (conversations, items, responses) = stores();
    let conversation = runtime
        .block_on(conversations.create_conversation(NewConversation::default()))
        .expect("a conversation");
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "conversation": conversation.id.0,
    }))
    .expect("a request");
    let linked = response("resp_linked");
    let (stored, _, bytes_linked) = allocations_during(|| {
        runtime.block_on(persist_conversation_items(
            conversations,
            items,
            responses,
            &linked,
            &request,
            None,
            None,
        ))
    });
    stored.expect("the response is stored and linked");

    let linking = bytes_linked.saturating_sub(bytes_alone);
    eprintln!(
        "bytes allocated by a persist of eight 64 KiB messages: without a conversation {bytes_alone}, with one {bytes_linked}, the linking {linking} ({}% of the content)",
        linking * 100 / content_bytes
    );
    assert!(
        linking < content_bytes * 3 / 2,
        "bytes allocated by linking eight 64 KiB messages: {linking} (the content is {content_bytes})"
    );
}
