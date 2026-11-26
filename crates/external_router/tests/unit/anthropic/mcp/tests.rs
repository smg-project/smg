use serde_json::json;
use smg_mcp::Tool as McpTool;

use super::*;
use crate::test_support::counting_alloc;

#[test]
fn mcp_tool_schema_keeps_the_servers_key_order() {
    // More keys than a hash map would keep in order by chance.
    let names = [
        "zeta", "alpha", "mu", "beta", "omega", "delta", "kappa", "gamma",
    ];
    let properties: Map<String, Value> = names
        .iter()
        .map(|name| (name.to_string(), json!({"type": "string"})))
        .collect();
    let schema = json!({
        "type": "object",
        "properties": properties,
        "required": ["mu", "alpha"],
        "title": "Lookup",
        "additionalProperties": false,
        "$defs": {}
    });
    let tool = McpTool::new(
        "lookup".to_string(),
        "look a value up",
        schema.as_object().unwrap().clone(),
    );
    let entry = ToolEntry::from_server_tool("server", tool);
    let custom = convert_tool_entry_to_anthropic_tool(&entry, false);
    assert_eq!(
        serde_json::to_string(&custom.input_schema).unwrap(),
        serde_json::to_string(&schema).unwrap()
    );
}

/// The answer's `mcp_tool_result` blocks say whether each call failed, as
/// the stream's do: `false` for a call that succeeded, `true` for one
/// that failed, never `null`.
#[test]
fn the_answers_mcp_tool_results_say_whether_each_call_failed() {
    let message: Message = serde_json::from_value(json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn", "stop_sequence": null, "stop_details": null,
        "usage": {"input_tokens": 1, "output_tokens": 1}, "container": null
    }))
    .unwrap();
    let call = |id: &str, is_error| McpToolCall {
        original_id: format!("toolu_{id}"),
        mcp_id: format!("mcptoolu_{id}"),
        name: "lookup".to_string(),
        server_name: "srv".to_string(),
        input: json!({}),
        result_content: "42".to_string(),
        is_error,
    };
    let answer =
        rebuild_response_with_mcp_blocks(message, &[call("1", false), call("2", true)], &[]);
    let answer = serde_json::to_value(answer).unwrap();
    let results: Vec<&Value> = answer["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["type"] == "mcp_tool_result")
        .collect();
    assert_eq!(results.len(), 2, "{answer}");
    assert_eq!(results[0]["is_error"], json!(false));
    assert_eq!(results[1]["is_error"], json!(true));
}

/// Every message as one line, `role: text` or `role: [block, ...]`, a
/// block named by its kind and id or text.
fn turns(request: &CreateMessageRequest) -> Vec<String> {
    let label = |block: &InputContentBlock| match block {
        InputContentBlock::ToolUse(call) => format!("tool_use {}", call.id),
        InputContentBlock::ToolResult(result) => format!("tool_result {}", result.tool_use_id),
        InputContentBlock::Text(text) => format!("text {}", text.text),
        other => format!("{other:?}"),
    };
    request
        .messages
        .iter()
        .map(|message| {
            let role = format!("{:?}", message.role).to_lowercase();
            match &message.content {
                InputContent::String(text) => format!("{role}: {text}"),
                InputContent::Blocks(blocks) => {
                    let blocks: Vec<String> = blocks.iter().map(label).collect();
                    format!("{role}: [{}]", blocks.join(", "))
                }
            }
        })
        .collect()
}

/// The content blocks of `messages[index]`.
fn blocks_of(request: &CreateMessageRequest, index: usize) -> &[InputContentBlock] {
    match &request.messages[index].content {
        InputContent::Blocks(blocks) => blocks,
        InputContent::String(text) => panic!("messages.{index} is the text {text:?}"),
    }
}

/// A client replaying an earlier MCP answer sends its `mcp_tool_use` and
/// `mcp_tool_result` blocks back; the upstream gets the plain pair this
/// loop sent the first time, the call in the assistant turn and the
/// result in a user turn right after it, and every other block as it came.
#[test]
fn a_replayed_mcp_exchange_goes_upstream_as_the_plain_pair() {
    let mut request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": [
                {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false},
                {"type": "text", "text": "it is 42"}
            ]},
            {"role": "user", "content": "thanks"}
        ]
    }))
    .expect("a request");
    plain_tool_pairs(&mut request.messages);
    assert_eq!(
        turns(&request),
        [
            "user: look it up",
            "assistant: [tool_use mcptoolu_1]",
            "user: [tool_result mcptoolu_1]",
            "assistant: [text it is 42]",
            "user: thanks",
        ]
    );
    match blocks_of(&request, 1) {
        [InputContentBlock::ToolUse(call)] => {
            assert_eq!(call.name, "lookup");
            assert_eq!(call.input, json!({"q": "x"}));
        }
        other => panic!("the plain call alone, not {other:?}"),
    }
    match blocks_of(&request, 2) {
        [InputContentBlock::ToolResult(result)] => {
            assert_eq!(result.is_error, Some(false));
            assert_eq!(
                serde_json::to_value(&result.content).unwrap(),
                json!([{"type": "text", "text": "42"}])
            );
        }
        other => panic!("the plain result alone, not {other:?}"),
    }
}

/// Two calls in one replayed answer, each with its result and a text
/// after it: a user turn per result, and the text between them opens the
/// assistant's next turn with the second call.
#[test]
fn a_replayed_exchange_of_two_calls_opens_a_user_turn_per_result() {
    let mut request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "look both up"},
            {"role": "assistant", "content": [
                {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false},
                {"type": "text", "text": "x is 42"},
                {"type": "mcp_tool_use", "id": "mcptoolu_2", "name": "lookup", "server_name": "s", "input": {"q": "y"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_2", "content": [{"type": "text", "text": "7"}], "is_error": false},
                {"type": "text", "text": "y is 7"}
            ]}
        ]
    }))
    .expect("a request");
    plain_tool_pairs(&mut request.messages);
    assert_eq!(
        turns(&request),
        [
            "user: look both up",
            "assistant: [tool_use mcptoolu_1]",
            "user: [tool_result mcptoolu_1]",
            "assistant: [text x is 42, tool_use mcptoolu_2]",
            "user: [tool_result mcptoolu_2]",
            "assistant: [text y is 7]",
        ]
    );
}

/// A non-streamed answer writes two parallel calls as sequential pairs;
/// the replay keeps them as one assistant turn with one user turn of
/// results, the shape the loop sent the first time.
#[test]
fn a_replayed_answer_of_sequential_pairs_keeps_its_parallel_calls_in_one_turn() {
    let mut request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "look both up"},
            {"role": "assistant", "content": [
                {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false},
                {"type": "mcp_tool_use", "id": "mcptoolu_2", "name": "lookup", "server_name": "s", "input": {"q": "y"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_2", "content": [{"type": "text", "text": "7"}], "is_error": false},
                {"type": "text", "text": "x is 42, y is 7"}
            ]}
        ]
    }))
    .expect("a request");
    plain_tool_pairs(&mut request.messages);
    assert_eq!(
        turns(&request),
        [
            "user: look both up",
            "assistant: [tool_use mcptoolu_1, tool_use mcptoolu_2]",
            "user: [tool_result mcptoolu_1, tool_result mcptoolu_2]",
            "assistant: [text x is 42, y is 7]",
        ]
    );
}

/// A replayed answer that ends with its result closes on the user turn
/// that carries it, nothing after.
#[test]
fn a_replayed_answer_ending_with_its_result_closes_on_the_user_turn() {
    let mut request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": [
                {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false}
            ]}
        ]
    }))
    .expect("a request");
    plain_tool_pairs(&mut request.messages);
    assert_eq!(
        turns(&request),
        [
            "user: look it up",
            "assistant: [tool_use mcptoolu_1]",
            "user: [tool_result mcptoolu_1]",
        ]
    );
}

/// A request without an MCP block is left as it came, a plain tool pair
/// included.
#[test]
fn a_request_without_mcp_blocks_is_left_as_it_came() {
    let messages = json!([
        {"role": "user", "content": "look it up"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}
        ]}
    ]);
    let mut request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 8, "messages": messages
    }))
    .expect("a request");
    let before: Vec<InputMessage> = serde_json::from_value(messages).expect("messages");
    plain_tool_pairs(&mut request.messages);
    assert_eq!(
        serde_json::to_value(&request.messages).unwrap(),
        serde_json::to_value(&before).unwrap()
    );
}

/// A round's message as the model answers it: text, tool calls and a
/// thought, read from the wire shape.
fn round_message(blocks: usize) -> Message {
    let content: Vec<Value> = (0..blocks)
        .map(|i| match i % 3 {
            0 => json!({"type": "text", "text": format!("paragraph {i} of the answer, long enough to live on the heap")}),
            1 => json!({"type": "tool_use", "id": format!("toolu_{i}"), "name": "lookup",
                "input": {"query": format!("q{i}"), "limit": 10, "fields": ["a", "b", "c"]}}),
            _ => json!({"type": "thinking", "thinking": format!("step {i}"), "signature": "sig"}),
        })
        .collect();
    serde_json::from_value(json!({
        "id": "msg_1", "type": "message", "role": "assistant", "content": content,
        "model": "m", "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
    .expect("a message")
}

/// A block of a kind the answer does not read goes back in the next
/// round's history in its input form when the request reads it (a
/// `search_result` here, which only the request names); one the request
/// cannot read either is the loop's error, naming the block, so no tool
/// runs with a history the upstream would get incomplete.
#[test]
fn an_answer_block_the_next_round_cannot_carry_stops_the_loop() {
    let readable: Vec<ContentBlock> = serde_json::from_value(json!([
        {"type": "text", "text": "before"},
        {"type": "search_result", "source": "https://example.com", "title": "t",
         "content": [{"type": "text", "text": "found"}]},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
    ]))
    .unwrap();
    assert!(matches!(readable[1], ContentBlock::Other(_)));
    let blocks = build_assistant_content_blocks(&readable).expect("the request reads it");
    assert_eq!(blocks.len(), 3);
    assert!(
        matches!(blocks[1], InputContentBlock::SearchResult(_)),
        "{:?}",
        blocks[1]
    );

    let unreadable: Vec<ContentBlock> = serde_json::from_value(json!([
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "a_tool", "input": {}},
        {"type": "a_block_kind_of_no_api", "tool_use_id": "srvtoolu_1"},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
    ]))
    .unwrap();
    let error = build_assistant_content_blocks(&unreadable).expect_err("the loop stops");
    assert!(error.contains("'a_block_kind_of_no_api'"), "{error}");
}

/// What a round does with the message's content: the iteration result,
/// the conversion into the next request's input blocks, the blocks kept
/// for the final response. Taken and moved, the blocks are copied once
/// (the conversion); the round that copied them into the result and again
/// into the prior blocks allocated the two copies on top. The counts are
/// printed for the Numbers table and the moved shape is pinned.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
fn a_round_takes_the_message_content_and_copies_it_once() {
    let blocks = 6;
    let copied = round_message(blocks);
    let mut moved = round_message(blocks);
    let expected = serde_json::to_value(&copied.content).expect("blocks as JSON");

    // The shape before: cloned into the result, converted, cloned again.
    let before = counting_alloc::calls();
    let content_blocks = copied.content.clone();
    let tool_use_blocks = extract_tool_calls(&content_blocks);
    let assistant = build_assistant_content_blocks(&content_blocks).expect("readable blocks");
    let mut prior: Vec<ContentBlock> = Vec::new();
    prior.extend(copied.content.clone());
    let copied_calls = counting_alloc::calls() - before;
    drop((content_blocks, tool_use_blocks, assistant, prior));

    // The shape now: taken, converted, moved.
    let before = counting_alloc::calls();
    let result = IterationResult::take_from(&mut moved);
    let assistant =
        build_assistant_content_blocks(&result.content_blocks).expect("readable blocks");
    let mut prior: Vec<ContentBlock> = Vec::new();
    prior.extend(result.content_blocks);
    let moved_calls = counting_alloc::calls() - before;

    eprintln!(
        "a round's content handling, {blocks} blocks: copied shape {copied_calls} allocation calls, moved shape {moved_calls}"
    );
    assert!(moved.content.is_empty(), "the message's content is taken");
    assert_eq!(
        serde_json::to_value(&prior).expect("blocks as JSON"),
        expected
    );
    assert_eq!(assistant.len(), blocks);
    assert_eq!(result.tool_use_blocks.len(), blocks / 3);
    assert!(moved_calls < copied_calls, "{moved_calls} < {copied_calls}");
    // Pinned at today's count: the conversion of the blocks (about 32
    // calls for these six), the tool-call extraction's copies (about
    // 25) and the prior vector's buffer (1).
    assert_eq!(
        moved_calls, PINNED_ROUND_CALLS,
        "the moved shape's allocation calls changed"
    );
}

/// Allocation calls of the moved shape for six blocks, today.
const PINNED_ROUND_CALLS: usize = 58;
