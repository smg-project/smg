use serde_json::json;

use super::*;
use crate::routers::grpc::{spec::ChatResponseSpec, utils::request_as_rendered};

fn names(tools: Option<&[Tool]>) -> Vec<&str> {
    tools
        .unwrap_or_default()
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect()
}

#[test]
fn the_final_turn_renders_the_requests_own_tools_and_is_read_against_every_tool() {
    let mut chat_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "final-turn-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}},
        ],
    }))
    .unwrap();
    let state = ToolLoopState {
        iteration: 2,
        final_turn: true,
        ..Default::default()
    };

    prepare_tool_choice(&mut chat_request, &state, |name| name == "search");

    let rendered = request_as_rendered(&chat_request);
    assert_eq!(names(rendered.tools.as_deref()), ["lookup"]);
    let spec = ChatResponseSpec::from(&chat_request);
    assert_eq!(names(spec.tools.as_deref()), ["lookup", "search"]);
}
