use serde_json::json;
use symphony::{adapt, CallEnd, Event, Events, FinishReason, Text};

use super::{messages, object_inputs, tool_use_input};

/// An answer's `tool_use.input` is always an object, as the request side
/// requires it: the parsed object as it is; an empty object for missing
/// arguments, arguments that are not JSON, and arguments that are JSON
/// but not an object (a double-encoded string, a bare string, an array, a
/// number, null).
#[test]
fn a_tool_use_input_is_always_an_object() {
    assert_eq!(
        tool_use_input(Some(r#"{"city":"Paris"}"#), "call_1"),
        json!({"city": "Paris"})
    );
    assert_eq!(tool_use_input(Some("{}"), "call_1"), json!({}));
    for arguments in [
        None,
        Some("not json"),
        Some(r#""{\"city\":\"Paris\"}""#),
        Some(r#""Paris""#),
        Some(r#"["Paris"]"#),
        Some("1"),
        Some("null"),
    ] {
        assert_eq!(
            tool_use_input(arguments, "call_1"),
            json!({}),
            "{arguments:?}"
        );
    }
}

/// The Symphony reader's collected answer keeps the same contract: the
/// adapter holds a whole value of any kind as the block's `input` (an
/// object as it is, `"Paris"` or `[1, 2]` as written, none of them
/// refused), and the reader leaves every non-object as `{}`.
#[test]
fn a_symphony_tool_use_input_is_an_object() {
    for (arguments, expected) in [
        (r#"{"city":"Paris"}"#, json!({"city": "Paris"})),
        (r#""Paris""#, json!({})),
        (r#""{\"city\":\"Paris\"}""#, json!({})),
        ("[1, 2]", json!({})),
    ] {
        let mut events = Events::new();
        events.push(Event::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
            source: Text::new("", 0),
        });
        events.push(Event::ToolCallArguments {
            index: 0,
            json: arguments.into(),
            source: Text::new("", 0),
        });
        events.push(Event::ToolCallEnd {
            index: 0,
            source: Text::new("", 0),
            ended: CallEnd::Closed,
        });
        events.push(Event::Finish {
            reason: FinishReason::ToolCalls,
            tool_calls: 1,
            reasoning_tokens: 0,
        });
        let mut output = adapt::messages::output(&events);
        assert!(
            output.refused.is_empty(),
            "{arguments}: the adapter holds the value"
        );
        object_inputs(&mut output.content, "read_messages_with_symphony");
        let messages::ContentBlock::ToolUse { input, .. } = &output.content[0] else {
            panic!("{arguments}: no tool_use block");
        };
        assert_eq!(input, &expected, "{arguments}");
    }
}
