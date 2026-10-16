//! GLM: reasoning between `<think>` and `</think>`, each call between `<tool_call>` and
//! `</tool_call>` as its name and keyed arguments (`<arg_key>`, `<arg_value>`), typed by the
//! request's tools, with nothing between the tags. Everything else is content. The prompt opens
//! the thought (`<|assistant|><think>`, whatever the request says: the template has no thinking
//! switch, only `clear_thinking` for the history), so an empty thought writes `</think>` at once;
//! a turn opens with `<|assistant|>`, so the prompt's replay starts there. The template strips the
//! content's edges and writes the calls right after it, so a call follows the content's last byte.
//! Recorded as `glm-5.3-flash` (zai-org/GLM-5.3-Flash). Ling writes the same syntax with newlines
//! between the tags and under its own turn opener ([`ling()`](super::ling())).

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The GLM table.
pub fn glm() -> Format {
    Format::new("glm")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("call", "call_close", "content")
        .transition("call", "call_open", "call")
        .calls(CallSyntax::Keyed(keyed::Tags::PLAIN))
        .opens_turn("<|assistant|>")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "ChaDri_change_drink".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "drink_id": {"type": "string"},
                    "new_preferences": {"type": "object"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The generation prompt's tail: the turn opened, and its thought.
    const PROMPT: &str = "<|assistant|><think>";

    /// A recorded output: an empty thought, no content, one call with a string and an object.
    const OUTPUT: &str = concat!(
        "</think><tool_call>ChaDri_change_drink<arg_key>drink_id</arg_key><arg_value>latte",
        "</arg_value><arg_key>new_preferences</arg_key><arg_value>",
        "{\"size\": \"large\", \"temperature\": \"hot\"}</arg_value></tool_call>"
    );

    /// The tools of an agentic turn, as a coding agent declares them: strings, numbers, an array.
    fn coding_tools() -> Declared {
        let tool = |name: &str, parameters: serde_json::Value| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict: None,
                extra: Default::default(),
            },
        };
        Declared::of(&[
            tool(
                "Read",
                value!({"type": "object", "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": "number"},
                }}),
            ),
            tool(
                "Bash",
                value!({"type": "object", "properties": {"command": {"type": "string"}}}),
            ),
            tool(
                "Ls",
                value!({"type": "object", "properties": {"path": {"type": "string"}}}),
            ),
            tool(
                "TodoWrite",
                value!({"type": "object", "properties": {"todos": {"type": "array"}}}),
            ),
        ])
    }

    /// The assistant turns the model's chat template renders for a coding agent's requests,
    /// after the prompt's `<think>`: a thought, content and three calls (strings and numbers), and
    /// a thought and two calls with no content (a string and an array). What the template writes
    /// for a request's `tool_calls` is what the table reads back.
    const AGENT_TURN: &str = concat!(
        "The user wants three things at once.</think>I'll read both files and check the status.",
        "<tool_call>Read<arg_key>file_path</arg_key><arg_value>/workspace/cli/help.go</arg_value>",
        "</tool_call><tool_call>Read<arg_key>file_path</arg_key>",
        "<arg_value>/workspace/cli/reproduce_issue_test.go</arg_value><arg_key>limit</arg_key>",
        "<arg_value>120</arg_value><arg_key>offset</arg_key><arg_value>85</arg_value></tool_call>",
        "<tool_call>Bash<arg_key>command</arg_key><arg_value>git status --short</arg_value>",
        "</tool_call>"
    );
    const AGENT_TURN_WITHOUT_CONTENT: &str = concat!(
        "Two calls, no prose.</think><tool_call>Ls<arg_key>path</arg_key>",
        "<arg_value>/workspace/cli</arg_value></tool_call><tool_call>TodoWrite",
        "<arg_key>todos</arg_key>",
        "<arg_value>[{\"content\": \"Fix help.go\", \"status\": \"in_progress\"}]</arg_value>",
        "</tool_call>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        run_with(declared(), prompt, pieces)
    }

    fn run_with(declared: Declared, prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(glm(), declared);
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        for piece in pieces {
            parser
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: piece,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("delta");
        }
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        out.drain()
    }

    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
    }

    fn arguments_of(events: &[Event], call: u32) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { index, json, .. } if *index == call => {
                    Some(json.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn content(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Each call's name and its arguments as a value, in order.
    fn calls_of(events: &[Event]) -> Vec<(String, serde_json::Value)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { index, name, .. } => Some((
                    name.clone(),
                    serde_json::from_str(&arguments_of(events, *index)).expect("arguments"),
                )),
                _ => None,
            })
            .collect()
    }

    fn dropped(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn a_recorded_output_gives_its_call_typed_by_the_tools_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                dropped("</think>"),
                Event::ReasoningEnd,
                dropped("<tool_call>"),
            ]
        );
        assert!(matches!(
            &events[4],
            Event::ToolCallStart { index: 0, name, .. } if name == "ChaDri_change_drink"
        ));
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"drink_id": "latte", "new_preferences": {"size": "large", "temperature": "hot"}}"#
        );
        assert_eq!(content(&events), "");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_thought_and_an_answer_are_the_regions_bytes() {
        let output =
            "Step 1: read.\n\nStep 2: answer.</think>Here it is:\n\n```python\nprint(1)\n```";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..3],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("Step 1: read.\n\nStep 2: answer.")),
                dropped("</think>"),
            ]
        );
        assert_eq!(content(&events), "Here it is:\n\n```python\nprint(1)\n```");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn thinking_off_still_starts_inside_the_thought_the_prompt_opened() {
        // The template has no thinking switch: the generation prompt ends with `<think>` whatever
        // the request says, and the model closes the empty thought at once.
        let events = run(PROMPT, &["</think>Hello!"]);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                dropped("</think>"),
                Event::ReasoningEnd,
                Event::Content(Text::uncounted("Hello!")),
            ]
        );
    }

    #[test]
    fn content_runs_to_the_calls_last_byte_and_two_calls_follow_each_other() {
        let output = concat!(
            "</think>Two drinks.<tool_call>ChaDri_change_drink<arg_key>drink_id</arg_key>",
            "<arg_value>123</arg_value></tool_call><tool_call>ChaDri_change_drink",
            "<arg_key>drink_id</arg_key><arg_value>tea</arg_value></tool_call>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Two drinks.");
        // A declared string keeps the digits as text.
        assert_eq!(arguments_of(&events, 0), r#"{"drink_id": "123"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"drink_id": "tea"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = run(PROMPT, &[OUTPUT]);
        let whole_arguments = arguments_of(&whole, 0);
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run(PROMPT, &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(arguments_of(&events, 0), whole_arguments, "cut at {cut}");
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_stream_cut_inside_a_call_ends_the_call_with_its_arguments_cut() {
        let output =
            "</think><tool_call>ChaDri_change_drink<arg_key>drink_id</arg_key><arg_value>lat";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        // The declared string streamed as it arrived and stays open: the client sees the cut.
        assert_eq!(arguments_of(&events, 0), r#"{"drink_id": "lat"#);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 0, source } if source.text.is_empty()
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn an_earlier_turns_call_in_the_prompt_moves_nothing() {
        // The replay starts at the last `<|assistant|>`, so an earlier turn's call block, closed or
        // not, and a user's quoted marker are not entered.
        for earlier in [
            "<think></think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
            "<think></think><tool_call>f<arg_key>a</arg_key>",
        ] {
            let prompt = format!(
                "<|user|>Hi<|assistant|>{earlier}<|observation|>ok<|user|>Why <tool_call>?{PROMPT}"
            );
            let events = run(&prompt, &["A plan.</think>An answer."]);
            assert_eq!(events[0], Event::ReasoningStart, "{earlier:?}");
            assert_eq!(
                events[1],
                Event::Reasoning(Text::uncounted("A plan.")),
                "{earlier:?}"
            );
            assert_eq!(content(&events), "An answer.", "{earlier:?}");
        }
    }

    #[test]
    fn the_templates_agent_turns_read_back_as_the_calls_that_rendered_them() {
        let events = run_with(coding_tools(), PROMPT, &[AGENT_TURN]);
        assert_eq!(bytes(&events), AGENT_TURN);
        assert_eq!(reasoning(&events), "The user wants three things at once.");
        assert_eq!(
            content(&events),
            "I'll read both files and check the status."
        );
        assert_eq!(
            calls_of(&events),
            [
                (
                    "Read".to_string(),
                    value!({"file_path": "/workspace/cli/help.go"})
                ),
                (
                    "Read".to_string(),
                    value!({"file_path": "/workspace/cli/reproduce_issue_test.go", "limit": 120, "offset": 85})
                ),
                (
                    "Bash".to_string(),
                    value!({"command": "git status --short"})
                ),
            ]
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 3, .. })
        ));
        let events = run_with(coding_tools(), PROMPT, &[AGENT_TURN_WITHOUT_CONTENT]);
        assert_eq!(bytes(&events), AGENT_TURN_WITHOUT_CONTENT);
        assert_eq!(reasoning(&events), "Two calls, no prose.");
        assert_eq!(content(&events), "");
        assert_eq!(
            calls_of(&events),
            [
                ("Ls".to_string(), value!({"path": "/workspace/cli"})),
                (
                    "TodoWrite".to_string(),
                    value!({"todos": [{"content": "Fix help.go", "status": "in_progress"}]})
                ),
            ]
        );
        // Every two-way split of the three-call turn says the same.
        let whole = calls_of(&run_with(coding_tools(), PROMPT, &[AGENT_TURN]));
        for cut in 1..AGENT_TURN.len() {
            let events = run_with(
                coding_tools(),
                PROMPT,
                &[&AGENT_TURN[..cut], &AGENT_TURN[cut..]],
            );
            assert_eq!(bytes(&events), AGENT_TURN, "cut at {cut}");
            assert_eq!(calls_of(&events), whole, "cut at {cut}");
        }
    }

    #[test]
    fn a_call_written_as_python_where_the_name_goes_names_no_call() {
        // On agentic turns the model sometimes writes a call as Python inside the block's markers,
        // whole, or cut short by a tag with the rest of the block after it. Neither is a name: the
        // block names no call and its bytes come back, and the well-formed call after it is the
        // turn's one call.
        let output = concat!(
            "</think>I'll read both files.",
            "<tool_call>Read(filed_path</arg_key><arg_value>/workspace/reproduce_issue.java",
            "</arg_value>></tool_call>",
            "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls -la /workspace</arg_value>",
            "</tool_call>"
        );
        let events = run_with(coding_tools(), PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls_of(&events),
            [("Bash".to_string(), value!({"command": "ls -la /workspace"}))]
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, .. } if text.text == "Read(filed_path"
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        let output =
            "</think>Let me look:<tool_call>Read(file_path=\"/workspace/x.java\")</tool_call>";
        let events = run_with(coding_tools(), PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert!(calls_of(&events).is_empty());
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
        // A whole pythonic call with a number and one with no arguments: no quote refuses
        // these, only the parentheses rule does.
        for output in [
            "</think>Let me look:<tool_call>Read(offset=10)</tool_call>",
            "</think>Let me look:<tool_call>Ls()</tool_call>",
        ] {
            let events = run_with(coding_tools(), PROMPT, &[output]);
            assert_eq!(bytes(&events), output);
            assert!(calls_of(&events).is_empty(), "{output}");
            assert!(
                events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{output}"
            );
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 0, .. })),
                "{output}"
            );
        }
    }
}
