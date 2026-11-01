//! GLM: reasoning between `<think>` and `</think>`, each call between `<tool_call>` and
//! `</tool_call>` as its name and keyed arguments (`<arg_key>`, `<arg_value>`), typed by the
//! request's tools, with nothing between the tags. Everything else is content; what a call block
//! holds that the syntax cannot read (a tag out of its place, prose or Python where the name goes)
//! is never content: refused and counted (`keyed`). The prompt opens
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
        event::{DropReason, Event, FinishReason, MalformedReason, Text},
        input::EngineFinish,
        testing::{
            arguments_of, bytes, content, drive, drive_to, dropped, reasoning,
            the_same_at_every_cut,
        },
        Declared,
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

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        run_with(declared(), prompt, pieces)
    }

    fn run_with(declared: Declared, prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(&mut Engine::new(glm(), declared), prompt, pieces)
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
        let events = drive_to(
            &mut Engine::new(glm(), declared()),
            PROMPT,
            &[output],
            EngineFinish::Length,
        );
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
        // The model's own end of output there is no cut: the string and the object close, and
        // one `Malformed { OpenAtTheStop }` event says the stop closed them.
        let stopped = run(PROMPT, &[output]);
        assert_eq!(bytes(&stopped), output);
        assert_eq!(arguments_of(&stopped, 0), r#"{"drink_id": "lat"}"#);
        assert!(stopped.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::OpenAtTheStop } if text.text.is_empty()
        )));
        // The same call cut inside its object value, which the assembler holds for its close:
        // the member leaves as written, the cut member's key and bytes included, nothing of the
        // call is a failed call's bytes, and the finish is `length`; whole, one character at a
        // time and at every two-way cut.
        let output = concat!(
            "</think><tool_call>ChaDri_change_drink<arg_key>drink_id</arg_key><arg_value>latte",
            "</arg_value><arg_key>new_preferences</arg_key><arg_value>{\"size\": \"large\", \"temp"
        );
        let cut = |pieces: &[&str]| {
            drive_to(
                &mut Engine::new(glm(), declared()),
                PROMPT,
                pieces,
                EngineFinish::Length,
            )
        };
        let said = the_same_at_every_cut(cut, output);
        assert_eq!(
            said.calls,
            vec![(
                "ChaDri_change_drink".to_string(),
                r#"{"drink_id": "latte", "new_preferences": {"size": "large", "temp"#.to_string()
            )]
        );
        assert_eq!(said.finish, Some((FinishReason::Length, 1)));
        assert_eq!(said.content, "");
        let events = cut(&[output]);
        assert_eq!(bytes(&events), output);
        assert!(
            !events.iter().any(|event| matches!(
                event,
                Event::Dropped {
                    why: DropReason::MalformedCall,
                    ..
                }
            )),
            "nothing of the call is a failed call's bytes: {events:?}"
        );
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
        // block names no call, its bytes are refused (never content, the failure counted by events
        // of no bytes), and the well-formed call after it is the turn's one call.
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
        assert_eq!(content(&events), "I'll read both files.");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::TextWhereTheNameShouldBe }
                if text.text.is_empty()
        )));
        assert_eq!(
            refused(&events),
            "Read(filed_path</arg_key><arg_value>/workspace/reproduce_issue.java</arg_value>></tool_call>"
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        let output =
            "</think>Let me look:<tool_call>Read(file_path=\"/workspace/x.java\")</tool_call>";
        let events = run_with(coding_tools(), PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Let me look:");
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

    /// The tools of the recorded pythonic cases: a string, an array with an object.
    fn stock_tools() -> Declared {
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
        let schema =
            |properties: serde_json::Value| value!({"type": "object", "properties": properties});
        Declared::of(&[
            tool(
                "get_stock_info",
                schema(value!({"company": {"type": "string"}})),
            ),
            tool(
                "todo",
                schema(
                    value!({"type": {"type": "string"}, "items": {"type": "array"}, "when": {"type": "object"}}),
                ),
            ),
            tool(
                "searchHotels",
                schema(value!({"location": {"type": "string"}})),
            ),
        ])
    }

    /// A call written as Python inside the block's markers, right after the thought, as the
    /// recorded cells show it: whole with one string, whole with nested arguments, and twice
    /// with the second block cut by the end of the stream. None names a call: the block's bytes
    /// are refused whole (the text where the name goes, then the marker that ended it, as the
    /// call's own bytes; the opener that led into the block is the table's row), behind events
    /// of no bytes, the first of them saying that text stands where the name should be; nothing
    /// of the block is content, so the content is empty, and what an answer of nothing else
    /// shows a client is the reader's decision. The same call in the template's own form is the
    /// call, typed by the tool. Every chunking says the same.
    #[test]
    fn a_call_written_as_python_inside_the_tag_is_refused_whole_and_the_tagged_form_reads() {
        struct Case {
            output: &'static str,
            refused: &'static str,
        }
        let cases = [
            Case {
                output: "</think><tool_call>get_stock_info(company=\"Apple\")</tool_call>",
                refused: "get_stock_info(company=\"Apple\")</tool_call>",
            },
            Case {
                output: concat!(
                    "</think><tool_call>todo(type=\"add\", items=[{\"title\": \"Buy milk\", ",
                    "\"tags\": [\"home\", \"food\"]}], when={\"day\": 1, \"hour\": 9})</tool_call>"
                ),
                refused: concat!(
                    "todo(type=\"add\", items=[{\"title\": \"Buy milk\", ",
                    "\"tags\": [\"home\", \"food\"]}], when={\"day\": 1, \"hour\": 9})</tool_call>"
                ),
            },
            Case {
                output: concat!(
                    "</think><tool_call>searchHotels(location=\"San Diego\")",
                    "<tool_call>searchHotels(location=\"San Diego\")"
                ),
                refused: concat!(
                    "searchHotels(location=\"San Diego\")",
                    "<tool_call>searchHotels(location=\"San Diego\")"
                ),
            },
        ];
        for case in &cases {
            for pieces in plans(case.output) {
                let events = run_with(stock_tools(), PROMPT, &pieces);
                let place = || format!("{:?} in {} pieces", &case.output[..40], pieces.len());
                assert_eq!(bytes(&events), case.output, "{}", place());
                assert_eq!(content(&events), "", "{}", place());
                assert!(calls_of(&events).is_empty(), "{}", place());
                let malformed = malformed(&events);
                assert!(
                    malformed.iter().all(|(_, text)| text.is_empty()),
                    "{}: {malformed:?}",
                    place()
                );
                assert_eq!(
                    malformed.first().map(|(why, _)| *why),
                    Some(MalformedReason::TextWhereTheNameShouldBe),
                    "{}",
                    place()
                );
                assert_eq!(refused(&events), case.refused, "{}", place());
                assert!(
                    matches!(events.last(), Some(Event::Finish { tool_calls: 0, .. })),
                    "{}",
                    place()
                );
            }
        }
        // The template's own form of the first case: the call, with the string the tool
        // declares, and nothing refused.
        let output = concat!(
            "</think><tool_call>get_stock_info<arg_key>company</arg_key>",
            "<arg_value>Apple</arg_value></tool_call>"
        );
        for pieces in plans(output) {
            let events = run_with(stock_tools(), PROMPT, &pieces);
            assert_eq!(bytes(&events), output, "{} pieces", pieces.len());
            assert_eq!(
                calls_of(&events),
                [("get_stock_info".to_string(), value!({"company": "Apple"}))],
                "{} pieces",
                pieces.len()
            );
            assert_eq!(content(&events), "", "{} pieces", pieces.len());
            assert!(malformed(&events).is_empty(), "{} pieces", pieces.len());
            assert_eq!(refused(&events), "", "{} pieces", pieces.len());
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                "{} pieces",
                pieces.len()
            );
        }
    }

    /// A call written as Python inside the block's markers after the answer's text (the recorded
    /// shape: a sentence, then the block): the text is content, and the block is refused whole as
    /// when it stands alone (the text where the name goes, then the marker that ended it), behind
    /// events of no bytes, the first saying that text stands where the name should be; the
    /// content ends where the text does, and what a client sees of the block is the reader's
    /// decision. Every chunking says the same.
    #[test]
    fn a_call_written_as_python_after_the_text_is_refused_whole_and_the_text_is_content() {
        let output =
            "</think>Let me check.<tool_call>get_stock_info(company=\"Apple\")</tool_call>";
        for pieces in plans(output) {
            let events = run_with(stock_tools(), PROMPT, &pieces);
            let place = || format!("{} pieces", pieces.len());
            assert_eq!(bytes(&events), output, "{}", place());
            assert_eq!(content(&events), "Let me check.", "{}", place());
            assert!(calls_of(&events).is_empty(), "{}", place());
            let malformed = malformed(&events);
            assert!(
                malformed.iter().all(|(_, text)| text.is_empty()),
                "{}: {malformed:?}",
                place()
            );
            assert_eq!(
                malformed.first().map(|(why, _)| *why),
                Some(MalformedReason::TextWhereTheNameShouldBe),
                "{}",
                place()
            );
            assert_eq!(
                refused(&events),
                "get_stock_info(company=\"Apple\")</tool_call>",
                "{}",
                place()
            );
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 0, .. })),
                "{}",
                place()
            );
        }
    }

    /// A recorded output: one well-formed call whose last value close the model wrote twice.
    const RECORDED_STRAY_CLOSE: &str = concat!(
        "The user wants to perform a t-test between two datasets:\n- Dataset A: [12, 24, 36]\n",
        "- Dataset B: [15, 30, 45]\n\nI have the t_test function available. Let me call it with ",
        "these datasets.</think>I'll perform a t-test to check if there's a statistically significant ",
        "difference between these two datasets.<tool_call>t_test<arg_key>dataset_A</arg_key><arg_value>",
        "[12, 24, 36]</arg_value><arg_key>dataset_B</arg_key><arg_value>[15, 30, 45]</arg_value>",
        "</arg_value></tool_call>",
    );
    /// A recorded output: two well-formed calls, then two whose key tag never opened (the name,
    /// then the key's close and a value).
    const RECORDED_KEY_NEVER_OPENED: &str = concat!(
        "The user wants to know:\n1. President in 1980\n2. President in 2016\n3. Vice President ",
        "in 1975\n4. Vice President in 2011\n\nI have a tool `get_president_and_vp_for_year` ",
        "that takes a year and position. I need to make 4 calls. These are all independent, so ",
        "I can make them all in the same block.\n\nWait, the tool takes a single year parameter. ",
        "I need to call it 4 times - once for each year. Since they're independent, I should ",
        "make all calls in the same block.</think>I'll look up the presidents and vice presidents ",
        "for those years. Since these are independent lookups, I'll make all the calls at once:<tool_call>",
        "get_president_and_vp_for_year<arg_key>year</arg_key><arg_value>1980</arg_value></tool_call>",
        "<tool_call>get_president_and_vp_for_year<arg_key>year</arg_key><arg_value>2016</arg_value>",
        "</tool_call><tool_call>get_president_and_vp_for_year</arg_key><arg_value>1975</arg_value>",
        "</tool_call><tool_call>get_president_and_vp_for_year</arg_key><arg_value>2011</arg_value>",
        "</tool_call>",
    );
    /// A recorded output: a call whose keys are written `key: value` and closed by value closes,
    /// then a well-formed one.
    const RECORDED_KEYS_CLOSED_BY_VALUE_CLOSES: &str = concat!(
        "The user wants to:\n1. Get transaction history for account '00125648' for the past 7 ",
        "days\n2. Calculate the total balance\n\nI have two tools available:\n- bank_getTransactionHistory ",
        "- retrieves transaction history\n- bank_calculateBalance - calculates balance\n\nThese ",
        "are independent calls (the balance calculation doesn't depend on the transaction history ",
        "result), so I can make both calls in the same block.\n\nLet me call both functions.</think>",
        "I'll retrieve your transaction history and calculate your balance for you.<tool_call>",
        "bank_get_transaction_history<arg_key>account: </arg_value>\"00125648\"</arg_value><arg_key>",
        "days: 7</arg_value></tool_call><tool_call>bank_calculate_balance<arg_key>account</arg_key>",
        "<arg_value>00125648</arg_value></tool_call>",
    );
    /// A recorded output: two blocks written as Python with key closes inside, no call.
    const RECORDED_PYTHON_WITH_TAGS: &str = concat!(
        "The user wants me to calculate:\n1. 3^5 = 243\n2. 2^3 = 8\n\nThese are independent calculations, ",
        "so I can call the power function twice. Actually, I could make both calls in the same ",
        "block since they're independent.</think>I'll calculate both operations for you.<tool_call>",
        "math_power_function(base</arg_key>3,exponent</arg_key>5)</tool_call><tool_call>math_power_function(base</arg_key>",
        "2,exponent</arg_key>3)</tool_call>",
    );
    /// A recorded output: prose where a name goes, closed by the next call's opener, then a
    /// well-formed call; no content at all.
    const RECORDED_PROSE_THEN_A_CALL: &str = concat!(
        "The user wants to predict the evolutionary rate for Homo sapiens over the next 50 years ",
        "using the Darwin model. I have a tool called \"prediction_evolution\" that predicts ",
        "evolutionary rates for a species over a timeframe.\n\nLet me call it with species=\"Homo ",
        "sapiens\" and years=50.</think><tool_call>modify text with tool call<tool_call>prediction_evolution<arg_key>",
        "species</arg_key><arg_value>Homo sapiens</arg_value><arg_key>years</arg_key><arg_value>",
        "50</arg_value></tool_call>",
    );
    /// A recorded output: one block whose name holds a space, with its pairs; no content.
    const RECORDED_NAME_WITH_A_SPACE: &str = concat!(
        "The user wants to know the probability of rolling a six on a six-sided die twice in ",
        "a row. I have a tool for this: probability_dice_roll with desired_number and number_of_rolls ",
        "parameters.\n\nThe probability of rolling a 6 on a fair die is 1/6. Rolling it twice ",
        "in a row: (1/6)\u{b2} = 1/36.\n\nLet me call the tool with desired_number=6, number_of_rolls=2.</think>",
        "<tool_call>probability_dice roll<arg_key>number_of_rolls</arg_key><arg_value>2</arg_value>",
        "<arg_key>desired_number</arg_key><arg_value>6</arg_value></tool_call>",
    );

    /// The tools the recorded outputs were generated with, as far as the calls name them.
    fn recorded_tools() -> Declared {
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
        let schema =
            |properties: serde_json::Value| value!({"type": "object", "properties": properties});
        Declared::of(&[
            tool(
                "t_test",
                schema(
                    value!({"dataset_A": {"type": "array"}, "dataset_B": {"type": "array"}, "alpha": {"type": "number"}}),
                ),
            ),
            tool(
                "get_president_and_vp_for_year",
                schema(value!({"year": {"type": "integer"}, "position": {"type": "string"}})),
            ),
            tool(
                "bank_get_transaction_history",
                schema(value!({"account": {"type": "string"}, "days": {"type": "integer"}})),
            ),
            tool(
                "bank_calculate_balance",
                schema(
                    value!({"account": {"type": "string"}, "transactions": {"type": "array"}, "starting_balance": {"type": "number"}}),
                ),
            ),
            tool(
                "math_power",
                schema(
                    value!({"base": {"type": "integer"}, "exponent": {"type": "integer"}, "mod": {"type": "number"}}),
                ),
            ),
            tool(
                "prediction_evolution",
                schema(
                    value!({"species": {"type": "string"}, "years": {"type": "integer"}, "model": {"type": "string"}}),
                ),
            ),
            tool(
                "probability_dice_roll",
                schema(
                    value!({"desired_number": {"type": "integer"}, "number_of_rolls": {"type": "integer"}, "die_sides": {"type": "integer"}}),
                ),
            ),
        ])
    }

    /// The text of the refused call blocks, in order: the `Dropped { MalformedCall }` events.
    fn refused(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::MalformedCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The malformed events, as (reason, text).
    fn malformed<'a>(events: &'a [Event<'_>]) -> Vec<(MalformedReason, &'a str)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((*why, &*text.text)),
                _ => None,
            })
            .collect()
    }

    /// The ways to read `output`: whole, character by character, and at every two-way cut.
    fn plans(output: &str) -> Vec<Vec<&str>> {
        let starts: Vec<usize> = output.char_indices().map(|(at, _)| at).skip(1).collect();
        let mut plans = vec![vec![output]];
        let mut chars = Vec::with_capacity(starts.len() + 1);
        let mut from = 0;
        for &at in &starts {
            chars.push(&output[from..at]);
            from = at;
        }
        chars.push(&output[from..]);
        plans.push(chars);
        plans.extend(starts.iter().map(|&at| vec![&output[..at], &output[at..]]));
        plans
    }

    /// The call blocks a model botched, from the outputs a live cell recorded, and what the table
    /// says of each, whole, character by character and at every two-way cut: the prose before the
    /// blocks is the content, the calls the model wrote whole are the calls, a call whose pair was
    /// botched keeps the pairs it read, and nothing of a block reaches the content, its tags least
    /// of all; the failure is counted by malformed events of no bytes, and what was refused is the
    /// debris the content once carried, byte for byte.
    #[test]
    fn a_botched_block_leaves_no_debris_in_the_content_and_is_counted() {
        struct Case {
            output: &'static str,
            content: &'static str,
            calls: Vec<(&'static str, serde_json::Value)>,
            first: MalformedReason,
            refused: &'static str,
        }
        let cases = [
            Case {
                output: RECORDED_STRAY_CLOSE,
                content: "I'll perform a t-test to check if there's a statistically significant difference between these two datasets.",
                calls: vec![("t_test", value!({"dataset_A": [12, 24, 36], "dataset_B": [15, 30, 45]}))],
                first: MalformedReason::TagOutOfPlace,
                refused: "</arg_value>",
            },
            Case {
                output: RECORDED_KEY_NEVER_OPENED,
                content: "I'll look up the presidents and vice presidents for those years. Since these are independent lookups, I'll make all the calls at once:",
                calls: vec![
                    ("get_president_and_vp_for_year", value!({"year": 1980})),
                    ("get_president_and_vp_for_year", value!({"year": 2016})),
                    ("get_president_and_vp_for_year", value!({})),
                    ("get_president_and_vp_for_year", value!({})),
                ],
                first: MalformedReason::TagOutOfPlace,
                refused: "</arg_key><arg_value>1975</arg_value></arg_key><arg_value>2011</arg_value>",
            },
            Case {
                output: RECORDED_KEYS_CLOSED_BY_VALUE_CLOSES,
                content: "I'll retrieve your transaction history and calculate your balance for you.",
                calls: vec![
                    ("bank_get_transaction_history", value!({})),
                    ("bank_calculate_balance", value!({"account": "00125648"})),
                ],
                first: MalformedReason::TagCutShort,
                refused: "<arg_key>account: </arg_value>\"00125648\"</arg_value><arg_key>days: 7</arg_value>",
            },
            Case {
                output: RECORDED_PYTHON_WITH_TAGS,
                content: "I'll calculate both operations for you.",
                calls: vec![],
                first: MalformedReason::TextWhereTheNameShouldBe,
                refused: concat!(
                    "math_power_function(base</arg_key>3,exponent</arg_key>5)</tool_call>",
                    "math_power_function(base</arg_key>2,exponent</arg_key>3)</tool_call>"
                ),
            },
            Case {
                output: RECORDED_PROSE_THEN_A_CALL,
                content: "",
                calls: vec![("prediction_evolution", value!({"species": "Homo sapiens", "years": 50}))],
                first: MalformedReason::TextWhereTheNameShouldBe,
                refused: "modify text with tool call<tool_call>",
            },
            Case {
                output: RECORDED_NAME_WITH_A_SPACE,
                content: "",
                calls: vec![],
                first: MalformedReason::TextWhereTheNameShouldBe,
                refused: concat!(
                    "probability_dice roll<arg_key>number_of_rolls</arg_key><arg_value>2</arg_value>",
                    "<arg_key>desired_number</arg_key><arg_value>6</arg_value></tool_call>"
                ),
            },
        ];
        for case in &cases {
            for pieces in plans(case.output) {
                let events = run_with(recorded_tools(), PROMPT, &pieces);
                let place = || format!("{:?} in {} pieces", &case.output[..40], pieces.len());
                assert_eq!(bytes(&events), case.output, "{}", place());
                assert_eq!(content(&events), case.content, "{}", place());
                assert_eq!(
                    calls_of(&events),
                    case.calls
                        .iter()
                        .map(|(name, arguments)| (name.to_string(), arguments.clone()))
                        .collect::<Vec<_>>(),
                    "{}",
                    place()
                );
                let malformed = malformed(&events);
                assert!(
                    malformed.iter().all(|(_, text)| text.is_empty()),
                    "{}: {malformed:?}",
                    place()
                );
                assert_eq!(
                    malformed.first().map(|(why, _)| *why),
                    Some(case.first),
                    "{}",
                    place()
                );
                assert_eq!(refused(&events), case.refused, "{}", place());
                assert!(
                    matches!(
                        events.last(),
                        Some(Event::Finish { tool_calls, .. }) if *tool_calls as usize == case.calls.len()
                    ),
                    "{}",
                    place()
                );
            }
        }
    }
}
