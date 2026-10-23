//! GLM 4.5 and 4.6: reasoning between `<think>` and `</think>`, each call between `<tool_call>`
//! and `</tool_call>` as its name and keyed arguments (`<arg_key>`, `<arg_value>`), typed by the
//! request's tools, with the template's newline after the name, after each `</arg_key>` and after
//! each `</arg_value>`. Everything else is content. A turn opens with `<|assistant|>`, so the
//! prompt's replay starts there, and the model writes its own `<think>`: the template puts
//! `\n<think></think>` into the prompt only when the request turns thinking off. Recorded as
//! `glm-4.6` (zai-org/GLM-4.6). GLM 4.7 and later write the tags with nothing between them and
//! open the thought in the prompt ([`glm()`](super::glm())); Ling writes the newline before the
//! close instead of after each pair, under its own turn opener ([`ling()`](super::ling())).
//!
//! The table carries the one part of its chat template's layout the tags write: after the
//! thought's close the template writes a newline before the content or the first call
//! (`</think>\n<tool_call>`), so a forced call written where the close left the engine begins on
//! its own line, as the template writes it, and the call's marker is not the grammar's first text
//! ([`Format::grammar_for`]). The thought the model opens and closes itself: a `<tool_call>`
//! inside it is thought text, as the template reads it.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The GLM 4.5 and 4.6 table.
pub fn glm4() -> Format {
    Format::new("glm4")
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
        .calls(CallSyntax::Keyed(keyed::Tags::GLM4))
        .opens_turn("<|assistant|>")
        .with_layout(super::thought_layout("</think>", "\n"))
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, FinishReason, MalformedReason, Text},
        testing::{
            a_forced_tag_begins_with_the_templates_text, arguments_of, bytes, content, drive,
            drive_to, reasoning, said, the_same_at_every_cut, Said,
        },
        Declared, EngineFinish,
    };

    /// The recorded tool and one without parameters, as a request lists them.
    fn tools() -> Vec<Tool> {
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
        vec![
            tool(
                "ChaDri_change_drink",
                value!({"type": "object", "properties": {
                    "drink_id": {"type": "string"},
                    "new_preferences": {"type": "object"},
                }}),
            ),
            tool("ping", value!({"type": "object", "properties": {}})),
        ]
    }

    fn declared() -> Declared {
        Declared::of(&tools()[..1])
    }

    /// The generation prompt's tail when the request leaves thinking on: the turn opened, and
    /// nothing more; the model writes the thought.
    const PROMPT: &str = "<|assistant|>";

    /// The generation prompt's tail when the request turns thinking off: the template closes an
    /// empty thought on its own line.
    const PROMPT_THINKING_OFF: &str = "<|assistant|>\n<think></think>";

    /// An output as the template renders a turn: the thought, then one call with a string and an
    /// object, each tag on its own line.
    const OUTPUT: &str = concat!(
        "\n<think>The user wants a latte, large and hot.</think>\n<tool_call>ChaDri_change_drink\n",
        "<arg_key>drink_id</arg_key>\n<arg_value>latte</arg_value>\n",
        "<arg_key>new_preferences</arg_key>\n",
        "<arg_value>{\"size\": \"large\", \"temperature\": \"hot\"}</arg_value>\n</tool_call>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(&mut Engine::new(glm4(), declared()), prompt, pieces)
    }

    fn dropped(text: &str) -> Event<'static> {
        Event::Dropped {
            text: Text::uncounted(text.to_owned()),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn a_rendered_turn_gives_its_thought_and_its_call_typed_by_the_tools_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        // The newline before the thought is the template's spacing, held until the thought's
        // open showed no text followed it, and dropped; the markers are wrappers.
        assert_eq!(
            events[..3],
            [
                Event::Dropped {
                    text: Text::uncounted("\n"),
                    why: DropReason::Whitespace,
                },
                dropped("<think>"),
                Event::ReasoningStart,
            ]
        );
        assert_eq!(
            events[3..8],
            [
                Event::Reasoning(Text::uncounted("The user wants a latte, large and hot.")),
                dropped("</think>"),
                Event::ReasoningEnd,
                // The newline between the thought and the call is the template's spacing,
                // held until the call's opener came and dropped.
                Event::Dropped {
                    text: Text::uncounted("\n"),
                    why: DropReason::Whitespace,
                },
                dropped("<tool_call>"),
            ]
        );
        assert!(matches!(
            &events[8],
            Event::ToolCallStart { index: 0, name, .. } if name == "ChaDri_change_drink"
        ));
        // The newlines between the tags are the template's, taken as wrappers, not as values.
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
    fn thinking_off_starts_in_content_after_the_thought_the_prompt_closed() {
        let events = run(PROMPT_THINKING_OFF, &["\nHello!"]);
        assert_eq!(
            events[..1],
            [Event::Content(Text::uncounted("\nHello!"))],
            "the prompt's closed thought is read, not reported"
        );
        assert_eq!(reasoning(&events), "");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn content_sits_between_the_thought_and_two_calls_each_on_its_own_lines() {
        let output = concat!(
            "\n<think></think>\nTwo drinks.\n<tool_call>ChaDri_change_drink\n",
            "<arg_key>drink_id</arg_key>\n<arg_value>123</arg_value>\n</tool_call>\n",
            "<tool_call>ChaDri_change_drink\n",
            "<arg_key>drink_id</arg_key>\n<arg_value>tea</arg_value>\n</tool_call>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        // The template's newlines around the words ride in the content stream (bellwether #17);
        // the newline before the thought, with no text after it in its stay, and the one between
        // the calls are the template's spacing, dropped.
        assert_eq!(content(&events), "\nTwo drinks.\n");
        // A declared string keeps the digits as text.
        assert_eq!(arguments_of(&events, 0), r#"{"drink_id": "123"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"drink_id": "tea"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_call_cut_between_its_tags_streams_the_string_and_closes_the_object_at_the_end() {
        let pieces = [
            "\n<think></think>\n<tool_call>ChaDri_",
            "change_drink\n<arg_key>drink_",
            "id</arg_key>\n<arg_value>lat",
            "te</arg_value>\n</tool_call>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(arguments_of(&events, 0), r#"{"drink_id": "latte"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn the_table_validates() {
        glm4().validate().expect("a sound table");
    }

    /// The tool of a recorded case: a question answered through one string parameter.
    fn answer_tool() -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "answer_string".to_string(),
                description: None,
                parameters: value!({"type": "object", "required": ["answer"], "properties": {
                    "answer": {"type": "string"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }]
    }

    /// A recorded shape: the model answers through the string parameter and ends its output
    /// with its own end-of-box marker, a newline and the end of the stream, writing neither
    /// `</arg_value>` nor `</tool_call>`.
    const OPEN_AT_THE_STOP: &str = concat!(
        "\n<think></think>\n<tool_call>answer_string\n<arg_key>answer</arg_key>\n",
        "<arg_value>Logistic regression is a statistical method for a binary outcome.",
        "<|end_of_box|>\n"
    );

    /// The value of that shape, whole: every byte the model wrote between the tag and its end.
    const ANSWER: &str =
        "Logistic regression is a statistical method for a binary outcome.<|end_of_box|>\n";

    /// The malformed events as (text, reason), in order.
    fn malformed(events: &[Event]) -> Vec<(String, MalformedReason)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((text.text.to_string(), *why)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_string_value_the_model_leaves_open_at_its_own_end_is_closed_there_and_said() {
        // Whole, one character at a time and at every two-way cut: the call carries the value
        // with every byte of the model's text, its arguments one JSON object under `stop` with
        // one call, which a client reading `tool_calls` parses; the one `Malformed` event of no
        // bytes says the stop closed the string, right before the closing quote.
        let run = |pieces: &[&str]| {
            drive(
                &mut Engine::new(glm4(), Declared::of(&answer_tool())),
                PROMPT,
                pieces,
            )
        };
        let arguments = format!(
            "{{\"answer\": {}}}",
            serde_json::to_string(ANSWER).expect("json")
        );
        assert_eq!(
            the_same_at_every_cut(run, OPEN_AT_THE_STOP),
            Said {
                calls: vec![("answer_string".to_string(), arguments.clone())],
                finish: Some((FinishReason::Stop, 1)),
                content: String::new(),
                reasoning: String::new(),
            }
        );
        let events = run(&[OPEN_AT_THE_STOP]);
        let read: serde_json::Value =
            serde_json::from_str(&arguments_of(&events, 0)).expect("whole JSON");
        assert_eq!(read["answer"], ANSWER);
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::OpenAtTheStop)]
        );
        let at = events
            .iter()
            .position(|event| matches!(event, Event::Malformed { .. }))
            .expect("the one word");
        assert!(
            matches!(&events[at + 1], Event::ToolCallArguments { json, .. } if json == "\""),
            "{events:?}"
        );
        assert_eq!(bytes(&events), OPEN_AT_THE_STOP);
    }

    /// The same shape as the live stream carries it: the model ends the value with its end-of-box
    /// marker and a newline, skips `</arg_value>` alone, writes `</tool_call>`, and then its
    /// turn's stop token, which the engine hides and reports as a stop. The closing marker, not
    /// the stop, ends the call with the string open.
    const OPEN_AT_THE_MARKER: &str = concat!(
        "\n<think></think>\n<tool_call>answer_string\n<arg_key>answer</arg_key>\n",
        "<arg_value>Logistic regression is a statistical method for a binary outcome.",
        "<|end_of_box|>\n</tool_call>"
    );

    #[test]
    fn a_string_value_the_model_leaves_open_at_the_calls_closing_marker_is_closed_there_and_said() {
        // Whole, one character at a time, at every two-way cut, and one piece per token at the
        // end as the stream carries it (the text, the box marker, the newline, the marker): the
        // call carries the value with every byte of the model's text, its arguments one JSON
        // object under `stop` with one call, which a client reading `tool_calls` parses; the one
        // `Malformed` event of no bytes says the marker closed the string, right before the
        // closing quote; the marker's bytes are the call's end.
        let run = |pieces: &[&str]| {
            drive(
                &mut Engine::new(glm4(), Declared::of(&answer_tool())),
                PROMPT,
                pieces,
            )
        };
        let arguments = format!(
            "{{\"answer\": {}}}",
            serde_json::to_string(ANSWER).expect("json")
        );
        let expected = Said {
            calls: vec![("answer_string".to_string(), arguments.clone())],
            finish: Some((FinishReason::Stop, 1)),
            content: String::new(),
            reasoning: String::new(),
        };
        assert_eq!(the_same_at_every_cut(run, OPEN_AT_THE_MARKER), expected);
        let as_streamed = [
            "\n<think></think>\n<tool_call>answer_string\n<arg_key>answer</arg_key>\n<arg_value>",
            "Logistic regression is a statistical method for a binary outcome",
            ".",
            "<|end_of_box|>",
            "\n",
            "</tool_call>",
        ];
        let events = run(&as_streamed);
        assert_eq!(said(&events), expected);
        let read: serde_json::Value =
            serde_json::from_str(&arguments_of(&events, 0)).expect("whole JSON");
        assert_eq!(read["answer"], ANSWER);
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::OpenAtTheMarker)]
        );
        let at = events
            .iter()
            .position(|event| matches!(event, Event::Malformed { .. }))
            .expect("the one word");
        assert!(
            matches!(&events[at + 1], Event::ToolCallArguments { json, .. } if json == "\""),
            "{events:?}"
        );
        assert!(
            events.iter().any(|event| matches!(
                event,
                Event::ToolCallEnd { source, .. } if source.text == "</tool_call>"
            )),
            "{events:?}"
        );
        assert_eq!(bytes(&events), as_streamed.concat());
        // Cut by the token limit after the marker: the marker closed the call before the cut, so
        // the arguments are the same whole object, and the engine's finish is kept.
        let cut = drive_to(
            &mut Engine::new(glm4(), Declared::of(&answer_tool())),
            PROMPT,
            &[OPEN_AT_THE_MARKER],
            EngineFinish::Length,
        );
        assert_eq!(arguments_of(&cut, 0), arguments);
        assert_eq!(
            malformed(&cut),
            [(String::new(), MalformedReason::OpenAtTheMarker)]
        );
        assert_eq!(said(&cut).finish, Some((FinishReason::Length, 1)));
    }

    #[test]
    fn the_same_value_cut_by_the_token_limit_stays_open_and_says_so() {
        // The twin under `length`: the prefix as written, no closing quote and no brace, the
        // one `Malformed { UnterminatedRegion }` event; whole, one character at a time and at
        // every two-way cut.
        let run = |pieces: &[&str]| {
            drive_to(
                &mut Engine::new(glm4(), Declared::of(&answer_tool())),
                PROMPT,
                pieces,
                EngineFinish::Length,
            )
        };
        let said = the_same_at_every_cut(run, OPEN_AT_THE_STOP);
        let mut prefix = format!(
            "{{\"answer\": {}}}",
            serde_json::to_string(ANSWER).expect("json")
        );
        prefix.truncate(prefix.len() - 2);
        assert_eq!(said.calls, vec![("answer_string".to_string(), prefix)]);
        assert_eq!(said.finish, Some((FinishReason::Length, 1)));
        assert!(serde_json::from_str::<serde_json::Value>(&said.calls[0].1).is_err());
        assert_eq!(
            malformed(&run(&[OPEN_AT_THE_STOP])),
            [(String::new(), MalformedReason::UnterminatedRegion)]
        );
    }

    #[test]
    fn a_call_the_token_limit_cut_inside_an_object_keeps_it_in_its_arguments() {
        // The report's shape under `length`: the string whole, the object cut inside a nested
        // string. The arguments are the prefix as written, the cut member's key and bytes
        // included (before this the member left the call as a failed call's bytes and the
        // arguments stopped at the string), no closing quote and no brace, the one `Malformed {
        // UnterminatedRegion }` event of no bytes, nothing dropped as a failed call; whole, one
        // character at a time and at every two-way cut.
        let output = concat!(
            "\n<think></think>\n<tool_call>ChaDri_change_drink\n",
            "<arg_key>drink_id</arg_key>\n<arg_value>latte</arg_value>\n",
            "<arg_key>new_preferences</arg_key>\n<arg_value>{\"size\": \"large\", \"temp"
        );
        let run = |pieces: &[&str]| {
            drive_to(
                &mut Engine::new(glm4(), declared()),
                PROMPT,
                pieces,
                EngineFinish::Length,
            )
        };
        let said = the_same_at_every_cut(run, output);
        assert_eq!(
            said.calls,
            vec![(
                "ChaDri_change_drink".to_string(),
                r#"{"drink_id": "latte", "new_preferences": {"size": "large", "temp"#.to_string()
            )]
        );
        assert_eq!(said.finish, Some((FinishReason::Length, 1)));
        assert_eq!(said.content, "");
        assert!(serde_json::from_str::<serde_json::Value>(&said.calls[0].1).is_err());
        let events = run(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::UnterminatedRegion)]
        );
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
    fn a_call_the_model_leaves_open_between_its_pairs_at_its_own_end_is_closed_there_and_said() {
        // The sibling shape: the second call's last value closed, then the end of the output
        // with no `</tool_call>`. Both calls read whole, the second's object closed at the stop,
        // and the one word says so.
        let output = concat!(
            "\n<think></think>\n<tool_call>ChaDri_change_drink\n",
            "<arg_key>drink_id</arg_key>\n<arg_value>latte</arg_value>\n</tool_call>\n",
            "<tool_call>ChaDri_change_drink\n",
            "<arg_key>drink_id</arg_key>\n<arg_value>tea</arg_value>\n"
        );
        assert_eq!(
            the_same_at_every_cut(|pieces| run(PROMPT, pieces), output),
            Said {
                calls: vec![
                    (
                        "ChaDri_change_drink".to_string(),
                        r#"{"drink_id": "latte"}"#.to_string()
                    ),
                    (
                        "ChaDri_change_drink".to_string(),
                        r#"{"drink_id": "tea"}"#.to_string()
                    ),
                ],
                finish: Some((FinishReason::Stop, 2)),
                content: String::new(),
                reasoning: String::new(),
            }
        );
        assert_eq!(
            malformed(&run(PROMPT, &[output])),
            [(String::new(), MalformedReason::OpenAtTheStop)]
        );
        // Cut by the token limit, the second call stays open, as before.
        let cut = drive_to(
            &mut Engine::new(glm4(), declared()),
            PROMPT,
            &[output],
            EngineFinish::Length,
        );
        assert_eq!(arguments_of(&cut, 1), r#"{"drink_id": "tea""#);
        assert_eq!(
            malformed(&cut),
            [(String::new(), MalformedReason::UnterminatedRegion)]
        );
    }

    /// One call as the tag writes it after the thought's close, and as the template renders one.
    const FORCED_CALL: &str = "<tool_call>ChaDri_change_drink\n<arg_key>drink_id</arg_key>\n\
                               <arg_value>latte</arg_value>\n</tool_call>";

    #[test]
    fn a_prompt_with_thinking_off_ends_with_the_bare_close_and_the_newline_stays_the_tags() {
        // The thinking-off prompt closes an empty thought without the newline the template writes
        // after the close before the content or a call, so the prompt has not written it: a
        // forced call written after the gate of the engine's reasoning parser still begins with
        // it, as with the thought left to the model.
        use crate::grammar::{GrammarStart, Options};
        let format = glm4();
        assert!(!format.thought_closed(PROMPT_THINKING_OFF));
        assert!(!format.thought_closed(PROMPT));
        assert!(!format.reasoning_open(PROMPT_THINKING_OFF));
        let tag = format
            .grammar_for(
                &tools(),
                Options {
                    at_least_one: true,
                    reasoning_open: format.reasoning_open(PROMPT_THINKING_OFF),
                    thought_closed: format.thought_closed(PROMPT_THINKING_OFF),
                    one_call: false,
                    start: GrammarStart::AfterReasoning,
                },
            )
            .expect("a tag")
            .payload();
        assert_eq!(tag["format"]["type"], "sequence");
        assert_eq!(tag["format"]["elements"][0]["type"], "const_string");
        assert_eq!(tag["format"]["elements"][0]["value"], "\n");
    }

    #[test]
    fn a_forced_call_after_the_thoughts_close_begins_with_the_templates_newline() {
        // The GLM 4.5 and 4.6 templates write a newline after the thought's close before the
        // content or the first call, and the table carries that part of its layout: where the
        // grammar starts after the close, the calls' tag begins with it, so the call's marker is
        // not the grammar's first text; with the thought closed, at the first token or where the
        // engine does not say, the calls at once, as before.
        a_forced_tag_begins_with_the_templates_text(
            &glm4(),
            &tools(),
            "</think>",
            "\n",
            "<tool_call>",
        );
    }

    #[test]
    fn a_forced_call_written_after_the_thoughts_close_reads_as_the_call_with_the_rehearsal_as_thought(
    ) {
        // The tag's shape on an engine whose grammar starts after the thought's close: the thought
        // with the call's format rehearsed inside it, its close, the template's newline and the
        // call. The rehearsal is thought text, the newline the template's spacing, the call the
        // call; whole, one character at a time and at every cut.
        let rehearsed = format!("The user wants a latte. The format is {FORCED_CALL}. So:");
        let output = format!("<think>{rehearsed}</think>\n{FORCED_CALL}");
        assert_eq!(
            the_same_at_every_cut(|pieces| run(PROMPT, pieces), &output),
            Said {
                calls: vec![(
                    "ChaDri_change_drink".to_string(),
                    r#"{"drink_id": "latte"}"#.to_string()
                )],
                finish: Some((FinishReason::Stop, 1)),
                content: String::new(),
                reasoning: rehearsed,
            }
        );
    }

    #[test]
    fn a_call_marker_inside_the_thought_stays_thought_text() {
        // An engine whose reasoning gate took the rehearsed marker for the thought's end and
        // started a forced call there wrote the call inside the thought, the output ending there;
        // the table reads it as the template does: thought text, no call, `stop`. The tag no
        // longer lets such an engine start there (its first text after the close is the
        // template's newline, not the marker); the reading is pinned here, whole, one character
        // at a time and at every cut.
        let thought = format!("The user wants a latte. The format is {FORCED_CALL}");
        let output = format!("<think>{thought}");
        assert_eq!(
            the_same_at_every_cut(|pieces| run(PROMPT, pieces), &output),
            Said {
                calls: Vec::new(),
                finish: Some((FinishReason::Stop, 0)),
                content: String::new(),
                reasoning: thought,
            }
        );
    }
}
