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
        event::{DropReason, Event, FinishReason, Text},
        testing::{
            a_forced_tag_begins_with_the_templates_text, arguments_of, bytes, content, drive,
            reasoning, the_same_at_every_cut, Said,
        },
        Declared,
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
