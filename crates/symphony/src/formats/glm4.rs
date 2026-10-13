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

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(glm4(), declared());
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

    fn dropped(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn a_rendered_turn_gives_its_thought_and_its_call_typed_by_the_tools_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        // The newline before the thought is the model's, so it is content; the markers are
        // wrappers.
        assert_eq!(
            events[..3],
            [
                Event::Content(Text::uncounted("\n")),
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
                Event::Content(Text::uncounted("\n")),
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
        assert_eq!(content(&events), "\n\n");
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
        // The template's newlines around the content and between the calls ride in the content
        // stream (bellwether #17); the words are the content.
        assert_eq!(content(&events), "\n\nTwo drinks.\n\n");
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
}
