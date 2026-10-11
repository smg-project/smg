//! K2-Horizon: reasoning between `<ifm|think>` and `</ifm|think>`, the calls of one turn between
//! `<ifm|tool_calls>` and `</ifm|tool_calls>`, each call between `<ifm|tool_call>` and
//! `</ifm|tool_call>` as its name and keyed arguments (`<ifm|arg_key>`, `<ifm|arg_value>`), typed
//! by the request's tools, with the template's newline after the name, after each `</ifm|arg_key>`
//! and after each `</ifm|arg_value>`, as GLM 4.5 and 4.6 write theirs, and the block's own before
//! each call and before its close, which are its wrapping. Everything else is content. The prompt
//! opens the thought (`<|ifm|im_start|>assistant\n<ifm|think>\n`, whatever the request says: the
//! template has no thinking switch; it raises on an assistant turn without a thinking field), and
//! a turn opens with `<|ifm|im_start|>assistant`, so the prompt's replay starts there. Recorded as
//! `k2-horizon-36b` (IFM/K2-Horizon-36B).
//!
//! What the table leaves out, for want of fixtures: the template picks the thought's tag by
//! `reasoning_effort` (`<ifm|think_fast>` and `<ifm|think_faster>` below the default), and it can
//! write the calls as JSON objects or with a typed tag per argument (`tool_call_format`); the
//! recorded sets hold the default thought and the default calls, read here. A thought under one of
//! the other tags is content until a set shows it.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The K2-Horizon table.
pub fn k2_horizon() -> Format {
    Format::new("k2_horizon")
        .terminal("think_open", "<ifm|think>")
        .terminal("think_close", "</ifm|think>")
        .terminal("calls_open", "<ifm|tool_calls>")
        .terminal("calls_close", "</ifm|tool_calls>")
        .terminal("call_open", "<ifm|tool_call>")
        .terminal("call_close", "</ifm|tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("calls", "call_open", "call")
        .transition("call", "call_close", "calls")
        .transition("call", "call_open", "call")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Keyed(keyed::Tags::K2_HORIZON))
        .opens_turn("<|ifm|im_start|>assistant")
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
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The generation prompt's tail: the turn opened, and its thought.
    const PROMPT: &str = "<|ifm|im_start|>assistant\n<ifm|think>\n";

    /// An output as the template renders a turn: the thought closed, then one call with a string
    /// and an integer in a block, each tag on its own line.
    const OUTPUT: &str = concat!(
        "The user asks about Paris.</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_weather\n",
        "<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Paris</ifm|arg_value>\n",
        "<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_value>3</ifm|arg_value>\n",
        "</ifm|tool_call>\n</ifm|tool_calls>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(k2_horizon(), declared());
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
        // The prompt opened the thought, so the output starts inside it.
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("The user asks about Paris.")),
                dropped("</ifm|think>"),
                Event::ReasoningEnd,
            ]
        );
        assert_eq!(reasoning(&events), "The user asks about Paris.");
        // The block's markers and its newlines are wrapping, then the call's opener.
        assert_eq!(events[4..6], [dropped("<ifm|tool_calls>"), dropped("\n")]);
        assert_eq!(events[6], dropped("<ifm|tool_call>"));
        assert!(matches!(
            &events[7],
            Event::ToolCallStart { index: 0, name, .. } if name == "get_weather"
        ));
        // The newlines between the tags are the template's, taken as wrappers, not as values; the
        // integer is typed by the tool.
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris", "days": 3}"#);
        assert_eq!(content(&events), "");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn content_sits_between_the_thought_and_two_calls_in_one_block() {
        let output = concat!(
            "Two cities.</ifm|think>Here are both.<ifm|tool_calls>\n",
            "<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n",
            "<ifm|arg_value>Paris</ifm|arg_value>\n</ifm|tool_call>\n",
            "<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n",
            "<ifm|arg_value>Rome</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "Two cities.");
        assert_eq!(content(&events), "Here are both.");
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"city": "Rome"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_call_without_arguments_is_its_name_and_a_newline() {
        // As the recorded zero-argument calls read: `<ifm|tool_call>NAME\n</ifm|tool_call>`.
        let output = concat!(
            "</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_weather\n</ifm|tool_call>\n",
            "</ifm|tool_calls>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        // An empty thought, the block's opener, its newline, the call's opener, then the call.
        assert_eq!(
            events[..6],
            [
                Event::ReasoningStart,
                dropped("</ifm|think>"),
                Event::ReasoningEnd,
                dropped("<ifm|tool_calls>"),
                dropped("\n"),
                dropped("<ifm|tool_call>"),
            ]
        );
        assert!(matches!(
            &events[6],
            Event::ToolCallStart { index: 0, name, .. } if name == "get_weather"
        ));
        assert_eq!(arguments_of(&events, 0), "{}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_call_cut_between_its_tags_streams_the_string_and_closes_the_object_at_the_end() {
        let pieces = [
            "</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_",
            "weather\n<ifm|arg_key>ci",
            "ty</ifm|arg_key>\n<ifm|arg_value>Par",
            "is</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn the_table_validates() {
        k2_horizon().validate().expect("a sound table");
    }
}
