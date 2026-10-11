//! MiniMax M2: reasoning between `<think>` and `</think>`, the calls of one turn between
//! `<minimax:tool_call>` and `</minimax:tool_call>`, each call between `<invoke name="…">` and
//! `</invoke>` as `<parameter name="…">` tags whose key is the attribute and whose value the
//! request's tools type, everything else content. The prompt opens the thought
//! (`]~b]ai\n<think>\n`, whatever the request says: the template has no thinking switch), and a
//! turn opens with `]~b]ai`, so the prompt's replay starts there. The template writes the thought
//! as `<think>\n`,
//! the reasoning, `\n</think>\n\n`, then the content, then `\n<minimax:tool_call>\n`, each invoke
//! as `<invoke name="NAME">\n`, each parameter on its own line, `</invoke>\n`, and the block's
//! close; the newlines around the thought and before the block are the model's to write and reach
//! the reasoning and content streams (bellwether #17's question, as for Qwen3), the ones inside the
//! block are its wrapping. Recorded as `minimax-m2` (MiniMaxAI/MiniMax-M2) and `minimax-m2.7`
//! (MiniMaxAI/MiniMax-M2.7), whose templates write the same turn:
//!
//! ```text
//! The user asks about Paris.
//! </think>
//!
//!
//! <minimax:tool_call>
//! <invoke name="get_weather">
//! <parameter name="city">Paris</parameter>
//! </invoke>
//! </minimax:tool_call>
//! ```
//!
//! MiniMax M3 writes its arguments as an XML tree under its own markers ([`minimax_m3()`]).
//!
//! [`minimax_m3()`]: super::minimax_m3()

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed,
};

/// The MiniMax M2 table.
pub fn minimax_m2() -> Format {
    Format::new("minimax_m2")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<minimax:tool_call>")
        .terminal("calls_close", "</minimax:tool_call>")
        .terminal("invoke_open", "<invoke name=\"")
        .terminal("invoke_close", attributed::INVOKE_CLOSE)
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("invoke", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("calls", "invoke_open", "invoke")
        .transition("invoke", "invoke_close", "calls")
        .transition("invoke", "invoke_open", "invoke")
        .transition("invoke", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Attributed(attributed::Tags::MINIMAX_M2))
        .opens_turn("]~b]ai")
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
                    "metric": {"type": "boolean"},
                    "options": {"type": "object"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The generation prompt's tail: the turn opened, and its thought on its own line.
    const PROMPT: &str = "]~b]ai\n<think>\n";

    /// An output as the template renders a turn: the thought closed, no content, one invoke with a
    /// string in a block, each tag on its own line.
    const OUTPUT: &str = concat!(
        "The user asks about Paris.\n</think>\n\n\n<minimax:tool_call>\n",
        "<invoke name=\"get_weather\">\n<parameter name=\"city\">Paris</parameter>\n",
        "</invoke>\n</minimax:tool_call>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(minimax_m2(), declared());
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
    fn a_rendered_turn_gives_its_thought_and_its_call_typed_by_the_tool_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        // The prompt opened the thought; the template's newline before `</think>` is the model's,
        // so it is reasoning text; the markers are wrappers.
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("The user asks about Paris.\n")),
                dropped("</think>"),
                Event::ReasoningEnd,
            ]
        );
        assert_eq!(reasoning(&events), "The user asks about Paris.\n");
        // The newlines between the thought and the block are content; the block's own are
        // wrapping.
        assert_eq!(content(&events), "\n\n\n");
        assert_eq!(
            events[5..8],
            [
                dropped("<minimax:tool_call>"),
                dropped("\n"),
                dropped("<invoke name=\""),
            ]
        );
        assert!(matches!(
            &events[8],
            Event::ToolCallStart { index: 0, name, .. } if name == "get_weather"
        ));
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn content_sits_between_the_thought_and_two_invokes_in_one_block() {
        let output = concat!(
            "Two cities.\n</think>\n\nHere are both.\n<minimax:tool_call>\n",
            "<invoke name=\"get_weather\">\n<parameter name=\"city\">Paris</parameter>\n",
            "</invoke>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Rome</parameter>\n",
            "</invoke>\n</minimax:tool_call>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "Two cities.\n");
        assert_eq!(content(&events), "\n\nHere are both.\n");
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"city": "Rome"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_boolean_and_an_object_are_typed_by_the_tool_and_a_string_streams() {
        let output = concat!(
            "\n</think>\n\n\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n",
            "<parameter name=\"city\">Paris</parameter>\n<parameter name=\"days\">3</parameter>\n",
            "<parameter name=\"metric\">true</parameter>\n",
            "<parameter name=\"options\">{\"a\": 1}</parameter>\n</invoke>\n</minimax:tool_call>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"city": "Paris", "days": 3, "metric": true, "options": {"a": 1}}"#
        );
        // The string's bytes come as they arrive: `Par` then `is`.
        let pieces = [
            "\n</think>\n\n\n<minimax:tool_call>\n<invoke name=\"get_weather\">\n",
            "<parameter name=\"city\">Par",
            "is</parameter>\n</invoke>\n</minimax:tool_call>",
        ];
        let streamed = run(PROMPT, &pieces);
        assert_eq!(bytes(&streamed), pieces.concat());
        let fragments: Vec<&str> = streamed
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(fragments, [r#"{"city": ""#, "Par", "is", "\"", "}"]);
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_closes_nothing_and_still_ends_the_call() {
        let pieces = [
            "\n</think>\n\n\n<minimax:tool_call>\n<invoke name=\"get_weat",
            "her\">\n<parameter name=\"city\">Par",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Par"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn the_table_validates() {
        minimax_m2().validate().expect("a sound table");
    }
}
