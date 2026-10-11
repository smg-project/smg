//! dots3: reasoning between `<think>` and `</think>`, each call in its own `<dots_function_call>`
//! block as `<invoke name="…">` with `<parameter name="…">` tags whose key is the attribute and
//! whose value the request's tools type, everything else content: MiniMax M2's attributed syntax
//! with a block per call and each value on a line of its own. The template writes the turn as
//! `<think>\n`, the reasoning, `\n</think>\n\n`, the content, then for each call
//! `\n<dots_function_call>\n<invoke name="NAME">`, for each argument `\n<parameter name="KEY">\n`,
//! the value, `\n</parameter>`, then `\n</invoke>\n</dots_function_call>`; a string value as it is,
//! every other value as JSON (`tojson`). The newlines the template writes after a parameter tag
//! and before its closing tag frame the value and are not part of it; the ones around the thought
//! and before a block are the model's to write and reach the reasoning and content streams
//! (bellwether #17's question, as for Qwen3). The thought is the model's to write: the generation
//! prompt is `<|assistant|>` alone, or with the thought closed empty (`<think>\n\n</think>\n\n`)
//! when the request turns thinking off, as Qwen3's does; a turn opens with `<|assistant|>`, so the
//! prompt's replay starts there. Recorded as `dots3-note-prev` (dots-studio/dots3-note-prev),
//! 54,195 parse cases, 15,446 of them with calls:
//!
//! ```text
//! <think>
//! The user wants two songs.
//! </think>
//!
//!
//! <dots_function_call>
//! <invoke name="spotify_play">
//! <parameter name="artist">
//! Taylor Swift
//! </parameter>
//! <parameter name="duration">
//! 20
//! </parameter>
//! </invoke>
//! </dots_function_call>
//! <dots_function_call>
//! <invoke name="spotify_play">
//! <parameter name="artist">
//! Maroon 5
//! </parameter>
//! </invoke>
//! </dots_function_call>
//! ```

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed,
};

/// The dots3 table.
pub fn dots3() -> Format {
    Format::new("dots3")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("call_open", "<dots_function_call>")
        .terminal("call_close", "</dots_function_call>")
        .terminal("invoke_open", "<invoke name=\"")
        .terminal("invoke_close", attributed::Tags::DOTS3.invoke_close)
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Wrapper)
        .state("invoke", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("reasoning", "call_open", "call")
        .transition("call", "invoke_open", "invoke")
        .transition("invoke", "invoke_close", "call")
        .transition("invoke", "call_close", "content")
        .transition("call", "call_close", "content")
        .calls(CallSyntax::Attributed(attributed::Tags::DOTS3))
        .opens_turn("<|assistant|>")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "spotify_play".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "artist": {"type": "string"},
                    "duration": {"type": "integer"},
                    "shuffle": {"type": "boolean"},
                    "tags": {"type": "array"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The generation prompt's tail when the request leaves thinking to the model.
    const PROMPT: &str = "<|assistant|>";

    /// An output as the template renders a turn: the thought, content, then two calls each in its
    /// own block, every value on a line of its own.
    const OUTPUT: &str = concat!(
        "<think>\nThe user wants two songs.\n</think>\n\nTwo songs, then.\n",
        "<dots_function_call>\n<invoke name=\"spotify_play\">\n",
        "<parameter name=\"artist\">\nTaylor Swift\n</parameter>\n",
        "<parameter name=\"duration\">\n20\n</parameter>\n</invoke>\n</dots_function_call>\n",
        "<dots_function_call>\n<invoke name=\"spotify_play\">\n",
        "<parameter name=\"artist\">\nMaroon 5\n</parameter>\n</invoke>\n</dots_function_call>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(dots3(), declared());
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
                Event::Content(Text { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(Text { text, .. }) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_rendered_turn_gives_its_thought_its_content_and_two_calls_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(reasoning(&events), "\nThe user wants two songs.\n");
        // The newlines around the thought and the ones between the blocks are the model's.
        assert_eq!(content(&events), "\n\nTwo songs, then.\n\n");
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Taylor Swift", "duration": 20}"#
        );
        assert_eq!(arguments_of(&events, 1), r#"{"artist": "Maroon 5"}"#);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 1, source } if source.text == "</invoke>"
        )));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_boolean_and_a_list_are_the_json_the_template_writes() {
        let call = concat!(
            "<dots_function_call>\n<invoke name=\"spotify_play\">\n",
            "<parameter name=\"shuffle\">\ntrue\n</parameter>\n",
            "<parameter name=\"tags\">\n[\"pop\", \"live\"]\n</parameter>\n",
            "</invoke>\n</dots_function_call>"
        );
        let events = run(PROMPT, &[call]);
        assert_eq!(bytes(&events), call);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"shuffle": true, "tags": ["pop", "live"]}"#
        );
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_string_streams_as_it_arrives_and_its_framing_newlines_are_not_its_text() {
        let pieces = [
            "<think>\n\n</think>\n\n<dots_function_call>\n<invoke name=\"spotify_play\">\n",
            "<parameter name=\"artist\">\n",
            "Tay",
            "lor Swift\n",
            "</parameter>\n</invoke>\n</dots_function_call>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(arguments_of(&events, 0), r#"{"artist": "Taylor Swift"}"#);
        // The opening, each piece of the string, the closing quote: the newline before the
        // closing tag is held until the tag says it was the template's.
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            fragments,
            ["{\"artist\": \"", "Tay", "lor Swift", "\"", "}"]
        );
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_newline_inside_a_string_stays_and_an_empty_string_is_empty() {
        let call = concat!(
            "<dots_function_call>\n<invoke name=\"spotify_play\">\n",
            "<parameter name=\"artist\">\nTwo\nlines\n</parameter>\n",
            "<parameter name=\"tags\">\n[]\n</parameter>\n</invoke>\n</dots_function_call>"
        );
        let events = run(PROMPT, &[call]);
        assert_eq!(bytes(&events), call);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Two\nlines", "tags": []}"#
        );
        let empty = concat!(
            "<dots_function_call>\n<invoke name=\"spotify_play\">\n",
            "<parameter name=\"artist\">\n\n</parameter>\n</invoke>\n</dots_function_call>"
        );
        let events = run(PROMPT, &[empty]);
        assert_eq!(arguments_of(&events, 0), r#"{"artist": ""}"#);
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_closes_nothing_and_still_ends_the_call() {
        let mut parser = Engine::new(dots3(), declared());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: PROMPT,
                },
                &mut out,
            )
            .expect("prompt");
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: concat!(
                        "<dots_function_call>\n<invoke name=\"spotify_play\">\n",
                        "<parameter name=\"artist\">\nTay"
                    ),
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Length,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        assert_eq!(arguments_of(&events, 0), "{\"artist\": \"Tay");
        assert!(events.iter().any(
            |event| matches!(event, Event::ToolCallEnd { source, .. } if source.text.is_empty())
        ));
    }

    #[test]
    fn the_table_validates() {
        dots3().validate().expect("a valid table");
    }
}
