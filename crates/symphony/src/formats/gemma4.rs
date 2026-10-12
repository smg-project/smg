//! Gemma 4: reasoning between `<|channel>thought\n` and `<channel|>`, each call between
//! `<|tool_call>` and `<tool_call|>` in Gemma's own spelling (`call:NAME{key:value,…}`, a string
//! between `<|"|>` tokens, [`gemma`](crate::gemma)), everything else content. The template writes
//! the thought as `<|channel>thought\n`, the reasoning, `\n<channel|>`, then a turn's calls one
//! right after another with nothing between them, then the content; the newline before
//! `<channel|>` is the model's to write and reaches the reasoning stream (bellwether #17's
//! question, as for Qwen3). The thought is the model's to write: the generation prompt is
//! `<|turn>model\n` alone whatever the request says (the template opens the thought channel only
//! after a tool response, and `enable_thinking` is off unless the request sets it), so the output
//! starts in content. A turn opens with `<|turn>model`, so the prompt's replay starts there.
//! Recorded as `gemma-4-e4b-it` (google/gemma-4-E4B-it), 54,416 parse cases, 15,669 of them with
//! calls:
//!
//! ```text
//! <|channel>thought
//! The user wants two songs.
//! <channel|><|tool_call>call:play{artist:<|"|>Taylor Swift<|"|>,n:20}<tool_call|>
//! ```
//!
//! The template sorts a call's keys, so a call's arguments come back with the keys in that order
//! whatever order the request gave them: the replay counts such a case as respelled, not as a
//! difference (bellwether #110). No grammar yet: `Format::grammar` gives `None` for this syntax
//! until its tag is derived.

use crate::format::{CallSyntax, Emits, Format};

/// The Gemma 4 table.
pub fn gemma4() -> Format {
    Format::new("gemma4")
        .terminal("think_open", "<|channel>thought\n")
        .terminal("think_close", "<channel|>")
        .terminal("call_open", "<|tool_call>")
        .terminal("call_close", "<tool_call|>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("reasoning", "call_open", "call")
        .transition("call", "call_close", "content")
        .calls(CallSyntax::Gemma)
        .opens_turn("<|turn>model")
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
                name: "play".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "artist": {"type": "string"},
                    "n": {"type": "integer"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The generation prompt's tail: the turn opened and nothing more.
    const PROMPT: &str = "<|turn>model\n";

    /// An output as the template renders a turn: the thought, two calls with nothing between
    /// them, then the content.
    const OUTPUT: &str = concat!(
        "<|channel>thought\nThe user wants two songs.\n<channel|>",
        "<|tool_call>call:play{artist:<|\"|>Taylor Swift<|\"|>,n:20}<tool_call|>",
        "<|tool_call>call:play{artist:<|\"|>Maroon 5<|\"|>,n:15}<tool_call|>Two songs, then."
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(gemma4(), declared());
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
    fn a_rendered_turn_gives_its_thought_two_calls_and_its_content_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(reasoning(&events), "The user wants two songs.\n");
        assert_eq!(content(&events), "Two songs, then.");
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Taylor Swift", "n": 20}"#
        );
        assert_eq!(
            arguments_of(&events, 1),
            r#"{"artist": "Maroon 5", "n": 15}"#
        );
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["play", "play"]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 1, source } if source.text == "<tool_call|>"
        )));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_call_cut_across_pieces_streams_its_string_and_reads_the_same() {
        let whole = run(PROMPT, &[OUTPUT]);
        let boundaries: Vec<usize> = OUTPUT.char_indices().map(|(at, _)| at).collect();
        for &at in boundaries.iter().skip(1).step_by(7) {
            let events = run(PROMPT, &[&OUTPUT[..at], &OUTPUT[at..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {at}");
            assert_eq!(
                arguments_of(&events, 0),
                arguments_of(&whole, 0),
                "cut at {at}"
            );
            assert_eq!(
                arguments_of(&events, 1),
                arguments_of(&whole, 1),
                "cut at {at}"
            );
            assert_eq!(content(&events), content(&whole), "cut at {at}");
            assert_eq!(reasoning(&events), reasoning(&whole), "cut at {at}");
        }
    }

    #[test]
    fn a_string_with_newlines_quotes_and_a_split_token_streams_as_json_text() {
        let pieces = [
            "<|tool_call>call:play{artist:<|\"",
            "|>Say \"hi\"\n",
            "twice<|\"|>}<tool_call|>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Say \"hi\"\ntwice"}"#
        );
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_stream_cut_inside_a_call_closes_the_object_for_the_client_and_ends_the_call() {
        let mut parser = Engine::new(gemma4(), declared());
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
                    text: "<|tool_call>call:play{artist:<|\"|>Tay",
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
        assert_eq!(arguments_of(&events, 0), r#"{"artist": "Tay"}"#);
        assert!(events.iter().any(
            |event| matches!(event, Event::ToolCallEnd { source, .. } if source.text.is_empty())
        ));
    }

    #[test]
    fn the_table_validates() {
        gemma4().validate().expect("a valid table");
    }
}
