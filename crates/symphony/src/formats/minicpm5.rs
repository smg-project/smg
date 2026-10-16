//! MiniCPM5: reasoning between `<think>` and `</think>`, each call between `<function name="…">`
//! and `</function>` as `<param name="…">` tags whose key is the attribute and whose value the
//! request's tools type, everything else content, and no block around the calls. Recorded as
//! `minicpm5-2b` (openbmb/MiniCPM5-2B), 54,181 parse cases, 15,432 of them with calls:
//!
//! ```text
//! <think>
//! The user wants two songs.
//! </think>
//!
//! <function name="play"><param name="artist">Swift</param><param name="n">20</param></function>
//! <function name="play"><param name="artist">Maroon 5</param><param name="n">15</param></function>
//! ```
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed,
};

/// The MiniCPM5 table.
pub fn minicpm5() -> Format {
    Format::new("minicpm5")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("function_open", "<function name=\"")
        .terminal("function_close", attributed::Tags::MINICPM5.invoke_close)
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("function", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "function_open", "function")
        .transition("reasoning", "function_open", "function")
        .transition("function", "function_close", "content")
        .calls(CallSyntax::Attributed(attributed::Tags::MINICPM5))
        .opens_turn("<|im_start|>assistant")
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
        Declared::of(&[
            Tool {
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
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "submit_patch".to_string(),
                    description: None,
                    parameters: value!({"type": "object", "properties": {
                        "patch": {"type": "string"},
                    }}),
                    strict: None,
                    extra: Default::default(),
                },
            },
        ])
    }

    /// The generation prompt's tail when the request leaves thinking to the model.
    const PROMPT: &str = "<|im_start|>assistant\n";

    /// An output as the template renders a turn: the thought, content, then two calls on lines
    /// of their own, nothing between a call's tags.
    const OUTPUT: &str = concat!(
        "<think>\nThe user wants two songs.\n</think>\n\nTwo songs, then.\n",
        "<function name=\"spotify_play\"><param name=\"artist\">Taylor Swift</param>",
        "<param name=\"duration\">20</param></function>\n",
        "<function name=\"spotify_play\"><param name=\"artist\">Maroon 5</param>",
        "<param name=\"duration\">15</param></function>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(minicpm5(), declared());
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
        // The newlines around the thought and the one after the content are the model's; the
        // one between the calls is the template's spacing, dropped.
        assert_eq!(content(&events), "\n\nTwo songs, then.\n");
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"artist": "Taylor Swift", "duration": 20}"#
        );
        assert_eq!(
            arguments_of(&events, 1),
            r#"{"artist": "Maroon 5", "duration": 15}"#
        );
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["spotify_play", "spotify_play"]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 1, source } if source.text == "</function>"
        )));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_cdata_value_is_the_text_inside_the_section_however_it_is_cut() {
        let call = concat!(
            "<function name=\"submit_patch\"><param name=\"patch\"><![CDATA[--- a/x\n+++ b/x\n",
            "+if a < b && c:]]></param></function>"
        );
        let whole = run(PROMPT, &[call]);
        assert_eq!(bytes(&whole), call);
        assert_eq!(
            arguments_of(&whole, 0),
            r#"{"patch": "--- a/x\n+++ b/x\n+if a < b && c:"}"#
        );
        assert!(malformed(&whole).is_empty());
        // Cut inside the opener, inside the text, between the closer's bytes.
        let pieces: Vec<&str> = {
            let a = call.find("CDATA").expect("opener");
            let b = call.find("+++").expect("text");
            let c = call.find("]]>").expect("closer") + 1;
            vec![&call[..a], &call[a..b], &call[b..c], &call[c..]]
        };
        let cut = run(PROMPT, &pieces);
        assert_eq!(bytes(&cut), call);
        assert_eq!(arguments_of(&cut, 0), arguments_of(&whole, 0));
        assert!(malformed(&cut).is_empty());
    }

    #[test]
    fn a_boolean_and_a_list_are_typed_by_the_tool_as_python_prints_them() {
        let call = concat!(
            "<function name=\"spotify_play\"><param name=\"shuffle\">True</param>",
            "<param name=\"tags\">['pop', 'live']</param></function>"
        );
        let events = run(PROMPT, &[call]);
        assert_eq!(bytes(&events), call);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"shuffle": true, "tags": ["pop", "live"]}"#
        );
    }

    #[test]
    fn a_call_right_after_the_thought_and_a_string_cut_across_pieces_stream() {
        let pieces = [
            "<think>\n\n</think>\n\n<function name=\"spotify_pl",
            "ay\"><param name=\"artist\">Tay",
            "lor Swift</param></function>",
        ];
        let events = run(PROMPT, &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        // An empty thought is the newline after `<think>` and the one before `</think>`; the two
        // between the thought and the call are the template's spacing, not content.
        assert_eq!(reasoning(&events), "\n\n");
        assert_eq!(content(&events), "");
        assert_eq!(arguments_of(&events, 0), r#"{"artist": "Taylor Swift"}"#);
        // The string streams as its pieces arrive: three fragments and the closing quote.
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
    }

    #[test]
    fn a_stream_cut_inside_a_call_closes_nothing_and_still_ends_the_call() {
        let mut parser = Engine::new(minicpm5(), declared());
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
                    text: "<function name=\"spotify_play\"><param name=\"artist\">Tay",
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
        minicpm5().validate().expect("a valid table");
    }
}
