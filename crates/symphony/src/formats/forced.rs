//! The forced call: the whole output is the arguments of the function the request forces, as the
//! engine writes it under the JSON-schema constraint the gateway builds when `tool_choice` names a
//! function (the function's `parameters` as the schema), with no marker and no name in the output.
//! The one format known from the request rather than from the model: the gateway picks it for such
//! a request, with the function's name in [`Declared::forcing`](crate::tagged::Declared::forcing),
//! and no operator names it. A required call under the same constraint is a list of
//! `{"name": …, "parameters": …}` objects, which the [`json`](super::json()) table reads.
//!
//! ```text
//! {"city": "Paris"}
//! ```
//!
//! Notes on this format, and on what it leaves open: `docs/tables.md`.

use crate::format::{CallSyntax, Emits, Format};

/// The forced call's format.
pub fn forced() -> Format {
    Format::new("forced")
        .state("arguments", Emits::Arguments)
        .calls(CallSyntax::ForcedArguments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events, FinishReason, MalformedReason},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn run(pieces: &[&str], declared: Declared) -> Vec<Event> {
        let mut parser = Engine::new(forced(), declared);
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "<|im_start|>assistant\n",
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

    fn calls(events: &[Event]) -> Vec<(String, String)> {
        let mut calls: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { name, .. } => calls.push((name.clone(), String::new())),
                Event::ToolCallArguments { json, .. } => {
                    calls.last_mut().expect("a call started").1.push_str(json);
                }
                _ => {}
            }
        }
        calls
    }

    fn finish(events: &[Event]) -> (FinishReason, u32) {
        events
            .iter()
            .find_map(|event| match event {
                Event::Finish {
                    reason, tool_calls, ..
                } => Some((reason.clone(), *tool_calls)),
                _ => None,
            })
            .expect("a finish")
    }

    #[test]
    fn the_output_is_the_forced_functions_arguments_however_it_is_cut() {
        // What the engine writes for `tool_choice: {"type": "function", "function": {"name":
        // "get_weather"}}` under the gateway's schema: the arguments object alone.
        let arguments = "{\"city\": \"Paris\", \"days\": [1, 2.5, true, null]}";
        let declared = Declared::default().forcing("get_weather");
        for pieces in [
            vec![arguments],
            vec!["{\"city\": \"Pa", "ris\", \"days\": [1, 2.5, true, null]}"],
        ] {
            let events = run(&pieces, declared.clone());
            assert_eq!(
                calls(&events),
                [("get_weather".to_string(), arguments.to_string())]
            );
            assert_eq!(finish(&events), (FinishReason::Stop, 1));
            assert!(!events
                .iter()
                .any(|event| matches!(event, Event::Content(_) | Event::Reasoning(_))));
        }
    }

    #[test]
    fn a_cut_object_a_blank_output_and_an_unforced_request_say_what_they_are() {
        let declared = Declared::default().forcing("get_weather");
        let events = run(&["{\"city\": \"Par"], declared.clone());
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
        let events = run(&["  "], declared);
        assert!(calls(&events).is_empty());
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        // The gateway never assigns the format without a forced function; the output is then
        // reported, not read as content.
        let events = run(&["{\"city\": \"Paris\"}"], Declared::default());
        assert!(calls(&events).is_empty());
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::Other(_),
                ..
            }
        )));
    }

    #[test]
    fn the_format_validates_derives_no_grammar_and_has_no_opener() {
        let format = forced();
        format.validate().expect("a valid format");
        assert_eq!(format.name(), "forced");
        assert_eq!(format.turn_opener(), None);
        assert!(format.grammar(&[], true, false).is_none());
    }
}
