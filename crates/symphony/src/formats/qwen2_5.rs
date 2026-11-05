//! Qwen2.5: each tool call between `<tool_call>` and `</tool_call>` as a JSON object, everything
//! else content. The template has no thought, so `<think>` and `</think>` are text where the model
//! writes them, which is what sets this table apart from [`qwen3()`](crate::formats::qwen3()): one
//! family, two tables, and the engine between them unchanged.
//!
//! Qwen2.5-Instruct, Qwen2.5-VL and Qwen2.5-Omni share the table; bellwether records them as
//! `qwen2.5-*`.

use crate::format::{CallSyntax, Emits, Format};

/// The Qwen2.5 table.
pub fn qwen2_5() -> Format {
    Format::new("qwen2.5")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("content", "call_open", "calls")
        .transition("calls", "call_close", "content")
        .transition("calls", "call_open", "calls")
        .calls(CallSyntax::Json)
        .opens_turn("<|im_start|>assistant")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapt::chat,
        engine::Engine,
        event::{DropReason, Event, Events, MalformedReason, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{arguments_of, bytes},
        Declared,
    };

    const CALL: &str = concat!(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
        "\n</tool_call>"
    );

    fn run(output: &str) -> Vec<Event<'static>> {
        run_pieces(&[output])
    }

    fn run_pieces(pieces: &[&str]) -> Vec<Event<'static>> {
        let mut parser = Engine::new(qwen2_5(), Declared::default());
        let mut out = Events::new();
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
        out.into_owned().drain()
    }

    /// Text after the call's object, before `</tool_call>`: the chat message's content and the
    /// stream's deltas are the model's words with the spaces between them, whole, at every cut
    /// and one character at a time alike; the call keeps its arguments, and the newline after the
    /// object and the one before the marker are the template's.
    #[test]
    fn text_after_the_calls_object_reaches_the_client_as_the_model_wrote_it() {
        let output = concat!(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n",
            "please note this and that\n</tool_call>"
        );
        let prose = "please note this and that";
        let shown = |events: &[Event]| chat::message(0, events).message.content;
        let streamed = |events: &[Event]| -> String {
            chat::deltas(0, events)
                .into_iter()
                .filter_map(|choice| choice.delta.content)
                .collect()
        };
        let words = |events: &[Event]| -> Vec<(String, MalformedReason)> {
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Malformed { text, why } => Some((text.text.to_string(), *why)),
                    _ => None,
                })
                .collect()
        };
        let whole = run(output);
        assert_eq!(bytes(&whole), output);
        assert_eq!(arguments_of(&whole, 0), r#"{"city": "Paris"}"#);
        let expected: Vec<(String, MalformedReason)> =
            ["please", " note", " this", " and", " that"]
                .into_iter()
                .map(|word| (word.to_string(), MalformedReason::TextAfterTheArguments))
                .collect();
        assert_eq!(words(&whole), expected);
        assert_eq!(shown(&whole).as_deref(), Some(prose));
        assert_eq!(streamed(&whole), prose);
        for cut in 1..output.len() {
            let events = run_pieces(&[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(
                arguments_of(&events, 0),
                r#"{"city": "Paris"}"#,
                "cut at {cut}"
            );
            assert_eq!(shown(&events).as_deref(), Some(prose), "cut at {cut}");
            assert_eq!(streamed(&events), prose, "cut at {cut}");
        }
        let pieces: Vec<&str> = output
            .char_indices()
            .map(|(at, c)| &output[at..at + c.len_utf8()])
            .collect();
        let events = run_pieces(&pieces);
        assert_eq!(bytes(&events), output);
        assert_eq!(shown(&events).as_deref(), Some(prose));
        assert_eq!(streamed(&events), prose);
    }

    #[test]
    fn a_think_block_is_content_and_a_call_is_a_call() {
        let events = run(&format!("<think>\nplan\n</think>\n\n{CALL}"));
        assert_eq!(
            events[..2],
            [
                Event::Content(Text::uncounted("<think>\nplan\n</think>\n\n")),
                Event::Dropped {
                    text: Text::uncounted("<tool_call>"),
                    why: DropReason::Wrapper,
                },
            ]
        );
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["get_weather"]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_prompt_that_opens_a_thought_opens_nothing_here() {
        let mut parser = Engine::new(qwen2_5(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "<|im_start|>assistant\n<think>\n",
                },
                &mut out,
            )
            .expect("prompt");
        assert!(
            out.is_empty(),
            "no state to enter: the table has no thought"
        );
    }
}
