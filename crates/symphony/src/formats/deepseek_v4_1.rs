//! DeepSeek V4.1: reasoning between `<think>` and `</think>`, the calls of one turn between
//! `<｜DSML｜ calls>` and `</｜DSML｜ calls>`, each call between `<｜DSML｜ invoke name="…">` and
//! `</｜DSML｜ invoke>`, everything else content. The design's section 4 example, as recorded for
//! `deepseek-ai/DeepSeek-V4.1-Flash`.
//!
//! What this table says beyond the Qwen ones:
//!
//! - **Two states for the calls.** `calls` is the template's wrapping between invokes (a newline,
//!   dropped; anything else malformed), and `invoke` is one call's arguments, read by the DSML
//!   assembler ([`tagged::dsml`](crate::tagged::dsml)). A block with several invokes gives several
//!   calls, each with its own index, through the same two rows.
//! - **The invoke tag's opening is the terminal** that enters the arguments state, so the
//!   assembler starts inside the function's name; the closing tag is the terminal that leaves it,
//!   and the call's end carries its bytes. A second invoke before the first closed ends the first
//!   and starts the second (`invoke + invoke_open = invoke`), and the block's close ends an invoke
//!   still open (`invoke + calls_close = content`), so the call closes into an object and the
//!   prose after the block stays prose.
//! - **A calls block ends the thought** (`reasoning + calls_open = calls`), as the design's table
//!   and SMG's V4.1 reasoning parser have it; the recorded outputs close the thought first.
//! - **The prompt opens the thought.** The template ends the generation prompt with `<think>`, and
//!   an empty thought writes `</think>` at once, so most recorded outputs start with
//!   `</think>\n\n`; the engine's prompt replay puts the output inside the thought first. The
//!   turn opener is `<｜Assistant｜>`, so a marker quoted in an earlier turn moves nothing.

use crate::format::{CallSyntax, Emits, Format};

/// The DeepSeek V4.1 table.
pub fn deepseek_v4_1() -> Format {
    Format::new("deepseek_v4_1")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜DSML｜ calls>")
        .terminal("calls_close", "</｜DSML｜ calls>")
        .terminal("invoke_open", "<｜DSML｜ invoke name=\"")
        .terminal("invoke_close", "</｜DSML｜ invoke>")
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
        .calls(CallSyntax::Dsml)
        .opens_turn("<｜Assistant｜>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    /// A recorded output, qwen-style spacing and all: an empty thought, two invokes, a string and
    /// a JSON value.
    const OUTPUT: &str = concat!(
        "</think>\n\n<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"ChaDri_change_drink\">\n",
        "<｜DSML｜ parameter name=\"drink_id\" string=\"true\">latte</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"new_preferences\" string=\"false\">",
        "{\"size\": \"large\", \"temperature\": \"hot\"}</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "<｜DSML｜ invoke name=\"get_stock_price\">\n\n</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(deepseek_v4_1(), Declared::default());
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

    fn dropped(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn a_recorded_output_gives_its_two_calls_with_their_types_and_every_byte() {
        let events = run("<｜Assistant｜><think>", &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..5],
            [
                Event::ReasoningStart,
                dropped("</think>"),
                Event::ReasoningEnd,
                Event::Content(Text::uncounted("\n\n")),
                dropped("<｜DSML｜ calls>"),
            ]
        );
        assert_eq!(
            events[5],
            dropped("\n"),
            "the wrapping before the first invoke"
        );
        assert_eq!(events[6], dropped("<｜DSML｜ invoke name=\""));
        assert_eq!(
            events[7],
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "ChaDri_change_drink".into(),
                source: Text::uncounted("ChaDri_change_drink\">"),
            }
        );
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"drink_id": "latte", "new_preferences": {"size": "large", "temperature": "hot"}}"#
        );
        assert_eq!(arguments_of(&events, 1), "{}");
        let ends: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(source.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ends, ["</｜DSML｜ invoke>", "</｜DSML｜ invoke>"]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn a_string_value_streams_as_it_arrives_and_a_json_value_is_written_at_its_close() {
        let pieces = [
            "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">Par",
            "is</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">1",
            "2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        ];
        let events = run("<think>", &pieces);
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            fragments,
            ["{\"city\": \"", "Par", "is", "\"", ", \"n\": 12", "}"]
        );
        assert_eq!(bytes(&events), pieces.concat());
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = run("<think>", &[OUTPUT]);
        let whole_arguments = (arguments_of(&whole, 0), arguments_of(&whole, 1));
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run("<think>", &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(
                (arguments_of(&events, 0), arguments_of(&events, 1)),
                whole_arguments,
                "cut at {cut}"
            );
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 2, .. })),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_ends_the_call_with_its_arguments_cut() {
        let events = run(
            "<think>",
            &[concat!(
                "</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
                "<｜DSML｜ parameter name=\"x\" string=\"false\">1"
            )],
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: crate::event::MalformedReason::UnterminatedRegion,
                ..
            }
        )));
        // The call ends, with no bytes of its own, as every assembler ends a started call at the
        // end of the stream; its arguments stay cut, `{` without its `}`.
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 0, source } if source.text.is_empty()
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_calls_block_inside_the_thought_ends_it_and_a_second_invoke_ends_the_first() {
        let output = concat!(
            "Let me call.<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
            "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n",
            "<｜DSML｜ invoke name=\"g\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        );
        let events = run("<think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("Let me call.")),
                dropped("<｜DSML｜ calls>"),
                Event::ReasoningEnd,
            ]
        );
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"}"#);
        assert_eq!(arguments_of(&events, 1), "{}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn the_blocks_close_ends_an_invoke_left_open_and_the_prose_after_it_stays() {
        let output = concat!(
            "</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
            "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n",
            "</｜DSML｜ calls>\nThe weather is sunny."
        );
        let events = run("<think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"}"#);
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        // The newline after `</think>` is content too.
        assert_eq!(content, "\n\nThe weather is sunny.");
    }

    #[test]
    fn a_missing_quote_on_the_name_starts_no_call_and_the_next_invoke_takes_index_zero() {
        let output = concat!(
            "</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f>\n",
            "<｜DSML｜ parameter name=\"b\" string=\"true\">y</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"g\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        );
        let events = run("<think>", &[output]);
        assert_eq!(bytes(&events), output);
        let starts: Vec<(u32, &str)> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { index, name, .. } => Some((*index, name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(starts, [(0, "g")]);
        // No fragment before the start: everything of the unnamed invoke is reported.
        let first_start = events
            .iter()
            .position(|event| matches!(event, Event::ToolCallStart { .. }))
            .expect("g starts");
        assert!(!events[..first_start]
            .iter()
            .any(|event| matches!(event, Event::ToolCallArguments { .. })));
        assert_eq!(arguments_of(&events, 0), "{}");
    }

    #[test]
    fn a_marker_quoted_in_an_earlier_turn_moves_nothing() {
        let prompt = concat!(
            "<｜User｜>Why did you write <｜DSML｜ calls> and <｜DSML｜ invoke name=\" there?",
            "<｜Assistant｜><think>"
        );
        let events = run(
            prompt,
            &["The user asks about the tag.</think>\n\nIt opens a calls block."],
        );
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(
            events[1],
            Event::Reasoning(Text::uncounted("The user asks about the tag."))
        );
    }

    #[test]
    fn a_calls_block_in_the_prompt_is_not_entered_and_thinking_off_starts_in_content() {
        let events = run("<think>\n\n</think>\n\n", &["Hello"]);
        assert_eq!(events[0], Event::Content(Text::uncounted("Hello")));
    }
}
