//! DeepSeek V4.1: reasoning between `<think>` and `</think>`, the calls of one turn between
//! `<｜DSML｜ calls>` and `</｜DSML｜ calls>`, each call between `<｜DSML｜ invoke name="…">` and
//! `</｜DSML｜ invoke>`, everything else content. The design's section 4 example, as recorded for
//! `deepseek-ai/DeepSeek-V4.1-Flash`.
//!
//! The block's opener is spelled as the family's reference parser spells it: the blank line the
//! template writes before the block and the tag, `\n\n<｜DSML｜ calls>`, so the blank line
//! leaves with the tag wherever the block opens, after a text preamble, after the thought's
//! close, or at the turn's start, and the content ends where the model's text does. The tag alone
//! opens the block too, for a block the model writes without the blank line.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed::Tags,
};

/// The DeepSeek V4.1 table.
pub fn deepseek_v4_1() -> Format {
    Format::new("deepseek_v4.1")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜DSML｜ calls>")
        // The opener with the blank line the template writes before the block (the syntax's
        // `before_block`), the spelling the reference parser looks for: the scanner takes the
        // longer marker where the blank line stands, and holds a newline that may begin it, so a
        // stream reads as the whole does. Its rows come after the bare opener's, which the
        // structural tags derive the block's opener from.
        .terminal("separated_calls_open", "\n\n<｜DSML｜ calls>")
        .terminal("calls_close", "</｜DSML｜ calls>")
        .terminal("invoke_open", "<｜DSML｜ invoke name=\"")
        .terminal("invoke_close", Tags::DSML.invoke_close)
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("invoke", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("content", "separated_calls_open", "calls")
        .transition("reasoning", "separated_calls_open", "calls")
        .transition("calls", "invoke_open", "invoke")
        .transition("invoke", "invoke_close", "calls")
        .transition("invoke", "invoke_open", "invoke")
        .transition("invoke", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Attributed(Tags::DSML))
        .opens_turn("<｜Assistant｜>")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        ids::CallIds,
        testing::{arguments_of, bytes, calls, content, reasoning},
    };

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        Declared,
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

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        let mut parser =
            Engine::with_call_ids(Arc::new(deepseek_v4_1()), Declared::default(), ids());
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
        out.into_owned().drain()
    }

    fn dropped(text: &str) -> Event<'static> {
        Event::Dropped {
            text: Text::uncounted(text.to_owned()),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn a_recorded_output_gives_its_two_calls_with_their_types_and_every_byte() {
        let events = run("<｜Assistant｜><think>", &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                dropped("</think>"),
                Event::ReasoningEnd,
                // The template's blank line before the block is the opener's: one marker.
                dropped("\n\n<｜DSML｜ calls>"),
            ]
        );
        assert_eq!(
            events[4],
            dropped("\n"),
            "the wrapping before the first invoke"
        );
        assert_eq!(events[5], dropped("<｜DSML｜ invoke name=\""));
        assert_eq!(
            events[6],
            Event::ToolCallStart {
                index: 0,
                id: ids().id(0),
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
                Event::ToolCallEnd { source, .. } => Some(&*source.text),
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
                Event::Content(text) => Some(&*text.text),
                _ => None,
            })
            .collect();
        // The newline after `</think>` is the template's spacing before the block, dropped; the
        // one after the block comes before text, so it is the text's.
        assert_eq!(content, "\nThe weather is sunny.");
    }

    #[test]
    fn a_missing_quote_on_the_name_starts_no_call_and_the_next_invoke_takes_index_zero() {
        // Two parameters: the first arrives while the name is still open and cuts it short, the
        // second arrives between tags with no call started, which is the guard this test holds.
        let output = concat!(
            "</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f>\n",
            "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n",
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
    fn only_the_invokes_closing_tag_is_a_calls_end() {
        // Three ways an invoke ends: its own closing tag, which the end carries; the next invoke's
        // opening and the block's close, which the engine drops as the region's, so the end
        // carries nothing.
        let output = concat!(
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n",
            "<｜DSML｜ invoke name=\"g\">\n<｜DSML｜ invoke name=\"h\">\n</｜DSML｜ calls>"
        );
        let events = run("<think>\n\n</think>\n\n", &[output]);
        assert_eq!(bytes(&events), output);
        let ends: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(&*source.text),
                _ => None,
            })
            .collect();
        assert_eq!(ends, ["</｜DSML｜ invoke>", "", ""]);
        // All three openers are dropped: `f`'s and `g`'s by the `calls + invoke_open` row, `h`'s by
        // the `invoke + invoke_open` row, which this test pins.
        assert_eq!(
            events
                .iter()
                .filter(|event| **event == dropped("<｜DSML｜ invoke name=\""))
                .count(),
            3
        );
        assert!(events.contains(&dropped("</｜DSML｜ calls>")));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 3, .. })
        ));
    }

    #[test]
    fn an_invoke_that_named_nothing_is_reported_however_it_ended() {
        // The invoke tag cut short before its name, ended by the block's close, by the next
        // invoke's opening, or by its own closing tag: each ending's bytes come back as
        // `Malformed`, and no call is counted for it.
        for (output, ending) in [
            (
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"</｜DSML｜ calls>",
                "</｜DSML｜ calls>",
            ),
            (
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"<｜DSML｜ invoke name=\"g\">\n\
                 </｜DSML｜ invoke>\n</｜DSML｜ calls>",
                "<｜DSML｜ invoke name=\"",
            ),
            (
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"</｜DSML｜ invoke>\n</｜DSML｜ calls>",
                "</｜DSML｜ invoke>",
            ),
        ] {
            let events = run("<think>\n\n</think>\n\n", &[output]);
            assert_eq!(bytes(&events), output, "{ending}");
            assert!(
                events.iter().any(|event| matches!(
                    event,
                    Event::Malformed { text, .. } if text.text == ending
                )),
                "{ending}: {events:?}"
            );
        }
        let events = run(
            "<think>\n\n</think>\n\n",
            &["<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"</｜DSML｜ calls>"],
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn a_tag_the_invokes_close_cut_short_inside_a_string_is_reported() {
        let output = concat!(
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
            "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ par</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls>"
        );
        let events = run("<think>\n\n</think>\n\n", &[output]);
        assert_eq!(bytes(&events), output);
        // The string the invoke's close cut stays open: nothing is made up for it, and the one
        // word before the end carries the tag's bytes.
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"#);
        let ended = events
            .iter()
            .position(|event| matches!(event, Event::ToolCallEnd { .. }))
            .expect("the call ends");
        assert!(matches!(
            &events[ended - 1],
            Event::Malformed { text, .. } if text.text == "</｜DSML｜ par"
        ));
    }

    #[test]
    fn thinking_off_starts_in_content() {
        let events = run("<think>\n\n</think>\n\n", &["Hello"]);
        assert_eq!(events[0], Event::Content(Text::uncounted("Hello")));
    }

    #[test]
    fn a_calls_block_in_an_earlier_turn_of_the_prompt_is_not_entered() {
        // The replay starts at the last `<｜Assistant｜>`, so the earlier turn's block, closed or
        // not, moves nothing; the thought the prompt opens is where the output starts.
        for earlier in [
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
        ] {
            let prompt = format!(
                "<｜User｜>q<｜Assistant｜></think>{earlier}<｜end▁of▁sentence｜>\
                 <｜User｜>r<｜Assistant｜><think>"
            );
            let events = run(&prompt, &["A thought.</think>\n\nAn answer."]);
            assert_eq!(events[0], Event::ReasoningStart, "{earlier:?}");
            assert_eq!(
                events[1],
                Event::Reasoning(Text::uncounted("A thought.")),
                "{earlier:?}"
            );
        }
    }

    #[test]
    fn a_parameter_tag_with_another_attribute_is_no_parameter() {
        let output = concat!(
            "</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
            "<｜DSML｜ parameter name=\"x\" kind=\"y\">v</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"a\" string=\"true\">1</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        );
        let events = run("<think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(arguments_of(&events, 0), r#"{"a": "1"}"#);
        let malformed: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert!(
            malformed
                .iter()
                .any(|text| text.ends_with("<｜DSML｜ parameter name=\"x\" kind=\"y\">")),
            "{malformed:?}"
        );
    }

    /// A recorded answer: a thought, a text preamble, the template's blank line, one call.
    const PREAMBLE_AND_CALL: &str = concat!(
        "The user wants detailed information about Apple Inc. stock. Let me call the tool.",
        "</think>I'll retrieve detailed stock information for Apple Inc.\n\n",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_stock_info\">\n",
        "<｜DSML｜ parameter name=\"company_name\" string=\"true\">Apple Inc.</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"detail_level\" string=\"true\">detailed</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    );

    /// A recorded answer: a preamble, then two calls in one block.
    const PREAMBLE_AND_PARALLEL_CALLS: &str = concat!(
        "We need to call get_exchange_rate for both pairs. Independent calls, so make both in ",
        "same block.</think>I'll fetch both exchange rates for you.\n\n",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_exchange_rate\">\n",
        "<｜DSML｜ parameter name=\"base_currency\" string=\"true\">USD</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"target_currency\" string=\"true\">AUD</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"get_exchange_rate\">\n",
        "<｜DSML｜ parameter name=\"base_currency\" string=\"true\">USD</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"target_currency\" string=\"true\">CAD</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    );

    /// A recorded answer with no preamble: the blank line after the thought's close, one call
    /// with a string, a number and a list.
    const CALL_WITHOUT_PREAMBLE: &str = concat!(
        "We need to call the weather tool.</think>\n\n",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"detailed_weather_forecast\">\n",
        "<｜DSML｜ parameter name=\"location\" string=\"true\">London, United Kingdom",
        "</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"details\" string=\"false\">",
        "[\"high_low_temperature\", \"humidity\", \"precipitation\"]</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    );

    /// A block with one call and no arguments.
    const BARE_BLOCK: &str =
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

    /// The content, the reasoning and the calls of `events`.
    fn said(events: &[Event]) -> (String, String, Vec<(String, String)>) {
        (content(events), reasoning(events), calls(events))
    }

    /// Reads `output` after `prompt` whole, a character at a time and at every two-way cut:
    /// each reading accounts for every byte and says what the whole does. What the whole says.
    fn however_cut(prompt: &str, output: &str) -> (String, String, Vec<(String, String)>) {
        let whole = run(prompt, &[output]);
        assert_eq!(bytes(&whole), output);
        let expected = said(&whole);
        let characters: Vec<&str> = output
            .char_indices()
            .map(|(at, c)| &output[at..at + c.len_utf8()])
            .collect();
        let by_character = run(prompt, &characters);
        assert_eq!(bytes(&by_character), output, "a character at a time");
        assert_eq!(said(&by_character), expected, "a character at a time");
        for cut in (1..output.len()).filter(|&cut| output.is_char_boundary(cut)) {
            let events = run(prompt, &[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(said(&events), expected, "cut at {cut}");
        }
        expected
    }

    /// The content deltas a client of the stream receives when `pieces` arrive one by one after
    /// `prompt`: the content of each feed's events, the end's included, empty ones left out.
    fn content_deltas(prompt: &str, pieces: &[&str]) -> Vec<String> {
        let mut engine =
            Engine::with_call_ids(Arc::new(deepseek_v4_1()), Declared::default(), ids());
        let mut out = Events::new();
        engine
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        let mut deltas = Vec::new();
        for piece in pieces {
            engine
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: piece,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("delta");
            deltas.push(content(&out.drain()));
        }
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        deltas.push(content(&out.drain()));
        deltas.retain(|delta| !delta.is_empty());
        deltas
    }

    #[test]
    fn a_preamble_ends_where_the_templates_blank_line_before_the_block_begins() {
        // The recorded answers with a preamble: the blank line between the text and the block is
        // the template's, read as the opener's, so the content is the preamble alone, whole, a
        // character at a time and at every cut; the calls read as before.
        let (content, reasoning, calls) = however_cut("<｜Assistant｜><think>", PREAMBLE_AND_CALL);
        assert_eq!(
            content,
            "I'll retrieve detailed stock information for Apple Inc."
        );
        assert_eq!(
            reasoning,
            "The user wants detailed information about Apple Inc. stock. Let me call the tool."
        );
        assert_eq!(
            calls,
            [(
                "get_stock_info".to_string(),
                r#"{"company_name": "Apple Inc.", "detail_level": "detailed"}"#.to_string()
            )]
        );
        let (content, _, calls) =
            however_cut("<｜Assistant｜><think>", PREAMBLE_AND_PARALLEL_CALLS);
        assert_eq!(content, "I'll fetch both exchange rates for you.");
        let rate = |target: &str| {
            (
                "get_exchange_rate".to_string(),
                format!(r#"{{"base_currency": "USD", "target_currency": "{target}"}}"#),
            )
        };
        assert_eq!(calls, [rate("AUD"), rate("CAD")]);
        // A blank line inside the preamble is the text's; the one before the block alone goes.
        let output = concat!(
            "</think>I'll work through this step by step.\n\n**Step 1: Total energy**\n\n",
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"biological_calc_energy\">\n",
            "<｜DSML｜ parameter name=\"mols\" string=\"false\">5</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        );
        let (content, _, calls) = however_cut("<｜Assistant｜><think>", output);
        assert_eq!(
            content,
            "I'll work through this step by step.\n\n**Step 1: Total energy**"
        );
        assert_eq!(
            calls,
            [(
                "biological_calc_energy".to_string(),
                r#"{"mols": 5}"#.to_string()
            )]
        );
    }

    #[test]
    fn a_call_without_a_preamble_has_no_content_and_its_values_their_types() {
        let (content, reasoning, calls) =
            however_cut("<｜Assistant｜><think>", CALL_WITHOUT_PREAMBLE);
        assert_eq!(content, "");
        assert_eq!(reasoning, "We need to call the weather tool.");
        assert_eq!(
            calls,
            [(
                "detailed_weather_forecast".to_string(),
                concat!(
                    r#"{"location": "London, United Kingdom", "days": 3, "#,
                    r#""details": ["high_low_temperature", "humidity", "precipitation"]}"#
                )
                .to_string()
            )]
        );
    }

    #[test]
    fn exactly_the_blank_line_before_the_block_is_the_openers() {
        // As the reference parser reads it: a third newline before the blank line is content,
        // and so is a space before it; a block opened without the blank line, or after one
        // newline, is still the block, the newline content.
        for (between, content) in [
            ("\n\n", "Sure."),
            ("\n\n\n", "Sure.\n"),
            (" \n\n", "Sure. "),
            ("", "Sure."),
            ("\n", "Sure.\n"),
        ] {
            let output = format!("</think>Sure.{between}{BARE_BLOCK}");
            let (said, _, calls) = however_cut("<think>", &output);
            assert_eq!(said, content, "{between:?}");
            assert_eq!(calls, [("f".to_string(), "{}".to_string())], "{between:?}");
        }
    }

    #[test]
    fn the_blank_line_before_the_block_leaves_no_content_delta() {
        // The recorded stream: the preamble's last token carries the blank line and the block's
        // opener comes next. The content deltas carry the preamble and nothing after it, so the
        // stream shows the content the whole answer has.
        let pieces = [
            "The user wants detailed information about Apple Inc. stock. Let me call the tool.",
            "</think>",
            "I",
            "'ll",
            " retrieve",
            " detailed",
            " stock",
            " information",
            " for",
            " Apple",
            " Inc",
            ".\n\n",
            "<｜DSML｜ calls>",
            "\n",
            "<｜DSML｜ invoke name=\"",
            "get_stock_info",
            "\">\n",
            "<｜DSML｜ parameter name=\"company_name\" string=\"true\">Apple Inc.</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"detail_level\" string=\"true\">detailed</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls>",
        ];
        assert_eq!(pieces.concat(), PREAMBLE_AND_CALL);
        let deltas = content_deltas("<｜Assistant｜><think>", &pieces);
        assert_eq!(
            deltas.concat(),
            "I'll retrieve detailed stock information for Apple Inc."
        );
        assert_eq!(deltas.last().map(String::as_str), Some("."));
        // The blank line alone in a delta, then the opener: no content at all.
        let deltas = content_deltas(
            "<think>",
            &["</think>", "\n\n", "<｜DSML｜ calls>", "\n", BARE_BLOCK],
        );
        assert_eq!(deltas, Vec::<String>::new());
    }

    #[test]
    fn the_turn_after_a_tool_result_reads_the_opener_as_the_turns_start_does() {
        // The template renders a tool result as a user turn and opens the model's turn after it,
        // `<｜Assistant｜>` with the thought opened (thinking) or closed (chat), so the output
        // starts in a stay of its own; and the blank line before a block is the opener's whether
        // or not text came before it in the stay, so neither mode shows it as content.
        let result = concat!(
            "<｜User｜>Weather in Paris?<｜Assistant｜><think>Need the weather.</think>\n\n",
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>",
            "<｜User｜><tool_result>{\"temperature\": 22}</tool_result>"
        );
        let thinking = format!("{result}<｜Assistant｜><think>");
        let chat = format!("{result}<｜Assistant｜></think>");
        for (prompt, output, expected) in [
            (
                &thinking,
                format!("Check again.</think>\n\n{BARE_BLOCK}"),
                ("", "Check again."),
            ),
            (
                &thinking,
                format!("Check again.</think>It is 22 degrees; checking again.\n\n{BARE_BLOCK}"),
                ("It is 22 degrees; checking again.", "Check again."),
            ),
            (&chat, format!("\n\n{BARE_BLOCK}"), ("", "")),
            (
                &chat,
                format!("Checking again.\n\n{BARE_BLOCK}"),
                ("Checking again.", ""),
            ),
        ] {
            let (content, reasoning, calls) = however_cut(prompt, &output);
            assert_eq!(
                (content.as_str(), reasoning.as_str()),
                expected,
                "{output:?}"
            );
            assert_eq!(calls, [("f".to_string(), "{}".to_string())], "{output:?}");
        }
        // The same at a turn's start in chat mode, with no result before it.
        let (content, reasoning, calls) = however_cut(
            "<｜User｜>Weather in Paris?<｜Assistant｜></think>",
            &format!("\n\n{BARE_BLOCK}"),
        );
        assert_eq!((content.as_str(), reasoning.as_str()), ("", ""));
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn a_block_the_thought_opens_after_a_blank_line_takes_the_blank_line_with_it() {
        let output = format!("Let me call.\n\n{BARE_BLOCK}");
        let (content, reasoning, calls) = however_cut("<think>", &output);
        assert_eq!((content.as_str(), reasoning.as_str()), ("", "Let me call."));
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn the_opener_with_its_blank_line_is_spelled_as_the_tags_write_it() {
        // The reading and the writing agree on what the template puts before the block.
        let spelled = format!("{}<｜DSML｜ calls>", Tags::DSML.before_block);
        assert!(deepseek_v4_1().terminal_texts().any(|text| text == spelled));
    }
}
