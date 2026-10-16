//! DeepSeek V4: reasoning between `<think>` and `</think>`, the calls of one turn between
//! `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`, each call between `<｜DSML｜invoke name="…">`
//! and `</｜DSML｜invoke>`, everything else content: the DSML spelling before V4.1's, with no space
//! after the sentinel and `tool_calls` for the block's name, as the reference encoder of
//! `deepseek-ai/DeepSeek-V4-Flash` renders a turn:
//!
//! ```text
//! <｜Assistant｜><think>The user wants the weather in Beijing.</think>
//!
//! <｜DSML｜tool_calls>
//! <｜DSML｜invoke name="get_weather">
//! <｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
//! <｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜tool_calls><｜end▁of▁sentence｜>
//! ```
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed::Tags,
};

/// The DeepSeek V4 table.
pub fn deepseek_v4() -> Format {
    Format::new("deepseek_v4")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜DSML｜tool_calls>")
        .terminal("calls_close", "</｜DSML｜tool_calls>")
        .terminal("invoke_open", "<｜DSML｜invoke name=\"")
        .terminal("invoke_close", Tags::DSML_V4.invoke_close)
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
        .calls(CallSyntax::Attributed(Tags::DSML_V4))
        .opens_turn("<｜Assistant｜>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, MalformedReason, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    /// The turn the reference encoder renders for its first test, after the prompt's `<think>`: a
    /// thought, one invoke with two strings.
    const OUTPUT: &str = concat!(
        "The user wants to know the weather in Beijing. I should use the get_weather tool.",
        "</think>\n\n<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );

    /// An answer with content before the block and two JSON values, as the gateway's older parser
    /// for this checkpoint records it.
    const ANSWER: &str = concat!(
        "Sure, adding them.\n\n",
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"add_integers\">\n",
        "<｜DSML｜parameter name=\"a\" string=\"false\">17</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"b\" string=\"false\">25</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(deepseek_v4(), Declared::default());
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

    fn content_of(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning_of(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn calls_of(events: &[Event]) -> Vec<(u32, String)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { index, name, .. } => Some((*index, name.clone())),
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
    fn the_encoders_turn_gives_its_call_with_its_strings_and_every_byte() {
        let events = run("<｜Assistant｜><think>", &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..6],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted(
                    "The user wants to know the weather in Beijing. I should use the get_weather \
                     tool."
                )),
                dropped("</think>"),
                Event::ReasoningEnd,
                // The template's two newlines before the block: held until the block's opener
                // said they were its spacing, not content.
                Event::Dropped {
                    text: Text::uncounted("\n\n"),
                    why: DropReason::Whitespace,
                },
                dropped("<｜DSML｜tool_calls>"),
            ]
        );
        assert_eq!(events[6], dropped("\n"), "the wrapping before the invoke");
        assert_eq!(events[7], dropped("<｜DSML｜invoke name=\""));
        assert_eq!(
            events[8],
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "get_weather".into(),
                source: Text::uncounted("get_weather\">"),
            }
        );
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"location": "Beijing", "unit": "celsius"}"#
        );
        let ends: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(source.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ends, ["</｜DSML｜invoke>"]);
        assert!(events.contains(&dropped("</｜DSML｜tool_calls>")));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn the_older_parsers_fixtures_give_the_same_calls_with_their_types() {
        // Content before the block stays content, its two newlines included; `string="false"`
        // values are JSON.
        let events = run("<｜Assistant｜></think>", &[ANSWER]);
        assert_eq!(bytes(&events), ANSWER);
        assert_eq!(content_of(&events), "Sure, adding them.\n\n");
        assert_eq!(calls_of(&events), [(0, "add_integers".to_string())]);
        assert_eq!(arguments_of(&events, 0), r#"{"a": 17, "b": 25}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        // A string, a number and a boolean in one invoke.
        let mixed = concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"process\">\n",
            "<｜DSML｜parameter name=\"text\" string=\"true\">hello</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"count\" string=\"false\">42</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"enabled\" string=\"false\">true</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>"
        );
        let events = run("<｜Assistant｜></think>", &[mixed]);
        assert_eq!(bytes(&events), mixed);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"text": "hello", "count": 42, "enabled": true}"#
        );
        assert_eq!(content_of(&events), "");
    }

    #[test]
    fn a_string_value_streams_as_it_arrives_and_a_json_value_is_written_at_its_close() {
        let pieces = [
            "</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
            "<｜DSML｜parameter name=\"city\" string=\"true\">Par",
            "is</｜DSML｜parameter>\n<｜DSML｜parameter name=\"n\" string=\"false\">1",
            "2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
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
        for (prompt, output) in [("<think>", OUTPUT), ("</think>", ANSWER)] {
            let whole = run(prompt, &[output]);
            let whole_arguments = arguments_of(&whole, 0);
            let whole_content = content_of(&whole);
            for cut in 1..output.len() {
                if !output.is_char_boundary(cut) {
                    continue;
                }
                let events = run(prompt, &[&output[..cut], &output[cut..]]);
                assert_eq!(bytes(&events), output, "cut at {cut}");
                assert_eq!(arguments_of(&events, 0), whole_arguments, "cut at {cut}");
                assert_eq!(content_of(&events), whole_content, "cut at {cut}");
                assert!(
                    matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                    "cut at {cut}"
                );
            }
        }
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_ends_the_call_with_its_arguments_cut() {
        let events = run(
            "<think>",
            &[concat!(
                "</think>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
                "<｜DSML｜parameter name=\"x\" string=\"false\">1"
            )],
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::UnterminatedRegion,
                ..
            }
        )));
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
    fn a_calls_block_inside_the_thought_ends_it_and_is_the_call() {
        // The forced-call tag the table writes when the prompt leaves the thought open lets the
        // engine write the block before `</think>`, so the table reads the call it forced: the
        // block ends the thought, as under V4.1's table.
        let output = concat!(
            "Let me call.<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
            "<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        let events = run("<think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("Let me call.")),
                dropped("<｜DSML｜tool_calls>"),
                Event::ReasoningEnd,
            ]
        );
        assert_eq!(calls_of(&events), [(0, "f".to_string())]);
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        // Closed first, as the encoder renders it, the same block is the same call.
        let closed = format!("Let me call.</think>{}", &output["Let me call.".len()..]);
        let events = run("<think>", &[&closed]);
        assert_eq!(reasoning_of(&events), "Let me call.");
        assert_eq!(calls_of(&events), [(0, "f".to_string())]);
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"}"#);
    }

    #[test]
    fn the_blocks_close_ends_an_invoke_left_open_and_the_prose_after_it_stays() {
        let output = concat!(
            "</think>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
            "<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n",
            "</｜DSML｜tool_calls>\nThe weather is sunny."
        );
        let events = run("<think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"}"#);
        // The newline between `</think>` and the block is the template's spacing, dropped; the
        // one after the block comes before text, so it is the text's.
        assert_eq!(content_of(&events), "\nThe weather is sunny.");
    }

    /// The whitespace the engine dropped as the template's spacing, event by event.
    fn spacing(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Whitespace,
                } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The content deltas a client of the stream receives when `pieces` arrive one by one after
    /// `prompt`: the content of each feed's events, the end's included, empty ones left out.
    fn content_deltas(prompt: &str, pieces: &[&str]) -> Vec<String> {
        let mut parser = Engine::new(deepseek_v4(), Declared::default());
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
        let mut deltas = Vec::new();
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
            deltas.push(content_of(&out.drain()));
        }
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        deltas.push(content_of(&out.drain()));
        deltas.retain(|delta| !delta.is_empty());
        deltas
    }

    #[test]
    fn the_spacing_between_the_thought_and_the_block_is_not_content_whole_or_streamed() {
        // The encoder's turn has nothing between `</think>` and the block but the template's two
        // newlines: whole, byte by byte and in a tokenizer's pieces they are dropped as the
        // spacing they are, and no feed's events carry content, so a client of the stream gets
        // what the whole answer has: no content.
        let prompt = "<｜Assistant｜><think>";
        let whole = run(prompt, &[OUTPUT]);
        assert_eq!(content_of(&whole), "");
        assert_eq!(spacing(&whole), ["\n\n"]);
        let per_char: Vec<&str> = OUTPUT
            .char_indices()
            .map(|(i, c)| &OUTPUT[i..i + c.len_utf8()])
            .collect();
        let (thought, block) = OUTPUT.split_once("</think>").expect("the thought's close");
        let mut tokens = vec![thought, "</think>", "\n\n"];
        tokens.extend(block["\n\n".len()..].split_inclusive('\n'));
        assert_eq!(tokens.concat(), OUTPUT);
        for pieces in [per_char, tokens] {
            let events = run(prompt, &pieces);
            assert_eq!(bytes(&events), OUTPUT, "{pieces:?}");
            assert_eq!(content_of(&events), "", "{pieces:?}");
            assert_eq!(spacing(&events), ["\n\n"], "{pieces:?}");
            assert_eq!(
                arguments_of(&events, 0),
                r#"{"location": "Beijing", "unit": "celsius"}"#,
                "{pieces:?}"
            );
            assert_eq!(
                content_deltas(prompt, &pieces),
                Vec::<String>::new(),
                "{pieces:?}"
            );
        }
        // Content before the block keeps its two newlines, whole and streamed: they follow text.
        let (text, block) = ANSWER.split_once("\n\n").expect("the content's end");
        let pieces = [text, "\n\n", block];
        let events = run("<｜Assistant｜></think>", &pieces);
        assert_eq!(content_of(&events), "Sure, adding them.\n\n");
        assert_eq!(spacing(&events), Vec::<&str>::new());
        assert_eq!(
            content_deltas("<｜Assistant｜></think>", &pieces),
            ["Sure, adding them.", "\n\n"]
        );
    }

    #[test]
    fn a_second_invoke_ends_the_first_and_only_the_invokes_closing_tag_is_a_calls_end() {
        let output = concat!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n</｜DSML｜invoke>\n",
            "<｜DSML｜invoke name=\"g\">\n<｜DSML｜invoke name=\"h\">\n</｜DSML｜tool_calls>"
        );
        let events = run("<think>\n\n</think>\n\n", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls_of(&events),
            [
                (0, "f".to_string()),
                (1, "g".to_string()),
                (2, "h".to_string())
            ]
        );
        let ends: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(source.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ends, ["</｜DSML｜invoke>", "", ""]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 3, .. })
        ));
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        // Thinking mode ends the prompt with `<think>`: the output starts inside the thought.
        let events = run(
            "<｜User｜>hi<｜Assistant｜><think>",
            &["A thought.</think>Hi"],
        );
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(reasoning_of(&events), "A thought.");
        assert_eq!(content_of(&events), "Hi");
        // Chat mode writes `</think>` right after the assistant token: the output is content.
        let events = run("<｜User｜>hi<｜Assistant｜></think>", &["Hello"]);
        assert_eq!(events[0], Event::Content(Text::uncounted("Hello")));
        assert_eq!(reasoning_of(&events), "");
    }

    #[test]
    fn a_marker_quoted_in_an_earlier_turn_moves_nothing() {
        let prompt = concat!(
            "<｜User｜>Why did you write <｜DSML｜tool_calls> and <｜DSML｜invoke name=\" there?",
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
    fn a_calls_block_in_an_earlier_turn_of_the_prompt_is_not_entered() {
        for earlier in [
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
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
    fn the_spaced_spelling_of_the_later_checkpoint_is_content_here() {
        // The registry reads each checkpoint with its own table; under this one the spaced tags
        // are no markers, so the whole block is content and no call.
        let output = concat!(
            "</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
            "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        );
        let events = run("<think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert!(calls_of(&events).is_empty());
        assert_eq!(content_of(&events), &output["</think>".len()..]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    /// A turn the family's reference encoder renders, with the reading its own parser gives it.
    struct EncoderTurn {
        /// What the prompt ends with after the turn opener: `<think>` in thinking mode,
        /// `</think>` in chat mode.
        tail: &'static str,
        /// The turn as the model writes it, up to the end-of-turn token.
        output: &'static str,
        reasoning: &'static str,
        content: &'static str,
        /// Each call's name and its arguments as the encoder's parser serializes them.
        calls: &'static [(&'static str, &'static str)],
    }

    /// The turns the family's reference encoder (the model repository's `encoding` module) renders
    /// for the shapes a tool-calling conversation takes: two of the encoder's own test
    /// conversations, then one call with a string, a number, a boolean, an array and an object,
    /// two calls in one block, a call after content, a call with no parameters (the encoder writes
    /// an empty line inside the invoke), and content alone, in thinking and in chat mode.
    const ENCODER_TURNS: &[EncoderTurn] = &[
        // vendor-test-1
        EncoderTurn {
            tail: "<think>",
            output: "Got the weather data. Let me format a nice response.</think>The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.",
            reasoning: "Got the weather data. Let me format a nice response.",
            content: "The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.",
            calls: &[],
        },
        // vendor-test-3
        EncoderTurn {
            tail: "<think>",
            output: "用户想知道小柴胡冲剂和布洛芬能否一起服用。</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">\n<｜DSML｜parameter name=\"queries\" string=\"true\">小柴胡冲剂 布洛芬 相互作用 一起吃</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "用户想知道小柴胡冲剂和布洛芬能否一起服用。",
            content: "",
            calls: &[("search", "{\"queries\": \"小柴胡冲剂 布洛芬 相互作用 一起吃\"}")],
        },
        // every-value-kind/thinking
        EncoderTurn {
            tail: "<think>",
            output: "Call the tool with every kind of value.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n<｜DSML｜parameter name=\"metric\" string=\"false\">true</｜DSML｜parameter>\n<｜DSML｜parameter name=\"hours\" string=\"false\">[9, 18]</｜DSML｜parameter>\n<｜DSML｜parameter name=\"options\" string=\"false\">{\"lang\": \"fr\", \"n\": 2}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "Call the tool with every kind of value.",
            content: "",
            calls: &[("get_weather", "{\"location\": \"Paris\", \"days\": 3, \"metric\": true, \"hours\": [9, 18], \"options\": {\"lang\": \"fr\", \"n\": 2}}")],
        },
        // every-value-kind/chat
        EncoderTurn {
            tail: "</think>",
            output: "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n<｜DSML｜parameter name=\"metric\" string=\"false\">true</｜DSML｜parameter>\n<｜DSML｜parameter name=\"hours\" string=\"false\">[9, 18]</｜DSML｜parameter>\n<｜DSML｜parameter name=\"options\" string=\"false\">{\"lang\": \"fr\", \"n\": 2}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "",
            content: "",
            calls: &[("get_weather", "{\"location\": \"Paris\", \"days\": 3, \"metric\": true, \"hours\": [9, 18], \"options\": {\"lang\": \"fr\", \"n\": 2}}")],
        },
        // parallel-calls/thinking
        EncoderTurn {
            tail: "<think>",
            output: "Two calls at once.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"search\">\n<｜DSML｜parameter name=\"query\" string=\"true\">Paris weather</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "Two calls at once.",
            content: "",
            calls: &[("get_weather", "{\"location\": \"Paris\"}"), ("search", "{\"query\": \"Paris weather\"}")],
        },
        // call-after-content/thinking
        EncoderTurn {
            tail: "<think>",
            output: "Say something first.</think>Let me check.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "Say something first.",
            content: "Let me check.",
            calls: &[("get_weather", "{\"location\": \"Paris\"}")],
        },
        // call-after-content/chat
        EncoderTurn {
            tail: "</think>",
            output: "Let me check.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "",
            content: "Let me check.",
            calls: &[("get_weather", "{\"location\": \"Paris\"}")],
        },
        // no-arguments/thinking
        EncoderTurn {
            tail: "<think>",
            output: "A call with nothing in it.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"noop\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "A call with nothing in it.",
            content: "",
            calls: &[("noop", "{}")],
        },
        // no-arguments/chat
        EncoderTurn {
            tail: "</think>",
            output: "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"noop\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            reasoning: "",
            content: "",
            calls: &[("noop", "{}")],
        },
        // content-only/chat
        EncoderTurn {
            tail: "</think>",
            output: "It is sunny in Paris.",
            reasoning: "",
            content: "It is sunny in Paris.",
            calls: &[],
        },
    ];

    #[test]
    fn every_turn_the_encoder_renders_reads_as_its_own_parser_reads_it() {
        for turn in ENCODER_TURNS {
            let prompt = format!("<｜Assistant｜>{}", turn.tail);
            let events = run(&prompt, &[turn.output]);
            assert_eq!(bytes(&events), turn.output, "{:?}", turn.output);
            assert_eq!(reasoning_of(&events), turn.reasoning, "{:?}", turn.output);
            // The two newlines the template writes before a block follow the content here,
            // exactly two of them, where there is content; the encoder's parser skips them
            // (bellwether #17). Where the thought's close is all that comes before them, they
            // are the template's spacing, dropped, as the encoder's parser has it.
            let content = if turn.calls.is_empty() || turn.content.is_empty() {
                turn.content.to_string()
            } else {
                format!("{}\n\n", turn.content)
            };
            assert_eq!(content_of(&events), content, "{:?}", turn.output);
            let read: Vec<(String, serde_json::Value)> = calls_of(&events)
                .into_iter()
                .map(|(index, name)| {
                    let arguments: serde_json::Value =
                        serde_json::from_str(&arguments_of(&events, index)).expect("arguments");
                    (name, arguments)
                })
                .collect();
            let expected: Vec<(String, serde_json::Value)> = turn
                .calls
                .iter()
                .map(|(name, arguments)| {
                    (
                        name.to_string(),
                        serde_json::from_str(arguments).expect("the encoder's arguments"),
                    )
                })
                .collect();
            assert_eq!(read, expected, "{:?}", turn.output);
            let count = turn.calls.len() as u32;
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls, .. }) if *tool_calls == count),
                "{:?}",
                turn.output
            );
        }
    }

    #[test]
    fn an_invoke_with_no_parameters_reads_the_same_at_every_cut() {
        let turn = ENCODER_TURNS
            .iter()
            .find(|turn| turn.output.contains("<｜DSML｜invoke name=\"noop\">"))
            .expect("the shape");
        let prompt = format!("<｜Assistant｜>{}", turn.tail);
        let whole = run(&prompt, &[turn.output]);
        assert!(
            !whole
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "the whole turn reads with every wrapper byte dropped, none malformed"
        );
        for cut in 1..turn.output.len() {
            if !turn.output.is_char_boundary(cut) {
                continue;
            }
            let events = run(&prompt, &[&turn.output[..cut], &turn.output[cut..]]);
            assert_eq!(bytes(&events), turn.output, "cut at {cut}");
            assert_eq!(calls_of(&events), calls_of(&whole), "cut at {cut}");
            assert_eq!(
                arguments_of(&events, 0),
                arguments_of(&whole, 0),
                "cut at {cut}"
            );
            assert_eq!(content_of(&events), content_of(&whole), "cut at {cut}");
            assert_eq!(reasoning_of(&events), reasoning_of(&whole), "cut at {cut}");
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "cut at {cut}: a wrapper byte read as malformed text"
            );
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                "cut at {cut}"
            );
        }
    }
}
