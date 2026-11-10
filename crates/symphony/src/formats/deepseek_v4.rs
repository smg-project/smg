//! DeepSeek V4: reasoning between `<think>` and `</think>`, the calls of one turn between
//! `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`, each call between `<｜DSML｜invoke name="…">`
//! and `</｜DSML｜invoke>`, an invoke written outside a block a call too, as the serving engines'
//! own parsers read it, everything else content: the DSML spelling before V4.1's, with no space
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
//! The template writes `<｜Assistant｜>` and the thought's opener (`<think>`, or `</think>` in chat
//! mode) after a user message, and after a system message that ends the conversation or that an
//! assistant turn follows; a system message a latest reminder follows stays bare, and the
//! checkpoint's own encoder file opens the turn after user and developer turns only, so a prompt
//! can still leave the turn unopened, and the model, which thinks unless told not to, starts
//! inside its thought. The table starts there too (the unopened row, `<｜end▁of▁sentence｜>` ending
//! a turn the model wrote), so `</think>` closes the thought instead of reaching the content.
//!
//! The block's opener is spelled as the family's reference parser spells it: the blank line the
//! template writes before the block and the tag, `\n\n<｜DSML｜tool_calls>`, so the blank line
//! leaves with the tag wherever the block opens, after a text preamble, after the thought's close,
//! at the turn's start or after a tool result, and the content ends where the model's text does.
//! The tag alone opens the block too, for a block the model writes without the blank line.
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
        // The opener with the blank line the template writes before the block (the syntax's
        // `before_block`), the spelling the reference parser looks for: the scanner takes the
        // longer marker where the blank line stands, and holds a newline that may begin it, so a
        // stream reads as the whole does. Its rows come after the bare opener's, which the
        // structural tags derive the block's opener from.
        .terminal("separated_calls_open", "\n\n<｜DSML｜tool_calls>")
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
        .transition("content", "separated_calls_open", "calls")
        .transition("reasoning", "separated_calls_open", "calls")
        .transition("calls", "invoke_open", "invoke")
        // An invoke the model writes outside a block is the call, as the serving engines' own
        // parsers for the family read it (their machines start a call at the invoke's opener from
        // the content as from the block); before this row the table kept such an invoke as
        // content, its two tags counted as markers out of place, and the call was lost. The
        // close leads into the block's wrapping state, as a block's invoke's does, so a second
        // bare invoke is the next call and prose after a bare call is text between calls; a
        // block opener met there is wrapping, as it is to the engines' machines, so a call the
        // model writes bare and then again inside a block is two calls and no markup.
        .transition("content", "invoke_open", "invoke")
        .transition("calls", "calls_open", "calls")
        .transition("calls", "separated_calls_open", "calls")
        .transition("invoke", "invoke_close", "calls")
        .transition("invoke", "invoke_open", "invoke")
        .transition("invoke", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Attributed(Tags::DSML_V4))
        .opens_turn("<｜Assistant｜>")
        .unopened_turn_starts_in("reasoning", "<｜end▁of▁sentence｜>")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        ids::CallIds,
        testing::{
            arguments_of, bytes, content, drive, drive_to, malformed, reasoning,
            the_same_at_every_cut, weather_tools,
        },
    };

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, MalformedReason, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        Declared,
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

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(
            &mut Engine::with_call_ids(Arc::new(deepseek_v4()), Declared::default(), ids()),
            prompt,
            pieces,
        )
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

    fn dropped(text: &str) -> Event<'static> {
        Event::Dropped {
            text: Text::uncounted(text.to_owned()),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn the_encoders_turn_gives_its_call_with_its_strings_and_every_byte() {
        let events = run("<｜Assistant｜><think>", &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..5],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted(
                    "The user wants to know the weather in Beijing. I should use the get_weather \
                     tool."
                )),
                dropped("</think>"),
                Event::ReasoningEnd,
                // The template's blank line before the block is the opener's: one marker.
                dropped("\n\n<｜DSML｜tool_calls>"),
            ]
        );
        assert_eq!(events[5], dropped("\n"), "the wrapping before the invoke");
        assert_eq!(events[6], dropped("<｜DSML｜invoke name=\""));
        assert_eq!(
            events[7],
            Event::ToolCallStart {
                index: 0,
                id: ids().id(0),
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
                Event::ToolCallEnd { source, .. } => Some(&*source.text),
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
        // Content before the block ends where the text does: the blank line the template writes
        // before the block is the opener's (the older parser strips it too); `string="false"`
        // values are JSON.
        let events = run("<｜Assistant｜></think>", &[ANSWER]);
        assert_eq!(bytes(&events), ANSWER);
        assert_eq!(content(&events), "Sure, adding them.");
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
        assert_eq!(content(&events), "");
    }

    #[test]
    fn a_string_value_streams_as_it_arrives_and_a_json_value_is_written_at_its_close() {
        let pieces = [
            "</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
            "<｜DSML｜parameter name=\"city\" string=\"true\">Par",
            "is</｜DSML｜parameter>\n<｜DSML｜parameter name=\"n\" string=\"false\">1",
            "2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ];
        let events = run("<｜Assistant｜><think>", &pieces);
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_ref()),
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
        for (prompt, output) in [
            ("<｜Assistant｜><think>", OUTPUT),
            ("<｜Assistant｜></think>", ANSWER),
        ] {
            let whole = run(prompt, &[output]);
            let whole_arguments = arguments_of(&whole, 0);
            let whole_content = content(&whole);
            for cut in 1..output.len() {
                if !output.is_char_boundary(cut) {
                    continue;
                }
                let events = run(prompt, &[&output[..cut], &output[cut..]]);
                assert_eq!(bytes(&events), output, "cut at {cut}");
                assert_eq!(arguments_of(&events, 0), whole_arguments, "cut at {cut}");
                assert_eq!(content(&events), whole_content, "cut at {cut}");
                assert!(
                    matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                    "cut at {cut}"
                );
            }
        }
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_ends_the_call_with_its_arguments_cut() {
        let output = concat!(
            "</think>\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
            "<｜DSML｜parameter name=\"x\" string=\"false\">1"
        );
        let events = drive_to(
            &mut Engine::with_call_ids(Arc::new(deepseek_v4()), Declared::default(), ids()),
            "<think>",
            &[output],
            EngineFinish::Length,
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
        // Cut by the token limit, the held value is the call's own, as written.
        assert_eq!(arguments_of(&events, 0), r#"{"x": 1"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        // The model's own end of output there ends the call as its closing tag would: the value
        // held for its close is refused, and no close is made up for a call the syntax refused
        // bytes in, so no word says the stop closed it.
        let stopped = run("<think>", &[output]);
        assert_eq!(arguments_of(&stopped, 0), "");
        assert!(!stopped.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::OpenAtTheStop,
                ..
            }
        )));
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
        let events = run("<｜Assistant｜><think>", &[output]);
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
        let events = run("<｜Assistant｜><think>", &[&closed]);
        assert_eq!(reasoning(&events), "Let me call.");
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
        let events = run("<｜Assistant｜><think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"}"#);
        // The newline between `</think>` and the block is the template's spacing, dropped; the
        // one after the block comes before text, so it is the text's.
        assert_eq!(content(&events), "\nThe weather is sunny.");
    }

    /// The whitespace the engine dropped as the template's spacing, event by event.
    fn spacing<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Whitespace,
                } => Some(&*text.text),
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
            deltas.push(content(&out.drain()));
        }
        parser
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
    fn the_blank_line_between_the_thought_and_the_block_is_the_openers_whole_or_streamed() {
        // The encoder's turn has nothing between `</think>` and the block but the template's
        // blank line: whole, byte by byte and in a tokenizer's pieces it leaves with the tag as
        // the block's opener, no whitespace dropped on its own, and no feed's events carry
        // content, so a client of the stream gets what the whole answer has: no content.
        let prompt = "<｜Assistant｜><think>";
        let whole = run(prompt, &[OUTPUT]);
        assert_eq!(content(&whole), "");
        assert_eq!(spacing(&whole), Vec::<&str>::new());
        assert!(whole.contains(&dropped("\n\n<｜DSML｜tool_calls>")));
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
            assert_eq!(content(&events), "", "{pieces:?}");
            assert_eq!(spacing(&events), Vec::<&str>::new(), "{pieces:?}");
            assert!(
                events.contains(&dropped("\n\n<｜DSML｜tool_calls>")),
                "{pieces:?}"
            );
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
        // Content before the block ends where the text does, whole and streamed: the blank line
        // after it is the opener's, and no content delta carries it.
        let (text, block) = ANSWER.split_once("\n\n").expect("the content's end");
        let pieces = [text, "\n\n", block];
        let events = run("<｜Assistant｜></think>", &pieces);
        assert_eq!(content(&events), "Sure, adding them.");
        assert_eq!(spacing(&events), Vec::<&str>::new());
        assert_eq!(
            content_deltas("<｜Assistant｜></think>", &pieces),
            ["Sure, adding them."]
        );
    }

    #[test]
    fn a_second_invoke_ends_the_first_and_only_the_invokes_closing_tag_is_a_calls_end() {
        let output = concat!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n</｜DSML｜invoke>\n",
            "<｜DSML｜invoke name=\"g\">\n<｜DSML｜invoke name=\"h\">\n</｜DSML｜tool_calls>"
        );
        let events = run("<｜Assistant｜></think>", &[output]);
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
                Event::ToolCallEnd { source, .. } => Some(&*source.text),
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
        assert_eq!(reasoning(&events), "A thought.");
        assert_eq!(content(&events), "Hi");
        // Chat mode writes `</think>` right after the assistant token: the output is content.
        let events = run("<｜User｜>hi<｜Assistant｜></think>", &["Hello"]);
        assert_eq!(events[0], Event::Content(Text::uncounted("Hello")));
        assert_eq!(reasoning(&events), "");
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

    /// The prompt the checkpoint's own encoder file writes for a conversation that ends with a
    /// system message, as a coding agent's client sends one (its environment notes after the user
    /// turn): that file writes the turn opener after user and developer turns only, so the prompt
    /// opens no turn, and the model starts inside its thought as it would after
    /// `<｜Assistant｜><think>`.
    const TRAILING_SYSTEM_PROMPT: &str = concat!(
        "<｜begin▁of▁sentence｜>You are a coding assistant. Be brief.",
        "<｜User｜>The test suite has a failing test. Find the bug under src/ and fix it.",
        "# Environment\nYou have been invoked in the following environment:\n",
        " - Platform: linux\n\nToday's date is 2026-10-16."
    );

    /// The prompt the checkpoint's own encoder file writes, with the thoughts kept, for the same
    /// client's later request: the environment notes after the first user turn, the model's turn
    /// after them, the tool's result merged into a user turn, and a note after that, with
    /// `</think>` quoted in the result. That file wrote no opener before the model's earlier turn
    /// either (a system message came before it, and it opens the turn after user and developer
    /// turns only), so the prompt holds none.
    const AGENT_LATER_TURN_PROMPT: &str = concat!(
        "<｜begin▁of▁sentence｜>You are a coding assistant. Be brief.",
        "<｜User｜>Fix the failing test.# Environment\n - Platform: linux\n",
        "Run the suite first.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"run\">\n",
        "<｜DSML｜parameter name=\"command\" string=\"true\">pytest -q</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>",
        "<｜User｜><tool_result>FAILED tests/test_versions.py::test_parse - ",
        "the fixture says </think> here</tool_result>",
        "<total_tokens>10000 tokens left</total_tokens>"
    );

    /// What the turn said: the reasoning, the content, and the first call's arguments.
    fn said(events: &[Event]) -> (String, String, String) {
        (reasoning(events), content(events), arguments_of(events, 0))
    }

    #[test]
    fn a_conversation_that_ends_with_a_system_message_starts_the_output_inside_the_thought() {
        let (thought, after) = OUTPUT.split_once("</think>").expect("the thought's close");
        // The thought's close alone, then the calls block: an empty thought and no content, the
        // close never reaching the content.
        let events = run(TRAILING_SYSTEM_PROMPT, &["</think>", after]);
        assert_eq!(events[0], Event::ReasoningStart);
        assert!(events.contains(&Event::ReasoningEnd), "{events:?}");
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Content(_))),
            "{events:?}"
        );
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "");
        assert_eq!(calls_of(&events), [(0, "get_weather".to_string())]);
        assert_eq!(bytes(&events), format!("</think>{after}"));
        // The thought, then the calls: read as after `<｜Assistant｜><think>`.
        let events = run(TRAILING_SYSTEM_PROMPT, &[OUTPUT]);
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(
            said(&events),
            said(&run("<｜User｜>q<｜Assistant｜><think>", &[OUTPUT]))
        );
        assert_eq!(reasoning(&events), thought);
        assert_eq!(content(&events), "");
        assert_eq!(calls_of(&events), [(0, "get_weather".to_string())]);
        // Content after the thought.
        let events = run(
            TRAILING_SYSTEM_PROMPT,
            &["The notes are context.</think>Running the suite."],
        );
        assert_eq!(reasoning(&events), "The notes are context.");
        assert_eq!(content(&events), "Running the suite.");
    }

    #[test]
    fn a_later_turn_whose_prompt_opens_no_turn_starts_inside_the_thought_too() {
        let (thought, _) = OUTPUT.split_once("</think>").expect("the thought's close");
        // The earlier turn's `</think>` and calls block, and the `</think>` quoted in the tool's
        // result, move nothing: the prompt opens no turn, and nothing of it is replayed.
        let events = run(AGENT_LATER_TURN_PROMPT, &[OUTPUT]);
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(reasoning(&events), thought);
        assert_eq!(content(&events), "");
        assert_eq!(calls_of(&events), [(0, "get_weather".to_string())]);
        // The prompt the checkpoint's own encoder file writes, with the thoughts kept, for an
        // earlier turn it opened, then a user turn and a system note: the turn's end after the
        // last opener says the prompt opens no turn for the model.
        let prompt = concat!(
            "<｜User｜>u1<｜Assistant｜><think>r1</think>a1<｜end▁of▁sentence｜>",
            "<｜User｜>u2# Environment\n - Platform: linux"
        );
        let events = run(prompt, &["r2</think>a2"]);
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(reasoning(&events), "r2");
        assert_eq!(content(&events), "a2");
        // The model's own turn continued without its end is a prefill the output goes on with.
        let events = run("<｜User｜>u<｜Assistant｜></think>Sure,", &[" here it is."]);
        assert!(!events.contains(&Event::ReasoningStart), "{events:?}");
        assert_eq!(content(&events), " here it is.");
    }

    #[test]
    fn an_unopened_turn_reads_the_same_whole_at_every_byte_and_in_a_tokenizers_pieces() {
        let whole = said(&run(TRAILING_SYSTEM_PROMPT, &[OUTPUT]));
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run(TRAILING_SYSTEM_PROMPT, &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(said(&events), whole, "cut at {cut}");
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                "cut at {cut}"
            );
        }
        let per_char: Vec<&str> = OUTPUT
            .char_indices()
            .map(|(i, c)| &OUTPUT[i..i + c.len_utf8()])
            .collect();
        let (thought, block) = OUTPUT.split_once("</think>").expect("the thought's close");
        let mut tokens = vec![thought, "</think>", "\n\n"];
        tokens.extend(block["\n\n".len()..].split_inclusive('\n'));
        for pieces in [per_char, tokens] {
            let events = run(TRAILING_SYSTEM_PROMPT, &pieces);
            assert_eq!(bytes(&events), OUTPUT, "{pieces:?}");
            assert_eq!(said(&events), whole, "{pieces:?}");
        }
    }

    #[test]
    fn the_chat_and_messages_adapters_keep_an_unopened_turns_thought_out_of_the_content() {
        use crate::adapt::{chat, messages};
        let (thought, after) = OUTPUT.split_once("</think>").expect("the thought's close");
        let kinds = |events: &[Event]| -> Vec<String> {
            let blocks = serde_json::to_value(messages::output(events).content).expect("blocks");
            blocks
                .as_array()
                .expect("an array")
                .iter()
                .map(|block| block["type"].as_str().unwrap_or("").to_string())
                .collect()
        };
        // The Messages API: a thinking block and the call, never a text block with the close.
        let events = run(TRAILING_SYSTEM_PROMPT, &[OUTPUT]);
        assert_eq!(kinds(&events), ["thinking", "tool_use"]);
        let blocks = serde_json::to_value(messages::output(&events).content).expect("blocks");
        assert_eq!(blocks[0]["thinking"], thought);
        // An empty thought is no block at all.
        let events = run(TRAILING_SYSTEM_PROMPT, &["</think>", after]);
        assert_eq!(kinds(&events), ["tool_use"]);
        // Chat completions: the thought in `reasoning_content`, no content.
        let events = run(TRAILING_SYSTEM_PROMPT, &[OUTPUT]);
        let choice = chat::message(0, &events);
        assert_eq!(choice.message.reasoning_content.as_deref(), Some(thought));
        assert_eq!(choice.message.content, None);
        assert_eq!(choice.message.tool_calls.as_ref().map(Vec::len), Some(1));
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
        let events = run("<｜Assistant｜><think>", &[output]);
        assert_eq!(bytes(&events), output);
        assert!(calls_of(&events).is_empty());
        assert_eq!(content(&events), &output["</think>".len()..]);
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
            assert_eq!(reasoning(&events), turn.reasoning, "{:?}", turn.output);
            // The blank line the template writes before a block is the opener's, after content
            // as after the thought's close: the content is what the encoder's parser reads, which
            // looks for the block as the blank line and the tag.
            assert_eq!(content(&events), turn.content, "{:?}", turn.output);
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
            assert_eq!(content(&events), content(&whole), "cut at {cut}");
            assert_eq!(reasoning(&events), reasoning(&whole), "cut at {cut}");
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

    /// A parameter tag the model wrote without its `<` (seen live: the third of four parallel
    /// calls lost its `action` this way): the invoke is read without that member and gets no
    /// close, so its arguments are no whole value and a client is told the call failed rather
    /// than handed a closed object with the member missing; the bytes of the slip, `｜DSML｜` and
    /// the closing tag included, are refused as the model's failed call, counted and never
    /// content, where they reached the client as content before.
    #[test]
    fn a_parameter_tag_without_its_angle_bracket_is_refused_never_content() {
        const OUTPUT: &str = concat!(
            "Let me estimate.</think>\n\n<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"emission_estimator\">\n",
            "<｜DSML｜parameter name=\"current_emissions\" string=\"false\">10</｜DSML｜parameter>\n",
            "｜DSML｜parameter name=\"action\" string=\"true\">plant trees</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"duration\" string=\"false\">1</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        const SLIP: &str =
            "｜DSML｜parameter name=\"action\" string=\"true\">plant trees</｜DSML｜parameter>";
        for pieces in [vec![OUTPUT], OUTPUT.split_inclusive(' ').collect()] {
            let events = run("<｜Assistant｜><think>", &pieces);
            assert_eq!(bytes(&events), OUTPUT, "{pieces:?}");
            assert_eq!(calls_of(&events), [(0, "emission_estimator".to_string())]);
            assert_eq!(
                arguments_of(&events, 0),
                r#"{"current_emissions": 10, "duration": 1"#,
                "{pieces:?}"
            );
            // What a chat client reads: content and malformed text alike. Nothing.
            let shown: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Content(text) | Event::Malformed { text, .. } => Some(&*text.text),
                    _ => None,
                })
                .collect();
            assert_eq!(shown, "", "{pieces:?}: {events:?}");
            let refused: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Dropped {
                        text,
                        why: DropReason::MalformedCall,
                    } => Some(&*text.text),
                    _ => None,
                })
                .collect();
            assert_eq!(refused, SLIP, "{pieces:?}");
            let reasons: Vec<MalformedReason> = events
                .iter()
                .filter_map(|event| match event {
                    Event::Malformed { why, .. } => Some(*why),
                    _ => None,
                })
                .collect();
            assert!(
                reasons.iter().all(|why| matches!(
                    why,
                    MalformedReason::TextBetweenTags | MalformedReason::TagOutOfPlace
                )) && reasons.contains(&MalformedReason::TagOutOfPlace),
                "{pieces:?}: {reasons:?}"
            );
        }
    }

    /// A block with one call and no arguments, as the encoder writes it.
    const BARE_BLOCK: &str =
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    /// Reads `output` after `prompt` whole, a character at a time and at every two-way cut, each
    /// reading saying what the whole does; the content, the reasoning and the calls.
    fn however_cut(prompt: &str, output: &str) -> (String, String, Vec<(String, String)>) {
        assert_eq!(bytes(&run(prompt, &[output])), output);
        let said = the_same_at_every_cut(|pieces| run(prompt, pieces), output);
        (said.content, said.reasoning, said.calls)
    }

    #[test]
    fn a_preamble_ends_where_the_templates_blank_line_before_the_block_begins() {
        // The turns the encoder renders for a call after content, in thinking and in chat mode:
        // the content is the text alone, as the encoder's parser reads it, whole, a character at
        // a time and at every cut; the call reads as before.
        let turns: Vec<&EncoderTurn> = ENCODER_TURNS
            .iter()
            .filter(|turn| turn.content == "Let me check.")
            .collect();
        assert_eq!(turns.len(), 2, "thinking and chat");
        for turn in turns {
            let prompt = format!("<｜Assistant｜>{}", turn.tail);
            let (content, reasoning, calls) = however_cut(&prompt, turn.output);
            assert_eq!(content, "Let me check.", "{:?}", turn.output);
            assert_eq!(reasoning, turn.reasoning, "{:?}", turn.output);
            assert_eq!(
                calls,
                [(
                    "get_weather".to_string(),
                    r#"{"location": "Paris"}"#.to_string()
                )],
                "{:?}",
                turn.output
            );
        }
        // A blank line inside the content is the text's; the one before the block alone goes.
        let output = format!("</think>Step by step.\n\n**Step 1**\n\n{BARE_BLOCK}");
        let (content, _, calls) = however_cut("<｜Assistant｜><think>", &output);
        assert_eq!(content, "Step by step.\n\n**Step 1**");
        assert_eq!(calls, [("f".to_string(), "{}".to_string())]);
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
            let (said, _, calls) = however_cut("<｜Assistant｜><think>", &output);
            assert_eq!(said, content, "{between:?}");
            assert_eq!(calls, [("f".to_string(), "{}".to_string())], "{between:?}");
        }
    }

    #[test]
    fn the_blank_line_before_the_block_leaves_no_content_delta() {
        // The encoder's turn with a preamble in a tokenizer's pieces: the content deltas carry
        // the preamble and nothing after it, so the stream shows the content the whole has.
        let turn = ENCODER_TURNS
            .iter()
            .find(|turn| turn.tail == "<think>" && turn.content == "Let me check.")
            .expect("the shape");
        let (before, block) = turn
            .output
            .split_once("\n\n<｜DSML｜tool_calls>")
            .expect("the block");
        assert_eq!(before, "Say something first.</think>Let me check.");
        let mut pieces = vec![
            "Say something first.",
            "</think>",
            "Let",
            " me",
            " check",
            ".\n\n",
            "<｜DSML｜tool_calls>",
        ];
        pieces.extend(block.split_inclusive('\n'));
        assert_eq!(pieces.concat(), turn.output);
        let deltas = content_deltas("<｜Assistant｜><think>", &pieces);
        assert_eq!(deltas, ["Let", " me", " check", "."]);
        // The blank line alone in a delta, then the opener, at a chat turn's start: no content.
        let deltas = content_deltas("<｜Assistant｜></think>", &["\n\n", BARE_BLOCK]);
        assert_eq!(deltas, Vec::<String>::new());
    }

    #[test]
    fn the_turn_after_a_tool_result_reads_the_opener_as_the_turns_start_does() {
        // The template renders a tool result as a block inside a user message and opens the
        // model's turn after it, `<｜Assistant｜>` with the thought opened (thinking) or closed
        // (chat), a trailing system note after the result included; only a latest reminder after
        // such a note leaves the turn unopened; the blank line before a block is the opener's
        // whether or not text came before it, so no mode and no start shows it as content.
        let result = concat!(
            "<｜User｜>Weather in Paris?<｜Assistant｜><think>Need the weather.</think>\n\n",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>",
            "<｜User｜><tool_result>{\"temperature\": 22}</tool_result>"
        );
        let thinking = format!("{result}<｜Assistant｜><think>");
        let chat = format!("{result}<｜Assistant｜></think>");
        let unopened = format!("{result}# Environment\n - Platform: linux");
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
            (
                &unopened,
                format!("Check again.</think>Checking again.\n\n{BARE_BLOCK}"),
                ("Checking again.", "Check again."),
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
    }

    #[test]
    fn the_opener_with_its_blank_line_is_spelled_as_the_tags_write_it() {
        // The reading and the writing agree on what the template puts before the block.
        let spelled = format!("{}<｜DSML｜tool_calls>", Tags::DSML_V4.before_block);
        assert!(deepseek_v4().terminal_texts().any(|text| text == spelled));
    }

    /// The draw a fleet verifier saw once in several hundred answers to one prompt: an invoke
    /// that names the tool and writes both parameter tags without their `<`, so the syntax reads
    /// no pair (text where it has tags, then a closing tag out of its place). The engine's own
    /// parser hands such a call out as `{}` under `tool_calls`, and so did this table: every
    /// client read a well-formed call with no arguments, and nothing on the wire said the call had
    /// failed. Now the call is no whole call: the syntax writes no close for it, so its arguments
    /// are no JSON value, which a chat client's own reader refuses and the Responses item says
    /// (`incomplete`); the refused bytes are counted and never content; the Messages block keeps
    /// the `{}` a cut call's has, as the adapter's notes say.
    #[test]
    fn a_call_whose_every_parameter_the_syntax_refused_is_no_whole_call_to_any_client() {
        use crate::adapt;
        const OUTPUT: &str = concat!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"add_integers\">\n",
            "｜DSML｜parameter name=\"a\" string=\"false\">17</｜DSML｜parameter>\n",
            "｜DSML｜parameter name=\"b\" string=\"false\">25</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        const SLIPS: [&str; 2] = [
            "｜DSML｜parameter name=\"a\" string=\"false\">17",
            "｜DSML｜parameter name=\"b\" string=\"false\">25",
        ];
        // Chat mode, thinking off: the turn opens with the thought's close.
        let prompt = "<｜Assistant｜></think>";
        let (content, reasoning, calls) = however_cut(prompt, OUTPUT);
        assert_eq!(content, "");
        assert_eq!(reasoning, "");
        assert_eq!(calls, [("add_integers".to_string(), String::new())]);

        let events = run(prompt, &[OUTPUT]);
        let reasons: Vec<MalformedReason> = events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => {
                    assert!(text.text.is_empty(), "{event:?}");
                    Some(*why)
                }
                _ => None,
            })
            .collect();
        assert!(
            reasons.contains(&MalformedReason::TextBetweenTags)
                && reasons.contains(&MalformedReason::TagOutOfPlace),
            "{reasons:?}"
        );
        let refused: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::MalformedCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert!(
            SLIPS.iter().all(|slip| refused.contains(*slip)),
            "{refused:?}"
        );
        let arguments = arguments_of(&events, 0);
        assert!(
            serde_json::from_str::<serde_json::Value>(&arguments).is_err(),
            "{arguments:?} reads as a whole value"
        );

        // Chat Completions: the call, under `tool_calls`, with arguments a strict reader refuses.
        let choice = adapt::chat::message(0, &events);
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(choice.message.content, None);
        let calls = choice.message.tool_calls.as_deref().expect("a call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "add_integers");
        let arguments = calls[0].function.arguments.as_deref().expect("arguments");
        assert!(
            serde_json::from_str::<serde_json::Value>(arguments).is_err(),
            "{arguments:?} reads as a whole value"
        );

        // Responses: the item says the call is incomplete.
        let output = adapt::responses::output("resp", &events);
        let items = serde_json::to_value(&output.items).expect("wire");
        assert_eq!(items.as_array().map(Vec::len), Some(1), "{items}");
        assert_eq!(items[0]["type"], "function_call", "{items}");
        assert_eq!(items[0]["name"], "add_integers", "{items}");
        assert_eq!(items[0]["status"], "incomplete", "{items}");

        // Messages: the block carries `{}`, as a cut call's does; the call is not refused.
        let message = adapt::messages::output(&events);
        assert!(message.refused.is_empty(), "{:?}", message.refused);
        let blocks = serde_json::to_value(&message.content).expect("wire");
        assert_eq!(blocks[0]["type"], "tool_use", "{blocks}");
        assert_eq!(blocks[0]["input"], serde_json::json!({}), "{blocks}");
    }

    /// A call written without its block, as the serving engines' own parsers for the family read
    /// it. The draw a fleet verifier saw once in several hundred answers to one prompt: `:tool` in
    /// place of the block's opener and the invoke after it. Before the `content + invoke_open`
    /// row the table kept the invoke as content, its two tags counted as markers out of place,
    /// and a chat client read four lines of markup and no call; now the invoke is the call, whole
    /// and at every cut, every byte in its event and nothing malformed, and no content delta
    /// carries a byte of it.
    const BARE_INVOKE: &str = concat!(
        "<｜DSML｜invoke name=\"add_integers\">\n",
        "<｜DSML｜parameter name=\"a\" string=\"false\">6</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"b\" string=\"false\">8</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>"
    );

    /// The call of [`BARE_INVOKE`], as the table says it.
    fn the_call() -> (String, String) {
        (
            "add_integers".to_string(),
            r#"{"a": 6, "b": 8}"#.to_string(),
        )
    }

    /// `text` in the pieces a tokenizer tends to make of DSML: a piece ends at a newline, a quote
    /// or a tag's close.
    fn tokenized(text: &str) -> Vec<&str> {
        let pieces: Vec<&str> = text.split_inclusive(['\n', '"', '>']).collect();
        assert_eq!(pieces.concat(), text);
        pieces
    }

    #[test]
    fn an_invoke_written_outside_a_block_is_the_call_as_the_engines_own_parsers_read_it() {
        use crate::adapt;
        let prompt = "<｜Assistant｜></think>";
        let (content, reasoning, calls) = however_cut(prompt, BARE_INVOKE);
        assert_eq!(content, "");
        assert_eq!(reasoning, "");
        assert_eq!(calls, [the_call()]);
        let events = run(prompt, &[BARE_INVOKE]);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "{events:?}"
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        // What a chat client reads: the call under `tool_calls`, no content.
        let choice = adapt::chat::message(0, &events);
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(choice.message.content, None);
        assert_eq!(choice.message.tool_calls.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            content_deltas(prompt, &tokenized(BARE_INVOKE)),
            Vec::<String>::new()
        );
        // After a preamble, the blank line before a bare invoke is the text's: only the block's
        // opener owns one, as the engines' own parsers have it.
        let output = format!("Let me check.\n\n{BARE_INVOKE}");
        let (content, _, calls) = however_cut(prompt, &output);
        assert_eq!(content, "Let me check.\n\n");
        assert_eq!(calls, [the_call()]);
        // Two bare invokes are two calls, the newline between them the template's.
        let output = format!("{BARE_INVOKE}\n{BARE_INVOKE}");
        let (content, _, calls) = however_cut(prompt, &output);
        assert_eq!(content, "");
        assert_eq!(calls, [the_call(), the_call()]);
    }

    /// The recorded draw whole: the slip, the bare invoke, then the call written again inside a
    /// block. The engines' own servers deliver two calls for it, and so does the table, with the
    /// model's text alone as content where the first attempt streamed as 37 deltas of markup; the
    /// block's opener after the bare call, met in the wrapping between calls, is wrapping
    /// (`calls + calls_open`, with the blank line the template writes before it), as the engines'
    /// machines drop it.
    #[test]
    fn an_invoke_outside_the_block_and_the_block_after_it_are_two_calls_and_the_text_is_the_models()
    {
        let output = format!(
            "\n\n:tool\n{BARE_INVOKE}\n\n<｜DSML｜tool_calls>\n{BARE_INVOKE}\n</｜DSML｜tool_calls>"
        );
        let prompt = "<｜Assistant｜></think>";
        let (content, _, calls) = however_cut(prompt, &output);
        assert_eq!(content, "\n\n:tool\n");
        assert_eq!(calls, [the_call(), the_call()]);
        let events = run(prompt, &[&output]);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "{events:?}"
        );
        assert!(
            events.contains(&dropped("\n\n<｜DSML｜tool_calls>")),
            "{events:?}"
        );
        // Streamed in a tokenizer's pieces, the content deltas carry the slip and no markup.
        let deltas = content_deltas(prompt, &tokenized(&output));
        assert_eq!(deltas.concat(), "\n\n:tool\n");
        assert!(
            deltas.iter().all(|delta| !delta.contains("DSML")),
            "{deltas:?}"
        );
    }

    /// Prose after a call written without its block is text between calls: counted, and shown
    /// to a chat client as content, where the engines' own parsers drop everything after a call.
    /// An invoke inside the thought stays in the thought, as under the engines' machines, which
    /// start no call there.
    #[test]
    fn prose_after_a_bare_invoke_is_text_between_calls_and_one_inside_the_thought_stays_there() {
        use crate::adapt;
        let output = format!("{BARE_INVOKE}\nThe sum is 14.");
        let prompt = "<｜Assistant｜></think>";
        let (content, _, calls) = however_cut(prompt, &output);
        assert_eq!(content, "");
        assert_eq!(calls, [the_call()]);
        let events = run(prompt, &[&output]);
        let between: String = malformed(&events)
            .into_iter()
            .filter(|(_, why)| *why == MalformedReason::TextBetweenCalls)
            .map(|(text, _)| text)
            .collect();
        assert_eq!(between, "The sum is 14.");
        let choice = adapt::chat::message(0, &events);
        assert_eq!(choice.message.content.as_deref(), Some("The sum is 14."));
        assert_eq!(choice.message.tool_calls.as_ref().map(Vec::len), Some(1));

        let output = format!("Let me add.{BARE_INVOKE}</think>Done.");
        let (content, reasoning, calls) = however_cut("<｜Assistant｜><think>", &output);
        assert_eq!(content, "Done.");
        assert_eq!(reasoning, format!("Let me add.{BARE_INVOKE}"));
        assert!(calls.is_empty());
    }

    /// The rows from the content into an invoke change no forced-call tag: the grammar takes the
    /// block from the wrapping state's row alone, so a forced call is still written as the block,
    /// the block's opener with the template's blank line first, the invoke inside it, the block's
    /// close last.
    #[test]
    fn a_forced_call_is_still_written_as_the_block() {
        let payload = deepseek_v4()
            .grammar(&weather_tools(), true, false)
            .expect("a grammar")
            .payload();
        let elements = &payload["format"]["elements"];
        assert_eq!(
            elements[0],
            serde_json::json!({"type": "const_string", "value": "\n\n<｜DSML｜tool_calls>\n"}),
            "{payload}"
        );
        assert_eq!(
            elements[1]["tags"][0]["begin"], "<｜DSML｜invoke name=\"get_weather\">\n",
            "{payload}"
        );
        assert_eq!(
            elements[2],
            serde_json::json!({"type": "const_string", "value": "</｜DSML｜tool_calls>"}),
            "{payload}"
        );
    }
}
