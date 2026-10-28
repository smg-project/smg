//! DeepSeek V3.2: reasoning between `<think>` and `</think>`, the calls of one turn between
//! `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`, each call between
//! `<｜DSML｜invoke name="…">` and `</｜DSML｜invoke>`, everything else content: DSML as V4 spells
//! it ([`deepseek_v4`]), with `function_calls` for the block's name where V4 writes `tool_calls`
//! and V4.1 puts a space after the sentinel. As the reference encoder of
//! `deepseek-ai/DeepSeek-V3.2` (the repository's `encoding` module, which the checkpoint ships in
//! place of a chat template) renders a turn in thinking mode:
//!
//! ```text
//! <｜Assistant｜><think>Need the weather.</think>
//!
//! <｜DSML｜function_calls>
//! <｜DSML｜invoke name="get_weather">
//! <｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>
//! <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜function_calls><｜end▁of▁sentence｜>
//! ```
//!
//! The encoder ends the prompt with `<think>` in thinking mode and `</think>` in chat mode, after
//! the last user message and again after a tool result (`</function_results>`, inside the same
//! assistant turn). Those four tails are the table's turn anchors: the replay starts after the
//! last of them, in the state it names, so the turn's earlier block and a tool result's text are
//! not walked, and the turn is never unopened (the table has no unopened row: V4's would read the
//! `<｜end▁of▁sentence｜>` after a tool result as one and start the output in the thought).
//! DeepSeek-V3.2-Speciale ships the identical encoder; DeepSeek-V3.2-Exp, before
//! it, writes the V3.1 spelling. The gateway's older `deepseek32` parser reads this format, and
//! the alias takes a deployment that names it to this table. bellwether has no recorded group for
//! the family yet: the table stands on the turns the encoder renders, each read as the encoder's
//! own parser reads it in the tests below, and on the probes; the recorded set replays from the
//! fixture pin once bellwether has it.
//!
//! The block's opener is spelled as the encoder's parser spells it: the blank line the encoder
//! writes before the block and the tag, `\n\n<｜DSML｜function_calls>`, so the blank line leaves
//! with the tag wherever the block opens, after a text preamble, after the thought's close, at the
//! turn's start or after a tool result, and the content ends where the model's text does. The tag
//! alone opens the block too, for a block the model writes without the blank line.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
//!
//! [`deepseek_v4`]: super::deepseek_v4()

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed::Tags,
};

/// The DeepSeek V3.2 table.
pub fn deepseek_v3_2() -> Format {
    Format::new("deepseek_v3.2")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜DSML｜function_calls>")
        // The opener with the blank line the encoder writes before the block (the syntax's
        // `before_block`), the spelling its parser looks for: the scanner takes the longer marker
        // where the blank line stands, and holds a newline that may begin it, so a stream reads
        // as the whole does. Its rows come after the bare opener's, which the structural tags
        // derive the block's opener from.
        .terminal("separated_calls_open", "\n\n<｜DSML｜function_calls>")
        .terminal("calls_close", "</｜DSML｜function_calls>")
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
        .transition("invoke", "invoke_close", "calls")
        .transition("invoke", "invoke_open", "invoke")
        .transition("invoke", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Attributed(Tags::DSML_V4))
        // The four tails the encoder writes before the model's turn: the thought opened or closed
        // right after the turn's opener, or right after a tool result's block inside the same
        // assistant turn. The replay starts after the last of them, in the state it names, so the
        // turn's earlier block and the result's text are not walked (a marker quoted in a result
        // moves nothing), and the turn is never unopened: no unopened row, unlike V4's.
        .turn_anchor_starting_in("<｜Assistant｜><think>", "reasoning")
        .turn_anchor_starting_in("<｜Assistant｜></think>", "content")
        .turn_anchor_starting_in("</function_results>\n\n<think>", "reasoning")
        .turn_anchor_starting_in("</function_results>\n\n</think>", "content")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::{json as value, Value};

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
        formats::{deepseek_v4, deepseek_v4_1},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{self, bytes, content, drive_to, reasoning, the_same_at_every_cut},
        Declared,
    };

    /// The generation prompt's tail in thinking mode: the turn's opener and the thought opened.
    const THINKING: &str = "<｜User｜>Weather in Paris?<｜Assistant｜><think>";
    /// The generation prompt's tail in chat mode: the turn's opener and the thought closed at once.
    const CHAT: &str = "<｜User｜>Weather in Paris?<｜Assistant｜></think>";

    /// The turn the encoder renders for one call with a string and a number, after a thought.
    const OUTPUT: &str = concat!(
        "Need the weather.</think>\n\n<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>"
    );

    fn run_table(
        format: Format,
        prompt: &str,
        pieces: &[&str],
        finish: EngineFinish,
    ) -> Vec<Event<'static>> {
        let mut parser = Engine::new(format, Declared::default());
        drive_to(&mut parser, prompt, pieces, finish)
    }

    /// The calls as (name, arguments) pairs, the arguments as the JSON value their fragments make.
    fn calls(events: &[Event]) -> Vec<(String, Value)> {
        testing::calls(events)
            .into_iter()
            .map(|(name, arguments)| {
                let value = serde_json::from_str(&arguments)
                    .unwrap_or_else(|e| panic!("{name}: arguments {arguments:?}: {e}"));
                (name, value)
            })
            .collect()
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        run_table(deepseek_v3_2(), prompt, pieces, EngineFinish::Stop)
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

    fn dropped(text: &str) -> Event<'static> {
        Event::Dropped {
            text: Text::uncounted(text.to_owned()),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn the_encoders_turn_gives_its_call_with_its_values_and_every_byte() {
        let events = run(THINKING, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..5],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("Need the weather.")),
                dropped("</think>"),
                Event::ReasoningEnd,
                // The encoder's blank line before the block is the opener's: one marker.
                dropped("\n\n<｜DSML｜function_calls>"),
            ]
        );
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                value!({"location": "Paris", "days": 3})
            )]
        );
        assert!(events.contains(&dropped("</｜DSML｜function_calls>")));
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
    }

    #[test]
    fn every_cut_says_the_same_and_accounts_for_every_byte() {
        let whole = run(THINKING, &[OUTPUT]);
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run(THINKING, &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
            assert_eq!(content(&events), content(&whole), "cut at {cut}");
            assert_eq!(reasoning(&events), reasoning(&whole), "cut at {cut}");
            assert_eq!(finish(&events), (FinishReason::Stop, 1), "cut at {cut}");
        }
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_ends_the_call_with_its_arguments_cut() {
        let events = run_table(
            deepseek_v3_2(),
            CHAT,
            &["\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"days\" string=\"false\">1"],
            EngineFinish::Length,
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::UnterminatedRegion,
                ..
            }
        )));
        // The held value is the call's own, as written; nothing of it is content.
        assert_eq!(testing::arguments_of(&events, 0), r#"{"days": 1"#);
        assert_eq!(content(&events), "");
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Length,
                tool_calls: 1,
                ..
            })
        ));
    }

    #[test]
    fn a_calls_block_inside_the_thought_ends_it_and_is_the_call() {
        // The forced-call tag lets the engine write the block before `</think>` when the prompt
        // leaves the thought open: the block ends the thought, as under the V4 table.
        let output = concat!(
            "Let me call.<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"search\">\n",
            "<｜DSML｜parameter name=\"query\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜function_calls>"
        );
        let events = run(THINKING, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "Let me call.");
        assert_eq!(
            calls(&events),
            [("search".to_string(), value!({"query": "Paris"}))]
        );
        assert_eq!(content(&events), "");
    }

    #[test]
    fn the_other_dsml_spellings_are_content_here_and_this_one_under_them() {
        // The registry reads each checkpoint with its own table: under this one the V4 block
        // name and the V4.1 spaced tags are no markers, and under V4's and V4.1's tables this
        // block is content.
        let v4 = &OUTPUT.replace("function_calls", "tool_calls");
        let v4_1 = &v4
            .replace("<｜DSML｜", "<｜DSML｜ ")
            .replace("</｜DSML｜", "</｜DSML｜ ");
        for other in [v4, v4_1] {
            let events = run(THINKING, &[other]);
            assert!(calls(&events).is_empty(), "{other:?}");
            assert_eq!(
                content(&events),
                &other["Need the weather.</think>".len()..]
            );
            assert_eq!(finish(&events), (FinishReason::Stop, 0));
        }
        for other in [deepseek_v4(), deepseek_v4_1()] {
            let name = other.name().to_string();
            let events = run_table(other, THINKING, &[OUTPUT], EngineFinish::Stop);
            assert!(calls(&events).is_empty(), "{name}");
            assert_eq!(
                content(&events),
                &OUTPUT["Need the weather.</think>".len()..],
                "{name}"
            );
        }
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        let format = deepseek_v3_2();
        assert!(format.reasoning_open(THINKING));
        assert!(!format.reasoning_open(CHAT));
        // After a tool result the encoder writes `</function_results>` and opens the thought
        // again (or closes it, in chat mode) inside the same assistant turn: the replay starts
        // after that tail, the turn's earlier block and the result left unwalked, and the output
        // starts inside the thought.
        let after_result = concat!(
            "<｜User｜>Weather in Paris?<｜Assistant｜><think>Need the weather.</think>\n\n",
            "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜function_calls><｜end▁of▁sentence｜>\n\n",
            "<function_results>\n<result>{\"temperature\": 18}</result>\n</function_results>\n\n<think>"
        );
        assert!(format.reasoning_open(after_result));
        let events = run(after_result, &["Got it.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), "Got it.");
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        let chat_after_result = after_result.replace(
            "</function_results>\n\n<think>",
            "</function_results>\n\n</think>",
        );
        assert!(!format.reasoning_open(&chat_after_result));
        let events = run(&chat_after_result, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        // The blank line before a block after a tool result is the opener's: in chat mode with
        // nothing before it and after a preamble, in thinking mode after the thought's close and
        // after a preamble; the encoder's parser reads the content up to the blank line and the
        // tag, and so does the table, wherever the block opens.
        let rome = concat!(
            "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">Rome</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜function_calls>"
        );
        for (prompt, output, expected) in [
            (chat_after_result.as_str(), format!("\n\n{rome}"), ("", "")),
            (
                chat_after_result.as_str(),
                format!("It is 18 degrees; checking Rome.\n\n{rome}"),
                ("It is 18 degrees; checking Rome.", ""),
            ),
            (
                after_result,
                format!("Check Rome.</think>\n\n{rome}"),
                ("", "Check Rome."),
            ),
            (
                after_result,
                format!("Check Rome.</think>It is 18 degrees; checking Rome.\n\n{rome}"),
                ("It is 18 degrees; checking Rome.", "Check Rome."),
            ),
        ] {
            let events = run(prompt, &[&output]);
            assert_eq!(bytes(&events), output);
            assert_eq!(
                (content(&events).as_str(), reasoning(&events).as_str()),
                expected,
                "{output:?}"
            );
            assert_eq!(
                calls(&events),
                vec![("get_weather".to_string(), value!({"location": "Rome"}))],
                "{output:?}"
            );
        }
        // A marker quoted in a tool result moves nothing: the replay starts after the tail the
        // encoder writes behind the result, so the result's text is never walked.
        let quoted_result = after_result.replace(
            "<result>{\"temperature\": 18}</result>",
            "<result>The tag <｜DSML｜function_calls> opens the calls.</result>",
        );
        assert!(format.reasoning_open(&quoted_result));
        let events = run(&quoted_result, &["Got it.</think>It is 18 degrees."]);
        assert_eq!(reasoning(&events), "Got it.");
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // Nor does a result that quotes an invoke's opening, in chat mode: the output starts in
        // the content, as the tail says.
        let open_invoke = chat_after_result.replace(
            "<result>{\"temperature\": 18}</result>",
            "<result>Write <｜DSML｜invoke name=\" and then the name.</result>",
        );
        assert!(!format.reasoning_open(&open_invoke));
        let events = run(&open_invoke, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // A marker quoted in an earlier turn moves nothing.
        let quoted = "<｜User｜>Why did you write <｜DSML｜function_calls>?<｜Assistant｜><think>";
        let events = run(
            quoted,
            &["The user asks about the tag.</think>It opens the calls."],
        );
        assert_eq!(reasoning(&events), "The user asks about the tag.");
        assert_eq!(content(&events), "It opens the calls.");
    }

    #[test]
    fn a_closing_tag_inside_a_string_value_reads_as_the_encoders_parser_reads_it() {
        // DSML writes a string value raw, so a value that holds a parameter's closing tag and
        // another parameter's opening is, on the wire, byte for byte a call with two parameters:
        // the encoder's own parser splits at the first closing tag and reads two, and so does
        // this table. Neither can tell the value's bytes from the tags; the two agree.
        let meant =
            "Paris</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">1";
        let output = format!(
            "\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n\
             <｜DSML｜parameter name=\"location\" string=\"true\">{meant}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜function_calls>"
        );
        let events = run(CHAT, &[&output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                value!({"location": "Paris", "days": 1})
            )],
            "two parameters, not one holding {meant:?}"
        );
    }

    #[test]
    fn the_forced_call_tag_writes_this_tables_block() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}),
                strict: None,
                extra: Default::default(),
            },
        }];
        let grammar = deepseek_v3_2()
            .grammar(&tools, true, false)
            .expect("a tag for a required call");
        let payload = serde_json::to_string(&grammar.payload()).expect("the payload as JSON");
        assert!(payload.contains("<｜DSML｜function_calls>"), "{payload}");
        assert!(!payload.contains("<｜DSML｜tool_calls>"), "{payload}");
        let v4 = deepseek_v4()
            .grammar(&tools, true, false)
            .expect("the V4 tag");
        let v4_payload = serde_json::to_string(&v4.payload()).expect("the payload as JSON");
        assert_eq!(payload, v4_payload.replace("tool_calls", "function_calls"));
    }

    #[test]
    fn the_table_validates_and_anchors_the_turn_at_the_encoders_tails() {
        let format = deepseek_v3_2();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "deepseek_v3.2");
        assert_eq!(format.turn_opener(), None);
        assert_eq!(
            format.turn_texts(),
            [
                "<｜Assistant｜><think>",
                "<｜Assistant｜></think>",
                "</function_results>\n\n<think>",
                "</function_results>\n\n</think>",
            ]
        );
    }

    /// A block with one call and no arguments, as the encoder writes it.
    const BARE_BLOCK: &str = "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"f\">\n\n\
                              </｜DSML｜invoke>\n</｜DSML｜function_calls>";

    /// The content deltas a client of the stream receives when `pieces` arrive one by one after
    /// `prompt`: the content of each feed's events, the end's included, empty ones left out.
    fn content_deltas(prompt: &str, pieces: &[&str]) -> Vec<String> {
        let mut parser = Engine::new(deepseek_v3_2(), Declared::default());
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
    fn a_preamble_ends_where_the_encoders_blank_line_before_the_block_begins() {
        // The turns the encoder renders for a call after content, in thinking and in chat mode:
        // the content is the text alone, as the encoder's parser reads it, whole, a character at
        // a time and at every cut; the call reads as before.
        let block = concat!(
            "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">Paris</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n</｜DSML｜function_calls>"
        );
        for (prompt, output, thought) in [
            (
                THINKING,
                format!("Say something first.</think>Let me check.\n\n{block}"),
                "Say something first.",
            ),
            (CHAT, format!("Let me check.\n\n{block}"), ""),
        ] {
            let events = run(prompt, &[&output]);
            assert_eq!(bytes(&events), output);
            let said = the_same_at_every_cut(|pieces| run(prompt, pieces), &output);
            assert_eq!(said.content, "Let me check.", "{output:?}");
            assert_eq!(said.reasoning, thought, "{output:?}");
            assert_eq!(
                calls(&events),
                [("get_weather".to_string(), value!({"location": "Paris"}))],
                "{output:?}"
            );
        }
        // Exactly the blank line is the opener's, as the encoder's parser reads it: a third
        // newline or a space before it stays content; the bare tag opens a block written without
        // it, and the one newline before such a block is content.
        for (between, expected) in [
            ("\n\n", "Sure."),
            ("\n\n\n", "Sure.\n"),
            (" \n\n", "Sure. "),
            ("", "Sure."),
            ("\n", "Sure.\n"),
        ] {
            let output = format!("Sure.{between}{BARE_BLOCK}");
            let said = the_same_at_every_cut(|pieces| run(CHAT, pieces), &output);
            assert_eq!(said.content, expected, "{between:?}");
            assert_eq!(
                said.calls,
                [("f".to_string(), "{}".to_string())],
                "{between:?}"
            );
        }
    }

    #[test]
    fn the_blank_line_before_the_block_leaves_no_content_delta() {
        // Streamed in a tokenizer's pieces, the content deltas carry the preamble and nothing
        // after it, so the stream shows the content the whole has; the blank line alone in a
        // delta, then the opener, is no content at all.
        assert_eq!(
            content_deltas(CHAT, &["Let", " me", " check", ".\n\n", BARE_BLOCK]),
            ["Let", " me", " check", "."]
        );
        assert_eq!(
            content_deltas(CHAT, &["\n\n", BARE_BLOCK]),
            Vec::<String>::new()
        );
        assert_eq!(
            content_deltas(THINKING, &["Thinking.", "</think>", "\n\n", BARE_BLOCK]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn the_opener_with_its_blank_line_is_spelled_as_the_tags_write_it() {
        // The reading and the writing agree on what the encoder puts before the block.
        let spelled = format!("{}<｜DSML｜function_calls>", Tags::DSML_V4.before_block);
        assert!(deepseek_v3_2().terminal_texts().any(|text| text == spelled));
    }
}
