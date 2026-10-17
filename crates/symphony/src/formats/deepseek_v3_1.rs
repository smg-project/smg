//! DeepSeek V3.1: the thought between `<think>` and `</think>`, the prompt opening it (`<think>`)
//! or closing it at once (`</think>`) right after `<｜Assistant｜>`, and the calls as one block,
//! `<｜tool▁calls▁begin｜>` to `<｜tool▁calls▁end｜>`, holding one `<｜tool▁call▁begin｜>` per call:
//! the tool's name, `<｜tool▁sep｜>`, the arguments object alone and `<｜tool▁call▁end｜>`, with
//! nothing between the calls. Content comes before the block. As the checkpoint's template renders
//! it for `deepseek-ai/DeepSeek-V3.1`:
//!
//! ```text
//! Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
//! ```
//!
//! The templates of DeepSeek V3.1, V3.1-Terminus and V3.2-Exp write this spelling (the gateway's
//! older `deepseek31` parser's); DeepSeek V3's puts `function<｜tool▁sep｜>` before the name and a
//! `json` code fence around the arguments ([`deepseek_v3`]), and V3.2 and V4 write DSML
//! ([`deepseek_v4`]). Recorded as `deepseek-v3.1`, whose parse sets hold content alone: the
//! template takes a call's arguments as a string, so the recorder, whose corpus holds them as an
//! object, refused every call case; the calls stand on the turns the template renders, held in the
//! tests below.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
//!
//! [`deepseek_v3`]: super::deepseek_v3()
//! [`deepseek_v4`]: super::deepseek_v4()

use crate::format::{CallSyntax, Emits, Format};

/// The marker between a call's name and its arguments.
pub const SEP: &str = "<｜tool▁sep｜>";

/// The DeepSeek V3.1 table.
pub fn deepseek_v3_1() -> Format {
    Format::new("deepseek_v3.1")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜tool▁calls▁begin｜>")
        .terminal("calls_close", "<｜tool▁calls▁end｜>")
        .terminal("call_open", "<｜tool▁call▁begin｜>")
        .terminal("call_close", "<｜tool▁call▁end｜>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("calls", "call_open", "call")
        .transition("call", "call_close", "calls")
        .transition("call", "call_open", "call")
        .transition("call", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::NamedArguments(SEP))
        .opens_turn("<｜Assistant｜>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason},
        input::{EngineFinish, Input},
        parser::Parser,
        Declared,
    };

    /// The generation prompt's tail in chat mode: the turn's opener and the thought closed at once.
    const CHAT: &str = "<｜Assistant｜></think>";
    /// The generation prompt's tail in thinking mode: the turn's opener and the thought opened.
    const THINKING: &str = "<｜Assistant｜><think>";

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(deepseek_v3_1(), Declared::default());
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

    /// Every byte the events carry, in order: the output, when the engine accounts for all of it.
    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|event| match event {
                Event::Content(text) | Event::Reasoning(text) => text.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
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

    fn content(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The engine's finish: its reason as the engine gave it (the adapters refine it to
    /// `tool_calls`), and the calls counted.
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

    /// A turn as the checkpoint's chat template renders it (`chat_template` of
    /// `deepseek-ai/DeepSeek-V3.1`, the arguments given as the strings the template concatenates):
    /// the prompt's tail, the turn, and the message the template rendered.
    struct RenderedTurn {
        tail: &'static str,
        output: &'static str,
        reasoning: &'static str,
        content: &'static str,
        calls: &'static [(&'static str, &'static str)],
    }

    /// The turns the template renders for the shapes a tool-calling conversation takes: one call,
    /// three calls in one block, a call after content, a call with no arguments, nested and
    /// non-ASCII values, and content alone, in chat mode; and, in thinking mode, the same turns
    /// after the thought the model writes before `</think>` (the template renders no thought back
    /// into the history, so the thought's bytes are this test's).
    const RENDERED_TURNS: &[RenderedTurn] = &[
        RenderedTurn {
            tail: CHAT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: CHAT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Rome\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_stock_price<｜tool▁sep｜>{\"ticker\": \"MSFT\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[
                ("get_weather", "{\"city\": \"Paris\"}"),
                ("get_weather", "{\"city\": \"Rome\"}"),
                ("get_stock_price", "{\"ticker\": \"MSFT\"}"),
            ],
        },
        RenderedTurn {
            tail: CHAT,
            output: "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\", \"unit\": \"celsius\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "Let me check.",
            calls: &[("get_weather", "{\"city\": \"Paris\", \"unit\": \"celsius\"}")],
        },
        RenderedTurn {
            tail: CHAT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_stock_price<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("get_stock_price", "{}")],
        },
        RenderedTurn {
            tail: CHAT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>save_note<｜tool▁sep｜>{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[(
                "save_note",
                "{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}",
            )],
        },
        RenderedTurn {
            tail: CHAT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>save_note<｜tool▁sep｜>{\"title\": \"計画 🌍\", \"body\": \"café\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("save_note", "{\"title\": \"計画 🌍\", \"body\": \"café\"}")],
        },
        RenderedTurn {
            tail: CHAT,
            output: "It is sunny in Paris.",
            reasoning: "",
            content: "It is sunny in Paris.",
            calls: &[],
        },
        RenderedTurn {
            tail: THINKING,
            output: "The user asks about Paris.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "The user asks about Paris.",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: THINKING,
            output: "Say something first.</think>Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "Say something first.",
            content: "Let me check.",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: THINKING,
            output: "97 is not divisible by 2, 3, 5 or 7.</think>Yes, 97 is prime.",
            reasoning: "97 is not divisible by 2, 3, 5 or 7.",
            content: "Yes, 97 is prime.",
            calls: &[],
        },
        RenderedTurn {
            tail: THINKING,
            output: "</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_stock_price<｜tool▁sep｜>{\"ticker\": \"MSFT\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("get_stock_price", "{\"ticker\": \"MSFT\"}")],
        },
    ];

    fn check(turn: &RenderedTurn, events: &[Event], how: &str) {
        assert_eq!(bytes(events), turn.output, "{how}: {:?}", turn.output);
        assert_eq!(
            reasoning(events),
            turn.reasoning,
            "{how}: {:?}",
            turn.output
        );
        assert_eq!(content(events), turn.content, "{how}: {:?}", turn.output);
        let expected: Vec<(String, String)> = turn
            .calls
            .iter()
            .map(|(name, arguments)| ((*name).to_string(), (*arguments).to_string()))
            .collect();
        assert_eq!(calls(events), expected, "{how}: {:?}", turn.output);
        assert_eq!(
            malformed(events),
            Vec::<&str>::new(),
            "{how}: {:?}",
            turn.output
        );
        let count = u32::try_from(turn.calls.len()).expect("a few calls");
        assert_eq!(
            finish(events),
            (FinishReason::Stop, count),
            "{how}: {:?}",
            turn.output
        );
    }

    #[test]
    fn every_turn_the_template_renders_reads_back_as_the_message_it_rendered() {
        for turn in RENDERED_TURNS {
            let events = run(turn.tail, &[turn.output]);
            check(turn, &events, "whole");
        }
    }

    #[test]
    fn every_turn_reads_the_same_at_every_cut() {
        for turn in RENDERED_TURNS {
            for (at, _) in turn.output.char_indices().skip(1) {
                let (head, rest) = turn.output.split_at(at);
                let events = run(turn.tail, &[head, rest]);
                check(turn, &events, &format!("cut at {at}"));
            }
            let pieces: Vec<&str> = turn
                .output
                .char_indices()
                .map(|(at, c)| &turn.output[at..at + c.len_utf8()])
                .collect();
            let events = run(turn.tail, &pieces);
            check(turn, &events, "character by character");
        }
    }

    #[test]
    fn the_markers_are_wrapping_and_the_arguments_are_the_models_bytes() {
        let events = run(CHAT, &[RENDERED_TURNS[2].output]);
        for marker in [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ] {
            assert!(
                events.iter().any(|event| matches!(
                    event,
                    Event::Dropped { text, why: DropReason::Wrapper } if text.text == marker
                )),
                "{marker} dropped as wrapping"
            );
        }
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallStart { index: 0, name, .. } if name == "get_weather"
        )));
    }

    #[test]
    fn content_and_a_thought_without_calls_keep_the_other_families_markers_as_text() {
        // The two probes: the fenced `<tool_call>` block is no marker of this table, so it stays
        // content; `</think>` is, so a thought that quotes it ends there (the engine's rule for a
        // marker inside a channel, bellwether #16).
        let fence =
            "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                     {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.";
        let events = run(CHAT, &[fence]);
        assert_eq!(content(&events), fence);
        assert!(calls(&events).is_empty());
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        let events = run(
            THINKING,
            &["I could write </think> here but it is just text... no, I will answer.</think>Paris."],
        );
        assert_eq!(reasoning(&events), "I could write ");
        assert_eq!(
            content(&events),
            " here but it is just text... no, I will answer.</think>Paris."
        );
    }

    #[test]
    fn a_block_inside_the_thought_ends_it_and_is_the_calls() {
        // The template closes the thought before the block; a model that opens the block inside
        // its thought still gets its call read, as the V4 table reads it.
        let events = run(
            THINKING,
            &["Calling now<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"],
        );
        assert_eq!(reasoning(&events), "Calling now");
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\"}".to_string()
            )]
        );
        assert_eq!(content(&events), "");
    }

    #[test]
    fn a_stream_cut_inside_a_call_ends_it_with_what_arrived() {
        let mut parser = Engine::new(deepseek_v3_1(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: CHAT,
                },
                &mut out,
            )
            .expect("prompt");
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Par",
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
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events).0, FinishReason::Length);
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        let format = deepseek_v3_1();
        assert!(format.reasoning_open("<｜User｜>Hi<｜Assistant｜><think>"));
        assert!(!format.reasoning_open("<｜User｜>Hi<｜Assistant｜></think>"));
        // After a tool result the prompt ends in `<｜tool▁output▁end｜>` with no new opener: the
        // replay walks the previous turn's calls block from its `<｜Assistant｜>` and leaves the
        // output in content, where the template puts it.
        let prompt = "<｜User｜>Weather in Paris?<｜Assistant｜></think><｜tool▁calls▁begin｜>\
                      <｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Paris\"}\
                      <｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>\
                      <｜tool▁output▁begin｜>{\"temperature\": 18}<｜tool▁output▁end｜>";
        assert_eq!(
            format.replay(prompt),
            &prompt[prompt.find("<｜Assistant｜>").expect("the opener")..]
        );
        assert!(!format.reasoning_open(prompt));
        let events = run(prompt, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // A marker quoted in an earlier turn moves nothing.
        let quoted = "<｜User｜>What does <think> do?<｜Assistant｜></think>It opens a thought.\
                      <｜end▁of▁sentence｜><｜User｜>And <｜tool▁calls▁begin｜>?<｜Assistant｜></think>";
        assert!(!format.reasoning_open(quoted));
        let events = run(quoted, &["It opens the calls."]);
        assert_eq!(content(&events), "It opens the calls.");
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = deepseek_v3_1();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "deepseek_v3.1");
        assert_eq!(format.turn_opener(), Some("<｜Assistant｜>"));
    }
}
