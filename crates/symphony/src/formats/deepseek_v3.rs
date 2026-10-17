//! DeepSeek V3: the calls as one block, `<｜tool▁calls▁begin｜>` to `<｜tool▁calls▁end｜>`, holding
//! one call after another, each as `<｜tool▁call▁begin｜>function<｜tool▁sep｜>`, the tool's name,
//! a `json` code fence around the arguments object, and `<｜tool▁call▁end｜>`, the calls after the
//! first on a new line. Content comes before the block. As the checkpoint's template renders it
//! for `deepseek-ai/DeepSeek-V3-0324`:
//!
//! ```text
//! <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
//! ```json
//! {"city": "Paris"}
//! ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
//! ```
//!
//! The spelling is the gateway's older `deepseek` parser's; DeepSeek V3.1 drops the type and the
//! fence ([`deepseek_v3_1`]). V3's template writes no thought; the table reads one between
//! `<think>` and `</think>` all the same, since the checkpoints that share the call spelling and
//! write a thought (DeepSeek R1-0528, which the gateway's older pairing `deepseek` + `deepseek_r1`
//! served) read with it under the `deepseek` alias, and a thought must never land in the content.
//! Recorded as `deepseek-v3-0324`, whose parse sets hold content alone: the template takes a
//! call's arguments as a string, so the recorder, whose corpus holds them as an object, refused
//! every call case; the calls stand on the turns the template renders, held in the tests below.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
//!
//! [`deepseek_v3_1`]: super::deepseek_v3_1()

use crate::format::{CallSyntax, Emits, Format};

/// The text between a call's name and its arguments: the fence the template opens.
pub const FENCE_OPEN: &str = "\n```json\n";

/// The DeepSeek V3 table.
pub fn deepseek_v3() -> Format {
    Format::new("deepseek_v3")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜tool▁calls▁begin｜>")
        .terminal("calls_close", "<｜tool▁calls▁end｜>")
        .terminal("call_open", "<｜tool▁call▁begin｜>function<｜tool▁sep｜>")
        .terminal("call_close", "```<｜tool▁call▁end｜>")
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
        .calls(CallSyntax::NamedArguments(FENCE_OPEN))
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

    /// The generation prompt's tail: the turn's opener and nothing more.
    const PROMPT: &str = "<｜Assistant｜>";
    /// The tail of a checkpoint whose template opens the thought in the prompt (the original R1
    /// and its distills). R1-0528 ends its prompt at the opener and writes `<think>` itself.
    const THINKING: &str = "<｜Assistant｜><think>\n";

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(deepseek_v3(), Declared::default());
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
    /// `deepseek-ai/DeepSeek-V3-0324`, the arguments given as the strings the template
    /// concatenates): the prompt's tail, the turn, and the message the template rendered.
    struct RenderedTurn {
        tail: &'static str,
        output: &'static str,
        reasoning: &'static str,
        content: &'static str,
        calls: &'static [(&'static str, &'static str)],
    }

    /// The turns the template renders for the shapes a tool-calling conversation takes: one call,
    /// three calls in one block (a newline before each call after the first), a call after
    /// content, a call with no arguments, nested and non-ASCII values, and content alone; then a
    /// call and content after a thought the prompt opens (the original R1 and its distills), and
    /// after a thought the model opens itself, the prompt ending at the opener (R1-0528).
    const RENDERED_TURNS: &[RenderedTurn] = &[
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Rome\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_stock_price\n```json\n{\"ticker\": \"MSFT\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[
                ("get_weather", "{\"city\": \"Paris\"}"),
                ("get_weather", "{\"city\": \"Rome\"}"),
                ("get_stock_price", "{\"ticker\": \"MSFT\"}"),
            ],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\", \"unit\": \"celsius\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "Let me check.",
            calls: &[("get_weather", "{\"city\": \"Paris\", \"unit\": \"celsius\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_stock_price\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("get_stock_price", "{}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>save_note\n```json\n{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[(
                "save_note",
                "{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}",
            )],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>save_note\n```json\n{\"title\": \"計画 🌍\", \"body\": \"café\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "",
            content: "",
            calls: &[("save_note", "{\"title\": \"計画 🌍\", \"body\": \"café\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "It is sunny in Paris.",
            reasoning: "",
            content: "It is sunny in Paris.",
            calls: &[],
        },
        RenderedTurn {
            tail: THINKING,
            output: "The user asks about Paris.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "The user asks about Paris.",
            content: "",
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
            tail: PROMPT,
            output: "<think>\nThe user asks about Paris.\n</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            reasoning: "\nThe user asks about Paris.\n",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        RenderedTurn {
            tail: PROMPT,
            output: "<think>\n97 is not divisible by 2, 3, 5 or 7.\n</think>\n\nYes, 97 is prime.",
            reasoning: "\n97 is not divisible by 2, 3, 5 or 7.\n",
            content: "\n\nYes, 97 is prime.",
            calls: &[],
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
    fn the_markers_and_the_fence_are_wrapping_and_the_arguments_are_the_models_bytes() {
        let events = run(PROMPT, &[RENDERED_TURNS[1].output]);
        for marker in [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>",
            "```<｜tool▁call▁end｜>",
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
        // The newline before the fence's close and the one between two calls are the template's.
        assert!(
            events.iter().all(|event| !matches!(
                event,
                Event::Content(text) if text.text.trim().is_empty() && !text.text.is_empty()
            )),
            "no whitespace reaches the content between the calls"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallStart { index: 2, name, .. } if name == "get_stock_price"
        )));
    }

    #[test]
    fn content_keeps_the_fenced_block_and_a_thought_the_model_writes_is_read() {
        // The two probes: the fenced `<tool_call>` block is no marker of this table, so it stays
        // content; `<think>` and `</think>` are, though V3's template never writes them, so a
        // thought the model writes is reasoning and a `</think>` quoted inside it ends it (the
        // engine's rule for a marker inside a channel, bellwether #16).
        let fence =
            "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                     {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.";
        let events = run(PROMPT, &[fence]);
        assert_eq!(content(&events), fence);
        assert!(calls(&events).is_empty());
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        let events = run(
            PROMPT,
            &["<think>I could write </think> here.</think>Paris."],
        );
        assert_eq!(reasoning(&events), "I could write ");
        assert_eq!(content(&events), " here.</think>Paris.");
        // A thought the prompt opened and the model closed at once leaves the output in content,
        // as the older pairing of the `deepseek` and `deepseek_r1` parsers read it.
        let events = run(THINKING, &["</think>Paris is sunny."]);
        assert_eq!(reasoning(&events), "");
        assert_eq!(content(&events), "Paris is sunny.");
    }

    #[test]
    fn a_stream_cut_inside_a_call_ends_it_with_what_arrived() {
        let mut parser = Engine::new(deepseek_v3(), Declared::default());
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
                    text: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Par",
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
        let format = deepseek_v3();
        assert!(!format.reasoning_open("<｜User｜>Hi<｜Assistant｜>"));
        assert!(format.reasoning_open(THINKING));
        // After a tool result the prompt ends in `<｜tool▁outputs▁end｜>` with no new opener: the
        // replay walks the previous turn's calls block from its `<｜Assistant｜>` and leaves the
        // output in content, where the template puts it.
        let prompt = "<｜User｜>Weather in Paris?<｜Assistant｜><｜tool▁calls▁begin｜>\
                      <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n\
                      {\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\
                      <｜end▁of▁sentence｜><｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>\
                      {\"temperature\": 18}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>";
        assert_eq!(
            format.replay(prompt),
            &prompt[prompt.find("<｜Assistant｜>").expect("the opener")..]
        );
        let events = run(prompt, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // A marker quoted in an earlier turn moves nothing.
        let quoted =
            "<｜User｜>What does <｜tool▁calls▁begin｜> do?<｜Assistant｜>It opens the calls.\
                      <｜end▁of▁sentence｜><｜User｜>Thanks.<｜Assistant｜>";
        let events = run(quoted, &["You are welcome."]);
        assert_eq!(content(&events), "You are welcome.");
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = deepseek_v3();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "deepseek_v3");
        assert_eq!(format.turn_opener(), Some("<｜Assistant｜>"));
    }
}
