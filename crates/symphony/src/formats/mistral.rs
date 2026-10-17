//! Mistral, the list: the calls as one JSON list after `[TOOL_CALLS]`, each object the tool's
//! name and its arguments, the content before the marker; no thought. As Mistral 7B v0.3's chat
//! template renders it (a space after the marker and an `id` of nine characters after each call's
//! arguments, which the model may or may not write):
//!
//! ```text
//! [TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "call00000"}]
//! ```
//!
//! The spelling of the Mistral tokenizers before v11, the gateway's older `mistral` parser's:
//! Mistral 7B v0.3 writes it through its template, and Mistral Small 3 (2501), Small 3.1,
//! Devstral Small 2505 and Magistral Small 2506 through their own tokenizer, since their templates
//! render no tool call. The v11 and v13 tokenizers write a call's name and `[ARGS]` instead, which
//! the [`mistral_v13`](super::mistral_v13()) table reads; the two names keep the two spellings
//! apart. The list reads framed, as Llama Nemotron's does: text in place of it is malformed. The
//! generation prompt ends at `[/INST]`, whatever the request says, so the prompt's replay starts
//! there. Recorded as `mistral-small-24b-instruct-2501`, whose sets hold content alone; the calls
//! stand on the turns 7B v0.3's template renders, held in the tests below and in the rendered-turn
//! fixtures.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::format::{CallSyntax, Emits, Format};

/// The Mistral list table.
pub fn mistral() -> Format {
    Format::new("mistral")
        .terminal("calls_open", "[TOOL_CALLS]")
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("content", "calls_open", "calls")
        .transition("calls", "calls_open", "calls")
        .calls(CallSyntax::JsonList)
        .opens_turn("[/INST]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason, MalformedReason},
        input::{EngineFinish, Input},
        parser::Parser,
        Declared,
    };

    /// The generation prompt's tail: the user turn's close, and nothing more.
    const PROMPT: &str = "[/INST]";

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        run_until(prompt, pieces, EngineFinish::Stop)
    }

    fn run_until(prompt: &str, pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut parser = Engine::new(mistral(), Declared::default());
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
        parser.feed(Input::End { finish }, &mut out).expect("end");
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

    /// The malformed text, whole: the events that carry it depend on the cuts, the text does not.
    fn malformed(events: &[Event]) -> String {
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

    /// A turn as Mistral 7B v0.3's chat template renders it, and the message it rendered.
    struct Turn {
        output: &'static str,
        content: &'static str,
        calls: &'static [(&'static str, &'static str)],
    }

    /// The turns the template renders: one call, a call with no arguments, three calls, nested
    /// and non-ASCII values, content alone; and the spellings the model writes beyond the template,
    /// no space after the marker and no `id`.
    const TURNS: &[Turn] = &[
        Turn {
            output: "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}, \"id\": \"call00000\"}]",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        Turn {
            output: "[TOOL_CALLS] [{\"name\": \"get_stock_price\", \"arguments\": {}, \"id\": \"call00000\"}]",
            content: "",
            calls: &[("get_stock_price", "{}")],
        },
        Turn {
            output: "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}, \"id\": \"call00000\"}, {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}, \"id\": \"call00001\"}, {\"name\": \"get_stock_price\", \"arguments\": {\"ticker\": \"MSFT\"}, \"id\": \"call00002\"}]",
            content: "",
            calls: &[
                ("get_weather", "{\"city\": \"Paris\"}"),
                ("get_weather", "{\"city\": \"Rome\"}"),
                ("get_stock_price", "{\"ticker\": \"MSFT\"}"),
            ],
        },
        Turn {
            output: "[TOOL_CALLS] [{\"name\": \"save_note\", \"arguments\": {\"title\": \"計画 🌍\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a ] bracket\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}, \"id\": \"call00000\"}]",
            content: "",
            calls: &[(
                "save_note",
                "{\"title\": \"計画 🌍\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a ] bracket\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}",
            )],
        },
        Turn {
            output: "Hello!",
            content: "Hello!",
            calls: &[],
        },
        Turn {
            output: "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        Turn {
            output: "Let me check.[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"celsius\"}, \"id\": \"call00000\"}]",
            content: "Let me check.",
            calls: &[("get_weather", "{\"city\": \"Paris\", \"unit\": \"celsius\"}")],
        },
        Turn {
            output: "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.",
            content: "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.",
            calls: &[],
        },
    ];

    fn check(turn: &Turn, events: &[Event], how: &str) {
        assert_eq!(bytes(events), turn.output, "{how}: {:?}", turn.output);
        assert_eq!(content(events), turn.content, "{how}: {:?}", turn.output);
        let expected: Vec<(String, String)> = turn
            .calls
            .iter()
            .map(|(name, arguments)| ((*name).to_string(), (*arguments).to_string()))
            .collect();
        assert_eq!(calls(events), expected, "{how}: {:?}", turn.output);
        assert_eq!(malformed(events), "", "{how}: {:?}", turn.output);
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
        for turn in TURNS {
            let events = run(PROMPT, &[turn.output]);
            check(turn, &events, "whole");
        }
    }

    #[test]
    fn every_turn_reads_the_same_at_every_cut() {
        for turn in TURNS {
            for (at, _) in turn.output.char_indices().skip(1) {
                let (head, rest) = turn.output.split_at(at);
                let events = run(PROMPT, &[head, rest]);
                check(turn, &events, &format!("cut at {at}"));
            }
            let pieces: Vec<&str> = turn
                .output
                .char_indices()
                .map(|(at, c)| &turn.output[at..at + c.len_utf8()])
                .collect();
            let events = run(PROMPT, &pieces);
            check(turn, &events, "character by character");
        }
    }

    #[test]
    fn the_marker_the_brackets_and_the_id_are_the_templates_and_the_arguments_the_models_bytes() {
        let events = run(PROMPT, &[TURNS[2].output]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "[TOOL_CALLS]"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == " ["
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallStart { index: 2, name, .. } if name == "get_stock_price"
        )));
        // The `id` member is the object's: it is in no call's arguments, and its bytes are the
        // call's source, not content and not malformed.
        assert!(calls(&events)
            .iter()
            .all(|(_, arguments)| !arguments.contains("call0000")));
        assert_eq!(content(&events), "");
        assert_eq!(malformed(&events), "");
        // A second marker opens another list: two regions, three calls.
        let events = run(
            PROMPT,
            &["[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}][TOOL_CALLS] [{\"name\": \"get_stock_price\", \"arguments\": {}}, {\"name\": \"get_stock_price\", \"arguments\": {\"ticker\": \"MSFT\"}}]"],
        );
        assert_eq!(calls(&events).len(), 3);
        assert_eq!(finish(&events), (FinishReason::Stop, 3));
    }

    #[test]
    fn text_in_place_of_the_list_is_malformed_and_a_cut_stream_closes_nothing() {
        // The whitespace held before the text that is not a list is reported with it.
        let events = run(PROMPT, &["[TOOL_CALLS] not a list"]);
        assert!(calls(&events).is_empty());
        assert_eq!(malformed(&events), " not a list");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::TextInPlaceOfTheList,
                ..
            }
        )));
        // The marker quoted in prose is a marker, as every marker parser reads it; the prose
        // after it is text in place of the list.
        let events = run(
            PROMPT,
            &["The marker is [TOOL_CALLS]; the list follows it."],
        );
        assert_eq!(content(&events), "The marker is ");
        assert_eq!(malformed(&events), "; the list follows it.");
        let events = run_until(
            PROMPT,
            &["[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par"],
            EngineFinish::Length,
        );
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events).0, FinishReason::Length);
    }

    #[test]
    fn the_prompt_after_a_tool_result_leaves_the_output_in_content() {
        // After a tool result the prompt ends in `[/TOOL_RESULTS]`, and its last `[/INST]` stands
        // before the previous turn's list: the replay walks into the call state, which the prompt
        // never enters, so the output starts in content, where the template puts it.
        let prompt = "[INST] Weather in Paris?[/INST][TOOL_CALLS] [{\"name\": \"get_weather\", \
                      \"arguments\": {\"city\": \"Paris\"}, \"id\": \"call00000\"}]</s>\
                      [TOOL_RESULTS] {\"content\": {\"temperature\": 18}, \"call_id\": \"call00000\"}\
                      [/TOOL_RESULTS]";
        assert_eq!(
            mistral().replay(prompt),
            &prompt[prompt.find("[/INST]").expect("the user turn's close")..]
        );
        let events = run(prompt, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // A marker quoted in an earlier turn moves nothing.
        let quoted = "[INST] What does [TOOL_CALLS] do?[/INST] It opens the list.</s>\
                      [INST] Thanks.[/INST]";
        let events = run(quoted, &["You are welcome."]);
        assert_eq!(content(&events), "You are welcome.");
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = mistral();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "mistral");
        assert_eq!(format.turn_opener(), Some("[/INST]"));
        assert!(!format.starts_in_calls());
    }
}
