//! Seed-OSS: the Qwen tagged syntax under its own markers. Reasoning between `<seed:think>` and
//! `</seed:think>`, the calls between `<seed:tool_call>` and `</seed:tool_call>`, each a
//! `<function=NAME>` block of `<parameter=KEY>` tags (one block per pair of markers as the
//! template writes it, several in one as the model writes a parallel answer), typed by the
//! request's tools (which reach the engine with the request), every value written with Python's
//! `str` where Qwen writes an object or a list as JSON and between the tags directly where Qwen
//! 3.5 puts it on a line of its own, everything else content. The same five rows as
//! [`qwen3()`](crate::formats::qwen3()), with the spellings ByteDance-Seed/Seed-OSS-36B-Instruct
//! writes; a plain `<tool_call>` is text here. The template is not ChatML: a turn opens with
//! `<seed:bos>assistant`, so the prompt's replay starts there.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::{Placement, Spelling},
};

/// The Seed-OSS table.
pub fn seed_oss() -> Format {
    Format::new("seed_oss")
        .terminal("think_open", "<seed:think>")
        .terminal("think_close", "</seed:think>")
        .terminal("call_open", "<seed:tool_call>")
        .terminal("call_close", "</seed:tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "calls")
        .transition("calls", "call_close", "content")
        .transition("calls", "call_open", "calls")
        .calls(CallSyntax::Tagged(Spelling::Python, Placement::Direct))
        .opens_turn("<seed:bos>assistant")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason, MalformedReason, Text},
        ids::CallIds,
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{content, drive, reasoning},
        Declared,
    };

    /// A call as the template writes it: the value between the parameter tags directly.
    const CALL: &str = concat!(
        "<seed:tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n",
        "</function>\n</seed:tool_call>"
    );

    #[test]
    fn the_prompt_is_replayed_from_seeds_own_turn_opener() {
        // A stray `<seed:think>` in the user's turn: the replay starts at `<seed:bos>assistant`,
        // so the answer is content (smg #2841, Alex's probe).
        let prompt = "<seed:bos>user\nWhy did you print <seed:think> there?<seed:eos>\
                      <seed:bos>assistant\n";
        let mut parser = Engine::new(seed_oss(), Declared::default());
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
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: "It opens a thought.",
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        let events = out.drain();
        assert_eq!(
            events[0],
            Event::Content(Text::uncounted("It opens a thought."))
        );
    }

    #[test]
    fn seeds_markers_are_read_and_qwens_are_text() {
        let output = format!("<seed:think>plan</seed:think>Sure.\n{CALL}<tool_call>x</tool_call>");
        let mut parser = Engine::new(seed_oss(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: &output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("end");
        let events = out.drain();
        assert_eq!(
            events[..3],
            [
                Event::Dropped {
                    text: Text::uncounted("<seed:think>"),
                    why: DropReason::Wrapper,
                },
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("plan")),
            ]
        );
        let arguments: String = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(arguments, r#"{"city": "Paris"}"#);
        assert_eq!(
            events
                .last()
                .map(|e| matches!(e, Event::Finish { tool_calls: 1, .. })),
            Some(true)
        );
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "Sure.\n<tool_call>x</tool_call>");
    }

    /// A `submit_patch` tool whose `patch` is a declared string, as bellwether's swebench call
    /// cases declare it.
    fn submit_patch() -> Declared {
        use openai_protocol::common::{Function, Tool};
        use serde_json::json as value;
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "submit_patch".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {"patch": {"type": "string"}}}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    #[test]
    fn a_value_keeps_the_newline_it_ends_with_since_the_template_writes_none_of_its_own() {
        // Seed-OSS writes `<parameter=KEY>`, the value and `</parameter>` with no newline between
        // them, so a newline the value ends with is the value's: bellwether's
        // seed-oss-36b-instruct/parse/swebench-*-call-* cases, whose patch ends in one, came back
        // a byte short (smg-lab #110). A value of spaces, of one newline and of nothing, the same,
        // whole and at every cut.
        for value in [
            "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n",
            "  two spaces  ",
            "\n",
            "\n\n",
            "",
        ] {
            let output = format!(
                "<seed:tool_call>\n<function=submit_patch>\n<parameter=patch>{value}</parameter>\n\
                 </function>\n</seed:tool_call>"
            );
            let expected = format!(
                "{{\"patch\": {}}}",
                serde_json::Value::String(value.to_string())
            );
            for cut in 0..output.len() {
                let pieces: Vec<&str> = if cut == 0 {
                    vec![&output]
                } else {
                    vec![&output[..cut], &output[cut..]]
                };
                let mut parser = Engine::new(seed_oss(), submit_patch());
                let mut out = Events::new();
                for piece in &pieces {
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
                let events = out.drain();
                let arguments: String = events
                    .iter()
                    .filter_map(|event| match event {
                        Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(arguments, expected, "{value:?} cut at {cut}");
                assert!(
                    matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                    "{value:?} cut at {cut}"
                );
            }
        }
    }

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }

    /// The tools of the recorded parallel answers below, typed as their requests declared them.
    fn recorded_tools() -> Declared {
        use openai_protocol::common::{Function, Tool};
        use serde_json::{json as value, Value};
        let tool = |name: &str, properties: Value| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": properties}),
                strict: None,
                extra: Default::default(),
            },
        };
        Declared::of(&[
            tool(
                "locate_tallest_mountains",
                value!({
                    "location": {"type": "string"},
                    "radius": {"type": "integer"},
                    "amount": {"type": "integer"},
                }),
            ),
            tool(
                "calculate_entropy_change",
                value!({
                    "initial_temp": {"type": "number"},
                    "final_temp": {"type": "number"},
                    "heat_capacity": {"type": "number"},
                    "isothermal": {"type": "boolean"},
                }),
            ),
            tool(
                "get_event_date",
                value!({"event": {"type": "string"}, "location": {"type": "string"}}),
            ),
            tool(
                "european_history_get_monarchs",
                value!({"country": {"type": "string"}, "century": {"type": "integer"}}),
            ),
            tool(
                "european_history_get_events",
                value!({
                    "country": {"type": "string"},
                    "century": {"type": "integer"},
                    "event_type": {"type": "string"},
                }),
            ),
        ])
    }

    /// What a client learns from an output: each call's index, id, name and arguments, the
    /// content, the reasoning, the malformed text by reason (adjacent runs of one reason joined)
    /// and the finish.
    #[derive(Debug, PartialEq)]
    struct Reading {
        calls: Vec<(u32, String, String, String)>,
        content: String,
        reasoning: String,
        malformed: Vec<(String, MalformedReason)>,
        finish: Option<(FinishReason, u32)>,
    }

    /// The reading of `pieces` fed one by one after an empty prompt and ended by a stop, with
    /// the call ids counting from one nonce so that two readings compare.
    fn read(pieces: &[&str]) -> Reading {
        let mut parser = Engine::with_call_ids(Arc::new(seed_oss()), recorded_tools(), ids());
        let events = drive(&mut parser, "", pieces);
        let mut calls: Vec<(u32, String, String, String)> = Vec::new();
        let mut malformed: Vec<(String, MalformedReason)> = Vec::new();
        let mut finish = None;
        for event in &events {
            match event {
                Event::ToolCallStart {
                    index, id, name, ..
                } => calls.push((*index, id.clone(), name.clone(), String::new())),
                Event::ToolCallArguments { json, .. } => {
                    calls.last_mut().expect("a call started").3.push_str(json);
                }
                // Runs of one reason joined, as the contract joins them: a cut inside a word
                // splits its report, not what is reported.
                Event::Malformed { text, why } => match malformed.last_mut() {
                    Some((joined, last)) if last == why => joined.push_str(&text.text),
                    _ => malformed.push((text.text.clone(), *why)),
                },
                Event::Finish {
                    reason, tool_calls, ..
                } => finish = Some((reason.clone(), *tool_calls)),
                _ => {}
            }
        }
        Reading {
            calls,
            content: content(&events),
            reasoning: reasoning(&events),
            malformed,
            finish,
        }
    }

    /// Every way of cutting `output` into deltas: whole, at every two-way split, and character
    /// by character.
    fn cuts(output: &str) -> Vec<Vec<&str>> {
        let mut cuts = vec![vec![output]];
        cuts.extend(
            output
                .char_indices()
                .skip(1)
                .map(|(at, _)| vec![&output[..at], &output[at..]]),
        );
        cuts.push(
            output
                .char_indices()
                .map(|(at, c)| &output[at..at + c.len_utf8()])
                .collect(),
        );
        cuts
    }

    fn call(index: u32, name: &str, arguments: &str) -> (u32, String, String, String) {
        (
            index,
            ids().id(index),
            name.to_string(),
            arguments.to_string(),
        )
    }

    #[test]
    fn several_function_blocks_in_one_region_are_as_many_calls_whole_and_at_every_cut() {
        // Recorded from parallel answers: the model writes its calls one after another between
        // one pair of `<seed:tool_call>` markers, where the template writes a pair per call, the
        // tags on one line or one per line. The engine's own parser returned every block as a
        // call where the table returned the first and showed the rest as text in the content.
        const THREE_ON_ONE_LINE: &str = concat!(
            "<seed:tool_call><function=locate_tallest_mountains><parameter=location>Tokyo",
            "</parameter><parameter=radius>200</parameter><parameter=amount>5</parameter>",
            "</function><function=calculate_entropy_change><parameter=initial_temp>300",
            "</parameter><parameter=final_temp>350</parameter><parameter=heat_capacity>1.5",
            "</parameter><parameter=isothermal>True</parameter></function>",
            "<function=get_event_date><parameter=event>Battle of Waterloo</parameter>",
            "<parameter=location>Belgium</parameter></function></seed:tool_call>"
        );
        const TWO_AFTER_A_THOUGHT: &str = concat!(
            "<seed:think>plan</seed:think>Sure.\n<seed:tool_call>\n",
            "<function=european_history_get_monarchs>\n<parameter=country>France</parameter>\n",
            "<parameter=century>17</parameter>\n</function>\n",
            "<function=european_history_get_events>\n<parameter=country>England</parameter>\n",
            "<parameter=century>18</parameter>\n<parameter=event_type>war</parameter>\n",
            "</function>\n</seed:tool_call>"
        );
        let expected = [
            (
                THREE_ON_ONE_LINE,
                Reading {
                    calls: vec![
                        call(
                            0,
                            "locate_tallest_mountains",
                            r#"{"location": "Tokyo", "radius": 200, "amount": 5}"#,
                        ),
                        call(
                            1,
                            "calculate_entropy_change",
                            r#"{"initial_temp": 300, "final_temp": 350, "heat_capacity": 1.5, "isothermal": true}"#,
                        ),
                        call(
                            2,
                            "get_event_date",
                            r#"{"event": "Battle of Waterloo", "location": "Belgium"}"#,
                        ),
                    ],
                    content: String::new(),
                    reasoning: String::new(),
                    malformed: vec![],
                    finish: Some((FinishReason::Stop, 3)),
                },
            ),
            (
                TWO_AFTER_A_THOUGHT,
                Reading {
                    calls: vec![
                        call(
                            0,
                            "european_history_get_monarchs",
                            r#"{"country": "France", "century": 17}"#,
                        ),
                        call(
                            1,
                            "european_history_get_events",
                            r#"{"country": "England", "century": 18, "event_type": "war"}"#,
                        ),
                    ],
                    content: "Sure.\n".to_string(),
                    reasoning: "plan".to_string(),
                    malformed: vec![],
                    finish: Some((FinishReason::Stop, 2)),
                },
            ),
        ];
        for (output, reading) in &expected {
            for pieces in cuts(output) {
                assert_eq!(&read(&pieces), reading, "{output:?} cut as {pieces:?}");
            }
        }
    }

    #[test]
    fn text_between_a_regions_blocks_is_reported_and_the_calls_around_it_are_read() {
        const BETWEEN: &str = concat!(
            "<seed:tool_call>\n<function=european_history_get_monarchs>\n",
            "<parameter=country>France</parameter>\n</function>\nthen\n",
            "<function=european_history_get_events>\n<parameter=country>England</parameter>\n",
            "</function>\n</seed:tool_call>"
        );
        let expected = Reading {
            calls: vec![
                call(
                    0,
                    "european_history_get_monarchs",
                    r#"{"country": "France"}"#,
                ),
                call(
                    1,
                    "european_history_get_events",
                    r#"{"country": "England"}"#,
                ),
            ],
            content: String::new(),
            reasoning: String::new(),
            malformed: vec![("then".to_string(), MalformedReason::TextBetweenCalls)],
            finish: Some((FinishReason::Stop, 2)),
        };
        for pieces in cuts(BETWEEN) {
            assert_eq!(read(&pieces), expected, "cut as {pieces:?}");
        }
    }

    #[test]
    fn a_region_of_two_blocks_and_a_region_of_one_number_their_calls_in_order() {
        const REGIONS: &str = concat!(
            "<seed:tool_call>\n<function=get_event_date>\n",
            "<parameter=event>Battle of Waterloo</parameter>\n</function>\n",
            "<function=get_event_date>\n<parameter=event>Battle of Hastings</parameter>\n",
            "</function>\n</seed:tool_call>\nAnd the mountains.\n<seed:tool_call>\n",
            "<function=locate_tallest_mountains>\n<parameter=location>Tokyo</parameter>\n",
            "</function>\n</seed:tool_call>"
        );
        let whole = read(&[REGIONS]);
        assert_eq!(
            whole.calls,
            vec![
                call(0, "get_event_date", r#"{"event": "Battle of Waterloo"}"#),
                call(1, "get_event_date", r#"{"event": "Battle of Hastings"}"#),
                call(2, "locate_tallest_mountains", r#"{"location": "Tokyo"}"#),
            ]
        );
        assert_eq!(whole.content.trim(), "And the mountains.");
        assert!(whole.malformed.is_empty(), "{:?}", whole.malformed);
        assert_eq!(whole.finish, Some((FinishReason::Stop, 3)));
        for pieces in cuts(REGIONS) {
            assert_eq!(read(&pieces), whole, "cut as {pieces:?}");
        }
    }
}
