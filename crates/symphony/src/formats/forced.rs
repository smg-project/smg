//! The forced call: the whole output is the arguments of the function the request forces, as the
//! engine writes it under the JSON-schema constraint the gateway builds when `tool_choice` names a
//! function (the function's `parameters` as the schema), with no marker and no name in the output.
//! The one format known from the request rather than from the model, with the function's name in
//! [`Declared::forcing`](crate::Declared::forcing); no operator names it. A required call under
//! the same constraint is a list of `{"name": …, "parameters": …}` objects, which the
//! [`json`](super::json()) table reads. A model that thinks first writes its thought before the
//! JSON where the engine lets it, so the gateway reads such a request with the variant of the
//! model's own table ([`Format::with_forced_call`]): the thought as the table reads one, then
//! what this format or the list reads; of a table without a thought the variant is this format
//! with the table's markers dropped before the JSON.
//!
//! ```text
//! {"city": "Paris"}
//! ```
//!
//! Notes on this format, and on what it leaves open: `docs/tables.md`.

use openai_protocol::common::Tool;
use serde_json::Value;

use crate::format::{CallSyntax, Emits, Format};

/// The forced call's format.
pub fn forced() -> Format {
    Format::new("forced")
        .state("arguments", Emits::Arguments)
        .calls(CallSyntax::ForcedArguments)
}

/// The JSON schema the engine writes the forced call under, which this table then reads: the
/// `parameters` of the first of `tools` named `function`, as the gateway builds it where the
/// model's own table has no tag for the call. `None` when no tool of that name is declared.
pub fn arguments_schema(tools: &[Tool], function: &str) -> Option<Value> {
    tools
        .iter()
        .find(|tool| tool.function.name == function)
        .map(|tool| tool.function.parameters_schema().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events, FinishReason, MalformedReason},
        format::ForcedCall,
        formats::{qwen3, Table},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{calls, drive, finish, said, the_same_at_every_cut, Said},
        Declared,
    };

    fn run(pieces: &[&str], declared: Declared) -> Vec<Event<'static>> {
        let mut parser = Engine::new(forced(), declared);
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "<|im_start|>assistant\n",
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

    #[test]
    fn the_output_is_the_forced_functions_arguments_however_it_is_cut() {
        // What the engine writes for `tool_choice: {"type": "function", "function": {"name":
        // "get_weather"}}` under the gateway's schema: the arguments object alone.
        let arguments = "{\"city\": \"Paris\", \"days\": [1, 2.5, true, null]}";
        let declared = Declared::default().forcing("get_weather");
        for pieces in [
            vec![arguments],
            vec!["{\"city\": \"Pa", "ris\", \"days\": [1, 2.5, true, null]}"],
        ] {
            let events = run(&pieces, declared.clone());
            assert_eq!(
                calls(&events),
                [("get_weather".to_string(), arguments.to_string())]
            );
            assert_eq!(finish(&events), (FinishReason::Stop, 1));
            assert!(!events
                .iter()
                .any(|event| matches!(event, Event::Content(_) | Event::Reasoning(_))));
        }
    }

    #[test]
    fn a_cut_object_a_blank_output_and_an_unforced_request_say_what_they_are() {
        let declared = Declared::default().forcing("get_weather");
        let events = run(&["{\"city\": \"Par"], declared.clone());
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
        let events = run(&["  "], declared);
        assert!(calls(&events).is_empty());
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        // The gateway never assigns the format without a forced function; the output is then
        // reported, not read as content.
        let events = run(&["{\"city\": \"Paris\"}"], Declared::default());
        assert!(calls(&events).is_empty());
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::NoForcedFunction,
                ..
            }
        )));
    }

    #[test]
    fn the_format_validates_derives_no_grammar_and_has_no_opener() {
        let format = forced();
        format.validate().expect("a valid format");
        assert_eq!(format.name(), "forced");
        assert_eq!(format.turn_opener(), None);
        assert!(format.grammar(&[], true, false).is_none());
    }

    const ARGUMENTS: &str = "{\"q\": 1}";
    const LIST: &str = "[{\"name\": \"lookup\", \"parameters\": {\"q\": 1}}]";
    /// A prompt that opens the model's turn and leaves the thought to the model, and one whose
    /// template opened the thought.
    const TURN: &str = "<|im_start|>assistant\n";
    const THINKING: &str = "<|im_start|>assistant\n<think>\n";

    /// Qwen3's table as a request that forces `lookup` reads with it, the JSON written as
    /// `written` says, after `prompt`.
    fn forced_qwen3(
        written: ForcedCall,
        prompt: &'static str,
    ) -> impl Fn(&[&str]) -> Vec<Event<'static>> {
        move |pieces| {
            let format = qwen3(CallSyntax::Json).with_forced_call(written);
            let mut engine = Engine::new(format, Declared::default().forcing("lookup"));
            drive(&mut engine, prompt, pieces)
        }
    }

    fn one_call(reasoning: &str) -> Said {
        Said {
            calls: vec![("lookup".to_string(), ARGUMENTS.to_string())],
            finish: Some((FinishReason::Stop, 1)),
            content: String::new(),
            reasoning: reasoning.to_string(),
        }
    }

    #[test]
    fn a_thought_before_the_forced_arguments_is_the_thought_and_the_rest_is_the_call() {
        // An engine that holds the output to the schema once the thought is over writes the
        // thought, its close, and the arguments alone. The template opened the thought; the
        // model's own blank line after the close, where it writes one, is wrapping.
        for output in ["plan</think>{\"q\": 1}", "plan</think>\n\n{\"q\": 1}"] {
            let read = forced_qwen3(ForcedCall::Arguments, THINKING);
            assert_eq!(
                the_same_at_every_cut(read, output),
                one_call("plan"),
                "{output}"
            );
        }
        // The model opens the thought itself, as its first piece, behind whitespace or not.
        for output in [
            "<think>plan</think>{\"q\": 1}",
            "\n<think>plan</think>{\"q\": 1}",
        ] {
            let read = forced_qwen3(ForcedCall::Arguments, TURN);
            assert_eq!(
                the_same_at_every_cut(read, output),
                one_call("plan"),
                "{output}"
            );
        }
        // No thought: the output is the arguments from its first byte, as `forced` reads it.
        for output in [ARGUMENTS, "\n{\"q\": 1}"] {
            let read = forced_qwen3(ForcedCall::Arguments, TURN);
            assert_eq!(
                the_same_at_every_cut(read, output),
                one_call(""),
                "{output}"
            );
        }
        // A prompt that closed the thought itself leaves the output in the call.
        let read = forced_qwen3(
            ForcedCall::Arguments,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        );
        assert_eq!(the_same_at_every_cut(read, ARGUMENTS), one_call(""));
    }

    #[test]
    fn a_marker_inside_the_forced_json_is_its_text_and_moves_nothing() {
        // A string of the arguments may hold any of the table's markers: once the value began,
        // none of them has a row.
        let arguments = "{\"q\": \"a</think><think>b<tool_call>c</tool_call>\"}";
        let output = format!("plan</think>{arguments}");
        let read = forced_qwen3(ForcedCall::Arguments, THINKING);
        let said = the_same_at_every_cut(read, &output);
        assert_eq!(said.calls, [("lookup".to_string(), arguments.to_string())]);
        assert_eq!(
            (said.reasoning.as_str(), said.content.as_str()),
            ("plan", "")
        );
        // The same inside a call of the list.
        let list = "[{\"name\": \"lookup\", \"parameters\": {\"q\": \"a</think><think>b\"}}]";
        let output = format!("plan</think>{list}");
        let read = forced_qwen3(ForcedCall::List, THINKING);
        let said = the_same_at_every_cut(read, &output);
        assert_eq!(
            said.calls,
            [(
                "lookup".to_string(),
                "{\"q\": \"a</think><think>b\"}".to_string()
            )]
        );
        assert_eq!(
            (said.reasoning.as_str(), said.content.as_str()),
            ("plan", "")
        );
    }

    #[test]
    fn the_list_a_required_call_writes_is_read_after_the_thought() {
        let two = concat!(
            "[{\"name\": \"lookup\", \"parameters\": {\"q\": 1}}, ",
            "{\"name\": \"fetch\", \"parameters\": {}}]"
        );
        for (prompt, output, reasoning) in [
            (THINKING, format!("plan</think>{two}"), "plan"),
            (TURN, format!("<think>plan</think>\n\n{two}"), "plan"),
            (TURN, format!("\n<think>plan</think>{two}"), "plan"),
            (TURN, two.to_string(), ""),
        ] {
            let read = forced_qwen3(ForcedCall::List, prompt);
            let said = the_same_at_every_cut(read, &output);
            assert_eq!(
                said.calls,
                [
                    ("lookup".to_string(), ARGUMENTS.to_string()),
                    ("fetch".to_string(), "{}".to_string())
                ],
                "{output}"
            );
            assert_eq!(said.finish, Some((FinishReason::Stop, 2)), "{output}");
            assert_eq!(
                (said.reasoning.as_str(), said.content.as_str()),
                (reasoning, ""),
                "{output}"
            );
        }
    }

    #[test]
    fn a_thought_the_output_ends_inside_is_the_thought_and_no_call() {
        let read = forced_qwen3(ForcedCall::Arguments, THINKING);
        let said = said(&read(&["plan that never ", "ends"]));
        assert!(said.calls.is_empty());
        assert_eq!(said.finish, Some((FinishReason::Stop, 0)));
        assert_eq!(
            (said.reasoning.as_str(), said.content.as_str()),
            ("plan that never ends", "")
        );
    }

    #[test]
    fn a_request_that_keeps_the_thought_in_the_content_reads_it_there_before_the_call() {
        // The variant without reasoning, made of the forced one: the thought is content and its
        // markers, the rows into and out of the call here, stay dropped.
        let format = qwen3(CallSyntax::Json)
            .with_forced_call(ForcedCall::Arguments)
            .without_reasoning();
        for (prompt, output) in [
            (THINKING, "plan</think>{\"q\": 1}"),
            (TURN, "<think>plan</think>{\"q\": 1}"),
        ] {
            let mut engine = Engine::new(format.clone(), Declared::default().forcing("lookup"));
            let said = said(&drive(&mut engine, prompt, &[output]));
            assert_eq!(
                said,
                Said {
                    content: "plan".to_string(),
                    ..one_call("")
                },
                "{output}"
            );
        }
    }

    /// The thoughts a forced variant keeps, each with a way into it and a marker that closes it:
    /// the prompt and the output's first bytes that leave the engine inside the thought, and
    /// the text of a terminal with a row from the thought to the call.
    fn thoughts(format: &Format) -> Vec<(String, String, String)> {
        let thought = |state: usize| format.emits(state) == Emits::Reasoning;
        let close = |state: usize| {
            format
                .transitions()
                .find(|&(from, _, to)| from == state && to == 0)
                .map(|(_, on, _)| format.terminal_text(on).to_string())
        };
        let mut found = Vec::new();
        for (from, on, to) in format.transitions() {
            if from == 0 && thought(to) {
                if let Some(close) = close(to) {
                    found.push((String::new(), format.terminal_text(on).to_string(), close));
                }
            }
        }
        for anchor in format.turn_texts() {
            if format.turn_opener().is_some() {
                continue;
            }
            let state = format.prompt_state(anchor);
            if thought(state) {
                if let Some(close) = close(state) {
                    found.push((anchor.to_string(), String::new(), close));
                }
            }
        }
        found
    }

    #[test]
    fn every_table_reads_a_forced_call_after_each_way_into_its_thought() {
        // Every table of the registry, the call written either way: an engine runs the variant
        // (it validates), and where the table reads a thought, each way into one (a marker's
        // row from outside it, a turn anchor that starts the output inside it) reads the
        // thought and then the JSON as the call. A table with a start row would start a turn of
        // text inside the thought, which the variant cannot say: none has one.
        let mut with_a_thought = Vec::new();
        for table in Table::ALL {
            for (written, json) in [(ForcedCall::Arguments, ARGUMENTS), (ForcedCall::List, LIST)] {
                let format = table.format().with_forced_call(written);
                format
                    .validate()
                    .unwrap_or_else(|why| panic!("{}: {why}", table.name()));
                assert_eq!(format.name(), table.format().name());
                let reads_a_thought =
                    (0..format.states()).any(|state| format.emits(state) == Emits::Reasoning);
                if reads_a_thought {
                    assert_eq!(table.format().start(), None, "{}", table.name());
                }
                let thoughts = thoughts(&format);
                assert_eq!(
                    reads_a_thought,
                    !thoughts.is_empty(),
                    "{}: a thought with no way into it or out of it",
                    table.name()
                );
                for (prompt, opened, close) in thoughts {
                    let output = format!("{opened}plan{close}{json}");
                    let read = |pieces: &[&str]| {
                        let mut engine =
                            Engine::new(format.clone(), Declared::default().forcing("lookup"));
                        drive(&mut engine, &prompt, pieces)
                    };
                    assert_eq!(
                        the_same_at_every_cut(read, &output),
                        one_call("plan"),
                        "{} after {prompt:?}: {output}",
                        table.name()
                    );
                    if written == ForcedCall::Arguments {
                        with_a_thought.push(table.name());
                    }
                }
                // With no thought written, the output is the JSON from its first byte.
                let mut engine = Engine::new(format, Declared::default().forcing("lookup"));
                assert_eq!(
                    said(&drive(&mut engine, "", &[json])),
                    one_call(""),
                    "{}",
                    table.name()
                );
            }
        }
        with_a_thought.dedup();
        assert!(with_a_thought.len() >= 20, "{with_a_thought:?}");
    }
}
