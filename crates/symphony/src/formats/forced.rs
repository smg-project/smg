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
        event::{CallEnd, DropReason, Event, Events, FinishReason, MalformedReason},
        format::ForcedCall,
        formats::{qwen3, Table},
        grammar::{GrammarStart, Options},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{
            calls, drive, drive_to, endings, finish, malformed, said, the_same_at_every_cut,
            weather_tools, Said,
        },
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

    /// The forced call has no delimiter to close it: its value is the whole output, and the
    /// output's end is the call's own end where the value is whole there. A number or a literal
    /// the bytes end on is whole only then, and its call ends closed at the model's stop, as
    /// the object that closed at its brace does. At a cut a literal is whole all the same,
    /// since no byte could have followed it, and a number is not: `12` may be what the limit
    /// left of `123`, so its call is the one the limit cut, its bytes delivered as they are.
    /// A value the end found unfinished is the call the stop found open, or the one the limit
    /// cut.
    #[test]
    fn a_value_whole_where_the_output_ends_closes_its_call_and_an_unfinished_one_does_not() {
        let declared = Declared::default().forcing("get_weather");
        let read = |output: &str, finish: EngineFinish| {
            drive_to(
                &mut Engine::new(forced(), declared.clone()),
                "",
                &[output],
                finish,
            )
        };
        for finish in [EngineFinish::Stop, EngineFinish::Length] {
            for (whole, a_number) in [
                ("7", true),
                ("-12.5", true),
                ("0", true),
                ("true", false),
                ("null", false),
                ("{\"city\": \"Paris\"}", false),
                ("\"Paris\"", false),
            ] {
                let events = read(whole, finish.clone());
                assert_eq!(
                    calls(&events),
                    [("get_weather".to_string(), whole.to_string())],
                    "{whole}"
                );
                let cut_short = a_number && finish == EngineFinish::Length;
                assert_eq!(
                    endings(&events),
                    [(
                        0,
                        if cut_short {
                            CallEnd::Cut
                        } else {
                            CallEnd::Closed
                        }
                    )],
                    "{whole} under {finish:?}"
                );
                assert_eq!(malformed(&events), [], "{whole}");
            }
        }
        let reasons = |events: &[Event]| -> Vec<MalformedReason> {
            malformed(events).into_iter().map(|(_, why)| why).collect()
        };
        let unfinished = ["{\"city\": \"Par", "[1, 2", "tru", "\"Par"]; // codespell:ignore tru
        for unfinished in unfinished {
            let stopped = read(unfinished, EngineFinish::Stop);
            assert_eq!(endings(&stopped), [(0, CallEnd::Stop)], "{unfinished}");
            assert_eq!(
                reasons(&stopped),
                [MalformedReason::UnterminatedRegion],
                "{unfinished}"
            );
            let cut = read(unfinished, EngineFinish::Length);
            assert_eq!(endings(&cut), [(0, CallEnd::Cut)], "{unfinished}");
            assert_eq!(
                reasons(&cut),
                [MalformedReason::UnterminatedRegion],
                "{unfinished}"
            );
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
        // With tools to write, neither it nor a table's forced variant derives one: the
        // request's own schema is the constraint, for a named call and a required list alike.
        let tools = weather_tools();
        assert!(format.grammar(&tools, true, false).is_none());
        for written in [ForcedCall::Arguments, ForcedCall::List] {
            let variant = qwen3(CallSyntax::Json).with_forced_call(written);
            assert!(
                variant.grammar(&tools, true, false).is_none(),
                "{written:?}"
            );
        }
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

    /// The markers of `table`, each with whether the table ignores it: those a JSON string can
    /// hold as they are written.
    fn markers(table: Table) -> Vec<(String, bool)> {
        let format = table.format();
        format
            .terminal_texts()
            .enumerate()
            .map(|(index, text)| (text.to_string(), format.ignored(index)))
            .filter(|(text, _)| !text.contains(['"', '\\']) && !text.chars().any(char::is_control))
            .collect()
    }

    /// `arguments` as a forced call writes it: alone, or as the one call of a required list.
    fn written_as(written: ForcedCall, arguments: &str) -> String {
        match written {
            ForcedCall::Arguments => arguments.to_string(),
            ForcedCall::List => format!("[{{\"name\": \"lookup\", \"parameters\": {arguments}}}]"),
        }
    }

    #[test]
    fn every_marker_of_every_table_is_the_forced_jsons_text_inside_it() {
        // A string of the JSON may hold any marker of the model's table, one a row moves on or
        // one the table ignores: once the value began the marker is the value's text. Every
        // table of the registry, every marker, the call written either way, whole and cut
        // anywhere.
        let mut ignoring = Vec::new();
        for table in Table::ALL {
            for (marker, ignored) in markers(table) {
                for written in [ForcedCall::Arguments, ForcedCall::List] {
                    let arguments = format!("{{\"q\": \"a{marker}b\"}}");
                    let read = |pieces: &[&str]| {
                        let format = table.format().with_forced_call(written);
                        let mut engine = Engine::new(format, Declared::default().forcing("lookup"));
                        drive(&mut engine, "", pieces)
                    };
                    let said = the_same_at_every_cut(read, &written_as(written, &arguments));
                    let case = format!("{} {marker:?} {written:?}", table.name());
                    assert_eq!(said.calls, [("lookup".to_string(), arguments)], "{case}");
                    assert_eq!(
                        (said.reasoning.as_str(), said.content.as_str()),
                        ("", ""),
                        "{case}"
                    );
                }
                if ignored {
                    ignoring.push(table.name());
                }
            }
        }
        ignoring.dedup();
        assert!(ignoring.len() >= 3, "{ignoring:?}");
    }

    #[test]
    fn a_marker_the_table_ignores_is_wrapping_still_around_the_forced_json() {
        // Before the value's first byte and after its last, a marker the table ignores is the
        // template's wrapping, dropped where it stands as the table drops it everywhere.
        for table in Table::ALL {
            for (marker, _) in markers(table).into_iter().filter(|(_, ignored)| *ignored) {
                for written in [ForcedCall::Arguments, ForcedCall::List] {
                    let read = |pieces: &[&str]| {
                        let format = table.format().with_forced_call(written);
                        let mut engine = Engine::new(format, Declared::default().forcing("lookup"));
                        drive(&mut engine, "", pieces)
                    };
                    let output = format!("{marker}{}{marker}", written_as(written, ARGUMENTS));
                    let case = format!("{} {marker:?} {written:?}", table.name());
                    assert_eq!(the_same_at_every_cut(read, &output), one_call(""), "{case}");
                    let is_the_marker = |event: &&Event| matches!(event, Event::Dropped { text, .. } if text.text == marker);
                    let dropped = read(&[&output]).iter().filter(is_the_marker).count();
                    assert_eq!(dropped, 2, "{case}");
                }
            }
        }
    }

    #[test]
    fn the_beginning_of_a_marker_the_end_cut_short_inside_the_forced_json_is_its_text() {
        // The output ends inside a string of the JSON, on bytes the engine would take for a tag
        // the end cut short where a state reads text (`Format::is_cut_tag`): the beginning of a
        // marker, itself beginning with a whole structural token. Inside the value they are the
        // value's bytes, as the whole marker's would be, at a cut and at the model's own stop.
        // Every table whose variant has such a tail, the call written either way.
        let mut tables = Vec::new();
        for table in Table::ALL {
            for written in [ForcedCall::Arguments, ForcedCall::List] {
                let format = table.format().with_forced_call(written);
                let beginnings = markers(table).into_iter().flat_map(|(marker, _)| {
                    let ends = marker.char_indices().skip(1).map(|(at, _)| at);
                    ends.map(|at| marker[..at].to_string()).collect::<Vec<_>>()
                });
                for tail in beginnings.filter(|tail| format.is_cut_tag(0, tail, false)) {
                    let arguments = format!("{{\"q\": \"a{tail}");
                    let output = match written {
                        ForcedCall::Arguments => arguments.clone(),
                        ForcedCall::List => {
                            format!("[{{\"name\": \"lookup\", \"parameters\": {arguments}")
                        }
                    };
                    for finish in [EngineFinish::Length, EngineFinish::Stop] {
                        let read = |pieces: &[&str]| {
                            let declared = Declared::default().forcing("lookup");
                            let mut engine = Engine::new(format.clone(), declared);
                            drive_to(&mut engine, "", pieces, finish.clone())
                        };
                        let said = the_same_at_every_cut(read, &output);
                        let case = format!("{} {tail:?} {written:?} {finish:?}", table.name());
                        assert_eq!(
                            said.calls,
                            [("lookup".to_string(), arguments.clone())],
                            "{case}"
                        );
                    }
                    tables.push(table.name());
                }
            }
        }
        tables.dedup();
        assert!(!tables.is_empty(), "no table declares a structural token");
    }

    #[test]
    fn a_required_list_under_the_json_answer_option_reads_as_a_json_syntax() {
        // A request may ask for a JSON answer and require a call: the required list is then
        // read under the JSON-answer option, as a list with no marker is. A marker spelled
        // inside a string of a call's arguments is the argument's text, with the option as
        // without it. Where the model wrote text in the list's place, a marker the table
        // ignores inside a string of that text is the string's under the option; without the
        // option the table drops it. Every table, whole and cut anywhere.
        let mut ignoring = Vec::new();
        for table in Table::ALL {
            let read = |json_answer: bool| {
                move |pieces: &[&str]| {
                    let format = table.format().with_forced_call(ForcedCall::List);
                    let engine = Engine::new(format, Declared::default().forcing("lookup"));
                    let mut engine = if json_answer {
                        engine.reads_json_answer()
                    } else {
                        engine
                    };
                    drive(&mut engine, "", pieces)
                }
            };
            for (marker, ignored) in markers(table) {
                let case = format!("{} {marker:?}", table.name());
                let arguments = format!("{{\"q\": \"a{marker}b\"}}");
                let list = written_as(ForcedCall::List, &arguments);
                let said = the_same_at_every_cut(read(true), &list);
                assert_eq!(said, the_same_at_every_cut(read(false), &list), "{case}");
                assert_eq!(said.calls, [("lookup".to_string(), arguments)], "{case}");
                if ignored {
                    let text = format!("say \"a{marker}b\" then");
                    let said = the_same_at_every_cut(read(true), &text);
                    assert_eq!(said.content, text, "{case}");
                    assert!(said.calls.is_empty(), "{case}");
                    let said = the_same_at_every_cut(read(false), &text);
                    assert_eq!(said.content, "say \"ab\" then", "{case}");
                    ignoring.push(table.name());
                }
            }
        }
        ignoring.dedup();
        assert!(ignoring.len() >= 3, "{ignoring:?}");
    }

    /// A table built by hand, as the builder lets anyone build one: content, and a thought
    /// between `<a>` and `</a>`.
    fn built() -> Format {
        Format::new("built")
            .terminal("open", "<a>")
            .terminal("close", "</a>")
            .state("content", Emits::Content)
            .state("inside", Emits::Reasoning)
            .transition("content", "open", "inside")
            .transition("inside", "close", "content")
    }

    /// What the forced variant of `table`, which must validate, says of `output` after `prompt`.
    fn variant_says(table: Format, prompt: &str, output: &str) -> Said {
        let format = table.with_forced_call(ForcedCall::Arguments);
        format.validate().expect("the variant validates");
        let mut engine = Engine::new(format, Declared::default().forcing("lookup"));
        said(&drive(&mut engine, prompt, &[output]))
    }

    #[test]
    fn a_start_row_goes_from_the_variant() {
        // The variant's first state holds the call, where no start row may stand: the row goes,
        // so the output is the JSON from its first byte, and a thought is read where its marker
        // opens it.
        let table = built().starts_in("inside");
        table.validate().expect("the table validates");
        assert_eq!(variant_says(table.clone(), "", ARGUMENTS), one_call(""));
        assert_eq!(
            variant_says(table, "", "<a>plan</a>{\"q\": 1}"),
            one_call("plan")
        );
    }

    #[test]
    fn an_unopened_row_stays_on_a_thought_and_goes_where_it_would_name_the_call() {
        // A prompt that opens no turn starts the output where the unopened row says. A row that
        // names a thought keeps its state in the variant. One that names any other state would
        // name the call, where no unopened turn may start, and goes: the output starts in the
        // call, as it does without the row.
        let unopened = "User: hi";
        let on_the_thought = built()
            .opens_turn("<|assistant|>")
            .unopened_turn_starts_in("inside", "<|end|>");
        assert_eq!(
            variant_says(on_the_thought, unopened, "plan</a>{\"q\": 1}"),
            one_call("plan")
        );
        let on_content = built()
            .opens_turn("<|assistant|>")
            .unopened_turn_starts_in("content", "<|end|>");
        on_content.validate().expect("the table validates");
        assert_eq!(variant_says(on_content, unopened, ARGUMENTS), one_call(""));
    }

    #[test]
    fn the_row_listed_first_stands_where_two_states_now_one_read_a_marker_differently() {
        // Content and the answer are one state in the variant, and each reads `<a>` its own way:
        // content opens the thought on it, the answer returns to content. The variant reads the
        // marker by the row the table lists first.
        let table = |rows: [(&str, &str, &str); 2]| {
            let format = Format::new("built")
                .terminal("open", "<a>")
                .terminal("close", "</a>")
                .terminal("answer", "<b>")
                .state("content", Emits::Content)
                .state("inside", Emits::Reasoning)
                .state("answer", Emits::Content)
                .transition("inside", "close", "content")
                .transition("content", "answer", "answer");
            rows.into_iter().fold(format, |format, (from, on, to)| {
                format.transition(from, on, to)
            })
        };
        let opens = ("content", "open", "inside");
        let returns = ("answer", "open", "content");
        // The thought's row first: the marker opens the thought.
        assert_eq!(
            variant_says(table([opens, returns]), "", "<a>plan</a>{\"q\": 1}"),
            one_call("plan")
        );
        // The answer's row first: the marker is wrapping before the JSON and opens no thought.
        assert_eq!(
            variant_says(table([returns, opens]), "", "<a>{\"q\": 1}"),
            one_call("")
        );
    }

    #[test]
    fn an_anchor_names_the_state_its_own_became_in_the_variant() {
        // The thought is the table's third state and the variant's second, and the answer is
        // the arguments state there: an anchor follows the state it named.
        let table = Format::new("built")
            .terminal("open", "<a>")
            .terminal("close", "</a>")
            .state("content", Emits::Content)
            .state("answer", Emits::Content)
            .state("inside", Emits::Reasoning)
            .transition("content", "open", "inside")
            .transition("inside", "close", "content")
            .turn_anchor_starting_in("<thinks>", "inside")
            .turn_anchor_starting_in("<answers>", "answer");
        table.validate().expect("the table validates");
        assert_eq!(
            variant_says(table.clone(), "<thinks>", "plan</a>{\"q\": 1}"),
            one_call("plan")
        );
        assert_eq!(variant_says(table, "<answers>", ARGUMENTS), one_call(""));
    }

    #[test]
    fn the_variant_has_no_layout() {
        // A layout says how the template writes the table's turn, which a tag is written from.
        // The variant's turn is not the table's, and no tag is derived from it.
        let table = built().with_layout(crate::shape::Layout::default());
        assert!(table.layout().is_some());
        for written in [ForcedCall::Arguments, ForcedCall::List] {
            assert!(table.clone().with_forced_call(written).layout().is_none());
        }
    }

    #[test]
    fn a_marker_led_back_into_the_call_is_wrapping_before_the_forced_json() {
        // Before the value's first byte a marker whose row leads from the arguments state back
        // into it moves nothing and is dropped: an answer's opener, a thought's closer the
        // table reads where no thought is open, by a row of its own or one the parser alone
        // read in the table. Every table, every such marker, the call written either way.
        let mut tables = Vec::new();
        for table in Table::ALL {
            for written in [ForcedCall::Arguments, ForcedCall::List] {
                let format = table.format().with_forced_call(written);
                let texts: Vec<String> = format.terminal_texts().map(str::to_string).collect();
                for (index, marker) in texts.iter().enumerate() {
                    if format.ignored(index) || format.next(0, index) != Some(0) {
                        continue;
                    }
                    let read = |pieces: &[&str]| {
                        let declared = Declared::default().forcing("lookup");
                        let mut engine = Engine::new(format.clone(), declared);
                        drive(&mut engine, "", pieces)
                    };
                    let output = format!("{marker}{}", written_as(written, ARGUMENTS));
                    let case = format!("{} {marker:?} {written:?}", table.name());
                    assert_eq!(the_same_at_every_cut(read, &output), one_call(""), "{case}");
                    tables.push(table.name());
                }
            }
        }
        tables.dedup();
        assert!(tables.contains(&"minimax_m3"), "{tables:?}");
    }

    #[test]
    fn bytes_no_function_names_are_reported_with_every_marker_after_their_first() {
        // With no function forced the bytes are reported, not read. A thought before them is
        // still the thought, and from their first byte a marker is among them, as it is inside
        // a named call's value.
        let read = |output: &str| {
            let format = qwen3(CallSyntax::Json).with_forced_call(ForcedCall::Arguments);
            let mut engine = Engine::new(format, Declared::default());
            let events = drive(&mut engine, TURN, &[output]);
            let reported: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Malformed {
                        text,
                        why: MalformedReason::NoForcedFunction,
                    } => Some(&*text.text),
                    _ => None,
                })
                .collect();
            (reported, said(&events))
        };
        let (reported, said) = read("<think>plan</think>{\"a\": 1}<think>b</think>");
        assert_eq!(reported, "{\"a\": 1}<think>b</think>");
        assert_eq!((said.reasoning.as_str(), said.calls.len()), ("plan", 0));
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

    /// A required call's list the end of the stream leaves open is no prose: the engine held
    /// the output to the list. What the list still held (the separators after a call, the next
    /// call's object the end cut before its arguments began, a started call's object behind
    /// its arguments) is refused: counted once a run and never content, behind a whole call or
    /// with none before it, however the bytes were cut. A call whose arguments began keeps the
    /// argument bytes that arrived.
    #[test]
    fn a_required_list_the_end_leaves_open_says_its_calls_and_no_content() {
        let whole = "[{\"name\": \"lookup\", \"parameters\": {\"q\": 1}}";
        let lookup = ("lookup".to_string(), ARGUMENTS.to_string());
        for (output, expected, refused) in [
            (format!("{whole},"), vec![lookup.clone()], vec![","]),
            (format!("{whole} ,\n"), vec![lookup.clone()], vec![" ,\n"]),
            (
                format!("{whole}, {{\"name\": \"fet"),
                vec![lookup.clone()],
                vec!["{\"name\": \"fet"],
            ),
            // The next call's name is whole and its arguments never began: no call started.
            (
                format!("{whole}, {{\"name\": \"fetch\", \"parameters\":"),
                vec![lookup.clone()],
                vec!["{\"name\": \"fetch\", \"parameters\":"],
            ),
            (
                "[{\"name\": \"loo".to_string(),
                vec![],
                vec!["{\"name\": \"loo"],
            ),
            (
                format!("{whole}, {{\"name\": \"fetch\", \"parameters\": {{\"u"),
                vec![lookup.clone(), ("fetch".to_string(), "{\"u".to_string())],
                vec![],
            ),
            // A started call the end cut before its item's closing brace keeps its arguments,
            // and what its object held behind them is refused: the newline and the indent an
            // indented list writes there, or the name written after the parameters.
            (
                "[\n  {\"name\": \"lookup\", \"parameters\": {\"q\": 1}\n  ".to_string(),
                vec![lookup.clone()],
                vec!["\n  "],
            ),
            (
                "[{\"parameters\": {\"q\": 1}, \"name\": \"lookup\"".to_string(),
                vec![lookup.clone()],
                vec![", \"name\": \"lookup\""],
            ),
        ] {
            let read = forced_qwen3(ForcedCall::List, TURN);
            // Every read of the output holds what follows: whole, one character at a time and
            // at every two-way cut.
            let said = the_same_at_every_cut(
                |pieces| {
                    let events = read(pieces);
                    // What a chat client reads as content: the content and the malformed text.
                    assert_eq!(crate::tagged::shown(&events), "", "{output}: {pieces:?}");
                    assert_eq!(
                        crate::tagged::refused(&events),
                        refused,
                        "{output}: {pieces:?}"
                    );
                    // Each refused run is counted once: the event of no bytes stands right
                    // before it.
                    let counted = events
                        .windows(2)
                        .filter(|pair| {
                            matches!(
                                pair,
                                [
                                    Event::Malformed {
                                        text,
                                        why: MalformedReason::UnterminatedRegion,
                                    },
                                    Event::Dropped {
                                        why: DropReason::MalformedCall,
                                        ..
                                    },
                                ] if text.text.is_empty()
                            )
                        })
                        .count();
                    assert_eq!(counted, refused.len(), "{output}: {pieces:?}: {events:?}");
                    events
                },
                &output,
            );
            assert_eq!(said.calls, expected, "{output}");
        }
        // The bracket alone, with no object after it, is what the list with no marker says of
        // it: the model's text, as before.
        let events = forced_qwen3(ForcedCall::List, TURN)(&["["]);
        assert_eq!(crate::tagged::shown(&events), "[");
        assert!(crate::tagged::refused(&events).is_empty(), "{events:?}");
        // One call written as an object alone, with no bracket, is a call too, as the list
        // with no marker reads it.
        let alone = "{\"name\": \"lookup\", \"parameters\": {\"q\": 1}}";
        let said = the_same_at_every_cut(forced_qwen3(ForcedCall::List, TURN), alone);
        assert_eq!(said.calls, vec![lookup]);
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
    /// the text of a terminal with a row from the thought to the call, once for every such row
    /// (a recovery row the table lists beside the thought's own close is one).
    fn thoughts(format: &Format) -> Vec<(String, String, String)> {
        let thought = |state: usize| format.emits(state) == Emits::Reasoning;
        let closes = |state: usize| -> Vec<String> {
            format
                .transitions()
                .filter(|&(from, _, to)| from == state && to == 0)
                .map(|(_, on, _)| format.terminal_text(on).to_string())
                .collect()
        };
        let mut found = Vec::new();
        for (from, on, to) in format.transitions() {
            if from == 0 && thought(to) {
                for close in closes(to) {
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
                for close in closes(state) {
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
                    // A prompt that wrote the thought and closed it leaves the output in the
                    // call, whichever way the thought was entered: the JSON is the call from
                    // its first byte.
                    let closed = format!(
                        "{}{prompt}{opened}plan{close}",
                        format.turn_opener().unwrap_or_default()
                    );
                    let mut engine =
                        Engine::new(format.clone(), Declared::default().forcing("lookup"));
                    assert_eq!(
                        said(&drive(&mut engine, &closed, &[json])),
                        one_call(""),
                        "{} after {closed:?}",
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

    /// A table's recovery rows are rows of its variant too: Kimi K3 reads a thought the model
    /// closed with the answer's close tag, and the variant reads the JSON after it as the call,
    /// as after the thought's own close. The sweep above reads the variant's own rows, so a
    /// variant made without them would pass it: this case names the row.
    #[test]
    fn a_thought_a_recovery_row_closes_is_the_thought_and_then_the_call() {
        for (written, json) in [(ForcedCall::Arguments, ARGUMENTS), (ForcedCall::List, LIST)] {
            for close in ["<|close|>think<|sep|>", "<|close|>response<|sep|>"] {
                let format = Table::KimiK3.format().with_forced_call(written);
                let output = format!("<|open|>think<|sep|>plan{close}{json}");
                let read = |pieces: &[&str]| {
                    let mut engine =
                        Engine::new(format.clone(), Declared::default().forcing("lookup"));
                    drive(&mut engine, "", pieces)
                };
                assert_eq!(
                    the_same_at_every_cut(read, &output),
                    one_call("plan"),
                    "{written:?} after {close}"
                );
            }
        }
    }

    /// The cells of the sweep that `holds` picks: every table of the registry, the prompt's
    /// thought never opened, closed by the prompt or open, wherever the engines start a grammar,
    /// one call and many. A cell is the table, whether the thought is open, the start and whether
    /// one call.
    fn swept(
        holds: impl Fn(Table, Options) -> bool,
    ) -> Vec<(&'static str, bool, GrammarStart, bool)> {
        let mut cells = Vec::new();
        for table in Table::ALL {
            for (reasoning_open, thought_closed) in [(false, false), (false, true), (true, false)] {
                for start in [
                    GrammarStart::Unknown,
                    GrammarStart::Either,
                    GrammarStart::AfterReasoning,
                    GrammarStart::FirstToken,
                ] {
                    for one_call in [true, false] {
                        let options = Options {
                            at_least_one: true,
                            reasoning_open,
                            thought_closed,
                            one_call,
                            start,
                        };
                        if holds(table, options) {
                            cells.push((table.name(), reasoning_open, start, one_call));
                        }
                    }
                }
            }
        }
        cells
    }

    /// Where the forced JSON arises under a table: the request's own schema constrains the
    /// output only where the table has no tag of its own for the forced call, and the variant
    /// reads the output there. Swept over every table of the registry, wherever the engines
    /// start a grammar, the prompt's thought open, closed by the prompt or never opened, one call
    /// and many, and over the tools a request may carry. For tools every syntax writes, every
    /// table has a tag but three: a table that reads no calls (`plain`), and the Inkling tables
    /// (`inkling`, `inkling2`) with the prompt's thought open where the grammar starts at the
    /// first token. A request whose every tool the table's syntax cannot write leaves that table
    /// without a tag in every cell: a name with a space, a quote or a newline on `kimi_k2` and
    /// `kimi`; a name or a required key with a quote and a bracket, or an empty required key, on
    /// `muse_spark`; a tool with no name on every table. One tool the syntax writes keeps the
    /// tag. A table that gains or loses a tag moves this list, and with it the outputs the
    /// variant reads.
    #[test]
    fn a_forced_call_has_its_own_tag_but_on_three_tables_and_for_tools_a_table_cannot_write() {
        use serde_json::json;
        let tool = |name: &str, parameters: Value| -> Tool {
            serde_json::from_value(json!({
                "type": "function",
                "function": {"name": name, "parameters": parameters},
            }))
            .expect("a tool")
        };
        let city = || json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let keyed = |key: &str| json!({"type": "object", "properties": {key: {"type": "string"}}, "required": [key]});
        // Each request's tools, with the tables that have no tag for them in any cell.
        let shapes: Vec<(&str, Vec<Tool>, &[&str])> = vec![
            ("two simple tools", weather_tools(), &[]),
            (
                "an object with no properties",
                vec![tool("lookup", json!({"type": "object"}))],
                &[],
            ),
            (
                "parameters that are a string",
                vec![tool("lookup", json!({"type": "string"}))],
                &[],
            ),
            (
                "a nested object and an array",
                vec![tool(
                    "lookup",
                    json!({"type": "object", "properties": {
                        "place": {"type": "object", "properties": {"city": {"type": "string"}}},
                        "days": {"type": "array", "items": {"type": "integer"}},
                    }, "required": ["place", "days"]}),
                )],
                &[],
            ),
            (
                "a required key with a space",
                vec![tool("lookup", keyed("the city"))],
                &[],
            ),
            (
                "a required key with a newline",
                vec![tool("lookup", keyed("a\nb"))],
                &[],
            ),
            (
                "a required key with a quote and a bracket",
                vec![tool("lookup", keyed("ci\"ty<"))],
                &["muse_spark"],
            ),
            (
                "a required empty key",
                vec![tool("lookup", keyed(""))],
                &["muse_spark"],
            ),
            (
                "a name with a space",
                vec![tool("look up", city())],
                &["kimi_k2", "kimi"],
            ),
            (
                "a name with a newline",
                vec![tool("look\nup", city())],
                &["kimi_k2", "kimi"],
            ),
            (
                "a name with a quote and a bracket",
                vec![tool("look\"up<", city())],
                &["kimi_k2", "kimi", "muse_spark"],
            ),
            (
                "a name with a space beside a tool every syntax writes",
                vec![tool("look up", city()), tool("lookup", city())],
                &[],
            ),
        ];
        for (what, tools, nowhere) in &shapes {
            let without =
                swept(|table, options| table.format().grammar_for(tools, options).is_none());
            let expected = swept(|table, options| {
                table.name() == "plain"
                    || nowhere.contains(&table.name())
                    || (matches!(table.name(), "inkling" | "inkling2")
                        && options.reasoning_open
                        && options.start == GrammarStart::FirstToken)
            });
            assert_eq!(without, expected, "{what}");
        }
        // A tool with no name cannot be called by any syntax: no table has a tag anywhere.
        let unnamed = [tool("", city())];
        assert_eq!(
            swept(|table, options| table.format().grammar_for(&unnamed, options).is_none()).len(),
            Table::ALL.len() * 24,
            "a tool with no name"
        );
    }
}
