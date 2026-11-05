//! JSON calls alone in the turn: the whole output is one call object, `{"name": …, "parameters":
//! {…}}` (`arguments` reads the same), or several with `;` or whitespace between them, or a list
//! of them, or content; Llama 3.1's format, whose template renders a call as the bare object and
//! whose models write the call behind the template's call token, `<|python_tag|>{"name": …,
//! "parameters": {…}}`, and their parallel calls as `<|python_tag|>{…}; {…}`; Llama 3.2 Instruct
//! writes its parallel calls as consecutive objects with a blank line between them, `{…}\n\n{…}`,
//! and the engines' own readers of the family read every object of either sequence. The marker
//! is read at the output's start and dropped, so an output with it and one without read the
//! same. The format the gateway's older `json` parser reads for a model that names it, and its
//! older `llama` parser's (Llama 3.2 writes the same). The recorded groups that write it (Nemotron
//! Ultra CPT, UltraLong, Safety Guard, the 70B Reward Principle, Llama 3.2 1B/3B Instruct and
//! 11B/90B Vision Instruct) record the template's bare object; the marker and the `;` form stand
//! on the unit tests, which hold a model's output verbatim.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
use crate::format::{CallSyntax, Emits, Format};

/// The token Llama 3.1's template reserves for a tool call, which its models write before the
/// call's object; the `json` table drops it where the output opens with it.
pub const PYTHON_TAG: &str = "<|python_tag|>";

/// The JSON table.
pub fn json() -> Format {
    Format::new("json")
        .state("calls", Emits::Arguments)
        .calls(CallSyntax::JsonAlone(Some(PYTHON_TAG)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adapt::chat,
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason, MalformedReason},
        input::{EngineFinish, Input},
        parser::Parser,
        testing::{bytes, calls, content},
        Declared,
    };

    fn run(pieces: &[&str]) -> Vec<Event<'static>> {
        run_with(json(), pieces)
    }

    fn run_with(format: Format, pieces: &[&str]) -> Vec<Event<'static>> {
        let mut parser = Engine::new(format, Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "",
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

    /// Prose between two objects of a list: the chat message's content and the stream's deltas
    /// are the model's words with the spaces between them, whole, at every cut and one
    /// character at a time alike; both objects are calls.
    #[test]
    fn text_between_the_lists_objects_reaches_the_client_as_the_model_wrote_it() {
        let output = concat!(
            "[{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}} some words here ",
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}]"
        );
        let prose = "some words here";
        let shown = |events: &[Event]| chat::message(0, events).message.content;
        let streamed = |events: &[Event]| -> String {
            chat::deltas(0, events)
                .into_iter()
                .filter_map(|choice| choice.delta.content)
                .collect()
        };
        let whole = run(&[output]);
        assert_eq!(bytes(&whole), output);
        assert_eq!(calls(&whole).len(), 2);
        assert_eq!(shown(&whole).as_deref(), Some(prose));
        assert_eq!(streamed(&whole), prose);
        for cut in 1..output.len() {
            let events = run(&[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(calls(&events).len(), 2, "cut at {cut}");
            assert_eq!(shown(&events).as_deref(), Some(prose), "cut at {cut}");
            assert_eq!(streamed(&events), prose, "cut at {cut}");
        }
        let pieces: Vec<String> = output.chars().map(String::from).collect();
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let events = run(&refs);
        assert_eq!(shown(&events).as_deref(), Some(prose));
        assert_eq!(streamed(&events), prose);
    }

    #[test]
    fn an_object_alone_is_a_call_a_list_is_calls_and_prose_is_content() {
        let events = run(&[r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#]);
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                r#"{"city": "Paris"}"#.to_string()
            )]
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        let events = run(&[r#"[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]"#]);
        assert_eq!(calls(&events).len(), 2);
        let events = run(&["The weather in Paris is mild."]);
        assert!(calls(&events).is_empty());
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::Content(text) if text.text.starts_with("The"))));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn text_around_the_object_is_wrapping_or_malformed_and_every_chunking_agrees() {
        let output = "\n{\"name\": \"f\", \"arguments\": {\"n\": 1}} tail";
        let whole = run(&[output]);
        assert!(whole.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "\n"
        )));
        assert!(whole.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::TextAfterTheArguments } if text.text == " tail"
        )));
        for cut in 1..output.len() {
            let events = run(&[&output[..cut], &output[cut..]]);
            assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
        }
    }

    /// Whole, and one byte at a time: the same calls, the same bytes accounted for, the same
    /// dropped and malformed text by reason, and the same finish.
    fn whole_and_per_byte(output: &str) -> Vec<Event<'static>> {
        let whole = run(&[output]);
        assert_eq!(bytes(&whole), output);
        let pieces: Vec<&str> = output
            .char_indices()
            .map(|(at, c)| &output[at..at + c.len_utf8()])
            .collect();
        let per_byte = run(&pieces);
        assert_eq!(bytes(&per_byte), output);
        assert_eq!(calls(&per_byte), calls(&whole), "{output:?}");
        assert_eq!(content(&per_byte), content(&whole), "{output:?}");
        assert_eq!(dropped(&per_byte), dropped(&whole), "{output:?}");
        assert_eq!(malformed(&per_byte), malformed(&whole), "{output:?}");
        assert_eq!(per_byte.last(), whole.last(), "{output:?}");
        whole
    }

    /// The dropped text by reason, each reason's text joined.
    fn dropped(events: &[Event]) -> Vec<(DropReason, String)> {
        let mut joined: Vec<(DropReason, String)> = Vec::new();
        for event in events {
            if let Event::Dropped { text, why } = event {
                match joined.iter_mut().find(|(reason, _)| reason == why) {
                    Some((_, so_far)) => so_far.push_str(&text.text),
                    None => joined.push((*why, text.text.to_string())),
                }
            }
        }
        joined
    }

    /// The malformed text by reason, each reason's text joined.
    fn malformed(events: &[Event]) -> Vec<(MalformedReason, String)> {
        let mut joined: Vec<(MalformedReason, String)> = Vec::new();
        for event in events {
            if let Event::Malformed { text, why } = event {
                match joined.iter_mut().find(|(reason, _)| reason == why) {
                    Some((_, so_far)) => so_far.push_str(&text.text),
                    None => joined.push((*why, text.text.to_string())),
                }
            }
        }
        joined
    }

    /// One Llama 3.1 call as the model writes it, verbatim from a served checkpoint's output.
    const ONE_CALL: &str = concat!(
        "<|python_tag|>{\"name\": \"get_stock_info\", \"parameters\": {\"company_name\": ",
        "\"Apple Inc.\", \"detail_level\": \"detailed\", \"market\": \"NASDAQ\"}}"
    );

    /// Four parallel calls as the model writes them, verbatim: one marker, `; ` between the
    /// objects.
    const PARALLEL_CALLS: &str = concat!(
        "<|python_tag|>{\"name\": \"calculate_average\", \"parameters\": {\"numbers\": \"[23, 45, 67, 89]\"}}; ",
        "{\"name\": \"calculate_average\", \"parameters\": {\"numbers\": \"[12, 34, 56, 78]\"}}; ",
        "{\"name\": \"calculate_average\", \"parameters\": {\"numbers\": \"[98, 76, 54, 32]\"}}; ",
        "{\"name\": \"calculate_average\", \"parameters\": {\"numbers\": \"[87, 65, 43, 21]\"}}"
    );

    /// Two calls of different tools, verbatim.
    const TWO_TOOLS: &str = concat!(
        "<|python_tag|>{\"name\": \"scientist_info_get_birthdate\", \"parameters\": {\"name\": \"Nikola Tesla\"}}; ",
        "{\"name\": \"scientist_info_get_famous_discovery\", \"parameters\": {\"name\": \"Nikola Tesla\", ",
        "\"discovery_order\": \"1\"}}"
    );

    #[test]
    fn the_call_token_and_the_semicolons_between_calls_are_the_templates_wrapping() {
        // The one call: the marker dropped, the arguments as written, nothing in content.
        let events = whole_and_per_byte(ONE_CALL);
        assert_eq!(
            calls(&events),
            [(
                "get_stock_info".to_string(),
                r#"{"company_name": "Apple Inc.", "detail_level": "detailed", "market": "NASDAQ"}"#
                    .to_string()
            )]
        );
        assert_eq!(content(&events), "");
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, PYTHON_TAG.to_string())]
        );
        assert!(malformed(&events).is_empty());
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 1,
                ..
            })
        ));
        // The parallel calls: one call per object, numbered in order, each with its own
        // arguments; the `; ` between them dropped with the marker.
        let events = whole_and_per_byte(PARALLEL_CALLS);
        let numbers = [
            "[23, 45, 67, 89]",
            "[12, 34, 56, 78]",
            "[98, 76, 54, 32]",
            "[87, 65, 43, 21]",
        ];
        assert_eq!(
            calls(&events),
            numbers.map(|numbers| (
                "calculate_average".to_string(),
                format!(r#"{{"numbers": "{numbers}"}}"#)
            ))
        );
        let indices: Vec<u32> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(indices, [0, 1, 2, 3]);
        assert_eq!(content(&events), "");
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, format!("{PYTHON_TAG}; ; ; "))]
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 4,
                ..
            })
        ));
        // Two tools, and the same two with the marker left out, or the `;` spaced otherwise.
        let expected = [
            (
                "scientist_info_get_birthdate".to_string(),
                r#"{"name": "Nikola Tesla"}"#.to_string(),
            ),
            (
                "scientist_info_get_famous_discovery".to_string(),
                r#"{"name": "Nikola Tesla", "discovery_order": "1"}"#.to_string(),
            ),
        ];
        assert_eq!(calls(&whole_and_per_byte(TWO_TOOLS)), expected);
        let bare = &TWO_TOOLS[PYTHON_TAG.len()..];
        assert_eq!(calls(&whole_and_per_byte(bare)), expected);
        let spaced = bare.replacen("; ", "\n;\n", 1);
        let events = whole_and_per_byte(&spaced);
        assert_eq!(calls(&events), expected);
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, "\n;\n".to_string())]
        );
        let tight = bare.replacen("; ", ";", 1);
        assert_eq!(calls(&whole_and_per_byte(&tight)), expected);
        // Whitespace around the marker is wrapping too, and a list after it is the list.
        let events = whole_and_per_byte(&format!("\n{PYTHON_TAG}\n{bare}"));
        assert_eq!(calls(&events), expected);
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, format!("\n{PYTHON_TAG}\n; "))]
        );
        let events = whole_and_per_byte(&format!(
            "{PYTHON_TAG}[{{\"name\": \"a\", \"arguments\": {{}}}}, {{\"name\": \"b\", \"arguments\": {{}}}}]"
        ));
        assert_eq!(calls(&events).len(), 2);
    }

    #[test]
    fn a_semicolon_nothing_follows_and_text_after_it_are_text_after_the_arguments() {
        let call = r#"{"name": "f", "parameters": {"a": 1}}"#;
        for (tail, text) in [
            (";", ";"),
            ("; ", "; "),
            ("; and then some", "; and then some"),
            (";\n[]", ";\n[]"),
            (
                "<|python_tag|>{\"name\": \"g\"}",
                "<|python_tag|>{\"name\": \"g\"}",
            ),
        ] {
            let events = whole_and_per_byte(&format!("{call}{tail}"));
            assert_eq!(calls(&events).len(), 1, "{tail:?}");
            assert_eq!(
                malformed(&events),
                [(MalformedReason::TextAfterTheArguments, text.to_string())],
                "{tail:?}"
            );
            assert!(dropped(&events).is_empty(), "{tail:?}");
            // Whole, the text after the arguments is one run and one event, with no event of no
            // bytes after it.
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, Event::Malformed { .. }))
                    .count(),
                1,
                "{tail:?}: {events:?}"
            );
        }
    }

    /// Two parallel calls of different tools as a served Llama 3.2 Instruct checkpoint writes
    /// them, verbatim: consecutive objects, each on its own lines, a blank line between them,
    /// no call token, no `;` and no list.
    const OBJECTS_ON_BLANK_LINES: &str = concat!(
        "{\n",
        "    \"name\": \"triangle_properties_get\",\n",
        "    \"parameters\": {\n",
        "        \"side1\": 5,\n",
        "        \"side2\": 7,\n",
        "        \"side3\": 9,\n",
        "        \"get_area\": true,\n",
        "        \"get_perimeter\": true,\n",
        "        \"get_angles\": true\n",
        "    }\n",
        "}\n",
        "\n",
        "{\n",
        "    \"name\": \"circle_properties_get\",\n",
        "    \"parameters\": {\n",
        "        \"radius\": 3,\n",
        "        \"get_area\": true,\n",
        "        \"get_circumference\": true\n",
        "    }\n",
        "}",
    );

    /// Four, verbatim from the same checkpoint, with one newline between the objects.
    const OBJECTS_ON_NEWLINES: &str = concat!(
        "{\n",
        "    \"name\": \"get_scientist_for_discovery\",\n",
        "    \"parameters\": {\n",
        "        \"discovery\": \"Relativity Theory\"\n",
        "    }\n",
        "}\n",
        "{\n",
        "    \"name\": \"flight_book\",\n",
        "    \"parameters\": {\n",
        "        \"departure_location\": \"Los Angeles\",\n",
        "        \"destination_location\": \"New York\",\n",
        "        \"date\": \"2022-12-25\",\n",
        "        \"time\": \"10:00 AM\",\n",
        "        \"direct_flight\": true\n",
        "    }\n",
        "}\n",
        "{\n",
        "    \"name\": \"game_stats_fetch_player_statistics\",\n",
        "    \"parameters\": {\n",
        "        \"game\": \"Call of Duty\",\n",
        "        \"username\": \"gamer123\",\n",
        "        \"platform\": \"PlayStation\"\n",
        "    }\n",
        "}\n",
        "{\n",
        "    \"name\": \"event_finder_find_upcoming\",\n",
        "    \"parameters\": {\n",
        "        \"location\": \"San Francisco, CA\",\n",
        "        \"genre\": \"rock\",\n",
        "        \"days_ahead\": 14\n",
        "    }\n",
        "}",
    );

    #[test]
    fn consecutive_objects_on_blank_lines_or_newlines_are_the_familys_parallel_calls() {
        // Every object of the sequence is a call, in order, with its own arguments as written;
        // nothing is content, the whitespace between the objects is the template's wrapping, and
        // the finish counts them: the engines' own readers of the family read every object.
        let events = whole_and_per_byte(OBJECTS_ON_BLANK_LINES);
        assert_eq!(
            calls(&events),
            [
                (
                    "triangle_properties_get".to_string(),
                    concat!(
                        "{\n        \"side1\": 5,\n        \"side2\": 7,\n        \"side3\": 9,\n",
                        "        \"get_area\": true,\n        \"get_perimeter\": true,\n",
                        "        \"get_angles\": true\n    }"
                    )
                    .to_string()
                ),
                (
                    "circle_properties_get".to_string(),
                    concat!(
                        "{\n        \"radius\": 3,\n        \"get_area\": true,\n",
                        "        \"get_circumference\": true\n    }"
                    )
                    .to_string()
                ),
            ]
        );
        assert_eq!(content(&events), "");
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, "\n\n".to_string())]
        );
        assert!(malformed(&events).is_empty(), "{events:?}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 2,
                ..
            })
        ));
        let events = whole_and_per_byte(OBJECTS_ON_NEWLINES);
        let names: Vec<String> = calls(&events).into_iter().map(|(name, _)| name).collect();
        assert_eq!(
            names,
            [
                "get_scientist_for_discovery",
                "flight_book",
                "game_stats_fetch_player_statistics",
                "event_finder_find_upcoming",
            ]
        );
        assert_eq!(content(&events), "");
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, "\n\n\n".to_string())]
        );
        assert!(malformed(&events).is_empty(), "{events:?}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 4,
                ..
            })
        ));
        // At every two-way cut as well, the sequence cut between and inside the objects.
        for output in [OBJECTS_ON_BLANK_LINES, OBJECTS_ON_NEWLINES] {
            let whole = run(&[output]);
            for cut in 1..output.len() {
                let events = run(&[&output[..cut], &output[cut..]]);
                assert_eq!(bytes(&events), output, "cut at {cut}");
                assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
                assert_eq!(content(&events), "", "cut at {cut}");
                assert_eq!(events.last(), whole.last(), "cut at {cut}");
            }
        }
    }

    #[test]
    fn the_call_token_before_prose_is_dropped_and_inside_prose_or_cut_short_it_is_text() {
        // The model declines after writing its call token: the token is the template's, the
        // sentence is the answer.
        let events = whole_and_per_byte("<|python_tag|>I cannot help with that.");
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "I cannot help with that.");
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, PYTHON_TAG.to_string())]
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 0,
                ..
            })
        ));
        // The token after prose, or an object after it then, is the model's text, whole: the
        // marker is read at the output's start only, as the engine's own reader reads it.
        let output = format!("Sure, here it is: {ONE_CALL}");
        let events = whole_and_per_byte(&output);
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), output);
        assert!(dropped(&events).is_empty());
        // Cut inside the token, or not the token after all: text.
        for output in ["<|pyth", "<|pyxx", "<|python_tag|", "<", "<b>bold</b>"] {
            let events = whole_and_per_byte(output);
            assert!(calls(&events).is_empty(), "{output:?}");
            assert_eq!(content(&events), output, "{output:?}");
            assert!(dropped(&events).is_empty(), "{output:?}");
        }
        // A second token after the first is text too.
        let events = whole_and_per_byte("<|python_tag|><|python_tag|>{\"name\": \"f\"}");
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "<|python_tag|>{\"name\": \"f\"}");
        assert_eq!(
            dropped(&events),
            [(DropReason::Wrapper, PYTHON_TAG.to_string())]
        );
    }

    #[test]
    fn without_its_calls_the_table_shows_the_output_as_written() {
        // A request without tools reads with the table without its calls: the output is the
        // text it was, the call token included, as the legacy parsers leave it when they run no
        // tool parser.
        let events = run_with(json().without_calls(), &[PARALLEL_CALLS]);
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), PARALLEL_CALLS);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn the_output_starts_in_calls_here_and_in_xlam_only() {
        use crate::formats::{inkling, llama_nemotron, plain, qwen3, xlam};
        assert!(json().starts_in_calls());
        assert!(xlam().starts_in_calls());
        assert!(!qwen3(CallSyntax::Json).starts_in_calls());
        assert!(!plain().starts_in_calls());
        // Inkling starts in an arguments state, but a row enters it and its call has a marker
        // after the name; Llama Nemotron writes xLAM's list between markers.
        assert!(!inkling().starts_in_calls());
        assert!(!llama_nemotron().starts_in_calls());
    }

    #[test]
    fn a_forced_call_is_written_as_the_list() {
        let tool: openai_protocol::common::Tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}
        }))
        .expect("tool");
        let grammar = json().grammar(&[tool], true, false).expect("grammar");
        assert_eq!(grammar.to_json(), crate::formats::xlam().grammar(&[serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}
        })).expect("tool")], true, false).expect("grammar").to_json());
    }
}
