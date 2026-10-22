//! Llama 4: the turn's calls as a Python list, `[get_weather(city="Paris"), get_stock_price()]`,
//! with no marker before or after it, right after the turn's content when the turn has both, and
//! every value between double quotes; everything else content, and no thought. As the checkpoint's
//! chat template renders it for `meta-llama/Llama-4-Scout-17B-16E-Instruct`:
//!
//! ```text
//! Let me check.[get_weather(city="Paris", unit="celsius")]
//! ```
//!
//! The table has one state, an arguments state the output starts in, and the pythonic assembler
//! reads the turn bare (`pythonic::Assembler::bare`): the list
//! as it arrives where the turn starts with one; content otherwise, and a list after the content
//! once it is whole, well formed and made of calls to the request's tools, since code holds
//! `[name(` of its own; the template's wrapping after the list. The
//! template writes every value as `"%s"` of it, so the quotes carry no type and a quoted value
//! reads by the type the request's tool declares for the parameter
//! (`Dialect::LLAMA4`). The model's `<|python_start|>` and
//! `<|python_end|>`, which it may write around its text or the list, are dropped wherever they
//! stand. A turn opens with `<|header_start|>assistant<|header_end|>`, so the prompt's replay
//! starts there. Recorded as `llama-4-scout-17b-16e-instruct` and
//! `llama-4-maverick-17b-128e-instruct`.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    pythonic::Dialect,
};

/// The Llama 4 table.
pub fn llama4() -> Format {
    Format::new("llama4")
        .ignores("python_start", "<|python_start|>")
        .ignores("python_end", "<|python_end|>")
        .state("calls", Emits::Arguments)
        .calls(CallSyntax::Pythonic(Dialect::LLAMA4))
        .opens_turn("<|header_start|>assistant<|header_end|>")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        adapt::chat,
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason, MalformedReason},
        input::{EngineFinish, Input},
        parser::Parser,
        Declared,
    };

    /// The generation prompt's tail: the turn's opener and the two newlines after it.
    const PROMPT: &str = "<|header_start|>assistant<|header_end|>\n\n";

    /// The request's tools, which type the quoted values.
    fn tools() -> Declared {
        Declared::of(&[
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: None,
                    parameters: value!({"type": "object", "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string"},
                        "days": {"type": "integer"},
                        "detailed": {"type": "boolean"},
                    }}),
                    strict: None,
                    extra: Default::default(),
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_stock_price".to_string(),
                    description: None,
                    parameters: value!({"type": "object", "properties": {
                        "ticker": {"type": "string"},
                    }}),
                    strict: None,
                    extra: Default::default(),
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "save_note".to_string(),
                    description: None,
                    parameters: value!({"type": "object", "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "meta": {"type": "object"},
                    }}),
                    strict: None,
                    extra: Default::default(),
                },
            },
        ])
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        run_until(prompt, pieces, EngineFinish::Stop)
    }

    fn run_until(prompt: &str, pieces: &[&str], finish: EngineFinish) -> Vec<Event<'static>> {
        let mut parser = Engine::new(llama4(), tools());
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
        out.into_owned().drain()
    }

    /// Every byte the events carry, in order: the output, when the engine accounts for all of it.
    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|event| match event {
                Event::Content(text) | Event::Reasoning(text) => &*text.text,
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => &*text.text,
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => &*source.text,
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
                Event::Content(text) => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The malformed text, whole: the events that carry it depend on the cuts, the text does not.
    fn malformed(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    fn dropped<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                } => Some(&*text.text),
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

    /// A turn as the checkpoint's chat template renders it, and the message it rendered.
    struct Turn {
        output: &'static str,
        content: &'static str,
        calls: &'static [(&'static str, &'static str)],
    }

    /// The turns bellwether's common set records for `llama-4-scout-17b-16e-instruct`: one call, a
    /// call with no arguments, three calls, a call after content, non-ASCII values with a list,
    /// the nested one (a quote inside a quoted value, which the template writes without escaping
    /// it and the table reads since the quotes inside pair up, and a value that ends in a
    /// backslash; `docs/tables.md`), a value that ends in a backslash (`dir C:\`, as the recorded
    /// benchmark sets write it) alone and after content, and content alone; then the probes.
    const TURNS: &[Turn] = &[
        Turn {
            output: "[get_weather(city=\"Paris\")]",
            content: "",
            calls: &[("get_weather", "{\"city\": \"Paris\"}")],
        },
        Turn {
            output: "[get_stock_price()]",
            content: "",
            calls: &[("get_stock_price", "{}")],
        },
        Turn {
            output: "[get_weather(city=\"Paris\"), get_weather(city=\"Rome\"), get_stock_price(ticker=\"MSFT\")]",
            content: "",
            calls: &[
                ("get_weather", "{\"city\": \"Paris\"}"),
                ("get_weather", "{\"city\": \"Rome\"}"),
                ("get_stock_price", "{\"ticker\": \"MSFT\"}"),
            ],
        },
        Turn {
            output: "Let me check.[get_weather(city=\"Paris\", unit=\"celsius\")]",
            content: "Let me check.",
            calls: &[("get_weather", "{\"city\": \"Paris\", \"unit\": \"celsius\"}")],
        },
        Turn {
            output: "[save_note(title=\"計画\", body=\"今日は晴れ 🌤️ — très bien\", tags=\"['日本語']\")]",
            content: "",
            calls: &[(
                "save_note",
                "{\"title\": \"計画\", \"body\": \"今日は晴れ 🌤️ — très bien\", \"tags\": [\"日本語\"]}",
            )],
        },
        Turn {
            output: "[save_note(title=\"Plan\", body=\"Line one\nLine two with \"quotes\" and a backslash \\\", tags=\"['a', 'b']\", meta=\"{'nested': {'deep': [1, 2.5, True, None]}}\")]",
            content: "",
            calls: &[(
                "save_note",
                "{\"title\": \"Plan\", \"body\": \"Line one\\nLine two with \\\"quotes\\\" and a backslash \\\\\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}",
            )],
        },
        Turn {
            output: "[get_weather(city=\"dir C:\\\", unit=\"N/A\")]",
            content: "",
            calls: &[("get_weather", "{\"city\": \"dir C:\\\\\", \"unit\": \"N/A\"}")],
        },
        Turn {
            output: "Listing.[get_weather(city=\"dir C:\\\")]",
            content: "Listing.",
            calls: &[("get_weather", "{\"city\": \"dir C:\\\\\"}")],
        },
        Turn {
            output: "Hello!",
            content: "Hello!",
            calls: &[],
        },
        Turn {
            output: "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.",
            content: "The syntax is:\n\n```\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n```\n\nThat is all.",
            calls: &[],
        },
        Turn {
            output: "See [1] and [the docs](https://example.com); the list [get_weather, get_stock_price] names two tools.",
            content: "See [1] and [the docs](https://example.com); the list [get_weather, get_stock_price] names two tools.",
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
    fn every_recorded_turn_reads_back_as_the_message_the_template_rendered() {
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
    fn a_quoted_value_reads_by_the_tools_declaration_and_a_bare_one_as_python() {
        // The template quotes every value: the declaration gives a number and a boolean back,
        // and a string of digits stays a string where the tool declares one.
        let events = run(
            PROMPT,
            &["[get_weather(city=\"75001\", days=\"3\", detailed=\"True\", extra=\"7\")]"],
        );
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"75001\", \"days\": 3, \"detailed\": true, \"extra\": 7}".to_string()
            )]
        );
        // What the model writes bare is Python's, as the model card spells the calls.
        let events = run(
            PROMPT,
            &["[get_weather(city='Paris', days=3, detailed=False)]"],
        );
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\", \"days\": 3, \"detailed\": false}".to_string()
            )]
        );
        // A nested value the template wrote as Python's repr inside the quotes reads by the
        // declaration as the object and the list it was.
        let events = run(
            PROMPT,
            &["[save_note(title=\"Plan\", tags=\"['a', 'b']\", meta=\"{'nested': {'deep': [1, 2.5, True, None]}}\")]"],
        );
        assert_eq!(
            calls(&events),
            [(
                "save_note".to_string(),
                "{\"title\": \"Plan\", \"tags\": [\"a\", \"b\"], \"meta\": {\"nested\": {\"deep\": [1, 2.5, true, null]}}}".to_string()
            )]
        );
    }

    #[test]
    fn a_list_after_content_reads_once_whole_and_code_with_brackets_stays_content() {
        // The template renders a message with content and calls as the content and then the list
        // (bellwether's `call-with-content` and `call-after-long-content`): the list is held
        // until its `]` and read when it is a well-formed list of calls.
        let events = run(
            PROMPT,
            &["Sure, two things.\n\n[get_weather(city=\"Paris\"), get_stock_price(ticker=\"MSFT\")]"],
        );
        assert_eq!(content(&events), "Sure, two things.\n\n");
        assert_eq!(calls(&events).len(), 2);
        assert_eq!(finish(&events), (FinishReason::Stop, 2));
        // Code holds `[name(` of its own: a positional argument, an operator, a nested call or a
        // name no tool declares leaves the held text content, and so does a list the end cut.
        for output in [
            "f'(a) = lim(h→0)[f(a+h) - f(a)]/h",
            "const m = movies[Math.floor(Math.random() * movies.length)];",
            "rows = [set(range(10)) for _ in range(3)]",
            "// managers only\n[Authorize(Roles = \"Manager\")]\npublic ActionResult Feedback()",
            "ingredients[exotic_ingredient.lower()] = 1",
            "Let me check.[get_weather(city=\"Paris\")",
        ] {
            let events = run(PROMPT, &[output]);
            assert_eq!(content(&events), output);
            assert!(calls(&events).is_empty(), "{output:?}");
            assert_eq!(malformed(&events), "");
            assert_eq!(finish(&events), (FinishReason::Stop, 0));
        }
        // Whitespace before the turn's list is its wrapping.
        let events = run(PROMPT, &["\n[get_weather(city=\"Paris\")]"]);
        assert_eq!(content(&events), "");
        assert_eq!(calls(&events).len(), 1);
        assert!(dropped(&events).contains(&"\n["));
    }

    #[test]
    fn the_python_markers_are_wrapping_wherever_the_model_puts_them() {
        let events = run(
            PROMPT,
            &["<|python_start|>[get_weather(city=\"Paris\")]<|python_end|>"],
        );
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\"}".to_string()
            )]
        );
        assert_eq!(content(&events), "");
        assert_eq!(malformed(&events), "");
        assert!(dropped(&events).contains(&"<|python_start|>"));
        assert!(dropped(&events).contains(&"<|python_end|>"));
        // Around text, as the family's earlier template wrote the content of a tool turn.
        let events = run(
            PROMPT,
            &["<|python_start|>Let me check.<|python_end|>[get_weather(city=\"Paris\")]"],
        );
        assert_eq!(content(&events), "Let me check.");
        assert_eq!(calls(&events).len(), 1);
        // Around code, which is content.
        let events = run(PROMPT, &["<|python_start|>print(1)<|python_end|>"]);
        assert_eq!(content(&events), "print(1)");
        assert!(calls(&events).is_empty());
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
    }

    #[test]
    fn text_after_the_list_is_the_models_prose_and_a_cut_call_stays_cut() {
        // Once text follows the list, every byte after the `]` is the model's prose, the
        // whitespace before its first word included; whitespace alone is the template's wrapping.
        let events = run(PROMPT, &["[get_weather(city=\"Paris\")]\n\nDone."]);
        assert_eq!(calls(&events).len(), 1);
        assert_eq!(content(&events), "");
        assert_eq!(malformed(&events), "\n\nDone.");
        assert_eq!(dropped(&events), ["[", "]"]);
        let events = run(PROMPT, &["[get_weather(city=\"Paris\")]\n\n"]);
        assert_eq!(malformed(&events), "");
        assert_eq!(dropped(&events), ["[", "]", "\n\n"]);
        let events = run_until(PROMPT, &["[get_weather(city=\"Par"], EngineFinish::Length);
        assert_eq!(content(&events), "");
        // A value is held until its extent is known; cut by the token limit it is the call's own
        // fragment, the open string as written, and the one word of no bytes before the call's
        // end says that the arguments are no whole value.
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), r#"{"city": "Par"#.to_string())]
        );
        assert_eq!(malformed(&events), "");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::UnterminatedRegion }
                if text.text.is_empty()
        )));
        assert_eq!(finish(&events), (FinishReason::Length, 1));
        // A bracket the end cut before any call is content, as it would have been.
        let events = run_until(PROMPT, &["[get_wea"], EngineFinish::Length);
        assert_eq!(content(&events), "[get_wea");
        assert!(calls(&events).is_empty());
    }

    /// A call the token limit cut, the shape of a tool request whose `max_tokens` is below the
    /// call's length: inside the first value's quotes, inside a second quoted value, after a
    /// key's `=` and inside a key. The call's arguments are the prefix as the model wrote it,
    /// the cut member's key and its open string included; nothing of the call reaches a chat
    /// client as content; the one word before the call's end says that the arguments are no
    /// whole value; and the finish is the engine's `length`, whole, a character at a time and
    /// at every two-way cut.
    #[test]
    fn a_call_the_token_limit_cut_keeps_its_bytes_as_its_arguments_and_nothing_as_content() {
        use crate::testing::the_same_at_every_cut;
        for (output, arguments) in [
            ("[get_weather(city=\"Par", r#"{"city": "Par"#),
            (
                "[get_weather(city=\"Paris\", days=\"3",
                r#"{"city": "Paris", "days": "3"#,
            ),
            (
                "[get_weather(city=\"Paris\", days=",
                r#"{"city": "Paris", "days": "#,
            ),
            ("[get_weather(city=\"Paris\", da", r#"{"city": "Paris""#),
        ] {
            let read = |pieces: &[&str]| run_until(PROMPT, pieces, EngineFinish::Length);
            let said = the_same_at_every_cut(read, output);
            assert_eq!(
                said.calls,
                vec![("get_weather".to_string(), arguments.to_string())],
                "{output:?}"
            );
            assert_eq!(said.finish, Some((FinishReason::Length, 1)), "{output:?}");
            assert_eq!(said.content, "", "{output:?}");
            let events = read(&[output]);
            assert_eq!(bytes(&events), output, "{output:?}");
            assert_eq!(
                malformed(&events),
                "",
                "nothing of the call is content: {events:?}"
            );
            assert!(
                matches!(
                    &events[events.len() - 3..],
                    [
                        Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                        Event::ToolCallEnd { source, .. },
                        Event::Finish { .. },
                    ] if text.text.is_empty() && source.text.is_empty()
                ),
                "{output:?}: {events:?}"
            );
        }
    }

    /// The model explains which call it would make, the list inside the sentence: the chat
    /// message's content and the stream's content deltas are the model's prose, byte for byte,
    /// the whitespace between the words after the list included, whole and at every cut alike;
    /// the list is the call.
    #[test]
    fn prose_after_a_list_written_mid_sentence_reaches_the_client_as_the_model_wrote_it() {
        let output = concat!(
            "I can look that up. For example, [get_weather(city=\"Honolulu\")] would give you ",
            "the current weather in Honolulu."
        );
        let prose =
            "I can look that up. For example,  would give you the current weather in Honolulu.";
        let content = |events: &[Event]| chat::message(0, events).message.content;
        let streamed = |events: &[Event]| -> String {
            chat::deltas(0, events)
                .into_iter()
                .filter_map(|choice| choice.delta.content)
                .collect()
        };
        let whole = run(PROMPT, &[output]);
        let message = chat::message(0, &whole).message;
        assert_eq!(message.content.as_deref(), Some(prose));
        let calls = message.tool_calls.expect("a call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments.as_deref(),
            Some(r#"{"city": "Honolulu"}"#)
        );
        assert_eq!(streamed(&whole), prose);
        assert_eq!(dropped(&whole), ["[", "]"]);
        for cut in (1..output.len()).filter(|&cut| output.is_char_boundary(cut)) {
            let events = run(PROMPT, &[&output[..cut], &output[cut..]]);
            assert_eq!(content(&events).as_deref(), Some(prose), "cut at {cut}");
            assert_eq!(streamed(&events), prose, "cut at {cut}");
        }
        let pieces: Vec<&str> = output
            .char_indices()
            .map(|(at, c)| &output[at..at + c.len_utf8()])
            .collect();
        let events = run(PROMPT, &pieces);
        assert_eq!(content(&events).as_deref(), Some(prose));
        assert_eq!(streamed(&events), prose);
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        let format = llama4();
        // After a tool result the prompt holds the previous turn's list; the replay starts at the
        // last opener, and the output is content, where the template puts it.
        let prompt = "<|header_start|>user<|header_end|>\n\nWeather in Paris?<|eot|>\
                      <|header_start|>assistant<|header_end|>\n\n[get_weather(city=\"Paris\")]\
                      <|eot|><|header_start|>ipython<|header_end|>\n\n\"{\\\"temperature\\\": 18}\"\
                      <|eot|><|header_start|>assistant<|header_end|>\n\n";
        assert_eq!(
            format.replay(prompt),
            &prompt[prompt
                .rfind("<|header_start|>assistant<|header_end|>")
                .expect("the opener")..]
        );
        let events = run(prompt, &["It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // A marker quoted in an earlier turn moves nothing.
        let quoted = "<|header_start|>user<|header_end|>\n\nWhat does <|python_start|> do?<|eot|>\
                      <|header_start|>assistant<|header_end|>\n\nIt opens code.<|eot|>\
                      <|header_start|>user<|header_end|>\n\nThanks.<|eot|>\
                      <|header_start|>assistant<|header_end|>\n\n";
        let events = run(quoted, &["You are welcome."]);
        assert_eq!(content(&events), "You are welcome.");
        assert!(!format.reasoning_open(quoted));
    }

    #[test]
    fn the_table_validates_and_names_its_opener() {
        let format = llama4();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "llama4");
        assert_eq!(
            format.turn_opener(),
            Some("<|header_start|>assistant<|header_end|>")
        );
        assert!(format.starts_in_calls());
    }
}
