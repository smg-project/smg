//! MiniMax M3: reasoning between `<mm:think>` and `</mm:think>`, the calls of one turn between
//! `<tool_call>` and `</tool_call>`, each call between `<invoke name="…">` and `</invoke>` as an
//! XML tree of arguments, everything else content, and the separator token `]<]minimax[>[` before
//! every tag, dropped wherever it stands. As recorded for `MiniMaxAI/MiniMax-M3`;
//! `MiniMax-M3.1-Flash` reads with it (the same tokenizer and wire; its template differs only
//! outside the wire, as `docs/tables.md` notes).
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::xml,
};

/// The separator token the template writes before every tag, dropped wherever it stands.
pub const SEPARATOR: &str = "]<]minimax[>[";

/// The MiniMax M3 table. With reasoning separation, closers before substantive content are
/// ignored. Under the default content policy, surrounding whitespace belongs to the answer
/// when text follows, and to the template otherwise. Closers after text or calls stay literal.
pub fn minimax_m3() -> Format {
    Format::new("minimax_m3")
        .terminal("think_open", "<mm:think>")
        .terminal("think_close", "</mm:think>")
        .terminal("calls_open", "<tool_call>")
        .terminal("calls_close", "</tool_call>")
        .terminal("invoke_open", "<invoke name=\"")
        .terminal("invoke_close", xml::INVOKE_CLOSE)
        .ignores("separator", SEPARATOR)
        .state("start", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        // A completed call must not reopen the boundary-closer tolerance.
        .state("after_calls", Emits::Content)
        .state("calls", Emits::Wrapper)
        .state("invoke", Emits::Arguments)
        .transition("start", "think_open", "reasoning")
        .transition("start", "think_close", "content")
        .transition("reasoning", "think_close", "content")
        .leading_transition("start", "think_close", "content")
        .leading_transition("content", "think_close", "content")
        .transition("start", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("content", "calls_open", "calls")
        .transition("after_calls", "calls_open", "calls")
        .transition("calls", "invoke_open", "invoke")
        .transition("invoke", "invoke_close", "calls")
        .transition("invoke", "invoke_open", "invoke")
        .transition("invoke", "calls_close", "after_calls")
        .transition("calls", "calls_close", "after_calls")
        .calls(CallSyntax::Xml)
        .opens_turn("]~b]ai\n")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        adapt::chat,
        engine::Engine,
        event::{DropReason, Event, MalformedReason},
        input::EngineFinish,
        testing::{arguments_of, bytes, content, drive, drive_to, dropped, names, reasoning},
        Declared,
    };

    /// A recorded output, separators and all: an empty thought, one invoke with a string and a
    /// nested object.
    const OUTPUT: &str = concat!(
        "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"ChaDri_change_drink\">",
        "]<]minimax[>[<drink_id>latte]<]minimax[>[</drink_id>]<]minimax[>[<new_preferences>",
        "]<]minimax[>[<size>large]<]minimax[>[</size>]<]minimax[>[<temperature>hot",
        "]<]minimax[>[</temperature>]<]minimax[>[</new_preferences>]<]minimax[>[</invoke>\n",
        "]<]minimax[>[</tool_call>"
    );

    fn tools() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "alert".to_string(),
                description: None,
                parameters: Some(value!({"type": "object", "properties": {
                    "name": {"type": "string"},
                    "threshold": {"type": "number"},
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "recipients": {"type": "array", "items": {"type": "string"}},
                    "points": {"type": "array", "items": {"type": "object", "properties": {
                        "x": {"type": "integer"}, "label": {"type": "string"}}}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "options": {"type": "object", "properties": {"id": {"type": "string"}}},
                    "orders": {"type": "array", "items": {"type": "object", "properties": {
                        "item": {"type": "string"}, "quantity": {"type": "integer"}}}},
                    "sep": {"type": "string"},
                    "meta": {"type": "object"},
                    "año": {"type": "integer"},
                    "cart": {"type": "object", "properties": {"item": {"type": "string"}}},
                }})),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    fn run_with(declared: Declared, prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        drive(&mut Engine::new(minimax_m3(), declared), prompt, pieces)
    }

    /// One function with `parameters`, as a request declares it.
    fn declared(name: &str, parameters: serde_json::Value) -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: Some(parameters),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The arguments of the first call, parsed, so a test compares values and not separators.
    fn arguments(events: &[Event]) -> serde_json::Value {
        serde_json::from_str(&arguments_of(events, 0)).expect("the arguments are JSON")
    }

    /// `body` with the template's separator before every tag, as the template writes it.
    fn separated(body: &str) -> String {
        body.replace('<', "]<]minimax[>[<")
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event<'static>> {
        run_with(Declared::default(), prompt, pieces)
    }

    fn fragments_of<'a>(events: &'a [Event<'_>], call: u32) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { index, json, .. } if *index == call => {
                    Some(json.as_ref())
                }
                _ => None,
            })
            .collect()
    }

    fn ends(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, Event::ToolCallEnd { .. }))
            .count()
    }

    /// A `get_weather` tool declared with one argument, the city, a string.
    fn get_weather() -> Declared {
        declared(
            "get_weather",
            value!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )
    }

    /// The chat message's content, as the adapter gives the whole response.
    fn shown(events: &[Event]) -> Option<String> {
        chat::message(0, events).message.content
    }

    /// The stream's content deltas, joined.
    fn streamed(events: &[Event]) -> String {
        chat::deltas(0, events)
            .into_iter()
            .filter_map(|choice| choice.delta.content)
            .collect()
    }

    /// The `Malformed` events' bytes with their reasons, in order.
    fn malformed(events: &[Event]) -> Vec<(String, MalformedReason)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((text.text.to_string(), *why)),
                _ => None,
            })
            .collect()
    }

    /// `inner` wrapped `times` times in objects under the key `d`.
    fn wrapped(inner: serde_json::Value, times: usize) -> serde_json::Value {
        (0..times).fold(inner, |value, _| value!({ "d": value }))
    }

    /// Prose where the invoke has tags, before its first child or after its last: refused word by
    /// word, each word with the whitespace before it, under one counted `Malformed` event of no
    /// bytes per word, the newline after the tag the syntax read dropped as the template's; the
    /// call keeps the member it read and gets no close, since the syntax refused bytes in it; the
    /// chat message's content and the stream's deltas never carry the words, at every cut and one
    /// character at a time alike, and the refused bytes joined are the model's prose, whole.
    #[test]
    fn text_where_the_syntax_has_tags_keeps_the_models_whitespace_between_its_words() {
        let prose = "please note this and that";
        let dropped = |events: &[Event]| -> Vec<String> {
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Dropped {
                        text,
                        why: DropReason::Wrapper,
                    } => Some(text.text.to_string()),
                    _ => None,
                })
                .collect()
        };
        let refused = |events: &[Event]| -> Vec<String> {
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Dropped {
                        text,
                        why: DropReason::MalformedCall,
                    } => Some(text.text.to_string()),
                    _ => None,
                })
                .collect()
        };
        for output in [
            "<tool_call>\n<invoke name=\"get_weather\">\nplease note this and that\n<city>Paris</city>\n</invoke>\n</tool_call>",
            "<tool_call>\n<invoke name=\"get_weather\">\n<city>Paris</city>\nplease note this and that\n</invoke>\n</tool_call>",
        ] {
            let whole = run_with(get_weather(), "", &[output]);
            assert_eq!(bytes(&whole), output);
            assert_eq!(arguments_of(&whole, 0), r#"{"city": "Paris""#, "{output:?}");
            assert_eq!(
                malformed(&whole),
                vec![(String::new(), MalformedReason::TextBetweenTags); 5],
                "{output:?}"
            );
            assert_eq!(
                refused(&whole),
                ["please", " note", " this", " and", " that"],
                "{output:?}"
            );
            assert!(!dropped(&whole).contains(&" ".to_string()), "{output:?}");
            assert_eq!(shown(&whole), None, "{output:?}");
            assert_eq!(streamed(&whole), "", "{output:?}");
            // A word a delta boundary cuts leaves in two pieces: the same bytes refused, joined.
            let joined = |events: &[Event]| -> String {
                for (word, why) in malformed(events) {
                    assert_eq!(why, MalformedReason::TextBetweenTags);
                    assert_eq!(word, "");
                }
                refused(events).concat()
            };
            for cut in (1..output.len()).filter(|&cut| output.is_char_boundary(cut)) {
                let events = run_with(get_weather(), "", &[&output[..cut], &output[cut..]]);
                assert_eq!(bytes(&events), output, "cut at {cut}");
                assert_eq!(joined(&events), prose, "cut at {cut}");
                assert_eq!(shown(&events), None, "cut at {cut}");
                assert_eq!(streamed(&events), "", "cut at {cut}");
            }
            let pieces: Vec<&str> = output
                .char_indices()
                .map(|(at, c)| &output[at..at + c.len_utf8()])
                .collect();
            let events = run_with(get_weather(), "", &pieces);
            assert_eq!(bytes(&events), output);
            assert_eq!(joined(&events), prose);
            assert_eq!(streamed(&events), "");
        }
        // The template's separator inside the prose keeps its place and the words their spaces.
        let output = "<tool_call><invoke name=\"get_weather\">please ]<]minimax[>[note this<city>Paris</city></invoke></tool_call>";
        let events = run_with(get_weather(), "", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(shown(&events), None);
        assert_eq!(refused(&events).concat(), "please note this");
        assert_eq!(arguments_of(&events, 0), r#"{"city": "Paris""#);
    }

    /// Prose between two invokes, where the block's template only wraps: the chat message's
    /// content and the stream's deltas are the model's words with the spaces between them, whole,
    /// at every cut and one character at a time alike; both invokes are calls, and the newline
    /// after the first invoke's close and the one before the next invoke are the template's.
    #[test]
    fn prose_between_two_invokes_reaches_the_client_as_the_model_wrote_it() {
        let output = concat!(
            "<tool_call>\n<invoke name=\"get_weather\">\n<city>Paris</city>\n</invoke>\n",
            "and also this one\n",
            "<invoke name=\"get_weather\">\n<city>Rome</city>\n</invoke>\n</tool_call>"
        );
        let prose = "and also this one";
        let whole = run_with(get_weather(), "", &[output]);
        assert_eq!(bytes(&whole), output);
        assert_eq!(names(&whole), ["get_weather", "get_weather"]);
        assert_eq!(arguments_of(&whole, 0), r#"{"city": "Paris"}"#);
        assert_eq!(arguments_of(&whole, 1), r#"{"city": "Rome"}"#);
        let expected: Vec<(String, MalformedReason)> = ["and", " also", " this", " one"]
            .into_iter()
            .map(|word| (word.to_string(), MalformedReason::TextBetweenCalls))
            .collect();
        assert_eq!(malformed(&whole), expected);
        assert_eq!(shown(&whole).as_deref(), Some(prose));
        assert_eq!(streamed(&whole), prose);
        for cut in (1..output.len()).filter(|&cut| output.is_char_boundary(cut)) {
            let events = run_with(get_weather(), "", &[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(
                names(&events),
                ["get_weather", "get_weather"],
                "cut at {cut}"
            );
            assert_eq!(shown(&events).as_deref(), Some(prose), "cut at {cut}");
            assert_eq!(streamed(&events), prose, "cut at {cut}");
        }
        let pieces: Vec<&str> = output
            .char_indices()
            .map(|(at, c)| &output[at..at + c.len_utf8()])
            .collect();
        let events = run_with(get_weather(), "", &pieces);
        assert_eq!(bytes(&events), output);
        assert_eq!(shown(&events).as_deref(), Some(prose));
        assert_eq!(streamed(&events), prose);
    }

    #[test]
    fn a_recorded_output_gives_its_call_with_its_nested_object_and_every_byte() {
        let events = run("", &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(names(&events), ["ChaDri_change_drink"]);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"drink_id": "latte", "new_preferences": {"size": "large", "temperature": "hot"}}"#
        );
        assert_eq!(ends(&events), 1);
        assert_eq!(content(&events), "");
        assert!(!events.iter().any(|e| matches!(e, Event::ReasoningStart)));
        // Every separator is dropped, and the one after `latte` stays in the fragment that holds
        // the value, so the bytes keep their order.
        assert!(events.contains(&dropped(SEPARATOR)));
        let held = format!("<drink_id>latte{SEPARATOR}</drink_id>");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallArguments { source, .. } if source.text.ends_with(&held)
        )));
    }

    #[test]
    fn leaves_are_typed_by_the_tools_at_every_depth_and_lists_come_from_items() {
        let body = concat!(
            "<name>12</name><threshold>5.0",
            "</threshold><count>007</count><enabled>true</enabled><recipients><item>a@x</item>",
            "<item>b@x</item></recipients><points><item><x>1</x><label>7</label></item></points>",
            "<tags></tags><options></options><extra>12</extra>",
            "<orders><item><item>burgers</item><quantity>5</quantity></item></orders>",
            "<sep> </sep><año>2024</año><meta><deep><item>1</item><item></item></deep></meta>",
        );
        let output = format!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"alert\">{}\
             ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            separated(body)
        );
        let output = output.as_str();
        let events = run_with(tools(), "", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            arguments_of(&events, 0),
            concat!(
                r#"{"name": "12", "threshold": 5.0, "count": 7, "enabled": true, "#,
                r#""recipients": ["a@x", "b@x"], "points": [{"x": 1, "label": "7"}], "#,
                r#""tags": [], "options": {}, "extra": 12, "#,
                r#""orders": [{"item": "burgers", "quantity": 5}], "sep": " ", "año": 2024, "#,
                r#""meta": {"deep": [1, null]}}"#
            )
        );
        // Without the tools every leaf is inferred, a top-level empty element is an empty string,
        // and an element whose first child is an `<item>` is a list: the order's `<item>` member
        // makes the order a list, in which the member is a leaf and `quantity` a second item.
        let events = run("", &[output]);
        assert_eq!(
            arguments_of(&events, 0),
            concat!(
                r#"{"name": 12, "threshold": 5.0, "count": "007", "enabled": true, "#,
                r#""recipients": ["a@x", "b@x"], "points": [{"x": 1, "label": 7}], "#,
                r#""tags": "", "options": "", "extra": 12, "#,
                r#""orders": [["burgers", 5]], "sep": " ", "año": 2024, "#,
                r#""meta": {"deep": [1, null]}}"#
            )
        );
    }

    #[test]
    fn a_declared_string_streams_as_it_arrives_and_a_separator_inside_keeps_its_place() {
        let pieces = [
            "</mm:think><tool_call>\n<invoke name=\"alert\"><name>Pa",
            "ris",
            SEPARATOR,
            "</name></invoke>\n</tool_call>",
        ];
        let events = run_with(tools(), "", &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(
            fragments_of(&events, 0),
            [r#"{"name": ""#, "Pa", "ris", "\"", "}"]
        );
        assert!(events.contains(&dropped(SEPARATOR)));
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = run("", &[OUTPUT]);
        let said = (arguments_of(&whole, 0), names(&whole).len(), ends(&whole));
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run("", &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(
                (
                    arguments_of(&events, 0),
                    names(&events).len(),
                    ends(&events)
                ),
                said,
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn the_thought_is_the_models_to_write_or_to_close_empty() {
        let events = run("", &["<mm:think>A plan.</mm:think>Hello."]);
        assert_eq!(reasoning(&events), "A plan.");
        assert_eq!(content(&events), "Hello.");
        assert!(events.contains(&dropped("<mm:think>")));

        let events = run("", &["</mm:think>Hello!"]);
        assert_eq!(content(&events), "Hello!");
        assert!(events.contains(&dropped("</mm:think>")));
        assert!(!events.iter().any(|e| matches!(e, Event::ReasoningStart)));

        // Later in the content both markers are text, and a separator anywhere is dropped.
        let events = run(
            "",
            &["</mm:think>Say <mm:think>x</mm:think>", SEPARATOR, " done"],
        );
        assert_eq!(content(&events), "Say <mm:think>x</mm:think> done");
        assert!(events.contains(&dropped(SEPARATOR)));

        // No marker at all: content.
        let events = run("", &["Hello."]);
        assert_eq!(content(&events), "Hello.");
    }

    /// The legacy parser removes only boundary closers, preserving whitespace and literal
    /// markers after content begins. Whole responses and every streaming cut must agree.
    #[test]
    fn repeated_boundary_closers_preserve_content_at_every_chunk_boundary() {
        for (prompt, output, thought, answer) in [
            ("", "</mm:think></mm:think>答案", "", "答案"),
            ("", " \n</mm:think>\t</mm:think> 答案", "", " \n\t 答案"),
            (
                "",
                "<mm:think>思考</mm:think> </mm:think>\n答案",
                "思考",
                " \n答案",
            ),
            ("", "正文</mm:think>后续", "", "正文</mm:think>后续"),
            ("", "</mm:think></mm:thi", "", "</mm:thi"),
            (
                "",
                "<mm:think>思考</mm:think>正文</mm:think>后续",
                "思考",
                "正文</mm:think>后续",
            ),
            (
                "",
                "</mm:think>Say <mm:think>x</mm:think>",
                "",
                "Say <mm:think>x</mm:think>",
            ),
            (
                "]~b]ai\n<mm:think>",
                "思考</mm:think></mm:think>答案",
                "思考",
                "答案",
            ),
            ("]~b]ai\n</mm:think>", "</mm:think>答案", "", "答案"),
            (
                "]~b]ai\n</mm:think>正文",
                "</mm:think>后续",
                "",
                "</mm:think>后续",
            ),
            // A whitespace-only stay retains the existing empty-content convention.
            ("", "</mm:think> \n</mm:think>\t", "", ""),
        ] {
            let check = |pieces: &[&str]| {
                let events = run(prompt, pieces);
                assert_eq!(bytes(&events), output);
                assert_eq!(reasoning(&events), thought, "{prompt:?}: {pieces:?}");
                assert_eq!(content(&events), answer, "{prompt:?}: {pieces:?}");
                assert_eq!(
                    shown(&events).as_deref(),
                    (!answer.trim().is_empty()).then_some(answer)
                );
                assert_eq!(streamed(&events), answer);
            };
            check(&[output]);
            for (cut, _) in output.char_indices().skip(1) {
                check(&[&output[..cut], &output[cut..]]);
            }
            let chars: Vec<_> = output
                .char_indices()
                .map(|(at, ch)| &output[at..at + ch.len_utf8()])
                .collect();
            check(&chars);
        }
    }

    /// Tool-call tails and requests without reasoning separation preserve literal closers.
    #[test]
    fn boundary_closers_do_not_strip_markers_after_calls_or_without_reasoning() {
        let output = concat!(
            "</mm:think></mm:think><tool_call><invoke name=\"f\"></invoke></tool_call>",
            "</mm:think>正文"
        );
        let events = run("", &[output]);
        assert_eq!(names(&events), ["f"]);
        assert_eq!(content(&events), "</mm:think>正文");
        assert_eq!(bytes(&events), output);

        let output = "</mm:think> <mm:think>思考</mm:think></mm:think>答案";
        let events = drive(
            &mut Engine::new(minimax_m3().without_reasoning(), Declared::default()),
            "",
            &[output],
        );
        assert_eq!(content(&events), output);
        assert_eq!(reasoning(&events), "");
    }

    /// Parser-only rows preserve byte order and literal tails under format policies.
    #[test]
    fn boundary_closers_respect_content_policies_and_structural_tails() {
        for (format, output, answer) in [
            (minimax_m3().framing("content"), " </mm:think> hello", ""),
            (
                minimax_m3().keeps("content"),
                " </mm:think>hello</mm:think>",
                " hello</mm:think>",
            ),
            (minimax_m3().trims("content"), " </mm:think> hello", "hello"),
            (
                minimax_m3().structural_token("</mm:th"),
                "answer</mm:th",
                "answer</mm:th",
            ),
            (minimax_m3().structural_token("</mm:th"), "</mm:th", ""),
            (
                minimax_m3().structural_token("</mm:th"),
                "</mm:think></mm:th",
                "",
            ),
        ] {
            for cut in 0..=output.len() {
                let events = drive(
                    &mut Engine::new(format.clone(), Declared::default()),
                    "",
                    &[&output[..cut], &output[cut..]],
                );
                assert_eq!(bytes(&events), output, "{output:?}: {cut}");
                assert_eq!(content(&events), answer, "{output:?}: {cut}");
            }
        }
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts_from_the_last_turn_opener() {
        let events = run("]~b]ai\n</mm:think>", &["Hello."]);
        assert_eq!(content(&events), "Hello.");
        assert!(!events.iter().any(|e| matches!(e, Event::ReasoningStart)));

        let events = run("]~b]ai\n<mm:think>", &["A plan.</mm:think>Hello."]);
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(reasoning(&events), "A plan.");
        assert_eq!(content(&events), "Hello.");

        // A calls block quoted in the user's turn moves nothing: the replay starts at the last
        // `]~b]ai\n`.
        let prompt = "]~b]user\nWhat does <tool_call>\n<invoke name=\"f\"> mean?[e~[\n]~b]ai\n";
        let events = run(prompt, &["</mm:think>It opens a call."]);
        assert_eq!(content(&events), "It opens a call.");
    }

    #[test]
    fn a_calls_block_inside_the_thought_ends_it_and_a_second_invoke_ends_the_first() {
        let output = concat!(
            "<mm:think>Plan.<tool_call>\n<invoke name=\"f\"><a>x</a><invoke name=\"g\"><b>y</b>",
            "</invoke>\n</tool_call>Done."
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(reasoning(&events), "Plan.");
        assert_eq!(names(&events), ["f", "g"]);
        assert_eq!(arguments_of(&events, 0), r#"{"a": "x"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"b": "y"}"#);
        assert_eq!(ends(&events), 2);
        assert_eq!(content(&events), "Done.");
    }

    #[test]
    fn the_blocks_close_ends_an_invoke_left_open_and_the_prose_after_it_stays() {
        let output = concat!(
            "</mm:think><tool_call>\n<invoke name=\"f\">]<]minimax[>[<opts>]<]minimax[>[<a>1",
            "]<]minimax[>[</a><b>tex</tool_call>The weather is sunny."
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(names(&events), ["f"]);
        // The leaf never written is refused, so no close is made up for the call: the open object
        // stands as the fragments wrote it, and the arguments are no whole value.
        assert_eq!(arguments_of(&events, 0), r#"{"opts": {"a": 1"#);
        assert_eq!(ends(&events), 1);
        assert_eq!(content(&events), "The weather is sunny.");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::MalformedCall } if text.text == "<b>tex"
        )));
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_ends_the_call_with_its_arguments_cut() {
        let output = "</mm:think><tool_call>\n<invoke name=\"alert\"><name>Par";
        // The model's own end of output there closes the string and the object, and says so.
        let stopped = run_with(tools(), "", &[output]);
        assert_eq!(bytes(&stopped), output);
        assert_eq!(
            fragments_of(&stopped, 0),
            [r#"{"name": ""#, "Par", "\"", "}"]
        );
        assert!(stopped.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::OpenAtTheStop } if text.text.is_empty()
        )));
        // Cut by the token limit, nothing closes.
        let events = drive_to(
            &mut Engine::new(minimax_m3(), tools()),
            "",
            &[output],
            EngineFinish::Length,
        );
        assert_eq!(bytes(&events), output);
        assert_eq!(fragments_of(&events, 0), [r#"{"name": ""#, "Par"]);
        assert_eq!(ends(&events), 1);
        // The arguments are no whole value and nothing is made up: one `Malformed` event of no
        // bytes says so, right before the call's end.
        let ended = events
            .iter()
            .position(|event| matches!(event, Event::ToolCallEnd { .. }))
            .expect("the call ends");
        assert!(
            matches!(
                &events[ended - 1],
                Event::Malformed { text, why: MalformedReason::UnterminatedRegion }
                    if text.text.is_empty()
            ),
            "{events:#?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Malformed { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { source, .. } if source.text.is_empty()
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn text_that_is_not_a_tag_is_a_values_text_and_text_between_tags_is_refused() {
        let output = concat!(
            "<tool_call>\n<invoke name=\"f\"><expr>a < b and c > d</expr>junk<n>1</n>",
            "<q>a <b> c</q></invoke>\n</tool_call>"
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        // The word between tags is refused, so the call gets no close after the members it read.
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"expr": "a < b and c > d", "n": 1, "q": "a <b> c""#
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::MalformedCall } if text.text == "junk"
        )));
        assert!(!content(&events).contains("junk"));
    }

    #[test]
    fn a_property_named_item_keeps_its_type_below_an_object() {
        let output = concat!(
            "</mm:think><tool_call>\n<invoke name=\"alert\"><cart><item>123</item></cart>",
            "</invoke>\n</tool_call>"
        );
        let events = run_with(tools(), "", &[output]);
        assert_eq!(arguments_of(&events, 0), r#"{"cart": {"item": "123"}}"#);
    }

    #[test]
    fn a_separator_inside_the_invoke_tag_is_carried_and_never_splits_a_character() {
        // The separator arriving between the name's quote and the `>`, after a multi-byte
        // character: the tail is refused around it, counted once, its bytes in order around the
        // separator's own event, and the call starts (smg #2850, claude[bot]).
        let output = concat!(
            "</mm:think><tool_call>\n<invoke name=\"add\" ñ]<]minimax[>[x><a>1</a></invoke>\n",
            "</tool_call>"
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(names(&events), ["add"]);
        // The tail is refused, so the call gets no close after the member it read.
        assert_eq!(arguments_of(&events, 0), r#"{"a": 1"#);
        let malformed_bytes: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(malformed_bytes, [""]);
        let refused_bytes: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::MalformedCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(refused_bytes, [" ñ", "x"]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, .. } if text.text == "]<]minimax[>["
        )));
    }

    #[test]
    fn only_the_invokes_closing_tag_is_a_started_calls_end() {
        let output = concat!(
            "</mm:think><tool_call>\n<invoke name=\"f\"><a>1</a></invoke>\n",
            "<invoke name=\"g\"><b>2</b><invoke name=\"h\"><c>3</c></tool_call>Done."
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        let ends: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(&*source.text),
                _ => None,
            })
            .collect();
        assert_eq!(ends, ["</invoke>", "", ""]);
        assert_eq!(
            events
                .iter()
                .filter(|event| **event == dropped("<invoke name=\""))
                .count(),
            3
        );
        assert!(events.contains(&dropped("</tool_call>")));
        assert_eq!(content(&events), "Done.");
        // An invoke that named no call is refused whole with whichever terminal ended it.
        let output = "<tool_call>\n<invoke name=\"</tool_call>";
        let events = run("", &[output]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::MalformedCall } if text.text.ends_with("</tool_call>")
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    /// The invoke's close inside a streamed string, by its own closing tag, the block's close or
    /// the next invoke: the model ended the call, so the value's bytes are the value whole. The
    /// string is closed, the start of a tag held there its text since no tag followed, under one
    /// `Malformed { OpenAtTheMarker }` event of no bytes right before the closing quote; the
    /// containers above it and the object close, and the arguments read as JSON.
    #[test]
    fn the_invokes_close_inside_a_string_closes_it_there_and_says_so() {
        for (output, value) in [
            (
                "</mm:think><tool_call>\n<invoke name=\"alert\"><name>Par</na</invoke>\n</tool_call>",
                "Par</na",
            ),
            ("</mm:think><tool_call>\n<invoke name=\"alert\"><name>Par</tool_call>", "Par"),
            (
                "</mm:think><tool_call>\n<invoke name=\"alert\"><name>Par<invoke name=\"alert\">\
                 </invoke>\n</tool_call>",
                "Par",
            ),
        ] {
            let events = run_with(tools(), "", &[output]);
            assert_eq!(bytes(&events), output, "{output:?}");
            assert_eq!(
                arguments_of(&events, 0),
                format!("{{\"name\": \"{value}\"}}"),
                "{output:?}"
            );
            let at = events
                .iter()
                .position(|event| matches!(event, Event::Malformed { .. }))
                .expect("the one word");
            assert!(
                matches!(
                    &events[at..at + 2],
                    [
                        Event::Malformed { text, why: MalformedReason::OpenAtTheMarker },
                        Event::ToolCallArguments { index: 0, json, .. }
                    ] if text.text.is_empty() && json == "\""
                ),
                "{output:?}: {events:#?}"
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, Event::Malformed { .. }))
                    .count(),
                1,
                "{output:?}"
            );
        }
    }

    #[test]
    fn an_invoke_with_no_name_starts_no_call_and_the_next_one_takes_index_zero() {
        let output = concat!(
            "<tool_call>\n<invoke name=\"\"><a>1</a></invoke>\n",
            "<invoke name=\"g\"></invoke>\n</tool_call>"
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(names(&events), ["g"]);
        assert_eq!(arguments_of(&events, 0), "{}");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    /// The review's probes (smg #2850), each rendered offline by MiniMax M3's template at
    /// f0e1c1e0 from the reference arguments.
    #[test]
    fn a_declared_string_that_starts_with_a_tag_is_its_text() {
        let vue = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"write_file\">",
            "]<]minimax[>[<path>App.vue]<]minimax[>[</path>]<]minimax[>[<content><template>\n",
            "  <div>{{ msg }}</div>\n</template>\n]<]minimax[>[</content>]<]minimax[>[</invoke>\n",
            "]<]minimax[>[</tool_call>"
        );
        let tools = value!({"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}});
        let events = run_with(declared("write_file", tools), "", &[vue]);
        assert_eq!(bytes(&events), vue);
        let content = "<template>\n  <div>{{ msg }}</div>\n</template>\n";
        assert_eq!(
            arguments(&events),
            value!({"path": "App.vue", "content": content})
        );
        let html = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"send_email\">",
            "]<]minimax[>[<to>a@b.c]<]minimax[>[</to>]<]minimax[>[<body><p>Hello</p><p>Bye</p>",
            "]<]minimax[>[</body>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
        );
        let tools = value!({"type": "object", "properties": {
            "to": {"type": "string"}, "body": {"type": "string"}}});
        let events = run_with(declared("send_email", tools), "", &[html]);
        assert_eq!(
            arguments(&events),
            value!({"to": "a@b.c", "body": "<p>Hello</p><p>Bye</p>"})
        );
    }

    #[test]
    fn without_a_declaration_a_tag_opens_a_child_only_after_the_separator() {
        // The template writes its separator before every tag it writes and never inside a
        // value, so a tag that follows no separator is the value's text.
        let text = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">",
            "]<]minimax[>[<body><p>Hi</p>]<]minimax[>[</body>]<]minimax[>[</invoke>\n",
            "]<]minimax[>[</tool_call>"
        );
        let events = run("", &[text]);
        assert_eq!(arguments(&events), value!({"body": "<p>Hi</p>"}));
        let child = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">",
            "]<]minimax[>[<body>]<]minimax[>[<p>Hi]<]minimax[>[</p>]<]minimax[>[</body>",
            "]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
        );
        let events = run("", &[child]);
        assert_eq!(arguments(&events), value!({"body": {"p": "Hi"}}));
    }

    #[test]
    fn a_key_is_any_run_of_characters_without_whitespace_or_angle_brackets() {
        let odata = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"list_users\">",
            "]<]minimax[>[<$filter>startswith(name, 'A')]<]minimax[>[</$filter>",
            "]<]minimax[>[<$top>5]<]minimax[>[</$top>]<]minimax[>[</invoke>\n",
            "]<]minimax[>[</tool_call>"
        );
        let tools = value!({"type": "object", "properties": {
            "$filter": {"type": "string"}, "$top": {"type": "integer"}}});
        let events = run_with(declared("list_users", tools), "", &[odata]);
        assert_eq!(bytes(&events), odata);
        assert_eq!(
            arguments(&events),
            value!({"$filter": "startswith(name, 'A')", "$top": 5})
        );
        let mongo = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"find\">",
            "]<]minimax[>[<collection>users]<]minimax[>[</collection>]<]minimax[>[<filter>",
            "]<]minimax[>[<age>]<]minimax[>[<$gt>30]<]minimax[>[</$gt>]<]minimax[>[</age>",
            "]<]minimax[>[</filter>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
        );
        let tools = value!({"type": "object", "properties": {
            "collection": {"type": "string"}, "filter": {"type": "object"}}});
        let events = run_with(declared("find", tools), "", &[mongo]);
        assert_eq!(
            arguments(&events),
            value!({"collection": "users", "filter": {"age": {"$gt": 30}}})
        );
    }

    #[test]
    fn an_empty_member_of_an_undeclared_object_is_an_empty_string_and_an_empty_item_null() {
        // The template skips a mapping's `None` member, so an empty element under an object
        // is an empty string; only a list writes `None` as an empty element.
        let member = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">",
            "]<]minimax[>[<opts>]<]minimax[>[<unit>]<]minimax[>[</unit>]<]minimax[>[<note>x",
            "]<]minimax[>[</note>]<]minimax[>[</opts>]<]minimax[>[</invoke>\n",
            "]<]minimax[>[</tool_call>"
        );
        let tools = value!({"type": "object", "properties": {"opts": {"type": "object"}}});
        let events = run_with(declared("f", tools), "", &[member]);
        assert_eq!(
            arguments(&events),
            value!({"opts": {"unit": "", "note": "x"}})
        );
        let item = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">",
            "]<]minimax[>[<tags>]<]minimax[>[<item>]<]minimax[>[</item>]<]minimax[>[</tags>",
            "]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
        );
        let events = run("", &[item]);
        assert_eq!(arguments(&events), value!({"tags": [null]}));
    }

    #[test]
    fn a_leaf_below_a_nullable_list_or_object_keeps_its_declared_type() {
        // Pydantic writes `Optional[List[str]]` as an `anyOf` around the inline array and
        // `Optional[Model]` as an `anyOf` around a `$ref` into `$defs`; the declared types reach
        // the leaves below both, so `1234` under a list of strings stays the string it is (main
        // made it the number).
        let output = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">",
            "]<]minimax[>[<tags>]<]minimax[>[<item>1234]<]minimax[>[</item>]<]minimax[>[</tags>",
            "]<]minimax[>[<meta>]<]minimax[>[<id>7]<]minimax[>[</id>]<]minimax[>[<flag>true",
            "]<]minimax[>[</flag>]<]minimax[>[</meta>]<]minimax[>[</invoke>\n",
            "]<]minimax[>[</tool_call>"
        );
        let tools = value!({"type": "object", "properties": {
        "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "meta": {"anyOf": [{"$ref": "#/$defs/Meta"}, {"type": "null"}]}},
        "$defs": {"Meta": {"type": "object", "properties": {
            "id": {"type": "integer"}, "flag": {"type": "string"}}}}});
        let events = run_with(declared("f", tools), "", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            arguments(&events),
            value!({"tags": ["1234"], "meta": {"id": 7, "flag": "true"}})
        );
    }

    /// A call whose elements nest past the bound the JSON reader applies to every syntax
    /// (`PartialJson::DEFAULT_MAX_DEPTH` containers, the arguments object counted) is refused
    /// from the element that would cross it: nothing of that member is written, the bytes from
    /// its own tag to the call's end are dropped as a malformed call's behind one counted
    /// `Malformed { InvalidArguments }` of no bytes, and nothing closes after it, since a call
    /// the syntax refused bytes in is no whole call: the containers read stand open, a prefix the
    /// reader takes and no whole value, and every byte is accounted for, whole and at every cut.
    /// Nesting up to the bound reads as before, for an object's members and a list's items alike.
    #[test]
    fn nesting_past_the_json_readers_bound_is_refused_from_the_element_that_crosses_it() {
        use crate::{json::PartialJson, testing::the_same_at_every_cut};
        let deepest = PartialJson::DEFAULT_MAX_DEPTH;
        // `levels` elements nested under the arguments object, the innermost a leaf, the
        // template's separator before every tag.
        let chain = |tag: &str, levels: usize| {
            let body = format!(
                "{}1{}",
                format!("<{tag}>").repeat(levels),
                format!("</{tag}>").repeat(levels)
            );
            format!(
                "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"alert\">{}\
                 ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
                separated(&body)
            )
        };
        // `inner` wrapped `times` times: in objects under the key, or in lists.
        let wrapped = |tag: &str, inner: serde_json::Value, times: usize| {
            (0..times).fold(inner, |value, _| {
                if tag == "item" {
                    value!([value])
                } else {
                    value!({ tag: value })
                }
            })
        };
        for tag in ["d", "item"] {
            // A chain of `deepest` elements makes `deepest - 1` containers below the arguments
            // object: as many as the reader takes, nothing refused.
            let within = chain(tag, deepest);
            let events = run("", &[&within]);
            assert_eq!(bytes(&events), within, "{tag}");
            assert_eq!(names(&events), ["alert"], "{tag}");
            let json = arguments_of(&events, 0);
            assert!(
                PartialJson::default().parse(&json, false).is_ok(),
                "{tag}: {json}"
            );
            assert_eq!(
                arguments(&events),
                value!({ tag: wrapped(tag, value!(1), deepest - 1) }),
                "{tag}"
            );
            assert!(malformed(&events).is_empty(), "{tag}: {events:?}");

            // One element more: the element below the deepest container would make it the
            // container past the bound, and is refused from its own tag to the call's end; no
            // container closes after it, the call being no whole call.
            let past = chain(tag, deepest + 1);
            let events = run("", &[&past]);
            assert_eq!(bytes(&events), past, "{tag}");
            assert_eq!(names(&events), ["alert"], "{tag}");
            let json = arguments_of(&events, 0);
            assert!(
                PartialJson::default().parse(&json, false).is_ok(),
                "{tag}: a prefix the parser takes: {json}"
            );
            let open = if tag == "item" {
                format!("{{\"item\": [{}", "[".repeat(deepest - 2))
            } else {
                format!("{{{}", "\"d\": {".repeat(deepest - 1))
            };
            assert_eq!(
                json, open,
                "{tag}: the containers read stand open, the member that crossed the bound is not written"
            );
            assert!(
                serde_json::from_str::<serde_json::Value>(&json).is_err(),
                "{tag}: {json} reads as a whole value"
            );
            // Counted once, with no bytes of its own; the bytes from the tag of the element
            // that would cross the bound (the `deepest`-th) to the last closing tag are
            // dropped as the call's, the separators among them as the wrapper's.
            assert_eq!(
                malformed(&events),
                [(String::new(), MalformedReason::InvalidArguments)],
                "{tag}"
            );
            let tail = format!(
                "{}1{}",
                format!("<{tag}>").repeat(2),
                format!("</{tag}>").repeat(deepest + 1)
            );
            assert_eq!(refused(&events), tail, "{tag}");
            assert_eq!(content(&events), "", "{tag}");
            assert_eq!(ends(&events), 1, "{tag}");
            // The same at every cut.
            the_same_at_every_cut(|pieces| run("", pieces), &past);
        }
    }

    /// A leaf's own JSON counts toward the same bound: a list under the deepest container the
    /// reader takes would be the container past it, and is refused from the leaf's own tag to
    /// the call's end, exactly as a tag there would be; one level up it is read.
    #[test]
    fn a_leafs_own_nesting_counts_toward_the_bound() {
        use crate::{json::PartialJson, testing::the_same_at_every_cut};
        let deepest = PartialJson::DEFAULT_MAX_DEPTH;
        let chain = |levels: usize, leaf: &str| {
            let body = format!("{}{leaf}{}", "<d>".repeat(levels), "</d>".repeat(levels));
            format!(
                "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"alert\">{}\
                 ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
                separated(&body)
            )
        };
        // `deepest - 1` elements: `deepest - 2` containers below the arguments object and a
        // leaf; its list is the last container the reader takes.
        let within = chain(deepest - 1, "[1]");
        let events = run("", &[&within]);
        assert_eq!(bytes(&events), within);
        let json = arguments_of(&events, 0);
        assert!(PartialJson::default().parse(&json, false).is_ok(), "{json}");
        assert_eq!(
            arguments(&events),
            value!({ "d": wrapped(value!([1]), deepest - 2) })
        );
        assert!(malformed(&events).is_empty(), "{events:?}");

        // One element more: the leaf's list would be the container past the bound; the leaf is
        // refused from its own tag, and no container closes after it.
        let past = chain(deepest, "[1]");
        let events = run("", &[&past]);
        assert_eq!(bytes(&events), past);
        assert_eq!(names(&events), ["alert"]);
        let json = arguments_of(&events, 0);
        assert!(PartialJson::default().parse(&json, false).is_ok(), "{json}");
        assert_eq!(
            json,
            format!("{{{}", "\"d\": {".repeat(deepest - 1)),
            "the containers read stand open, the leaf that crossed the bound is not written"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(&json).is_err(),
            "{json} reads as a whole value"
        );
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::InvalidArguments)]
        );
        assert_eq!(
            refused(&events),
            format!("<d>[1]{}", "</d>".repeat(deepest))
        );
        assert_eq!(content(&events), "");
        assert_eq!(ends(&events), 1);
        the_same_at_every_cut(|pieces| run("", pieces), &past);

        // The same leaf at the same depth as a scalar is read: only a container uses the depth.
        let scalar = chain(deepest, "1");
        let events = run("", &[&scalar]);
        assert!(malformed(&events).is_empty(), "{events:?}");
        assert_eq!(
            arguments(&events),
            value!({ "d": wrapped(value!(1), deepest - 1) })
        );
    }

    /// An empty element the tool declares a container is written as `[]` or `{}`: at the bound
    /// that would be the container past it, so the element is refused from its own tag as a
    /// leaf there would be; one level up it is written.
    #[test]
    fn an_empty_element_declared_a_container_at_the_bound_is_refused() {
        use crate::json::PartialJson;
        let deepest = PartialJson::DEFAULT_MAX_DEPTH;
        // `levels` nested `<d>` elements, the innermost empty; the tool declares an object at
        // every level but the innermost, an array there.
        let output = |levels: usize| {
            let body = format!("{}{}", "<d>".repeat(levels), "</d>".repeat(levels));
            format!(
                "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"alert\">{}\
                 ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
                separated(&body)
            )
        };
        let schema = |levels: usize| {
            let innermost = value!({"type": "array", "items": {"type": "integer"}});
            let nested = (1..levels).fold(
                innermost,
                |schema, _| value!({"type": "object", "properties": {"d": schema}}),
            );
            declared(
                "alert",
                value!({"type": "object", "properties": {"d": nested}}),
            )
        };
        // `deepest - 1` elements: the empty array is the last container the reader takes.
        let within = output(deepest - 1);
        let events = run_with(schema(deepest - 1), "", &[&within]);
        assert_eq!(bytes(&events), within);
        let json = arguments_of(&events, 0);
        assert!(PartialJson::default().parse(&json, false).is_ok(), "{json}");
        assert_eq!(
            arguments(&events),
            value!({ "d": wrapped(value!([]), deepest - 2) })
        );
        assert!(malformed(&events).is_empty(), "{events:?}");

        // One element more: the empty array would be the container past the bound.
        let past = output(deepest);
        let events = run_with(schema(deepest), "", &[&past]);
        assert_eq!(bytes(&events), past);
        assert_eq!(names(&events), ["alert"]);
        let json = arguments_of(&events, 0);
        assert!(PartialJson::default().parse(&json, false).is_ok(), "{json}");
        assert_eq!(
            json,
            format!("{{{}", "\"d\": {".repeat(deepest - 1)),
            "the containers read stand open, the element that crossed the bound is not written"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(&json).is_err(),
            "{json} reads as a whole value"
        );
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::InvalidArguments)]
        );
        assert_eq!(refused(&events), format!("<d>{}", "</d>".repeat(deepest)));
        assert_eq!(content(&events), "");
        assert_eq!(ends(&events), 1);
    }

    /// A refused run inside an invoke that read an element before it and one after it: the
    /// members stand as the call's fragments, and nothing is made up for the call's close at
    /// `</invoke>`, so the arguments are a strict JSON prefix and no whole value (a client that
    /// reads them as JSON is told the call failed, where it read a closed object with the word
    /// missing before); the word is counted and never content; whole, one character at a time
    /// and at every cut. The stop makes nothing up for such a call either, nor for a leaf it
    /// found held for its close, so it announces no close; a call read whole announces its close.
    #[test]
    fn a_call_the_syntax_refused_bytes_inside_keeps_its_members_and_gets_no_close() {
        use crate::{json::Prefix, testing::the_same_at_every_cut};
        let output = format!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"alert\">{}\
             ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            separated("<name>x</name>junk<count>3</count>")
        );
        let read = |pieces: &[&str]| run_with(tools(), "", pieces);
        let said = the_same_at_every_cut(|pieces| read(pieces), &output);
        assert_eq!(
            said.calls,
            [(
                "alert".to_string(),
                r#"{"name": "x", "count": 3"#.to_string()
            )]
        );
        assert_eq!(said.content, "");
        let events = read(&[&output]);
        assert_eq!(bytes(&events), output);
        let written = arguments_of(&events, 0);
        assert!(
            Prefix::default().feed(&written).is_none(),
            "{written:?} is no prefix the crate's JSON judge takes"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(&written).is_err(),
            "{written:?} reads as a whole value"
        );
        assert_eq!(
            malformed(&events),
            [(String::new(), MalformedReason::TextBetweenTags)]
        );
        assert_eq!(refused(&events), "junk");
        assert_eq!(ends(&events), 1);
        // The same invoke read whole closes its object.
        let clean = output.replace("junk", "");
        let events = read(&[&clean]);
        assert_eq!(arguments(&events), value!({"name": "x", "count": 3}));

        // The stop: a leaf held for its close or a word the syntax refused leaves the call no
        // whole call, so nothing closes and the stop announces no close.
        for output in [
            "</mm:think><tool_call>\n<invoke name=\"alert\"><count>3",
            "</mm:think><tool_call>\n<invoke name=\"alert\"><name>x</name>junk",
        ] {
            let stopped = run_with(tools(), "", &[output]);
            assert_eq!(bytes(&stopped), output);
            assert!(
                !stopped.iter().any(|event| matches!(
                    event,
                    Event::Malformed {
                        why: MalformedReason::OpenAtTheStop,
                        ..
                    }
                )),
                "{output:?}: {stopped:?}"
            );
            let written = arguments_of(&stopped, 0);
            assert!(
                serde_json::from_str::<serde_json::Value>(&written).is_err(),
                "{output:?}: {written:?} reads as a whole value"
            );
        }
        let whole = run_with(
            tools(),
            "",
            &["</mm:think><tool_call>\n<invoke name=\"alert\"><count>3</count>"],
        );
        assert_eq!(arguments(&whole), value!({"count": 3}));
        assert!(whole.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::OpenAtTheStop,
                ..
            }
        )));
    }

    /// The reasons of the `Malformed` events of no bytes, in order: the refusals counted.
    fn reported(events: &[Event]) -> Vec<MalformedReason> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } if text.text.is_empty() => Some(*why),
                _ => None,
            })
            .collect()
    }

    /// The bytes refused as a malformed call's, concatenated.
    fn refused(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::MalformedCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// A stray closing tag between elements, one that names nothing open, is refused and
    /// counted but costs the call nothing: it carries no member the client could lose, so the
    /// call closes whole. A tag the invoke's end cut short between elements is a member the
    /// model had begun: refused as cut short, the call gets no close, and the stop announces
    /// none for it.
    #[test]
    fn a_stray_close_between_elements_costs_the_call_nothing_and_a_cut_tag_its_close() {
        use crate::testing::the_same_at_every_cut;
        let output = format!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"alert\">{}\
             ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            separated("<name>x</name></count><count>3</count>")
        );
        let read = |pieces: &[&str]| run_with(tools(), "", pieces);
        let said = the_same_at_every_cut(|pieces| read(pieces), &output);
        assert_eq!(
            said.calls,
            [(
                "alert".to_string(),
                r#"{"name": "x", "count": 3}"#.to_string()
            )]
        );
        assert_eq!(said.content, "");
        let events = read(&[&output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(arguments(&events), value!({"name": "x", "count": 3}));
        assert_eq!(reported(&events), [MalformedReason::TagOutOfPlace]);
        assert_eq!(refused(&events), "</count>");
        assert_eq!(ends(&events), 1);

        let cut = "</mm:think><tool_call>\n<invoke name=\"alert\"><name>x</name>\n<cou</invoke>\n</tool_call>";
        let events = read(&[cut]);
        assert_eq!(bytes(&events), cut);
        assert_eq!(arguments_of(&events, 0), r#"{"name": "x""#);
        assert_eq!(reported(&events), [MalformedReason::TagCutShort]);
        assert_eq!(refused(&events), "<cou");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if &*text.text == "\n"
        )));
        assert_eq!(ends(&events), 1);
        let at_the_stop = "</mm:think><tool_call>\n<invoke name=\"alert\"><name>x</name>\n<cou";
        let events = read(&[at_the_stop]);
        assert_eq!(bytes(&events), at_the_stop);
        assert_eq!(arguments_of(&events, 0), r#"{"name": "x""#);
        assert_eq!(reported(&events), [MalformedReason::TagCutShort]);
        assert_eq!(refused(&events), "<cou");
    }

    /// A closing tag that names an open element further out, an inner close left out (`</range>`
    /// missing), closes it out of order: the members after it would land in the wrong container
    /// (`limit` inside `range`), so it is refused as any tag out of its place and the call is no
    /// whole call. A closing tag that names nothing open costs the call nothing inside a
    /// container as at the top.
    #[test]
    fn a_close_that_skips_an_inner_one_is_refused_and_one_that_names_nothing_open_is_free() {
        use crate::testing::the_same_at_every_cut;
        let search = || {
            declared(
                "search",
                value!({"type": "object", "properties": {
                    "filter": {"type": "object", "properties": {
                        "range": {"type": "object", "properties": {"gte": {"type": "integer"}}},
                    }},
                    "limit": {"type": "integer"},
                }}),
            )
        };
        let invoke = |body: &str| {
            format!(
                "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"search\">{}\
                 ]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
                separated(body)
            )
        };
        let read = |pieces: &[&str]| run_with(search(), "", pieces);
        let misnested = invoke("<filter><range><gte>5</gte></filter><limit>10</limit>");
        let said = the_same_at_every_cut(|pieces| read(pieces), &misnested);
        assert_eq!(
            said.calls,
            [(
                "search".to_string(),
                r#"{"filter": {"range": {"gte": 5, "limit": 10"#.to_string()
            )]
        );
        assert_eq!(said.content, "");
        let events = read(&[&misnested]);
        assert_eq!(bytes(&events), misnested);
        let written = arguments_of(&events, 0);
        assert!(
            serde_json::from_str::<serde_json::Value>(&written).is_err(),
            "{written:?} reads as a whole value"
        );
        assert_eq!(reported(&events), [MalformedReason::TagOutOfPlace]);
        assert_eq!(refused(&events), "</filter>");
        assert_eq!(ends(&events), 1);

        let stray = invoke("<filter><range><gte>5</gte></range></bogus></filter><limit>10</limit>");
        let said = the_same_at_every_cut(|pieces| read(pieces), &stray);
        assert_eq!(
            said.calls,
            [(
                "search".to_string(),
                r#"{"filter": {"range": {"gte": 5}}, "limit": 10}"#.to_string()
            )]
        );
        let events = read(&[&stray]);
        assert_eq!(bytes(&events), stray);
        assert_eq!(
            arguments(&events),
            value!({"filter": {"range": {"gte": 5}}, "limit": 10})
        );
        assert_eq!(reported(&events), [MalformedReason::TagOutOfPlace]);
        assert_eq!(refused(&events), "</bogus>");
        assert_eq!(ends(&events), 1);
    }

    /// The draw a fleet verifier once saw on the attributed syntax, on this one: an invoke that
    /// names the tool and writes its arguments as text where the syntax has elements, so it read
    /// no member. The call used to close as `{}`, a well-formed call under `tool_calls` with
    /// nothing on the wire saying it had failed. Now it is no whole call: the syntax writes no
    /// close for it, so its arguments are no JSON value, which a chat client's own reader
    /// refuses and the Responses item says (`incomplete`); the refused bytes are counted and
    /// never content; the Messages block keeps the `{}` a cut call's has, as the adapter's notes
    /// say.
    #[test]
    fn a_call_whose_every_argument_the_syntax_refused_is_no_whole_call_to_any_client() {
        use crate::{adapt, testing::the_same_at_every_cut};
        const OUTPUT: &str = concat!(
            "</mm:think>]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"add_integers\">",
            "a: 17, b: 25]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
        );
        let read = |pieces: &[&str]| run_with(tools(), "", pieces);
        let said = the_same_at_every_cut(|pieces| read(pieces), OUTPUT);
        assert_eq!(said.content, "");
        assert_eq!(said.reasoning, "");
        assert_eq!(said.calls, [("add_integers".to_string(), String::new())]);

        let events = read(&[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::TextBetweenTags } if text.text.is_empty()
        )));
        let refused = refused(&events);
        assert!(
            refused.contains("17") && refused.contains("25"),
            "{refused:?}"
        );

        // Chat Completions: the call, under `tool_calls`, with arguments a strict reader refuses.
        let choice = chat::message(0, &events);
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
}
