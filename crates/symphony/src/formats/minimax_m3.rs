//! MiniMax M3: reasoning between `<mm:think>` and `</mm:think>`, the calls of one turn between
//! `<tool_call>` and `</tool_call>`, each call between `<invoke name="…">` and `</invoke>` as an
//! XML tree of arguments, everything else content, and the separator token `]<]minimax[>[` before
//! every tag, dropped wherever it stands. As recorded for `MiniMaxAI/MiniMax-M3`.
//!
//! What this table says beyond the DSML one:
//!
//! - **An ignored terminal.** The template writes `]<]minimax[>[` before `<tool_call>`, before
//!   every tag inside an invoke and before `</tool_call>`; it is one token, and it never stands
//!   inside prose. The table ignores it: the engine drops it in every state and moves nowhere,
//!   and inside an invoke the assembler keeps its bytes in place.
//! - **The model writes the thought, or closes an empty one.** The template renders an assistant
//!   turn as `<mm:think>…</mm:think>` when it has reasoning and as `</mm:think>` alone when it has
//!   none, and its generation prompt adds nothing in its default mode (`thinking_mode` unset or
//!   `adaptive`), so an output starts with one or the other. The table starts in a `start` state
//!   whose text is content: `<mm:think>` there opens the thought and `</mm:think>` there closes
//!   the empty one, both dropped; later in the content both are text. With `thinking_mode` set,
//!   the generation prompt ends with `<mm:think>` (`enabled`) or `</mm:think>` (`disabled`) after
//!   the turn opener `]~b]ai\n`, and the replay puts the output inside the thought or in content.
//! - **A calls block ends the thought** (`reasoning + calls_open = calls`), as the design's table
//!   has it; the recorded outputs close the thought first.
//! - **Two states for the calls, as for DSML:** `calls` is the template's wrapping (a newline after
//!   `<tool_call>` and after `</invoke>`), `invoke` one call's arguments, read by the XML assembler
//!   ([`tagged::xml`](crate::tagged::xml)); a second invoke before the first closed ends the first,
//!   and the block's close ends an invoke still open.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::xml,
};

/// The MiniMax M3 table.
pub fn minimax_m3() -> Format {
    Format::new("minimax_m3")
        .terminal("think_open", "<mm:think>")
        .terminal("think_close", "</mm:think>")
        .terminal("calls_open", "<tool_call>")
        .terminal("calls_close", "</tool_call>")
        .terminal("invoke_open", "<invoke name=\"")
        .terminal("invoke_close", xml::INVOKE_CLOSE)
        .ignores("separator", "]<]minimax[>[")
        .state("start", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("calls", Emits::Wrapper)
        .state("invoke", Emits::Arguments)
        .transition("start", "think_open", "reasoning")
        .transition("start", "think_close", "content")
        .transition("reasoning", "think_close", "content")
        .transition("start", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("content", "calls_open", "calls")
        .transition("calls", "invoke_open", "invoke")
        .transition("invoke", "invoke_close", "calls")
        .transition("invoke", "invoke_open", "invoke")
        .transition("invoke", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Xml)
        .opens_turn("]~b]ai\n")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    const SEP: &str = "]<]minimax[>[";

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
                parameters: value!({"type": "object", "properties": {
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
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    fn run_with(declared: Declared, prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(minimax_m3(), declared);
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

    /// One function with `parameters`, as a request declares it.
    fn declared(name: &str, parameters: serde_json::Value) -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
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

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        run_with(Declared::default(), prompt, pieces)
    }

    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
    }

    fn arguments_of(events: &[Event], call: u32) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { index, json, .. } if *index == call => {
                    Some(json.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn fragments_of(events: &[Event], call: u32) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { index, json, .. } if *index == call => {
                    Some(json.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn names(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect()
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

    fn ends(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, Event::ToolCallEnd { .. }))
            .count()
    }

    fn dropped(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
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
        assert!(events.contains(&dropped(SEP)));
        let held = format!("<drink_id>latte{SEP}</drink_id>");
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
            SEP,
            "</name></invoke>\n</tool_call>",
        ];
        let events = run_with(tools(), "", &pieces);
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(
            fragments_of(&events, 0),
            [r#"{"name": ""#, "Pa", "ris", "\"", "}"]
        );
        assert!(events.contains(&dropped(SEP)));
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
        let events = run("", &["</mm:think>Say <mm:think>x</mm:think>", SEP, " done"]);
        assert_eq!(content(&events), "Say <mm:think>x</mm:think> done");
        assert!(events.contains(&dropped(SEP)));

        // No marker at all: content.
        let events = run("", &["Hello."]);
        assert_eq!(content(&events), "Hello.");
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
        // The open object closes, and the leaf never written is reported.
        assert_eq!(arguments_of(&events, 0), r#"{"opts": {"a": 1}}"#);
        assert_eq!(ends(&events), 1);
        assert_eq!(content(&events), "The weather is sunny.");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, .. } if text.text == "<b>tex"
        )));
    }

    #[test]
    fn a_stream_cut_inside_an_invoke_ends_the_call_with_its_arguments_cut() {
        let output = "</mm:think><tool_call>\n<invoke name=\"alert\"><name>Par";
        let events = run_with(tools(), "", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(fragments_of(&events, 0), [r#"{"name": ""#, "Par"]);
        assert_eq!(ends(&events), 1);
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
    fn text_that_is_not_a_tag_is_a_values_text_and_text_between_tags_is_reported() {
        let output = concat!(
            "<tool_call>\n<invoke name=\"f\"><expr>a < b and c > d</expr>junk<n>1</n>",
            "<q>a <b> c</q></invoke>\n</tool_call>"
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"expr": "a < b and c > d", "n": 1, "q": "a <b> c"}"#
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, .. } if text.text == "junk"
        )));
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
        // character: the tail is reported whole, and the call starts (smg #2850, claude[bot]).
        let output = concat!(
            "</mm:think><tool_call>\n<invoke name=\"add\" ñ]<]minimax[>[x><a>1</a></invoke>\n",
            "</tool_call>"
        );
        let events = run("", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(names(&events), ["add"]);
        assert_eq!(arguments_of(&events, 0), r#"{"a": 1}"#);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, .. } if text.text == " ñ]<]minimax[>[x"
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
                Event::ToolCallEnd { source, .. } => Some(source.text.as_str()),
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
        // An invoke that named no call is reported with whichever terminal ended it.
        let output = "<tool_call>\n<invoke name=\"</tool_call>";
        let events = run("", &[output]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, .. } if text.text.ends_with("</tool_call>")
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn a_tag_the_invokes_close_cut_short_inside_a_string_is_reported() {
        let output = concat!(
            "</mm:think><tool_call>\n<invoke name=\"alert\"><name>Par</na</invoke>\n",
            "</tool_call>"
        );
        let events = run_with(tools(), "", &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(arguments_of(&events, 0), r#"{"name": "Par"}"#);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, .. } if text.text == "</na"
        )));
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
}
