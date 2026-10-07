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

use crate::format::{CallSyntax, Emits, Format};

/// The MiniMax M3 table.
pub fn minimax_m3() -> Format {
    Format::new("minimax_m3")
        .terminal("think_open", "<mm:think>")
        .terminal("think_close", "</mm:think>")
        .terminal("calls_open", "<tool_call>")
        .terminal("calls_close", "</tool_call>")
        .terminal("invoke_open", "<invoke name=\"")
        .terminal("invoke_close", "</invoke>")
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
                }}),
                strict: None,
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
        let output = concat!(
            "</mm:think><tool_call>\n<invoke name=\"alert\"><name>12</name><threshold>5.0",
            "</threshold><count>007</count><enabled>true</enabled><recipients><item>a@x</item>",
            "<item>b@x</item></recipients><points><item><x>1</x><label>7</label></item></points>",
            "<tags></tags><options></options><extra>12</extra>",
            // A list of objects whose key is `item`: the declared array decides the shapes.
            "<orders><item><item>burgers</item><quantity>5</quantity></item></orders>",
            // One space is a value; a key in another script; an empty element under an object
            // the tool declares nothing below is null, the template's spelling of `None` there.
            "<sep> </sep><año>2024</año><meta><deep><item>1</item><item></item></deep></meta>",
            "</invoke>\n</tool_call>"
        );
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
            "</mm:think><tool_call>\n<invoke name=\"f\"><opts><a>1</a><b>tex",
            "</tool_call>The weather is sunny."
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
}
