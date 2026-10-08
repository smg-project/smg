//! GLM: reasoning between `<think>` and `</think>`, each call between `<tool_call>` and
//! `</tool_call>` as its name and keyed arguments (`<arg_key>`, `<arg_value>`), typed by the
//! request's tools, with nothing between the tags. Everything else is content. The prompt opens
//! the thought (`<|assistant|><think>`, whatever the request says: the template has no thinking
//! switch, only `clear_thinking` for the history), so an empty thought writes `</think>` at once;
//! a turn opens with `<|assistant|>`, so the prompt's replay starts there. The template strips the
//! content's edges and writes the calls right after it, so a call follows the content's last byte.
//! Recorded as `glm-5.3-flash` (zai-org/GLM-5.3-Flash). Ling writes the same syntax with newlines
//! between the tags and under its own turn opener ([`ling()`](super::ling())).

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The GLM table.
pub fn glm() -> Format {
    Format::new("glm")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("call", "call_close", "content")
        .transition("call", "call_open", "call")
        .calls(CallSyntax::Keyed(keyed::Tags::PLAIN))
        .opens_turn("<|assistant|>")
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

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "ChaDri_change_drink".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "drink_id": {"type": "string"},
                    "new_preferences": {"type": "object"},
                }}),
                strict: None,
            },
        }])
    }

    /// The generation prompt's tail: the turn opened, and its thought.
    const PROMPT: &str = "<|assistant|><think>";

    /// A recorded output: an empty thought, no content, one call with a string and an object.
    const OUTPUT: &str = concat!(
        "</think><tool_call>ChaDri_change_drink<arg_key>drink_id</arg_key><arg_value>latte",
        "</arg_value><arg_key>new_preferences</arg_key><arg_value>",
        "{\"size\": \"large\", \"temperature\": \"hot\"}</arg_value></tool_call>"
    );

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(glm(), declared());
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

    fn content(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn dropped(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    #[test]
    fn a_recorded_output_gives_its_call_typed_by_the_tools_and_every_byte() {
        let events = run(PROMPT, &[OUTPUT]);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                dropped("</think>"),
                Event::ReasoningEnd,
                dropped("<tool_call>"),
            ]
        );
        assert!(matches!(
            &events[4],
            Event::ToolCallStart { index: 0, name, .. } if name == "ChaDri_change_drink"
        ));
        assert_eq!(
            arguments_of(&events, 0),
            r#"{"drink_id": "latte", "new_preferences": {"size": "large", "temperature": "hot"}}"#
        );
        assert_eq!(content(&events), "");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_thought_and_an_answer_are_the_regions_bytes() {
        let output =
            "Step 1: read.\n\nStep 2: answer.</think>Here it is:\n\n```python\nprint(1)\n```";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..3],
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("Step 1: read.\n\nStep 2: answer.")),
                dropped("</think>"),
            ]
        );
        assert_eq!(content(&events), "Here it is:\n\n```python\nprint(1)\n```");
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }

    #[test]
    fn thinking_off_still_starts_inside_the_thought_the_prompt_opened() {
        // The template has no thinking switch: the generation prompt ends with `<think>` whatever
        // the request says, and the model closes the empty thought at once.
        let events = run(PROMPT, &["</think>Hello!"]);
        assert_eq!(
            events[..4],
            [
                Event::ReasoningStart,
                dropped("</think>"),
                Event::ReasoningEnd,
                Event::Content(Text::uncounted("Hello!")),
            ]
        );
    }

    #[test]
    fn content_runs_to_the_calls_last_byte_and_two_calls_follow_each_other() {
        let output = concat!(
            "</think>Two drinks.<tool_call>ChaDri_change_drink<arg_key>drink_id</arg_key>",
            "<arg_value>123</arg_value></tool_call><tool_call>ChaDri_change_drink",
            "<arg_key>drink_id</arg_key><arg_value>tea</arg_value></tool_call>"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Two drinks.");
        // A declared string keeps the digits as text.
        assert_eq!(arguments_of(&events, 0), r#"{"drink_id": "123"}"#);
        assert_eq!(arguments_of(&events, 1), r#"{"drink_id": "tea"}"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = run(PROMPT, &[OUTPUT]);
        let whole_arguments = arguments_of(&whole, 0);
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let events = run(PROMPT, &[&OUTPUT[..cut], &OUTPUT[cut..]]);
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(arguments_of(&events, 0), whole_arguments, "cut at {cut}");
            assert!(
                matches!(events.last(), Some(Event::Finish { tool_calls: 1, .. })),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_stream_cut_inside_a_call_ends_the_call_with_its_arguments_cut() {
        let events = run(
            PROMPT,
            &["</think><tool_call>ChaDri_change_drink<arg_key>drink_id</arg_key><arg_value>lat"],
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallEnd { index: 0, source } if source.text.is_empty()
        )));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn an_earlier_turns_call_in_the_prompt_moves_nothing() {
        // The replay starts at the last `<|assistant|>`, so an earlier turn's call block, closed or
        // not, and a user's quoted marker are not entered.
        for earlier in [
            "<think></think><tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
            "<think></think><tool_call>f<arg_key>a</arg_key>",
        ] {
            let prompt = format!(
                "<|user|>Hi<|assistant|>{earlier}<|observation|>ok<|user|>Why <tool_call>?{PROMPT}"
            );
            let events = run(&prompt, &["A plan.</think>An answer."]);
            assert_eq!(events[0], Event::ReasoningStart, "{earlier:?}");
            assert_eq!(
                events[1],
                Event::Reasoning(Text::uncounted("A plan.")),
                "{earlier:?}"
            );
            assert_eq!(content(&events), "An answer.", "{earlier:?}");
        }
    }
}
