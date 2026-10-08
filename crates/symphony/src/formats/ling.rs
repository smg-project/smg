//! Ling: reasoning between `<think>` and `</think>`, each call between `<tool_call>` and
//! `</tool_call>` as its name and keyed arguments (`<arg_key>`, `<arg_value>`), typed by the
//! request's tools, with the template's newline after the name and after `</arg_key>`. Everything
//! else is content. The prompt opens the thought, and a turn opens with `<role>ASSISTANT</role>`,
//! so the prompt's replay starts there. Recorded as `ling-3.0-flash` (inclusionAI/Ling-3.0-flash);
//! GLM 4.5 and 4.6 write the same syntax; GLM 4.7 and later write it with nothing between the
//! tags, under their own turn opener ([`glm()`](super::glm())).

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The Ling table.
pub fn ling() -> Format {
    Format::new("ling")
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
        .calls(CallSyntax::Keyed(keyed::Tags::LING))
        .opens_turn("<role>ASSISTANT</role>")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                }}),
                strict: None,
            },
        }])
    }

    const PROMPT: &str = "<think>\n";
    const OUTPUT: &str = concat!(
        "The user asks.</think><tool_call>get_weather\n<arg_key>city</arg_key>\n",
        "<arg_value>Paris</arg_value><arg_key>days</arg_key>\n<arg_value>3</arg_value>\n",
        "</tool_call>\n<tool_call>get_weather\n</tool_call>"
    );

    fn run(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(ling(), declared());
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
                    text: output,
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

    fn arguments(events: &[Event]) -> Vec<(String, String)> {
        let mut calls: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { name, .. } => calls.push((name.clone(), String::new())),
                Event::ToolCallArguments { json, .. } => {
                    if let Some(last) = calls.last_mut() {
                        last.1.push_str(json);
                    }
                }
                _ => {}
            }
        }
        calls
    }

    #[test]
    fn the_prompt_is_replayed_from_lings_own_turn_opener() {
        // A call marker quoted in the user's turn, then the generation prompt opening the thought:
        // the replay starts at Ling's own turn opener, so the output is the thought and then
        // content (smg #2842, Alex's probe with the rendered prompt).
        let prompt = "user: Why did you print <tool_call> there?\n<role>ASSISTANT</role><think>\n";
        let output = "The user asks about the tag.</think>It opens a call.";
        let events = run(prompt, output);
        assert_eq!(events[0], Event::ReasoningStart);
        assert_eq!(bytes(&events), output);
        assert!(arguments(&events).is_empty());
        let reasoning: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "The user asks about the tag.");
        assert_eq!(content, "It opens a call.");
    }

    #[test]
    fn a_recorded_output_gives_its_calls_typed_by_the_tools_and_every_byte() {
        let events = run(PROMPT, OUTPUT);
        assert_eq!(bytes(&events), OUTPUT);
        assert_eq!(
            arguments(&events),
            [
                (
                    "get_weather".to_string(),
                    r#"{"city": "Paris", "days": 3}"#.to_string()
                ),
                ("get_weather".to_string(), "{}".to_string()),
            ]
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 2, .. })
        ));
        let reasoning: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "The user asks.");
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = arguments(&run(PROMPT, OUTPUT));
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            let mut parser = Engine::new(ling(), declared());
            let mut out = Events::new();
            parser
                .feed(
                    Input::Prompt {
                        token_ids: &[],
                        text: PROMPT,
                    },
                    &mut out,
                )
                .expect("prompt");
            for piece in [&OUTPUT[..cut], &OUTPUT[cut..]] {
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
            assert_eq!(bytes(&events), OUTPUT, "cut at {cut}");
            assert_eq!(arguments(&events), whole, "cut at {cut}");
        }
    }
}
