//! IQuest: reasoning between `<think>` and `</think>`, each call between `<iquest_tool_call>` and
//! `</iquest_tool_call>` as its name and keyed arguments (`<arg_key>`, `<arg_value>`), typed by
//! the request's tools, with no newline anywhere. Everything else is content. The prompt opens
//! the thought. Recorded as `iquest-q1` (IQuestLab/IQuest-Q1).

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::keyed,
};

/// The IQuest table.
pub fn iquest() -> Format {
    Format::new("iquest")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("call_open", "<iquest_tool_call>")
        .terminal("call_close", "</iquest_tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("call", "call_close", "content")
        .transition("call", "call_open", "call")
        .calls(CallSyntax::Keyed(keyed::Tags::PLAIN))
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
        "The user asks.</think><iquest_tool_call>get_weather<arg_key>city</arg_key>",
        "<arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>",
        "</iquest_tool_call><iquest_tool_call>get_weather</iquest_tool_call>"
    );

    fn run(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(iquest(), declared());
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
            let mut parser = Engine::new(iquest(), declared());
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
