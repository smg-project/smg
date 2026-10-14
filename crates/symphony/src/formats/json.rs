//! JSON calls alone in the turn: the whole output is one call object, `{"name": …, "arguments":
//! {…}}`, or a list of them, or content. The format the gateway's older `json` parser reads for a
//! model that names it. No recorded group writes it as its own format, so the table stands on its
//! unit tests until one does.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.
use crate::format::{CallSyntax, Emits, Format};

/// The JSON table.
pub fn json() -> Format {
    Format::new("json")
        .state("calls", Emits::Arguments)
        .calls(CallSyntax::JsonAlone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, MalformedReason},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn run(pieces: &[&str]) -> Vec<Event> {
        let mut parser = Engine::new(json(), Declared::default());
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
        out.drain()
    }

    fn calls(events: &[Event]) -> Vec<(String, String)> {
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
            Event::Malformed { text, why: MalformedReason::Other(_) } if text.text == "tail"
        )));
        for cut in 1..output.len() {
            let events = run(&[&output[..cut], &output[cut..]]);
            assert_eq!(calls(&events), calls(&whole), "cut at {cut}");
        }
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
