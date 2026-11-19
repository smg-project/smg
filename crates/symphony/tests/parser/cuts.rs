//! A call the stream cuts short says so on every call syntax. The shaped syntax (tagged
//! parameters under an invoke tag) marks the cut as the tagged and JSON syntaxes mark theirs:
//! `Malformed` with `UnterminatedRegion` before the call's end, its text the bytes of a parameter
//! the cut left open, never dropped as wrapping.

use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::{
    formats::Table, Declared, DropReason, Engine, EngineFinish, Event, Events, Input,
    MalformedReason, Parser,
};

use crate::common::{delta, prompt};

/// The one declared tool of the reproduction: two string parameters.
fn save_note() -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "save_note".to_string(),
            description: None,
            parameters: Some(json!({
                "type": "object",
                "properties": {"title": {"type": "string"}, "body": {"type": "string"}}
            })),
            strict: None,
            extra: Default::default(),
        },
    }]
}

#[test]
fn a_shaped_call_the_stream_cuts_short_is_marked_cut_before_its_end() {
    // The events of `text` read whole after an empty prompt, the stream ended by a length stop.
    let events_of = |text: &str| -> Vec<Event<'static>> {
        let mut engine = Engine::new(Table::MuseSpark.format(), Declared::of(&save_note()));
        let mut out = Events::new();
        engine.feed(prompt(), &mut out).expect("the prompt");
        engine.feed(delta(text), &mut out).expect("the output");
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Length,
                },
                &mut out,
            )
            .expect("the end");
        out.into_owned().drain()
    };
    let head = " to=save_note<|message|><atem:function_calls>\n<atem:invoke name=\"save_note\">";
    let one_parameter = format!("{head}\n<atem:parameter name=\"title\">Lines</atem:parameter>\n");
    let mid_value = format!("{one_parameter}<atem:parameter name=\"body\">Line o");
    for (text, open) in [
        (head.to_owned(), ""),
        (one_parameter.clone(), ""),
        (mid_value, "<atem:parameter name=\"body\">Line o"),
    ] {
        let events = events_of(&text);
        let end = events
            .iter()
            .position(|event| matches!(event, Event::ToolCallEnd { .. }))
            .expect("the call ends");
        let markers: Vec<&str> = events[..end]
            .iter()
            .filter_map(|event| match event {
                Event::Malformed {
                    text,
                    why: MalformedReason::UnterminatedRegion,
                } => Some(text.text.as_ref()),
                _ => None,
            })
            .collect();
        assert_eq!(
            markers,
            [open],
            "the cut is said once before the end of the call the stream cut, its text the open \
             parameter's bytes: {events:#?}"
        );
        assert!(
            !events.iter().any(|event| matches!(
                event,
                Event::Dropped { text, why: DropReason::Wrapper } if text.text.contains("atem:parameter")
            )),
            "a half-written parameter was dropped as wrapping: {events:#?}"
        );
    }
}
