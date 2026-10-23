//! xLAM: the whole output is a JSON list of calls, `[{"name": …, "arguments": {…}}, …]`, or
//! content, with no marker either way. The table has one state, an arguments state the output
//! starts in, and the list assembler (`json::list`) decides at the first byte
//! that is not whitespace. Recorded as `llama-xlam-2-8b-fc-r` (Salesforce/Llama-xLAM-2-8b-fc-r).

use crate::format::{CallSyntax, Emits, Format};

/// The xLAM table.
pub fn xlam() -> Format {
    Format::new("xlam")
        .state("calls", Emits::Arguments)
        .calls(CallSyntax::JsonList)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events, MalformedReason, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        Declared,
    };

    fn run(output: &str) -> Vec<Event<'static>> {
        let mut parser = Engine::new(xlam(), Declared::default());
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
        out.into_owned().drain()
    }

    /// A call whose arguments are not JSON carries them as the model wrote them and says once,
    /// with no bytes, that they are no JSON; nothing of the list leaks into content.
    #[test]
    fn arguments_that_are_not_json_stay_the_calls_own_and_are_said_to_be_no_json() {
        let list = "[{\"name\": \"plot_sine_wave\", \"arguments\": {\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}}]";
        let events = run(list);
        let arguments: String = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            arguments,
            "{\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}"
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::Content(_))));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Malformed { .. }))
                .collect::<Vec<_>>(),
            [&Event::Malformed {
                text: Text::default(),
                why: MalformedReason::InvalidArguments,
            }]
        );
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_list_is_calls_and_prose_is_content() {
        let events = run(r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
        let started = &events[1];
        assert!(
            matches!(started, Event::ToolCallStart { index: 0, name, .. } if name == "get_weather")
        );

        let events = run("Hello!");
        assert_eq!(events[0], Event::Content(Text::uncounted("Hello!")));
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 0, .. })
        ));
    }
}
