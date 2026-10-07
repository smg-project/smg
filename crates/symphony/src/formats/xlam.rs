//! xLAM: the whole output is a JSON list of calls, `[{"name": …, "arguments": {…}}, …]`, or
//! content, with no marker either way. The table has one state, an arguments state the output
//! starts in, and the list assembler ([`json::list`](crate::json::list)) decides at the first byte
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
        event::{Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn run(output: &str) -> Vec<Event> {
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
        out.drain()
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
