//! Qwen2.5: each tool call between `<tool_call>` and `</tool_call>` as a JSON object, everything
//! else content. The template has no thought, so `<think>` and `</think>` are text where the model
//! writes them, which is what sets this table apart from [`qwen3()`](crate::formats::qwen3()): one
//! family, two tables, and the engine between them unchanged.
//!
//! Qwen2.5-Instruct, Qwen2.5-VL and Qwen2.5-Omni share the table; bellwether records them as
//! `qwen2.5-*`.

use crate::format::{CallSyntax, Emits, Format};

/// The Qwen2.5 table.
pub fn qwen2_5() -> Format {
    Format::new("qwen2.5")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("content", "call_open", "calls")
        .transition("calls", "call_close", "content")
        .transition("calls", "call_open", "calls")
        .calls(CallSyntax::Json)
        .opens_turn("<|im_start|>assistant")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    const CALL: &str = concat!(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
        "\n</tool_call>"
    );

    fn run(output: &str) -> Vec<Event> {
        let mut parser = Engine::new(qwen2_5(), Declared::default());
        let mut out = Events::new();
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
    fn a_think_block_is_content_and_a_call_is_a_call() {
        let events = run(&format!("<think>\nplan\n</think>\n\n{CALL}"));
        assert_eq!(
            events[..2],
            [
                Event::Content(Text::uncounted("<think>\nplan\n</think>\n\n")),
                Event::Dropped {
                    text: Text::uncounted("<tool_call>"),
                    why: DropReason::Wrapper,
                },
            ]
        );
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["get_weather"]);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

    #[test]
    fn a_prompt_that_opens_a_thought_opens_nothing_here() {
        let mut parser = Engine::new(qwen2_5(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: "<|im_start|>assistant\n<think>\n",
                },
                &mut out,
            )
            .expect("prompt");
        assert!(
            out.is_empty(),
            "no state to enter: the table has no thought"
        );
    }
}
