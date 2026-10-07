//! Seed-OSS: the Qwen tagged syntax under its own markers. Reasoning between `<seed:think>` and
//! `</seed:think>`, each call between `<seed:tool_call>` and `</seed:tool_call>` as
//! `<function=NAME>` and `<parameter=KEY>` tags, typed by the request's tools (which reach the
//! engine with the request), everything else content. The same five rows as
//! [`qwen3()`](crate::formats::qwen3()), with the spellings ByteDance-Seed/Seed-OSS-36B-Instruct
//! writes; a plain `<tool_call>` is text here. The template is not ChatML: a turn opens with
//! `<seed:bos>assistant`, so the prompt's replay starts there.

use crate::format::{CallSyntax, Emits, Format};

/// The Seed-OSS table.
pub fn seed_oss() -> Format {
    Format::new("seed_oss")
        .terminal("think_open", "<seed:think>")
        .terminal("think_close", "</seed:think>")
        .terminal("call_open", "<seed:tool_call>")
        .terminal("call_close", "</seed:tool_call>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "calls")
        .transition("calls", "call_close", "content")
        .transition("calls", "call_open", "calls")
        .calls(CallSyntax::Tagged)
        .opens_turn("<seed:bos>assistant")
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
        "<seed:tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
        "</function>\n</seed:tool_call>"
    );

    #[test]
    fn the_prompt_is_replayed_from_seeds_own_turn_opener() {
        // A stray `<seed:think>` in the user's turn: the replay starts at `<seed:bos>assistant`,
        // so the answer is content (smg #2841, Alex's probe).
        let prompt = "<seed:bos>user\nWhy did you print <seed:think> there?<seed:eos>\
                      <seed:bos>assistant\n";
        let mut parser = Engine::new(seed_oss(), Declared::default());
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
                    text: "It opens a thought.",
                    spans: &[],
                },
                &mut out,
            )
            .expect("delta");
        let events = out.drain();
        assert_eq!(
            events[0],
            Event::Content(Text::uncounted("It opens a thought."))
        );
    }

    #[test]
    fn seeds_markers_are_read_and_qwens_are_text() {
        let output = format!("<seed:think>plan</seed:think>Sure.\n{CALL}<tool_call>x</tool_call>");
        let mut parser = Engine::new(seed_oss(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: &output,
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
        let events = out.drain();
        assert_eq!(
            events[..3],
            [
                Event::Dropped {
                    text: Text::uncounted("<seed:think>"),
                    why: DropReason::Wrapper,
                },
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("plan")),
            ]
        );
        let arguments: String = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(arguments, r#"{"city": "Paris"}"#);
        assert_eq!(
            events
                .last()
                .map(|e| matches!(e, Event::Finish { tool_calls: 1, .. })),
            Some(true)
        );
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "Sure.\n<tool_call>x</tool_call>");
    }
}
