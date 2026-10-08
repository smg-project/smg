//! LFM2.5: reasoning between `<think>` and `</think>`, the turn's calls between
//! `<|tool_call_start|>` and `<|tool_call_end|>` as a Python list (`[get_weather(city='Paris')]`),
//! everything else content. The model writes its own `<think>`, and a turn opens with
//! `<|im_start|>assistant`, so the prompt's replay starts there. Recorded as
//! `lfm2.5-1.2b-instruct` (LiquidAI/LFM2.5-1.2B-Instruct).

use crate::format::{CallSyntax, Emits, Format};

/// The LFM2.5 table.
pub fn lfm2_5() -> Format {
    Format::new("lfm2.5")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<|tool_call_start|>")
        .terminal("calls_close", "<|tool_call_end|>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Pythonic)
        .opens_turn("<|im_start|>assistant")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, Events},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    fn run(prompt: &str, output: &str) -> Vec<Event> {
        let mut parser = Engine::new(lfm2_5(), Declared::default());
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

    fn text_of(events: &[Event], reasoning: bool) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) if reasoning => Some(text.text.as_str()),
                Event::Content(text) if !reasoning => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_prompt_is_replayed_from_the_turn_opener_the_table_names() {
        // A marker quoted in the user's turn moves nothing: the replay starts at the last
        // `<|im_start|>assistant` (smg #2843, claude[bot]'s probe).
        let prompt = "<|im_start|>user\nWhy did you print <think> there?<|im_end|>\n\
                      <|im_start|>assistant\n";
        let events = run(
            prompt,
            "<think>A thought.</think>The marker opens a thought.",
        );
        assert_eq!(text_of(&events, true), "A thought.");
        assert_eq!(text_of(&events, false), "The marker opens a thought.");
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { .. })));
    }
}
