//! Olmo 3: the turn's calls between `<function_calls>` and `</function_calls>`, written as
//! Python, one per line (`get_weather(city="Paris")`); everything else content, and no thought.
//! A turn opens with `<|im_start|>assistant`, so the prompt's replay starts there. Recorded as
//! `olmo-3-7b-instruct` (allenai/Olmo-3-7B-Instruct).

use crate::format::{CallSyntax, Emits, Format};

/// The Olmo 3 table.
pub fn olmo3() -> Format {
    Format::new("olmo3")
        .terminal("calls_open", "<function_calls>")
        .terminal("calls_close", "</function_calls>")
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
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
        let mut parser = Engine::new(olmo3(), Declared::default());
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
        let prompt = "<|im_start|>user\nWhy did you print <function_calls> there?<|im_end|>\n\
                      <|im_start|>assistant\n";
        let events = run(prompt, "The marker opens a calls block.");
        assert_eq!(text_of(&events, true), "");
        assert_eq!(text_of(&events, false), "The marker opens a calls block.");
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { .. })));
    }
}
