//! LFM2.5: reasoning between `<think>` and `</think>`, the turn's calls between
//! `<|tool_call_start|>` and `<|tool_call_end|>` as a Python list (`[get_weather(city='Paris')]`),
//! everything else content. The model writes its own `<think>`, and a turn opens with
//! `<|im_start|>assistant`, so the prompt's replay starts there. Recorded as
//! `lfm2.5-1.2b-instruct` (LiquidAI/LFM2.5-1.2B-Instruct).
//!
//! A JSON answer's whitespace is bounded: held to a schema's JSON after a long thought by an
//! engine that starts the grammar at the first token, the model drifted into runs of tabs and
//! newlines inside the object until the budget ended, where its own server, which lets no thought
//! precede the answer, stopped every time. A run of at most thirty-two characters ends a drift of
//! thousands and leaves pretty-printing its room: a newline and four-space indentation to a depth
//! of seven, two-space to fifteen; past that the model writes the next token early, the JSON
//! still valid. The bound goes with the table to every model that reads with it (the d1 family's
//! checkpoints too, on which the drift was not measured).

use crate::{
    format::{CallSyntax, Emits, Format},
    pythonic::Dialect,
};

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
        .calls(CallSyntax::Pythonic(Dialect::LFM2_5))
        .opens_turn("<|im_start|>assistant")
        .bounds_answer_whitespace(32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::Engine, event::Event, testing::drive, Declared};

    fn run(prompt: &str, output: &str) -> Vec<Event<'static>> {
        drive(
            &mut Engine::new(lfm2_5(), Declared::default()),
            prompt,
            &[output],
        )
    }

    fn text_of(events: &[Event], reasoning: bool) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) if reasoning => Some(&*text.text),
                Event::Content(text) if !reasoning => Some(&*text.text),
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
