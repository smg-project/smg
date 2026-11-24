//! Olmo 3: the turn's calls between `<function_calls>` and `</function_calls>`, written as
//! Python, one per line (`get_weather(city="Paris")`); everything else content, and no thought.
//! A turn opens with `<|im_start|>assistant`, so the prompt's replay starts there. Recorded as
//! `olmo-3-7b-instruct` (allenai/Olmo-3-7B-Instruct).

use crate::{
    format::{CallSyntax, Emits, Format},
    pythonic::Dialect,
};

/// The Olmo 3 table.
pub fn olmo3() -> Format {
    Format::new("olmo3")
        .terminal("calls_open", "<function_calls>")
        .terminal("calls_close", "</function_calls>")
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("content", "calls_open", "calls")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Pythonic(Dialect::OLMO3))
        .opens_turn("<|im_start|>assistant")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{Event, MalformedReason},
        formats::{lfm2_5, llama4},
        testing::drive,
        Declared,
    };

    fn run(prompt: &str, output: &str) -> Vec<Event<'static>> {
        drive(
            &mut Engine::new(olmo3(), Declared::default()),
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
        let prompt = "<|im_start|>user\nWhy did you print <function_calls> there?<|im_end|>\n\
                      <|im_start|>assistant\n";
        let events = run(prompt, "The marker opens a calls block.");
        assert_eq!(text_of(&events, true), "");
        assert_eq!(text_of(&events, false), "The marker opens a calls block.");
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { .. })));
    }

    #[test]
    fn a_stray_sentence_where_a_call_belongs_is_text_between_calls() {
        // Words no call follows are prose, the reason a sentence between calls has in every
        // table; a name alone on its line still names a tool and writes no call. One reader
        // serves the three pythonic tables, so each has a row.
        let text = MalformedReason::TextBetweenCalls;
        let name = MalformedReason::NameWithoutACall;
        let chatml = "<|im_start|>assistant\n";
        let rows = [
            (
                olmo3(),
                chatml,
                "<function_calls>\nParis is sunny today.\n</function_calls>",
                vec![text; 4],
            ),
            (
                olmo3(),
                chatml,
                "<function_calls>\nget_weather\n</function_calls>",
                vec![name],
            ),
            // The prose ends with its line: the name on the next line stands alone again.
            (
                olmo3(),
                chatml,
                "<function_calls>\nParis is sunny\nget_weather\n</function_calls>",
                vec![text, text, text, name],
            ),
            // The list of LFM2.5: the last word ends at the `]`, not at a newline.
            (
                lfm2_5(),
                chatml,
                "<|tool_call_start|>[Paris is sunny today.]<|tool_call_end|>",
                vec![text; 4],
            ),
            // The bare list of Llama 4, once a call opened it.
            (
                llama4(),
                "<|header_start|>assistant<|header_end|>\n\n",
                "[get_weather(city=\"Paris\"), Paris is sunny]",
                vec![text; 3],
            ),
        ];
        for (format, prompt, output, expected) in rows {
            let events = drive(
                &mut Engine::new(format, Declared::default()),
                prompt,
                &[output],
            );
            let reasons: Vec<MalformedReason> = events
                .iter()
                .filter_map(|event| match event {
                    Event::Malformed { why, .. } => Some(*why),
                    _ => None,
                })
                .collect();
            assert_eq!(reasons, expected, "{output}");
        }
    }
}
