//! Step 3: the thought the prompt opens and `</think>` closes, then the answer, or the calls in
//! one block between `<｜tool_calls_begin｜>` and `<｜tool_calls_end｜>`. Each call stands between
//! `<｜tool_call_begin｜>` and `<｜tool_call_end｜>`: the word `function`, `<｜tool_sep｜>`, and one
//! invoke whose parameters are tags of their own, `<steptml:parameter name="city">Paris`
//! and `</steptml:parameter>`, as Step 3's chat template spells the block to the model:
//!
//! ```text
//! <｜tool_calls_begin｜>
//! <｜tool_call_begin｜>
//! function<｜tool_sep｜><steptml:invoke name="get_weather">…</steptml:invoke><｜tool_call_end｜>
//! <｜tool_calls_end｜>
//! ```
//!
//! The format of the gateway's older `step3` parsers, whose tests write `function` right behind
//! `<｜tool_call_begin｜>` and each tag on a line of its own: a call's opener is a terminal in
//! both spellings, and the table reads both. A value is typed by what the request's tool
//! declares, as MiniMax M2's is.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    tagged::attributed,
};

/// The Step 3 table.
pub fn steptml() -> Format {
    Format::new("steptml")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("calls_open", "<｜tool_calls_begin｜>")
        .terminal("calls_close", "<｜tool_calls_end｜>")
        .terminal(
            "invoke_open",
            "<｜tool_call_begin｜>\nfunction<｜tool_sep｜><steptml:invoke name=\"",
        )
        .terminal(
            "invoke_open_inline",
            "<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name=\"",
        )
        .terminal("invoke_close", attributed::Tags::STEPTML.invoke_close)
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("calls", Emits::Wrapper)
        .state("invoke", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "calls_open", "calls")
        .transition("reasoning", "calls_open", "calls")
        .transition("calls", "invoke_open", "invoke")
        .transition("calls", "invoke_open_inline", "invoke")
        .transition("invoke", "invoke_close", "calls")
        .transition("invoke", "invoke_open", "invoke")
        .transition("invoke", "invoke_open_inline", "invoke")
        .transition("invoke", "calls_close", "content")
        .transition("calls", "calls_close", "content")
        .calls(CallSyntax::Attributed(attributed::Tags::STEPTML))
        .opens_turn("<|BOT|>assistant\n")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{
        engine::Engine,
        testing::{calls, content, drive, reasoning},
        Declared,
    };

    #[test]
    fn both_spellings_of_a_call_read_as_the_calls_behind_the_thought() {
        let tools = [Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                }})),
                strict: None,
                extra: Default::default(),
            },
        }];
        let run = |output: &str| {
            drive(
                &mut Engine::new(steptml(), Declared::of(&tools)),
                "<|BOT|>assistant\n<think>\n",
                &[output],
            )
        };
        // The template's spelling: a newline before `function`, a call's tags on one line.
        let events = run(
            "Two lookups.\n</think>\n<｜tool_calls_begin｜>\n<｜tool_call_begin｜>\nfunction<｜tool_sep｜>\
             <steptml:invoke name=\"get_weather\"><steptml:parameter name=\"city\">Paris\
             </steptml:parameter><steptml:parameter name=\"days\">3</steptml:parameter>\
             </steptml:invoke><｜tool_call_end｜>\n<｜tool_call_begin｜>\nfunction<｜tool_sep｜>\
             <steptml:invoke name=\"get_weather\"><steptml:parameter name=\"city\">Rome\
             </steptml:parameter></steptml:invoke><｜tool_call_end｜>\n<｜tool_calls_end｜>",
        );
        assert_eq!(reasoning(&events), "Two lookups.\n");
        assert_eq!(
            calls(&events),
            [
                (
                    "get_weather".to_string(),
                    "{\"city\": \"Paris\", \"days\": 3}".to_string()
                ),
                (
                    "get_weather".to_string(),
                    "{\"city\": \"Rome\"}".to_string()
                ),
            ]
        );
        assert_eq!(content(&events), "");
        // The older parser's tests: `function` right behind the opener, each tag on its own
        // line; a declared string that looks like a number stays a string.
        let events = run(
            "Short.\n</think>\nLet me check.\n<｜tool_calls_begin｜>\n<｜tool_call_begin｜>function\
             <｜tool_sep｜><steptml:invoke name=\"get_weather\">\n<steptml:parameter name=\"city\">75001\
             </steptml:parameter>\n<steptml:parameter name=\"days\">3</steptml:parameter>\n\
             </steptml:invoke><｜tool_call_end｜>\n<｜tool_calls_end｜>",
        );
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"75001\", \"days\": 3}".to_string()
            )]
        );
        assert_eq!(content(&events), "\nLet me check.\n");
    }
}
