//! LFM2.5: reasoning between `<think>` and `</think>`, the turn's calls between
//! `<|tool_call_start|>` and `<|tool_call_end|>` as a Python list (`[get_weather(city='Paris')]`),
//! everything else content. The model writes its own `<think>`. Recorded as
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
}
