//! Olmo 3: the turn's calls between `<function_calls>` and `</function_calls>`, written as
//! Python, one per line (`get_weather(city="Paris")`); everything else content, and no thought.
//! Recorded as `olmo-3-7b-instruct` (allenai/Olmo-3-7B-Instruct).

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
}
