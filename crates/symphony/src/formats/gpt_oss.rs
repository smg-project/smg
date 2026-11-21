//! gpt-oss (the Harmony response format): a turn of messages, each a header closed by
//! `<|message|>`, a body, and `<|end|>` (another message follows), `<|call|>` (a call, the
//! turn stops) or `<|return|>` (the answer, the turn stops). The header names a channel after
//! `<|channel|>` and, for a call, a recipient `to=functions.NAME` before or after the channel,
//! then an optional `<|constrain|>json`:
//!
//! ```text
//! <|channel|>analysis<|message|>The user asks about Paris.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Paris"}<|call|>
//! ```
//!
//! The thought is the `analysis` message, the answer the `final` message, and a call a message
//! whose header holds `to=functions.NAME`, on any channel, its body one JSON value. A
//! `commentary` message with no recipient, and a message to a tool of the model's own
//! (`to=python`, `to=browser`, `to=container`), read as the thought. Everything between two
//! messages is framing.

use crate::{
    format::{CallSyntax, Emits, Format},
    shape::{self, Arguments, Atom, CallShape, Class, Dialect, NameSource, Pattern, Slot},
};

/// The most bytes a call's header holds undecided, the matcher's own default.
const HOLD_CAP_BYTES: u32 = 8192;

/// The gpt-oss table.
pub fn gpt_oss() -> Format {
    Format::new("gpt_oss")
        .terminal("analysis", "<|channel|>analysis<|message|>")
        .terminal("final", "<|channel|>final<|message|>")
        .terminal("preamble", "<|channel|>commentary<|message|>")
        .terminal("to_python", "to=python")
        .terminal("to_browser", "to=browser")
        .terminal("to_container", "to=container")
        .terminal("body", "<|message|>")
        .terminal("end", "<|end|>")
        .terminal("start", "<|start|>")
        .terminal("call_open", "to=functions.")
        .terminal("call_close", "<|call|>")
        .state("message", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .state("aside", Emits::Content)
        .transition("message", "analysis", "reasoning")
        .transition("message", "final", "content")
        .transition("message", "preamble", "reasoning")
        .transition("message", "body", "content")
        .transition("message", "to_python", "aside")
        .transition("message", "to_browser", "aside")
        .transition("message", "to_container", "aside")
        .transition("aside", "body", "reasoning")
        .transition("reasoning", "end", "message")
        .transition("reasoning", "call_close", "message")
        .transition("content", "end", "message")
        .transition("message", "call_open", "call")
        .transition("call", "call_close", "message")
        .transition("call", "end", "message")
        .transition("call", "start", "message")
        .calls(CallSyntax::Shaped(shape::intern(call_shape())))
        .framing("message")
        .framing("aside")
        .closes_at_stop("call")
        .turn_anchor("<|start|>assistant")
        .stop_tokens(["<|return|>", "<|call|>"])
}

/// A call's header after `to=functions.`: the tool's name up to a space or the next marker,
/// whatever else the header holds (the channel, `<|constrain|>json`), and `<|message|>`.
fn call_shape() -> CallShape {
    let name = Class::new([' ', '\t', '\n', '\x0C', '\r', '<'].map(|c| (c, c)), true);
    CallShape {
        header: Pattern(
            [
                Atom::Run {
                    class: name,
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                Atom::Run {
                    class: Class::any(),
                    min: 0,
                    max: None,
                    greedy: false,
                    capture: None,
                },
                Atom::Lit("<|message|>".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Json(Dialect::default()),
        trailer: Pattern::default(),
        word: Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false),
        hold_cap: HOLD_CAP_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adapt, engine::Engine, testing::drive, Declared};

    /// Each turn reads as the Harmony token parser reads it: the thought, the answer, and a
    /// call under either order of its header, the stop token hidden as the stop decoder hides it.
    #[test]
    fn a_turn_reads_as_the_harmony_parser_reads_it() {
        let call = ("get_weather", r#"{"city": "Paris"}"#);
        let turns = [
            (
                "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant\
                 <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>\
                 {\"city\": \"Paris\"}",
                Some("Need the weather."),
                None,
                &[call][..],
            ),
            (
                " to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Paris\"}",
                None,
                None,
                &[call][..],
            ),
            (
                "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final\
                 <|message|>The answer.",
                Some("Think."),
                Some("The answer."),
                &[][..],
            ),
            (
                "<|channel|>commentary to=functions.get_weather json<|message|><|start|>assistant",
                None,
                None,
                &[("get_weather", "")][..],
            ),
        ];
        for (output, reasoning, content, calls) in turns {
            let mut engine = Engine::new(gpt_oss(), Declared::default());
            let events = drive(
                &mut engine,
                "<|start|>user<|message|>Hi<|end|><|start|>assistant",
                &[output],
            );
            let message = adapt::chat::message(0, &events).message;
            assert_eq!(message.reasoning_content.as_deref(), reasoning, "{output}");
            assert_eq!(message.content.as_deref(), content, "{output}");
            let read: Vec<(String, String)> = message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    (
                        call.function.name,
                        call.function.arguments.unwrap_or_default(),
                    )
                })
                .collect();
            let expected: Vec<(String, String)> = calls
                .iter()
                .map(|(n, a)| (n.to_string(), a.to_string()))
                .collect();
            assert_eq!(read, expected, "{output}");
        }
    }
}
