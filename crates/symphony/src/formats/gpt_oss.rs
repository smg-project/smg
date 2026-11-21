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
//! `commentary` message with no recipient, an `analysis` or `commentary` message to any other
//! recipient, and a message to a tool of the model's own (`to=python`, `to=browser`,
//! `to=container`) on any channel read as the thought. Everything between two messages is
//! framing.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

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
        .terminal("analysis_to", "<|channel|>analysis")
        .terminal("commentary_to", "<|channel|>commentary")
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
        .state("thought_header", Emits::Content)
        .state("aside", Emits::Content)
        .transition("message", "analysis", "reasoning")
        .transition("message", "final", "content")
        .transition("message", "preamble", "reasoning")
        .transition("message", "body", "content")
        .transition("message", "analysis_to", "thought_header")
        .transition("message", "commentary_to", "thought_header")
        .transition("thought_header", "body", "reasoning")
        .transition("thought_header", "call_open", "call")
        .transition("thought_header", "analysis", "reasoning")
        .transition("thought_header", "final", "content")
        .transition("thought_header", "preamble", "reasoning")
        .transition("message", "to_python", "aside")
        .transition("message", "to_browser", "aside")
        .transition("message", "to_container", "aside")
        .transition("thought_header", "to_python", "aside")
        .transition("thought_header", "to_browser", "aside")
        .transition("thought_header", "to_container", "aside")
        .transition("aside", "body", "reasoning")
        .transition("aside", "analysis", "reasoning")
        .transition("aside", "final", "reasoning")
        .transition("aside", "preamble", "reasoning")
        .transition("reasoning", "end", "message")
        .transition("reasoning", "call_close", "message")
        .transition("content", "end", "message")
        .transition("message", "call_open", "call")
        .transition("call", "call_close", "message")
        .transition("call", "end", "message")
        .transition("call", "start", "message")
        .calls(CallSyntax::Shaped(shape::intern(call_shape())))
        .framing("message")
        .framing("thought_header")
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

    /// Each turn reads as the Harmony token parser reads it: the thought, the answer, a call
    /// under either order of its header, and every header whose body is a thought, the stop
    /// token hidden as the stop decoder hides it. The one turn it reads otherwise says so.
    #[test]
    fn a_turn_reads_as_the_harmony_parser_reads_it() {
        let call = ("get_weather", r#"{"city": "Paris"}"#);
        let mut turns = vec![
            (
                "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant\
                 <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>\
                 {\"city\": \"Paris\"}"
                    .into(),
                Some("Need the weather."),
                None,
                vec![call],
            ),
            (
                " to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Paris\"}"
                    .into(),
                None,
                None,
                vec![call],
            ),
            (
                "<|channel|>analysis to=functions.get_weather <|constrain|>json<|message|>\
                 {\"city\": \"Paris\"}"
                    .into(),
                None,
                None,
                vec![call],
            ),
            (
                "<|channel|>commentary to=functions.get-weather.v2 json<|message|>{}".into(),
                None,
                None,
                vec![("get-weather.v2", "{}")],
            ),
            (
                "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final\
                 <|message|>The answer."
                    .into(),
                Some("Think."),
                Some("The answer."),
                vec![],
            ),
            // Not the token parser's read: it takes a start marker in a call's body for the
            // arguments' text. The table ends the call there, with no arguments.
            (
                "<|channel|>commentary to=functions.get_weather json<|message|><|start|>assistant"
                    .into(),
                None,
                None,
                vec![("get_weather", "")],
            ),
            // A preamble, then the call it announces.
            (
                "<|channel|>commentary<|message|>Looking it up.<|end|><|start|>assistant\
                 <|channel|>commentary to=functions.get_weather json<|message|>\
                 {\"city\": \"Paris\"}"
                    .into(),
                Some("Looking it up."),
                None,
                vec![call],
            ),
            // A recipient that is no function and no tool of the model's own: the channel decides.
            (
                "<|channel|>commentary to=repo_browser.print_tree <|constrain|>json<|message|>\
                 {\"path\": \".\"}"
                    .into(),
                Some(r#"{"path": "."}"#),
                None,
                vec![],
            ),
            (
                "<|channel|>analysis to=foo<|message|>Let me think.".into(),
                Some("Let me think."),
                None,
                vec![],
            ),
            // A header that names no channel.
            ("<|message|>Plain.".into(), None, Some("Plain."), vec![]),
        ];
        // The model's own tools, under either order of the header and on every channel.
        for tool in ["python", "browser", "container"] {
            for header in [
                format!(" to={tool}<|channel|>analysis<|message|>"),
                format!("<|channel|>analysis to={tool} code<|message|>"),
                format!(" to={tool}<|channel|>commentary code<|message|>"),
                format!(" to={tool}<|channel|>commentary<|message|>"),
                format!(" to={tool}<|channel|>final<|message|>"),
            ] {
                turns.push((
                    format!(
                        "{header}print(1)<|end|><|start|>assistant<|channel|>final<|message|>Done."
                    ),
                    Some("print(1)"),
                    Some("Done."),
                    vec![],
                ));
            }
        }
        for (output, reasoning, content, calls) in turns {
            let output = output.as_str();
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
