//! The JSON family's spellings: a call as the call markers around one JSON object (`json`),
//! xLAM's and Llama Nemotron's lists of such objects, Inkling's named JSON and Mistral's named
//! arguments. Each builder writes the tag the family's assembler reads back, the tool's schema as
//! the arguments' grammar.

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{CallMarkers, Count, Grammar, Tag};
use crate::format::{Emits, Format, Whitespace};

/// The JSON family: each call is the call markers around one JSON object whose `name` is the
/// tool's and whose `arguments` the tool's schema accepts, the template's newlines included.
pub(super) fn calls(markers: &CallMarkers<'_>, tools: &[&Tool], count: Count<'_>) -> Grammar {
    let tags = tools
        .iter()
        .map(|tool| {
            let name = Value::String(tool.function.name.clone());
            Tag::new(
                format!("{}\n{{\"name\": {name}, \"arguments\": ", markers.call_open),
                Grammar::JsonSchema {
                    schema: tool.function.parameters.clone(),
                    style: None,
                    excludes: Vec::new(),
                    whitespace: Whitespace::Free,
                },
                format!("}}\n{}", markers.call_close),
            )
        })
        .collect();
    count.triggered(vec![markers.call_open.to_string()], tags)
}

/// Inkling's named JSON: the tag the gateway's old Inkling parser hand-writes. The marker that
/// says a message is a call triggers it, inside whatever message the model is writing, the
/// first one included: the marker, `{"name":"NAME","args":` as the template writes it, the
/// arguments the tool's schema accepts, `}` and the message's close (the terminal of the row
/// from the turn's own state, where a message starts, to the wrapper state a closed message
/// leaves the engine in). The tool's name before the marker stays free text: the assembler
/// takes the call's name from the object and drops the header as the template's framing. The
/// messages before the call (a thought, the template's empty answer) are free text too, and a
/// thought is a message of its own, so the tag is the same whether the prompt left the model
/// inside one or not. The tag is wider than the parser at one point, as the gateway's is: a
/// marker outside a message's header (after an answer's text, or after a message's close with
/// no opener between) is writable here and reads as that message's text, so an output written
/// there satisfies `at_least_one` and yields no call; the model does not write it, the
/// sequence being off its training. `None` for a table whose messages have no close.
pub(super) fn named_calls(
    format: &Format,
    marker: &str,
    tools: &[&Tool],
    count: Count<'_>,
) -> Option<Grammar> {
    let (_, close, _) = format
        .transitions()
        .find(|&(source, _, target)| source == 0 && format.emits(target) == Emits::Wrapper)?;
    let close = format.terminal_text(close);
    let tags = tools
        .iter()
        .map(|tool| {
            let quoted = Value::String(tool.function.name.clone());
            Tag::new(
                format!("{marker}{{\"name\":{quoted},\"args\":"),
                Grammar::JsonSchema {
                    schema: tool.function.parameters.clone(),
                    style: None,
                    excludes: Vec::new(),
                    whitespace: Whitespace::Free,
                },
                format!("}}{close}"),
            )
        })
        .collect();
    Some(count.triggered(vec![marker.to_string()], tags))
}

/// Mistral's named arguments: the call's opener triggers a tag per tool, the opener, the
/// tool's name and the marker as its `begin`, the arguments the tool's schema accepts as its
/// content, and nothing as its `end`, since the template writes no closing marker: the object
/// the schema accepts ends the tag, and the next opener or the end of the turn follows. The
/// opener is the terminal of the row from the turn's own state into the arguments state;
/// `None` for a table without one.
pub(super) fn named_arguments_calls(
    format: &Format,
    marker: &str,
    tools: &[&Tool],
    count: Count<'_>,
) -> Option<Grammar> {
    let (_, open, _) = format
        .transitions()
        .find(|&(source, _, target)| source == 0 && format.emits(target) == Emits::Arguments)?;
    let open = format.terminal_text(open);
    let tags = tools
        .iter()
        .map(|tool| {
            Tag::new(
                format!("{open}{}{marker}", tool.function.name),
                Grammar::JsonSchema {
                    schema: tool.function.parameters.clone(),
                    style: None,
                    excludes: Vec::new(),
                    whitespace: Whitespace::Free,
                },
                "",
            )
        })
        .collect();
    Some(count.triggered(vec![open.to_string()], tags))
}

/// The list of the tools' calls as xLAM and Llama Nemotron write it: `[`, each call as
/// `{"name": NAME, "arguments": ` and the tool's schema, `, ` between calls, `]`; with one
/// call, that call alone, and after the `]` what the count writes after the last call.
fn list(tools: &[&Tool], count: Count<'_>) -> Grammar {
    let one_call = Grammar::Or(
        tools
            .iter()
            .map(|tool| {
                let name = Value::String(tool.function.name.clone());
                Grammar::Sequence(vec![
                    Grammar::ConstString(format!("{{\"name\": {name}, \"arguments\": ")),
                    Grammar::JsonSchema {
                        schema: tool.function.parameters.clone(),
                        style: None,
                        excludes: Vec::new(),
                        whitespace: Whitespace::Free,
                    },
                    Grammar::ConstString("}".to_string()),
                ])
            })
            .collect(),
    );
    let mut list = vec![Grammar::ConstString("[".to_string()), one_call.clone()];
    if !count.one {
        list.push(Grammar::Star(Box::new(Grammar::Sequence(vec![
            Grammar::ConstString(", ".to_string()),
            one_call,
        ]))));
    }
    list.push(Grammar::ConstString(count.end("]")));
    Grammar::Sequence(list)
}

/// xLAM's list: the whole output is `[`, the calls joined by `, `, `]`, each call a JSON object
/// whose `name` is the tool's and whose `arguments` the tool's schema accepts, spelled as the
/// template spells it (`{"name": "NAME", "arguments": {...}}`, a space after each colon and
/// comma); with `at_least_one`, the list alone, and otherwise the list or prose, since the
/// model answers in prose instead. The prose is the assembler's rule exactly: its first byte
/// that is not whitespace is not `[`, and the rest is free, a `[` later in it included (a
/// Markdown link, in 592 of the 13,085 recorded prose answers). The one place the grammar is
/// narrower than the assembler: an output that starts with `[` and opens no object is content
/// to the assembler and not writable here.
pub(super) fn list_calls(tools: &[&Tool], count: Count<'_>) -> Grammar {
    let list = list(tools, count);
    if count.at_least_one {
        list
    } else {
        Grammar::Or(vec![
            Grammar::Sequence(vec![
                Grammar::Regex(r"\s*[^\s\[]".to_string()),
                Grammar::AnyText {
                    excludes: Vec::new(),
                },
            ]),
            list,
        ])
    }
}

/// Llama Nemotron's list between its markers: one tag, the opener, the list, the closer.
pub(super) fn list_tag(markers: &CallMarkers<'_>, tools: &[&Tool], count: Count<'_>) -> Grammar {
    // The tag's end carries the end of the turn, so the list inside it writes none.
    let inside = Count { last: "", ..count };
    count.triggered(
        vec![markers.call_open.to_string()],
        vec![Tag::new(
            markers.call_open,
            list(tools, inside),
            markers.call_close,
        )],
    )
}
