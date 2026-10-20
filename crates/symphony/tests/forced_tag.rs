//! Every table that reads tool calls writes a structural tag for a forced or required call in
//! every configuration a request and an engine present: `tool_choice` naming a function or
//! `required`, parallel calls on or off, the prompt's thought closed or left open, the engine's
//! grammar started where it does not say, after its reasoning parser, at the first token, or on
//! an engine of either kind (the engines a request may run on name different starts). A
//! table that writes none leaves the request to the gateway's older path, a JSON-schema
//! constraint whose output the table never reads, so the forced call comes back as raw text. The
//! gaps known today are listed with their cause and checked both ways: a listed gap that closes
//! fails the test until its entry goes, so the list stays what is true.

use openai_protocol::common::{Function, Tool};
use serde_json::json;
use symphony::{
    formats::Table,
    grammar::{GrammarStart, Options},
    CallSyntax,
};

/// Where a table writes no tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gap {
    /// With the prompt's thought open and the grammar started at the first token.
    OpenThoughtAtFirstToken,
}

/// The tables that write no forced tag today, each with its cause; an entry goes with the change
/// that makes its table write the tag.
const KNOWN_GAPS: &[(Table, Gap, &str)] = &[
    (
        Table::Inkling,
        Gap::OpenThoughtAtFirstToken,
        "the thought is a message of its own, and the tag finds no close of it to write before the call",
    ),
];

fn tool(name: &str, parameters: serde_json::Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
            extra: Default::default(),
        },
    }
}

/// Two tools: one with parameters, one with none.
fn tools() -> Vec<Tool> {
    vec![
        tool(
            "get_weather",
            json!({
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            }),
        ),
        tool("ping", json!({"type": "object", "properties": {}})),
    ]
}

/// Every configuration: the request's ask, the tools the gateway passes for it (a named function
/// narrows the list to that one tool, `ToolChoice::narrow_tools`: the one with parameters or the
/// one without; `required` passes both), and where the prompt and the engine leave the model.
fn configurations() -> Vec<(String, Options, std::ops::Range<usize>)> {
    let mut all = Vec::new();
    for (ask, one_call, passed) in [
        ("a named function with parameters", true, 0..1),
        ("a named function without parameters", true, 1..2),
        ("required", false, 0..2),
        ("required with parallel calls off", true, 0..2),
    ] {
        for (thought, reasoning_open) in [("thought closed", false), ("thought open", true)] {
            for start in [
                GrammarStart::Unknown,
                GrammarStart::AfterReasoning,
                GrammarStart::FirstToken,
                GrammarStart::Either,
            ] {
                all.push((
                    format!("{ask}, {thought}, start {start:?}"),
                    Options {
                        at_least_one: true,
                        reasoning_open,
                        thought_closed: false,
                        one_call,
                        start,
                    },
                    passed.clone(),
                ));
            }
        }
    }
    all
}

fn gap_of(table: Table) -> Option<Gap> {
    KNOWN_GAPS
        .iter()
        .find(|(known, ..)| *known == table)
        .map(|(_, gap, _)| *gap)
}

#[test]
fn every_table_that_reads_calls_writes_a_forced_tag_in_every_configuration() {
    let tools = tools();
    let mut checked = 0;
    for table in Table::ALL {
        let format = table.format();
        let Some(syntax) = format.call_syntax() else {
            continue;
        };
        assert!(
            !matches!(syntax, CallSyntax::ForcedArguments),
            "{}: the forced table is not a registry table",
            table.name()
        );
        let gap = gap_of(table);
        for (configuration, options, passed) in configurations() {
            let expected = match gap {
                Some(Gap::OpenThoughtAtFirstToken) => {
                    options.reasoning_open && options.start == GrammarStart::FirstToken
                }
                None => false,
            };
            let tag = format.grammar_for(&tools[passed], options);
            if expected {
                assert!(
                    tag.is_none(),
                    "{} ({configuration}): the table writes a tag here now; remove its entry from \
                     KNOWN_GAPS",
                    table.name()
                );
            } else {
                assert!(
                    tag.is_some(),
                    "{} ({configuration}): no structural tag, so a forced call on this table goes \
                     to the older path",
                    table.name()
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no table reads calls");
}

#[test]
fn every_known_gap_names_a_table_that_reads_calls_once() {
    for (at, (table, _, why)) in KNOWN_GAPS.iter().enumerate() {
        assert!(
            table.format().call_syntax().is_some(),
            "{}: a table without calls has no tag to miss",
            table.name()
        );
        assert!(
            !KNOWN_GAPS[..at].iter().any(|(other, ..)| other == table),
            "{} is listed twice",
            table.name()
        );
        assert!(!why.is_empty(), "{}: a gap says why", table.name());
    }
}
