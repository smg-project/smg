//! Every table that reads tool calls writes a structural tag for a forced or required call in
//! every configuration a request and an engine present: `tool_choice` naming a function or
//! `required`, parallel calls on or off, the prompt's thought closed or left open, the engine's
//! grammar started where it does not say, after its reasoning parser or at the first token. A
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
    /// In every configuration.
    Every,
    /// With the prompt's thought open and the grammar started at the first token.
    OpenThoughtAtFirstToken,
}

/// The tables that write no forced tag today, each with its cause; an entry goes with the change
/// that makes its table write the tag.
const KNOWN_GAPS: &[(Table, Gap, &str)] = &[
    (
        Table::KimiK2,
        Gap::Every,
        "a shaped call with JSON arguments has no tag without a layout, and the table carries none",
    ),
    (
        Table::DeepSeekV3_1,
        Gap::Every,
        "the named-arguments builder writes Mistral's shape alone and does not see through the block",
    ),
    (
        Table::DeepSeekV3,
        Gap::Every,
        "the same block as V3.1's, read by the same builder",
    ),
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

/// Every configuration: the request's ask, and where the prompt and the engine leave the model.
fn configurations() -> Vec<(String, Options)> {
    let mut all = Vec::new();
    for (ask, one_call) in [
        ("a named function", true),
        ("required", false),
        ("required with parallel calls off", true),
    ] {
        for (thought, reasoning_open) in [("thought closed", false), ("thought open", true)] {
            for start in [
                GrammarStart::Unknown,
                GrammarStart::AfterReasoning,
                GrammarStart::FirstToken,
            ] {
                all.push((
                    format!("{ask}, {thought}, start {start:?}"),
                    Options {
                        at_least_one: true,
                        reasoning_open,
                        one_call,
                        start,
                    },
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
        for (configuration, options) in configurations() {
            let expected = match gap {
                Some(Gap::Every) => true,
                Some(Gap::OpenThoughtAtFirstToken) => {
                    options.reasoning_open && options.start == GrammarStart::FirstToken
                }
                None => false,
            };
            let tag = format.grammar_for(&tools, options);
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
