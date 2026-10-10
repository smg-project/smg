//! Kimi K3's structural tag: the block of calls as the gateway's Kimi K3 parser writes it by hand,
//! here from the table's markers and the request's tools.
//!
//! The table gives the block its close and its ways in (`Format::ways_into`): Kimi K3's prompt
//! leaves the thought open in thinking mode and the answer otherwise, and a block the model could
//! not reach from there would force the calls inside a region the model can never close. So the
//! tag has one way in per state the block is entered from, each a path the table has: from the
//! thought or the answer, its close and then the row into the block from where the close returns;
//! from the turn itself, its own row. Each way is closed by the block's close, and its first text
//! triggers the tag. Inside, one or more calls, each `<|open|>call tool="NAME" index="N"<|sep|>`,
//! its arguments and the call's close, with the name as the request gives it (`&` and `"` escaped
//! as the template does).
//!
//! The arguments follow the tool's schema: one argument tag per property, in the schema's order,
//! each optional unless the schema requires it, its `type` attribute and value pinned by the
//! property's type. A string is written as it is, so its value is any text up to a marker; a
//! string `enum` is one of its values; a number, an object or an array is JSON the property's
//! schema accepts, with the tool's definitions attached so a `$ref` inside it still resolves; a
//! boolean is JSON `true` or `false`; `null` is the word. A property the schema leaves untyped, or
//! one whose `$ref` points at nothing, keeps its key but takes any type and any value; a schema
//! without properties takes any number of such arguments. The engine then accepts exactly what
//! the XTML assembler reads ([`tagged::xtml`](crate::tagged::xtml)).

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{
    block_of,
    schema::{self, shape, Definitions},
    CallMarkers, Grammar, WayIn,
};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
/// The call's count of the turn's calls, from one.
const INDEX: &str = "[1-9][0-9]*";

/// The block of calls for `tools`, entered by `ways_in`, or `None` for a table whose calls sit
/// in no block.
pub(super) fn calls(
    markers: &CallMarkers<'_>,
    ways_in: &[WayIn<'_>],
    tools: &[&Tool],
    at_least_one: bool,
) -> Option<Grammar> {
    let block = markers.block.as_ref()?;
    let one_call = Grammar::Or(
        tools
            .iter()
            .map(|tool| call(markers, &tool.function.name, &tool.function.parameters))
            .collect(),
    );
    let content = Grammar::Plus(Box::new(one_call));
    Some(block_of(ways_in, content, block.close, at_least_one))
}

/// One call: the call's opener and the tool's name, its index, the arguments and the call's close.
fn call(markers: &CallMarkers<'_>, name: &str, parameters: &Value) -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(format!("{}{}\" index=\"", markers.call_open, escape(name))),
        Grammar::Regex(INDEX.to_string()),
        Grammar::ConstString(format!("\"{SEP}")),
        arguments(parameters),
        Grammar::ConstString(markers.call_close.to_string()),
    ])
}

/// The arguments the tool's schema asks for: one tag per property, in the schema's order, each
/// optional unless required; any arguments at all for a schema without properties.
fn arguments(parameters: &Value) -> Grammar {
    schema::arguments(parameters, argument, || {
        Grammar::Star(Box::new(any_argument()))
    })
}

/// One argument tag: the key, the type the property's schema pins and a value of that type, or
/// the key with any type and any value when the schema pins none.
fn argument(key: &str, schema: &Value, definitions: Definitions<'_>) -> Grammar {
    let key = escape(key);
    match shape(schema, definitions, any_value) {
        Some((type_name, value)) => Grammar::Sequence(vec![
            Grammar::ConstString(format!(
                "{OPEN}argument key=\"{key}\" type=\"{type_name}\"{SEP}"
            )),
            value,
            Grammar::ConstString(format!("{CLOSE}argument{SEP}")),
        ]),
        None => Grammar::Sequence(vec![
            Grammar::ConstString(format!("{OPEN}argument key=\"{key}\" type=\"")),
            attribute_text(),
            Grammar::ConstString(format!("\"{SEP}")),
            any_value(),
            Grammar::ConstString(format!("{CLOSE}argument{SEP}")),
        ]),
    }
}

/// An argument tag with any key, any type and any value.
fn any_argument() -> Grammar {
    Grammar::Sequence(vec![
        Grammar::ConstString(format!("{OPEN}argument key=\"")),
        attribute_text(),
        Grammar::ConstString("\" type=\"".to_string()),
        attribute_text(),
        Grammar::ConstString(format!("\"{SEP}")),
        any_value(),
        Grammar::ConstString(format!("{CLOSE}argument{SEP}")),
    ])
}

/// The text of a quoted attribute: no quote, and not the start of a marker.
fn attribute_text() -> Grammar {
    Grammar::AnyText {
        excludes: vec!["\"".to_string(), "<|".to_string()],
    }
}

/// A value written as it is: any text up to a marker, a lone `<` or `|` included.
fn any_value() -> Grammar {
    Grammar::AnyText {
        excludes: vec![CLOSE.to_string(), OPEN.to_string(), SEP.to_string()],
    }
}

/// What the template escapes in an attribute: `&` first, then `"`.
fn escape(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;
    use crate::formats;

    fn tool(name: &str, parameters: Value) -> Tool {
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

    /// An argument tag's opener for `key`, typed `type_name`.
    fn argument_open(key: &str, type_name: &str) -> String {
        format!("<|open|>argument key=\"{key}\" type=\"{type_name}\"<|sep|>")
    }

    const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";

    /// The first call of the tag's first way in, as JSON.
    fn first_call(tools: &[Tool]) -> Value {
        let payload = formats::kimi_k3()
            .grammar(tools, true, false)
            .expect("a grammar")
            .payload();
        payload["format"]["tags"][0]["content"]["content"]["elements"][0].clone()
    }

    #[test]
    fn a_call_pins_its_framing_and_the_tools_name() {
        // The old crate's `build_call_format`: the opener with the name, the index, `"<|sep|>`,
        // the arguments, the call's close.
        let call = first_call(&[tool(
            "get_weather",
            value!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )]);
        assert_eq!(call["type"], "sequence");
        let elements = call["elements"].as_array().expect("five elements");
        assert_eq!(elements.len(), 5);
        assert_eq!(
            elements[0]["value"],
            "<|open|>call tool=\"get_weather\" index=\""
        );
        assert_eq!(
            elements[1],
            value!({"type": "regex", "pattern": "[1-9][0-9]*"})
        );
        assert_eq!(elements[2]["value"], "\"<|sep|>");
        assert_eq!(elements[3]["type"], "sequence");
        assert_eq!(elements[4]["value"], "<|close|>call<|sep|>");
    }

    #[test]
    fn the_arguments_follow_the_schema_one_tag_per_property_in_its_order() {
        // The old crate's `build_args_format` and `classify_arg`: a required string as any text up
        // to a marker, a required integer as `number` with the property's schema, an optional
        // string enum as one of its values inside `optional`.
        let call = first_call(&[tool(
            "get_weather",
            value!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1},
                    "units": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city", "days"],
            }),
        )]);
        let markers = value!(["<|close|>", "<|open|>", "<|sep|>"]);
        assert_eq!(
            call["elements"][3],
            value!({"type": "sequence", "elements": [
                {"type": "sequence", "elements": [
                    {"type": "const_string", "value": argument_open("city", "string")},
                    {"type": "any_text", "excludes": markers},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]},
                {"type": "sequence", "elements": [
                    {"type": "const_string", "value": argument_open("days", "number")},
                    {"type": "json_schema", "json_schema": {"type": "integer", "minimum": 1}},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]},
                {"type": "optional", "content": {"type": "sequence", "elements": [
                    {"type": "const_string", "value": argument_open("units", "string")},
                    {"type": "or", "elements": [
                        {"type": "const_string", "value": "c"},
                        {"type": "const_string", "value": "f"},
                    ]},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]}},
            ]})
        );
    }

    #[test]
    fn every_type_the_syntax_has_pins_its_attribute_and_value() {
        let call = first_call(&[tool(
            "t",
            value!({
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean", "description": "ignored"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "where": {"type": "object", "properties": {"x": {"type": "number"}}},
                    "nothing": {"type": "null"},
                    "ratio": {"type": "number"},
                },
                "required": ["flag", "tags", "where", "nothing", "ratio"],
            }),
        )]);
        let slots = call["elements"][3]["elements"]
            .as_array()
            .expect("five slots");
        let opener = |slot: &Value| slot["elements"][0]["value"].clone();
        let pinned = |slot: &Value| slot["elements"][1].clone();
        assert_eq!(
            opener(&slots[0]),
            "<|open|>argument key=\"flag\" type=\"boolean\"<|sep|>"
        );
        assert_eq!(
            pinned(&slots[0]),
            value!({"type": "json_schema", "json_schema": {"type": "boolean"}})
        );
        assert_eq!(
            opener(&slots[1]),
            "<|open|>argument key=\"tags\" type=\"array\"<|sep|>"
        );
        assert_eq!(
            pinned(&slots[1])["json_schema"],
            value!({"type": "array", "items": {"type": "string"}})
        );
        assert_eq!(
            opener(&slots[2]),
            "<|open|>argument key=\"where\" type=\"object\"<|sep|>"
        );
        assert_eq!(
            opener(&slots[3]),
            "<|open|>argument key=\"nothing\" type=\"null\"<|sep|>"
        );
        assert_eq!(
            pinned(&slots[3]),
            value!({"type": "const_string", "value": "null"})
        );
        assert_eq!(
            opener(&slots[4]),
            "<|open|>argument key=\"ratio\" type=\"number\"<|sep|>"
        );
    }

    #[test]
    fn a_reference_brings_the_tools_definitions_along_and_a_dangling_one_pins_nothing() {
        let call = first_call(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {"Count": {"type": "integer"}, "Unused": {"type": "string"}},
                "properties": {
                    "count": {"$ref": "#/$defs/Count"},
                    "lost": {"$ref": "#/$defs/Missing"},
                    "untyped": {"description": "anything"},
                    "whole": {"$ref": "#"},
                },
                "required": ["count", "lost", "untyped", "whole"],
            }),
        )]);
        let slots = call["elements"][3]["elements"]
            .as_array()
            .expect("four slots");
        // The `$ref` lends its target's type, and the definitions travel with the schema.
        assert_eq!(
            slots[0]["elements"][0]["value"],
            "<|open|>argument key=\"count\" type=\"number\"<|sep|>"
        );
        assert_eq!(
            slots[0]["elements"][1]["json_schema"],
            value!({
                "$ref": "#/$defs/Count",
                "$defs": {"Count": {"type": "integer"}, "Unused": {"type": "string"}},
            })
        );
        // A pointer at nothing, no type, or a pointer to the root (the property's own document once
        // the engine compiles it alone): the key stays, the type and the value are the model's.
        for (slot, key) in [
            (&slots[1], "lost"),
            (&slots[2], "untyped"),
            (&slots[3], "whole"),
        ] {
            let begin = format!("<|open|>argument key=\"{key}\" type=\"");
            assert_eq!(
                *slot,
                value!({"type": "sequence", "elements": [
                    {"type": "const_string", "value": begin},
                    {"type": "any_text", "excludes": ["\"", "<|"]},
                    {"type": "const_string", "value": "\"<|sep|>"},
                    {"type": "any_text", "excludes": ["<|close|>", "<|open|>", "<|sep|>"]},
                    {"type": "const_string", "value": ARGUMENT_CLOSE},
                ]})
            );
        }
    }

    #[test]
    fn a_pointer_inside_a_definition_counts_as_the_propertys_own() {
        // The definitions travel with the property's schema, so a pointer at nothing, or to the
        // root, inside a definition the property names reaches the engine as surely as one in the
        // property itself: that property pins nothing. A definition that names itself is followed
        // once and resolves, so a recursive tree keeps its schema, the definitions attached.
        let call = first_call(&[tool(
            "t",
            value!({
                "type": "object",
                "$defs": {
                    "Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}},
                    "Broken": {"type": "object", "properties": {"x": {"$ref": "#/$defs/Gone"}}},
                    "Rooted": {"type": "object", "properties": {"up": {"$ref": "#"}}},
                },
                "properties": {
                    "tree": {"$ref": "#/$defs/Node"},
                    "broken": {"$ref": "#/$defs/Broken"},
                    "rooted": {"$ref": "#/$defs/Rooted"},
                },
                "required": ["tree", "broken", "rooted"],
            }),
        )]);
        let slots = call["elements"][3]["elements"]
            .as_array()
            .expect("three slots");
        assert_eq!(
            slots[0]["elements"][0]["value"],
            argument_open("tree", "object")
        );
        assert_eq!(
            slots[0]["elements"][1]["json_schema"]["$ref"],
            "#/$defs/Node"
        );
        for (slot, key) in [(&slots[1], "broken"), (&slots[2], "rooted")] {
            assert_eq!(
                slot["elements"][0]["value"],
                format!("<|open|>argument key=\"{key}\" type=\""),
                "{key} pins nothing"
            );
        }
    }

    #[test]
    fn attribute_values_are_escaped_as_the_template_writes_them() {
        let parameters = value!({
            "type": "object",
            "properties": {"k\"q": {"type": "string"}},
            "required": ["k\"q"],
        });
        let call = first_call(&[tool("a&b\"c", parameters)]);
        assert_eq!(
            call["elements"][0]["value"],
            "<|open|>call tool=\"a&amp;b&quot;c\" index=\""
        );
        assert_eq!(
            call["elements"][3]["elements"][0]["elements"][0]["value"],
            "<|open|>argument key=\"k&quot;q\" type=\"string\"<|sep|>"
        );
    }

    #[test]
    fn a_schema_without_properties_takes_any_arguments() {
        for parameters in [
            value!({"type": "object"}),
            value!({}),
            value!({"properties": {}}),
        ] {
            let call = first_call(&[tool("ping", parameters)]);
            assert_eq!(call["elements"][3]["type"], "star");
            assert_eq!(
                call["elements"][3]["content"]["elements"][0]["value"],
                "<|open|>argument key=\""
            );
        }
    }
}
